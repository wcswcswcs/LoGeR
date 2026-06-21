from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.sim3 import Sim3Transform
from stream4d_native.sim3 import apply_sim3_to_xyz, fit_sim3_umeyama
from tools.trace_v25_real_geometry_flow import load_scene_chunks_from_cache


ROOT = Path(__file__).resolve().parents[1]
PROBE5_SCENES = ["scene0030_00", "scene0081_01", "scene0591_00", "scene0011_00", "scene0050_00"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in keys})


def _parse_scenes(value: str) -> list[str]:
    if str(value).strip() in {"", "probe5"}:
        return list(PROBE5_SCENES)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _backproject_uv_world(stream: ScanNetStream, frame_id: int, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = stream.load_depth(int(frame_id))
    pose = stream.load_pose(int(frame_id))
    intr = stream.load_intrinsics()
    h, w = depth.shape[:2]
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    out = np.full((uv.shape[0], 3), np.nan, dtype=np.float64)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h) & np.isfinite(pose).all()
    if not np.any(ok):
        return out, np.isfinite(out).all(axis=1)
    idx = np.flatnonzero(ok)
    z = depth[y[idx], x[idx]].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.0)
    idx = idx[valid]
    if idx.size == 0:
        return out, np.isfinite(out).all(axis=1)
    z = depth[y[idx], x[idx]].astype(np.float64)
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    cam = np.stack(
        [
            (x[idx].astype(np.float64) - cx) * z / fx,
            (y[idx].astype(np.float64) - cy) * z / fy,
            z,
            np.ones_like(z),
        ],
        axis=1,
    )
    out[idx] = (pose.astype(np.float64) @ cam.T).T[:, :3]
    return out, np.isfinite(out).all(axis=1)


def _robust_fit(source: np.ndarray, target: np.ndarray) -> tuple[dict[str, Any], np.ndarray, int]:
    first = fit_sim3_umeyama(source, target)
    pred = apply_sim3_to_xyz(source, transform=first).astype(np.float64)
    residual = np.linalg.norm(pred - target, axis=1)
    threshold = float(np.quantile(residual, 0.80))
    keep = np.isfinite(residual) & (residual <= threshold)
    if int(np.count_nonzero(keep)) >= 32:
        final = fit_sim3_umeyama(source[keep], target[keep])
        pred = apply_sim3_to_xyz(source[keep], transform=final).astype(np.float64)
        residual = np.linalg.norm(pred - target[keep], axis=1)
        return final, residual, int(np.count_nonzero(keep))
    return first, residual, int(source.shape[0])


def _window_scale_row(
    *,
    scene: str,
    stream: ScanNetStream,
    carrier_path: Path,
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
) -> dict[str, Any]:
    window = int(carrier_path.stem.replace("carriers_window", ""))
    manifest = json.loads(carrier_path.with_name(carrier_path.stem + "_manifest.json").read_text())
    frame_ids = [int(v) for v in manifest["frame_ids"]]
    with np.load(carrier_path) as data:
        xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
        uv = np.asarray(data["uv_pred"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)
        visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
        confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    per_frame_cap = max(16, int(max_anchors) // max(len(frame_ids), 1))
    for local_idx, frame_id in enumerate(frame_ids):
        ok = (
            valid[local_idx]
            & (visibility[local_idx] >= float(min_visibility))
            & (confidence[local_idx] >= float(min_confidence))
            & np.isfinite(uv[local_idx]).all(axis=1)
            & np.isfinite(xyz[local_idx]).all(axis=1)
            & (uv[local_idx, :, 0] >= 0.0)
            & (uv[local_idx, :, 0] <= 1.0)
            & (uv[local_idx, :, 1] >= 0.0)
            & (uv[local_idx, :, 1] <= 1.0)
        )
        indices = np.flatnonzero(ok)
        if indices.size == 0:
            continue
        if indices.size > per_frame_cap:
            indices = indices[np.linspace(0, indices.size - 1, per_frame_cap, dtype=np.int64)]
        world, world_ok = _backproject_uv_world(stream, frame_id, uv[local_idx, indices])
        if np.any(world_ok):
            src_parts.append(xyz[local_idx, indices][world_ok].astype(np.float64))
            tgt_parts.append(world[world_ok].astype(np.float64))
    if not src_parts:
        return {"scene": scene, "window": window, "status": "no_anchors", "carrier_path": str(carrier_path)}
    source = np.concatenate(src_parts, axis=0)
    target = np.concatenate(tgt_parts, axis=0)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 32:
        return {
            "scene": scene,
            "window": window,
            "status": "too_few_anchors",
            "anchor_count": int(source.shape[0]),
            "carrier_path": str(carrier_path),
        }
    fit, residual, kept = _robust_fit(source, target)
    return {
        "scene": scene,
        "window": window,
        "status": "ok",
        "scale_native_to_gt_world": float(fit["scale"]),
        "anchor_count": int(source.shape[0]),
        "kept_anchor_count": int(kept),
        "residual_p50": float(np.median(residual)),
        "residual_p90": float(np.quantile(residual, 0.90)),
        "frame_min": int(min(frame_ids)),
        "frame_max": int(max(frame_ids)),
        "carrier_path": str(carrier_path),
    }


def _ratio_rows(window_rows: list[dict[str, Any]], scenes: list[str]) -> list[dict[str, Any]]:
    ratio_rows: list[dict[str, Any]] = []
    for scene in scenes:
        rows = sorted(
            [row for row in window_rows if row.get("scene") == scene and row.get("status") == "ok"],
            key=lambda row: int(row["window"]),
        )
        for left, right in zip(rows, rows[1:]):
            prev = float(left["scale_native_to_gt_world"])
            nxt = float(right["scale_native_to_gt_world"])
            ratio = nxt / prev if prev > 0.0 else float("nan")
            ratio_rows.append(
                {
                    "scene": scene,
                    "window_pair": f"{left['window']}-{right['window']}",
                    "scale_prev": prev,
                    "scale_next": nxt,
                    "scale_next_over_prev": float(ratio),
                    "abs_log_scale_ratio": float(abs(math.log(ratio))) if ratio > 0.0 else None,
                    "scale_aligned_within_5pct": bool(0.95 <= ratio <= 1.05),
                    "scale_aligned_within_10pct": bool(0.90 <= ratio <= 1.10),
                }
            )
    return ratio_rows


def _variant_summary(variant: str, window_rows: list[dict[str, Any]], ratio_rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_abs_log = max(
        (float(row["abs_log_scale_ratio"]) for row in ratio_rows if row["abs_log_scale_ratio"] is not None),
        default=None,
    )
    return {
        "variant": variant,
        "ok_window_count": int(sum(1 for row in window_rows if row.get("status") == "ok")),
        "ratio_count": int(len(ratio_rows)),
        "max_abs_log_scale_ratio": max_abs_log,
        "outside_5pct_pair_count": int(sum(1 for row in ratio_rows if not bool(row["scale_aligned_within_5pct"]))),
        "outside_10pct_pair_count": int(sum(1 for row in ratio_rows if not bool(row["scale_aligned_within_10pct"]))),
        "all_adjacent_within_5pct": bool(ratio_rows and all(bool(row["scale_aligned_within_5pct"]) for row in ratio_rows)),
        "all_adjacent_within_10pct": bool(ratio_rows and all(bool(row["scale_aligned_within_10pct"]) for row in ratio_rows)),
    }


def _stitch_with_min_inlier_gate(
    builder: D4RTNativeSceneBuilder,
    local_chunks: list[dict[str, Any]],
    *,
    min_inlier_abs010: float,
    alignment_source: str,
) -> dict[str, Any]:
    stitched: list[dict[str, Any]] = []
    weak_alignment = 0
    pairwise: list[dict[str, Any] | None] = []
    identity = Sim3Transform(scale=1.0, rot=np.eye(3, dtype=np.float64), trans=np.zeros((3,), dtype=np.float64))
    for idx, chunk in enumerate(local_chunks):
        if idx == 0:
            stitched.append(
                builder._apply_chunk_canonical_transform(
                    chunk,
                    transform=identity,
                    transform_id="chunk000_identity",
                    submap_id=0,
                    alignment_quality={"pass_gate": True, "anchor_count": None, "fail_reason": None},
                    alignment_source="same_chunk_identity",
                    allow_metric_merge=True,
                    weak_alignment=False,
                )
            )
            continue
        fit = builder.estimate_overlap_self_sim3(
            stitched[-1],
            chunk,
            min_inlier_abs010=float(min_inlier_abs010),
        )
        pairwise.append(fit)
        if fit is None or not bool(fit.get("pass_gate", False)):
            weak_alignment += 1
            submap_id = int(stitched[-1].get("submap_id", -1)) + 1
            stitched.append(
                builder._apply_chunk_canonical_transform(
                    {**chunk, "self_sim3_to_previous": fit},
                    transform=identity,
                    transform_id=f"chunk{idx:03d}_submap_identity",
                    submap_id=submap_id,
                    alignment_quality={
                        "pass_gate": False,
                        "anchor_count": None if fit is None else int(fit.get("anchor_count", 0)),
                        "fail_reason": "missing_fit" if fit is None else fit.get("fail_reason"),
                    },
                    alignment_source="submap_identity_after_failed_self_sim3",
                    allow_metric_merge=False,
                    weak_alignment=True,
                )
            )
            continue
        transform = Sim3Transform(
            scale=float(fit["scale"]),
            rot=np.asarray(fit["rot"], dtype=np.float64),
            trans=np.asarray(fit["trans"], dtype=np.float64),
        )
        alignment_quality = {
            "pass_gate": True,
            "anchor_count": int(fit.get("anchor_count", 0)),
            "used_anchor_count": int(fit.get("match_stats", {}).get("used_anchor_count", fit.get("anchor_count", 0))),
            "residual_median": fit.get("residual_median"),
            "residual_p90": fit.get("residual_p90"),
            "inlier_ratio_abs010": fit.get("inlier_ratio_abs010"),
            "fail_reason": None,
        }
        stitched.append(
            builder._apply_chunk_canonical_transform(
                {**chunk, "self_sim3_to_previous": fit},
                transform=transform,
                transform_id=f"chunk{idx:03d}_{alignment_source}",
                submap_id=int(stitched[-1].get("submap_id", 0)),
                alignment_quality=alignment_quality,
                alignment_source=alignment_source,
                allow_metric_merge=True,
                weak_alignment=False,
            )
        )
    return {
        "chunks": stitched,
        "diagnostics": {
            "num_chunks": int(len(local_chunks)),
            "weak_alignment_chunk_count": int(weak_alignment),
            "submap_count": int(len({int(chunk.get("submap_id", 0)) for chunk in stitched})),
            "canonicalized_chunk_count": int(sum(1 for chunk in stitched if chunk.get("tubes"))),
            "pairwise_self_sim3": pairwise,
            "min_inlier_abs010": float(min_inlier_abs010),
            "alignment_source": alignment_source,
        },
    }


def _canonical_window_scale_row(
    *,
    scene: str,
    stream: ScanNetStream,
    chunk: dict[str, Any],
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
    variant: str,
) -> dict[str, Any]:
    meta = chunk.get("chunk", chunk)
    window = int(meta.get("chunk_id", 0))
    frame_ids = [int(v) for v in meta.get("frame_ids", [])]
    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    per_frame_cap = max(16, int(max_anchors) // max(len(frame_ids), 1))
    for local_idx, frame_id in enumerate(frame_ids):
        uv_values: list[np.ndarray] = []
        xyz_values: list[np.ndarray] = []
        for tube in chunk.get("tubes", []):
            uv = np.asarray(tube.get("uv_norm", tube.get("uv")), dtype=np.float32)
            xyz = np.asarray(tube.get("xyz_canonical"), dtype=np.float32)
            valid = np.asarray(tube.get("valid", np.ones((uv.shape[0],), dtype=bool)), dtype=bool)
            visibility = np.asarray(tube.get("visibility", np.ones((uv.shape[0],), dtype=np.float32)), dtype=np.float32)
            confidence = np.asarray(tube.get("confidence", np.ones((uv.shape[0],), dtype=np.float32)), dtype=np.float32)
            if local_idx >= uv.shape[0] or local_idx >= xyz.shape[0]:
                continue
            if not bool(valid[local_idx]):
                continue
            if float(visibility[local_idx]) < float(min_visibility) or float(confidence[local_idx]) < float(min_confidence):
                continue
            if not (np.isfinite(uv[local_idx]).all() and np.isfinite(xyz[local_idx]).all()):
                continue
            if not (0.0 <= float(uv[local_idx, 0]) <= 1.0 and 0.0 <= float(uv[local_idx, 1]) <= 1.0):
                continue
            uv_values.append(uv[local_idx].astype(np.float32))
            xyz_values.append(xyz[local_idx].astype(np.float32))
        if not uv_values:
            continue
        indices = np.arange(len(uv_values), dtype=np.int64)
        if indices.size > per_frame_cap:
            indices = indices[np.linspace(0, indices.size - 1, per_frame_cap, dtype=np.int64)]
        uv_arr = np.stack([uv_values[int(idx)] for idx in indices], axis=0)
        xyz_arr = np.stack([xyz_values[int(idx)] for idx in indices], axis=0)
        world, world_ok = _backproject_uv_world(stream, frame_id, uv_arr)
        if np.any(world_ok):
            src_parts.append(xyz_arr[world_ok].astype(np.float64))
            tgt_parts.append(world[world_ok].astype(np.float64))
    if not src_parts:
        return {
            "scene": scene,
            "window": window,
            "variant": variant,
            "status": "no_anchors",
            "chunk_alignment_source": chunk.get("alignment_source"),
            "submap_id": chunk.get("submap_id"),
        }
    source = np.concatenate(src_parts, axis=0)
    target = np.concatenate(tgt_parts, axis=0)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 32:
        return {
            "scene": scene,
            "window": window,
            "variant": variant,
            "status": "too_few_anchors",
            "anchor_count": int(source.shape[0]),
            "chunk_alignment_source": chunk.get("alignment_source"),
            "submap_id": chunk.get("submap_id"),
        }
    fit, residual, kept = _robust_fit(source, target)
    return {
        "scene": scene,
        "window": window,
        "variant": variant,
        "status": "ok",
        "scale_native_to_gt_world": float(fit["scale"]),
        "anchor_count": int(source.shape[0]),
        "kept_anchor_count": int(kept),
        "residual_p50": float(np.median(residual)),
        "residual_p90": float(np.quantile(residual, 0.90)),
        "frame_min": int(min(frame_ids)) if frame_ids else None,
        "frame_max": int(max(frame_ids)) if frame_ids else None,
        "chunk_alignment_source": chunk.get("alignment_source"),
        "allow_metric_merge": chunk.get("allow_metric_merge"),
        "weak_alignment": chunk.get("weak_alignment"),
        "submap_id": chunk.get("submap_id"),
    }


def _canonical_variant_rows(
    *,
    scenes: list[str],
    cache_root: Path,
    variant: str,
    relaxed_min_inlier_abs010: float | None,
    max_tubes_per_window: int,
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for scene in scenes:
        stream = ScanNetStream(seq_name=scene)
        chunks, load_diag = load_scene_chunks_from_cache(
            cache_root / scene,
            max_tubes_per_window=int(max_tubes_per_window),
            image_width=640,
            image_height=480,
        )
        builder = D4RTNativeSceneBuilder(
            object(),
            {"model": {"input": {"clip_frames": 32}}},
            temporal_chunk_size=32,
            temporal_chunk_stride=16,
        )
        if relaxed_min_inlier_abs010 is None:
            stitched = builder.stitch_to_canonical(chunks)
        else:
            stitched = _stitch_with_min_inlier_gate(
                builder,
                chunks,
                min_inlier_abs010=float(relaxed_min_inlier_abs010),
                alignment_source=f"d4rt_self_sim3_relaxed{float(relaxed_min_inlier_abs010):.2f}".replace(".", ""),
            )
        diagnostics.append(
            {
                "scene": scene,
                "variant": variant,
                "load_diag": load_diag,
                "stitch_diagnostics": stitched.get("diagnostics", {}),
            }
        )
        for chunk in stitched.get("chunks", []):
            rows.append(
                _canonical_window_scale_row(
                    scene=scene,
                    stream=stream,
                    chunk=chunk,
                    max_anchors=max_anchors,
                    min_visibility=min_visibility,
                    min_confidence=min_confidence,
                    variant=variant,
                )
            )
    return rows, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_scenes(args.scenes)
    cache_root = ROOT / args.cache_root
    output_root = ROOT / args.output_root
    raw_window_rows: list[dict[str, Any]] = []
    for scene in scenes:
        stream = ScanNetStream(seq_name=scene)
        for carrier_path in sorted((cache_root / scene).glob("carriers_window*.npz")):
            raw_window_rows.append(
                _window_scale_row(
                    scene=scene,
                    stream=stream,
                    carrier_path=carrier_path,
                    max_anchors=int(args.max_anchors),
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                )
            )
    raw_ratio_rows = _ratio_rows(raw_window_rows, scenes)
    canonical_strict_rows, canonical_strict_diag = _canonical_variant_rows(
        scenes=scenes,
        cache_root=cache_root,
        variant="canonical_strict_existing",
        relaxed_min_inlier_abs010=None,
        max_tubes_per_window=int(args.max_tubes_per_window),
        max_anchors=int(args.max_anchors),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )
    canonical_strict_ratios = _ratio_rows(canonical_strict_rows, scenes)
    canonical_relaxed_rows, canonical_relaxed_diag = _canonical_variant_rows(
        scenes=scenes,
        cache_root=cache_root,
        variant="canonical_relaxed030",
        relaxed_min_inlier_abs010=float(args.relaxed_min_inlier_abs010),
        max_tubes_per_window=int(args.max_tubes_per_window),
        max_anchors=int(args.max_anchors),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )
    canonical_relaxed_ratios = _ratio_rows(canonical_relaxed_rows, scenes)
    variant_summaries = {
        "raw_input": _variant_summary("raw_input", raw_window_rows, raw_ratio_rows),
        "canonical_strict_existing": _variant_summary(
            "canonical_strict_existing",
            canonical_strict_rows,
            canonical_strict_ratios,
        ),
        "canonical_relaxed030": _variant_summary("canonical_relaxed030", canonical_relaxed_rows, canonical_relaxed_ratios),
    }
    summary = {
        "phase": "v44_chunk_scale_diagnostic",
        "cache_root": str(cache_root),
        "scenes": scenes,
        "uses_gt_depth_pose_for_diagnostic": True,
        "uses_gt_for_prediction": False,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "raw_window_rows": raw_window_rows,
        "raw_ratio_rows": raw_ratio_rows,
        "canonical_strict_window_rows": canonical_strict_rows,
        "canonical_strict_ratio_rows": canonical_strict_ratios,
        "canonical_strict_diagnostics": canonical_strict_diag,
        "canonical_relaxed_window_rows": canonical_relaxed_rows,
        "canonical_relaxed_ratio_rows": canonical_relaxed_ratios,
        "canonical_relaxed_diagnostics": canonical_relaxed_diag,
        "variant_summaries": variant_summaries,
        "max_abs_log_scale_ratio": variant_summaries["raw_input"]["max_abs_log_scale_ratio"],
        "all_adjacent_within_5pct": variant_summaries["raw_input"]["all_adjacent_within_5pct"],
        "all_adjacent_within_10pct": variant_summaries["raw_input"]["all_adjacent_within_10pct"],
        "conclusion": (
            "chunk_scale_repair_partial_one_pair_outside_10pct"
            if variant_summaries["canonical_relaxed030"]["outside_10pct_pair_count"] > 0
            else "chunk_scale_repair_candidate_within_10pct"
        ),
    }
    _write_json(output_root / "chunk_scale_diagnostic_summary.json", summary)
    _write_csv(output_root / "chunk_scale_window_rows.csv", raw_window_rows)
    _write_csv(output_root / "chunk_scale_ratio_rows.csv", raw_ratio_rows)
    _write_csv(output_root / "chunk_scale_raw_window_rows.csv", raw_window_rows)
    _write_csv(output_root / "chunk_scale_raw_ratio_rows.csv", raw_ratio_rows)
    _write_csv(output_root / "chunk_scale_canonical_strict_window_rows.csv", canonical_strict_rows)
    _write_csv(output_root / "chunk_scale_canonical_strict_ratio_rows.csv", canonical_strict_ratios)
    _write_csv(output_root / "chunk_scale_canonical_relaxed030_window_rows.csv", canonical_relaxed_rows)
    _write_csv(output_root / "chunk_scale_canonical_relaxed030_ratio_rows.csv", canonical_relaxed_ratios)
    print(
        json.dumps(
            _json_safe(
                {
                    "conclusion": summary["conclusion"],
                    "variant_summaries": variant_summaries,
                    "canonical_relaxed_ratio_rows": canonical_relaxed_ratios,
                }
            ),
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose inter-window chunk scale alignment for v44 native D4RT cache.")
    parser.add_argument("--scenes", default="probe5")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit/v44_chunk_scale_diagnostic_probe5")
    parser.add_argument("--max-anchors", type=int, default=2048)
    parser.add_argument("--max-tubes-per-window", type=int, default=1920)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--relaxed-min-inlier-abs010", type=float, default=0.30)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
