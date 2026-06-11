from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

STREAM3D_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider
from stream4d.scannet_stream import ScanNetStream
from tools.diagnose_v22_ref0_trajectory_scale import _finite_values
from tools.run_v22_direct_reconstruction_benchmark import (
    _fit_ref0_pose_scale,
    _json_safe,
    _read_seq_list,
    _sample_indices,
)


_CANDIDATE_KEYS = [
    "loger_z_over_d4rt_local_z_median",
    "loger_z_over_d4rt_ref_z_median",
    "loger_norm_over_d4rt_local_norm_median",
    "loger_norm_over_d4rt_ref_norm_median",
    "d4rt_local_z_over_loger_z_median",
    "scannet_depth_over_loger_z_median",
    "scannet_depth_over_d4rt_local_z_median",
]

_NO_GT_PROXY_KEYS = [
    "loger_z_over_d4rt_local_z_median",
    "loger_z_over_d4rt_ref_z_median",
    "loger_norm_over_d4rt_local_norm_median",
    "loger_norm_over_d4rt_ref_norm_median",
    "d4rt_local_z_over_loger_z_median",
]


def _median_positive(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 1e-8)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 1e-8:
        return None
    return float(num / den)


def _candidate_error(value: float | None, eval_scale: float | None) -> float | None:
    if value is None or eval_scale is None:
        return None
    val = float(value)
    ref = float(eval_scale)
    if not np.isfinite(val) or not np.isfinite(ref) or ref <= 1e-8:
        return None
    return float(abs(val - ref) / ref)


def _sample_pointmap_uv(pointmap: np.ndarray, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor sample [H,W,C] pointmap at normalized UV coordinates."""
    pts = np.asarray(pointmap, dtype=np.float64)
    uv = np.asarray(uv_norm, dtype=np.float64).reshape(-1, 2)
    if pts.ndim != 3 or pts.shape[-1] == 0:
        raise ValueError(f"Expected pointmap [H,W,C], got {pts.shape}")
    h, w = pts.shape[:2]
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    ok = (
        np.isfinite(uv).all(axis=1)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )
    out = np.full((uv.shape[0], pts.shape[-1]), np.nan, dtype=np.float64)
    out[ok] = pts[y[ok], x[ok]]
    ok &= np.isfinite(out).all(axis=1)
    return out, ok


def _sample_map_uv(map_2d: np.ndarray, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(map_2d, dtype=np.float64)
    uv = np.asarray(uv_norm, dtype=np.float64).reshape(-1, 2)
    h, w = arr.shape[:2]
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    ok = (
        np.isfinite(uv).all(axis=1)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )
    out = np.full((uv.shape[0],), np.nan, dtype=np.float64)
    out[ok] = arr[y[ok], x[ok]]
    ok &= np.isfinite(out)
    return out, ok


def _load_loger_output(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "local_points" not in payload:
        raise KeyError(f"{path} does not contain local_points")
    local = payload["local_points"]
    if hasattr(local, "detach"):
        local = local.detach().cpu().numpy()
    confidence = payload.get("confidence")
    if confidence is not None and hasattr(confidence, "detach"):
        confidence = confidence.detach().cpu().numpy()
    return {
        "local_points": np.asarray(local, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32) if confidence is not None else None,
    }


def _loger_path(root: Path, template: str, scene: str) -> Path:
    scene_short = scene.split("_")[0]
    return root / template.format(scene=scene, scene_short=scene_short)


def _frame_ids_for_loger(stream: ScanNetStream, frame_stride: int, max_frames: int | None, actual_count: int) -> list[int]:
    frame_ids = stream.frame_ids(stride=int(frame_stride), max_frames=max_frames)
    if len(frame_ids) < actual_count:
        raise ValueError(f"LoGeR output has {actual_count} frames but only {len(frame_ids)} inferred frame ids")
    return frame_ids[:actual_count]


def _diagnose_scene(
    scene: str,
    *,
    stream: ScanNetStream,
    provider: D4RTCarrierProjectionProvider,
    loger_payload: dict[str, np.ndarray],
    frame_stride: int,
    max_frames: int | None,
    max_windows_per_scene: int | None,
    max_anchors: int,
    robust_trim_percentile: float,
    min_visibility: float,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache = provider._load_scene(scene)
    windows = list(cache["windows"])
    if max_windows_per_scene is not None:
        windows = windows[: int(max_windows_per_scene)]
    local_points = loger_payload["local_points"]
    confidence = loger_payload.get("confidence")
    loger_frame_ids = _frame_ids_for_loger(stream, frame_stride, max_frames, int(local_points.shape[0]))
    loger_frame_to_idx = {int(fid): idx for idx, fid in enumerate(loger_frame_ids)}

    frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        _, eval_diag = _fit_ref0_pose_scale(
            stream,
            window,
            robust_trim_percentile=float(robust_trim_percentile),
            max_anchors=int(max_anchors),
        )
        eval_scale = eval_diag.get("ref0_pose_scale")
        with np.load(window.path) as data:
            if "xyz_local" not in data.files:
                window_rows.append(
                    {
                        "scene": scene,
                        "window_index": int(window_index),
                        "window_path": str(window.path),
                        "status": "missing_xyz_local",
                        **eval_diag,
                    }
                )
                continue
            xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
            xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float64)
        if xyz_local.shape != np.asarray(window.xyz).shape or xyz_ref.shape != xyz_local.shape:
            window_rows.append(
                {
                    "scene": scene,
                    "window_index": int(window_index),
                    "window_path": str(window.path),
                    "status": "shape_mismatch",
                    **eval_diag,
                }
            )
            continue
        per_frame_cap = max(16, int(max_anchors) // max(len(window.frame_ids), 1))
        row_count_before = len(frame_rows)
        for local_idx, frame_id in enumerate(window.frame_ids):
            if int(frame_id) not in loger_frame_to_idx:
                continue
            loger_idx = loger_frame_to_idx[int(frame_id)]
            uv = np.asarray(window.uv[local_idx], dtype=np.float64)
            ok = (
                np.asarray(window.valid[local_idx], dtype=bool)
                & np.isfinite(xyz_local[local_idx]).all(axis=1)
                & np.isfinite(xyz_ref[local_idx]).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= float(min_visibility))
                & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= float(min_confidence))
            )
            indices = np.flatnonzero(ok)
            if indices.shape[0] < 8:
                continue
            indices = _sample_indices(indices, per_frame_cap)
            loger_xyz, loger_ok = _sample_pointmap_uv(local_points[loger_idx], uv[indices])
            if confidence is not None:
                loger_conf, loger_conf_ok = _sample_map_uv(confidence[loger_idx], uv[indices])
                loger_ok &= loger_conf_ok
            else:
                loger_conf = np.full((indices.shape[0],), np.nan, dtype=np.float64)
            target_depth, target_ok = _sample_map_uv(stream.load_depth(int(frame_id)), uv[indices])
            use = loger_ok & target_ok
            if np.count_nonzero(use) < 8:
                continue
            idx = indices[use]
            loger_xyz = loger_xyz[use]
            loger_conf = loger_conf[use]
            target_depth = target_depth[use]
            d4rt_local = xyz_local[local_idx, idx]
            d4rt_ref = xyz_ref[local_idx, idx]

            loger_z = np.abs(loger_xyz[:, 2])
            loger_norm = np.linalg.norm(loger_xyz, axis=1)
            d4rt_local_z = np.abs(d4rt_local[:, 2])
            d4rt_ref_z = np.abs(d4rt_ref[:, 2])
            d4rt_local_norm = np.linalg.norm(d4rt_local, axis=1)
            d4rt_ref_norm = np.linalg.norm(d4rt_ref, axis=1)

            loger_z_median = _median_positive(loger_z)
            loger_norm_median = _median_positive(loger_norm)
            d4rt_local_z_median = _median_positive(d4rt_local_z)
            d4rt_ref_z_median = _median_positive(d4rt_ref_z)
            d4rt_local_norm_median = _median_positive(d4rt_local_norm)
            d4rt_ref_norm_median = _median_positive(d4rt_ref_norm)
            target_depth_median = _median_positive(target_depth)

            row: dict[str, Any] = {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "ref_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                "frame_id": int(frame_id),
                "local_idx": int(local_idx),
                "loger_frame_index": int(loger_idx),
                "anchor_count": int(idx.shape[0]),
                "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                "loger_z_median": loger_z_median,
                "loger_norm_median": loger_norm_median,
                "loger_confidence_median": _median_positive(loger_conf),
                "d4rt_local_z_median": d4rt_local_z_median,
                "d4rt_ref_z_median": d4rt_ref_z_median,
                "d4rt_local_norm_median": d4rt_local_norm_median,
                "d4rt_ref_norm_median": d4rt_ref_norm_median,
                "target_depth_median": target_depth_median,
                "loger_z_over_d4rt_local_z_median": _safe_ratio(loger_z_median, d4rt_local_z_median),
                "loger_z_over_d4rt_ref_z_median": _safe_ratio(loger_z_median, d4rt_ref_z_median),
                "loger_norm_over_d4rt_local_norm_median": _safe_ratio(loger_norm_median, d4rt_local_norm_median),
                "loger_norm_over_d4rt_ref_norm_median": _safe_ratio(loger_norm_median, d4rt_ref_norm_median),
                "d4rt_local_z_over_loger_z_median": _safe_ratio(d4rt_local_z_median, loger_z_median),
                "scannet_depth_over_loger_z_median": _safe_ratio(target_depth_median, loger_z_median),
                "scannet_depth_over_d4rt_local_z_median": _safe_ratio(target_depth_median, d4rt_local_z_median),
            }
            for key in _CANDIDATE_KEYS:
                row[f"{key}_abs_rel_vs_eval_scale"] = _candidate_error(row.get(key), row.get("eval_ref0_depth_scale"))
            frame_rows.append(row)
        window_rows.append(
            {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "status": "ok",
                "frame_row_count": int(len(frame_rows) - row_count_before),
                "loger_frame_count": int(local_points.shape[0]),
                "loger_frame_ids": loger_frame_ids,
                **eval_diag,
            }
        )
    return frame_rows, window_rows


def _scene_summary(scene: str, window_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_values = _finite_values(window_rows, "ref0_pose_scale")
    eval_scale = float(np.median(eval_values)) if eval_values.size else None
    summary: dict[str, Any] = {
        "scene": scene,
        "status": "ok" if frame_rows else "no_frame_rows",
        "window_count": int(len(window_rows)),
        "frame_count": int(len(frame_rows)),
        "eval_ref0_depth_scale": eval_scale,
    }
    for key in [
        "anchor_count",
        "loger_z_median",
        "d4rt_local_z_median",
        "target_depth_median",
        *_CANDIDATE_KEYS,
    ]:
        values = _finite_values(frame_rows, key)
        summary[f"{key}_median"] = float(np.median(values)) if values.size else None
        summary[f"{key}_mean"] = float(np.mean(values)) if values.size else None
    for key in _CANDIDATE_KEYS:
        value = summary.get(f"{key}_median")
        summary[f"{key}_median_abs_rel_vs_eval_scale"] = _candidate_error(value, eval_scale)
    return summary


def _candidate_error_rows(scene_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in _CANDIDATE_KEYS:
        errors = _finite_values(scene_summaries, f"{key}_median_abs_rel_vs_eval_scale")
        values = _finite_values(scene_summaries, f"{key}_median")
        if errors.size == 0:
            continue
        rows.append(
            {
                "candidate": key,
                "uses_scannet_depth_for_proxy": bool(key not in _NO_GT_PROXY_KEYS),
                "scene_count": int(errors.size),
                "mean_abs_rel_vs_eval_scale": float(np.mean(errors)),
                "median_abs_rel_vs_eval_scale": float(np.median(errors)),
                "max_abs_rel_vs_eval_scale": float(np.max(errors)),
                "mean_candidate_scale": float(np.mean(values)) if values.size else None,
            }
        )
    rows.sort(key=lambda row: (bool(row["uses_scannet_depth_for_proxy"]), float(row["mean_abs_rel_vs_eval_scale"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6f}"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _write_md(path: Path, scene_summaries: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines: list[str] = [
        "# v22.14 LoGeR geometry scale-proxy diagnostic",
        "",
        "Diagnostic-only: samples LoGeR local pointmap depth at D4RT target UVs and compares candidate scale proxies against R23 eval-only ref0 depth scale.",
        "",
        "LoGeR proxy candidates do not use ScanNet depth; rows explicitly marked `uses_scannet_depth_for_proxy=True` are positive-control diagnostics only.",
        "",
        "## Metadata",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key in sorted(metadata.keys()):
        lines.append(f"| {key} | {_fmt(metadata[key])} |")
    lines.extend(
        [
            "",
            "## Candidate Error",
            "",
            "| candidate | uses ScanNet depth for proxy | mean absrel vs eval scale | median absrel | max absrel | mean candidate | scenes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("candidate")),
                    _fmt(row.get("uses_scannet_depth_for_proxy")),
                    _fmt(row.get("mean_abs_rel_vs_eval_scale")),
                    _fmt(row.get("median_abs_rel_vs_eval_scale")),
                    _fmt(row.get("max_abs_rel_vs_eval_scale")),
                    _fmt(row.get("mean_candidate_scale")),
                    _fmt(row.get("scene_count")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-scene",
            "",
            "| scene | eval scale | loger/local z | loger/ref z | local/loger z | GT/local z | GT/loger z | frames |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scene_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scene")),
                    _fmt(row.get("eval_ref0_depth_scale")),
                    _fmt(row.get("loger_z_over_d4rt_local_z_median_median")),
                    _fmt(row.get("loger_z_over_d4rt_ref_z_median_median")),
                    _fmt(row.get("d4rt_local_z_over_loger_z_median_median")),
                    _fmt(row.get("scannet_depth_over_d4rt_local_z_median_median")),
                    _fmt(row.get("scannet_depth_over_loger_z_median_median")),
                    _fmt(row.get("frame_count")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose LoGeR pointmap depth as a non-GT scale proxy for D4RT.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--loger-output-root", default="outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke")
    parser.add_argument("--loger-template", default="loger_{scene_short}_4f.pt")
    parser.add_argument("--audit-root", default="outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]

    provider = D4RTCarrierProjectionProvider(
        debug_root=args.cache_root,
        mode="raw",
        max_anchors=int(args.max_anchors),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    missing_loger: list[str] = []
    for scene in scenes:
        loger_path = _loger_path(Path(args.loger_output_root), args.loger_template, scene)
        if not loger_path.exists():
            missing_loger.append(scene)
            scene_summaries.append({"scene": scene, "status": "missing_loger_pt", "loger_path": str(loger_path)})
            continue
        scene_dir = Path(args.cache_root) / scene
        if not scene_dir.exists():
            scene_summaries.append({"scene": scene, "status": "missing_d4rt_cache", "loger_path": str(loger_path)})
            continue
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        payload = _load_loger_output(loger_path)
        frame_rows, window_rows = _diagnose_scene(
            scene,
            stream=stream,
            provider=provider,
            loger_payload=payload,
            frame_stride=int(args.frame_stride),
            max_frames=int(args.max_frames) if args.max_frames is not None else None,
            max_windows_per_scene=args.max_windows_per_scene,
            max_anchors=int(args.max_anchors),
            robust_trim_percentile=float(args.robust_trim_percentile),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
        )
        for row in window_rows:
            row["loger_path"] = str(loger_path)
        all_frame_rows.extend(frame_rows)
        all_window_rows.extend(window_rows)
        scene_summaries.append(_scene_summary(scene, window_rows, frame_rows))

    candidate_rows = _candidate_error_rows(scene_summaries)
    metadata = {
        "is_diagnostic_only": True,
        "loger_proxy_uses_scannet_depth": False,
        "evaluation_uses_scannet_depth_or_pose": True,
        "missing_loger_scene_count": int(len(missing_loger)),
        "missing_loger_scenes": ",".join(missing_loger) if missing_loger else "",
        "scene_count_requested": int(len(scenes)),
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames) if args.max_frames is not None else None,
        "max_windows_per_scene": args.max_windows_per_scene,
    }

    _write_csv(audit_root / "loger_scale_proxy_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "loger_scale_proxy_window_rows.csv", all_window_rows)
    _write_csv(audit_root / "loger_scale_proxy_scene_summary.csv", scene_summaries)
    _write_csv(audit_root / "loger_scale_proxy_candidate_errors.csv", candidate_rows)
    (audit_root / "loger_scale_proxy_frame_rows.json").write_text(json.dumps(_json_safe(all_frame_rows), indent=2), encoding="utf-8")
    (audit_root / "loger_scale_proxy_window_rows.json").write_text(json.dumps(_json_safe(all_window_rows), indent=2), encoding="utf-8")
    (audit_root / "loger_scale_proxy_scene_summary.json").write_text(json.dumps(_json_safe(scene_summaries), indent=2), encoding="utf-8")
    (audit_root / "loger_scale_proxy_candidate_errors.json").write_text(json.dumps(_json_safe(candidate_rows), indent=2), encoding="utf-8")
    (audit_root / "loger_scale_proxy_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    _write_md(audit_root / "loger_scale_proxy.md", scene_summaries, candidate_rows, metadata)
    print(f"Wrote v22.14 LoGeR scale-proxy diagnostic to {audit_root}")


if __name__ == "__main__":
    main()
