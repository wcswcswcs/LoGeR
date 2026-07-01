#!/usr/bin/env python3
"""Apply method-safe D4RT overlap-stitch to v97 Phase2 micro-track xyz.

This is intentionally not a visualization/eval alignment step.  It estimates
chunk-to-chunk Sim3 only from D4RT overlap carriers and rewrites the Phase2
micro-track 3D coordinates for downstream affinity features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.self_stitch import fit_sim3_with_diagnostics, match_overlap_carriers
from stream4d_native.sim3 import Sim3Transform, apply_sim3_to_xyz, compose_sim3


PHASE_ID = "v97_phase2_overlap_stitch_micro_tracks"
RUN_ID = "v97_phase2_overlap_stitch_micro_tracks"
DEFAULT_IN = ROOT / "outputs/audit/v97_phase2_d4rt_micro_tracks_overlap48_48clip_cap1024_gpu6"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase2_d4rt_micro_tracks_overlap48_48clip_cap1024_stitched"


@dataclass
class BatchRecord:
    decode_variant: str
    scene_id: str
    window_id: str
    path: Path
    frame_ids: list[int]
    transform_to_method: Sim3Transform


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _rel(value)
    return value


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _batch_dict(path: Path) -> dict[str, np.ndarray | list[int]]:
    with np.load(path, allow_pickle=True) as payload:
        frame_ids = [int(v) for v in np.asarray(payload["frame_ids"], dtype=np.int64).reshape(-1).tolist()]
        return {
            "frame_ids": frame_ids,
            "xyz": np.asarray(payload["xyz_ref"], dtype=np.float32),
            "uv": np.asarray(payload["uv_pred"], dtype=np.float32),
            "valid": np.asarray(payload["valid"], dtype=bool),
            "visibility": np.asarray(payload["visibility_prob"], dtype=np.float32),
            "confidence": np.asarray(payload["confidence_prob"], dtype=np.float32),
            "carrier_id": np.asarray(payload["carrier_id"], dtype=np.int64),
            "src_frame_global": np.asarray(payload["src_frame_global"], dtype=np.int64),
            "src_xy": np.asarray(payload["src_xy"], dtype=np.int64),
        }


def _discover_batches(root: Path, decode_variants: set[str]) -> dict[tuple[str, str], list[BatchRecord]]:
    base = root / "carrier_batches"
    if not base.exists():
        raise FileNotFoundError(f"missing carrier_batches under {root}")
    grouped: dict[tuple[str, str], list[BatchRecord]] = {}
    for path in sorted(base.glob("*/*/*.npz")):
        decode_variant = path.parent.parent.name
        if decode_variants and decode_variant not in decode_variants:
            continue
        scene_id = path.parent.name
        window_id = path.stem
        with np.load(path, allow_pickle=True) as payload:
            frame_ids = [int(v) for v in np.asarray(payload["frame_ids"], dtype=np.int64).reshape(-1).tolist()]
        grouped.setdefault((decode_variant, scene_id), []).append(
            BatchRecord(
                decode_variant=decode_variant,
                scene_id=scene_id,
                window_id=window_id,
                path=path,
                frame_ids=frame_ids,
                transform_to_method=Sim3Transform(
                    scale=1.0,
                    rot=np.eye(3, dtype=np.float64),
                    trans=np.zeros((3,), dtype=np.float64),
                ),
            )
        )
    for records in grouped.values():
        records.sort(key=lambda item: (min(item.frame_ids), max(item.frame_ids), item.window_id))
    if not grouped:
        raise RuntimeError(f"no carrier batch npz files selected under {base}")
    return grouped


def _fit_curr_to_prev(prev: BatchRecord, curr: BatchRecord, args: argparse.Namespace) -> dict[str, Any]:
    prev_payload = _batch_dict(prev.path)
    curr_payload = _batch_dict(curr.path)
    match = match_overlap_carriers(
        prev_payload,
        curr_payload,
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        uv_radius=float(args.uv_radius),
        max_matches_per_frame=int(args.max_matches_per_frame),
    )
    source = match.curr_xyz.reshape(-1, 3)
    target = match.prev_xyz.reshape(-1, 3)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < int(args.min_overlap_anchors):
        raise RuntimeError(
            f"not enough overlap anchors for {prev.window_id}->{curr.window_id}: "
            f"{source.shape[0]} < {args.min_overlap_anchors}; stats={match.stats}"
        )
    first = fit_sim3_with_diagnostics(source, target)
    residual = np.asarray(first["residual"], dtype=np.float64)
    kept = np.ones((source.shape[0],), dtype=bool)
    trim = float(args.fit_trim_percentile)
    if 0.0 < trim < 100.0 and source.shape[0] >= 16:
        kept = residual <= float(np.percentile(residual, trim))
        if int(np.count_nonzero(kept)) >= int(args.min_overlap_anchors) and int(np.count_nonzero(kept)) < source.shape[0]:
            fit = fit_sim3_with_diagnostics(source[kept], target[kept])
        else:
            fit = first
            kept = np.ones((source.shape[0],), dtype=bool)
    else:
        fit = first
    transform = Sim3Transform(
        scale=float(fit["scale"]),
        rot=np.asarray(fit["rot"], dtype=np.float64),
        trans=np.asarray(fit["trans"], dtype=np.float64),
    )
    return {
        "transform": transform,
        "row": {
            **match.stats,
            "fit_anchor_count": int(source.shape[0]),
            "fit_kept_anchor_count": int(np.count_nonzero(kept)),
            "fit_trim_percentile": trim,
            "scale_curr_to_prev": float(transform.scale),
            "rotation_det_curr_to_prev": float(np.linalg.det(transform.rot)),
            "translation_norm_curr_to_prev": float(np.linalg.norm(transform.trans)),
            "residual_median_curr_to_prev": fit.get("residual_median"),
            "residual_p90_curr_to_prev": fit.get("residual_p90"),
            "residual_p95_curr_to_prev": fit.get("residual_p95"),
            "inlier_ratio_abs005_curr_to_prev": fit.get("inlier_ratio_abs005"),
            "inlier_ratio_abs010_curr_to_prev": fit.get("inlier_ratio_abs010"),
        },
    }


def _fit_transforms(grouped: dict[tuple[str, str], list[BatchRecord]], args: argparse.Namespace) -> list[dict[str, Any]]:
    stitch_rows: list[dict[str, Any]] = []
    for (decode_variant, scene_id), records in sorted(grouped.items()):
        if not records:
            continue
        records[0].transform_to_method = Sim3Transform(
            scale=1.0,
            rot=np.eye(3, dtype=np.float64),
            trans=np.zeros((3,), dtype=np.float64),
        )
        for prev, curr in zip(records[:-1], records[1:]):
            fit = _fit_curr_to_prev(prev, curr, args)
            curr.transform_to_method = compose_sim3(fit["transform"], prev.transform_to_method)
            stitch_rows.append(
                {
                    "schema_version": "stream4d_v97_phase2_overlap_stitch_row_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "decode_variant": decode_variant,
                    "scene_id": scene_id,
                    "prev_window_id": prev.window_id,
                    "curr_window_id": curr.window_id,
                    "prev_frame_start": min(prev.frame_ids),
                    "prev_frame_end": max(prev.frame_ids),
                    "curr_frame_start": min(curr.frame_ids),
                    "curr_frame_end": max(curr.frame_ids),
                    **fit["row"],
                    "transform_scale_to_method": float(curr.transform_to_method.scale),
                    "transform_trans_norm_to_method": float(np.linalg.norm(curr.transform_to_method.trans)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return stitch_rows


def _copy_sidecars(input_root: Path, output_root: Path) -> None:
    names = [
        "micro_query_rows.csv",
        "micro_track_quality_rows.csv",
        "micro_occupancy_rows.csv",
        "decode_group_rows.csv",
        "decode_error_rows.csv",
        "d4rt_quality_rows.csv",
        "variant_config_rows.csv",
        "variant_metric_rows.csv",
        "variant_gate_rows.csv",
        "phase2_gate_rows.csv",
        "variant_failure_rows.csv",
        "casebook_rows.csv",
        "best_variant_summary.json",
    ]
    for name in names:
        src = input_root / name
        if src.exists():
            shutil.copy2(src, output_root / name)


def _rewrite_micro_tracks(input_root: Path, output_root: Path, grouped: dict[tuple[str, str], list[BatchRecord]]) -> dict[str, Any]:
    transform_by_key: dict[tuple[str, str, str], Sim3Transform] = {}
    for (decode_variant, scene_id), records in grouped.items():
        for rec in records:
            transform_by_key[(decode_variant, scene_id, rec.window_id)] = rec.transform_to_method
    src = input_root / "micro_track_rows.csv"
    dst = output_root / "micro_track_rows.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    output_root.mkdir(parents=True, exist_ok=True)
    row_count = 0
    stitched_count = 0
    missing_transform_count = 0
    with src.open(newline="", encoding="utf-8") as in_handle, dst.open("w", newline="", encoding="utf-8") as out_handle:
        reader = csv.DictReader(in_handle)
        fields = list(reader.fieldnames or [])
        extra = [
            "x_3d_raw",
            "y_3d_raw",
            "z_3d_raw",
            "overlap_stitch_applied",
            "overlap_stitch_transform_scale_to_method",
            "overlap_stitch_transform_trans_norm_to_method",
            "geometry_coordinate_mode",
        ]
        for field in extra:
            if field not in fields:
                fields.append(field)
        writer = csv.DictWriter(out_handle, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            row_count += 1
            key = (row.get("decode_variant", ""), row.get("scene_id", ""), row.get("window_id", ""))
            transform = transform_by_key.get(key)
            row["x_3d_raw"] = row.get("x_3d", "")
            row["y_3d_raw"] = row.get("y_3d", "")
            row["z_3d_raw"] = row.get("z_3d", "")
            row["geometry_coordinate_mode"] = "d4rt_overlap_self_stitched_no_final_gt_sim3"
            if transform is None:
                row["overlap_stitch_applied"] = False
                row["overlap_stitch_transform_scale_to_method"] = ""
                row["overlap_stitch_transform_trans_norm_to_method"] = ""
                missing_transform_count += 1
            else:
                xyz = np.asarray([[_num(row.get("x_3d")), _num(row.get("y_3d")), _num(row.get("z_3d"))]], dtype=np.float32)
                stitched = apply_sim3_to_xyz(xyz, transform=transform).reshape(3)
                row["x_3d"] = float(stitched[0])
                row["y_3d"] = float(stitched[1])
                row["z_3d"] = float(stitched[2])
                row["overlap_stitch_applied"] = True
                row["overlap_stitch_transform_scale_to_method"] = float(transform.scale)
                row["overlap_stitch_transform_trans_norm_to_method"] = float(np.linalg.norm(transform.trans))
                stitched_count += 1
            writer.writerow({field: row.get(field, "") for field in fields})
    return {
        "micro_track_rows": _rel(dst),
        "micro_track_row_count": int(row_count),
        "stitched_track_row_count": int(stitched_count),
        "missing_transform_track_row_count": int(missing_transform_count),
    }


def _write_stitch_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    input_root = _project(args.input_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    decode_variants = {part.strip() for part in args.decode_variants.split(",") if part.strip()}
    grouped = _discover_batches(input_root, decode_variants)
    stitch_rows = _fit_transforms(grouped, args)
    _copy_sidecars(input_root, output_root)
    track_stats = _rewrite_micro_tracks(input_root, output_root, grouped)
    stitch_path = output_root / "overlap_stitch_rows.csv"
    _write_stitch_rows(stitch_path, stitch_rows)
    source_summary = _read_json(input_root / "summary.json")
    all_required_edges = sum(max(0, len(records) - 1) for records in grouped.values())
    passed = (
        int(track_stats["missing_transform_track_row_count"]) == 0
        and len(stitch_rows) == all_required_edges
        and all(int(row.get("fit_kept_anchor_count", 0)) >= int(args.min_overlap_anchors) for row in stitch_rows)
    )
    summary = dict(source_summary)
    summary.update(
        {
            "schema": "stream4d_v97_phase2_overlap_stitch_micro_tracks_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_phase2_root": _rel(input_root),
            "output_root": _rel(output_root),
            "decision": "PASS_V97_PHASE2_OVERLAP_STITCH_MICRO_TRACKS" if passed else "NO_GO_V97_PHASE2_OVERLAP_STITCH_MICRO_TRACKS",
            "can_enter_phase3": bool(passed and bool(source_summary.get("can_enter_phase3", True))),
            "method_geometry_policy": "D4RT-only overlap self-stitch; final GT Sim3 is not applied.",
            "geometry_coordinate_mode": "d4rt_overlap_self_stitched_no_final_gt_sim3",
            "d4rt_applies_overlap_stitch": True,
            "d4rt_applies_final_gt_sim3": False,
            "overlap_stitch_rows": _rel(stitch_path),
            "overlap_stitch_edge_count": int(len(stitch_rows)),
            "required_overlap_stitch_edge_count": int(all_required_edges),
            "min_overlap_anchors": int(args.min_overlap_anchors),
            "min_visibility": float(args.min_visibility),
            "min_confidence": float(args.min_confidence),
            "uv_radius": float(args.uv_radius),
            "fit_trim_percentile": float(args.fit_trim_percentile),
            "runtime_overlap_stitch_sec": float(time.time() - started),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            **track_stats,
        }
    )
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "output_root": _rel(output_root),
                "overlap_stitch_edge_count": len(stitch_rows),
                "micro_track_row_count": track_stats["micro_track_row_count"],
                "missing_transform_track_row_count": track_stats["missing_transform_track_row_count"],
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_IN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--decode-variants", default="D3_adaptive1024")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--uv-radius", type=float, default=0.002)
    parser.add_argument("--max-matches-per-frame", type=int, default=4096)
    parser.add_argument("--fit-trim-percentile", type=float, default=90.0)
    parser.add_argument("--min-overlap-anchors", type=int, default=16)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
