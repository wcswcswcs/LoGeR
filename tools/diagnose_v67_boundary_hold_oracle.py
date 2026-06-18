#!/usr/bin/env python3
"""Single reset-boundary hold oracle for ACL2 v67 O3 diagnostics.

For each source run and each estimated reset-boundary chunk, this tool replaces
only that boundary chunk's scale with the first chunk scale and evaluates the
trajectory. It is posthoc attribution only, not a deployable controller.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from diagnose_v67_offline_scale_controller import (
    DEFAULT_GT,
    _apply_scale_control,
    _float,
    _load_kitti_gt,
    _load_observability,
    _load_postmerge_trajectory,
    _load_trace,
    _metric_result_row,
)


def _parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be LABEL=RUN_DIR")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--source label must be non-empty")
    return label, Path(path)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _origin_modes(value: str) -> List[str]:
    if value == "both":
        return ["transform_translation", "first_pose"]
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--observability-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--origin-mode", choices=["transform_translation", "first_pose", "both"], default="both")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    args = parser.parse_args()

    _, _, gt_pos = _load_kitti_gt(args.gt)
    obs = _load_observability(args.observability_csv)
    rows: List[Dict[str, Any]] = []
    missing_inputs: List[Dict[str, Any]] = []

    for label, run_dir in args.source:
        trace_path = run_dir / "merge_state_trace.jsonl"
        pose_path = run_dir / "postmerge_global_pose.jsonl"
        if not trace_path.is_file() or not pose_path.is_file():
            missing_inputs.append({
                "source_label": label,
                "source_run": str(run_dir),
                "missing_trace": not trace_path.is_file(),
                "missing_pose": not pose_path.is_file(),
            })
            continue

        trace = _load_trace(trace_path)
        frames, poses, frame_to_chunk = _load_postmerge_trajectory(pose_path)
        source_scales = {c: float(trace[c]["scale"]) for c in trace}
        source_has_scale_state = any(abs(s - 1.0) > 1e-6 for s in source_scales.values())
        source_nonunit_count = sum(abs(s - 1.0) > 1e-6 for s in source_scales.values())
        baseline_rows = [
            {
                "chunk_idx": c,
                "source_scale": source_scales[c],
                "ctrl_scale": source_scales[c],
                "action": "baseline",
            }
            for c in sorted(trace)
        ]
        baseline = _metric_result_row(
            "BASELINE_SOURCE_NOOP",
            frames,
            poses,
            gt_pos,
            None,
            baseline_rows,
            source_nonunit_count,
            source_has_scale_state,
            args.chunk_size,
            args.chunk_overlap,
            args.head_len,
        )
        boundary_chunks = [
            c for c in sorted(trace)
            if str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform"
        ]
        first_scale = source_scales[min(trace)]

        for origin_mode in _origin_modes(args.origin_mode):
            for boundary in boundary_chunks:
                ctrl_scales = dict(source_scales)
                ctrl_scales[boundary] = first_scale
                candidate_poses = _apply_scale_control(
                    frames,
                    poses,
                    frame_to_chunk,
                    trace,
                    ctrl_scales,
                    origin_mode=origin_mode,
                )
                controller_rows = [
                    {
                        "chunk_idx": c,
                        "source_scale": source_scales[c],
                        "ctrl_scale": ctrl_scales[c],
                        "action": "hold" if c == boundary else "baseline",
                    }
                    for c in sorted(trace)
                ]
                metric = _metric_result_row(
                    f"hold_only_{boundary}",
                    frames,
                    candidate_poses,
                    gt_pos,
                    baseline,
                    controller_rows,
                    source_nonunit_count,
                    source_has_scale_state,
                    args.chunk_size,
                    args.chunk_overlap,
                    args.head_len,
                )
                obs_row = obs.get(boundary, {})
                metric.update({
                    "source_label": label,
                    "source_run": str(run_dir),
                    "origin_mode": origin_mode,
                    "held_boundary_chunk": int(boundary),
                    "baseline_global_ate": baseline.get("global_ate"),
                    "boundary_source_scale": source_scales[boundary],
                    "first_chunk_scale": first_scale,
                    "boundary_Q_scale": _float(obs_row.get("Q_scale")),
                    "boundary_Q_scale_smoothed": _float(obs_row.get("Q_scale_smoothed")),
                    "boundary_anchor_type": obs_row.get("anchor_type"),
                    "boundary_straight_road_anchor_sparse": obs_row.get("straight_road_anchor_sparse"),
                    "boundary_geometry_confidence_mean": _float(obs_row.get("geometry_confidence_mean")),
                    "boundary_condition_mean": _float(obs_row.get("condition_score")),
                    "boundary_vertical_mean": _float(obs_row.get("vertical_static_total_ratio")),
                    "boundary_road_mean": _float(obs_row.get("road_plane_dominance")),
                    "boundary_vegetation_mean": _float(obs_row.get("vegetation_ratio")),
                    "boundary_dynamic_mean": _float(obs_row.get("dynamic_ratio")),
                    "boundary_future_err_mean": _float(obs_row.get("future_after_overlap_error")),
                    "boundary_headtail_mean": _float(obs_row.get("head_to_tail_transfer_ratio")),
                    "boundary_H35_gap_mean": _float(obs_row.get("H35_minus_C9_gap")),
                })
                rows.append(metric)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "single_boundary_hold_results.csv", rows)
    summary = {
        "sources": [{"label": label, "run_dir": str(path)} for label, path in args.source],
        "origin_mode": args.origin_mode,
        "rows": len(rows),
        "missing_inputs": missing_inputs,
        "out_csv": str(args.out_dir / "single_boundary_hold_results.csv"),
        "note": "Posthoc attribution oracle. Do not treat as method success without a pre-registered selector.",
    }
    (args.out_dir / "single_boundary_hold_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
