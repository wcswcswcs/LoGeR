#!/usr/bin/env python3
"""Phase 1 scale drift ledger for ACL2 v73."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from v73_semantic_memory_common import (
    TARGET_CHUNKS,
    finite_quantile,
    load_jsonl,
    parse_chunks,
    rotation_deg_from_trace,
    safe_float,
    utc_now,
    write_csv,
    write_json,
)

try:
    from diagnose_acl2_v67_segments import DEFAULT_GT, _chunk_rows
    from diagnose_v67_offline_scale_controller import _load_postmerge_trajectory
    from kitti_trajectory_diagnostics import _load_kitti_gt, _umeyama_sim3
except ImportError:  # pragma: no cover
    from tools.diagnose_acl2_v67_segments import DEFAULT_GT, _chunk_rows
    from tools.diagnose_v67_offline_scale_controller import _load_postmerge_trajectory
    from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _umeyama_sim3


DEFAULT_OUT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final/phase1_scale_drift_ledger")
DEFAULT_H35 = Path(
    "results/kitti01_hmc_v2/acl2_v67_dense_semantic_reconstruction/"
    "phaseO2_h35_trace_geom_merge_full/rollouts/V67S_H35_TRACE_GEOM_MERGE_FULL_H35_PARITY"
)


def _trace_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        chunk = int(row.get("chunk_idx", row.get("local_chunk_idx", len(rows))))
        rows[chunk] = row
    return rows


def _global_aligned_errors(frames: np.ndarray, poses: np.ndarray, gt_pos: np.ndarray) -> dict[int, float]:
    valid = (frames >= 0) & (frames < gt_pos.shape[0])
    frames_v = frames[valid]
    pos = poses[valid, :3, 3]
    if frames_v.size < 3:
        return {}
    scale, rot, trans = _umeyama_sim3(pos, gt_pos[frames_v], with_scale=True)
    aligned = (scale * (rot @ pos.T)).T + trans
    errors = np.linalg.norm(aligned - gt_pos[frames_v], axis=1)
    return {int(frame): float(err) for frame, err in zip(frames_v, errors)}


def _rmse(vals: list[float]) -> float | None:
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    if not finite:
        return None
    arr = np.asarray(finite, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def _boundary_jump(frame_to_pose: dict[int, np.ndarray], end_frame: int) -> float | None:
    a = int(end_frame) - 1
    b = int(end_frame)
    if a not in frame_to_pose or b not in frame_to_pose:
        return None
    return float(np.linalg.norm(frame_to_pose[b][:3, 3] - frame_to_pose[a][:3, 3]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h35-run-dir", type=Path, default=DEFAULT_H35)
    parser.add_argument("--target-chunks", default=",".join(map(str, TARGET_CHUNKS)))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    args = parser.parse_args()

    _, _, gt_pos = _load_kitti_gt(args.gt)
    frames, poses, frame_to_chunk = _load_postmerge_trajectory(args.h35_run_dir / "postmerge_global_pose.jsonl")
    frame_to_pose = {int(frame): poses[i] for i, frame in enumerate(frames)}
    pred_pos = poses[:, :3, 3]
    segment_rows = _chunk_rows(
        args.h35_run_dir.name,
        frames,
        pred_pos,
        gt_pos,
        int(args.chunk_size),
        int(args.chunk_overlap),
        int(args.head_len),
    )
    segment_by_chunk = {int(row["chunk_idx"]): row for row in segment_rows}
    trace_by_chunk = _trace_by_chunk(args.h35_run_dir / "merge_state_trace.jsonl")
    global_errors = _global_aligned_errors(frames, poses, gt_pos)
    target_set = set(parse_chunks(args.target_chunks))
    all_rows: list[dict[str, Any]] = []
    for chunk_id in sorted(segment_by_chunk):
        seg = segment_by_chunk[chunk_id]
        trace = trace_by_chunk.get(chunk_id, {})
        start = int(seg.get("start", chunk_id * (args.chunk_size - args.chunk_overlap)))
        end = int(seg.get("end", start + args.chunk_size))
        chunk_err = [global_errors[f] for f in range(start, end) if f in global_errors]
        scales = [safe_float(seg.get(key)) for key in ("head_scale", "mid_scale", "tail_scale")]
        finite_scales = [float(x) for x in scales if x is not None]
        intra_var = float(np.var(finite_scales)) if finite_scales else None
        local_sim3 = safe_float(seg.get("whole_ate_rmse_m"))
        global_chunk_ate = _rmse(chunk_err)
        row = {
            "chunk_id": chunk_id,
            "is_target_chunk": chunk_id in target_set,
            "frame_start": start,
            "frame_end": end,
            "local_sim3_ate": local_sim3,
            "global_chunk_ate": global_chunk_ate,
            "oracle_gap": None if local_sim3 is None or global_chunk_ate is None else float(global_chunk_ate - local_sim3),
            "head_to_tail": safe_float(seg.get("head_to_tail_ate_rmse_m")),
            "scale_cv": safe_float(seg.get("scale_cv_head_mid_tail")),
            "intra_scale_variance": intra_var,
            "future_after_overlap": safe_float(seg.get("overlap_to_future_ate_rmse_m")),
            "boundary_jump": _boundary_jump(frame_to_pose, end),
            "reset_relative_index": int(chunk_id % int(trace.get("reset_every", 5) or 5)),
            "merge_transform_log_scale": None,
            "merge_transform_rotation_deg": rotation_deg_from_trace(trace.get("transform_rot_trace")),
            "merge_transform_translation_norm": safe_float(trace.get("transform_trans_norm")),
            "transform_reason": trace.get("transform_reason"),
            "transform_kind": trace.get("transform_kind"),
        }
        scale_value = safe_float(trace.get("transform_scale_value"))
        if scale_value is not None and scale_value > 0:
            row["merge_transform_log_scale"] = float(math.log(scale_value))
        all_rows.append(row)

    targets = [row for row in all_rows if row.get("is_target_chunk")]
    thresholds = {
        "head_to_tail_p75_target": finite_quantile((row.get("head_to_tail") for row in targets), 0.75),
        "scale_cv_p75_target": finite_quantile((row.get("scale_cv") for row in targets), 0.75),
        "future_after_overlap_p75_target": finite_quantile((row.get("future_after_overlap") for row in targets), 0.75),
        "boundary_jump_p75_target": finite_quantile((row.get("boundary_jump") for row in targets), 0.75),
        "oracle_gap_p75_target": finite_quantile((row.get("oracle_gap") for row in targets), 0.75),
    }
    for row in all_rows:
        head = safe_float(row.get("head_to_tail"))
        cv = safe_float(row.get("scale_cv"))
        future = safe_float(row.get("future_after_overlap"))
        boundary = safe_float(row.get("boundary_jump"))
        gap = safe_float(row.get("oracle_gap"))
        row["Y_short"] = (
            int((head is not None and thresholds["head_to_tail_p75_target"] is not None and head >= thresholds["head_to_tail_p75_target"])
                or (cv is not None and thresholds["scale_cv_p75_target"] is not None and cv >= thresholds["scale_cv_p75_target"]))
            if row.get("is_target_chunk") else None
        )
        row["Y_mid"] = (
            int((future is not None and thresholds["future_after_overlap_p75_target"] is not None and future >= thresholds["future_after_overlap_p75_target"])
                or (boundary is not None and thresholds["boundary_jump_p75_target"] is not None and boundary >= thresholds["boundary_jump_p75_target"]))
            if row.get("is_target_chunk") else None
        )
        row["Y_long"] = None
        row["Y_scale_drift"] = (
            int(bool(row.get("Y_short")) or bool(row.get("Y_mid"))
                or (gap is not None and thresholds["oracle_gap_p75_target"] is not None and gap >= thresholds["oracle_gap_p75_target"]))
            if row.get("is_target_chunk") else None
        )

    summary = {
        "schema": "acl2_v73_phase1_scale_drift_ledger_v1",
        "created_at": utc_now(),
        "h35_run_dir": str(args.h35_run_dir),
        "rows": len(all_rows),
        "target_rows": len(targets),
        "target_chunks": sorted(target_set),
        "thresholds": thresholds,
        "label_rule": "Target-chunk p75 high-risk labels; continuous metrics remain authoritative.",
        "Y_long_note": "No current v73 long-term future-scale label was derivable from trajectory alone; kept null until TTT action evidence exists.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "scale_drift_ledger.csv", all_rows)
    write_json(args.out_dir / "scale_drift_ledger.json", {"summary": summary, "rows": all_rows})
    print({"out_dir": str(args.out_dir), "rows": len(all_rows), "target_rows": len(targets)})


if __name__ == "__main__":
    main()
