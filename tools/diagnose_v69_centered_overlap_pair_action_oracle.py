#!/usr/bin/env python3
"""Centered overlap-pair action oracle for v69 transform-origin diagnostics.

This diagnostic follows the v69 plan's 16.3/16.4 repair direction. It keeps the
same materialized overlap pairs and metric gates as the v67 point-pair oracle,
but compares the original origin-linear damping against centered Sim3
application modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

try:
    from diagnose_v67_overlap_pair_action_oracle import (
        _fit_pair_correction,
        _load_label_to_id,
        _load_pair,
        _parse_filter_spec,
        _parse_source,
        _safe_tag,
        _select_fit_points,
    )
    from diagnose_v67_boundary_sim3_action_oracle import (
        _best_mechanism_improvement,
        _rotation_delta_deg,
        _rotation_power,
        _rmse_dist,
        _safe_improvement_ratio,
        _target_chunks,
    )
    from diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )
except ImportError:  # pragma: no cover
    from tools.diagnose_v67_overlap_pair_action_oracle import (
        _fit_pair_correction,
        _load_label_to_id,
        _load_pair,
        _parse_filter_spec,
        _parse_source,
        _safe_tag,
        _select_fit_points,
    )
    from tools.diagnose_v67_boundary_sim3_action_oracle import (
        _best_mechanism_improvement,
        _rotation_delta_deg,
        _rotation_power,
        _rmse_dist,
        _safe_improvement_ratio,
        _target_chunks,
    )
    from tools.diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )


CENTER_MODES = (
    "origin_linear",
    "fit_centroid_interp",
    "current_chunk_first_pose",
    "current_chunk_mean_pose",
)


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


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


def _make_baseline_row(
    frames: np.ndarray,
    poses: np.ndarray,
    gt_pos: np.ndarray,
    trace: Dict[int, Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    head_len: int,
) -> Dict[str, Any]:
    scales = [float(trace[c]["scale"]) for c in trace]
    controller_rows = [
        {
            "chunk_idx": int(c),
            "action": "baseline",
            "source_scale": float(trace[c]["scale"]),
            "ctrl_scale": float(trace[c]["scale"]),
            "is_estimated_boundary": str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform",
        }
        for c in sorted(trace)
    ]
    return _metric_result_row(
        "BASELINE_SOURCE_NOOP",
        frames,
        poses,
        gt_pos,
        None,
        controller_rows,
        sum(abs(s - 1.0) > 1e-6 for s in scales),
        any(abs(s - 1.0) > 1e-6 for s in scales),
        chunk_size,
        chunk_overlap,
        head_len,
    )


def _pose_centers_for_chunk(
    frames: np.ndarray,
    poses: np.ndarray,
    frame_to_chunk: Mapping[int, int],
    chunk_idx: int,
) -> np.ndarray:
    centers = [
        poses[i, :3, 3].astype(np.float64)
        for i, frame in enumerate(frames)
        if int(frame_to_chunk.get(int(frame), -999999)) == int(chunk_idx)
    ]
    if not centers:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(centers, axis=0)


def _center_for_mode(
    mode: str,
    curr_points: np.ndarray,
    frames: np.ndarray,
    poses: np.ndarray,
    frame_to_chunk: Mapping[int, int],
    curr_chunk: int,
) -> Optional[np.ndarray]:
    if mode in {"origin_linear", "fit_centroid_interp"}:
        return np.mean(curr_points, axis=0).astype(np.float64)
    centers = _pose_centers_for_chunk(frames, poses, frame_to_chunk, curr_chunk)
    if centers.shape[0] == 0:
        return None
    if mode == "current_chunk_first_pose":
        return centers[0]
    if mode == "current_chunk_mean_pose":
        return np.mean(centers, axis=0)
    raise ValueError(f"unsupported center mode: {mode}")


def _transform_points(
    points: np.ndarray,
    *,
    center_mode: str,
    alpha: float,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    full_scale: float,
    full_rot: np.ndarray,
    full_trans: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    if center_mode == "origin_linear":
        return float(scale) * (rot @ points.T).T + trans
    full_center = float(full_scale) * (full_rot @ center) + full_trans
    center_after = center + float(alpha) * (full_center - center)
    return center_after[None, :] + float(scale) * (rot @ (points - center[None, :]).T).T


def _apply_centered_correction(
    frames: np.ndarray,
    poses: np.ndarray,
    frame_to_chunk: Mapping[int, int],
    target_chunks: Sequence[int],
    *,
    center_mode: str,
    alpha: float,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    full_scale: float,
    full_rot: np.ndarray,
    full_trans: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    out = poses.copy()
    targets = {int(c) for c in target_chunks}
    for idx, frame in enumerate(frames):
        chunk_idx = frame_to_chunk.get(int(frame))
        if chunk_idx not in targets:
            continue
        point = out[idx, :3, 3][None, :]
        out[idx, :3, 3] = _transform_points(
            point,
            center_mode=center_mode,
            alpha=alpha,
            scale=scale,
            rot=rot,
            trans=trans,
            full_scale=full_scale,
            full_rot=full_rot,
            full_trans=full_trans,
            center=center,
        )[0]
        out[idx, :3, :3] = rot @ out[idx, :3, :3]
    return out


def _boundary_jump(
    frames: np.ndarray,
    poses: np.ndarray,
    frame_a: int,
    frame_b: int,
) -> float:
    index = {int(frame): i for i, frame in enumerate(frames)}
    if frame_a not in index or frame_b not in index:
        return float("nan")
    return float(np.linalg.norm(poses[index[frame_b], :3, 3] - poses[index[frame_a], :3, 3]))


def _tail100_delta(
    frames: np.ndarray,
    poses: np.ndarray,
    baseline_poses: np.ndarray,
    gt_pos: np.ndarray,
) -> float:
    valid = (frames >= 0) & (frames < gt_pos.shape[0])
    idx = np.where(valid)[0]
    if idx.size < 3:
        return float("nan")
    idx = idx[-min(100, idx.size):]
    cand = poses[idx, :3, 3]
    base = baseline_poses[idx, :3, 3]
    gt = gt_pos[frames[idx]]
    cand_err = np.linalg.norm(cand - gt, axis=1)
    base_err = np.linalg.norm(base - gt, axis=1)
    return float(np.sqrt(np.mean(cand_err * cand_err)) - np.sqrt(np.mean(base_err * base_err)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=_parse_source, required=True)
    parser.add_argument("--overlap-pairs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    parser.add_argument("--max-fit-points", type=int, default=20000)
    parser.add_argument("--semantic-full-pt", type=Path, default=Path("results/kitti_preprocess/01/sparse_masklets_with_semantic.pt"))
    parser.add_argument("--fit-semantic-filter", action="append", default=None)
    parser.add_argument("--semantic-min-conf", type=float, default=0.0)
    parser.add_argument("--min-filter-fit-points", type=int, default=256)
    parser.add_argument("--max-safe-rotation-deg", type=float, default=2.0)
    parser.add_argument("--max-safe-overlap-displacement-m", type=float, default=0.5)
    parser.add_argument("--max-safe-log-scale-delta", type=float, default=0.03)
    parser.add_argument("--min-raw-overlap-improvement-ratio", type=float, default=0.20)
    parser.add_argument("--min-mechanism-improvement-ratio", type=float, default=0.10)
    parser.add_argument("--max-ate-regression-m", type=float, default=0.30)
    parser.add_argument("--damping-alpha", type=float, action="append", default=None)
    parser.add_argument("--center-mode", action="append", choices=CENTER_MODES, default=None)
    args = parser.parse_args()

    source_label, run_dir = args.source
    pair_files = sorted(args.overlap_pairs_dir.glob("chunk_*_*.pt"))
    if not pair_files:
        raise FileNotFoundError(f"No overlap pair files in {args.overlap_pairs_dir}")
    trace = _load_trace(run_dir / "merge_state_trace.jsonl")
    frames, poses, frame_to_chunk = _load_postmerge_trajectory(run_dir / "postmerge_global_pose.jsonl")
    _, _, gt_pos = _load_kitti_gt(args.gt)
    baseline = _make_baseline_row(frames, poses, gt_pos, trace, args.chunk_size, args.chunk_overlap, args.head_len)
    damping_alphas = args.damping_alpha if args.damping_alpha is not None else [1.0, 0.5, 0.25]
    center_modes = args.center_mode if args.center_mode is not None else list(CENTER_MODES)
    scopes = ["current", "boundary_span", "future"]
    actions = [("SE3_PAIR", False), ("SIM3_PAIR", True)]
    label_to_id = _load_label_to_id(args.semantic_full_pt)
    fit_filter_specs = args.fit_semantic_filter if args.fit_semantic_filter is not None else ["all"]
    fit_filters = [_parse_filter_spec(spec, label_to_id) for spec in fit_filter_specs]
    rows: List[Dict[str, Any]] = []
    fit_failures: List[Dict[str, Any]] = []

    for pair_file in pair_files:
        pair = _load_pair(pair_file)
        curr_chunk = int(pair.get("curr_chunk"))
        prev_chunk = int(pair.get("prev_chunk", curr_chunk - 1))
        for fit_filter_name, fit_label_ids in fit_filters:
            prev_points, curr_points, valid_count, filter_fit_point_count, filter_reason = _select_fit_points(
                pair,
                int(args.max_fit_points),
                label_ids=fit_label_ids,
                semantic_min_conf=float(args.semantic_min_conf),
                min_filter_fit_points=int(args.min_filter_fit_points),
            )
            raw_before = _rmse_dist(prev_points, curr_points)
            for action_name, with_scale in actions:
                full_scale, full_rot, full_trans, fail_reason = _fit_pair_correction(
                    prev_points,
                    curr_points,
                    with_scale=with_scale,
                )
                if filter_reason is not None:
                    fail_reason = filter_reason if fail_reason is None else f"{filter_reason};{fail_reason}"
                if fail_reason is not None or full_scale is None or full_rot is None or full_trans is None:
                    fit_failures.append({
                        "pair_file": str(pair_file),
                        "prev_chunk": int(prev_chunk),
                        "curr_chunk": int(curr_chunk),
                        "fit_semantic_filter": fit_filter_name,
                        "fit_semantic_label_ids": sorted(fit_label_ids) if fit_label_ids is not None else [],
                        "semantic_min_conf": float(args.semantic_min_conf),
                        "action_family": action_name,
                        "valid_pair_count": int(valid_count),
                        "filter_fit_point_count": int(filter_fit_point_count),
                        "fit_failure": fail_reason,
                    })
                    continue
                for damping_alpha in damping_alphas:
                    scale = float(full_scale) ** float(damping_alpha) if with_scale else 1.0
                    rot = _rotation_power(full_rot, float(damping_alpha))
                    trans = float(damping_alpha) * full_trans
                    rot_deg = _rotation_delta_deg(rot)
                    abs_log_scale = abs(math.log(max(float(scale), 1e-12)))
                    safe_rotation = rot_deg <= float(args.max_safe_rotation_deg)
                    safe_scale = abs_log_scale <= float(args.max_safe_log_scale_delta)
                    alpha_tag = str(damping_alpha).replace(".", "p")
                    damped_action_name = f"{action_name}_A{alpha_tag}"
                    for center_mode in center_modes:
                        center = _center_for_mode(
                            center_mode,
                            curr_points,
                            frames,
                            poses,
                            frame_to_chunk,
                            curr_chunk,
                        )
                        if center is None:
                            fit_failures.append({
                                "pair_file": str(pair_file),
                                "prev_chunk": int(prev_chunk),
                                "curr_chunk": int(curr_chunk),
                                "fit_semantic_filter": fit_filter_name,
                                "semantic_min_conf": float(args.semantic_min_conf),
                                "action_family": action_name,
                                "center_mode": center_mode,
                                "fit_failure": "missing_center",
                            })
                            continue
                        corrected = _transform_points(
                            curr_points,
                            center_mode=center_mode,
                            alpha=float(damping_alpha),
                            scale=scale,
                            rot=rot,
                            trans=trans,
                            full_scale=float(full_scale),
                            full_rot=full_rot,
                            full_trans=full_trans,
                            center=center,
                        )
                        raw_after = _rmse_dist(prev_points, corrected)
                        raw_improvement = _safe_improvement_ratio(raw_before, raw_after)
                        overlap_displacement = _rmse_dist(curr_points, corrected)
                        safe_displacement = overlap_displacement <= float(args.max_safe_overlap_displacement_m)
                        safe_correction = bool(safe_rotation and safe_displacement and (safe_scale or not with_scale))
                        for scope in scopes:
                            targets = _target_chunks(trace, curr_chunk, scope)
                            candidate_poses = _apply_centered_correction(
                                frames,
                                poses,
                                frame_to_chunk,
                                targets,
                                center_mode=center_mode,
                                alpha=float(damping_alpha),
                                scale=scale,
                                rot=rot,
                                trans=trans,
                                full_scale=float(full_scale),
                                full_rot=full_rot,
                                full_trans=full_trans,
                                center=center,
                            )
                            controller_rows = [
                                {
                                    "chunk_idx": int(c),
                                    "action": "centered_overlap_pair_oracle" if c in targets else "baseline",
                                    "source_scale": float(trace[c]["scale"]),
                                    "ctrl_scale": float(trace[c]["scale"]),
                                    "is_estimated_boundary": str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform",
                                }
                                for c in sorted(trace)
                            ]
                            candidate_name = f"{damped_action_name}_{scope}_b{curr_chunk}_center_{_safe_tag(center_mode)}"
                            metric = _metric_result_row(
                                candidate_name,
                                frames,
                                candidate_poses,
                                gt_pos,
                                baseline,
                                controller_rows,
                                int(baseline.get("source_nonunit_scale_count", 0)),
                                bool(baseline.get("source_has_scale_state", False)),
                                args.chunk_size,
                                args.chunk_overlap,
                                args.head_len,
                            )
                            if fit_filter_name != "all" or len(fit_filters) > 1:
                                metric["candidate"] = f"{metric.get('candidate')}_fit_{_safe_tag(fit_filter_name)}"
                            best_mech = _best_mechanism_improvement(metric)
                            ate_delta = _float(metric.get("delta_vs_baseline_global_ate"))
                            raw_support_pass = bool(
                                math.isfinite(raw_improvement)
                                and raw_improvement >= float(args.min_raw_overlap_improvement_ratio)
                            )
                            mechanism_pass = bool(
                                math.isfinite(best_mech)
                                and best_mech >= float(args.min_mechanism_improvement_ratio)
                            )
                            ate_guard_pass = bool(
                                math.isfinite(ate_delta)
                                and ate_delta <= float(args.max_ate_regression_m)
                            )
                            gate_pass = bool(raw_support_pass and mechanism_pass and ate_guard_pass and safe_correction)
                            trace_row = trace.get(curr_chunk, {}).get("row", {})
                            emitted = [int(x) for x in trace_row.get("emitted_frame_ids", [])]
                            entry_frame = emitted[0] if emitted else int(trace_row.get("start_frame", -1))
                            exit_frame = emitted[-1] if emitted else int(trace_row.get("end_frame", -1)) - 1
                            boundary_jump_before = _boundary_jump(frames, poses, entry_frame - 1, entry_frame)
                            boundary_jump_after = _boundary_jump(frames, candidate_poses, entry_frame - 1, entry_frame)
                            metric.update({
                                "source_label": source_label,
                                "source_run": str(run_dir),
                                "overlap_pair_file": str(pair_file),
                                "prev_chunk": int(prev_chunk),
                                "curr_chunk": int(curr_chunk),
                                "fit_semantic_filter": fit_filter_name,
                                "fit_semantic_label_ids": sorted(fit_label_ids) if fit_label_ids is not None else [],
                                "semantic_min_conf": float(args.semantic_min_conf),
                                "filter_fit_point_count": int(filter_fit_point_count),
                                "filter_reason": filter_reason or "",
                                "action_family": action_name,
                                "damped_action_family": damped_action_name,
                                "damping_alpha": float(damping_alpha),
                                "center_mode": center_mode,
                                "center_x": float(center[0]),
                                "center_y": float(center[1]),
                                "center_z": float(center[2]),
                                "scope": scope,
                                "target_chunk_count": int(len(targets)),
                                "target_chunks_first": int(targets[0]) if targets else None,
                                "target_chunks_last": int(targets[-1]) if targets else None,
                                "fit_point_count": int(prev_points.shape[0]),
                                "valid_pair_count": int(valid_count),
                                "raw_overlap_before_m": raw_before,
                                "raw_overlap_after_m": raw_after,
                                "raw_overlap_improvement_ratio": raw_improvement,
                                "correction_overlap_displacement_m": overlap_displacement,
                                "correction_scale": float(scale),
                                "correction_abs_log_scale_delta": abs_log_scale,
                                "correction_rotation_deg": rot_deg,
                                "correction_translation_norm_m": float(np.linalg.norm(trans)),
                                "safe_rotation_pass": safe_rotation,
                                "safe_overlap_displacement_pass": safe_displacement,
                                "safe_scale_pass": safe_scale if with_scale else True,
                                "safe_correction_pass": safe_correction,
                                "best_mechanism_improvement": best_mech,
                                "raw_support_pass": raw_support_pass,
                                "mechanism_pass": mechanism_pass,
                                "ate_guard_pass": ate_guard_pass,
                                "oracle_action_gate_pass": gate_pass,
                                "boundary_jump_before_m": boundary_jump_before,
                                "boundary_jump_after_m": boundary_jump_after,
                                "boundary_jump_delta_m": (
                                    boundary_jump_after - boundary_jump_before
                                    if math.isfinite(boundary_jump_before) and math.isfinite(boundary_jump_after)
                                    else float("nan")
                                ),
                                "tail100_rmse_delta_raw_pose_m": _tail100_delta(frames, candidate_poses, poses, gt_pos),
                            })
                            rows.append(metric)

    rows.sort(key=lambda row: (
        not bool(row.get("oracle_action_gate_pass")),
        -_float(row.get("best_mechanism_improvement")),
        -_float(row.get("raw_overlap_improvement_ratio")),
        _float(row.get("delta_vs_baseline_global_ate")),
        str(row.get("candidate")),
    ))
    gate_rows = [row for row in rows if bool(row.get("oracle_action_gate_pass"))]
    gate_chunks = sorted({int(row.get("curr_chunk")) for row in gate_rows})
    future_pass_chunks = sorted({
        int(row.get("curr_chunk"))
        for row in rows
        if bool(row.get("raw_support_pass"))
        and bool(row.get("safe_correction_pass"))
        and bool(row.get("ate_guard_pass"))
        and _float(row.get("future_after_overlap_mean_improvement_vs_baseline")) >= float(args.min_mechanism_improvement_ratio)
    })
    headtail_pass_chunks = sorted({
        int(row.get("curr_chunk"))
        for row in rows
        if bool(row.get("raw_support_pass"))
        and bool(row.get("safe_correction_pass"))
        and bool(row.get("ate_guard_pass"))
        and _float(row.get("head_to_tail_transfer_ratio_mean_improvement_vs_baseline")) >= float(args.min_mechanism_improvement_ratio)
    })
    summary = {
        "schema": "acl2_v69_centered_overlap_pair_action_oracle_summary_v1",
        "source_label": source_label,
        "source_run": str(run_dir),
        "overlap_pairs_dir": str(args.overlap_pairs_dir),
        "center_modes": center_modes,
        "pair_files": len(pair_files),
        "rows": len(rows),
        "fit_failures": fit_failures,
        "counts": {
            "safe_correction_pass": sum(bool(row.get("safe_correction_pass")) for row in rows),
            "raw_support_pass": sum(bool(row.get("raw_support_pass")) for row in rows),
            "mechanism_pass": sum(bool(row.get("mechanism_pass")) for row in rows),
            "ate_guard_pass": sum(bool(row.get("ate_guard_pass")) for row in rows),
            "oracle_action_gate_pass": len(gate_rows),
        },
        "gate_chunks": gate_chunks,
        "future_pass_chunks": future_pass_chunks,
        "headtail_pass_chunks": headtail_pass_chunks,
        "center_mode_gate_chunks": {
            mode: sorted({int(row.get("curr_chunk")) for row in gate_rows if row.get("center_mode") == mode})
            for mode in center_modes
        },
        "center_mode_best_rows": {
            mode: next((row for row in rows if row.get("center_mode") == mode), {})
            for mode in center_modes
        },
        "gate_rule": {
            "semantic_full_pt": str(args.semantic_full_pt),
            "fit_semantic_filters": [name for name, _ in fit_filters],
            "semantic_min_conf": float(args.semantic_min_conf),
            "min_filter_fit_points": int(args.min_filter_fit_points),
            "min_raw_overlap_improvement_ratio": float(args.min_raw_overlap_improvement_ratio),
            "min_mechanism_improvement_ratio": float(args.min_mechanism_improvement_ratio),
            "max_ate_regression_m": float(args.max_ate_regression_m),
            "max_safe_rotation_deg": float(args.max_safe_rotation_deg),
            "max_safe_overlap_displacement_m": float(args.max_safe_overlap_displacement_m),
            "max_safe_log_scale_delta": float(args.max_safe_log_scale_delta),
            "damping_alphas": [float(x) for x in damping_alphas],
        },
        "best_row": rows[0] if rows else {},
        "mean_raw_overlap_improvement_ratio": _finite_mean(row.get("raw_overlap_improvement_ratio") for row in rows),
        "mean_best_mechanism_improvement": _finite_mean(row.get("best_mechanism_improvement") for row in rows),
        "oracle_action_gate_pass": bool(gate_rows),
        "decision": "diagnostic_only",
        "note": (
            "Centered application tests transform-origin/scale-center hypotheses only. "
            "It is not an online controller or semantic-anchor success claim."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "centered_overlap_pair_action_oracle_results.csv", rows)
    (args.out_dir / "centered_overlap_pair_action_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
