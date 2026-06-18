#!/usr/bin/env python3
"""Offline boundary SE(3)/Sim(3) action oracle for ACL2 v67 failure path.

This is a diagnostic-only follow-up to Phase O3 failure mode 18.3. It asks
whether raw premerge overlap pose support justifies adding rotation/translation
damping after scale-only boundary hold failed the action-sensitive checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )
    from kitti_trajectory_diagnostics import _umeyama_sim3
except ImportError:  # pragma: no cover
    from tools.diagnose_v67_offline_scale_controller import (
        DEFAULT_GT,
        _load_kitti_gt,
        _load_postmerge_trajectory,
        _load_trace,
        _metric_result_row,
    )
    from tools.kitti_trajectory_diagnostics import _umeyama_sim3


def _parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be LABEL=RUN_DIR")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--source label must be non-empty")
    return label, Path(path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _safe_improvement_ratio(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(candidate) or abs(baseline) < 1e-12:
        return float("nan")
    return float((baseline - candidate) / abs(baseline))


def _rotation_delta_deg(rot: np.ndarray) -> float:
    value = float((np.trace(rot) - 1.0) / 2.0)
    value = max(-1.0, min(1.0, value))
    return float(math.degrees(math.acos(value)))


def _rotation_power(rot: np.ndarray, alpha: float) -> np.ndarray:
    value = float((np.trace(rot) - 1.0) / 2.0)
    value = max(-1.0, min(1.0, value))
    angle = math.acos(value)
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1],
    ], dtype=np.float64) / (2.0 * sin_angle)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis /= norm
    a = float(alpha) * angle
    kx, ky, kz = axis
    k = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def _rmse_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=1)
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")


def _load_premerge_windows(run_dir: Path) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    path = run_dir / "premerge_local_pose.jsonl"
    if not path.is_file():
        return out
    for row in _read_jsonl(path):
        chunk_idx = int(row.get("chunk_idx", row.get("local_chunk_idx", len(out))))
        poses = np.asarray(row.get("camera_poses", []), dtype=np.float64)
        if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
            out[chunk_idx] = poses
    return out


def _transform_local_positions(local_poses: np.ndarray, scale: float, mat: np.ndarray) -> np.ndarray:
    rot = np.asarray(mat[:3, :3], dtype=np.float64)
    trans = np.asarray(mat[:3, 3], dtype=np.float64)
    local_t = np.asarray(local_poses[:, :3, 3], dtype=np.float64)
    return (float(scale) * (rot @ local_t.T)).T + trans


def _overlap_positions(
    trace: Dict[int, Dict[str, Any]],
    premerge: Dict[int, np.ndarray],
    boundary: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    prev = int(boundary) - 1
    if prev not in trace or boundary not in trace or prev not in premerge or boundary not in premerge:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), 0
    overlap = int(trace[boundary].get("row", {}).get("overlap_size", 0) or 0)
    if overlap <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), 0
    prev_poses = premerge[prev]
    curr_poses = premerge[boundary]
    actual = min(overlap, prev_poses.shape[0], curr_poses.shape[0])
    if actual <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), 0
    prev_pos = _transform_local_positions(
        prev_poses[-actual:],
        float(trace[prev]["scale"]),
        np.asarray(trace[prev]["matrix"], dtype=np.float64),
    )
    curr_pos = _transform_local_positions(
        curr_poses[:actual],
        float(trace[boundary]["scale"]),
        np.asarray(trace[boundary]["matrix"], dtype=np.float64),
    )
    return prev_pos, curr_pos, actual


def _fit_overlap_correction(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    *,
    with_scale: bool,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if prev_pos.shape != curr_pos.shape or prev_pos.shape[0] < 3:
        return None, None, None, "not_enough_overlap_points"
    try:
        scale, rot, trans = _umeyama_sim3(curr_pos, prev_pos, with_scale=with_scale)
    except Exception as exc:  # noqa: BLE001 - diagnostic should record fit failure.
        return None, None, None, f"fit_error:{type(exc).__name__}:{exc}"
    if not np.isfinite(scale) or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
        return None, None, None, "fit_nonfinite"
    return float(scale), rot, trans, None


def _target_chunks(trace: Dict[int, Dict[str, Any]], boundary: int, scope: str) -> List[int]:
    chunks = sorted(trace)
    if scope == "current":
        return [int(boundary)]
    if scope == "future":
        return [c for c in chunks if c >= boundary]
    if scope == "boundary_span":
        estimated_after = [
            c for c in chunks
            if c > boundary and str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform"
        ]
        stop = min(estimated_after) if estimated_after else max(chunks) + 1
        return [c for c in chunks if boundary <= c < stop]
    raise ValueError(f"Unsupported scope={scope!r}")


def _apply_global_correction(
    frames: np.ndarray,
    poses: np.ndarray,
    frame_to_chunk: Dict[int, int],
    target_chunks: Sequence[int],
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    out = poses.copy()
    targets = {int(c) for c in target_chunks}
    for idx, frame in enumerate(frames):
        chunk_idx = frame_to_chunk.get(int(frame))
        if chunk_idx not in targets:
            continue
        out[idx, :3, 3] = float(scale) * (rot @ out[idx, :3, 3]) + trans
        out[idx, :3, :3] = rot @ out[idx, :3, :3]
    return out


def _step_distance(frames: np.ndarray, poses: np.ndarray, frame_a: int, frame_b: int) -> float:
    index = {int(frame): i for i, frame in enumerate(frames)}
    if frame_a not in index or frame_b not in index:
        return float("nan")
    return float(np.linalg.norm(poses[index[frame_b], :3, 3] - poses[index[frame_a], :3, 3]))


def _boundary_frames(trace: Dict[int, Dict[str, Any]], boundary: int) -> Tuple[int, int]:
    trace_row = trace[boundary].get("row", {})
    emitted = [int(x) for x in trace_row.get("emitted_frame_ids", [])]
    if emitted:
        return int(emitted[0]), int(emitted[-1])
    return int(trace_row.get("start_frame", -1)), int(trace_row.get("end_frame", -1)) - 1


def _best_mechanism_improvement(row: Dict[str, Any]) -> float:
    values = [
        _float(row.get("head_to_tail_transfer_ratio_mean_improvement_vs_baseline")),
        _float(row.get("future_after_overlap_mean_improvement_vs_baseline")),
        _float(row.get("intra_scale_variance_mean_improvement_vs_baseline")),
    ]
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    parser.add_argument("--max-safe-rotation-deg", type=float, default=2.0)
    parser.add_argument("--max-safe-overlap-displacement-m", type=float, default=0.5)
    parser.add_argument("--max-safe-log-scale-delta", type=float, default=0.03)
    parser.add_argument("--min-raw-overlap-improvement-ratio", type=float, default=0.20)
    parser.add_argument("--min-mechanism-improvement-ratio", type=float, default=0.10)
    parser.add_argument("--max-ate-regression-m", type=float, default=0.30)
    parser.add_argument(
        "--damping-alpha",
        type=float,
        action="append",
        default=None,
        help="Correction damping factors to test. Defaults to 1.0, 0.5, 0.25.",
    )
    args = parser.parse_args()

    _, _, gt_pos = _load_kitti_gt(args.gt)
    rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    missing_inputs: List[Dict[str, Any]] = []
    fit_failures: List[Dict[str, Any]] = []
    scopes = ["current", "boundary_span", "future"]
    actions = [("SE3_OVERLAP", False), ("SIM3_OVERLAP", True)]
    damping_alphas = args.damping_alpha if args.damping_alpha is not None else [1.0, 0.5, 0.25]

    for source_label, run_dir in args.source:
        trace_path = run_dir / "merge_state_trace.jsonl"
        pose_path = run_dir / "postmerge_global_pose.jsonl"
        premerge_path = run_dir / "premerge_local_pose.jsonl"
        if not trace_path.is_file() or not pose_path.is_file() or not premerge_path.is_file():
            missing_inputs.append({
                "source_label": source_label,
                "source_run": str(run_dir),
                "missing_trace": not trace_path.is_file(),
                "missing_postmerge_pose": not pose_path.is_file(),
                "missing_premerge_pose": not premerge_path.is_file(),
            })
            continue

        trace = _load_trace(trace_path)
        frames, poses, frame_to_chunk = _load_postmerge_trajectory(pose_path)
        premerge = _load_premerge_windows(run_dir)
        baseline = _make_baseline_row(
            frames,
            poses,
            gt_pos,
            trace,
            args.chunk_size,
            args.chunk_overlap,
            args.head_len,
        )
        baseline.update({"source_label": source_label, "source_run": str(run_dir)})
        baseline_rows.append(baseline)
        boundary_chunks = [
            c for c in sorted(trace)
            if str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform"
        ]

        for boundary in boundary_chunks:
            prev_pos, curr_pos, overlap_n = _overlap_positions(trace, premerge, boundary)
            raw_before = _rmse_dist(prev_pos, curr_pos)
            entry_frame, exit_frame = _boundary_frames(trace, boundary)
            base_entry_jump = _step_distance(frames, poses, entry_frame - 1, entry_frame)
            base_exit_jump = _step_distance(frames, poses, exit_frame, exit_frame + 1)
            for action_name, with_scale in actions:
                full_scale, full_rot, full_trans, fail_reason = _fit_overlap_correction(
                    prev_pos,
                    curr_pos,
                    with_scale=with_scale,
                )
                if fail_reason is not None or full_scale is None or full_rot is None or full_trans is None:
                    fit_failures.append({
                        "source_label": source_label,
                        "source_run": str(run_dir),
                        "held_boundary_chunk": int(boundary),
                        "action_family": action_name,
                        "overlap_n": int(overlap_n),
                        "fit_failure": fail_reason,
                    })
                    continue
                for damping_alpha in damping_alphas:
                    scale = float(full_scale) ** float(damping_alpha) if with_scale else 1.0
                    rot = _rotation_power(full_rot, float(damping_alpha))
                    trans = float(damping_alpha) * full_trans
                    corrected_overlap = float(scale) * (rot @ curr_pos.T).T + trans
                    raw_after = _rmse_dist(prev_pos, corrected_overlap)
                    raw_improvement = _safe_improvement_ratio(raw_before, raw_after)
                    overlap_displacement = _rmse_dist(curr_pos, corrected_overlap)
                    rot_deg = _rotation_delta_deg(rot)
                    abs_log_scale = abs(math.log(max(float(scale), 1e-12)))
                    safe_rotation = rot_deg <= float(args.max_safe_rotation_deg)
                    safe_displacement = overlap_displacement <= float(args.max_safe_overlap_displacement_m)
                    safe_scale = abs_log_scale <= float(args.max_safe_log_scale_delta)
                    safe_correction = bool(safe_rotation and safe_displacement and (safe_scale or not with_scale))

                    alpha_tag = str(damping_alpha).replace(".", "p")
                    damped_action_name = f"{action_name}_A{alpha_tag}"
                    for scope in scopes:
                        targets = _target_chunks(trace, boundary, scope)
                        candidate_poses = _apply_global_correction(
                            frames,
                            poses,
                            frame_to_chunk,
                            targets,
                            float(scale),
                            rot,
                            trans,
                        )
                        controller_rows = [
                            {
                                "chunk_idx": int(c),
                                "action": "boundary_sim3_oracle" if c in targets else "baseline",
                                "source_scale": float(trace[c]["scale"]),
                                "ctrl_scale": float(trace[c]["scale"]),
                                "is_estimated_boundary": str(trace[c].get("row", {}).get("transform_reason", "")) == "estimated_overlap_transform",
                            }
                            for c in sorted(trace)
                        ]
                        metric = _metric_result_row(
                            f"{damped_action_name}_{scope}_b{boundary}",
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
                        cand_entry_jump = _step_distance(frames, candidate_poses, entry_frame - 1, entry_frame)
                        cand_exit_jump = _step_distance(frames, candidate_poses, exit_frame, exit_frame + 1)
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
                        ate_guard_pass = bool(math.isfinite(ate_delta) and ate_delta <= float(args.max_ate_regression_m))
                        gate_pass = bool(raw_support_pass and mechanism_pass and ate_guard_pass and safe_correction)
                        metric.update({
                            "source_label": source_label,
                            "source_run": str(run_dir),
                            "held_boundary_chunk": int(boundary),
                            "action_family": action_name,
                            "damped_action_family": damped_action_name,
                            "damping_alpha": float(damping_alpha),
                            "scope": scope,
                            "target_chunk_count": int(len(targets)),
                            "target_chunks_first": int(targets[0]) if targets else None,
                            "target_chunks_last": int(targets[-1]) if targets else None,
                            "overlap_n": int(overlap_n),
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
                            "baseline_entry_jump_m": base_entry_jump,
                            "candidate_entry_jump_m": cand_entry_jump,
                            "entry_jump_improvement_m": (
                                float(base_entry_jump - cand_entry_jump)
                                if math.isfinite(base_entry_jump) and math.isfinite(cand_entry_jump)
                                else float("nan")
                            ),
                            "baseline_exit_jump_m": base_exit_jump,
                            "candidate_exit_jump_m": cand_exit_jump,
                            "exit_jump_improvement_m": (
                                float(base_exit_jump - cand_exit_jump)
                                if math.isfinite(base_exit_jump) and math.isfinite(cand_exit_jump)
                                else float("nan")
                            ),
                            "best_mechanism_improvement": best_mech,
                            "raw_support_pass": raw_support_pass,
                            "mechanism_pass": mechanism_pass,
                            "ate_guard_pass": ate_guard_pass,
                            "oracle_action_gate_pass": gate_pass,
                            "gate_rule": (
                                "raw_overlap_improvement>=20%, best mechanism improvement>=10%, "
                                "ATE regression<=0.3m, rotation<=2deg, overlap displacement<=0.5m, "
                                "and abs log scale delta<=0.03 for Sim3"
                            ),
                        })
                        rows.append(metric)

    rows.sort(key=lambda row: (
        not bool(row.get("oracle_action_gate_pass")),
        -_float(row.get("best_mechanism_improvement")),
        -_float(row.get("raw_overlap_improvement_ratio")),
        _float(row.get("delta_vs_baseline_global_ate")),
        str(row.get("source_label")),
        str(row.get("candidate")),
    ))
    gate_rows = [row for row in rows if bool(row.get("oracle_action_gate_pass"))]
    safe_rows = [row for row in rows if bool(row.get("safe_correction_pass"))]
    raw_support_rows = [row for row in rows if bool(row.get("raw_support_pass"))]
    mechanism_rows = [row for row in rows if bool(row.get("mechanism_pass"))]
    ate_guard_rows = [row for row in rows if bool(row.get("ate_guard_pass"))]
    best_row = rows[0] if rows else {}
    summary = {
        "sources": [{"label": label, "run_dir": str(path)} for label, path in args.source],
        "rows": len(rows),
        "baseline_rows": len(baseline_rows),
        "missing_inputs": missing_inputs,
        "fit_failures": fit_failures,
        "overlap_pose_support_note": "Uses only premerge overlap camera poses; current KITTI01 reset boundaries have overlap_n=3, so this is weak diagnostic support, not point-pair proof.",
        "gate_rule": {
            "min_raw_overlap_improvement_ratio": float(args.min_raw_overlap_improvement_ratio),
            "min_mechanism_improvement_ratio": float(args.min_mechanism_improvement_ratio),
            "max_ate_regression_m": float(args.max_ate_regression_m),
            "max_safe_rotation_deg": float(args.max_safe_rotation_deg),
            "max_safe_overlap_displacement_m": float(args.max_safe_overlap_displacement_m),
            "max_safe_log_scale_delta": float(args.max_safe_log_scale_delta),
            "damping_alphas": [float(x) for x in damping_alphas],
        },
        "counts": {
            "safe_correction_pass": len(safe_rows),
            "raw_support_pass": len(raw_support_rows),
            "mechanism_pass": len(mechanism_rows),
            "ate_guard_pass": len(ate_guard_rows),
            "oracle_action_gate_pass": len(gate_rows),
        },
        "best_row": {k: v for k, v in best_row.items() if k != "gate_rule"},
        "mean_raw_overlap_improvement_ratio": _finite_mean(row.get("raw_overlap_improvement_ratio") for row in rows),
        "mean_best_mechanism_improvement": _finite_mean(row.get("best_mechanism_improvement") for row in rows),
        "oracle_action_gate_pass": bool(gate_rows),
        "note": "Diagnostic-only oracle. Passing this would justify implementing a controlled action and selector; it is not itself a semantic method result.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "boundary_sim3_action_oracle_results.csv", rows)
    _write_csv(args.out_dir / "baseline_rows.csv", baseline_rows)
    (args.out_dir / "boundary_sim3_action_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
