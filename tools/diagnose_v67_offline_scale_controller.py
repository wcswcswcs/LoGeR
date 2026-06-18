#!/usr/bin/env python3
"""Offline scale-state controller diagnostic for ACL2 v67 Phase O3.

The tool is intentionally trajectory-only: it does not rerun HMC, and it refuses
to promote results when the source merge trace has no non-unit scale state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from kitti_trajectory_diagnostics import _load_kitti_gt, _mat_to_quat_xyzw, _umeyama_sim3
    from diagnose_acl2_v67_segments import _chunk_rows, _run_summary
except ImportError:  # pragma: no cover
    from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _mat_to_quat_xyzw, _umeyama_sim3
    from tools.diagnose_acl2_v67_segments import _chunk_rows, _run_summary


DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def _safe_ratio_improvement(baseline: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if baseline is None or candidate is None:
        return None
    if not math.isfinite(float(baseline)) or not math.isfinite(float(candidate)):
        return None
    if abs(float(baseline)) < 1e-12:
        return None
    return float((float(baseline) - float(candidate)) / abs(float(baseline)))


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


def _write_tum(path: Path, frames: np.ndarray, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = _mat_to_quat_xyzw(poses[:, :3, :3])
    t = poses[:, :3, 3]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for frame, tt, qq in zip(frames, t, q):
            handle.write(
                f"{float(frame):.6f} {tt[0]:.9f} {tt[1]:.9f} {tt[2]:.9f} "
                f"{qq[0]:.9f} {qq[1]:.9f} {qq[2]:.9f} {qq[3]:.9f}\n"
            )


def _load_postmerge_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    frame_to_pose: Dict[int, np.ndarray] = {}
    frame_to_chunk: Dict[int, int] = {}
    for row in _read_jsonl(path):
        frames = [int(x) for x in row.get("emitted_frame_ids", [])]
        poses = np.asarray(row.get("camera_poses", []), dtype=np.float64)
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(f"{path}: invalid camera_poses shape for chunk {row.get('chunk_idx')}: {poses.shape}")
        if len(frames) != poses.shape[0]:
            raise ValueError(
                f"{path}: emitted_frame_ids length {len(frames)} != pose count {poses.shape[0]} "
                f"for chunk {row.get('chunk_idx')}"
            )
        chunk_idx = int(row.get("chunk_idx", row.get("local_chunk_idx", -1)))
        for frame, pose in zip(frames, poses):
            frame_to_pose[int(frame)] = pose
            frame_to_chunk[int(frame)] = chunk_idx
    frames_np = np.asarray(sorted(frame_to_pose), dtype=np.int64)
    poses_np = np.stack([frame_to_pose[int(f)] for f in frames_np], axis=0)
    return frames_np, poses_np, frame_to_chunk


def _load_trace(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        chunk_idx = int(row.get("chunk_idx", row.get("local_chunk_idx", len(out))))
        mat = np.asarray(row.get("transform_matrix", np.eye(4)), dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"{path}: invalid transform_matrix for chunk {chunk_idx}: {mat.shape}")
        scale = _float(row.get("transform_scale_value"), 1.0)
        out[chunk_idx] = {
            "row": row,
            "scale": float(scale),
            "matrix": mat,
            "translation": mat[:3, 3].copy(),
        }
    return out


def _load_observability(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in _read_csv(path):
        chunk_idx = int(row.get("chunk_id", row.get("chunk_idx", len(out))))
        out[chunk_idx] = row
    return out


def _eval_global(
    frames: np.ndarray,
    poses: np.ndarray,
    gt_pos: np.ndarray,
) -> Dict[str, Any]:
    valid = (frames >= 0) & (frames < gt_pos.shape[0])
    frames_v = frames[valid]
    poses_v = poses[valid]
    if frames_v.size < 3:
        return {"global_ate": None, "FinalErr": None, "sim3_scale": None, "frame_count": int(frames_v.size)}
    pred_pos = poses_v[:, :3, 3]
    scale, rot, trans = _umeyama_sim3(pred_pos, gt_pos[frames_v], with_scale=True)
    aligned = (scale * (rot @ pred_pos.T)).T + trans
    errors = np.linalg.norm(aligned - gt_pos[frames_v], axis=1)
    return {
        "global_ate": float(np.sqrt(np.mean(errors * errors))),
        "FinalErr": float(errors[-1]),
        "sim3_scale": float(scale),
        "frame_count": int(frames_v.size),
    }


def _segment_summary(
    label: str,
    frames: np.ndarray,
    poses: np.ndarray,
    gt_pos: np.ndarray,
    chunk_size: int,
    chunk_overlap: int,
    head_len: int,
) -> Dict[str, Optional[float]]:
    rows = _chunk_rows(label, frames, poses[:, :3, 3], gt_pos, chunk_size, chunk_overlap, head_len)
    return {
        "local_sim3_mean": _finite_mean(r.get("whole_ate_rmse_m") for r in rows),
        "head_to_tail_transfer_ratio_mean": _finite_mean(r.get("head_to_tail_ate_rmse_m") for r in rows),
        "intra_scale_variance_mean": _finite_mean(r.get("scale_cv_head_mid_tail") for r in rows),
        "future_after_overlap_mean": _finite_mean(r.get("overlap_to_future_ate_rmse_m") for r in rows),
        "chunk_whole_ate_median": _run_summary(rows, "whole_ate_rmse_m").get("median"),
    }


def _category_q_values(base_q: Dict[int, float], seed: int) -> Dict[int, float]:
    chunks = sorted(base_q)
    low = sum(base_q[c] <= 0.35 for c in chunks)
    high = sum(base_q[c] >= 0.60 for c in chunks)
    rng = random.Random(seed)
    shuffled = chunks[:]
    rng.shuffle(shuffled)
    out = {c: 0.50 for c in chunks}
    for c in shuffled[:low]:
        out[c] = 0.20
    for c in shuffled[low:low + high]:
        out[c] = 0.70
    return out


def _controller_scales(
    trace: Dict[int, Dict[str, Any]],
    q_values: Dict[int, float],
    *,
    fixed_alpha: Optional[float] = None,
    alpha_multiplier: float = 1.0,
    boundary_only_estimated: bool = False,
    forced_actions: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    scales: Dict[int, float] = {}
    rows: List[Dict[str, Any]] = []
    ell_trust: Optional[float] = None
    active_ctrl_scale: Optional[float] = None
    for chunk_idx in sorted(trace):
        source_scale = max(float(trace[chunk_idx]["scale"]), 1e-12)
        ell_hat = math.log(source_scale)
        reason = str(trace[chunk_idx].get("row", {}).get("transform_reason", ""))
        is_estimated_boundary = reason == "estimated_overlap_transform"
        if ell_trust is None:
            ell_ctrl = ell_hat
            ell_trust = ell_ctrl
            action = "init"
            alpha = 1.0
            raw_delta = 0.0
            clipped_delta = 0.0
            active_ctrl_scale = float(math.exp(ell_ctrl))
        elif forced_actions is not None and forced_actions.get(chunk_idx) == "source_noop":
            ell_ctrl = ell_hat
            action = "source_noop"
            alpha = 0.0
            raw_delta = 0.0
            clipped_delta = 0.0
            active_ctrl_scale = float(math.exp(ell_ctrl))
        elif boundary_only_estimated and not is_estimated_boundary:
            scale_ctrl = float(active_ctrl_scale if active_ctrl_scale is not None else math.exp(ell_trust))
            scales[chunk_idx] = scale_ctrl
            rows.append({
                "chunk_idx": int(chunk_idx),
                "source_scale": float(source_scale),
                "ctrl_scale": scale_ctrl,
                "Q_used": float(q_values.get(chunk_idx, float("nan"))),
                "action": "reuse_boundary_ctrl",
                "alpha": 0.0,
                "raw_log_scale_delta": 0.0,
                "applied_log_scale_delta": 0.0,
                "ell_trust_after": float(ell_trust),
                "boundary_only_estimated": True,
                "transform_reason": reason,
                "is_estimated_boundary": False,
            })
            continue
        else:
            q = float(q_values.get(chunk_idx, 0.5))
            forced_action = forced_actions.get(chunk_idx) if forced_actions is not None else None
            if forced_action in {"hold", "refresh", "ambiguous"}:
                action = forced_action
                alpha = {"hold": 0.05, "refresh": 0.35, "ambiguous": 0.15}[forced_action]
            elif fixed_alpha is not None:
                alpha = float(fixed_alpha)
                action = "fixed_alpha"
            elif q <= 0.35:
                alpha = 0.05
                action = "hold"
            elif q >= 0.60:
                alpha = 0.35
                action = "refresh"
            else:
                alpha = 0.15
                action = "ambiguous"
            alpha *= float(alpha_multiplier)
            raw_delta = ell_hat - ell_trust
            clipped_delta = max(-0.03, min(0.03, alpha * raw_delta))
            ell_ctrl = ell_trust + clipped_delta
            if q >= 0.60:
                ell_trust = 0.7 * ell_trust + 0.3 * ell_ctrl
            elif q <= 0.35:
                ell_trust = ell_trust
            else:
                ell_trust = 0.9 * ell_trust + 0.1 * ell_ctrl
            active_ctrl_scale = float(math.exp(ell_ctrl))
        scale_ctrl = float(math.exp(ell_ctrl))
        scales[chunk_idx] = scale_ctrl
        rows.append({
            "chunk_idx": int(chunk_idx),
            "source_scale": float(source_scale),
            "ctrl_scale": scale_ctrl,
            "Q_used": float(q_values.get(chunk_idx, float("nan"))),
            "action": action,
            "alpha": float(alpha),
            "raw_log_scale_delta": float(raw_delta),
            "applied_log_scale_delta": float(clipped_delta),
            "ell_trust_after": float(ell_trust if ell_trust is not None else ell_ctrl),
            "boundary_only_estimated": bool(boundary_only_estimated),
            "transform_reason": reason,
            "is_estimated_boundary": bool(is_estimated_boundary),
        })
    return scales, rows


def _apply_scale_control(
    frames: np.ndarray,
    poses: np.ndarray,
    frame_to_chunk: Dict[int, int],
    trace: Dict[int, Dict[str, Any]],
    ctrl_scales: Dict[int, float],
    origin_mode: str,
) -> np.ndarray:
    out = poses.copy()
    first_pose_origin: Dict[int, np.ndarray] = {}
    if origin_mode == "first_pose":
        for i, frame in enumerate(frames):
            chunk_idx = frame_to_chunk.get(int(frame))
            if chunk_idx is not None and chunk_idx not in first_pose_origin:
                first_pose_origin[chunk_idx] = poses[i, :3, 3].copy()
    for i, frame in enumerate(frames):
        chunk_idx = frame_to_chunk.get(int(frame))
        if chunk_idx is None or chunk_idx not in trace:
            continue
        source_scale = max(float(trace[chunk_idx]["scale"]), 1e-12)
        ratio = float(ctrl_scales.get(chunk_idx, source_scale)) / source_scale
        if origin_mode == "first_pose":
            origin = first_pose_origin.get(chunk_idx, trace[chunk_idx]["translation"])
        elif origin_mode == "transform_translation":
            origin = trace[chunk_idx]["translation"]
        else:
            raise ValueError(f"Unsupported origin_mode={origin_mode!r}")
        out[i, :3, 3] = origin + ratio * (poses[i, :3, 3] - origin)
    return out


def _metric_result_row(
    candidate: str,
    frames: np.ndarray,
    poses: np.ndarray,
    gt_pos: np.ndarray,
    baseline: Optional[Dict[str, Any]],
    chunk_rows: Sequence[Dict[str, Any]],
    source_nonunit_scale_count: int,
    source_has_scale_state: bool,
    chunk_size: int,
    chunk_overlap: int,
    head_len: int,
) -> Dict[str, Any]:
    result = _eval_global(frames, poses, gt_pos)
    result.update(_segment_summary(candidate, frames, poses, gt_pos, chunk_size, chunk_overlap, head_len))
    result["candidate"] = candidate
    if baseline:
        result["delta_vs_baseline_global_ate"] = (
            None if result["global_ate"] is None or baseline.get("global_ate") is None
            else float(result["global_ate"] - baseline["global_ate"])
        )
        for key in ("head_to_tail_transfer_ratio_mean", "future_after_overlap_mean", "intra_scale_variance_mean"):
            result[f"{key}_improvement_vs_baseline"] = _safe_ratio_improvement(baseline.get(key), result.get(key))
    else:
        result["delta_vs_baseline_global_ate"] = 0.0
    result["source_nonunit_scale_count"] = int(source_nonunit_scale_count)
    result["source_has_scale_state"] = bool(source_has_scale_state)
    result["hold_chunks"] = sum(r.get("action") == "hold" for r in chunk_rows)
    result["refresh_chunks"] = sum(r.get("action") == "refresh" for r in chunk_rows)
    result["ambiguous_chunks"] = sum(r.get("action") == "ambiguous" for r in chunk_rows)
    result["reuse_boundary_ctrl_chunks"] = sum(r.get("action") == "reuse_boundary_ctrl" for r in chunk_rows)
    result["estimated_boundary_updates"] = sum(bool(r.get("is_estimated_boundary")) for r in chunk_rows)
    result["source_noop_chunks"] = sum(r.get("action") == "source_noop" for r in chunk_rows)
    result["accepted_chunks"] = sum(r.get("action") in {"init", "hold", "refresh", "ambiguous", "fixed_alpha"} for r in chunk_rows)
    deltas = [abs(math.log(max(_float(r.get("ctrl_scale"), 1.0), 1e-12)) - math.log(max(_float(r.get("source_scale"), 1.0), 1e-12))) for r in chunk_rows]
    result["mean_abs_scale_delta"] = float(np.mean(deltas)) if deltas else 0.0
    result["max_abs_scale_delta"] = float(np.max(deltas)) if deltas else 0.0
    return result


def _parse_official_ate(run_dir: Path) -> Optional[float]:
    path = run_dir / "results_sim3" / "results_ate.txt"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Average:"):
            parts = line.split()
            if len(parts) >= 2:
                return _float(parts[1], float("nan"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--observability-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--source-label", default="")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--scale-state-eps", type=float, default=1e-6)
    parser.add_argument("--origin-mode", choices=["transform_translation", "first_pose"], default="transform_translation")
    parser.add_argument("--alpha-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--boundary-only-estimated",
        action="store_true",
        help=(
            "Update the trusted scale state only on trace rows whose transform_reason is "
            "estimated_overlap_transform; reuse rows inherit the last boundary control scale."
        ),
    )
    args = parser.parse_args()

    source_run = args.source_run
    trace_path = source_run / "merge_state_trace.jsonl"
    pose_path = source_run / "postmerge_global_pose.jsonl"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    label = args.source_label or source_run.name
    _, _, gt_pos = _load_kitti_gt(args.gt)
    trace = _load_trace(trace_path)
    obs = _load_observability(args.observability_csv)
    frames, poses, frame_to_chunk = _load_postmerge_trajectory(pose_path)

    source_scales = [float(v["scale"]) for v in trace.values()]
    nonunit_count = sum(abs(s - 1.0) > float(args.scale_state_eps) for s in source_scales)
    source_has_scale_state = nonunit_count > 0
    source_kinds = sorted({str(v["row"].get("transform_kind", "")) for v in trace.values()})

    q_scale = {c: _float(row.get("Q_scale"), 0.5) for c, row in obs.items()}
    q_smooth = {c: _float(row.get("Q_scale_smoothed"), q_scale.get(c, 0.5)) for c, row in obs.items()}
    q_geom = {c: _float(row.get("geometry_confidence_mean"), 0.5) for c, row in obs.items()}
    q_shuffle = {c: q_scale[src] for c, src in zip(sorted(q_scale), random.Random(args.random_seed).sample(sorted(q_scale), len(q_scale)))}
    q_random = _category_q_values(q_scale, args.random_seed)
    straight_road = {
        c: str(row.get("straight_road_anchor_sparse", "")).lower() in {"1", "true", "yes", "on"}
        for c, row in obs.items()
    }
    anchor_type = {c: str(row.get("anchor_type", "")) for c, row in obs.items()}
    policy_straight_hold = {
        c: "hold" if straight_road.get(c, False) else "source_noop"
        for c in sorted(trace)
    }
    policy_anchor_refresh = {
        c: "refresh" if anchor_type.get(c) == "anchor_rich" else "source_noop"
        for c in sorted(trace)
    }
    policy_straight_hold_anchor_refresh = {
        c: "hold" if straight_road.get(c, False) else "refresh" if anchor_type.get(c) == "anchor_rich" else "source_noop"
        for c in sorted(trace)
    }

    candidates: List[Tuple[str, Dict[int, float], Optional[float], Optional[Dict[int, str]]]] = [
        ("BASELINE_SOURCE_NOOP", q_scale, 1.0, None),
        ("O3A_Q_FIXED_HOLD_REFRESH", q_scale, None, None),
        ("O3B_Q_SMOOTHED_HOLD_REFRESH", q_smooth, None, None),
        ("O3C_STRAIGHT_ROAD_HOLD_ONLY", q_scale, None, policy_straight_hold),
        ("O3D_ANCHOR_RICH_REFRESH_ONLY", q_scale, None, policy_anchor_refresh),
        ("O3E_COMBINED_STRAIGHT_HOLD_ANCHOR_REFRESH", q_scale, None, policy_straight_hold_anchor_refresh),
        ("O3F_GEOMETRY_ONLY_Q_CONTROL_PROXY", q_geom, None, None),
        ("O3G_SHUFFLED_SEMANTIC_Q_CONTROL", q_shuffle, None, None),
        ("O3H_RANDOM_CHUNK_HOLD_CONTROL", q_random, None, None),
    ]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: List[Dict[str, Any]] = []
    all_chunk_controller_rows: List[Dict[str, Any]] = []

    baseline_metrics: Optional[Dict[str, Any]] = None
    baseline_poses = poses
    baseline_controller_rows: List[Dict[str, Any]] = []
    for candidate, q_values, fixed_alpha, forced_actions in candidates:
        if candidate == "BASELINE_SOURCE_NOOP":
            ctrl_scales = {c: float(trace[c]["scale"]) for c in trace}
            controller_rows = [
                {
                    "chunk_idx": int(c),
                    "source_scale": float(trace[c]["scale"]),
                    "ctrl_scale": float(trace[c]["scale"]),
                    "Q_used": float(q_values.get(c, float("nan"))),
                    "action": "baseline",
                    "alpha": 0.0,
                    "raw_log_scale_delta": 0.0,
                    "applied_log_scale_delta": 0.0,
                    "ell_trust_after": math.log(max(float(trace[c]["scale"]), 1e-12)),
                }
                for c in sorted(trace)
            ]
            candidate_poses = baseline_poses
        else:
            ctrl_scales, controller_rows = _controller_scales(
                trace,
                q_values,
                fixed_alpha=fixed_alpha,
                alpha_multiplier=float(args.alpha_multiplier),
                boundary_only_estimated=bool(args.boundary_only_estimated),
                forced_actions=forced_actions,
            )
            candidate_poses = _apply_scale_control(
                frames,
                poses,
                frame_to_chunk,
                trace,
                ctrl_scales,
                origin_mode=args.origin_mode,
            )
        for row in controller_rows:
            row["candidate"] = candidate
            row["source_label"] = label
        if candidate == "BASELINE_SOURCE_NOOP":
            baseline_metrics = _metric_result_row(
                candidate,
                frames,
                candidate_poses,
                gt_pos,
                None,
                controller_rows,
                nonunit_count,
                source_has_scale_state,
                args.chunk_size,
                args.chunk_overlap,
                args.head_len,
            )
            result_rows.append(baseline_metrics)
            baseline_controller_rows = controller_rows
        else:
            result_rows.append(
                _metric_result_row(
                    candidate,
                    frames,
                    candidate_poses,
                    gt_pos,
                    baseline_metrics,
                    controller_rows,
                    nonunit_count,
                    source_has_scale_state,
                    args.chunk_size,
                    args.chunk_overlap,
                    args.head_len,
                )
            )
        all_chunk_controller_rows.extend(controller_rows)
        _write_tum(out_dir / "trajectories" / f"{candidate}.txt", frames, candidate_poses)

    control_names = {
        "geometry": "O3F_GEOMETRY_ONLY_Q_CONTROL_PROXY",
        "shuffled": "O3G_SHUFFLED_SEMANTIC_Q_CONTROL",
        "random": "O3H_RANDOM_CHUNK_HOLD_CONTROL",
    }
    by_name = {str(row["candidate"]): row for row in result_rows}
    for row in result_rows:
        metric_key = "future_after_overlap_mean"
        for label_control, control_name in control_names.items():
            control = by_name.get(control_name)
            row[f"beats_{label_control}"] = (
                False if control is None or row.get(metric_key) is None or control.get(metric_key) is None
                else float(row[metric_key]) <= 0.95 * float(control[metric_key])
            )

    primary = by_name.get("O3A_Q_FIXED_HOLD_REFRESH", {})
    mechanism_keys = [
        "head_to_tail_transfer_ratio_mean_improvement_vs_baseline",
        "future_after_overlap_mean_improvement_vs_baseline",
        "intra_scale_variance_mean_improvement_vs_baseline",
    ]
    best_primary_improvement = max(
        [float(primary.get(k)) for k in mechanism_keys if primary.get(k) is not None and math.isfinite(float(primary.get(k)))],
        default=float("nan"),
    )
    ate_regression = primary.get("delta_vs_baseline_global_ate")
    offline_gate = bool(
        source_has_scale_state
        and math.isfinite(best_primary_improvement)
        and best_primary_improvement >= 0.10
        and ate_regression is not None
        and float(ate_regression) <= 0.3
        and bool(primary.get("beats_geometry"))
        and bool(primary.get("beats_shuffled"))
        and bool(primary.get("beats_random"))
    )

    summary = {
        "source_label": label,
        "source_run": str(source_run),
        "gt": str(args.gt),
        "observability_csv": str(args.observability_csv),
        "origin_mode": args.origin_mode,
        "alpha_multiplier": float(args.alpha_multiplier),
        "boundary_only_estimated": bool(args.boundary_only_estimated),
        "source_transform_kinds": source_kinds,
        "source_scale_min": float(np.min(source_scales)) if source_scales else None,
        "source_scale_mean": float(np.mean(source_scales)) if source_scales else None,
        "source_scale_max": float(np.max(source_scales)) if source_scales else None,
        "source_nonunit_scale_count": int(nonunit_count),
        "source_has_scale_state": bool(source_has_scale_state),
        "precondition_pass": bool(source_has_scale_state),
        "precondition_blocker": None if source_has_scale_state else (
            "source merge_state_trace has no non-unit scale state; faithful Phase O3 ell_hat cannot be derived"
        ),
        "official_source_ate_if_present": _parse_official_ate(source_run),
        "internal_baseline": baseline_metrics,
        "baseline_controller_rows": baseline_controller_rows[:3],
        "offline_scale_controller_gate_pass": offline_gate,
        "gate_note": (
            "This is a faithful Phase O3 source only if source_run is the H35 O2 artifact and precondition_pass is true. "
            "Non-H35 semantic sources are repair diagnostics and must not be promoted directly."
        ),
        "best_primary_mechanism_improvement": None if not math.isfinite(best_primary_improvement) else best_primary_improvement,
        "primary_ate_regression": ate_regression,
        "result_count": len(result_rows),
    }

    _write_csv(out_dir / "offline_scale_controller_results.csv", result_rows)
    _write_csv(out_dir / "controller_chunk_trace.csv", all_chunk_controller_rows)
    (out_dir / "offline_scale_controller_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
