#!/usr/bin/env python3
"""Evaluate ACL2 v68 Phase D READ online smoke runs.

This is a short-window, trajectory-only evaluator for the Phase D smoke gate.
It does not rerun HMC.  The "overlap_to_future" and "head_to_tail" metrics are
pose Sim(3) transfer proxies computed from the emitted TUM trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _angle_diff_deg,
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
    _yaw_from_pose,
)


DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_BASE_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/"
    "phaseD_read_online_smoke/chunk10_32f"
)
DEFAULT_RUNS = [
    "candidate",
    "geometry_only",
    "same_cue_random",
    "label_shuffled",
    "confidence_shuffled",
    "joint_shuffled",
    "native_no_read",
]
DEFAULT_CONTROLS = [
    "geometry_only",
    "same_cue_random",
    "label_shuffled",
    "confidence_shuffled",
    "joint_shuffled",
]
LOWER_IS_BETTER_KEYS = [
    "local_sim3_ate_rmse_m",
    "segment_200_300_intersection_allfit_rmse_m",
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]
MECHANISM_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


def _read_jsonl_first(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _read_jsonl_all(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _finite_max(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.max(xs)) if xs else None


def _safe_ratio_improvement(baseline: Any, candidate: Any) -> Optional[float]:
    base = _finite(baseline)
    cand = _finite(candidate)
    if base is None or cand is None or abs(base) < 1e-12:
        return None
    return float((base - cand) / abs(base))


def _aggregate_hmc_debug(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl_all(run_dir / "hmc_state_hash.jsonl")
    if not rows:
        return {"hmc_debug_rows": 0}
    implemented = sorted(
        {
            str(path)
            for row in rows
            for path in (row.get("control_trace", {}).get("implemented_paths") or [])
        }
    )
    swa_rows = [
        row.get("control_trace", {}).get("hook_effect_summary", {}).get("swa_read", {})
        for row in rows
        if isinstance(row.get("control_trace", {}).get("hook_effect_summary", {}).get("swa_read", {}), dict)
    ]
    frame_rows = [
        row.get("control_trace", {}).get("hook_effect_summary", {}).get("frame_attention", {})
        for row in rows
        if isinstance(row.get("control_trace", {}).get("hook_effect_summary", {}).get("frame_attention", {}), dict)
    ]
    return {
        "hmc_debug_rows": int(len(rows)),
        "implemented_paths_all": implemented,
        "v68_read_available_count": int(sum(1 for row in rows if bool(row.get("prior_v68_read_available", False)))),
        "v68_read_reason_set": sorted({str(row.get("prior_v68_read_reason")) for row in rows if row.get("prior_v68_read_reason") is not None}),
        "swa_num_calls_sum": int(sum(int(row.get("num_calls", 0) or 0) for row in swa_rows)),
        "swa_num_overlap_source_gate_applied_sum": int(sum(int(row.get("num_swa_overlap_source_gate_applied", 0) or 0) for row in swa_rows)),
        "swa_num_overlap_source_replace_applied_sum": int(sum(int(row.get("num_swa_overlap_source_replace_applied", 0) or 0) for row in swa_rows)),
        "swa_mean_overlap_source_gate_delta": _finite_mean(row.get("mean_swa_overlap_source_gate_delta") for row in swa_rows),
        "swa_max_overlap_source_gate_delta": _finite_max(row.get("max_swa_overlap_source_gate_delta") for row in swa_rows),
        "swa_mean_overlap_source_score": _finite_mean(row.get("mean_swa_overlap_source_score") for row in swa_rows),
        "swa_mean_overlap_source_score_q90": _finite_mean(row.get("mean_swa_overlap_source_score_q90") for row in swa_rows),
        "swa_mean_overlap_source_replace_alpha": _finite_mean(row.get("mean_swa_overlap_source_replace_alpha") for row in swa_rows),
        "swa_mean_overlap_source_replace_alpha_p90": _finite_mean(row.get("mean_swa_overlap_source_replace_alpha_p90") for row in swa_rows),
        "swa_mean_overlap_source_replace_score": _finite_mean(row.get("mean_swa_overlap_source_replace_score") for row in swa_rows),
        "swa_mean_semantic_selected_ratio": _finite_mean(row.get("mean_swa_overlap_source_semantic_selected_ratio") for row in swa_rows),
        "swa_max_semantic_selected_tokens": _finite_max(row.get("max_swa_overlap_source_semantic_selected_tokens") for row in swa_rows),
        "swa_frac_semantic_random_same_mass": _finite_mean(row.get("frac_swa_overlap_source_semantic_random_same_mass") for row in swa_rows),
        "swa_frac_semantic_missing_labels": _finite_mean(row.get("frac_swa_overlap_source_semantic_missing_labels") for row in swa_rows),
        "frame_attention_mean_abs_bias_mean": _finite_mean(row.get("mean_abs_bias") for row in frame_rows),
        "frame_attention_max_abs_bias_max": _finite_max(row.get("max_abs_bias") for row in frame_rows),
    }


def _fit_eval_by_indices(
    frames: np.ndarray,
    raw_pos: np.ndarray,
    gt_pos_all: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "fit_n": int(fit_idx.size),
        "eval_n": int(eval_idx.size),
        "ate_rmse_m": None,
        "ate_mean_m": None,
        "ate_p90_m": None,
        "sim3_scale": None,
        "valid": False,
    }
    if fit_idx.size < 3 or eval_idx.size < 1:
        out["reason"] = "insufficient_frames"
        return out
    fit_frames = frames[fit_idx]
    eval_frames = frames[eval_idx]
    try:
        scale, rot, trans = _umeyama_sim3(raw_pos[fit_idx], gt_pos_all[fit_frames], with_scale=True)
    except Exception as exc:
        out["reason"] = f"sim3_fit_failed:{type(exc).__name__}"
        return out
    aligned = (scale * (rot @ raw_pos[eval_idx].T)).T + trans
    err = np.linalg.norm(aligned - gt_pos_all[eval_frames], axis=1)
    out.update(
        {
            "ate_rmse_m": float(_rmse(err)),
            "ate_mean_m": float(np.mean(err)),
            "ate_p90_m": float(np.percentile(err, 90)),
            "sim3_scale": float(scale),
            "valid": True,
        }
    )
    return out


def _window_indices(n: int, start: int, end: int) -> np.ndarray:
    start = max(0, min(n, int(start)))
    end = max(start, min(n, int(end)))
    return np.arange(start, end, dtype=np.int64)


def _parse_run_arg(base_dir: Path, spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path)
    path = Path(spec)
    if path.exists():
        return path.name, path
    return spec, base_dir / spec


def _eval_run(run_name: str, run_dir: Path, gt_poses_all: np.ndarray, gt_pos_all: np.ndarray) -> Dict[str, Any]:
    traj_path = run_dir / "01.txt"
    frames, raw_poses, raw_pos = _load_tum_prediction(traj_path, gt_pos_all.shape[0])
    if frames.size < 3:
        raise ValueError(f"{traj_path}: need at least 3 valid frames, got {frames.size}")

    gt_pos = gt_pos_all[frames]
    scale, rot, trans = _umeyama_sim3(raw_pos, gt_pos, with_scale=True)
    aligned = _apply_alignment(raw_poses, scale, rot, trans)
    aligned_pos = aligned[:, :3, 3]
    err = aligned_pos - gt_pos
    err_norm = np.linalg.norm(err, axis=1)
    yaw_err = _angle_diff_deg(_yaw_from_pose(aligned, "xz"), _yaw_from_pose(gt_poses_all[frames], "xz"))

    n = int(frames.size)
    head = min(10, n)
    overlap = min(3, n)
    third = max(3, n // 3)
    third_windows = [
        _window_indices(n, 0, third),
        _window_indices(n, max(0, (n - third) // 2), max(0, (n - third) // 2) + third),
        _window_indices(n, n - third, n),
    ]
    third_scales: List[float] = []
    for idx in third_windows:
        part = _fit_eval_by_indices(frames, raw_pos, gt_pos_all, idx, idx)
        scale_v = _finite(part.get("sim3_scale"))
        if scale_v is not None:
            third_scales.append(scale_v)
    if third_scales:
        scale_cv = float(np.std(np.asarray(third_scales)) / max(abs(float(np.mean(third_scales))), 1e-12))
    else:
        scale_cv = None

    head_idx = _window_indices(n, 0, head)
    tail_idx = _window_indices(n, n - head, n)
    overlap_idx = _window_indices(n, 0, overlap)
    future_idx = _window_indices(n, overlap, n)
    head_tail = _fit_eval_by_indices(frames, raw_pos, gt_pos_all, head_idx, tail_idx)
    overlap_future = _fit_eval_by_indices(frames, raw_pos, gt_pos_all, overlap_idx, future_idx)

    seg_mask = (frames >= 200) & (frames < 300)
    if bool(seg_mask.any()):
        seg_rmse = float(_rmse(err_norm[seg_mask]))
        seg_n = int(seg_mask.sum())
        seg_start = int(frames[seg_mask].min())
        seg_end = int(frames[seg_mask].max()) + 1
    else:
        seg_rmse = None
        seg_n = 0
        seg_start = None
        seg_end = None

    hmc = _read_jsonl_first(run_dir / "hmc_state_hash.jsonl")
    hmc_agg = _aggregate_hmc_debug(run_dir)
    hook = _read_jsonl_first(run_dir / "hook_effect_summary.jsonl")
    hook_summary = hook.get("hook_effect_summary") or hmc.get("control_trace", {}).get("hook_effect_summary", {})
    frame_hook = hook_summary.get("frame_attention", {}) if isinstance(hook_summary, dict) else {}

    row: Dict[str, Any] = {
        "run": run_name,
        "run_dir": str(run_dir),
        "trajectory": str(traj_path),
        "frame_count": n,
        "frame_start": int(frames.min()),
        "frame_end_exclusive": int(frames.max()) + 1,
        "local_sim3_ate_rmse_m": float(_rmse(err_norm)),
        "local_sim3_finalerr_m": float(err_norm[-1]),
        "local_sim3_yaw_rmse_deg": float(_rmse(yaw_err)),
        "local_sim3_scale": float(scale),
        "segment_200_300_intersection_allfit_rmse_m": seg_rmse,
        "segment_200_300_intersection_n": seg_n,
        "segment_200_300_intersection_start": seg_start,
        "segment_200_300_intersection_end_exclusive": seg_end,
        "head10_to_tail10_pose_sim3_rmse_m": head_tail.get("ate_rmse_m"),
        "head10_to_tail10_pose_sim3_scale": head_tail.get("sim3_scale"),
        "head10_to_tail10_fit_n": head_tail.get("fit_n"),
        "head10_to_tail10_eval_n": head_tail.get("eval_n"),
        "overlap3_to_future_pose_sim3_rmse_m": overlap_future.get("ate_rmse_m"),
        "overlap3_to_future_pose_sim3_scale": overlap_future.get("sim3_scale"),
        "overlap3_to_future_fit_n": overlap_future.get("fit_n"),
        "overlap3_to_future_eval_n": overlap_future.get("eval_n"),
        "scale_cv_head_mid_tail_pose_sim3": scale_cv,
        "scale_head_mid_tail_values": third_scales,
        "prior_v68_read_available": hmc.get("prior_v68_read_available"),
        "prior_v68_read_reason": hmc.get("prior_v68_read_reason"),
        "prior_v68_read_control": hmc.get("prior_v68_read_control"),
        "prior_v68_read_fusion": hmc.get("prior_v68_read_fusion"),
        "prior_v68_read_layers": hmc.get("prior_v68_read_layers"),
        "prior_v68_read_output_mean": hmc.get("prior_v68_read_output_mean"),
        "prior_v68_read_output_q90": hmc.get("prior_v68_read_output_q90"),
        "prior_v68_read_output_gt050_mass": hmc.get("prior_v68_read_output_gt050_mass"),
        "prior_v68_read_gram_mean": hmc.get("prior_v68_read_gram_mean"),
        "prior_v68_read_gram_q90": hmc.get("prior_v68_read_gram_q90"),
        "prior_v68_read_semantic_risk_mean": hmc.get("prior_v68_read_semantic_risk_mean"),
        "prior_v68_read_semantic_trust_mean": hmc.get("prior_v68_read_semantic_trust_mean"),
        "prior_v68_read_corr_output_motion": hmc.get("prior_v68_read_corr_output_motion"),
        "prior_v68_read_corr_output_sem_risk": hmc.get("prior_v68_read_corr_output_sem_risk"),
        "prior_beta_frame_effective": hmc.get("prior_beta_frame_effective"),
        "pass1_pass2_pose_t_mean": hmc.get("pass1_pass2_pose_t_mean"),
        "pass1_pass2_pose_t_max": hmc.get("pass1_pass2_pose_t_max"),
        "pass1_pass2_pose_matrix_abs_max": hmc.get("pass1_pass2_pose_matrix_abs_max"),
        "implemented_paths": hmc.get("control_trace", {}).get("implemented_paths"),
        "frame_attention_num_calls": frame_hook.get("num_calls"),
        "frame_attention_num_enabled_layers": frame_hook.get("num_enabled_layers"),
        "frame_attention_mean_abs_bias": frame_hook.get("mean_abs_bias"),
        "frame_attention_max_abs_bias": frame_hook.get("max_abs_bias"),
    }
    row.update(hmc_agg)
    return row


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def _best_control_value(rows_by_name: Dict[str, Dict[str, Any]], controls: Sequence[str], key: str) -> Optional[float]:
    vals = [_finite(rows_by_name.get(name, {}).get(key)) for name in controls]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _build_decision(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    controls: Sequence[str],
) -> Dict[str, Any]:
    rows_by_name = {str(row["run"]): row for row in rows}
    cand = rows_by_name.get(candidate)
    base = rows_by_name.get(baseline)
    if cand is None:
        return {"phaseD_gate_pass": False, "reason": f"missing_candidate:{candidate}"}
    if base is None:
        return {"phaseD_gate_pass": False, "reason": f"missing_baseline:{baseline}"}

    comparisons: Dict[str, Dict[str, Any]] = {}
    metric_passes: List[str] = []
    for key in LOWER_IS_BETTER_KEYS:
        cand_v = _finite(cand.get(key))
        base_v = _finite(base.get(key))
        best_ctrl = _best_control_value(rows_by_name, controls, key)
        beats_controls = cand_v is not None and best_ctrl is not None and cand_v < best_ctrl
        abs_improvement = (base_v - cand_v) if base_v is not None and cand_v is not None else None
        ratio_improvement = _safe_ratio_improvement(base_v, cand_v)
        if key == "segment_200_300_intersection_allfit_rmse_m":
            key_pass = bool(beats_controls and abs_improvement is not None and abs_improvement >= 0.5)
        elif key in MECHANISM_KEYS:
            key_pass = bool(beats_controls and ratio_improvement is not None and ratio_improvement >= 0.05)
        else:
            key_pass = False
        if key_pass:
            metric_passes.append(key)
        comparisons[key] = {
            "candidate": cand_v,
            "baseline": base_v,
            "best_control": best_ctrl,
            "candidate_minus_baseline": (cand_v - base_v) if cand_v is not None and base_v is not None else None,
            "candidate_minus_best_control": (cand_v - best_ctrl) if cand_v is not None and best_ctrl is not None else None,
            "improvement_vs_baseline_ratio": ratio_improvement,
            "beats_all_controls_for_key": beats_controls,
            "phaseD_key_pass": key_pass,
        }

    return {
        "phaseD_gate_pass": bool(metric_passes),
        "candidate": candidate,
        "baseline": baseline,
        "controls": list(controls),
        "metric_passes": metric_passes,
        "comparisons": comparisons,
        "rule": (
            "candidate must beat all listed controls for the same lower-is-better metric; "
            "segment_200_300_intersection needs >=0.5m absolute improvement vs baseline, "
            "head_tail/overlap_to_future/scale_cv pose proxies need >=5% improvement vs baseline"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--run", action="append", default=[], help="Run name under --base-dir or NAME=run_dir, repeatable")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate run name to gate; repeatable")
    parser.add_argument("--baseline", default="native_no_read")
    parser.add_argument("--control", action="append", default=[], help="Control run name, repeatable")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    run_specs = args.run or DEFAULT_RUNS
    controls = args.control or DEFAULT_CONTROLS
    out_json = args.out_json or args.base_dir / "phaseD_read_smoke_metrics.json"
    out_csv = args.out_csv or args.base_dir / "phaseD_read_smoke_metrics.csv"

    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    rows = [
        _eval_run(name, run_dir, gt_poses_all, gt_pos_all)
        for name, run_dir in (_parse_run_arg(args.base_dir, spec) for spec in run_specs)
    ]
    candidates = args.candidate or ["candidate"]
    decisions = {
        cand: _build_decision(rows, candidate=cand, baseline=args.baseline, controls=controls)
        for cand in candidates
    }
    decision = decisions[candidates[0]]

    payload = {
        "base_dir": str(args.base_dir),
        "gt": str(args.gt),
        "runs": rows,
        "decision": decision,
        "decisions": decisions,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, rows)

    print(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")


if __name__ == "__main__":
    main()
