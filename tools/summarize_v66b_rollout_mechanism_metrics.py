#!/usr/bin/env python3
"""Posthoc mechanism metrics for v66B rollout trajectories.

This tool complements the ATE/RPE table. It uses landed trajectories plus KITTI
GT to compute pose-level proxies for the v62/v66B mechanism questions:

* per-chunk local Sim(3) geometry quality,
* head/mid/tail scale consistency,
* overlap/head Sim(3) transfer to future/tail frames,
* READ attention and TTT write/action fidelity from HMC JSONL logs.

It does not fabricate unavailable raw point/overlap metrics; transfer metrics
are explicitly trajectory-level GT proxies.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_ROLLOUT_DIR = (
    ROOT
    / "results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/"
    "phase9_parallel_continuation/rollouts"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/"
    "report_final/phase10_mechanism_posthoc"
)
DEFAULT_EXTRA_RUN_DIRS = [
    ROOT
    / "results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/"
    "phase5_read_smoke/rollouts/V66B_P5_READ_BASE_DENSE_IGNORE_96F_AFTER_PRIORFIX"
]


def _load_v62_helpers():
    path = ROOT / "tools/diagnose_v62_kitti01_error_source_autopsy.py"
    spec = importlib.util.spec_from_file_location("_v62_autopsy_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load v62 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


V62 = _load_v62_helpers()
EPS = 1e-12


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        val = _safe_float(value)
        if val is not None:
            out.append(val)
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else None


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.median(vals)) if vals else None


def _max(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.max(vals)) if vals else None


def _sum(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.sum(vals)) if vals else None


def _parse_average(path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not path.is_file():
        return None, None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split()
        if len(parts) >= 3 and parts[0].lower().rstrip(":") == "average":
            return _safe_float(parts[1]), _safe_float(parts[2])
    return None, None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fields:
                value = _clean(row.get(key))
                if value is None:
                    out[key] = ""
                elif isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _category(run_name: str) -> str:
    if "_READ_" in run_name:
        return "read"
    if "_TTT_" in run_name:
        return "ttt"
    return "other"


def _frame_scope(run_name: str) -> str:
    if "_704_" in run_name:
        return "704F"
    if run_name.endswith("_96F") or "_96F_" in run_name:
        return "96F"
    return "unknown"


def _schedule_from_hmc(rows: Sequence[Mapping[str, Any]], frames_used: int) -> List[Dict[str, int]]:
    seen = set()
    sched: List[Dict[str, int]] = []
    for row in rows:
        if not {"chunk_idx", "start_frame", "end_frame"} <= set(row):
            continue
        cid = int(row["chunk_idx"])
        if cid in seen:
            continue
        seen.add(cid)
        start = max(0, min(frames_used, int(row["start_frame"])))
        end = max(0, min(frames_used, int(row["end_frame"])))
        if end <= start:
            continue
        sched.append(
            {
                "chunk_idx": cid,
                "start_frame": start,
                "end_frame": end,
                "chunk_size": int(row.get("chunk_size") or 32),
                "chunk_overlap": int(row.get("chunk_overlap") or 3),
            }
        )
    sched.sort(key=lambda row: int(row["chunk_idx"]))
    return sched


def _fallback_schedule(frames_used: int, chunk_size: int = 32, overlap: int = 3) -> List[Dict[str, int]]:
    step = max(chunk_size - overlap, 1)
    out: List[Dict[str, int]] = []
    start = 0
    cid = 0
    while start < frames_used:
        end = min(frames_used, start + chunk_size)
        out.append(
            {
                "chunk_idx": cid,
                "start_frame": start,
                "end_frame": end,
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
            }
        )
        if end >= frames_used:
            break
        start += step
        cid += 1
    return out


def _split_indices(n: int) -> Dict[str, np.ndarray]:
    one = max(n // 3, 1)
    two = max((2 * n) // 3, one + 1)
    return {
        "head": np.arange(0, min(one, n), dtype=np.int64),
        "mid": np.arange(min(one, n), min(two, n), dtype=np.int64),
        "tail": np.arange(min(two, n), n, dtype=np.int64),
    }


def _segment_fit(pred: np.ndarray, gt: np.ndarray, idx: np.ndarray) -> Tuple[Optional[Tuple[float, np.ndarray, np.ndarray]], Optional[float]]:
    if idx.size < 3:
        return None, None
    fit = V62._fit_sim3(pred[idx], gt[idx])
    if fit is None:
        return None, None
    scale = _safe_float(fit[0])
    return fit, (math.log(abs(scale)) if scale is not None and abs(scale) > EPS else None)


def _hook_value(row: Mapping[str, Any], hook_name: str, key: str) -> Any:
    control = row.get("control_trace")
    if not isinstance(control, Mapping):
        return None
    hooks = control.get("hook_effect_summary")
    if not isinstance(hooks, Mapping):
        return None
    hook = hooks.get(hook_name)
    if not isinstance(hook, Mapping):
        return None
    return hook.get(key)


def _rows_by_chunk(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if "chunk_idx" in row:
            out.setdefault(int(row["chunk_idx"]), row)
    return out


def _summarize_run(run_dir: Path, gt_full: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    run_name = run_dir.name
    pred_path = run_dir / "01.txt"
    if not pred_path.is_file():
        return [], {
            "run": run_name,
            "artifact_dir": str(run_dir),
            "skipped": True,
            "skip_reason": "missing_01_txt",
        }
    pred = V62._read_poses(pred_path)
    frames_used = min(int(pred.centers.shape[0]), int(gt_full.centers.shape[0]))
    if frames_used < 3:
        return [], {
            "run": run_name,
            "artifact_dir": str(run_dir),
            "skipped": True,
            "skip_reason": "too_few_frames",
        }
    pred_centers = pred.centers[:frames_used]
    gt_centers = gt_full.centers[:frames_used]
    hmc_rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    hmc_by_chunk = _rows_by_chunk(hmc_rows)
    sched = _schedule_from_hmc(hmc_rows, frames_used) or _fallback_schedule(frames_used)
    global_fit = V62._fit_sim3(pred_centers, gt_centers)
    global_aligned = V62._apply_sim3(pred_centers, global_fit) if global_fit is not None else np.full_like(pred_centers, np.nan)
    global_stats = V62._residual_stats(global_aligned, gt_centers)
    ate_rmse, ate_sse = _parse_average(run_dir / "results_sim3/results_ate.txt")
    rpe_trans, rpe_rot = _parse_average(run_dir / "results_sim3/results_rpe.txt")

    chunk_rows: List[Dict[str, Any]] = []
    for item in sched:
        cid = int(item["chunk_idx"])
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        if end - start < 3:
            continue
        pred_chunk = pred_centers[start:end]
        gt_chunk = gt_centers[start:end]
        local_fit = V62._fit_sim3(pred_chunk, gt_chunk)
        local_aligned = V62._apply_sim3(pred_chunk, local_fit) if local_fit is not None else np.full_like(pred_chunk, np.nan)
        local_stats = V62._residual_stats(local_aligned, gt_chunk)
        global_chunk_stats = V62._residual_stats(global_aligned[start:end], gt_chunk)

        seg_idx = _split_indices(end - start)
        fits: Dict[str, Optional[Tuple[float, np.ndarray, np.ndarray]]] = {}
        log_scales: Dict[str, Optional[float]] = {}
        for segment, idx in seg_idx.items():
            fit, log_scale = _segment_fit(pred_chunk, gt_chunk, idx)
            fits[segment] = fit
            log_scales[segment] = log_scale

        scale_vals = _finite(log_scales.values())
        tail_idx = seg_idx["tail"]
        h2t_rmse = None
        m2t_rmse = None
        if fits.get("head") is not None and tail_idx.size >= 1:
            aligned_tail = V62._apply_sim3(pred_chunk[tail_idx], fits["head"])
            h2t_rmse = V62._residual_stats(aligned_tail, gt_chunk[tail_idx]).get("rmse")
        if fits.get("mid") is not None and tail_idx.size >= 1:
            aligned_tail = V62._apply_sim3(pred_chunk[tail_idx], fits["mid"])
            m2t_rmse = V62._residual_stats(aligned_tail, gt_chunk[tail_idx]).get("rmse")

        overlap = max(0, min(int(item.get("chunk_overlap", 3) or 3), end - start))
        overlap_fit = None
        overlap_residual = None
        future_error = None
        tail_error = None
        if cid > 0 and overlap >= 3:
            overlap_idx = np.arange(0, overlap, dtype=np.int64)
            overlap_fit = V62._fit_sim3(pred_chunk[overlap_idx], gt_chunk[overlap_idx])
            if overlap_fit is not None:
                overlap_aligned = V62._apply_sim3(pred_chunk[overlap_idx], overlap_fit)
                overlap_residual = V62._residual_stats(overlap_aligned, gt_chunk[overlap_idx]).get("rmse")
                future_idx = np.arange(overlap, end - start, dtype=np.int64)
                if future_idx.size:
                    future_aligned = V62._apply_sim3(pred_chunk[future_idx], overlap_fit)
                    future_error = V62._residual_stats(future_aligned, gt_chunk[future_idx]).get("rmse")
                if tail_idx.size:
                    tail_aligned = V62._apply_sim3(pred_chunk[tail_idx], overlap_fit)
                    tail_error = V62._residual_stats(tail_aligned, gt_chunk[tail_idx]).get("rmse")

        hmc = hmc_by_chunk.get(cid, {})
        row = {
            "run": run_name,
            "artifact_dir": str(run_dir),
            "category": _category(run_name),
            "frame_scope": _frame_scope(run_name),
            "metric_scope": "pose_gt_proxy",
            "raw_overlap_available": False,
            "chunk_idx": cid,
            "start_frame": start,
            "end_frame": end,
            "chunk_overlap": overlap,
            "global_sim3_rmse": global_stats.get("rmse"),
            "global_chunk_ate": global_chunk_stats.get("rmse"),
            "local_sim3_chunk_ate": local_stats.get("rmse"),
            "local_to_global_ate_ratio": (
                _safe_float(local_stats.get("rmse")) / _safe_float(global_chunk_stats.get("rmse"))
                if _safe_float(local_stats.get("rmse")) is not None and _safe_float(global_chunk_stats.get("rmse")) not in (None, 0.0)
                else None
            ),
            "log_scale_head": log_scales.get("head"),
            "log_scale_mid": log_scales.get("mid"),
            "log_scale_tail": log_scales.get("tail"),
            "intra_scale_variance": float(np.var(scale_vals)) if len(scale_vals) >= 2 else None,
            "head_to_tail_transfer_error": h2t_rmse,
            "head_to_tail_transfer_ratio": (
                _safe_float(h2t_rmse) / _safe_float(local_stats.get("rmse"))
                if _safe_float(h2t_rmse) is not None and _safe_float(local_stats.get("rmse")) not in (None, 0.0)
                else None
            ),
            "mid_to_tail_transfer_error": m2t_rmse,
            "overlap_residual_pose_gt_proxy": overlap_residual,
            "future_after_overlap_error_pose_gt_proxy": future_error,
            "tail_after_overlap_error_pose_gt_proxy": tail_error,
            "prior_semantic_role_consumed_any": hmc.get("prior_semantic_role_consumed_any"),
            "prior_semantic_role_control_mode": hmc.get("prior_semantic_role_control_mode"),
            "prior_semantic_role_control_applied": hmc.get("prior_semantic_role_control_applied"),
            "prior_semantic_role_control_changed_fraction": hmc.get("prior_semantic_role_control_changed_fraction"),
            "prior_ttt_write_present": hmc.get("prior_ttt_write_present"),
            "prior_ttt_write_mean": hmc.get("prior_ttt_write_mean"),
            "prior_condition_signal_conflict_available": hmc.get("prior_condition_signal_conflict_available"),
            "prior_condition_signal_conflict_level": hmc.get("prior_condition_signal_conflict_level"),
            "prior_condition_signal_conflict_source": hmc.get("prior_condition_signal_conflict_source"),
            "prior_condition_signal_conflict_value": hmc.get("prior_condition_signal_conflict_value"),
            "prior_condition_signal_conflict_token_exact": hmc.get("prior_condition_signal_conflict_token_exact"),
            "prior_condition_signal_conflict_token_mean": hmc.get("prior_condition_signal_conflict_token_mean"),
            "prior_condition_signal_conflict_token_p90": hmc.get("prior_condition_signal_conflict_token_p90"),
            "prior_condition_signal_scale_risk_available": hmc.get("prior_condition_signal_scale_risk_available"),
            "prior_condition_signal_scale_risk_level": hmc.get("prior_condition_signal_scale_risk_level"),
            "prior_condition_signal_scale_risk_source": hmc.get("prior_condition_signal_scale_risk_source"),
            "prior_condition_signal_scale_risk_value": hmc.get("prior_condition_signal_scale_risk_value"),
            "prior_condition_signal_scale_risk_token_exact": hmc.get("prior_condition_signal_scale_risk_token_exact"),
            "prior_condition_signal_scale_risk_token_mean": hmc.get("prior_condition_signal_scale_risk_token_mean"),
            "prior_condition_signal_scale_risk_token_p90": hmc.get("prior_condition_signal_scale_risk_token_p90"),
            "probe_ttt_write_action_delta_norm_mean": hmc.get("probe_ttt_write_action_delta_norm_mean"),
            "probe_ttt_write_post_delta_norm_mean": hmc.get("probe_ttt_write_post_delta_norm_mean"),
            "probe_ttt_write_native_delta_norm_mean": hmc.get("probe_ttt_write_native_delta_norm_mean"),
            "frame_context_source_skip_applied": _hook_value(hmc, "frame_attention", "num_context_source_skip_applied"),
            "frame_semantic_anchor_boost_applied": _hook_value(hmc, "frame_attention", "num_semantic_anchor_boost_applied"),
            "frame_attention_mass_removed_before": _hook_value(hmc, "frame_attention", "mean_attention_mass_removed_before"),
            "frame_attention_mass_removed_after": _hook_value(hmc, "frame_attention", "mean_attention_mass_removed_after"),
            "frame_mean_abs_bias": _hook_value(hmc, "frame_attention", "mean_abs_bias"),
            "prior_semantic_swa_role_control_applied": hmc.get("prior_semantic_swa_role_control_applied"),
            "prior_semantic_swa_role_control_changed_fraction": hmc.get("prior_semantic_swa_role_control_changed_fraction"),
            "prior_semantic_role_swa_protect_adjusted": hmc.get("prior_semantic_role_swa_protect_adjusted"),
            "prior_semantic_role_swa_score_before_mean": hmc.get("prior_semantic_role_swa_score_before_mean"),
            "prior_semantic_role_swa_score_after_mean": hmc.get("prior_semantic_role_swa_score_after_mean"),
            "prior_semantic_role_swa_score_protect_before_mean": hmc.get("prior_semantic_role_swa_score_protect_before_mean"),
            "prior_semantic_role_swa_score_protect_after_mean": hmc.get("prior_semantic_role_swa_score_protect_after_mean"),
            "swa_overlap_source_gate_applied": _hook_value(hmc, "swa_read", "num_swa_overlap_source_gate_applied"),
            "swa_overlap_source_gate_delta": _hook_value(hmc, "swa_read", "mean_swa_overlap_source_gate_delta"),
            "swa_overlap_source_gate_score_mean": _hook_value(hmc, "swa_read", "mean_swa_overlap_source_score"),
            "swa_overlap_source_replace_applied": _hook_value(hmc, "swa_read", "num_swa_overlap_source_replace_applied"),
            "swa_overlap_source_replace_alpha": _hook_value(hmc, "swa_read", "mean_swa_overlap_source_replace_alpha"),
            "swa_overlap_source_replace_score_mean": _hook_value(hmc, "swa_read", "mean_swa_overlap_source_replace_score"),
        }
        chunk_rows.append(row)

    summary = {
        "run": run_name,
        "artifact_dir": str(run_dir),
        "category": _category(run_name),
        "frame_scope": _frame_scope(run_name),
        "metric_scope": "pose_gt_proxy",
        "raw_overlap_available": False,
        "skipped": False,
        "frames_used": frames_used,
        "num_chunks": len(chunk_rows),
        "ate_rmse": ate_rmse,
        "ate_sse": ate_sse,
        "rpe_trans": rpe_trans,
        "rpe_rot": rpe_rot,
        "global_sim3_rmse": global_stats.get("rmse"),
        "local_sim3_chunk_ate_mean": _mean(row.get("local_sim3_chunk_ate") for row in chunk_rows),
        "local_sim3_chunk_ate_median": _median(row.get("local_sim3_chunk_ate") for row in chunk_rows),
        "local_to_global_ate_ratio_median": _median(row.get("local_to_global_ate_ratio") for row in chunk_rows),
        "intra_scale_variance_mean": _mean(row.get("intra_scale_variance") for row in chunk_rows),
        "intra_scale_variance_median": _median(row.get("intra_scale_variance") for row in chunk_rows),
        "head_to_tail_transfer_error_mean": _mean(row.get("head_to_tail_transfer_error") for row in chunk_rows),
        "head_to_tail_transfer_ratio_mean": _mean(row.get("head_to_tail_transfer_ratio") for row in chunk_rows),
        "overlap_residual_pose_gt_proxy_mean": _mean(row.get("overlap_residual_pose_gt_proxy") for row in chunk_rows),
        "future_after_overlap_error_pose_gt_proxy_mean": _mean(row.get("future_after_overlap_error_pose_gt_proxy") for row in chunk_rows),
        "tail_after_overlap_error_pose_gt_proxy_mean": _mean(row.get("tail_after_overlap_error_pose_gt_proxy") for row in chunk_rows),
        "semantic_role_consumed_rows": sum(1 for row in chunk_rows if bool(row.get("prior_semantic_role_consumed_any"))),
        "semantic_role_control_applied_rows": sum(1 for row in chunk_rows if bool(row.get("prior_semantic_role_control_applied"))),
        "semantic_role_control_changed_fraction_mean": _mean(row.get("prior_semantic_role_control_changed_fraction") for row in chunk_rows),
        "ttt_write_present_rows": sum(1 for row in chunk_rows if bool(row.get("prior_ttt_write_present"))),
        "prior_ttt_write_mean_avg": _mean(row.get("prior_ttt_write_mean") for row in chunk_rows),
        "condition_conflict_available_rows": sum(1 for row in chunk_rows if bool(row.get("prior_condition_signal_conflict_available"))),
        "condition_conflict_chunk_broadcast_rows": sum(1 for row in chunk_rows if row.get("prior_condition_signal_conflict_level") == "chunk_broadcast"),
        "condition_conflict_token_exact_rows": sum(1 for row in chunk_rows if bool(row.get("prior_condition_signal_conflict_token_exact"))),
        "condition_conflict_value_avg": _mean(row.get("prior_condition_signal_conflict_value") for row in chunk_rows),
        "condition_conflict_token_mean_avg": _mean(row.get("prior_condition_signal_conflict_token_mean") for row in chunk_rows),
        "condition_conflict_token_p90_avg": _mean(row.get("prior_condition_signal_conflict_token_p90") for row in chunk_rows),
        "condition_scale_risk_available_rows": sum(1 for row in chunk_rows if bool(row.get("prior_condition_signal_scale_risk_available"))),
        "condition_scale_risk_chunk_broadcast_rows": sum(1 for row in chunk_rows if row.get("prior_condition_signal_scale_risk_level") == "chunk_broadcast"),
        "condition_scale_risk_token_exact_rows": sum(1 for row in chunk_rows if bool(row.get("prior_condition_signal_scale_risk_token_exact"))),
        "condition_scale_risk_value_avg": _mean(row.get("prior_condition_signal_scale_risk_value") for row in chunk_rows),
        "condition_scale_risk_token_mean_avg": _mean(row.get("prior_condition_signal_scale_risk_token_mean") for row in chunk_rows),
        "condition_scale_risk_token_p90_avg": _mean(row.get("prior_condition_signal_scale_risk_token_p90") for row in chunk_rows),
        "probe_ttt_write_action_delta_norm_mean": _mean(row.get("probe_ttt_write_action_delta_norm_mean") for row in chunk_rows),
        "probe_ttt_write_post_delta_norm_mean": _mean(row.get("probe_ttt_write_post_delta_norm_mean") for row in chunk_rows),
        "frame_context_source_skip_applied_total": _sum(row.get("frame_context_source_skip_applied") for row in chunk_rows),
        "frame_semantic_anchor_boost_applied_total": _sum(row.get("frame_semantic_anchor_boost_applied") for row in chunk_rows),
        "frame_attention_mass_removed_before_mean": _mean(row.get("frame_attention_mass_removed_before") for row in chunk_rows),
        "frame_attention_mass_removed_after_mean": _mean(row.get("frame_attention_mass_removed_after") for row in chunk_rows),
        "frame_mean_abs_bias_mean": _mean(row.get("frame_mean_abs_bias") for row in chunk_rows),
        "semantic_swa_role_control_applied_rows": sum(1 for row in chunk_rows if bool(row.get("prior_semantic_swa_role_control_applied"))),
        "semantic_swa_role_control_changed_fraction_mean": _mean(row.get("prior_semantic_swa_role_control_changed_fraction") for row in chunk_rows),
        "semantic_role_swa_protect_adjusted_rows": sum(1 for row in chunk_rows if bool(row.get("prior_semantic_role_swa_protect_adjusted"))),
        "semantic_role_swa_score_before_mean": _mean(row.get("prior_semantic_role_swa_score_before_mean") for row in chunk_rows),
        "semantic_role_swa_score_after_mean": _mean(row.get("prior_semantic_role_swa_score_after_mean") for row in chunk_rows),
        "semantic_role_swa_score_protect_before_mean": _mean(row.get("prior_semantic_role_swa_score_protect_before_mean") for row in chunk_rows),
        "semantic_role_swa_score_protect_after_mean": _mean(row.get("prior_semantic_role_swa_score_protect_after_mean") for row in chunk_rows),
        "swa_overlap_source_gate_applied_total": _sum(row.get("swa_overlap_source_gate_applied") for row in chunk_rows),
        "swa_overlap_source_gate_delta_mean": _mean(row.get("swa_overlap_source_gate_delta") for row in chunk_rows),
        "swa_overlap_source_gate_score_mean": _mean(row.get("swa_overlap_source_gate_score_mean") for row in chunk_rows),
        "swa_overlap_source_replace_applied_total": _sum(row.get("swa_overlap_source_replace_applied") for row in chunk_rows),
        "swa_overlap_source_replace_alpha_mean": _mean(row.get("swa_overlap_source_replace_alpha") for row in chunk_rows),
        "swa_overlap_source_replace_score_mean": _mean(row.get("swa_overlap_source_replace_score_mean") for row in chunk_rows),
    }
    return chunk_rows, summary


def _base_for(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    category = row.get("category")
    frame_scope = row.get("frame_scope")
    if category == "read":
        needles = ["V66B_P9_704_READ_BASE_DENSE_IGNORE"] if frame_scope == "704F" else [
            "V66B_P5_READ_BASE_DENSE_IGNORE_96F_AFTER_PRIORFIX",
            "V66B_P9_READ_BASE_DENSE_IGNORE",
        ]
    elif category == "ttt":
        needles = [f"V66B_P9_{'704_' if frame_scope == '704F' else ''}TTT_BASE_DENSE_IGNORE"]
    else:
        return None
    candidates = [
        base for base in rows
        if not base.get("skipped")
        and base.get("category") == category
        and base.get("frame_scope") == frame_scope
        and any(str(base.get("run", "")).startswith(needle) for needle in needles)
    ]
    return candidates[0] if candidates else None


def _delta(row: Mapping[str, Any], base: Mapping[str, Any], key: str) -> Optional[float]:
    value = _safe_float(row.get(key))
    base_value = _safe_float(base.get(key))
    return value - base_value if value is not None and base_value is not None else None


def _improvement(row: Mapping[str, Any], base: Mapping[str, Any], key: str) -> Optional[float]:
    value = _safe_float(row.get(key))
    base_value = _safe_float(base.get(key))
    if value is None or base_value is None or abs(base_value) <= EPS:
        return None
    return (base_value - value) / abs(base_value)


def summarize(rollout_dir: Path, out_dir: Path, gt_poses: Path, extra_run_dirs: Sequence[Path]) -> Dict[str, Any]:
    gt = V62._read_poses(gt_poses)
    run_dirs = sorted(path for path in rollout_dir.iterdir() if path.is_dir() and path.name.startswith("V66B_P"))
    run_dirs.extend(path for path in extra_run_dirs if path.is_dir())
    chunk_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        chunks, summary = _summarize_run(run_dir, gt)
        chunk_rows.extend(chunks)
        summary_rows.append(summary)

    for row in summary_rows:
        if row.get("skipped"):
            continue
        base = _base_for(row, summary_rows)
        if base is None:
            continue
        row["base_run"] = base.get("run")
        for key in (
            "ate_rmse",
            "local_sim3_chunk_ate_mean",
            "intra_scale_variance_mean",
            "head_to_tail_transfer_ratio_mean",
            "future_after_overlap_error_pose_gt_proxy_mean",
            "tail_after_overlap_error_pose_gt_proxy_mean",
        ):
            row[f"{key}_delta_vs_base"] = _delta(row, base, key)
            row[f"{key}_improvement_vs_base"] = _improvement(row, base, key)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows.sort(key=lambda row: (str(row.get("frame_scope")), str(row.get("category")), str(row.get("run"))))
    chunk_rows.sort(key=lambda row: (str(row.get("run")), int(row.get("chunk_idx", -1))))
    _write_csv(out_dir / "rollout_mechanism_by_chunk.csv", chunk_rows)
    _write_csv(out_dir / "rollout_mechanism_summary.csv", summary_rows)
    _write_json(out_dir / "rollout_mechanism_by_chunk.json", chunk_rows)
    _write_json(out_dir / "rollout_mechanism_summary.json", summary_rows)

    report = _build_report(summary_rows, chunk_rows, out_dir)
    (out_dir / "rollout_mechanism_report.md").write_text(report, encoding="utf-8")
    result = {
        "out_dir": str(out_dir),
        "rollout_dir": str(rollout_dir),
        "gt_poses": str(gt_poses),
        "run_count": len(summary_rows),
        "skipped_runs": [row for row in summary_rows if row.get("skipped")],
        "summary_csv": str(out_dir / "rollout_mechanism_summary.csv"),
        "chunk_csv": str(out_dir / "rollout_mechanism_by_chunk.csv"),
        "report": str(out_dir / "rollout_mechanism_report.md"),
    }
    _write_json(out_dir / "rollout_mechanism_index.json", result)
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    return f"{val:.{digits}f}" if val is not None else "NA"


def _build_report(summary_rows: Sequence[Mapping[str, Any]], chunk_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> str:
    rows = [row for row in summary_rows if not row.get("skipped")]
    ttt704 = [row for row in rows if row.get("category") == "ttt" and row.get("frame_scope") == "704F"]
    read704 = [row for row in rows if row.get("category") == "read" and row.get("frame_scope") == "704F"]

    def table(subset: Sequence[Mapping[str, Any]]) -> List[str]:
        out = [
            "| run | ATE | local Sim3 mean | intra-scale var | head->tail ratio | future-after-overlap | action mass | scale-cond rows | delta future |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in subset:
            out.append(
                f"| `{row.get('run')}` | {_fmt(row.get('ate_rmse'))} | "
                f"{_fmt(row.get('local_sim3_chunk_ate_mean'))} | "
                f"{_fmt(row.get('intra_scale_variance_mean'))} | "
                f"{_fmt(row.get('head_to_tail_transfer_ratio_mean'))} | "
                f"{_fmt(row.get('future_after_overlap_error_pose_gt_proxy_mean'))} | "
                f"{_fmt(row.get('prior_ttt_write_mean_avg'))} | "
                f"{row.get('condition_scale_risk_available_rows', 'NA')} | "
                f"{_fmt(row.get('future_after_overlap_error_pose_gt_proxy_mean_delta_vs_base'))} |"
            )
        return out

    lines = [
        "# v66B rollout mechanism posthoc",
        "",
        "This report uses landed trajectory poses and KITTI GT. It is a pose-level proxy for local Sim(3), head/mid/tail scale consistency, and overlap/head-to-future transfer; raw pointmap overlap metrics are not inferred.",
        "",
        "## 704F TTT",
        "",
        *table(ttt704),
        "",
        "## 704F READ",
        "",
        *table(read704),
        "",
        "## Artifacts",
        "",
        f"- chunk CSV: `{out_dir / 'rollout_mechanism_by_chunk.csv'}`",
        f"- summary CSV: `{out_dir / 'rollout_mechanism_summary.csv'}`",
        f"- summary JSON: `{out_dir / 'rollout_mechanism_summary.json'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gt-poses", type=Path, default=DEFAULT_GT)
    parser.add_argument("--extra-run-dir", type=Path, action="append", default=None)
    args = parser.parse_args()
    extra = args.extra_run_dir if args.extra_run_dir is not None else DEFAULT_EXTRA_RUN_DIRS
    result = summarize(args.rollout_dir, args.out_dir, args.gt_poses, extra)
    print(json.dumps(_clean(result), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
