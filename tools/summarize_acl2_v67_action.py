#!/usr/bin/env python3
"""Summarize ACL2 v67 semantic READ and TTT action audit fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _parse_pred(arg: str) -> Tuple[str, Path]:
    if "=" in arg:
        name, path = arg.split("=", 1)
        return name.strip(), Path(path)
    path = Path(arg)
    return path.name, path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _get(row: Dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    prefixed = f"prior_{key}"
    if prefixed in row:
        return row.get(prefixed)
    return None


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    xs = [_num(_get(r, key)) for r in rows]
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _frac_true(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for row in rows:
        value = _get(row, key)
        if value is None:
            continue
        vals.append(bool(value))
    return float(np.mean(vals)) if vals else None


def _int_values(rows: Iterable[Dict[str, Any]], key: str) -> List[int]:
    vals: List[int] = []
    for row in rows:
        value = _get(row, key)
        if value is None:
            continue
        try:
            vals.append(int(value))
        except (TypeError, ValueError):
            continue
    return vals


def _list_values(rows: Iterable[Dict[str, Any]], key: str) -> List[Any]:
    vals: List[Any] = []
    for row in rows:
        value = _get(row, key)
        if value in (None, ""):
            continue
        if isinstance(value, list):
            vals.extend(value)
        else:
            vals.append(value)
    return vals


def _hook_mean(rows: Iterable[Dict[str, Any]], path: List[str]) -> Optional[float]:
    vals = []
    for row in rows:
        cur: Any = row
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        v = _num(cur)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _hook_mean_context_applied(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for row in rows:
        frame = row.get("hook_effect_summary", {}).get("frame_attention", {})
        if not isinstance(frame, dict):
            continue
        applied = _num(frame.get("num_context_source_skip_applied"))
        if applied is None or applied <= 0.0:
            continue
        v = _num(frame.get(key))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _read_results_ate(path: Path) -> Dict[str, Any]:
    candidates = [
        path / "results_sim3" / "results_ate.txt",
        path / "results_ate.txt",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        ate_rmse: Optional[float] = None
        ate_rot: Optional[float] = None
        for line in text.splitlines():
            parts = line.replace(":", " ").split()
            if "Average" in parts:
                idx = parts.index("Average")
                if idx + 2 < len(parts):
                    ate_rmse = _num(parts[idx + 1])
                    ate_rot = _num(parts[idx + 2])
        if ate_rmse is None:
            numeric = [_num(p) for p in text.replace(":", " ").split()]
            numeric = [v for v in numeric if v is not None]
            if len(numeric) >= 2:
                ate_rmse = numeric[-2]
                ate_rot = numeric[-1]
        return {
            "ate_results_path": str(candidate),
            "ate_rmse": ate_rmse,
            "ate_rot_deg": ate_rot,
        }
    return {
        "ate_results_path": None,
        "ate_rmse": None,
        "ate_rot_deg": None,
    }


def _ttt_action_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx in range(3):
        action = f"ttt_post_zp_w{idx}_action_delta_norm"
        committed = f"ttt_post_zp_w{idx}_committed_delta_norm"
        native = f"ttt_post_zp_w{idx}_native_delta_norm"
        action_mean = _mean(rows, action)
        committed_mean = _mean(rows, committed)
        native_mean = _mean(rows, native)
        out[f"w{idx}_action_delta_norm_mean"] = action_mean
        out[f"w{idx}_committed_delta_norm_mean"] = committed_mean
        out[f"w{idx}_native_delta_norm_mean"] = native_mean
        if action_mean is not None and committed_mean not in (None, 0.0):
            out[f"w{idx}_action_over_committed_mean_ratio"] = float(action_mean / max(abs(committed_mean), 1e-12))
    return out


def _summarize_run(path: Path) -> Dict[str, Any]:
    prior_rows = _read_jsonl(path / "prior_debug.jsonl")
    state_rows = _read_jsonl(path / "hmc_state_hash.jsonl")
    hook_rows = _read_jsonl(path / "hook_effect_summary.jsonl")
    probe_rows = _read_jsonl(path / "hmc_probe_summary.jsonl")
    control_rows = _read_jsonl(path / "hmc_control_summary.jsonl")
    semz_rows = [
        r for r in state_rows
        if _get(r, "v67_semz_available") is not None or _get(r, "v67_semz_reason") is not None
    ]
    label_counts = []
    for row in semz_rows:
        v = _num(_get(row, "v67_semz_label_count"))
        if v is not None:
            label_counts.append(v)
    control_modes = sorted({
        str(_get(r, "v67_semz_control_mode"))
        for r in semz_rows
        if _get(r, "v67_semz_control_mode") not in (None, "")
    })
    read_cues = sorted({
        str(_get(r, "v67_semz_read_cue_source"))
        for r in semz_rows
        if _get(r, "v67_semz_read_cue_source") not in (None, "")
    })
    source_rows = state_rows or prior_rows
    semcue_rows = [
        r for r in source_rows
        if _get(r, "v67_semcue_available") is not None or _get(r, "v67_semcue_reason") is not None
    ]
    semgeo_rows = [
        r for r in source_rows
        if _get(r, "v67_semgeo_available") is not None or _get(r, "v67_semgeo_reason") is not None
    ]
    gate_rows = [
        r for r in source_rows
        if _get(r, "semantic_action_chunk_gate_active") is not None
    ]
    gate_active_rows = [
        r for r in gate_rows
        if bool(_get(r, "semantic_action_chunk_gate_active"))
    ]
    gate_inactive_rows = [
        r for r in gate_rows
        if not bool(_get(r, "semantic_action_chunk_gate_active"))
    ]
    active_chunk_values = sorted(set(_int_values(gate_active_rows, "semantic_action_chunk_idx")))
    inactive_chunk_values = sorted(set(_int_values(gate_inactive_rows, "semantic_action_chunk_idx")))
    configured_chunks = sorted({
        int(v) for v in _list_values(source_rows, "semantic_action_active_chunks")
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.lstrip("-").isdigit())
    })
    out: Dict[str, Any] = {
        "run_dir": str(path),
        **_read_results_ate(path),
        "prior_rows": len(prior_rows),
        "state_rows": len(state_rows),
        "hook_rows": len(hook_rows),
        "probe_rows": len(probe_rows),
        "control_rows": len(control_rows),
        "chunks_in_prior": len({r.get("chunk_idx") for r in prior_rows if "chunk_idx" in r}),
        "chunks_in_state": len({r.get("chunk_idx") for r in state_rows if "chunk_idx" in r}),
        "dense_semantic_available_frac": (
            _frac_true(source_rows, "dense_semantic_available")
            or _frac_true(prior_rows, "prior_debug_dense_semantic_available")
        ),
        "dense_semantic_projection_nonempty_frac": (
            _frac_true(source_rows, "dense_semantic_token_projection_nonempty")
            or _frac_true(prior_rows, "prior_debug_dense_semantic_token_projection_nonempty")
        ),
        "semantic_prior_present_frac": _frac_true(source_rows, "semantic_prior_present"),
        "semantic_prior_consumed_frac": _frac_true(source_rows, "semantic_prior_consumed"),
        "semantic_action_gate_rows": len(gate_rows),
        "semantic_action_gate_active_count": len(gate_active_rows),
        "semantic_action_gate_inactive_count": len(gate_inactive_rows),
        "semantic_action_gate_active_frac": _frac_true(gate_rows, "semantic_action_chunk_gate_active"),
        "semantic_action_active_chunk_indices_observed": active_chunk_values,
        "semantic_action_inactive_chunk_indices_observed": inactive_chunk_values,
        "semantic_action_active_chunks_configured": configured_chunks,
        "semantic_action_inactive_read_cues": sorted({
            str(_get(r, "semantic_action_inactive_read_cue_source"))
            for r in source_rows
            if _get(r, "semantic_action_inactive_read_cue_source") not in (None, "")
        }),
        "v67_semz_rows": len(semz_rows),
        "v67_semz_available_frac": _frac_true(semz_rows, "v67_semz_available"),
        "v67_semz_fallback_ratio_mean": _mean(semz_rows, "v67_semz_fallback_ratio"),
        "v67_semz_read_mean_after": _mean(semz_rows, "v67_semz_read_mean_after"),
        "v67_semz_base_mean": _mean(semz_rows, "v67_semz_base_mean"),
        "v67_semz_read_delta_mean": (
            _mean(semz_rows, "v67_semz_read_mean_after") - _mean(semz_rows, "v67_semz_base_mean")
            if _mean(semz_rows, "v67_semz_read_mean_after") is not None
            and _mean(semz_rows, "v67_semz_base_mean") is not None
            else None
        ),
        "v67_semz_label_count_mean": float(np.mean(label_counts)) if label_counts else None,
        "v67_semz_control_modes": control_modes,
        "v67_semz_read_cues": read_cues,
        "v67_semcue_available_frac": _frac_true(semcue_rows, "v67_semcue_available"),
        "v67_semcue_kinds": sorted({str(v) for v in _list_values(semcue_rows, "v67_semcue_kind")}),
        "v67_semcue_random_same_distribution_frac": _frac_true(semcue_rows, "v67_semcue_random_same_distribution"),
        "v67_semcue_output_mean": _mean(semcue_rows, "v67_semcue_output_mean"),
        "v67_semcue_output_q90": _mean(semcue_rows, "v67_semcue_output_q90"),
        "v67_semcue_output_gt050_mass": _mean(semcue_rows, "v67_semcue_output_gt050_mass"),
        "v67_semcue_output_gt075_mass": _mean(semcue_rows, "v67_semcue_output_gt075_mass"),
        "v67_semcue_support_mean": _mean(semcue_rows, "v67_semcue_support_mean"),
        "v67_semcue_support_q90": _mean(semcue_rows, "v67_semcue_support_q90"),
        "v67_semcue_risk_mean": _mean(semcue_rows, "v67_semcue_risk_mean"),
        "v67_semcue_risk_q90": _mean(semcue_rows, "v67_semcue_risk_q90"),
        "v67_semcue_trust_mean": _mean(semcue_rows, "v67_semcue_trust_mean"),
        "v67_semcue_vertical_static_tokens_mean": _mean(semcue_rows, "v67_semcue_vertical_static_tokens"),
        "v67_semcue_ground_static_tokens_mean": _mean(semcue_rows, "v67_semcue_ground_static_tokens"),
        "v67_semcue_sky_tokens_mean": _mean(semcue_rows, "v67_semcue_sky_tokens"),
        "v67_semcue_vegetation_tokens_mean": _mean(semcue_rows, "v67_semcue_vegetation_tokens"),
        "v67_semcue_movable_tokens_mean": _mean(semcue_rows, "v67_semcue_movable_tokens"),
        "v67_semgeo_available_frac": _frac_true(semgeo_rows, "v67_semgeo_available"),
        "v67_semgeo_modes": sorted({str(v) for v in _list_values(semgeo_rows, "v67_semgeo_mode")}),
        "v67_semgeo_geo_kinds": sorted({str(v) for v in _list_values(semgeo_rows, "v67_semgeo_geo_kind")}),
        "v67_semgeo_sem_kinds": sorted({str(v) for v in _list_values(semgeo_rows, "v67_semgeo_sem_kind")}),
        "v67_semgeo_fusions": sorted({str(v) for v in _list_values(semgeo_rows, "v67_semgeo_fusion")}),
        "v67_semgeo_random_same_distribution_frac": _frac_true(semgeo_rows, "v67_semgeo_random_same_distribution"),
        "v67_semgeo_geo_mean": _mean(semgeo_rows, "v67_semgeo_geo_mean"),
        "v67_semgeo_geo_q90": _mean(semgeo_rows, "v67_semgeo_geo_q90"),
        "v67_semgeo_geo_gt050_mass": _mean(semgeo_rows, "v67_semgeo_geo_gt050_mass"),
        "v67_semgeo_geo_gt075_mass": _mean(semgeo_rows, "v67_semgeo_geo_gt075_mass"),
        "v67_semgeo_geo_corr_dyn": _mean(semgeo_rows, "v67_semgeo_geo_corr_dyn"),
        "v67_semgeo_geo_corr_unc": _mean(semgeo_rows, "v67_semgeo_geo_corr_unc"),
        "v67_semgeo_geo_corr_occ": _mean(semgeo_rows, "v67_semgeo_geo_corr_occ"),
        "v67_semgeo_geo_corr_conf": _mean(semgeo_rows, "v67_semgeo_geo_corr_conf"),
        "v67_semgeo_geo_corr_anchor": _mean(semgeo_rows, "v67_semgeo_geo_corr_anchor"),
        "v67_semgeo_sem_available_frac": _frac_true(semgeo_rows, "v67_semgeo_sem_available"),
        "v67_semgeo_sem_support_mean": _mean(semgeo_rows, "v67_semgeo_sem_support_mean"),
        "v67_semgeo_sem_support_q90": _mean(semgeo_rows, "v67_semgeo_sem_support_q90"),
        "v67_semgeo_sem_risk_mean": _mean(semgeo_rows, "v67_semgeo_sem_risk_mean"),
        "v67_semgeo_sem_risk_q90": _mean(semgeo_rows, "v67_semgeo_sem_risk_q90"),
        "v67_semgeo_output_mean": _mean(semgeo_rows, "v67_semgeo_output_mean"),
        "v67_semgeo_output_q90": _mean(semgeo_rows, "v67_semgeo_output_q90"),
        "v67_semgeo_output_gt050_mass": _mean(semgeo_rows, "v67_semgeo_output_gt050_mass"),
        "v67_semgeo_output_gt075_mass": _mean(semgeo_rows, "v67_semgeo_output_gt075_mass"),
        "v67_semgeo_output_corr_geo": _mean(semgeo_rows, "v67_semgeo_output_corr_geo"),
        "v67_semgeo_output_corr_sem": _mean(semgeo_rows, "v67_semgeo_output_corr_sem"),
        "v67_semgeo_output_corr_dyn": _mean(semgeo_rows, "v67_semgeo_output_corr_dyn"),
        "v67_semgeo_output_corr_conf": _mean(semgeo_rows, "v67_semgeo_output_corr_conf"),
        "v67_semgeo_output_corr_unc": _mean(semgeo_rows, "v67_semgeo_output_corr_unc"),
        "frame_attention_mean_abs_bias": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_abs_bias"]),
        "frame_attention_max_abs_bias": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "max_abs_bias"]),
        "frame_attention_mean_abs_bias_applied_mean": _hook_mean_context_applied(hook_rows, "mean_abs_bias"),
        "frame_attention_max_abs_bias_applied_mean": _hook_mean_context_applied(hook_rows, "max_abs_bias"),
        "frame_context_source_skip_applied_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "num_context_source_skip_applied"]),
        "frame_context_empty_source_events_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "num_context_empty_source_events"]),
        "frame_context_keep_ratio_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_context_source_keep_ratio"]),
        "frame_context_keep_ratio_applied_mean": _hook_mean_context_applied(hook_rows, "mean_context_source_keep_ratio"),
        "frame_context_max_source_control_tokens": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "max_context_source_control_tokens"]),
        "frame_attention_mass_removed_before_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_removed_before"]),
        "frame_attention_mass_removed_after_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_removed_after"]),
        "frame_attention_mass_actual_after_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_actual_after"]),
        "frame_attention_mass_retained_before_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_retained_before"]),
        "frame_attention_mass_retained_after_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_retained_after"]),
        "frame_attention_mass_removed_tokens_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_removed_tokens"]),
        "frame_attention_mass_retained_tokens_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_retained_tokens"]),
        "frame_attention_mass_query_sample_tokens_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_attention_mass_query_sample_tokens"]),
        "frame_context_selected_token_ratio_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_context_source_selected_token_ratio"]),
        "frame_context_selected_sky_frac_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_context_source_selected_fine_sky_frac"]),
        "frame_context_selected_structure_frac_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_context_source_selected_group_structure_frac"]),
        "frame_context_selected_lowstuff_frac_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_context_source_selected_group_lowstuff_frac"]),
        "frame_context_r2_sky_token_ratio_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v40_r2_sky_token_ratio"]),
        "frame_context_r2_sky_highd_token_ratio_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v40_r2_sky_highd_token_ratio"]),
        "frame_context_r2_source_mass_proxy_ratio_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v40_r2_source_mass_proxy_ratio"]),
        "frame_context_r2_source_mass_proxy_gate_pass_frac": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "frac_v40_r2_source_mass_proxy_gate_pass"]),
        "frame_context_r2_global_keep_proxy_after_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v40_r2_global_keep_proxy_after"]),
        "frame_context_r3_highd_threshold_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v40_r3_highd_threshold"]),
        "frame_context_v67_phase2_highd_quantile_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v67_phase2_highd_quantile"]),
        "frame_context_v67_phase2_highd_threshold_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v67_phase2_highd_threshold"]),
        "frame_context_v67_source_attention_top_quantile_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v67_source_attention_top_quantile"]),
        "frame_context_v67_source_attention_random_same_mass_frac": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "frac_v67_source_attention_random_same_mass"]),
        "frame_context_v67_source_attention_semantic_group_id_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v67_source_attention_semantic_group_id"]),
        "frame_context_v67_source_attention_group_eligible_tokens_mean": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "mean_v67_source_attention_group_eligible_tokens"]),
        "frame_context_v67_source_attention_group_missing_semantic_frac": _hook_mean(hook_rows, ["hook_effect_summary", "frame_attention", "frac_v67_source_attention_group_missing_semantic"]),
        "swa_read_max_history_tokens": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "max_history_tokens"]),
        "swa_read_overlap_source_gate_applied_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "num_swa_overlap_source_gate_applied"]),
        "swa_read_overlap_source_replace_applied_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "num_swa_overlap_source_replace_applied"]),
        "swa_read_overlap_source_gate_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_gate"]),
        "swa_read_overlap_source_gate_delta_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_gate_delta"]),
        "swa_read_overlap_source_gate_delta_max": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "max_swa_overlap_source_gate_delta"]),
        "swa_read_overlap_source_score_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_score"]),
        "swa_read_overlap_source_score_q90_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_score_q90"]),
        "swa_read_overlap_source_replace_alpha_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_replace_alpha"]),
        "swa_read_overlap_source_replace_alpha_p90_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_replace_alpha_p90"]),
        "swa_read_overlap_source_replace_score_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_replace_score"]),
        "swa_read_overlap_source_semantic_selected_tokens_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_semantic_selected_tokens"]),
        "swa_read_overlap_source_semantic_selected_ratio_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_semantic_selected_ratio"]),
        "swa_read_overlap_source_semantic_selected_index_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_semantic_selected_index_mean"]),
        "swa_read_overlap_source_semantic_selected_D_mean": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "mean_swa_overlap_source_semantic_selected_D_mean"]),
        "swa_read_overlap_source_semantic_random_same_mass_frac": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "frac_swa_overlap_source_semantic_random_same_mass"]),
        "swa_read_overlap_source_semantic_missing_labels_frac": _hook_mean(hook_rows, ["hook_effect_summary", "swa_read", "frac_swa_overlap_source_semantic_missing_labels"]),
        "ttt_write_scope_mass_mean": _mean(prior_rows, "ttt_write_scope_mass"),
        "ttt_write_scope_tokens_mean": _mean(prior_rows, "ttt_write_scope_tokens"),
        "hmc_write_score_mean": _mean(source_rows, "hmc_write_score_mean"),
        "hmc_write_selected_mass_mean": _mean(source_rows, "hmc_write_selected_mass"),
        "hmc_write_corr_score_dyn_mean": _mean(source_rows, "hmc_write_corr_score_dyn"),
        "hmc_write_corr_score_exp_dyn_mean": _mean(source_rows, "hmc_write_corr_score_exp_dyn"),
        "dynamic_mass_D_gt_001_mean": _mean(source_rows, "dynamic_mass_D_gt_001"),
        "dynamic_mass_D_gt_050_mean": _mean(source_rows, "dynamic_mass_D_gt_050"),
        "dynamic_mass_D_gt_075_mean": _mean(source_rows, "dynamic_mass_D_gt_075"),
        "old_dyn_iou_mean": _mean(source_rows, "old_dyn_iou"),
        "old_dyn_recall_mean": _mean(source_rows, "old_dyn_recall"),
        "corr_D_old_dyn_mean": _mean(source_rows, "corr_D_old_dyn"),
        "probe_ttt_write_debug_available_frac": _frac_true(control_rows, "probe_ttt_write_debug_available"),
        "probe_ttt_write_post_delta_norm_mean": _mean(control_rows, "probe_ttt_write_post_delta_norm_mean"),
        "probe_ttt_write_post_delta_norm_count_mean": _mean(control_rows, "probe_ttt_write_post_delta_norm_count"),
        "probe_ttt_write_action_delta_norm_mean": _mean(control_rows, "probe_ttt_write_action_delta_norm_mean"),
        "probe_ttt_write_native_delta_norm_mean": _mean(control_rows, "probe_ttt_write_native_delta_norm_mean"),
        "probe_ttt_write_action_native_cosine_mean": _mean(control_rows, "probe_ttt_write_action_native_cosine_mean"),
        "probe_ttt_write_native_cosine_mean": _mean(control_rows, "probe_ttt_write_native_cosine_mean"),
        "probe_ttt_write_tri_replay_applied_count_mean": _mean(control_rows, "probe_ttt_write_tri_replay_applied_count"),
        "probe_ttt_write_tri_pos_mass_mean": _mean(control_rows, "probe_ttt_write_tri_pos_mass_mean"),
        "probe_ttt_write_tri_neu_mass_mean": _mean(control_rows, "probe_ttt_write_tri_neu_mass_mean"),
        "probe_ttt_write_tri_neg_mass_mean": _mean(control_rows, "probe_ttt_write_tri_neg_mass_mean"),
        "probe_ttt_write_tri_delta_norm_mean": _mean(control_rows, "probe_ttt_write_tri_delta_norm_mean"),
        "state_double_write_safe_frac": _frac_true(control_rows, "state_double_write_safe"),
        "memory_ttt_mean_rel_diff_mean": _mean(control_rows, "memory_ttt_mean_rel_diff"),
        "memory_ttt_max_rel_diff_mean": _mean(control_rows, "memory_ttt_max_rel_diff"),
        "probe_no_commit_hash_equal_frac": _frac_true(probe_rows, "probe_no_commit_hash_equal"),
        "probe_token_count_mean": _mean(probe_rows, "probe_token_count"),
    }
    out.update(_ttt_action_summary(prior_rows))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=run_dir; repeatable")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    runs: Dict[str, Any] = {}
    for spec in args.run:
        name, path = _parse_pred(spec)
        runs[name] = _summarize_run(path)
    payload = {"runs": runs}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
