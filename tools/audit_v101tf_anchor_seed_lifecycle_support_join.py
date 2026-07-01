#!/usr/bin/env python3
"""Materialize diagnostic anchor-seed lifecycle joins for v101.

This is an audit-only bridge.  It expands no-action TTT lifecycle diagnostic
rows, joins them to existing Track U/S2/JL4 rows by (case_id, anchor_id), and
records whether the bridge is broad enough to change the final action status.
It never authorizes runtime or rewrites production track artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"

TRACE_ROOT_NAMES = [
    "stage_c_seed_bridge_target_traces",
    "stage_c_seed_bridge_clean_eval_traces_q128",
    "stage_c_seed_anchor_probe_ttt_write_smoke",
    "stage_c_seed_anchor_probe_ttt_write_diag_smoke",
    "stage_c_seed_anchor_probe_ttt_write_diag_clean_eval_q128",
    "stage_c_seed_anchor_probe_ttt_write_diag_target_q16",
]

SUPPORT_PATH = ROOT / "trackU_true_current_support/anchor_current_support_rows.csv"
STATE_PATH = ROOT / "trackS2_anchor_state_estimator/anchor_state_rows.csv"
JL4_PATH = ROOT / "trackJL4_semantic_anchor_instance_atlas/identity_resolution_gap_rows.csv"
SEED_SUPPORT_PATHS = [
    ROOT / "trackU_true_current_support/stage_c_seed_current_support_rows.csv",
    ROOT / "trackU_true_current_support/stage_c_seed_current_support_clean_eval_q128_rows.csv",
]

EXPANDED_ROWS_PATH = FINAL / "anchor_seed_lifecycle_expanded_rows.csv"
JOIN_ROWS_PATH = FINAL / "anchor_seed_lifecycle_support_join_rows.csv"
SEED_JOIN_ROWS_PATH = FINAL / "anchor_seed_lifecycle_stage_c_seed_support_join_rows.csv"
SUMMARY_PATH = FINAL / "anchor_seed_lifecycle_support_join_summary.json"
REPORT_PATH = FINAL / "anchor_seed_lifecycle_support_join_report.md"
SEED_REPORT_PATH = FINAL / "anchor_seed_lifecycle_stage_c_seed_support_join_report.md"

SCALAR_KEYS = [
    "anchor_id",
    "source_token_count",
    "topk_hit_position_count",
    "query_head_hit_frac",
    "query_head_hit_max",
    "query_head_ge50_frac",
    "query_head_ge75_frac",
    "topk_route_mass_mean",
    "topk_route_mass_max",
    "source_retention_mean",
    "source_residual_mean",
    "source_label_mode",
    "source_label_mode_frac",
    "source_stage_c_seed_global_track_idx_mode",
    "source_stage_c_seed_global_track_idx_mode_frac",
    "current_feature_residual_mean",
    "z_write_key_norm_mean",
    "z_write_key_sketch_norm_mean",
    "z_write_key_sketch_abs_mean",
    "z_write_key_sketch_dim",
    "z_cache_current_k_sketch_residual",
    "z_cache_current_v_sketch_residual",
    "z_write_current_q_sketch_residual",
    "z_write_cache_k_sketch_residual",
    "z_cache_current_k_vec_residual",
    "z_cache_current_k_native_vec_residual",
    "z_ref_current_k_native_vec_residual",
    "z_ref_cache_k_vec_residual",
    "z_cache_current_v_vec_residual",
    "z_ref_current_v_vec_residual",
    "z_ref_cache_v_vec_residual",
    "z_write_cache_hidden_vec_residual",
    "z_write_current_hidden_vec_residual",
    "z_ref_current_hidden_vec_residual",
    "z_ref_cache_hidden_vec_residual",
    "z_write_current_q_vec_residual",
    "z_write_current_k_vec_residual",
    "z_write_cache_k_vec_residual",
    "z_write_current_q_vec_projected_residual",
    "z_write_current_k_vec_projected_residual",
    "z_write_cache_k_vec_projected_residual",
    "z_write_ref_cache_k_vec_projected_residual",
    "z_cache_current_pair_count",
    "z_cache_current_cos_mean",
    "z_cache_current_cos_route_weighted_mean",
    "z_cache_current_l2_mean",
    "z_cache_current_l2_route_weighted_mean",
    "z_cache_current_v_cos_mean",
    "z_cache_current_v_cos_route_weighted_mean",
    "z_cache_current_v_l2_mean",
    "z_cache_current_v_l2_route_weighted_mean",
    "source_chunk_idx",
    "current_chunk_idx",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def as_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return ""
        value = value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    return ""


def as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def mean(values: list[Any]) -> float:
    finite = [as_float(value) for value in values if math.isfinite(as_float(value))]
    return sum(finite) / len(finite) if finite else math.nan


def max_finite(values: list[Any]) -> float:
    finite = [as_float(value) for value in values if math.isfinite(as_float(value))]
    return max(finite) if finite else math.nan


def limited_join(values: list[Any], *, limit: int = 16) -> str:
    text_values = sorted({str(value) for value in values if value not in (None, "")})
    if len(text_values) > limit:
        return ";".join(text_values[:limit]) + f";...(+{len(text_values) - limit})"
    return ";".join(text_values)


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id", "")), str(row.get("anchor_id", ""))


def seed_key_from_lifecycle(row: dict[str, Any]) -> tuple[str, str]:
    seed_value = as_float(row.get("source_stage_c_seed_global_track_idx_mode"))
    seed_text = str(int(seed_value)) if math.isfinite(seed_value) else ""
    return str(row.get("case_id", "")), seed_text


def seed_key_from_support(row: dict[str, Any]) -> tuple[str, str]:
    seed_value = row.get("stage_c_seed_global_track_idx", "")
    seed_text = str(int(as_float(seed_value))) if math.isfinite(as_float(seed_value)) else str(seed_value)
    return str(row.get("case_id", "")), seed_text


def case_id_from_trace_path(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if parts else ""


def load_lifecycle_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_stats: list[dict[str, Any]] = []
    for root_name in TRACE_ROOT_NAMES:
        root = ROOT / root_name
        payload_count = 0
        payload_with_lifecycle = 0
        payload_with_lifecycle_seed_mode = 0
        lifecycle_count = 0
        lifecycle_seed_count = 0
        if root.is_dir():
            trace_files = sorted(root.rglob("*swa_raw_transport*.pt"))
        else:
            trace_files = []
        for path in trace_files:
            payload_count += 1
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "trace_root": str(root),
                        "trace_root_name": root_name,
                        "case_id": case_id_from_trace_path(root, path),
                        "trace_payload_path": str(path),
                        "load_error": f"{type(exc).__name__}:{exc}",
                    }
                )
                continue
            lifecycle = payload.get("ttt_prev_stable_anchor_lifecycle_rows") or []
            if lifecycle:
                payload_with_lifecycle += 1
            payload_seed = False
            for idx, item in enumerate(lifecycle):
                if not isinstance(item, dict):
                    continue
                row: dict[str, Any] = {
                    "trace_root": str(root),
                    "trace_root_name": root_name,
                    "case_id": case_id_from_trace_path(root, path),
                    "trace_payload_path": str(path),
                    "lifecycle_row_idx": idx,
                }
                for scalar_key in SCALAR_KEYS:
                    row[scalar_key] = as_scalar(item.get(scalar_key))
                rows.append(row)
                lifecycle_count += 1
                if row.get("source_stage_c_seed_global_track_idx_mode") != "":
                    lifecycle_seed_count += 1
                    payload_seed = True
            if payload_seed:
                payload_with_lifecycle_seed_mode += 1
        trace_stats.append(
            {
                "trace_root_name": root_name,
                "trace_root": str(root),
                "exists": root.is_dir(),
                "trace_payload_file_count": payload_count,
                "payload_with_lifecycle_count": payload_with_lifecycle,
                "payload_with_lifecycle_stage_c_seed_mode_count": payload_with_lifecycle_seed_mode,
                "lifecycle_row_count": lifecycle_count,
                "lifecycle_rows_with_stage_c_seed_mode_count": lifecycle_seed_count,
            }
        )
    return rows, trace_stats


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row_key = key(row)
        if row_key[0] and row_key[1]:
            out[row_key].append(row)
    return out


def load_seed_support_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in SEED_SUPPORT_PATHS:
        for row in read_rows(path):
            out = dict(row)
            out["seed_support_source_file"] = path.name
            rows.append(out)
    return rows


def index_seed_support(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row_key = seed_key_from_support(row)
        if row_key[0] and row_key[1]:
            out[row_key].append(row)
    return out


def lifecycle_join_summary(lifecycle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lifecycle_join_available": bool(lifecycle_rows),
        "lifecycle_join_count": len(lifecycle_rows),
        "lifecycle_trace_roots": limited_join([row.get("trace_root_name") for row in lifecycle_rows]),
        "lifecycle_source_stage_c_seed_modes": limited_join(
            [row.get("source_stage_c_seed_global_track_idx_mode") for row in lifecycle_rows]
        ),
        "lifecycle_source_label_modes": limited_join([row.get("source_label_mode") for row in lifecycle_rows]),
        "lifecycle_source_token_count_sum": sum(int(as_float(row.get("source_token_count"))) for row in lifecycle_rows if math.isfinite(as_float(row.get("source_token_count")))),
        "lifecycle_topk_hit_position_count_sum": sum(int(as_float(row.get("topk_hit_position_count"))) for row in lifecycle_rows if math.isfinite(as_float(row.get("topk_hit_position_count")))),
        "lifecycle_query_head_hit_frac_max": max_finite([row.get("query_head_hit_frac") for row in lifecycle_rows]),
        "lifecycle_query_head_hit_max": max_finite([row.get("query_head_hit_max") for row in lifecycle_rows]),
        "lifecycle_topk_route_mass_mean": mean([row.get("topk_route_mass_mean") for row in lifecycle_rows]),
        "lifecycle_topk_route_mass_max": max_finite([row.get("topk_route_mass_max") for row in lifecycle_rows]),
        "lifecycle_current_feature_residual_mean": mean([row.get("current_feature_residual_mean") for row in lifecycle_rows]),
        "lifecycle_z_cache_current_k_sketch_residual_mean": mean(
            [row.get("z_cache_current_k_sketch_residual") for row in lifecycle_rows]
        ),
        "lifecycle_z_ref_current_hidden_vec_residual_mean": mean(
            [row.get("z_ref_current_hidden_vec_residual") for row in lifecycle_rows]
        ),
    }


def bool_any(rows: list[dict[str, str]], field: str) -> bool:
    return any(str(row.get(field, "")).strip().lower() in {"1", "true", "yes", "y"} for row in rows)


def seed_support_join_summary(seed_rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "seed_support_join_available": bool(seed_rows),
        "seed_support_join_count": len(seed_rows),
        "seed_support_source_files": limited_join([row.get("seed_support_source_file") for row in seed_rows]),
        "seed_presence_status_values": limited_join([row.get("seed_presence_status") for row in seed_rows]),
        "seed_support_quality_values": limited_join([row.get("support_quality") for row in seed_rows]),
        "seed_current_sample_token_count_max": max_finite([row.get("current_sample_token_count") for row in seed_rows]),
        "seed_cache_topk_token_count_max": max_finite([row.get("cache_topk_token_count") for row in seed_rows]),
        "seed_current_topk_slot_count_max": max_finite([row.get("current_topk_slot_count") for row in seed_rows]),
        "seed_same_seed_topk_true_count_max": max_finite([row.get("same_seed_topk_true_count") for row in seed_rows]),
        "seed_current_seed_topk_same_frac_max": max_finite([row.get("current_seed_topk_same_frac") for row in seed_rows]),
        "seed_cache_seed_same_frac_max": max_finite([row.get("cache_seed_same_frac") for row in seed_rows]),
        "seed_current_component_visible_any": bool_any(seed_rows, "current_component_visible_in_sample"),
        "seed_cache_component_available_any": bool_any(seed_rows, "cache_component_available_in_topk"),
        "seed_current_cache_same_seed_supported_any": bool_any(seed_rows, "current_cache_same_seed_supported"),
        "seed_strict_current_support_pass_any": bool_any(seed_rows, "strict_current_support_pass"),
        "seed_runtime_action_allowed_any": bool_any(seed_rows, "runtime_action_allowed"),
    }


def summarize_by_taxonomy(join_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in join_rows:
        grouped[str(row.get("target_taxonomy", ""))].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for taxonomy, rows in sorted(grouped.items()):
        joined = [row for row in rows if row.get("lifecycle_join_available") is True]
        unique_all = {key(row) for row in rows}
        unique_joined = {key(row) for row in joined}
        summary[taxonomy or "UNKNOWN"] = {
            "support_row_count": len(rows),
            "support_joined_row_count": len(joined),
            "support_unique_case_anchor_count": len(unique_all),
            "support_joined_unique_case_anchor_count": len(unique_joined),
            "support_join_unique_coverage": len(unique_joined) / len(unique_all) if unique_all else 0.0,
        }
    return summary


def main() -> None:
    support_rows = read_rows(SUPPORT_PATH)
    state_rows = read_rows(STATE_PATH)
    jl4_rows = read_rows(JL4_PATH)
    seed_support_rows = load_seed_support_rows()
    lifecycle_rows, trace_stats = load_lifecycle_rows()
    write_rows(EXPANDED_ROWS_PATH, lifecycle_rows)

    lifecycle_index = index_rows(lifecycle_rows)  # type: ignore[arg-type]
    state_index = index_rows(state_rows)
    jl4_index = index_rows(jl4_rows)
    seed_support_index = index_seed_support(seed_support_rows)

    join_rows: list[dict[str, Any]] = []
    for support in support_rows:
        row_key = key(support)
        lifecycle_for_anchor = lifecycle_index.get(row_key, [])
        state = state_index.get(row_key, [{}])[0]
        jl4 = jl4_index.get(row_key, [{}])[0]
        joined = {
            "case_id": support.get("case_id", ""),
            "boundary_id": support.get("boundary_id", ""),
            "anchor_id": support.get("anchor_id", ""),
            "semantic_label": support.get("semantic_label", ""),
            "target_taxonomy": support.get("target_taxonomy", ""),
            "support_identity_resolution_level": support.get("identity_resolution_level", ""),
            "support_proxy_only": support.get("proxy_only", ""),
            "support_quality": support.get("support_quality", ""),
            "support_source_flags": support.get("support_source_flags", ""),
            "S_cur_combined": support.get("S_cur_combined", ""),
            "R_same": support.get("R_same", ""),
            "query_hit_max": support.get("query_hit_max", ""),
            "L3_handoff_transfer_penalty_proxy": support.get("L3_handoff_transfer_penalty_proxy", ""),
            **lifecycle_join_summary(lifecycle_for_anchor),
            "state_join_available": bool(state),
            "state_O_scale": state.get("O_scale", ""),
            "state_role": state.get("role", ""),
            "state_role_confidence": state.get("role_confidence", ""),
            "state_status": state.get("state_status", ""),
            "state_K_anchor": state.get("K_anchor", ""),
            "state_P_anchor": state.get("P_anchor", ""),
            "state_Q_anchor": state.get("Q_anchor", ""),
            "state_O_anchor": state.get("O_anchor", ""),
            "jl4_gap_join_available": bool(jl4),
            "jl4_component_id": jl4.get("component_id", ""),
            "jl4_identity_resolution_level": jl4.get("identity_resolution_level", ""),
            "jl4_memory_role": jl4.get("memory_role", ""),
            "jl4_runtime_action_allowed": jl4.get("runtime_action_allowed", ""),
            "jl4_claim_level": jl4.get("claim_level", ""),
            "claim_level": "diagnostic_lifecycle_support_join_no_action",
        }
        join_rows.append(joined)
    write_rows(JOIN_ROWS_PATH, join_rows)

    seed_join_rows: list[dict[str, Any]] = []
    for lifecycle in lifecycle_rows:
        row_key = seed_key_from_lifecycle(lifecycle)
        seed_rows = seed_support_index.get(row_key, [])
        seed_join_rows.append(
            {
                "case_id": lifecycle.get("case_id", ""),
                "anchor_id": lifecycle.get("anchor_id", ""),
                "source_stage_c_seed_global_track_idx_mode": lifecycle.get(
                    "source_stage_c_seed_global_track_idx_mode", ""
                ),
                "source_label_mode": lifecycle.get("source_label_mode", ""),
                "trace_root_name": lifecycle.get("trace_root_name", ""),
                "trace_payload_path": lifecycle.get("trace_payload_path", ""),
                "lifecycle_row_idx": lifecycle.get("lifecycle_row_idx", ""),
                "source_token_count": lifecycle.get("source_token_count", ""),
                "topk_hit_position_count": lifecycle.get("topk_hit_position_count", ""),
                "query_head_hit_frac": lifecycle.get("query_head_hit_frac", ""),
                "topk_route_mass_max": lifecycle.get("topk_route_mass_max", ""),
                "current_feature_residual_mean": lifecycle.get("current_feature_residual_mean", ""),
                **seed_support_join_summary(seed_rows),
                "claim_level": "diagnostic_lifecycle_stage_c_seed_support_join_no_action",
            }
        )
    write_rows(SEED_JOIN_ROWS_PATH, seed_join_rows)

    support_unique = {key(row) for row in support_rows}
    state_unique = {key(row) for row in state_rows}
    jl4_unique = {key(row) for row in jl4_rows}
    lifecycle_unique = {key(row) for row in lifecycle_rows if key(row)[0] and key(row)[1]}
    lifecycle_seed_rows = [row for row in lifecycle_rows if seed_key_from_lifecycle(row)[0] and seed_key_from_lifecycle(row)[1]]
    lifecycle_seed_unique = {seed_key_from_lifecycle(row) for row in lifecycle_seed_rows}
    seed_support_unique = {seed_key_from_support(row) for row in seed_support_rows}
    lifecycle_seed_support_joined_rows = [
        row for row in seed_join_rows if row.get("seed_support_join_available") is True
    ]
    lifecycle_seed_support_joined_unique = lifecycle_seed_unique & seed_support_unique
    joined_unique = lifecycle_unique & support_unique
    state_joined_unique = lifecycle_unique & state_unique
    jl4_joined_unique = lifecycle_unique & jl4_unique
    joined_rows = [row for row in join_rows if row.get("lifecycle_join_available") is True]

    current_support = read_json(ROOT / "trackU_true_current_support/current_support_summary.json")
    state_summary = read_json(ROOT / "trackS2_anchor_state_estimator/state_estimator_summary.json")
    jl4_summary = read_json(ROOT / "trackJL4_semantic_anchor_instance_atlas/JL4_summary.json")
    q2_summary = read_json(ROOT / "trackQ2_scale_update_admission/Q2_summary.json")
    v_summary = read_json(ROOT / "trackV_anchor_scale_observability/observability_summary.json")
    final = read_json(FINAL / "final_decision.json")

    taxonomy_summary = summarize_by_taxonomy(join_rows)
    blockers = [
        "Lifecycle join is diagnostic-only and comes from READ_NO_ACTION/probe_ttt_write traces.",
        "Track U remains proxy_only with true_current_support_strict_pass=false.",
        "Track V remains proxy_only with gate_pass=false.",
        "Track Q2 true_stage_pass=false and proxy_stage_pass=false.",
        "Track JL4 still lacks explicit component/instance identity rescue.",
        "M4/runtime/full validation are not authorized.",
    ]
    strict_action_ready = False
    summary = {
        "schema": "acl2_v101_anchor_seed_lifecycle_support_join_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "trace_root_count": len(trace_stats),
        "trace_payload_file_count": sum(int(row["trace_payload_file_count"]) for row in trace_stats),
        "trace_payload_with_lifecycle_count": sum(int(row["payload_with_lifecycle_count"]) for row in trace_stats),
        "trace_payload_with_lifecycle_stage_c_seed_mode_count": sum(
            int(row["payload_with_lifecycle_stage_c_seed_mode_count"]) for row in trace_stats
        ),
        "trace_root_stats": trace_stats,
        "lifecycle_expanded_row_count": len(lifecycle_rows),
        "lifecycle_rows_with_stage_c_seed_mode_count": sum(
            1 for row in lifecycle_rows if row.get("source_stage_c_seed_global_track_idx_mode") != ""
        ),
        "lifecycle_unique_case_anchor_count": len(lifecycle_unique),
        "support_row_count": len(support_rows),
        "support_unique_case_anchor_count": len(support_unique),
        "support_joined_row_count": len(joined_rows),
        "support_joined_unique_case_anchor_count": len(joined_unique),
        "support_join_row_coverage": len(joined_rows) / len(support_rows) if support_rows else 0.0,
        "support_join_unique_coverage": len(joined_unique) / len(support_unique) if support_unique else 0.0,
        "state_row_count": len(state_rows),
        "state_unique_case_anchor_count": len(state_unique),
        "state_joined_row_count": sum(1 for row in state_rows if key(row) in lifecycle_unique),
        "state_joined_unique_case_anchor_count": len(state_joined_unique),
        "state_join_unique_coverage": len(state_joined_unique) / len(state_unique) if state_unique else 0.0,
        "jl4_gap_row_count": len(jl4_rows),
        "jl4_gap_unique_case_anchor_count": len(jl4_unique),
        "jl4_gap_joined_row_count": sum(1 for row in jl4_rows if key(row) in lifecycle_unique),
        "jl4_gap_joined_unique_case_anchor_count": len(jl4_joined_unique),
        "jl4_gap_join_unique_coverage": len(jl4_joined_unique) / len(jl4_unique) if jl4_unique else 0.0,
        "stage_c_seed_support_row_count": len(seed_support_rows),
        "stage_c_seed_support_unique_case_seed_count": len(seed_support_unique),
        "lifecycle_rows_with_stage_c_seed_support_join_count": len(lifecycle_seed_support_joined_rows),
        "lifecycle_unique_case_seed_count": len(lifecycle_seed_unique),
        "lifecycle_stage_c_seed_support_joined_unique_case_seed_count": len(
            lifecycle_seed_support_joined_unique
        ),
        "lifecycle_stage_c_seed_support_join_row_coverage": (
            len(lifecycle_seed_support_joined_rows) / len(lifecycle_seed_rows) if lifecycle_seed_rows else 0.0
        ),
        "lifecycle_stage_c_seed_support_join_unique_coverage": (
            len(lifecycle_seed_support_joined_unique) / len(lifecycle_seed_unique)
            if lifecycle_seed_unique
            else 0.0
        ),
        "lifecycle_seed_support_presence_status_counts": dict(
            Counter(
                row.get("seed_presence_status_values", "")
                for row in lifecycle_seed_support_joined_rows
                if row.get("seed_presence_status_values", "")
            )
        ),
        "lifecycle_seed_support_quality_counts": dict(
            Counter(
                row.get("seed_support_quality_values", "")
                for row in lifecycle_seed_support_joined_rows
                if row.get("seed_support_quality_values", "")
            )
        ),
        "lifecycle_seed_support_strict_current_support_pass_any_count": sum(
            1 for row in lifecycle_seed_support_joined_rows if row.get("seed_strict_current_support_pass_any") is True
        ),
        "lifecycle_seed_support_runtime_action_allowed_any_count": sum(
            1 for row in lifecycle_seed_support_joined_rows if row.get("seed_runtime_action_allowed_any") is True
        ),
        "stage_c_seed_support_strict_current_support_ready": False,
        "support_joined_taxonomy_counts": dict(Counter(str(row.get("target_taxonomy", "")) for row in joined_rows)),
        "taxonomy_summary": taxonomy_summary,
        "trackU_gate_pass": current_support.get("gate_pass"),
        "trackU_proxy_only": current_support.get("proxy_only"),
        "trackU_true_current_support_strict_pass": current_support.get("true_current_support_strict_pass"),
        "trackV_gate_pass": v_summary.get("gate_pass"),
        "trackV_proxy_only": v_summary.get("proxy_only"),
        "trackS2_gate_pass": state_summary.get("gate_pass"),
        "trackS2_proxy_only": state_summary.get("proxy_only"),
        "trackJL4_gate_pass": jl4_summary.get("gate_pass"),
        "trackJL4_distinguishes_region_label_from_instance": jl4_summary.get(
            "distinguishes_region_label_from_instance"
        ),
        "trackQ2_gate_pass": q2_summary.get("gate_pass"),
        "trackQ2_proxy_only": q2_summary.get("proxy_only"),
        "trackQ2_proxy_stage_pass": q2_summary.get("proxy_stage_pass"),
        "trackQ2_true_stage_pass": q2_summary.get("true_stage_pass"),
        "final_runtime_action_allowed": final.get("runtime_action_allowed"),
        "final_full_validation_run": final.get("full_validation_run"),
        "trackU_strict_current_support_ready": False,
        "jl4_identity_rescue_available": False,
        "q2_true_stage_ready": False,
        "strict_action_ready": strict_action_ready,
        "status": "complete_diagnostic_lifecycle_support_join_no_action",
        "blockers": blockers,
        "expanded_rows_path": str(EXPANDED_ROWS_PATH),
        "join_rows_path": str(JOIN_ROWS_PATH),
        "seed_join_rows_path": str(SEED_JOIN_ROWS_PATH),
    }
    write_json(SUMMARY_PATH, summary)

    report_lines = [
        "# ACL2 v101 Anchor-Seed Lifecycle Support Join",
        "",
        "This report is audit-only. It materializes diagnostic lifecycle rows and joins them to Track U/S2/JL4 by `(case_id, anchor_id)`. It does not authorize runtime action.",
        "",
        "## Key Counts",
        "",
        f"- trace_payload_file_count: {summary['trace_payload_file_count']}",
        f"- lifecycle_expanded_row_count: {summary['lifecycle_expanded_row_count']}",
        f"- lifecycle_rows_with_stage_c_seed_mode_count: {summary['lifecycle_rows_with_stage_c_seed_mode_count']}",
        f"- lifecycle_unique_case_anchor_count: {summary['lifecycle_unique_case_anchor_count']}",
        f"- support_row_count: {summary['support_row_count']}",
        f"- support_joined_row_count: {summary['support_joined_row_count']}",
        f"- support_joined_unique_case_anchor_count: {summary['support_joined_unique_case_anchor_count']}",
        f"- support_join_unique_coverage: {summary['support_join_unique_coverage']}",
        f"- state_joined_unique_case_anchor_count: {summary['state_joined_unique_case_anchor_count']}",
        f"- jl4_gap_joined_unique_case_anchor_count: {summary['jl4_gap_joined_unique_case_anchor_count']}",
        f"- lifecycle_stage_c_seed_support_join_row_coverage: {summary['lifecycle_stage_c_seed_support_join_row_coverage']}",
        f"- lifecycle_stage_c_seed_support_join_unique_coverage: {summary['lifecycle_stage_c_seed_support_join_unique_coverage']}",
        "",
        "## Gate Status",
        "",
        f"- Track U strict current support ready: {summary['trackU_strict_current_support_ready']}",
        f"- Track V gate pass: {summary['trackV_gate_pass']}",
        f"- Track S2 gate pass: {summary['trackS2_gate_pass']}",
        f"- Track Q2 true stage pass: {summary['trackQ2_true_stage_pass']}",
        f"- JL4 identity rescue available: {summary['jl4_identity_rescue_available']}",
        f"- strict_action_ready: {summary['strict_action_ready']}",
        "",
        "## Blockers",
        "",
        *[f"- {item}" for item in blockers],
        "",
        "## Artifacts",
        "",
        f"- `{EXPANDED_ROWS_PATH}`",
        f"- `{JOIN_ROWS_PATH}`",
        f"- `{SEED_JOIN_ROWS_PATH}`",
        f"- `{SUMMARY_PATH}`",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    seed_report_lines = [
        "# ACL2 v101 Anchor-Seed Lifecycle Stage-C Seed Support Join",
        "",
        "This report is audit-only. It joins lifecycle `source_stage_c_seed_global_track_idx_mode` to Stage-C seed current-support rows by `(case_id, seed_id)`.",
        "",
        "## Key Counts",
        "",
        f"- lifecycle_rows_with_seed: {len(lifecycle_seed_rows)}",
        f"- lifecycle_unique_case_seed_count: {summary['lifecycle_unique_case_seed_count']}",
        f"- stage_c_seed_support_row_count: {summary['stage_c_seed_support_row_count']}",
        f"- stage_c_seed_support_unique_case_seed_count: {summary['stage_c_seed_support_unique_case_seed_count']}",
        f"- lifecycle_rows_with_stage_c_seed_support_join_count: {summary['lifecycle_rows_with_stage_c_seed_support_join_count']}",
        f"- lifecycle_stage_c_seed_support_joined_unique_case_seed_count: {summary['lifecycle_stage_c_seed_support_joined_unique_case_seed_count']}",
        f"- lifecycle_stage_c_seed_support_join_row_coverage: {summary['lifecycle_stage_c_seed_support_join_row_coverage']}",
        f"- lifecycle_stage_c_seed_support_join_unique_coverage: {summary['lifecycle_stage_c_seed_support_join_unique_coverage']}",
        f"- lifecycle_seed_support_strict_current_support_pass_any_count: {summary['lifecycle_seed_support_strict_current_support_pass_any_count']}",
        f"- lifecycle_seed_support_runtime_action_allowed_any_count: {summary['lifecycle_seed_support_runtime_action_allowed_any_count']}",
        "",
        "## Interpretation",
        "",
        "Seed-level provenance is join-complete, but the joined support rows remain diagnostic/proxy and do not authorize runtime action.",
        "",
        "## Artifacts",
        "",
        f"- `{SEED_JOIN_ROWS_PATH}`",
        f"- `{SUMMARY_PATH}`",
    ]
    SEED_REPORT_PATH.write_text("\n".join(seed_report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "lifecycle_expanded_row_count": summary["lifecycle_expanded_row_count"],
                "lifecycle_unique_case_anchor_count": summary["lifecycle_unique_case_anchor_count"],
                "support_joined_unique_case_anchor_count": summary["support_joined_unique_case_anchor_count"],
                "support_join_unique_coverage": summary["support_join_unique_coverage"],
                "lifecycle_stage_c_seed_support_join_unique_coverage": summary[
                    "lifecycle_stage_c_seed_support_join_unique_coverage"
                ],
                "strict_action_ready": summary["strict_action_ready"],
                "runtime_action_allowed": summary["runtime_action_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
