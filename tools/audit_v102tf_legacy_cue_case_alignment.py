#!/usr/bin/env python3
"""Align legacy B6-B10 cue/action evidence to the v102 case universe.

No-action Stage3 fail-forward audit: it joins legacy diagnostic/provider cues
back to v102 drift-source cases and records why they cannot be promoted to a
strict oracle or runtime action.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
OUT_DIR = ROOT / "stage3_semantic_oracle_upper_bound"
OUT_ROWS = OUT_DIR / "stage3_legacy_cue_case_alignment_rows.csv"
OUT_CASE_ROWS = OUT_DIR / "stage3_legacy_cue_case_alignment_case_rows.csv"
OUT_SUMMARY = OUT_DIR / "stage3_legacy_cue_case_alignment_summary.json"
OUT_REPORT = OUT_DIR / "legacy_cue_case_alignment_report.md"

STAGE1_ROWS = ROOT / "stage1_drift_source_autopsy/drift_source_taxonomy.csv"
STAGE2_ROWS = ROOT / "stage2_base_case_selection/base_case_rows.csv"
CLEAN_HANDOFF_SUMMARY = OUT_DIR / "stage3_clean_handoff_candidate_expansion_summary.json"
READ_BRIDGE_SUMMARY = OUT_DIR / "read_local_oracle_bridge_summary.json"

P_V98 = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
P_V101 = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")

SRC = {
    "v98_7e_summary": P_V98 / "stage7e_ttt_stable_anchor_id_hook/summary.json",
    "v98_7e_cues": P_V98 / "stage7e_ttt_stable_anchor_id_hook/cue_control_metrics.csv",
    "v98_7g_summary": P_V98 / "stage7g_anchor_id_query_head_risk_attribution/summary.json",
    "v98_7g_cues": P_V98 / "stage7g_anchor_id_query_head_risk_attribution/cue_control_metrics.csv",
    "v101_internal_qk_summary": P_V101 / "final_decision/internal_qk_actuator_reentry_summary.json",
    "v101_cache_summary": P_V101 / "final_decision/cache_topk_identity_query_reentry_summary.json",
    "v101_cache_cases": P_V101 / "final_decision/cache_topk_identity_query_reentry_case_outcomes.csv",
    "v101_f5_summary": P_V101 / "trackF5_ttt_write_to_use_state_chain/F5_summary.json",
    "v101_f5_metrics": P_V101 / "trackF5_ttt_write_to_use_state_chain/latent_support_interaction_metrics.csv",
    "v101_f5_rows": P_V101 / "trackF5_ttt_write_to_use_state_chain/write_to_use_proxy_rows.csv",
    "v101_c5_summary": P_V101 / "trackC5_identity_latent_gauge_with_support/C5_summary.json",
    "v101_c5_metrics": P_V101 / "trackC5_identity_latent_gauge_with_support/latent_support_interaction_metrics.csv",
    "v101_n3_summary": P_V101 / "trackN3_anchor_identity_graph_cleaned_targets/N3_summary.json",
    "v101_n3_rows": P_V101 / "trackN3_anchor_identity_graph_cleaned_targets/anchor_graph_pattern_rows.csv",
    "v101_dh4_summary": P_V101 / "trackDH4_read_current_support_refresh_provider/DH4_summary.json",
    "v101_dh4_rows": P_V101 / "trackDH4_read_current_support_refresh_provider/read_provider_case_rows.csv",
    "v101_q2_summary": P_V101 / "trackQ2_scale_update_admission/Q2_summary.json",
    "v101_q2_metrics": P_V101 / "trackQ2_scale_update_admission/admission_metric_summary.csv",
    "v101_geometry_summary": P_V101 / "final_decision/anchor_seed_lifecycle_geometry_observability_summary.json",
    "v101_seed_smoke_summary": P_V101 / "final_decision/stage_c_seed_geometry_smoke_target28_summary.json",
    "v101_combined_summary": P_V101 / "final_decision/combined_masklet_geometry_admission_summary.json",
    "v101_frontier_summary": P_V101 / "final_decision/strict_action_frontier_summary.json",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def split_cases(value: Any) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in str(value).replace(",", ";").split(";"):
        case_id = item.strip()
        if case_id and case_id not in seen:
            out.append(case_id)
            seen.add(case_id)
    return out


def rel(path: Path) -> str:
    return path.as_posix()


def case_seq(case_id: str) -> str:
    return case_id.split("_", 1)[0] if "_" in case_id else ""


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def context_maps() -> tuple[dict[str, dict[str, str]], set[str]]:
    stage1 = by_case(read_rows(STAGE1_ROWS))
    stage2 = by_case(read_rows(STAGE2_ROWS))
    merged = {case_id: dict(row) for case_id, row in stage1.items()}
    for case_id, row in stage2.items():
        merged.setdefault(case_id, {}).update(row)
    return merged, set(stage2)


def case_role(row: dict[str, str]) -> str:
    if not row:
        return "missing_v102_context"
    tax = str(row.get("target_taxonomy_v101", ""))
    labels = str(row.get("drift_source_labels", ""))
    primary = str(row.get("primary_drift_source", ""))
    label = str(row.get("label_original", ""))
    if tax == "HANDOFF_SCALE_GAUGE_TARGET":
        return "strict_clean_handoff_positive"
    if primary == "SAFE_GOOD" or tax == "SAFE_GOOD":
        return "safe_good_control"
    if "READ_LOCAL_SCALE" in labels or primary == "READ_LOCAL_SCALE":
        return "read_local_or_mixed"
    if "SWA_HANDOFF" in labels or primary.startswith("SWA_HANDOFF"):
        return "swa_handoff_ambiguous_or_nonclean"
    if "UNRELIABLE" in labels or "AMBIGUOUS" in tax or "GOOD_HIGH_L3" in tax or label == "good":
        return "ambiguous_or_control_like"
    return "other_v102_case"


def selected_from_cue(rows: list[dict[str, str]], cue_name: str = "") -> tuple[list[str], dict[str, Any]]:
    chosen = next((row for row in rows if row.get("cue_name") == cue_name), {}) if cue_name else {}
    if not chosen:
        chosen = next((row for row in rows if b(row.get("gate_pass"))), {})
    if not chosen and rows:
        chosen = max(rows, key=lambda row: f(row.get("balanced_accuracy"), -1.0))
    selected = split_cases(chosen.get("true_positive_cases")) + split_cases(chosen.get("false_positive_cases"))
    return selected, {
        "cue_name": chosen.get("cue_name", ""),
        "balanced_accuracy": chosen.get("balanced_accuracy", ""),
        "bad_recall": chosen.get("bad_recall", ""),
        "good_FPR": chosen.get("good_FPR", ""),
        "gate_pass": chosen.get("gate_pass", ""),
        "true_positive_cases": chosen.get("true_positive_cases", ""),
        "false_positive_cases": chosen.get("false_positive_cases", ""),
    }


def select_from_score_rows(rows: list[dict[str, str]], metrics: list[dict[str, str]]) -> list[str]:
    if not metrics:
        return []
    best = max(metrics, key=lambda row: f(row.get("balanced_accuracy"), -1.0))
    score_field = str(best.get("score_field", ""))
    if not score_field:
        return []
    selected_count = int(f(best.get("selected_case_count"), 1.0) or 1)
    reverse = str(best.get("direction", "")) != "lower_bad"
    eval_rows = [row for row in rows if row.get("target_taxonomy") in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}] or rows
    ranked = sorted(eval_rows, key=lambda row: f(row.get(score_field), -math.inf if reverse else math.inf), reverse=reverse)
    return [row.get("case_id", "") for row in ranked[:selected_count] if row.get("case_id")]


def summarize_cases(
    cases: list[str],
    context: dict[str, dict[str, str]],
    base_cases: set[str],
    strict_universe_count: int,
) -> dict[str, Any]:
    unique = split_cases(";".join(cases))
    roles: dict[str, list[str]] = {}
    for case_id in unique:
        roles.setdefault(case_role(context.get(case_id, {})), []).append(case_id)
    strict_cases = roles.get("strict_clean_handoff_positive", [])
    missing_cases = roles.get("missing_v102_context", [])
    strict_seqs = {case_seq(case_id) for case_id in strict_cases if case_seq(case_id)}
    seqs = {case_seq(case_id) for case_id in unique if case_seq(case_id)}
    global_target_blocked = strict_universe_count < 3
    alignment_pass = (
        not global_target_blocked
        and len(strict_cases) >= 3
        and len(strict_seqs) >= 2
        and not roles.get("safe_good_control")
        and not missing_cases
    )
    return {
        "selected_cases": ";".join(unique),
        "selected_case_count": len(unique),
        "selected_sequence_coverage": len(seqs),
        "selected_in_v102_context_count": sum(1 for case_id in unique if case_id in context),
        "selected_in_v102_base_count": sum(1 for case_id in unique if case_id in base_cases),
        "selected_missing_v102_context_count": len(missing_cases),
        "selected_missing_v102_context_cases": ";".join(missing_cases),
        "selected_strict_clean_handoff_count": len(strict_cases),
        "selected_strict_clean_handoff_cases": ";".join(strict_cases),
        "selected_strict_clean_sequence_coverage": len(strict_seqs),
        "selected_safe_good_count": len(roles.get("safe_good_control", [])),
        "selected_safe_good_cases": ";".join(roles.get("safe_good_control", [])),
        "selected_read_local_or_mixed_count": len(roles.get("read_local_or_mixed", [])),
        "selected_read_local_or_mixed_cases": ";".join(roles.get("read_local_or_mixed", [])),
        "selected_swa_ambiguous_count": len(roles.get("swa_handoff_ambiguous_or_nonclean", [])),
        "selected_swa_ambiguous_cases": ";".join(roles.get("swa_handoff_ambiguous_or_nonclean", [])),
        "selected_ambiguous_or_control_like_count": len(roles.get("ambiguous_or_control_like", [])),
        "selected_ambiguous_or_control_like_cases": ";".join(roles.get("ambiguous_or_control_like", [])),
        "v102_case_alignment_pass": alignment_pass,
        "global_target_coverage_blocked": global_target_blocked,
    }


def main() -> int:
    context, base_cases = context_maps()
    strict_cases = [
        case_id
        for case_id, row in context.items()
        if row.get("target_taxonomy_v101") == "HANDOFF_SCALE_GAUGE_TARGET"
    ]
    safe_good_cases = [
        case_id
        for case_id, row in context.items()
        if row.get("target_taxonomy_v101") == "SAFE_GOOD" or row.get("primary_drift_source") == "SAFE_GOOD"
    ]
    strict_universe_count = len(strict_cases)
    route_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    def add_route(
        route_id: str,
        plan_oracle: str,
        claim_type: str,
        source_path: Path,
        selected_cases: list[str],
        legacy_gate_pass: bool,
        legacy_action_gate_pass: bool,
        legacy_runtime_allowed: bool,
        blocker: str,
        next_repair: str,
        **metrics: Any,
    ) -> None:
        align = summarize_cases(selected_cases, context, base_cases, strict_universe_count)
        strict_promotion_allowed = bool(align["v102_case_alignment_pass"] and legacy_gate_pass and not legacy_action_gate_pass)
        row = {
            "route_id": route_id,
            "plan_oracle": plan_oracle,
            "claim_type": claim_type,
            "source_path": rel(source_path),
            "legacy_gate_pass": legacy_gate_pass,
            "legacy_action_gate_pass": legacy_action_gate_pass,
            "legacy_runtime_allowed": legacy_runtime_allowed,
            "strict_promotion_allowed": strict_promotion_allowed,
            "runtime_action_allowed": False,
            "blocker": blocker,
            "next_repair": next_repair,
            **align,
            **metrics,
        }
        route_rows.append(row)
        for case_id in split_cases(row["selected_cases"]):
            ctx = context.get(case_id, {})
            case_rows.append(
                {
                    "route_id": route_id,
                    "case_id": case_id,
                    "case_role": case_role(ctx),
                    "in_v102_context": bool(ctx),
                    "in_v102_base_cases": case_id in base_cases,
                    "seq": ctx.get("seq", case_seq(case_id)),
                    "label_original": ctx.get("label_original", ""),
                    "target_taxonomy_v101": ctx.get("target_taxonomy_v101", ""),
                    "primary_drift_source": ctx.get("primary_drift_source", ""),
                    "drift_source_labels": ctx.get("drift_source_labels", ""),
                    "L3_handoff_transfer_penalty_proxy": ctx.get("L3_handoff_transfer_penalty_proxy", ""),
                    "runtime_action_allowed": False,
                }
            )

    geometry = read_json(SRC["v101_geometry_summary"])
    seed = read_json(SRC["v101_seed_smoke_summary"])
    add_route(
        "B6_STAGE_C_GEOMETRY_SMOKE_TARGET28",
        "B6/B7",
        "provider_oracle_cue",
        SRC["v101_seed_smoke_summary"],
        split_cases(seed.get("best_geometry_policy_selected_cases")),
        seed.get("smoke_status") == "complete",
        False,
        b(seed.get("runtime_action_allowed")),
        "Stage-C geometry is materialized, but strict target universe has only one clean handoff positive and geometry_smoke_alignment_pass is false.",
        "Expand clean handoff target coverage and add query-head/rotation controls before strict promotion.",
        source_case_count=seed.get("selected_case_count"),
        positive_case_count=seed.get("positive_case_count"),
        safe_good_case_count=seed.get("safe_good_count"),
        lifecycle_geometry_same_payload_join_coverage=seed.get("lifecycle_geometry_same_payload_join_coverage"),
        trackv_handoff_target_case_count=geometry.get("trackv_handoff_target_case_count"),
    )

    combined = read_json(SRC["v101_combined_summary"])
    add_route(
        "B6_B7_COMBINED_MASKLET_GEOMETRY_ADMISSION",
        "B6/B7",
        "oracle_cue",
        SRC["v101_combined_summary"],
        split_cases(combined.get("clean_safe_best_policy_selected_cases")),
        b(combined.get("proxy_stage_signal_observed")),
        False,
        b(combined.get("runtime_action_allowed")),
        str(combined.get("blocker", "combined masklet/geometry admission is diagnostic only.")),
        "Do not promote a one-positive clean-safe policy; materialize target coverage and controls.",
        source_case_count=combined.get("case_count"),
        positive_case_count=combined.get("positive_case_count"),
        safe_good_case_count=combined.get("safe_good_count"),
        balanced_accuracy=combined.get("clean_safe_best_policy_balanced_accuracy"),
        selected_positive_sequence_coverage=combined.get("selected_positive_sequence_coverage"),
        same_count_margin_available=combined.get("same_count_margin_available"),
        semantic_rotation_margin_available=combined.get("semantic_rotation_margin_available"),
    )

    internal = read_json(SRC["v101_internal_qk_summary"])
    add_route(
        "B7_INTERNAL_QK_TRACKG_REENTRY",
        "B7",
        "oracle_cue_with_failed_actuator",
        SRC["v101_internal_qk_summary"],
        split_cases(internal.get("trackG_best_selected_pair_ids")),
        b(internal.get("trackG_gate_pass")),
        b(internal.get("best_family_action_surface_gate_pass")),
        b(internal.get("runtime_action_allowed")),
        str(internal.get("claim", "internal-QK cue is diagnostic but measured action families fail.")),
        "Use internal-QK as selector input only; old action bodies remain forbidden/failed.",
        trackG_best_balanced_accuracy=internal.get("trackG_best_balanced_accuracy"),
        best_bad_handoff_median_improvement=internal.get("best_bad_handoff_median_improvement"),
        candidate_action_surface_pass_row_count=internal.get("candidate_action_surface_pass_row_count"),
    )

    v98_7g = read_json(SRC["v98_7g_summary"])
    v98_7g_cases, v98_7g_metric = selected_from_cue(read_rows(SRC["v98_7g_cues"]), str(v98_7g.get("best_cue", "")))
    add_route(
        "B7_V98_QUERY_HEAD_RISK_CUE",
        "B7/B8",
        "oracle_cue",
        SRC["v98_7g_summary"],
        v98_7g_cases,
        b(v98_7g.get("gate_pass")),
        False,
        b(v98_7g.get("runtime_action_allowed")),
        str(v98_7g.get("primary_blocker", "query-head risk cue has no validated action.")),
        "Align query/head cue to v102 clean targets and add anchor/semantic/query-head controls.",
        **{f"cue_{k}": v for k, v in v98_7g_metric.items()},
    )

    v98_7e = read_json(SRC["v98_7e_summary"])
    v98_7e_cases, v98_7e_metric = selected_from_cue(read_rows(SRC["v98_7e_cues"]), str(v98_7e.get("best_cue", "")))
    add_route(
        "B8_B9_V98_ANCHOR_ID_TOPK_QUERY_CUE",
        "B8/B9",
        "carrier_oracle_cue",
        SRC["v98_7e_summary"],
        v98_7e_cases,
        b(v98_7e.get("gate_pass")),
        False,
        b(v98_7e.get("runtime_action_allowed")),
        str(v98_7e.get("primary_blocker", "anchor-id/top-k cue has no validated runtime policy.")),
        "Materialize write/cache/current residual chain and action target effect before TTT/SWA policy.",
        **{f"cue_{k}": v for k, v in v98_7e_metric.items()},
    )

    cache = read_json(SRC["v101_cache_summary"])
    cache_cases = read_rows(SRC["v101_cache_cases"])
    best_variant = str(cache.get("best_action_variant", ""))
    best_variant_rows = [row for row in cache_cases if row.get("variant") == best_variant] or cache_cases
    improvements = [f(row.get("improvement_ratio_vs_baseline")) for row in best_variant_rows if math.isfinite(f(row.get("improvement_ratio_vs_baseline")))]
    add_route(
        "B8_CACHE_TOPK_IDENTITY_ACTION_REENTRY",
        "B8",
        "negative_control_action_body_failed",
        SRC["v101_cache_summary"],
        [row.get("case_id", "") for row in best_variant_rows],
        int(f(cache.get("diagnostic_gate_pass_count"), 0.0) or 0) > 0,
        b(cache.get("best_action_gate_pass")),
        b(cache.get("runtime_action_allowed")),
        str(cache.get("claim", "cache/top-k identity cue diagnostic pass but action variants fail.")),
        "Do not repeat Stage7f/7h action bodies; keep identity as diagnostic carrier only.",
        best_action_variant=best_variant,
        best_action_median_improvement_ratio_vs_baseline=cache.get("best_action_median_improvement_ratio_vs_baseline"),
        best_variant_case_median_improvement_ratio=statistics.median(improvements) if improvements else "",
        action_variant_gate_pass_count=cache.get("action_variant_gate_pass_count"),
    )

    f5 = read_json(SRC["v101_f5_summary"])
    f5_metrics = read_rows(SRC["v101_f5_metrics"])
    f5_best = max(f5_metrics, key=lambda row: f(row.get("balanced_accuracy"), -1.0)) if f5_metrics else {}
    add_route(
        "B9_TTT_WRITE_TO_USE_F5_STATE_CHAIN",
        "B9",
        "write_to_use_oracle_proxy",
        SRC["v101_f5_summary"],
        select_from_score_rows(read_rows(SRC["v101_f5_rows"]), f5_metrics),
        f(f5.get("best_balanced_accuracy"), 0.0) >= 1.0,
        b(f5.get("gate_pass")),
        b(f5.get("runtime_action_allowed")),
        "F5 proxy separates the one clean target from safe-good controls, but target coverage is one and write/cache/current residual chain is not action-ready.",
        "Materialize write/cache/current residuals and expand clean target coverage before persistent TTT write action.",
        best_cue=f5.get("best_cue"),
        best_balanced_accuracy=f5.get("best_balanced_accuracy"),
        best_bad_recall=f5.get("best_bad_recall"),
        best_good_FPR=f5.get("best_good_FPR"),
        control_margins_available=f5_best.get("control_margins_available"),
        selected_positive_sequence_coverage=f5_best.get("selected_positive_sequence_coverage"),
    )

    c5 = read_json(SRC["v101_c5_summary"])
    c5_best = (read_rows(SRC["v101_c5_metrics"]) or [{}])[0]
    add_route(
        "B4_C5_LATENT_GAUGE_WITH_SUPPORT",
        "B4/B6",
        "negative_control_failed_cue",
        SRC["v101_c5_summary"],
        [],
        b(c5.get("gate_pass")),
        False,
        b(c5.get("runtime_action_allowed")),
        "C5 support-conditioned latent gauge cue fails bad recall and safe-good protection.",
        "Do not use R_same/S_cur/O_scale alone as selector; require true action target and controls.",
        balanced_accuracy=c5.get("balanced_accuracy"),
        bad_recall=c5.get("bad_recall"),
        good_FPR=c5.get("good_FPR"),
        metric_gate_pass=c5_best.get("gate_pass"),
    )

    n3 = read_json(SRC["v101_n3_summary"])
    n3_rows = read_rows(SRC["v101_n3_rows"])
    n3_selected = [row.get("case_id", "") for row in sorted(n3_rows, key=lambda row: f(row.get("stale_minus_fresh_score"), -math.inf), reverse=True)[:1]]
    add_route(
        "B6_N3_ANCHOR_IDENTITY_GRAPH",
        "B6",
        "negative_control_failed_cue",
        SRC["v101_n3_summary"],
        n3_selected,
        b(n3.get("gate_pass")),
        False,
        b(n3.get("runtime_action_allowed")),
        "N3 identity graph cleaned target cue fails to select the positive and has safe-good false positives.",
        "Materialize strict instance identity and role transitions before identity graph action.",
        best_balanced_accuracy=n3.get("best_balanced_accuracy"),
        best_bad_recall=n3.get("best_bad_recall"),
        best_good_FPR=n3.get("best_good_FPR"),
    )

    dh4 = read_json(SRC["v101_dh4_summary"])
    threshold = f(dh4.get("low_read_support_threshold_q25"))
    dh4_cases = [
        row.get("case_id", "")
        for row in read_rows(SRC["v101_dh4_rows"])
        if math.isfinite(threshold) and f(row.get("READ_current_support_mean")) <= threshold
    ]
    add_route(
        "B10_DH4_READ_CURRENT_SUPPORT_PROVIDER",
        "B10",
        "provider_only",
        SRC["v101_dh4_summary"],
        dh4_cases,
        f(dh4.get("handoff_low_read_support_recall"), 0.0) >= 1.0,
        False,
        b(dh4.get("runtime_action_allowed")),
        "READ current-support is provider-only; v102 READ_LOCAL/control overlap and full-sequence READ evidence remain insufficient.",
        "Feed READ support into future SWA/TTT oracle only after strict Stage3 target/control coverage is repaired.",
        READ_current_support_mean_corr_L3=dh4.get("READ_current_support_mean_corr_L3"),
        safe_good_low_read_support_fpr=dh4.get("safe_good_low_read_support_fpr"),
        read_bridge_v102_case_aligned_pass=read_json(READ_BRIDGE_SUMMARY).get("v102_case_aligned_read_local_oracle_pass"),
    )

    q2 = read_json(SRC["v101_q2_summary"])
    q2_metric = (read_rows(SRC["v101_q2_metrics"]) or [{}])[0]
    add_route(
        "B5_Q2_SCALE_UPDATE_ADMISSION",
        "B5",
        "proxy_only_negative_control",
        SRC["v101_q2_summary"],
        split_cases(q2_metric.get("true_positive_cases")) + split_cases(q2_metric.get("false_positive_cases")),
        b(q2.get("proxy_stage_pass")),
        b(q2.get("true_stage_pass")),
        b(q2.get("runtime_action_allowed")),
        str(q2.get("blocker", "Q2 true stage is unavailable.")),
        "Rebuild Q2 with true current support, scale observability, parallax-depth, and enough clean target coverage.",
        bad_recall=q2_metric.get("bad_recall"),
        good_FPR=q2_metric.get("good_FPR"),
        balanced_accuracy=q2_metric.get("balanced_accuracy"),
        selected_positive_sequence_coverage=q2_metric.get("selected_positive_sequence_coverage"),
        proxy_only=q2.get("proxy_only"),
        true_stage_pass=q2.get("true_stage_pass"),
    )

    frontier = read_json(SRC["v101_frontier_summary"])
    best_route = max(route_rows, key=lambda row: (int(row["selected_strict_clean_handoff_count"]), -int(row["selected_safe_good_count"])))
    strict_promotions = [row for row in route_rows if row["strict_promotion_allowed"]]
    summary = {
        "schema": "acl2_v102_legacy_cue_case_alignment_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "stage3_strict_repaired": False,
        "route_count": len(route_rows),
        "route_case_alignment_pass_count": sum(1 for row in route_rows if row["v102_case_alignment_pass"]),
        "strict_promotion_allowed_count": len(strict_promotions),
        "legacy_diagnostic_gate_pass_count": sum(1 for row in route_rows if row["legacy_gate_pass"]),
        "legacy_action_gate_pass_count": sum(1 for row in route_rows if row["legacy_action_gate_pass"]),
        "v102_stage1_case_count": len(context),
        "v102_base_case_count": len(base_cases),
        "v102_strict_clean_handoff_universe_count": strict_universe_count,
        "v102_strict_clean_handoff_cases": ";".join(strict_cases),
        "v102_safe_good_universe_count": len(safe_good_cases),
        "v102_safe_good_cases": ";".join(safe_good_cases),
        "clean_handoff_summary_strict_count": read_json(CLEAN_HANDOFF_SUMMARY).get("strict_clean_handoff_positive_count"),
        "max_selected_strict_clean_handoff_count": max(int(row["selected_strict_clean_handoff_count"]) for row in route_rows),
        "best_aligned_route_id": best_route["route_id"],
        "best_aligned_route_selected_strict_cases": best_route["selected_strict_clean_handoff_cases"],
        "strict_action_frontier_ready_count": frontier.get("strict_action_ready_clean_candidate_count"),
        "strict_action_frontier_blocked_reason": frontier.get("blocked_reason"),
        "blocker": (
            "Legacy diagnostic/provider cues are real, but strict promotion remains blocked: "
            "the v102 strict clean handoff universe has fewer than 3 positives, selected legacy cases mix READ/local/ambiguous roles, "
            "and audited legacy action bodies are runtime-disallowed or gate-failed."
        ),
        "next_repair": (
            "Do not repeat forbidden old action bodies.  The next meaningful repair must expand clean handoff target/control coverage "
            "or materialize a new true action-surface L3/L4 upper-bound with query-head, anchor/semantic rotation, and current-support controls."
        ),
        "outputs": {"rows": rel(OUT_ROWS), "case_rows": rel(OUT_CASE_ROWS), "report": rel(OUT_REPORT)},
    }
    write_rows(OUT_ROWS, route_rows)
    write_rows(OUT_CASE_ROWS, case_rows)
    write_json(OUT_SUMMARY, summary)
    write_text(
        OUT_REPORT,
        "# Legacy Cue Case Alignment\n\n"
        "This no-action audit aligns legacy B6-B10 evidence to v102 Stage1/Stage2 cases. "
        "Diagnostic/provider cues are preserved, but no strict oracle or runtime action is promoted.\n\n"
        f"- route_count: {summary['route_count']}\n"
        f"- legacy_diagnostic_gate_pass_count: {summary['legacy_diagnostic_gate_pass_count']}\n"
        f"- legacy_action_gate_pass_count: {summary['legacy_action_gate_pass_count']}\n"
        f"- v102_strict_clean_handoff_universe_count: {summary['v102_strict_clean_handoff_universe_count']}\n"
        f"- strict_promotion_allowed_count: {summary['strict_promotion_allowed_count']}\n"
        f"- best_aligned_route_id: {summary['best_aligned_route_id']}\n\n"
        "| route_id | oracle | claim | legacy_gate | legacy_action_gate | selected | strict | safe | read_local | missing | promotion |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {row['route_id']} | {row['plan_oracle']} | {row['claim_type']} | {row['legacy_gate_pass']} | "
            f"{row['legacy_action_gate_pass']} | {row['selected_case_count']} | {row['selected_strict_clean_handoff_count']} | "
            f"{row['selected_safe_good_count']} | {row['selected_read_local_or_mixed_count']} | "
            f"{row['selected_missing_v102_context_count']} | {row['strict_promotion_allowed']} |"
            for row in route_rows
        )
        + "\n\nConclusion:\n\n"
        + summary["blocker"]
        + "\n\nNext repair:\n\n"
        + summary["next_repair"]
        + "\n",
    )
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
