#!/usr/bin/env python3
"""Build a broader-carrier re-entry audit for ACL2 v101 Outcome D.

The audit is read-only over existing ACL2 artifacts. It does not claim a new
runtime action, and it does not modify v101 gate states.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


RESULTS = Path("results")
V101_ROOT = RESULTS / "acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission"
OUT = V101_ROOT / "final_decision"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_text(path: Path, max_chars: int = 1200) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace").strip()
    return text[:max_chars]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def value(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def compact(value_obj: Any, max_len: int = 700) -> str:
    if isinstance(value_obj, (dict, list)):
        text = json.dumps(value_obj, sort_keys=True)
    else:
        text = str(value_obj)
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def version_from_root(root: Path) -> str:
    match = re.search(r"acl2_v(\d+)", root.name)
    return match.group(1) if match else ""


def carrier_family(root: Path) -> str:
    name = root.name.lower()
    if "same_space" in name:
        return "same_space_semantic_anchor_state"
    if "semantic_anchor" in name:
        return "semantic_anchor_identity_or_state"
    if "cache_topk" in name:
        return "swa_cache_topk_identity"
    if "semantic_scale_evidence" in name:
        return "semantic_scale_evidence_cache_or_retention"
    if "vggt4d" in name:
        return "read_swa_ttt_instrumentation"
    if "multiroute" in name:
        return "multiroute_read_swa_ttt_internal_cues"
    if "gauge_failure_localization" in name:
        return "merge_gauge_object_source_localization"
    if "object_identity" in name:
        return "semantic_object_identity_carrier"
    if "policy_carrier" in name:
        return "semantic_policy_carrier"
    if "topology" in name:
        return "semantic_topology_or_regime"
    if "scale_mode" in name:
        return "scale_mode_observability"
    if "latent" in name or "ruler" in name:
        return "latent_memory_ruler"
    if "swa_carrier" in name:
        return "swa_carrier"
    return "other_acl2_carrier"


def load_final_summary(root: Path) -> tuple[dict[str, Any], list[str]]:
    candidates = [
        root / "final_decision" / "summary.json",
        root / "final_decision" / "final_decision.json",
        root / "decision_matrix" / "summary.json",
        root / "report_final" / "final_decision.json",
    ]
    for path in candidates:
        data = read_json(path)
        if data:
            return data, [str(path)]
    return {}, []


def scan_historical_roots() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in sorted(RESULTS.glob("acl2_v*")):
        data, evidence = load_final_summary(root)
        recommendation = ""
        rec_files = [
            root / "final_decision" / "next_route_recommendation.md",
            root / "final_decision" / "next_attempt_recommendation.md",
            root / "decision_matrix" / "next_route_recommendation.md",
            root / "report_final" / "next_route_recommendation.md",
        ]
        for path in rec_files:
            text = read_text(path, 500)
            if text:
                recommendation = text
                evidence.append(str(path))
                break

        final_status = value(data, "final_taxonomy", "final_status", "status", default="NO_UNIFIED_FINAL_SUMMARY")
        method_success = truthy(value(data, "method_success", "full_method_success", default=False))
        runtime_allowed = truthy(value(data, "runtime_action_allowed", default=False))
        full_success = truthy(value(data, "full_method_success", default=False))
        primary_blocker = value(data, "primary_blocker", "blocker", default="")
        signal_bits: list[str] = []
        for key in sorted(data):
            if key.endswith("gate_pass") and truthy(data[key]):
                signal_bits.append(f"{key}=true")
            if key in {"mechanism_success", "diagnostic_success"} and truthy(data[key]):
                signal_bits.append(f"{key}=true")

        rows.append(
            {
                "root": str(root),
                "version": version_from_root(root),
                "carrier_family": carrier_family(root),
                "final_status": compact(final_status, 180),
                "method_success": method_success,
                "runtime_action_allowed": runtime_allowed,
                "full_method_success": full_success,
                "positive_signal_fields": ";".join(signal_bits[:20]),
                "primary_blocker": compact(primary_blocker, 500),
                "next_recommendation_excerpt": compact(recommendation, 500),
                "evidence_files": ";".join(evidence),
            }
        )
    return rows


def build_reentry_routes() -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []

    v94_root = RESULTS / "acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control"
    v94 = read_json(v94_root / "report_final" / "final_decision.json")
    v94_metrics = v94.get("key_metrics", {})
    selector_audit = read_json(OUT / "merge_gauge_selector_reentry_summary.json")
    rich_selector_audit = read_json(OUT / "merge_gauge_rich_selector_reentry_summary.json")
    selector_followup = {}
    if selector_audit:
        selector_followup = {
            "simple_numeric_selector_policy_count": selector_audit.get("candidate_policy_count"),
            "simple_numeric_selector_passing_count": selector_audit.get("passing_candidate_count"),
            "simple_numeric_best_policy": selector_audit.get("best_policy_id"),
            "simple_numeric_best_margin_vs_random_p95": selector_audit.get(
                "best_bad_median_margin_vs_random_p95"
            ),
        }
    rich_selector_followup = {}
    if rich_selector_audit:
        rich_selector_followup = {
            "rich_selector_candidate_policy_count": rich_selector_audit.get("candidate_policy_count"),
            "rich_selector_retrospective_passing_count": rich_selector_audit.get(
                "retrospective_passing_candidate_count"
            ),
            "rich_selector_promotion_floor_passing_count": rich_selector_audit.get(
                "promotion_floor_passing_candidate_count"
            ),
            "rich_selector_action_authorized_candidate_count": rich_selector_audit.get(
                "action_authorized_candidate_count"
            ),
            "rich_selector_passing_selected_signature_count": rich_selector_audit.get(
                "passing_selected_signature_count"
            ),
            "rich_selector_best_policy": rich_selector_audit.get("best_policy_id"),
            "rich_selector_best_BA": rich_selector_audit.get("best_balanced_accuracy"),
            "rich_selector_best_margin_vs_random_p95": rich_selector_audit.get(
                "best_bad_median_margin_vs_random_p95"
            ),
            "rich_selector_best_action_authorized": rich_selector_audit.get("best_action_authorized"),
            "rich_selector_best_action_block_reason": rich_selector_audit.get("best_action_block_reason"),
            "rich_selector_best_selected_sequences": rich_selector_audit.get("best_selected_sequence_ids"),
            "rich_selector_leave_one_selected_sequence_out_pass_count": rich_selector_audit.get(
                "best_leave_one_selected_sequence_out_pass_count"
            ),
            "rich_selector_leave_one_selected_sequence_out_trial_count": rich_selector_audit.get(
                "best_leave_one_selected_sequence_out_trial_count"
            ),
            "rich_selector_stability_status": rich_selector_audit.get("best_signal_stability_status"),
            "rich_selector_loso_holdout_status": rich_selector_audit.get("loso_holdout_status"),
            "rich_selector_loso_train_split_with_retrospective_pass_count": rich_selector_audit.get(
                "loso_train_split_with_retrospective_pass_count"
            ),
            "rich_selector_loso_train_split_with_promotion_floor_pass_count": rich_selector_audit.get(
                "loso_train_split_with_promotion_floor_pass_count"
            ),
        }
    if rich_selector_audit and rich_selector_audit.get("retrospective_passing_candidate_count"):
        merge_gauge_next = (
            "The richer v101 retrospective selector screen found a weak existing-row signal, but it is not "
            "runtime-authorized: no candidate meets the stricter promotion floor, the best policy has low BA/tiny "
            "margin, fails sequence-drop stability, and LOSO train-side searches find no retrospective selector. "
            "The next merge/gauge work should freeze a predeclared family/holdout rerun with measured controls; "
            "do not promote this screen directly."
        )
    else:
        merge_gauge_next = (
            "Simple native/carrier numeric selector thresholds did not pass if the v101 follow-up audit is present; "
            "next merge/gauge work needs a richer predeclared selector family and measured selection controls, or should "
            "move to the V83/V85 internal-QK cue route."
        )
    routes.append(
        {
            "priority": 1,
            "route_id": "merge_gauge_actuator_selector_reentry",
            "route_label": "Merge/gauge carrier with non-semantic selector redesign",
            "evidence_strength": "measured_actuator_signal_but_selector_blocked",
            "positive_evidence": compact(
                {
                    "phase3_formal_repaired_gate_pass": v94.get("phase3_formal_repaired_gate_pass"),
                    "selected_carrier_body": v94_metrics.get("phase3_formal_selected_carrier_body"),
                    "selected_actuator_variant": v94_metrics.get("phase3_formal_selected_actuator_variant"),
                    "phase3s_actuator_probe_gate_pass": v94_metrics.get("phase3s_actuator_probe_gate_pass"),
                    "bad_median_I_J_runtime_proxy": v94_metrics.get("phase3_formal_bad_median_I_J_runtime_proxy"),
                    "sequence_coverage": v94.get("phase3_formal_merge_alpha_sensitivity", {}).get("sequence_coverage"),
                }
            ),
            "blocking_evidence": compact(
                {
                    "runtime_action_allowed": v94.get("runtime_action_allowed"),
                    "phase6_object_source_action_surface_gate_pass": v94.get(
                        "phase6_object_source_action_surface_gate_pass"
                    ),
                    "phase6_action_surface_actual_minus_best_control": v94_metrics.get(
                        "phase6_action_surface_actual_minus_best_control"
                    ),
                    "blocker": v94.get("blocker"),
                    "semantic_not_specific": v94_metrics.get("phase6_action_surface_semantic_not_specific"),
                    **selector_followup,
                    **rich_selector_followup,
                }
            ),
            "action_allowed": False,
            "recommended_next_step": merge_gauge_next,
            "evidence_files": ";".join(
                [
                    str(v94_root / "report_final" / "final_decision.json"),
                    str(v94_root / "report_final" / "next_route_recommendation.md"),
                    str(OUT / "merge_gauge_selector_reentry_summary.json") if selector_audit else "",
                    str(OUT / "merge_gauge_rich_selector_reentry_summary.json") if rich_selector_audit else "",
                ]
            ),
        }
    )

    v95_root = RESULTS / "acl2_v95tf_multiroute_semantic_memory_evidence_control"
    v95_g = read_json(v95_root / "trackG_swa_internal_cue_eval_v1" / "summary.json")
    v95_e = read_json(v95_root / "trackE_internal_cue_action_surface_v1" / "summary.json")
    internal_qk_audit = read_json(OUT / "internal_qk_actuator_reentry_summary.json")
    internal_qk_followup = {}
    if internal_qk_audit:
        internal_qk_followup = {
            "audited_actuator_family_count": internal_qk_audit.get("family_count"),
            "audited_metric_candidate_row_count": internal_qk_audit.get("metric_candidate_row_count"),
            "audited_action_surface_passing_family_count": internal_qk_audit.get(
                "action_surface_passing_family_count"
            ),
            "audited_best_family": internal_qk_audit.get("best_family_id"),
            "audited_best_bad_handoff_median_improvement": internal_qk_audit.get(
                "best_bad_handoff_median_improvement"
            ),
            "audited_global_per_pair_bad_ge_0p05_count": internal_qk_audit.get(
                "global_per_pair_bad_ge_0p05_count"
            ),
        }
    internal_qk_next = (
        "Follow-up audit found 0 measured actuator families passing if present; design a genuinely new "
        "predeclared actuator family and measured controls before any runtime pilot."
        if internal_qk_audit and not internal_qk_audit.get("action_surface_passing_family_count")
        else "Keep the internal cue as a carrier candidate, but design or enumerate a new measured actuator; "
        "prior measured variants did not reach the 5pct handoff-improvement gate."
    )
    g_best = v95_g.get("best_method_safe_internal", {})
    e_best = v95_e.get("best_cue_variant", {})
    routes.append(
        {
            "priority": 2,
            "route_id": "v83_v85_internal_qk_swa_cue_reentry",
            "route_label": "V83/V85 internal cue and QK feature carrier",
            "evidence_strength": "strong_offline_selector_but_measured_action_surface_failed",
            "positive_evidence": compact(
                {
                    "trackG_gate_pass": v95_g.get("gate_pass"),
                    "cue_id": g_best.get("cue_id"),
                    "balanced_accuracy": g_best.get("balanced_accuracy"),
                    "bad_recall": g_best.get("bad_recall"),
                    "good_FPR": g_best.get("good_FPR"),
                    "selected_sequence_coverage": g_best.get("selected_sequence_coverage"),
                    "selected_pair_ids": g_best.get("selected_pair_ids"),
                }
            ),
            "blocking_evidence": compact(
                {
                    "trackE_gate_pass": v95_e.get("gate_pass"),
                    "blocker": v95_e.get("blocker"),
                    "candidate_action_surface_gate_pass": e_best.get("candidate_action_surface_gate_pass"),
                    "bad_handoff_median_improvement": e_best.get("bad_handoff_median_improvement"),
                    "actual_minus_best_same_count_control": e_best.get("actual_minus_best_same_count_control"),
                    **internal_qk_followup,
                }
            ),
            "action_allowed": False,
            "recommended_next_step": internal_qk_next,
            "evidence_files": ";".join(
                [
                    str(v95_root / "trackG_swa_internal_cue_eval_v1" / "summary.json"),
                    str(v95_root / "trackE_internal_cue_action_surface_v1" / "summary.json"),
                    str(OUT / "internal_qk_actuator_reentry_summary.json") if internal_qk_audit else "",
                ]
            ),
        }
    )

    v97_root = RESULTS / "acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control"
    v98_root = RESULTS / "acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control"
    v97 = read_json(v97_root / "final_decision" / "summary.json")
    v98 = read_json(v98_root / "final_decision" / "summary.json")
    cache_query_audit = read_json(OUT / "cache_topk_identity_query_reentry_summary.json")
    cache_query_followup = {}
    if cache_query_audit:
        cache_query_followup = {
            "audited_action_variant_count": cache_query_audit.get("action_variant_count"),
            "audited_action_case_outcome_row_count": cache_query_audit.get("action_case_outcome_row_count"),
            "audited_action_variant_gate_pass_count": cache_query_audit.get("action_variant_gate_pass_count"),
            "audited_best_action_variant": cache_query_audit.get("best_action_variant"),
            "audited_best_median_improvement_ratio": cache_query_audit.get(
                "best_action_median_improvement_ratio_vs_baseline"
            ),
            "audited_best_improved_cases": cache_query_audit.get("best_action_improved_ate_case_count"),
            "audited_best_worse_cases": cache_query_audit.get("best_action_worse_ate_case_count"),
        }
    cache_query_next = (
        "Follow-up audit found 0 existing Stage7f/Stage7h action variants passing if present; any cache/top-k "
        "query-head re-entry needs a new predeclared action design and measured controls."
        if cache_query_audit and not cache_query_audit.get("action_variant_gate_pass_count")
        else "Re-enter as a causal cache/top-k or query-head-control carrier search with non-GT controls; "
        "do not repeat v98 soft query action directly."
    )
    routes.append(
        {
            "priority": 3,
            "route_id": "cache_topk_identity_query_head_reentry",
            "route_label": "SWA cache/top-k stability and query-head identity carrier",
            "evidence_strength": "diagnostic_identity_signal_but_action_pilots_no_go",
            "positive_evidence": compact(
                {
                    "v97_trackE2_gate_pass": v97.get("trackE2_gate_pass"),
                    "v97_trackK_any_eligibility_gate_pass": v97.get("trackK_any_eligibility_gate_pass"),
                    "v98_stage7g_query_head_gate_pass": v98.get("stage7g_query_head_gate_pass"),
                    "v98_stage7g_anchor_id_query_head_risk_attribution_gate_pass": v98.get(
                        "stage7g_anchor_id_query_head_risk_attribution_gate_pass"
                    ),
                    "v98_stage7e_write_to_use_chain_available": v98.get("stage7e_write_to_use_chain_available"),
                    "v98_stage7g_best_cue": v98.get("stage7g_best_cue"),
                }
            ),
            "blocking_evidence": compact(
                {
                    "v97_runtime_action_allowed": v97.get("runtime_action_allowed"),
                    "v98_runtime_action_run": v98.get("runtime_action_run"),
                    "v98_stage7f_action_pilot_gate_pass": v98.get("stage7f_prev_ttt_anchor_gate_action_pilot_gate_pass"),
                    "v98_stage7h_query_soft_action_pilot_gate_pass": v98.get(
                        "stage7h_prev_ttt_anchor_query_soft_action_pilot_gate_pass"
                    ),
                    "v98_primary_blocker": v98.get("primary_blocker"),
                    **cache_query_followup,
                }
            ),
            "action_allowed": False,
            "recommended_next_step": cache_query_next,
            "evidence_files": ";".join(
                [
                    str(v97_root / "final_decision" / "summary.json"),
                    str(v98_root / "final_decision" / "summary.json"),
                    str(OUT / "cache_topk_identity_query_reentry_summary.json") if cache_query_audit else "",
                ]
            ),
        }
    )

    v96_root = RESULTS / "acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control"
    v96 = read_json(v96_root / "final_decision" / "summary.json")
    remaining_closure = read_json(OUT / "remaining_reentry_route_closure_summary.json")
    routes.append(
        {
            "priority": 4,
            "route_id": "read_provider_only_reentry",
            "route_label": "READ as instrumentation/provider, not runtime action",
            "evidence_strength": "local_mechanism_signal_full_no_go",
            "positive_evidence": compact(
                {
                    "mechanism_success": v96.get("mechanism_success"),
                    "diagnostic_success": v96.get("diagnostic_success"),
                    "final_status": v96.get("final_status"),
                    "closure_read_provider_mechanism_success": remaining_closure.get(
                        "read_provider_mechanism_success"
                    ),
                }
            ),
            "blocking_evidence": compact(
                {
                    "method_success": v96.get("method_success"),
                    "full_method_success": v96.get("full_method_success"),
                    "runtime_action_allowed": v96.get("runtime_action_allowed"),
                    "primary_blocker": v96.get("primary_blocker"),
                    "closure_stage7_candidate_count": remaining_closure.get(
                        "read_provider_stage7_candidate_count"
                    ),
                    "closure_stage7_gate_pass": remaining_closure.get("read_provider_stage7_gate_pass"),
                    "closure_stage7_best_delta_ate": remaining_closure.get(
                        "read_provider_stage7_best_delta_ate"
                    ),
                    "closure_stage7_best_delta_final_error": remaining_closure.get(
                        "read_provider_stage7_best_delta_final_error"
                    ),
                    "closure_stage7_strict_reason": remaining_closure.get("read_provider_stage7_strict_reason"),
                }
            ),
            "action_allowed": False,
            "recommended_next_step": (
                "Use READ only to materialize current-support or before/after latent dumps tied to global final-error/yaw; "
                "do not revive READ L07 as the main runtime method."
            ),
            "evidence_files": ";".join(
                [
                    str(v96_root / "final_decision" / "summary.json"),
                    str(OUT / "remaining_reentry_route_closure_summary.json") if remaining_closure else "",
                ]
            ),
        }
    )

    v101 = read_json(V101_ROOT / "final_decision" / "summary.json")
    v101_new = read_json(V101_ROOT / "final_decision" / "new_v100_schema_universe_feasibility_summary.json")
    routes.append(
        {
            "priority": 5,
            "route_id": "semantic_anchor_state_freeze_until_new_universe",
            "route_label": "Semantic anchor state route held behind target universe",
            "evidence_strength": "instrumentation_available_action_blocked",
            "positive_evidence": compact(
                {
                    "trackU_gate_pass": v101.get("trackU_gate_pass"),
                    "trackW_gate_pass": v101.get("trackW_gate_pass"),
                    "core_v100_schema_ready_clean_candidate_count": v101_new.get(
                        "core_v100_schema_ready_clean_candidate_count"
                    ),
                    "closure_strict_action_ready_clean_candidate_count": remaining_closure.get(
                        "semantic_anchor_strict_action_ready_clean_candidate_count"
                    ),
                }
            ),
            "blocking_evidence": compact(
                {
                    "trackT_gate_pass": v101.get("trackT_gate_pass"),
                    "trackQ2_true_stage_pass": v101.get("trackQ2_true_stage_pass"),
                    "trackV_gate_pass": v101.get("trackV_gate_pass"),
                    "strict_action_ready_clean_candidate_count": v101_new.get(
                        "strict_action_ready_clean_candidate_count"
                    ),
                    "new_universe_available_from_existing_artifacts": v101_new.get(
                        "new_universe_available_from_existing_artifacts"
                    ),
                    "closure_new_universe_available_from_existing_artifacts": remaining_closure.get(
                        "semantic_anchor_new_universe_available_from_existing_artifacts"
                    ),
                }
            ),
            "action_allowed": False,
            "recommended_next_step": (
                "Freeze v101 runtime action until a new v100-schema clean handoff target universe is acquired; "
                "do not tune current semantic-anchor selectors."
            ),
            "evidence_files": ";".join(
                [
                    str(V101_ROOT / "final_decision" / "summary.json"),
                    str(V101_ROOT / "final_decision" / "new_v100_schema_universe_feasibility_summary.json"),
                    str(OUT / "remaining_reentry_route_closure_summary.json") if remaining_closure else "",
                ]
            ),
        }
    )

    return routes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    historical_rows = scan_historical_roots()
    reentry_routes = build_reentry_routes()

    historical_fields = [
        "root",
        "version",
        "carrier_family",
        "final_status",
        "method_success",
        "runtime_action_allowed",
        "full_method_success",
        "positive_signal_fields",
        "primary_blocker",
        "next_recommendation_excerpt",
        "evidence_files",
    ]
    route_fields = [
        "priority",
        "route_id",
        "route_label",
        "evidence_strength",
        "positive_evidence",
        "blocking_evidence",
        "action_allowed",
        "recommended_next_step",
        "evidence_files",
    ]

    write_csv(OUT / "broader_carrier_historical_root_scan.csv", historical_rows, historical_fields)
    write_csv(OUT / "broader_carrier_reentry_routes.csv", reentry_routes, route_fields)

    summary = {
        "schema": "acl2_v101_broader_carrier_reentry_audit_v1",
        "historical_root_count": len(historical_rows),
        "roots_with_unified_or_fallback_summary": sum(
            1 for row in historical_rows if row["final_status"] != "NO_UNIFIED_FINAL_SUMMARY"
        ),
        "reentry_route_count": len(reentry_routes),
        "action_allowed_route_count": sum(1 for row in reentry_routes if row["action_allowed"]),
        "runtime_action_allowed": False,
        "full_validation_allowed": False,
        "v101_goal_achieved": False,
        "top_reentry_route": reentry_routes[0]["route_id"] if reentry_routes else "",
        "stop_condition_interpretation": (
            "Outcome D is supported: semantic-anchor action remains blocked, so the next work is broader "
            "carrier search. This audit only ranks re-entry hypotheses; it does not authorize runtime."
        ),
    }
    with (OUT / "broader_carrier_reentry_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    report = [
        "# Broader Carrier Re-entry Audit",
        "",
        "This audit implements the v101 Outcome D handoff: stop semantic-anchor action work and return to broader carrier search.",
        "",
        "## Summary",
        "",
        f"- historical ACL2 roots scanned: `{summary['historical_root_count']}`",
        f"- roots with usable final/decision summary: `{summary['roots_with_unified_or_fallback_summary']}`",
        f"- re-entry routes emitted: `{summary['reentry_route_count']}`",
        f"- action-allowed routes: `{summary['action_allowed_route_count']}`",
        f"- runtime action allowed: `{summary['runtime_action_allowed']}`",
        f"- full validation allowed: `{summary['full_validation_allowed']}`",
        "",
        "## Ranked Re-entry Routes",
        "",
    ]
    for row in reentry_routes:
        report.extend(
            [
                f"### {row['priority']}. {row['route_label']}",
                "",
                f"- route_id: `{row['route_id']}`",
                f"- evidence_strength: `{row['evidence_strength']}`",
                f"- positive_evidence: `{row['positive_evidence']}`",
                f"- blocking_evidence: `{row['blocking_evidence']}`",
                f"- action_allowed: `{row['action_allowed']}`",
                f"- recommended_next_step: {row['recommended_next_step']}",
                "",
            ]
        )
    report.extend(
        [
            "## Interpretation",
            "",
            "No route in this audit is action-authorized. The strongest re-entry direction is not another semantic-anchor threshold sweep; it is a measured carrier/actuator search that starts from prior merge-gauge and internal-cue evidence while preserving control beating as the gate.",
        ]
    )
    (OUT / "broader_carrier_reentry_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
