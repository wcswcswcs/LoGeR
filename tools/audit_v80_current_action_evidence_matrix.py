#!/usr/bin/env python3
"""Audit current v80 semantic three-memory action evidence.

This tool is intentionally read-only. It gathers existing landed summaries,
maps them to the v80 plan gates, and writes a compact evidence matrix plus a
next-action recommendation. It does not claim success from diagnostic positives.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_current_action_evidence_matrix_20260622_2213"


EVIDENCE_SPECS: list[dict[str, Any]] = [
    {
        "requirement": "Phase0 artifact audit",
        "memory_body": "prerequisite",
        "family": "phase0",
        "path": "phase0_multiseq_artifact_audit/phase0_artifact_audit_summary.json",
        "gate_keys": ["phase0_gate_pass"],
        "positive_when": "any",
        "note": "Multi-sequence artifact readiness gate.",
    },
    {
        "requirement": "Phase1 good/bad case bank",
        "memory_body": "prerequisite",
        "family": "phase1",
        "path": "phase1_three_memory_case_bank/case_bank_summary.json",
        "gate_keys": ["phase1_gate_pass", "phase1_balance_gate_pass", "semantic_diagnosis_gate_pass"],
        "positive_when": "all",
        "note": "Case bank balance and semantic diagnosis gate.",
    },
    {
        "requirement": "Phase2 visual confirmation",
        "memory_body": "prerequisite",
        "family": "phase2",
        "path": "phase2_case_visual_confirmation/visual_integrity_audit.json",
        "gate_keys": ["gate_pass", "action_ready_gate_pass"],
        "positive_when": "all",
        "note": "Visual confirmation gate with direct QKV/TTT artifacts.",
    },
    {
        "requirement": "Phase2 all-seq enhanced visual aggregate",
        "memory_body": "prerequisite",
        "family": "phase2",
        "path": "phase2_direct_hook_enhanced_visual_review_allseq_aggregate/visual_integrity_audit_allseq.json",
        "gate_keys": ["aggregate_phase2_direct_hook_visual_gate_pass"],
        "positive_when": "all",
        "note": "All-sequence direct hook visual review aggregate.",
    },
    {
        "requirement": "Phase9 completed rediscovery visual bundle",
        "memory_body": "prerequisite",
        "family": "phase9_visual_completion",
        "path": "phase10_seq01_phase9_rediscovery_visual_completion_20260622_2305/visual_integrity_audit.json",
        "gate_keys": ["gate_pass", "visual_audit_gate_pass", "required_phase9_panel_sets_present"],
        "positive_when": "all",
        "note": "Plan-required Phase9 short-QK/mid-SWA/long-TTT/merge-boundary visual artifact completion.",
    },
    {
        "requirement": "Phase3 short READ/QK action",
        "memory_body": "short",
        "family": "READ/QK",
        "path": "phase3_short_read_existing_actuator_seq01_badgood1_fastfix/phase3_qk_dedicated_seq01_badgood1_gate_summary.json",
        "gate_keys": ["phase3_existing_actuator_gate_pass", "actual_method_progress"],
        "positive_when": "any",
        "note": "Dedicated QK-pair smoke on seq01 bad/good targets.",
    },
    {
        "requirement": "Phase3 short READ key-stable variant",
        "memory_body": "short",
        "family": "READ/QK",
        "path": "phase3_short_read_existing_actuator_seq01_badgood1_fastfix/phase3_qk_keystable_seq01_badgood1_gate_summary.json",
        "gate_keys": ["phase3_existing_actuator_gate_pass", "actual_method_progress"],
        "positive_when": "any",
        "note": "Key-stable QK-pair variant.",
    },
    {
        "requirement": "Phase5 long TTT selected-write support/no-persistent",
        "memory_body": "long",
        "family": "TTT selected-write",
        "path": "phase5_ttt_lw38_lw42_seq01_badgood1_fastfix/ttt_long_case_accelerated_decision.json",
        "gate_keys": ["any_representative_smoke_signal_pass", "method_gate_claimed"],
        "positive_when": "any",
        "note": "LW38/LW42 representative long-window write-control family.",
    },
    {
        "requirement": "Phase5 long TTT selected full veto",
        "memory_body": "long",
        "family": "TTT selected-write",
        "path": "phase5_ttt_lw44_frame279_selected_full_seq01_badgood1_fastfix/ttt_long_case_accelerated_decision.json",
        "gate_keys": ["any_representative_smoke_signal_pass", "method_gate_claimed"],
        "positive_when": "any",
        "note": "LW44 selected-write full-veto family.",
    },
    {
        "requirement": "Phase5 long TTT control-delta full veto",
        "memory_body": "long",
        "family": "TTT selected-write",
        "path": "phase5_ttt_lw46_chunk009_control_delta_full_seq01_badgood1_fastfix/ttt_long_case_accelerated_decision.json",
        "gate_keys": ["any_representative_smoke_signal_pass", "method_gate_claimed"],
        "positive_when": "any",
        "note": "LW46 control-delta full-veto family.",
    },
    {
        "requirement": "Phase6 cross-memory handshake",
        "memory_body": "cross",
        "family": "handshake",
        "path": "phase6_cross_memory_handshake_seq00_chunk142_read_ttt_minimal/phase5_cross_memory_handshake_decision.json",
        "gate_keys": ["phase5_any_gate_pass", "method_gate_claimed"],
        "positive_when": "any",
        "note": "READ/TTT minimal cross-memory handshake check.",
    },
    {
        "requirement": "Phase8 RET query-conditioned retrieval",
        "memory_body": "short",
        "family": "RET/QK retrieval",
        "path": "phase9_ret_qk_pair_read_seq00_chunk142_smoke/ret_qk_pair_gate_summary.json",
        "gate_keys": ["phase3_existing_actuator_gate_pass", "actual_method_progress"],
        "positive_when": "any",
        "note": "RET query-conditioned QK pair read smoke.",
    },
    {
        "requirement": "Phase8 OUT4 merge overlap outlier",
        "memory_body": "mid",
        "family": "OUT4 merge overlap",
        "path": "phase9_out4_merge_overlap_outlier_downweight_seq00_chunk142/phaseE_multichunk_summary_fast_nocompile.json",
        "gate_keys": ["phaseE_gate_pass", "phaseE_head_tail_pass", "phaseE_overlap_pass"],
        "positive_when": "any",
        "note": "Merge overlap outlier downweight PhaseE smoke.",
    },
    {
        "requirement": "Seq01 RADIO/thingstuff qscale canary5",
        "memory_body": "mid",
        "family": "RADIO qscale merge",
        "path": "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012/thingstuff_radio_qscale_ref055_canary5_gate_summary.json",
        "gate_keys": ["phaseE_gate_pass", "phaseE_head_tail_pass", "phaseE_overlap_pass"],
        "positive_when": "any",
        "note": "Ref055 qscale canary5 head-tail/overlap gate.",
    },
    {
        "requirement": "Seq01 proxy controller risk budget",
        "memory_body": "mid",
        "family": "proxy controller",
        "path": "phase9_seq01_proxy_controller_v1b_riskbudget_canary5/proxy_controller_v1b_canary5_gate_summary.json",
        "gate_keys": ["phaseE_gate_pass", "phaseE_head_tail_pass", "phaseE_overlap_pass"],
        "positive_when": "any",
        "note": "Non-GT qscale/native risk-budget controller.",
    },
    {
        "requirement": "Seq01 multiobjective controller",
        "memory_body": "mid",
        "family": "multiobjective controller",
        "path": "phase9_seq01_canary5_multiobjective_merge_controller_v1/canary_multiobjective_controller_gate_summary.json",
        "gate_keys": ["phaseE_gate_pass", "phaseE_head_tail_pass", "phaseE_overlap_pass"],
        "positive_when": "any",
        "note": "Canary5 multiobjective merge/gauge controller.",
    },
    {
        "requirement": "Seq01 safe-positive controller",
        "memory_body": "mid",
        "family": "safe-positive controller",
        "path": "phase9_seq01_canary5_safe_positive_controller_v2/safe_positive_controller_gate_summary.json",
        "gate_keys": ["phaseE_gate_pass", "phaseE_head_tail_pass", "phaseE_overlap_pass"],
        "positive_when": "any",
        "note": "Canary5 safe-positive controller.",
    },
    {
        "requirement": "Phase9 non-GT gauge direction signal",
        "memory_body": "mid",
        "family": "non-GT gauge signal",
        "path": "phase10_seq01_non_gt_gauge_direction_signal_audit_20260622_2045/non_gt_gauge_signal_audit_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "Non-GT gauge direction selector audit.",
    },
    {
        "requirement": "Phase9 trajectory-only future-overlap proxy",
        "memory_body": "mid",
        "family": "motion future-overlap proxy",
        "path": "phase10_seq01_motion_future_overlap_proxy_audit_20260622_2055/motion_future_overlap_proxy_audit_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "Trajectory-only future-overlap proxy audit.",
    },
    {
        "requirement": "Phase9 RADIO object topology separability",
        "memory_body": "mid",
        "family": "RADIO object topology",
        "path": "phase10_seq01_radio_object_topology_separability_audit_20260622_2115/radio_object_topology_separability_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "RADIO lowtrust/boundary/sky-context support-vs-harm audit.",
    },
    {
        "requirement": "Phase9 RADIO guard extra chunk coverage",
        "memory_body": "mid",
        "family": "RADIO object topology",
        "path": "phase10_seq01_radio_guard_extra_chunk_coverage_20260622_2140/radio_guard_extra_chunk_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "Projection of chunk08-preserving RADIO guards onto extra seq01 sidecar chunks.",
    },
    {
        "requirement": "Phase9 TTSA temporal-spatial carrier",
        "memory_body": "long",
        "family": "TTSA temporal-spatial",
        "path": "phase10_seq01_ttsa_temporal_spatial_carrier_audit_20260622_2205/ttsa_temporal_spatial_carrier_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "TTSA temporal memory evolution plus spatial observation quality carrier audit.",
    },
    {
        "requirement": "Phase9 per-region TTT post-delta carrier",
        "memory_body": "long",
        "family": "TTT post-delta region",
        "path": "phase10_seq01_ttt_postdelta_region_carrier_audit_20260622_2220/ttt_postdelta_region_carrier_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "Per-region TTT post-delta top-region alignment with semantic/geometry support.",
    },
    {
        "requirement": "Phase9 geometry error / TTT / semantic explanation",
        "memory_body": "long",
        "family": "TTT geometry-error semantic explanation",
        "path": "phase10_seq01_geometry_error_ttt_semantic_explanation_20260622_2245/geometry_error_ttt_semantic_explanation_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "Joined geometry error maps, TTT selected-write support, visual rediscovery, and canary decisions.",
    },
    {
        "requirement": "Phase9 semantic false-positive separator",
        "memory_body": "long",
        "family": "semantic false-positive separator",
        "path": "phase10_seq01_semantic_false_positive_separator_20260622_2153/semantic_false_positive_separator_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "Non-GT separator audit for chunk08 local explanation versus chunk10/chunk12 overlap-harm false positives.",
    },
    {
        "requirement": "Phase10 selected-write low-support held-out coverage",
        "memory_body": "long",
        "family": "selected-write heldout coverage",
        "path": "phase10_selected_write_low_support_coverage_20260622_2212/selected_write_low_support_coverage_summary.json",
        "gate_keys": ["method_gate_claimed", "runtime_promotion_allowed", "v80_goal_achieved"],
        "positive_when": "any",
        "note": "Coverage audit for low-support selected-write separator across Phase1 long cases and held-out sequences.",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _gate_value(payload: dict[str, Any], key: str) -> bool | None:
    return _boolish(payload.get(key))


def _evaluate_spec(report_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = report_root / spec["path"]
    payload = _read_json(path)
    if payload is None:
        return {
            **spec,
            "source": str(path),
            "source_exists": False,
            "status": "missing",
            "gate_pass": False,
            "gate_values": {},
            "selected_metrics": {},
        }

    gate_values = {key: _gate_value(payload, key) for key in spec["gate_keys"]}
    raw_gate_values = {key: payload.get(key) for key in spec["gate_keys"]}
    known_values = [value for value in gate_values.values() if value is not None]
    if spec.get("positive_when") == "all":
        gate_pass = bool(known_values) and all(known_values) and len(known_values) == len(spec["gate_keys"])
    else:
        gate_pass = any(bool(value) for value in known_values)

    selected_metrics = {}
    for key in (
        "phaseE_gate_pass",
        "phaseE_head_tail_pass",
        "phaseE_overlap_pass",
        "head_tail_pass_count",
        "overlap_pass_count",
        "head_tail_median_improvement_vs_baseline_ratio",
        "overlap_median_improvement_vs_baseline_ratio",
        "phase3_existing_actuator_gate_pass",
        "actual_method_progress",
        "any_representative_smoke_signal_pass",
        "method_gate_claimed",
        "runtime_promotion_allowed",
        "v80_goal_achieved",
        "rules_evaluated",
        "deployable_gate_pass_rules",
        "diagnostic_separator_rules",
        "separator_rules_preserving_known_canary",
        "separator_rules_with_extra_good_candidates",
        "extra_good_case_runtime_candidate_chunks",
        "recommended_runtime_chunks",
        "status",
        "status_detail",
        "core_blocker",
        "diagnostic_separator_found",
        "best_separator_rule",
        "best_separator_selected_chunks",
        "overlap_harm_false_positive_chunks",
        "high_dq_false_positive_selected_chunks",
        "heldout_multi_case_gate",
        "coverage_blockers",
        "selected_write_positive_seq_chunks",
        "selected_write_positive_seqs",
        "long_case_rows_with_positive_low_support_separator",
        "support_only_low_semantic_error_no_ttt_write_seq_chunks",
        "next_action",
    ):
        if key in payload:
            selected_metrics[key] = payload[key]

    return {
        **spec,
        "source": str(path),
        "source_exists": True,
        "status": "pass" if gate_pass else "fail",
        "gate_pass": gate_pass,
        "gate_values": raw_gate_values,
        "selected_metrics": selected_metrics,
    }


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prereq_rows = [row for row in rows if row["memory_body"] == "prerequisite"]
    action_rows = [row for row in rows if row["memory_body"] != "prerequisite"]
    prereq_pass = all(row["gate_pass"] for row in prereq_rows)
    action_pass_rows = [row for row in action_rows if row["gate_pass"]]
    missing_rows = [row for row in rows if not row["source_exists"]]
    failed_action_families = sorted({row["family"] for row in action_rows if row["source_exists"] and not row["gate_pass"]})

    phase8_families_seen = {
        row["family"]
        for row in action_rows
        if row["family"]
        in {
            "RET/QK retrieval",
            "OUT4 merge overlap",
            "RADIO qscale merge",
            "TTT selected-write",
            "motion future-overlap proxy",
            "non-GT gauge signal",
            "RADIO object topology",
            "TTSA temporal-spatial",
            "TTT post-delta region",
            "TTT geometry-error semantic explanation",
            "semantic false-positive separator",
            "selected-write heldout coverage",
        }
        and row["source_exists"]
    }

    uncovered_or_weak = []
    if "OUT4 merge overlap" in phase8_families_seen and not any(
        row["gate_pass"] for row in action_rows if row["family"] == "OUT4 merge overlap"
    ):
        uncovered_or_weak.append(
            "OUT4 already has seq00 evidence but seq01 boundary conflict remains unresolved; do not repeat same downweight rule."
        )
    if "RET/QK retrieval" in phase8_families_seen and not any(
        row["gate_pass"] for row in action_rows if row["family"] == "RET/QK retrieval"
    ):
        uncovered_or_weak.append("RET/QK retrieval has only a failed seq00 smoke; it is weak but not a current seq01 mid-memory fix.")
    radio_topology_seen = any(row["family"] == "RADIO object topology" and row["source_exists"] for row in action_rows)
    radio_topology_pass = any(row["family"] == "RADIO object topology" and row["gate_pass"] for row in action_rows)
    if not radio_topology_seen:
        uncovered_or_weak.append(
            "RADIO object topology separability is not represented as a canary5 support-vs-harm audit; this is the fastest remaining non-runtime diagnostic."
        )
    elif not radio_topology_pass:
        uncovered_or_weak.append(
            "RADIO topology now has both support-vs-harm and extra-chunk coverage audits; it remains diagnostic-only and has no extra phase1-good runtime candidate."
        )
    ttsa_seen = any(row["family"] == "TTSA temporal-spatial" and row["source_exists"] for row in action_rows)
    ttsa_pass = any(row["family"] == "TTSA temporal-spatial" and row["gate_pass"] for row in action_rows)
    if ttsa_seen and not ttsa_pass:
        uncovered_or_weak.append(
            "TTSA temporal-spatial scalar trace audit is diagnostic-only: chunk08-local rules do not pass PhaseE, and broader rules select overlap-harm false positives."
        )
    postdelta_seen = any(row["family"] == "TTT post-delta region" and row["source_exists"] for row in action_rows)
    postdelta_pass = any(row["family"] == "TTT post-delta region" and row["gate_pass"] for row in action_rows)
    if postdelta_seen and not postdelta_pass:
        uncovered_or_weak.append(
            "Per-region TTT post-delta/support audit has no separator: low-support top-region rules select no chunks, and high-Dq selects non-helpful or overlap-harm chunks."
        )
    geom_ttt_seen = any(
        row["family"] == "TTT geometry-error semantic explanation" and row["source_exists"] for row in action_rows
    )
    geom_ttt_pass = any(
        row["family"] == "TTT geometry-error semantic explanation" and row["gate_pass"] for row in action_rows
    )
    if geom_ttt_seen and not geom_ttt_pass:
        uncovered_or_weak.append(
            "Geometry-error/TTT/semantic join confirms a clean chunk08-local selected-write low-support explanation, but it is diagnostic-only and lacks held-out coverage or a promoted runtime gate."
        )
    false_positive_seen = any(
        row["family"] == "semantic false-positive separator" and row["source_exists"] for row in action_rows
    )
    false_positive_pass = any(
        row["family"] == "semantic false-positive separator" and row["gate_pass"] for row in action_rows
    )
    false_positive_diag_found = any(
        row["family"] == "semantic false-positive separator"
        and bool((row.get("selected_metrics") or {}).get("diagnostic_separator_found"))
        for row in action_rows
    )
    if false_positive_seen and not false_positive_pass:
        if false_positive_diag_found:
            uncovered_or_weak.append(
                "Semantic false-positive separator audit finds a diagnostic split: low-support selected-write preserves chunk08 and rejects chunk10/chunk12, but it remains diagnostic-only because the actuator/control gate still fails."
            )
        else:
            uncovered_or_weak.append(
                "Semantic false-positive separator audit did not find a diagnostic split strong enough to define a new runtime candidate."
            )
    heldout_cov_seen = any(
        row["family"] == "selected-write heldout coverage" and row["source_exists"] for row in action_rows
    )
    heldout_cov_pass = any(
        row["family"] == "selected-write heldout coverage" and row["gate_pass"] for row in action_rows
    )
    if heldout_cov_seen and not heldout_cov_pass:
        heldout_metrics = next(
            (
                row.get("selected_metrics") or {}
                for row in action_rows
                if row["family"] == "selected-write heldout coverage" and row["source_exists"]
            ),
            {},
        )
        positive_seqs = heldout_metrics.get("selected_write_positive_seqs") or []
        blockers = heldout_metrics.get("coverage_blockers") or []
        if any(str(seq) != "01" for seq in positive_seqs):
            uncovered_or_weak.append(
                "Selected-write low-support coverage audit now has non-seq01 positive attribution, but heldout_multi_case_gate remains false because coverage spans too few sequences and has no good-case safety coverage."
            )
        else:
            uncovered_or_weak.append(
                "Selected-write low-support coverage audit shows the positive separator evidence is not held-out multi-case coverage; current positives are seq01-only."
            )
        if blockers:
            uncovered_or_weak.append(f"Selected-write heldout coverage blockers: {', '.join(str(item) for item in blockers)}.")

    phase9_visual_done = any(
        row["family"] == "phase9_visual_completion" and row["source_exists"] and row["gate_pass"]
        for row in prereq_rows
    )

    if not prereq_pass:
        next_action = "repair_prerequisite_gates"
        next_action_reason = "Prerequisite gates are not all passing."
    elif action_pass_rows:
        next_action = "promote_only_after_heldout_validation"
        next_action_reason = "At least one action family reports a gate pass; heldout validation is required before promotion."
    elif (
        radio_topology_seen
        and ttsa_seen
        and postdelta_seen
        and geom_ttt_seen
        and false_positive_seen
        and heldout_cov_seen
        and phase9_visual_done
    ):
        next_action = "formal_no_go_ready_no_runtime_candidate_after_phase9_visual_completion"
        next_action_reason = (
            "All audited runtime/action families fail method gates. RADIO topology, TTSA scalar trace, and "
            "per-region TTT post-delta/support audits have all been run. The geometry-error/TTT/semantic join "
            "confirms a chunk08-local explanation, and the false-positive separator audit shows chunk10/chunk12 "
            "are high-Dq overlap-harm rather than low-support selected-write cases. Fresh seq00 attribution adds "
            "non-seq01 low-support selected-write positive evidence, but the held-out coverage gate remains false "
            "because positives cover too few sequences and no good-case safety rows. This sharpens the blocker but "
            "still does not provide a deployable multi-case runtime candidate. The plan-required Phase9 visual "
            "bundle is complete enough for formal No-Go review, while method success is still false."
        )
    elif radio_topology_seen and ttsa_seen and postdelta_seen and geom_ttt_seen and false_positive_seen and phase9_visual_done:
        next_action = "audit_selected_write_low_support_heldout_coverage"
        next_action_reason = (
            "False-positive separator evidence is diagnostic-positive on seq01. Before any runtime promotion, audit "
            "whether the selected-write low-support signal has held-out multi-case coverage across Phase1 long cases."
        )
    elif radio_topology_seen and ttsa_seen and postdelta_seen and geom_ttt_seen and phase9_visual_done:
        next_action = "audit_semantic_false_positive_separator_before_final_no_go"
        next_action_reason = (
            "Geometry-error/TTT/semantic evidence is chunk08-local and Phase9 visuals are complete. Before final "
            "No-Go, run the plan-directed semantic false-positive separator audit to explain why chunk10/chunk12 "
            "remain overlap-harm false positives."
        )
    elif radio_topology_seen and ttsa_seen and postdelta_seen and geom_ttt_seen:
        next_action = "complete_phase9_rediscovery_visual_bundle_before_final_no_go"
        next_action_reason = (
            "All audited runtime/action families fail method gates, but the Phase9 rediscovery visual bundle must "
            "be completed before any final No-Go review."
        )
    elif radio_topology_seen and ttsa_seen and postdelta_seen:
        next_action = "audit_geometry_error_ttt_semantic_join_before_any_runtime"
        next_action_reason = (
            "All audited runtime/action families fail method gates. The next cheap step is to join geometry error "
            "maps, TTT selected-write evidence, and semantic support before launching any runtime."
        )
    elif radio_topology_seen and ttsa_seen:
        next_action = "no_fast_plan_backed_runtime_candidate_after_radio_topology_and_ttsa"
        next_action_reason = (
            "All audited runtime/action families fail method gates. RADIO topology and TTSA scalar trace audits have "
            "both been run; their chunk08-local separators have no deployable coverage, and broader rules select "
            "overlap-harm false positives. Avoid repeating scalar sweeps, qscale thresholds, selected-write thresholds, "
            "RADIO topology guards, or TTSA scalar thresholds."
        )
    elif radio_topology_seen:
        next_action = "no_fast_plan_backed_runtime_candidate_after_radio_topology"
        next_action_reason = (
            "All audited runtime/action families fail method gates. The previously remaining cheap plan-backed "
            "RADIO topology audit has now been run; its chunk08-preserving guards have no extra phase1-good runtime "
            "candidate. Avoid repeating scalar sweeps, qscale thresholds, selected-write thresholds, or the same "
            "RADIO topology guard."
        )
    else:
        next_action = "audit_radio_object_topology_support_harm_separability_before_any_new_runtime"
        next_action_reason = (
            "All audited runtime/action families fail method gates. The remaining cheap plan-backed step is a "
            "RADIO object-topology separability audit on existing canary5 support/harm rows, not another scalar sweep."
        )

    return {
        "schema": "acl2_v80_current_action_evidence_matrix_v1",
        "prerequisite_gate_pass": prereq_pass,
        "action_gate_pass_any": bool(action_pass_rows),
        "action_gate_pass_requirements": [row["requirement"] for row in action_pass_rows],
        "missing_evidence_count": len(missing_rows),
        "missing_requirements": [row["requirement"] for row in missing_rows],
        "failed_action_families": failed_action_families,
        "uncovered_or_weak_points": uncovered_or_weak,
        "v80_goal_achieved": bool(prereq_pass and action_pass_rows),
        "method_gate_claimed": bool(action_pass_rows),
        "runtime_promotion_allowed": False,
        "next_action": next_action,
        "next_action_reason": next_action_reason,
    }


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 Current Action Evidence Matrix",
        "",
        f"- prerequisite_gate_pass: `{summary['prerequisite_gate_pass']}`",
        f"- action_gate_pass_any: `{summary['action_gate_pass_any']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        f"- next_action: `{summary['next_action']}`",
        "",
        "## Failed Action Families",
        "",
    ]
    for family in summary["failed_action_families"]:
        lines.append(f"- {family}")
    lines.extend(["", "## Evidence Rows", ""])
    lines.append("| requirement | memory | family | status | source | key metrics |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        metrics = row.get("selected_metrics") or {}
        short_metrics = {
            key: metrics.get(key)
            for key in (
                "phaseE_gate_pass",
                "head_tail_pass_count",
                "overlap_pass_count",
                "head_tail_median_improvement_vs_baseline_ratio",
                "overlap_median_improvement_vs_baseline_ratio",
                "method_gate_claimed",
                "runtime_promotion_allowed",
                "v80_goal_achieved",
                "core_blocker",
            )
            if key in metrics
        }
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["requirement"]),
                    str(row["memory_body"]),
                    str(row["family"]),
                    str(row["status"]),
                    str(row["source"]),
                    json.dumps(short_metrics, ensure_ascii=False, sort_keys=True),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    rows = [_evaluate_spec(args.report_root, spec) for spec in EVIDENCE_SPECS]
    summary = _build_summary(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "current_action_evidence_matrix_summary.json", summary)
    _write_csv(args.out_dir / "current_action_evidence_matrix_rows.csv", rows)
    _write_report(args.out_dir / "current_action_evidence_matrix_report.md", summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
