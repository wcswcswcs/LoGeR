from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"payload": payload}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def build_failure_autopsy(output_root: Path) -> dict[str, Any]:
    phase0 = _load_json(ROOT / "outputs/audit/v53_fact_lock/fact_lock.json")
    phase1_default = _load_json(ROOT / "outputs/audit/v53_mask_component_support/support_summary.json")
    phase1_repair = _load_json(ROOT / "outputs/audit/v53_mask_component_support_tau005/support_summary.json")
    phase2 = _load_json(ROOT / "outputs/audit/v53_chunk_universe/chunk_summary.json")
    phase3_k5 = _load_json(ROOT / "outputs/audit/v53_representative_observations_fixed/representative_summary.json")
    phase3_k8 = _load_json(ROOT / "outputs/audit/v53_representative_observations_k8_underseg_cap_fixed/representative_summary.json")
    phase4_default = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger/reprojection_summary.json")
    phase4_sparse = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger_skip_no_related/reprojection_summary.json")
    phase4_minvis_sparse = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger_minvis10_skip_no_related/reprojection_summary.json")
    phase4_top1_sparse = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger_top1_skip_no_related/reprojection_summary.json")
    phase4_conflict_veto = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger_conflict_veto018/reprojection_summary.json")
    phase4_k0_conflict_veto = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger_k0_conflict_veto018_max800/reprojection_summary.json")
    phase4_k0_conflict_veto025 = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger_k0_conflict_veto025_max4000/reprojection_summary.json")
    phase4_k0_conflict_veto025_repeated_sig = _load_json(
        ROOT / "outputs/audit/v53_reprojection_ledger_k0_conflict_veto025_skip_repeated_sig/reprojection_summary.json"
    )
    phase4_k0_conflict_veto030 = _load_json(ROOT / "outputs/audit/v53_reprojection_ledger_k0_conflict_veto030_max4000/reprojection_summary.json")
    local_k8 = _load_json(ROOT / "outputs/audit/v53_local_objectlets_conflict_veto018/local_objectlet_summary.json")
    local_k0 = _load_json(ROOT / "outputs/audit/v53_local_objectlets_k0_conflict_veto018_max800/local_objectlet_summary.json")
    local_k0_025 = _load_json(ROOT / "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/local_objectlet_summary.json")
    local_k0_025_l11 = _load_json(ROOT / "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000_l11_dynamic/local_objectlet_summary.json")
    local_k0_025_repeated_sig_l12 = _load_json(
        ROOT / "outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/local_objectlet_summary.json"
    )
    local_k0_030 = _load_json(ROOT / "outputs/audit/v53_local_objectlets_k0_conflict_veto030_max4000/local_objectlet_summary.json")
    ap = _load_json(ROOT / "outputs/audit/v53_ap_diagnostic/ap_export_summary.json")
    local_control_gap = _load_json(ROOT / "outputs/audit/v53_local_control_gap_audit/local_control_gap_summary.json")
    local_control_gap_repeated_sig_l12 = _load_json(
        ROOT / "outputs/audit/v53_local_control_gap_audit_skip_repeated_sig_l12/local_control_gap_summary.json"
    )
    local_control_gap_latest = local_control_gap_repeated_sig_l12 if not local_control_gap_repeated_sig_l12.get("missing") else local_control_gap
    latest_component_gap_path = ROOT / "outputs/audit/v53_local_control_gap_audit_skip_repeated_sig_l12/component_gap_rows.csv"
    if not latest_component_gap_path.exists():
        latest_component_gap_path = ROOT / "outputs/audit/v53_local_control_gap_audit/component_gap_rows.csv"
    local_component_gap_rows = _read_csv(latest_component_gap_path)

    ledger_sparse_rows = _read_csv(ROOT / "outputs/audit/v53_reprojection_ledger_skip_no_related/reprojection_ledger_rows.csv")
    conflict_rows = [row for row in ledger_sparse_rows if str(row.get("same_frame_exclusion_violation")).lower() == "true"]
    high_outside_default = [
        row
        for row in _read_csv(ROOT / "outputs/audit/v53_reprojection_ledger/reprojection_ledger_rows.csv")
        if float(row.get("outside_all_related_masks_ratio") or 0.0) > 0.35
    ]
    _write_csv(output_root / "same_frame_conflict_rows.csv", conflict_rows[:1000])
    _write_csv(output_root / "high_outside_residual_objectlets.csv", high_outside_default[:1000])
    _write_csv(output_root / "false_merge_objectlets.csv", conflict_rows[:1000])
    _write_csv(
        output_root / "false_cut_components.csv",
        [
            row
            for row in local_component_gap_rows
            if str(row.get("gap_type")) == "mask_only_only"
        ][:1000],
    )
    _write_csv(output_root / "underseg_mask_error_rows.csv", [])
    _write_csv(output_root / "semantic_outlier_error_rows.csv", [])
    _write_csv(output_root / "candidate_not_selected_rows.csv", [])
    ap_gate = ap.get("gate", {}) if isinstance(ap.get("gate"), dict) else {}
    ap_summary = ap.get("summary", {}) if isinstance(ap.get("summary"), dict) else {}
    ap_rows = ap.get("ap_rows", []) if isinstance(ap.get("ap_rows"), list) else []
    ap_rows_by_variant = {str(row.get("variant")): row for row in ap_rows}
    native_repair_audit = ap.get("native_method_export_repair_audit", {}) if isinstance(ap.get("native_method_export_repair_audit"), dict) else {}
    ap_diagnostic_ran = bool(ap_gate.get("ap_diagnostic_useful"))
    method_safe_ap_available = bool(ap_gate.get("method_safe_ap_available"))
    method_safe_native_support_available = bool(ap_gate.get("method_safe_native_support_available"))

    phase_status = [
        {"phase": "Phase 0 fact lock", "status": "pass", "summary": "v53_fact_lock/fact_lock.json"},
        {
            "phase": "Phase 1 incidence default",
            "status": "fail",
            "gate": phase1_default.get("gate"),
            "summary": "v53_mask_component_support/support_summary.json",
        },
        {
            "phase": "Phase 1 incidence repair tau0.05",
            "status": "pass" if phase1_repair.get("gate", {}).get("pass") else "fail",
            "gate": phase1_repair.get("gate"),
            "summary": "v53_mask_component_support_tau005/support_summary.json",
        },
        {"phase": "Phase 2 chunk universe", "status": "pass" if phase2.get("gate", {}).get("pass") else "fail", "gate": phase2.get("gate")},
        {"phase": "Phase 3 K5 representative", "status": "fail", "gate": phase3_k5.get("gate")},
        {"phase": "Phase 3 K8 representative repair", "status": "pass" if phase3_k8.get("gate", {}).get("pass") else "fail", "gate": phase3_k8.get("gate")},
        {"phase": "Phase 4 default reprojection", "status": "fail", "gate": phase4_default.get("gate")},
        {"phase": "Phase 4 sparse-skip repair", "status": "fail", "gate": phase4_sparse.get("gate")},
        {"phase": "Phase 4 minvis10 sparse-skip repair", "status": "fail", "gate": phase4_minvis_sparse.get("gate")},
        {"phase": "Phase 4 top1 sparse-skip diagnostic", "status": "fail", "gate": phase4_top1_sparse.get("gate")},
        {
            "phase": "Phase 4 K8 conflict-veto repair",
            "status": "pass" if phase4_conflict_veto.get("gate", {}).get("pass") else "fail",
            "gate": phase4_conflict_veto.get("gate"),
        },
        {
            "phase": "Phase 4 K0 repeated-partial conflict-veto repair",
            "status": "pass" if phase4_k0_conflict_veto.get("gate", {}).get("pass") else "fail",
            "gate": phase4_k0_conflict_veto.get("gate"),
        },
        {
            "phase": "Phase 4 K0 conflict-veto 0.25 best legal repair",
            "status": "pass" if phase4_k0_conflict_veto025.get("gate", {}).get("pass") else "fail",
            "gate": phase4_k0_conflict_veto025.get("gate"),
        },
        {
            "phase": "Phase 4 K0 0.25 repeated-support signature repair",
            "status": "pass" if phase4_k0_conflict_veto025_repeated_sig.get("gate", {}).get("pass") else "fail",
            "gate": phase4_k0_conflict_veto025_repeated_sig.get("gate"),
            "repeated_support_candidate_count": phase4_k0_conflict_veto025_repeated_sig.get("repeated_support_candidate_count"),
        },
        {
            "phase": "Phase 4 K0 conflict-veto 0.30 boundary check",
            "status": "pass" if phase4_k0_conflict_veto030.get("gate", {}).get("pass") else "fail",
            "gate": phase4_k0_conflict_veto030.get("gate"),
        },
        {
            "phase": "Phase 6 K8 local objectlets",
            "status": "success" if local_k8.get("any_success_gate_pass") else ("relaxed" if local_k8.get("any_relaxed_gate_pass") else "fail"),
            "best_real_variant": local_k8.get("best_real_variant"),
        },
        {
            "phase": "Phase 6 K0 repeated-partial local objectlets",
            "status": "success" if local_k0.get("any_success_gate_pass") else ("relaxed" if local_k0.get("any_relaxed_gate_pass") else "fail"),
            "best_real_variant": local_k0.get("best_real_variant"),
        },
        {
            "phase": "Phase 6 K0 0.25 best legal local objectlets",
            "status": "success" if local_k0_025.get("any_success_gate_pass") else ("relaxed" if local_k0_025.get("any_relaxed_gate_pass") else "fail"),
            "best_real_variant": local_k0_025.get("best_real_variant"),
        },
        {
            "phase": "Phase 7 local selection repair L11 dynamic beam",
            "status": (
                "success"
                if local_k0_025_l11.get("any_success_gate_pass")
                else ("relaxed_gap_not_closed" if local_k0_025_l11.get("any_relaxed_gate_pass") else "fail")
            ),
            "best_real_variant": local_k0_025_l11.get("best_real_variant"),
            "control_gap_repair_conclusion": local_control_gap.get("repair_conclusion"),
        },
        {
            "phase": "Phase 7 repeated-support signature/L12 repair",
            "status": (
                "success"
                if local_k0_025_repeated_sig_l12.get("any_success_gate_pass")
                else ("relaxed_gap_not_closed" if local_k0_025_repeated_sig_l12.get("any_relaxed_gate_pass") else "fail")
            ),
            "best_real_variant": local_k0_025_repeated_sig_l12.get("best_real_variant"),
            "control_gap_repair_conclusion": local_control_gap_latest.get("repair_conclusion"),
        },
        {
            "phase": "Phase 6 K0 0.30 boundary local objectlets",
            "status": "blocked_phase4_fail" if not phase4_k0_conflict_veto030.get("gate", {}).get("pass") else ("success" if local_k0_030.get("any_success_gate_pass") else ("relaxed" if local_k0_030.get("any_relaxed_gate_pass") else "fail")),
            "best_real_variant": local_k0_030.get("best_real_variant"),
        },
        {
            "phase": "Phase 11 AP diagnostic",
            "status": (
                "diagnostic_ran_method_safe_blocked"
                if ap_diagnostic_ran and not method_safe_ap_available
                else ("diagnostic_ran" if ap_diagnostic_ran else "not_run")
            ),
            "gate": ap_gate,
            "best_AP": ap_summary.get("best_AP"),
            "best_AP_variant": ap_summary.get("best_AP_variant"),
        },
        {
            "phase": "Phase 11 AP3 native carrier materialization",
            "status": (
                "native_support_available_scannet_ap_blocked"
                if method_safe_native_support_available and not method_safe_ap_available
                else ("method_safe_ap_available" if method_safe_ap_available else "native_support_not_available")
            ),
            "repair_result": native_repair_audit.get("repair_result"),
            "native_observation_row_count": native_repair_audit.get("v53_native_carrier_observation_row_count"),
            "native_unique_carrier_count": native_repair_audit.get("v53_native_unique_carrier_count"),
        },
    ]
    best_local_source = (
        local_k0_025_repeated_sig_l12
        if local_k0_025_repeated_sig_l12.get("best_real_row")
        else (local_k0_025_l11 if local_k0_025_l11.get("best_real_row") else (local_k0_025 if local_k0_025.get("best_real_row") else local_k0))
    )
    best_local = best_local_source.get("best_real_row") if isinstance(best_local_source.get("best_real_row"), dict) else {}
    mask_only_row = {}
    for row in best_local_source.get("variant_rows", []):
        if row.get("variant") == "L9_mask_only_representative_support":
            mask_only_row = row
            break
    local_relaxed = bool(best_local_source.get("any_relaxed_gate_pass"))
    local_success = bool(best_local_source.get("any_success_gate_pass"))
    if local_success:
        final_label = "LOCAL_OBJECTLET_SUCCESS_HISTORY_NOT_RUN"
        primary_blocker = "HISTORY_NOT_RUN"
        secondary_blocker = "LOCAL_SUCCESS_REQUIRES_HISTORY_STAGE"
        history_reason = "Local success gate passed, but history memory was not run in this continuation."
        ap_reason = None if ap_diagnostic_ran else "AP diagnostic was not run because history/native materialization was not promoted in this continuation."
    elif local_relaxed:
        final_label = "NO_GO_LOCAL_CONTROL_GAP"
        primary_blocker = "LOCAL_FAIL_CONTROLS"
        real_minus_mask_only = float(best_local.get("real_minus_mask_only_ARI") or 0.0)
        if real_minus_mask_only > 0.0:
            secondary_blocker = "MASK_ONLY_CONTROL_MARGIN_BELOW_GATE"
            history_reason = (
                "Local relaxed gate passed, but local success gate failed because "
                f"D4RT local objectlets beat mask-only by only {real_minus_mask_only:.6f} ARI, below the required +0.10 margin; "
                "do not promote to history method."
            )
        else:
            secondary_blocker = "MASK_ONLY_CONTROL_BEATS_D4RT_LOCAL_OBJECTLETS"
            history_reason = "Local relaxed gate passed, but local success gate failed because mask-only control outperformed D4RT local objectlets; do not promote to history method."
        ap_reason = None if ap_diagnostic_ran else "No method-eligible v53 local/history object fields were promoted after local control failure."
    else:
        final_label = "NO_GO_LOCAL_RELAXED_FAIL"
        primary_blocker = "LOCAL_FAIL_SELECTION"
        secondary_blocker = "LOCAL_RELAXED_GATE_NOT_PASSED"
        history_reason = "Local relaxed gate failed; Stop Rule 4 allows history only as diagnostic, not method."
        ap_reason = None if ap_diagnostic_ran else "No valid v53 local/history object fields were promoted after local relaxed failure."
    if ap_diagnostic_ran and not method_safe_ap_available:
        ap_materialization_blocker = (
            "METHOD_SAFE_SCANNET_AP_EXPORT_UNAVAILABLE"
            if method_safe_native_support_available
            else "METHOD_SAFE_NATIVE_EXPORT_UNAVAILABLE"
        )
    elif ap_diagnostic_ran:
        ap_materialization_blocker = None
    else:
        ap_materialization_blocker = "AP_DIAGNOSTIC_NOT_RUN"
    summary = {
        "phase": "v53_failure_autopsy",
        "final_label": final_label,
        "primary_blocker": primary_blocker,
        "secondary_blocker": secondary_blocker,
        "local_objectlet_formation_success": bool(local_success),
        "local_relaxed_gate_pass": bool(local_relaxed),
        "local_success_gate_pass": bool(local_success),
        "history_memory_ran": False,
        "history_not_run_reason": history_reason,
        "ap_diagnostic_ran": bool(ap_diagnostic_ran),
        "ap_not_run_reason": ap_reason,
        "method_safe_ap_available": bool(method_safe_ap_available),
        "method_safe_native_support_available": bool(method_safe_native_support_available),
        "ap_materialization_blocker": ap_materialization_blocker,
        "d4rt_controls_ran": True,
        "semantic_guard_ran": False,
        "phase_status": phase_status,
        "key_evidence": {
            "phase4_default": {
                "reprojection_success_rate": phase4_default.get("reprojection_success_rate"),
                "outside_all_related_masks_ratio_mean": phase4_default.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": phase4_default.get("same_frame_exclusion_violation_rate"),
                "same_gt_precision": phase4_default.get("reprojection_success_same_GT_precision"),
            },
            "phase4_sparse_skip": {
                "reprojection_success_rate": phase4_sparse.get("reprojection_success_rate"),
                "outside_all_related_masks_ratio_mean": phase4_sparse.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": phase4_sparse.get("same_frame_exclusion_violation_rate"),
                "same_gt_precision": phase4_sparse.get("reprojection_success_same_GT_precision"),
                "sparse_no_related_frame_count": phase4_sparse.get("sparse_no_related_frame_count"),
            },
            "phase4_k8_conflict_veto": {
                "reprojection_success_rate": phase4_conflict_veto.get("reprojection_success_rate"),
                "outside_all_related_masks_ratio_mean": phase4_conflict_veto.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": phase4_conflict_veto.get("same_frame_exclusion_violation_rate"),
                "same_gt_precision": phase4_conflict_veto.get("reprojection_success_same_GT_precision"),
                "candidate_count": phase4_conflict_veto.get("candidate_count"),
            },
            "phase4_k0_conflict_veto": {
                "reprojection_success_rate": phase4_k0_conflict_veto.get("reprojection_success_rate"),
                "outside_all_related_masks_ratio_mean": phase4_k0_conflict_veto.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": phase4_k0_conflict_veto.get("same_frame_exclusion_violation_rate"),
                "same_gt_precision": phase4_k0_conflict_veto.get("reprojection_success_same_GT_precision"),
                "candidate_count": phase4_k0_conflict_veto.get("candidate_count"),
            },
            "phase4_k0_conflict_veto025_best_legal": {
                "reprojection_success_rate": phase4_k0_conflict_veto025.get("reprojection_success_rate"),
                "outside_all_related_masks_ratio_mean": phase4_k0_conflict_veto025.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": phase4_k0_conflict_veto025.get("same_frame_exclusion_violation_rate"),
                "same_gt_precision": phase4_k0_conflict_veto025.get("reprojection_success_same_GT_precision"),
                "candidate_count": phase4_k0_conflict_veto025.get("candidate_count"),
                "gate": phase4_k0_conflict_veto025.get("gate"),
            },
            "phase4_k0_conflict_veto025_repeated_signature": {
                "reprojection_success_rate": phase4_k0_conflict_veto025_repeated_sig.get("reprojection_success_rate"),
                "outside_all_related_masks_ratio_mean": phase4_k0_conflict_veto025_repeated_sig.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": phase4_k0_conflict_veto025_repeated_sig.get("same_frame_exclusion_violation_rate"),
                "same_gt_precision": phase4_k0_conflict_veto025_repeated_sig.get("reprojection_success_same_GT_precision"),
                "candidate_count": phase4_k0_conflict_veto025_repeated_sig.get("candidate_count"),
                "repeated_support_candidate_count": phase4_k0_conflict_veto025_repeated_sig.get("repeated_support_candidate_count"),
                "gate": phase4_k0_conflict_veto025_repeated_sig.get("gate"),
            },
            "phase4_k0_conflict_veto030_boundary": {
                "reprojection_success_rate": phase4_k0_conflict_veto030.get("reprojection_success_rate"),
                "outside_all_related_masks_ratio_mean": phase4_k0_conflict_veto030.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": phase4_k0_conflict_veto030.get("same_frame_exclusion_violation_rate"),
                "same_gt_precision": phase4_k0_conflict_veto030.get("reprojection_success_same_GT_precision"),
                "candidate_count": phase4_k0_conflict_veto030.get("candidate_count"),
                "gate": phase4_k0_conflict_veto030.get("gate"),
            },
            "phase6_k0_best_local": {
                "variant": best_local.get("variant"),
                "4D_ARI": best_local.get("4D_ARI"),
                "4D_purity": best_local.get("4D_purity"),
                "4D_completeness": best_local.get("4D_completeness"),
                "conflict_rate": best_local.get("conflict_rate"),
                "outside_residual_mean": best_local.get("outside_residual_mean"),
                "real_minus_shuffled_ARI": best_local.get("real_minus_shuffled_ARI"),
                "real_minus_no_temporal_ARI": best_local.get("real_minus_no_temporal_ARI"),
                "real_minus_mask_only_ARI": best_local.get("real_minus_mask_only_ARI"),
                "success_gate": best_local.get("success_gate"),
            },
            "phase6_mask_only_control": {
                "4D_ARI": mask_only_row.get("4D_ARI"),
                "4D_purity": mask_only_row.get("4D_purity"),
                "4D_completeness": mask_only_row.get("4D_completeness"),
                "component_coverage_ratio": mask_only_row.get("component_coverage_ratio"),
            },
            "phase7_local_selection_repair": {
                "best_method_variant": local_control_gap_latest.get("best_method_variant"),
                "best_method_real_minus_mask_only_ARI": local_control_gap_latest.get("best_method_real_minus_mask_only_ARI"),
                "best_l11_variant": local_control_gap_latest.get("best_l11_variant"),
                "best_l11_ARI": local_control_gap_latest.get("best_l11_ARI"),
                "best_l11_real_minus_mask_only_ARI": local_control_gap_latest.get("best_l11_real_minus_mask_only_ARI"),
                "mask_only_only_component_count": local_control_gap_latest.get("mask_only_only_component_count"),
                "method_only_component_count": local_control_gap_latest.get("method_only_component_count"),
                "repair_conclusion": local_control_gap_latest.get("repair_conclusion"),
                "blocker_location": local_control_gap_latest.get("blocker_location"),
                "evidence_chain": local_control_gap_latest.get("evidence_chain"),
                "previous_l11_best_method_real_minus_mask_only_ARI": local_control_gap.get("best_method_real_minus_mask_only_ARI"),
                "previous_l11_best_l11_real_minus_mask_only_ARI": local_control_gap.get("best_l11_real_minus_mask_only_ARI"),
            },
            "phase11_ap_diagnostic": {
                "ap_smoke_pass": ap_gate.get("ap_smoke_pass"),
                "ap_diagnostic_identity_gate_pass": ap_gate.get("ap_diagnostic_identity_gate_pass"),
                "ap_diagnostic_useful": ap_gate.get("ap_diagnostic_useful"),
                "method_safe_ap_available": ap_gate.get("method_safe_ap_available"),
                "rgbd_bridge_ap_ran": ap_gate.get("rgbd_bridge_ap_ran"),
                "ap6_constant_score_min_region_ran": ap_gate.get("ap6_constant_score_min_region_ran"),
                "ap7_wta_conflict_suppression_ran": ap_gate.get("ap7_wta_conflict_suppression_ran"),
                "best_AP": ap_summary.get("best_AP"),
                "best_AP50": ap_summary.get("best_AP50"),
                "best_AP25": ap_summary.get("best_AP25"),
                "best_AP_variant": ap_summary.get("best_AP_variant"),
                "metric_scope": ap_summary.get("metric_scope"),
                "native_export_status": ap_rows_by_variant.get("AP3_v53_local_objectlet_native_export", {}).get("status"),
                "history_export_status": ap_rows_by_variant.get("AP4_v53_history_native_export", {}).get("status"),
                "method_safe_native_support_available": ap_gate.get("method_safe_native_support_available"),
                "v53_native_carrier_support_available": ap_gate.get("v53_native_carrier_support_available"),
                "v53_native_carrier_summary_path": native_repair_audit.get("v53_native_carrier_summary_path"),
                "v53_native_carrier_observation_row_count": native_repair_audit.get("v53_native_carrier_observation_row_count"),
                "v53_native_unique_carrier_count": native_repair_audit.get("v53_native_unique_carrier_count"),
                "v53_native_AP_bridge_status": native_repair_audit.get("v53_native_AP_bridge_status"),
                "native_method_export_repair_attempted": native_repair_audit.get("repair_attempted"),
                "native_method_export_repair_result": native_repair_audit.get("repair_result"),
                "carrier_has_native_point_or_mesh_vertex_mapping": native_repair_audit.get("carrier_has_native_point_or_mesh_vertex_mapping"),
                "objectlet_has_native_or_object_field_link": native_repair_audit.get("objectlet_has_native_or_object_field_link"),
                "chunk_component_has_native_or_object_field_link": native_repair_audit.get("chunk_component_has_native_or_object_field_link"),
                "v42_AP_bridge_status": native_repair_audit.get("v42_AP_bridge_status"),
                "required_future_change": native_repair_audit.get("required_future_change"),
            },
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    _write_json(output_root / "local_blocker_summary.json", summary)
    _write_json(output_root / "failure_summary.json", summary)
    dashboard = [
        "# Stream4D v53 Diagnostic Dashboard",
        "",
        f"Final label: `{summary['final_label']}`",
        f"Primary blocker: `{summary['primary_blocker']}`",
        "",
        "## Phase Status",
        "",
    ]
    for row in phase_status:
        dashboard.append(f"- {row['phase']}: {row['status']}")
    dashboard.extend(
        [
            "",
            "## Key Evidence",
            "",
            f"- Phase4 default success rate: `{phase4_default.get('reprojection_success_rate')}`",
            f"- Phase4 default outside mean: `{phase4_default.get('outside_all_related_masks_ratio_mean')}`",
            f"- Phase4 sparse-skip success rate: `{phase4_sparse.get('reprojection_success_rate')}`",
            f"- Phase4 sparse-skip conflict rate: `{phase4_sparse.get('same_frame_exclusion_violation_rate')}`",
            f"- Phase4 same-GT precision: `{phase4_sparse.get('reprojection_success_same_GT_precision')}`",
            f"- Phase4 K0 conflict-veto success rate: `{phase4_k0_conflict_veto.get('reprojection_success_rate')}`",
            f"- Phase4 K0 0.25 best legal success rate: `{phase4_k0_conflict_veto025.get('reprojection_success_rate')}`",
            f"- Phase4 R5 repeated signature success rate: `{phase4_k0_conflict_veto025_repeated_sig.get('reprojection_success_rate')}`",
            f"- Phase4 R5 repeated signature candidate count: `{phase4_k0_conflict_veto025_repeated_sig.get('repeated_support_candidate_count')}`",
            f"- Phase4 K0 0.30 boundary conflict rate: `{phase4_k0_conflict_veto030.get('same_frame_exclusion_violation_rate')}`",
            f"- Phase6 K0 best local ARI: `{best_local.get('4D_ARI')}`",
            f"- Phase6 K0 real-minus-mask-only ARI: `{best_local.get('real_minus_mask_only_ARI')}`",
            f"- Phase7 latest selection repair conclusion: `{local_control_gap_latest.get('repair_conclusion')}`",
            f"- Phase7 latest best L11 real-minus-mask-only ARI: `{local_control_gap_latest.get('best_l11_real_minus_mask_only_ARI')}`",
            f"- Phase11 AP diagnostic best AP/AP50/AP25: `{ap_summary.get('best_AP')}` / `{ap_summary.get('best_AP50')}` / `{ap_summary.get('best_AP25')}`",
            f"- Phase11 method-safe AP available: `{method_safe_ap_available}`",
            f"- Phase11 method-safe native support available: `{method_safe_native_support_available}`",
            f"- Phase11 v53 native carrier rows: `{native_repair_audit.get('v53_native_carrier_observation_row_count')}`",
            f"- Phase11 native AP repair result: `{native_repair_audit.get('repair_result')}`",
            "",
            "## Visualizations",
            "",
            "- `outputs/audit/v53_visualizations/local_objectlets_tau005/dominant_collapse_check.png`",
            "- `outputs/audit/v53_visualizations/local_objectlets_k8_underseg_cap_fixed/coverage_progress_curve_scene0050_00_chunk000.png`",
            "- `outputs/audit/v53_visualizations/reprojection_skip_no_related/outside_residual_heatmap_scene0050_00.png`",
            "- `outputs/audit/v53_visualizations/reprojection_skip_no_related/conflict_frame_gallery_scene0050_00.png`",
        ]
    )
    (ROOT / "outputs/audit/v53_visualizations").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/audit/v53_visualizations/v53_diagnostic_dashboard.md").write_text(
        "\n".join(dashboard) + "\n", encoding="utf-8"
    )
    _write_json(ROOT / "outputs/audit/v53_full_stage1/final_decision.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v53 failure autopsy and final decision.")
    parser.add_argument("--output-root", default="outputs/audit/v53_failure_autopsy")
    args = parser.parse_args()
    output_root = ROOT / args.output_root
    summary = build_failure_autopsy(output_root)
    print(
        {
            "summary": str(output_root / "failure_summary.json"),
            "final_decision": "outputs/audit/v53_full_stage1/final_decision.json",
            "final_label": summary["final_label"],
            "primary_blocker": summary["primary_blocker"],
        }
    )


if __name__ == "__main__":
    main()
