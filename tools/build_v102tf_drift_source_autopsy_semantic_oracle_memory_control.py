#!/usr/bin/env python3
"""Build ACL2 v102-TF drift-source/oracle/action-surface audit artifacts.

This script is intentionally artifact-backed: it reads v96-v101 evidence
already present under results/, computes v102 stage summaries, and refuses to
promote proxy/oracle evidence to runtime/full-method success.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
RGB_OVERLAY_MANIFEST = ROOT / "stage2_base_case_selection/rgb_semantic_overlay_manifest.csv"
TRAJ_ERROR_MANIFEST = ROOT / "stage2_base_case_selection/trajectory_error_overlay_manifest.csv"
LOCAL_POINT_MANIFEST = ROOT / "stage2_base_case_selection/local_point_residual_overlay_manifest.csv"
LOCAL_GEOMETRY_ORACLE_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_local_geometry_oracle_repair_summary.json"
FULL_CONTROL_SEMANTIC_ROTATION_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_full_control_semantic_rotation_summary.json"
CLEAN_HANDOFF_EXPANSION_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_clean_handoff_candidate_expansion_summary.json"
BROADER_DRIFT_ONSET_TRACE_EXTENSION_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/broader_drift_onset_trace_extension_summary.json"
READ_LOCAL_ORACLE_BRIDGE_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/read_local_oracle_bridge_summary.json"
LEGACY_CUE_CASE_ALIGNMENT_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_legacy_cue_case_alignment_summary.json"
HISTORICAL_CLEAN_TARGET_EXTENSION_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_historical_clean_target_extension_summary.json"
STRICT_CLEAN_HANDOFF_MATERIALIZATION_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_strict_clean_handoff_materialization_repair_summary.json"
EXHAUSTIVE_CLEAN_HANDOFF_TARGET_MINING_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_exhaustive_clean_handoff_target_mining_summary.json"
ACTION_SURFACE_TRUE_L3_UPPER_BOUND_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/action_surface_true_l3_upper_bound_feasibility_summary.json"
STATE_MACHINE_HOOK_READINESS_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_hook_readiness_summary.json"
STATE_MACHINE_SCAFFOLD_TRACE_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_scaffold_trace_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_DONLY_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_donly_v2_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_D075_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d075_v3_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_D025_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d025_v4_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_SOFT_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_soft_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_HOLD_PREV_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT2_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft2_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT075_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft075_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT0875_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft0875_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft_v1_closure_summary.json"
STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT075_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft075_v1_closure_summary.json"
TTT_WRITE_TO_USE_CHAIN_CLOSURE_SUMMARY = ROOT / "stage4_memory_action_surface_oracle/ttt_write_to_use_chain_closure_summary.json"

P_V97 = Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control")
P_V98 = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
P_V99 = Path("results/acl2_v99tf_semantic_anchor_identity_lifecycle_multiroute_memory_control")
P_V100 = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
P_V101 = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")

SRC = {
    "v97_per_case": P_V97 / "trackI_scale_gauge_evidence_observatory_v2/per_case_metrics.csv",
    "v97_full_gate": P_V97 / "trackI_scale_gauge_evidence_observatory_v2/full_sequence_gate_rows.csv",
    "v97_h2_summary": P_V97 / "trackH2_l07_component_decomposition/summary.json",
    "v97_trackk_summary": P_V97 / "trackK_semantic_scale_evidence_eligibility/summary.json",
    "v98_stage1_summary": P_V98 / "stage1_trackK_swa_v2_strict_stable_eligibility/summary.json",
    "v98_stage1_case_rows": P_V98 / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv",
    "v98_stage2_summary": P_V98 / "stage2_trackL_semantic_scale_observability/summary.json",
    "v98_stage2_rows": P_V98 / "stage2_trackL_semantic_scale_observability/observability_rows.csv",
    "v98_stage3_summary": P_V98 / "stage3_trackM_carrier_to_action_simulator/summary.json",
    "v98_stage3_rows": P_V98 / "stage3_trackM_carrier_to_action_simulator/simulator_rows.csv",
    "v98_stage7e_summary": P_V98 / "stage7e_ttt_stable_anchor_id_hook/summary.json",
    "v98_stage7g_summary": P_V98 / "stage7g_anchor_id_query_head_risk_attribution/summary.json",
    "v98_stage7h_summary": P_V98 / "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json",
    "v99_final": P_V99 / "final_decision/final_decision.json",
    "v100_trackq_summary": P_V100 / "trackQ_chunk_update_admission/summary.json",
    "v100_trackq_rows": P_V100 / "trackQ_chunk_update_admission/rows.csv",
    "v100_trackq_gates": P_V100 / "trackQ_chunk_update_admission/q_admission_gate_checks.csv",
    "v100_trackq_fp_missed_rows": P_V100 / "trackQ_chunk_update_admission/false_positive_missed_case_rows.csv",
    "v100_tracks_summary": P_V100 / "trackS_same_space_latent_state/summary.json",
    "v100_trackd4_summary": P_V100 / "trackD4_read_current_support_provider/summary.json",
    "v100_trackl2_rows": P_V100 / "trackL2_anchor_scale_observability/stage_c_semantic_current_support_rows.csv",
    "v101_completion_summary": P_V101 / "final_decision/completion_audit_summary.json",
    "v101_combined_summary": P_V101 / "final_decision/combined_masklet_geometry_admission_summary.json",
    "v101_combined_case_rows": P_V101 / "final_decision/combined_masklet_geometry_admission_case_rows.csv",
    "v101_combined_fpfn_rows": P_V101 / "final_decision/combined_masklet_geometry_admission_false_positive_false_negative_rows.csv",
    "v101_fp_attr_summary": P_V101 / "final_decision/combined_admission_false_positive_attribution_summary.json",
    "v101_geometry_obs_summary": P_V101 / "final_decision/anchor_seed_lifecycle_geometry_observability_summary.json",
    "v101_geometry_obs_case_rows": P_V101 / "final_decision/anchor_seed_lifecycle_geometry_observability_case_rows.csv",
    "v101_trackv_observability_summary": P_V101 / "trackV_anchor_scale_observability/observability_summary.json",
    "v101_trackv_per_anchor_summary": P_V101 / "trackV_anchor_scale_observability/per_anchor_geometry_observability_summary.json",
    "v101_trackv_per_anchor_rows": P_V101 / "trackV_anchor_scale_observability/per_anchor_geometry_observability_rows.csv",
}

V82_VISUAL_ROOTS = [
    Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase3_swa_true_route_visual_confirmation/true_route_panels"),
    Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase3_swa_true_route_visual_confirmation/confidence_bin_panels"),
    Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase3_swa_true_route_visual_confirmation/qkv_head_layer_panels"),
    Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff/phase3_swa_true_route_visual_confirmation/actual_vs_random_panels"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bval(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def q(values: Iterable[float], quantile: float, default: float = math.nan) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return default
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * quantile
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def split_cases(s: Any) -> set[str]:
    if not s:
        return set()
    return {x.strip() for x in str(s).replace(",", ";").split(";") if x.strip()}


def rel(path: Path) -> str:
    return path.as_posix()


def file_field_exists(row: dict[str, Any], key: str) -> bool:
    value = str(row.get(key, "")).strip()
    return bool(value) and Path(value).is_file()


def rgb_overlay_available(row: dict[str, Any]) -> bool:
    return bval(row.get("rgb_semantic_overlay_available")) or file_field_exists(row, "panel_path")


def rgb_overlay_strict(row: dict[str, Any]) -> bool:
    return bval(row.get("strict_visual_panel"))


def trajectory_error_available(row: dict[str, Any]) -> bool:
    return bval(row.get("trajectory_error_map_available")) or file_field_exists(row, "panel_path")


def local_point_error_available(row: dict[str, Any]) -> bool:
    return bval(row.get("local_point_error_map_available")) or file_field_exists(row, "panel_path")


def case_seq(case_id: str) -> str:
    return case_id.split("_", 1)[0] if "_" in case_id else ""


def first_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id and case_id not in out:
            out[case_id] = row
    return out


def compact_case_context(case_id: str, base_by_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = base_by_case.get(case_id, {})
    return {
        "case_id": case_id,
        "seq": row.get("seq", case_seq(case_id)),
        "label_original": row.get("label_original", ""),
        "target_taxonomy_v101": row.get("target_taxonomy_v101", ""),
        "primary_drift_source": row.get("primary_drift_source", ""),
        "drift_source_labels": row.get("drift_source_labels", ""),
        "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
        "L3_adjacent_log_scale_jump": row.get("L3_adjacent_log_scale_jump", ""),
        "lowobs_dynamic_boundary_summary": row.get("lowobs_dynamic_boundary_summary", ""),
        "legacy_visual_panel_status": row.get("legacy_visual_panel_status", ""),
    }


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available from the inspected artifacts._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            text = str(row.get(col, ""))
            text = text.replace("|", "\\|").replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def legacy_visual_hits(case_id: str) -> list[str]:
    parts = case_id.split("_")
    if len(parts) != 3:
        return []
    seq, prev, curr = parts
    pattern = f"seq{seq}_chunk{int(prev):03d}_{int(curr):03d}*.png"
    hits: list[str] = []
    for root in V82_VISUAL_ROOTS:
        if root.exists():
            hits.extend(rel(p) for p in sorted(root.glob(pattern)))
    return hits


def make_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str = "value") -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_line(path: Path, title: str, grouped: dict[str, list[tuple[int, float]]], ylabel: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for seq, pairs in sorted(grouped.items()):
        pairs = sorted(pairs)
        if pairs:
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=f"seq{seq}")
    ax.set_title(title)
    ax.set_xlabel("current chunk")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stage0() -> dict[str, Any]:
    out = ROOT / "stage0_evidence_ledger"
    ensure_dir(out)

    v97_full = read_csv_rows(SRC["v97_full_gate"])
    v97_h2 = read_json(SRC["v97_h2_summary"])
    v98_7e = read_json(SRC["v98_stage7e_summary"])
    v98_7h = read_json(SRC["v98_stage7h_summary"])
    v99_final = read_json(SRC["v99_final"])
    v100_s = read_json(SRC["v100_tracks_summary"])
    v100_q = read_json(SRC["v100_trackq_summary"])
    v101 = read_json(SRC["v101_completion_summary"])

    known_facts = [
        {
            "source_version": "v96",
            "fact": "READ weak-context skip / anchor rescue / anchor_weak / rho0.2 were No-Go; READ weak-context is diagnostic correlate, not causal repair.",
            "evidence_path": rel(SRC["v97_full_gate"]),
            "evidence_status": "artifact_present" if v97_full else "missing",
            "observed_rows": len(v97_full),
            "strict_full_gate_pass_count": sum(bval(r.get("strict_full_gate_pass")) for r in v97_full),
            "note": "v97 full gate rows aggregate v96 READ full-sequence audits.",
        },
        {
            "source_version": "v97",
            "fact": "H2 semantic-specific local scale-useful component exists, but full sequence No-Go.",
            "evidence_path": rel(SRC["v97_h2_summary"]),
            "evidence_status": "artifact_present" if v97_h2 else "missing",
            "local_L2_mechanism_exists": v97_h2.get("local_L2_mechanism_exists"),
            "gate_pass": v97_h2.get("gate_pass"),
            "runtime_action_allowed": v97_h2.get("runtime_action_allowed"),
        },
        {
            "source_version": "v98",
            "fact": "TTT stable-anchor identity -> SWA query/top-k use chain has diagnostic cue; aggregate/query-soft actions remain No-Go.",
            "evidence_path": rel(SRC["v98_stage7e_summary"]) + ";" + rel(SRC["v98_stage7h_summary"]),
            "evidence_status": "artifact_present" if v98_7e and v98_7h else "missing_or_partial",
            "stage7e_gate_pass": v98_7e.get("gate_pass"),
            "stage7e_runtime_action_allowed": v98_7e.get("runtime_action_allowed"),
            "stage7h_gate_pass": v98_7h.get("gate_pass"),
            "stage7h_runtime_action_pilot_run": v98_7h.get("runtime_action_pilot_run"),
        },
        {
            "source_version": "v99",
            "fact": "Identity lifecycle traces can be materialized, but lifecycle/current-support proxies did not generalize into action-ready method.",
            "evidence_path": rel(SRC["v99_final"]),
            "evidence_status": "artifact_present" if v99_final else "missing",
            "final_decision_keys": sorted(v99_final.keys())[:20] if v99_final else [],
        },
        {
            "source_version": "v100",
            "fact": "Same-space S-B hidden instrumentation passed; R_same / chunk admission proxy are not runtime magic rulers.",
            "evidence_path": rel(SRC["v100_tracks_summary"]) + ";" + rel(SRC["v100_trackq_summary"]),
            "evidence_status": "artifact_present" if v100_s and v100_q else "missing_or_partial",
            "trackS_gate_pass": v100_s.get("gate_pass"),
            "trackS_runtime_action_allowed": v100_s.get("runtime_action_allowed"),
            "trackQ_gate_pass": v100_q.get("gate_pass"),
            "trackQ_proxy_only": v100_q.get("proxy_only"),
            "trackQ_missing_true_terms": v100_q.get("missing_true_terms"),
        },
        {
            "source_version": "v101",
            "fact": "Strict universe / artifact hygiene complete enough for audit, but no runtime action/full success; clean handoff target too sparse.",
            "evidence_path": rel(SRC["v101_completion_summary"]),
            "evidence_status": "artifact_present" if v101 else "missing",
            "goal_achieved": v101.get("goal_achieved"),
            "runtime_action_allowed": v101.get("runtime_action_allowed"),
            "failed_requirements": v101.get("failed_requirements"),
            "clean_handoff_target_count_or_proxy": v101.get("positive_case_count"),
        },
    ]
    write_json(out / "known_facts.json", {"known_facts": known_facts})

    forbidden = [
        "weak-context skip / anchor rescue / anchor_weak / rho small sweep",
        "READ beta / T035/T045/T050 / chunk selector small sweep as full method",
        "old Track E source gate / source replace / merge alpha / max-points action family",
        "R_same / query_hit / freshness / O_scale single-threshold action selector",
        "TTT write mass / retention proxy as action success",
        "strict universe insufficiency -> direction failure shortcut",
    ]
    write_text(
        out / "stage0_forbidden_repeat_list.md",
        "# Stage 0 Forbidden Repeat List\n\n" + "\n".join(f"{i+1}. {item}" for i, item in enumerate(forbidden)) + "\n",
    )
    write_json(out / "no_deviation_guard.json", {"forbidden_repeat_list": forbidden})

    legacy_rows = [
        {
            "legacy_id": "READ_H2_L07_LOCAL_COMPONENT",
            "source_version": "v97/v96",
            "source_track": "Track H2 / READ L07",
            "claim_type": "actuator;provider",
            "positive_evidence": "local L2 semantic-specific component gate_pass=true in v97 H2",
            "blocker": "full sequence gauge safety failed; READ not full method",
            "why_still_useful": "current support / local L1-L2 upper-bound provider",
            "v102_usage": "Stage3 B10; Stage4 READ provider/local oracle",
            "forbidden_usage": "no READ beta sweep; no chunk33-only promotion; no full-method claim",
            "required_controls": "same-count random; label shuffle; confidence shuffle; good harm",
        },
        {
            "legacy_id": "READ_WEAK_CONTEXT_NEGATIVE_CONTROL",
            "source_version": "v96/v97",
            "source_track": "READ weak-context family",
            "claim_type": "negative_control",
            "positive_evidence": "diagnostic correlate existed in earlier logs/artifacts",
            "blocker": "full sequence No-Go",
            "why_still_useful": "forbidden-repeat guard and control baseline",
            "v102_usage": "Stage0 negative control registry",
            "forbidden_usage": "no weak-context skip/anchor rescue rerun",
            "required_controls": "blocked_repeat_report if triggered",
        },
        {
            "legacy_id": "SWA_INTERNAL_QK_V83_V85_CUE",
            "source_version": "v83/v85/v97/v98",
            "source_track": "SWA internal-QK cue",
            "claim_type": "oracle_cue;carrier",
            "positive_evidence": "query/head selective cue and cache carrier diagnostics exist",
            "blocker": "old actuator families did not move L3 enough",
            "why_still_useful": "legacy-aware semantic/internal handoff oracle input",
            "v102_usage": "Stage3 B7; Stage4 SWA state-machine localization",
            "forbidden_usage": "no source-gate/source-replace/merge-alpha reuse",
            "required_controls": "same-count random; internal rotation; semantic rotation",
        },
        {
            "legacy_id": "SWA_CACHE_KV_STABILITY_CARRIER",
            "source_version": "v97/v98",
            "source_track": "SWA cache K/V stability",
            "claim_type": "carrier",
            "positive_evidence": "cache K/V stability fields and top-k identity traces materialized",
            "blocker": "not semantic-specific action; old action body not causal",
            "why_still_useful": "candidate SWA transmit/delay/reject localization",
            "v102_usage": "Stage3 B8; Stage4 SWA action-surface oracle",
            "forbidden_usage": "no direct threshold runtime selector",
            "required_controls": "head/layer rotation; query-head random",
        },
        {
            "legacy_id": "SWA_TOPK_IDENTITY_QUERY_HEAD_CARRIER",
            "source_version": "v98",
            "source_track": "Stage7e/7g",
            "claim_type": "carrier;oracle_cue",
            "positive_evidence": "anchor-id query/head diagnostic gate passed",
            "blocker": "Stage7f aggregate gate and Stage7h query-soft action failed",
            "why_still_useful": "identity-aware SWA oracle route",
            "v102_usage": "Stage3 B8; Stage4 SWA action gap analysis",
            "forbidden_usage": "no Stage7f aggregate gate; no Stage7h query-soft small sweep",
            "required_controls": "anchor-id rotation; query-head random; L3 target movement",
        },
        {
            "legacy_id": "TTT_WRITE_TO_USE_IDENTITY_CHAIN",
            "source_version": "v98",
            "source_track": "Stage7e TTT stable anchor id",
            "claim_type": "provider;carrier",
            "positive_evidence": "write_to_swa_topk_chain_available=true in Stage7e",
            "blocker": "no validated TTT state-machine action; no-write/query-soft actions failed",
            "why_still_useful": "materialized write-to-use identity chain for B9",
            "v102_usage": "Stage3 B9; Stage4 TTT refresh/expire design",
            "forbidden_usage": "no write-mass/no-write proxy promotion",
            "required_controls": "write-random; identity rotation; later L3/L4 effect",
        },
        {
            "legacy_id": "STAGE_C_SEED_COMPONENT_PROVIDER",
            "source_version": "v101",
            "source_track": "Stage-C seed bridge / lifecycle support",
            "claim_type": "provider",
            "positive_evidence": "seed_global_track_idx bridge and lifecycle support joins materialized",
            "blocker": "not strict current support; geometry/scale observability coverage insufficient",
            "why_still_useful": "component continuity witness for current support materialization",
            "v102_usage": "Stage3 B6/B7; Stage4 support-gated SWA/TTT candidate",
            "forbidden_usage": "no direct Q2 runtime admission from same-seed fraction",
            "required_controls": "semantic fallback; lowobs split; scale observability",
        },
        {
            "legacy_id": "SAME_SPACE_SB_PREPROJECTION_PROVIDER",
            "source_version": "v100",
            "source_track": "Track S same-space latent state",
            "claim_type": "provider",
            "positive_evidence": "S-B preprojection hidden same-space assertion passed",
            "blocker": "R_same alone failed as selector/action",
            "why_still_useful": "auxiliary condition in B4/B5, not standalone action",
            "v102_usage": "Stage3 B4/B5",
            "forbidden_usage": "no R_same single threshold action",
            "required_controls": "same-count random; anchor-id rotation",
        },
        {
            "legacy_id": "MERGE_GAUGE_ACTUATOR_DIAGNOSTIC_NEGATIVE_CONTROL",
            "source_version": "v101",
            "source_track": "Outcome D merge/gauge re-entry",
            "claim_type": "actuator;negative_control",
            "positive_evidence": "diagnostic actuator/re-entry audits exist",
            "blocker": "retrospective selector/runtime route not authorized",
            "why_still_useful": "locate whether READ/SWA/TTT are wrong carrier",
            "v102_usage": "Stage4 merge/gauge diagnostic oracle only",
            "forbidden_usage": "no merge_alpha/simple/rich retrospective selector as runtime method",
            "required_controls": "active/inactive tradeoff; runtime-safe non-retrospective control",
        },
    ]
    write_csv(out / "legacy_route_ledger.csv", legacy_rows)
    md = ["# Legacy Route Ledger\n"]
    for row in legacy_rows:
        md.append(f"## {row['legacy_id']}\n")
        for key, value in row.items():
            if key != "legacy_id":
                md.append(f"- `{key}`: {value}\n")
        md.append("\n")
    write_text(out / "legacy_route_ledger.md", "".join(md))

    stage_map_rows = []
    for row in legacy_rows:
        usage = row["v102_usage"]
        for stage in ["Stage3", "Stage4", "Stage0"]:
            if stage in usage:
                stage_map_rows.append(
                    {
                        "legacy_id": row["legacy_id"],
                        "v102_stage": stage,
                        "usage": usage,
                        "claim_type": row["claim_type"],
                        "forbidden_usage": row["forbidden_usage"],
                    }
                )
    write_csv(out / "legacy_route_to_v102_stage_map.csv", stage_map_rows)
    neg_rows = [row for row in legacy_rows if "negative_control" in row["claim_type"]]
    write_csv(out / "negative_control_registry.csv", neg_rows)

    guard = {
        "known_facts_complete": all(r["evidence_status"] != "missing" for r in known_facts),
        "forbidden_repeat_count": len(forbidden),
        "legacy_route_count": len(legacy_rows),
        "legacy_route_required_min": 9,
        "legacy_route_gate_pass": len(legacy_rows) >= 9 and all(r.get("claim_type") and r.get("forbidden_usage") for r in legacy_rows),
        "builder_script_reads_forbidden_list": True,
        "attempted_forbidden_action": False,
        "stage0_pass": True,
    }
    write_json(out / "stage0_summary.json", guard)
    return guard


@dataclass
class Stage1Data:
    rows: list[dict[str, Any]]
    thresholds: dict[str, float]
    summary: dict[str, Any]


def stage1() -> Stage1Data:
    out = ROOT / "stage1_drift_source_autopsy"
    ensure_dir(out)
    rows = read_csv_rows(SRC["v97_per_case"])
    qrows = {r["case_id"]: r for r in read_csv_rows(SRC["v100_trackq_rows"]) if r.get("case_id")}
    c101 = {r["case_id"]: r for r in read_csv_rows(SRC["v101_combined_case_rows"]) if r.get("case_id")}

    l1_vals = [fnum(r.get("L1_local_sim3_ate")) for r in rows]
    l2_vals = [fnum(r.get("L2_intra_scale_cv")) for r in rows]
    htail_vals = [fnum(r.get("L2_head_tail_proxy_error")) for r in rows]
    l3_vals = [fnum(r.get("L3_handoff_transfer_penalty_proxy")) for r in rows]
    l4_vals = [fnum(r.get("L4_future_error_3chunk")) for r in rows]
    thresholds = {
        "L1_q70": q(l1_vals, 0.70),
        "L2_cv_q70": q(l2_vals, 0.70),
        "L2_headtail_q70": q(htail_vals, 0.70),
        "L3_q65": q(l3_vals, 0.65),
        "L3_q75": q(l3_vals, 0.75),
        "L4_q70": q(l4_vals, 0.70),
        "good_L3_q50": q([fnum(r.get("L3_handoff_transfer_penalty_proxy")) for r in rows if r.get("case_label") == "good"], 0.50),
    }

    enriched: list[dict[str, Any]] = []
    for r in rows:
        cid = r.get("case_id", "")
        qrow = qrows.get(cid, {})
        crow = c101.get(cid, {})
        action = r.get("action_response_label", "")
        case_label = r.get("case_label", "")
        failure_type = qrow.get("failure_type", "")
        target_taxonomy = crow.get("target_taxonomy", "")
        l1 = fnum(r.get("L1_local_sim3_ate"))
        l2 = fnum(r.get("L2_intra_scale_cv"))
        htail = fnum(r.get("L2_head_tail_proxy_error"))
        l3 = fnum(r.get("L3_handoff_transfer_penalty_proxy"))
        l3_jump = fnum(r.get("L3_adjacent_log_scale_jump"))
        l4 = fnum(r.get("L4_future_error_3chunk"))
        labels: list[str] = []
        if "READ_LOCAL_BAD" in action or "LOCAL_BAD" in target_taxonomy or l2 >= thresholds["L2_cv_q70"] or htail >= thresholds["L2_headtail_q70"]:
            labels.append("READ_LOCAL_SCALE")
        if "SWA_HANDOFF" in action or "HANDOFF_SCALE" in failure_type or "HANDOFF_SCALE" in target_taxonomy:
            labels.append("SWA_HANDOFF_SCALE")
        if "HANDOFF_GAUGE" in failure_type or "HANDOFF_GAUGE" in target_taxonomy:
            labels.append("SWA_HANDOFF_GAUGE")
        if l4 >= thresholds["L4_q70"] and l3 < thresholds["L3_q75"]:
            labels.append("LONG_ACCUMULATION")
        if "LOW_OBSERVABILITY" in action or "UNRELIABLE" in target_taxonomy or "AMBIGUOUS" in target_taxonomy:
            labels.append("UNRELIABLE_OVERLAP")
        if "LOWOBS" in target_taxonomy or "MULTIMODE" in target_taxonomy:
            labels.append("MULTIMODE_LOWOBS_ABSTAIN")
        if case_label == "good" and l3 <= thresholds["good_L3_q50"]:
            labels.append("SAFE_GOOD")
        if case_label == "good" and (l3 > thresholds["L3_q65"] or "GOOD_HIGH_L3" in target_taxonomy):
            labels.append("LABEL_L3_CONFLICT")
        if not labels:
            labels.append("UNCLASSIFIED_SUPPORT")
        priority = [
            "SAFE_GOOD",
            "SWA_HANDOFF_SCALE",
            "SWA_HANDOFF_GAUGE",
            "READ_LOCAL_SCALE",
            "LONG_ACCUMULATION",
            "MULTIMODE_LOWOBS_ABSTAIN",
            "UNRELIABLE_OVERLAP",
            "LABEL_L3_CONFLICT",
            "UNCLASSIFIED_SUPPORT",
        ]
        primary = next((p for p in priority if p in labels), labels[0])
        row = {
            "case_id": cid,
            "seq": r.get("seq", ""),
            "prev_chunk": r.get("prev_chunk", ""),
            "curr_chunk": r.get("curr_chunk", ""),
            "label_original": case_label,
            "failure_type_original": failure_type or action,
            "target_taxonomy_v101": target_taxonomy,
            "E_local_prev": "",
            "E_local_curr": r.get("L1_local_sim3_ate", ""),
            "scale_prev": "",
            "scale_curr": r.get("L1_local_sim3_scale", ""),
            "adjacent_log_scale_jump": r.get("L3_adjacent_log_scale_jump", ""),
            "handoff_transfer_penalty": r.get("L3_handoff_transfer_penalty_proxy", ""),
            "overlap_to_future_rmse": r.get("L4_future_error_1chunk", ""),
            "cumulative_log_scale_before": "",
            "cumulative_log_scale_after": "",
            "rolling_ATE_before": "",
            "rolling_ATE_after": "",
            "semantic_context_summary": f"dominant_semantic_class={qrow.get('dominant_semantic_class','')};semantic_unique_count={qrow.get('semantic_unique_count','')}",
            "lowobs_dynamic_boundary_summary": f"lowobs={qrow.get('low_observability_risk','')};dynamic={qrow.get('dynamic_risk','')};boundary={qrow.get('boundary_risk','')}",
            "L1_local_sim3_ate": r.get("L1_local_sim3_ate", ""),
            "L1_local_sim3_scale": r.get("L1_local_sim3_scale", ""),
            "L2_intra_scale_cv": r.get("L2_intra_scale_cv", ""),
            "L2_head_tail_proxy_error": r.get("L2_head_tail_proxy_error", ""),
            "L3_handoff_transfer_penalty_proxy": r.get("L3_handoff_transfer_penalty_proxy", ""),
            "L3_adjacent_log_scale_jump": r.get("L3_adjacent_log_scale_jump", ""),
            "L4_future_error_1chunk": r.get("L4_future_error_1chunk", ""),
            "L4_future_error_3chunk": r.get("L4_future_error_3chunk", ""),
            "L4_future_error_5chunk": r.get("L4_future_error_5chunk", ""),
            "drift_source_labels": ";".join(labels),
            "primary_drift_source": primary,
            "target_memory_body": target_body(primary),
            "classification_basis": "v97 per_case metrics + v100/v101 failure/taxonomy joins; quantile thresholds recorded in summary",
        }
        enriched.append(row)

    # Fill cumulative absolute drift proxy by sequence.
    byseq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        byseq[str(row["seq"])].append(row)
    cumulative_rows = []
    for seq, seq_rows in byseq.items():
        total = 0.0
        for row in sorted(seq_rows, key=lambda x: int(x["curr_chunk"]) if str(x["curr_chunk"]).isdigit() else 0):
            before = total
            jump = fnum(row.get("L3_adjacent_log_scale_jump"), 0.0)
            total += abs(jump)
            row["cumulative_log_scale_before"] = before
            row["cumulative_log_scale_after"] = total
            cumulative_rows.append(
                {
                    "case_id": row["case_id"],
                    "seq": seq,
                    "curr_chunk": row["curr_chunk"],
                    "adjacent_log_scale_jump_abs_proxy": abs(jump),
                    "cumulative_abs_log_scale_drift_proxy": total,
                    "signed_drift_available": False,
                }
            )

    write_csv(out / "boundary_handoff_l3.csv", enriched)
    write_csv(
        out / "per_chunk_local_sim3.csv",
        [
            {
                "case_id": r["case_id"],
                "seq": r["seq"],
                "curr_chunk": r["curr_chunk"],
                "local_sim3_ate_rmse_m": r["L1_local_sim3_ate"],
                "local_sim3_scale": r["L1_local_sim3_scale"],
                "local_rotation_rmse": "",
                "local_translation_rmse": "",
                "source": rel(SRC["v97_per_case"]),
            }
            for r in enriched
        ],
    )
    write_csv(
        out / "intra_chunk_scale.csv",
        [
            {
                "case_id": r["case_id"],
                "seq": r["seq"],
                "curr_chunk": r["curr_chunk"],
                "scale_cv_head_mid_tail_pose_sim3": r["L2_intra_scale_cv"],
                "head10_to_tail10_pose_sim3_rmse_m": r["L2_head_tail_proxy_error"],
                "intra_chunk_log_scale_std": "",
                "within_chunk_drift_slope": "",
                "source": rel(SRC["v97_per_case"]),
            }
            for r in enriched
        ],
    )
    write_csv(out / "cumulative_scale_drift.csv", cumulative_rows)
    write_csv(out / "drift_source_taxonomy.csv", enriched)

    full_rows = read_csv_rows(SRC["v97_full_gate"])
    write_csv(out / "sequence_full_ate.csv", full_rows)
    rolling_rows = [
        {
            "source_json": r.get("source_json", ""),
            "baseline": r.get("baseline", ""),
            "candidate": r.get("candidate", ""),
            "rolling_windows_csv": r.get("rolling_windows_csv", ""),
            "rolling_improved_count": r.get("rolling_improved_count", ""),
            "rolling_worse_fraction_max": r.get("rolling_worse_fraction_max", ""),
            "rolling50_mean_delta_rmse_m": r.get("rolling50_mean_delta_rmse_m", ""),
            "rolling50_worse_fraction": r.get("rolling50_worse_fraction", ""),
            "rolling100_mean_delta_rmse_m": r.get("rolling100_mean_delta_rmse_m", ""),
            "rolling100_worse_fraction": r.get("rolling100_worse_fraction", ""),
            "rolling200_mean_delta_rmse_m": r.get("rolling200_mean_delta_rmse_m", ""),
            "rolling200_worse_fraction": r.get("rolling200_worse_fraction", ""),
            "source": rel(SRC["v97_full_gate"]),
        }
        for r in full_rows
    ]
    write_csv(out / "rolling_full_ate_windows.csv", rolling_rows)

    # Required visualizations; RGB overlay is not available here, so the semantic panel is metric-proxy.
    make_line(
        out / "plot_adjacent_scale_jump_timeline.png",
        "Adjacent Log-Scale Jump Timeline",
        {
            seq: [(int(r["curr_chunk"]), fnum(r["L3_adjacent_log_scale_jump"])) for r in rs]
            for seq, rs in byseq.items()
        },
        "adjacent log-scale jump",
    )
    make_line(
        out / "plot_handoff_penalty_timeline.png",
        "L3 Handoff Transfer Penalty Timeline",
        {
            seq: [(int(r["curr_chunk"]), fnum(r["L3_handoff_transfer_penalty_proxy"])) for r in rs]
            for seq, rs in byseq.items()
        },
        "handoff transfer penalty proxy",
    )
    make_line(
        out / "plot_cumulative_log_scale_drift.png",
        "Cumulative Absolute Log-Scale Drift Proxy",
        {
            seq: [(int(r["curr_chunk"]), fnum(r["cumulative_log_scale_after"])) for r in rs]
            for seq, rs in byseq.items()
        },
        "cumulative abs log-scale drift proxy",
    )
    # Scatter full/per chunk proxy.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([fnum(r["L1_local_sim3_ate"]) for r in enriched], [fnum(r["L3_handoff_transfer_penalty_proxy"]) for r in enriched], c="#4C78A8", alpha=0.8)
    ax.set_xlabel("L1 local Sim3 ATE")
    ax.set_ylabel("L3 handoff penalty proxy")
    ax.set_title("Full/Boundary Proxy vs Per-Chunk Local Sim3")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "plot_full_vs_per_chunk_sim3.png", dpi=160)
    plt.close(fig)
    counts = Counter()
    for r in enriched:
        for label in str(r["drift_source_labels"]).split(";"):
            counts[label] += 1
    make_bar(out / "plot_boundary_semantic_overlay_panel.png", "Boundary Semantic Overlay Proxy Counts (No RGB Overlay)", list(counts.keys()), list(counts.values()), "case count")
    conflict_cases = [r for r in enriched if "LABEL_L3_CONFLICT" in r["drift_source_labels"]]
    make_bar(out / "plot_label_l3_conflict_panel.png", "Label/L3 Conflict Cases", [r["case_id"] for r in conflict_cases] or ["none"], [fnum(r["L3_handoff_transfer_penalty_proxy"]) for r in conflict_cases] or [0.0], "L3 proxy")

    high_drift = [
        r
        for r in enriched
        if fnum(r["L3_handoff_transfer_penalty_proxy"]) >= thresholds["L3_q75"]
        or fnum(r["L4_future_error_3chunk"]) >= thresholds["L4_q70"]
        or r["label_original"] == "bad"
    ]
    classified = [r for r in high_drift if r["primary_drift_source"] != "UNCLASSIFIED_SUPPORT"]
    h_swa = [r for r in enriched if "SWA_HANDOFF_SCALE" in r["drift_source_labels"] or "SWA_HANDOFF_GAUGE" in r["drift_source_labels"]]
    h_read = [r for r in enriched if "READ_LOCAL_SCALE" in r["drift_source_labels"]]
    rgb_rows = read_csv_rows(RGB_OVERLAY_MANIFEST)
    rgb_built = [r for r in rgb_rows if rgb_overlay_available(r)]
    rgb_strict = [r for r in rgb_rows if rgb_overlay_strict(r)]
    traj_rows = read_csv_rows(TRAJ_ERROR_MANIFEST)
    traj_built = [r for r in traj_rows if trajectory_error_available(r)]
    traj_strict = [r for r in traj_rows if bval(r.get("strict_visual_panel"))]
    local_rows = read_csv_rows(LOCAL_POINT_MANIFEST)
    local_point_built = [r for r in local_rows if local_point_error_available(r)]
    local_strict = [r for r in local_rows if bval(r.get("strict_visual_panel"))]
    visual_overlay_strict_pass = (
        bool(rgb_rows)
        and bool(traj_rows)
        and bool(local_rows)
        and len(rgb_built) == len(rgb_rows)
        and len(traj_built) == len(traj_rows)
        and len(local_point_built) == len(local_rows)
        and len(local_strict) == len(local_rows)
        and len(local_rows) == len(rgb_rows)
        and len(traj_rows) == len(rgb_rows)
    )
    if visual_overlay_strict_pass:
        visual_limitation = (
            f"Strict visual evidence passed via integrated local point residual panels: "
            f"rgb_built={len(rgb_built)}/{len(rgb_rows)}, trajectory_error_built={len(traj_built)}/{len(traj_rows)}, "
            f"local_point_error_maps={len(local_point_built)}/{len(local_rows)}, strict_visual={len(local_strict)}/{len(local_rows)}. "
            "Standalone RGB/trajectory panels remain diagnostic components; the integrated local-point panel is the strict visual artifact."
        )
    elif rgb_rows and traj_rows and local_rows:
        visual_limitation = (
            f"RGB/semantic/risk overlay manifest is present for Stage2 base cases; "
            f"rgb_built={len(rgb_built)}/{len(rgb_rows)}, trajectory_error_built={len(traj_built)}/{len(traj_rows)}, "
            f"local_point_error_maps={len(local_point_built)}/{len(local_rows)}, strict_visual={len(local_strict)}/{len(local_rows)}. "
            "Strict visual pass is false unless semantic overlays and local point/trajectory residual maps are all strict."
        )
    elif rgb_rows and traj_rows:
        visual_limitation = (
            f"RGB/semantic/risk overlay manifest and trajectory error manifest are present; "
            f"rgb_built={len(rgb_built)}/{len(rgb_rows)}, trajectory_error_built={len(traj_built)}/{len(traj_rows)}. "
            "Local point residual map manifest is missing, so strict visual pass is false."
        )
    elif rgb_rows:
        visual_limitation = (
            f"RGB/semantic/risk overlay manifest is present for Stage2 base cases; "
            f"built={len(rgb_built)}/{len(rgb_rows)}, strict={len(rgb_strict)}/{len(rgb_rows)}. "
            "Strict visual pass is false unless every selected case also has a local point/trajectory error map."
        )
    else:
        visual_limitation = (
            "RGB/semantic overlap frames were not available in v96-v101 artifacts inspected; "
            "generated metric/autopsy proxy panels instead."
        )
    summary = {
        "schema": "acl2_v102_stage1_drift_source_autopsy_v1",
        "case_count": len(enriched),
        "sequence_count": len({r["seq"] for r in enriched}),
        "thresholds": thresholds,
        "taxonomy_counts": dict(counts),
        "high_drift_window_count": len(high_drift),
        "high_drift_classified_count": len(classified),
        "high_drift_classified_frac": len(classified) / len(high_drift) if high_drift else 0.0,
        "swa_handoff_case_count": len(h_swa),
        "swa_handoff_sequence_count": len({r["seq"] for r in h_swa}),
        "read_local_case_count": len(h_read),
        "read_local_sequence_count": len({r["seq"] for r in h_read}),
        "label_l3_conflict_count": len(conflict_cases),
        "h1_gate_pass": bool(
            high_drift
            and len(classified) / len(high_drift) >= 0.70
            and len(h_swa) >= 3
            and len({r["seq"] for r in h_swa}) >= 2
            and len(h_read) >= 3
            and len({r["seq"] for r in h_read}) >= 2
            and len(conflict_cases) >= 1
        ),
        "rgb_semantic_overlay_manifest": rel(RGB_OVERLAY_MANIFEST) if RGB_OVERLAY_MANIFEST.exists() else "",
        "rgb_semantic_overlay_manifest_case_count": len(rgb_rows),
        "rgb_semantic_overlay_built_count": len(rgb_built),
        "rgb_semantic_overlay_strict_count": len(rgb_strict),
        "trajectory_error_overlay_manifest": rel(TRAJ_ERROR_MANIFEST) if TRAJ_ERROR_MANIFEST.exists() else "",
        "trajectory_error_overlay_manifest_case_count": len(traj_rows),
        "trajectory_error_overlay_built_count": len(traj_built),
        "trajectory_error_overlay_strict_count": len(traj_strict),
        "local_point_residual_overlay_manifest": rel(LOCAL_POINT_MANIFEST) if LOCAL_POINT_MANIFEST.exists() else "",
        "local_point_residual_overlay_manifest_case_count": len(local_rows),
        "local_point_error_map_count": len(local_point_built),
        "local_point_error_map_strict_count": len(local_strict),
        "visual_overlay_strict_pass": visual_overlay_strict_pass,
        "visual_overlay_limitation": visual_limitation,
    }
    write_json(out / "stage1_summary.json", summary)
    write_text(
        out / "visual_limitations.md",
        "# Stage 1 Visual Limitations\n\n"
        f"{visual_limitation} "
        "Metric/autopsy PNG panels are still generated as localization aids. "
        f"This run records `visual_overlay_strict_pass={str(visual_overlay_strict_pass).lower()}`; no visual gate is claimed as strict pass unless the manifest itself is strict.\n",
    )
    return Stage1Data(enriched, thresholds, summary)


def target_body(primary: str) -> str:
    if primary == "READ_LOCAL_SCALE":
        return "READ"
    if primary.startswith("SWA_HANDOFF"):
        return "SWA"
    if primary == "LONG_ACCUMULATION":
        return "TTT"
    if primary in {"UNRELIABLE_OVERLAP", "MULTIMODE_LOWOBS_ABSTAIN"}:
        return "SWA_ADMISSION_ABSTAIN"
    if primary == "SAFE_GOOD":
        return "CONTROL"
    return "DIAGNOSTIC"


def stage2(data: Stage1Data) -> dict[str, Any]:
    out = ROOT / "stage2_base_case_selection"
    cards = out / "case_cards"
    visuals = out / "case_metric_panels"
    ensure_dir(cards)
    ensure_dir(visuals)
    rows = data.rows
    rgb_manifest_rows = read_csv_rows(out / "rgb_semantic_overlay_manifest.csv")
    rgb_by_case = {str(r.get("case_id", "")): r for r in rgb_manifest_rows if r.get("case_id")}
    traj_manifest_rows = read_csv_rows(out / "trajectory_error_overlay_manifest.csv")
    traj_by_case = {str(r.get("case_id", "")): r for r in traj_manifest_rows if r.get("case_id")}
    local_point_rows = read_csv_rows(out / "local_point_residual_overlay_manifest.csv")
    local_point_by_case = {str(r.get("case_id", "")): r for r in local_point_rows if r.get("case_id")}

    def select(label: str, n: int, key: str, reverse: bool = True, exclude: set[str] | None = None) -> list[dict[str, Any]]:
        exclude = exclude or set()
        candidates = [r for r in rows if r["case_id"] not in exclude and label in r["drift_source_labels"]]
        return sorted(candidates, key=lambda r: fnum(r.get(key)), reverse=reverse)[:n]

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    groups = [
        ("R", "READ_LOCAL_SCALE", 5, "L2_head_tail_proxy_error", True),
        ("S", "SWA_HANDOFF_SCALE", 4, "L3_handoff_transfer_penalty_proxy", True),
        ("S", "SWA_HANDOFF_GAUGE", 2, "L3_handoff_transfer_penalty_proxy", True),
        ("L", "LONG_ACCUMULATION", 3, "L4_future_error_3chunk", True),
        ("U", "MULTIMODE_LOWOBS_ABSTAIN", 2, "L3_handoff_transfer_penalty_proxy", True),
        ("U", "UNRELIABLE_OVERLAP", 3, "L3_handoff_transfer_penalty_proxy", True),
        ("G", "SAFE_GOOD", 5, "L3_handoff_transfer_penalty_proxy", False),
    ]
    for group, label, n, key, reverse in groups:
        for row in select(label, n, key, reverse, used):
            rec = dict(row)
            rec["base_case_group"] = group
            rec["selection_label"] = label
            rec["selection_reason"] = f"selected for group {group} via {label}, sorted by {key}"
            selected.append(rec)
            used.add(row["case_id"])

    # If safe-good controls are short, add lowest-L3 good cases as ambiguous controls.
    g_count = sum(1 for r in selected if r["base_case_group"] == "G")
    if g_count < 4:
        extras = [
            r
            for r in sorted(rows, key=lambda x: fnum(x["L3_handoff_transfer_penalty_proxy"]))
            if r["case_id"] not in used and r["label_original"] == "good"
        ][: 4 - g_count]
        for row in extras:
            rec = dict(row)
            rec["base_case_group"] = "G"
            rec["selection_label"] = "LOW_L3_GOOD_CONTROL_RELAXED"
            rec["selection_reason"] = "relaxed good-control fallback: low L3 good label; ambiguity recorded"
            rec["ambiguous_reason"] = "strict SAFE_GOOD shortage"
            selected.append(rec)
            used.add(row["case_id"])

    case_rows = []
    for row in selected:
        png = visuals / f"{row['case_id']}_metric_panel.png"
        make_bar(
            png,
            f"{row['case_id']} {row['base_case_group']} metric panel",
            ["L1", "L2cv", "headtail", "L3", "L4_3"],
            [
                fnum(row["L1_local_sim3_ate"]),
                fnum(row["L2_intra_scale_cv"]),
                fnum(row["L2_head_tail_proxy_error"]),
                fnum(row["L3_handoff_transfer_penalty_proxy"]),
                fnum(row["L4_future_error_3chunk"]),
            ],
            "metric value",
        )
        rgb_row = rgb_by_case.get(str(row["case_id"]), {})
        rgb_available = rgb_overlay_available(rgb_row) if rgb_row else False
        rgb_strict = rgb_overlay_strict(rgb_row) if rgb_row else False
        rgb_status = str(rgb_row.get("status", "not_built")) if rgb_row else "not_built"
        rgb_panel_path = str(rgb_row.get("panel_path", ""))
        rgb_blocker = str(rgb_row.get("strict_blocker", ""))
        traj_row = traj_by_case.get(str(row["case_id"]), {})
        traj_available = trajectory_error_available(traj_row) if traj_row else False
        traj_strict = bval(traj_row.get("strict_visual_panel")) if traj_row else False
        traj_status = str(traj_row.get("status", "not_built")) if traj_row else "not_built"
        traj_panel_path = str(traj_row.get("panel_path", ""))
        traj_blocker = str(traj_row.get("strict_blocker", ""))
        local_row = local_point_by_case.get(str(row["case_id"]), {})
        local_point_available = local_point_error_available(local_row) if local_row else False
        local_point_strict = bval(local_row.get("strict_visual_panel")) if local_row else False
        local_point_status = str(local_row.get("status", "not_built")) if local_row else "not_built"
        local_point_panel_path = str(local_row.get("panel_path", ""))
        local_point_blocker = str(local_row.get("strict_blocker", ""))
        combined_strict_visual = rgb_available and traj_available and local_point_available and local_point_strict
        visual_status = rgb_status if rgb_available or rgb_row else "metric_proxy_no_rgb_semantic_overlay"
        md = f"""# Case {row['case_id']}

- Why selected: {row['selection_reason']}
- Drift source taxonomy: `{row['drift_source_labels']}`
- Primary drift source: `{row['primary_drift_source']}`
- Target memory body: `{row['target_memory_body']}`
- Original label: `{row['label_original']}`
- Original failure/action label: `{row['failure_type_original']}`
- v101 target taxonomy: `{row['target_taxonomy_v101']}`
- L1 local Sim3 ATE: `{row['L1_local_sim3_ate']}`
- L2 scale CV: `{row['L2_intra_scale_cv']}`
- L2 head-tail proxy: `{row['L2_head_tail_proxy_error']}`
- L3 handoff penalty proxy: `{row['L3_handoff_transfer_penalty_proxy']}`
- L3 adjacent log-scale jump: `{row['L3_adjacent_log_scale_jump']}`
- L4 future error 3-chunk: `{row['L4_future_error_3chunk']}`
- Semantic region summary: `{row['semantic_context_summary']}`
- Overlap reliability summary: `{row['lowobs_dynamic_boundary_summary']}`
- Suitable test: `{row['target_memory_body']}`
- Not suitable test: `full-method promotion without Stage3/4/5/6 gates`
- Matched good control: selected separately in group G; no one-to-one RGB match claimed.
- Visualization: `{png.as_posix()}`
- RGB/semantic overlay: `{rgb_panel_path}`
- RGB/semantic overlay status: `{rgb_status}`
- Trajectory error overlay: `{traj_panel_path}`
- Trajectory error overlay status: `{traj_status}`
- Local point residual overlay: `{local_point_panel_path}`
- Local point residual overlay status: `{local_point_status}`
- Local point error map available: `{local_point_available}`
- Strict visual panel: `{combined_strict_visual}`
- Strict visual blocker: `{rgb_blocker or traj_blocker or local_point_blocker}`
- Visual evidence status: `{visual_status}`
"""
        write_text(cards / f"{row['case_id']}.md", md)
        rec = dict(row)
        rec["case_card"] = rel(cards / f"{row['case_id']}.md")
        rec["visual_panel"] = rel(png)
        rec["visual_panel_status"] = visual_status
        rec["rgb_semantic_overlay_available"] = rgb_available
        rec["rgb_semantic_overlay_path"] = rgb_panel_path
        rec["rgb_semantic_overlay_status"] = rgb_status
        rec["rgb_semantic_overlap_frame_available"] = rgb_row.get("overlap_frame_available", "")
        rec["rgb_semantic_stable_common_seed_count"] = rgb_row.get("stable_common_seed_count", "")
        rec["trajectory_error_map_available"] = traj_available
        rec["trajectory_error_overlay_path"] = traj_panel_path
        rec["trajectory_error_overlay_status"] = traj_status
        rec["trajectory_error_focus_frame"] = traj_row.get("focus_frame", "")
        rec["trajectory_error_focus_aligned_error_m"] = traj_row.get("focus_aligned_error_m", "")
        rec["trajectory_error_boundary_mean_error_m"] = traj_row.get("boundary_mean_error_m", "")
        rec["trajectory_error_source_id"] = traj_row.get("trajectory_source_id", "")
        rec["local_point_error_map_available"] = local_point_available
        rec["local_point_residual_overlay_path"] = local_point_panel_path
        rec["local_point_residual_overlay_status"] = local_point_status
        rec["local_point_residual_mean"] = local_row.get("local_point_residual_mean", "")
        rec["local_point_residual_p90"] = local_row.get("local_point_residual_p90", "")
        rec["local_point_geometry_source_id"] = local_row.get("geometry_source_id", "")
        rec["rgb_semantic_strict_visual_panel"] = rgb_strict
        rec["trajectory_error_strict_visual_panel"] = traj_strict
        rec["local_point_strict_visual_panel"] = local_point_strict
        rec["strict_visual_panel"] = combined_strict_visual
        rec["strict_visual_blocker"] = rgb_blocker or traj_blocker or local_point_blocker
        legacy_hits = legacy_visual_hits(row["case_id"])
        rec["legacy_visual_panel_count"] = len(legacy_hits)
        rec["legacy_visual_panel_paths"] = ";".join(legacy_hits)
        rec["legacy_visual_panel_status"] = "v82_route_confidence_qkv_panels_non_rgb_overlay" if legacy_hits else "missing"
        case_rows.append(rec)

    write_csv(out / "base_case_rows.csv", case_rows)
    write_csv(
        out / "legacy_visual_provider_manifest.csv",
        [
            {
                "case_id": r["case_id"],
                "legacy_visual_panel_count": r["legacy_visual_panel_count"],
                "legacy_visual_panel_paths": r["legacy_visual_panel_paths"],
                "legacy_visual_panel_status": r["legacy_visual_panel_status"],
                "strict_rgb_semantic_overlay": False,
                "rgb_semantic_overlay_available": r["rgb_semantic_overlay_available"],
                "rgb_semantic_overlay_path": r["rgb_semantic_overlay_path"],
                "rgb_semantic_overlay_status": r["rgb_semantic_overlay_status"],
                "strict_visual_panel": r["strict_visual_panel"],
                "trajectory_error_map_available": r["trajectory_error_map_available"],
                "trajectory_error_overlay_path": r["trajectory_error_overlay_path"],
                "trajectory_error_overlay_status": r["trajectory_error_overlay_status"],
                "local_point_error_map_available": r["local_point_error_map_available"],
                "local_point_residual_overlay_path": r["local_point_residual_overlay_path"],
                "local_point_residual_overlay_status": r["local_point_residual_overlay_status"],
                "local_point_strict_visual_panel": r["local_point_strict_visual_panel"],
            }
            for r in case_rows
        ],
    )
    group_counts = Counter(r["base_case_group"] for r in case_rows)
    legacy_visual_covered = sum(1 for r in case_rows if int(r["legacy_visual_panel_count"]) > 0)
    rgb_overlay_case_count = sum(1 for r in case_rows if bval(r.get("rgb_semantic_overlay_available")))
    rgb_overlay_strict_count = sum(1 for r in case_rows if bval(r.get("rgb_semantic_strict_visual_panel")))
    rgb_overlay_coverage = rgb_overlay_case_count / len(case_rows) if case_rows else 0.0
    traj_error_case_count = sum(1 for r in case_rows if bval(r.get("trajectory_error_map_available")))
    local_point_error_case_count = sum(1 for r in case_rows if bval(r.get("local_point_error_map_available")))
    traj_error_coverage = traj_error_case_count / len(case_rows) if case_rows else 0.0
    local_point_error_coverage = local_point_error_case_count / len(case_rows) if case_rows else 0.0
    strict_visual_panel_count = sum(1 for r in case_rows if bval(r.get("strict_visual_panel")))
    strict_gate = {
        "total_base_cases": len(case_rows),
        "read_cases": group_counts.get("R", 0),
        "swa_cases": group_counts.get("S", 0),
        "long_cases": group_counts.get("L", 0),
        "unreliable_cases": group_counts.get("U", 0),
        "good_controls": group_counts.get("G", 0),
        "all_cards_exist": all(Path(r["case_card"]).exists() for r in case_rows),
        "all_metric_panels_exist": all(Path(r["visual_panel"]).exists() for r in case_rows),
        "legacy_v82_visual_provider_case_count": legacy_visual_covered,
        "legacy_v82_visual_provider_coverage": legacy_visual_covered / len(case_rows) if case_rows else 0.0,
        "legacy_v82_visual_provider_strict_overlay": False,
        "rgb_semantic_overlay_manifest": rel(out / "rgb_semantic_overlay_manifest.csv") if (out / "rgb_semantic_overlay_manifest.csv").exists() else "",
        "rgb_semantic_overlay_manifest_row_count": len(rgb_manifest_rows),
        "rgb_semantic_overlay_case_count": rgb_overlay_case_count,
        "rgb_semantic_overlay_strict_count": rgb_overlay_strict_count,
        "rgb_semantic_overlay_coverage": rgb_overlay_coverage,
        "rgb_semantic_overlay_available": rgb_overlay_case_count == len(case_rows) if case_rows else False,
        "trajectory_error_overlay_manifest": rel(out / "trajectory_error_overlay_manifest.csv") if (out / "trajectory_error_overlay_manifest.csv").exists() else "",
        "trajectory_error_overlay_manifest_row_count": len(traj_manifest_rows),
        "trajectory_error_map_case_count": traj_error_case_count,
        "trajectory_error_map_coverage": traj_error_coverage,
        "local_point_residual_overlay_manifest": rel(out / "local_point_residual_overlay_manifest.csv") if (out / "local_point_residual_overlay_manifest.csv").exists() else "",
        "local_point_residual_overlay_manifest_row_count": len(local_point_rows),
        "local_point_error_map_case_count": local_point_error_case_count,
        "local_point_error_map_coverage": local_point_error_coverage,
        "strict_visual_panel_count": strict_visual_panel_count,
    }
    strict_gate["stage2_count_gate_pass"] = (
        strict_gate["total_base_cases"] >= 12
        and strict_gate["read_cases"] >= 4
        and strict_gate["swa_cases"] >= 4
        and strict_gate["good_controls"] >= 4
    )
    strict_gate["stage2_strict_visual_gate_pass"] = (
        bool(case_rows)
        and rgb_overlay_case_count == len(case_rows)
        and traj_error_case_count == len(case_rows)
        and strict_visual_panel_count == len(case_rows)
    )
    strict_gate["stage2_strict_gate_pass"] = strict_gate["stage2_count_gate_pass"] and strict_gate["stage2_strict_visual_gate_pass"]
    strict_gate["stage2_diagnostic_gate_pass"] = strict_gate["stage2_count_gate_pass"] and strict_gate["all_cards_exist"] and strict_gate["all_metric_panels_exist"]
    if strict_gate["stage2_strict_visual_gate_pass"]:
        strict_gate["visual_limitation"] = (
            "Strict visual gate passed: RGB/semantic/risk overlays, trajectory error maps, and integrated local point residual panels "
            f"are available for every selected base case; strict_visual_panel={strict_visual_panel_count}/{len(case_rows)} and "
            f"local_point_error_map={local_point_error_case_count}/{len(case_rows)}."
        )
    elif rgb_overlay_case_count == len(case_rows) and traj_error_case_count == len(case_rows) and case_rows:
        strict_gate["visual_limitation"] = (
            "RGB/semantic/risk overlays and trajectory error maps are available for every selected base case, but strict visual gate is false because "
            f"strict_visual_panel={strict_visual_panel_count}/{len(case_rows)} and local_point_error_map={local_point_error_case_count}/{len(case_rows)}."
        )
    elif rgb_overlay_case_count == len(case_rows) and traj_error_case_count and case_rows:
        strict_gate["visual_limitation"] = (
            f"RGB/semantic/risk overlays are complete and trajectory error maps are partial ({traj_error_case_count}/{len(case_rows)}); "
            "strict visual gate remains false."
        )
    elif rgb_overlay_case_count == len(case_rows) and case_rows:
        strict_gate["visual_limitation"] = (
            "RGB/semantic/risk overlays are available for every selected base case, but strict visual gate is false because "
            f"strict_visual_panel={rgb_overlay_strict_count}/{len(case_rows)}; local point/trajectory error maps are not materialized."
        )
    elif rgb_overlay_case_count:
        strict_gate["visual_limitation"] = (
            f"RGB/semantic/risk overlay coverage is partial ({rgb_overlay_case_count}/{len(case_rows)}); "
            "strict visual gate remains false."
        )
    else:
        strict_gate["visual_limitation"] = "case cards and metric panels exist, but required RGB/semantic overlay panels are not available; strict visual gate is false."
    write_json(out / "stage2_summary.json", strict_gate)
    return strict_gate


def stage2_5() -> dict[str, Any]:
    src = ROOT / "stage0_evidence_ledger/legacy_route_ledger.csv"
    rows = read_csv_rows(src)
    out = ROOT / "stage2_5_legacy_reentry_matrix"
    ensure_dir(out)
    matrix = []
    oracle_inputs = []
    action_candidates = []
    neg = []
    for r in rows:
        rec = dict(r)
        usage = r.get("v102_usage", "")
        rec["assigned_stage3_oracle"] = ";".join(x for x in ["B6", "B7", "B8", "B9", "B10", "B4", "B5"] if x in usage)
        rec["assigned_stage4_surface"] = ""
        if "Stage4" in usage:
            if "READ" in usage:
                rec["assigned_stage4_surface"] = "READ_CURRENT_SUPPORT_PROVIDER"
            elif "SWA" in usage:
                rec["assigned_stage4_surface"] = "SWA_STATE_MACHINE_ORACLE"
            elif "TTT" in usage:
                rec["assigned_stage4_surface"] = "TTT_WRITE_TO_USE_STATE_MACHINE"
            elif "merge" in usage.lower():
                rec["assigned_stage4_surface"] = "MERGE_GAUGE_DIAGNOSTIC_ORACLE"
        matrix.append(rec)
        if rec["assigned_stage3_oracle"]:
            oracle_inputs.append(rec)
        if rec["assigned_stage4_surface"]:
            action_candidates.append(rec)
        if "negative_control" in rec.get("claim_type", ""):
            neg.append(rec)

    write_csv(out / "legacy_route_matrix.csv", matrix)
    write_csv(out / "oracle_inputs_from_legacy.csv", oracle_inputs)
    write_csv(out / "negative_control_registry.csv", neg)
    write_csv(out / "action_surface_candidate_registry.csv", action_candidates)
    md = ["# Stage 2.5 Legacy Re-entry Matrix\n\n"]
    for rec in matrix:
        md.append(f"- `{rec['legacy_id']}` -> oracle `{rec['assigned_stage3_oracle']}` / action `{rec['assigned_stage4_surface']}` / forbidden `{rec['forbidden_usage']}`\n")
    write_text(out / "legacy_route_matrix.md", "".join(md))
    summary = {
        "legacy_route_count": len(matrix),
        "required_legacy_route_count": 9,
        "claim_type_complete": all(bool(r.get("claim_type")) for r in matrix),
        "forbidden_usage_complete": all(bool(r.get("forbidden_usage")) for r in matrix),
        "stage3_route_count": len(oracle_inputs),
        "stage4_route_count": len(action_candidates),
        "negative_control_count": len(neg),
        "stage2_5_gate_pass": len(matrix) >= 9 and len(oracle_inputs) >= 4 and len(action_candidates) >= 3 and all(r.get("claim_type") and r.get("forbidden_usage") for r in matrix),
    }
    write_json(out / "stage2_5_summary.json", summary)
    return summary


def stage3() -> dict[str, Any]:
    out = ROOT / "stage3_semantic_oracle_upper_bound"
    ensure_dir(out)
    v97_h2 = read_json(SRC["v97_h2_summary"])
    v97_k = read_json(SRC["v97_trackk_summary"])
    v98_s1 = read_json(SRC["v98_stage1_summary"])
    v98_s2 = read_json(SRC["v98_stage2_summary"])
    v98_7e = read_json(SRC["v98_stage7e_summary"])
    v98_7g = read_json(SRC["v98_stage7g_summary"])
    v100_q = read_json(SRC["v100_trackq_summary"])
    v100_s = read_json(SRC["v100_tracks_summary"])
    v101_c = read_json(SRC["v101_combined_summary"])
    v101_g = read_json(SRC["v101_geometry_obs_summary"])
    v101_trackv = read_json(SRC["v101_trackv_observability_summary"])
    v101_trackv_anchor = read_json(SRC["v101_trackv_per_anchor_summary"])
    v102_local_geometry = read_json(LOCAL_GEOMETRY_ORACLE_SUMMARY)
    v102_local_best = v102_local_geometry.get("best_candidate") if isinstance(v102_local_geometry.get("best_candidate"), dict) else {}
    v102_full_control = read_json(FULL_CONTROL_SEMANTIC_ROTATION_SUMMARY)
    v102_full_best = v102_full_control.get("best_policy") if isinstance(v102_full_control.get("best_policy"), dict) else {}
    v102_clean_handoff = read_json(CLEAN_HANDOFF_EXPANSION_SUMMARY)
    v102_broader_trace = read_json(BROADER_DRIFT_ONSET_TRACE_EXTENSION_SUMMARY)
    v102_read_bridge = read_json(READ_LOCAL_ORACLE_BRIDGE_SUMMARY)
    v102_legacy_alignment = read_json(LEGACY_CUE_CASE_ALIGNMENT_SUMMARY)
    v102_historical_extension = read_json(HISTORICAL_CLEAN_TARGET_EXTENSION_SUMMARY)
    v102_strict_materialization = read_json(STRICT_CLEAN_HANDOFF_MATERIALIZATION_SUMMARY)
    v102_exhaustive_target_mining = read_json(EXHAUSTIVE_CLEAN_HANDOFF_TARGET_MINING_SUMMARY)

    b5_true_cases = split_cases((v100_q.get("best_composite_rule") or {}).get("true_positive_cases", ""))
    b7_cases = split_cases(v98_7g.get("best_cue_true_positive_cases", ""))
    b7_overlap = len(b7_cases & b5_true_cases) / len(b7_cases) if b7_cases else 0.0

    oracle_rows = [
        {
            "oracle_id": "B1",
            "name": "Stable Semantic Anchor Only",
            "evidence_path": rel(SRC["v98_stage1_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": False,
            "key_evidence": f"v98 raw/hygiene gate_pass={v98_s1.get('gate_pass')}; cue_gate_pass_count={v98_s1.get('cue_gate_pass_count')}",
            "blocker": v98_s1.get("blocker", "missing evidence"),
        },
        {
            "oracle_id": "B2",
            "name": "Reject Unreliable Evidence",
            "evidence_path": rel(SRC["v98_stage2_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": False,
            "key_evidence": f"v98 TrackL gate_pass={v98_s2.get('gate_pass')}; weak_context_collapse={v98_s2.get('weak_context_collapse')}",
            "blocker": v98_s2.get("blocker", "missing evidence"),
        },
        {
            "oracle_id": "B3",
            "name": "Scale-Observable Semantic Anchor",
            "evidence_path": rel(SRC["v98_stage2_summary"]) + ";" + rel(SRC["v101_trackv_per_anchor_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v101_trackv_anchor.get("geometry_materialization_pass")),
            "key_evidence": (
                f"v98 best_cue={v98_s2.get('best_cue')}; v98 gate_pass={v98_s2.get('gate_pass')}; "
                f"v101 geometry_materialization_pass={v101_trackv_anchor.get('geometry_materialization_pass')}; "
                f"true_geometry_source_available_frac={v101_trackv_anchor.get('true_geometry_source_available_frac')}"
            ),
            "blocker": f"TrackV gate_pass=false; blockers={v101_trackv_anchor.get('blockers')}",
        },
        {
            "oracle_id": "B4",
            "name": "Same-Space Consistent Anchor",
            "evidence_path": rel(SRC["v100_tracks_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v100_s.get("gate_pass")),
            "key_evidence": f"TrackS same-space instrumentation gate_pass={v100_s.get('gate_pass')}; runtime_action_allowed={v100_s.get('runtime_action_allowed')}",
            "blocker": "same-space instrumentation is provider evidence only; R_same alone is forbidden as action selector",
        },
        {
            "oracle_id": "B5",
            "name": "Chunk-Level Update Admission Oracle",
            "evidence_path": rel(SRC["v100_trackq_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v100_q.get("best_composite_q_gate_without_true_terms_pass")),
            "bad_recall": v100_q.get("best_composite_bad_recall"),
            "good_FPR": v100_q.get("best_composite_good_FPR"),
            "sequence_coverage": v100_q.get("sequence_coverage"),
            "same_count_random_margin": v100_q.get("best_same_count_random_margin"),
            "key_evidence": f"proxy q gate without true terms={v100_q.get('best_composite_q_gate_without_true_terms_pass')}; gate_pass={v100_q.get('gate_pass')}",
            "blocker": f"proxy_only={v100_q.get('proxy_only')}; missing_true_terms={v100_q.get('missing_true_terms')}",
        },
        {
            "oracle_id": "B6",
            "name": "Stage-C Seed/Component Continuity",
            "evidence_path": rel(SRC["v101_geometry_obs_summary"]) + ";" + rel(SRC["v101_trackv_per_anchor_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v101_g.get("lifecycle_combined_geometry_join_materialized")),
            "key_evidence": (
                f"combined_geometry_unique_coverage={v101_g.get('lifecycle_combined_geometry_unique_coverage')}; "
                f"handoff_join_coverage={v101_g.get('lifecycle_handoff_geometry_join_coverage')}; "
                f"per_anchor_true_geometry_source_available_frac={v101_trackv_anchor.get('true_geometry_source_available_frac')}; "
                f"handoff_target_case_count={v101_trackv_anchor.get('handoff_target_case_count')}"
            ),
            "blocker": "Stage-C seed/lifecycle join exists and true per-anchor geometry exists, but handoff target coverage/correlation/control reruns are insufficient for strict B6/B7.",
        },
        {
            "oracle_id": "B7",
            "name": "Internal-QK Semantic Handoff Oracle",
            "evidence_path": rel(SRC["v98_stage7g_summary"]) + ";" + rel(SRC["v100_trackq_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v98_7g.get("gate_pass") and b7_overlap >= 0.60),
            "selected_case_overlap_with_b5_proxy": b7_overlap,
            "best_cue_bad_recall": v98_7g.get("best_cue_bad_recall"),
            "best_cue_good_FPR": v98_7g.get("best_cue_good_FPR"),
            "sequence_coverage": v98_7g.get("sequence_coverage"),
            "key_evidence": f"Stage7g gate_pass={v98_7g.get('gate_pass')}; overlap_with_B5_proxy={b7_overlap}",
            "blocker": "legacy internal-QK cue overlaps proxy admission, but depends on B5 proxy/true-term gap; not a strict semantic oracle",
        },
        {
            "oracle_id": "B8",
            "name": "Cache K/V Stability and Top-k Identity Oracle",
            "evidence_path": rel(SRC["v98_stage7e_summary"]) + ";" + rel(SRC["v98_stage7g_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v98_7e.get("gate_pass") or v98_7g.get("gate_pass")),
            "key_evidence": f"Stage7e gate_pass={v98_7e.get('gate_pass')}; Stage7g gate_pass={v98_7g.get('gate_pass')}",
            "blocker": "diagnostic identity/cache carrier exists; no validated transmit/delay/reject action target effect",
        },
        {
            "oracle_id": "B9",
            "name": "TTT Identity Write-to-Use Oracle",
            "evidence_path": rel(SRC["v98_stage7e_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v98_7e.get("write_to_use_chain_available")),
            "key_evidence": f"write_to_use_chain_available={v98_7e.get('write_to_use_chain_available')}; gate_pass={v98_7e.get('gate_pass')}",
            "blocker": "write-to-use chain diagnostic exists, but stale/supported use groups are not tied to validated L3/L4 intervention",
        },
        {
            "oracle_id": "B10",
            "name": "READ Current-Support Oracle",
            "evidence_path": rel(SRC["v97_h2_summary"]) + ";" + rel(SRC["v100_trackd4_summary"]),
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v97_h2.get("gate_pass")),
            "bad_L2_improvement": (v97_h2.get("best_passing_component") or {}).get("bad_L2_improvement"),
            "good_worsen": (v97_h2.get("best_passing_component") or {}).get("good_worsen"),
            "key_evidence": f"v97 H2 local_L2_mechanism_exists={v97_h2.get('local_L2_mechanism_exists')}; v100 D4 provider gate={read_json(SRC['v100_trackd4_summary']).get('gate_pass')}",
            "blocker": "READ has local provider/mechanism evidence but no full method; current-support integration into SWA/TTT remains blocked",
        },
        {
            "oracle_id": "B10_READ_LOCAL_BRIDGE",
            "name": "READ Local Oracle Bridge Case Alignment",
            "evidence_path": rel(READ_LOCAL_ORACLE_BRIDGE_SUMMARY) if READ_LOCAL_ORACLE_BRIDGE_SUMMARY.exists() else "",
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v102_read_bridge.get("legacy_read_local_upper_bound_pass")),
            "bad_L2_improvement": v102_read_bridge.get("legacy_h2_best_bad_L2_improvement"),
            "good_worsen": v102_read_bridge.get("legacy_h2_best_good_worsen"),
            "control_margin": v102_read_bridge.get("legacy_h2_best_control_margin"),
            "v102_case_aligned_read_local_oracle_pass": v102_read_bridge.get("v102_case_aligned_read_local_oracle_pass"),
            "v102_strict_read_local_source_overlap_count": v102_read_bridge.get("v102_strict_read_local_source_overlap_count"),
            "v102_good_control_source_overlap_count": v102_read_bridge.get("v102_good_control_source_overlap_count"),
            "full_sequence_no_go": v102_read_bridge.get("full_sequence_no_go"),
            "key_evidence": (
                f"legacy_upper_bound={v102_read_bridge.get('legacy_read_local_upper_bound_pass')}; "
                f"legacy_bad_L2={v102_read_bridge.get('legacy_h2_best_bad_L2_improvement')}; "
                f"legacy_good_worsen={v102_read_bridge.get('legacy_h2_best_good_worsen')}; "
                f"v102_strict_overlap={v102_read_bridge.get('v102_strict_read_local_source_overlap_count')}; "
                f"v102_good_control_overlap={v102_read_bridge.get('v102_good_control_source_overlap_count')}; "
                f"full_sequence_strict_pass_count={v102_read_bridge.get('full_sequence_strict_pass_count')}"
            ),
            "blocker": v102_read_bridge.get(
                "blocker",
                "READ local bridge audit missing; legacy READ provider cannot be promoted without v102 case-aligned coverage.",
            ),
        },
        {
            "oracle_id": "B6_B10_LEGACY_CUE_CASE_ALIGNMENT",
            "name": "Legacy Cue Case Alignment",
            "evidence_path": rel(LEGACY_CUE_CASE_ALIGNMENT_SUMMARY) if LEGACY_CUE_CASE_ALIGNMENT_SUMMARY.exists() else "",
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v102_legacy_alignment.get("legacy_diagnostic_gate_pass_count")),
            "sequence_coverage": v102_legacy_alignment.get("max_selected_strict_clean_handoff_count"),
            "key_evidence": (
                f"route_count={v102_legacy_alignment.get('route_count')}; "
                f"legacy_diagnostic_gate_pass_count={v102_legacy_alignment.get('legacy_diagnostic_gate_pass_count')}; "
                f"legacy_action_gate_pass_count={v102_legacy_alignment.get('legacy_action_gate_pass_count')}; "
                f"v102_strict_clean_handoff_universe_count={v102_legacy_alignment.get('v102_strict_clean_handoff_universe_count')}; "
                f"strict_promotion_allowed_count={v102_legacy_alignment.get('strict_promotion_allowed_count')}; "
                f"best_aligned_route_id={v102_legacy_alignment.get('best_aligned_route_id')}"
            ),
            "blocker": v102_legacy_alignment.get(
                "blocker",
                "legacy cue case-alignment audit missing; B6-B10 legacy cues remain proxy/provider evidence only.",
            ),
        },
        {
            "oracle_id": "B6_B10_HISTORICAL_CLEAN_TARGET_EXTENSION",
            "name": "Historical Clean Target Extension",
            "evidence_path": rel(HISTORICAL_CLEAN_TARGET_EXTENSION_SUMMARY) if HISTORICAL_CLEAN_TARGET_EXTENSION_SUMMARY.exists() else "",
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v102_historical_extension.get("historical_row_count")),
            "sequence_coverage": v102_historical_extension.get("v102_strict_extension_candidate_count"),
            "key_evidence": (
                f"historical_row_count={v102_historical_extension.get('historical_row_count')}; "
                f"historical_clean_handoff_candidate_count={v102_historical_extension.get('historical_clean_handoff_candidate_count')}; "
                f"usable_new_extension_case_count={v102_historical_extension.get('usable_new_extension_case_count')}; "
                f"v102_strict_extension_candidate_count={v102_historical_extension.get('v102_strict_extension_candidate_count')}; "
                f"rich_holdout_fresh_labelled_bad_good_pair_count={v102_historical_extension.get('rich_holdout_fresh_labelled_bad_good_pair_count')}; "
                f"trace_rescue_available={v102_historical_extension.get('trace_rescue_available')}"
            ),
            "blocker": v102_historical_extension.get(
                "blocker",
                "historical clean-target extension audit missing; strict clean handoff target expansion is unproven.",
            ),
        },
        {
            "oracle_id": "B3_B5_LOCAL_GEOMETRY_REPAIR",
            "name": "Selected-Case Local Geometry Oracle Repair",
            "evidence_path": rel(LOCAL_GEOMETRY_ORACLE_SUMMARY) if LOCAL_GEOMETRY_ORACLE_SUMMARY.exists() else "",
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v102_local_geometry.get("selected_case_oracle_pass_count")),
            "bad_recall": v102_local_best.get("bad_recall"),
            "good_FPR": v102_local_best.get("good_FPR"),
            "sequence_coverage": v102_local_best.get("sequence_coverage"),
            "same_count_random_margin": v102_local_best.get("same_count_random_margin"),
            "key_evidence": (
                f"selected_case_oracle_pass_count={v102_local_geometry.get('selected_case_oracle_pass_count')}; "
                f"best_target={v102_local_best.get('target_label')}; best_feature={v102_local_best.get('feature')}; "
                f"strict_visual_case_count={v102_local_geometry.get('strict_visual_case_count')}"
            ),
            "blocker": v102_local_geometry.get(
                "strict_blocker",
                "selected-case local geometry audit missing; full target-universe controls not rerun",
            ),
        },
        {
            "oracle_id": "B2_B3_B6_FULL_CONTROL_SEMANTIC_ROTATION",
            "name": "Full-Control Semantic/Observability Rotation Audit",
            "evidence_path": rel(FULL_CONTROL_SEMANTIC_ROTATION_SUMMARY) if FULL_CONTROL_SEMANTIC_ROTATION_SUMMARY.exists() else "",
            "strict_oracle_pass": False,
            "diagnostic_or_proxy_pass": bool(v102_full_control.get("exploratory_control_signal_present")),
            "bad_recall": v102_full_best.get("bad_recall"),
            "good_FPR": v102_full_best.get("good_FPR"),
            "sequence_coverage": v102_full_best.get("sequence_coverage"),
            "same_count_random_margin": v102_full_best.get("same_count_random_margin"),
            "semantic_label_rotation_margin": v102_full_best.get("semantic_label_rotation_margin"),
            "anchor_id_geometry_rotation_margin": v102_full_best.get("anchor_id_geometry_rotation_margin"),
            "key_evidence": (
                f"input_case_count={v102_full_control.get('input_case_count')}; "
                f"input_per_anchor_row_count={v102_full_control.get('input_per_anchor_row_count')}; "
                f"best_scope={v102_full_best.get('eval_scope')}; best_feature={v102_full_best.get('score_field')}; "
                f"full_control_oracle_pass_count={v102_full_control.get('full_control_oracle_pass_count')}; "
                f"strict_promotion_pass_count={v102_full_control.get('strict_promotion_pass_count')}"
            ),
            "blocker": "; ".join(v102_full_control.get("blockers", []))
            or "full-control semantic rotation audit missing or did not materialize controls",
        },
    ]
    write_csv(out / "semantic_oracle_rows.csv", oracle_rows)

    fpfn = [
        {
            "failure_axis": "anchor_mask_alignment",
            "status": "repaired_for_selected_base_cases",
            "evidence": "RGB/semantic/risk overlays, trajectory maps, and integrated local point residual panels cover 21/21 selected base cases.",
            "repair_direction": "no further visual repair required before Stage3; remaining blocker is full-universe oracle/control evidence",
        },
        {
            "failure_axis": "current_support_missing",
            "status": "confirmed_gap",
            "evidence": f"v101 geometry/lifecycle support coverage={v101_g.get('lifecycle_combined_geometry_unique_coverage')}",
            "repair_direction": "improve anchor_id -> Stage-C seed -> geometry sidecar support join; do not use semantic class fallback",
        },
        {
            "failure_axis": "scale_observability_missing",
            "status": "materialized_but_not_passing",
            "evidence": (
                f"v101 TrackV true_geometry_source_available_frac={v101_trackv_anchor.get('true_geometry_source_available_frac')}; "
                f"gate_pass={v101_trackv_anchor.get('gate_pass')}; blockers={v101_trackv_anchor.get('blockers')}; "
                f"v102 selected_case_oracle_pass_count={v102_local_geometry.get('selected_case_oracle_pass_count')}; "
                f"best_selected_case_feature={v102_local_best.get('feature')}; "
                f"best_selected_case_margin={v102_local_best.get('same_count_random_margin')}; "
                f"v102_full_control_best_feature={v102_full_best.get('score_field')}; "
                f"v102_full_control_same_count_margin={v102_full_best.get('same_count_random_margin')}; "
                f"v102_full_control_semantic_rotation_margin={v102_full_best.get('semantic_label_rotation_margin')}; "
                f"v102_full_control_strict_promotion_pass_count={v102_full_control.get('strict_promotion_pass_count')}"
            ),
            "repair_direction": "full-control semantic/anchor rotation is now materialized; next repair must address strict clean handoff positive coverage and true action-surface L3/L4 effect rather than selected-case-only promotion.",
        },
        {
            "failure_axis": "full_control_semantic_rotation",
            "status": "exploratory_signal_but_strict_blocked",
            "evidence": (
                f"best_scope={v102_full_best.get('eval_scope')}; best_feature={v102_full_best.get('score_field')}; "
                f"bad_recall={v102_full_best.get('bad_recall')}; good_FPR={v102_full_best.get('good_FPR')}; "
                f"same_count_margin={v102_full_best.get('same_count_random_margin')}; "
                f"semantic_rotation_margin={v102_full_best.get('semantic_label_rotation_margin')}; "
                f"clean_handoff_positive_count={v102_full_control.get('clean_handoff_positive_count')}; "
                f"strict_promotion_pass_count={v102_full_control.get('strict_promotion_pass_count')}"
            ),
            "repair_direction": "do not authorize Stage4 from the exploratory contaminated/multimode scope; materialize more clean handoff positives or an action-surface upper bound with true L3 effect.",
        },
        {
            "failure_axis": "clean_handoff_candidate_expansion",
            "status": "exploration_candidates_found_but_strict_coverage_unrepaired",
            "evidence": (
                f"swa_handoff_candidate_count={v102_clean_handoff.get('swa_handoff_candidate_count')}; "
                f"clean_handoff_like_candidate_count={v102_clean_handoff.get('clean_handoff_like_candidate_count')}; "
                f"clean_handoff_like_sequence_count={v102_clean_handoff.get('clean_handoff_like_sequence_count')}; "
                f"clean_handoff_like_trace_sidecar_ready_count={v102_clean_handoff.get('clean_handoff_like_trace_sidecar_ready_count')}; "
                f"strict_clean_handoff_positive_count={v102_clean_handoff.get('strict_clean_handoff_positive_count')}; "
                f"exploration_only_candidate_cases={v102_clean_handoff.get('exploration_only_candidate_cases')}; "
                f"clean_like_selected_by_best_unreliable_semantic_cases={v102_clean_handoff.get('clean_like_selected_by_best_unreliable_semantic_cases')}; "
                f"broader_local_goodish_high_l3_count={v102_clean_handoff.get('broader_local_goodish_high_l3_count')}; "
                f"broader_exploration_only_cases={v102_clean_handoff.get('broader_exploration_only_cases')}; "
                f"broader_missing_full_control_cases={v102_clean_handoff.get('broader_missing_full_control_cases')}; "
                f"broader_trace_sidecar_ready_count={v102_clean_handoff.get('broader_trace_sidecar_ready_count')}; "
                f"broader_v102_trace_extension_ready_cases={v102_clean_handoff.get('broader_v102_trace_extension_ready_cases')}"
            ),
            "repair_direction": "ambiguous clean-handoff-like and broader drift-onset cases can guide exploration, but strict Stage3 still needs clean labels, full-control support for missing broader cases, or a true action-surface L3 upper bound.",
        },
        {
            "failure_axis": "strict_clean_handoff_materialization_repair",
            "status": "no_existing_candidate_upgradable_to_strict",
            "evidence": (
                f"candidate_case_count={v102_strict_materialization.get('candidate_case_count')}; "
                f"current_strict_clean_handoff_positive_count={v102_strict_materialization.get('current_strict_clean_handoff_positive_count')}; "
                f"current_strict_clean_handoff_positive_cases={v102_strict_materialization.get('current_strict_clean_handoff_positive_cases')}; "
                f"additional_strict_positive_needed_count={v102_strict_materialization.get('additional_strict_positive_needed_count')}; "
                f"strict_upgrade_possible_from_existing_evidence_count={v102_strict_materialization.get('strict_upgrade_possible_from_existing_evidence_count')}; "
                f"ambiguous_swa_handoff_cases={v102_strict_materialization.get('ambiguous_swa_handoff_cases')}"
            ),
            "repair_direction": "do not promote ambiguous cases from existing evidence; materialize fresh labelled clean bad SWA handoff targets with reliable overlap, full-control rows, trace/sidecar, current support, query-head controls, and true action-surface terms.",
        },
        {
            "failure_axis": "exhaustive_clean_handoff_target_mining",
            "status": "no_hidden_strict_positive_in_stage1",
            "evidence": (
                f"stage1_case_count={v102_exhaustive_target_mining.get('stage1_case_count')}; "
                f"exhaustive_strict_clean_handoff_positive_count={v102_exhaustive_target_mining.get('exhaustive_strict_clean_handoff_positive_count')}; "
                f"exhaustive_strict_clean_handoff_positive_cases={v102_exhaustive_target_mining.get('exhaustive_strict_clean_handoff_positive_cases')}; "
                f"candidate_generation_missed_strict_positive_count={v102_exhaustive_target_mining.get('candidate_generation_missed_strict_positive_count')}; "
                f"additional_strict_positive_needed_count={v102_exhaustive_target_mining.get('additional_strict_positive_needed_count')}; "
                f"top_external_materialization_worklist_cases={v102_exhaustive_target_mining.get('top_external_materialization_worklist_cases')}"
            ),
            "repair_direction": "candidate generation is not the blocker; fresh/manual labelled clean handoff positives or a measured true L3/L4 state-machine action surface are still required before Stage4/5/6.",
        },
        {
            "failure_axis": "broader_drift_onset_trace_extension",
            "status": "missing_support_materialized_but_strict_coverage_unrepaired",
            "evidence": (
                f"runner_status={v102_broader_trace.get('runner_status')}; "
                f"trace_sidecar_materialized_count={v102_broader_trace.get('trace_sidecar_materialized_count')}; "
                f"trace_sidecar_materialized_cases={v102_broader_trace.get('trace_sidecar_materialized_cases')}; "
                f"all_targets_trace_sidecar_materialized={v102_broader_trace.get('all_targets_trace_sidecar_materialized')}; "
                f"strict_stage3_coverage_repaired={v102_broader_trace.get('strict_stage3_coverage_repaired')}"
            ),
            "repair_direction": "the missing broader drift-onset trace support is now materialized; next evidence must either add strict clean handoff positives or show true L3 action-surface movement.",
        },
        {
            "failure_axis": "read_local_oracle_bridge",
            "status": "legacy_local_upper_bound_but_v102_case_alignment_insufficient",
            "evidence": (
                f"legacy_read_local_upper_bound_pass={v102_read_bridge.get('legacy_read_local_upper_bound_pass')}; "
                f"legacy_h2_best_bad_L2_improvement={v102_read_bridge.get('legacy_h2_best_bad_L2_improvement')}; "
                f"legacy_h2_best_good_worsen={v102_read_bridge.get('legacy_h2_best_good_worsen')}; "
                f"v102_strict_read_local_source_overlap_count={v102_read_bridge.get('v102_strict_read_local_source_overlap_count')}; "
                f"v102_good_control_source_overlap_count={v102_read_bridge.get('v102_good_control_source_overlap_count')}; "
                f"full_sequence_strict_pass_count={v102_read_bridge.get('full_sequence_strict_pass_count')}; "
                f"global_scale_shift_available={v102_read_bridge.get('global_scale_shift_available')}"
            ),
            "repair_direction": "keep READ as provider/local oracle; do not promote to strict Stage3 or runtime until v102 case-aligned READ_LOCAL/control coverage and global scale/yaw evidence are materialized.",
        },
        {
            "failure_axis": "legacy_cue_case_alignment",
            "status": "legacy_diagnostics_real_but_case_alignment_and_action_bodies_fail",
            "evidence": (
                f"route_count={v102_legacy_alignment.get('route_count')}; "
                f"legacy_diagnostic_gate_pass_count={v102_legacy_alignment.get('legacy_diagnostic_gate_pass_count')}; "
                f"legacy_action_gate_pass_count={v102_legacy_alignment.get('legacy_action_gate_pass_count')}; "
                f"v102_strict_clean_handoff_universe_count={v102_legacy_alignment.get('v102_strict_clean_handoff_universe_count')}; "
                f"max_selected_strict_clean_handoff_count={v102_legacy_alignment.get('max_selected_strict_clean_handoff_count')}; "
                f"strict_action_frontier_ready_count={v102_legacy_alignment.get('strict_action_frontier_ready_count')}; "
                f"best_aligned_route_id={v102_legacy_alignment.get('best_aligned_route_id')}"
            ),
            "repair_direction": "preserve B6-B10 legacy cues as provider/oracle-cue evidence, but do not choose one cue and continue action until clean target coverage, query-head/rotation controls, and true action-surface L3/L4 effects are materialized.",
        },
        {
            "failure_axis": "historical_clean_target_extension",
            "status": "no_new_strict_clean_target_universe_in_existing_artifacts",
            "evidence": (
                f"historical_row_count={v102_historical_extension.get('historical_row_count')}; "
                f"historical_clean_handoff_candidate_count={v102_historical_extension.get('historical_clean_handoff_candidate_count')}; "
                f"historical_clean_handoff_candidate_cases={v102_historical_extension.get('historical_clean_handoff_candidate_cases')}; "
                f"usable_new_extension_case_count={v102_historical_extension.get('usable_new_extension_case_count')}; "
                f"v102_strict_extension_candidate_count={v102_historical_extension.get('v102_strict_extension_candidate_count')}; "
                f"rich_holdout_fresh_labelled_bad_good_pair_count={v102_historical_extension.get('rich_holdout_fresh_labelled_bad_good_pair_count')}; "
                f"trace_rescue_available={v102_historical_extension.get('trace_rescue_available')}"
            ),
            "repair_direction": "existing artifacts cannot expand the strict clean handoff universe; a real repair needs newly materialized labelled clean handoff targets with same-space trace, per-anchor geometry, strict instance identity, query-head controls, write/cache/current chain, and Q2 true-stage evidence.",
        },
        {
            "failure_axis": "target_mixed",
            "status": "partially_repaired_by_stage1_taxonomy",
            "evidence": "Stage1 writes drift_source_taxonomy.csv and separates READ/SWA/LOWOBS/SAFE_GOOD labels.",
            "repair_direction": "run oracle metrics per drift source before any runtime pilot",
        },
    ]
    write_csv(out / "semantic_oracle_failure_decomposition.csv", fpfn)

    base_rows = read_csv_rows(ROOT / "stage2_base_case_selection/base_case_rows.csv")
    base_by_case = first_by_case(base_rows)
    q_fp_missed_rows = read_csv_rows(SRC["v100_trackq_fp_missed_rows"])
    combined_fpfn_rows = read_csv_rows(SRC["v101_combined_fpfn_rows"])
    b5_rule = v100_q.get("best_composite_rule") or {}
    b5_false_positive_cases = split_cases(b5_rule.get("false_positive_cases"))
    b5_missed_cases = split_cases(b5_rule.get("missed_positive_cases"))
    b5_true_positive_cases = split_cases(b5_rule.get("true_positive_cases"))
    b7_true_positive_cases = split_cases(v98_7g.get("best_cue_true_positive_cases"))

    panel_rows: list[dict[str, Any]] = []
    q_by_kind_case = {(r.get("row_kind", ""), r.get("case_id", "")): r for r in q_fp_missed_rows}

    for case_id in sorted(b5_false_positive_cases):
        ctx = compact_case_context(case_id, base_by_case)
        qrow = q_by_kind_case.get(("false_positive", case_id), {})
        if not ctx.get("L3_handoff_transfer_penalty_proxy"):
            ctx["L3_handoff_transfer_penalty_proxy"] = qrow.get("L3_handoff_transfer_penalty_proxy", "")
        if not ctx.get("target_taxonomy_v101"):
            ctx["target_taxonomy_v101"] = qrow.get("case_label", "")
        panel_rows.append(
            {
                **ctx,
                "panel_type": "semantic_oracle_false_positive",
                "source_oracle": "B5_chunk_update_admission_proxy",
                "source_path": rel(SRC["v100_trackq_fp_missed_rows"]),
                "proxy_score": qrow.get("field_value", ""),
                "evidence_note": "B5 proxy selected this good/control case; strict promotion blocked by good-FPR and missing true terms.",
                "conflict_type": "good_control_or_label_L3_conflict_selected_by_proxy",
            }
        )
    for row in combined_fpfn_rows:
        if row.get("row_kind") != "false_positive_control":
            continue
        case_id = str(row.get("case_id", ""))
        if not case_id:
            continue
        ctx = compact_case_context(case_id, base_by_case)
        if not ctx.get("L3_handoff_transfer_penalty_proxy"):
            ctx["L3_handoff_transfer_penalty_proxy"] = row.get("L3_handoff_transfer_penalty_proxy", "")
        if not ctx.get("target_taxonomy_v101"):
            ctx["target_taxonomy_v101"] = row.get("target_taxonomy", "")
        panel_rows.append(
            {
                **ctx,
                "panel_type": "semantic_oracle_false_positive",
                "source_oracle": "B6_B7_combined_masklet_geometry",
                "source_path": rel(SRC["v101_combined_fpfn_rows"]),
                "proxy_score": row.get("score_value", ""),
                "evidence_note": f"v101 combined geometry policy `{row.get('policy_name', '')}` selected a non-handoff/control row.",
                "conflict_type": "geometry_or_masklet_policy_false_positive_control",
            }
        )

    for case_id in sorted(b5_missed_cases):
        ctx = compact_case_context(case_id, base_by_case)
        qrow = q_by_kind_case.get(("missed_positive", case_id), {})
        if not ctx.get("L3_handoff_transfer_penalty_proxy"):
            ctx["L3_handoff_transfer_penalty_proxy"] = qrow.get("L3_handoff_transfer_penalty_proxy", "")
        if not ctx.get("target_taxonomy_v101"):
            ctx["target_taxonomy_v101"] = qrow.get("case_label", "")
        panel_rows.append(
            {
                **ctx,
                "panel_type": "semantic_oracle_missed_positive",
                "source_oracle": "B5_chunk_update_admission_proxy",
                "source_path": rel(SRC["v100_trackq_fp_missed_rows"]),
                "proxy_score": qrow.get("field_value", ""),
                "evidence_note": "B5 proxy missed this non-good/high-L3 positive; true current-support and scale-observability terms are not sufficient to rescue it.",
                "conflict_type": "positive_missed_by_proxy_admission",
            }
        )
    for row in combined_fpfn_rows:
        if row.get("row_kind") != "missed_handoff_positive":
            continue
        case_id = str(row.get("case_id", ""))
        if not case_id:
            continue
        ctx = compact_case_context(case_id, base_by_case)
        if not ctx.get("L3_handoff_transfer_penalty_proxy"):
            ctx["L3_handoff_transfer_penalty_proxy"] = row.get("L3_handoff_transfer_penalty_proxy", "")
        if not ctx.get("target_taxonomy_v101"):
            ctx["target_taxonomy_v101"] = row.get("target_taxonomy", "")
        panel_rows.append(
            {
                **ctx,
                "panel_type": "semantic_oracle_missed_positive",
                "source_oracle": "B6_B7_combined_masklet_geometry",
                "source_path": rel(SRC["v101_combined_fpfn_rows"]),
                "proxy_score": row.get("score_value", ""),
                "evidence_note": f"v101 combined geometry policy `{row.get('policy_name', '')}` missed a handoff-positive row.",
                "conflict_type": "geometry_or_masklet_policy_missed_handoff_positive",
            }
        )

    # Deduplicate exact source/case/type repeats while preserving different policy rows.
    dedup_panel_rows: list[dict[str, Any]] = []
    seen_panel = set()
    for row in panel_rows:
        key = (row.get("panel_type"), row.get("source_oracle"), row.get("case_id"), row.get("proxy_score"), row.get("evidence_note"))
        if key in seen_panel:
            continue
        seen_panel.add(key)
        if not row.get("primary_drift_source"):
            row["primary_drift_source"] = "not_in_v102_base_cases"
        dedup_panel_rows.append(row)
    write_csv(out / "semantic_oracle_case_panel_rows.csv", dedup_panel_rows)

    false_positive_rows = [r for r in dedup_panel_rows if r.get("panel_type") == "semantic_oracle_false_positive"]
    missed_rows = [r for r in dedup_panel_rows if r.get("panel_type") == "semantic_oracle_missed_positive"]
    panel_columns = [
        "case_id",
        "source_oracle",
        "target_taxonomy_v101",
        "primary_drift_source",
        "L3_handoff_transfer_penalty_proxy",
        "proxy_score",
        "conflict_type",
        "evidence_note",
        "legacy_visual_panel_status",
    ]
    write_text(
        out / "semantic_oracle_false_positive_panels.md",
        "# Semantic Oracle False Positive Panels\n\n"
        "These rows are artifact-backed false-positive examples from v100/v101 oracle inputs. They explain why diagnostic/proxy cues cannot be promoted to a strict semantic oracle.\n\n"
        + md_table(false_positive_rows, panel_columns)
        + "\nStrict interpretation: false positives remain blockers unless true current-support, scale-observability controls, and RGB/semantic visual confirmation are materialized.\n",
    )
    write_text(
        out / "semantic_oracle_missed_positive_panels.md",
        "# Semantic Oracle Missed Positive Panels\n\n"
        "These rows are artifact-backed missed-positive examples from v100/v101 oracle inputs. They identify target cases that current proxy/geometry policies did not capture.\n\n"
        + md_table(missed_rows, panel_columns)
        + "\nStrict interpretation: missed positives show that the current semantic/proxy oracle is not yet a reliable upper-bound selector for all handoff/local target cases.\n",
    )

    conflict_rows: list[dict[str, Any]] = []
    conflict_cases = sorted(b5_true_positive_cases | b5_false_positive_cases | b5_missed_cases | b7_true_positive_cases)
    for case_id in conflict_cases:
        ctx = compact_case_context(case_id, base_by_case)
        if not ctx.get("primary_drift_source"):
            ctx["primary_drift_source"] = "not_in_v102_base_cases"
        if not ctx.get("target_taxonomy_v101"):
            ctx["target_taxonomy_v101"] = "not_in_v102_base_cases"
        b5_state = (
            "true_positive" if case_id in b5_true_positive_cases
            else "false_positive" if case_id in b5_false_positive_cases
            else "missed_positive" if case_id in b5_missed_cases
            else "not_selected"
        )
        b7_state = "true_positive" if case_id in b7_true_positive_cases else "not_selected"
        if b5_state == "true_positive" and b7_state == "true_positive":
            conflict_type = "proxy_and_internal_qk_agree_but_true_terms_missing"
        elif b5_state == "missed_positive" and b7_state == "not_selected":
            conflict_type = "both_proxy_and_internal_qk_missed_or_not_materialized"
        elif b5_state == "false_positive":
            conflict_type = "proxy_false_positive_requires_control_repair"
        else:
            conflict_type = "cue_disagreement_or_partial_overlap"
        conflict_rows.append(
            {
                **ctx,
                "b5_proxy_state": b5_state,
                "b7_internal_qk_state": b7_state,
                "trackv_true_geometry_available_frac": v101_trackv_anchor.get("true_geometry_source_available_frac"),
                "trackv_handoff_target_case_count": v101_trackv_anchor.get("handoff_target_case_count"),
                "conflict_type": conflict_type,
                "strict_action_allowed": False,
            }
        )
    write_csv(out / "three_way_legacy_cue_conflict_rows.csv", conflict_rows)
    write_text(
        out / "three_way_legacy_cue_conflict_panel.md",
        "# Three-Way Legacy Cue Conflict Panel\n\n"
        "This panel compares B5 proxy admission, B7 internal-QK selected cases, and v101 TrackV true-geometry availability. "
        "It is diagnostic only: TrackV true geometry availability is high, but handoff target coverage/control is insufficient, so no cue is promoted alone.\n\n"
        + md_table(
            conflict_rows,
            [
                "case_id",
                "b5_proxy_state",
                "b7_internal_qk_state",
                "target_taxonomy_v101",
                "primary_drift_source",
                "trackv_true_geometry_available_frac",
                "trackv_handoff_target_case_count",
                "conflict_type",
            ],
        )
        + "\nPlan A5.3 interpretation: when semantic/proxy, internal-QK, and Stage-C/geometry evidence disagree or only partially overlap, do not choose one cue and continue action; repair support/control first.\n",
    )
    write_text(
        out / "oracle_false_positive_false_negative.md",
        "# Oracle FP/FN Notes\n\n"
        "B5/B7/B8 show diagnostic/proxy separation, but strict semantic oracle promotion is blocked by true current-support, "
        "scale-observability controls, and missing full target-universe control reruns. Strict visual evidence is now available for selected base cases. No runtime action is authorized here.\n\n"
        "Detailed panels:\n\n"
        "- `semantic_oracle_false_positive_panels.md`\n"
        "- `semantic_oracle_missed_positive_panels.md`\n"
        "- `three_way_legacy_cue_conflict_panel.md`\n",
    )
    diagnostic_passes = [r for r in oracle_rows if bval(r.get("diagnostic_or_proxy_pass"))]
    strict_passes = [r for r in oracle_rows if bval(r.get("strict_oracle_pass"))]
    summary = {
        "schema": "acl2_v102_stage3_semantic_oracle_upper_bound_v1",
        "oracle_count": len(oracle_rows),
        "diagnostic_or_proxy_pass_count": len(diagnostic_passes),
        "diagnostic_or_proxy_pass_oracles": [r["oracle_id"] for r in diagnostic_passes],
        "strict_oracle_pass_count": len(strict_passes),
        "strict_oracle_pass_oracles": [r["oracle_id"] for r in strict_passes],
        "selected_case_local_geometry_oracle_signal_present": bool(v102_local_geometry.get("selected_case_oracle_pass_count")),
        "selected_case_local_geometry_best_candidate": v102_local_best,
        "full_control_semantic_rotation_exploratory_signal_present": bool(v102_full_control.get("exploratory_control_signal_present")),
        "full_control_semantic_rotation_best_policy": v102_full_best,
        "full_control_semantic_rotation_strict_promotion_pass_count": v102_full_control.get("strict_promotion_pass_count"),
        "full_control_semantic_rotation_blockers": v102_full_control.get("blockers"),
        "clean_handoff_expansion_summary": {
            "clean_handoff_like_candidate_count": v102_clean_handoff.get("clean_handoff_like_candidate_count"),
            "clean_handoff_like_sequence_count": v102_clean_handoff.get("clean_handoff_like_sequence_count"),
            "strict_clean_handoff_positive_count": v102_clean_handoff.get("strict_clean_handoff_positive_count"),
            "exploration_only_candidate_count": v102_clean_handoff.get("exploration_only_candidate_count"),
            "broader_local_goodish_high_l3_count": v102_clean_handoff.get("broader_local_goodish_high_l3_count"),
            "broader_local_goodish_high_l3_sequence_count": v102_clean_handoff.get("broader_local_goodish_high_l3_sequence_count"),
            "broader_exploration_only_count": v102_clean_handoff.get("broader_exploration_only_count"),
            "broader_missing_full_control_case_row_count": v102_clean_handoff.get("broader_missing_full_control_case_row_count"),
            "broader_missing_full_control_cases": v102_clean_handoff.get("broader_missing_full_control_cases"),
            "broader_trace_sidecar_ready_count": v102_clean_handoff.get("broader_trace_sidecar_ready_count"),
            "broader_v102_trace_extension_ready_cases": v102_clean_handoff.get("broader_v102_trace_extension_ready_cases"),
            "broader_trace_extension_materialized_count": v102_broader_trace.get("trace_sidecar_materialized_count"),
            "broader_trace_extension_materialized_cases": v102_broader_trace.get("trace_sidecar_materialized_cases"),
            "strict_stage3_coverage_repaired": v102_clean_handoff.get("strict_stage3_coverage_repaired"),
        },
        "read_local_oracle_bridge_summary": {
            "legacy_read_local_upper_bound_pass": v102_read_bridge.get("legacy_read_local_upper_bound_pass"),
            "v102_case_aligned_read_local_oracle_pass": v102_read_bridge.get("v102_case_aligned_read_local_oracle_pass"),
            "v102_strict_read_local_source_overlap_count": v102_read_bridge.get("v102_strict_read_local_source_overlap_count"),
            "v102_good_control_source_overlap_count": v102_read_bridge.get("v102_good_control_source_overlap_count"),
            "full_sequence_no_go": v102_read_bridge.get("full_sequence_no_go"),
            "global_scale_shift_available": v102_read_bridge.get("global_scale_shift_available"),
        },
        "legacy_cue_case_alignment_summary": {
            "route_count": v102_legacy_alignment.get("route_count"),
            "legacy_diagnostic_gate_pass_count": v102_legacy_alignment.get("legacy_diagnostic_gate_pass_count"),
            "legacy_action_gate_pass_count": v102_legacy_alignment.get("legacy_action_gate_pass_count"),
            "v102_strict_clean_handoff_universe_count": v102_legacy_alignment.get("v102_strict_clean_handoff_universe_count"),
            "strict_promotion_allowed_count": v102_legacy_alignment.get("strict_promotion_allowed_count"),
            "best_aligned_route_id": v102_legacy_alignment.get("best_aligned_route_id"),
        },
        "historical_clean_target_extension_summary": {
            "historical_clean_handoff_candidate_count": v102_historical_extension.get("historical_clean_handoff_candidate_count"),
            "historical_clean_handoff_candidate_cases": v102_historical_extension.get("historical_clean_handoff_candidate_cases"),
            "usable_new_extension_case_count": v102_historical_extension.get("usable_new_extension_case_count"),
            "v102_strict_extension_candidate_count": v102_historical_extension.get("v102_strict_extension_candidate_count"),
            "rich_holdout_fresh_labelled_bad_good_pair_count": v102_historical_extension.get("rich_holdout_fresh_labelled_bad_good_pair_count"),
            "trace_rescue_available": v102_historical_extension.get("trace_rescue_available"),
        },
        "strict_clean_handoff_materialization_repair_summary": {
            "candidate_case_count": v102_strict_materialization.get("candidate_case_count"),
            "current_strict_clean_handoff_positive_count": v102_strict_materialization.get("current_strict_clean_handoff_positive_count"),
            "current_strict_clean_handoff_positive_cases": v102_strict_materialization.get("current_strict_clean_handoff_positive_cases"),
            "additional_strict_positive_needed_count": v102_strict_materialization.get("additional_strict_positive_needed_count"),
            "strict_upgrade_possible_from_existing_evidence_count": v102_strict_materialization.get("strict_upgrade_possible_from_existing_evidence_count"),
            "ambiguous_swa_handoff_cases": v102_strict_materialization.get("ambiguous_swa_handoff_cases"),
        },
        "exhaustive_clean_handoff_target_mining_summary": {
            "stage1_case_count": v102_exhaustive_target_mining.get("stage1_case_count"),
            "exhaustive_strict_clean_handoff_positive_count": v102_exhaustive_target_mining.get("exhaustive_strict_clean_handoff_positive_count"),
            "exhaustive_strict_clean_handoff_positive_cases": v102_exhaustive_target_mining.get("exhaustive_strict_clean_handoff_positive_cases"),
            "candidate_generation_missed_strict_positive_count": v102_exhaustive_target_mining.get("candidate_generation_missed_strict_positive_count"),
            "additional_strict_positive_needed_count": v102_exhaustive_target_mining.get("additional_strict_positive_needed_count"),
            "top_external_materialization_worklist_cases": v102_exhaustive_target_mining.get("top_external_materialization_worklist_cases"),
        },
        "stage3_exploration_oracle_signal_present": len(diagnostic_passes) > 0,
        "stage3_strict_semantic_oracle_pass": False,
        "stage3_runtime_action_allowed": False,
        "reason": "legacy/provider/proxy oracle signals exist, selected-case local geometry has diagnostic signal, TrackV true geometry exists, and full-control semantic rotation found exploratory signal, but no strict semantic oracle upper bound passes clean target-coverage/action-control requirements.",
    }
    write_json(out / "stage3_summary.json", summary)
    return summary


def stage4() -> dict[str, Any]:
    out = ROOT / "stage4_memory_action_surface_oracle"
    ensure_dir(out)
    stage3_summary = read_json(ROOT / "stage3_semantic_oracle_upper_bound/stage3_summary.json")
    formal_stage4_authorized = bool(stage3_summary.get("stage3_strict_semantic_oracle_pass"))
    v97_h2 = read_json(SRC["v97_h2_summary"])
    v98_s3 = read_json(SRC["v98_stage3_summary"])
    v98_7h = read_json(SRC["v98_stage7h_summary"])
    v98_7e = read_json(SRC["v98_stage7e_summary"])
    v100_q = read_json(SRC["v100_trackq_summary"])
    true_l3_upper_bound = read_json(ACTION_SURFACE_TRUE_L3_UPPER_BOUND_SUMMARY)
    state_machine_readiness = read_json(STATE_MACHINE_HOOK_READINESS_SUMMARY)
    scaffold_trace_closure = read_json(STATE_MACHINE_SCAFFOLD_TRACE_CLOSURE_SUMMARY)
    action_probe_closure = read_json(STATE_MACHINE_ACTION_PROBE_CLOSURE_SUMMARY)
    action_probe_donly_closure = read_json(STATE_MACHINE_ACTION_PROBE_DONLY_CLOSURE_SUMMARY)
    action_probe_d075_closure = read_json(STATE_MACHINE_ACTION_PROBE_D075_CLOSURE_SUMMARY)
    action_probe_d025_closure = read_json(STATE_MACHINE_ACTION_PROBE_D025_CLOSURE_SUMMARY)
    action_probe_transmit_supported_closure = read_json(STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_CLOSURE_SUMMARY)
    action_probe_transmit_supported_soft_closure = read_json(STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_SOFT_CLOSURE_SUMMARY)
    action_probe_hold_prev_closure = read_json(STATE_MACHINE_ACTION_PROBE_HOLD_PREV_CLOSURE_SUMMARY)
    action_probe_hold_prev_soft_closure = read_json(STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT_CLOSURE_SUMMARY)
    action_probe_hold_prev_soft2_closure = read_json(STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT2_CLOSURE_SUMMARY)
    action_probe_delay_update_closure = read_json(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_CLOSURE_SUMMARY)
    action_probe_delay_update_soft_closure = read_json(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT_CLOSURE_SUMMARY)
    action_probe_delay_update_soft075_closure = read_json(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT075_CLOSURE_SUMMARY)
    action_probe_delay_update_soft0875_closure = read_json(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT0875_CLOSURE_SUMMARY)
    action_probe_context_only_soft_closure = read_json(STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT_CLOSURE_SUMMARY)
    action_probe_context_only_soft075_closure = read_json(STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT075_CLOSURE_SUMMARY)
    ttt_write_to_use_chain_closure = read_json(TTT_WRITE_TO_USE_CHAIN_CLOSURE_SUMMARY)

    best_h2 = v97_h2.get("best_passing_component") or {}
    rows = [
        {
            "surface_id": "C1_READ_CURRENT_SUPPORT_PROVIDER",
            "memory_body": "READ",
            "trace_fidelity_pass": True,
            "target_metric": "L2 local scale CV",
            "bad_metric_improvement": best_h2.get("bad_L2_improvement"),
            "good_harm": best_h2.get("good_worsen"),
            "control_margin": best_h2.get("candidate_min_margin_vs_required_controls"),
            "action_surface_pass": bool(v97_h2.get("gate_pass")),
            "runtime_action_allowed": False,
            "classification": "READ_PROVIDER_ONLY",
            "evidence_path": rel(SRC["v97_h2_summary"]),
            "blocker": "READ local/provider pass does not authorize full method or SWA/TTT handoff action.",
        },
        {
            "surface_id": "C2_SWA_TRANSMIT_DELAY_REJECT_HOLD",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(v98_s3.get("status")),
            "target_metric": "L3 handoff transfer penalty",
            "bad_metric_improvement": v98_s3.get("trace_target_improvement_proxy"),
            "good_harm": v98_s3.get("good_simulated_harm"),
            "control_margin": v98_s3.get("actual_vs_random_margin"),
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "SWA_TRACE_PROXY_GOOD_HARM_FAIL",
            "evidence_path": rel(SRC["v98_stage3_summary"]),
            "blocker": v98_s3.get("blocker", "No validated SWA action surface L3 pass."),
        },
        {
            "surface_id": "C3_TTT_WRITE_TO_USE_STATE_MACHINE",
            "memory_body": "TTT",
            "trace_fidelity_pass": bool(v98_7e.get("write_to_use_chain_available")),
            "target_metric": "later L3/L4 after write-to-use",
            "bad_metric_improvement": "",
            "good_harm": "",
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "TTT_CHAIN_DIAGNOSTIC_NO_STATE_MACHINE_ACTION",
            "evidence_path": rel(SRC["v98_stage7e_summary"]) + ";" + rel(SRC["v98_stage7h_summary"]),
            "blocker": f"Stage7e chain diagnostic={v98_7e.get('write_to_use_chain_available')}; Stage7h query-soft pilot gate_pass={v98_7h.get('gate_pass')}",
        },
        {
            "surface_id": "C4_CHUNK_ADMISSION_PROXY",
            "memory_body": "ADMISSION",
            "trace_fidelity_pass": False,
            "target_metric": "boundary update admission / L3 proxy",
            "bad_metric_improvement": "",
            "good_harm": "",
            "control_margin": v100_q.get("best_same_count_random_margin"),
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "PROXY_ONLY_TRUE_TERMS_MISSING",
            "evidence_path": rel(SRC["v100_trackq_summary"]),
            "blocker": f"proxy_only={v100_q.get('proxy_only')}; missing_true_terms={v100_q.get('missing_true_terms')}",
        },
        {
            "surface_id": "C5_EXISTING_ACTION_TRUE_L3_UPPER_BOUND_FEASIBILITY",
            "memory_body": "READ/SWA/TTT",
            "trace_fidelity_pass": False,
            "target_metric": "true L3/L4 action-surface effect search over existing measured artifacts",
            "bad_metric_improvement": true_l3_upper_bound.get("best_eligible_read_swa_ttt_bad_median_or_best_improvement", ""),
            "good_harm": "",
            "control_margin": "",
            "action_surface_pass": bool(true_l3_upper_bound.get("strict_action_surface_upper_bound_pass_count")),
            "runtime_action_allowed": False,
            "classification": "NO_EXISTING_TRUE_L3_L4_UPPER_BOUND",
            "evidence_path": rel(ACTION_SURFACE_TRUE_L3_UPPER_BOUND_SUMMARY) if ACTION_SURFACE_TRUE_L3_UPPER_BOUND_SUMMARY.exists() else "",
            "blocker": true_l3_upper_bound.get(
                "conclusion",
                "true L3/L4 upper-bound feasibility audit missing; run audit_v102tf_action_surface_true_l3_upper_bound_feasibility.py",
            ),
        },
        {
            "surface_id": "C6_NEW_STATE_MACHINE_HOOK_READINESS",
            "memory_body": "SWA/TTT",
            "trace_fidelity_pass": False,
            "target_metric": "new non-forbidden state-machine hook readiness with true L3/L4 closure",
            "bad_metric_improvement": "",
            "good_harm": "",
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "NEW_STATE_MACHINE_HOOK_MISSING_TRUE_L3_CLOSURE",
            "evidence_path": rel(STATE_MACHINE_HOOK_READINESS_SUMMARY) if STATE_MACHINE_HOOK_READINESS_SUMMARY.exists() else "",
            "blocker": state_machine_readiness.get(
                "conclusion",
                "state-machine hook readiness audit missing; run audit_v102tf_state_machine_hook_readiness.py",
            ),
        },
        {
            "surface_id": "C7_V102_STATE_MACHINE_SCAFFOLD_TRACE_CLOSURE",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(scaffold_trace_closure.get("scaffold_trace_materialization_pass")),
            "target_metric": "diagnostic-only state-machine trace field materialization",
            "bad_metric_improvement": "",
            "good_harm": "",
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "SCAFFOLD_TRACE_MATERIALIZED_NO_TRUE_L3_ACTION_EFFECT",
            "evidence_path": (
                rel(STATE_MACHINE_SCAFFOLD_TRACE_CLOSURE_SUMMARY)
                if STATE_MACHINE_SCAFFOLD_TRACE_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": scaffold_trace_closure.get(
                "conclusion",
                "state-machine scaffold trace closure audit missing; run audit_v102tf_state_machine_scaffold_trace_closure.py",
            ),
        },
        {
            "surface_id": "C8_V102_STATE_MACHINE_ACTION_PROBE_REJECT_UNRELIABLE",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for compact_kv reject-unreliable action probe",
            "bad_metric_improvement": action_probe_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_ACTION_PROBE_MATERIALIZED_BUT_HARM_AND_STRICT_TARGET_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_closure.get(
                "conclusion",
                "state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py",
            ),
        },
        {
            "surface_id": "C9_V102_STATE_MACHINE_ACTION_PROBE_REJECT_UNRELIABLE_DONLY",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_donly_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for D-only compact_kv reject-unreliable action probe",
            "bad_metric_improvement": action_probe_donly_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_donly_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_DONLY_ACTION_PROBE_REDUCED_HARM_BUT_STRICT_TARGET_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_DONLY_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_DONLY_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_donly_closure.get(
                "conclusion",
                "D-only state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C10_V102_STATE_MACHINE_ACTION_PROBE_REJECT_UNRELIABLE_D075",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_d075_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for D-only compact_kv reject-unreliable action probe, D>=0.75",
            "bad_metric_improvement": action_probe_d075_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_d075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_D075_ACTION_PROBE_LOW_HARM_BUT_NEAR_NOOP_AND_STRICT_TARGET_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_D075_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_D075_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_d075_closure.get(
                "conclusion",
                "D>=0.75 state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C11_V102_STATE_MACHINE_ACTION_PROBE_REJECT_UNRELIABLE_D025",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_d025_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for D-only compact_kv reject-unreliable action probe, D>=0.25",
            "bad_metric_improvement": action_probe_d025_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_d025_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_D025_ACTION_PROBE_STRONGER_DELETE_WORSE_STRICT_AND_CONTROL_HARM",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_D025_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_D025_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_d025_closure.get(
                "conclusion",
                "D>=0.25 state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C12_V102_STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_COMPACT",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_transmit_supported_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for compact_kv transmit-supported action probe",
            "bad_metric_improvement": action_probe_transmit_supported_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_transmit_supported_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_TRANSMIT_SUPPORTED_COMPACT_MOVED_BUT_STRICT_AND_CONTROL_HARM",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_transmit_supported_closure.get(
                "conclusion",
                "compact transmit-supported state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C13_V102_STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_SOFT",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_transmit_supported_soft_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft transmit-supported action probe",
            "bad_metric_improvement": action_probe_transmit_supported_soft_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_transmit_supported_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_TRANSMIT_SUPPORTED_SOFT_REDUCED_HARM_BUT_STRICT_TARGET_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_SOFT_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_SOFT_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_transmit_supported_soft_closure.get(
                "conclusion",
                "source-soft transmit-supported state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C14_V102_STATE_MACHINE_ACTION_PROBE_HOLD_PREV_REFERENCE_COMPACT",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_hold_prev_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for compact_kv hold-previous-reference action probe",
            "bad_metric_improvement": action_probe_hold_prev_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_hold_prev_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_HOLD_PREV_COMPACT_SEVERE_CONTROL_HARM",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_HOLD_PREV_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_HOLD_PREV_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_hold_prev_closure.get(
                "conclusion",
                "compact hold-previous-reference state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C15_V102_STATE_MACHINE_ACTION_PROBE_HOLD_PREV_REFERENCE_SOFT",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_hold_prev_soft_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft hold-previous-reference action probe, 1 frame",
            "bad_metric_improvement": action_probe_hold_prev_soft_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_hold_prev_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_HOLD_PREV_SOFT_REDUCED_HARM_BUT_STRICT_AND_CONTROL_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_hold_prev_soft_closure.get(
                "conclusion",
                "source-soft hold-previous-reference state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C16_V102_STATE_MACHINE_ACTION_PROBE_HOLD_PREV_REFERENCE_SOFT2",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_hold_prev_soft2_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft hold-previous-reference action probe, 2 frames",
            "bad_metric_improvement": action_probe_hold_prev_soft2_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_hold_prev_soft2_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_HOLD_PREV_SOFT2_STRICT_AND_CONTROL_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT2_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_HOLD_PREV_SOFT2_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_hold_prev_soft2_closure.get(
                "conclusion",
                "source-soft two-frame hold-previous-reference state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C17_V102_STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_COMPACT",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_delay_update_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for compact_kv delay-update action probe",
            "bad_metric_improvement": action_probe_delay_update_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_delay_update_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_DELAY_UPDATE_COMPACT_STRICT_IMPROVES_BUT_SEVERE_CONTROL_HARM",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_delay_update_closure.get(
                "conclusion",
                "compact delay-update state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C18_V102_STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT050",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_delay_update_soft_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft delay-update action probe, current min keep 0.50",
            "bad_metric_improvement": action_probe_delay_update_soft_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_delay_update_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_DELAY_UPDATE_SOFT050_STRICT_IMPROVES_BUT_CONTROL_AND_OVERALL_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_delay_update_soft_closure.get(
                "conclusion",
                "source-soft delay-update 0.50 state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C19_V102_STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT075",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_delay_update_soft075_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft delay-update action probe, current min keep 0.75",
            "bad_metric_improvement": action_probe_delay_update_soft075_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_delay_update_soft075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_DELAY_UPDATE_SOFT075_REDUCED_LOCAL_HARM_BUT_SCALE_AND_OVERALL_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT075_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT075_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_delay_update_soft075_closure.get(
                "conclusion",
                "source-soft delay-update 0.75 state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C20_V102_STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT0875",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_delay_update_soft0875_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft delay-update action probe, current min keep 0.875",
            "bad_metric_improvement": action_probe_delay_update_soft0875_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_delay_update_soft0875_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_DELAY_UPDATE_SOFT0875_NEAR_SAFE_LOCAL_BUT_SCALE_AND_OVERALL_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT0875_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE_SOFT0875_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_delay_update_soft0875_closure.get(
                "conclusion",
                "source-soft delay-update 0.875 state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C21_V102_STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_DEMOTION_SOFT050",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_context_only_soft_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft context-only demotion action probe, min keep 0.50",
            "bad_metric_improvement": action_probe_context_only_soft_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_context_only_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_CONTEXT_ONLY_SOFT050_STRICT_TARGET_AND_SAFE_LOCAL_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_context_only_soft_closure.get(
                "conclusion",
                "source-soft context-only demotion 0.50 state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C22_V102_STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_DEMOTION_SOFT075",
            "memory_body": "SWA",
            "trace_fidelity_pass": bool(action_probe_context_only_soft075_closure.get("action_probe_materialization_pass")),
            "target_metric": "diagnostic paired trajectory metrics for source_soft context-only demotion action probe, min keep 0.75",
            "bad_metric_improvement": action_probe_context_only_soft075_closure.get(
                "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median", ""
            ),
            "good_harm": action_probe_context_only_soft075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median", ""
            ),
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_CONTEXT_ONLY_SOFT075_NEAR_SAFE_BUT_STRICT_TARGET_FAIL",
            "evidence_path": (
                rel(STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT075_CLOSURE_SUMMARY)
                if STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_SOFT075_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": action_probe_context_only_soft075_closure.get(
                "conclusion",
                "source-soft context-only demotion 0.75 state-machine action-probe closure audit missing; run audit_v102tf_state_machine_action_probe_closure.py with --trace-root/--metric-summary overrides",
            ),
        },
        {
            "surface_id": "C23_V102_TTT_WRITE_TO_USE_CHAIN_CLOSURE",
            "memory_body": "TTT",
            "trace_fidelity_pass": bool(ttt_write_to_use_chain_closure.get("stage7e_write_to_use_chain_available")),
            "target_metric": "diagnostic write/cache/current/later-use chain coverage for TTT write-to-use",
            "bad_metric_improvement": "",
            "good_harm": "",
            "control_margin": "",
            "action_surface_pass": False,
            "runtime_action_allowed": False,
            "classification": "DIAGNOSTIC_TTT_CHAIN_COVERAGE_AND_PER_ANCHOR_LINKS_INSUFFICIENT",
            "evidence_path": (
                rel(TTT_WRITE_TO_USE_CHAIN_CLOSURE_SUMMARY)
                if TTT_WRITE_TO_USE_CHAIN_CLOSURE_SUMMARY.exists()
                else ""
            ),
            "blocker": ttt_write_to_use_chain_closure.get(
                "conclusion",
                "v102 TTT write-to-use chain closure audit missing; run audit_v102tf_ttt_write_to_use_chain_closure.py",
            ),
        },
    ]
    write_csv(out / "action_surface_rows.csv", rows)
    write_text(
        out / "action_surface_gap_report.md",
        "# Stage 4 Action Surface Gap Report\n\n"
        "Formal Stage 4 action-surface promotion is not authorized because Stage 3 strict semantic oracle did not pass. "
        "The rows below are a diagnostic inventory of inherited providers/carriers/action attempts, not a runtime permission.\n\n"
        "- READ has a local/provider pass, but this is not a full memory-control route.\n"
        "- SWA has diagnostic/proxy movement but fails good-harm / true L3 action requirements.\n"
        "- TTT write-to-use identity chain exists, but no refresh/expire/context-only state-machine action has validated L3/L4 effect.\n"
        "- Chunk admission remains proxy-only until true support/observability/runtime outcome terms are materialized.\n"
        "- Existing measured action artifacts were audited for a strict true L3/L4 upper bound; no READ/SWA/TTT route passed.\n"
        "- New v102 transmit/delay/reject/hold/refresh/expire/context-only state-machine hooks are not ready with true L3/L4 closure.\n"
        "- v102 state-machine scaffold trace closure may repair instrumentation visibility, but it remains diagnostic-only until a true L3/L4 action effect is measured.\n"
        "- v102 diagnostic compact_kv reject-unreliable action probe materialized a true SWA KV-cache intervention and paired metrics, but strict-positive L3/scale worsened and safe controls show harm; it is not a Stage4 pass.\n"
        "- v102 D-only reject-unreliable action probe reduced deletion/harm, but strict-positive L3/scale still did not pass; it is also not a Stage4 pass.\n"
        "- v102 D>=0.75 D-only threshold probe reduced harm further but became near-no-op and still worsened strict-positive L3/scale.\n"
        "- v102 D>=0.25 D-only threshold probe increased overall movement but worsened strict-positive/control harm; no tested threshold in this reject body passed.\n"
        "- v102 compact transmit-supported action probe moved the KV carrier, but strict-positive head/overlap/scale and safe controls failed.\n"
        "- v102 source-soft transmit-supported action probe reduced hard-deletion harm, but strict-positive head/overlap/scale remained negative and safe-control local still worsened.\n"
        "- v102 compact hold-previous-reference action probe preserved only the last reference frame, but caused severe safe-control harm and worsened strict-positive metrics.\n"
        "- v102 source-soft hold-previous-reference reduced the hard-hold harm, including a two-frame variant, but strict-positive head/overlap/scale stayed negative and safe-control local still worsened.\n"
        "- v102 delay-update action probes can improve the single strict-positive handoff case, but hard delay causes severe overall/control harm and soft delay still fails scale/control/overall requirements.\n"
        "- v102 context-only demotion action probes materialize source-soft demotion of semantic but scale-unobservable history tokens; 0.50 and 0.75 both fail strict-positive head/overlap/scale, and 0.50 also fails safe-control local.\n"
        "- v102 TTT write-to-use chain closure shows Stage7e id-to-SWA-top-k exists only on a subset of v102 target cases and the strict-positive target lacks a full per-anchor write/cache/current/later-use chain.\n",
    )
    write_csv(
        out / "carrier_trace_moved_no_L3_effect.csv",
        [
            {
                "surface_id": "C2_SWA_TRANSMIT_DELAY_REJECT_HOLD",
                "trace_or_proxy_moved": True,
                "target_l3_effect_validated": False,
                "classification": "SWA_TRACE_PROXY_NO_TRUE_L3_EFFECT",
                "evidence_path": rel(SRC["v98_stage3_summary"]),
            },
            {
                "surface_id": "C3_TTT_WRITE_TO_USE_STATE_MACHINE",
                "trace_or_proxy_moved": bool(v98_7e.get("write_to_use_chain_available")),
                "target_l3_effect_validated": False,
                "classification": "TTT_CHAIN_NO_STATE_MACHINE_L3_EFFECT",
                "evidence_path": rel(SRC["v98_stage7e_summary"]),
            },
        ],
    )
    write_csv(
        out / "action_body_failure_taxonomy.csv",
        [
            {"route": "READ", "failure_taxonomy": "PROVIDER_ONLY_NOT_FULL_METHOD", "next_required": "feed current_support into SWA/TTT and remeasure L3"},
            {"route": "SWA", "failure_taxonomy": "GOOD_HARM_AND_TRUE_L3_EFFECT_FAIL", "next_required": "new non-forbidden state-machine hook with measured L3"},
            {"route": "TTT", "failure_taxonomy": "WRITE_TO_USE_CHAIN_ONLY_NO_ACTION", "next_required": "refresh/expire/context-only state machine with later-use L3/L4"},
            {"route": "ADMISSION", "failure_taxonomy": "TRUE_TERMS_MISSING", "next_required": "materialize true current support / parallax-depth / runtime update outcome"},
            {"route": "EXISTING_ACTION_ARTIFACTS", "failure_taxonomy": "NO_STRICT_TRUE_L3_L4_UPPER_BOUND_FOUND", "next_required": "new labelled clean handoff targets or new state-machine hook with measured true L3/L4 and controls"},
            {"route": "NEW_STATE_MACHINE_HOOK", "failure_taxonomy": "HOOK_NOT_IMPLEMENTED_WITH_TRUE_L3_CLOSURE", "next_required": "implement diagnostic-first v102 SWA/TTT state-machine hook and measure true L3/L4 with controls"},
            {"route": "V102_STATE_MACHINE_SCAFFOLD_TRACE", "failure_taxonomy": "INSTRUMENTATION_ONLY_NO_ACTION_EFFECT", "next_required": "use the materialized trace to design a default-off action hook, then measure true L3/L4 with strict controls"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_REJECT_UNRELIABLE", "failure_taxonomy": "DIAGNOSTIC_ACTION_MOVED_KV_BUT_HARM_AND_STRICT_TARGET_FAIL", "next_required": "repair action selectivity, especially G-heavy over-rejection, and remeasure paired strict target/control metrics before any promotion"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_REJECT_UNRELIABLE_DONLY", "failure_taxonomy": "DIAGNOSTIC_ACTION_REDUCED_HARM_BUT_STRICT_TARGET_STILL_FAILS", "next_required": "test stricter/localized D threshold or query-conditioned reject; do not promote until strict-positive L3 and good-control harm gates pass"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_COMPACT", "failure_taxonomy": "DIAGNOSTIC_ACTION_MOVED_KV_BUT_STRICT_AND_CONTROL_HARM", "next_required": "do not hard-drop unsupported history; if revisited, require softer source weighting and paired strict/control pass before promotion"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_TRANSMIT_SUPPORTED_SOFT", "failure_taxonomy": "DIAGNOSTIC_ACTION_SOFTENED_HARM_BUT_STRICT_TARGET_STILL_FAILS", "next_required": "treat SWA current action surface as unproven carrier unless a new hook improves strict clean handoff without good-control harm"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_HOLD_PREV_REFERENCE_COMPACT", "failure_taxonomy": "DIAGNOSTIC_HOLD_PREV_REFERENCE_SEVERE_CONTROL_HARM", "next_required": "do not hard-hold a single reference frame; if revisited, require soft/wider reference and strict/control pass"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_HOLD_PREV_REFERENCE_SOFT", "failure_taxonomy": "DIAGNOSTIC_HOLD_PREV_REFERENCE_SOFT_STRICT_AND_CONTROL_FAIL", "next_required": "hold action is not a valid SWA carrier unless a different hook fixes strict-positive head/overlap/scale without local good-control harm"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_DELAY_UPDATE", "failure_taxonomy": "STRICT_TARGET_CAN_MOVE_BUT_OVERALL_AND_GOOD_CONTROLS_FAIL", "next_required": "do not promote delay update unless strict gain reaches the required margin and good-control local/scale harms are both within bound"},
            {"route": "V102_STATE_MACHINE_ACTION_PROBE_CONTEXT_ONLY_DEMOTION", "failure_taxonomy": "CONTEXT_ONLY_DEMOTION_TRACE_APPLIES_BUT_STRICT_TARGET_FAILS", "next_required": "do not continue context-only strength sweeps; if SWA is revisited, require a different carrier or query/head-conditioned path with strict-positive and good-control pass"},
            {"route": "V102_TTT_WRITE_TO_USE_CHAIN_CLOSURE", "failure_taxonomy": "TTT_CHAIN_COVERAGE_AND_PER_ANCHOR_LINKS_MISSING", "next_required": "do not run TTT write/expire/refresh actions until per-anchor write/cache/current residual links and v102 target coverage are materialized"},
        ],
    )
    summary = {
        "schema": "acl2_v102_stage4_memory_action_surface_oracle_v1",
        "formal_stage4_authorized_by_stage3_gate": formal_stage4_authorized,
        "stage4_diagnostic_inventory_only": not formal_stage4_authorized,
        "surface_count": len(rows),
        "read_provider_pass": bool(v97_h2.get("gate_pass")),
        "swa_action_surface_pass": False,
        "ttt_action_surface_pass": False,
        "existing_action_true_l3_upper_bound_pass": bool(true_l3_upper_bound.get("strict_action_surface_upper_bound_pass_count")),
        "existing_action_true_l3_upper_bound_summary": {
            "audited_route_count": true_l3_upper_bound.get("audited_route_count"),
            "true_l3_l4_metric_route_count": true_l3_upper_bound.get("true_l3_l4_metric_route_count"),
            "eligible_read_swa_ttt_true_l3_l4_route_count": true_l3_upper_bound.get("eligible_read_swa_ttt_true_l3_l4_route_count"),
            "strict_action_surface_upper_bound_pass_count": true_l3_upper_bound.get("strict_action_surface_upper_bound_pass_count"),
            "best_eligible_read_swa_ttt_route_id": true_l3_upper_bound.get("best_eligible_read_swa_ttt_route_id"),
            "best_eligible_read_swa_ttt_bad_median_or_best_improvement": true_l3_upper_bound.get(
                "best_eligible_read_swa_ttt_bad_median_or_best_improvement"
            ),
        },
        "state_machine_hook_readiness_summary": {
            "old_or_forbidden_hook_patterns_present": state_machine_readiness.get("old_or_forbidden_hook_patterns_present"),
            "new_v102_state_machine_hook_pattern_count": state_machine_readiness.get("new_v102_state_machine_hook_pattern_count"),
            "state_machine_action_ready_count": state_machine_readiness.get("state_machine_action_ready_count"),
            "state_machine_action_blocked_count": state_machine_readiness.get("state_machine_action_blocked_count"),
            "measured_true_l3_l4_effect_available": state_machine_readiness.get("measured_true_l3_l4_effect_available"),
        },
        "state_machine_scaffold_trace_closure_summary": {
            "scaffold_trace_materialization_pass": scaffold_trace_closure.get("scaffold_trace_materialization_pass"),
            "target_count": scaffold_trace_closure.get("target_count"),
            "completed_job_count": scaffold_trace_closure.get("completed_job_count"),
            "failed_job_count": scaffold_trace_closure.get("failed_job_count"),
            "raw_case_with_v102_trace_count": scaffold_trace_closure.get("raw_case_with_v102_trace_count"),
            "hmc_case_with_v102_trace_count": scaffold_trace_closure.get("hmc_case_with_v102_trace_count"),
            "raw_trace_applied_count": scaffold_trace_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": scaffold_trace_closure.get("raw_runtime_action_allowed_count"),
            "true_l3_measurement_ready": scaffold_trace_closure.get("true_l3_measurement_ready"),
        },
        "state_machine_action_probe_closure_summary": {
            "action_probe_materialization_pass": action_probe_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_closure.get("target_count"),
            "completed_job_count": action_probe_closure.get("completed_job_count"),
            "failed_job_count": action_probe_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_closure.get("raw_runtime_action_allowed_count"),
            "paired_metric_case_count": action_probe_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_scale_cv_relative_improvement": action_probe_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
            ),
        },
        "state_machine_action_probe_donly_closure_summary": {
            "action_probe_materialization_pass": action_probe_donly_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_donly_closure.get("target_count"),
            "completed_job_count": action_probe_donly_closure.get("completed_job_count"),
            "failed_job_count": action_probe_donly_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_donly_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_donly_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_donly_closure.get("raw_rejected_history_frac_mean"),
            "paired_metric_case_count": action_probe_donly_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_donly_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_donly_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_donly_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_d075_closure_summary": {
            "action_probe_materialization_pass": action_probe_d075_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_d075_closure.get("target_count"),
            "completed_job_count": action_probe_d075_closure.get("completed_job_count"),
            "failed_job_count": action_probe_d075_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_d075_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_d075_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_d075_closure.get("raw_rejected_history_frac_mean"),
            "paired_metric_case_count": action_probe_d075_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_d075_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_d075_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_d075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_d025_closure_summary": {
            "action_probe_materialization_pass": action_probe_d025_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_d025_closure.get("target_count"),
            "completed_job_count": action_probe_d025_closure.get("completed_job_count"),
            "failed_job_count": action_probe_d025_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_d025_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_d025_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_d025_closure.get("raw_rejected_history_frac_mean"),
            "paired_metric_case_count": action_probe_d025_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_d025_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_d025_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_d025_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_transmit_supported_closure_summary": {
            "action_probe_materialization_pass": action_probe_transmit_supported_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_transmit_supported_closure.get("target_count"),
            "completed_job_count": action_probe_transmit_supported_closure.get("completed_job_count"),
            "failed_job_count": action_probe_transmit_supported_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_transmit_supported_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_transmit_supported_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_transmit_supported_closure.get("raw_rejected_history_frac_mean"),
            "raw_supported_history_tokens_mean": action_probe_transmit_supported_closure.get("raw_supported_history_tokens_mean"),
            "paired_metric_case_count": action_probe_transmit_supported_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_transmit_supported_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_transmit_supported_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_transmit_supported_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_transmit_supported_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_transmit_supported_soft_closure_summary": {
            "action_probe_materialization_pass": action_probe_transmit_supported_soft_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_transmit_supported_soft_closure.get("target_count"),
            "completed_job_count": action_probe_transmit_supported_soft_closure.get("completed_job_count"),
            "failed_job_count": action_probe_transmit_supported_soft_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_transmit_supported_soft_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_transmit_supported_soft_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_transmit_supported_soft_closure.get("raw_rejected_history_frac_mean"),
            "raw_supported_history_tokens_mean": action_probe_transmit_supported_soft_closure.get("raw_supported_history_tokens_mean"),
            "raw_soft_unsupported_min_keep_mean": action_probe_transmit_supported_soft_closure.get("raw_soft_unsupported_min_keep_mean"),
            "paired_metric_case_count": action_probe_transmit_supported_soft_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_transmit_supported_soft_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_transmit_supported_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_transmit_supported_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_transmit_supported_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_hold_prev_reference_closure_summary": {
            "action_probe_materialization_pass": action_probe_hold_prev_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_hold_prev_closure.get("target_count"),
            "completed_job_count": action_probe_hold_prev_closure.get("completed_job_count"),
            "failed_job_count": action_probe_hold_prev_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_hold_prev_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_hold_prev_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_hold_prev_closure.get("raw_rejected_history_frac_mean"),
            "raw_hold_prev_frames_mean": action_probe_hold_prev_closure.get("raw_hold_prev_frames_mean"),
            "raw_hold_soft_min_keep_mean": action_probe_hold_prev_closure.get("raw_hold_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_hold_prev_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_hold_prev_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_hold_prev_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_hold_prev_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_hold_prev_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_hold_prev_reference_soft_closure_summary": {
            "action_probe_materialization_pass": action_probe_hold_prev_soft_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_hold_prev_soft_closure.get("target_count"),
            "completed_job_count": action_probe_hold_prev_soft_closure.get("completed_job_count"),
            "failed_job_count": action_probe_hold_prev_soft_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_hold_prev_soft_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_hold_prev_soft_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_hold_prev_soft_closure.get("raw_rejected_history_frac_mean"),
            "raw_hold_prev_frames_mean": action_probe_hold_prev_soft_closure.get("raw_hold_prev_frames_mean"),
            "raw_hold_soft_min_keep_mean": action_probe_hold_prev_soft_closure.get("raw_hold_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_hold_prev_soft_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_hold_prev_soft_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_hold_prev_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_hold_prev_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_hold_prev_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_hold_prev_reference_soft2_closure_summary": {
            "action_probe_materialization_pass": action_probe_hold_prev_soft2_closure.get("action_probe_materialization_pass"),
            "target_count": action_probe_hold_prev_soft2_closure.get("target_count"),
            "completed_job_count": action_probe_hold_prev_soft2_closure.get("completed_job_count"),
            "failed_job_count": action_probe_hold_prev_soft2_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_hold_prev_soft2_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_hold_prev_soft2_closure.get("raw_runtime_action_allowed_count"),
            "raw_rejected_history_frac_mean": action_probe_hold_prev_soft2_closure.get("raw_rejected_history_frac_mean"),
            "raw_hold_prev_frames_mean": action_probe_hold_prev_soft2_closure.get("raw_hold_prev_frames_mean"),
            "raw_hold_soft_min_keep_mean": action_probe_hold_prev_soft2_closure.get("raw_hold_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_hold_prev_soft2_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_hold_prev_soft2_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_hold_prev_soft2_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_hold_prev_soft2_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_hold_prev_soft2_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
        },
        "state_machine_action_probe_delay_update_closure_summary": {
            "action_probe_materialization_pass": action_probe_delay_update_closure.get("action_probe_materialization_pass"),
            "completed_job_count": action_probe_delay_update_closure.get("completed_job_count"),
            "failed_job_count": action_probe_delay_update_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_delay_update_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_delay_update_closure.get("raw_runtime_action_allowed_count"),
            "raw_delay_current_frac_mean": action_probe_delay_update_closure.get("raw_delay_current_frac_mean"),
            "raw_delay_current_soft_min_keep_mean": action_probe_delay_update_closure.get("raw_delay_current_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_delay_update_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_delay_update_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_delay_update_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_delay_update_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_delay_update_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
            "safe_good_control_scale_relative_improvement": action_probe_delay_update_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
            ),
        },
        "state_machine_action_probe_delay_update_soft_closure_summary": {
            "action_probe_materialization_pass": action_probe_delay_update_soft_closure.get("action_probe_materialization_pass"),
            "completed_job_count": action_probe_delay_update_soft_closure.get("completed_job_count"),
            "failed_job_count": action_probe_delay_update_soft_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_delay_update_soft_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_delay_update_soft_closure.get("raw_runtime_action_allowed_count"),
            "raw_delay_current_frac_mean": action_probe_delay_update_soft_closure.get("raw_delay_current_frac_mean"),
            "raw_delay_current_soft_min_keep_mean": action_probe_delay_update_soft_closure.get("raw_delay_current_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_delay_update_soft_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_delay_update_soft_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_delay_update_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_delay_update_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_delay_update_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
            "safe_good_control_scale_relative_improvement": action_probe_delay_update_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
            ),
        },
        "state_machine_action_probe_delay_update_soft075_closure_summary": {
            "action_probe_materialization_pass": action_probe_delay_update_soft075_closure.get("action_probe_materialization_pass"),
            "completed_job_count": action_probe_delay_update_soft075_closure.get("completed_job_count"),
            "failed_job_count": action_probe_delay_update_soft075_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_delay_update_soft075_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_delay_update_soft075_closure.get("raw_runtime_action_allowed_count"),
            "raw_delay_current_frac_mean": action_probe_delay_update_soft075_closure.get("raw_delay_current_frac_mean"),
            "raw_delay_current_soft_min_keep_mean": action_probe_delay_update_soft075_closure.get("raw_delay_current_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_delay_update_soft075_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_delay_update_soft075_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_delay_update_soft075_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_delay_update_soft075_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_delay_update_soft075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
            "safe_good_control_scale_relative_improvement": action_probe_delay_update_soft075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
            ),
        },
        "state_machine_action_probe_delay_update_soft0875_closure_summary": {
            "action_probe_materialization_pass": action_probe_delay_update_soft0875_closure.get("action_probe_materialization_pass"),
            "completed_job_count": action_probe_delay_update_soft0875_closure.get("completed_job_count"),
            "failed_job_count": action_probe_delay_update_soft0875_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_delay_update_soft0875_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_delay_update_soft0875_closure.get("raw_runtime_action_allowed_count"),
            "raw_delay_current_frac_mean": action_probe_delay_update_soft0875_closure.get("raw_delay_current_frac_mean"),
            "raw_delay_current_soft_min_keep_mean": action_probe_delay_update_soft0875_closure.get("raw_delay_current_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_delay_update_soft0875_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_delay_update_soft0875_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_delay_update_soft0875_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_delay_update_soft0875_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_delay_update_soft0875_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
            "safe_good_control_scale_relative_improvement": action_probe_delay_update_soft0875_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
            ),
        },
        "state_machine_action_probe_context_only_soft_closure_summary": {
            "action_probe_materialization_pass": action_probe_context_only_soft_closure.get("action_probe_materialization_pass"),
            "completed_job_count": action_probe_context_only_soft_closure.get("completed_job_count"),
            "failed_job_count": action_probe_context_only_soft_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_context_only_soft_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_context_only_soft_closure.get("raw_runtime_action_allowed_count"),
            "raw_context_semantic_tokens_mean": action_probe_context_only_soft_closure.get("raw_context_semantic_tokens_mean"),
            "raw_context_scale_observable_tokens_mean": action_probe_context_only_soft_closure.get("raw_context_scale_observable_tokens_mean"),
            "raw_context_demoted_tokens_mean": action_probe_context_only_soft_closure.get("raw_context_demoted_tokens_mean"),
            "raw_context_soft_min_keep_mean": action_probe_context_only_soft_closure.get("raw_context_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_context_only_soft_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_context_only_soft_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_context_only_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_context_only_soft_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_context_only_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
            "safe_good_control_scale_relative_improvement": action_probe_context_only_soft_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
            ),
        },
        "state_machine_action_probe_context_only_soft075_closure_summary": {
            "action_probe_materialization_pass": action_probe_context_only_soft075_closure.get("action_probe_materialization_pass"),
            "completed_job_count": action_probe_context_only_soft075_closure.get("completed_job_count"),
            "failed_job_count": action_probe_context_only_soft075_closure.get("failed_job_count"),
            "raw_trace_applied_count": action_probe_context_only_soft075_closure.get("raw_trace_applied_count"),
            "raw_runtime_action_allowed_count": action_probe_context_only_soft075_closure.get("raw_runtime_action_allowed_count"),
            "raw_context_semantic_tokens_mean": action_probe_context_only_soft075_closure.get("raw_context_semantic_tokens_mean"),
            "raw_context_scale_observable_tokens_mean": action_probe_context_only_soft075_closure.get("raw_context_scale_observable_tokens_mean"),
            "raw_context_demoted_tokens_mean": action_probe_context_only_soft075_closure.get("raw_context_demoted_tokens_mean"),
            "raw_context_soft_min_keep_mean": action_probe_context_only_soft075_closure.get("raw_context_soft_min_keep_mean"),
            "paired_metric_case_count": action_probe_context_only_soft075_closure.get("paired_metric_case_count"),
            "true_l3_measurement_ready": action_probe_context_only_soft075_closure.get("true_l3_measurement_ready"),
            "strict_positive_head10_relative_improvement": action_probe_context_only_soft075_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
            ),
            "strict_positive_overlap_relative_improvement": action_probe_context_only_soft075_closure.get(
                "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
            ),
            "safe_good_control_local_relative_improvement": action_probe_context_only_soft075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
            ),
            "safe_good_control_scale_relative_improvement": action_probe_context_only_soft075_closure.get(
                "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
            ),
        },
        "ttt_write_to_use_chain_closure_summary": {
            "stage7e_write_to_use_chain_available": ttt_write_to_use_chain_closure.get("stage7e_write_to_use_chain_available"),
            "stage7e_gate_pass": ttt_write_to_use_chain_closure.get("stage7e_gate_pass"),
            "stage7h_query_soft_gate_pass": ttt_write_to_use_chain_closure.get("stage7h_query_soft_gate_pass"),
            "target_count": ttt_write_to_use_chain_closure.get("target_count"),
            "target_with_stage7e_swa_topk_use_count": ttt_write_to_use_chain_closure.get(
                "target_with_stage7e_swa_topk_use_count"
            ),
            "target_stage7e_swa_topk_use_coverage": ttt_write_to_use_chain_closure.get(
                "target_stage7e_swa_topk_use_coverage"
            ),
            "strict_clean_handoff_positive_with_stage7e_count": ttt_write_to_use_chain_closure.get(
                "strict_clean_handoff_positive_with_stage7e_count"
            ),
            "target_full_write_cache_current_swa_l3_chain_count": ttt_write_to_use_chain_closure.get(
                "target_full_write_cache_current_swa_l3_chain_count"
            ),
            "f5_per_anchor_write_chain_materialized": ttt_write_to_use_chain_closure.get(
                "f5_per_anchor_write_chain_materialized"
            ),
            "f5_r_write_cache_nonempty": ttt_write_to_use_chain_closure.get("f5_r_write_cache_nonempty"),
            "f5_r_cache_current_nonempty": ttt_write_to_use_chain_closure.get("f5_r_cache_current_nonempty"),
            "c3_chain_coverage_pass": ttt_write_to_use_chain_closure.get("c3_chain_coverage_pass"),
            "c3_full_chain_materialization_pass": ttt_write_to_use_chain_closure.get(
                "c3_full_chain_materialization_pass"
            ),
            "c3_ttt_action_surface_pass": ttt_write_to_use_chain_closure.get("c3_ttt_action_surface_pass"),
        },
        "strict_memory_action_surface_pass": False,
        "runtime_action_allowed": False,
        "stage5_allowed": False,
        "stage6_runtime_pilot_allowed": False,
        "reason": "Stage3 strict semantic oracle did not pass, so Stage4 is diagnostic inventory only; READ provider/local L2 exists, but SWA/TTT/admission do not have true L3/L4 action-surface pass. Existing measured action artifacts do not contain a strict true L3/L4 upper bound. The v102 state-machine scaffold trace repairs instrumentation visibility. The v102 compact_kv reject-unreliable action probes materialize real diagnostic KV-cache interventions with paired metrics; no tested reject threshold passes strict-positive L3/scale and safe-control requirements. The transmit-supported probes also materialize real KV/source-soft interventions, but compact transmit creates severe control harm and source-soft transmit still fails strict-positive head/overlap/scale plus safe-control local. Hold-previous-reference hard/soft probes likewise materialize, but hard hold causes severe control harm, while 1-frame and 2-frame soft hold still fail strict-positive head/overlap/scale and good-control local. Delay-update probes can move the strict-positive target, but hard/soft delay fail overall and/or safe-control scale requirements and do not satisfy the >=5% action-surface improvement criterion. Context-only demotion probes materialize source-soft demotion of scale-unobservable semantic history tokens, but both 0.50 and 0.75 variants fail strict-positive head/overlap/scale. TTT write-to-use chain closure shows insufficient v102 target coverage and missing per-anchor write/cache/current residual links, so no TTT write/expire/refresh action is allowed.",
    }
    write_json(out / "stage4_summary.json", summary)
    return summary


def stage5_blocked(stage4_summary: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "stage5_training_free_cue_distillation"
    ensure_dir(out)
    v97_k = read_json(SRC["v97_trackk_summary"])
    v100_q = read_json(SRC["v100_trackq_summary"])
    rows = [
        {
            "cue_family": "D1/D4 READ semantic/current support",
            "source": rel(SRC["v97_trackk_summary"]),
            "oracle_selection_F1": "",
            "boundary_admission_BA": "",
            "good_FPR": (v97_k.get("read_metrics") or {}).get("good_FPR"),
            "sequence_coverage": (v97_k.get("read_metrics") or {}).get("positive_sequence_coverage"),
            "cue_distillation_pass": False,
            "blocker": "READ eligibility cue had high good FPR and sequence fragility; READ remains provider/local mechanism.",
        },
        {
            "cue_family": "D5 chunk-level admission",
            "source": rel(SRC["v100_trackq_summary"]),
            "oracle_selection_F1": "",
            "boundary_admission_BA": v100_q.get("best_composite_balanced_accuracy"),
            "good_FPR": v100_q.get("best_composite_good_FPR"),
            "sequence_coverage": v100_q.get("sequence_coverage"),
            "cue_distillation_pass": False,
            "blocker": f"proxy-only; missing true terms {v100_q.get('missing_true_terms')}; no Stage4 action benefit retention measured.",
        },
    ]
    write_csv(out / "cue_distillation_rows.csv", rows)
    summary = {
        "schema": "acl2_v102_stage5_training_free_cue_distillation_v1",
        "stage5_run": False,
        "stage5_allowed": bool(stage4_summary.get("stage5_allowed")),
        "cue_distillation_pass": False,
        "reason": "Stage4 strict memory action surface did not pass; only blocked/proxy cue audit rows were written.",
    }
    write_json(out / "stage5_summary.json", summary)
    return summary


def final_decision(stage0_s: dict[str, Any], stage1_s: dict[str, Any], stage2_s: dict[str, Any], stage25_s: dict[str, Any], stage3_s: dict[str, Any], stage4_s: dict[str, Any], stage5_s: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "final_decision"
    ensure_dir(out)
    local_point_count = int(stage2_s.get("local_point_error_map_case_count") or 0)
    trajectory_count = int(stage2_s.get("trajectory_error_map_case_count") or 0)
    total_base_cases = int(stage2_s.get("total_base_cases") or 0)
    visual_gate_pass = bool(stage1_s.get("visual_overlay_strict_pass")) and bool(stage2_s.get("stage2_strict_visual_gate_pass"))
    visual_status = (
        "Strict visual evidence passed after RGB/semantic/risk overlays, trajectory maps, and local point residual maps covered all selected base cases."
        if visual_gate_pass
        else ""
    )
    if visual_gate_pass:
        visual_blocker = ""
    elif local_point_count > 0:
        visual_blocker = (
            "RGB/semantic/risk overlays, trajectory error maps, and local point residual maps were materialized for selected base cases, "
            f"but strict visual evidence is still blocked because local point residual coverage is {local_point_count}/{total_base_cases} "
            f"and strict_visual_panel_count={stage2_s.get('strict_visual_panel_count')}."
        )
    elif trajectory_count > 0:
        visual_blocker = (
            "RGB/semantic/risk overlays and trajectory error maps were materialized for selected base cases, but strict visual evidence is still blocked "
            "because local point-level residual maps / strict local point attribution are not materialized."
        )
    elif int(stage2_s.get("rgb_semantic_overlay_case_count") or 0) > 0:
        visual_blocker = (
            "RGB/semantic/risk overlays were materialized for selected base cases, but strict visual evidence is still blocked "
            "because local point/trajectory error maps are not materialized."
        )
    else:
        visual_blocker = "strict RGB/semantic overlay visual evidence missing"
    decision = {
        "schema": "acl2_v102_final_decision_v1",
        "goal_achieved": False,
        "full_method_success": False,
        "runtime_action_allowed": False,
        "stage6_runtime_pilot_run": False,
        "stage7_full_validation_run": False,
        "final_taxonomy": "READ_PROVIDER_ONLY__SWA_TTT_ACTION_SURFACE_FAIL__TRUE_GEOMETRY_CONTROL_BLOCKED",
        "stage0_pass": stage0_s.get("stage0_pass"),
        "stage1_h1_gate_pass": stage1_s.get("h1_gate_pass"),
        "stage1_visual_overlay_strict_pass": stage1_s.get("visual_overlay_strict_pass"),
        "stage2_diagnostic_gate_pass": stage2_s.get("stage2_diagnostic_gate_pass"),
        "stage2_strict_gate_pass": stage2_s.get("stage2_strict_gate_pass"),
        "stage2_rgb_semantic_overlay_case_count": stage2_s.get("rgb_semantic_overlay_case_count"),
        "stage2_rgb_semantic_overlay_coverage": stage2_s.get("rgb_semantic_overlay_coverage"),
        "stage2_trajectory_error_map_case_count": stage2_s.get("trajectory_error_map_case_count"),
        "stage2_trajectory_error_map_coverage": stage2_s.get("trajectory_error_map_coverage"),
        "stage2_local_point_error_map_case_count": stage2_s.get("local_point_error_map_case_count"),
        "stage2_local_point_error_map_coverage": stage2_s.get("local_point_error_map_coverage"),
        "stage2_5_gate_pass": stage25_s.get("stage2_5_gate_pass"),
        "stage3_exploration_oracle_signal_present": stage3_s.get("stage3_exploration_oracle_signal_present"),
        "stage3_strict_semantic_oracle_pass": stage3_s.get("stage3_strict_semantic_oracle_pass"),
        "stage4_read_provider_pass": stage4_s.get("read_provider_pass"),
        "stage4_strict_memory_action_surface_pass": stage4_s.get("strict_memory_action_surface_pass"),
        "stage5_run": stage5_s.get("stage5_run"),
        "visual_evidence_status": visual_status or visual_blocker,
        "blocking_requirements": [
            "strict semantic oracle upper bound did not pass target-coverage/correlation/control requirements",
            "SWA/TTT/admission action surfaces did not validate true L3/L4 metric movement",
            "Stage5/6/7 not allowed by gates",
        ],
    }
    if visual_blocker:
        decision["blocking_requirements"].insert(0, visual_blocker)
    write_json(out / "final_decision.json", decision)

    evidence_rows = []
    for stage, summary_path in [
        ("stage0", ROOT / "stage0_evidence_ledger/stage0_summary.json"),
        ("stage1", ROOT / "stage1_drift_source_autopsy/stage1_summary.json"),
        ("stage2", ROOT / "stage2_base_case_selection/stage2_summary.json"),
        ("stage2_5", ROOT / "stage2_5_legacy_reentry_matrix/stage2_5_summary.json"),
        ("stage3", ROOT / "stage3_semantic_oracle_upper_bound/stage3_summary.json"),
        ("stage4", ROOT / "stage4_memory_action_surface_oracle/stage4_summary.json"),
        ("stage5", ROOT / "stage5_training_free_cue_distillation/stage5_summary.json"),
    ]:
        evidence_rows.append({"stage": stage, "summary_path": rel(summary_path), "exists": summary_path.exists()})
    write_csv(out / "evidence_table.csv", evidence_rows)

    write_text(
        out / "what_passed_what_failed.md",
        "# What Passed / What Failed\n\n"
        "Passed or useful:\n\n"
        "- Stage 0 evidence freeze and legacy route ledger were materialized.\n"
        "- Stage 1 drift-source taxonomy was computed from v97 metrics joined with v100/v101 evidence.\n"
        + ("- Strict visual evidence passed after local point residual sidecar repair.\n" if visual_gate_pass else "")
        +
        "- Stage 2.5 legacy route matrix passed: old results are mapped into provider/oracle/carrier/negative-control roles.\n"
        "- Diagnostic/proxy oracle signals exist in B5/B7/B8/B10.\n"
        "- READ H2/L07 remains a local L2/provider mechanism, not a full method.\n\n"
        "Failed / blocked:\n\n"
        + (f"- {visual_blocker}\n" if visual_blocker else "")
        +
        "- Strict Stage 3 semantic oracle pass is false: full-control semantic rotation now has exploratory signal, but strict clean handoff coverage/action-control is insufficient.\n"
        "- SWA/TTT action surfaces do not have validated true L3/L4 metric improvement.\n"
        "- Stage 5/6/7 were not allowed; no runtime/full validation was run.\n",
    )
    write_text(
        out / "next_route_recommendation.md",
        "# Next Route Recommendation\n\n"
        "1. Treat `semantic_unreliable_anchor_frac` as an exploratory B2/B5 risk cue only; do not promote it from contaminated/multimode scope.\n"
        "2. Existing measured action artifacts have been audited and do not contain a strict true L3/L4 READ/SWA/TTT upper bound; materialize new labelled clean handoff positives or a new state-machine hook with measured true L3/L4 effect before Stage4/5/6.\n"
        "3. If strict B5/B7 passes after true terms, implement a new SWA state-machine action hook and measure true L3, not proxy transport mass.\n"
        "4. Keep READ as provider/local mechanism; do not promote READ to full method.\n"
        "5. Use TTT only after refresh/expire/context-only write-to-use state machine can be measured on later READ/SWA use and L3/L4.\n",
    )
    write_text(
        out / "do_not_repeat.md",
        "# Do Not Repeat\n\n"
        "- weak-context skip / anchor rescue / rho sweep\n"
        "- READ beta / chunk selector full-method promotion\n"
        "- Track E source-gate/source-replace/merge-alpha family\n"
        "- R_same/query_hit/freshness/O_scale single-threshold runtime selector\n"
        "- TTT write mass / retention proxy promotion\n"
        "- query-soft/aggregate TTT small sweep as if it were a new state-machine action\n",
    )
    write_text(
        out / "final_report.md",
        "# ACL2 v102 Final Report\n\n"
        f"Final taxonomy: `{decision['final_taxonomy']}`\n\n"
        "v102 produced a useful diagnosis but not a runtime method. Drift sources can be separated in the artifact-backed exploration set, and legacy cues/providers remain useful. "
        f"{visual_status + ' ' if visual_status else ''}"
        "However, the strict semantic oracle does not pass: full-control semantic rotation finds exploratory unreliable-semantic risk signal, but the strict clean handoff universe still has insufficient positives and no action-surface L3/L4 effect. "
        "READ local/provider evidence passes, but SWA/TTT/admission action surfaces do not validate true L3/L4 movement. Therefore Stage 6 runtime and Stage 7 full validation were not run.\n",
    )
    # Completion audit.
    requirements = [
        ("dual_logs_created", True, "docs execution/retrospective logs created"),
        ("stage0_known_facts_and_legacy_ledger", bool(stage0_s.get("stage0_pass")), rel(ROOT / "stage0_evidence_ledger/stage0_summary.json")),
        ("stage1_drift_autopsy_outputs", bool(stage1_s.get("case_count")), rel(ROOT / "stage1_drift_source_autopsy/stage1_summary.json")),
        ("stage1_strict_visual_overlay", bool(stage1_s.get("visual_overlay_strict_pass")), stage1_s.get("visual_overlay_limitation", "strict visual overlay not passed")),
        ("stage2_base_cases", bool(stage2_s.get("stage2_diagnostic_gate_pass")), rel(ROOT / "stage2_base_case_selection/stage2_summary.json")),
        (
            "stage2_rgb_semantic_overlay_fail_forward_artifacts",
            int(stage2_s.get("rgb_semantic_overlay_case_count") or 0) > 0,
            rel(ROOT / "stage2_base_case_selection/rgb_semantic_overlay_manifest.csv"),
        ),
        (
            "stage2_trajectory_error_map_fail_forward_artifacts",
            int(stage2_s.get("trajectory_error_map_case_count") or 0) > 0,
            rel(ROOT / "stage2_base_case_selection/trajectory_error_overlay_manifest.csv"),
        ),
        (
            "stage2_local_point_residual_map_fail_forward_artifacts",
            int(stage2_s.get("local_point_error_map_case_count") or 0) > 0,
            rel(ROOT / "stage2_base_case_selection/local_point_residual_overlay_manifest.csv"),
        ),
        ("stage2_strict_visual_gate", bool(stage2_s.get("stage2_strict_gate_pass")), stage2_s.get("visual_limitation", "strict visual gate not passed")),
        ("stage2_5_legacy_matrix", bool(stage25_s.get("stage2_5_gate_pass")), rel(ROOT / "stage2_5_legacy_reentry_matrix/stage2_5_summary.json")),
        ("stage3_strict_semantic_oracle", bool(stage3_s.get("stage3_strict_semantic_oracle_pass")), rel(ROOT / "stage3_semantic_oracle_upper_bound/stage3_summary.json")),
        (
            "stage3_fail_forward_artifacts",
            all(
                path.exists()
                for path in (
                    ROOT / "stage3_semantic_oracle_upper_bound/semantic_oracle_failure_decomposition.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/semantic_oracle_false_positive_panels.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/semantic_oracle_missed_positive_panels.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/three_way_legacy_cue_conflict_panel.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_local_geometry_oracle_repair_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_full_control_rerun_readiness_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_full_control_semantic_rotation_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_full_control_semantic_rotation_policy_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_full_control_semantic_rotation_control_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_clean_handoff_candidate_expansion_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_clean_handoff_candidate_expansion_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_broader_drift_onset_candidate_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_clean_handoff_candidate_expansion_report.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/broader_drift_onset_trace_extension_targets.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/broader_drift_onset_trace_extension_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/broader_drift_onset_trace_extension_audit_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/broader_drift_onset_trace_extension_report.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/read_local_oracle_bridge_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/read_local_oracle_bridge_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/read_local_oracle_bridge_report.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/read_local_help_global_harm_report.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/read_local_active_inactive_global_tradeoff.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/read_local_global_sim3_scale_yaw_shift.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_legacy_cue_case_alignment_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_legacy_cue_case_alignment_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_legacy_cue_case_alignment_case_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/legacy_cue_case_alignment_report.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_historical_clean_target_extension_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_historical_clean_target_extension_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/historical_clean_target_extension_report.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_strict_clean_handoff_materialization_repair_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_strict_clean_handoff_materialization_repair_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/strict_clean_handoff_materialization_repair_report.md",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_exhaustive_clean_handoff_target_mining_summary.json",
                    ROOT / "stage3_semantic_oracle_upper_bound/stage3_exhaustive_clean_handoff_target_mining_rows.csv",
                    ROOT / "stage3_semantic_oracle_upper_bound/exhaustive_clean_handoff_target_mining_report.md",
                )
            ),
            rel(ROOT / "stage3_semantic_oracle_upper_bound"),
        ),
        ("stage4_strict_memory_action_surface", bool(stage4_s.get("strict_memory_action_surface_pass")), rel(ROOT / "stage4_memory_action_surface_oracle/stage4_summary.json")),
        (
            "stage4_action_gap_artifacts",
            (ROOT / "stage4_memory_action_surface_oracle/action_surface_gap_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/action_body_failure_taxonomy.csv").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_hook_readiness_artifacts",
            (ROOT / "stage4_memory_action_surface_oracle/state_machine_hook_readiness_summary.json").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_hook_readiness_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_hook_code_loci.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_hook_readiness_report.md").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_true_l3_upper_bound_feasibility_artifacts",
            (ROOT / "stage4_memory_action_surface_oracle/action_surface_true_l3_upper_bound_feasibility_summary.json").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/action_surface_true_l3_upper_bound_feasibility_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/action_surface_true_l3_upper_bound_feasibility_report.md").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_scaffold_trace_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_scaffold_trace_closure_summary.json").get("scaffold_trace_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_scaffold_trace_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_scaffold_trace_closure_report.md").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_donly_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_donly_v2_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_donly_v2_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_donly_v2_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_donly_v2_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_d075_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d075_v3_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d075_v3_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d075_v3_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d075_v3_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_d025_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d025_v4_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d025_v4_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d025_v4_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_reject_unreliable_d025_v4_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_transmit_supported_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_transmit_supported_soft_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_soft_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_soft_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_soft_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_transmit_supported_soft_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_hold_prev_reference_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_hold_prev_reference_soft_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_hold_prev_reference_soft2_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft2_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft2_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft2_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_hold_prev_reference_soft2_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_delay_update_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_delay_update_soft_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_delay_update_soft075_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft075_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft075_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft075_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft075_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_delay_update_soft0875_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft0875_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft0875_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft0875_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_delay_update_soft0875_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_context_only_soft_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_state_machine_action_probe_context_only_soft075_closure_artifacts",
            bool(read_json(ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft075_v1_closure_summary.json").get("action_probe_materialization_pass"))
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft075_v1_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft075_v1_closure_report.md").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/state_machine_action_probe_context_only_demotion_soft075_v1_metrics/state_machine_trace_run_metrics_summary.json").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        (
            "stage4_ttt_write_to_use_chain_closure_artifacts",
            (ROOT / "stage4_memory_action_surface_oracle/ttt_write_to_use_chain_closure_summary.json").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/ttt_write_to_use_chain_closure_rows.csv").exists()
            and (ROOT / "stage4_memory_action_surface_oracle/ttt_write_to_use_chain_closure_report.md").exists(),
            rel(ROOT / "stage4_memory_action_surface_oracle"),
        ),
        ("stage5_cue_distillation_allowed_and_passed", False, rel(ROOT / "stage5_training_free_cue_distillation/stage5_summary.json")),
        ("stage6_runtime_pilot_allowed_and_run", False, "not allowed by Stage3/4/5 gates"),
        ("stage7_full_validation_allowed_and_run", False, "not allowed by Stage6 gate"),
        ("final_decision_artifacts", True, rel(ROOT / "final_decision/final_decision.json")),
    ]
    audit_rows = [
        {"requirement": req, "pass": ok, "evidence": evidence}
        for req, ok, evidence in requirements
    ]
    write_csv(out / "completion_audit.csv", audit_rows)
    completion = {
        "requirement_count": len(audit_rows),
        "passed_requirement_count": sum(1 for r in audit_rows if r["pass"]),
        "failed_requirement_count": sum(1 for r in audit_rows if not r["pass"]),
        "failed_requirements": [r["requirement"] for r in audit_rows if not r["pass"]],
        "goal_achieved": False,
        "runtime_action_allowed": False,
        "full_method_success": False,
    }
    write_json(out / "completion_audit_summary.json", completion)
    return decision


def main() -> None:
    ensure_dir(ROOT)
    stage0_s = stage0()
    s1 = stage1()
    stage2_s = stage2(s1)
    stage25_s = stage2_5()
    stage3_s = stage3()
    stage4_s = stage4()
    stage5_s = stage5_blocked(stage4_s)
    decision = final_decision(stage0_s, s1.summary, stage2_s, stage25_s, stage3_s, stage4_s, stage5_s)
    print(
        json.dumps(
            {
                "result_root": rel(ROOT),
                "final_taxonomy": decision["final_taxonomy"],
                "goal_achieved": decision["goal_achieved"],
                "runtime_action_allowed": decision["runtime_action_allowed"],
                "stage3_strict_semantic_oracle_pass": decision["stage3_strict_semantic_oracle_pass"],
                "stage4_strict_memory_action_surface_pass": decision["stage4_strict_memory_action_surface_pass"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
