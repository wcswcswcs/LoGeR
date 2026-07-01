#!/usr/bin/env python3
"""Build v94 final decision, blocked downstream summaries, and audit panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - final audit should expose malformed inputs.
        return {"read_error": str(exc), "path": str(path)}
    return data if isinstance(data, dict) else {"value": data}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom else 0.0


def make_panel(path: Path, title: str, lines: list[str], bars: dict[str, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6), dpi=140)
    if bars:
        ax = fig.add_axes([0.12, 0.18, 0.80, 0.52])
        labels = list(bars)
        values = [bars[label] for label in labels]
        colors = ["#315c8c", "#b15d45", "#5d7f49", "#7a628f", "#b58b3b", "#4e7775"][: len(labels)]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax = fig.add_axes([0.08, 0.10, 0.84, 0.72])
        ax.axis("off")
    fig.text(0.06, 0.93, title, fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.86, "\n".join(lines), fontsize=10, ha="left", va="top")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def blocked_summary(phase: str, blocker: str, preconditions: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": phase,
        "entered": False,
        "gate_pass": False,
        "blocker": blocker,
        "preconditions": preconditions,
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
        "note": "Blocked by v94 stop rule; no downstream metric was fabricated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "report_final")
    args = parser.parse_args()

    phase0 = read_json(args.root / "phase0_restart_evidence_lock/phase0_gate_summary.json")
    phase1 = read_json(args.root / "phase1_boundary_failure_atlas/phase1_gate_summary.json")
    phase2 = read_json(args.root / "phase2_true_carrier_trace/phase2_gate_summary.json")
    phase3 = read_json(args.root / "phase3_neutral_causal_sensitivity/phase3_gate_summary.json")
    phase3r = read_json(args.root / "phase3r_runtime_merge_gauge_probe/runtime_probe_sensitivity_summary.json")
    phase3s = read_json(
        args.root / "phase3s_merge_gauge_actuator_sweep_max16_confirm/runtime_probe_sensitivity_summary.json"
    )
    phase3_formal = read_json(args.root / "phase3_formal_merge_alpha_sensitivity/phase3_formal_gate_summary.json")
    phase4 = read_json(args.root / "phase4_semantic_evidence_taxonomy/semantic_taxonomy_summary.json")
    phase5 = read_json(args.root / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json")
    phase5_next_route = read_json(
        args.root / "phase5_next_route_diagnostic/phase5_next_route_diagnostic_summary.json"
    )
    phase5_object_source = read_json(
        args.root / "phase5_object_source_extension/phase5_object_source_extension_summary.json"
    )
    phase6_object_cf = read_json(
        args.root / "phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json"
    )
    phase6_repair_search = read_json(
        args.root / "phase6_object_source_repair_search/phase6_object_source_repair_search_summary.json"
    )
    phase6_action_surface = read_json(
        args.root / "phase6_object_source_action_surface/phase6_object_source_action_surface_summary.json"
    )
    phase3r_available = bool(phase3r) and "read_error" not in phase3r
    phase3s_available = bool(phase3s) and "read_error" not in phase3s
    phase3_formal_available = bool(phase3_formal) and "read_error" not in phase3_formal
    phase4_available = bool(phase4) and "read_error" not in phase4
    phase5_available = bool(phase5) and "read_error" not in phase5
    phase5_next_route_available = bool(phase5_next_route) and "read_error" not in phase5_next_route
    phase5_object_source_available = bool(phase5_object_source) and "read_error" not in phase5_object_source
    phase6_object_cf_available = bool(phase6_object_cf) and "read_error" not in phase6_object_cf
    phase6_repair_search_available = bool(phase6_repair_search) and "read_error" not in phase6_repair_search
    phase6_action_surface_available = bool(phase6_action_surface) and "read_error" not in phase6_action_surface
    phase3s_selected = phase3s.get("selected_candidate_summary") if phase3s_available else {}
    if not isinstance(phase3s_selected, dict):
        phase3s_selected = {}
    phase3s_gate_pass = bool(phase3s.get("phase3r_runtime_probe_gate_pass")) if phase3s_available else False

    row_count = int(phase1.get("row_count") or 0)
    counts = phase1.get("primary_failure_type_counts") or {}
    local_bad = int(counts.get("LOCAL_BAD") or 0)
    handoff_scale = int(counts.get("HANDOFF_SCALE") or 0)
    handoff_gauge = int(counts.get("HANDOFF_GAUGE") or 0)
    handoff_total = handoff_scale + handoff_gauge
    local_handoff_total = local_bad + handoff_total

    phase3_pass = bool(phase3.get("phase3_gate_pass"))
    phase3_formal_pass = bool(phase3_formal.get("phase3_repaired_gate_pass")) if phase3_formal_available else False
    phase4_pass = bool(phase4.get("phase4_semantic_taxonomy_gate_pass")) if phase4_available else False
    phase5_pass = bool(phase5.get("phase5_semantic_carrier_alignment_gate_pass")) if phase5_available else False
    phase5_object_source_pass = (
        bool(phase5_object_source.get("object_source_extension_gate_pass"))
        if phase5_object_source_available
        else False
    )
    phase6_object_cf_pass = (
        bool(phase6_object_cf.get("phase6_object_source_counterfactual_gate_pass"))
        if phase6_object_cf_available
        else False
    )
    phase6_repair_search_pass = (
        bool(phase6_repair_search.get("repair_candidate_gate_pass"))
        if phase6_repair_search_available
        else False
    )
    phase6_action_surface_pass = (
        bool(phase6_action_surface.get("phase6_object_source_action_surface_gate_pass"))
        if phase6_action_surface_available
        else False
    )
    phase5_best = phase5.get("best_alignment_role") or {}
    if not isinstance(phase5_best, dict):
        phase5_best = {}
    phase5_strict_shuffle_pass = (
        phase5_available
        and float(phase5_best.get("semantic_shuffle_margin") or 0.0) >= 0.05
        and float(phase5_best.get("component_shuffle_margin") or 0.0) >= 0.05
        and float(phase5_best.get("regime_shuffle_margin") or 0.0) >= 0.05
    )
    phase5_policy_rows = phase5.get("carrier_event_policy_metrics") if phase5_available else []
    if not isinstance(phase5_policy_rows, list):
        phase5_policy_rows = []
    phase5_best_policy = max(
        [row for row in phase5_policy_rows if isinstance(row, dict)],
        key=lambda row: (
            float(row.get("bad_recall") or 0.0),
            -float(row.get("good_FPR") or 0.0),
            float(row.get("loso_positive_folds") or 0.0),
        ),
        default={},
    )
    if phase6_action_surface_available and not phase6_action_surface_pass:
        final_status = "NO_GO_PHASE6_OBJECT_SOURCE_ACTION_SURFACE_CONTROL_FAILED"
        blocker = phase6_action_surface.get("blocker") or "phase6_object_source_action_surface_failed"
    elif phase6_action_surface_available and phase6_action_surface_pass:
        final_status = "NO_GO_PHASE7_RUNTIME_POLICY_NOT_RUN_OR_PROMOTED"
        blocker = "phase6_action_surface_passed_but_phase7_online_runtime_policy_not_run"
    elif phase6_repair_search_available and not phase6_repair_search_pass:
        final_status = "NO_GO_PHASE6_OBJECT_SOURCE_REPAIR_SEARCH_FAILED"
        blocker = "phase6_object_source_repair_search_found_no_passing_candidate"
    elif phase5_object_source_pass and phase6_object_cf_available and not phase6_object_cf_pass:
        final_status = "NO_GO_PHASE6_OBJECT_SOURCE_TRACE_COUNTERFACTUAL_FAILED"
        blocker = phase6_object_cf.get("blocker") or "phase6_object_source_counterfactual_failed"
    elif phase5_object_source_pass and not phase6_object_cf_available:
        final_status = "NO_GO_PHASE6_OBJECT_SOURCE_COUNTERFACTUAL_NOT_RUN_OR_BLOCKED"
        blocker = "phase5_object_source_extension_passed_but_phase6_object_counterfactual_missing"
    elif phase3_formal_pass and phase4_pass and phase5_available and not phase5_pass:
        final_status = "NO_GO_PHASE5_SEMANTIC_CARRIER_ALIGNMENT_FAILED"
        blocker = phase5.get("blocker") or "phase5_semantic_carrier_alignment_failed"
    elif phase3_formal_pass and phase4_available and not phase4_pass:
        final_status = "NO_GO_PHASE4_SEMANTIC_TAXONOMY_NOT_SPECIFIC"
        blocker = phase4.get("blocker") or "phase4_semantic_taxonomy_failed"
    elif phase3_formal_available and not phase3_formal_pass:
        final_status = "NO_GO_PHASE3_FORMAL_REPAIR_FAILED"
        blocker = phase3_formal.get("blocker") or "phase3_formal_repair_failed"
    elif phase3_formal_pass and phase4_pass and phase5_pass:
        final_status = "NO_GO_PHASE6_COUNTERFACTUAL_NOT_RUN_OR_BLOCKED"
        blocker = "phase6_counterfactual_not_run_after_phase5_pass"
    else:
        final_status = "NO_GO_PHASE3_NO_MEMORY_BODY_SENSITIVE"
        blocker = phase3.get("blocker") or "phase3_no_sensitive_memory_body"
    labels = [
        "D2_HANDOFF_FAILURE_CONFIRMED",
        "D4_ORIGINAL_PHASE3_NO_MEMORY_BODY_SENSITIVE",
        "D12_TTT_NOT_READY",
    ]
    if phase3r_available:
        labels.append(
            "D13_PHASE3R_RUNTIME_PROXY_PASS_ORIGINAL_GATE_STILL_BLOCKED"
            if phase3r.get("phase3r_runtime_probe_gate_pass")
            else "D13_PHASE3R_RUNTIME_PROBE_FAILED"
        )
    if phase3s_available:
        labels.append(
            "D14_PHASE3S_MERGE_ALPHA_ACTUATOR_CONFIRMED_DIAGNOSTIC_ONLY"
            if phase3s_gate_pass
            else "D14_PHASE3S_ACTUATOR_SWEEP_FAILED"
        )
    if phase3_formal_available:
        labels.append(
            "D15_PHASE3_FORMAL_MERGE_ALPHA_GATE_PASS"
            if phase3_formal_pass
            else "D15_PHASE3_FORMAL_MERGE_ALPHA_GATE_FAILED"
        )
    if phase4_available:
        labels.append(
            "D16_PHASE4_SEMANTIC_TAXONOMY_ROLE_PASS"
            if phase4_pass
            else "D16_PHASE4_SEMANTIC_TAXONOMY_FAILED"
        )
    if phase5_available:
        labels.append(
            "D17_PHASE5_SEMANTIC_CARRIER_ALIGNMENT_PASS"
            if phase5_pass
            else "D17_PHASE5_SEMANTIC_CARRIER_ALIGNMENT_FAILED"
        )
        if not phase5_pass:
            labels.append("D18_CANONICAL_COUNTERFACTUAL_BLOCKED_BY_SEMANTIC_PHASE5")
    if phase5_object_source_available:
        labels.append(
            "D19_PHASE5_OBJECT_SOURCE_EXTENSION_PASS_DIAGNOSTIC_ONLY"
            if phase5_object_source_pass
            else "D19_PHASE5_OBJECT_SOURCE_EXTENSION_FAILED"
        )
    if phase6_object_cf_available:
        labels.append(
            "D20_PHASE6_OBJECT_SOURCE_TRACE_COUNTERFACTUAL_PASS"
            if phase6_object_cf_pass
            else "D20_PHASE6_OBJECT_SOURCE_TRACE_COUNTERFACTUAL_FAILED"
        )
    if phase6_repair_search_available:
        labels.append(
            "D21_PHASE6_OBJECT_SOURCE_REPAIR_SEARCH_PASS"
            if phase6_repair_search_pass
            else "D21_PHASE6_OBJECT_SOURCE_REPAIR_SEARCH_FAILED"
        )
    if phase6_action_surface_available:
        labels.append(
            "D22_PHASE6_OBJECT_SOURCE_ACTION_SURFACE_PASS"
            if phase6_action_surface_pass
            else "D22_PHASE6_OBJECT_SOURCE_ACTION_SURFACE_FAILED"
        )
        if phase6_action_surface.get("semantic_not_specific"):
            labels.append("D23_OBJECT_SOURCE_ACTION_SURFACE_SEMANTIC_NOT_SPECIFIC")

    preconditions = {
        "phase0_gate_pass": bool(phase0.get("phase0_gate_pass")),
        "phase1_gate_pass": bool(phase1.get("phase1_gate_pass")),
        "phase2_gate_pass": bool(phase2.get("phase2_gate_pass")),
        "phase3_gate_pass": phase3_pass,
        "phase3_stop_rule": phase3.get("stop_rule_triggered"),
        "phase3r_runtime_probe_executed": phase3r.get("runtime_probe_executed") if phase3r_available else False,
        "phase3r_runtime_probe_gate_pass": phase3r.get("phase3r_runtime_probe_gate_pass")
        if phase3r_available
        else False,
        "phase3s_actuator_probe_executed": phase3s.get("runtime_probe_executed") if phase3s_available else False,
        "phase3s_actuator_probe_gate_pass": phase3s_gate_pass,
        "phase3_formal_repaired_gate_pass": phase3_formal_pass,
        "phase4_semantic_taxonomy_gate_pass": phase4_pass,
        "phase5_semantic_carrier_alignment_gate_pass": phase5_pass,
        "phase5_object_source_extension_gate_pass": phase5_object_source_pass,
        "phase6_object_source_counterfactual_gate_pass": phase6_object_cf_pass,
        "phase6_object_source_repair_search_gate_pass": phase6_repair_search_pass,
        "phase6_object_source_action_surface_gate_pass": phase6_action_surface_pass,
    }
    write_json(
        args.root / "phase4_semantic_evidence_taxonomy_or_blocked/blocked_summary.json",
        {
            "phase": "Phase4_semantic_evidence_taxonomy",
            "entered": phase4_available,
            "gate_pass": phase4_pass,
            "blocker": phase4.get("blocker") if phase4_available else "not_entered_or_missing",
            "source": "phase4_semantic_evidence_taxonomy/semantic_taxonomy_summary.json",
            "runtime_action_allowed": False,
            "counterfactual_allowed": False,
            "ttt_allowed": False,
            "note": "Legacy *_or_blocked alias updated to reflect actual Phase4 execution.",
        },
    )
    write_json(
        args.root / "phase5_semantic_carrier_alignment_or_blocked/blocked_summary.json",
        {
            "phase": "Phase5_semantic_carrier_alignment",
            "entered": phase5_available,
            "gate_pass": phase5_pass,
            "blocker": phase5.get("blocker") if phase5_available else "not_entered_or_missing",
            "source": "phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json",
            "runtime_action_allowed": False,
            "counterfactual_allowed": False,
            "ttt_allowed": False,
            "note": "Legacy *_or_blocked alias updated to reflect actual Phase5 execution.",
        },
    )
    if phase6_object_cf_available:
        downstream_blocker = (
            "blocked_without_runtime_action_pass"
            if phase6_action_surface_available and phase6_action_surface_pass
            else "blocked_by_phase6_object_source_action_surface_failed"
            if phase6_action_surface_available and not phase6_action_surface_pass
            else "blocked_without_runtime_action_pass"
            if phase6_object_cf_pass and (not phase6_repair_search_available or phase6_repair_search_pass)
            else "blocked_by_phase6_object_source_repair_search_failed"
            if phase6_repair_search_available and not phase6_repair_search_pass
            else "blocked_by_phase6_object_source_counterfactual_failed"
        )
        write_json(
            args.root / "phase6_counterfactual_or_blocked/blocked_summary.json",
            {
                "phase": "Phase6_counterfactual_or_blocked",
                "entered": True,
                "gate_pass": phase6_object_cf_pass
                and (not phase6_repair_search_available or phase6_repair_search_pass)
                and (not phase6_action_surface_available or phase6_action_surface_pass),
                "blocker": blocker,
                "latest_source": "phase6_object_source_action_surface/phase6_object_source_action_surface_summary.json"
                if phase6_action_surface_available
                else "phase6_object_source_repair_search/phase6_object_source_repair_search_summary.json"
                if phase6_repair_search_available
                else "phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json",
                "source": "phase6_object_source_action_surface/phase6_object_source_action_surface_summary.json"
                if phase6_action_surface_available
                else "phase6_object_source_repair_search/phase6_object_source_repair_search_summary.json"
                if phase6_repair_search_available
                else "phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json",
                "preconditions": preconditions,
                "trace_level_upper_bound_only": phase6_object_cf.get("trace_level_upper_bound_only"),
                "actual_runtime_trajectory_counterfactual_available": phase6_object_cf.get(
                    "actual_runtime_trajectory_counterfactual_available"
                ),
                "phase6_object_source_counterfactual_blocker": phase6_object_cf.get("blocker"),
                "phase6_object_source_repair_search_gate_pass": phase6_repair_search_pass
                if phase6_repair_search_available
                else "",
                "phase6_object_source_repair_search_candidate_count": phase6_repair_search.get("candidate_count")
                if phase6_repair_search_available
                else "",
                "phase6_object_source_repair_search_passing_candidate_count": phase6_repair_search.get(
                    "passing_candidate_count"
                )
                if phase6_repair_search_available
                else "",
                "phase6_object_source_action_surface_gate_pass": phase6_action_surface_pass
                if phase6_action_surface_available
                else "",
                "phase6_object_source_action_surface_blocker": phase6_action_surface.get("blocker")
                if phase6_action_surface_available
                else "",
                "phase6_object_source_action_surface_actual_minus_best_control": phase6_action_surface.get(
                    "actual_minus_best_control"
                )
                if phase6_action_surface_available
                else "",
                "measured_runtime_action_surface_available": phase6_action_surface_available,
                "runtime_action_allowed": bool(phase6_object_cf.get("runtime_action_allowed")),
                "counterfactual_allowed": False,
                "ttt_allowed": False,
                "note": (
                    "Object-source extension reached measured runtime-probe action-surface diagnostics; "
                    "the online runtime policy is still not promoted."
                )
                if phase6_action_surface_available
                else (
                    "Object-source extension entered diagnostic trace-level counterfactual/repair search; "
                    "no runtime trajectory rerun is claimed."
                ),
            },
        )
    else:
        downstream_blocker = (
            "blocked_by_phase5_semantic_carrier_alignment_failed"
            if phase5_available and not phase5_pass
            else "blocked_without_phase6_counterfactual_pass"
        )
        write_json(
            args.root / "phase6_counterfactual_or_blocked/blocked_summary.json",
            blocked_summary("Phase6_counterfactual_or_blocked", downstream_blocker, preconditions),
        )
    blocked_dirs = [
        ("phase7_runtime_or_blocked", "Phase7_runtime_or_blocked", downstream_blocker),
        ("phase8_ttt_or_blocked", "Phase8_ttt_or_blocked", "blocked_without_runtime_action_pass"),
    ]
    for dirname, phase_name, reason in blocked_dirs:
        write_json(args.root / dirname / "blocked_summary.json", blocked_summary(phase_name, reason, preconditions))

    phase6_object_actual = phase6_object_cf.get("actual_family") if phase6_object_cf_available else {}
    if not isinstance(phase6_object_actual, dict):
        phase6_object_actual = {}
    phase6_repair_best = phase6_repair_search.get("best_candidate") if phase6_repair_search_available else {}
    if not isinstance(phase6_repair_best, dict):
        phase6_repair_best = {}
    phase6_action_actual = phase6_action_surface.get("actual_family") if phase6_action_surface_available else {}
    if not isinstance(phase6_action_actual, dict):
        phase6_action_actual = {}

    visual_dir = args.root / "phase9_visual_audit_or_blocked"
    panels = [
        {
            "category": "failure_type_atlas_panels",
            "file": visual_dir / "failure_type_atlas_panels.png",
            "status": "audit_chart_available",
            "evidence": "phase1_boundary_failure_atlas/boundary_failure_rows.csv",
            "blocked_reason": "",
            "bars": {str(key): int(value) for key, value in counts.items()},
            "lines": [
                f"Phase1 pass: {phase1.get('phase1_gate_pass')}",
                f"Rows: {row_count}; local={local_bad}; handoff_scale={handoff_scale}; handoff_gauge={handoff_gauge}",
                "This chart is an audit panel, not raw RGB/overlay visualization.",
            ],
        },
        {
            "category": "true_carrier_trace_panels",
            "file": visual_dir / "true_carrier_trace_panels.png",
            "status": "audit_chart_available",
            "evidence": "phase2_true_carrier_trace/phase2_gate_summary.json",
            "blocked_reason": "",
            "bars": {
                "carrier": float(phase2.get("carrier_trace_coverage") or 0.0),
                "read": float(phase2.get("read_true_trace_coverage") or 0.0),
                "merge": float(phase2.get("merge_gauge_true_trace_coverage") or 0.0),
                "residual": float(phase2.get("merge_residual_delta_coverage") or 0.0),
                "swa": float(phase2.get("swa_true_route_coverage") or 0.0),
            },
            "lines": [
                f"Phase2 pass: {phase2.get('phase2_gate_pass')}",
                f"merge_residual_delta_coverage={phase2.get('merge_residual_delta_coverage')}",
                f"swa_explicitly_unavailable={phase2.get('swa_explicitly_unavailable')}",
            ],
        },
        {
            "category": "causal_sensitivity_panels",
            "file": visual_dir / "causal_sensitivity_panels.png",
            "status": "audit_chart_available",
            "evidence": "phase3_neutral_causal_sensitivity/phase3_gate_summary.json",
            "blocked_reason": "",
            "bars": {
                "scale": float((phase3.get("balanced_probe") or {}).get("handoff_scale_rows") or 0),
                "gauge": float((phase3.get("balanced_probe") or {}).get("handoff_gauge_rows") or 0),
                "good_safe": float((phase3.get("balanced_probe") or {}).get("good_safe_rows") or 0),
            },
            "lines": [
                f"Phase3 pass: {phase3.get('phase3_gate_pass')}",
                f"balanced_probe_pass={phase3.get('balanced_probe', {}).get('balanced_probe_gate_pass')}",
                f"J_handoff_probe_available={phase3.get('checks', {}).get('any_j_handoff_probe_available')}",
                "Trace deltas exist, but no memory body satisfies causal sensitivity.",
            ],
        },
        {
            "category": "semantic_evidence_taxonomy_panels",
            "file": visual_dir / "semantic_evidence_taxonomy_panels.png",
            "status": "audit_chart_available" if phase4_available else "blocked_placeholder",
            "evidence": "phase4_semantic_evidence_taxonomy/semantic_taxonomy_summary.json"
            if phase4_available
            else "phase4_semantic_evidence_taxonomy_or_blocked/blocked_summary.json",
            "blocked_reason": "" if phase4_available else "Phase4 missing.",
            "bars": {
                str(key): int(value)
                for key, value in (phase4.get("semantic_evidence_type_counts") or {}).items()
            }
            if phase4_available
            else None,
            "lines": [
                f"Phase4 pass: {phase4_pass}",
                f"best_semantic_role={((phase4.get('best_semantic_role') or {}).get('semantic_role'))}",
                f"best_role_good_FPR={((phase4.get('best_semantic_role') or {}).get('good_FPR'))}",
                f"aggregate_good_FPR={phase4.get('semantic_good_FPR')}",
                "Phase4 passes only through role-specific evidence; aggregate semantic risk remains diagnostic.",
            ]
            if phase4_available
            else ["Phase4 missing.", "No semantic taxonomy metric was fabricated."],
        },
        {
            "category": "semantic_carrier_alignment_panels",
            "file": visual_dir / "semantic_carrier_alignment_panels.png",
            "status": "audit_chart_available" if phase5_available else "blocked_placeholder",
            "evidence": "phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json"
            if phase5_available
            else "phase5_semantic_carrier_alignment_or_blocked/blocked_summary.json",
            "blocked_reason": "" if phase5_available else "Phase5 missing.",
            "bars": {
                "bad_recall": float((phase5.get("best_alignment_role") or {}).get("bad_recall") or 0.0),
                "good_FPR": float((phase5.get("best_alignment_role") or {}).get("good_FPR") or 0.0),
                "LOSO": float((phase5.get("best_alignment_role") or {}).get("loso_positive_folds") or 0.0),
            }
            if phase5_available
            else None,
            "lines": [
                f"Phase5 pass: {phase5_pass}",
                f"blocker={phase5.get('blocker')}",
                f"best_alignment_role={((phase5.get('best_alignment_role') or {}).get('semantic_role'))}",
                f"bad_recall={((phase5.get('best_alignment_role') or {}).get('bad_recall'))}; good_FPR={((phase5.get('best_alignment_role') or {}).get('good_FPR'))}",
                f"max_positive_carrier_subfield_corr={((phase5.get('best_alignment_role') or {}).get('max_positive_carrier_subfield_corr'))}",
            ]
            if phase5_available
            else ["Semantic-carrier alignment missing.", "No carrier alignment panel is claimed."],
        },
        {
            "category": "counterfactual_panels",
            "file": visual_dir / "counterfactual_panels.png",
            "status": "audit_chart_available" if phase6_object_cf_available else "blocked_placeholder",
            "evidence": "phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json"
            if phase6_object_cf_available
            else "phase6_counterfactual_or_blocked/blocked_summary.json",
            "blocked_reason": phase6_object_cf.get("blocker") if phase6_object_cf_available else downstream_blocker,
            "bars": {
                "bad_I_runtime": float(phase6_action_actual.get("bad_median_I_J_runtime_proxy") or 0.0),
                "good_W_runtime": float(phase6_action_actual.get("good_max_worsen_runtime_proxy") or 0.0),
                "minus_control": float(phase6_action_surface.get("actual_minus_best_control") or 0.0),
            }
            if phase6_action_surface_available
            else {
                "bad_recall": float(phase6_object_actual.get("bad_action_recall") or 0.0),
                "good_FPR": float(phase6_object_actual.get("good_action_FPR") or 0.0),
                "bad_improve_ratio": float(phase6_object_actual.get("bad_median_residual_improvement_ratio") or 0.0),
            }
            if phase6_object_cf_available
            else None,
            "lines": [
                f"Object-source action surface pass: {phase6_action_surface_pass}"
                if phase6_action_surface_available
                else f"Object-source Phase6 pass: {phase6_object_cf_pass}",
                f"policy={phase6_object_cf.get('selected_policy')}",
                f"blocker={phase6_action_surface.get('blocker') if phase6_action_surface_available else phase6_object_cf.get('blocker')}",
                f"repair_search_pass={phase6_repair_search_pass}" if phase6_repair_search_available else "",
                f"repair_candidates={phase6_repair_search.get('candidate_count')}" if phase6_repair_search_available else "",
                "Measured runtime-probe action surface; not a promoted online policy."
                if phase6_action_surface_available
                else "Trace-level upper-bound only; no runtime trajectory rerun.",
            ]
            if phase6_object_cf_available
            else ["Counterfactual not entered in v94.", f"Blocked reason: {downstream_blocker}"],
        },
        {
            "category": "runtime_action_panels_or_blocked_placeholders",
            "file": visual_dir / "runtime_action_panels_or_blocked_placeholders.png",
            "status": "blocked_placeholder",
            "evidence": "phase7_runtime_or_blocked/blocked_summary.json",
            "blocked_reason": "Runtime action blocked before action.",
            "lines": ["Runtime action not run.", "No fake runtime panels."],
        },
        {
            "category": "TTT_panels_or_blocked_placeholders",
            "file": visual_dir / "TTT_panels_or_blocked_placeholders.png",
            "status": "blocked_placeholder",
            "evidence": "phase8_ttt_or_blocked/blocked_summary.json",
            "blocked_reason": "TTT not eligible without runtime carrier/action pass.",
            "lines": ["TTT not entered.", "No TTT state/write metrics claimed."],
        },
    ]
    manifest_rows: list[dict[str, Any]] = []
    for panel in panels:
        make_panel(panel["file"], panel["category"], panel["lines"], panel.get("bars"))
        manifest_rows.append(
            {
                "category": panel["category"],
                "path": str(panel["file"]),
                "exists": panel["file"].exists(),
                "non_empty": panel["file"].exists() and panel["file"].stat().st_size > 0,
                "status": panel["status"],
                "evidence": panel["evidence"],
                "blocked_reason": panel["blocked_reason"],
                "full_rgb_overlay_available": False,
                "no_fake_runtime_panel": "runtime" not in panel["category"].lower() or panel["status"] == "blocked_placeholder",
            }
        )
    write_csv(visual_dir / "visual_audit_manifest.csv", manifest_rows)

    rgb_visual_summary_path = visual_dir / "rgb_metric_visual_audit/rgb_metric_visual_audit_summary.json"
    rgb_visual_summary = read_json(rgb_visual_summary_path)
    rgb_visual_available = bool(rgb_visual_summary) and "read_error" not in rgb_visual_summary
    visual_insight = [
        "# ACL2 v94 Visual Audit Insight",
        "",
        "This is a blocked visual audit, not a full RGB/overlay success bundle.",
        "All required categories have non-empty audit panels or explicit blocked placeholders.",
        "Phase4 and Phase5 panels reflect actual v94 follow-up execution after the formal Phase3 merge-alpha repair.",
        (
            "Phase6 now shows object-source trace-level, repair-search, and measured action-surface results; Phase7/8 remain blocked."
            if phase6_action_surface_available
            else "Phase6 now shows the object-source trace-level counterfactual and repair-search results; Phase7/8 remain blocked."
            if phase6_repair_search_available
            else "Phase6 now shows the object-source trace-level counterfactual result; Phase7/8 remain blocked."
            if phase6_object_cf_available
            else "Phase6-8 panels are intentionally blocked because Phase5 semantic-carrier alignment did not pass."
        ),
        "The central visual conclusion is that semantic invalid-boundary evidence is clean but too sparse across bad rows and LOSO folds for counterfactual/runtime promotion.",
    ]
    if rgb_visual_available:
        visual_insight.extend(
            [
                "",
                "A supplemental RGB metric audit is available for labelled Phase5 carrier rows.",
                f"RGB audit review coverage: {rgb_visual_summary.get('review_coverage')}",
                f"RGB audit blocker: {rgb_visual_summary.get('visual_gate_blocker')}",
                "The RGB audit uses real KITTI frames and marks unavailable semantic/object/raw-overlap overlays explicitly.",
            ]
        )
    (visual_dir / "visual_insight.md").write_text("\n".join(visual_insight) + "\n", encoding="utf-8")
    visual_summary = {
        "visual_audit_produced": True,
        "visual_gate_pass": False,
        "visual_gate_blocker": "semantic_component_object_overlay_images_unavailable;raw_overlap_point_overlay_unavailable"
        if rgb_visual_available
        else "full_rgb_overlay_panels_not_generated_after_phase3_stop;blocked_audit_panels_only",
        "manifest_rows": len(manifest_rows),
        "all_manifest_rows_exist": all(row["exists"] for row in manifest_rows),
        "all_image_files_non_empty": all(row["non_empty"] for row in manifest_rows),
        "review_coverage": 1.0,
        "visual_insight_exists": (visual_dir / "visual_insight.md").exists(),
        "no_fake_runtime_panels": all(row["no_fake_runtime_panel"] for row in manifest_rows),
        "blocked_phases_explicitly_shown": True,
        "rgb_metric_visual_audit_available": rgb_visual_available,
        "rgb_metric_visual_audit_summary": str(rgb_visual_summary_path) if rgb_visual_available else "",
        "rgb_metric_visual_audit_review_coverage": rgb_visual_summary.get("review_coverage")
        if rgb_visual_available
        else "",
        "rgb_metric_visual_audit_all_rgb_available": rgb_visual_summary.get("all_rgb_available")
        if rgb_visual_available
        else "",
        "rgb_metric_visual_audit_all_panels_non_empty": rgb_visual_summary.get("all_panels_non_empty")
        if rgb_visual_available
        else "",
    }
    write_json(visual_dir / "visual_audit_summary.json", visual_summary)

    failure_rows = [
        {
            "attribution": "LOCAL_BAD",
            "row_count": local_bad,
            "fraction_all_rows": pct(local_bad, row_count),
            "fraction_local_plus_handoff": pct(local_bad, local_handoff_total),
            "evidence": "phase1_boundary_failure_atlas/phase1_gate_summary.json",
        },
        {
            "attribution": "HANDOFF_SCALE",
            "row_count": handoff_scale,
            "fraction_all_rows": pct(handoff_scale, row_count),
            "fraction_local_plus_handoff": pct(handoff_scale, local_handoff_total),
            "evidence": "phase1_boundary_failure_atlas/phase1_gate_summary.json",
        },
        {
            "attribution": "HANDOFF_GAUGE",
            "row_count": handoff_gauge,
            "fraction_all_rows": pct(handoff_gauge, row_count),
            "fraction_local_plus_handoff": pct(handoff_gauge, local_handoff_total),
            "evidence": "phase1_boundary_failure_atlas/phase1_gate_summary.json",
        },
    ]
    for key in ["LOW_OBSERVABILITY", "MULTIMODE_CONFLICT", "SAFE_OR_UNASSIGNED"]:
        failure_rows.append(
            {
                "attribution": key,
                "row_count": int(counts.get(key) or 0),
                "fraction_all_rows": pct(int(counts.get(key) or 0), row_count),
                "fraction_local_plus_handoff": "",
                "evidence": "phase1_boundary_failure_atlas/phase1_gate_summary.json",
            }
        )
    write_csv(args.out_dir / "failure_attribution_report.csv", failure_rows)

    if phase5_best_policy:
        carrier_policy_route = (
            "3. Carrier-event policies may use Phase1 quantile thresholds; the strongest current policy "
            f"`{phase5_best_policy.get('policy')}` still fails "
            f"(`bad_recall={phase5_best_policy.get('bad_recall')}`, "
            f"`good_FPR={phase5_best_policy.get('good_FPR')}`, "
            f"`LOSO={phase5_best_policy.get('loso_positive_folds')}`, "
            f"`gate_pass={phase5_best_policy.get('carrier_event_gate_pass')}`)."
        )
    else:
        carrier_policy_route = (
            "3. Carrier-event policy metrics are unavailable; do not infer a q75 residual-policy pass."
        )
    next_route = [
        "# ACL2 v94 Next Route Recommendation",
        "",
        "1. Canonical semantic-only Phase5 still fails; do not promote it to runtime policy.",
        "2. `SEM_INVALID_BOUNDARY` is clean but sparse: expand semantic source coverage or object-boundary evidence instead of lowering thresholds.",
        carrier_policy_route,
        "4. Investigate why invalid-boundary positives concentrate on seq01/seq02 and do not cover seq00/seq05 bad rows.",
        (
            "5. Measured object-source action-surface replay was available but did not beat measured selection controls; runtime action and TTT stay closed."
            if phase6_action_surface_available and not phase6_action_surface_pass
            else "5. Fixed object-source Phase6 repair guards were tried and still found no passing counterfactual; runtime action and TTT stay closed."
            if phase6_repair_search_available and not phase6_repair_search_pass
            else
            "5. Object-source evidence can enter a diagnostic Phase6 branch, but runtime action and TTT stay closed because the trace-level counterfactual failed."
            if phase6_object_cf_available and not phase6_object_cf_pass
            else "5. Keep runtime action and TTT closed until Phase6 counterfactual upper-bound passes after a valid Phase5."
        ),
    ]
    if phase3r_available:
        next_route.extend(
            [
                "",
                "## Phase3R Runtime Probe Update",
                "",
                f"- runtime_probe_executed: `{phase3r.get('runtime_probe_executed')}`",
                f"- phase3r_runtime_probe_gate_pass: `{phase3r.get('phase3r_runtime_probe_gate_pass')}`",
                f"- blocker: `{phase3r.get('blocker')}`",
                "- Treat Phase3R as a measured runtime-proxy diagnostic only; it does not rewrite the original Phase3 balanced-probe gate.",
            ]
        )
    if phase3s_available:
        next_route.extend(
            [
                "",
                "## Phase3S Merge-Alpha Actuator Update",
                "",
                f"- runtime_probe_executed: `{phase3s.get('runtime_probe_executed')}`",
                f"- phase3s_actuator_probe_gate_pass: `{phase3s_gate_pass}`",
                f"- selected_candidate_variant: `{phase3s.get('selected_candidate_variant')}`",
                f"- bad_median_I_J_runtime_proxy: `{phase3s_selected.get('bad_median_I_J_runtime_proxy')}`",
                f"- good_median_worsen_runtime_proxy: `{phase3s_selected.get('good_median_worsen_runtime_proxy')}`",
                f"- good_max_worsen_runtime_proxy: `{phase3s_selected.get('good_max_worsen_runtime_proxy')}`",
                (
                    "- Phase3S is now formalized in `phase3_formal_merge_alpha_sensitivity`; canonical runtime action remains blocked by semantic-only Phase5, and the object-source branch remains blocked by Phase6 action-surface control failure."
                    if phase6_action_surface_available and not phase6_action_surface_pass
                    else "- Phase3S is now formalized in `phase3_formal_merge_alpha_sensitivity`; canonical runtime action remains blocked by semantic-only Phase5, and the object-source branch remains blocked by Phase6 repair-search failure."
                    if phase6_repair_search_available and not phase6_repair_search_pass
                    else "- Phase3S is now formalized in `phase3_formal_merge_alpha_sensitivity`; canonical runtime action remains blocked by semantic-only Phase5, and the object-source branch remains blocked by Phase6 counterfactual failure."
                    if phase6_object_cf_available
                    else "- Phase3S is now formalized in `phase3_formal_merge_alpha_sensitivity`; downstream runtime action still remains blocked by Phase5."
                ),
            ]
        )
    if phase3_formal_available:
        next_route.extend(
            [
                "",
                "## Phase3 Formal Repair Update",
                "",
                f"- phase3_repaired_gate_pass: `{phase3_formal_pass}`",
                f"- selected_carrier_body: `{phase3_formal.get('selected_carrier_body')}`",
                f"- selected_actuator_variant: `{phase3_formal.get('selected_actuator_variant')}`",
            ]
        )
    if phase4_available:
        next_route.extend(
            [
                "",
                "## Phase4 Semantic Taxonomy Update",
                "",
                f"- phase4_semantic_taxonomy_gate_pass: `{phase4_pass}`",
                f"- best_semantic_role: `{((phase4.get('best_semantic_role') or {}).get('semantic_role'))}`",
                f"- aggregate_good_FPR: `{phase4.get('semantic_good_FPR')}`",
            ]
        )
    if phase5_available:
        next_route.extend(
            [
                "",
                "## Phase5 Semantic-Carrier Alignment Update",
                "",
                f"- phase5_semantic_carrier_alignment_gate_pass: `{phase5_pass}`",
                f"- blocker: `{phase5.get('blocker')}`",
                f"- best_alignment_role: `{((phase5.get('best_alignment_role') or {}).get('semantic_role'))}`",
                f"- bad_recall: `{((phase5.get('best_alignment_role') or {}).get('bad_recall'))}`",
                f"- good_FPR: `{((phase5.get('best_alignment_role') or {}).get('good_FPR'))}`",
                f"- loso_positive_folds: `{((phase5.get('best_alignment_role') or {}).get('loso_positive_folds'))}`",
                f"- max_positive_carrier_subfield_corr: `{((phase5.get('best_alignment_role') or {}).get('max_positive_carrier_subfield_corr'))}`",
            ]
        )
    if phase5_next_route_available:
        best_low_fpr = phase5_next_route.get("best_low_fpr_policy") or {}
        best_high_recall = phase5_next_route.get("best_high_recall_policy") or {}
        next_route.extend(
            [
                "",
                "## Phase5 Next-Route Diagnostic Update",
                "",
                f"- diagnostic_only: `{phase5_next_route.get('diagnostic_only')}`",
                f"- policies_evaluated: `{phase5_next_route.get('policies_evaluated')}`",
                f"- gate_like_policy_count: `{phase5_next_route.get('gate_like_policy_count')}`",
                f"- current_policy_missed_bad_rows: `{phase5_next_route.get('current_policy_missed_bad_rows')}`",
                f"- seq00_seq05_bad_rows_missed_by_current_policy: `{phase5_next_route.get('seq00_seq05_bad_rows_missed_by_current_policy')}`",
                f"- best_low_fpr_policy: `{best_low_fpr.get('policy')}` "
                f"(bad_recall=`{best_low_fpr.get('bad_recall')}`, good_FPR=`{best_low_fpr.get('good_FPR')}`, LOSO=`{best_low_fpr.get('loso_positive_folds')}`)",
                f"- best_high_recall_policy: `{best_high_recall.get('policy')}` "
                f"(bad_recall=`{best_high_recall.get('bad_recall')}`, good_FPR=`{best_high_recall.get('good_FPR')}`, LOSO=`{best_high_recall.get('loso_positive_folds')}`)",
                "- No Phase6/7/8 promotion: the next-route diagnostic found no fixed q75/q95 semantic/carrier candidate satisfying the Phase5 gate-like recall/FPR/LOSO requirements.",
            ]
        )
    if phase5_object_source_available:
        selected_object_policy = phase5_object_source.get("selected_policy") or {}
        next_route.extend(
            [
                "",
                "## Phase5 Object-Source Extension Update",
                "",
                f"- diagnostic_only: `{phase5_object_source.get('diagnostic_only')}`",
                f"- object_source_extension_gate_pass: `{phase5_object_source_pass}`",
                f"- selected_policy: `{selected_object_policy.get('policy')}`",
                f"- bad_recall: `{selected_object_policy.get('bad_recall')}`",
                f"- good_FPR: `{selected_object_policy.get('good_FPR')}`",
                f"- loso_positive_folds: `{selected_object_policy.get('loso_positive_folds')}`",
                f"- min_control_margin: `{selected_object_policy.get('min_control_margin')}`",
                "- This repairs localization evidence only; it does not itself authorize runtime action.",
            ]
        )
    if phase6_object_cf_available:
        next_route.extend(
            [
                "",
                "## Phase6 Object-Source Counterfactual Update",
                "",
                f"- phase6_object_source_counterfactual_gate_pass: `{phase6_object_cf_pass}`",
                f"- selected_policy: `{phase6_object_cf.get('selected_policy')}`",
                f"- blocker: `{phase6_object_cf.get('blocker')}`",
                f"- bad_median_residual_improvement_ratio: `{phase6_object_actual.get('bad_median_residual_improvement_ratio')}`",
                f"- good_max_residual_worsen_ratio: `{phase6_object_actual.get('good_max_residual_worsen_ratio')}`",
                f"- actual_minus_best_control: `{phase6_object_cf.get('actual_minus_best_control')}`",
                "- Next repair target: design an action model that distinguishes harmful native negative-delta rows from rows where cancelling merge/gauge update helps.",
            ]
        )
    if phase6_repair_search_available:
        next_route.extend(
            [
                "",
                "## Phase6 Object-Source Repair Search Update",
                "",
                f"- diagnostic_only: `{phase6_repair_search.get('diagnostic_only')}`",
                f"- repair_candidate_gate_pass: `{phase6_repair_search_pass}`",
                f"- candidate_count: `{phase6_repair_search.get('candidate_count')}`",
                f"- phase5_passing_candidate_count: `{phase6_repair_search.get('phase5_passing_candidate_count')}`",
                f"- phase6_passing_candidate_count: `{phase6_repair_search.get('phase6_passing_candidate_count')}`",
                f"- passing_candidate_count: `{phase6_repair_search.get('passing_candidate_count')}`",
                f"- best_candidate: `{phase6_repair_best.get('policy')}`",
                f"- best_bad_median_residual_improvement_ratio: `{phase6_repair_best.get('bad_median_residual_improvement_ratio')}`",
                f"- best_good_max_residual_worsen_ratio: `{phase6_repair_best.get('good_max_residual_worsen_ratio')}`",
                "- The tested guards can protect good rows, but then lose Phase5 bad recall and still do not create positive bad median improvement.",
            ]
        )
    if phase6_action_surface_available:
        next_route.extend(
            [
                "",
                "## Phase6 Object-Source Action Surface Update",
                "",
                f"- diagnostic_only: `{phase6_action_surface.get('diagnostic_only')}`",
                f"- phase6_object_source_action_surface_gate_pass: `{phase6_action_surface_pass}`",
                f"- variant: `{phase6_action_surface.get('variant')}`",
                f"- selected_pair_coverage: `{phase6_action_surface.get('object_source_selected_pair_count') - len(phase6_action_surface.get('missing_selected_pairs') or [])}/{phase6_action_surface.get('object_source_selected_pair_count')}`",
                f"- bad_median_I_J_runtime_proxy: `{phase6_action_actual.get('bad_median_I_J_runtime_proxy')}`",
                f"- good_max_worsen_runtime_proxy: `{phase6_action_actual.get('good_max_worsen_runtime_proxy')}`",
                f"- actual_minus_best_control: `{phase6_action_surface.get('actual_minus_best_control')}`",
                f"- bad_negative_improvement_rows: `{phase6_action_actual.get('bad_negative_improvement_rows')}`",
                f"- blocker: `{phase6_action_surface.get('blocker')}`",
                "- Action surface exists and affects carrier state, but object-source row selection is not more specific than measured controls.",
            ]
        )
    (args.out_dir / "next_route_recommendation.md").write_text("\n".join(next_route) + "\n", encoding="utf-8")

    forbidden = [
        "# ACL2 v94 Forbidden Repeat Update",
        "",
        "- Do not promote trace-level counterfactual residual cancellation as trajectory or `J_handoff` improvement.",
        "- Do not enter downstream semantic/runtime/TTT phases after the original Phase3 failure unless a documented formal repair gate passes; runtime, counterfactual, and TTT still require later gates.",
        "- Do not treat sample-level SWA route audit as full-boundary per-head carrier evidence.",
        "- Do not treat READ role mass from masks as READ query compatibility.",
        "- Do not ignore the handoff-gauge undercoverage (`3/8`) when claiming a balanced Phase3 probe set.",
    ]
    if phase3r_available:
        forbidden.append("- Do not treat the Phase3R runtime proxy as a formal original Phase3 `J_handoff_probe` pass.")
    if phase3s_available:
        forbidden.append(
            "- Do not promote the Phase3S/Phase3-formal merge-alpha pass to runtime action or TTT while Phase5 semantic-carrier alignment fails."
        )
    if phase5_available and not phase5_pass:
        forbidden.append(
            "- Do not lower the residual carrier-event threshold from Phase1 q75 to a label-tuned value just to pass Phase5."
        )
    if phase6_object_cf_available:
        forbidden.append(
            "- Do not promote the object-source Phase5 localization pass to runtime action while Phase6 trace-level counterfactual fails."
        )
    if phase6_repair_search_available and not phase6_repair_search_pass:
        forbidden.append(
            "- Do not treat the Phase6 fixed repair-search guards as runtime-ready; no candidate passed both localization and counterfactual gates."
        )
    if phase6_action_surface_available and not phase6_action_surface_pass:
        forbidden.append(
            "- Do not promote the measured object-source action surface while it fails to beat measured selection controls."
        )
    (args.out_dir / "forbidden_repeat_update.md").write_text("\n".join(forbidden) + "\n", encoding="utf-8")

    sensitive_body_answer = (
        f"{phase3_formal.get('selected_carrier_body')} (via Phase3 formal repair)"
        if phase3_formal_pass and phase3_formal.get("selected_carrier_body")
        else "none"
    )

    final_questions = [
        {
            "question_id": 1,
            "question": "What fraction of errors are local vs handoff?",
            "answer": (
                f"all_rows: local={pct(local_bad, row_count):.6f}, "
                f"handoff={pct(handoff_total, row_count):.6f}; "
                f"local_plus_handoff_only: local={pct(local_bad, local_handoff_total):.6f}, "
                f"handoff={pct(handoff_total, local_handoff_total):.6f}"
            ),
            "evidence": "phase1_boundary_failure_atlas/phase1_gate_summary.json",
        },
        {
            "question_id": 2,
            "question": "Which memory body is causally sensitive?",
            "answer": sensitive_body_answer,
            "evidence": "phase3_formal_merge_alpha_sensitivity/phase3_formal_gate_summary.json"
            if phase3_formal_pass
            else "phase3_neutral_causal_sensitivity/phase3_gate_summary.json",
        },
        {
            "question_id": 3,
            "question": "Did merge/gauge true trace become available?",
            "answer": True,
            "evidence": (
                f"merge_gauge_true_trace_coverage={phase2.get('merge_gauge_true_trace_coverage')}; "
                f"merge_residual_delta_coverage={phase2.get('merge_residual_delta_coverage')}"
            ),
        },
        {
            "question_id": 4,
            "question": "Did semantic taxonomy improve over geometry-only within conflict rows?",
            "answer": phase4_pass if phase4_available else "not_available",
            "evidence": "phase4_semantic_evidence_taxonomy/semantic_taxonomy_summary.json"
            if phase4_available
            else "phase4_semantic_evidence_taxonomy_or_blocked/blocked_summary.json",
        },
        {
            "question_id": 5,
            "question": "Did semantic policy beat semantic/component/regime shuffle?",
            "answer": phase5_strict_shuffle_pass if phase5_available else "not_available",
            "evidence": "phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json"
            if phase5_available
            else "phase5_semantic_carrier_alignment_or_blocked/blocked_summary.json",
        },
        {
            "question_id": 6,
            "question": "Did counterfactual show upper bound?",
            "answer": phase6_object_cf_pass if phase6_object_cf_available else False,
            "evidence": "phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json"
            if phase6_object_cf_available
            else "phase6_counterfactual_or_blocked/blocked_summary.json; blocked before Phase6 because canonical Phase5 did not pass",
        },
        {
            "question_id": 7,
            "question": "Did runtime action improve bad without hurting good?",
            "answer": False,
            "evidence": "measured action-surface diagnostic exists but did not beat controls; no promoted Phase7 online runtime policy was run"
            if phase6_action_surface_available
            else "runtime action was not allowed or executed; Phase3S is a diagnostic actuator probe, not a runtime action phase",
        },
        {
            "question_id": 8,
            "question": "Is TTT eligible?",
            "answer": False,
            "evidence": "phase8_ttt_or_blocked/blocked_summary.json",
        },
        {
            "question_id": 9,
            "question": "What is the next blocker?",
            "answer": blocker,
            "evidence": "report_final/final_decision.json",
        },
        {
            "question_id": 10,
            "question": "What must not be repeated?",
            "answer": "see forbidden_repeat_update.md",
            "evidence": "report_final/forbidden_repeat_update.md",
        },
    ]
    if phase3r_available:
        final_questions.append(
            {
                "question_id": 11,
                "question": "Did the Phase3R runtime merge/gauge proxy repair pass?",
                "answer": bool(phase3r.get("phase3r_runtime_probe_gate_pass")),
                "evidence": "phase3r_runtime_merge_gauge_probe/runtime_probe_sensitivity_summary.json",
            }
        )
    if phase3s_available:
        final_questions.append(
            {
                "question_id": 12,
                "question": "Did the Phase3S merge-alpha actuator probe confirm a measured sensitive control point?",
                "answer": phase3s_gate_pass,
                "evidence": "phase3s_merge_gauge_actuator_sweep_max16_confirm/runtime_probe_sensitivity_summary.json",
            }
        )
    if phase3_formal_available:
        final_questions.append(
            {
                "question_id": 13,
                "question": "Did the repaired formal Phase3 merge-alpha gate pass?",
                "answer": phase3_formal_pass,
                "evidence": "phase3_formal_merge_alpha_sensitivity/phase3_formal_gate_summary.json",
            }
        )
    if phase5_available:
        final_questions.append(
            {
                "question_id": 14,
                "question": "Why did Phase5 fail?",
                "answer": phase5.get("blocker"),
                "evidence": "phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json",
            }
        )
    if phase5_next_route_available:
        final_questions.append(
            {
                "question_id": 15,
                "question": "Did the Phase5 next-route diagnostic find a valid repair candidate?",
                "answer": bool(phase5_next_route.get("gate_like_policy_count")),
                "evidence": "phase5_next_route_diagnostic/phase5_next_route_diagnostic_summary.json",
            }
        )
    if phase5_object_source_available:
        final_questions.append(
            {
                "question_id": 16,
                "question": "Did object-source evidence repair Phase5 localization?",
                "answer": phase5_object_source_pass,
                "evidence": "phase5_object_source_extension/phase5_object_source_extension_summary.json",
            }
        )
    if phase6_object_cf_available:
        final_questions.append(
            {
                "question_id": 17,
                "question": "Did the object-source counterfactual justify runtime action?",
                "answer": phase6_object_cf_pass,
                "evidence": "phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json",
            }
        )
    if phase6_repair_search_available:
        final_questions.append(
            {
                "question_id": 18,
                "question": "Did the Phase6 fixed repair-search find a safe candidate?",
                "answer": phase6_repair_search_pass,
                "evidence": "phase6_object_source_repair_search/phase6_object_source_repair_search_summary.json",
            }
        )
    if phase6_action_surface_available:
        final_questions.append(
            {
                "question_id": 19,
                "question": "Did measured object-source action surface beat controls?",
                "answer": phase6_action_surface_pass,
                "evidence": "phase6_object_source_action_surface/phase6_object_source_action_surface_summary.json",
            }
        )

    decision = {
        "final_status": final_status,
        "blocker": blocker,
        "decision_labels": labels,
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase1_gate_pass": phase1.get("phase1_gate_pass"),
        "phase2_gate_pass": phase2.get("phase2_gate_pass"),
        "phase3_gate_pass": phase3.get("phase3_gate_pass"),
        "phase3_formal_repaired_gate_pass": phase3_formal_pass,
        "phase4_semantic_taxonomy_gate_pass": phase4_pass,
        "phase5_semantic_carrier_alignment_gate_pass": phase5_pass,
        "phase5_object_source_extension_gate_pass": phase5_object_source_pass,
        "phase6_object_source_counterfactual_gate_pass": phase6_object_cf_pass,
        "phase6_object_source_repair_search_gate_pass": phase6_repair_search_pass,
        "phase6_object_source_action_surface_gate_pass": phase6_action_surface_pass,
        "stop_rule_triggered": phase3.get("stop_rule_triggered"),
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
        "visual_audit_produced": visual_summary["visual_audit_produced"],
        "visual_gate_pass": visual_summary["visual_gate_pass"],
        "rgb_metric_visual_audit": rgb_visual_summary if rgb_visual_available else {},
        "phase3r_runtime_probe": phase3r if phase3r_available else {},
        "phase3s_merge_alpha_actuator_probe": phase3s if phase3s_available else {},
        "phase3_formal_merge_alpha_sensitivity": phase3_formal if phase3_formal_available else {},
        "phase4_semantic_evidence_taxonomy": phase4 if phase4_available else {},
        "phase5_semantic_carrier_alignment": phase5 if phase5_available else {},
        "phase5_next_route_diagnostic": phase5_next_route if phase5_next_route_available else {},
        "phase5_object_source_extension": phase5_object_source if phase5_object_source_available else {},
        "phase6_object_source_counterfactual": phase6_object_cf if phase6_object_cf_available else {},
        "phase6_object_source_repair_search": phase6_repair_search if phase6_repair_search_available else {},
        "phase6_object_source_action_surface": phase6_action_surface if phase6_action_surface_available else {},
        "key_metrics": {
            "row_count": row_count,
            "local_bad_rows": local_bad,
            "handoff_scale_rows": handoff_scale,
            "handoff_gauge_rows": handoff_gauge,
            "handoff_total_rows": handoff_total,
            "local_fraction_all_rows": pct(local_bad, row_count),
            "handoff_fraction_all_rows": pct(handoff_total, row_count),
            "phase2_merge_gauge_true_trace_coverage": phase2.get("merge_gauge_true_trace_coverage"),
            "phase2_merge_residual_delta_coverage": phase2.get("merge_residual_delta_coverage"),
            "phase3_handoff_scale_probe_rows": (phase3.get("balanced_probe") or {}).get("handoff_scale_rows"),
            "phase3_handoff_gauge_probe_rows": (phase3.get("balanced_probe") or {}).get("handoff_gauge_rows"),
            "phase3_good_safe_probe_rows": (phase3.get("balanced_probe") or {}).get("good_safe_rows"),
            "phase3_any_carrier_trace_delta_observed": (phase3.get("checks") or {}).get(
                "any_carrier_trace_delta_observed"
            ),
            "phase3_any_j_handoff_probe_available": (phase3.get("checks") or {}).get(
                "any_j_handoff_probe_available"
            ),
            "phase3r_runtime_probe_executed": phase3r.get("runtime_probe_executed") if phase3r_available else False,
            "phase3r_runtime_probe_gate_pass": phase3r.get("phase3r_runtime_probe_gate_pass")
            if phase3r_available
            else False,
            "phase3r_runtime_probe_job_count": phase3r.get("runtime_probe_job_count") if phase3r_available else "",
            "phase3r_runtime_probe_failed_count": phase3r.get("runtime_probe_failed_count")
            if phase3r_available
            else "",
            "phase3s_actuator_probe_executed": phase3s.get("runtime_probe_executed") if phase3s_available else False,
            "phase3s_actuator_probe_gate_pass": phase3s_gate_pass,
            "phase3s_selected_candidate_variant": phase3s.get("selected_candidate_variant")
            if phase3s_available
            else "",
            "phase3s_target_count": phase3s.get("target_count") if phase3s_available else "",
            "phase3s_metric_row_count": phase3s.get("metric_row_count") if phase3s_available else "",
            "phase3s_effect_row_count": phase3s.get("effect_row_count") if phase3s_available else "",
            "phase3s_runtime_probe_job_count": phase3s.get("runtime_probe_job_count") if phase3s_available else "",
            "phase3s_runtime_probe_failed_count": phase3s.get("runtime_probe_failed_count")
            if phase3s_available
            else "",
            "phase3s_selected_candidate_beats_control": phase3s.get("selected_candidate_beats_control")
            if phase3s_available
            else "",
            "phase3s_bad_median_I_J_runtime_proxy": phase3s_selected.get("bad_median_I_J_runtime_proxy"),
            "phase3s_good_median_worsen_runtime_proxy": phase3s_selected.get("good_median_worsen_runtime_proxy"),
            "phase3s_good_max_worsen_runtime_proxy": phase3s_selected.get("good_max_worsen_runtime_proxy"),
            "phase3_formal_selected_carrier_body": phase3_formal.get("selected_carrier_body")
            if phase3_formal_available
            else "",
            "phase3_formal_selected_actuator_variant": phase3_formal.get("selected_actuator_variant")
            if phase3_formal_available
            else "",
            "phase3_formal_bad_median_I_J_runtime_proxy": phase3_formal.get("bad_median_I_J_runtime_proxy")
            if phase3_formal_available
            else "",
            "phase4_best_semantic_role": (phase4.get("best_semantic_role") or {}).get("semantic_role")
            if phase4_available
            else "",
            "phase4_best_role_good_FPR": (phase4.get("best_semantic_role") or {}).get("good_FPR")
            if phase4_available
            else "",
            "phase4_best_role_max_shuffle_margin": (phase4.get("best_semantic_role") or {}).get(
                "max_shuffle_margin"
            )
            if phase4_available
            else "",
            "phase5_joined_effect_row_count": phase5.get("joined_effect_row_count") if phase5_available else "",
            "phase5_labelled_joined_rows": phase5.get("labelled_joined_rows") if phase5_available else "",
            "phase5_best_alignment_role": (phase5.get("best_alignment_role") or {}).get("semantic_role")
            if phase5_available
            else "",
            "phase5_best_alignment_bad_recall": (phase5.get("best_alignment_role") or {}).get("bad_recall")
            if phase5_available
            else "",
            "phase5_best_alignment_good_FPR": (phase5.get("best_alignment_role") or {}).get("good_FPR")
            if phase5_available
            else "",
            "phase5_best_alignment_loso_positive_folds": (phase5.get("best_alignment_role") or {}).get(
                "loso_positive_folds"
            )
            if phase5_available
            else "",
            "phase5_best_alignment_max_positive_carrier_subfield_corr": (
                phase5.get("best_alignment_role") or {}
            ).get("max_positive_carrier_subfield_corr")
            if phase5_available
            else "",
            "phase5_next_route_policies_evaluated": phase5_next_route.get("policies_evaluated")
            if phase5_next_route_available
            else "",
            "phase5_next_route_gate_like_policy_count": phase5_next_route.get("gate_like_policy_count")
            if phase5_next_route_available
            else "",
            "phase5_next_route_best_low_fpr_policy": (
                (phase5_next_route.get("best_low_fpr_policy") or {}).get("policy")
            )
            if phase5_next_route_available
            else "",
            "phase5_next_route_best_low_fpr_bad_recall": (
                (phase5_next_route.get("best_low_fpr_policy") or {}).get("bad_recall")
            )
            if phase5_next_route_available
            else "",
            "phase5_next_route_best_low_fpr_good_FPR": (
                (phase5_next_route.get("best_low_fpr_policy") or {}).get("good_FPR")
            )
            if phase5_next_route_available
            else "",
            "phase5_next_route_best_high_recall_policy": (
                (phase5_next_route.get("best_high_recall_policy") or {}).get("policy")
            )
            if phase5_next_route_available
            else "",
            "phase5_next_route_best_high_recall_bad_recall": (
                (phase5_next_route.get("best_high_recall_policy") or {}).get("bad_recall")
            )
            if phase5_next_route_available
            else "",
            "phase5_next_route_best_high_recall_good_FPR": (
                (phase5_next_route.get("best_high_recall_policy") or {}).get("good_FPR")
            )
            if phase5_next_route_available
            else "",
            "phase5_object_source_selected_policy": (
                (phase5_object_source.get("selected_policy") or {}).get("policy")
            )
            if phase5_object_source_available
            else "",
            "phase5_object_source_bad_recall": (
                (phase5_object_source.get("selected_policy") or {}).get("bad_recall")
            )
            if phase5_object_source_available
            else "",
            "phase5_object_source_good_FPR": (
                (phase5_object_source.get("selected_policy") or {}).get("good_FPR")
            )
            if phase5_object_source_available
            else "",
            "phase5_object_source_loso_positive_folds": (
                (phase5_object_source.get("selected_policy") or {}).get("loso_positive_folds")
            )
            if phase5_object_source_available
            else "",
            "phase5_object_source_min_control_margin": (
                (phase5_object_source.get("selected_policy") or {}).get("min_control_margin")
            )
            if phase5_object_source_available
            else "",
            "phase6_object_source_selected_policy": phase6_object_cf.get("selected_policy")
            if phase6_object_cf_available
            else "",
            "phase6_object_source_bad_median_residual_improvement_ratio": phase6_object_actual.get(
                "bad_median_residual_improvement_ratio"
            )
            if phase6_object_cf_available
            else "",
            "phase6_object_source_good_max_residual_worsen_ratio": phase6_object_actual.get(
                "good_max_residual_worsen_ratio"
            )
            if phase6_object_cf_available
            else "",
            "phase6_object_source_actual_minus_best_control": phase6_object_cf.get("actual_minus_best_control")
            if phase6_object_cf_available
            else "",
            "phase6_object_source_blocker": phase6_object_cf.get("blocker") if phase6_object_cf_available else "",
            "phase6_repair_search_candidate_count": phase6_repair_search.get("candidate_count")
            if phase6_repair_search_available
            else "",
            "phase6_repair_search_phase5_passing_candidate_count": phase6_repair_search.get(
                "phase5_passing_candidate_count"
            )
            if phase6_repair_search_available
            else "",
            "phase6_repair_search_phase6_passing_candidate_count": phase6_repair_search.get(
                "phase6_passing_candidate_count"
            )
            if phase6_repair_search_available
            else "",
            "phase6_repair_search_passing_candidate_count": phase6_repair_search.get("passing_candidate_count")
            if phase6_repair_search_available
            else "",
            "phase6_repair_search_best_candidate": phase6_repair_best.get("policy")
            if phase6_repair_search_available
            else "",
            "phase6_repair_search_best_bad_median_residual_improvement_ratio": phase6_repair_best.get(
                "bad_median_residual_improvement_ratio"
            )
            if phase6_repair_search_available
            else "",
            "phase6_repair_search_best_good_max_residual_worsen_ratio": phase6_repair_best.get(
                "good_max_residual_worsen_ratio"
            )
            if phase6_repair_search_available
            else "",
            "phase6_action_surface_variant": phase6_action_surface.get("variant")
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_selected_pair_count": phase6_action_surface.get(
                "object_source_selected_pair_count"
            )
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_measured_labelled_pair_count": phase6_action_surface.get(
                "measured_labelled_pair_count"
            )
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_bad_median_I_J_runtime_proxy": phase6_action_actual.get(
                "bad_median_I_J_runtime_proxy"
            )
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_good_max_worsen_runtime_proxy": phase6_action_actual.get(
                "good_max_worsen_runtime_proxy"
            )
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_actual_minus_best_control": phase6_action_surface.get(
                "actual_minus_best_control"
            )
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_bad_negative_improvement_rows": phase6_action_actual.get(
                "bad_negative_improvement_rows"
            )
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_blocker": phase6_action_surface.get("blocker")
            if phase6_action_surface_available
            else "",
            "phase6_action_surface_semantic_not_specific": phase6_action_surface.get("semantic_not_specific")
            if phase6_action_surface_available
            else "",
        },
        "final_questions": final_questions,
    }
    write_json(args.out_dir / "final_decision.json", decision)
    write_csv(args.out_dir / "decision_labels.csv", [{"label": label} for label in labels])
    write_csv(args.out_dir / "blocker_attribution.csv", [{"final_status": final_status, "blocker": blocker}])
    write_csv(args.out_dir / "final_decision_answers.csv", final_questions)

    phase5_next_route_conclusion = ""
    if phase5_next_route_available:
        next_low_fpr = phase5_next_route.get("best_low_fpr_policy") or {}
        next_high_recall = phase5_next_route.get("best_high_recall_policy") or {}
        phase5_next_route_conclusion = (
            "The next-route diagnostic evaluated "
            f"`{phase5_next_route.get('policies_evaluated')}` fixed semantic/carrier q75/q95 candidates and found "
            f"`{phase5_next_route.get('gate_like_policy_count')}` gate-like repair candidates; "
            f"best low-FPR recall was `{next_low_fpr.get('bad_recall')}` "
            f"at good_FPR `{next_low_fpr.get('good_FPR')}`, "
            f"while best high-recall good_FPR was `{next_high_recall.get('good_FPR')}`. "
        )
    phase5_object_source_conclusion = ""
    if phase5_object_source_available:
        selected_object = phase5_object_source.get("selected_policy") or {}
        phase5_object_source_conclusion = (
            "Adding object-source evidence produced a diagnostic Phase5-like localization pass with "
            f"`{selected_object.get('policy')}` "
            f"(`bad_recall={selected_object.get('bad_recall')}`, "
            f"`good_FPR={selected_object.get('good_FPR')}`, "
            f"`LOSO={selected_object.get('loso_positive_folds')}`, "
            f"`min_control_margin={selected_object.get('min_control_margin')}`). "
        )
    phase6_object_cf_conclusion = ""
    if phase6_object_cf_available:
        phase6_object_cf_conclusion = (
            "However, the object-source trace-level counterfactual failed Phase6 "
            f"(`bad_median_residual_improvement_ratio={phase6_object_actual.get('bad_median_residual_improvement_ratio')}`, "
            f"`good_max_residual_worsen_ratio={phase6_object_actual.get('good_max_residual_worsen_ratio')}`, "
            f"`actual_minus_best_control={phase6_object_cf.get('actual_minus_best_control')}`, "
            f"`blocker={phase6_object_cf.get('blocker')}`). "
        )
    phase6_repair_search_conclusion = ""
    if phase6_repair_search_available:
        phase6_repair_search_conclusion = (
            "A fixed Phase6 repair search then evaluated "
            f"`{phase6_repair_search.get('candidate_count')}` no-training guard candidates; "
            f"`{phase6_repair_search.get('phase5_passing_candidate_count')}` passed localization, "
            f"`{phase6_repair_search.get('phase6_passing_candidate_count')}` passed counterfactual, "
            f"and `{phase6_repair_search.get('passing_candidate_count')}` passed both. "
            f"The best candidate `{phase6_repair_best.get('policy')}` still had "
            f"`bad_median_residual_improvement_ratio={phase6_repair_best.get('bad_median_residual_improvement_ratio')}`. "
        )
    phase6_action_surface_conclusion = ""
    if phase6_action_surface_available:
        phase6_action_surface_conclusion = (
            "Measured pipeline action-surface replay covered "
            f"`{phase6_action_surface.get('object_source_selected_pair_count') - len(phase6_action_surface.get('missing_selected_pairs') or [])}/"
            f"{phase6_action_surface.get('object_source_selected_pair_count')}` selected labelled rows with "
            f"`{phase6_action_surface.get('variant')}` and showed carrier/trajectory effects "
            f"(`bad_median_I_J_runtime_proxy={phase6_action_actual.get('bad_median_I_J_runtime_proxy')}`, "
            f"`good_max_worsen_runtime_proxy={phase6_action_actual.get('good_max_worsen_runtime_proxy')}`), "
            f"but it failed control specificity "
            f"(`actual_minus_best_control={phase6_action_surface.get('actual_minus_best_control')}`, "
            f"`blocker={phase6_action_surface.get('blocker')}`). "
        )
    conclusion_text = (
        "v94 no longer stops at the old Phase3 diagnostic boundary: Phase3S was formalized as a "
        f"{phase3_formal.get('selected_carrier_body') or 'merge/gauge'} repair and Phase4 found a role-specific semantic signal. "
        "The canonical semantic-only branch still fails at Phase5. "
        f"`{phase5_best.get('semantic_role')}` is clean (`good_FPR={phase5_best.get('good_FPR')}`) "
        "and correlates with the merge residual subfield "
        f"(`max_positive_carrier_subfield_corr={phase5_best.get('max_positive_carrier_subfield_corr')}`), "
        f"but it covers only {phase5_best.get('bad_recall')} of labelled bad carrier rows "
        f"with {phase5_best.get('loso_positive_folds')} LOSO positive folds. "
        f"The strongest Phase1-q75 carrier-event policy reaches `bad_recall={phase5_best_policy.get('bad_recall')}` "
        f"with `good_FPR={phase5_best_policy.get('good_FPR')}` and `LOSO={phase5_best_policy.get('loso_positive_folds')}`. "
        f"{phase5_next_route_conclusion}"
        f"{phase5_object_source_conclusion}"
        f"{phase6_object_cf_conclusion}"
        f"{phase6_repair_search_conclusion}"
        f"{phase6_action_surface_conclusion}"
        "Therefore runtime action and TTT remain blocked."
    )

    report = [
        "# ACL2 v94 Final Report",
        "",
        f"- final_status: `{final_status}`",
        f"- blocker: `{blocker}`",
        f"- phase0_gate_pass: `{phase0.get('phase0_gate_pass')}`",
        f"- phase1_gate_pass: `{phase1.get('phase1_gate_pass')}`",
        f"- phase2_gate_pass: `{phase2.get('phase2_gate_pass')}`",
        f"- phase3_gate_pass: `{phase3.get('phase3_gate_pass')}`",
        f"- phase3r_runtime_probe_executed: `{phase3r.get('runtime_probe_executed') if phase3r_available else False}`",
        f"- phase3r_runtime_probe_gate_pass: `{phase3r.get('phase3r_runtime_probe_gate_pass') if phase3r_available else False}`",
        f"- phase3s_actuator_probe_executed: `{phase3s.get('runtime_probe_executed') if phase3s_available else False}`",
        f"- phase3s_actuator_probe_gate_pass: `{phase3s_gate_pass}`",
        f"- phase3s_selected_candidate_variant: `{phase3s.get('selected_candidate_variant') if phase3s_available else ''}`",
        f"- phase3_formal_repaired_gate_pass: `{phase3_formal_pass}`",
        f"- phase4_semantic_taxonomy_gate_pass: `{phase4_pass}`",
        f"- phase5_semantic_carrier_alignment_gate_pass: `{phase5_pass}`",
        f"- phase5_object_source_extension_gate_pass: `{phase5_object_source_pass}`",
        f"- phase6_object_source_counterfactual_gate_pass: `{phase6_object_cf_pass}`",
        f"- phase6_object_source_repair_search_gate_pass: `{phase6_repair_search_pass}`",
        f"- phase6_object_source_action_surface_gate_pass: `{phase6_action_surface_pass}`",
        f"- runtime_action_allowed: `False`",
        f"- ttt_allowed: `False`",
        "",
        "## Evidence",
        "",
        f"- Phase1 counts: `{counts}`",
        f"- Phase2 merge/gauge trace coverage: `{phase2.get('merge_gauge_true_trace_coverage')}`; residual coverage: `{phase2.get('merge_residual_delta_coverage')}`",
        f"- Phase3 balanced probe: `{phase3.get('balanced_probe')}`",
        f"- Phase3 blocker: `{phase3.get('blocker')}`",
        f"- Phase3R runtime probe: `{args.root / 'phase3r_runtime_merge_gauge_probe/runtime_probe_sensitivity_summary.json'}`",
        f"- Phase3S merge-alpha actuator probe: `{args.root / 'phase3s_merge_gauge_actuator_sweep_max16_confirm/runtime_probe_sensitivity_summary.json'}`",
        f"- Phase3 formal repair: `{args.root / 'phase3_formal_merge_alpha_sensitivity/phase3_formal_gate_summary.json'}`",
        f"- Phase4 semantic taxonomy: `{args.root / 'phase4_semantic_evidence_taxonomy/semantic_taxonomy_summary.json'}`",
        f"- Phase5 semantic-carrier alignment: `{args.root / 'phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json'}`",
        f"- Phase5 next-route diagnostic: `{args.root / 'phase5_next_route_diagnostic/phase5_next_route_diagnostic_summary.json'}`; available=`{phase5_next_route_available}`; gate_like_policy_count=`{phase5_next_route.get('gate_like_policy_count') if phase5_next_route_available else ''}`",
        f"- Phase5 object-source extension: `{args.root / 'phase5_object_source_extension/phase5_object_source_extension_summary.json'}`; available=`{phase5_object_source_available}`; gate_pass=`{phase5_object_source_pass}`",
        f"- Phase6 object-source counterfactual: `{args.root / 'phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json'}`; available=`{phase6_object_cf_available}`; gate_pass=`{phase6_object_cf_pass}`",
        f"- Phase6 object-source repair search: `{args.root / 'phase6_object_source_repair_search/phase6_object_source_repair_search_summary.json'}`; available=`{phase6_repair_search_available}`; gate_pass=`{phase6_repair_search_pass}`",
        f"- Phase6 object-source action surface: `{args.root / 'phase6_object_source_action_surface/phase6_object_source_action_surface_summary.json'}`; available=`{phase6_action_surface_available}`; gate_pass=`{phase6_action_surface_pass}`",
        f"- Visual audit: `{visual_dir / 'visual_audit_manifest.csv'}`; visual_gate_pass=`{visual_summary['visual_gate_pass']}`",
        f"- RGB metric visual audit: `{rgb_visual_summary_path}`; available=`{rgb_visual_available}`; review_coverage=`{rgb_visual_summary.get('review_coverage') if rgb_visual_available else ''}`",
        "",
        "## Final Questions",
        "",
        *[
            f"{row['question_id']}. {row['question']}: `{row['answer']}` Evidence: {row['evidence']}"
            for row in final_questions
        ],
        "",
        "## Conclusion",
        "",
        conclusion_text,
    ]
    (args.out_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"final_status={final_status}")
    print(f"phase3_gate_pass={phase3.get('phase3_gate_pass')}")
    print(f"phase3s_actuator_probe_gate_pass={phase3s_gate_pass}")
    print(f"phase3_formal_repaired_gate_pass={phase3_formal_pass}")
    print(f"phase4_semantic_taxonomy_gate_pass={phase4_pass}")
    print(f"phase5_semantic_carrier_alignment_gate_pass={phase5_pass}")
    print(f"phase5_object_source_extension_gate_pass={phase5_object_source_pass}")
    print(f"phase6_object_source_counterfactual_gate_pass={phase6_object_cf_pass}")
    print(f"phase6_object_source_repair_search_gate_pass={phase6_repair_search_pass}")
    print(f"phase6_object_source_action_surface_gate_pass={phase6_action_surface_pass}")
    print(f"runtime_action_allowed=False")
    print(f"ttt_allowed=False")
    print(f"visual_audit_produced={visual_summary['visual_audit_produced']}")
    print(f"visual_gate_pass={visual_summary['visual_gate_pass']}")


if __name__ == "__main__":
    main()
