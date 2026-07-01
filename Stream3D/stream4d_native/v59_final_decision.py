from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_json


DEFAULT_PHASE0 = "outputs/audit/v59_phase0_fact_lock/fact_lock.json"
DEFAULT_PHASE1 = "outputs/audit/v59_phase1_graph/graph_summary.json"
DEFAULT_PHASE2 = "outputs/audit/v59_phase2_paths_repair_margin070_noexcl_semcat/path_summary.json"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v59_final_decision(
    phase0_path: str | Path = DEFAULT_PHASE0,
    phase1_path: str | Path = DEFAULT_PHASE1,
    phase2_path: str | Path = DEFAULT_PHASE2,
) -> dict[str, Any]:
    phase0 = read_json(_project(phase0_path))
    phase1 = read_json(_project(phase1_path))
    phase2 = read_json(_project(phase2_path))
    phase0_pass = bool((phase0.get("gate") or {}).get("pass"))
    phase1_pass = bool((phase1.get("gate") or {}).get("pass"))
    phase2_gate = phase2.get("gate") or {}
    phase2_pass = bool(phase2_gate.get("pass"))
    same_category_blocked = bool(phase2_gate.get("same_category_false_path_rate_metric_available")) and not bool(
        phase2_gate.get("same_category_false_path_rate_le_semantic_pairwise_baseline_minus_0_05")
    )
    shortcut_signal_pass = bool(
        phase2_gate.get("path_precision_diagnostic_ge_0_80")
        and phase2_gate.get("part_to_core_path_precision_ge_0_80")
        and phase2_gate.get("shortcut_quarantine_precision_ge_0_75")
    )
    if phase0_pass and phase1_pass and shortcut_signal_pass and same_category_blocked:
        final_label = "NO_GO_PHASE2_SAME_CATEGORY_GATE"
        partial_label = "PARTIAL_TYPED_GRAPH_PATH_SHORTCUT_SIGNAL"
    elif phase0_pass and phase1_pass:
        final_label = "NO_GO_PHASE2_PATH_GATE"
        partial_label = "PARTIAL_TYPED_GRAPH_SIGNAL"
    else:
        final_label = "NO_GO_PHASE0_OR_GRAPH_GATE"
        partial_label = "PARTIAL_FACT_LOCK_ONLY" if phase0_pass else "NO_PARTIAL_SIGNAL"
    return {
        "phase": "v59_final_decision",
        "created_at": utc_now(),
        "goal_achieved": False,
        "final_label": final_label,
        "partial_label": partial_label,
        "phase0_gate_pass": phase0_pass,
        "phase1_gate_pass": phase1_pass,
        "phase2_gate_pass": phase2_pass,
        "phase2_stop_rule_triggered": not phase2_pass,
        "stop_reason": (
            "Phase2 same-category gate failed after planned shortcut repairs; full embedding/query/stress/native method claims are blocked."
            if same_category_blocked
            else "Phase2 path gate failed; full embedding/query/stress/native method claims are blocked."
        ),
        "key_metrics": {
            "phase0_v58_dino_recall@3": phase0.get("v58_phase1_dino_recall@3"),
            "phase0_v58_deferred_count": phase0.get("v58_phase2_deferred_count"),
            "phase1_history_manifold_count": phase1.get("history_manifold_count"),
            "phase1_underseg_bridge_edge_count": phase1.get("underseg_bridge_edge_count"),
            "phase2_accepted_path_count": phase2.get("accepted_path_count"),
            "phase2_path_precision_diagnostic": phase2.get("path_precision_diagnostic"),
            "phase2_part_to_core_path_precision": phase2.get("part_to_core_path_precision"),
            "phase2_shortcut_quarantine_precision": phase2.get("shortcut_quarantine_precision"),
            "phase2_same_category_false_path_rate": phase2.get("same_category_false_path_rate"),
            "phase2_same_category_baseline_false_path_rate": phase2.get("same_category_baseline_false_path_rate"),
            "phase2_same_category_required_max_rate": phase2.get("same_category_required_max_rate"),
        },
        "blocked_downstream_phases": [
            "Phase3 manifold-constrained embedding",
            "Phase4 manifold refinement and promotion",
            "Phase5 manifold-aware active query",
            "Phase6 stress/dynamic evaluation",
            "Phase7 native field method output",
            "Phase8 GO_SOMA_MANIFOLD final claim",
        ],
        "input_paths": {
            "phase0": _rel(phase0_path),
            "phase1": _rel(phase1_path),
            "phase2": _rel(phase2_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v59_final_decision(decision: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "final_decision.json"
    write_json(path, decision)
    return {"final_decision": _rel(path)}
