#!/usr/bin/env python3
"""Build ACL2 v84 Phase12 decision matrix and final report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ROOT = Path("results/acl2_v84tf_memory_ruler_audit")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "phase12_decision_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def main() -> None:
    args = parse_args()
    root = args.root
    phase0 = read_json(root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = read_json(root / "phase1_ruler_candidate_universe/phase1_gate_summary.json")
    phase2 = read_json(root / "phase2_memory_ruler_ledger/phase2_ledger_summary.json")
    phase10_plan = read_json(root / "phase10_support_expansion/support_expansion_plan.json")
    phase10 = read_json(root / "phase10_support_expansion_audit/support_expansion_audit_summary.json")
    phase11 = read_json(root / "phase11_memory_ruler_rediscovery/visual_integrity_audit.json")

    labels = ["D1_RULER_CLUES_INSUFFICIENT", "D10_TTT_NOT_READY"]
    supporting_blockers = []
    if phase10.get("semantic_shuffle_specificity_available") is False:
        supporting_blockers.append("semantic_shuffle_and_same_mass_controls_unavailable")
    if phase10.get("labelled_bad_anchor_recall_observed", 0.0) < 0.60:
        supporting_blockers.append("labelled_bad_anchor_recall_too_low")
    if phase2.get("score_available_high_quality_ratio", 0.0) < 0.80:
        supporting_blockers.append("phase2_score_availability_too_low")

    final_decision = {
        "schema": "acl2_v84_phase12_final_decision_v1",
        "final_status": "No-Go_before_runtime_action",
        "decision_labels": labels,
        "supporting_blockers": supporting_blockers,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "phase_gate_chain": {
            "phase0_evidence_lock_pass": bool(phase0.get("phase0_gate_pass")),
            "phase1_main_candidate_gate_pass": bool(phase1.get("phase1_gate_pass")),
            "phase2_main_ledger_gate_pass": bool(phase2.get("phase2_gate_pass")),
            "phase10_rows_2x_pre_candidate_pass": bool(phase10_plan.get("pre_candidate_rows_2x_gate_pass")),
            "phase10_expansion_useful_gate_pass": bool(phase10.get("expansion_useful_gate_pass")),
            "phase11_visual_integrity_pass": bool(phase11.get("visual_integrity_pass")),
            "phase3_sufficiency_ran": False,
            "phase4_carrier_localization_ran": False,
            "phase6_counterfactual_ran": False,
        },
        "key_metrics": {
            "phase1_main_token_rows": phase1.get("token_rows"),
            "phase1_main_adjacent_pair_rows": phase1.get("adjacent_pair_rows"),
            "phase2_score_available_high_quality_ratio": phase2.get("score_available_high_quality_ratio"),
            "phase2_ruler_anchor_pair_count": phase2.get("ruler_anchor_pair_count"),
            "phase2_ruler_anchor_token_count": phase2.get("ruler_anchor_token_count"),
            "phase10_expanded_pair_rows": phase10.get("expanded_pair_rows"),
            "phase10_rows_increase_ratio": phase10.get("rows_increase_ratio"),
            "phase10_anchor_support_pair_count": phase10.get("anchor_support_pair_count"),
            "phase10_default_high_quality_support_anchor_pair_count": phase10.get(
                "default_high_quality_support_anchor_pair_count"
            ),
            "phase10_labelled_bad_anchor_recall_observed": phase10.get("labelled_bad_anchor_recall_observed"),
            "phase10_labelled_good_anchor_fpr_observed": phase10.get("labelled_good_anchor_fpr_observed"),
            "phase11_visual_manifest_rows": phase11.get("manifest_rows"),
        },
        "evidence_paths": {
            "phase0": str(root / "phase0_evidence_lock/phase0_gate_summary.json"),
            "phase1": str(root / "phase1_ruler_candidate_universe/phase1_gate_summary.json"),
            "phase2": str(root / "phase2_memory_ruler_ledger/phase2_ledger_summary.json"),
            "phase10_plan": str(root / "phase10_support_expansion/support_expansion_plan.json"),
            "phase10_audit": str(root / "phase10_support_expansion_audit/support_expansion_audit_summary.json"),
            "phase11_visual": str(root / "phase11_memory_ruler_rediscovery/visual_integrity_audit.json"),
        },
        "final_interpretation": (
            "Current memory-ruler definition and available usage proxies do not provide enough labelled bad/good "
            "support or control specificity to justify carrier localization, counterfactuals, runtime action, or TTT."
        ),
    }

    blockers = [
        {
            "phase": "Phase2",
            "blocker": "score_available_high_quality_ratio_below_gate",
            "observed": phase2.get("score_available_high_quality_ratio"),
            "required": ">=0.80",
            "evidence_path": str(root / "phase2_memory_ruler_ledger/phase2_ledger_summary.json"),
            "action_taken": "risk_formula_fix; usage_neighborhood_radius1; support expansion",
            "resolved": False,
        },
        {
            "phase": "Phase2",
            "blocker": "labelled_main_anchor_support_sparse",
            "observed": f"anchor_pairs={phase2.get('ruler_anchor_pair_count')}; anchor_tokens={phase2.get('ruler_anchor_token_count')}",
            "required": "enough positive support before Phase3",
            "evidence_path": str(root / "phase2_memory_ruler_ledger/phase2_ledger_summary.json"),
            "action_taken": "Phase10 expanded observable rows to >=2x",
            "resolved": False,
        },
        {
            "phase": "Phase10",
            "blocker": "specificity_controls_unavailable",
            "observed": f"semantic_shuffle_specificity_available={phase10.get('semantic_shuffle_specificity_available')}",
            "required": "semantic-shuffle and same-mass/random controls before useful expansion success",
            "evidence_path": str(root / "phase10_support_expansion_audit/support_expansion_audit_summary.json"),
            "action_taken": "Phase11 rediscovery generated visual/question bundle",
            "resolved": False,
        },
        {
            "phase": "Phase10",
            "blocker": "bad_anchor_recall_too_low",
            "observed": phase10.get("labelled_bad_anchor_recall_observed"),
            "required": ">=0.60 bad recall for mechanism route",
            "evidence_path": str(root / "phase10_support_expansion_audit/support_expansion_audit_summary.json"),
            "action_taken": "Visual rediscovery separates bad absence, contradiction, good false positives, low observability",
            "resolved": False,
        },
        {
            "phase": "Phase11",
            "blocker": "per_head_route_mass_unavailable",
            "observed": "SWA_usage is PCA QK proxy, not true route attention mass",
            "required": "true per-head route mass for carrier localization and same-head random controls",
            "evidence_path": str(root / "phase11_memory_ruler_rediscovery/per_head_carrier_panels/per_head_route_mass_unavailable.png"),
            "action_taken": "Explicit unavailable-evidence panel and hypothesis bank",
            "resolved": False,
        },
    ]

    out = args.out_dir
    write_json(out / "final_decision.json", final_decision)
    write_csv(out / "blocker_attribution.csv", blockers)
    memory_report = [
        "# ACL2 v84 Memory Ruler Report",
        "",
        f"Final status: `{final_decision['final_status']}`",
        f"Decision labels: `{', '.join(labels)}`",
        "",
        "## Key Evidence",
        "",
        f"- Phase1 main: {phase1.get('adjacent_pair_rows')} pair rows, {phase1.get('token_rows')} token rows, READ/SWA availability 1.0.",
        f"- Phase2 main: gate pass `{phase2.get('phase2_gate_pass')}`, anchor pairs {phase2.get('ruler_anchor_pair_count')}, anchor tokens {phase2.get('ruler_anchor_token_count')}, score availability {phase2.get('score_available_high_quality_ratio')}.",
        f"- Phase10: expanded to {phase10.get('expanded_pair_rows')} rows ({phase10.get('rows_increase_ratio')}x), anchor-support pairs {phase10.get('anchor_support_pair_count')}, useful gate `{phase10.get('expansion_useful_gate_pass')}`.",
        f"- Labelled bad anchor recall after expansion: {phase10.get('labelled_bad_anchor_recall_observed')}; labelled good anchor FPR: {phase10.get('labelled_good_anchor_fpr_observed')}.",
        f"- Phase11 visual integrity pass: `{phase11.get('visual_integrity_pass')}` with {phase11.get('manifest_rows')} manifest rows.",
        "",
        "## Conclusion",
        "",
        "The current Memory Ruler candidate definition finds some plausible anchors after support expansion, but they do not separate labelled bad/good handoffs strongly enough and lack required specificity controls. No runtime action or TTT route is eligible.",
        "",
    ]
    (out / "memory_ruler_report.md").write_text("\n".join(memory_report), encoding="utf-8")

    carrier_report = [
        "# Carrier Localization Report",
        "",
        "Carrier localization did not run because Phase2 and Phase10 did not pass the required support/specificity gates.",
        "",
        "Current SWA evidence is limited to PCA QK-compatibility proxy. True per-head route mass, semantic-shuffle route controls, same-head random controls, and QK/V carrier margins remain unavailable.",
        "",
        "Therefore no READ, SWA QK, SWA V, merge/gauge, or TTT carrier is confirmed in v84.",
        "",
    ]
    (out / "carrier_localization_report.md").write_text("\n".join(carrier_report), encoding="utf-8")

    action_report = [
        "# Action Misuse Report",
        "",
        "No runtime action was executed.",
        "",
        "The run did not start SWA action, merge/gauge action, or TTT because the audit gates required before action failed. This avoids repeating the v82/v83 misuse pattern where visual/audit artifacts were mistaken for method success.",
        "",
    ]
    (out / "action_misuse_report.md").write_text("\n".join(action_report), encoding="utf-8")

    next_route = [
        "# Next Route Recommendation",
        "",
        "1. Add true per-head READ/SWA route mass dumps for the same support rows, then rerun semantic-shuffle and same-head random controls.",
        "2. Refine geometry leverage toward structural edges and local shape, because many observed anchors are degenerate-dominant.",
        "3. Keep seq01 low-confidence/minconf0 rows as stress/risk only.",
        "4. If true SWA route controls remain nonspecific, move the carrier search to merge/gauge boundary state rather than SWA alpha/action sweeps.",
        "5. Do not run TTT until a confirmed ruler carrier passes support, specificity, visual, and counterfactual gates.",
        "",
    ]
    (out / "next_route_recommendation.md").write_text("\n".join(next_route), encoding="utf-8")

    print(json.dumps({"out_dir": str(out), "final_status": final_decision["final_status"], "decision_labels": labels}, ensure_ascii=False))


if __name__ == "__main__":
    main()
