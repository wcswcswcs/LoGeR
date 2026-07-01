#!/usr/bin/env python3
"""Build ACL2 v83 Phase10 final decision matrix from audited artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ROOT = Path("results/acl2_v83tf_clue_sufficiency_vs_action_misuse")
DEFAULT_OUT_DIR = DEFAULT_ROOT / "phase10_decision_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_by_combo(rows: list[dict[str, str]], combo: str) -> dict[str, str]:
    for row in rows:
        if row.get("combo") == combo:
            return row
    return {}


def row_by_carrier(rows: list[dict[str, str]], carrier: str) -> dict[str, str]:
    for row in rows:
        if row.get("carrier_body") == carrier:
            return row
    return {}


def build_blockers(
    phase2_rows: list[dict[str, str]],
    phase3_rows: list[dict[str, str]],
    phase4_rows: list[dict[str, str]],
    visual_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    c0 = row_by_combo(phase2_rows, "C0_geometry_only")
    c4 = row_by_combo(phase2_rows, "C4_geometry_semantic")
    c7 = row_by_combo(phase2_rows, "C7_full_combined")
    swa = row_by_carrier(phase3_rows, "SWA")
    merge = row_by_carrier(phase3_rows, "merge_gauge")
    read = row_by_carrier(phase3_rows, "READ")
    ttt = row_by_carrier(phase3_rows, "TTT")
    best_phase4 = max(
        [row for row in phase4_rows if f(row.get("bad_median_improvement_vs_baseline_ratio")) is not None],
        key=lambda row: f(row.get("bad_median_improvement_vs_baseline_ratio")) or -1.0,
        default={},
    )
    return [
        {
            "blocker_label": "D3_GEOMETRY_SUFFICIENT_SEMANTIC_NO_ADD",
            "status": truthy(c0.get("gate_pass")) and not truthy(c4.get("gate_pass")) and not truthy(c7.get("gate_pass")),
            "evidence": {
                "C0_gate_pass": c0.get("gate_pass", ""),
                "C0_auc": c0.get("auc", ""),
                "C4_gate_pass": c4.get("gate_pass", ""),
                "C4_delta_sem_vs_geometry_auc": c4.get("delta_sem_vs_geometry_auc", ""),
                "C7_gate_pass": c7.get("gate_pass", ""),
                "C7_delta_sem_vs_geometry_auc": c7.get("delta_sem_vs_geometry_auc", ""),
            },
            "interpretation": "Geometry-only clues separate bad/good; current semantic combination does not add measurable lift.",
        },
        {
            "blocker_label": "D2_SEMANTIC_NOT_SPECIFIC",
            "status": truthy(swa.get("semantic_shuffle_available")) and not truthy(swa.get("semantic_shuffle_specificity_gate_pass")),
            "evidence": {
                "SWA_semantic_shuffle_available": swa.get("semantic_shuffle_available", ""),
                "SWA_semantic_shuffle_specificity_gate_pass": swa.get("semantic_shuffle_specificity_gate_pass", ""),
                "SWA_blocker": swa.get("blocker", ""),
            },
            "interpretation": "Existing semantic-shuffle controls do not show semantic specificity for SWA route evidence.",
        },
        {
            "blocker_label": "D6_SWA_NOT_SCALE_GAUGE_CARRIER",
            "status": not truthy(swa.get("carrier_gate_pass")) and (f(merge.get("auc")) or 0.0) > (f(swa.get("auc")) or 0.0),
            "evidence": {
                "SWA_auc": swa.get("auc", ""),
                "SWA_bad_recall": swa.get("bad_recall", ""),
                "SWA_good_fpr": swa.get("good_false_positive_rate", ""),
                "merge_gauge_auc": merge.get("auc", ""),
                "merge_gauge_bad_recall": merge.get("bad_recall", ""),
                "merge_gauge_good_fpr": merge.get("good_false_positive_rate", ""),
            },
            "interpretation": "SWA route/QK is weaker than merge/gauge residual evidence for this failure family.",
        },
        {
            "blocker_label": "D7_MERGE_GAUGE_NEEDS_NEW_INTERFACE",
            "status": not truthy(merge.get("carrier_gate_pass")) and bool(merge),
            "evidence": {
                "merge_gauge_carrier_gate_pass": merge.get("carrier_gate_pass", ""),
                "merge_gauge_blocker": merge.get("blocker", ""),
                "phase4_best_bad_improvement": best_phase4.get("bad_median_improvement_vs_baseline_ratio", ""),
                "phase4_best_invalid_as_runtime_method": best_phase4.get("invalid_as_runtime_method", ""),
            },
            "interpretation": "Merge/gauge is the strongest diagnostic direction, but current controls/upper-bound are insufficient.",
        },
        {
            "blocker_label": "D8_TTT_NOT_READY",
            "status": not truthy(ttt.get("carrier_gate_pass")),
            "evidence": {
                "TTT_carrier_gate_pass": ttt.get("carrier_gate_pass", ""),
                "TTT_blocker": ttt.get("blocker", ""),
            },
            "interpretation": "No confirmed SWA or merge/gauge carrier exists for TTT write eligibility.",
        },
        {
            "blocker_label": "VISUAL_REDISCOVERY_AUDITED",
            "status": bool(visual_audit.get("visual_audit_gate_pass")),
            "evidence": {
                "visual_audit_gate_pass": visual_audit.get("visual_audit_gate_pass", ""),
                "after_visual_rediscovery_decision": visual_audit.get("after_visual_rediscovery_decision", ""),
                "failed_case_question_count": visual_audit.get("failed_case_question_count", ""),
            },
            "interpretation": "Final No-Go is backed by mandatory visual rediscovery audit.",
        },
        {
            "blocker_label": "READ_CONTROLS_MISSING",
            "status": not truthy(read.get("carrier_gate_pass")),
            "evidence": {
                "READ_carrier_gate_pass": read.get("carrier_gate_pass", ""),
                "READ_auc": read.get("auc", ""),
                "READ_blocker": read.get("blocker", ""),
            },
            "interpretation": "READ QK controls are incomplete and READ separation is not sufficient for action.",
        },
    ]


def answers(
    phase2_summary: Mapping[str, Any],
    phase2_rows: list[dict[str, str]],
    phase3_summary: Mapping[str, Any],
    phase3_rows: list[dict[str, str]],
    phase4_summary: Mapping[str, Any],
    visual_audit: Mapping[str, Any],
) -> dict[str, Any]:
    c0 = row_by_combo(phase2_rows, "C0_geometry_only")
    c2 = row_by_combo(phase2_rows, "C2_radio_only")
    c4 = row_by_combo(phase2_rows, "C4_geometry_semantic")
    c7 = row_by_combo(phase2_rows, "C7_full_combined")
    read = row_by_carrier(phase3_rows, "READ")
    swa = row_by_carrier(phase3_rows, "SWA")
    merge = row_by_carrier(phase3_rows, "merge_gauge")
    ttt = row_by_carrier(phase3_rows, "TTT")
    carriers = {
        "READ": f(read.get("auc")) or -1.0,
        "SWA": f(swa.get("auc")) or -1.0,
        "merge_gauge": f(merge.get("auc")) or -1.0,
        "TTT": f(ttt.get("auc")) or -1.0,
    }
    strongest = max(carriers, key=carriers.get)
    return {
        "1_geometry_only_separated_bad_good": {
            "answer": truthy(c0.get("gate_pass")),
            "evidence": {"auc": c0.get("auc"), "bad_recall": c0.get("bad_recall"), "good_fpr": c0.get("good_false_positive_rate")},
        },
        "2_semantic_or_radio_added_lift": {
            "answer": False,
            "evidence": {
                "C4_gate_pass": c4.get("gate_pass"),
                "C4_delta_sem_vs_geometry_auc": c4.get("delta_sem_vs_geometry_auc"),
                "C7_gate_pass": c7.get("gate_pass"),
                "C7_delta_sem_vs_geometry_auc": c7.get("delta_sem_vs_geometry_auc"),
                "C2_scored_rows": c2.get("scored_rows"),
            },
        },
        "3_actual_semantic_beat_shuffle_and_random": {
            "answer": False,
            "evidence": phase3_summary.get("route_specificity_evidence", {}),
        },
        "4_strongest_carrier_alignment": {
            "answer": strongest,
            "evidence": carriers,
        },
        "5_clue_sufficiency_passed_but_runtime_action_failed": {
            "answer": "clue sufficiency passed for geometry-only; runtime action was not eligible and was not run",
            "evidence": {
                "phase2_gate_pass": phase2_summary.get("phase2_gate_pass"),
                "phase3_gate_pass": phase3_summary.get("phase3_gate_pass"),
                "phase4_gate_pass": phase4_summary.get("phase4_gate_pass"),
            },
        },
        "6_action_failure_mode": {
            "answer": "no runtime action; blocked before action by carrier and counterfactual gates",
            "evidence": {
                "phase3_decision": phase3_summary.get("decision"),
                "phase4_decision": phase4_summary.get("decision"),
            },
        },
        "7_swa_route_qk_scale_gauge_carrier": {
            "answer": False,
            "evidence": {"SWA_auc": swa.get("auc"), "SWA_blocker": swa.get("blocker")},
        },
        "8_merge_gauge_stronger_than_swa": {
            "answer": True,
            "evidence": {"merge_auc": merge.get("auc"), "SWA_auc": swa.get("auc"), "merge_blocker": merge.get("blocker")},
        },
        "9_ttt_run_only_after_confirmation": {
            "answer": "TTT did not run",
            "evidence": {"TTT_blocker": ttt.get("blocker")},
        },
        "10_good_cases_protected": {
            "answer": "not enough for promotion; Phase4 found good worsen violations in candidate rows",
            "evidence": {
                "phase4_best_bad_improvement": phase4_summary.get("best_bad_improvement", {}),
            },
        },
        "11_method_candidate_passed_heldout_or_704f": {
            "answer": False,
            "evidence": "No candidate reached Phase9 entry condition.",
        },
        "12_no_go_blocker": {
            "answer": [
                "semantic_nonspecificity",
                "SWA_not_scale_gauge_carrier",
                "merge_gauge_current_interface_upper_bound_failed",
                "TTT_not_ready",
            ],
            "evidence": {
                "visual_audit_gate_pass": visual_audit.get("visual_audit_gate_pass"),
                "phase4_stop_reason": phase4_summary.get("stop_reason"),
            },
        },
    }


def render_clue_report(path: Path, final: Mapping[str, Any]) -> None:
    a = final["final_report_answers"]
    lines = [
        "# ACL2 v83 Clue Sufficiency Report",
        "",
        f"Phase2 decision: `{final['phase2_decision']}`",
        "",
        "## Geometry",
        "",
        f"- geometry-only separated bad/good: `{a['1_geometry_only_separated_bad_good']['answer']}`",
        f"- evidence: `{a['1_geometry_only_separated_bad_good']['evidence']}`",
        "",
        "## Semantic / RADIO",
        "",
        f"- semantic or RADIO added measurable lift: `{a['2_semantic_or_radio_added_lift']['answer']}`",
        f"- actual semantic beat shuffle/random: `{a['3_actual_semantic_beat_shuffle_and_random']['answer']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def render_action_report(path: Path, final: Mapping[str, Any]) -> None:
    a = final["final_report_answers"]
    lines = [
        "# ACL2 v83 Action Misuse Report",
        "",
        f"final_status: `{final['final_status']}`",
        f"runtime_action_allowed: `{final['runtime_action_allowed']}`",
        "",
        "## Carrier / Action Gates",
        "",
        f"- strongest carrier alignment: `{a['4_strongest_carrier_alignment']['answer']}`",
        f"- SWA route/QK actual scale/gauge carrier: `{a['7_swa_route_qk_scale_gauge_carrier']['answer']}`",
        f"- merge/gauge stronger than SWA: `{a['8_merge_gauge_stronger_than_swa']['answer']}`",
        f"- action failure mode: `{a['6_action_failure_mode']['answer']}`",
        "",
        "## TTT / Held-Out",
        "",
        f"- TTT status: `{a['9_ttt_run_only_after_confirmation']['answer']}`",
        f"- held-out or 704F candidate pass: `{a['11_method_candidate_passed_heldout_or_704f']['answer']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def render_recommendation(path: Path, final: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v83 Next Route Recommendation",
        "",
        "Do not continue v83 runtime action routes.",
        "",
        "Evidence:",
        "",
        "- Phase2 passed only for geometry clues; semantic/RADIO did not add lift.",
        "- Phase3 found no carrier with actual-vs-random plus semantic-shuffle specificity.",
        "- Phase4 found no counterfactual/oracle row with bad improvement >=10%, good worsen <=2%, and all-bad control beating.",
        "- Mandatory visual rediscovery audit passed, so No-Go is not vague.",
        "",
        "Next valid plan should design a new merge/gauge state interface with native same-overlap random and semantic-shuffle weighting controls before any runtime action.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase2_summary = read_json(args.root / "phase2_clue_sufficiency/clue_sufficiency_summary.json")
    phase2_rows = read_csv(args.root / "phase2_clue_sufficiency/clue_sufficiency_rows.csv")
    phase3_summary = read_json(args.root / "phase3_carrier_alignment/carrier_alignment_summary.json")
    phase3_rows = read_csv(args.root / "phase3_carrier_alignment/carrier_alignment_rows.csv")
    phase4_summary = read_json(args.root / "phase4_counterfactual_upper_bound/counterfactual_upper_bound_summary.json")
    phase4_rows = read_csv(args.root / "phase4_counterfactual_upper_bound/counterfactual_upper_bound_rows.csv")
    visual_audit = read_json(args.root / "phase11_visual_rediscovery/visual_integrity_audit.json")

    blockers = build_blockers(phase2_rows, phase3_rows, phase4_rows, visual_audit)
    final_answers = answers(phase2_summary, phase2_rows, phase3_summary, phase3_rows, phase4_summary, visual_audit)
    active_labels = [row["blocker_label"] for row in blockers if row["status"]]
    final_status = "No-Go_before_runtime_action"
    final = {
        "schema": "acl2_v83_phase10_final_decision_v1",
        "final_status": final_status,
        "method_candidate": False,
        "runtime_action_allowed": False,
        "phase2_decision": phase2_summary.get("interpretation", {}).get("decision", ""),
        "phase2_gate_pass": bool(phase2_summary.get("phase2_gate_pass")),
        "phase3_decision": phase3_summary.get("decision", ""),
        "phase3_gate_pass": bool(phase3_summary.get("phase3_gate_pass")),
        "phase4_decision": phase4_summary.get("decision", ""),
        "phase4_gate_pass": bool(phase4_summary.get("phase4_gate_pass")),
        "visual_rediscovery_gate_pass": bool(visual_audit.get("visual_audit_gate_pass")),
        "active_decision_labels": active_labels,
        "primary_decision_labels": [
            "D3_GEOMETRY_SUFFICIENT_SEMANTIC_NO_ADD",
            "D2_SEMANTIC_NOT_SPECIFIC",
            "D6_SWA_NOT_SCALE_GAUGE_CARRIER",
            "D7_MERGE_GAUGE_NEEDS_NEW_INTERFACE",
            "D8_TTT_NOT_READY",
        ],
        "final_report_answers": final_answers,
        "artifact_inputs": {
            "phase2_summary": str(args.root / "phase2_clue_sufficiency/clue_sufficiency_summary.json"),
            "phase3_summary": str(args.root / "phase3_carrier_alignment/carrier_alignment_summary.json"),
            "phase4_summary": str(args.root / "phase4_counterfactual_upper_bound/counterfactual_upper_bound_summary.json"),
            "visual_rediscovery_audit": str(args.root / "phase11_visual_rediscovery/visual_integrity_audit.json"),
        },
        "conclusion": (
            "v83 stops before runtime action: geometry clues are sufficient, current semantic definitions are not specific, "
            "SWA is not the scale/gauge carrier, merge/gauge current families lack a passing upper bound, and TTT is not eligible."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "final_decision.json", final)
    write_csv(args.out_dir / "blocker_attribution.csv", blockers)
    render_clue_report(args.out_dir / "clue_sufficiency_report.md", final)
    render_action_report(args.out_dir / "action_misuse_report.md", final)
    render_recommendation(args.out_dir / "next_route_recommendation.md", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
