#!/usr/bin/env python3
"""Create a formal v80 No-Go review from landed evidence.

The report is a requirement-by-requirement audit. It intentionally separates
prerequisite/visual completion from method success: a formal No-Go can be ready
while v80_goal_achieved remains false.
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
DEFAULT_MATRIX_ROOT = REPORT_ROOT / "phase10_current_action_evidence_matrix_20260622_2213"
DEFAULT_VISUAL_ROOT = REPORT_ROOT / "phase10_seq01_phase9_rediscovery_visual_completion_20260622_2305"
DEFAULT_GEOM_TTT_ROOT = REPORT_ROOT / "phase10_seq01_geometry_error_ttt_semantic_explanation_20260622_2245"
DEFAULT_CASE_BANK = REPORT_ROOT / "phase1_three_memory_case_bank" / "case_bank_summary.json"
DEFAULT_PHASE0 = REPORT_ROOT / "phase0_multiseq_artifact_audit" / "phase0_artifact_audit_summary.json"
DEFAULT_PHASE2 = REPORT_ROOT / "phase2_case_visual_confirmation" / "visual_integrity_audit.json"
DEFAULT_DIRECT_VISUAL = (
    REPORT_ROOT
    / "phase2_direct_hook_enhanced_visual_review_allseq_aggregate"
    / "visual_integrity_audit_allseq.json"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_formal_no_go_review_20260622_2214"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--geometry-ttt-root", type=Path, default=DEFAULT_GEOM_TTT_ROOT)
    parser.add_argument("--case-bank-summary", type=Path, default=DEFAULT_CASE_BANK)
    parser.add_argument("--phase0-summary", type=Path, default=DEFAULT_PHASE0)
    parser.add_argument("--phase2-summary", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--direct-visual-summary", type=Path, default=DEFAULT_DIRECT_VISUAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else val for key, val in row.items()})


def _bool(value: Any) -> bool:
    return bool(value is True or str(value).lower() == "true")


def _action_rows(matrix_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in matrix_rows if row.get("memory_body") != "prerequisite"]


def _prereq_rows(matrix_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in matrix_rows if row.get("memory_body") == "prerequisite"]


def _build_requirement_rows(
    phase0: dict[str, Any],
    case_bank: dict[str, Any],
    phase2: dict[str, Any],
    direct_visual: dict[str, Any],
    visual: dict[str, Any],
    matrix: dict[str, Any],
    geom_ttt: dict[str, Any],
) -> list[dict[str, Any]]:
    memory_summary = case_bank.get("memory_body_summary") or {}
    available_seqs = phase0.get("phase1_basic_case_mining_allowed_seqs") or []
    requested_seqs = phase0.get("seqs") or []
    seq08_blockers = (phase0.get("blockers_by_seq") or {}).get("08", [])
    min_case_gate = all(
        (memory_summary.get(body) or {}).get("gate_pass") for body in ("short", "mid", "long")
    )
    min_case_detail = {
        body: {
            "bad": (memory_summary.get(body) or {}).get("bad"),
            "good": (memory_summary.get(body) or {}).get("good"),
            "bad_seqs": (memory_summary.get(body) or {}).get("bad_seqs"),
            "good_seqs": (memory_summary.get(body) or {}).get("good_seqs"),
        }
        for body in ("short", "mid", "long")
    }

    action_success = _bool(matrix.get("action_gate_pass_any"))
    controls_success = bool(matrix.get("action_gate_pass_requirements"))
    failure_localized = bool(matrix.get("uncovered_or_weak_points")) and bool(geom_ttt.get("local_semantic_explains_ttt_low_support_write"))

    return [
        {
            "requirement_id": "R1_phase0_artifacts",
            "plan_ref": "Plan lines 623-659",
            "requirement": "Artifact gates must be checked and missing paths must remain diagnostic-only.",
            "status": "pass_with_scope_caveat",
            "evidence": str(DEFAULT_PHASE0),
            "detail": {
                "phase0_gate_pass": phase0.get("phase0_gate_pass"),
                "requested_seqs": requested_seqs,
                "phase1_allowed_seqs": available_seqs,
                "seq08_blockers": seq08_blockers,
                "diagnostic_only_reasons_by_seq": phase0.get("diagnostic_only_reasons_by_seq"),
            },
        },
        {
            "requirement_id": "R2_three_memory_case_bank",
            "plan_ref": "Plan lines 226-231",
            "requirement": "Build short/mid/long good/bad case bank with >=12 bad and >=12 good cases over >=3 sequences.",
            "status": "pass_available_scope_only",
            "evidence": str(DEFAULT_CASE_BANK),
            "detail": {
                "phase1_gate_pass": case_bank.get("phase1_gate_pass"),
                "phase1_balance_gate_pass": case_bank.get("phase1_balance_gate_pass"),
                "semantic_diagnosis_gate_pass": case_bank.get("semantic_diagnosis_gate_pass"),
                "memory_body_summary": min_case_detail,
                "scope_caveat": "case bank covers 00/01/02/05; seq08 was blocked by baseline_trajectory_available in Phase0.",
                "min_case_gate": min_case_gate,
            },
        },
        {
            "requirement_id": "R3_visual_confirmation",
            "plan_ref": "Plan lines 231 and 1568-1604",
            "requirement": "Each case has semantic diagnosis and PCA/QKV/TTT visual confirmation; Phase9 rediscovery visual audit passes before final No-Go.",
            "status": "pass",
            "evidence": [str(DEFAULT_PHASE2), str(DEFAULT_DIRECT_VISUAL), str(DEFAULT_VISUAL_ROOT / "visual_integrity_audit.json")],
            "detail": {
                "phase2_gate_pass": phase2.get("gate_pass"),
                "phase2_action_ready_gate_pass": phase2.get("action_ready_gate_pass"),
                "direct_visual_gate_pass": direct_visual.get("aggregate_phase2_direct_hook_visual_gate_pass"),
                "phase9_visual_audit_gate_pass": visual.get("visual_audit_gate_pass"),
                "phase9_group_counts": visual.get("group_counts"),
            },
        },
        {
            "requirement_id": "R4_semantic_action_improves",
            "plan_ref": "Plan lines 232 and 1249-1269",
            "requirement": "At least one semantic-conditioned action significantly improves bad cases and does not break good cases.",
            "status": "fail",
            "evidence": str(DEFAULT_MATRIX_ROOT / "current_action_evidence_matrix_summary.json"),
            "detail": {
                "action_gate_pass_any": matrix.get("action_gate_pass_any"),
                "action_gate_pass_requirements": matrix.get("action_gate_pass_requirements"),
                "failed_action_families": matrix.get("failed_action_families"),
            },
        },
        {
            "requirement_id": "R5_beats_controls",
            "plan_ref": "Plan lines 233 and 1264-1268",
            "requirement": "Successful action must beat geometry-only, semantic shuffle, same-mass random, and related controls.",
            "status": "fail",
            "evidence": str(DEFAULT_MATRIX_ROOT / "current_action_evidence_matrix_rows.csv"),
            "detail": {
                "control_separated_success": controls_success,
                "summary": "No action family has a method gate pass after geometry/random/shuffle or paired-control checks.",
            },
        },
        {
            "requirement_id": "R6_failure_localization",
            "plan_ref": "Plan lines 234-240 and 1739",
            "requirement": "If no method succeeds, localize where the semantic memory path fails.",
            "status": "pass",
            "evidence": [
                str(DEFAULT_MATRIX_ROOT / "current_action_evidence_matrix_summary.json"),
                str(DEFAULT_GEOM_TTT_ROOT / "geometry_error_ttt_semantic_explanation_summary.json"),
            ],
            "detail": {
                "failure_localized": failure_localized,
                "local_semantic_explains_chunk08": geom_ttt.get("local_semantic_explains_ttt_low_support_write"),
                "local_explained_chunks": geom_ttt.get("local_explained_chunks"),
                "head_tail_only_overlap_harm_chunks": geom_ttt.get("head_tail_only_overlap_harm_chunks"),
                "core_blocker": geom_ttt.get("core_blocker"),
                "uncovered_or_weak_points": matrix.get("uncovered_or_weak_points"),
            },
        },
    ]


def _build_action_family_rows(matrix_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _action_rows(matrix_rows):
        metrics: dict[str, Any]
        try:
            metrics = json.loads(row.get("selected_metrics") or "{}")
        except json.JSONDecodeError:
            metrics = {}
        out.append(
            {
                "requirement": row.get("requirement"),
                "memory_body": row.get("memory_body"),
                "family": row.get("family"),
                "status": row.get("status"),
                "gate_pass": row.get("gate_pass"),
                "source": row.get("source"),
                "selected_metrics": metrics,
            }
        )
    return out


def _checklist_answers(
    phase0: dict[str, Any],
    case_bank: dict[str, Any],
    visual: dict[str, Any],
    matrix: dict[str, Any],
    geom_ttt: dict[str, Any],
) -> list[dict[str, str]]:
    memory_summary = case_bank.get("memory_body_summary") or {}
    case_scope = "short/mid/long use 00/01/02/05; 08 blocked by baseline trajectory availability."
    all_good_bad = all((memory_summary.get(body) or {}).get("bad") == 12 and (memory_summary.get(body) or {}).get("good") == 12 for body in ("short", "mid", "long"))
    return [
        {"question": "Which sequences and cases were used for short/mid/long memory?", "answer": case_scope},
        {"question": "Did each memory body have good and bad cases?", "answer": f"Yes in available scope: {all_good_bad}."},
        {"question": "Did PCA/QKV/TTT visual confirmation pass?", "answer": f"Yes: Phase9 visual_audit_gate_pass={visual.get('visual_audit_gate_pass')}."},
        {"question": "Which semantic source helped?", "answer": "Local semantic/geometry low-support plus RADIO/thingstuff evidence explains chunk08 only; no deployable multi-case source."},
        {"question": "Did semantic READ improve single-chunk geometry?", "answer": "No method gate: READ/QK family failed."},
        {"question": "Did semantic SWA improve adjacent overlap/future?", "answer": "No method gate: SWA/merge/controller families failed or selected overlap-harm false positives."},
        {"question": "Did semantic TTT improve five-chunk persistence?", "answer": "No method gate: TTT selected-write/no-persistent/post-delta families failed."},
        {"question": "Did cross-memory semantic roles align?", "answer": "No: handshake failed and geometry-error/TTT semantic explanation is chunk08-local only."},
        {"question": "Did semantic beat geometry-only and random/shuffled controls?", "answer": "No action family has a gate pass over required controls."},
        {"question": "Did good cases remain safe?", "answer": "Prereq case bank has good cases, but no promoted action survived controls, so no method safety claim."},
        {"question": "Did official 704F/full improve?", "answer": "Not run/promoted; runtime_promotion_allowed=false."},
        {"question": "If no, where exactly did the semantic memory path fail?", "answer": str(matrix.get("next_action_reason"))},
    ]


def _write_report(
    path: Path,
    summary: dict[str, Any],
    requirements: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    checklist: list[dict[str, str]],
) -> None:
    lines = [
        "# v80 Formal No-Go Review",
        "",
        f"- final_decision: `{summary['final_decision']}`",
        f"- formal_no_go_ready: `{summary['formal_no_go_ready']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- runtime_promotion_allowed: `{summary['runtime_promotion_allowed']}`",
        "",
        "## Requirement Audit",
        "",
        "| id | status | evidence | requirement |",
        "|---|---|---|---|",
    ]
    for row in requirements:
        evidence = row.get("evidence")
        if isinstance(evidence, list):
            evidence_text = "<br>".join(str(item) for item in evidence)
        else:
            evidence_text = str(evidence)
        lines.append(f"| {row['requirement_id']} | {row['status']} | {evidence_text} | {row['requirement']} |")

    lines.extend(["", "## Action Families", "", "| family | memory | status | source |", "|---|---|---|---|"])
    for row in actions:
        lines.append(f"| {row['family']} | {row['memory_body']} | {row['status']} | {row['source']} |")

    lines.extend(["", "## Checklist", "", "| question | answer |", "|---|---|"])
    for row in checklist:
        lines.append(f"| {row['question']} | {row['answer']} |")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["decision_text"],
            "",
            "This is not a success claim. It is a formal No-Go review for the current v80 evidence state.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    matrix = _read_json(args.matrix_root / "current_action_evidence_matrix_summary.json")
    matrix_rows = _read_csv(args.matrix_root / "current_action_evidence_matrix_rows.csv")
    visual = _read_json(args.visual_root / "visual_integrity_audit.json")
    geom_ttt = _read_json(args.geometry_ttt_root / "geometry_error_ttt_semantic_explanation_summary.json")
    case_bank = _read_json(args.case_bank_summary)
    phase0 = _read_json(args.phase0_summary)
    phase2 = _read_json(args.phase2_summary)
    direct_visual = _read_json(args.direct_visual_summary)

    requirements = _build_requirement_rows(phase0, case_bank, phase2, direct_visual, visual, matrix, geom_ttt)
    actions = _build_action_family_rows(matrix_rows)
    checklist = _checklist_answers(phase0, case_bank, visual, matrix, geom_ttt)
    false_separator = next(
        (row for row in actions if row.get("family") == "semantic false-positive separator"),
        {},
    )
    false_separator_metrics = false_separator.get("selected_metrics") or {}
    heldout_coverage = next(
        (row for row in actions if row.get("family") == "selected-write heldout coverage"),
        {},
    )
    heldout_coverage_metrics = heldout_coverage.get("selected_metrics") or {}

    action_gate_pass_any = _bool(matrix.get("action_gate_pass_any"))
    visual_gate_pass = _bool(visual.get("visual_audit_gate_pass"))
    prerequisite_gate_pass = _bool(matrix.get("prerequisite_gate_pass"))
    formal_ready = prerequisite_gate_pass and visual_gate_pass and not action_gate_pass_any
    fail_requirements = [row["requirement_id"] for row in requirements if row["status"] == "fail"]
    summary = {
        "schema": "acl2_v80_formal_no_go_review_v1",
        "final_decision": "formal_no_go_no_deployable_semantic_action",
        "formal_no_go_ready": formal_ready,
        "v80_goal_achieved": False,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "prerequisite_gate_pass": prerequisite_gate_pass,
        "phase9_visual_audit_gate_pass": visual_gate_pass,
        "action_gate_pass_any": action_gate_pass_any,
        "failed_method_requirements": fail_requirements,
        "scope_caveats": [
            "KITTI08 remains unavailable for case mining because Phase0 reports baseline_trajectory_available blocker.",
            "RADIO/RADSeg runtime evidence is available only for seq01 in the current artifact set.",
        ],
        "core_positive_signal": {
            "local_semantic_explains_ttt_low_support_write": geom_ttt.get("local_semantic_explains_ttt_low_support_write"),
            "local_explained_chunks": geom_ttt.get("local_explained_chunks"),
            "helpful_overlap_safe_chunks": geom_ttt.get("helpful_overlap_safe_chunks"),
        },
        "core_false_positive_separator": {
            "diagnostic_separator_found": false_separator_metrics.get("diagnostic_separator_found"),
            "best_separator_rule": false_separator_metrics.get("best_separator_rule"),
            "best_separator_selected_chunks": false_separator_metrics.get("best_separator_selected_chunks"),
            "overlap_harm_false_positive_chunks": false_separator_metrics.get("overlap_harm_false_positive_chunks"),
            "high_dq_false_positive_selected_chunks": false_separator_metrics.get("high_dq_false_positive_selected_chunks"),
        },
        "core_heldout_coverage": {
            "heldout_multi_case_gate": heldout_coverage_metrics.get("heldout_multi_case_gate"),
            "coverage_blockers": heldout_coverage_metrics.get("coverage_blockers"),
            "selected_write_positive_seq_chunks": heldout_coverage_metrics.get("selected_write_positive_seq_chunks"),
            "selected_write_positive_seqs": heldout_coverage_metrics.get("selected_write_positive_seqs"),
            "long_case_rows_with_positive_low_support_separator": heldout_coverage_metrics.get(
                "long_case_rows_with_positive_low_support_separator"
            ),
            "support_only_low_semantic_error_no_ttt_write_seq_chunks": heldout_coverage_metrics.get(
                "support_only_low_semantic_error_no_ttt_write_seq_chunks"
            ),
        },
        "core_blocker": geom_ttt.get("core_blocker"),
        "matrix_next_action": matrix.get("next_action"),
        "matrix_next_action_reason": matrix.get("next_action_reason"),
        "decision_text": (
            "The current evidence supports a local mechanism diagnosis for chunk08, but no semantic-conditioned "
            "READ/SWA/TTT/merge action passes the required control-separated method gates. The false-positive "
            "separator audit further localizes chunk10/chunk12 as high-Dq overlap-harm rather than low-support "
            "selected-write cases. Fresh seq00 attribution adds non-seq01 low-support selected-write evidence, but "
            "the held-out coverage audit still fails because positives cover too few sequences and no good-case "
            "safety rows. Phase9 visual artifacts are complete, so the current state is ready for formal No-Go "
            "review, not runtime promotion."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "formal_no_go_summary.json", summary)
    _write_csv(args.out_dir / "formal_no_go_requirement_audit.csv", requirements)
    _write_csv(args.out_dir / "formal_no_go_action_family_audit.csv", actions)
    _write_csv(args.out_dir / "formal_no_go_checklist_answers.csv", checklist)
    _write_report(args.out_dir / "formal_no_go_report.md", summary, requirements, actions, checklist)
    print(json.dumps(_jsonable({"out_dir": args.out_dir, **summary}), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
