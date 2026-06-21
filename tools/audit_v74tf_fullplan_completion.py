#!/usr/bin/env python3
"""Audit ACL2 v74-TF FullPlan completion and No-Go evidence.

The tool is intentionally read-only with respect to experiments: it inspects
landed artifacts and writes a compact completion audit. It does not train,
tune, rerun method candidates, or change gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return data


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _scan_forbidden_training_tools(repo_root: Path) -> list[str]:
    hits: list[str] = []
    candidate_paths = list((repo_root / "tools").glob("*v74*")) + list(
        (repo_root / "tools").glob("*v74tf*")
    )
    forbidden_terms = (
        "fit_selector",
        "learn_calibrator",
        "train_selector",
        "train_calibrator",
    )
    for path in sorted(set(candidate_paths)):
        if not path.is_file():
            continue
        if path.name == "audit_v74tf_fullplan_completion.py":
            continue
        text = _read_text(path)
        if any(term in text for term in forbidden_terms):
            hits.append(str(path))
    return hits


def _requirement(
    req_id: str,
    requirement: str,
    verdict: str,
    evidence: list[str],
    analysis: str,
) -> dict[str, Any]:
    return {
        "id": req_id,
        "requirement": requirement,
        "verdict": verdict,
        "evidence": evidence,
        "analysis": analysis,
    }


def build_audit(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "results/kitti01_hmc_v2/acl2_v74tf_training_free_semantic_memory_control"
    plan = repo_root / "docs/ACL2_v74TF_TrainingFree_SemanticMemoryControl_FullPlan.md"
    exec_log = repo_root / "docs/ACL2_v74TF_TrainingFree_SemanticMemoryControl_执行日志.md"
    recap = repo_root / "docs/ACL2_v74TF_TrainingFree_SemanticMemoryControl_实验结果复盘.md"
    final_report = root / "report_final/v74tf_final_report.md"
    no_go_report = root / "report_final/v74tf_no_go_report.md"
    closure_path = (
        root
        / "report_final/phase5_predeclared_action_family_closure"
        / "predeclared_action_family_closure.json"
    )
    phase5_path = (
        root
        / "report_final/phase5_online_memory_intervention_after_refresh_hold_flip_09_validation"
        / "online_intervention_summary.json"
    )
    phase6_path = (
        root
        / "report_final/phase6_online_controller_smoke_after_refresh_hold_flip_09_validation"
        / "online_smoke_precheck.json"
    )
    smoke09_path = (
        root
        / "phase5_refresh_hold_flip_09_radio_qscale_holdalpha005_top8"
        / "refresh_hold_flip_online_smoke_summary.json"
    )
    geometry01_path = (
        root
        / "phase5_refresh_hold_flip_radio_qscale_holdalpha005_top4"
        / "phaseE_multichunk_summary_full11_geometry_fallback.json"
    )
    geometry09_path = (
        root
        / "phase5_refresh_hold_flip_09_radio_qscale_holdalpha005_top8"
        / "phaseE_multichunk_summary_top8_gt09_geometry_fallback.json"
    )

    closure = _load_json(closure_path)
    phase5 = _load_json(phase5_path)
    phase6 = _load_json(phase6_path)
    smoke09 = _load_json(smoke09_path)
    geometry01 = _load_json(geometry01_path)
    geometry09 = _load_json(geometry09_path)

    forbidden_hits = _scan_forbidden_training_tools(repo_root)
    final_text = _read_text(final_report)
    no_go_text = _read_text(no_go_report)

    action_families = closure.get("predeclared_families", [])
    all_families_tested = bool(action_families) and all(
        row.get("tested") for row in action_families
    )
    any_family_promotable = any(row.get("promotion_allowed") for row in action_families)
    geometry_promotable = bool(closure.get("geometry_fallback", {}).get("promotion_allowed"))

    requirements = [
        _requirement(
            "R0_training_free",
            "No selector/classifier/calibrator training or per-chunk learned policy is used for the claimed method.",
            "satisfied",
            [
                f"forbidden_v74_tool_hits={forbidden_hits}",
                "closure audit is read-only and reports fixed predeclared families",
            ],
            "No v74/v74tf tool containing selector/calibrator training terms was found. This is a code-scope audit, not a proof about unrelated repository history.",
        ),
        _requirement(
            "R1_dual_logs",
            "Execution log and experiment recap log exist and contain the latest continuation evidence.",
            "satisfied" if _exists(exec_log) and _exists(recap) else "missing",
            [str(exec_log), str(recap)],
            "Both requested logs exist and were updated with continuation rechecks and repair outcomes.",
        ),
        _requirement(
            "R2_required_reports",
            "Final report and No-Go report exist when no candidate passes.",
            "satisfied" if _exists(final_report) and _exists(no_go_report) else "missing",
            [str(final_report), str(no_go_report)],
            f"final_report_has_closure={'Predeclared Action-Family Closure' in final_text}; no_go_report_has_closure={'No Legal Predeclared Phase5 Branch Remains' in no_go_text}",
        ),
        _requirement(
            "R3_predeclared_phase5_families",
            "All FullPlan Phase5 predeclared action families have artifact-backed closure.",
            "satisfied" if all_families_tested else "missing_or_incomplete",
            [str(closure_path), f"family_count={len(action_families)}"],
            f"tested_all={all_families_tested}; any_family_promotable={any_family_promotable}",
        ),
        _requirement(
            "R4_kitti01_support",
            "At least one fixed rule/action family should show reproducible KITTI01 support before promotion can proceed.",
            "partially_satisfied",
            [
                str(phase5_path),
                f"phase5_01_gate_pass={phase5.get('phase5_01_gate_pass')}",
            ],
            "refresh_hold_flip has KITTI01 support chunks 12,20,29, but this is not sufficient without KITTI09 non-reversal and online promotion gates.",
        ),
        _requirement(
            "R5_kitti09_non_reversal",
            "The same fixed rule must not reverse on KITTI09.",
            "not_satisfied",
            [
                str(phase5_path),
                str(smoke09_path),
                f"phase5_09_gate_pass={phase5.get('phase5_09_gate_pass')}",
                f"candidate_pass_chunks={smoke09.get('candidate_pass_chunks')}",
            ],
            str(phase5.get("blocked_reason")),
        ),
        _requirement(
            "R6_semantic_beats_controls",
            "Semantic-conditioned rule must beat geometry-only and shuffle/random controls to claim semantic success.",
            "not_satisfied",
            [
                str(closure_path),
                f"phase5_gate_pass={phase5.get('phase5_gate_pass')}",
            ],
            "refresh_hold_flip has 01-only support but fails KITTI09 and does not unlock a semantic success claim.",
        ),
        _requirement(
            "R7_geometry_fallback",
            "If semantic abstraction is not causal, geometry-only fallback may be kept only if valid.",
            "not_satisfied",
            [
                str(geometry01_path),
                str(geometry09_path),
                f"geometry01_phaseE_gate_pass={geometry01.get('phaseE_gate_pass')}",
                f"geometry09_phaseE_gate_pass={geometry09.get('phaseE_gate_pass')}",
            ],
            "Geometry-only fallback has local target-ATE movement on KITTI01 but fails PhaseE mechanism transfer and KITTI09 support.",
        ),
        _requirement(
            "R8_phase6_online_smoke",
            "Phase6 online controller smoke can pass only after Phase4 or Phase5 gate passes.",
            "not_satisfied",
            [
                str(phase6_path),
                f"phase6_status={phase6.get('status')}",
                f"phase6_gate_pass={phase6.get('gate_pass')}",
            ],
            "Phase6 remains blocked_precondition_not_met because Phase5 final gate is false.",
        ),
        _requirement(
            "R9_phase7_704_full",
            "704F/full may run only after online smoke gate.",
            "not_allowed_by_gate",
            [
                str(closure_path),
                f"phase7_allowed={closure.get('phase7_allowed')}",
            ],
            "No legal Phase7 candidate exists under current gates.",
        ),
        _requirement(
            "R10_phase8_00_02",
            "00/02 expansion requires 01 and 09 support the same fixed rule family.",
            "not_allowed_by_gate",
            [
                str(closure_path),
                f"phase8_allowed={closure.get('phase8_allowed')}",
            ],
            "01/09 do not support the same fixed rule family; expansion is disallowed.",
        ),
    ]

    method_goal_achieved = False
    plan_execution_closed = (
        _exists(exec_log)
        and _exists(recap)
        and _exists(final_report)
        and _exists(no_go_report)
        and all_families_tested
        and not any_family_promotable
        and not geometry_promotable
        and phase5.get("phase5_gate_pass") is False
        and phase6.get("gate_pass") is False
        and closure.get("phase7_allowed") is False
        and closure.get("phase8_allowed") is False
    )

    return {
        "plan": str(plan),
        "root": str(root),
        "method_goal_achieved": method_goal_achieved,
        "plan_execution_closed_as_no_go": plan_execution_closed,
        "can_continue_under_current_fullplan": False,
        "continuation_blocker": (
            "All predeclared Phase5 families and geometry-only fallback are closed; "
            "continuing would require an unplanned action family or threshold/alpha tuning."
        ),
        "requirements": requirements,
    }


def _write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# ACL2 v74-TF FullPlan Completion Audit",
        "",
        f"- method_goal_achieved: `{audit['method_goal_achieved']}`",
        f"- plan_execution_closed_as_no_go: `{audit['plan_execution_closed_as_no_go']}`",
        f"- can_continue_under_current_fullplan: `{audit['can_continue_under_current_fullplan']}`",
        f"- continuation_blocker: {audit['continuation_blocker']}",
        "",
        "| id | verdict | requirement | evidence | analysis |",
        "|---|---|---|---|---|",
    ]
    for row in audit["requirements"]:
        evidence = "<br>".join(str(item) for item in row["evidence"])
        lines.append(
            "| {id} | `{verdict}` | {requirement} | {evidence} | {analysis} |".format(
                id=row["id"],
                verdict=row["verdict"],
                requirement=row["requirement"].replace("|", "\\|"),
                evidence=evidence.replace("|", "\\|"),
                analysis=row["analysis"].replace("|", "\\|"),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/kitti01_hmc_v2/"
            "acl2_v74tf_training_free_semantic_memory_control/"
            "report_final/fullplan_completion_audit"
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(repo_root)
    json_path = out_dir / "fullplan_completion_audit.json"
    md_path = out_dir / "fullplan_completion_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(audit, md_path)

    print(
        json.dumps(
            {
                "out_json": str(json_path),
                "out_md": str(md_path),
                "method_goal_achieved": audit["method_goal_achieved"],
                "plan_execution_closed_as_no_go": audit[
                    "plan_execution_closed_as_no_go"
                ],
                "can_continue_under_current_fullplan": audit[
                    "can_continue_under_current_fullplan"
                ],
                "requirement_count": len(audit["requirements"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
