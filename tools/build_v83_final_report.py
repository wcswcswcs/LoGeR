#!/usr/bin/env python3
"""Build the ACL2 v83 final report from the Phase10 decision JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUESTION_TEXT = {
    "1": "Did geometry-only clues separate bad/good cases?",
    "2": "Did semantic or RADIO add measurable lift over geometry-only?",
    "3": "Did actual semantic beat semantic-shuffled and same-mass random controls?",
    "4": "Which memory body had the strongest carrier alignment: READ, SWA, merge/gauge, or TTT?",
    "5": "Did clue sufficiency pass but runtime action fail?",
    "6": "If yes, did action fail fidelity or fail geometry despite fidelity?",
    "7": "Was SWA route/QK an actual scale/gauge carrier?",
    "8": "If not, did merge/gauge show stronger carrier evidence?",
    "9": "Did TTT run only after SWA/merge confirmation?",
    "10": "Did good cases remain protected?",
    "11": "Did any method candidate pass held-out or official 704F?",
    "12": "If No-Go, is the blocker clue insufficiency, semantic nonspecificity, action misuse, weak carrier surface, merge/gauge missing interface, or TTT not ready?",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object at {path}")
    return data


def question_sort_key(item: tuple[str, Any]) -> int:
    prefix = item[0].split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return 999


def as_fenced_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_report(decision: dict[str, Any], decision_path: Path) -> tuple[dict[str, Any], str]:
    answers = decision.get("final_report_answers", {})
    if not isinstance(answers, dict):
        raise TypeError("final_report_answers must be an object")

    question_rows: list[dict[str, Any]] = []
    for key, payload in sorted(answers.items(), key=question_sort_key):
        number = key.split("_", 1)[0]
        if number not in QUESTION_TEXT:
            raise KeyError(f"unknown final report question key: {key}")
        if not isinstance(payload, dict):
            raise TypeError(f"answer payload for {key} must be an object")
        question_rows.append(
            {
                "number": int(number),
                "key": key,
                "question": QUESTION_TEXT[number],
                "answer": payload.get("answer"),
                "evidence": payload.get("evidence"),
            }
        )

    final_report = {
        "schema": "acl2_v83_phase20_final_report_v1",
        "source_decision_json": str(decision_path),
        "final_status": decision.get("final_status"),
        "method_candidate": decision.get("method_candidate"),
        "runtime_action_allowed": decision.get("runtime_action_allowed"),
        "phase2_gate_pass": decision.get("phase2_gate_pass"),
        "phase3_gate_pass": decision.get("phase3_gate_pass"),
        "phase4_gate_pass": decision.get("phase4_gate_pass"),
        "visual_rediscovery_gate_pass": decision.get("visual_rediscovery_gate_pass"),
        "primary_decision_labels": decision.get("primary_decision_labels", []),
        "questions": question_rows,
        "conclusion": decision.get("conclusion"),
    }

    lines = [
        "# ACL2 v83 Final Report",
        "",
        "## Source",
        "",
        f"- source_decision_json: `{decision_path}`",
        "",
        "## Final Status",
        "",
        f"- final_status: `{decision.get('final_status')}`",
        f"- method_candidate: `{decision.get('method_candidate')}`",
        f"- runtime_action_allowed: `{decision.get('runtime_action_allowed')}`",
        f"- phase2_gate_pass: `{decision.get('phase2_gate_pass')}`",
        f"- phase3_gate_pass: `{decision.get('phase3_gate_pass')}`",
        f"- phase4_gate_pass: `{decision.get('phase4_gate_pass')}`",
        f"- visual_rediscovery_gate_pass: `{decision.get('visual_rediscovery_gate_pass')}`",
        "",
        "## Primary Decision Labels",
        "",
    ]
    for label in decision.get("primary_decision_labels", []):
        lines.append(f"- `{label}`")

    lines.extend(["", "## Required Questions", ""])
    for row in question_rows:
        lines.append(f"### {row['number']}. {row['question']}")
        lines.append("")
        lines.append("Answer:")
        lines.append("")
        lines.append("```json")
        lines.append(as_fenced_json(row["answer"]))
        lines.append("```")
        lines.append("")
        lines.append("Evidence:")
        lines.append("")
        lines.append("```json")
        lines.append(as_fenced_json(row["evidence"]))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Conclusion",
            "",
            str(decision.get("conclusion")),
            "",
            "This report is generated only from Phase10 decision evidence. It does not add new metrics and does not promote any runtime action.",
            "",
        ]
    )

    return final_report, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/acl2_v83tf_clue_sufficiency_vs_action_misuse"),
    )
    parser.add_argument("--decision", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    decision_path = args.decision or args.root / "phase10_decision_matrix" / "final_decision.json"
    out_dir = args.out_dir or args.root / "phase20_final_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    decision = read_json(decision_path)
    final_report, markdown = build_report(decision, decision_path)

    json_path = out_dir / "final_report.json"
    md_path = out_dir / "final_report.md"
    json_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "json_path": str(json_path),
                "md_path": str(md_path),
                "final_status": final_report["final_status"],
                "runtime_action_allowed": final_report["runtime_action_allowed"],
                "question_count": len(final_report["questions"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
