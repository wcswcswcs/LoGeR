#!/usr/bin/env python3
"""Summarize ACL2 v83 Phase4 counterfactual upper-bound artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase4_counterfactual_upper_bound"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def fnum(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    rows_path = args.root / "counterfactual_upper_bound_rows.csv"
    summary_path = args.root / "counterfactual_upper_bound_summary.json"
    rows = read_csv(rows_path)
    summary = read_json(summary_path)
    passing = [row for row in rows if truthy(row.get("counterfactual_gate_pass"))]
    non_invalid_passing = [row for row in passing if not truthy(row.get("invalid_as_runtime_method"))]
    best = summary.get("best_bad_improvement", {})

    lines = [
        "# ACL2 v83 Phase4 Counterfactual Upper-Bound Summary",
        "",
        f"phase4_gate_pass: `{summary.get('phase4_gate_pass')}`",
        f"runtime_action_allowed: `{summary.get('runtime_action_allowed')}`",
        f"decision: `{summary.get('decision')}`",
        f"non_invalid_passing_count: `{summary.get('non_invalid_passing_count')}`",
        f"oracle_passing_count: `{summary.get('oracle_passing_count')}`",
        "",
        "## Best Bad Improvement",
        "",
        f"- family: `{best.get('family', '')}`",
        f"- candidate_or_controller: `{best.get('candidate_or_controller', '')}`",
        f"- metric: `{best.get('metric', '')}`",
        f"- bad_median_improvement_vs_baseline_ratio: `{best.get('bad_median_improvement_vs_baseline_ratio', '')}`",
        f"- good_max_worsen_vs_baseline_ratio: `{best.get('good_max_worsen_vs_baseline_ratio', '')}`",
        f"- bad_control_beat_count: `{best.get('bad_control_beat_count', '')}`",
        f"- invalid_as_runtime_method: `{best.get('invalid_as_runtime_method', '')}`",
        "",
        "## Passing Rows",
        "",
    ]
    if passing:
        for row in passing:
            lines.append(
                "- `{family}` `{candidate}` `{metric}` bad={bad} good_worsen={good} invalid={invalid}".format(
                    family=row.get("family", ""),
                    candidate=row.get("candidate_or_controller", ""),
                    metric=row.get("metric_alias") or row.get("metric", ""),
                    bad=fnum(row.get("bad_median_improvement_vs_baseline_ratio")),
                    good=fnum(row.get("good_max_worsen_vs_baseline_ratio")),
                    invalid=row.get("invalid_as_runtime_method", ""),
                )
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Runtime Eligibility",
            "",
            (
                "No runtime action is eligible."
                if not non_invalid_passing or not summary.get("phase3_gate_pass")
                else "A non-invalid counterfactual row exists and Phase3 passed."
            ),
            "",
            f"stop_reason: `{summary.get('stop_reason', '')}`",
            "",
        ]
    )
    out_path = args.root / "counterfactual_upper_bound_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_path": str(out_path), "phase4_gate_pass": summary.get("phase4_gate_pass")}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
