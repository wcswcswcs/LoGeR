#!/usr/bin/env python3
"""Summarize v82 SWA carrier ledger and decide Phase4 gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase4_swa_carrier_ledger"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _family_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    seqs = sorted({row.get("seq", "") for row in rows if row.get("seq", "")})
    visual_ok = all(_truthy(row.get("carrier_visual_evidence_confirmed")) for row in rows)
    per_head = all(_truthy(row.get("per_head_available")) for row in rows)
    same_head_random = all(_truthy(row.get("same_head_random_available")) for row in rows)
    shuffled = all(_truthy(row.get("shuffled_semantic_available")) for row in rows)
    bad = [row for row in rows if row.get("base_case_type") == "bad"]
    good = [row for row in rows if row.get("base_case_type") == "good"]
    bad_deltas = [_float(row.get("stable_alignment_delta")) for row in bad]
    bad_deltas = [value for value in bad_deltas if value is not None]
    same_sign_bad = bool(bad_deltas) and (all(value >= 0 for value in bad_deltas) or all(value <= 0 for value in bad_deltas))
    good_false_positive_rate = (
        sum(_truthy(row.get("good_case_false_positive")) for row in good) / len(good)
        if good
        else None
    )
    good_ok = good_false_positive_rate is not None and good_false_positive_rate <= 0.02
    gate = {
        "coverage_ge_3_sequences": len(seqs) >= 3,
        "carrier_visual_evidence_confirmed": visual_ok,
        "selected_vs_random_same_sign_on_bad": same_sign_bad,
        "good_false_positive_rate_le_2pct": good_ok,
        "per_head_route_available": per_head,
        "same_head_random_available": same_head_random,
        "shuffled_semantic_control_available": shuffled,
    }
    gate["carrier_family_gate_pass"] = all(gate.values())
    return {
        "seq_coverage": seqs,
        "rows": len(rows),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "good_false_positive_rate": good_false_positive_rate,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    ledger = _read_csv(args.root / "swa_carrier_ledger.csv")
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ledger:
        by_family[row.get("carrier_family", "")].append(row)
    family_decisions = {family: _family_gate(rows) for family, rows in sorted(by_family.items())}
    passing = [family for family, item in family_decisions.items() if item["gate"]["carrier_family_gate_pass"]]
    gate = {
        "ledger_rows_ge_96": len(ledger) >= 96,
        "families_present": sorted(by_family),
        "at_least_one_carrier_family_passes": bool(passing),
        "coverage_ge_3_sequences": all(item["gate"]["coverage_ge_3_sequences"] for item in family_decisions.values()),
        "carrier_visual_evidence_confirmed": all(item["gate"]["carrier_visual_evidence_confirmed"] for item in family_decisions.values()),
    }
    gate["phase4_gate_pass"] = bool(gate["ledger_rows_ge_96"] and gate["at_least_one_carrier_family_passes"])
    decision = {
        "schema": "acl2_v82_swa_carrier_ledger_decision_v1",
        "rows": len(ledger),
        "family_counts": dict(Counter(row.get("carrier_family", "") for row in ledger)),
        "family_decisions": family_decisions,
        "passing_families": passing,
        "gate": gate,
        "decision": "pass_to_phase5" if gate["phase4_gate_pass"] else "fail_to_phase8_merge_gauge_diagnostic",
        "blocker": (
            ""
            if gate["phase4_gate_pass"]
            else "SWA semantic carrier is not localized: current evidence lacks per-head route, same-head random, and shuffled semantic controls."
        ),
    }
    (args.root / "swa_carrier_ledger_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# v82 Phase4 SWA Carrier Ledger Decision",
        "",
        f"rows: {decision['rows']}",
        f"decision: {decision['decision']}",
        f"blocker: {decision['blocker'] or 'none'}",
        "",
        "## Family Gates",
    ]
    for family, item in family_decisions.items():
        report.extend(
            [
                "",
                f"### {family}",
                f"rows: {item['rows']}",
                f"seq_coverage: {item['seq_coverage']}",
                f"good_false_positive_rate: {item['good_false_positive_rate']}",
                f"gate: {item['gate']}",
            ]
        )
    (args.root / "swa_carrier_ledger_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
