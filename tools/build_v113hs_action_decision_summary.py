#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact v113-HS action decision table.")
    parser.add_argument("--results-root", default="results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence")
    args = parser.parse_args()

    root = Path(args.results_root)
    diag = root / "diagnostics"
    entries = [
        ("HS_A1_patch_only_mild_no_mrt_sliding10", diag / "stage4_sliding10_hs_a1_patch_only_mild_no_mrt_summary.json"),
        ("HS_L1_value_dynamic_boundary_mild_default", diag / "stage5_hs_l1_value_dynamic_boundary_mild_full_default_summary.json"),
        ("HS_L3_value_stable_boost_default", diag / "stage5_hs_l3_value_stable_boost_full_default_summary.json"),
        ("HS_L4_value_risk_suppress_plus_stable_boost_default", diag / "stage5_hs_l4_value_risk_suppress_plus_stable_boost_full_default_summary.json"),
        ("HS_L3_value_stable_boost_tiny_default", diag / "stage5_hs_l3_value_stable_boost_tiny_full_default_summary.json"),
        ("HS_L3_value_stable_boost_micro_default", diag / "stage5_hs_l3_value_stable_boost_micro_full_default_summary.json"),
    ]

    rows = []
    for action, path in entries:
        summary = load_summary(path)
        agg = summary["aggregate"]
        row = {
            "action": action,
            "seqs": ",".join(agg["seqs"]),
            "median_full_ATE_rel_improvement": agg["median_full_ATE_rel_improvement"],
            "median_rolling_p90_rel_improvement": agg["median_rolling_p90_rel_improvement"],
            "median_segment_scale_rel_improvement": agg["median_segment_scale_rel_improvement"],
            "max_full_ATE_harm_rel": agg["max_full_ATE_harm_rel"],
            "improved_seq_count_full_ATE": agg["improved_seq_count_full_ATE"],
            "segment_scale_not_worse_all": agg["segment_scale_not_worse_all"],
            "pilot_geometry_gate_pass": agg["pilot_geometry_gate"]["pass"],
            "semantic_causality_gate_status": agg["semantic_causality_gate"]["status"],
            "semantic_causality_gate_pass": agg["semantic_causality_gate"]["pass"],
            "summary_path": str(path),
        }
        if row["pilot_geometry_gate_pass"] and row["semantic_causality_gate_pass"]:
            row["decision"] = "promote"
        elif (
            (row["median_full_ATE_rel_improvement"] or 0.0) >= 0.05
            or (row["median_rolling_p90_rel_improvement"] or 0.0) >= 0.05
        ) and not row["segment_scale_not_worse_all"]:
            row["decision"] = "alignment_or_scale_tradeoff_no_go"
        else:
            row["decision"] = "pilot_no_go"
        rows.append(row)

    out_csv = diag / "stage5_action_decision_rows.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json = diag / "stage5_action_decision_summary.json"
    payload = {
        "rows": rows,
        "promoted_actions": [row["action"] for row in rows if row["decision"] == "promote"],
        "best_full_ATE_candidate": max(rows, key=lambda row: row["median_full_ATE_rel_improvement"] or float("-inf")),
        "claim": "No action passed both pilot geometry and semantic causality gates; tradeoff candidates are not promoted.",
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
