#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_aggregate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["aggregate"]


def decision_for(row: dict[str, Any], target: dict[str, Any] | None) -> str:
    if row["role"] == "target":
        if row["pilot_geometry_gate_pass"]:
            return "candidate_requires_semantic_controls"
        return "pilot_no_go"
    if row["role"] == "control":
        if not row["pilot_geometry_gate_pass"]:
            return "control_not_matching"
        if target is None:
            return "control_pass_no_target"
        target_ate = target["median_full_ATE_rel_improvement"]
        target_roll = target["median_rolling_p90_rel_improvement"]
        if (
            row["median_full_ATE_rel_improvement"] >= target_ate
            or row["median_rolling_p90_rel_improvement"] >= target_roll
        ):
            return "control_matches_or_exceeds_target"
        return "control_pass_but_below_target"
    if row["pilot_geometry_gate_pass"]:
        return "diagnostic_pass_not_promoted"
    if row["segment_scale_not_worse_all"] and row["median_segment_scale_rel_improvement"] > 0:
        return "scale_only_pose_harm_no_go"
    return "pilot_no_go"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v113-HS Stage6 decision summary.")
    parser.add_argument("--results-root", default="results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence")
    args = parser.parse_args()

    root = Path(args.results_root)
    diag = root / "diagnostics"
    entries = [
        ("target", "HS_L3_tiny_norm05_plus_HS_M1_default", diag / "stage6_hs_l3_tiny_norm05_plus_m1_full_default_summary.json"),
        ("control", "ctrl_semantic_shuffle", diag / "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_semantic_shuffle_full_default_summary.json"),
        ("control", "ctrl_same_count_high_risk_frame_random", diag / "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_same_count_high_risk_frame_random_full_default_summary.json"),
        ("control", "ctrl_role_rotation_dynamic_stable", diag / "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_role_rotation_full_default_summary.json"),
        ("control", "ctrl_low_risk_reverse", diag / "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_low_risk_reverse_full_default_summary.json"),
        ("repair", "HS_L3_tiny_normpreserve_plus_HS_M1_default", diag / "stage6_hs_l3_tiny_normpreserve_plus_m1_full_default_summary.json"),
        ("repair", "HS_L5_stable_centered_tiny_plus_HS_M1_default", diag / "stage6_hs_l5_stable_centered_tiny_plus_m1_full_default_summary.json"),
        ("ablation", "HS_L3_tiny_norm05_ctrl_semantic_shuffle_no_M1", diag / "stage6_hs_l3_tiny_norm05_ctrl_semantic_shuffle_full_default_summary.json"),
    ]

    target: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for role, name, path in entries:
        agg = load_aggregate(path)
        row = {
            "role": role,
            "name": name,
            "seqs": ",".join(agg["seqs"]),
            "median_full_ATE_rel_improvement": agg["median_full_ATE_rel_improvement"],
            "median_rolling_p90_rel_improvement": agg["median_rolling_p90_rel_improvement"],
            "median_segment_scale_rel_improvement": agg["median_segment_scale_rel_improvement"],
            "max_full_ATE_harm_rel": agg["max_full_ATE_harm_rel"],
            "improved_seq_count_full_ATE": agg["improved_seq_count_full_ATE"],
            "segment_scale_not_worse_all": agg["segment_scale_not_worse_all"],
            "pilot_geometry_gate_pass": agg["pilot_geometry_gate"]["pass"],
            "summary_path": str(path),
        }
        if role == "target":
            target = row
        rows.append(row)

    for row in rows:
        row["decision"] = decision_for(row, target)

    control_matches = [row for row in rows if row["role"] == "control" and row["decision"] == "control_matches_or_exceeds_target"]
    promoted_actions: list[str] = []
    if target and target["pilot_geometry_gate_pass"] and not control_matches:
        promoted_actions.append(target["name"])
    claim = (
        "HS_GEOMETRY_PASS_SEMANTIC_CAUSALITY_FAIL: target pilot geometry passed, "
        "but semantic_shuffle and same_count controls matched or exceeded the target."
        if control_matches
        else "No Stage6 target was promoted."
    )

    csv_path = diag / "stage6_action_decision_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "rows": rows,
        "target": target,
        "control_matches_or_exceeds_target": [row["name"] for row in control_matches],
        "promoted_actions": promoted_actions,
        "claim": claim,
    }
    json_path = diag / "stage6_action_decision_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
