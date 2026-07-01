#!/usr/bin/env python3
"""Audit v90 Phase2 topology scale-mode ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import read_json, write_csv, write_json
from v90_semantic_topology_utils import ROOT


DEFAULT_LEDGER = ROOT / "phase2_semantic_topology_scale_mode_ledger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.ledger_dir / "phase2_topology_ledger_summary.json") if (args.ledger_dir / "phase2_topology_ledger_summary.json").exists() else {}
    modes = pd.read_csv(args.ledger_dir / "topology_mode_rows.csv") if (args.ledger_dir / "topology_mode_rows.csv").exists() else pd.DataFrame()
    pairs = pd.read_csv(args.ledger_dir / "topology_pair_rows.csv") if (args.ledger_dir / "topology_pair_rows.csv").exists() else pd.DataFrame()
    counts = modes["topology_mode_type"].value_counts().to_dict() if len(modes) and "topology_mode_type" in modes else {}
    labelled = pairs[pd.to_numeric(pairs.get("abs_log_scale_jump_gt", pd.Series(dtype=float)), errors="coerce").notna()] if len(pairs) else pd.DataFrame()
    checks = [
        {"check": "pair_rows_ge_49", "pass": int(len(pairs)) >= 49, "value": int(len(pairs))},
        {"check": "mode_rows_positive", "pass": int(len(modes)) > 0, "value": int(len(modes))},
        {"check": "valid_support_or_conflict_ge_20pct", "pass": summary.get("valid_support_or_conflict_mode_ratio", 0.0) >= 0.20, "value": summary.get("valid_support_or_conflict_mode_ratio")},
        {"check": "invalid_conflict_ge_20pct", "pass": summary.get("invalid_conflict_mode_ratio", 0.0) >= 0.20, "value": summary.get("invalid_conflict_mode_ratio")},
        {"check": "topology_coverage_ge_90pct", "pass": summary.get("topology_coverage_ratio", 0.0) >= 0.90, "value": summary.get("topology_coverage_ratio")},
        {"check": "semantic_shuffle_controls_available", "pass": bool(summary.get("semantic_shuffle_controls_generated", False)), "value": summary.get("semantic_shuffle_controls_generated")},
        {"check": "component_shuffle_controls_available", "pass": bool(summary.get("component_shuffle_controls_generated", False)), "value": summary.get("component_shuffle_controls_generated")},
        {"check": "scale_labels_audit_only", "pass": bool(pairs.get("offline_audit_label_only", pd.Series([True])).astype(str).str.lower().isin(["true", "1"]).all()) if len(pairs) else False, "value": int(len(labelled))},
    ]
    gate = bool(all(row["pass"] for row in checks))
    audit = {
        "phase": "Phase2_semantic_topology_scale_mode_ledger_audit",
        "phase2_topology_ledger_audit_gate_pass": gate,
        "pair_rows": int(len(pairs)),
        "mode_rows": int(len(modes)),
        "labelled_pair_rows": int(len(labelled)),
        "sequence_coverage": int(pairs["seq"].astype(str).str.zfill(2).nunique()) if len(pairs) else 0,
        "topology_mode_type_counts": counts,
        "valid_support_or_conflict_mode_ratio": summary.get("valid_support_or_conflict_mode_ratio"),
        "invalid_conflict_mode_ratio": summary.get("invalid_conflict_mode_ratio"),
        "topology_coverage_ratio": summary.get("topology_coverage_ratio"),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        audit["blocker"] = "topology_scale_mode_ledger_audit_failed"
    write_csv(args.ledger_dir / "topology_ledger_audit_checks.csv", checks)
    write_json(args.ledger_dir / "phase2_topology_ledger_audit.json", audit)
    print(f"phase2_topology_ledger_audit_gate_pass={audit['phase2_topology_ledger_audit_gate_pass']}")
    print(f"pair_rows={audit['pair_rows']}")
    print(f"mode_rows={audit['mode_rows']}")
    print(f"sequence_coverage={audit['sequence_coverage']}")
    print(f"topology_mode_type_counts={audit['topology_mode_type_counts']}")
    if audit.get("blocker"):
        print(f"blocker={audit['blocker']}")


if __name__ == "__main__":
    main()
