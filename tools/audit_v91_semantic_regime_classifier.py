#!/usr/bin/env python3
"""Audit v91 semantic regime classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import read_json, write_csv, write_json
from v91_semantic_regime_utils import ROOT


DEFAULT_DIR = ROOT / "phase2_semantic_regime_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.regime_dir / "semantic_regime_rows.csv")
    summary = read_json(args.regime_dir / "semantic_regime_summary.json")
    checks = [
        {"check": "regime_rows_equal_pair_rows", "pass": summary.get("regime_rows") == summary.get("pair_rows"), "value": f"{summary.get('regime_rows')}/{summary.get('pair_rows')}"},
        {"check": "sequence_coverage_ge_4", "pass": summary.get("sequence_coverage", 0) >= 4, "value": summary.get("sequence_coverage")},
        {"check": "known_regime_ratio_ge_0_90", "pass": summary.get("known_regime_ratio", 0) >= 0.90, "value": summary.get("known_regime_ratio")},
        {"check": "single_regime_not_gt_70pct", "pass": summary.get("max_single_regime_ratio", 1) <= 0.70, "value": summary.get("max_single_regime_ratio")},
        {"check": "bad_good_not_used", "pass": not bool(summary.get("bad_good_label_used_for_assignment")), "value": summary.get("bad_good_label_used_for_assignment")},
        {"check": "scale_label_not_used", "pass": not bool(summary.get("scale_label_used_for_assignment")), "value": summary.get("scale_label_used_for_assignment")},
        {"check": "regime_confidence_available_ge_0_90", "pass": summary.get("regime_confidence_available_ratio", 0) >= 0.90, "value": summary.get("regime_confidence_available_ratio")},
    ]
    gate = bool(all(row["pass"] for row in checks))
    audit = {
        "phase": "Phase2_semantic_regime_classifier_audit",
        "phase2_regime_classifier_audit_gate_pass": gate,
        "regime_rows": int(len(rows)),
        "sequence_coverage": int(rows["seq"].astype(str).str.zfill(2).nunique()),
        "regime_counts": rows["regime"].value_counts().to_dict(),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        audit["blocker"] = "semantic_regime_classifier_audit_failed"
    write_csv(args.regime_dir / "semantic_regime_audit_checks.csv", checks)
    write_json(args.regime_dir / "semantic_regime_audit.json", audit)
    print(f"phase2_regime_classifier_audit_gate_pass={audit['phase2_regime_classifier_audit_gate_pass']}")
    print(f"regime_rows={audit['regime_rows']}")
    print(f"sequence_coverage={audit['sequence_coverage']}")
    print(f"regime_counts={audit['regime_counts']}")
    if audit.get("blocker"):
        print(f"blocker={audit['blocker']}")


if __name__ == "__main__":
    main()
