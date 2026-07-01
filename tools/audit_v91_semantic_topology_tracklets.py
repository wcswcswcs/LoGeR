#!/usr/bin/env python3
"""Audit v91 semantic topology tracklet construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import read_json, write_csv, write_json
from v91_semantic_regime_utils import ROOT


DEFAULT_DIR = ROOT / "phase1_semantic_topology_tracklets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklet-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.tracklet_dir / "semantic_topology_tracklet_summary.json") if (args.tracklet_dir / "semantic_topology_tracklet_summary.json").exists() else {}
    rows = pd.read_csv(args.tracklet_dir / "semantic_topology_tracklet_rows.csv") if (args.tracklet_dir / "semantic_topology_tracklet_rows.csv").exists() else pd.DataFrame()
    pair_summary = pd.read_csv(args.tracklet_dir / "semantic_topology_tracklet_pair_summary.csv") if (args.tracklet_dir / "semantic_topology_tracklet_pair_summary.csv").exists() else pd.DataFrame()
    checks = [
        {"check": "pair_rows_ge_49", "pass": summary.get("pair_rows", 0) >= 49, "value": summary.get("pair_rows")},
        {"check": "sequence_coverage_ge_4", "pass": summary.get("sequence_coverage", 0) >= 4, "value": summary.get("sequence_coverage")},
        {"check": "pairs_with_tracklets_ratio_ge_0_90", "pass": summary.get("pairs_with_tracklets_ratio", 0.0) >= 0.90, "value": summary.get("pairs_with_tracklets_ratio")},
        {"check": "tracklet_rows_ge_pair_rows_x5", "pass": summary.get("tracklet_rows", 0) >= summary.get("pair_rows", 0) * 5, "value": summary.get("tracklet_rows")},
        {"check": "raw_or_match_backed_tracklet_ratio_ge_0_80", "pass": summary.get("raw_or_match_backed_tracklet_ratio", 0.0) >= 0.80, "value": summary.get("raw_or_match_backed_tracklet_ratio")},
        {"check": "semantic_confidence_available_ratio_ge_0_90", "pass": summary.get("semantic_confidence_available_ratio", 0.0) >= 0.90, "value": summary.get("semantic_confidence_available_ratio")},
        {"check": "tracklet_type_coverage_ratio_ge_0_90", "pass": summary.get("tracklet_type_coverage_ratio", 0.0) >= 0.90, "value": summary.get("tracklet_type_coverage_ratio")},
        {"check": "zero_conf_positive_tracklet_ratio_le_0_05", "pass": summary.get("zero_conf_positive_tracklet_ratio", 1.0) <= 0.05, "value": summary.get("zero_conf_positive_tracklet_ratio")},
        {"check": "compact_label_status_recorded", "pass": summary.get("label_mapping_status") == "compact_project_local_id_no_class_names", "value": summary.get("label_mapping_status")},
        {"check": "no_radio_track_fabricated", "pass": summary.get("has_radio") is False and summary.get("has_track") is False, "value": f"has_radio={summary.get('has_radio')} has_track={summary.get('has_track')}"},
    ]
    gate = bool(all(row["pass"] for row in checks))
    audit = {
        "phase": "Phase1_semantic_topology_tracklet_audit",
        "phase1_tracklet_audit_gate_pass": gate,
        "tracklet_rows": int(len(rows)),
        "pair_rows": int(pair_summary["pair_id"].nunique()) if len(pair_summary) else 0,
        "sequence_coverage": int(rows["seq"].astype(str).str.zfill(2).nunique()) if len(rows) else 0,
        "tracklet_type_counts": rows["tracklet_type"].value_counts().to_dict() if len(rows) else {},
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        audit["blocker"] = "semantic_topology_tracklet_audit_failed"
    write_csv(args.tracklet_dir / "semantic_topology_tracklet_audit_checks.csv", checks)
    write_json(args.tracklet_dir / "semantic_topology_tracklet_audit.json", audit)
    print(f"phase1_tracklet_audit_gate_pass={audit['phase1_tracklet_audit_gate_pass']}")
    print(f"tracklet_rows={audit['tracklet_rows']}")
    print(f"pair_rows={audit['pair_rows']}")
    print(f"sequence_coverage={audit['sequence_coverage']}")
    print(f"tracklet_type_counts={audit['tracklet_type_counts']}")
    if audit.get("blocker"):
        print(f"blocker={audit['blocker']}")


if __name__ == "__main__":
    main()
