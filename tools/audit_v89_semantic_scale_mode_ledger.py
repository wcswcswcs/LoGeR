#!/usr/bin/env python3
"""Audit v89 Phase1 semantic scale-mode ledger outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import read_json, write_json


DEFAULT_DIR = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/phase1_semantic_scale_mode_ledger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.ledger_dir / "phase1_semantic_ledger_summary.json")
    pair = pd.read_csv(args.ledger_dir / "semantic_scale_pair_rows.csv")
    mode = pd.read_csv(args.ledger_dir / "semantic_scale_mode_rows.csv")
    edge_header = pd.read_csv(args.ledger_dir / "semantic_scale_mode_edge_rows.csv", nrows=10)
    required_edge_cols = {
        "seq",
        "prev_chunk",
        "curr_chunk",
        "raw_pair_index_a",
        "raw_pair_index_b",
        "prev_patch_yx",
        "curr_patch_yx",
        "signed_log_shape_ratio",
        "mode_id",
        "prev_label",
        "curr_label",
        "prev_conf",
        "curr_conf",
        "semantic_conf_mean",
        "semantic_purity",
        "zero_conf_flag",
        "low_conf_flag",
    }
    required_mode_cols = {
        "mode_id",
        "mode_center_mu",
        "mode_mad",
        "mode_mass",
        "H_sem",
        "S_valid",
        "S_invalid",
        "S_context",
        "S_lowobs",
        "semantic_mode_type",
    }
    required_pair_cols = {
        "geometry_dominant_mode_mu",
        "semantic_valid_dominant_mode_mu",
        "semantic_invalid_dominant_mode_mu",
        "semantic_valid_mass",
        "semantic_invalid_mass",
        "O_sem_scale",
        "offline_audit_label_only",
        "no_gt_runtime_feature",
    }
    audit = {
        "phase": "Phase1_semantic_scale_mode_ledger_audit",
        "phase1_audit_gate_pass": bool(
            summary.get("phase1_gate_pass")
            and required_edge_cols.issubset(edge_header.columns)
            and required_mode_cols.issubset(mode.columns)
            and required_pair_cols.issubset(pair.columns)
            and len(pair) >= 49
            and pair["seq"].astype(str).str.zfill(2).nunique() >= 4
            and len(mode) > 0
        ),
        "summary_phase1_gate_pass": summary.get("phase1_gate_pass"),
        "pair_rows": int(len(pair)),
        "mode_rows": int(len(mode)),
        "sequence_coverage": int(pair["seq"].astype(str).str.zfill(2).nunique()),
        "edge_required_columns_present": required_edge_cols.issubset(edge_header.columns),
        "mode_required_columns_present": required_mode_cols.issubset(mode.columns),
        "pair_required_columns_present": required_pair_cols.issubset(pair.columns),
        "semantic_mode_type_counts": mode["semantic_mode_type"].value_counts().to_dict(),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not audit["phase1_audit_gate_pass"]:
        audit["blocker"] = "phase1_semantic_ledger_audit_failed"
    write_json(args.ledger_dir / "phase1_semantic_ledger_audit.json", audit)
    print(f"phase1_audit_gate_pass={audit['phase1_audit_gate_pass']}")
    print(f"summary_phase1_gate_pass={audit['summary_phase1_gate_pass']}")
    print(f"pair_rows={audit['pair_rows']}")
    print(f"mode_rows={audit['mode_rows']}")
    print(f"sequence_coverage={audit['sequence_coverage']}")
    print(f"semantic_mode_type_counts={audit['semantic_mode_type_counts']}")
    if audit.get("blocker"):
        print(f"blocker={audit['blocker']}")


if __name__ == "__main__":
    main()
