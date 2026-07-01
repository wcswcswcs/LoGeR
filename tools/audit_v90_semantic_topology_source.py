#!/usr/bin/env python3
"""Audit v90 Phase1 semantic topology source construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import read_json, write_csv, write_json
from v90_semantic_topology_utils import ROOT


DEFAULT_SOURCE = ROOT / "phase1_semantic_topology_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_summary = read_json(args.source_dir / "topology_node_summary.json") if (args.source_dir / "topology_node_summary.json").exists() else {}
    edge_summary = read_json(args.source_dir / "topology_source_summary.json") if (args.source_dir / "topology_source_summary.json").exists() else {}
    nodes = pd.read_csv(args.source_dir / "topology_nodes.csv") if (args.source_dir / "topology_nodes.csv").exists() else pd.DataFrame()
    edges = pd.read_csv(args.source_dir / "topology_edges.csv") if (args.source_dir / "topology_edges.csv").exists() else pd.DataFrame()
    pair_summary = pd.read_csv(args.source_dir / "topology_pair_summary.csv") if (args.source_dir / "topology_pair_summary.csv").exists() else pd.DataFrame()
    label_statuses = sorted(set(nodes.get("label_mapping_status", pd.Series(dtype=str)).astype(str).tolist()))
    audit_checks = [
        {"check": "pair_rows_ge_49", "pass": edge_summary.get("pair_rows", 0) >= 49, "value": edge_summary.get("pair_rows")},
        {"check": "sequence_coverage_ge_4", "pass": edge_summary.get("sequence_coverage", 0) >= 4, "value": edge_summary.get("sequence_coverage")},
        {"check": "node_rows_positive_all_sequences", "pass": node_summary.get("node_rows", 0) > 0 and node_summary.get("node_rows_all_sequences", 0) >= 4, "value": node_summary.get("node_rows")},
        {"check": "topology_edges_positive_90pct_pairs", "pass": edge_summary.get("pairs_with_topology_edges", 0) >= 0.90 * max(edge_summary.get("pair_rows", 0), 1), "value": edge_summary.get("pairs_with_topology_edges")},
        {"check": "semantic_confidence_available_ratio_ge_0_90", "pass": edge_summary.get("semantic_confidence_available_ratio", 0.0) >= 0.90, "value": edge_summary.get("semantic_confidence_available_ratio")},
        {"check": "component_boundary_available_ratio_ge_0_90", "pass": edge_summary.get("component_boundary_available_ratio", 0.0) >= 0.90, "value": edge_summary.get("component_boundary_available_ratio")},
        {"check": "feature_match_or_raw_overlap_support_available_ratio_ge_0_90", "pass": edge_summary.get("feature_match_or_raw_overlap_support_available_ratio", 0.0) >= 0.90, "value": edge_summary.get("feature_match_or_raw_overlap_support_available_ratio")},
        {"check": "label_mapping_status_recorded", "pass": bool(label_statuses) and all(bool(x) for x in label_statuses), "value": "|".join(label_statuses)},
        {"check": "no_class_name_claim_without_mapping", "pass": not bool(edge_summary.get("class_name_available", False)), "value": edge_summary.get("class_name_available")},
    ]
    gate = bool(all(row["pass"] for row in audit_checks))
    summary = {
        "phase": "Phase1_semantic_topology_source_audit",
        "phase1_topology_source_gate_pass": gate,
        "pair_rows": int(edge_summary.get("pair_rows", 0)),
        "sequence_coverage": int(edge_summary.get("sequence_coverage", 0)),
        "node_rows": int(len(nodes)),
        "topology_edge_rows": int(len(edges)),
        "topology_pair_rows": int(len(pair_summary)),
        "semantic_confidence_available_ratio": edge_summary.get("semantic_confidence_available_ratio"),
        "component_boundary_available_ratio": edge_summary.get("component_boundary_available_ratio"),
        "feature_match_or_raw_overlap_support_available_ratio": edge_summary.get("feature_match_or_raw_overlap_support_available_ratio"),
        "label_mapping_statuses": label_statuses,
        "class_name_available": False,
        "has_radio": False,
        "has_track": False,
        "patch_radius": edge_summary.get("patch_radius"),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "semantic_source_topology_insufficient"
    write_csv(args.source_dir / "topology_source_audit_checks.csv", audit_checks)
    write_json(args.source_dir / "topology_source_audit.json", summary)
    print(f"phase1_topology_source_gate_pass={summary['phase1_topology_source_gate_pass']}")
    print(f"pair_rows={summary['pair_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"node_rows={summary['node_rows']}")
    print(f"topology_edge_rows={summary['topology_edge_rows']}")
    print(f"semantic_confidence_available_ratio={summary['semantic_confidence_available_ratio']}")
    print(f"component_boundary_available_ratio={summary['component_boundary_available_ratio']}")
    print(f"feature_match_or_raw_overlap_support_available_ratio={summary['feature_match_or_raw_overlap_support_available_ratio']}")
    print(f"patch_radius={summary['patch_radius']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
