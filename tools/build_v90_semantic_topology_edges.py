#!/usr/bin/env python3
"""Build v90 Phase1 semantic topology edges from component maps and feature support."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v90_semantic_topology_utils import (
    ROOT,
    V89_FEATURE,
    V89_LEDGER,
    build_components_for_side,
    load_raw,
    nearest_component_for_point,
    pair_id,
    patch_coords,
    seq_norm,
)


DEFAULT_OUT = ROOT / "phase1_semantic_topology_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v89-ledger-dir", type=Path, default=V89_LEDGER)
    parser.add_argument("--v89-feature-dir", type=Path, default=V89_FEATURE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--patch-radius", type=int, default=0)
    return parser.parse_args()


def _feature_counts(path: Path) -> dict[tuple[str, int, int, int, int], int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    counts: dict[tuple[str, int, int, int, int], int] = defaultdict(int)
    for _, row in df.iterrows():
        seq = seq_norm(row["seq"])
        prev = int(row["prev_chunk"])
        curr = int(row["curr_chunk"])
        pl = int(row["semantic_prev_label"])
        cl = int(row["semantic_curr_label"])
        counts[(seq, prev, curr, pl, cl)] += 1
    return counts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.v89_ledger_dir / "semantic_scale_pair_rows.csv")
    feature_counts = _feature_counts(args.v89_feature_dir / "feature_match_semantic_rows.csv")
    edge_acc: dict[tuple[Any, ...], dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    raw_missing = 0
    for _, pair in pairs.iterrows():
        seq = seq_norm(pair["seq"])
        prev = int(pair["prev_chunk"])
        curr = int(pair["curr_chunk"])
        pid = pair_id(seq, prev, curr)
        source_path = str(pair["source_path"])
        raw = load_raw(source_path)
        if raw is None:
            raw_missing += 1
            pair_rows.append(
                {
                    "seq": seq,
                    "prev_chunk": prev,
                    "curr_chunk": curr,
                    "pair_id": pid,
                    "source_path": source_path,
                    "raw_available": False,
                    "topology_edge_rows": 0,
                    "raw_overlap_support_count": 0,
                    "semantic_confidence_available": False,
                    "component_boundary_available": False,
                    "feature_match_or_raw_overlap_support_available": False,
                    "patch_radius": args.patch_radius,
                }
            )
            continue
        prev_rows, prev_map, _ = build_components_for_side(raw, "prev", radius=args.patch_radius)
        curr_rows, curr_map, _ = build_components_for_side(raw, "curr", radius=args.patch_radius)
        prev_cells = patch_coords(raw["prev_pixel_coords"], radius=0)
        curr_cells = patch_coords(raw["curr_pixel_coords"], radius=0)
        prev_labels = np.asarray(raw["prev_semantic_labels"]).astype(int)
        curr_labels = np.asarray(raw["curr_semantic_labels"]).astype(int)
        prev_conf = np.asarray(raw["prev_semantic_conf"]).astype(float)
        curr_conf = np.asarray(raw["curr_semantic_conf"]).astype(float)
        n = min(len(prev_cells), len(curr_cells), len(prev_labels), len(curr_labels))
        missing_component = 0
        pair_edge_keys: set[tuple[Any, ...]] = set()
        for i in range(n):
            pcell = prev_cells[i][0] if prev_cells[i] else (0, 0)
            ccell = curr_cells[i][0] if curr_cells[i] else (0, 0)
            pcomp = nearest_component_for_point(prev_map, pcell)
            ccomp = nearest_component_for_point(curr_map, ccell)
            if pcomp is None or ccomp is None:
                missing_component += 1
                continue
            plabel = int(prev_labels[i])
            clabel = int(curr_labels[i])
            same_label = plabel == clabel and plabel > 0
            prev_boundary = float(pcomp["boundary_ratio"]) >= 0.5
            curr_boundary = float(ccomp["boundary_ratio"]) >= 0.5
            trans = "same_label_component" if same_label and pcomp["component_id"] == ccomp["component_id"] else ("same_label_cross_component" if same_label else "cross_label_boundary")
            if prev_boundary or curr_boundary:
                trans = f"{trans}_boundary"
            key = (
                seq,
                prev,
                curr,
                int(pcomp["label_compact_id"]),
                int(pcomp["component_id"]),
                int(ccomp["label_compact_id"]),
                int(ccomp["component_id"]),
                trans,
            )
            rec = edge_acc.setdefault(
                key,
                {
                    "seq": seq,
                    "prev_chunk": prev,
                    "curr_chunk": curr,
                    "pair_id": pid,
                    "prev_label_compact_id": int(pcomp["label_compact_id"]),
                    "prev_component_id": int(pcomp["component_id"]),
                    "curr_label_compact_id": int(ccomp["label_compact_id"]),
                    "curr_component_id": int(ccomp["component_id"]),
                    "component_transition_type": trans,
                    "raw_overlap_support_count": 0,
                    "same_label": bool(same_label),
                    "cross_component_boundary": bool(not same_label or prev_boundary or curr_boundary),
                    "prev_boundary_ratio_sum": 0.0,
                    "curr_boundary_ratio_sum": 0.0,
                    "semantic_conf_sum": 0.0,
                    "source_path": source_path,
                    "patch_radius": args.patch_radius,
                },
            )
            rec["raw_overlap_support_count"] += 1
            rec["prev_boundary_ratio_sum"] += float(pcomp["boundary_ratio"])
            rec["curr_boundary_ratio_sum"] += float(ccomp["boundary_ratio"])
            rec["semantic_conf_sum"] += 0.5 * (float(prev_conf[i]) + float(curr_conf[i]))
            pair_edge_keys.add(key)
        pair_records = [edge_acc[k] for k in pair_edge_keys]
        for rec in pair_records:
            support = max(int(rec["raw_overlap_support_count"]), 1)
            rec["prev_boundary_ratio"] = float(rec["prev_boundary_ratio_sum"] / support)
            rec["curr_boundary_ratio"] = float(rec["curr_boundary_ratio_sum"] / support)
            rec["mean_semantic_conf"] = float(rec["semantic_conf_sum"] / support)
            rec["feature_match_support_count"] = int(
                feature_counts.get(
                    (
                        seq,
                        prev,
                        curr,
                        int(rec["prev_label_compact_id"]),
                        int(rec["curr_label_compact_id"]),
                    ),
                    0,
                )
            )
        raw_support = sum(int(r["raw_overlap_support_count"]) for r in pair_records)
        same_support = sum(int(r["raw_overlap_support_count"]) for r in pair_records if bool(r["same_label"]))
        boundary_support = sum(int(r["raw_overlap_support_count"]) for r in pair_records if bool(r["cross_component_boundary"]))
        feature_support = sum(int(r.get("feature_match_support_count", 0)) for r in pair_records)
        pair_rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "pair_id": pid,
                "source_path": source_path,
                "raw_available": True,
                "topology_edge_rows": len(pair_records),
                "raw_overlap_support_count": raw_support,
                "same_label_support_ratio": float(same_support / max(raw_support, 1)),
                "cross_component_boundary_ratio": float(boundary_support / max(raw_support, 1)),
                "feature_match_support_count": feature_support,
                "semantic_confidence_available": bool(np.nanmean(np.r_[prev_conf, curr_conf]) > 0.0),
                "component_boundary_available": bool(len(prev_rows) > 0 and len(curr_rows) > 0),
                "feature_match_or_raw_overlap_support_available": bool(raw_support > 0 or feature_support > 0),
                "missing_component_point_count": int(missing_component),
                "patch_radius": args.patch_radius,
            }
        )
    edge_rows = []
    for rec in edge_acc.values():
        out = dict(rec)
        out.pop("prev_boundary_ratio_sum", None)
        out.pop("curr_boundary_ratio_sum", None)
        out.pop("semantic_conf_sum", None)
        edge_rows.append(out)
    write_csv(args.out_dir / "topology_edges.csv", edge_rows)
    write_csv(args.out_dir / "topology_pair_summary.csv", pair_rows)
    pair_df = pd.DataFrame(pair_rows)
    summary = {
        "phase": "Phase1_semantic_topology_edges",
        "pair_rows": int(len(pair_rows)),
        "sequence_coverage": int(pair_df["seq"].nunique()) if len(pair_df) else 0,
        "topology_edge_rows": int(len(edge_rows)),
        "pairs_with_topology_edges": int((pair_df["topology_edge_rows"] > 0).sum()) if len(pair_df) else 0,
        "semantic_confidence_available_ratio": float(pair_df["semantic_confidence_available"].astype(float).mean()) if len(pair_df) else 0.0,
        "component_boundary_available_ratio": float(pair_df["component_boundary_available"].astype(float).mean()) if len(pair_df) else 0.0,
        "feature_match_or_raw_overlap_support_available_ratio": float(pair_df["feature_match_or_raw_overlap_support_available"].astype(float).mean()) if len(pair_df) else 0.0,
        "raw_missing_pairs": int(raw_missing),
        "patch_radius": int(args.patch_radius),
        "label_mapping_status": "compact_project_local_id_no_class_names",
        "class_name_available": False,
        "has_radio": False,
        "has_track": False,
    }
    summary["edge_build_gate_pass"] = bool(
        summary["pair_rows"] >= 49
        and summary["sequence_coverage"] >= 4
        and summary["topology_edge_rows"] > 0
        and summary["pairs_with_topology_edges"] >= math.ceil(0.90 * summary["pair_rows"])
        and summary["semantic_confidence_available_ratio"] >= 0.90
        and summary["component_boundary_available_ratio"] >= 0.90
        and summary["feature_match_or_raw_overlap_support_available_ratio"] >= 0.90
    )
    if not summary["edge_build_gate_pass"]:
        summary["blocker"] = "semantic_topology_edge_source_insufficient"
    write_json(args.out_dir / "topology_source_summary.json", summary)
    print(f"edge_build_gate_pass={summary['edge_build_gate_pass']}")
    print(f"pair_rows={summary['pair_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"topology_edge_rows={summary['topology_edge_rows']}")
    print(f"pairs_with_topology_edges={summary['pairs_with_topology_edges']}")
    print(f"semantic_confidence_available_ratio={summary['semantic_confidence_available_ratio']}")
    print(f"component_boundary_available_ratio={summary['component_boundary_available_ratio']}")
    print(f"feature_match_or_raw_overlap_support_available_ratio={summary['feature_match_or_raw_overlap_support_available_ratio']}")
    print(f"patch_radius={summary['patch_radius']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    import math

    main()
