#!/usr/bin/env python3
"""Build v90 Phase1 semantic topology nodes from raw overlap semantic labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v90_semantic_topology_utils import ROOT, V89_LEDGER, build_components_for_side, load_raw, median_frame, pair_id, seq_norm


DEFAULT_OUT = ROOT / "phase1_semantic_topology_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v89-ledger-dir", type=Path, default=V89_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--patch-radius", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.v89_ledger_dir / "semantic_scale_pair_rows.csv")
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    raw_missing = 0
    for _, pair in pairs.iterrows():
        seq = seq_norm(pair["seq"])
        prev = int(pair["prev_chunk"])
        curr = int(pair["curr_chunk"])
        source_path = str(pair["source_path"])
        raw = load_raw(source_path)
        if raw is None:
            raw_missing += 1
            source_rows.append(
                {
                    "seq": seq,
                    "prev_chunk": prev,
                    "curr_chunk": curr,
                    "pair_id": pair_id(seq, prev, curr),
                    "source_path": source_path,
                    "raw_available": False,
                    "node_rows": 0,
                    "label_mapping_status": "raw_overlap_unavailable",
                    "patch_radius": args.patch_radius,
                }
            )
            continue
        pair_node_count = 0
        for side in ("prev", "curr"):
            comp_rows, _, label_counts = build_components_for_side(raw, side, radius=args.patch_radius)
            frame_id = median_frame(raw, side)
            for comp in comp_rows:
                out = dict(comp)
                out.update(
                    {
                        "seq": seq,
                        "prev_chunk": prev,
                        "curr_chunk": curr,
                        "pair_id": pair_id(seq, prev, curr),
                        "frame_id": frame_id,
                        "source_path": source_path,
                        "patch_radius": args.patch_radius,
                        "label_component_count_for_label": label_counts.get(int(comp["label_compact_id"]), 0),
                    }
                )
                rows.append(out)
                pair_node_count += 1
        source_rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "pair_id": pair_id(seq, prev, curr),
                "source_path": source_path,
                "raw_available": True,
                "node_rows": pair_node_count,
                "label_mapping_status": "compact_project_local_id_no_class_names",
                "patch_radius": args.patch_radius,
            }
        )
    write_csv(args.out_dir / "topology_nodes.csv", rows)
    write_csv(args.out_dir / "topology_source_rows.csv", source_rows)
    node_df = pd.DataFrame(rows)
    src_df = pd.DataFrame(source_rows)
    summary = {
        "phase": "Phase1_semantic_topology_nodes",
        "pair_rows": int(len(pairs)),
        "sequence_coverage": int(pairs["seq"].astype(str).str.zfill(2).nunique()),
        "node_rows": int(len(rows)),
        "node_rows_all_sequences": int(node_df["seq"].nunique()) if len(node_df) else 0,
        "raw_missing_pairs": int(raw_missing),
        "patch_radius": int(args.patch_radius),
        "semantic_source_format": "raw_overlap_compact_labels_confidence",
        "class_name_available": False,
        "label_mapping_status": "compact_project_local_id_no_class_names",
        "has_radio": False,
        "has_track": False,
        "pairs_with_nodes": int((src_df["node_rows"] > 0).sum()) if len(src_df) else 0,
    }
    summary["node_build_gate_pass"] = bool(
        summary["pair_rows"] >= 49
        and summary["sequence_coverage"] >= 4
        and summary["node_rows"] > 0
        and summary["node_rows_all_sequences"] >= 4
        and summary["raw_missing_pairs"] == 0
        and summary["pairs_with_nodes"] == summary["pair_rows"]
    )
    if not summary["node_build_gate_pass"]:
        summary["blocker"] = "semantic_topology_node_source_insufficient"
    write_json(args.out_dir / "topology_node_summary.json", summary)
    print(f"node_build_gate_pass={summary['node_build_gate_pass']}")
    print(f"pair_rows={summary['pair_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"node_rows={summary['node_rows']}")
    print(f"raw_missing_pairs={summary['raw_missing_pairs']}")
    print(f"patch_radius={summary['patch_radius']}")
    print(f"label_mapping_status={summary['label_mapping_status']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
