from __future__ import annotations

import argparse

from stream4d_native.v47_common import ROOT, read_csv, write_csv, write_json
from stream4d_native.v47_min_cost_flow import build_sparse_temporal_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 sparse temporal flow greedy min-cost proxy.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables")
    parser.add_argument("--edge-root", default="outputs/audit/v47_adjacent_edges")
    parser.add_argument("--score-key", default="A5_d4rt_semantic_confirmation")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--output-root", default="outputs/audit/v47_flow")
    args = parser.parse_args()
    payload = build_sparse_temporal_flow(
        mask_rows=read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv"),
        edge_rows=read_csv(ROOT / str(args.edge_root) / "temporal_candidate_edge_table.csv"),
        score_key=str(args.score_key),
        min_score=float(args.min_score),
    )
    out = ROOT / str(args.output_root)
    write_csv(out / "flow_track_rows.csv", payload["tracklet_rows"])
    write_csv(out / "flow_selected_edge_rows.csv", payload["selected_edge_rows"])
    write_json(out / "min_cost_flow.json", payload["summary"])
    print({"summary": str(out / "min_cost_flow.json"), "gate": payload["summary"]["gate"]})


if __name__ == "__main__":
    main()

