from __future__ import annotations

import argparse

from stream4d_native.v47_common import ROOT, read_csv, write_csv, write_json
from stream4d_native.v47_tracklet_builder import build_tracklets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 short tracklet construction.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables")
    parser.add_argument("--edge-root", default="outputs/audit/v47_adjacent_edges")
    parser.add_argument("--score-key", default="A5_d4rt_semantic_confirmation")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--edge-types", default="adjacent")
    parser.add_argument("--ignore-edge-accept-candidate", action="store_true")
    parser.add_argument("--output-root", default="outputs/audit/v47_tracklets")
    args = parser.parse_args()
    edge_types = {item.strip() for item in str(args.edge_types).split(",") if item.strip()}
    payload = build_tracklets(
        mask_rows=read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv"),
        edge_rows=read_csv(ROOT / str(args.edge_root) / "temporal_candidate_edge_table.csv"),
        score_key=str(args.score_key),
        min_score=float(args.min_score),
        edge_types=edge_types,
        respect_edge_accept_candidate=not bool(args.ignore_edge_accept_candidate),
    )
    out = ROOT / str(args.output_root)
    write_csv(out / "tracklet_rows.csv", payload["tracklet_rows"])
    write_csv(out / "tracklet_selected_edge_rows.csv", payload["selected_edge_rows"])
    write_json(out / "tracklet_construction.json", payload["summary"])
    print({"summary": str(out / "tracklet_construction.json"), "gate": payload["summary"]["gate"]})


if __name__ == "__main__":
    main()
