from __future__ import annotations

import argparse

from stream4d_native.v47_common import ROOT, read_csv, write_csv, write_json
from stream4d_native.v47_temporal_edge_builder import build_temporal_candidate_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 adjacent/short temporal edge audit.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables")
    parser.add_argument("--gap-max", type=int, default=2)
    parser.add_argument("--reactivation-gap-max", type=int, default=5)
    parser.add_argument("--min-forward-visible-carriers", type=int, default=0)
    parser.add_argument("--min-backward-visible-carriers", type=int, default=0)
    parser.add_argument("--max-visible-outside", type=float, default=1.0)
    parser.add_argument("--output-root", default="outputs/audit/v47_adjacent_edges")
    args = parser.parse_args()
    obs_root = ROOT / str(args.observation_root)
    payload = build_temporal_candidate_edges(
        mask_rows=read_csv(obs_root / "mask_observation_table.csv"),
        carrier_rows=read_csv(obs_root / "carrier_observation_table.csv"),
        gap_max=int(args.gap_max),
        reactivation_gap_max=int(args.reactivation_gap_max),
        min_forward_visible_carriers=int(args.min_forward_visible_carriers),
        min_backward_visible_carriers=int(args.min_backward_visible_carriers),
        max_visible_outside=float(args.max_visible_outside),
    )
    out = ROOT / str(args.output_root)
    write_csv(out / "temporal_candidate_edge_table.csv", payload["edge_rows"])
    write_csv(out / "edge_summary_rows.csv", payload["summary_rows"])
    write_json(out / "adjacent_edge_audit.json", payload["summary"])
    print({"summary": str(out / "adjacent_edge_audit.json"), "gate": payload["summary"]["gate"]})


if __name__ == "__main__":
    main()
