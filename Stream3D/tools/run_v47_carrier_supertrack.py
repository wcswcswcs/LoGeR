from __future__ import annotations

import argparse

from stream4d_native.v47_carrier_supertrack import build_carrier_supertrack_summary
from stream4d_native.v47_common import ROOT, read_csv, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v47 carrier super-track diagnostic.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--max-union-unique-carriers", type=int, default=-1)
    parser.add_argument("--output-root", default="outputs/audit/v47_carrier_supertrack")
    args = parser.parse_args()

    obs_root = ROOT / str(args.observation_root)
    payload = build_carrier_supertrack_summary(
        carrier_rows=read_csv(obs_root / "carrier_observation_table.csv"),
        mask_rows=read_csv(obs_root / "mask_observation_table.csv"),
        max_union_unique_carriers=None
        if int(args.max_union_unique_carriers) < 0
        else int(args.max_union_unique_carriers),
    )
    out = ROOT / str(args.output_root)
    write_csv(out / "carrier_supertrack_rows.csv", payload["supertrack_rows"])
    write_csv(out / "carrier_supertrack_mask_vote_rows.csv", payload["mask_vote_rows"])
    write_csv(out / "carrier_supertrack_scene_rows.csv", payload["scene_rows"])
    write_json(out / "carrier_supertrack_summary.json", payload["summary"])
    print({"summary": str(out / "carrier_supertrack_summary.json"), "gate": payload["summary"]["gate"]})


if __name__ == "__main__":
    main()
