from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import build_fact_lock, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v46 fact lock and scale/input contract.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_semantic_part_graph_radio")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_combined_dino_prepared_allframe_r1")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v46_fact_lock")
    args = parser.parse_args()
    payload = build_fact_lock(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    out = ROOT / args.output_root
    write_json(out / "fact_lock.json", payload)
    write_csv(out / "fact_lock_rows.csv", payload["fact_rows"])
    print(json.dumps({"summary": str(out / "fact_lock.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
