from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import load_scene_artifacts, solver_comparison, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v46 signed graph solvers.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_semantic_part_graph_radio")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_combined_dino_prepared_allframe_r1")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v46_solver_comparison")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    payload = solver_comparison(artifacts)
    out = ROOT / args.output_root
    write_json(out / "solver_comparison.json", payload)
    write_csv(out / "solver_aggregate_rows.csv", payload["aggregate_rows"])
    write_csv(out / "solver_scene_rows.csv", payload["scene_rows"])
    write_csv(out / "solver_merge_trace.csv", payload["solver_merge_trace"])
    print(json.dumps({"summary": str(out / "solver_comparison.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
