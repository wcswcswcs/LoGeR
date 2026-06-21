from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import load_scene_artifacts, object_field_export, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v46 object fields from mask clusters.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_semantic_part_graph_radio")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_combined_dino_prepared_allframe_r1")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--solver", default="S5_multicut_local_search")
    parser.add_argument("--output-root", default="outputs/audit/v46_object_field_export")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    payload = object_field_export(artifacts, solver=args.solver)
    out = ROOT / args.output_root
    write_json(out / "object_field_export.json", payload)
    write_csv(out / "object_field_scene_rows.csv", payload["scene_rows"])
    write_csv(out / "object_field_rows.csv", payload["object_rows"])
    print(json.dumps({"summary": str(out / "object_field_export.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
