from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import full_stage1_controls, load_scene_artifacts, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v46 full Stage-1 controls over available artifacts.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_semantic_part_graph_radio")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_combined_dino_prepared_allframe_r1")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v46_full_stage1")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    payload = full_stage1_controls(artifacts)
    out = ROOT / args.output_root
    write_json(out / "full_stage1_controls.json", payload)
    write_csv(out / "full_stage1_variant_rows.csv", payload["rows"])
    write_csv(out / "full_stage1_scene_rows.csv", payload["scene_rows"])
    print(json.dumps({"summary": str(out / "full_stage1_controls.json"), "gate": payload["gate"], "controls": payload["controls"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
