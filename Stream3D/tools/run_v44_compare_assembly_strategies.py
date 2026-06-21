from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import load_scene_artifacts, strategy_comparison, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v44 typed mask assembly strategies.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v43_2_current_v37_semantic_part_graph_dinov2_sample8")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_dino_q5_stride1_r3")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v44_strategy_comparison")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    payload = strategy_comparison(artifacts)
    out = ROOT / args.output_root
    write_json(out / "strategy_comparison.json", payload)
    write_csv(out / "stage1_variant_rows.csv", payload["variant_rows"])
    write_csv(out / "stage1_scene_rows.csv", payload["scene_rows"])
    write_csv(out / "object_field_rows.csv", payload["object_rows"])
    print(json.dumps({"summary": str(out / "strategy_comparison.json"), "best_variant": payload["best_variant"], "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
