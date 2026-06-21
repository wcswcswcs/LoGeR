from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import controls_and_significance, load_scene_artifacts, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v44 controls and significance diagnostics.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v43_2_current_v37_semantic_part_graph_dinov2_sample8")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_dino_q5_stride1_r3")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--real-variant", default="E_lattice_plus_temporal_reactivation")
    parser.add_argument("--output-root", default="outputs/audit/v44_controls")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    payload = controls_and_significance(artifacts, real_variant=args.real_variant)
    out = ROOT / args.output_root
    write_json(out / "controls_and_significance.json", payload)
    write_csv(out / "control_rows.csv", payload["control_rows"])
    write_csv(out / "control_variant_rows.csv", payload["variant_rows"])
    write_csv(out / "per_scene_delta_rows.csv", payload["per_scene_delta_rows"])
    print(json.dumps({"summary": str(out / "controls_and_significance.json"), "gate": payload["gate"], "checks": payload["checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
