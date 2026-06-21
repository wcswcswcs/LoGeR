from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import incidence_audit, load_scene_artifacts, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v46 mask-carrier incidence.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_semantic_part_graph_radio")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_combined_dino_prepared_allframe_r1")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v46_incidence")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    payload = incidence_audit(artifacts)
    out = ROOT / args.output_root
    write_json(out / "incidence_audit.json", payload)
    write_csv(out / "incidence_scene_rows.csv", payload["scene_rows"])
    write_csv(out / "incidence_rows.csv", payload["incidence_rows"])
    print(json.dumps({"summary": str(out / "incidence_audit.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
