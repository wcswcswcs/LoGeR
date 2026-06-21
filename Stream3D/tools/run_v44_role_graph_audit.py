from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import load_scene_artifacts, role_graph_audit, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v44 same-frame role graph.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v43_2_current_v37_semantic_part_graph_dinov2_sample8")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v44_role_graph")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, scenes=args.scenes)
    payload = role_graph_audit(artifacts)
    out = ROOT / args.output_root
    write_json(out / "role_graph_audit.json", payload)
    write_csv(out / "role_graph_rows.csv", payload["rows"])
    write_csv(out / "role_assignments.csv", payload["role_rows"])
    print(json.dumps({"summary": str(out / "role_graph_audit.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
