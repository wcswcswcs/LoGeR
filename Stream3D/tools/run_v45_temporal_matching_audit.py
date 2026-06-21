from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.stage1_scale_aware_typed_assembly import load_scene_artifacts, temporal_matching_audit_v45, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v45 role-aware temporal matching.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v43_2_current_v37_semantic_part_graph_dinov2_sample8")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_dino_q5_stride1_r3")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--role-profile", default="R4_full_role_lattice")
    parser.add_argument("--output-root", default="outputs/audit/v45_temporal_matching")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    payload = temporal_matching_audit_v45(artifacts, role_profile=args.role_profile)
    out = ROOT / args.output_root
    write_json(out / "temporal_matching_audit.json", payload)
    write_csv(out / "temporal_matching_rows.csv", payload["rows"])
    write_csv(out / "d4rt_verification_rows.csv", payload["d4rt_rows"])
    print(json.dumps({"summary": str(out / "temporal_matching_audit.json"), "best_profile": payload["best_profile"], "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

