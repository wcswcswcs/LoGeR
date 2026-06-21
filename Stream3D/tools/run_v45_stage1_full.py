from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.stage1_scale_aware_typed_assembly import (
    load_scene_artifacts,
    preconditioned_stage1_status,
    read_json,
    typed_energy_selection_diagnostic_v45,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or block v45 Stage-1 according to precondition gates.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v43_2_current_v37_semantic_part_graph_dinov2_sample8")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_dino_q5_stride1_r3")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v45_stage1_full")
    parser.add_argument("--descriptor-summary", default="outputs/audit/v45_mask_descriptor/mask_descriptor_audit.json")
    parser.add_argument("--role-summary", default="outputs/audit/v45_role_lattice/role_lattice_audit.json")
    parser.add_argument("--operations-summary", default="outputs/audit/v45_typed_operations/typed_operation_audit.json")
    parser.add_argument("--temporal-summary", default="outputs/audit/v45_temporal_matching/temporal_matching_audit.json")
    args = parser.parse_args()
    descriptor = read_json(ROOT / args.descriptor_summary) or {}
    role = read_json(ROOT / args.role_summary) or {}
    operations = read_json(ROOT / args.operations_summary) or {}
    temporal = read_json(ROOT / args.temporal_summary) or {}
    status = preconditioned_stage1_status(descriptor=descriptor, role=role, operations=operations, temporal=temporal)
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    diagnostic = typed_energy_selection_diagnostic_v45(artifacts)
    payload = {**status, "typed_energy_diagnostic": diagnostic}
    if not status["stage1_run_as_method"]:
        payload["note"] = "Stop Rules block full Stage-1 method promotion; typed-energy rows are diagnostic only."
    out = ROOT / args.output_root
    write_json(out / "stage1_full_summary.json", payload)
    write_csv(out / "stage1_variant_rows.csv", diagnostic["variant_rows"])
    write_csv(out / "stage1_scene_rows.csv", diagnostic["scene_rows"])
    write_csv(out / "object_field_rows.csv", diagnostic["object_rows"])
    print(json.dumps({"summary": str(out / "stage1_full_summary.json"), "status": payload["status"], "gate": payload["gate"], "typed_energy_gate": diagnostic["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
