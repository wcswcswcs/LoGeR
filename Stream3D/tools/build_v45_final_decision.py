from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stream4d_native.stage1_scale_aware_typed_assembly import read_json, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]


def _gate(payload: dict[str, Any]) -> bool:
    return bool(payload.get("gate", {}).get("pass"))


def build_decision(
    root: Path,
    *,
    fact_path: str = "outputs/audit/v45_fact_lock/fact_lock.json",
    scale_path: str = "outputs/audit/v45_scale_alignment/scale_alignment_audit.json",
    descriptor_path: str = "outputs/audit/v45_mask_descriptor/mask_descriptor_audit.json",
    role_path: str = "outputs/audit/v45_role_lattice/role_lattice_audit.json",
    operations_path: str = "outputs/audit/v45_typed_operations/typed_operation_audit.json",
    temporal_path: str = "outputs/audit/v45_temporal_matching/temporal_matching_audit.json",
    stage1_path: str = "outputs/audit/v45_stage1_full/stage1_full_summary.json",
    ap_path: str = "outputs/audit/v45_eval_aligned_ap/eval_aligned_ap_summary.json",
    stage2_path: str = "outputs/audit/v45_stage2_geometry/stage2_geometry_eligibility.json",
) -> dict[str, Any]:
    fact = read_json(root / fact_path) or {}
    scale = read_json(root / scale_path) or {}
    descriptor = read_json(root / descriptor_path) or {}
    role = read_json(root / role_path) or {}
    operations = read_json(root / operations_path) or {}
    temporal = read_json(root / temporal_path) or {}
    stage1 = read_json(root / stage1_path) or {}
    ap = read_json(root / ap_path) or {}
    stage2 = read_json(root / stage2_path) or {}
    if not _gate(scale):
        label = "NO_GO_SCALE_ALIGNMENT"
        reason = "Scale guard failed or was unavailable."
    elif not _gate(descriptor):
        label = "NO_GO_DESCRIPTOR"
        reason = "Semantic descriptor did not pass v45 signal gate over available artifacts."
    elif not _gate(role):
        label = "NO_GO_ROLE"
        reason = "Role lattice did not pass core/part/mixed/duplicate gates."
    elif not _gate(operations):
        label = "NO_GO_TYPED_OPERATIONS"
        reason = "Absorb/Reject/Replace diagnostics failed v45 operation gate."
    elif not _gate(temporal):
        label = "PARTIAL_D4RT_SIGNAL_ONLY"
        reason = "Temporal/D4RT controls did not pass; D4RT cannot be claimed as reliable merge driver."
    elif not _gate(stage1):
        label = "NO_GO_STAGE1_NOT_SIGNIFICANT"
        reason = "Stage-1 preconditions or significant gate failed."
    elif not _gate(ap):
        label = "NO_GO_AP_BRIDGE"
        reason = "Stage-1 passed but AP bridge did not pass or was not run."
    elif not stage2.get("stage2_allowed", False):
        label = "NO_GO_STAGE2"
        reason = "Stage-2 was not allowed by eligibility gate."
    else:
        label = "GO_STAGE1_SCALE_AWARE_TYPED_ASSEMBLY"
        reason = "All required v45 gates passed."
    answers = {
        "scale_alignment_pass": _gate(scale),
        "cross_chunk_local_metric_reads": scale.get("cross_chunk_local_metric_reads"),
        "cross_chunk_eval_reads": scale.get("cross_chunk_eval_reads"),
        "ap_uses_eval_only_gt_alignment": ap.get("manifest_gate", {}).get("pass"),
        "semantic_descriptor_pass": _gate(descriptor),
        "role_lattice_pass": _gate(role),
        "typed_operations_pass": _gate(operations),
        "temporal_d4rt_pass": _gate(temporal),
        "stage1_significant": _gate(stage1),
        "AP": ap.get("AP"),
        "AP50": ap.get("AP50"),
        "AP25": ap.get("AP25"),
        "stage2_allowed": bool(stage2.get("stage2_allowed", False)),
        "failure_location": label,
    }
    return {
        "phase": "v45_final_decision",
        "created_at": utc_now(),
        "final_label": label,
        "reason": reason,
        "answers": answers,
        "gate_sources": {
            "fact": fact.get("gate"),
            "scale": scale.get("gate"),
            "descriptor": descriptor.get("gate"),
            "role": role.get("gate"),
            "operations": operations.get("gate"),
            "temporal": temporal.get("gate"),
            "stage1": stage1.get("gate"),
            "ap": ap.get("gate"),
            "stage2": stage2.get("gate"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v45 final decision.")
    parser.add_argument("--output-root", default="outputs/audit/v45_final_decision")
    parser.add_argument("--fact-path", default="outputs/audit/v45_fact_lock/fact_lock.json")
    parser.add_argument("--scale-path", default="outputs/audit/v45_scale_alignment/scale_alignment_audit.json")
    parser.add_argument("--descriptor-path", default="outputs/audit/v45_mask_descriptor/mask_descriptor_audit.json")
    parser.add_argument("--role-path", default="outputs/audit/v45_role_lattice/role_lattice_audit.json")
    parser.add_argument("--operations-path", default="outputs/audit/v45_typed_operations/typed_operation_audit.json")
    parser.add_argument("--temporal-path", default="outputs/audit/v45_temporal_matching/temporal_matching_audit.json")
    parser.add_argument("--stage1-path", default="outputs/audit/v45_stage1_full/stage1_full_summary.json")
    parser.add_argument("--ap-path", default="outputs/audit/v45_eval_aligned_ap/eval_aligned_ap_summary.json")
    parser.add_argument("--stage2-path", default="outputs/audit/v45_stage2_geometry/stage2_geometry_eligibility.json")
    args = parser.parse_args()
    payload = build_decision(
        ROOT,
        fact_path=args.fact_path,
        scale_path=args.scale_path,
        descriptor_path=args.descriptor_path,
        role_path=args.role_path,
        operations_path=args.operations_path,
        temporal_path=args.temporal_path,
        stage1_path=args.stage1_path,
        ap_path=args.ap_path,
        stage2_path=args.stage2_path,
    )
    out = ROOT / args.output_root
    write_json(out / "v45_final_decision.json", payload)
    print(json.dumps({"summary": str(out / "v45_final_decision.json"), "final_label": payload["final_label"], "reason": payload["reason"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
