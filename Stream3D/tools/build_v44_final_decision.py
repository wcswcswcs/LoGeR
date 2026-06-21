from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import final_decision, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final v44 decision from audit artifacts.")
    parser.add_argument("--fact", default="outputs/audit/v44_fact_lock/fact_lock.json")
    parser.add_argument("--descriptor", default="outputs/audit/v44_mask_descriptor/mask_descriptor_audit.json")
    parser.add_argument("--role", default="outputs/audit/v44_role_graph/role_graph_audit.json")
    parser.add_argument("--operations", default="outputs/audit/v44_typed_operations/typed_operation_audit.json")
    parser.add_argument("--temporal", default="outputs/audit/v44_temporal_matching/temporal_objectlet_matching.json")
    parser.add_argument("--strategies", default="outputs/audit/v44_strategy_comparison/strategy_comparison.json")
    parser.add_argument("--controls", default="outputs/audit/v44_controls/controls_and_significance.json")
    parser.add_argument("--stage2", default="outputs/audit/v44_stage2_geometry/stage2_geometry_eligibility.json")
    parser.add_argument("--ap", default="outputs/audit/v44_ap_bridge/ap_bridge_diagnostic.json")
    parser.add_argument("--output-root", default="outputs/audit/v44_final_decision")
    args = parser.parse_args()
    payload = final_decision(
        read_json(ROOT / args.fact) or {},
        read_json(ROOT / args.descriptor) or {},
        read_json(ROOT / args.role) or {},
        read_json(ROOT / args.operations) or {},
        read_json(ROOT / args.temporal) or {},
        read_json(ROOT / args.strategies) or {},
        read_json(ROOT / args.controls) or {},
        read_json(ROOT / args.stage2) or {},
        read_json(ROOT / args.ap) or {},
    )
    out = ROOT / args.output_root
    write_json(out / "v44_final_decision.json", payload)
    print(json.dumps({"summary": str(out / "v44_final_decision.json"), "final_label": payload["final_label"], "reason": payload["reason"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
