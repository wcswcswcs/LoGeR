from __future__ import annotations

import argparse

from stream4d_native.v63_action_utility_repair_audit import (
    V63ActionUtilityRepairAuditConfig,
    build_v63_action_utility_repair_audit,
    write_v63_action_utility_repair_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v63 Phase 4 action-utility/control repair targets.")
    parser.add_argument("--action-outcome-rows", default="outputs/audit/v63_action_outcome/action_outcome_rows.csv")
    parser.add_argument("--action-utility-rows", default="outputs/audit/v63_action_outcome/action_utility_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v63_action_utility_repair_audit")
    parser.add_argument("--cq-margin", type=float, default=0.15)
    args = parser.parse_args()
    cfg = V63ActionUtilityRepairAuditConfig(
        action_outcome_rows=args.action_outcome_rows,
        action_utility_rows=args.action_utility_rows,
        output_root=args.output_root,
        cq_margin=args.cq_margin,
    )
    result = build_v63_action_utility_repair_audit(cfg)
    outputs = write_v63_action_utility_repair_audit(result, cfg)
    print({"outputs": outputs, "gate": result["summary"]["gate"], "repair_assessment": result["summary"]["repair_assessment"]})


if __name__ == "__main__":
    main()

