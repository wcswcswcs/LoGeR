#!/usr/bin/env python3
"""Phase 6 precondition-checked online controller smoke entrypoint for v74-TF."""

from __future__ import annotations

import argparse
from pathlib import Path

from v73_semantic_memory_common import load_json, utc_now, write_json, write_text
from v74tf_common import REPORT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable_v74tf_memory_control", action="store_true")
    parser.add_argument("--v74tf_rule_config", type=Path, default=Path("configs/v74tf_fixed_rules.yaml"))
    parser.add_argument("--v74tf_action_family", default="")
    parser.add_argument("--v74tf_semantic_source", default="")
    parser.add_argument("--v74tf_control_mode", default="native")
    parser.add_argument("--v74tf_seq", default="01")
    parser.add_argument("--v74tf_chunks", default="")
    parser.add_argument("--phase4-summary", type=Path, default=REPORT_ROOT / "phase4_action_family_oracle" / "action_family_summary_by_seq.json")
    parser.add_argument("--phase5-summary", type=Path, default=REPORT_ROOT / "phase5_counterfactual_memory_intervention" / "counterfactual_intervention_summary.json")
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "phase6_online_controller_smoke")
    parser.add_argument("--dry-run", action="store_true", help="Record precondition status without launching run_pipeline.")
    args = parser.parse_args()

    phase4 = load_json(args.phase4_summary) or {}
    phase5 = load_json(args.phase5_summary) or {}
    precondition_pass = bool(phase4.get("phase4_gate_pass") or phase5.get("phase5_gate_pass"))
    status = "blocked_precondition_not_met"
    if precondition_pass and args.enable_v74tf_memory_control and not args.dry_run:
        status = "blocked_runner_not_wired_for_online_v74tf_in_this_entrypoint"
    elif precondition_pass:
        status = "dry_run_precondition_pass"
    payload = {
        "schema": "acl2_v74tf_phase6_online_controller_smoke_precheck_v1",
        "created_at": utc_now(),
        "enable_v74tf_memory_control": bool(args.enable_v74tf_memory_control),
        "rule_config": str(args.v74tf_rule_config),
        "rule_config_exists": args.v74tf_rule_config.exists(),
        "action_family": args.v74tf_action_family,
        "semantic_source": args.v74tf_semantic_source,
        "control_mode": args.v74tf_control_mode,
        "seq": str(args.v74tf_seq).zfill(2),
        "chunks": args.v74tf_chunks,
        "phase4_gate_pass": bool(phase4.get("phase4_gate_pass")),
        "phase5_gate_pass": bool(phase5.get("phase5_gate_pass")),
        "online_smoke_precondition_pass": precondition_pass,
        "status": status,
        "gate_pass": False,
        "note": "Plan forbids online smoke until Phase 4 or Phase 5 gate passes. This script records that guard.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "online_smoke_precheck.json", payload)
    write_text(
        args.out_dir / "online_smoke_report.md",
        "\n".join(
            [
                "# v74-TF Phase 6 Online Controller Smoke",
                "",
                f"- status: `{status}`",
                f"- phase4_gate_pass: `{payload['phase4_gate_pass']}`",
                f"- phase5_gate_pass: `{payload['phase5_gate_pass']}`",
                f"- online_smoke_precondition_pass: `{precondition_pass}`",
                f"- gate_pass: `False`",
                "",
            ]
        ),
    )
    print({"out_dir": str(args.out_dir), "status": status, "gate_pass": False})


if __name__ == "__main__":
    main()

