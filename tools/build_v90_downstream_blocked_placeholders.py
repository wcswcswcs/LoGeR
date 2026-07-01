#!/usr/bin/env python3
"""Write v90 Phase6-Phase9 blocked summaries when upstream gates do not allow entry."""

from __future__ import annotations

import argparse
from pathlib import Path

from v86_soft_latent_utils import read_json, write_csv, write_json
from v90_semantic_topology_utils import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _write_blocked(out: Path, name: str, reason: str, preconditions: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": name,
        "entered": False,
        "gate_pass": False,
        "blocker": reason,
        "preconditions": preconditions,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "note": "Blocked summary only; no carrier/counterfactual/runtime/TTT data were fabricated.",
    }
    write_json(out / f"{name}_summary.json", payload)
    write_csv(out / f"{name}_preconditions.csv", [{"precondition": k, "value": v} for k, v in preconditions.items()])
    (out / f"{name}_blocked.md").write_text(
        "\n".join(
            [
                f"# {name} Blocked",
                "",
                f"- entered: `{payload['entered']}`",
                f"- gate_pass: `{payload['gate_pass']}`",
                f"- blocker: `{payload['blocker']}`",
                "",
                "This file is a blocked placeholder. It does not contain measured carrier/action/counterfactual results.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    phase3 = _load(args.root / "phase3_semantic_topology_relevance/topology_relevance_summary.json")
    phase4 = _load(args.root / "phase4_semantic_topology_observability_policy/topology_observability_policy_audit_summary.json")
    phase5 = _load(args.root / "phase5_feature_match_topology_ruler/feature_match_topology_audit_summary.json")
    p3 = bool(phase3.get("phase3_topology_relevance_global_gate_pass", False))
    p4 = bool(phase4.get("semantic_topology_observability_policy_gate_pass", False))
    p5 = bool(phase5.get("feature_match_topology_ruler_gate_pass", False))
    carrier_pre = {
        "phase3_topology_global_pass": p3,
        "phase4_topology_policy_pass": p4,
        "phase5_feature_match_topology_pass": p5,
        "has_topology_specific_pass": bool(p3 or p4 or p5),
    }
    _write_blocked(args.root / "phase6_topology_carrier_attribution", "phase6_carrier_attribution", "carrier_not_entered_preconditions_failed", carrier_pre)
    cf_pre = {**carrier_pre, "phase6_carrier_candidate": False, "offline_upper_bound_owner_allowance": False}
    _write_blocked(args.root / "phase7_topology_counterfactual_upper_bound", "phase7_counterfactual_upper_bound", "counterfactual_not_entered_preconditions_failed", cf_pre)
    runtime_pre = {**cf_pre, "phase7_counterfactual_pass": False, "phase10_visual_audit_ready": False}
    _write_blocked(args.root / "phase8_runtime_memory_action", "phase8_runtime_memory_action", "runtime_action_not_entered_preconditions_failed", runtime_pre)
    ttt_pre = {**runtime_pre, "phase8_runtime_action_pass": False, "merge_gauge_runtime_action_pass": False}
    _write_blocked(args.root / "phase9_ttt_write_policy", "phase9_ttt_write_policy", "ttt_not_entered_preconditions_failed", ttt_pre)
    print(f"phase6_entered=False")
    print(f"carrier_not_entered_preconditions_failed={not carrier_pre['has_topology_specific_pass']}")
    print(f"phase7_entered=False")
    print(f"phase8_runtime_action_allowed=False")
    print(f"phase9_ttt_allowed=False")


if __name__ == "__main__":
    main()
