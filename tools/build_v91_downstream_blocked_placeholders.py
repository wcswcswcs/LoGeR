#!/usr/bin/env python3
"""Write v91 downstream blocked artifacts for Phase8/9/10 when gates are closed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json
from v91_semantic_regime_utils import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def main() -> None:
    args = parse_args()
    phase7 = _json(args.root / "phase7_carrier_attribution_or_blocked/phase7_carrier_summary.json")
    phase8_dir = args.root / "phase8_counterfactual_or_blocked"
    phase9_dir = args.root / "phase9_runtime_or_blocked"
    phase10_dir = args.root / "phase10_ttt_or_blocked"
    phase8_dir.mkdir(parents=True, exist_ok=True)
    phase9_dir.mkdir(parents=True, exist_ok=True)
    phase10_dir.mkdir(parents=True, exist_ok=True)
    phase8_entered = bool(phase7.get("phase7_carrier_gate_pass"))
    phase8 = {
        "phase": "Phase8_counterfactual_or_blocked",
        "entered": phase8_entered,
        "phase8_counterfactual_gate_pass": False,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "blocker": "counterfactual_not_entered_carrier_failed" if not phase8_entered else "counterfactual_upper_bound_not_available",
    }
    candidates = [
        "CF0_NATIVE",
        "CF1_GEOMETRY_ONLY_REGIME",
        "CF2_SEMANTIC_TOPOLOGY_UPDATE_HOLD_REJECT",
        "CF3_SEMANTIC_DELAYED_COMMIT",
        "CF4_FEATURE_MATCH_TOPOLOGY_SUPPORT",
        "CF5_COMPONENT_SHUFFLE_CONTROL",
        "CF6_SEMANTIC_LABEL_SHUFFLE_CONTROL",
        "CF7_REGIME_SHUFFLE_CONTROL",
        "CF8_SAME_OVERLAP_RANDOM_CONTROL",
    ]
    write_json(phase8_dir / "counterfactual_or_blocked_summary.json", phase8)
    write_csv(phase8_dir / "counterfactual_candidates.csv", [{"candidate": item, "status": "blocked", "reason": phase8["blocker"]} for item in candidates])
    phase9_entered = bool(phase8.get("phase8_counterfactual_gate_pass"))
    phase9 = {
        "phase": "Phase9_runtime_memory_action_or_blocked",
        "entered": phase9_entered,
        "phase9_runtime_action_gate_pass": False,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "blocker": "runtime_not_entered_counterfactual_failed" if not phase9_entered else "runtime_action_gate_failed",
    }
    runtime_candidates = [
        "R0_NATIVE",
        "R1_GEOMETRY_ONLY_UPDATE_HOLD",
        "R2_SEMANTIC_TOPOLOGY_READ_GATE",
        "R3_SEMANTIC_TOPOLOGY_SWA_QK_PAIR_BIAS",
        "R4_SEMANTIC_TOPOLOGY_MERGE_GAUGE_UPDATE_GATE",
        "R5_SEMANTIC_TOPOLOGY_DELAYED_COMMIT",
        "R6_FULL_READ_SWA_MERGE_HANDSHAKE",
        "R7_SEMANTIC_SHUFFLE_CONTROL",
        "R8_COMPONENT_SHUFFLE_CONTROL",
        "R9_REGIME_SHUFFLE_CONTROL",
        "R10_SAME_MASS_RANDOM_CONTROL",
    ]
    write_json(phase9_dir / "runtime_or_blocked_summary.json", phase9)
    write_csv(phase9_dir / "runtime_candidates.csv", [{"candidate": item, "status": "blocked", "reason": phase9["blocker"]} for item in runtime_candidates])
    phase10_entered = bool(phase9.get("phase9_runtime_action_gate_pass"))
    phase10 = {
        "phase": "Phase10_ttt_write_policy_or_blocked",
        "entered": phase10_entered,
        "phase10_ttt_gate_pass": False,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "blocker": "ttt_not_entered_runtime_action_failed" if not phase10_entered else "ttt_write_policy_gate_failed",
    }
    write_json(phase10_dir / "ttt_or_blocked_summary.json", phase10)
    write_csv(
        phase10_dir / "ttt_policy_rows.csv",
        [
            {"policy": "persistent_write", "status": "blocked", "reason": phase10["blocker"]},
            {"policy": "one_hop_transient", "status": "blocked", "reason": phase10["blocker"]},
            {"policy": "neutral", "status": "blocked", "reason": phase10["blocker"]},
        ],
    )
    print(f"phase8_entered={phase8['entered']}")
    print(f"phase8_counterfactual_gate_pass={phase8['phase8_counterfactual_gate_pass']}")
    print(f"phase9_entered={phase9['entered']}")
    print(f"runtime_action_allowed={phase9['runtime_action_allowed']}")
    print(f"phase10_entered={phase10['entered']}")
    print(f"ttt_allowed={phase10['ttt_allowed']}")


if __name__ == "__main__":
    main()
