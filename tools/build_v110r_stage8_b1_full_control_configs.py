#!/usr/bin/env python3
"""Generate ACL2 v110R continuation configs for B1 full-sequence controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v110r_stage2_candidate_configs as stage2  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
STAGE8 = RESULT_ROOT / "stage8_b1_full_controls"
CONFIG_OUT = STAGE8 / "config_generation"
SEQUENCES = ("00", "01", "02", "05")
SELECTED_CANDIDATES = ("B1",)
CONTROL_POLICY_FAMILIES = (
    "internal_only",
    "semantic_shuffle",
    "same_count_random",
    "low_risk_reverse",
)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stage2.clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    stage2.OUT = CONFIG_OUT
    stage2.STAGE3 = STAGE8
    stage2.CONFIG_ROOT = STAGE8 / "configs"
    stage2.METHOD_DIR = stage2.CONFIG_ROOT / "methods"
    stage2.DATASET_DIR = stage2.CONFIG_ROOT / "datasets"
    stage2.WORKSPACE = STAGE8 / "workspace"
    stage2.RAW_ACTION = STAGE8 / "raw_action"
    stage2.SEQUENCES = SEQUENCES
    stage2.POLICY_FAMILIES = CONTROL_POLICY_FAMILIES
    stage2.RUNNABLE_CANDIDATES = {
        candidate_id: stage2.RUNNABLE_CANDIDATES[candidate_id]
        for candidate_id in SELECTED_CANDIDATES
    }

    summary = stage2.build()
    stage8_summary = {
        "schema": "acl2_v110r_stage8_b1_full_control_config_summary_v1",
        "stage8_config_ready": bool(summary.get("stage2_config_ready")),
        "blocker": summary.get("blocker", ""),
        "continuation_reason": (
            "v110R final decision found B1 full geometry success but semantic causality failed because "
            "four-sequence stronger controls were absent or matched."
        ),
        "sequences": list(SEQUENCES),
        "candidate_ids": list(SELECTED_CANDIDATES),
        "policy_families": list(CONTROL_POLICY_FAMILIES),
        "action_config_rows": summary.get("action_config_rows"),
        "manifest_rows": summary.get("manifest_rows"),
        "prepare_manifest_rows": summary.get("prepare_manifest_rows"),
        "run_worker_manifest_rows": summary.get("run_worker_manifest_rows"),
        "evaluate_manifest_rows": summary.get("evaluate_manifest_rows"),
        "report_manifest_rows": summary.get("report_manifest_rows"),
        "claim_boundary": (
            "This continuation only tests whether B1 semantic content beats internal/shuffle/random/reverse "
            "controls on the same four-sequence full ATE protocol; it does not introduce a new method claim."
        ),
        "outputs": {
            "action_config_rows": rel(STAGE8 / "action_config_rows.csv"),
            "candidate_policy_rows": rel(STAGE8 / "candidate_policy_rows.csv"),
            "run_manifest": rel(STAGE8 / "run_manifest.csv"),
            "workspace": rel(stage2.WORKSPACE),
            "raw_action": rel(stage2.RAW_ACTION),
            "config_generation_summary": rel(STAGE8 / "stage8_config_generation_summary.json"),
        },
    }
    write_json(STAGE8 / "stage8_config_generation_summary.json", stage8_summary)
    print(json.dumps(stage2.clean_json(stage8_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
