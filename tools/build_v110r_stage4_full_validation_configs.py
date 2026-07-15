#!/usr/bin/env python3
"""Generate ACL2 v110R Stage4 full KITTI validation configs from Stage3 pass set."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v110r_stage2_candidate_configs as stage2  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
STAGE4 = RESULT_ROOT / "stage4_full_00_01_02_05_validation"
CONFIG_OUT = STAGE4 / "config_generation"
STAGE3_SELECTION = RESULT_ROOT / "stage3_pilot_00_02/stage4_candidate_selection.csv"
SEQUENCES = ("00", "01", "02", "05")
SELECTED_CANDIDATES = ("B1", "E1", "F2")
SELECTED_POLICY_FAMILIES = ("semantic_only", "semantic_plus_internal")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stage2.clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    stage2.OUT = CONFIG_OUT
    stage2.STAGE3 = STAGE4
    stage2.CONFIG_ROOT = STAGE4 / "configs"
    stage2.METHOD_DIR = stage2.CONFIG_ROOT / "methods"
    stage2.DATASET_DIR = stage2.CONFIG_ROOT / "datasets"
    stage2.WORKSPACE = STAGE4 / "workspace"
    stage2.RAW_ACTION = STAGE4 / "raw_action"
    stage2.SEQUENCES = SEQUENCES
    stage2.POLICY_FAMILIES = SELECTED_POLICY_FAMILIES
    stage2.RUNNABLE_CANDIDATES = {
        candidate_id: stage2.RUNNABLE_CANDIDATES[candidate_id]
        for candidate_id in SELECTED_CANDIDATES
    }

    summary = stage2.build()
    stage4_summary = {
        "schema": "acl2_v110r_stage4_config_generation_summary_v1",
        "stage4_config_ready": bool(summary.get("stage2_config_ready")),
        "blocker": summary.get("blocker", ""),
        "selected_from_stage3": rel(STAGE3_SELECTION),
        "sequences": list(SEQUENCES),
        "candidate_ids": list(SELECTED_CANDIDATES),
        "policy_families": list(SELECTED_POLICY_FAMILIES),
        "action_config_rows": summary.get("action_config_rows"),
        "manifest_rows": summary.get("manifest_rows"),
        "prepare_manifest_rows": summary.get("prepare_manifest_rows"),
        "run_worker_manifest_rows": summary.get("run_worker_manifest_rows"),
        "evaluate_manifest_rows": summary.get("evaluate_manifest_rows"),
        "report_manifest_rows": summary.get("report_manifest_rows"),
        "fixed_control_boundary": (
            "NO_ACTION and F19 are frozen from Stage0/v109. Same-count/internal controls are kept in Stage3 summaries; "
            "this Stage4 manifest runs only Stage3-passing candidate policies."
        ),
        "claim_boundary": "Stage4 run validates four-sequence geometry; semantic causality still requires Stage6 controls.",
        "outputs": {
            "action_config_rows": rel(STAGE4 / "action_config_rows.csv"),
            "candidate_policy_rows": rel(STAGE4 / "candidate_policy_rows.csv"),
            "run_manifest": rel(STAGE4 / "run_manifest.csv"),
            "workspace": rel(stage2.WORKSPACE),
            "raw_action": rel(stage2.RAW_ACTION),
            "config_generation_summary": rel(STAGE4 / "stage4_config_generation_summary.json"),
        },
    }
    write_json(STAGE4 / "stage4_config_generation_summary.json", stage4_summary)
    print(json.dumps(stage2.clean_json(stage4_summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
