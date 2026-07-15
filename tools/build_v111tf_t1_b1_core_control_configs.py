#!/usr/bin/env python3
"""Generate ACL2 v111TF T1 B1 core-control full-sequence configs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v110r_stage2_candidate_configs as stage2  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
T1 = RESULT_ROOT / "batch_t_t1_b1_core_controls"
CONFIG_OUT = T1 / "config_generation"
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


def frame_counts() -> dict[str, int]:
    rows = stage2.read_csv(RESULT_ROOT / "stage0_evidence_freeze/full_kitti_baseline_table.csv")
    return {row["seq"]: int(float(row["frames"])) for row in rows if row.get("seq") in SEQUENCES}


def allowed_surface_rows() -> dict[str, dict[str, str]]:
    # v111 Stage0 uses memory-family rows. The v110 generator only needs these
    # fields for reporting, so provide a compact B-surface compatibility view.
    return {
        "B": {
            "surface_id": "B",
            "operation_type": "cache_append_write_control",
            "v110_status": "stage2_abef_candidate_allowed",
            "claim_boundary": "full_00_01_02_05_t1_core_control_required",
            "note": "v111 T1 core control generation reuses B1 force_non_keyframe action family",
            "implementation_status": "implementable_now",
            "has_existing_runtime_knob": "True",
            "new_hook_needed": "False",
            "full_sequence_pilot_allowed": "True",
        }
    }


def main() -> None:
    stage2.RESULT_ROOT = RESULT_ROOT
    stage2.STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
    stage2.OUT = CONFIG_OUT
    stage2.STAGE3 = T1
    stage2.CONFIG_ROOT = T1 / "configs"
    stage2.METHOD_DIR = stage2.CONFIG_ROOT / "methods"
    stage2.DATASET_DIR = stage2.CONFIG_ROOT / "datasets"
    stage2.WORKSPACE = T1 / "workspace"
    stage2.RAW_ACTION = T1 / "raw_action"
    stage2.SEQUENCES = SEQUENCES
    stage2.POLICY_FAMILIES = CONTROL_POLICY_FAMILIES
    stage2.RUNNABLE_CANDIDATES = {
        candidate_id: stage2.RUNNABLE_CANDIDATES[candidate_id]
        for candidate_id in SELECTED_CANDIDATES
    }
    stage2.frame_counts = frame_counts
    stage2.allowed_surface_rows = allowed_surface_rows

    summary = stage2.build()
    t1_summary = {
        "schema": "acl2_v111tf_t1_b1_core_control_config_summary_v1",
        "t1_core_config_ready": bool(summary.get("stage2_config_ready")),
        "blocker": summary.get("blocker", ""),
        "continuation_reason": (
            "v111 T1 starts by closing four-sequence B1 core controls missing from the v110R final claim boundary."
        ),
        "claim_boundary": (
            "This is a core-control subset only. It does not satisfy the full v111 T1 stronger-control set "
            "until same-bucket, schedule-only matched, role rotation, semantic-shuffle multi-seed, and random multi-seed controls are added."
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
        "outputs": {
            "action_config_rows": rel(T1 / "action_config_rows.csv"),
            "candidate_policy_rows": rel(T1 / "candidate_policy_rows.csv"),
            "run_manifest": rel(T1 / "run_manifest.csv"),
            "workspace": rel(stage2.WORKSPACE),
            "raw_action": rel(stage2.RAW_ACTION),
            "config_generation_summary": rel(T1 / "t1_core_config_generation_summary.json"),
        },
    }
    write_json(T1 / "t1_core_config_generation_summary.json", t1_summary)
    print(json.dumps(stage2.clean_json(t1_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
