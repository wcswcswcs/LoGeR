#!/usr/bin/env python3
"""Generate v109TF Stage2 role-specific hard-negative safety candidate configs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_role_specific_full_candidate_configs as fullcfg  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
OUT = RESULT_ROOT / "stage2_role_specific_safety_candidates"
POLICY_ID = "F19_dynamic_or_special_admitted_high_risk_else_weak_context"


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def install_overrides() -> None:
    fullcfg.OUT = OUT
    fullcfg.CONFIG_ROOT = OUT / "configs"
    fullcfg.METHOD_DIR = fullcfg.CONFIG_ROOT / "methods"
    fullcfg.DATASET_DIR = fullcfg.CONFIG_ROOT / "datasets"
    fullcfg.WORKSPACE = OUT / "workspace"
    fullcfg.RAW_ACTION = OUT / "raw_action"
    fullcfg.POLICIES = ((POLICY_ID, "dynamic_or_special_admitted_high_risk_else_weak_context"),)

    original_role_score = fullcfg.rolecfg.role_score

    def role_score(policy_id: str, case: dict[str, Any]) -> float:
        if policy_id != POLICY_ID:
            return original_role_score(policy_id, case)
        admitted = fnum(case.get("dynamic_mass")) >= 0.02
        if admitted:
            return 1000.0 + original_role_score("F13_dynamic_boundary_only", case)
        return original_role_score("F14_weak_context_only", case)

    fullcfg.rolecfg.role_score = role_score


def build() -> dict[str, Any]:
    install_overrides()
    summary = fullcfg.build()
    summary["schema"] = "acl2_v109tf_stage2_role_safety_config_summary_v1"
    summary["safety_candidate_config_ready"] = summary.pop("role_full_candidate_config_ready")
    summary["candidate_basis"] = (
        "F19 hard-negative safety repair from role-full attribution: use high-risk/boundary frames only "
        "when dynamic_mass>=0.02, otherwise fall back to weak-context ranking. Special-token count is logged "
        "but not used for admission because seq01 had special-only false positives."
    )
    summary["policies"] = [POLICY_ID]
    summary["outputs"]["hard_negative_attribution"] = fullcfg.rel(
        RESULT_ROOT
        / "stage2_role_specific_full_candidates/F_seq01_seq05_hard_negative_attribution.md"
    )
    fullcfg.write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
