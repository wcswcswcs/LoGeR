#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_v103_supp_r5_phaseR5_4_support_weighted_local_ap as base  # noqa: E402


PHASE_ID = "v103_supp_r6_phaseR6_2_support_conditioned_local_ap"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_local_ap"
DEFAULT_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_feature"
R6_VARIANTS = [
    "R6F0_anchor_only_replay",
    "R6F1_support005_specificity",
    "R6F2_support010_specificity_semantic",
    "R6F3_support010_specificity_semantic_vetoatten",
    "R6F4_support020_specificity_semantic_vetoatten",
    "R6F5_support010_semantic_gate_strict",
]


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    return base._jsonable(value)


def _patch_base() -> None:
    base.PHASE_ID = PHASE_ID
    base.DEFAULT_OUT = DEFAULT_OUT
    base.DEFAULT_R5_FEATURE_ROOT = DEFAULT_FEATURE_ROOT
    original_read_json = base._read_json

    def read_json_with_r6_feature_alias(path: Path) -> dict[str, Any]:
        data = original_read_json(path)
        if path.name == "summary.json" and "phase_r6_2_feature_pass" in data and "phase_r5_1_pass" not in data:
            data = dict(data)
            data["phase_r5_1_pass"] = bool(data.get("phase_r6_2_feature_pass"))
            data["passing_support_variants"] = list(data.get("passing_r6_feature_variants", []))
        return data

    base._read_json = read_json_with_r6_feature_alias


def build(args: argparse.Namespace) -> dict[str, Any]:
    _patch_base()
    summary = base.build(args)
    out = _project(args.output_root)
    summary = dict(summary)
    phase_pass = bool(summary.pop("phase_r5_4_diag_pass", False))
    summary.update(
        {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_2_local_ap_summary_v1",
            "phase_id": PHASE_ID,
            "decision": "PASS_R6_2_SUPPORT_CONDITIONED_LOCAL_AP_SIGNAL"
            if phase_pass
            else "NO_GO_R6_2_SUPPORT_CONDITIONED_LOCAL_AP",
            "phase_r6_2_local_ap_pass": phase_pass,
            "tested_r6_feature_variants": summary.get("tested_r5_feature_variants", []),
            "fully_passing_r6_feature_variants": summary.get("fully_passing_r5_feature_variants", []),
            "partially_passing_r6_gate_variants": summary.get("partially_passing_gate_variants", []),
            "r6_feature_root": base._rel(_project(args.r5_feature_root)),
            "truthfulness_note": (
                "R6-2 local AP wrapper runs the current Phase6d/v65 subset evaluator on pre-registered R6F0-R6F5 features. "
                "It is subset-only, uses GT only for evaluation/diagnostics, and does not allow full-dev/holdout/history unless the R6 subset gate passes."
            ),
        }
    )
    base._write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="R6 support-conditioned local AP diagnostic using Phase6d/v65.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r5-feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--variants", default=",".join(R6_VARIANTS))
    parser.add_argument("--max-variants", type=int, default=6)
    parser.add_argument("--phase6d-script", default=str(base.DEFAULT_PHASE6D_SCRIPT))
    parser.add_argument("--f2-root", default=str(base.DEFAULT_F2_ROOT))
    parser.add_argument("--phase9n-root", default=str(base.DEFAULT_PHASE9N_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(base.DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(base.DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--subset-baseline-rows", default=str(base.DEFAULT_SUBSET_BASELINE))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phase_r6_2_local_ap_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
