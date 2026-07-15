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

import build_v103_supp_r5_phaseR5_1_support_weighted_affinity as base  # noqa: E402


PHASE_ID = "v103_supp_r6_phaseR6_2_support_conditioned_feature"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_feature"
DEFAULT_FACT_LOCK_ROOT = AUDIT_ROOT / "v103_supp_r6_phase0_fact_lock"

R6_VARIANTS = [
    {
        "variant_id": "R6F0_anchor_only_replay",
        "support_lambda": 0.0,
        "semantic_filter": False,
        "veto_attenuation": False,
        "description": "R6F0 control: z = z_A",
    },
    {
        "variant_id": "R6F1_support005_specificity",
        "support_lambda": 0.05,
        "semantic_filter": False,
        "veto_attenuation": False,
        "description": "R6F1: z = z_A + 0.05 z_S_spec",
    },
    {
        "variant_id": "R6F2_support010_specificity_semantic",
        "support_lambda": 0.10,
        "semantic_filter": True,
        "veto_attenuation": False,
        "description": "R6F2: z = z_A + 0.10 z_S_spec_sem",
    },
    {
        "variant_id": "R6F3_support010_specificity_semantic_vetoatten",
        "support_lambda": 0.10,
        "semantic_filter": True,
        "veto_attenuation": True,
        "description": "R6F3: z = z_A + 0.10 z_S_spec_sem_veto",
    },
    {
        "variant_id": "R6F4_support020_specificity_semantic_vetoatten",
        "support_lambda": 0.20,
        "semantic_filter": True,
        "veto_attenuation": True,
        "description": "R6F4: z = z_A + 0.20 z_S_spec_sem_veto",
    },
    {
        "variant_id": "R6F5_support010_semantic_gate_strict",
        "support_lambda": 0.10,
        "semantic_filter": True,
        "semantic_gate_min": 0.75,
        "veto_attenuation": True,
        "description": "R6F5: z = z_A + 0.10 z_S_spec_sem_veto with alpha_sem >= 0.75",
    },
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
    base.DEFAULT_FACT_LOCK_ROOT = DEFAULT_FACT_LOCK_ROOT
    base.VARIANTS = R6_VARIANTS
    original_read_json = base._read_json

    def read_json_with_r6_fact_alias(path: Path) -> dict[str, Any]:
        data = original_read_json(path)
        if path.name == "summary.json" and "phase_r6_0_pass" in data and "phase_r5_0_pass" not in data:
            data = dict(data)
            data["phase_r5_0_pass"] = bool(data.get("phase_r6_0_pass"))
        return data

    base._read_json = read_json_with_r6_fact_alias


def build(args: argparse.Namespace) -> dict[str, Any]:
    _patch_base()
    summary = base.build(args)
    out = _project(args.output_root)
    summary = dict(summary)
    phase_pass = bool(summary.pop("phase_r5_1_pass", False))
    summary.update(
        {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_2_feature_summary_v1",
            "phase_id": PHASE_ID,
            "decision": "PASS_ENTER_PHASE_R6_2_LOCAL_AP"
            if phase_pass
            else "NO_GO_REPAIR_PHASE_R6_2_SUPPORT_CONDITIONED_FEATURE",
            "phase_r6_2_feature_pass": phase_pass,
            "r6_feature_family": "low_weight_specificity_semantic_veto_attenuated_support",
            "tested_r6_feature_variants": summary.get("variant_ids", []),
            "passing_r6_feature_variants": summary.get("passing_support_variants", []),
            "r6F0_anchor_only_pass": summary.get("f0_anchor_only_pass", ""),
            "fact_lock_root": base._rel(_project(args.fact_lock_root)),
            "truthfulness_note": (
                "R6-2 feature builder reuses the audited R5-1 primitive/mask-level arithmetic with R6 pre-registered variants. "
                "It does not run AP, does not use GT, and does not claim method success. "
                "R6F5 uses a GT-free fixed semantic gate alpha_sem >= 0.75."
            ),
        }
    )
    base._write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R6 Phase R6-2 support-conditioned feature builder.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--fact-lock-root", default=str(DEFAULT_FACT_LOCK_ROOT))
    parser.add_argument("--phaseS1-root", default=str(base.DEFAULT_PHASES1_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(base.DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(base.DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--exact-subset-size", type=int, default=4096)
    parser.add_argument("--max-pair-rows", type=int, default=4096)
    parser.add_argument("--topk-carriers", type=int, default=64)
    parser.add_argument("--trim-quantile", type=float, default=0.10)
    parser.add_argument("--specificity-mode", default="idf_object_preserve_downweight")
    parser.add_argument("--specificity-alpha", type=float, default=1.0)
    parser.add_argument("--affinity-risk-mode", default="source_and_competing_penalty")
    parser.add_argument("--exact-p95-threshold", type=float, default=0.005)
    parser.add_argument("--valid-rate-threshold", type=float, default=0.95)
    parser.add_argument("--support-ratio-min", type=float, default=0.02)
    parser.add_argument("--support-ratio-max", type=float, default=0.45)
    parser.add_argument("--broad-plus-budget", type=float, default=0.10)
    parser.add_argument("--veto-overlap-fraction-max", type=float, default=0.60)
    parser.add_argument("--save-primitive-features", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phase_r6_2_feature_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
