#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R44 uniform-frame token-only source-value configs."""

from __future__ import annotations

import json
import os
from statistics import median
from typing import Any


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r44")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r44_lingbot_ar_uniform_token_source_value")
os.environ.setdefault("ACL2_V118_AR_BACKEND_LABEL", "sdpa")
os.environ.setdefault("ACL2_V118_AR_USE_SDPA", "true")
os.environ.setdefault("ACL2_V118_AR_ACTION_MODE", "anchor_source_value_scaling")

import build_v118tf_stage4_r41_lingbot_ar_token_gated_oriented_source_value_configs as r41


ROOT = r41.ROOT
STAGE = r41.STAGE
STAGE_TAG = r41.STAGE_TAG
BACKEND_LABEL = r41.BACKEND_LABEL
CONFIG_BASENAME = r41.CONFIG_BASENAME
DATASET = r41.DATASET
BASELINE_METHOD = r41.BASELINE_METHOD
ANCHOR_FRAMES = r41.ANCHOR_FRAMES
SEQS = r41.SEQS
METHOD_PREFIX = r41.METHOD_PREFIX
TOKEN_ROOT = r41.TOKEN_ROOT
base = r41.base


METHODS = {
    f"{METHOD_PREFIX}_ar10_uniform_token_source_value": {
        "policy": "AR10_UNIFORM_TOKEN_SOURCE_VALUE",
        "role": "candidate",
        "token_weight_mode": "risk_suppress_plus_stable_x_frame",
        "description": "uniform frame weights with token-level risk_suppress_plus_stable semantic source-value weights",
    },
    f"{METHOD_PREFIX}_ar10_uniform_token_reverse_source_value_control": {
        "policy": "AR10_UNIFORM_TOKEN_REVERSE_SOURCE_VALUE_CONTROL",
        "role": "token_reverse_control",
        "token_weight_mode": "reverse_risk_x_frame",
        "description": "uniform frame weights with reverse token semantic formula",
    },
    f"{METHOD_PREFIX}_ar10_uniform_token_random_source_value_control": {
        "policy": "AR10_UNIFORM_TOKEN_RANDOM_SOURCE_VALUE_CONTROL",
        "role": "token_random_control",
        "token_weight_mode": "same_magnitude_random_logit_x_frame",
        "description": "uniform frame weights with deterministic same-magnitude random token signs",
    },
}


def main() -> None:
    if not TOKEN_ROOT.exists():
        raise FileNotFoundError(TOKEN_ROOT)
    config_dir = STAGE / "configs"
    method_dir = config_dir / "methods"
    config_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    base.write_dataset_config(config_dir)

    weight_map = {frame: 1.0 for frame in ANCHOR_FRAMES}
    manifest_rows: list[dict[str, Any]] = []
    manifest_methods: dict[str, dict[str, Any]] = {}
    concrete_methods: list[str] = []
    weight_maps_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for method_base, meta in METHODS.items():
        token_weight_mode = str(meta["token_weight_mode"])
        manifest_methods[method_base] = {
            "policy": meta["policy"],
            "role": meta["role"],
            "description": meta["description"],
            "query_roles": [],
            "weight_mode": "uniform",
            "token_weight_mode": token_weight_mode,
            "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
            "uniform_frame_weight": 1.0,
        }
        weight_maps_by_method[method_base] = {}
        for seq in SEQS:
            method = f"{method_base}_seq{seq}"
            concrete_methods.append(method)
            (method_dir / f"{method}.yaml").write_text(
                r41.concrete_method_yaml(
                    policy=str(meta["policy"]),
                    query_roles=[],
                    weight_map=weight_map,
                    token_weight_mode=token_weight_mode,
                ),
                encoding="utf-8",
            )
            weight_maps_by_method[method_base][seq] = {
                str(k): v for k, v in sorted(weight_map.items())
            }
            weights = list(weight_map.values())
            manifest_rows.append(
                {
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_uniform_token_source_value_manifest_row_v1",
                    "seq": seq,
                    "method_base": method_base,
                    "method": method,
                    "branch": "LB-AR",
                    "policy": meta["policy"],
                    "role": meta["role"],
                    "source_context_roles": "scale_reference_context",
                    "token_roles": "patch",
                    "query_roles": "",
                    "source_frames": ";".join(str(frame) for frame in ANCHOR_FRAMES),
                    "weight_min": min(weights),
                    "weight_max": max(weights),
                    "weight_median": median(weights),
                    "weight_mode": "uniform",
                    "token_weight_mode": token_weight_mode,
                }
            )

    base.write_main_config(config_dir, concrete_methods)
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    manifest_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_uniform_token_source_value_manifest.csv"
    manifest_json_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
    r41.write_csv(out / manifest_csv_name, manifest_rows)
    manifest = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_uniform_token_source_value_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_tag": STAGE_TAG,
        "backend_label": BACKEND_LABEL,
        "use_sdpa": True,
        "action_mode": base.ACTION_MODE,
        "intervention_form": f"{base.INTERVENTION_FORM}_uniform_token_only",
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(base.SUPPORT.relative_to(ROOT)),
        "trace_dir": str(base.TRACE_DIR.relative_to(ROOT)),
        "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
        "requires_token_weight_key_count": True,
        "branch": "LB-AR",
        "operation": "Uniform-frame token-only source-value repair",
        "fixed_anchor_source_frames": list(ANCHOR_FRAMES),
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "uniform_frame_weight": 1.0,
        "token_frame_combination_rule": "frame weights are all 1.0; token_weight_mode suffix _x_frame is retained only to use the existing hook path",
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "r43_reference": "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r43_lingbot_ar_agreement_token_source_value/summary/stage4_r43_lingbot_ar_anchor_read_summary.json",
        "boundary": (
            "R44 removes frame-weight placement as a degree of freedom and tests whether token semantic weights alone "
            "carry beneficial operation-specific source-value information on 00/02. No global success claim is possible from this dev run alone."
        ),
        "manifest_csv": str((out / manifest_csv_name).relative_to(ROOT)),
    }
    (out / manifest_json_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": str(STAGE.relative_to(ROOT)),
                "config": str((config_dir / CONFIG_BASENAME).relative_to(ROOT)),
                "manifest_rows": len(manifest_rows),
                "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
                "concrete_methods": concrete_methods,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
