#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R43 frame-local agreement token source-value configs."""

from __future__ import annotations

import json
import math
import os
from statistics import median
from typing import Any


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r43")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r43_lingbot_ar_agreement_token_source_value")
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
    f"{METHOD_PREFIX}_ar9_agreement_token_source_value": {
        "policy": "AR9_AGREEMENT_TOKEN_SOURCE_VALUE",
        "role": "candidate",
        "weight_mode": "agreement",
        "token_weight_mode": "risk_suppress_plus_stable_x_frame",
        "description": "frame-local internal-semantic agreement weights multiplied by token-level semantic weights",
    },
    f"{METHOD_PREFIX}_ar9_opposite_agreement_token_source_value_control": {
        "policy": "AR9_OPPOSITE_AGREEMENT_TOKEN_SOURCE_VALUE_CONTROL",
        "role": "opposite_frame_control",
        "weight_mode": "opposite_agreement",
        "token_weight_mode": "risk_suppress_plus_stable_x_frame",
        "description": "reciprocal frame-local agreement control with matched token semantic weights",
    },
    f"{METHOD_PREFIX}_ar9_rotated_agreement_token_source_value_control": {
        "policy": "AR9_ROTATED_AGREEMENT_TOKEN_SOURCE_VALUE_CONTROL",
        "role": "rotated_frame_control",
        "weight_mode": "rotated_agreement",
        "token_weight_mode": "risk_suppress_plus_stable_x_frame",
        "description": "rotated frame-local agreement control with matched token semantic weights",
    },
    f"{METHOD_PREFIX}_ar9_token_reverse_agreement_source_value_control": {
        "policy": "AR9_TOKEN_REVERSE_AGREEMENT_SOURCE_VALUE_CONTROL",
        "role": "token_reverse_control",
        "weight_mode": "agreement",
        "token_weight_mode": "reverse_risk_x_frame",
        "description": "same frame-local agreement weights with reverse token semantic formula",
    },
    f"{METHOD_PREFIX}_ar9_token_random_agreement_source_value_control": {
        "policy": "AR9_TOKEN_RANDOM_AGREEMENT_SOURCE_VALUE_CONTROL",
        "role": "token_random_control",
        "weight_mode": "agreement",
        "token_weight_mode": "same_magnitude_random_logit_x_frame",
        "description": "same frame-local agreement weights with deterministic same-magnitude random token signs",
    },
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def agreement_maps_for_seq(scores: dict[int, dict[str, Any]]) -> tuple[dict[str, dict[int, float]], dict[int, dict[str, float]]]:
    internal_mid = median(float(scores[frame]["internal_score"]) for frame in ANCHOR_FRAMES)
    semantic_mid = median(float(scores[frame]["semantic_score_norm"]) for frame in ANCHOR_FRAMES)
    agreement: dict[int, float] = {}
    weights: dict[int, float] = {}
    meta: dict[int, dict[str, float]] = {}
    for frame in ANCHOR_FRAMES:
        internal_centered = float(scores[frame]["internal_score"]) - internal_mid
        semantic_centered = float(scores[frame]["semantic_score_norm"]) - semantic_mid
        value = internal_centered * semantic_centered
        weight = clamp(math.exp(3.0 * value), 0.65, 1.45)
        agreement[frame] = value
        weights[frame] = weight
        meta[frame] = {
            "agreement_internal_centered": internal_centered,
            "agreement_semantic_centered": semantic_centered,
            "agreement_score": value,
        }
    opposite = {frame: clamp(1.0 / weight, 0.65, 1.45) for frame, weight in weights.items()}
    return (
        {
            "agreement": weights,
            "opposite_agreement": opposite,
            "rotated_agreement": r41.rotate_map(weights),
        },
        meta,
    )


def main() -> None:
    if not TOKEN_ROOT.exists():
        raise FileNotFoundError(TOKEN_ROOT)
    support = base.read_support()
    config_dir = STAGE / "configs"
    method_dir = config_dir / "methods"
    config_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    base.write_dataset_config(config_dir)

    scores_by_seq = {
        seq: base.frame_scores(seq, support.get(seq, {}), base.read_anchor_read_stats(seq))
        for seq in SEQS
    }
    maps_by_seq: dict[str, dict[str, dict[int, float]]] = {}
    agreement_meta_by_seq: dict[str, dict[int, dict[str, float]]] = {}
    for seq in SEQS:
        maps_by_seq[seq], agreement_meta_by_seq[seq] = agreement_maps_for_seq(scores_by_seq[seq])

    frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    manifest_methods: dict[str, dict[str, Any]] = {}
    concrete_methods: list[str] = []
    weight_maps_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for seq in SEQS:
        for frame in ANCHOR_FRAMES:
            row = dict(scores_by_seq[seq][frame])
            row.update(agreement_meta_by_seq[seq][frame])
            for key, weight_map in maps_by_seq[seq].items():
                row[f"{key}_weight"] = weight_map[frame]
            frame_rows.append(row)

    for method_base, meta in METHODS.items():
        mode = str(meta["weight_mode"])
        token_weight_mode = str(meta["token_weight_mode"])
        manifest_methods[method_base] = {
            "policy": meta["policy"],
            "role": meta["role"],
            "description": meta["description"],
            "query_roles": [],
            "weight_mode": mode,
            "token_weight_mode": token_weight_mode,
            "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
            "agreement_rule": "weight=clip(exp(3.0*(internal_score-med_internal)*(semantic_score_norm-med_semantic)),0.65,1.45)",
        }
        weight_maps_by_method[method_base] = {}
        for seq in SEQS:
            weight_map = maps_by_seq[seq][mode]
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
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_agreement_token_source_value_manifest_row_v1",
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
                    "weight_mode": mode,
                    "token_weight_mode": token_weight_mode,
                }
            )

    base.write_main_config(config_dir, concrete_methods)
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    weight_rows_name = f"stage4_{STAGE_TAG}_lingbot_ar_agreement_token_source_value_weight_rows.csv"
    manifest_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_agreement_token_source_value_manifest.csv"
    manifest_json_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
    r41.write_csv(out / weight_rows_name, frame_rows)
    r41.write_csv(out / manifest_csv_name, manifest_rows)
    manifest = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_agreement_token_source_value_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_tag": STAGE_TAG,
        "backend_label": BACKEND_LABEL,
        "use_sdpa": True,
        "action_mode": base.ACTION_MODE,
        "intervention_form": f"{base.INTERVENTION_FORM}_agreement_token_gated",
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(base.SUPPORT.relative_to(ROOT)),
        "trace_dir": str(base.TRACE_DIR.relative_to(ROOT)),
        "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
        "requires_token_weight_key_count": True,
        "branch": "LB-AR",
        "operation": "Frame-local internal-semantic agreement source-value repair",
        "fixed_anchor_source_frames": list(ANCHOR_FRAMES),
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "agreement_rule": "weight=clip(exp(3.0*(internal_score-med_internal)*(semantic_score_norm-med_semantic)),0.65,1.45)",
        "token_frame_combination_rule": "token_weight_mode suffix _x_frame multiplies token semantic weights by frame-local agreement weights",
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "r41_reference": "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r41_lingbot_ar_token_gated_oriented_source_value/summary/stage4_r41_lingbot_ar_anchor_read_summary.json",
        "r42_boundary": "R42 failed fresh 01/05; this R43 run is a new 00/02 development mechanism and cannot use 01/05 as fresh validation without a new held-out protocol.",
        "boundary": (
            "R43 replaces sequence-level orientation with frame-local internal-semantic agreement. "
            "It is evaluated first on 00/02 with matched controls; no global success claim is possible from this dev run alone."
        ),
        "manifest_csv": str((out / manifest_csv_name).relative_to(ROOT)),
        "weight_rows": str((out / weight_rows_name).relative_to(ROOT)),
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
                "weight_rows": len(frame_rows),
                "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
                "concrete_methods": concrete_methods,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
