#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R45 uniform-frame token-polarity source-value configs."""

from __future__ import annotations

import json
import os
from statistics import median
from typing import Any


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r45")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r45_lingbot_ar_uniform_token_polarity_source_value")
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
    f"{METHOD_PREFIX}_ar11_uniform_token_polarity_source_value": {
        "policy": "AR11_UNIFORM_TOKEN_POLARITY_SOURCE_VALUE",
        "role": "candidate",
        "mode_kind": "selected_polarity",
        "description": "uniform frame weights with internal/semantic-correlation selected token polarity",
    },
    f"{METHOD_PREFIX}_ar11_uniform_token_opposite_polarity_source_value_control": {
        "policy": "AR11_UNIFORM_TOKEN_OPPOSITE_POLARITY_SOURCE_VALUE_CONTROL",
        "role": "token_opposite_polarity_control",
        "mode_kind": "opposite_polarity",
        "description": "uniform frame weights with the opposite token polarity selected by the same rule",
    },
    f"{METHOD_PREFIX}_ar11_uniform_token_random_source_value_control": {
        "policy": "AR11_UNIFORM_TOKEN_RANDOM_SOURCE_VALUE_CONTROL",
        "role": "token_random_control",
        "mode_kind": "same_magnitude_random",
        "description": "uniform frame weights with deterministic same-magnitude random token signs",
    },
}


def token_polarity_for_seq(scores: dict[int, dict[str, Any]]) -> dict[str, Any]:
    correlation = r41.corr(
        [float(scores[frame]["internal_score"]) for frame in ANCHOR_FRAMES],
        [float(scores[frame]["semantic_score_norm"]) for frame in ANCHOR_FRAMES],
    )
    selected = "reverse_risk_x_frame" if correlation <= 0.0 else "risk_suppress_plus_stable_x_frame"
    opposite = "risk_suppress_plus_stable_x_frame" if correlation <= 0.0 else "reverse_risk_x_frame"
    return {
        "token_polarity_rule": "reverse_risk_if_internal_semantic_corr_le_0_else_risk_suppress_plus_stable",
        "internal_semantic_corr": correlation,
        "selected_token_weight_mode": selected,
        "opposite_token_weight_mode": opposite,
    }


def token_mode(meta: dict[str, Any], polarity_meta: dict[str, Any]) -> str:
    kind = str(meta["mode_kind"])
    if kind == "selected_polarity":
        return str(polarity_meta["selected_token_weight_mode"])
    if kind == "opposite_polarity":
        return str(polarity_meta["opposite_token_weight_mode"])
    if kind == "same_magnitude_random":
        return "same_magnitude_random_logit_x_frame"
    raise ValueError(f"unknown mode_kind: {kind}")


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
    polarity_meta_by_seq = {
        seq: token_polarity_for_seq(scores_by_seq[seq])
        for seq in SEQS
    }
    weight_map = {frame: 1.0 for frame in ANCHOR_FRAMES}

    polarity_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    manifest_methods: dict[str, dict[str, Any]] = {}
    concrete_methods: list[str] = []
    weight_maps_by_method: dict[str, dict[str, dict[str, float]]] = {}
    token_modes_by_method: dict[str, dict[str, str]] = {}

    for seq in SEQS:
        row = {
            "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_uniform_token_polarity_row_v1",
            "seq": seq,
            **polarity_meta_by_seq[seq],
        }
        polarity_rows.append(row)

    for method_base, meta in METHODS.items():
        manifest_methods[method_base] = {
            "policy": meta["policy"],
            "role": meta["role"],
            "description": meta["description"],
            "query_roles": [],
            "weight_mode": "uniform",
            "mode_kind": meta["mode_kind"],
            "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
            "uniform_frame_weight": 1.0,
            "token_polarity_rule": "reverse_risk_if_internal_semantic_corr_le_0_else_risk_suppress_plus_stable",
            "token_weight_mode_by_seq": {},
        }
        weight_maps_by_method[method_base] = {}
        token_modes_by_method[method_base] = {}
        for seq in SEQS:
            seq_polarity = polarity_meta_by_seq[seq]
            token_weight_mode = token_mode(meta, seq_polarity)
            manifest_methods[method_base]["token_weight_mode_by_seq"][seq] = token_weight_mode
            token_modes_by_method[method_base][seq] = token_weight_mode
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
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_uniform_token_polarity_manifest_row_v1",
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
                    **seq_polarity,
                }
            )

    base.write_main_config(config_dir, concrete_methods)
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    polarity_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_uniform_token_polarity_rows.csv"
    manifest_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_uniform_token_polarity_manifest.csv"
    manifest_json_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
    r41.write_csv(out / polarity_csv_name, polarity_rows)
    r41.write_csv(out / manifest_csv_name, manifest_rows)
    manifest = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_uniform_token_polarity_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_tag": STAGE_TAG,
        "backend_label": BACKEND_LABEL,
        "use_sdpa": True,
        "action_mode": base.ACTION_MODE,
        "intervention_form": f"{base.INTERVENTION_FORM}_uniform_token_polarity",
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(base.SUPPORT.relative_to(ROOT)),
        "trace_dir": str(base.TRACE_DIR.relative_to(ROOT)),
        "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
        "requires_token_weight_key_count": True,
        "branch": "LB-AR",
        "operation": "Uniform-frame token-polarity source-value repair",
        "fixed_anchor_source_frames": list(ANCHOR_FRAMES),
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "uniform_frame_weight": 1.0,
        "token_polarity_rule": "reverse_risk_if_internal_semantic_corr_le_0_else_risk_suppress_plus_stable",
        "polarity_meta_by_seq": polarity_meta_by_seq,
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "token_modes_by_method": token_modes_by_method,
        "r44_reference": "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r44_lingbot_ar_uniform_token_source_value/summary/stage4_r44_lingbot_ar_anchor_read_summary.json",
        "boundary": (
            "R45 keeps uniform source-frame weights fixed and tests whether a dev-side internal/semantic-correlation "
            "token-polarity rule solves the R44 reverse-control failure on 00/02. This is not a fresh holdout claim; "
            "R42 01/05 remains failed."
        ),
        "polarity_csv": str((out / polarity_csv_name).relative_to(ROOT)),
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
                "polarity_meta_by_seq": polarity_meta_by_seq,
                "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
                "concrete_methods": concrete_methods,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
