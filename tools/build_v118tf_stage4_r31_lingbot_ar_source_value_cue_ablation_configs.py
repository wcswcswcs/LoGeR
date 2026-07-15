#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R31 LingBot AR source-value cue ablations."""

from __future__ import annotations

import csv
import json
import math
import os
from statistics import median
from typing import Any


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r31")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r31_lingbot_ar_source_value_cue_ablation")
os.environ.setdefault("ACL2_V118_AR_BACKEND_LABEL", "sdpa")
os.environ.setdefault("ACL2_V118_AR_USE_SDPA", "true")
os.environ.setdefault("ACL2_V118_AR_ACTION_MODE", "anchor_source_value_scaling")

import build_v118tf_stage4_r28_lingbot_ar_anchor_read_configs as base


ROOT = base.ROOT
STAGE = base.STAGE
STAGE_TAG = base.STAGE_TAG
BACKEND_LABEL = base.BACKEND_LABEL
CONFIG_BASENAME = base.CONFIG_BASENAME
DATASET = base.DATASET
BASELINE_METHOD = base.BASELINE_METHOD
ANCHOR_FRAMES = base.ANCHOR_FRAMES
SEQS = base.SEQS
METHOD_PREFIX = base.METHOD_PREFIX
ACTION_MODE = base.ACTION_MODE
INTERVENTION_FORM = base.INTERVENTION_FORM

CUE_VARIANTS = [
    {
        "cue_variant": "internal_only",
        "suffix": "ar5_source_value_internal_only",
        "policy": "AR5_SOURCE_VALUE_INTERNAL_ONLY",
        "role": "candidate_internal_only",
        "cue_terms": ["internal"],
        "description": "source-value scaling using only internal anchor-read score",
    },
    {
        "cue_variant": "semantic_only",
        "suffix": "ar5_source_value_semantic_only",
        "policy": "AR5_SOURCE_VALUE_SEMANTIC_ONLY",
        "role": "candidate_semantic_only",
        "cue_terms": ["semantic"],
        "description": "source-value scaling using only semantic provenance score",
    },
    {
        "cue_variant": "internal_semantic",
        "suffix": "ar5_source_value_internal_semantic",
        "policy": "AR5_SOURCE_VALUE_INTERNAL_SEMANTIC",
        "role": "candidate_internal_semantic",
        "cue_terms": ["internal", "semantic"],
        "description": "source-value scaling using internal and semantic cues with original full-cue weights renormalized",
    },
    {
        "cue_variant": "internal_reliability",
        "suffix": "ar5_source_value_internal_reliability",
        "policy": "AR5_SOURCE_VALUE_INTERNAL_RELIABILITY",
        "role": "candidate_internal_reliability",
        "cue_terms": ["internal", "reliability"],
        "description": "source-value scaling using internal and reliability cues with original full-cue weights renormalized",
    },
    {
        "cue_variant": "semantic_reliability",
        "suffix": "ar5_source_value_semantic_reliability",
        "policy": "AR5_SOURCE_VALUE_SEMANTIC_RELIABILITY",
        "role": "candidate_semantic_reliability",
        "cue_terms": ["semantic", "reliability"],
        "description": "source-value scaling using semantic and reliability cues with original full-cue weights renormalized",
    },
    {
        "cue_variant": "full_three_way",
        "suffix": "ar5_source_value_full_three_way",
        "policy": "AR5_SOURCE_VALUE_FULL_THREE_WAY_REFERENCE",
        "role": "candidate_full_three_way",
        "cue_terms": ["internal", "semantic", "reliability"],
        "description": "R31 self-contained full three-way source-value scaling reference",
    },
]

ORIGINAL_FULL_COEFFICIENTS = {
    "internal": 0.55,
    "semantic": 0.30,
    "reliability": 0.15,
}


def cue_score(row: dict[str, Any], terms: list[str]) -> float:
    values = {
        "internal": float(row["internal_score"]),
        "semantic": float(row["semantic_score_norm"]),
        "reliability": float(row["reliability_score"]),
    }
    denom = sum(ORIGINAL_FULL_COEFFICIENTS[term] for term in terms)
    if denom <= 0:
        raise ValueError(f"empty cue terms: {terms}")
    return sum(ORIGINAL_FULL_COEFFICIENTS[term] * values[term] for term in terms) / denom


def weight_map_for_variant(scores: dict[int, dict[str, Any]], terms: list[str]) -> tuple[dict[int, float], dict[int, float]]:
    cue_scores = {frame: cue_score(scores[frame], terms) for frame in ANCHOR_FRAMES}
    score_median = median(cue_scores.values())
    weights = {
        frame: base.clamp(math.exp(1.20 * (cue_scores[frame] - score_median)), 0.55, 1.55)
        for frame in ANCHOR_FRAMES
    }
    return weights, cue_scores


def write_csv(path: Any, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    support = base.read_support()
    config_dir = STAGE / "configs"
    method_dir = config_dir / "methods"
    config_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    base.write_dataset_config(config_dir)

    scores_by_seq: dict[str, dict[int, dict[str, Any]]] = {}
    for seq in SEQS:
        scores_by_seq[seq] = base.frame_scores(seq, support.get(seq, {}), base.read_anchor_read_stats(seq))

    frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    manifest_methods: dict[str, dict[str, Any]] = {}
    concrete_methods: list[str] = []
    weight_maps_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for variant in CUE_VARIANTS:
        method_base = f"{METHOD_PREFIX}_{variant['suffix']}"
        cue_terms = list(variant["cue_terms"])
        manifest_methods[method_base] = {
            "policy": variant["policy"],
            "role": variant["role"],
            "description": variant["description"],
            "query_roles": [],
            "weight_mode": "cue_ablation",
            "cue_variant": variant["cue_variant"],
            "cue_terms": cue_terms,
            "cue_coefficients": {
                term: ORIGINAL_FULL_COEFFICIENTS[term] / sum(ORIGINAL_FULL_COEFFICIENTS[t] for t in cue_terms)
                for term in cue_terms
            },
        }
        weight_maps_by_method[method_base] = {}
        for seq in SEQS:
            weight_map, cue_scores = weight_map_for_variant(scores_by_seq[seq], cue_terms)
            method = f"{method_base}_seq{seq}"
            concrete_methods.append(method)
            (method_dir / f"{method}.yaml").write_text(
                base.concrete_method_yaml(
                    policy=str(variant["policy"]),
                    query_roles=[],
                    weight_map=weight_map,
                ),
                encoding="utf-8",
            )
            weight_maps_by_method[method_base][seq] = {str(k): v for k, v in sorted(weight_map.items())}
            weights = list(weight_map.values())
            manifest_rows.append(
                {
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_cue_ablation_manifest_row_v1",
                    "seq": seq,
                    "method_base": method_base,
                    "method": method,
                    "branch": "LB-AR",
                    "policy": variant["policy"],
                    "role": variant["role"],
                    "cue_variant": variant["cue_variant"],
                    "cue_terms": ";".join(cue_terms),
                    "source_context_roles": "scale_reference_context",
                    "token_roles": "patch",
                    "query_roles": "",
                    "source_frames": ";".join(str(frame) for frame in ANCHOR_FRAMES),
                    "weight_min": min(weights),
                    "weight_max": max(weights),
                    "weight_median": median(weights),
                    "weight_mode": "cue_ablation",
                }
            )
            cue_median = median(cue_scores.values())
            for frame in ANCHOR_FRAMES:
                row = dict(scores_by_seq[seq][frame])
                row.update(
                    {
                        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_cue_ablation_weight_row_v1",
                        "cue_variant": variant["cue_variant"],
                        "cue_terms": ";".join(cue_terms),
                        "cue_score": cue_scores[frame],
                        "cue_score_median": cue_median,
                        "cue_ablation_weight": weight_map[frame],
                    }
                )
                frame_rows.append(row)

    base.write_main_config(config_dir, concrete_methods)
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    weight_rows_name = f"stage4_{STAGE_TAG}_lingbot_ar_source_value_cue_ablation_weight_rows.csv"
    manifest_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_source_value_cue_ablation_manifest.csv"
    manifest_json_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
    write_csv(out / weight_rows_name, frame_rows)
    write_csv(out / manifest_csv_name, manifest_rows)
    manifest = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_source_value_cue_ablation_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_tag": STAGE_TAG,
        "backend_label": BACKEND_LABEL,
        "use_sdpa": True,
        "action_mode": ACTION_MODE,
        "intervention_form": INTERVENTION_FORM,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(base.SUPPORT.relative_to(ROOT)),
        "trace_dir": str(base.TRACE_DIR.relative_to(ROOT)),
        "branch": "LB-AR",
        "operation": "Anchor read source-value scaling cue ablation",
        "fixed_anchor_source_frames": list(ANCHOR_FRAMES),
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "r30_control_reference": str(
            (
                base.RESULT_ROOT
                / "stage4_r30_lingbot_ar_source_value_scaling/summary/stage4_r30_lingbot_ar_anchor_read_summary.json"
            ).relative_to(ROOT)
        ),
        "boundary": (
            "R31 keeps source frames, token roles, action mode, value-scaling hook, and weight formula fixed, "
            "then swaps the cue terms used to produce source-frame weights. It is a dev 00/02 mechanism "
            "dissection and cannot by itself claim global success."
        ),
        "manifest_csv": str((out / manifest_csv_name).relative_to(ROOT)),
        "weight_rows": str((out / weight_rows_name).relative_to(ROOT)),
    }
    (out / manifest_json_name).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": str(STAGE.relative_to(ROOT)),
                "config": str((config_dir / CONFIG_BASENAME).relative_to(ROOT)),
                "manifest_rows": len(manifest_rows),
                "weight_rows": len(frame_rows),
                "concrete_methods": concrete_methods,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
