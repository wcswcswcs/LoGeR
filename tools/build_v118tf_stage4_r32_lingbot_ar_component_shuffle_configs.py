#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R32 LingBot AR component-shuffle mechanism configs."""

from __future__ import annotations

import csv
import json
import math
import os
from statistics import median
from typing import Any


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r32")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r32_lingbot_ar_component_shuffle")
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
ROTATED_FRAMES = list(ANCHOR_FRAMES[3:]) + list(ANCHOR_FRAMES[:3])

COMPONENT_VARIANTS = [
    {
        "cue_variant": "full_internal_shuffled",
        "suffix": "ar6_full_internal_shuffled",
        "policy": "AR6_FULL_INTERNAL_SHUFFLED",
        "role": "candidate_full_internal_shuffled",
        "shuffle_component": "internal",
        "description": "full three-way source-value score with internal component shuffled across source frames",
    },
    {
        "cue_variant": "full_semantic_shuffled",
        "suffix": "ar6_full_semantic_shuffled",
        "policy": "AR6_FULL_SEMANTIC_SHUFFLED",
        "role": "candidate_full_semantic_shuffled",
        "shuffle_component": "semantic",
        "description": "full three-way source-value score with semantic component shuffled across source frames",
    },
    {
        "cue_variant": "full_reliability_shuffled",
        "suffix": "ar6_full_reliability_shuffled",
        "policy": "AR6_FULL_RELIABILITY_SHUFFLED",
        "role": "candidate_full_reliability_shuffled",
        "shuffle_component": "reliability",
        "description": "full three-way source-value score with reliability component shuffled across source frames",
    },
]


def component_value(scores: dict[int, dict[str, Any]], frame: int, component: str, shuffle_component: str) -> float:
    source_frame = ROTATED_FRAMES[list(ANCHOR_FRAMES).index(frame)] if component == shuffle_component else frame
    row = scores[source_frame]
    if component == "internal":
        return float(row["internal_score"])
    if component == "semantic":
        return float(row["semantic_score_norm"])
    if component == "reliability":
        return float(row["reliability_score"])
    raise ValueError(component)


def full_score_with_shuffle(scores: dict[int, dict[str, Any]], frame: int, shuffle_component: str) -> float:
    internal = component_value(scores, frame, "internal", shuffle_component)
    semantic = component_value(scores, frame, "semantic", shuffle_component)
    reliability = component_value(scores, frame, "reliability", shuffle_component)
    return 0.55 * internal + 0.30 * semantic + 0.15 * reliability


def weight_map_for_variant(scores: dict[int, dict[str, Any]], shuffle_component: str) -> tuple[dict[int, float], dict[int, float]]:
    cue_scores = {
        frame: full_score_with_shuffle(scores, frame, shuffle_component)
        for frame in ANCHOR_FRAMES
    }
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

    scores_by_seq = {
        seq: base.frame_scores(seq, support.get(seq, {}), base.read_anchor_read_stats(seq))
        for seq in SEQS
    }
    frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    manifest_methods: dict[str, dict[str, Any]] = {}
    concrete_methods: list[str] = []
    weight_maps_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for variant in COMPONENT_VARIANTS:
        method_base = f"{METHOD_PREFIX}_{variant['suffix']}"
        shuffle_component = str(variant["shuffle_component"])
        manifest_methods[method_base] = {
            "policy": variant["policy"],
            "role": variant["role"],
            "description": variant["description"],
            "query_roles": [],
            "weight_mode": "component_shuffle",
            "cue_variant": variant["cue_variant"],
            "shuffle_component": shuffle_component,
            "shuffle_rotation": 3,
            "cue_coefficients": {"internal": 0.55, "semantic": 0.30, "reliability": 0.15},
        }
        weight_maps_by_method[method_base] = {}
        for seq in SEQS:
            weight_map, cue_scores = weight_map_for_variant(scores_by_seq[seq], shuffle_component)
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
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_component_shuffle_manifest_row_v1",
                    "seq": seq,
                    "method_base": method_base,
                    "method": method,
                    "branch": "LB-AR",
                    "policy": variant["policy"],
                    "role": variant["role"],
                    "cue_variant": variant["cue_variant"],
                    "shuffle_component": shuffle_component,
                    "source_context_roles": "scale_reference_context",
                    "token_roles": "patch",
                    "query_roles": "",
                    "source_frames": ";".join(str(frame) for frame in ANCHOR_FRAMES),
                    "weight_min": min(weights),
                    "weight_max": max(weights),
                    "weight_median": median(weights),
                    "weight_mode": "component_shuffle",
                }
            )
            cue_median = median(cue_scores.values())
            for frame in ANCHOR_FRAMES:
                source_frame_for_shuffle = ROTATED_FRAMES[list(ANCHOR_FRAMES).index(frame)]
                row = dict(scores_by_seq[seq][frame])
                row.update(
                    {
                        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_component_shuffle_weight_row_v1",
                        "cue_variant": variant["cue_variant"],
                        "shuffle_component": shuffle_component,
                        "shuffle_source_frame_for_component": source_frame_for_shuffle,
                        "cue_score": cue_scores[frame],
                        "cue_score_median": cue_median,
                        "component_shuffle_weight": weight_map[frame],
                    }
                )
                frame_rows.append(row)

    base.write_main_config(config_dir, concrete_methods)
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    weight_rows_name = f"stage4_{STAGE_TAG}_lingbot_ar_component_shuffle_weight_rows.csv"
    manifest_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_component_shuffle_manifest.csv"
    manifest_json_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
    write_csv(out / weight_rows_name, frame_rows)
    write_csv(out / manifest_csv_name, manifest_rows)
    manifest = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_component_shuffle_manifest_v1",
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
        "operation": "Anchor read source-value full-cue component shuffle",
        "fixed_anchor_source_frames": list(ANCHOR_FRAMES),
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "r31_leave_one_out_reference": str(
            (
                base.RESULT_ROOT
                / "stage4_r31_lingbot_ar_source_value_cue_ablation/summary/stage4_r31_lingbot_ar_source_value_cue_ablation_summary.json"
            ).relative_to(ROOT)
        ),
        "r30_control_reference": str(
            (
                base.RESULT_ROOT
                / "stage4_r30_lingbot_ar_source_value_scaling/summary/stage4_r30_lingbot_ar_anchor_read_summary.json"
            ).relative_to(ROOT)
        ),
        "boundary": (
            "R32 shuffles one full-cue component at a time while keeping the source-value hook, source frames, "
            "token roles, and full score coefficients fixed. It is a 00/02 mechanism-control run."
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
