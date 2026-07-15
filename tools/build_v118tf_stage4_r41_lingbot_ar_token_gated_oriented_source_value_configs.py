#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R41 token-gated oriented source-value configs."""

from __future__ import annotations

import csv
import json
import math
import os
from statistics import median
from typing import Any


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r41")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r41_lingbot_ar_token_gated_oriented_source_value")
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
ROTATED_FRAMES = list(ANCHOR_FRAMES[3:]) + list(ANCHOR_FRAMES[:3])
TOKEN_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence/task2_l2t/token_semantics"


METHODS = {
    f"{METHOD_PREFIX}_ar8_token_frame_semantic_source_value": {
        "policy": "AR8_TOKEN_FRAME_SEMANTIC_SOURCE_VALUE",
        "role": "candidate",
        "weight_mode": "corr_oriented",
        "token_weight_mode": "risk_suppress_plus_stable_x_frame",
        "description": "R40 correlation-oriented frame source-value weights multiplied by token-level risk_suppress_plus_stable semantic weights",
    },
    f"{METHOD_PREFIX}_ar8_opposite_frame_token_source_value_control": {
        "policy": "AR8_OPPOSITE_FRAME_TOKEN_SOURCE_VALUE_CONTROL",
        "role": "opposite_frame_control",
        "weight_mode": "opposite_sign",
        "token_weight_mode": "risk_suppress_plus_stable_x_frame",
        "description": "matched token semantic weights with opposite frame orientation",
    },
    f"{METHOD_PREFIX}_ar8_rotated_frame_token_source_value_control": {
        "policy": "AR8_ROTATED_FRAME_TOKEN_SOURCE_VALUE_CONTROL",
        "role": "rotated_frame_control",
        "weight_mode": "rotated_source",
        "token_weight_mode": "risk_suppress_plus_stable_x_frame",
        "description": "matched token semantic weights with rotated oriented frame assignment",
    },
    f"{METHOD_PREFIX}_ar8_token_reverse_source_value_control": {
        "policy": "AR8_TOKEN_REVERSE_SOURCE_VALUE_CONTROL",
        "role": "token_reverse_control",
        "weight_mode": "corr_oriented",
        "token_weight_mode": "reverse_risk_x_frame",
        "description": "matched oriented frame weights with reverse token semantic formula",
    },
    f"{METHOD_PREFIX}_ar8_token_random_source_value_control": {
        "policy": "AR8_TOKEN_RANDOM_SOURCE_VALUE_CONTROL",
        "role": "token_random_control",
        "weight_mode": "corr_oriented",
        "token_weight_mode": "same_magnitude_random_logit_x_frame",
        "description": "matched oriented frame weights with deterministic same-magnitude random token signs",
    },
}


def corr(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def orientation_for_seq(scores: dict[int, dict[str, Any]]) -> tuple[str, float]:
    value = corr(
        [float(scores[frame]["internal_score"]) for frame in ANCHOR_FRAMES],
        [float(scores[frame]["semantic_score_norm"]) for frame in ANCHOR_FRAMES],
    )
    return ("direct" if value <= 0.0 else "reverse"), value


def direct_map(scores: dict[int, dict[str, Any]]) -> dict[int, float]:
    return {frame: float(scores[frame]["candidate_weight"]) for frame in ANCHOR_FRAMES}


def reverse_map(scores: dict[int, dict[str, Any]]) -> dict[int, float]:
    return {frame: float(scores[frame]["reverse_weight"]) for frame in ANCHOR_FRAMES}


def rotate_map(weight_map: dict[int, float]) -> dict[int, float]:
    return {
        frame: float(weight_map[source_frame])
        for frame, source_frame in zip(ANCHOR_FRAMES, ROTATED_FRAMES)
    }


def maps_for_seq(scores: dict[int, dict[str, Any]]) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    orientation, correlation = orientation_for_seq(scores)
    direct = direct_map(scores)
    reverse = reverse_map(scores)
    oriented = direct if orientation == "direct" else reverse
    opposite = reverse if orientation == "direct" else direct
    return (
        {
            "corr_oriented": oriented,
            "opposite_sign": opposite,
            "rotated_source": rotate_map(oriented),
        },
        {
            "orientation_rule": "direct_if_internal_semantic_corr_le_0_else_reverse",
            "orientation": orientation,
            "internal_semantic_corr": correlation,
        },
    )


def write_csv(path: Any, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def concrete_method_yaml(
    *,
    policy: str,
    query_roles: list[str],
    weight_map: dict[int, float],
    token_weight_mode: str,
) -> str:
    return "\n".join(
        [
            "model: lingbot_map",
            "env: loger",
            f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
            "_device: cuda",
            "_use_amp: true",
            f"_use_sdpa: {str(base.USE_SDPA).lower()}",
            "_image_size: 518",
            "_patch_size: 14",
            "_enable_3d_rope: true",
            "_num_scale_frames: 8",
            "_max_frame_num: 1024",
            "_kv_cache_sliding_window: 64",
            "_kv_cache_scale_frames: 8",
            "_auto_keyframe_threshold: 320",
            "_area_budget: 255000",
            "_align: 14",
            "_mode: streaming",
            "_keyframe_interval: auto",
            f"_stage4_action_mode: {base.ACTION_MODE}",
            f"_stage4_action_label: {policy}",
            f"_stage4_anchor_source_weight_map: {json.dumps({str(k): v for k, v in sorted(weight_map.items())}, sort_keys=True)}",
            '_stage4_anchor_source_token_roles: ["patch"]',
            f"_stage4_anchor_source_query_roles: {json.dumps(query_roles)}",
            '_stage4_anchor_source_context_roles: ["scale_reference_context"]',
            f"_stage4_anchor_source_token_weight_root: {json.dumps(str(TOKEN_ROOT.resolve()))}",
            f"_stage4_anchor_source_token_weight_mode: {json.dumps(token_weight_mode)}",
            "",
        ]
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
    orientation_meta_by_seq: dict[str, dict[str, Any]] = {}
    for seq in SEQS:
        maps_by_seq[seq], orientation_meta_by_seq[seq] = maps_for_seq(scores_by_seq[seq])

    frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    manifest_methods: dict[str, dict[str, Any]] = {}
    concrete_methods: list[str] = []
    weight_maps_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for seq in SEQS:
        for frame in ANCHOR_FRAMES:
            row = dict(scores_by_seq[seq][frame])
            row.update(orientation_meta_by_seq[seq])
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
            "orientation_rule": "direct_if_internal_semantic_corr_le_0_else_reverse",
        }
        weight_maps_by_method[method_base] = {}
        for seq in SEQS:
            weight_map = maps_by_seq[seq][mode]
            method = f"{method_base}_seq{seq}"
            concrete_methods.append(method)
            (method_dir / f"{method}.yaml").write_text(
                concrete_method_yaml(
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
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_token_gated_oriented_source_value_manifest_row_v1",
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
                    **orientation_meta_by_seq[seq],
                }
            )

    base.write_main_config(config_dir, concrete_methods)
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    weight_rows_name = f"stage4_{STAGE_TAG}_lingbot_ar_token_gated_oriented_source_value_weight_rows.csv"
    manifest_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_token_gated_oriented_source_value_manifest.csv"
    manifest_json_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
    write_csv(out / weight_rows_name, frame_rows)
    write_csv(out / manifest_csv_name, manifest_rows)
    manifest = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_token_gated_oriented_source_value_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_tag": STAGE_TAG,
        "backend_label": BACKEND_LABEL,
        "use_sdpa": True,
        "action_mode": base.ACTION_MODE,
        "intervention_form": f"{base.INTERVENTION_FORM}_token_gated",
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(base.SUPPORT.relative_to(ROOT)),
        "trace_dir": str(base.TRACE_DIR.relative_to(ROOT)),
        "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
        "branch": "LB-AR",
        "operation": "Token-gated source-value correlation-oriented repair",
        "fixed_anchor_source_frames": list(ANCHOR_FRAMES),
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "orientation_rule": "direct_if_internal_semantic_corr_le_0_else_reverse",
        "token_frame_combination_rule": "token_weight_mode suffix _x_frame multiplies token semantic weights by frame orientation weights",
        "orientation_meta_by_seq": orientation_meta_by_seq,
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "r40_reference": "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r40_lingbot_ar_oriented_source_value/summary/stage4_r40_lingbot_ar_anchor_read_summary.json",
        "boundary": (
            "R41 tests whether R40 source-value orientation becomes operation-specific after token-level semantic gating. "
            "A pass requires beating opposite-frame, rotated-frame, token-reverse, and token-random controls on 00/02, "
            "then fresh 01/05 validation; this dev run alone is not a global success claim."
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
                "orientation_meta_by_seq": orientation_meta_by_seq,
                "token_weight_root": str(TOKEN_ROOT.relative_to(ROOT)),
                "concrete_methods": concrete_methods,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
