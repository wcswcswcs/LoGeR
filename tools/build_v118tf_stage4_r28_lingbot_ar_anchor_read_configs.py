#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R28 LingBot anchor-read configs.

R28 reopens the LB-AR branch with a real existing action surface:
``anchor_source_attention_weight``.  It keeps the default anchor set fixed
(source frames 0-7, ``scale_reference_context``) and only changes how selected
queries read those anchor source tokens.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import median, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_RESULT_ROOT_ENV = os.environ.get("ACL2_V118_AR_RESULT_ROOT", "").strip()
if _RESULT_ROOT_ENV:
    _result_root_path = Path(_RESULT_ROOT_ENV).expanduser()
    RESULT_ROOT = _result_root_path if _result_root_path.is_absolute() else ROOT / _result_root_path
else:
    RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
_REFERENCE_ROOT_ENV = os.environ.get("ACL2_V118_AR_REFERENCE_ROOT", "").strip()
if _REFERENCE_ROOT_ENV:
    _reference_root_path = Path(_REFERENCE_ROOT_ENV).expanduser()
    REFERENCE_ROOT = _reference_root_path if _reference_root_path.is_absolute() else ROOT / _reference_root_path
else:
    REFERENCE_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE_TAG = os.environ.get("ACL2_V118_AR_STAGE_TAG", "r28").strip().lower() or "r28"
BACKEND_LABEL = os.environ.get("ACL2_V118_AR_BACKEND_LABEL", "flashinfer").strip().lower() or "flashinfer"
ACTION_MODE = os.environ.get("ACL2_V118_AR_ACTION_MODE", "anchor_source_attention_weight").strip() or "anchor_source_attention_weight"
VALUE_WEIGHT_NORMALIZATION = (
    os.environ.get("ACL2_V118_AR_VALUE_WEIGHT_NORMALIZATION", "legacy_geometric_mean_1").strip()
    or "legacy_geometric_mean_1"
)
INTERVENTION_FORM = (
    "source_value_scaling"
    if ACTION_MODE == "anchor_source_value_scaling"
    else "selected_query_attention_weight"
)
DEFAULT_STAGE_SLUG = "stage4_r28_lingbot_ar_anchor_read"
STAGE_SLUG = os.environ.get("ACL2_V118_AR_STAGE_SLUG", DEFAULT_STAGE_SLUG).strip() or DEFAULT_STAGE_SLUG
USE_SDPA = os.environ.get("ACL2_V118_AR_USE_SDPA", "").strip().lower() in {"1", "true", "yes", "y"}
if STAGE_TAG == "r28" and BACKEND_LABEL == "flashinfer":
    CONFIG_BASENAME = "kitti_lingbot_flashinfer_r28_ar_anchor_read_full_reuse_v105gt.yaml"
else:
    CONFIG_BASENAME = f"kitti_lingbot_{BACKEND_LABEL}_v118_{STAGE_TAG}_ar_anchor_read_full_reuse_v105gt.yaml"
STAGE = RESULT_ROOT / STAGE_SLUG
SUPPORT = REFERENCE_ROOT / "stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_frame_semantic_support_rows.csv"
TRACE_DIR = REFERENCE_ROOT / "stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
ANCHOR_FRAMES = tuple(range(8))
SEQS = ("00", "02")
SPECIAL_QUERY_ROLES = ["camera_special", "register_special", "scale_special"]

METHOD_PREFIX = f"lingbot_map_stream_{BACKEND_LABEL}_v118_{STAGE_TAG}"

if ACTION_MODE == "anchor_source_value_scaling":
    METHODS = {
        f"{METHOD_PREFIX}_ar3_source_value_scaling": {
            "policy": "AR3_SOURCE_VALUE_SCALING_SEMANTIC_INTERNAL_RELIABILITY",
            "role": "candidate",
            "query_roles": [],
            "weight_mode": "combined",
            "description": "source value scaling over fixed default anchor source frames",
        },
        f"{METHOD_PREFIX}_ar3_reverse_source_value_scaling_control": {
            "policy": "AR3_REVERSE_SOURCE_VALUE_SCALING_CONTROL",
            "role": "reverse_control",
            "query_roles": [],
            "weight_mode": "reverse",
            "description": "matched source-value negative control with reciprocal weights",
        },
        f"{METHOD_PREFIX}_ar3_shuffle_source_value_scaling_control": {
            "policy": "AR3_SHUFFLED_SOURCE_VALUE_SCALING_CONTROL",
            "role": "shuffled_control",
            "query_roles": [],
            "weight_mode": "shuffle",
            "description": "matched source-value control that shuffles the source-frame weight assignment",
        },
    }
else:
    METHODS = {
        f"{METHOD_PREFIX}_ar4_selected_special_anchor_read_weight": {
            "policy": "AR4_SELECTED_SPECIAL_SEMANTIC_INTERNAL_RELIABILITY_ANCHOR_READ",
            "role": "candidate",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "combined",
            "description": "selected special-query logit bias over fixed default anchor source frames",
        },
        f"{METHOD_PREFIX}_ar2_patch_query_anchor_read_control": {
            "policy": "AR2_PATCH_QUERY_LOGIT_BIAS_CONTROL",
            "role": "patch_query_control",
            "query_roles": ["patch"],
            "weight_mode": "combined",
            "description": "same source weights as candidate, but applied to patch queries",
        },
        f"{METHOD_PREFIX}_ar4_reverse_selected_special_anchor_read_control": {
            "policy": "AR4_REVERSE_SELECTED_SPECIAL_CONTROL",
            "role": "reverse_control",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "reverse",
            "description": "matched selected-query negative control with reciprocal weights",
        },
    }


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_role_count(raw: str, role: str) -> int:
    total = 0
    prefix = f"{role}:"
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part.startswith(prefix):
            continue
        try:
            total += int(part.split(":", 1)[1])
        except ValueError:
            pass
    return total


def read_support() -> dict[str, dict[int, dict[str, str]]]:
    out: dict[str, dict[int, dict[str, str]]] = {}
    with SUPPORT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(str(row["seq"]), {})[int(float(row["frame_id"]))] = row
    return out


def read_anchor_read_stats(seq: str) -> dict[int, dict[str, Any]]:
    path = TRACE_DIR / f"seq{seq}_flashinfer_trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    stats: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "read":
                continue
            if row.get("memory_family") != "anchor":
                continue
            if row.get("token_type") != "image_patch":
                continue
            frame_raw = row.get("source_frame_id")
            if frame_raw is None:
                continue
            frame = int(frame_raw)
            if frame not in ANCHOR_FRAMES:
                continue
            bucket = stats.setdefault(
                frame,
                {
                    "frame_id": frame,
                    "read_rows": 0,
                    "qk_cosines": [],
                    "qk_softmaxes": [],
                    "qk_ranks": [],
                    "entropies": [],
                },
            )
            bucket["read_rows"] += 1
            bucket["qk_cosines"].append(fnum(row.get("qk_relevance_cosine")))
            bucket["qk_softmaxes"].append(fnum(row.get("qk_relevance_softmax")))
            bucket["qk_ranks"].append(fnum(row.get("qk_relevance_rank")))
            bucket["entropies"].append(fnum(row.get("read_entropy_normalized")))
    missing = [frame for frame in ANCHOR_FRAMES if frame not in stats]
    if missing:
        raise RuntimeError(f"missing anchor read rows for seq {seq}: {missing}")
    for bucket in stats.values():
        n = int(bucket["read_rows"])
        bucket["mean_qk_cosine"] = sum(bucket["qk_cosines"]) / n if n else 0.0
        bucket["mean_qk_softmax"] = sum(bucket["qk_softmaxes"]) / n if n else 0.0
        bucket["mean_qk_rank"] = sum(bucket["qk_ranks"]) / n if n else 0.0
        bucket["mean_read_entropy"] = sum(bucket["entropies"]) / n if n else 0.0
        bucket["qk_cosine_std"] = pstdev(bucket["qk_cosines"]) if n > 1 else 0.0
    return stats


def normalize(values: dict[int, float], *, invert: bool = False) -> dict[int, float]:
    vals = list(values.values())
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-12:
        return {frame: 0.5 for frame in values}
    out: dict[int, float] = {}
    for frame, value in values.items():
        norm = (value - lo) / (hi - lo)
        out[frame] = 1.0 - norm if invert else norm
    return out


def semantic_score(row: dict[str, str]) -> float:
    visible = max(1, int(fnum(row.get("visible_track_rows"))))
    stable = parse_role_count(row.get("top_roles", ""), "stable_landmark") / visible
    vegetation = parse_role_count(row.get("top_roles", ""), "vegetation_repetitive") / visible
    weak = parse_role_count(row.get("top_roles", ""), "weak_context") / visible
    dynamic = parse_role_count(row.get("top_roles", ""), "dynamic") / visible
    sky = parse_role_count(row.get("top_roles", ""), "sky_lowobs") / visible
    persistence = fnum(row.get("mean_semantic_persistence_prefix"))
    confidence = fnum(row.get("mean_semantic_confidence_prefix"))
    role_prior = stable + 0.25 * vegetation + 0.10 * weak - 0.35 * dynamic - 0.45 * sky
    return role_prior + 0.20 * persistence + 0.10 * confidence


def frame_scores(seq: str, support_by_seq: dict[int, dict[str, str]], internal: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    qk_norm = normalize({frame: fnum(internal[frame]["mean_qk_cosine"]) for frame in ANCHOR_FRAMES})
    softmax_norm = normalize({frame: fnum(internal[frame]["mean_qk_softmax"]) for frame in ANCHOR_FRAMES})
    rank_quality = normalize({frame: fnum(internal[frame]["mean_qk_rank"]) for frame in ANCHOR_FRAMES}, invert=True)
    stability = normalize({frame: -fnum(internal[frame]["qk_cosine_std"]) for frame in ANCHOR_FRAMES})
    sem_norm = normalize({frame: semantic_score(support_by_seq.get(frame, {})) for frame in ANCHOR_FRAMES})
    out: dict[int, dict[str, Any]] = {}
    combined_scores: dict[int, float] = {}
    for frame in ANCHOR_FRAMES:
        internal_score = 0.55 * qk_norm[frame] + 0.30 * softmax_norm[frame] + 0.15 * rank_quality[frame]
        reliability_score = 0.50 * rank_quality[frame] + 0.30 * stability[frame] + 0.20 * softmax_norm[frame]
        combined = 0.55 * internal_score + 0.30 * sem_norm[frame] + 0.15 * reliability_score
        combined_scores[frame] = combined
        row = support_by_seq.get(frame, {})
        out[frame] = {
            "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_anchor_read_weight_row_v1",
            "seq": seq,
            "source_frame": frame,
            "read_rows": int(internal[frame]["read_rows"]),
            "mean_qk_cosine": internal[frame]["mean_qk_cosine"],
            "mean_qk_softmax": internal[frame]["mean_qk_softmax"],
            "mean_qk_rank": internal[frame]["mean_qk_rank"],
            "qk_cosine_std": internal[frame]["qk_cosine_std"],
            "internal_score": internal_score,
            "semantic_score_norm": sem_norm[frame],
            "reliability_score": reliability_score,
            "combined_score": combined,
            "visible_track_rows": row.get("visible_track_rows", ""),
            "top_roles": row.get("top_roles", ""),
            "best_track_role": row.get("best_track_role", ""),
        }
    score_median = median(combined_scores.values())
    for frame in ANCHOR_FRAMES:
        weight = clamp(math.exp(1.20 * (combined_scores[frame] - score_median)), 0.55, 1.55)
        out[frame]["candidate_weight"] = weight
        out[frame]["reverse_weight"] = clamp(1.0 / weight, 0.55, 1.55)
    return out


def method_yaml(
    *,
    method: str,
    policy: str,
    query_roles: list[str],
    weight_by_seq: dict[str, dict[int, float]],
) -> str:
    lines = [
        "model: lingbot_map",
        "env: loger",
        f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
        "_device: cuda",
        "_use_amp: true",
            f"_use_sdpa: {str(USE_SDPA).lower()}",
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
        f"_stage4_action_mode: {ACTION_MODE}",
        f"_stage4_action_label: {policy}",
        "_stage4_anchor_source_token_roles: [\"patch\"]",
        f"_stage4_anchor_source_query_roles: {json.dumps(query_roles)}",
        "_stage4_anchor_source_context_roles: [\"scale_reference_context\"]",
        "_stage4_anchor_source_weight_map_by_seq:",
    ]
    # The local wrapper does not expose a by-sequence field for this action, so
    # each method file is seq-specific at runtime.  The marker below is retained
    # only for audit readability and ignored by the wrapper.
    for seq in sorted(weight_by_seq):
        lines.append(f"  \"{seq}\": {json.dumps({str(k): v for k, v in sorted(weight_by_seq[seq].items())}, sort_keys=True)}")
    # Actual wrapper field.  run_worker is launched one seq at a time, and the
    # generated top-level config points to method variants with the seq-specific
    # map already materialized.
    lines.append(f"# method_template: {method}")
    return "\n".join(lines) + "\n"


def concrete_method_yaml(
    *,
    policy: str,
    query_roles: list[str],
    weight_map: dict[int, float],
) -> str:
    lines = [
        "model: lingbot_map",
        "env: loger",
        f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
        "_device: cuda",
        "_use_amp: true",
        f"_use_sdpa: {str(USE_SDPA).lower()}",
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
        f"_stage4_action_mode: {ACTION_MODE}",
        f"_stage4_action_label: {policy}",
        f"_stage4_anchor_source_weight_map: {json.dumps({str(k): v for k, v in sorted(weight_map.items())}, sort_keys=True)}",
        "_stage4_anchor_source_token_roles: [\"patch\"]",
        f"_stage4_anchor_source_query_roles: {json.dumps(query_roles)}",
        "_stage4_anchor_source_context_roles: [\"scale_reference_context\"]",
    ]
    if ACTION_MODE == "anchor_source_value_scaling":
        lines.append(f"_stage4_anchor_source_value_weight_normalization: {VALUE_WEIGHT_NORMALIZATION}")
    lines.append("")
    return "\n".join(lines)


def write_main_config(config_dir: Path, methods: list[str]) -> None:
    lines = [
        f"workspace: {WORKSPACE}",
        "",
        "evaluation:",
        "  traj:",
        "    enable: true",
        "    vis: true",
        "  auc:",
        "    enable: false",
        "  depth:",
        "    enable: false",
        "  points:",
        "    enable: false",
        "",
        "datasets:",
        f"  - {DATASET}",
        "",
        "methods:",
        *[f"  - {method}" for method in methods],
        "",
    ]
    (config_dir / CONFIG_BASENAME).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_dataset_config(config_dir: Path) -> None:
    ds_dir = config_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / f"{DATASET}.yaml").write_text(
        "\n".join(
            [
                "dataset: kitti",
                f"raw_data_root: {ROOT / 'data/kitti/dataset'}",
                "_target_size: [504, 280]",
                '_sequences: ["00", "01", "02", "05"]',
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    support = read_support()
    config_dir = STAGE / "configs"
    method_dir = config_dir / "methods"
    config_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_config(config_dir)

    frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    manifest_methods: dict[str, dict[str, Any]] = {}
    concrete_methods: list[str] = []
    weight_maps_by_method: dict[str, dict[str, dict[str, float]]] = {}

    scores_by_seq: dict[str, dict[int, dict[str, Any]]] = {}
    for seq in SEQS:
        scores = frame_scores(seq, support.get(seq, {}), read_anchor_read_stats(seq))
        scores_by_seq[seq] = scores
        frame_rows.extend(scores[frame] for frame in ANCHOR_FRAMES)

    for method_base, meta in METHODS.items():
        manifest_methods[method_base] = {
            key: meta[key]
            for key in ("policy", "role", "description", "query_roles", "weight_mode")
        }
        weight_maps_by_method[method_base] = {}
        for seq in SEQS:
            if meta["weight_mode"] == "reverse":
                weight_map = {frame: float(scores_by_seq[seq][frame]["reverse_weight"]) for frame in ANCHOR_FRAMES}
            elif meta["weight_mode"] == "shuffle":
                rotated = list(ANCHOR_FRAMES[3:]) + list(ANCHOR_FRAMES[:3])
                weight_map = {
                    frame: float(scores_by_seq[seq][source_frame]["candidate_weight"])
                    for frame, source_frame in zip(ANCHOR_FRAMES, rotated)
                }
            else:
                weight_map = {frame: float(scores_by_seq[seq][frame]["candidate_weight"]) for frame in ANCHOR_FRAMES}
            method = f"{method_base}_seq{seq}"
            concrete_methods.append(method)
            (method_dir / f"{method}.yaml").write_text(
                concrete_method_yaml(
                    policy=str(meta["policy"]),
                    query_roles=list(meta["query_roles"]),
                    weight_map=weight_map,
                ),
                encoding="utf-8",
            )
            weight_maps_by_method[method_base][seq] = {str(k): v for k, v in sorted(weight_map.items())}
            weights = list(weight_map.values())
            manifest_rows.append(
                {
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest_row_v1",
                    "seq": seq,
                    "method_base": method_base,
                    "method": method,
                    "branch": "LB-AR",
                    "policy": meta["policy"],
                    "role": meta["role"],
                    "source_context_roles": "scale_reference_context",
                    "token_roles": "patch",
                    "query_roles": ",".join(meta["query_roles"]),
                    "source_frames": ";".join(str(frame) for frame in ANCHOR_FRAMES),
                    "weight_min": min(weights),
                    "weight_max": max(weights),
                    "weight_median": median(weights),
                    "weight_mode": meta["weight_mode"],
                }
            )

    write_main_config(config_dir, concrete_methods)
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    weight_rows_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_weight_rows.csv"
    manifest_csv_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.csv"
    manifest_json_name = f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
    write_csv(out / weight_rows_name, frame_rows)
    write_csv(out / manifest_csv_name, manifest_rows)
    manifest = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "stage_tag": STAGE_TAG,
        "backend_label": BACKEND_LABEL,
        "use_sdpa": USE_SDPA,
        "action_mode": ACTION_MODE,
        "intervention_form": INTERVENTION_FORM,
        "value_weight_normalization": VALUE_WEIGHT_NORMALIZATION if ACTION_MODE == "anchor_source_value_scaling" else "",
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(SUPPORT.relative_to(ROOT)),
        "trace_dir": str(TRACE_DIR.relative_to(ROOT)),
        "branch": "LB-AR",
        "operation": "Anchor read",
        "fixed_anchor_source_frames": list(ANCHOR_FRAMES),
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "boundary": (
            f"{STAGE_TAG.upper()} uses 00/02 default-trace aggregate anchor-read cues over fixed default anchors "
            f"with intervention_form={INTERVENTION_FORM}. A pass can only create a fresh-validation candidate, "
            "not a global success claim."
        ),
        "manifest_csv": str((out / manifest_csv_name).relative_to(ROOT)),
        "weight_rows": str((out / weight_rows_name).relative_to(ROOT)),
    }
    (out / manifest_json_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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
