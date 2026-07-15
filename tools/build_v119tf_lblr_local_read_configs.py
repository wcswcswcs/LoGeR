#!/usr/bin/env python3
"""Build ACL2 v119 LB-LR local-read configs.

This reuses the existing LingBot source attention/value action surfaces, but
targets ``local_window_context`` instead of the anchor ``scale_reference`` path.
The default source set is deliberately small and auditable: local frames 8..63.
Set ``ACL2_V119_LBLR_SOURCE_FRAMES`` for a runtime-carrier-aligned repair
universe, e.g. the keyframe pages actually reachable by the streaming cache.
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
_RESULT_ROOT_ENV = os.environ.get("ACL2_V119_LBLR_RESULT_ROOT", "").strip()
if _RESULT_ROOT_ENV:
    _result_root_path = Path(_RESULT_ROOT_ENV).expanduser()
    RESULT_ROOT = _result_root_path if _result_root_path.is_absolute() else ROOT / _result_root_path
else:
    RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
REFERENCE_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE_TAG = os.environ.get("ACL2_V119_LBLR_STAGE_TAG", "v119_lblr_logit").strip().lower() or "v119_lblr_logit"
BACKEND_LABEL = os.environ.get("ACL2_V119_LBLR_BACKEND_LABEL", "sdpa").strip().lower() or "sdpa"
ACTION_MODE = (
    os.environ.get("ACL2_V119_LBLR_ACTION_MODE", "anchor_source_attention_weight").strip()
    or "anchor_source_attention_weight"
)
VALUE_WEIGHT_NORMALIZATION = (
    os.environ.get("ACL2_V119_LBLR_VALUE_WEIGHT_NORMALIZATION", "arithmetic_mean_1").strip()
    or "arithmetic_mean_1"
)
INTERVENTION_FORM = (
    "selected_source_value_routing"
    if ACTION_MODE == "anchor_source_value_scaling"
    else "selected_special_query_logit_bias"
)
DEFAULT_STAGE_SLUG = "stage2_lblr_local_read_value" if ACTION_MODE == "anchor_source_value_scaling" else "stage2_lblr_local_read_logit"
STAGE_SLUG = os.environ.get("ACL2_V119_LBLR_STAGE_SLUG", DEFAULT_STAGE_SLUG).strip() or DEFAULT_STAGE_SLUG
USE_SDPA = os.environ.get("ACL2_V119_LBLR_USE_SDPA", "true").strip().lower() in {"1", "true", "yes", "y"}
CONFIG_BASENAME = f"kitti_lingbot_{BACKEND_LABEL}_{STAGE_TAG}_local_read_full_reuse_v105gt.yaml"
STAGE = RESULT_ROOT / STAGE_SLUG
SUPPORT = REFERENCE_ROOT / "stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_frame_semantic_support_rows.csv"
TRACE_DIR = REFERENCE_ROOT / "stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"


def parse_source_frames(raw: str) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple(range(8, 64))
    frames: list[int] = []
    for part in text.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
            lo = int(lo_raw.strip())
            hi = int(hi_raw.strip())
            step = 1 if hi >= lo else -1
            frames.extend(range(lo, hi + step, step))
        else:
            frames.append(int(item))
    unique = tuple(sorted(set(frames)))
    if not unique:
        raise ValueError("ACL2_V119_LBLR_SOURCE_FRAMES resolved to an empty frame set")
    return unique


SOURCE_FRAMES = parse_source_frames(os.environ.get("ACL2_V119_LBLR_SOURCE_FRAMES", ""))
SEQS = tuple(
    part.strip().zfill(2)
    for part in os.environ.get("ACL2_V119_LBLR_SEQS", "00,02").replace(";", ",").split(",")
    if part.strip()
) or ("00", "02")
SPECIAL_QUERY_ROLES = ["camera_special", "register_special", "scale_special"]
METHOD_PREFIX = f"lingbot_map_stream_{BACKEND_LABEL}_{STAGE_TAG}"


def methods() -> dict[str, dict[str, Any]]:
    common = {
        f"{METHOD_PREFIX}_lr1_internal_only": {
            "policy": "LR1_INTERNAL_ONLY_QK_ENTROPY_SINK",
            "role": "internal_only_candidate",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "internal",
            "description": "local read score from internal QK/rank/entropy proxies only",
        },
        f"{METHOD_PREFIX}_lr2_semantic_only": {
            "policy": "LR2_SEMANTIC_ONLY_LOCAL_ROLE",
            "role": "semantic_only_candidate",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "semantic",
            "description": "local read score from semantic role/support only",
        },
        f"{METHOD_PREFIX}_lr3_internal_semantic": {
            "policy": "LR3_INTERNAL_SEMANTIC_ROLE_COMPATIBILITY",
            "role": "candidate",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "combined",
            "description": "internal plus semantic role-compatible local read routing",
        },
        f"{METHOD_PREFIX}_lr4_dynamic_aligned": {
            "policy": "LR4_DYNAMIC_ALIGNED_LOCAL_ALLOW_LONG_BLOCK",
            "role": "dynamic_aligned_candidate",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "dynamic_aligned",
            "description": "allow dynamic local evidence only when internal score is aligned",
        },
        f"{METHOD_PREFIX}_lr5_same_qk_bucket_shuffle": {
            "policy": "LR5_SAME_QK_BUCKET_FRAME_SHUFFLE_CONTROL",
            "role": "same_qk_bucket_shuffle_control",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "same_qk_bucket_shuffle",
            "description": "shuffle candidate weights within internal-score buckets",
        },
        f"{METHOD_PREFIX}_lr7_reverse_role": {
            "policy": "LR7_REVERSE_ROLE_CONTROL",
            "role": "reverse_role_control",
            "query_roles": SPECIAL_QUERY_ROLES,
            "weight_mode": "reverse",
            "description": "reciprocal matched local read weights",
        },
    }
    if ACTION_MODE == "anchor_source_value_scaling":
        for meta in common.values():
            meta["query_roles"] = []
        common[f"{METHOD_PREFIX}_lr6_uniform_value_noop"] = {
            "policy": "LR6_UNIFORM_VALUE_NOOP_CONTROL_QUERY_SELECTION_UNAVAILABLE",
            "role": "uniform_value_noop_control",
            "query_roles": [],
            "weight_mode": "uniform",
            "description": "value-routing no-op control; value action has no query-role selector",
        }
    else:
        common[f"{METHOD_PREFIX}_lr6_patch_query_control"] = {
            "policy": "LR6_PATCH_QUERY_CONTROL",
            "role": "patch_query_control",
            "query_roles": ["patch"],
            "weight_mode": "combined",
            "description": "same local source weights as LR3, applied to patch queries",
        }
    return common


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
            out.setdefault(str(row["seq"]).zfill(2), {})[int(float(row["frame_id"]))] = row
    return out


def read_local_stats(seq: str) -> dict[int, dict[str, Any]]:
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
            if row.get("memory_family") != "local":
                continue
            if row.get("token_type") != "image_patch":
                continue
            frame_raw = row.get("source_frame_id")
            if frame_raw is None:
                continue
            frame = int(frame_raw)
            if frame not in SOURCE_FRAMES:
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
    missing = [frame for frame in SOURCE_FRAMES if frame not in stats]
    if missing:
        raise RuntimeError(f"missing local read rows for seq {seq}: {missing[:20]}")
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
    out = {}
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


def dynamic_ratio(row: dict[str, str]) -> float:
    visible = max(1, int(fnum(row.get("visible_track_rows"))))
    return parse_role_count(row.get("top_roles", ""), "dynamic") / visible


def weights_from_scores(scores: dict[int, float], strength: float = 1.20) -> dict[int, float]:
    center = median(scores.values())
    return {frame: clamp(math.exp(strength * (score - center)), 0.55, 1.55) for frame, score in scores.items()}


def frame_scores(seq: str, support_by_frame: dict[int, dict[str, str]], internal: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    qk_norm = normalize({frame: fnum(internal[frame]["mean_qk_cosine"]) for frame in SOURCE_FRAMES})
    softmax_norm = normalize({frame: fnum(internal[frame]["mean_qk_softmax"]) for frame in SOURCE_FRAMES})
    rank_quality = normalize({frame: fnum(internal[frame]["mean_qk_rank"]) for frame in SOURCE_FRAMES}, invert=True)
    entropy_quality = normalize({frame: -fnum(internal[frame]["mean_read_entropy"]) for frame in SOURCE_FRAMES})
    stability = normalize({frame: -fnum(internal[frame]["qk_cosine_std"]) for frame in SOURCE_FRAMES})
    sem_norm = normalize({frame: semantic_score(support_by_frame.get(frame, {})) for frame in SOURCE_FRAMES})
    dyn = {frame: dynamic_ratio(support_by_frame.get(frame, {})) for frame in SOURCE_FRAMES}
    internal_scores = {
        frame: 0.45 * qk_norm[frame] + 0.30 * softmax_norm[frame] + 0.15 * rank_quality[frame] + 0.10 * entropy_quality[frame]
        for frame in SOURCE_FRAMES
    }
    internal_center = median(internal_scores.values())
    dynamic_scores = {}
    combined_scores = {}
    for frame in SOURCE_FRAMES:
        aligned_dynamic = dyn[frame] if internal_scores[frame] >= internal_center else -dyn[frame]
        dynamic_scores[frame] = 0.70 * internal_scores[frame] + 0.20 * sem_norm[frame] + 0.10 * aligned_dynamic
        combined_scores[frame] = 0.55 * internal_scores[frame] + 0.35 * sem_norm[frame] + 0.10 * dynamic_scores[frame]

    weight_sets = {
        "internal": weights_from_scores(internal_scores),
        "semantic": weights_from_scores(sem_norm),
        "combined": weights_from_scores(combined_scores),
        "dynamic_aligned": weights_from_scores(dynamic_scores),
    }
    out: dict[int, dict[str, Any]] = {}
    for frame in SOURCE_FRAMES:
        row = support_by_frame.get(frame, {})
        out[frame] = {
            "schema": f"acl2_v119tf_{STAGE_TAG}_lblr_local_read_weight_row_v1",
            "seq": seq,
            "source_frame": frame,
            "read_rows": int(internal[frame]["read_rows"]),
            "mean_qk_cosine": internal[frame]["mean_qk_cosine"],
            "mean_qk_softmax": internal[frame]["mean_qk_softmax"],
            "mean_qk_rank": internal[frame]["mean_qk_rank"],
            "mean_read_entropy": internal[frame]["mean_read_entropy"],
            "qk_cosine_std": internal[frame]["qk_cosine_std"],
            "internal_score": internal_scores[frame],
            "semantic_score_norm": sem_norm[frame],
            "dynamic_ratio": dyn[frame],
            "dynamic_aligned_score": dynamic_scores[frame],
            "combined_score": combined_scores[frame],
            "internal_weight": weight_sets["internal"][frame],
            "semantic_weight": weight_sets["semantic"][frame],
            "candidate_weight": weight_sets["combined"][frame],
            "dynamic_aligned_weight": weight_sets["dynamic_aligned"][frame],
            "reverse_weight": clamp(1.0 / weight_sets["combined"][frame], 0.55, 1.55),
            "uniform_weight": 1.0,
            "visible_track_rows": row.get("visible_track_rows", ""),
            "top_roles": row.get("top_roles", ""),
            "best_track_role": row.get("best_track_role", ""),
        }
    return out


def same_qk_bucket_shuffle(scores: dict[int, dict[str, Any]]) -> dict[int, float]:
    frames = list(SOURCE_FRAMES)
    sorted_frames = sorted(frames, key=lambda frame: float(scores[frame]["internal_score"]))
    buckets = [sorted_frames[i::4] for i in range(4)]
    shuffled = {frame: float(scores[frame]["candidate_weight"]) for frame in frames}
    for bucket in buckets:
        if len(bucket) < 2:
            continue
        rotated = bucket[1:] + bucket[:1]
        for frame, source in zip(bucket, rotated):
            shuffled[frame] = float(scores[source]["candidate_weight"])
    return shuffled


def weight_map_for(mode: str, scores: dict[int, dict[str, Any]]) -> dict[int, float]:
    if mode == "internal":
        return {frame: float(scores[frame]["internal_weight"]) for frame in SOURCE_FRAMES}
    if mode == "semantic":
        return {frame: float(scores[frame]["semantic_weight"]) for frame in SOURCE_FRAMES}
    if mode == "dynamic_aligned":
        return {frame: float(scores[frame]["dynamic_aligned_weight"]) for frame in SOURCE_FRAMES}
    if mode == "same_qk_bucket_shuffle":
        return same_qk_bucket_shuffle(scores)
    if mode == "reverse":
        return {frame: float(scores[frame]["reverse_weight"]) for frame in SOURCE_FRAMES}
    if mode == "uniform":
        return {frame: 1.0 for frame in SOURCE_FRAMES}
    return {frame: float(scores[frame]["candidate_weight"]) for frame in SOURCE_FRAMES}


def concrete_method_yaml(*, policy: str, query_roles: list[str], weight_map: dict[int, float]) -> str:
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
        "_stage4_anchor_source_context_roles: [\"local_window_context\"]",
    ]
    if ACTION_MODE == "anchor_source_value_scaling":
        lines.append(f"_stage4_anchor_source_value_weight_normalization: {VALUE_WEIGHT_NORMALIZATION}")
    lines.append("")
    return "\n".join(lines)


def write_main_config(config_dir: Path, concrete_methods: list[str]) -> None:
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
        *[f"  - {method}" for method in concrete_methods],
        "",
    ]
    (config_dir / CONFIG_BASENAME).write_text("\n".join(lines), encoding="utf-8")


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
    method_defs = methods()
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
        scores = frame_scores(seq, support.get(seq, {}), read_local_stats(seq))
        scores_by_seq[seq] = scores
        frame_rows.extend(scores[frame] for frame in SOURCE_FRAMES)

    for method_base, meta in method_defs.items():
        manifest_methods[method_base] = {
            key: meta[key]
            for key in ("policy", "role", "description", "query_roles", "weight_mode")
        }
        weight_maps_by_method[method_base] = {}
        for seq in SEQS:
            weight_map = weight_map_for(str(meta["weight_mode"]), scores_by_seq[seq])
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
                    "schema": f"acl2_v119tf_{STAGE_TAG}_lblr_local_read_manifest_row_v1",
                    "seq": seq,
                    "method_base": method_base,
                    "method": method,
                    "branch": "LB-LR",
                    "policy": meta["policy"],
                    "role": meta["role"],
                    "source_context_roles": "local_window_context",
                    "token_roles": "patch",
                    "query_roles": ",".join(meta["query_roles"]),
                    "source_frames": ";".join(str(frame) for frame in SOURCE_FRAMES),
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
        "schema": f"acl2_v119tf_{STAGE_TAG}_lblr_local_read_manifest_v1",
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
        "branch": "LB-LR",
        "operation": "Local read",
        "fixed_anchor_source_frames": list(SOURCE_FRAMES),
        "source_context_roles": ["local_window_context"],
        "token_roles": ["patch"],
        "methods": manifest_methods,
        "concrete_methods": concrete_methods,
        "weight_maps_by_method": weight_maps_by_method,
        "boundary": (
            f"{STAGE_TAG.upper()} uses fixed local source frames {','.join(str(frame) for frame in SOURCE_FRAMES)} "
            "from v118 R14 local read traces "
            f"with intervention_form={INTERVENTION_FORM}. Runtime action rows must prove local_window_context "
            "coverage; this is not a global success claim."
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
                "concrete_method_count": len(concrete_methods),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
