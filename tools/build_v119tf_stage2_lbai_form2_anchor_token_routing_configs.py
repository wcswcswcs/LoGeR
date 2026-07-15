#!/usr/bin/env python3
"""Generate ACL2 v119-TF LB-AI-FIX Form2 anchor-token routing configs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_NAME = os.environ.get("ACL2_V119_LBAI_FORM2_RUN_NAME", "stage2_lbai_form2_anchor_token_routing").strip()
RUN_ROOT = RESULT_ROOT / RUN_NAME
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
RAW_ACTION = RUN_ROOT / "raw_action"
SEM_V3 = RESULT_ROOT / "stage1_semv3_sidecar/semv3_prefix_rows.parquet"
SEM_V3_SUMMARY = RESULT_ROOT / "stage1_semv3_sidecar/semv3_summary.json"
INTERNAL_TRACE = Path(
    os.environ.get(
        "ACL2_V119_LBAI_INTERNAL_TRACE_ROWS",
        str(RESULT_ROOT / "stage2_lbai_internal_anchor_trace/lbai_internal_anchor_utility_rows.csv"),
    )
)
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"
RAW_DATA_ROOT = ROOT / "data/kitti/dataset"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
ENV_NAME = "loger"
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
SEQ_LENGTHS = {"00": 4541, "02": 4661}
AUTO_KEYFRAME_THRESHOLD = 320
SCALE_FRAMES = 8
RANDOM_SEEDS = list(range(21))
ACTION_MODE = os.environ.get("ACL2_V119_LBAI_FORM2_ACTION_MODE", "anchor_source_attention_weight").strip()
USE_SDPA = os.environ.get("ACL2_V119_LBAI_FORM2_USE_SDPA", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
TOKEN_ROLES = [
    part.strip()
    for part in os.environ.get("ACL2_V119_LBAI_FORM2_TOKEN_ROLES", "scale_special,patch").split(",")
    if part.strip()
]
QUERY_ROLES = [
    part.strip()
    for part in os.environ.get(
        "ACL2_V119_LBAI_FORM2_QUERY_ROLES",
        "camera_special,register_special,scale_special",
    ).split(",")
    if part.strip()
]
SOURCE_CONTEXT_ROLES = [
    part.strip()
    for part in os.environ.get("ACL2_V119_LBAI_FORM2_CONTEXT_ROLES", "scale_reference_context").split(",")
    if part.strip()
]
VALUE_WEIGHT_NORMALIZATION = os.environ.get(
    "ACL2_V119_LBAI_FORM2_VALUE_WEIGHT_NORMALIZATION", "arithmetic_mean_1"
).strip()
GPU_IDS = [
    part.strip()
    for part in os.environ.get("ACL2_V119_LBAI_FORM2_GPU_IDS", "0,1,2,3,4,5").split(",")
    if part.strip()
]
ROLE_WEIGHTS = {
    "stable_landmark": 1.0,
    "weak_context": 0.65,
    "vegetation_repetitive": 0.45,
    "boundary_lowpurity": 0.2,
    "dynamic": 0.15,
    "sky_lowobs": 0.0,
    "unknown_lowtrust": 0.0,
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def yaml_list(items: list[int] | list[str]) -> str:
    chunks = []
    for item in items:
        chunks.append(json.dumps(item) if isinstance(item, str) else str(int(item)))
    return "[" + ", ".join(chunks) + "]"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_frozen_indices(num_frames: int) -> list[int]:
    interval = math.ceil(num_frames / AUTO_KEYFRAME_THRESHOLD)
    stream = [idx for idx in range(num_frames) if idx >= SCALE_FRAMES]
    return [idx for pos, idx in enumerate(stream) if interval <= 1 or pos % interval == 0]


def semantic_sidecar_version() -> str:
    if not SEM_V3_SUMMARY.exists():
        return "v119_semv3_visibility_role_shape_mad_reobs_motion_v1"
    payload = json.loads(SEM_V3_SUMMARY.read_text(encoding="utf-8"))
    return str(payload.get("semv3_formula_version") or "v119_semv3_visibility_role_shape_mad_reobs_motion_v1")


def semantic_frame_scores() -> dict[str, dict[int, dict[str, Any]]]:
    if not SEM_V3.exists():
        raise FileNotFoundError(SEM_V3)
    df = pd.read_parquet(
        SEM_V3,
        columns=[
            "seq",
            "frame_id",
            "track_id",
            "dominant_role_prefix",
            "dominant_label_prefix",
            "semv3_visibility_prefix",
            "semv3_role_prior_prefix",
            "semv3_shape_score_prefix",
            "semv3_reobs_score_prefix",
            "semv3_motion_residual_prefix",
            "semv3_identity_key",
        ],
    )
    df = df[df["frame_id"].astype(int) < SCALE_FRAMES].copy()
    rows_by_seq: dict[str, dict[int, dict[str, Any]]] = {}
    for seq, seq_df in df.groupby(df["seq"].astype(str), sort=True):
        frame_rows: dict[int, dict[str, Any]] = {}
        for frame_id, frame_df in seq_df.groupby(seq_df["frame_id"].astype(int), sort=True):
            best_score = -1.0
            best_row: dict[str, Any] | None = None
            score_sum = 0.0
            for row in frame_df.itertuples(index=False):
                role = str(row.dominant_role_prefix)
                role_weight = ROLE_WEIGHTS.get(role, 0.0)
                score = (
                    role_weight
                    * float(row.semv3_role_prior_prefix)
                    * float(row.semv3_shape_score_prefix)
                    * (0.5 + 0.5 * float(row.semv3_reobs_score_prefix))
                    * min(1.0, 4.0 * float(row.semv3_visibility_prefix))
                )
                score_sum += score
                if score > best_score:
                    best_score = score
                    best_row = {
                        "best_track_id": int(row.track_id),
                        "best_identity_key": str(row.semv3_identity_key),
                        "best_role": role,
                        "best_label": str(row.dominant_label_prefix),
                        "best_motion_residual": float(row.semv3_motion_residual_prefix),
                    }
            frame_rows[int(frame_id)] = {
                "schema": "acl2_v119tf_lbai_form2_anchor_frame_score_row_v1",
                "seq": str(seq).zfill(2),
                "source_frame": int(frame_id),
                "visible_track_count": int(len(frame_df)),
                "semantic_anchor_score": float(max(best_score, 0.0)),
                "semantic_anchor_score_mean": float(score_sum / max(1, len(frame_df))),
                **(best_row or {}),
            }
        rows_by_seq[str(seq).zfill(2)] = frame_rows
    return rows_by_seq


def internal_frame_scores() -> dict[str, dict[int, dict[str, Any]]]:
    if not INTERNAL_TRACE.exists():
        return {}
    rows_by_seq: dict[str, dict[int, dict[str, Any]]] = {}
    with INTERNAL_TRACE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seq = str(row.get("seq", "")).zfill(2)
            frame = int(row["frame_id"])
            rows_by_seq.setdefault(seq, {})[frame] = row
    return rows_by_seq


def normalized(values: dict[int, float]) -> dict[int, float]:
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {key: 0.5 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def semantic_weights(rows_by_frame: dict[int, dict[str, Any]]) -> dict[int, float]:
    scores = normalized({frame: float(rows_by_frame[frame]["semantic_anchor_score"]) for frame in range(SCALE_FRAMES)})
    center = median(scores.values())
    return {frame: clamp(math.exp(1.20 * (scores[frame] - center)), 0.70, 1.30) for frame in range(SCALE_FRAMES)}


def internal_weights(rows_by_frame: dict[int, dict[str, Any]]) -> dict[int, float]:
    scores = normalized(
        {frame: float(rows_by_frame[frame]["internal_anchor_utility_score"]) for frame in range(SCALE_FRAMES)}
    )
    center = median(scores.values())
    return {frame: clamp(math.exp(1.20 * (scores[frame] - center)), 0.70, 1.30) for frame in range(SCALE_FRAMES)}


def combined_weights(semantic: dict[int, float], internal: dict[int, float]) -> dict[int, float]:
    return {
        frame: clamp(math.sqrt(float(semantic[frame]) * float(internal[frame])), 0.70, 1.30)
        for frame in range(SCALE_FRAMES)
    }


def shuffled_weights(weights: dict[int, float], seed: int) -> dict[int, float]:
    frames = list(range(SCALE_FRAMES))
    values = [weights[frame] for frame in frames]
    rng = random.Random(119200 + seed)
    rng.shuffle(values)
    return {frame: value for frame, value in zip(frames, values)}


def same_internal_bucket_shuffle_weights(
    semantic: dict[int, float],
    internal_rows_by_frame: dict[int, dict[str, Any]],
    seq: str,
) -> dict[int, float]:
    buckets: dict[str, list[int]] = {}
    for frame in range(SCALE_FRAMES):
        bucket = str(internal_rows_by_frame[frame].get("internal_anchor_bucket_q2", "q1_of_2"))
        buckets.setdefault(bucket, []).append(frame)
    rng = random.Random(119207 + int(seq) * 100)
    out = dict(semantic)
    for frames in buckets.values():
        if len(frames) < 2:
            continue
        values = [out[frame] for frame in frames]
        rng.shuffle(values)
        for frame, value in zip(frames, values):
            out[frame] = value
    return out


def variants_for_seq(
    seq: str,
    rows_by_frame: dict[int, dict[str, Any]],
    internal_rows_by_frame: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    semantic = semantic_weights(rows_by_frame)
    reverse = {frame: clamp(1.0 / semantic[frame], 0.70, 1.30) for frame in range(SCALE_FRAMES)}
    variants: list[dict[str, Any]] = [
        {
            "variant": "ai0_default_no_action",
            "policy": "AI0_DEFAULT_FIRSTN_ZERO_DELAY_NO_ROUTING",
            "role": "default_control",
            "weight_mode": "none",
            "weights": {},
            "action_mode": "force_non_keyframe",
            "internal_score_version": "none_default_no_internal_utility",
        },
    ]
    if internal_rows_by_frame and all(frame in internal_rows_by_frame for frame in range(SCALE_FRAMES)):
        internal = internal_weights(internal_rows_by_frame)
        variants.extend(
            [
                {
                    "variant": "ai1_internal_anchor_utility",
                    "policy": "AI1_INTERNAL_VIEW_DIVERSITY_GEOM_QUALITY_ANCHOR_TOKEN_ROUTING",
                    "role": "internal_only_control",
                    "weight_mode": "internal_anchor_utility",
                    "weights": internal,
                    "action_mode": ACTION_MODE,
                    "internal_score_version": "v119_lbai_noaction_pose_confidence_viewnovelty_redundancy_v1",
                },
                {
                    "variant": "ai3_internal_semantic_anchor_role",
                    "policy": "AI3_INTERNAL_PLUS_SEMV3_ANCHOR_TOKEN_ROUTING",
                    "role": "candidate_internal_plus_semantic",
                    "weight_mode": "internal_semantic_geomean",
                    "weights": combined_weights(semantic, internal),
                    "action_mode": ACTION_MODE,
                    "internal_score_version": "v119_lbai_noaction_pose_confidence_viewnovelty_redundancy_v1",
                },
                {
                    "variant": "ai7_same_internal_bucket_shuffle",
                    "policy": "AI7_SAME_INTERNAL_BUCKET_SEMV3_FRAME_SHUFFLE",
                    "role": "same_internal_bucket_shuffle_control",
                    "weight_mode": "same_internal_bucket_semantic_shuffle",
                    "weights": same_internal_bucket_shuffle_weights(semantic, internal_rows_by_frame, seq),
                    "action_mode": ACTION_MODE,
                    "internal_score_version": "v119_lbai_noaction_pose_confidence_viewnovelty_redundancy_v1",
                },
            ]
        )
    variants.extend(
        [
        {
            "variant": "ai2_semantic_anchor_role",
            "policy": "AI2_SEMV3_SEMANTIC_ONLY_ANCHOR_TOKEN_ROUTING",
            "role": "candidate_semantic_only",
            "weight_mode": "semantic",
            "weights": semantic,
            "action_mode": ACTION_MODE,
            "internal_score_version": "none_semantic_only_no_internal_utility",
        },
        {
            "variant": "ai5_reverse_semantic_anchor_role",
            "policy": "AI5_REVERSE_SEMV3_ANCHOR_TOKEN_ROUTING",
            "role": "reverse_control",
            "weight_mode": "reverse_semantic",
            "weights": reverse,
            "action_mode": ACTION_MODE,
            "internal_score_version": "none_reverse_semantic_control",
        },
        {
            "variant": "ai8_uniform_hook_noop",
            "policy": "AI8_SAME_SOURCE_SAME_HOOK_UNIFORM_NOOP",
            "role": "same_source_hook_noop_control",
            "weight_mode": "uniform_noop",
            "weights": {frame: 1.0 for frame in range(SCALE_FRAMES)},
            "action_mode": ACTION_MODE,
            "internal_score_version": "none_uniform_hook_control",
        },
        ]
    )
    for seed in RANDOM_SEEDS:
        variants.append(
            {
                "variant": f"ai6_random_seed{seed:02d}",
                "policy": f"AI6_SAME_SOURCE_WEIGHT_RANDOM_SEED{seed:02d}",
                "role": "matched_random_control",
                "weight_mode": f"semantic_weight_shuffle_seed{seed:02d}",
                "weights": shuffled_weights(semantic, seed + int(seq) * 100),
                "action_mode": ACTION_MODE,
                "internal_score_version": "none_same_weight_random_control",
            }
        )
    return variants


def weight_hash(weights: dict[int, float]) -> str:
    payload = json.dumps({str(k): round(float(v), 12) for k, v in sorted(weights.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def track_distribution_hash(rows_by_frame: dict[int, dict[str, Any]]) -> str:
    payload = [
        {
            "frame": frame,
            "identity": rows_by_frame[frame].get("best_identity_key", ""),
            "role": rows_by_frame[frame].get("best_role", ""),
            "label": rows_by_frame[frame].get("best_label", ""),
        }
        for frame in sorted(rows_by_frame)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def internal_trace_hash() -> str:
    return sha256_file(INTERNAL_TRACE) if INTERNAL_TRACE.exists() else ""


def weight_stats(weights: dict[int, float]) -> dict[str, Any]:
    if not weights:
        return {
            "weight_hash": "",
            "source_frames": "",
            "weight_min": "",
            "weight_max": "",
            "weight_mean": "",
            "weight_l1": "",
            "weight_l2": "",
            "changed_source_frame_count": 0,
        }
    values = [float(weights[frame]) for frame in sorted(weights)]
    return {
        "weight_hash": weight_hash(weights),
        "source_frames": ";".join(str(frame) for frame in sorted(weights)),
        "weight_min": min(values),
        "weight_max": max(values),
        "weight_mean": sum(values) / len(values),
        "weight_l1": sum(abs(value) for value in values),
        "weight_l2": math.sqrt(sum(value * value for value in values)),
        "changed_source_frame_count": sum(1 for value in values if abs(value - 1.0) > 1e-12),
    }


def method_yaml(variant: dict[str, Any], frozen: list[int]) -> str:
    weights = {str(k): float(v) for k, v in sorted(dict(variant["weights"]).items())}
    lines = [
        "model: lingbot_map",
        f"env: {ENV_NAME}",
        f"_checkpoint: {CHECKPOINT}",
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
        "_keyframe_schedule_mode: global_frozen",
        f"_frozen_keyframe_indices: {yaml_list(frozen)}",
        f"_stage4_action_mode: {variant['action_mode']}",
        f"_stage4_action_label: v119_lbai_form2_{variant['variant']}",
    ]
    if weights:
        lines.extend(
            [
                f"_stage4_anchor_source_weight_map: {json.dumps(weights, sort_keys=True)}",
                f"_stage4_anchor_source_token_roles: {json.dumps(TOKEN_ROLES)}",
                f"_stage4_anchor_source_query_roles: {json.dumps(QUERY_ROLES)}",
                f"_stage4_anchor_source_context_roles: {json.dumps(SOURCE_CONTEXT_ROLES)}",
            ]
        )
        if variant["action_mode"] == "anchor_source_value_scaling":
            lines.append(f"_stage4_anchor_source_value_weight_normalization: {VALUE_WEIGHT_NORMALIZATION}")
    lines.append("")
    return "\n".join(lines)


def base_yaml(workspace: Path, dataset: str, method: str) -> str:
    return f"""workspace: {workspace}

evaluation:
  traj:
    enable: true
    vis: true
  auc:
    enable: false
  depth:
    enable: false
  points:
    enable: false

datasets:
  - {dataset}

methods:
  - {method}
"""


def dataset_yaml(seq: str) -> str:
    return f"""dataset: kitti
raw_data_root: {RAW_DATA_ROOT}
_target_size: [504, 280]
_sequences: {yaml_list([seq])}
"""


def main() -> None:
    if ACTION_MODE not in {"anchor_source_attention_weight", "anchor_source_value_scaling"}:
        raise ValueError(f"Unsupported ACL2_V119_LBAI_FORM2_ACTION_MODE={ACTION_MODE!r}")
    if not GPU_IDS:
        raise ValueError("ACL2_V119_LBAI_FORM2_GPU_IDS produced no usable GPU ids")
    scores_by_seq = semantic_frame_scores()
    internal_by_seq = internal_frame_scores()
    internal_available_by_seq = {
        seq: bool(seq in internal_by_seq and all(frame in internal_by_seq[seq] for frame in range(SCALE_FRAMES)))
        for seq in SEQ_LENGTHS
    }
    internal_available = all(internal_available_by_seq.values())
    internal_hash = internal_trace_hash()
    sidecar_hash = sha256_file(SEM_V3)
    sidecar_version = semantic_sidecar_version()
    logs = RUN_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"
    run_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    frame_score_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    if not internal_available:
        missing_seqs = ",".join(seq for seq, ok in sorted(internal_available_by_seq.items()) if not ok)
        blocked_rows.extend(
            [
                {
                    "variant": "AI1 internal-only view-diversity + geom-quality",
                    "status": "blocked_pending_internal_trace_repair",
                    "reason": (
                        "current-code zero-delay anchor internal utility trace is not materialized for "
                        f"seqs={missing_seqs}; not replaced by frame-index heuristic"
                    ),
                },
                {
                    "variant": "AI3 internal + semantic",
                    "status": "blocked_pending_internal_trace_repair",
                    "reason": "requires AI1 internal utility trace joined with SEM-V3; semantic-only Form2 runs are still valid controls but not AI3",
                },
                {
                    "variant": "AI7 same-internal-bucket track shuffle",
                    "status": "blocked_pending_internal_trace_repair",
                    "reason": "same-internal-bucket control needs current-code internal buckets; random controls are generated separately as AI6",
                },
            ]
        )

    for seq, num_frames in SEQ_LENGTHS.items():
        rows_by_frame = scores_by_seq[seq]
        frame_score_rows.extend(rows_by_frame[frame] for frame in sorted(rows_by_frame))
        dataset = f"kitti_v119_lbai_form2_seq{seq}"
        dataset_cfg = CONFIG_ROOT / "datasets" / f"{dataset}.yaml"
        write_text(dataset_cfg, dataset_yaml(seq))
        frozen = default_frozen_indices(num_frames)
        variants = variants_for_seq(seq, rows_by_frame, internal_by_seq.get(seq))
        prepare_method = f"lingbot_map_v119_lbai_f2_{variants[0]['variant']}_{seq}"
        prepare_cfg = CONFIG_ROOT / f"kitti_lbai_form2_prepare_seq{seq}.yaml"
        write_text(prepare_cfg, base_yaml(WORKSPACE, dataset, prepare_method))
        prepare_log = logs / f"prepare_seq{seq}.log"
        run_rows.append(
            {
                "phase": "prepare",
                "seq": seq,
                "variant": "prepare",
                "gpu": "",
                "cwd": str(BENCH),
                "config": str(prepare_cfg),
                "dataset": dataset,
                "method": "",
                "action_file": "",
                "log": str(prepare_log),
                "command": (
                    f"{env_prefix} {CONDA} run -n {ENV_NAME} --no-capture-output "
                    f"python prepare.py --config {prepare_cfg} --force > {prepare_log} 2>&1"
                ),
            }
        )
        seq_track_hash = track_distribution_hash(rows_by_frame)
        for idx, variant in enumerate(variants):
            method = f"lingbot_map_v119_lbai_f2_{variant['variant']}_{seq}"
            method_cfg = CONFIG_ROOT / "methods" / f"{method}.yaml"
            base_cfg = CONFIG_ROOT / f"kitti_lbai_form2_{variant['variant']}_seq{seq}.yaml"
            write_text(method_cfg, method_yaml(variant, frozen))
            write_text(base_cfg, base_yaml(WORKSPACE, dataset, method))
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            if not action_file.exists():
                action_file.write_text("", encoding="utf-8")
            gpu = GPU_IDS[idx % len(GPU_IDS)]
            stats = weight_stats(dict(variant["weights"]))
            carrier_form = f"LB-AI-FIX_Form2_zero_delay_anchor_source_{variant['action_mode']}"
            common_env = (
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                f"ACL2_V105_STAGE4_ACTION_LABEL=v119_lbai_form2_{variant['variant']} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method} "
                f"ACL2_V112_A2_ACTION_GLOBAL_IDXS= "
                f"ACL2_V112_A2_ACTION_MAX_ROWS=120000 "
                f"ACL2_V119_SEMANTIC_SIDECAR_VERSION={sidecar_version} "
                f"ACL2_V119_SEMANTIC_SIDECAR_HASH={sidecar_hash} "
                f"ACL2_V119_TRACK_DISTRIBUTION_HASH={seq_track_hash} "
                f"ACL2_V119_INTERNAL_SCORE_VERSION={variant['internal_score_version']} "
                f"ACL2_V119_POLICY_VERSION=v119_lbai_form2_anchor_token_routing_v1 "
                f"ACL2_V119_CARRIER_FORM={carrier_form}"
            )
            run_log = logs / f"run_{variant['variant']}_seq{seq}_gpu{gpu}.log"
            eval_log = logs / f"evaluate_{variant['variant']}_seq{seq}.log"
            run_rows.append(
                {
                    "phase": "run_worker",
                    "seq": seq,
                    "variant": variant["variant"],
                    "gpu": gpu,
                    "cwd": str(BENCH),
                    "config": str(base_cfg),
                    "dataset": dataset,
                    "method": method,
                    "action_file": str(action_file),
                    "log": str(run_log),
                    "command": (
                        f"{common_env} {CONDA} run -n {ENV_NAME} --no-capture-output "
                        f"python run_worker.py --config {base_cfg} --method {method} "
                        f"--dataset {dataset} --scene {seq} --force > {run_log} 2>&1"
                    ),
                }
            )
            run_rows.append(
                {
                    "phase": "evaluate",
                    "seq": seq,
                    "variant": variant["variant"],
                    "gpu": gpu,
                    "cwd": str(BENCH),
                    "config": str(base_cfg),
                    "dataset": dataset,
                    "method": method,
                    "action_file": "",
                    "log": str(eval_log),
                    "command": (
                        f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {CONDA} run -n {ENV_NAME} --no-capture-output "
                        f"python evaluate.py --config {base_cfg} --force > {eval_log} 2>&1"
                    ),
                }
            )
            config_rows.append(
                {
                    "schema": "acl2_v119tf_lbai_form2_config_row_v1",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "variant": variant["variant"],
                    "policy": variant["policy"],
                    "role": variant["role"],
                    "form": "Form2 zero-delay default first-n + anchor source token-role routing",
                    "backend": "sdpa" if USE_SDPA else "flashinfer",
                    "action_mode": variant["action_mode"],
                    "weight_mode": variant["weight_mode"],
                    "token_roles": ",".join(TOKEN_ROLES),
                    "query_roles": ",".join(QUERY_ROLES) if QUERY_ROLES else "all",
                    "source_context_roles": ",".join(SOURCE_CONTEXT_ROLES),
                    "scale_frame_indices": ";".join(str(frame) for frame in range(SCALE_FRAMES)),
                    "frozen_keyframe_count": len(frozen),
                    "schedule_mode": "global_frozen",
                    "semantic_sidecar_version": sidecar_version,
                    "semantic_sidecar_hash": sidecar_hash,
                    "internal_trace": rel(INTERNAL_TRACE) if INTERNAL_TRACE.exists() else "",
                    "internal_trace_hash": internal_hash,
                    "track_distribution_hash": seq_track_hash,
                    "internal_score_version": variant["internal_score_version"],
                    "policy_version": "v119_lbai_form2_anchor_token_routing_v1",
                    "carrier_form": carrier_form,
                    "config": rel(base_cfg),
                    "method_config": rel(method_cfg),
                    "action_file": rel(action_file),
                    **stats,
                }
            )

    write_csv(RUN_ROOT / "frame_semv3_anchor_scores_scale_frames.csv", frame_score_rows)
    write_csv(RUN_ROOT / "config_rows.csv", config_rows)
    write_csv(RUN_ROOT / "run_manifest.csv", run_rows)
    if blocked_rows:
        write_csv(RUN_ROOT / "blocked_variants.csv", blocked_rows)
    else:
        write_text(RUN_ROOT / "blocked_variants.csv", "variant,status,reason\n")
    summary = {
        "schema": "acl2_v119tf_lbai_form2_config_summary_v1",
        "run_root": rel(RUN_ROOT),
        "workspace": rel(WORKSPACE),
        "semv3_source": rel(SEM_V3),
        "semantic_sidecar_version": sidecar_version,
        "semantic_sidecar_hash": sidecar_hash,
        "internal_trace": rel(INTERNAL_TRACE) if INTERNAL_TRACE.exists() else "",
        "internal_trace_hash": internal_hash,
        "internal_trace_available_by_seq": internal_available_by_seq,
        "sequences": list(SEQ_LENGTHS),
        "backend": "sdpa" if USE_SDPA else "flashinfer",
        "action_mode": ACTION_MODE,
        "token_roles": TOKEN_ROLES,
        "query_roles": QUERY_ROLES,
        "source_context_roles": SOURCE_CONTEXT_ROLES,
        "runnable_variant_count": len(config_rows),
        "blocked_variant_count": len(blocked_rows),
        "config_rows": rel(RUN_ROOT / "config_rows.csv"),
        "frame_score_rows": rel(RUN_ROOT / "frame_semv3_anchor_scores_scale_frames.csv"),
        "blocked_variants": rel(RUN_ROOT / "blocked_variants.csv"),
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "truthfulness_boundary": (
            "Config generation only. Form2 semantic/reverse/random/default controls are generated with zero-delay "
            f"default first-n anchors and anchor source token-role routing on backend={'sdpa' if USE_SDPA else 'flashinfer'}. "
            + (
                "AI1/AI3/AI7 are generated from current-code no-action internal trace; no runtime pass is claimed here."
                if internal_available
                else "AI1/AI3/AI7 remain blocked until a current-code internal anchor utility trace is materialized; no runtime pass is claimed here."
            )
        ),
    }
    write_text(RUN_ROOT / "config_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
