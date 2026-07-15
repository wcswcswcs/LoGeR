#!/usr/bin/env python3
"""Generate ACL2 v119-TF LB-AR-FIX anchor-read intervention configs."""

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


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_NAME = os.environ.get("ACL2_V119_LBAR_RUN_NAME", "stage2_lbar_anchor_read_form1_attention_sdpa").strip()
RUN_ROOT = RESULT_ROOT / RUN_NAME
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
RAW_ACTION = RUN_ROOT / "raw_action"
LOGS = RUN_ROOT / "logs"
SEM_SCORE_ROWS = Path(
    os.environ.get(
        "ACL2_V119_LBAR_SEM_FRAME_ROWS",
        str(
            RESULT_ROOT
            / "stage2_lbai_form2_anchor_token_routing_value_scaling_sdpa_repair/frame_semv3_anchor_scores_scale_frames.csv"
        ),
    )
)
SEM_V3 = RESULT_ROOT / "stage1_semv3_sidecar/semv3_prefix_rows.parquet"
SEM_V3_SUMMARY = RESULT_ROOT / "stage1_semv3_sidecar/semv3_summary.json"
INTERNAL_READ_ROWS = Path(
    os.environ.get(
        "ACL2_V119_LBAR_INTERNAL_READ_ROWS",
        str(RESULT_ROOT / "stage2_lbar_internal_anchor_read_trace/lbar_internal_anchor_read_rows.csv"),
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
ACTION_MODE = os.environ.get("ACL2_V119_LBAR_ACTION_MODE", "anchor_source_attention_weight").strip()
USE_SDPA = os.environ.get("ACL2_V119_LBAR_USE_SDPA", "1").strip().lower() in {"1", "true", "yes", "y"}
TOKEN_ROLES = [
    part.strip()
    for part in os.environ.get("ACL2_V119_LBAR_TOKEN_ROLES", "scale_special,patch").split(",")
    if part.strip()
]
QUERY_ROLES = [
    part.strip()
    for part in os.environ.get(
        "ACL2_V119_LBAR_QUERY_ROLES",
        "camera_special,register_special,scale_special",
    ).split(",")
    if part.strip()
]
SOURCE_CONTEXT_ROLES = [
    part.strip()
    for part in os.environ.get("ACL2_V119_LBAR_CONTEXT_ROLES", "scale_reference_context").split(",")
    if part.strip()
]
VALUE_WEIGHT_NORMALIZATION = os.environ.get("ACL2_V119_LBAR_VALUE_WEIGHT_NORMALIZATION", "arithmetic_mean_1").strip()
GPU_IDS = [
    part.strip()
    for part in os.environ.get("ACL2_V119_LBAR_GPU_IDS", "0,1,2,3,4,5").split(",")
    if part.strip()
]


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
    return "[" + ", ".join(json.dumps(item) if isinstance(item, str) else str(int(item)) for item in items) + "]"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def semantic_sidecar_version() -> str:
    if not SEM_V3_SUMMARY.exists():
        return "v119_semv3_visibility_role_shape_mad_reobs_motion_v1"
    payload = json.loads(SEM_V3_SUMMARY.read_text(encoding="utf-8"))
    return str(payload.get("semv3_formula_version") or "v119_semv3_visibility_role_shape_mad_reobs_motion_v1")


def default_frozen_indices(num_frames: int) -> list[int]:
    interval = math.ceil(num_frames / AUTO_KEYFRAME_THRESHOLD)
    stream = [idx for idx in range(num_frames) if idx >= SCALE_FRAMES]
    return [idx for pos, idx in enumerate(stream) if interval <= 1 or pos % interval == 0]


def read_semantic_rows() -> dict[str, dict[int, dict[str, Any]]]:
    if not SEM_SCORE_ROWS.exists():
        raise FileNotFoundError(SEM_SCORE_ROWS)
    out: dict[str, dict[int, dict[str, Any]]] = {}
    with SEM_SCORE_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seq = str(row["seq"]).zfill(2)
            frame = int(row["source_frame"])
            if frame < SCALE_FRAMES:
                out.setdefault(seq, {})[frame] = row
    return out


def read_internal_rows() -> dict[str, dict[int, dict[str, Any]]]:
    if not INTERNAL_READ_ROWS.exists():
        return {}
    out: dict[str, dict[int, dict[str, Any]]] = {}
    with INTERNAL_READ_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seq = str(row["seq"]).zfill(2)
            frame = int(row["source_frame"])
            out.setdefault(seq, {})[frame] = row
    return out


def normalized(values: dict[int, float]) -> dict[int, float]:
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {key: 0.5 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def weights_from_scores(scores: dict[int, float], gain: float = 1.20) -> dict[int, float]:
    normed = normalized(scores)
    center = median(normed.values())
    return {frame: clamp(math.exp(gain * (normed[frame] - center)), 0.70, 1.30) for frame in range(SCALE_FRAMES)}


def semantic_weights(rows_by_frame: dict[int, dict[str, Any]]) -> dict[int, float]:
    return weights_from_scores({frame: float(rows_by_frame[frame]["semantic_anchor_score"]) for frame in range(SCALE_FRAMES)})


def internal_read_weights(rows_by_frame: dict[int, dict[str, Any]]) -> dict[int, float]:
    return weights_from_scores({frame: float(rows_by_frame[frame]["internal_anchor_read_score"]) for frame in range(SCALE_FRAMES)})


def combined_weights(semantic: dict[int, float], internal: dict[int, float]) -> dict[int, float]:
    return {frame: clamp(math.sqrt(semantic[frame] * internal[frame]), 0.70, 1.30) for frame in range(SCALE_FRAMES)}


def shuffled_weights(weights: dict[int, float], seq: str) -> dict[int, float]:
    frames = list(range(SCALE_FRAMES))
    values = [weights[frame] for frame in frames]
    rng = random.Random(119420 + int(seq))
    rng.shuffle(values)
    return {frame: value for frame, value in zip(frames, values)}


def variants_for_seq(
    seq: str,
    semantic_rows: dict[int, dict[str, Any]],
    internal_rows: dict[int, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic = semantic_weights(semantic_rows)
    reverse = {frame: clamp(1.0 / semantic[frame], 0.70, 1.30) for frame in range(SCALE_FRAMES)}
    variants: list[dict[str, Any]] = [
        {
            "variant": "ar0_no_action",
            "policy": "AR0_NO_ACTION",
            "role": "default_control",
            "weight_mode": "none",
            "weights": {},
            "action_mode": "force_non_keyframe",
            "internal_score_version": "none_default_no_internal_read",
            "value_weight_normalization": "",
        }
    ]
    blocked: list[dict[str, Any]] = []
    if internal_rows and all(frame in internal_rows for frame in range(SCALE_FRAMES)):
        internal = internal_read_weights(internal_rows)
        variants.extend(
            [
                {
                    "variant": "ar1_internal_qk_entropy",
                    "policy": "AR1_INTERNAL_QK_ENTROPY_ONLY",
                    "role": "internal_only_control",
                    "weight_mode": "internal_qk_entropy",
                    "weights": internal,
                    "action_mode": ACTION_MODE,
                    "internal_score_version": "v119_lbar_noaction_qk_topk_entropy_global23_head0_special_v1",
                    "value_weight_normalization": VALUE_WEIGHT_NORMALIZATION,
                },
                {
                    "variant": "ar3_internal_semantic",
                    "policy": "AR3_INTERNAL_PLUS_SEMV3_ADDRESSABILITY",
                    "role": "candidate_internal_plus_semantic",
                    "weight_mode": "internal_semantic_geomean",
                    "weights": combined_weights(semantic, internal),
                    "action_mode": ACTION_MODE,
                    "internal_score_version": "v119_lbar_noaction_qk_topk_entropy_global23_head0_special_v1",
                    "value_weight_normalization": VALUE_WEIGHT_NORMALIZATION,
                },
            ]
        )
    else:
        blocked.append(
            {
                "variant": "AR1/AR3 internal QK entropy",
                "status": "blocked_pending_lbar_internal_read_trace",
                "reason": "requires current-code no-action SDPA QK top-k/entropy trace for source frames 0..7",
            }
        )
    variants.extend(
        [
            {
                "variant": "ar2_semantic_addressability_role",
                "policy": "AR2_SEMV3_ADDRESSABILITY_ROLE_ONLY",
                "role": "candidate_semantic_only",
                "weight_mode": "semantic_addressability_role",
                "weights": semantic,
                "action_mode": ACTION_MODE,
                "internal_score_version": "none_semantic_only_no_internal_read",
                "value_weight_normalization": VALUE_WEIGHT_NORMALIZATION,
            },
            {
                "variant": "ar4_same_magnitude_random",
                "policy": "AR4_SAME_MAGNITUDE_RANDOM_SEED00",
                "role": "matched_random_control",
                "weight_mode": "same_magnitude_semantic_shuffle_seed00",
                "weights": shuffled_weights(semantic, seq),
                "action_mode": ACTION_MODE,
                "internal_score_version": "none_same_magnitude_random_control",
                "value_weight_normalization": VALUE_WEIGHT_NORMALIZATION,
            },
            {
                "variant": "ar5_same_source_reverse_role",
                "policy": "AR5_SAME_SOURCE_REVERSE_ROLE",
                "role": "reverse_control",
                "weight_mode": "reverse_semantic_role",
                "weights": reverse,
                "action_mode": ACTION_MODE,
                "internal_score_version": "none_reverse_semantic_control",
                "value_weight_normalization": VALUE_WEIGHT_NORMALIZATION,
            },
        ]
    )
    if ACTION_MODE == "anchor_source_value_scaling":
        variants.append(
            {
                "variant": "ar6_legacy_geomean_norm",
                "policy": "AR6_LEGACY_GEOMETRIC_MEAN_NORMALIZATION_NEGATIVE_CONTROL",
                "role": "legacy_normalization_negative_control",
                "weight_mode": "semantic_legacy_geomean_norm",
                "weights": semantic,
                "action_mode": ACTION_MODE,
                "internal_score_version": "none_legacy_norm_control",
                "value_weight_normalization": "legacy_geometric_mean_1",
            }
        )
    else:
        blocked.append(
            {
                "variant": "AR6 legacy geometric-mean normalization",
                "status": "not_applicable_to_form1_attention_logit_bias",
                "reason": "normalization control belongs to Form2 source value scaling; Form1 has no value normalization operator",
            }
        )
    return variants, blocked


def weight_hash(weights: dict[int, float]) -> str:
    payload = json.dumps({str(k): round(float(v), 12) for k, v in sorted(weights.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        f"_stage4_action_label: v119_lbar_{variant['variant']}",
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
            lines.append(f"_stage4_anchor_source_value_weight_normalization: {variant['value_weight_normalization']}")
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
        raise ValueError(f"Unsupported ACL2_V119_LBAR_ACTION_MODE={ACTION_MODE!r}")
    if not GPU_IDS:
        raise ValueError("ACL2_V119_LBAR_GPU_IDS produced no usable GPU ids")
    semantic_by_seq = read_semantic_rows()
    internal_by_seq = read_internal_rows()
    sidecar_hash = sha256_file(SEM_V3)
    sidecar_version = semantic_sidecar_version()
    internal_hash = sha256_file(INTERNAL_READ_ROWS) if INTERNAL_READ_ROWS.exists() else ""
    LOGS.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"
    run_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for seq, num_frames in SEQ_LENGTHS.items():
        rows_by_frame = semantic_by_seq[seq]
        dataset = f"kitti_v119_lbar_seq{seq}"
        dataset_cfg = CONFIG_ROOT / "datasets" / f"{dataset}.yaml"
        write_text(dataset_cfg, dataset_yaml(seq))
        frozen = default_frozen_indices(num_frames)
        variants, blocked = variants_for_seq(seq, rows_by_frame, internal_by_seq.get(seq))
        blocked_rows.extend({"seq": seq, **row} for row in blocked)
        prepare_method = f"lingbot_map_v119_lbar_f1_{variants[0]['variant']}_{seq}"
        prepare_cfg = CONFIG_ROOT / f"kitti_lbar_prepare_seq{seq}.yaml"
        write_text(prepare_cfg, base_yaml(WORKSPACE, dataset, prepare_method))
        prepare_log = LOGS / f"prepare_seq{seq}.log"
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
                "command": f"{env_prefix} {CONDA} run -n {ENV_NAME} --no-capture-output python prepare.py --config {prepare_cfg} --force > {prepare_log} 2>&1",
            }
        )
        seq_track_hash = track_distribution_hash(rows_by_frame)
        for idx, variant in enumerate(variants):
            method = f"lingbot_map_v119_lbar_f1_{variant['variant']}_{seq}"
            method_cfg = CONFIG_ROOT / "methods" / f"{method}.yaml"
            base_cfg = CONFIG_ROOT / f"kitti_lbar_{variant['variant']}_seq{seq}.yaml"
            write_text(method_cfg, method_yaml(variant, frozen))
            write_text(base_cfg, base_yaml(WORKSPACE, dataset, method))
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            if not action_file.exists():
                action_file.write_text("", encoding="utf-8")
            gpu = GPU_IDS[idx % len(GPU_IDS)]
            stats = weight_stats(dict(variant["weights"]))
            form = (
                "LB-AR-FIX_Form1_selected_query_attention_logit_bias"
                if ACTION_MODE == "anchor_source_attention_weight"
                else "LB-AR-FIX_Form2_arithmetic_mean_neutral_source_value_scaling"
            )
            common_env = (
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                f"ACL2_V105_STAGE4_ACTION_LABEL=v119_lbar_{variant['variant']} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method} "
                f"ACL2_V112_A2_ACTION_GLOBAL_IDXS= "
                f"ACL2_V112_A2_ACTION_MAX_ROWS=120000 "
                f"ACL2_V119_SEMANTIC_SIDECAR_VERSION={sidecar_version} "
                f"ACL2_V119_SEMANTIC_SIDECAR_HASH={sidecar_hash} "
                f"ACL2_V119_TRACK_DISTRIBUTION_HASH={seq_track_hash} "
                f"ACL2_V119_INTERNAL_SCORE_VERSION={variant['internal_score_version']} "
                f"ACL2_V119_POLICY_VERSION=v119_lbar_anchor_read_fix_v1 "
                f"ACL2_V119_CARRIER_FORM={form}"
            )
            run_log = LOGS / f"run_{variant['variant']}_seq{seq}_gpu{gpu}.log"
            eval_log = LOGS / f"evaluate_{variant['variant']}_seq{seq}.log"
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
                    "command": f"{common_env} {CONDA} run -n {ENV_NAME} --no-capture-output python run_worker.py --config {base_cfg} --method {method} --dataset {dataset} --scene {seq} --force > {run_log} 2>&1",
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
                    "command": f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {CONDA} run -n {ENV_NAME} --no-capture-output python evaluate.py --config {base_cfg} --force > {eval_log} 2>&1",
                }
            )
            config_rows.append(
                {
                    "schema": "acl2_v119tf_lbar_config_row_v1",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "variant": variant["variant"],
                    "policy": variant["policy"],
                    "role": variant["role"],
                    "form": form,
                    "backend": "sdpa" if USE_SDPA else "flashinfer",
                    "action_mode": variant["action_mode"],
                    "weight_mode": variant["weight_mode"],
                    "value_weight_normalization": variant["value_weight_normalization"],
                    "token_roles": ",".join(TOKEN_ROLES),
                    "query_roles": ",".join(QUERY_ROLES) if QUERY_ROLES else "all",
                    "source_context_roles": ",".join(SOURCE_CONTEXT_ROLES),
                    "scale_frame_indices": ";".join(str(frame) for frame in range(SCALE_FRAMES)),
                    "frozen_keyframe_count": len(frozen),
                    "schedule_mode": "global_frozen",
                    "semantic_sidecar_version": sidecar_version,
                    "semantic_sidecar_hash": sidecar_hash,
                    "semantic_frame_rows": rel(SEM_SCORE_ROWS),
                    "internal_read_rows": rel(INTERNAL_READ_ROWS) if INTERNAL_READ_ROWS.exists() else "",
                    "internal_read_hash": internal_hash,
                    "track_distribution_hash": seq_track_hash,
                    "internal_score_version": variant["internal_score_version"],
                    "policy_version": "v119_lbar_anchor_read_fix_v1",
                    "carrier_form": form,
                    "config": rel(base_cfg),
                    "method_config": rel(method_cfg),
                    "action_file": rel(action_file),
                    **stats,
                }
            )

    write_csv(RUN_ROOT / "config_rows.csv", config_rows)
    write_csv(RUN_ROOT / "run_manifest.csv", run_rows)
    if blocked_rows:
        write_csv(RUN_ROOT / "blocked_variants.csv", blocked_rows)
    else:
        write_text(RUN_ROOT / "blocked_variants.csv", "seq,variant,status,reason\n")
    summary = {
        "schema": "acl2_v119tf_lbar_config_summary_v1",
        "run_root": rel(RUN_ROOT),
        "workspace": rel(WORKSPACE),
        "backend": "sdpa" if USE_SDPA else "flashinfer",
        "action_mode": ACTION_MODE,
        "form": "Form1 selected-query attention-logit bias" if ACTION_MODE == "anchor_source_attention_weight" else "Form2 source value scaling",
        "semantic_frame_rows": rel(SEM_SCORE_ROWS),
        "internal_read_rows": rel(INTERNAL_READ_ROWS) if INTERNAL_READ_ROWS.exists() else "",
        "internal_read_hash": internal_hash,
        "runnable_variant_count": len(config_rows),
        "blocked_variant_count": len(blocked_rows),
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "config_rows": rel(RUN_ROOT / "config_rows.csv"),
        "blocked_variants": rel(RUN_ROOT / "blocked_variants.csv"),
        "truthfulness_boundary": (
            "Config generation only. LB-AR-FIX uses fixed source frames 0..7, fixed query roles, fixed token roles, "
            "and SDPA anchor-source action rows. No runtime pass is claimed here."
        ),
    }
    write_text(RUN_ROOT / "config_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
