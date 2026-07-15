#!/usr/bin/env python3
"""Generate v119 LB-TA trajectory-admission configs and run manifest."""

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

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_NAME = os.environ.get("ACL2_V119_LBTA_RUN_NAME", "stage2_lbta_trajectory_admission").strip()
RUN_ROOT = RESULT_ROOT / RUN_NAME
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
RAW_ACTION = RUN_ROOT / "raw_action"
LOGS = RUN_ROOT / "logs"
SUMMARY = RUN_ROOT / "summary"
SEM_V3 = RESULT_ROOT / "stage1_semv3_sidecar/semv3_prefix_rows.parquet"
SEM_V3_SUMMARY = RESULT_ROOT / "stage1_semv3_sidecar/semv3_summary.json"
BASELINE_ROOT = RESULT_ROOT / "stage0_lingbot_fresh_baselines/workspace/kitti_v119_stage0_00_01_02_05"
B1_CONFIG_ROOT = RESULT_ROOT / "stage0_lingbot_b1_fresh_baselines/configs/methods"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"
RAW_DATA_ROOT = ROOT / "data/kitti/dataset"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
ENV_NAME = "loger"
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
SEQ_LENGTHS = {"00": 4541, "02": 4661}
SCALE_FRAMES = 8
AUTO_KEYFRAME_THRESHOLD = 320
ACTION_BUDGET_FRACTION = float(os.environ.get("ACL2_V119_LBTA_ACTION_BUDGET_FRACTION", "0.10"))
TEMPORAL_BUCKETS = 4
INTERNAL_BUCKETS = 4
MIN_FRAME_GAP = int(os.environ.get("ACL2_V119_LBTA_MIN_FRAME_GAP", "60"))
RANDOM_SEED = int(os.environ.get("ACL2_V119_LBTA_RANDOM_SEED", "119740"))
USE_SDPA = os.environ.get("ACL2_V119_LBTA_USE_SDPA", "0").strip().lower() in {"1", "true", "yes", "y"}
GPU_IDS = [
    part.strip()
    for part in os.environ.get("ACL2_V119_LBTA_GPU_IDS", "0,1,2,3,4,5").split(",")
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

SURFACES = {
    "hard": {
        "family": "hard_noappend",
        "action_mode": "force_non_keyframe",
        "context_token_mask": None,
        "description": "force selected default keyframes to non-keyframes",
    },
    "soft_all": {
        "family": "soft_compact_context_all_special",
        "action_mode": "context_only_special",
        "context_token_mask": None,
        "description": "convert selected default keyframes to all-special context-only compact entries",
    },
    "soft_anchor": {
        "family": "soft_compact_context_anchor_only",
        "action_mode": "anchor_special_only",
        "context_token_mask": None,
        "description": "convert selected default keyframes to scale/anchor-only compact entries",
    },
    "soft_local": {
        "family": "soft_compact_context_local_register_only",
        "action_mode": "trajectory_context_token_mask",
        "context_token_mask": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        "description": "convert selected default keyframes to camera/register-only compact entries",
    },
}

VARIANTS = {
    "ta0_default": {
        "policy": "TA0_DEFAULT_ADMISSION",
        "role": "default_control",
        "selector": "default_no_action",
    },
    "ta1_internal_low_utility_drop": {
        "policy": "TA1_INTERNAL_ONLY_LOW_UTILITY_DROP",
        "role": "internal_only_control",
        "selector": "internal_low",
    },
    "ta2_semantic_low_support_drop": {
        "policy": "TA2_SEMANTIC_PERSISTENT_LANDMARK_LOW_SUPPORT_DROP",
        "role": "semantic_only_candidate",
        "selector": "semantic_low",
    },
    "ta3_internal_semantic_low_combined_drop": {
        "policy": "TA3_INTERNAL_PLUS_SEMANTIC_LOW_COMBINED_DROP",
        "role": "candidate_internal_plus_semantic",
        "selector": "combined_low",
    },
    "ta4_b1_legacy_highrisk_reference": {
        "policy": "TA4_B1_LEGACY_HIGH_RISK_NOAPPEND_REFERENCE",
        "role": "legacy_reference",
        "selector": "b1_legacy",
    },
    "ta5_reverse_high_support_drop": {
        "policy": "TA5_REVERSE_SEMANTIC_HIGH_SUPPORT_DROP",
        "role": "reverse_control",
        "selector": "semantic_high",
    },
    "ta6_temporal_random": {
        "policy": "TA6_SAME_COUNT_SAME_TEMPORAL_BUCKET_RANDOM",
        "role": "matched_temporal_random_control",
        "selector": "temporal_random",
    },
    "ta7_same_internal_bucket_shuffle": {
        "policy": "TA7_SAME_INTERNAL_SCORE_BUCKET_SHUFFLE",
        "role": "same_internal_bucket_shuffle_control",
        "selector": "internal_bucket_shuffle",
    },
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if abs(hi - lo) < 1e-12:
        return {key: 0.5 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def quantile_bucket(values: dict[int, float], frame: int, bucket_count: int) -> int:
    ordered = sorted(values.values())
    value = values[frame]
    rank = sum(1 for item in ordered if item <= value) - 1
    denom = max(1, len(ordered) - 1)
    return min(bucket_count - 1, int(math.floor((rank / denom) * bucket_count)))


def default_frozen_indices(num_frames: int) -> list[int]:
    interval = math.ceil(num_frames / AUTO_KEYFRAME_THRESHOLD)
    return [
        frame
        for stream_pos, frame in enumerate(range(SCALE_FRAMES, num_frames))
        if interval <= 1 or stream_pos % interval == 0
    ]


def greedy_gap_select(ranked: list[int], budget: int, min_gap: int) -> list[int]:
    selected: list[int] = []
    for frame in ranked:
        if all(abs(frame - old) >= min_gap for old in selected):
            selected.append(frame)
            if len(selected) == budget:
                return sorted(selected)
    for frame in ranked:
        if frame not in selected:
            selected.append(frame)
            if len(selected) == budget:
                return sorted(selected)
    return sorted(selected)


def temporal_bucket_counts(eligible: list[int], selected: list[int]) -> dict[int, int]:
    pos = {frame: idx for idx, frame in enumerate(eligible)}
    counts = {idx: 0 for idx in range(TEMPORAL_BUCKETS)}
    denom = max(1, len(eligible))
    for frame in selected:
        counts[min(TEMPORAL_BUCKETS - 1, int(pos[frame] * TEMPORAL_BUCKETS / denom))] += 1
    return counts


def temporal_matched_random(eligible: list[int], selected: list[int], seq: str) -> list[int]:
    selected_set = set(selected)
    counts = temporal_bucket_counts(eligible, selected)
    rng = random.Random(RANDOM_SEED + int(seq) * 17)
    out: list[int] = []
    denom = max(1, len(eligible))
    for bucket, count in counts.items():
        bucket_frames = [
            frame
            for idx, frame in enumerate(eligible)
            if min(TEMPORAL_BUCKETS - 1, int(idx * TEMPORAL_BUCKETS / denom)) == bucket
        ]
        pool = [frame for frame in bucket_frames if frame not in selected_set]
        rng.shuffle(pool)
        picked = greedy_gap_select(pool, count, MIN_FRAME_GAP)
        if len(picked) < count:
            fallback = [frame for frame in bucket_frames if frame not in set(picked)]
            rng.shuffle(fallback)
            picked.extend(frame for frame in fallback if frame not in picked)
        out.extend(picked[:count])
    if len(out) < len(selected):
        fallback = [frame for frame in eligible if frame not in set(out) and frame not in selected_set]
        rng.shuffle(fallback)
        out.extend(fallback[: len(selected) - len(out)])
    return sorted(out[: len(selected)])


def internal_bucket_shuffle(
    eligible: list[int],
    selected: list[int],
    internal_scores: dict[int, float],
    seq: str,
) -> list[int]:
    selected_set = set(selected)
    counts = {idx: 0 for idx in range(INTERNAL_BUCKETS)}
    buckets = {frame: quantile_bucket(internal_scores, frame, INTERNAL_BUCKETS) for frame in eligible}
    for frame in selected:
        counts[buckets[frame]] += 1
    rng = random.Random(RANDOM_SEED + int(seq) * 31)
    out: list[int] = []
    for bucket, count in counts.items():
        pool = [frame for frame in eligible if buckets[frame] == bucket and frame not in selected_set]
        rng.shuffle(pool)
        picked = greedy_gap_select(pool, count, MIN_FRAME_GAP)
        if len(picked) < count:
            fallback = [frame for frame in eligible if buckets[frame] == bucket and frame not in set(picked)]
            rng.shuffle(fallback)
            picked.extend(frame for frame in fallback if frame not in picked)
        out.extend(picked[:count])
    if len(out) < len(selected):
        fallback = [frame for frame in eligible if frame not in set(out) and frame not in selected_set]
        rng.shuffle(fallback)
        out.extend(fallback[: len(selected) - len(out)])
    return sorted(out[: len(selected)])


def read_traj(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            frame = int(parts[0])
            vals = [float(value) for value in parts[1:13]]
            poses[frame] = np.array([vals[3], vals[7], vals[11]], dtype=np.float64)
    return poses


def confidence_mean(path: Path) -> float:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return float(arr.mean())


def baseline_method_root(seq: str) -> Path:
    return BASELINE_ROOT / seq / "lingbot_map_stream_default_flashinfer_v119_stage0"


def internal_rows_for_seq(seq: str, eligible: list[int]) -> tuple[list[dict[str, Any]], dict[int, float]]:
    root = baseline_method_root(seq)
    poses = read_traj(root / "traj.txt")
    missing = [frame for frame in eligible if frame not in poses]
    if missing:
        raise RuntimeError(f"missing baseline pose frames for seq{seq}: {missing[:10]}")
    step = {}
    baseline = {}
    min_prior = {}
    conf = {}
    origin = poses[eligible[0]]
    prior_frames: list[int] = []
    for frame in eligible:
        conf_path = root / "confidence" / f"{frame:06d}.jpg"
        if not conf_path.exists():
            raise FileNotFoundError(conf_path)
        conf[frame] = confidence_mean(conf_path)
        prev = poses.get(frame - 1, poses[frame])
        step[frame] = float(np.linalg.norm(poses[frame] - prev))
        baseline[frame] = float(np.linalg.norm(poses[frame] - origin))
        if not prior_frames:
            min_prior[frame] = 0.0
        else:
            min_prior[frame] = float(min(np.linalg.norm(poses[frame] - poses[old]) for old in prior_frames))
        prior_frames.append(frame)
    step_n = normalize(step)
    baseline_n = normalize(baseline)
    min_prior_n = normalize(min_prior)
    conf_n = normalize(conf)
    scores: dict[int, float] = {}
    rows = []
    for frame in eligible:
        redundancy = 1.0 - min_prior_n[frame]
        score = (
            0.35 * step_n[frame]
            + 0.25 * baseline_n[frame]
            + 0.30 * conf_n[frame]
            + 0.10 * min_prior_n[frame]
            - 0.20 * redundancy
        )
        scores[frame] = max(0.0, min(1.0, float(score)))
    for frame in eligible:
        rows.append(
            {
                "schema": "acl2_v119tf_lbta_internal_admission_row_v1",
                "seq": seq,
                "frame_id": frame,
                "eligible_default_base_keyframe": True,
                "source_no_action_root": rel(root),
                "pose_step_delta": step[frame],
                "pose_baseline_delta": baseline[frame],
                "pose_min_prior_delta": min_prior[frame],
                "confidence_mean": conf[frame],
                "pose_step_delta_norm": step_n[frame],
                "pose_baseline_delta_norm": baseline_n[frame],
                "pose_min_prior_delta_norm": min_prior_n[frame],
                "confidence_mean_norm": conf_n[frame],
                "redundancy_penalty": 1.0 - min_prior_n[frame],
                "internal_admission_score": scores[frame],
                "internal_admission_bucket_q4": f"q{quantile_bucket(scores, frame, 4) + 1}_of_4",
                "internal_score_version": "v119_lbta_noaction_pose_confidence_viewnovelty_redundancy_all_default_keyframes_v1",
                "runtime_cue_boundary": (
                    "current-code no-action output only: predicted C2W translation and LingBot confidence JPG; "
                    "no GT, no external depth, no SLAM, no post-hoc ATE feedback"
                ),
            }
        )
    return rows, scores


def semantic_sidecar_version() -> str:
    if not SEM_V3_SUMMARY.exists():
        return "v119_semv3_visibility_role_shape_mad_reobs_motion_v1"
    payload = json.loads(SEM_V3_SUMMARY.read_text(encoding="utf-8"))
    return str(payload.get("semv3_formula_version") or "v119_semv3_visibility_role_shape_mad_reobs_motion_v1")


def semantic_rows_by_seq() -> dict[str, pd.DataFrame]:
    if not SEM_V3.exists():
        raise FileNotFoundError(SEM_V3)
    df = pd.read_parquet(
        SEM_V3,
        columns=[
            "seq",
            "track_id",
            "frame_id",
            "dominant_role_prefix",
            "dominant_label_prefix",
            "semv3_visibility_prefix",
            "semv3_role_prior_prefix",
            "semv3_shape_score_prefix",
            "semv3_reobs_score_prefix",
            "semv3_motion_residual_prefix",
            "current_area_ratio",
            "current_mask_quality",
            "semv3_identity_key",
        ],
    )
    return {
        str(seq).zfill(2): seq_df.copy()
        for seq, seq_df in df.groupby(df["seq"].astype(str), sort=True)
    }


def semantic_support_for_seq(
    seq: str,
    eligible: list[int],
    seq_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    rows: list[dict[str, Any]] = []
    raw_support: dict[int, dict[str, float]] = {}
    best_identity: dict[int, str] = {}
    grouped = {
        int(frame): frame_df
        for frame, frame_df in seq_df.groupby(seq_df["frame_id"].astype(int), sort=False)
    }
    for frame in eligible:
        frame_df = grouped.get(frame)
        if frame_df is None or frame_df.empty:
            raw_support[frame] = {
                "max_semantic_persistence_prefix": 0.0,
                "mean_semantic_persistence_prefix": 0.0,
                "mean_current_mask_quality": 0.0,
                "sum_current_area_ratio": 0.0,
                "visible_track_rows": 0.0,
                "unique_track_count": 0.0,
                "motion_stability": 0.0,
            }
            best_identity[frame] = ""
            continue
        scores = []
        best_score = -1.0
        best_row: Any = None
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
            scores.append(max(0.0, float(score)))
            if score > best_score:
                best_score = float(score)
                best_row = row
        motion_values = [max(0.0, 1.0 - float(value)) for value in frame_df["semv3_motion_residual_prefix"]]
        raw_support[frame] = {
            "max_semantic_persistence_prefix": max(scores) if scores else 0.0,
            "mean_semantic_persistence_prefix": float(sum(scores) / max(1, len(scores))),
            "mean_current_mask_quality": float(frame_df["current_mask_quality"].fillna(0.0).astype(float).mean()),
            "sum_current_area_ratio": float(frame_df["current_area_ratio"].fillna(0.0).astype(float).sum()),
            "visible_track_rows": float(len(frame_df)),
            "unique_track_count": float(frame_df["track_id"].nunique()),
            "motion_stability": float(sum(motion_values) / max(1, len(motion_values))),
        }
        best_identity[frame] = str(getattr(best_row, "semv3_identity_key", "")) if best_row is not None else ""
    fields = [
        "max_semantic_persistence_prefix",
        "mean_semantic_persistence_prefix",
        "mean_current_mask_quality",
        "sum_current_area_ratio",
        "visible_track_rows",
        "unique_track_count",
        "motion_stability",
    ]
    norm = {field: normalize({frame: raw_support[frame][field] for frame in eligible}) for field in fields}
    support_scores: dict[int, float] = {}
    for frame in eligible:
        support_scores[frame] = (
            0.30 * norm["max_semantic_persistence_prefix"][frame]
            + 0.20 * norm["mean_semantic_persistence_prefix"][frame]
            + 0.15 * norm["mean_current_mask_quality"][frame]
            + 0.10 * norm["sum_current_area_ratio"][frame]
            + 0.10 * norm["visible_track_rows"][frame]
            + 0.05 * norm["unique_track_count"][frame]
            + 0.10 * norm["motion_stability"][frame]
        )
    for frame in eligible:
        rows.append(
            {
                "schema": "acl2_v119tf_lbta_semantic_support_row_v1",
                "seq": seq,
                "frame_id": frame,
                "eligible_default_base_keyframe": True,
                "best_identity_key": best_identity[frame],
                "semantic_admission_score": support_scores[frame],
                "semantic_admission_bucket_q4": f"q{quantile_bucket(support_scores, frame, 4) + 1}_of_4",
                **raw_support[frame],
                "semv3_formula_version": semantic_sidecar_version(),
                "runtime_cue_boundary": (
                    "causal SEM-V3 prefix rows only; no GT geometry, no external depth, no SLAM, no ATE feedback"
                ),
            }
        )
    return rows, support_scores


def read_b1_indices(seq: str) -> list[int]:
    path = B1_CONFIG_ROOT / f"lingbot_map_v119_stage0_B1_semantic_only_flashinfer_{seq}.yaml"
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("_force_non_keyframe_indices:"):
            return [int(value) for value in json.loads(line.split(":", 1)[1].strip())]
    return []


def select_variants(
    seq: str,
    eligible: list[int],
    semantic: dict[int, float],
    internal: dict[int, float],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    budget = max(1, int(len(eligible) * ACTION_BUDGET_FRACTION))
    combined = {frame: math.sqrt(max(0.0, semantic[frame]) * max(0.0, internal[frame])) for frame in eligible}
    ranked_internal_low = sorted(eligible, key=lambda frame: (internal[frame], frame))
    ranked_semantic_low = sorted(eligible, key=lambda frame: (semantic[frame], frame))
    ranked_semantic_high = sorted(eligible, key=lambda frame: (-semantic[frame], frame))
    ranked_combined_low = sorted(eligible, key=lambda frame: (combined[frame], frame))
    candidate = greedy_gap_select(ranked_combined_low, budget, MIN_FRAME_GAP)
    selections = {
        "ta0_default": [],
        "ta1_internal_low_utility_drop": greedy_gap_select(ranked_internal_low, budget, MIN_FRAME_GAP),
        "ta2_semantic_low_support_drop": greedy_gap_select(ranked_semantic_low, budget, MIN_FRAME_GAP),
        "ta3_internal_semantic_low_combined_drop": candidate,
        "ta4_b1_legacy_highrisk_reference": sorted(frame for frame in read_b1_indices(seq) if frame in set(eligible)),
        "ta5_reverse_high_support_drop": greedy_gap_select(ranked_semantic_high, budget, MIN_FRAME_GAP),
        "ta6_temporal_random": temporal_matched_random(eligible, candidate, seq),
        "ta7_same_internal_bucket_shuffle": internal_bucket_shuffle(eligible, candidate, internal, seq),
    }
    return selections, {
        "budget": budget,
        "eligible_count": len(eligible),
        "candidate_temporal_bucket_counts": json.dumps(temporal_bucket_counts(eligible, candidate), sort_keys=True),
        "combined_score_median_selected": median([combined[frame] for frame in candidate]) if candidate else "",
    }


def method_yaml(surface: str, variant: str, seq: str, selected: list[int], frozen: list[int]) -> str:
    surface_meta = SURFACES[surface]
    variant_meta = VARIANTS[variant]
    action_mode = surface_meta["action_mode"] if selected else "force_non_keyframe"
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
        f"_stage4_action_mode: {action_mode}",
        f"_stage4_action_label: v119_lbta_{surface}_{variant}",
        "_stage4_force_non_keyframe_indices_by_seq:",
        f'  "{seq}": {json.dumps(selected, separators=(",", ":"))}',
    ]
    mask = surface_meta.get("context_token_mask")
    if selected and mask is not None:
        lines.append(f"_stage4_context_token_mask: {json.dumps(mask, separators=(',', ':'))}")
    lines.append("")
    return "\n".join(lines)


def base_yaml(dataset: str, method: str) -> str:
    return f"""workspace: {WORKSPACE}

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
    if not GPU_IDS:
        raise RuntimeError("ACL2_V119_LBTA_GPU_IDS produced no usable GPU ids")
    sem_by_seq = semantic_rows_by_seq()
    sidecar_hash = sha256_file(SEM_V3)
    sidecar_version = semantic_sidecar_version()
    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    all_internal_rows: list[dict[str, Any]] = []
    all_semantic_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for seq, num_frames in SEQ_LENGTHS.items():
        frozen = default_frozen_indices(num_frames)
        eligible = list(frozen)
        internal_rows, internal_scores = internal_rows_for_seq(seq, eligible)
        semantic_rows, semantic_scores = semantic_support_for_seq(seq, eligible, sem_by_seq.get(seq, pd.DataFrame()))
        all_internal_rows.extend(internal_rows)
        all_semantic_rows.extend(semantic_rows)
        selections, seq_meta = select_variants(seq, eligible, semantic_scores, internal_scores)
        for surface in SURFACES:
            dataset = f"kitti_v119_lbta_{surface}_seq{seq}"
            dataset_cfg = CONFIG_ROOT / "datasets" / f"{dataset}.yaml"
            write_text(dataset_cfg, dataset_yaml(seq))
            for idx, (variant, selected) in enumerate(selections.items()):
                method = f"lingbot_map_v119_lbta_{surface}_{variant}_{seq}"
                method_cfg = CONFIG_ROOT / "methods" / f"{method}.yaml"
                base_cfg = CONFIG_ROOT / f"kitti_lbta_{surface}_{variant}_seq{seq}.yaml"
                write_text(method_cfg, method_yaml(surface, variant, seq, selected, frozen))
                write_text(base_cfg, base_yaml(dataset, method))
                action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
                action_file.write_text("", encoding="utf-8")
                gpu = GPU_IDS[idx % len(GPU_IDS)]
                common_env = (
                    f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} "
                    f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                    f"ACL2_V105_STAGE4_ACTION_LABEL=v119_lbta_{surface}_{variant} "
                    f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                    f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                    f"ACL2_V105_GCA_TRACE_METHOD={method} "
                    f"ACL2_V108_STAGE4_TRAJECTORY_ONLY_OUTPUT=1 "
                    f"ACL2_V112_A2_ACTION_GLOBAL_IDXS= "
                    f"ACL2_V112_A2_ACTION_MAX_ROWS=140000 "
                    f"OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 "
                    f"ACL2_V119_SEMANTIC_SIDECAR_VERSION={sidecar_version} "
                    f"ACL2_V119_SEMANTIC_SIDECAR_HASH={sidecar_hash} "
                    f"ACL2_V119_POLICY_VERSION=v119_lbta_trajectory_admission_v1"
                )
                run_log = LOGS / f"run_{surface}_{variant}_seq{seq}_gpu{gpu}.log"
                eval_log = LOGS / f"evaluate_{surface}_{variant}_seq{seq}.log"
                run_rows.append(
                    {
                        "phase": "run_worker",
                        "seq": seq,
                        "variant": f"{surface}_{variant}",
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
                        "variant": f"{surface}_{variant}",
                        "gpu": gpu,
                        "cwd": str(BENCH),
                        "config": str(base_cfg),
                        "dataset": dataset,
                        "method": method,
                        "action_file": "",
                        "log": str(eval_log),
                        "command": (
                            f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {CONDA} run -n {ENV_NAME} "
                            f"--no-capture-output python evaluate.py --config {base_cfg} --force > {eval_log} 2>&1"
                        ),
                    }
                )
                action_scores = {
                    "semantic_selected_median": median([semantic_scores[frame] for frame in selected]) if selected else "",
                    "internal_selected_median": median([internal_scores[frame] for frame in selected]) if selected else "",
                }
                row = {
                    "schema": "acl2_v119tf_lbta_trajectory_admission_config_row_v1",
                    "seq": seq,
                    "surface": surface,
                    "surface_family": SURFACES[surface]["family"],
                    "surface_action_mode": SURFACES[surface]["action_mode"],
                    "variant": variant,
                    "variant_key": f"{surface}_{variant}",
                    "policy": VARIANTS[variant]["policy"],
                    "role": VARIANTS[variant]["role"],
                    "selector": VARIANTS[variant]["selector"],
                    "dataset": dataset,
                    "method": method,
                    "action_budget_fraction": ACTION_BUDGET_FRACTION,
                    "min_frame_gap": MIN_FRAME_GAP,
                    "selected_count": len(selected),
                    "selected_frame_ids": ";".join(str(frame) for frame in selected),
                    "eligible_default_base_keyframes": len(eligible),
                    "frozen_keyframe_count": len(frozen),
                    "backend": "sdpa" if USE_SDPA else "flashinfer",
                    "config": rel(base_cfg),
                    "method_config": rel(method_cfg),
                    "action_file": rel(action_file),
                    "semantic_sidecar": rel(SEM_V3),
                    "semantic_sidecar_hash": sidecar_hash,
                    "semantic_sidecar_version": sidecar_version,
                    **seq_meta,
                    **action_scores,
                }
                manifest_rows.append(row)
                config_rows.append(row)

    write_csv(RUN_ROOT / "internal_admission_rows.csv", all_internal_rows)
    write_csv(RUN_ROOT / "semantic_support_rows.csv", all_semantic_rows)
    write_csv(RUN_ROOT / "config_rows.csv", config_rows)
    write_csv(RUN_ROOT / "run_manifest.csv", run_rows)
    write_csv(SUMMARY / "lbta_trajectory_admission_manifest.csv", manifest_rows)
    payload = {
        "schema": "acl2_v119tf_lbta_trajectory_admission_manifest_v1",
        "run_root": rel(RUN_ROOT),
        "dataset_scope": sorted(SEQ_LENGTHS),
        "seq_lengths": SEQ_LENGTHS,
        "surfaces": SURFACES,
        "variants": VARIANTS,
        "action_budget_fraction": ACTION_BUDGET_FRACTION,
        "temporal_buckets": TEMPORAL_BUCKETS,
        "internal_buckets": INTERNAL_BUCKETS,
        "min_frame_gap": MIN_FRAME_GAP,
        "random_seed": RANDOM_SEED,
        "backend": "sdpa" if USE_SDPA else "flashinfer",
        "semantic_sidecar": rel(SEM_V3),
        "semantic_sidecar_hash": sidecar_hash,
        "semantic_sidecar_version": sidecar_version,
        "internal_admission_rows": rel(RUN_ROOT / "internal_admission_rows.csv"),
        "semantic_support_rows": rel(RUN_ROOT / "semantic_support_rows.csv"),
        "config_rows": rel(RUN_ROOT / "config_rows.csv"),
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "manifest_csv": rel(SUMMARY / "lbta_trajectory_admission_manifest.csv"),
        "truthfulness_boundary": (
            "TA1 internal scores use current-code no-action outputs only. TA2/TA3 semantic scores use causal SEM-V3 prefix rows. "
            "No GT, external depth, SLAM, external geometry model, training, or output trajectory post-processing is used as runtime cue."
        ),
    }
    write_json(SUMMARY / "lbta_trajectory_admission_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
