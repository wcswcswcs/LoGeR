#!/usr/bin/env python3
"""Generate ACL2 v119-TF LB-AI-FIX Form1 frozen-schedule configs."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage2_lbai_form1_anchor_initialization"
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
SEM_V3 = RESULT_ROOT / "stage1_semv3_sidecar/semv3_prefix_rows.parquet"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"
RAW_DATA_ROOT = ROOT / "data/kitti/dataset"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
ENV_NAME = "loger"
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
SEQ_LENGTHS = {"00": 4541, "02": 4661}
AUTO_KEYFRAME_THRESHOLD = 320
SCALE_FRAMES = 8
BUFFER_SIZE = 32
BIN_SIZE = 4
RANDOM_SEEDS = list(range(21))
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


def yaml_list(items: list[int | str]) -> str:
    return "[" + ", ".join(f'"{item}"' if isinstance(item, str) else str(int(item)) for item in items) + "]"


def keyframe_interval(num_frames: int) -> int:
    return (num_frames + AUTO_KEYFRAME_THRESHOLD - 1) // AUTO_KEYFRAME_THRESHOLD


def default_frozen_indices(num_frames: int) -> list[int]:
    interval = keyframe_interval(num_frames)
    stream = [idx for idx in range(num_frames) if idx >= SCALE_FRAMES]
    return [idx for pos, idx in enumerate(stream) if interval <= 1 or pos % interval == 0]


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
    df = df[df["frame_id"].astype(int) < BUFFER_SIZE].copy()
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
                # Semantic-only anchor score: role prior, prefix visibility, shape stability,
                # reobservation evidence, and role prior. Motion is recorded but not used as
                # a positive semantic score to avoid turning it into a geometry utility.
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
                "frame_id": int(frame_id),
                "visible_track_count": int(len(frame_df)),
                "semantic_anchor_score": float(max(best_score, 0.0)),
                "semantic_anchor_score_mean": float(score_sum / max(1, len(frame_df))),
                **(best_row or {}),
            }
        rows_by_seq[str(seq)] = frame_rows
    return rows_by_seq


def bins() -> list[list[int]]:
    return [list(range(start, start + BIN_SIZE)) for start in range(0, BUFFER_SIZE, BIN_SIZE)]


def pick_per_bin(rows_by_frame: dict[int, dict[str, Any]], mode: str, seed: int | None = None) -> list[int]:
    selected: list[int] = []
    rng = random.Random(119200 + int(seed or 0))
    for frame_bin in bins():
        eligible = [frame for frame in frame_bin if frame in rows_by_frame]
        if not eligible:
            raise RuntimeError(f"empty first32 bin: {frame_bin}")
        if mode == "high_semantic":
            chosen = max(eligible, key=lambda frame: (rows_by_frame[frame]["semantic_anchor_score"], frame))
        elif mode == "low_semantic":
            chosen = min(eligible, key=lambda frame: (rows_by_frame[frame]["semantic_anchor_score"], frame))
        elif mode == "random":
            chosen = rng.choice(eligible)
        elif mode == "first_in_bin":
            chosen = min(eligible)
        else:
            raise ValueError(mode)
        selected.append(int(chosen))
    return sorted(selected)


def variants_for_seq(seq: str, rows_by_frame: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "variant": "ai0_default_firstn",
            "policy": "AI0_DEFAULT_FIRSTN_FROZEN_SCHEDULE",
            "role": "default_control",
            "scale_frame_indices": list(range(SCALE_FRAMES)),
        },
        {
            "variant": "ai2_semantic_only",
            "policy": "AI2_SEMV3_SEMANTIC_ONLY_PERSISTENT_LANDMARK",
            "role": "candidate_semantic_only",
            "scale_frame_indices": pick_per_bin(rows_by_frame, "high_semantic"),
        },
        {
            "variant": "ai5_reverse_semantic",
            "policy": "AI5_REVERSE_LOW_SEMV3_SEMANTIC",
            "role": "reverse_control",
            "scale_frame_indices": pick_per_bin(rows_by_frame, "low_semantic"),
        },
        {
            "variant": "ai8_latency_control",
            "policy": "AI8_SAME_LATENCY_SAME_FRAME_COUNT_CONTROL",
            "role": "latency_control",
            "scale_frame_indices": pick_per_bin(rows_by_frame, "first_in_bin"),
        },
    ]
    for seed in RANDOM_SEEDS:
        variants.append(
            {
                "variant": f"ai6_random_seed{seed:02d}",
                "policy": f"AI6_SAME_FIRSTM_RANDOM_SEED{seed:02d}",
                "role": "matched_random_control",
                "scale_frame_indices": pick_per_bin(rows_by_frame, "random", seed=seed + int(seq) * 100),
            }
        )
    return variants


def score_stats(rows_by_frame: dict[int, dict[str, Any]], frames: list[int]) -> dict[str, Any]:
    scores = [float(rows_by_frame.get(frame, {}).get("semantic_anchor_score", 0.0)) for frame in frames]
    roles = [str(rows_by_frame.get(frame, {}).get("best_role", "")) for frame in frames]
    labels = [str(rows_by_frame.get(frame, {}).get("best_label", "")) for frame in frames]
    return {
        "semantic_score_min": min(scores) if scores else "",
        "semantic_score_max": max(scores) if scores else "",
        "semantic_score_mean": sum(scores) / len(scores) if scores else "",
        "best_roles": ";".join(roles),
        "best_labels": ";".join(labels),
    }


def main() -> None:
    frame_scores_by_seq = semantic_frame_scores()
    logs = RUN_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    raw_action = RUN_ROOT / "raw_action"
    run_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    frame_score_rows: list[dict[str, Any]] = []
    blocked_rows = [
        {
            "variant": "AI1 internal-only view-diversity + geom-quality",
            "status": "blocked_not_generated",
            "reason": "current-code v119 internal trace for first32 non-default anchor frames not materialized yet",
        },
        {
            "variant": "AI3 internal + semantic",
            "status": "blocked_not_generated",
            "reason": "requires AI1 current-code internal trace plus SEM-V3 join",
        },
        {
            "variant": "AI4 low-dynamic legacy reference under frozen schedule",
            "status": "blocked_not_generated",
            "reason": "requires explicit v119 low-dynamic legacy rule port and frozen-schedule validation",
        },
        {
            "variant": "AI7 same-internal-bucket track shuffle",
            "status": "blocked_not_generated",
            "reason": "requires current-code internal bucket trace; SEM-V3 score shuffle would not satisfy the planned AI7 control",
        },
        {
            "variant": "Form2 zero-delay default first-n + token-level anchor role routing",
            "status": "blocked_not_generated",
            "reason": "Form2 runtime hook not implemented in this config generator",
        },
    ]
    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"
    for seq, num_frames in SEQ_LENGTHS.items():
        dataset = f"kitti_v119_lbai_form1_seq{seq}"
        dataset_cfg = CONFIG_ROOT / "datasets" / f"{dataset}.yaml"
        write_text(
            dataset_cfg,
            f"""dataset: kitti
raw_data_root: {RAW_DATA_ROOT}
_target_size: [504, 280]
_sequences: {yaml_list([seq])}
""",
        )
        frozen = default_frozen_indices(num_frames)
        variants = variants_for_seq(seq, frame_scores_by_seq[seq])
        for frame, row in sorted(frame_scores_by_seq[seq].items()):
            frame_score_rows.append(
                {
                    "schema": "acl2_v119tf_lbai_form1_frame_score_row_v1",
                    "seq": seq,
                    **row,
                }
            )
        prepare_cfg = CONFIG_ROOT / f"kitti_lbai_form1_prepare_seq{seq}.yaml"
        first_method = f"lingbot_map_v119_lbai_f1_{variants[0]['variant']}_{seq}"
        write_text(
            prepare_cfg,
            f"""workspace: {WORKSPACE}

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
  - {first_method}
""",
        )
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
        for idx, variant in enumerate(variants):
            method = f"lingbot_map_v119_lbai_f1_{variant['variant']}_{seq}"
            method_cfg = CONFIG_ROOT / "methods" / f"{method}.yaml"
            base_cfg = CONFIG_ROOT / f"kitti_lbai_form1_{variant['variant']}_seq{seq}.yaml"
            action_file = raw_action / f"{dataset}_{seq}_{method}.jsonl"
            action_file.parent.mkdir(parents=True, exist_ok=True)
            action_file.write_text("", encoding="utf-8")
            write_text(
                method_cfg,
                f"""model: lingbot_map
env: {ENV_NAME}
_checkpoint: {CHECKPOINT}
_device: cuda
_use_amp: true
_use_sdpa: false
_image_size: 518
_patch_size: 14
_enable_3d_rope: true
_num_scale_frames: 8
_max_frame_num: 1024
_kv_cache_sliding_window: 64
_kv_cache_scale_frames: 8
_auto_keyframe_threshold: 320
_area_budget: 255000
_align: 14
_mode: streaming
_keyframe_interval: auto
_keyframe_schedule_mode: global_frozen
_frozen_keyframe_indices: {yaml_list(frozen)}
_stage4_action_mode: anchor_scale_frame_indices
_stage4_scale_frame_indices: {yaml_list(variant['scale_frame_indices'])}
_stage4_action_label: v119_lbai_form1_{variant['variant']}
""",
            )
            write_text(
                base_cfg,
                f"""workspace: {WORKSPACE}

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
""",
            )
            gpu = idx % 5
            run_log = logs / f"run_{variant['variant']}_seq{seq}_gpu{gpu}.log"
            eval_log = logs / f"evaluate_{variant['variant']}_seq{seq}.log"
            common_env = (
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                f"ACL2_V105_STAGE4_ACTION_LABEL=v119_lbai_form1_{variant['variant']} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method}"
            )
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
                    "schema": "acl2_v119tf_lbai_form1_config_row_v1",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "variant": variant["variant"],
                    "policy": variant["policy"],
                    "role": variant["role"],
                    "scale_frame_indices": ";".join(str(v) for v in variant["scale_frame_indices"]),
                    "frozen_keyframe_count": len(frozen),
                    "schedule_mode": "global_frozen",
                    "form": "Form1 delayed first-M anchor selection frozen downstream schedule",
                    "config": rel(base_cfg),
                    "method_config": rel(method_cfg),
                    "action_file": rel(action_file),
                    **score_stats(frame_scores_by_seq[seq], variant["scale_frame_indices"]),
                }
            )
    write_csv(RUN_ROOT / "frame_semv3_anchor_scores_first32.csv", frame_score_rows)
    write_csv(RUN_ROOT / "config_rows.csv", config_rows)
    write_csv(RUN_ROOT / "run_manifest.csv", run_rows)
    write_csv(RUN_ROOT / "blocked_variants.csv", blocked_rows)
    summary = {
        "schema": "acl2_v119tf_lbai_form1_config_summary_v1",
        "run_root": rel(RUN_ROOT),
        "workspace": rel(WORKSPACE),
        "semv3_source": rel(SEM_V3),
        "sequences": list(SEQ_LENGTHS),
        "runnable_variant_count": len(config_rows),
        "blocked_variant_count": len(blocked_rows),
        "config_rows": rel(RUN_ROOT / "config_rows.csv"),
        "frame_score_rows": rel(RUN_ROOT / "frame_semv3_anchor_scores_first32.csv"),
        "blocked_variants": rel(RUN_ROOT / "blocked_variants.csv"),
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "truthfulness_boundary": (
            "Config generation only. AI1/AI3/AI4/AI7 and Form2 are not generated because their current-code "
            "internal trace, low-dynamic rule, or runtime hook prerequisites are not yet materialized. "
            "No LB-AI runtime pass is claimed."
        ),
    }
    write_text(RUN_ROOT / "config_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
