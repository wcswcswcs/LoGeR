#!/usr/bin/env python3
"""Generate ACL2 v119-TF Stage1 LB-SCHED parity configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage1_lbsched_parity"
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"
RAW_DATA_ROOT = ROOT / "data/kitti/dataset"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
ENV_NAME = "loger"
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
SEQ_LENGTHS = {"00": 4541, "02": 4661}
AUTO_KEYFRAME_THRESHOLD = 320
SCALE_FRAMES = 8


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def yaml_list(items: list[int | str]) -> str:
    return "[" + ", ".join(f'"{item}"' if isinstance(item, str) else str(item) for item in items) + "]"


def keyframe_interval(num_frames: int) -> int:
    return (num_frames + AUTO_KEYFRAME_THRESHOLD - 1) // AUTO_KEYFRAME_THRESHOLD


def default_frozen_indices(num_frames: int) -> list[int]:
    interval = keyframe_interval(num_frames)
    stream = [idx for idx in range(num_frames) if idx >= SCALE_FRAMES]
    return [idx for pos, idx in enumerate(stream) if interval <= 1 or pos % interval == 0]


def method_variants(seq: str) -> list[dict[str, Any]]:
    if seq == "00":
        delayed = [0, 1, 2, 3, 668, 683, 3113, 3128]
        spread = [0, 180, 540, 900, 1440, 2160, 3240, 4320]
    else:
        delayed = [0, 1, 2, 3, 2813, 2843, 3818, 3833]
        spread = [0, 240, 720, 1200, 1800, 2520, 3600, 4440]
    return [
        {
            "variant": "legacy_default",
            "schedule_mode": "legacy_stream_pos",
            "scale_frame_indices": "",
            "parity_group": "default_legacy_reference",
        },
        {
            "variant": "frozen_default_a",
            "schedule_mode": "global_frozen",
            "scale_frame_indices": "",
            "parity_group": "default_frozen_vs_legacy",
        },
        {
            "variant": "frozen_default_b",
            "schedule_mode": "global_frozen",
            "scale_frame_indices": "",
            "parity_group": "default_frozen_repeat",
        },
        {
            "variant": "frozen_delayed_b1_like",
            "schedule_mode": "global_frozen",
            "scale_frame_indices": delayed,
            "parity_group": "random_anchor_schedule_match",
        },
        {
            "variant": "frozen_spread_seed0",
            "schedule_mode": "global_frozen",
            "scale_frame_indices": spread,
            "parity_group": "random_anchor_schedule_match",
        },
    ]


def main() -> None:
    rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    logs = RUN_ROOT / "logs"
    raw_action = RUN_ROOT / "raw_action"
    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"
    for seq, num_frames in SEQ_LENGTHS.items():
        dataset = f"kitti_v119_stage1_lbsched_seq{seq}"
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
        prepare_cfg = CONFIG_ROOT / f"kitti_lbsched_prepare_seq{seq}.yaml"
        first_method = f"lingbot_map_v119_lbsched_legacy_default_{seq}"
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
        rows.append(
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
        for method_idx, variant in enumerate(method_variants(seq)):
            method = f"lingbot_map_v119_lbsched_{variant['variant']}_{seq}"
            method_cfg = CONFIG_ROOT / "methods" / f"{method}.yaml"
            base_cfg = CONFIG_ROOT / f"kitti_lbsched_{variant['variant']}_seq{seq}.yaml"
            action_file = raw_action / f"{dataset}_{seq}_{method}.jsonl"
            action_file.parent.mkdir(parents=True, exist_ok=True)
            action_file.write_text("", encoding="utf-8")
            scale_line = ""
            if variant["scale_frame_indices"]:
                scale_line = (
                    "_stage4_action_mode: anchor_scale_frame_indices\n"
                    f"_stage4_scale_frame_indices: {yaml_list(variant['scale_frame_indices'])}\n"
                )
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
_keyframe_schedule_mode: {variant['schedule_mode']}
_frozen_keyframe_indices: {yaml_list(frozen)}
_stage4_action_label: v119_stage1_lbsched_{variant['variant']}
{scale_line}""",
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
            gpu = method_idx
            run_log = logs / f"run_{variant['variant']}_seq{seq}_gpu{gpu}.log"
            eval_log = logs / f"evaluate_{variant['variant']}_seq{seq}.log"
            common_env = (
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                f"ACL2_V105_STAGE4_ACTION_LABEL=v119_stage1_lbsched_{variant['variant']} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method}"
            )
            rows.append(
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
            rows.append(
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
                    "schema": "acl2_v119tf_stage1_lbsched_parity_config_row_v1",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "variant": variant["variant"],
                    "parity_group": variant["parity_group"],
                    "schedule_mode": variant["schedule_mode"],
                    "scale_frame_indices": ";".join(str(idx) for idx in variant["scale_frame_indices"])
                    if variant["scale_frame_indices"]
                    else "",
                    "frozen_keyframe_count": len(frozen),
                    "config": rel(base_cfg),
                    "method_config": rel(method_cfg),
                    "action_file": rel(action_file),
                }
            )
    write_csv(RUN_ROOT / "run_manifest.csv", rows)
    write_csv(RUN_ROOT / "config_rows.csv", config_rows)
    summary = {
        "schema": "acl2_v119tf_stage1_lbsched_parity_config_summary_v1",
        "run_root": rel(RUN_ROOT),
        "workspace": rel(WORKSPACE),
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "config_rows": rel(RUN_ROOT / "config_rows.csv"),
        "sequences": list(SEQ_LENGTHS),
        "variants": [row["variant"] for row in config_rows if row["seq"] == "00"],
        "truthfulness_boundary": "Config generation only; no runtime parity pass is claimed.",
    }
    write_text(RUN_ROOT / "config_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
