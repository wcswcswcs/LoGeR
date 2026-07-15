#!/usr/bin/env python3
"""Generate ACL2 v119-TF LingBot Stage0 fresh FlashInfer baseline configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage0_lingbot_fresh_baselines"
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
BENCH = ROOT / "third_party/lingbot-map/benchmark"

SEQS = ["00", "01", "02", "05"]
METHOD = "lingbot_map_stream_default_flashinfer_v119_stage0"
DATASET = "kitti_v119_stage0_00_01_02_05"
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"
RAW_DATA_ROOT = ROOT / "data/kitti/dataset"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
ENV_NAME = "loger"
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"


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


def yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    dataset_cfg = f"""dataset: kitti
raw_data_root: {RAW_DATA_ROOT}
_target_size: [504, 280]
_sequences: {yaml_list(SEQS)}
"""
    method_cfg = f"""model: lingbot_map
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
"""
    base_cfg = f"""workspace: {WORKSPACE}

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
  - {DATASET}

methods:
  - {METHOD}
"""
    write_text(CONFIG_ROOT / "datasets" / f"{DATASET}.yaml", dataset_cfg)
    write_text(CONFIG_ROOT / "methods" / f"{METHOD}.yaml", method_cfg)
    base_path = CONFIG_ROOT / "kitti_lingbot_flashinfer_stage0_00_01_02_05.yaml"
    write_text(base_path, base_cfg)

    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"
    logs = RUN_ROOT / "logs"
    rows: list[dict[str, Any]] = [
        {
            "phase": "prepare",
            "seq": ",".join(SEQS),
            "gpu": "",
            "cwd": str(BENCH),
            "config": str(base_path),
            "log": str(logs / "prepare.log"),
            "command": (
                f"{env_prefix} {CONDA} run -n {ENV_NAME} --no-capture-output "
                f"python prepare.py --config {base_path} --force > {logs / 'prepare.log'} 2>&1"
            ),
        },
        {
            "phase": "evaluate",
            "seq": ",".join(SEQS),
            "gpu": "",
            "cwd": str(BENCH),
            "config": str(base_path),
            "log": str(logs / "evaluate.log"),
            "command": (
                f"{env_prefix} {CONDA} run -n {ENV_NAME} --no-capture-output "
                f"python evaluate.py --config {base_path} --force > {logs / 'evaluate.log'} 2>&1"
            ),
        },
    ]
    for idx, seq in enumerate(SEQS):
        gpu = idx
        rows.append(
            {
                "phase": "run_worker",
                "seq": seq,
                "gpu": gpu,
                "cwd": str(BENCH),
                "config": str(base_path),
                "log": str(logs / f"run_seq{seq}_gpu{gpu}.log"),
                "command": (
                    f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {CONDA} run -n {ENV_NAME} --no-capture-output "
                    f"python run_worker.py --config {base_path} --method {METHOD} --dataset {DATASET} "
                    f"--scene {seq} --force > {logs / f'run_seq{seq}_gpu{gpu}.log'} 2>&1"
                ),
            }
        )

    write_csv(RUN_ROOT / "run_manifest.csv", rows)
    summary = {
        "schema": "acl2_v119tf_lingbot_stage0_fresh_baseline_config_v1",
        "base_config": rel(base_path),
        "dataset_config": rel(CONFIG_ROOT / "datasets" / f"{DATASET}.yaml"),
        "method_config": rel(CONFIG_ROOT / "methods" / f"{METHOD}.yaml"),
        "workspace": rel(WORKSPACE),
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "method": METHOD,
        "dataset": DATASET,
        "sequences": SEQS,
        "use_sdpa": False,
        "backend_intent": "FlashInfer via _use_sdpa=false",
        "checkpoint": rel(CHECKPOINT),
        "raw_data_root": rel(RAW_DATA_ROOT),
        "env_name": ENV_NAME,
        "pythonpath": PYTHONPATH,
        "truthfulness_boundary": "Config generation only; no metric or gate pass is claimed.",
    }
    write_text(RUN_ROOT / "config_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
