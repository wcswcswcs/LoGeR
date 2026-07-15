#!/usr/bin/env python3
"""Generate ACL2 v119-TF LingBot B1 Stage0 fresh FlashInfer configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage0_lingbot_b1_fresh_baselines"
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
BENCH = ROOT / "third_party/lingbot-map/benchmark"

SEQS = {
    "00": [668, 683, 3113, 3128, 3143, 3158, 3173],
    "02": [2813, 2843, 3818, 3833, 3848, 3863, 3893],
}
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"
RAW_DATA_ROOT = ROOT / "data/kitti/dataset"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
ENV_NAME = "loger"
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
ACTION_LABEL = "v119_stage0_B1_semantic_only_flashinfer"


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


def yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"


def main() -> None:
    logs = RUN_ROOT / "logs"
    raw_action = RUN_ROOT / "raw_action"
    rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"

    for idx, (seq, forced_indices) in enumerate(SEQS.items()):
        dataset = f"kitti_v119_stage0_b1_fullseq_{seq}"
        method = f"lingbot_map_v119_stage0_B1_semantic_only_flashinfer_{seq}"
        base_cfg = CONFIG_ROOT / f"kitti_lingbot_b1_flashinfer_stage0_{seq}.yaml"
        dataset_cfg = CONFIG_ROOT / "datasets" / f"{dataset}.yaml"
        method_cfg = CONFIG_ROOT / "methods" / f"{method}.yaml"
        action_file = raw_action / f"{dataset}_{seq}_{method}.jsonl"
        action_file.parent.mkdir(parents=True, exist_ok=True)
        action_file.write_text("", encoding="utf-8")

        write_text(
            dataset_cfg,
            f"""dataset: kitti
raw_data_root: {RAW_DATA_ROOT}
_target_size: [504, 280]
_sequences: {yaml_list([seq])}
""",
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
_force_non_keyframe_indices: {forced_indices}
_stage4_action_label: {ACTION_LABEL}
_stage4_action_mode: force_non_keyframe
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

        gpu = idx
        prepare_log = logs / f"prepare_seq{seq}.log"
        run_log = logs / f"run_seq{seq}_gpu{gpu}.log"
        eval_log = logs / f"evaluate_seq{seq}.log"
        common_env = (
            f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} "
            f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
            f"ACL2_V105_STAGE4_ACTION_LABEL={ACTION_LABEL} "
            f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
            f"ACL2_V105_GCA_TRACE_SEQ={seq} "
            f"ACL2_V105_GCA_TRACE_METHOD={method} "
            "ACL2_V108_STAGE4_POLICY_ID=B1_semantic_only "
            "ACL2_V108_STAGE4_SURFACE_ID=B "
            "ACL2_V110R_STAGE3_POLICY_ID=B1_semantic_only "
            "ACL2_V110R_STAGE3_CANDIDATE_ID=B1"
        )
        rows.extend(
            [
                {
                    "phase": "prepare",
                    "seq": seq,
                    "gpu": gpu,
                    "cwd": str(BENCH),
                    "config": str(base_cfg),
                    "dataset": dataset,
                    "method": method,
                    "action_file": "",
                    "log": str(prepare_log),
                    "command": (
                        f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {CONDA} run -n {ENV_NAME} --no-capture-output "
                        f"python prepare.py --config {base_cfg} --force > {prepare_log} 2>&1"
                    ),
                },
                {
                    "phase": "run_worker",
                    "seq": seq,
                    "gpu": gpu,
                    "cwd": str(BENCH),
                    "config": str(base_cfg),
                    "dataset": dataset,
                    "method": method,
                    "action_file": str(action_file),
                    "forced_indices": ";".join(str(value) for value in forced_indices),
                    "log": str(run_log),
                    "command": (
                        f"{common_env} {CONDA} run -n {ENV_NAME} --no-capture-output "
                        f"python run_worker.py --config {base_cfg} --method {method} "
                        f"--dataset {dataset} --scene {seq} --force > {run_log} 2>&1"
                    ),
                },
                {
                    "phase": "evaluate",
                    "seq": seq,
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
                },
            ]
        )
        config_rows.append(
            {
                "schema": "acl2_v119tf_lingbot_b1_stage0_fresh_config_row_v1",
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "config": rel(base_cfg),
                "dataset_config": rel(dataset_cfg),
                "method_config": rel(method_cfg),
                "action_file": rel(action_file),
                "forced_indices": ";".join(str(value) for value in forced_indices),
                "use_sdpa": False,
                "backend_intent": "v119 fresh FlashInfer replacement for B1 strong carrier",
            }
        )

    write_csv(RUN_ROOT / "run_manifest.csv", rows)
    write_csv(RUN_ROOT / "config_rows.csv", config_rows)
    summary = {
        "schema": "acl2_v119tf_lingbot_b1_stage0_fresh_config_summary_v1",
        "run_root": rel(RUN_ROOT),
        "workspace": rel(WORKSPACE),
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "config_rows": rel(RUN_ROOT / "config_rows.csv"),
        "sequences": list(SEQS),
        "use_sdpa": False,
        "backend_intent": "FlashInfer matched with v119 default LingBot baseline; not a strict v110 SDPA replay",
        "truthfulness_boundary": "Config generation only; no metric or gate pass is claimed.",
    }
    write_text(RUN_ROOT / "config_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
