#!/usr/bin/env python3
"""Generate ACL2 v105-TF LingBot Stage 1 benchmark configs and manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE0_SUMMARY = RESULT_ROOT / "stage0_repo_env_audit/stage0_summary.json"
CONFIG_ROOT = RESULT_ROOT / "configs"
STAGE1 = RESULT_ROOT / "stage1_lingbot_baseline"
BENCH = ROOT / "third_party/lingbot-map/benchmark"


def read_stage0() -> dict[str, Any]:
    with STAGE0_SUMMARY.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"


def build() -> dict[str, Any]:
    stage0 = read_stage0()
    if not stage0.get("stage1_baseline_allowed"):
        raise SystemExit("Stage1 baseline is not allowed by stage0_summary.json")
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    raw_data_root = stage0["kitti"]["resolved_kitti_root"]
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"]["flashinfer_available_in_recommended_env"])
    workspace = (RESULT_ROOT / "stage1_lingbot_baseline/workspace").resolve()
    sequences = ["00", "01", "02", "05"]

    dataset_full = f"""dataset: kitti
raw_data_root: {raw_data_root}
_target_size: [504, 280]
_sequences: {yaml_list(sequences)}
"""
    dataset_debug = dataset_full + """sampling:
  strategy: sequence
  num_frames: 96
"""
    write_text(CONFIG_ROOT / "datasets/kitti_v105_00_01_02_05.yaml", dataset_full)
    write_text(CONFIG_ROOT / "datasets/kitti_v105_00_01_02_05_debug96.yaml", dataset_debug)
    debug_sequences = ["00", "01", "02", "05"]
    for seq in debug_sequences:
        write_text(
            CONFIG_ROOT / f"datasets/kitti_v105_seq{seq}_debug96.yaml",
            f"""dataset: kitti
raw_data_root: {raw_data_root}
_target_size: [504, 280]
_sequences: ["{seq}"]
sampling:
  strategy: sequence
  num_frames: 96
""",
        )

    common_method = f"""model: lingbot_map
env: {env_name}
_checkpoint: {checkpoint}
_device: cuda
_use_amp: true
_use_sdpa: {str(use_sdpa).lower()}
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
"""
    methods = {
        "lingbot_map_stream_default": common_method
        + """_mode: streaming
_keyframe_interval: auto
""",
        "lingbot_map_stream_kf1": common_method
        + """_mode: streaming
_keyframe_interval: 1
""",
        "lingbot_map_stream_kf4": common_method
        + """_mode: streaming
_keyframe_interval: 4
""",
        "lingbot_map_window64": common_method
        + """_mode: windowed
_window_size: 64
_overlap_size: 16
_keyframe_interval: 1
""",
    }
    for name, text in methods.items():
        write_text(CONFIG_ROOT / f"methods/{name}.yaml", text)

    base_common = f"""workspace: {workspace}

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
"""
    base_map = {
        "kitti_lingbot_stream_default.yaml": ("kitti_v105_00_01_02_05", "lingbot_map_stream_default"),
        "kitti_lingbot_stream_kf1.yaml": ("kitti_v105_00_01_02_05", "lingbot_map_stream_kf1"),
        "kitti_lingbot_stream_kf4.yaml": ("kitti_v105_00_01_02_05", "lingbot_map_stream_kf4"),
        "kitti_lingbot_window64.yaml": ("kitti_v105_00_01_02_05", "lingbot_map_window64"),
        "kitti_lingbot_stream_default_debug96.yaml": ("kitti_v105_00_01_02_05_debug96", "lingbot_map_stream_default"),
        "kitti_lingbot_stream_default_seq00_debug96.yaml": ("kitti_v105_seq00_debug96", "lingbot_map_stream_default"),
        "kitti_lingbot_stream_default_seq01_debug96.yaml": ("kitti_v105_seq01_debug96", "lingbot_map_stream_default"),
        "kitti_lingbot_stream_default_seq02_debug96.yaml": ("kitti_v105_seq02_debug96", "lingbot_map_stream_default"),
        "kitti_lingbot_stream_default_seq05_debug96.yaml": ("kitti_v105_seq05_debug96", "lingbot_map_stream_default"),
        "kitti_lingbot_stream_kf1_seq00_debug96.yaml": ("kitti_v105_seq00_debug96", "lingbot_map_stream_kf1"),
        "kitti_lingbot_stream_kf4_seq00_debug96.yaml": ("kitti_v105_seq00_debug96", "lingbot_map_stream_kf4"),
        "kitti_lingbot_window64_seq00_debug96.yaml": ("kitti_v105_seq00_debug96", "lingbot_map_window64"),
    }
    for filename, (dataset_name, method_name) in base_map.items():
        write_text(
            CONFIG_ROOT / filename,
            base_common + f"\ndatasets:\n  - {dataset_name}\n\nmethods:\n  - {method_name}\n",
        )

    env_prefix = (
        f"PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={pythonpath}"
    )
    rows: list[dict[str, Any]] = []
    for filename, (dataset_name, _method_name) in base_map.items():
        config_abs = (CONFIG_ROOT / filename).resolve()
        run_name = filename.removesuffix(".yaml")
        for phase, script, extra in [
            ("prepare", "prepare.py", "--force"),
            ("run", "run.py", "--force"),
            ("evaluate", "evaluate.py", "--force"),
            ("report", "report.py", ""),
        ]:
            if phase == "report":
                command = (
                    f"{env_prefix} CUDA_VISIBLE_DEVICES=0 "
                    f"{conda_path} run -n {env_name} python {script} --workspace {workspace} --dataset {dataset_name}"
                )
            else:
                command = (
                    f"{env_prefix} CUDA_VISIBLE_DEVICES=0 "
                    f"{conda_path} run -n {env_name} python {script} --config {config_abs}"
                )
                if extra:
                    command += f" {extra}"
            rows.append(
                {
                    "run_name": run_name,
                    "phase": phase,
                    "cwd": str(BENCH),
                    "config": str(config_abs),
                    "command": command,
                    "status": "planned",
                    "notes": "debug96 is the initial smoke; full configs are generated but not yet run",
                }
            )
    write_csv(STAGE1 / "run_manifest.csv", rows)
    write_csv(
        STAGE1 / "job_results.csv",
        [
            {
                "run_name": name.removesuffix(".yaml"),
                "phase": "config_generation",
                "returncode": 0,
                "status": "config_written",
                "config": str((CONFIG_ROOT / name).resolve()),
            }
            for name in base_map
        ],
    )
    summary = {
        "schema": "acl2_v105tf_lingbot_stage1_config_generation_v1",
        "configs": [str((CONFIG_ROOT / name).relative_to(ROOT)) for name in base_map],
        "dataset_configs": [
            str((CONFIG_ROOT / "datasets/kitti_v105_00_01_02_05.yaml").relative_to(ROOT)),
            str((CONFIG_ROOT / "datasets/kitti_v105_00_01_02_05_debug96.yaml").relative_to(ROOT)),
            *[
                str((CONFIG_ROOT / f"datasets/kitti_v105_seq{seq}_debug96.yaml").relative_to(ROOT))
                for seq in debug_sequences
            ],
        ],
        "method_configs": [str((CONFIG_ROOT / f"methods/{name}.yaml").relative_to(ROOT)) for name in methods],
        "checkpoint": checkpoint,
        "raw_data_root": raw_data_root,
        "env_name": env_name,
        "conda_path": conda_path,
        "pythonpath": pythonpath,
        "use_sdpa": use_sdpa,
        "initial_smoke_config": str((CONFIG_ROOT / "kitti_lingbot_stream_default_debug96.yaml").relative_to(ROOT)),
    }
    write_text(STAGE1 / "config_generation_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
