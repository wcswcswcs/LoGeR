#!/usr/bin/env python3
"""Generate ACL2 v105-TF LingBot Stage 2 trace parity configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE0_SUMMARY = RESULT_ROOT / "stage0_repo_env_audit/stage0_summary.json"
CONFIG_ROOT = RESULT_ROOT / "configs"
STAGE2 = RESULT_ROOT / "stage2_gca_trace"
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


def build() -> dict[str, Any]:
    stage0 = read_stage0()
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    raw_data_root = stage0["kitti"]["resolved_kitti_root"]
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"]["flashinfer_available_in_recommended_env"])
    workspace = (STAGE2 / "workspace").resolve()
    trace_dir = (STAGE2 / "raw_trace").resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    sequences = ["00", "02"]
    for seq in sequences:
        write_text(
            CONFIG_ROOT / f"datasets/kitti_v105_seq{seq}_trace32.yaml",
            f"""dataset: kitti
raw_data_root: {raw_data_root}
_target_size: [504, 280]
_sequences: ["{seq}"]
sampling:
  strategy: sequence
  num_frames: 32
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
_mode: streaming
_keyframe_interval: auto
"""
    method_names = [
        "lingbot_map_stream_default_stage2_notrace",
        "lingbot_map_stream_default_stage2_trace",
    ]
    for method_name in method_names:
        write_text(CONFIG_ROOT / f"methods/{method_name}.yaml", common_method)

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
    config_map: dict[str, tuple[str, str, str]] = {}
    for seq in sequences:
        dataset_name = f"kitti_v105_seq{seq}_trace32"
        for mode, method_name in [
            ("notrace", "lingbot_map_stream_default_stage2_notrace"),
            ("trace", "lingbot_map_stream_default_stage2_trace"),
        ]:
            filename = f"kitti_lingbot_stage2_{mode}_seq{seq}_trace32.yaml"
            config_map[filename] = (dataset_name, method_name, seq)
            write_text(
                CONFIG_ROOT / filename,
                base_common + f"\ndatasets:\n  - {dataset_name}\n\nmethods:\n  - {method_name}\n",
            )

    env_prefix = (
        "PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={pythonpath}"
    )
    rows: list[dict[str, Any]] = []
    for filename, (dataset_name, method_name, seq) in config_map.items():
        config_abs = (CONFIG_ROOT / filename).resolve()
        run_name = filename.removesuffix(".yaml")
        gpu = "0" if seq == "00" else "2"
        trace_file = trace_dir / f"{dataset_name}_{seq}_{method_name}.jsonl"
        trace_env = ""
        if method_name.endswith("_trace"):
            trace_env = (
                f"ACL2_V105_GCA_TRACE_FILE={trace_file} "
                f"ACL2_V105_GCA_TRACE_CASE={dataset_name}/{seq}/{method_name} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset_name} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method_name} "
                "ACL2_V105_GCA_TRACE_GLOBAL_IDXS=0,11,23 "
                "ACL2_V105_GCA_TRACE_TOPK=5 "
                "ACL2_V105_GCA_TRACE_MAX_ROWS=20000 "
            )
        commands = [
            (
                "prepare",
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config_abs} --force",
            ),
            (
                "run_worker",
                f"{env_prefix} {trace_env}CUDA_VISIBLE_DEVICES={gpu} {conda_path} run -n {env_name} "
                f"python run_worker.py --config {config_abs} --method {method_name} "
                f"--dataset {dataset_name} --scene {seq} --force",
            ),
            (
                "evaluate",
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {conda_path} run -n {env_name} "
                f"python evaluate.py --config {config_abs} --force",
            ),
            (
                "report",
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {conda_path} run -n {env_name} "
                f"python report.py --workspace {workspace} --dataset {dataset_name}",
            ),
        ]
        for phase, command in commands:
            rows.append(
                {
                    "run_name": run_name,
                    "phase": phase,
                    "cwd": str(BENCH),
                    "config": str(config_abs),
                    "dataset": dataset_name,
                    "seq": seq,
                    "method": method_name,
                    "trace_file": str(trace_file) if method_name.endswith("_trace") else "",
                    "command": command,
                    "status": "planned",
                }
            )

    write_csv(STAGE2 / "run_manifest.csv", rows)
    summary = {
        "schema": "acl2_v105tf_lingbot_stage2_trace_config_generation_v1",
        "sequences": sequences,
        "frames_per_sequence": 32,
        "workspace": str(workspace),
        "raw_trace_dir": str(trace_dir),
        "configs": [str((CONFIG_ROOT / name).relative_to(ROOT)) for name in sorted(config_map)],
        "dataset_configs": [
            str((CONFIG_ROOT / f"datasets/kitti_v105_seq{seq}_trace32.yaml").relative_to(ROOT))
            for seq in sequences
        ],
        "method_configs": [
            str((CONFIG_ROOT / f"methods/{name}.yaml").relative_to(ROOT))
            for name in method_names
        ],
        "checkpoint": checkpoint,
        "raw_data_root": raw_data_root,
        "env_name": env_name,
        "conda_path": conda_path,
        "pythonpath": pythonpath,
        "use_sdpa": use_sdpa,
        "trace_backend": "SDPA_TRACE",
        "trace_global_idxs": "0,11,23",
    }
    write_text(
        STAGE2 / "config_generation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
