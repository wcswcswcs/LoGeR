#!/usr/bin/env python3
"""Generate no-action head-resolved trace configs for ACL2 v105-TF Stage 4."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE0_SUMMARY = RESULT_ROOT / "stage0_repo_env_audit/stage0_summary.json"
CONFIG_ROOT = RESULT_ROOT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
STAGE4_HEAD = RESULT_ROOT / "stage4_lingbot_headlocal_trace"
WORKSPACE = STAGE4_HEAD / "workspace"
RAW_TRACE = STAGE4_HEAD / "raw_trace"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
SEQUENCES = ["00", "02"]
HEAD_IDXS = ",".join(str(i) for i in range(16))


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


def method_yaml(method: str, checkpoint: str, env_name: str, use_sdpa: bool) -> str:
    return "\n".join(
        [
            "model: lingbot_map",
            f"env: {env_name}",
            f"_checkpoint: {checkpoint}",
            "_device: cuda",
            "_use_amp: true",
            f"_use_sdpa: {str(use_sdpa).lower()}",
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
            "_stage4_action_label: headlocal_trace_only",
            "_stage4_action_mode: force_non_keyframe",
            "",
        ]
    )


def base_yaml(dataset: str, method: str) -> str:
    return "\n".join(
        [
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
            f"  - {dataset}",
            "",
            "methods:",
            f"  - {method}",
            "",
        ]
    )


def build() -> dict[str, Any]:
    stage0 = json.loads(STAGE0_SUMMARY.read_text(encoding="utf-8"))
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"]["flashinfer_available_in_recommended_env"])

    STAGE4_HEAD.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        dataset = f"kitti_v105_seq{seq}_trace32"
        method = f"lingbot_map_stage4_headlocal_trace_seq{seq}"
        config = CONFIG_ROOT / f"kitti_lingbot_stage4_headlocal_trace_seq{seq}_trace32.yaml"
        method_path = METHOD_DIR / f"{method}.yaml"
        trace_file = RAW_TRACE / f"{dataset}_{seq}_{method}.jsonl"
        gpu = "0" if seq == "00" else "2"

        method_path.write_text(method_yaml(method, checkpoint, env_name, use_sdpa), encoding="utf-8")
        config.write_text(base_yaml(dataset, method), encoding="utf-8")

        env_prefix = (
            "PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
            f"PYTHONPATH={pythonpath} "
            f"CUDA_VISIBLE_DEVICES={gpu}"
        )
        trace_env = (
            f"ACL2_V105_GCA_TRACE_FILE={trace_file} "
            f"ACL2_V105_GCA_TRACE_CASE={dataset}/{seq}/{method} "
            f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
            f"ACL2_V105_GCA_TRACE_SEQ={seq} "
            f"ACL2_V105_GCA_TRACE_METHOD={method} "
            "ACL2_V105_GCA_TRACE_GLOBAL_IDXS=0,11,23 "
            f"ACL2_V105_GCA_TRACE_HEAD_IDXS={HEAD_IDXS} "
            "ACL2_V105_GCA_TRACE_TOPK=5 "
            "ACL2_V105_GCA_TRACE_MAX_ROWS=120000"
        )
        commands = {
            "prepare": f"{env_prefix} {conda_path} run -n {env_name} python prepare.py --config {config} --force",
            "run_worker": (
                f"{env_prefix} {trace_env} {conda_path} run -n {env_name} python run_worker.py "
                f"--config {config} --method {method} --dataset {dataset} --scene {seq} --force"
            ),
            "evaluate": f"{env_prefix} {conda_path} run -n {env_name} python evaluate.py --config {config} --force",
            "report": f"{env_prefix} {conda_path} run -n {env_name} python report.py --workspace {WORKSPACE} --dataset {dataset}",
        }
        for phase, command in commands.items():
            rows.append(
                {
                    "schema": "acl2_v105tf_lingbot_stage4_headlocal_manifest_v1",
                    "run_name": f"kitti_lingbot_stage4_headlocal_trace_seq{seq}",
                    "phase": phase,
                    "cwd": str(BENCHMARK),
                    "config": str(config),
                    "dataset": dataset,
                    "seq": seq,
                    "method": method,
                    "trace_file": str(trace_file),
                    "command": command,
                    "status": "planned",
                }
            )

    write_csv(STAGE4_HEAD / "run_manifest.csv", rows)
    summary = {
        "schema": "acl2_v105tf_lingbot_stage4_headlocal_config_summary_v1",
        "sequences": SEQUENCES,
        "head_idxs": HEAD_IDXS,
        "manifest_rows": len(rows),
        "workspace": str(WORKSPACE),
        "raw_trace_dir": str(RAW_TRACE),
        "use_sdpa": use_sdpa,
    }
    (STAGE4_HEAD / "config_generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
