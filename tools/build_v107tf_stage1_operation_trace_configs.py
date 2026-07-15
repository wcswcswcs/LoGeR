#!/usr/bin/env python3
"""Generate ACL2 v107TF Stage1 cache-operation trace configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106R_TRACE = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control/stage1_memory_operation_map/targeted_trace"
V107 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
V105_STAGE0 = V105 / "stage0_repo_env_audit/stage0_summary.json"
SOURCE_TARGET_MANIFEST = V106R_TRACE / "target_manifest.csv"
CONFIG_ROOT = V107 / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
STAGE1 = V107 / "stage1_cache_operation_instrumentation"
WORKSPACE = STAGE1 / "workspace"
RAW_TRACE = STAGE1 / "raw_trace"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def method_yaml(checkpoint: str, env_name: str, use_sdpa: bool) -> str:
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
            "_stage4_action_label: v107tf_operation_trace_only",
            "_stage4_action_mode: force_non_keyframe",
            "",
        ]
    )


def run_config_yaml(dataset: str, method: str) -> str:
    return "\n".join(
        [
            f"workspace: {WORKSPACE.resolve()}",
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


def normalize_targets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(SOURCE_TARGET_MANIFEST):
        item = dict(row)
        item["schema"] = "acl2_v107tf_stage1_operation_trace_target_v1"
        item["source_manifest"] = str(SOURCE_TARGET_MANIFEST.relative_to(ROOT))
        rows.append(item)
    return rows


def build() -> dict[str, Any]:
    stage0 = json.loads(V105_STAGE0.read_text(encoding="utf-8"))
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    raw_data_root = stage0["kitti"]["resolved_kitti_root"]
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"]["flashinfer_available_in_recommended_env"])

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)

    targets = normalize_targets()
    write_csv(STAGE1 / "target_manifest.csv", targets)

    notrace_method = "lingbot_map_v107tf_stage1_operation_notrace"
    trace_method = "lingbot_map_v107tf_stage1_operation_trace"
    write_text(METHOD_DIR / f"{notrace_method}.yaml", method_yaml(checkpoint, env_name, use_sdpa))
    write_text(METHOD_DIR / f"{trace_method}.yaml", method_yaml(checkpoint, env_name, use_sdpa))

    env_prefix_base = (
        "PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={pythonpath}"
    )
    gpu_cycle = ["0", "1", "2", "3", "4"]
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        seq = str(target["seq"])
        dataset = f"kitti_v107tf_{target['target_id']}_trace{target['trace_frame_count']}"
        gpu = gpu_cycle[index % len(gpu_cycle)]
        dataset_path = DATASET_DIR / f"{dataset}.yaml"
        write_text(
            dataset_path,
            "\n".join(
                [
                    "dataset: kitti",
                    f"raw_data_root: {raw_data_root}",
                    "_target_size: [504, 280]",
                    f"_sequences: [\"{seq}\"]",
                    "sampling:",
                    "  strategy: sequence",
                    f"  start_idx: {target['trace_start_idx']}",
                    f"  end_idx: {target['trace_end_idx_exclusive']}",
                    "  stride: 1",
                    "",
                ]
            ),
        )
        notrace_config = CONFIG_ROOT / f"kitti_lingbot_v107tf_stage1_{target['target_id']}_notrace.yaml"
        trace_config = CONFIG_ROOT / f"kitti_lingbot_v107tf_stage1_{target['target_id']}_trace.yaml"
        write_text(notrace_config, run_config_yaml(dataset, notrace_method))
        write_text(trace_config, run_config_yaml(dataset, trace_method))
        trace_file = RAW_TRACE / f"{dataset}_{seq}_{trace_method}.jsonl"
        trace_env = (
            f"ACL2_V107_CACHE_TRACE_FILE={trace_file} "
            f"ACL2_V107_CACHE_TRACE_RUN_ID={target['target_id']} "
            f"ACL2_V107_CACHE_TRACE_CASE={dataset}/{seq}/{trace_method} "
            f"ACL2_V107_CACHE_TRACE_DATASET={dataset} "
            f"ACL2_V107_CACHE_TRACE_SEQ={seq} "
            f"ACL2_V107_CACHE_TRACE_METHOD={trace_method} "
            f"ACL2_V107_CACHE_TRACE_WINDOW_ID={target['window_index']} "
            f"ACL2_V107_CACHE_TRACE_FRAME_START_IDX={target['trace_start_idx']} "
            f"ACL2_V107_CACHE_TRACE_GLOBAL_IDXS={target['trace_global_idxs']} "
            "ACL2_V107_CACHE_TRACE_MAX_ROWS=240000"
        )
        commands = [
            (
                "prepare",
                notrace_config,
                notrace_method,
                "",
                f"{env_prefix_base} CUDA_VISIBLE_DEVICES={gpu} {conda_path} run -n {env_name} "
                f"python prepare.py --config {notrace_config.resolve()} --force",
            ),
            (
                "run_worker_notrace",
                notrace_config,
                notrace_method,
                "",
                f"{env_prefix_base} CUDA_VISIBLE_DEVICES={gpu} {conda_path} run -n {env_name} "
                f"python run_worker.py --config {notrace_config.resolve()} --method {notrace_method} "
                f"--dataset {dataset} --scene {seq} --force",
            ),
            (
                "run_worker_trace",
                trace_config,
                trace_method,
                str(trace_file),
                f"{env_prefix_base} {trace_env} CUDA_VISIBLE_DEVICES={gpu} {conda_path} run -n {env_name} "
                f"python run_worker.py --config {trace_config.resolve()} --method {trace_method} "
                f"--dataset {dataset} --scene {seq} --force",
            ),
        ]
        for phase, config, method, trace_file_value, command in commands:
            rows.append(
                {
                    "schema": "acl2_v107tf_stage1_operation_trace_manifest_row_v1",
                    "target_id": target["target_id"],
                    "target_kind": target["target_kind"],
                    "run_name": f"kitti_lingbot_v107tf_stage1_{target['target_id']}_{phase}",
                    "phase": phase,
                    "cwd": str(BENCHMARK),
                    "config": str(config.resolve()),
                    "dataset": dataset,
                    "seq": seq,
                    "method": method,
                    "gpu": gpu,
                    "trace_file": trace_file_value,
                    "trace_global_idxs": target["trace_global_idxs"],
                    "trace_start_idx": target["trace_start_idx"],
                    "trace_end_idx_exclusive": target["trace_end_idx_exclusive"],
                    "window_index": target["window_index"],
                    "command": command,
                    "status": "planned",
                }
            )

    write_csv(STAGE1 / "run_manifest.csv", rows)
    summary = {
        "schema": "acl2_v107tf_stage1_operation_trace_config_summary_v1",
        "target_count": len(targets),
        "manifest_rows": len(rows),
        "workspace": str(WORKSPACE.resolve()),
        "raw_trace_dir": str(RAW_TRACE.resolve()),
        "target_manifest": str((STAGE1 / "target_manifest.csv").relative_to(ROOT)),
        "run_manifest": str((STAGE1 / "run_manifest.csv").relative_to(ROOT)),
        "notrace_method": notrace_method,
        "trace_method": trace_method,
        "use_sdpa": use_sdpa,
        "source_target_manifest": str(SOURCE_TARGET_MANIFEST.relative_to(ROOT)),
        "trace_env_schema": "ACL2_V107_CACHE_TRACE_*",
    }
    write_text(STAGE1 / "config_generation_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
