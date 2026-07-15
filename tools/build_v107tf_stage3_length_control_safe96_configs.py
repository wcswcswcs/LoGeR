#!/usr/bin/env python3
"""Generate v107TF Stage3 96F safe-good length-control trace configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V107 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
V105_STAGE0 = V105 / "stage0_repo_env_audit/stage0_summary.json"
SOURCE_TARGETS = V107 / "stage1_cache_operation_instrumentation/target_manifest.csv"
OUT = V107 / "stage3_operation_discovery/length_control_safe96"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_TRACE = OUT / "raw_trace"
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


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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
            "_stage4_action_label: v107tf_stage3_length_control_trace_only",
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


def sequence_frame_count(raw_root: Path, seq: str) -> int:
    image_dir = raw_root / "sequences" / seq / "image_2"
    return len(sorted(image_dir.glob("*.png")))


def build() -> dict[str, Any]:
    stage0 = json.loads(V105_STAGE0.read_text(encoding="utf-8"))
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(stage0["kitti"]["resolved_kitti_root"])
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"]["flashinfer_available_in_recommended_env"])

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)

    notrace_method = "lingbot_map_v107tf_stage3_safe96_notrace"
    trace_method = "lingbot_map_v107tf_stage3_safe96_trace"
    write_text(METHOD_DIR / f"{notrace_method}.yaml", method_yaml(checkpoint, env_name, use_sdpa))
    write_text(METHOD_DIR / f"{trace_method}.yaml", method_yaml(checkpoint, env_name, use_sdpa))

    env_prefix_base = (
        "PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={pythonpath}"
    )
    gpu_cycle = ["0", "1", "2", "3", "4"]
    target_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    source_rows = read_csv(SOURCE_TARGETS)
    controls = [
        row for row in source_rows
        if row.get("target_kind") == "safe_good_low_drift" and int(float(row.get("trace_frame_count", "0"))) < 96
    ]
    for index, source in enumerate(controls):
        seq = str(source["seq"])
        frame_count = sequence_frame_count(raw_data_root, seq)
        trace_start = int(float(source["trace_start_idx"]))
        trace_end = trace_start + 96
        if trace_end > frame_count:
            skipped_rows.append({
                **source,
                "skip_reason": f"trace_end_{trace_end}_gt_frame_count_{frame_count}",
            })
            continue

        target_id = f"{source['target_id']}_safe96_length_control"
        dataset = f"kitti_v107tf_{target_id}_trace96"
        gpu = gpu_cycle[index % len(gpu_cycle)]
        target = {
            **source,
            "schema": "acl2_v107tf_stage3_safe96_length_control_target_v1",
            "target_id": target_id,
            "original_target_id": source["target_id"],
            "target_kind": "safe_good_low_drift_length_control",
            "trace_end_idx_exclusive": trace_end,
            "trace_frame_count": 96,
            "length_control_reason": "match_high_l3_trace_frame_count_to_test_cache_budget_operation_confound",
            "source_manifest": rel(SOURCE_TARGETS),
        }
        target_rows.append(target)

        write_text(
            DATASET_DIR / f"{dataset}.yaml",
            "\n".join(
                [
                    "dataset: kitti",
                    f"raw_data_root: {raw_data_root}",
                    "_target_size: [504, 280]",
                    f"_sequences: [\"{seq}\"]",
                    "sampling:",
                    "  strategy: sequence",
                    f"  start_idx: {trace_start}",
                    f"  end_idx: {trace_end}",
                    "  stride: 1",
                    "",
                ]
            ),
        )
        notrace_config = CONFIG_ROOT / f"kitti_lingbot_v107tf_stage3_{target_id}_notrace.yaml"
        trace_config = CONFIG_ROOT / f"kitti_lingbot_v107tf_stage3_{target_id}_trace.yaml"
        write_text(notrace_config, run_config_yaml(dataset, notrace_method))
        write_text(trace_config, run_config_yaml(dataset, trace_method))
        trace_file = RAW_TRACE / f"{dataset}_{seq}_{trace_method}.jsonl"
        trace_env = (
            f"ACL2_V107_CACHE_TRACE_FILE={trace_file} "
            f"ACL2_V107_CACHE_TRACE_RUN_ID={target_id} "
            f"ACL2_V107_CACHE_TRACE_CASE={dataset}/{seq}/{trace_method} "
            f"ACL2_V107_CACHE_TRACE_DATASET={dataset} "
            f"ACL2_V107_CACHE_TRACE_SEQ={seq} "
            f"ACL2_V107_CACHE_TRACE_METHOD={trace_method} "
            f"ACL2_V107_CACHE_TRACE_WINDOW_ID={source['window_index']} "
            f"ACL2_V107_CACHE_TRACE_FRAME_START_IDX={trace_start} "
            f"ACL2_V107_CACHE_TRACE_GLOBAL_IDXS={source['trace_global_idxs']} "
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
            manifest_rows.append({
                "schema": "acl2_v107tf_stage3_safe96_length_control_manifest_row_v1",
                "target_id": target_id,
                "target_kind": target["target_kind"],
                "run_name": f"kitti_lingbot_v107tf_stage3_{target_id}_{phase}",
                "phase": phase,
                "cwd": str(BENCHMARK),
                "config": str(config.resolve()),
                "dataset": dataset,
                "seq": seq,
                "method": method,
                "gpu": gpu,
                "trace_file": trace_file_value,
                "trace_global_idxs": source["trace_global_idxs"],
                "trace_start_idx": trace_start,
                "trace_end_idx_exclusive": trace_end,
                "window_index": source["window_index"],
                "command": command,
                "status": "planned",
            })

    write_csv(OUT / "target_manifest.csv", target_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_csv(OUT / "skipped_targets.csv", skipped_rows)
    summary = {
        "schema": "acl2_v107tf_stage3_safe96_length_control_config_summary_v1",
        "source_target_manifest": rel(SOURCE_TARGETS),
        "target_count": len(target_rows),
        "skipped_count": len(skipped_rows),
        "manifest_rows": len(manifest_rows),
        "workspace": rel(WORKSPACE),
        "raw_trace_dir": rel(RAW_TRACE),
        "target_manifest": rel(OUT / "target_manifest.csv"),
        "run_manifest": rel(OUT / "run_manifest.csv"),
        "notrace_method": notrace_method,
        "trace_method": trace_method,
        "use_sdpa": use_sdpa,
        "control_reason": "safe_good_64F_to_96F_trace_length_matching_for_cache_budget_operation_confound",
    }
    write_text(OUT / "config_generation_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
