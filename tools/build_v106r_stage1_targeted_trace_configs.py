#!/usr/bin/env python3
"""Generate ACL2 v106R Stage1 targeted LingBot trace configs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106R = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control"
V105_STAGE0 = V105 / "stage0_repo_env_audit/stage0_summary.json"
LOCAL_WINDOWS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/local_window_rows.csv"
CONFIG_ROOT = V106R / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
STAGE1_TRACE = V106R / "stage1_memory_operation_map/targeted_trace"
WORKSPACE = STAGE1_TRACE / "workspace"
RAW_TRACE = STAGE1_TRACE / "raw_trace"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

SEQUENCES = ["00", "01", "02", "05"]
HEAD_IDXS = ",".join(str(i) for i in range(16))
CONTEXT_BEFORE = 32
TARGET_WINDOW = 32
TRACE_GLOBAL_IDXS = "0,5,11,17,23"


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
    path.write_text(text, encoding="utf-8")


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def select_targets() -> list[dict[str, Any]]:
    rows_by_seq: dict[str, list[dict[str, Any]]] = {seq: [] for seq in SEQUENCES}
    for row in read_csv(LOCAL_WINDOWS):
        seq = row["seq"]
        if seq not in rows_by_seq:
            continue
        if row.get("handoff_transfer_penalty", "") in {"", None}:
            continue
        item = dict(row)
        item["frame_start_i"] = int(float(row["frame_start"]))
        item["frame_end_i"] = int(float(row["frame_end"]))
        item["window_index_i"] = int(float(row["window_index"]))
        item["handoff_transfer_penalty_f"] = fnum(row, "handoff_transfer_penalty")
        item["local_sim3_ate_rmse_m_f"] = fnum(row, "local_sim3_ate_rmse_m")
        item["adjacent_log_scale_jump_f"] = fnum(row, "adjacent_log_scale_jump")
        rows_by_seq[seq].append(item)

    targets: list[dict[str, Any]] = []
    for seq, rows in rows_by_seq.items():
        if not rows:
            continue
        local_median = statistics.median(row["local_sim3_ate_rmse_m_f"] for row in rows)
        adjacent_median = statistics.median(row["adjacent_log_scale_jump_f"] for row in rows)
        adjacent_q75 = statistics.quantiles(
            [row["adjacent_log_scale_jump_f"] for row in rows],
            n=4,
        )[2]
        high = max(rows, key=lambda row: row["handoff_transfer_penalty_f"])
        safe_pool = [
            row for row in rows
            if row["local_sim3_ate_rmse_m_f"] <= local_median
            and row["handoff_transfer_penalty_f"] > 0
            and row["adjacent_log_scale_jump_f"] <= adjacent_median
        ] or rows
        safe_pool_rule = "local_le_median_and_handoff_positive_and_adjacent_le_median"
        if safe_pool is rows:
            safe_pool = [
                row for row in rows
                if row["local_sim3_ate_rmse_m_f"] <= local_median
                and row["handoff_transfer_penalty_f"] > 0
                and row["adjacent_log_scale_jump_f"] <= adjacent_q75
            ] or rows
            safe_pool_rule = "fallback_local_le_median_and_handoff_positive_and_adjacent_le_q75"
        if safe_pool is rows:
            safe_pool_rule = "fallback_all_rows"
        safe = min(
            safe_pool,
            key=lambda row: (
                row["handoff_transfer_penalty_f"],
                row["adjacent_log_scale_jump_f"],
                row["local_sim3_ate_rmse_m_f"],
            ),
        )
        for kind, row in [("high_l3", high), ("safe_good_low_drift", safe)]:
            trace_start = max(0, row["frame_start_i"] - CONTEXT_BEFORE)
            trace_end_exclusive = row["frame_end_i"] + 1
            target_start_rel = row["frame_start_i"] - trace_start
            target_end_rel = row["frame_end_i"] - trace_start
            target_sample_positions = [
                target_start_rel,
                target_start_rel + 7,
                target_start_rel + 15,
                target_start_rel + 23,
                target_end_rel,
            ]
            frame_count = trace_end_exclusive - trace_start
            target_sample_positions = sorted({idx for idx in target_sample_positions if 0 <= idx < frame_count})
            target_id = f"seq{seq}_{kind}_w{row['window_index_i']:04d}"
            targets.append(
                {
                    "schema": "acl2_v106r_stage1_targeted_trace_target_v1",
                    "target_id": target_id,
                    "seq": seq,
                    "target_kind": kind,
                    "window_index": row["window_index_i"],
                    "target_frame_start": row["frame_start_i"],
                    "target_frame_end": row["frame_end_i"],
                    "trace_start_idx": trace_start,
                    "trace_end_idx_exclusive": trace_end_exclusive,
                    "trace_frame_count": frame_count,
                    "target_sample_positions": ",".join(str(idx) for idx in target_sample_positions),
                    "trace_global_idxs": TRACE_GLOBAL_IDXS,
                    "handoff_transfer_penalty": row["handoff_transfer_penalty_f"],
                    "adjacent_log_scale_jump": row["adjacent_log_scale_jump_f"],
                    "local_sim3_ate_rmse_m": row["local_sim3_ate_rmse_m_f"],
                    "safe_local_median": local_median,
                    "safe_adjacent_median": adjacent_median,
                    "safe_adjacent_q75": adjacent_q75,
                    "safe_pool_rule": safe_pool_rule if kind == "safe_good_low_drift" else "",
                    "selection_rule": "per_seq_top_handoff_for_high_l3_and_min_handoff_among_local_median_and_low_adjacent_safe_pool",
                }
            )
    return targets


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
            "_stage4_action_label: v106r_targeted_trace_only",
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


def build() -> dict[str, Any]:
    stage0 = json.loads(V105_STAGE0.read_text(encoding="utf-8"))
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    raw_data_root = stage0["kitti"]["resolved_kitti_root"]
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"]["flashinfer_available_in_recommended_env"])

    STAGE1_TRACE.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    targets = select_targets()
    write_csv(STAGE1_TRACE / "target_manifest.csv", targets)

    notrace_method = "lingbot_map_v106r_stage1_targeted_notrace"
    trace_method = "lingbot_map_v106r_stage1_targeted_trace"
    write_text(METHOD_DIR / f"{notrace_method}.yaml", method_yaml(checkpoint, env_name, use_sdpa))
    write_text(METHOD_DIR / f"{trace_method}.yaml", method_yaml(checkpoint, env_name, use_sdpa))

    env_prefix_base = (
        "PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={pythonpath}"
    )
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        seq = str(target["seq"])
        dataset = f"kitti_v106r_{target['target_id']}_trace{target['trace_frame_count']}"
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
        notrace_config = CONFIG_ROOT / f"kitti_lingbot_v106r_stage1_{target['target_id']}_notrace.yaml"
        trace_config = CONFIG_ROOT / f"kitti_lingbot_v106r_stage1_{target['target_id']}_trace.yaml"
        write_text(notrace_config, run_config_yaml(dataset, notrace_method))
        write_text(trace_config, run_config_yaml(dataset, trace_method))
        trace_file = RAW_TRACE / f"{dataset}_{seq}_{trace_method}.jsonl"
        trace_env = (
            f"ACL2_V105_GCA_TRACE_FILE={trace_file} "
            f"ACL2_V105_GCA_TRACE_CASE={dataset}/{seq}/{trace_method} "
            f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
            f"ACL2_V105_GCA_TRACE_SEQ={seq} "
            f"ACL2_V105_GCA_TRACE_METHOD={trace_method} "
            f"ACL2_V105_GCA_TRACE_GLOBAL_IDXS={target['trace_global_idxs']} "
            f"ACL2_V105_GCA_TRACE_HEAD_IDXS={HEAD_IDXS} "
            "ACL2_V105_GCA_TRACE_TOPK=5 "
            "ACL2_V105_GCA_TRACE_MAX_ROWS=240000"
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
                    "schema": "acl2_v106r_stage1_targeted_trace_manifest_row_v1",
                    "target_id": target["target_id"],
                    "target_kind": target["target_kind"],
                    "run_name": f"kitti_lingbot_v106r_stage1_{target['target_id']}_{phase}",
                    "phase": phase,
                    "cwd": str(BENCHMARK),
                    "config": str(config.resolve()),
                    "dataset": dataset,
                    "seq": seq,
                    "method": method,
                    "gpu": gpu,
                    "trace_file": trace_file_value,
                    "trace_global_idxs": target["trace_global_idxs"],
                    "command": command,
                    "status": "planned",
                }
            )

    write_csv(STAGE1_TRACE / "run_manifest.csv", rows)
    summary = {
        "schema": "acl2_v106r_stage1_targeted_trace_config_summary_v1",
        "target_count": len(targets),
        "manifest_rows": len(rows),
        "sequences": SEQUENCES,
        "head_idxs": HEAD_IDXS,
        "workspace": str(WORKSPACE.resolve()),
        "raw_trace_dir": str(RAW_TRACE.resolve()),
        "target_manifest": str((STAGE1_TRACE / "target_manifest.csv").relative_to(ROOT)),
        "run_manifest": str((STAGE1_TRACE / "run_manifest.csv").relative_to(ROOT)),
        "notrace_method": notrace_method,
        "trace_method": trace_method,
        "use_sdpa": use_sdpa,
        "trace_global_idx_policy": "lingbot_attention_global_block_indices_0_5_11_17_23; target frame positions are recorded separately as target_sample_positions",
    }
    write_text(
        STAGE1_TRACE / "config_generation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
