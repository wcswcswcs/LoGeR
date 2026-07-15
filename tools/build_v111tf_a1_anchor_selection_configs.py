#!/usr/bin/env python3
"""Generate ACL2 v111TF A1 delayed semantic anchor-frame selection configs."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
A1 = RESULT_ROOT / "batch_a_a1_anchor_selection"
CONFIG_ROOT = A1 / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = A1 / "workspace"
RAW_ACTION = A1 / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
SEMANTIC = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage2_semantic_cue_bank/frame_semantic_summary.csv"
SEQUENCES = ("00", "01", "02", "05")
NUM_ANCHOR = 8

LAMBDA_D = 1.0
LAMBDA_B = 0.7
LAMBDA_W = 0.3

BASE_POLICIES = [
    {"policy_id": "A1_default_first_n", "policy_family": "default_first_n", "mode": "default", "M": 8},
    {"policy_id": "A1_topQ_from_first16", "policy_family": "topQ", "mode": "topQ", "M": 16},
    {"policy_id": "A1_topQ_from_first32", "policy_family": "topQ", "mode": "topQ", "M": 32},
    {"policy_id": "A1_topQ_from_first64", "policy_family": "topQ", "mode": "topQ", "M": 64},
    {"policy_id": "A1_low_dynamic_from_first32", "policy_family": "low_dynamic", "mode": "low_dynamic", "M": 32},
    {"policy_id": "A1_high_stable_from_first32", "policy_family": "high_stable", "mode": "high_stable", "M": 32},
    {"policy_id": "A1_low_risk_reverse_from_first32", "policy_family": "low_risk_reverse", "mode": "low_risk_reverse", "M": 32},
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def semantic_by_key() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(SEMANTIC):
        seq = row.get("seq_id", "")
        if seq not in SEQUENCES:
            continue
        frame = int(float(row.get("frame_id", 0)))
        row = dict(row)
        stable = safe_float(row.get("stable_structure_mass"))
        dynamic = safe_float(row.get("dynamic_mass"))
        boundary = safe_float(row.get("boundary_mass"))
        weak = safe_float(row.get("weak_context_mass"))
        row["a1_anchor_quality"] = stable - LAMBDA_D * dynamic - LAMBDA_B * boundary - LAMBDA_W * weak
        row["a1_risk"] = dynamic + LAMBDA_B * boundary + LAMBDA_W * weak
        out[(seq, frame)] = row
    return out


def rank_frames(seq: str, M: int, sem: dict[tuple[str, int], dict[str, Any]], key: str, reverse: bool) -> list[int]:
    candidates: list[tuple[float, int]] = []
    for frame in range(M):
        row = sem.get((seq, frame), {})
        candidates.append((safe_float(row.get(key)), frame))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=reverse)
    return sorted(frame for _score, frame in candidates[:NUM_ANCHOR])


def select_indices(policy: dict[str, Any], seq: str, sem: dict[tuple[str, int], dict[str, Any]]) -> list[int]:
    mode = str(policy["mode"])
    M = int(policy["M"])
    if mode == "default":
        return list(range(NUM_ANCHOR))
    if mode == "topQ":
        return rank_frames(seq, M, sem, "a1_anchor_quality", True)
    if mode == "low_dynamic":
        return rank_frames(seq, M, sem, "dynamic_mass", False)
    if mode == "high_stable":
        return rank_frames(seq, M, sem, "stable_structure_mass", True)
    if mode == "low_risk_reverse":
        low_risk = rank_frames(seq, M, sem, "a1_risk", False)
        return sorted(reversed(low_risk))
    if mode.startswith("random_same_firstM_seed"):
        seed = int(mode.rsplit("seed", 1)[1])
        rng = random.Random(seed + int(seq))
        return sorted(rng.sample(list(range(M)), NUM_ANCHOR))
    raise ValueError(f"unknown A1 mode: {mode}")


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    scale_indices: list[int],
) -> str:
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
            f"_num_scale_frames: {NUM_ANCHOR}",
            "_max_frame_num: 1024",
            "_kv_cache_sliding_window: 64",
            f"_kv_cache_scale_frames: {NUM_ANCHOR}",
            "_auto_keyframe_threshold: 320",
            "_area_budget: 255000",
            "_align: 14",
            "_mode: streaming",
            "_keyframe_interval: auto",
            f"_stage4_action_label: {action_label}",
            "_stage4_action_mode: anchor_scale_frame_indices",
            f"_stage4_scale_frame_indices: {json.dumps(scale_indices)}",
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


def main() -> int:
    include_random = "--include-random" in sys.argv[1:]
    policies = list(BASE_POLICIES)
    if include_random:
        for seed in range(21):
            policies.append(
                {
                    "policy_id": f"A1_random_same_first32_seed{seed}",
                    "policy_family": "random_same_first32",
                    "mode": f"random_same_firstM_seed{seed}",
                    "M": 32,
                }
            )

    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    sem = semantic_by_key()
    gpu_cycle = ["0", "1", "2", "3", "4"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    A1.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v111tf_a1_fullseq_{seq}"
        write_text(
            DATASET_DIR / f"{dataset}.yaml",
            "\n".join(
                [
                    "dataset: kitti",
                    f"raw_data_root: {raw_data_root}",
                    "_target_size: [504, 280]",
                    f"_sequences: [\"{seq}\"]",
                    "",
                ]
            ),
        )

    config_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    row_index = 0
    missing_semantic: list[dict[str, Any]] = []

    for policy in policies:
        policy_id = str(policy["policy_id"])
        for seq in SEQUENCES:
            scale_indices = select_indices(policy, seq, sem)
            M = int(policy["M"])
            for frame in range(M):
                if (seq, frame) not in sem:
                    missing_semantic.append({"seq": seq, "frame": frame, "policy_id": policy_id})
            selected_metrics = [sem.get((seq, frame), {}) for frame in scale_indices]
            dataset = f"kitti_v111tf_a1_fullseq_{seq}"
            method = f"lingbot_map_v111tf_a1_{policy_id}_{seq}"
            action_label = f"v111tf_a1_{policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v111tf_a1_{policy_id}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = gpu_cycle[row_index % len(gpu_cycle)]
            row_index += 1

            write_text(
                method_path,
                method_yaml(
                    checkpoint=checkpoint,
                    env_name=env_name,
                    use_sdpa=use_sdpa,
                    action_label=action_label,
                    scale_indices=scale_indices,
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            selected_string = ";".join(str(x) for x in scale_indices)
            row = {
                "schema": "acl2_v111tf_a1_anchor_selection_policy_row_v1",
                "surface_id": "A",
                "candidate_id": "A1",
                "policy_id": policy_id,
                "policy_family": policy["policy_family"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage2_action_mode": "anchor_scale_frame_indices",
                "stage4_action_mode": "anchor_scale_frame_indices",
                "expected_action_field": "anchor_scale_frame",
                "M": M,
                "num_anchor": NUM_ANCHOR,
                "selected_global_frame_indices": selected_string,
                "scale_frame_indices": selected_string,
                "latency_frames": max(scale_indices) + 1,
                "latency_policy": "delayed_initialization" if max(scale_indices) >= NUM_ANCHOR else "zero_delay_default_equivalent",
                "anchor_quality_mean": sum(safe_float(row.get("a1_anchor_quality")) for row in selected_metrics) / len(selected_metrics),
                "dynamic_mass_mean": sum(safe_float(row.get("dynamic_mass")) for row in selected_metrics) / len(selected_metrics),
                "stable_structure_mass_mean": sum(safe_float(row.get("stable_structure_mass")) for row in selected_metrics) / len(selected_metrics),
                "boundary_mass_mean": sum(safe_float(row.get("boundary_mass")) for row in selected_metrics) / len(selected_metrics),
                "weak_context_mass_mean": sum(safe_float(row.get("weak_context_mass")) for row in selected_metrics) / len(selected_metrics),
                "runtime_boundary": "A1 selects scale/anchor initialization frames; it restores output order after delayed initialization.",
                "claim_boundary": "A1 semantic anchor selection needs four-sequence geometry gate and random P95 controls before semantic causality claim.",
                "config": str(config.resolve()),
                "method_config": str(method_path.resolve()),
                "action_file": str(action_file.resolve()),
                "gpu": gpu,
            }
            config_rows.append(row)
            for frame in scale_indices:
                sem_row = sem.get((seq, frame), {})
                frame_rows.append(
                    {
                        "schema": "acl2_v111tf_a1_selected_anchor_frame_row_v1",
                        "policy_id": policy_id,
                        "policy_family": policy["policy_family"],
                        "seq": seq,
                        "frame": frame,
                        "M": M,
                        "a1_anchor_quality": sem_row.get("a1_anchor_quality", ""),
                        "a1_risk": sem_row.get("a1_risk", ""),
                        "dynamic_mass": sem_row.get("dynamic_mass", ""),
                        "stable_structure_mass": sem_row.get("stable_structure_mass", ""),
                        "boundary_mass": sem_row.get("boundary_mass", ""),
                        "weak_context_mass": sem_row.get("weak_context_mass", ""),
                    }
                )

            prefix = v108.command_prefix(conda_path, pythonpath, gpu)
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v111tf_a1_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v111tf_a1_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"a1_fullseq_{seq}",
                    "target_kind": "full_sequence_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "deduplicated_prepare_once_per_dataset_seq_to_avoid_parallel_rmtree_race",
                    "selected_count": 0,
                    "gpu": gpu,
                    "cwd": str(BENCHMARK.resolve()),
                    "config": str(config.resolve()),
                    "trace_file": "",
                    "action_file": "",
                    "command": prepare_command,
                    "status": "planned",
                },
            )
            action_env = (
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method} "
                f"ACL2_V108_STAGE4_POLICY_ID={policy_id} "
                f"ACL2_V108_STAGE4_SURFACE_ID=A "
                f"ACL2_V111TF_A1_POLICY_ID={policy_id}"
            )
            commands = {
                "run_worker": (
                    f"{prefix} {action_env} {conda_path} run -n {env_name} "
                    f"python run_worker.py --config {config.resolve()} --method {method} "
                    f"--dataset {dataset} --scene {seq} --force"
                ),
                "evaluate": (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python evaluate.py --config {config.resolve()} --force"
                ),
                "report": (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python report.py --workspace {WORKSPACE.resolve()} --dataset {dataset}"
                ),
            }
            for phase, command in commands.items():
                manifest_rows.append(
                    {
                        "schema": "acl2_v111tf_a1_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v111tf_a1_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"a1_fullseq_{seq}",
                        "target_kind": "full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": "anchor_scale_frame_indices",
                        "selector": "first_M_anchor_semantic_selection",
                        "selected_count": len(scale_indices),
                        "force_non_keyframe_indices": "",
                        "scale_frame_indices": selected_string,
                        "gpu": gpu,
                        "cwd": str(BENCHMARK.resolve()),
                        "config": str(config.resolve()),
                        "trace_file": "",
                        "action_file": str(action_file.resolve()),
                        "command": command,
                        "status": "planned",
                    }
                )

    manifest_rows = [prepare_rows_by_seq[seq] for seq in SEQUENCES] + manifest_rows
    summary = {
        "schema": "acl2_v111tf_a1_anchor_selection_config_summary_v1",
        "a1_config_ready": not missing_semantic,
        "blocker": "missing_semantic_rows" if missing_semantic else "",
        "missing_semantic_rows": missing_semantic,
        "sequences": list(SEQUENCES),
        "num_anchor": NUM_ANCHOR,
        "policy_count": len(policies),
        "policy_ids": [str(policy["policy_id"]) for policy in policies],
        "include_random_controls": include_random,
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_manifest_rows": len(prepare_rows_by_seq),
        "run_worker_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "quality_formula": {
            "Q_anchor": "stable_structure_mass - lambda_d*dynamic_mass - lambda_b*boundary_mass - lambda_w*weak_context_mass",
            "lambda_d": LAMBDA_D,
            "lambda_b": LAMBDA_B,
            "lambda_w": LAMBDA_W,
        },
        "outputs": {
            "action_config_rows": rel(A1 / "action_config_rows.csv"),
            "selected_anchor_frame_rows": rel(A1 / "selected_anchor_frame_rows.csv"),
            "run_manifest": rel(A1 / "run_manifest.csv"),
            "summary": rel(A1 / "a1_config_generation_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(A1 / "action_config_rows.csv", config_rows)
    write_csv(A1 / "selected_anchor_frame_rows.csv", frame_rows)
    write_csv(A1 / "run_manifest.csv", manifest_rows)
    write_json(A1 / "a1_config_generation_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    return 0 if summary["a1_config_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
