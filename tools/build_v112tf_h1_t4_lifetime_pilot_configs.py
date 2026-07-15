#!/usr/bin/env python3
"""Generate ACL2 v112TF H1/T4 semantic lifetime pilot/full configs."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
MODE = os.environ.get("ACL2_V112TF_H1_T4_MODE", "pilot").strip().lower()
if MODE not in {"pilot", "full"}:
    raise ValueError("ACL2_V112TF_H1_T4_MODE must be 'pilot' or 'full'")
RUN_LABEL = "full" if MODE == "full" else "pilot"
PILOT = RESULT_ROOT / (
    "stage6_h1_t4_full_validation_00_01_02_05"
    if MODE == "full"
    else "stage5_h1_t4_semantic_lifetime_pilot_00_02"
)
CONFIG_ROOT = PILOT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = PILOT / "workspace"
RAW_ACTION = PILOT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
SOURCE = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality/stage4_full_00_01_02_05_validation/candidate_policy_rows.csv"
CUE = RESULT_ROOT / "stage2_memory_specific_cue_bank/anchor_memory_cue_rows.csv"
SEQUENCES = ("00", "01", "02", "05") if MODE == "full" else ("00", "02")
SOURCE_POLICY_ID = "B1_semantic_plus_internal"


POLICIES = [
    {
        "policy_id": "H1_no_action_mask_all1_default_off",
        "policy_family": "default_off_parity",
        "mask_mode": "all1",
        "source_selected": False,
    },
    {
        "policy_id": "H1_B1_hard_lifetime_zero",
        "policy_family": "hard_lifetime_zero",
        "mask_mode": "zero_all",
        "source_selected": True,
    },
    {
        "policy_id": "H1_semantic_lifetime_soft_raw",
        "policy_family": "semantic_lifetime_soft",
        "mask_mode": "stage2_soft_g",
        "source_selected": True,
    },
    {
        "policy_id": "T4_role_adaptive_policy",
        "policy_family": "role_adaptive_policy",
        "mask_mode": "role_adaptive",
        "source_selected": True,
    },
    {
        "policy_id": "T4_boundary_register_downweight",
        "policy_family": "boundary_register_downweight",
        "mask_mode": "boundary_register_downweight",
        "source_selected": True,
    },
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


def parse_indices(raw: str) -> list[int]:
    return [int(float(part)) for part in str(raw).split(";") if str(part).strip()]


def source_rows() -> dict[str, dict[str, str]]:
    return {
        row["seq"]: row
        for row in read_csv(SOURCE)
        if row.get("policy_id") == SOURCE_POLICY_ID and row.get("seq") in SEQUENCES
    }


def cue_rows() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(CUE):
        seq = row.get("seq", "")
        if seq not in SEQUENCES:
            continue
        frame = int(float(row.get("frame_id", 0)))
        out[(seq, frame)] = row
    return out


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mask_for(mode: str, cue: dict[str, Any]) -> list[float]:
    if mode == "all1":
        return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if mode == "zero_all":
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if mode == "stage2_soft_g":
        g_camera = clamp(safe_float(cue.get("g_camera"), 1.0), 0.0, 1.0)
        g_register = clamp(safe_float(cue.get("g_register"), 1.0), 0.0, 1.0)
        g_anchor = clamp(safe_float(cue.get("g_anchor"), 1.0), 0.0, 1.0)
        return [g_camera, g_register, g_register, g_register, g_register, g_anchor]
    dynamic = safe_float(cue.get("dynamic_mass"), 0.0)
    boundary = safe_float(cue.get("boundary_mass"), 0.0)
    weak = safe_float(cue.get("weak_context_mass"), 0.0)
    stable = safe_float(cue.get("stable_landmark_mass"), 0.0)
    if mode == "boundary_register_downweight":
        if boundary >= 0.04:
            return [0.5, 0.25, 0.25, 0.25, 0.25, 1.0]
        return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if mode == "role_adaptive":
        if dynamic >= 0.02:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        if boundary >= 0.04:
            return [0.5, 0.25, 0.25, 0.25, 0.25, 1.0]
        if weak >= 0.60:
            return [0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
        if stable >= 0.18:
            return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    raise ValueError(f"unknown mask mode: {mode}")


def mask_text(mask: list[float]) -> str:
    return ",".join(f"{value:.6g}" for value in mask)


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    indices: list[int],
    mask_map: dict[int, list[float]],
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
            "_num_scale_frames: 8",
            "_max_frame_num: 1024",
            "_kv_cache_sliding_window: 64",
            "_kv_cache_scale_frames: 8",
            "_auto_keyframe_threshold: 320",
            "_area_budget: 255000",
            "_align: 14",
            "_mode: streaming",
            "_keyframe_interval: auto",
            f"_force_non_keyframe_indices: {json.dumps(indices)}",
            f"_stage4_action_label: {action_label}",
            "_stage4_action_mode: context_token_mask",
            f"_stage4_context_token_mask_map: {json.dumps({str(k): v for k, v in mask_map.items()}, sort_keys=True)}",
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
    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    source_by_seq = source_rows()
    cue_by_key = cue_rows()
    missing_source = [seq for seq in SEQUENCES if seq not in source_by_seq]
    gpu_cycle = ["0", "1", "2", "3"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    PILOT.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v112tf_h1_t4_{RUN_LABEL}_{seq}"
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
    missing_cue: list[dict[str, Any]] = []
    row_index = 0

    for policy in POLICIES:
        policy_id = policy["policy_id"]
        for seq in SEQUENCES:
            source = source_by_seq.get(seq, {})
            selected = parse_indices(source.get("selected_global_frame_indices", ""))
            if not policy["source_selected"]:
                selected = []
            mask_map: dict[int, list[float]] = {}
            for frame in selected:
                cue = cue_by_key.get((seq, frame), {})
                if not cue:
                    missing_cue.append({"seq": seq, "frame": frame, "policy_id": policy_id})
                    cue = {}
                mask = mask_for(policy["mask_mode"], cue)
                mask_map[frame] = mask
                frame_rows.append(
                    {
                        "schema": "acl2_v112tf_h1_t4_lifetime_frame_mask_row_v1",
                        "policy_id": policy_id,
                        "policy_family": policy["policy_family"],
                        "seq": seq,
                        "frame": frame,
                        "mask_mode": policy["mask_mode"],
                        "dynamic_mass": cue.get("dynamic_mass", ""),
                        "boundary_mass": cue.get("boundary_mass", ""),
                        "weak_context_mass": cue.get("weak_context_mass", ""),
                        "stable_landmark_mass": cue.get("stable_landmark_mass", ""),
                        "g_camera_stage2": cue.get("g_camera", ""),
                        "g_register_stage2": cue.get("g_register", ""),
                        "g_anchor_stage2": cue.get("g_anchor", ""),
                        "token_type_mask": mask_text(mask),
                        "camera_weight": mask[0],
                        "register_weight": mask[1],
                        "anchor_weight": mask[5],
                    }
                )

            dataset = f"kitti_v112tf_h1_t4_{RUN_LABEL}_{seq}"
            method = f"lingbot_map_v112tf_h1_t4_{policy_id}_{seq}"
            action_label = f"v112tf_h1_t4_{policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v112tf_h1_t4_{policy_id}_{seq}.yaml"
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
                    indices=selected,
                    mask_map=mask_map,
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            selected_string = ";".join(str(x) for x in selected)
            mask_values = list(mask_map.values())
            cam_mean = sum(mask[0] for mask in mask_values) / len(mask_values) if mask_values else ""
            reg_mean = sum(mask[1] for mask in mask_values) / len(mask_values) if mask_values else ""
            anchor_mean = sum(mask[5] for mask in mask_values) / len(mask_values) if mask_values else ""
            row = {
                "schema": "acl2_v112tf_h1_t4_lifetime_policy_row_v1",
                "surface_id": "T",
                "candidate_id": "H1_T4",
                "policy_id": policy_id,
                "policy_family": policy["policy_family"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage2_action_mode": "context_token_mask",
                "stage4_action_mode": "context_token_mask",
                "source_policy_id": SOURCE_POLICY_ID,
                "selected_count": len(selected),
                "selected_global_frame_indices": selected_string,
                "expected_action_field": "forced_context_only",
                "mask_mode": policy["mask_mode"],
                "camera_token_weight_mean": cam_mean,
                "register_token_weight_mean": reg_mean,
                "anchor_token_weight_mean": anchor_mean,
                "runtime_boundary": f"v112 H1/T4 {RUN_LABEL} uses existing compact context-token mask hook on B1-selected keyframes.",
                "claim_boundary": (
                    "00/01/02/05 full validation; semantic-aware claim still requires matched controls."
                    if MODE == "full"
                    else "00/02 pilot only; no full KITTI or semantic-aware claim allowed from this run alone."
                ),
                "config": str(config.resolve()),
                "method_config": str(method_path.resolve()),
                "action_file": str(action_file.resolve()),
                "gpu": gpu,
            }
            config_rows.append(row)

            prefix = v108.command_prefix(conda_path, pythonpath, gpu)
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v112tf_h1_t4_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v112tf_h1_t4_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"h1_t4_{RUN_LABEL}_{seq}",
                    "target_kind": f"{RUN_LABEL}_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "deduplicated_prepare_once_per_dataset_seq",
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
                f"ACL2_V108_STAGE4_SURFACE_ID=T "
                f"ACL2_V112TF_H1_T4_POLICY_ID={policy_id}"
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
                        "schema": "acl2_v112tf_h1_t4_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v112tf_h1_t4_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"h1_t4_{RUN_LABEL}_{seq}",
                        "target_kind": f"{RUN_LABEL}_full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": "context_token_mask",
                        "selector": "v110_B1_selected_keyframes_with_v112_lifetime_mask",
                        "selected_count": len(selected),
                        "force_non_keyframe_indices": selected_string,
                        "context_token_type_mask": "per_frame_mask_map",
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
        "schema": f"acl2_v112tf_h1_t4_lifetime_{RUN_LABEL}_config_summary_v1",
        "mode": MODE,
        "config_ready": not missing_source and not missing_cue,
        "blocker": "missing_source_or_stage2_cue_rows" if (missing_source or missing_cue) else "",
        "missing_source_sequences": missing_source,
        "missing_cue_rows": missing_cue,
        "source_policy_id": SOURCE_POLICY_ID,
        "sequences": list(SEQUENCES),
        "policy_ids": [policy["policy_id"] for policy in POLICIES],
        "config_rows": len(config_rows),
        "prepare_manifest_rows": len(prepare_rows_by_seq),
        "run_worker_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "gpu_cycle": gpu_cycle,
        "outputs": {
            "action_config_rows": rel(PILOT / "action_config_rows.csv"),
            "frame_mask_rows": rel(PILOT / "frame_mask_rows.csv"),
            "run_manifest": rel(PILOT / "run_manifest.csv"),
            "summary": rel(PILOT / "h1_t4_config_generation_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(PILOT / "action_config_rows.csv", config_rows)
    write_csv(PILOT / "frame_mask_rows.csv", frame_rows)
    write_csv(PILOT / "run_manifest.csv", manifest_rows)
    write_json(PILOT / "h1_t4_config_generation_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    return 0 if summary["config_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
