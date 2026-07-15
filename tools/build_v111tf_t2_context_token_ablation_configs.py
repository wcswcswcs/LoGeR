#!/usr/bin/env python3
"""Generate ACL2 v111TF T2 trajectory context-token ablation configs."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
T2 = RESULT_ROOT / "batch_t_t2_context_token_ablation"
CONFIG_ROOT = T2 / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = T2 / "workspace"
RAW_ACTION = T2 / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
SOURCE = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality/stage4_full_00_01_02_05_validation/candidate_policy_rows.csv"
SEQUENCES = ("00", "01", "02", "05")
SOURCE_POLICY_ID = "B1_semantic_plus_internal"

POLICIES = [
    {
        "policy_id": "T2_no_action_mask_all1_default_off",
        "policy_family": "no_action_parity",
        "action_mode": "context_token_mask",
        "mask": [1, 1, 1, 1, 1, 1],
        "source_selected": False,
        "claim_role": "default_off_no_action_parity",
    },
    {
        "policy_id": "T2_default_context_tokens",
        "policy_family": "default_context_tokens",
        "action_mode": "context_token_mask",
        "mask": [1, 1, 1, 1, 1, 1],
        "source_selected": True,
        "claim_role": "new_hook_all_context_tokens",
    },
    {
        "policy_id": "T2_default_context_tokens_legacy_context_only",
        "policy_family": "legacy_context_only_parity",
        "action_mode": "context_only_special",
        "mask": "",
        "source_selected": True,
        "claim_role": "legacy_context_only_all_special_reference",
    },
    {
        "policy_id": "T2_camera_only",
        "policy_family": "camera_only",
        "action_mode": "context_token_mask",
        "mask": [1, 0, 0, 0, 0, 0],
        "source_selected": True,
        "claim_role": "camera_special_only",
    },
    {
        "policy_id": "T2_register_only",
        "policy_family": "register_only",
        "action_mode": "context_token_mask",
        "mask": [0, 1, 1, 1, 1, 0],
        "source_selected": True,
        "claim_role": "register_special_only",
    },
    {
        "policy_id": "T2_anchor_only",
        "policy_family": "anchor_only",
        "action_mode": "context_token_mask",
        "mask": [0, 0, 0, 0, 0, 1],
        "source_selected": True,
        "claim_role": "new_hook_anchor_special_only",
    },
    {
        "policy_id": "T2_camera_plus_anchor",
        "policy_family": "camera_plus_anchor",
        "action_mode": "context_token_mask",
        "mask": [1, 0, 0, 0, 0, 1],
        "source_selected": True,
        "claim_role": "camera_and_anchor_special",
    },
    {
        "policy_id": "T2_register_plus_anchor",
        "policy_family": "register_plus_anchor",
        "action_mode": "context_token_mask",
        "mask": [0, 1, 1, 1, 1, 1],
        "source_selected": True,
        "claim_role": "register_and_anchor_special",
    },
    {
        "policy_id": "T2_camera_plus_register",
        "policy_family": "camera_plus_register",
        "action_mode": "context_token_mask",
        "mask": [1, 1, 1, 1, 1, 0],
        "source_selected": True,
        "claim_role": "camera_and_register_special",
    },
    {
        "policy_id": "T2_zero_all_context_for_high_risk",
        "policy_family": "zero_all_context",
        "action_mode": "context_token_mask",
        "mask": [0, 0, 0, 0, 0, 0],
        "source_selected": True,
        "claim_role": "zero_compact_context_for_selected_high_risk",
    },
    {
        "policy_id": "T2_anchor_only_for_high_risk_else_default",
        "policy_family": "legacy_anchor_only",
        "action_mode": "anchor_special_only",
        "mask": [0, 0, 0, 0, 0, 1],
        "source_selected": True,
        "claim_role": "legacy_anchor_special_only_reference",
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


def parse_indices(raw: str) -> list[int]:
    return [int(float(part)) for part in str(raw).split(";") if str(part).strip()]


def source_rows() -> dict[str, dict[str, str]]:
    rows = {
        row["seq"]: row
        for row in read_csv(SOURCE)
        if row.get("policy_id") == SOURCE_POLICY_ID and row.get("seq") in SEQUENCES
    }
    return rows


def mask_text(mask: Any) -> str:
    if mask == "" or mask is None:
        return ""
    return ",".join(f"{float(value):g}" for value in mask)


def expected_action_field(action_mode: str) -> str:
    if action_mode == "anchor_special_only":
        return "forced_anchor_only"
    if action_mode in {"context_token_mask", "trajectory_context_token_mask", "context_only_special"}:
        return "forced_context_only"
    if action_mode == "force_non_keyframe":
        return "forced_non_keyframe"
    return ""


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    action_mode: str,
    indices: list[int],
    mask: Any,
) -> str:
    lines = [
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
        f"_stage4_action_mode: {action_mode}",
    ]
    if mask != "" and mask is not None:
        lines.append(f"_stage4_context_token_mask: {json.dumps(mask)}")
    lines.append("")
    return "\n".join(lines)


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
    selected_by_seq = source_rows()
    missing = [seq for seq in SEQUENCES if seq not in selected_by_seq]
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    T2.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v111tf_t2_fullseq_{seq}"
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
    policy_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    row_index = 0
    for policy in POLICIES:
        policy_id = str(policy["policy_id"])
        action_mode = str(policy["action_mode"])
        mask = policy["mask"]
        for seq in SEQUENCES:
            source = selected_by_seq.get(seq, {})
            source_indices = parse_indices(source.get("source_selected_global_frame_indices", ""))
            selected_indices = parse_indices(source.get("selected_global_frame_indices", ""))
            if not policy["source_selected"]:
                selected_indices = []
            frames = int(float(source.get("frames", "0"))) if source else 0
            dataset = f"kitti_v111tf_t2_fullseq_{seq}"
            method = f"lingbot_map_v111tf_t2_{policy_id}_{seq}"
            action_label = f"v111tf_t2_{policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v111tf_t2_{policy_id}_{seq}.yaml"
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
                    action_mode=action_mode,
                    indices=selected_indices,
                    mask=mask,
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            selected_string = ";".join(str(x) for x in selected_indices)
            source_selected_string = ";".join(str(x) for x in source_indices)
            row = {
                "schema": "acl2_v111tf_t2_context_token_policy_row_v1",
                "surface_id": "T",
                "candidate_id": "T2",
                "policy_id": policy_id,
                "policy_family": policy["policy_family"],
                "claim_role": policy["claim_role"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage2_action_mode": action_mode,
                "stage4_action_mode": action_mode,
                "source_policy_id": SOURCE_POLICY_ID,
                "source_selected_count": len(source_indices),
                "source_selected_global_frame_indices": source_selected_string,
                "selected_count": len(selected_indices),
                "selected_global_frame_indices": selected_string,
                "context_token_type_mask": mask_text(mask),
                "mask_camera": mask[0] if isinstance(mask, list) else "",
                "mask_register_0": mask[1] if isinstance(mask, list) else "",
                "mask_register_1": mask[2] if isinstance(mask, list) else "",
                "mask_register_2": mask[3] if isinstance(mask, list) else "",
                "mask_register_3": mask[4] if isinstance(mask, list) else "",
                "mask_anchor": mask[5] if isinstance(mask, list) else "",
                "frames": frames,
                "full_sequence_keyframe_interval": v108.keyframe_interval(frames) if frames else "",
                "expected_action_field": expected_action_field(action_mode),
                "runtime_boundary": (
                    "Selected B1 high-risk base keyframes use context-only compact special-token persistence; "
                    "new context_token_mask mode multiplies camera/register/anchor special slices before special-cache write."
                ),
                "claim_boundary": (
                    "T2 mechanism/geometry study only. Semantic-aware claim still requires causality controls; "
                    "no-action parity and legacy context-only parity must pass before promotion."
                ),
                "config": str(config.resolve()),
                "method_config": str(method_path.resolve()),
                "action_file": str(action_file.resolve()),
                "gpu": gpu,
            }
            policy_rows.append({k: v for k, v in row.items() if k not in {"config", "method_config", "action_file", "gpu"}})
            config_rows.append(row)

            prefix = v108.command_prefix(conda_path, pythonpath, gpu)
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v111tf_t2_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v111tf_t2_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"t2_fullseq_{seq}",
                    "target_kind": "full_sequence_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "deduplicated_prepare_once_per_dataset_seq_to_avoid_parallel_rmtree_race",
                    "selected_count": 0,
                    "force_non_keyframe_indices": "",
                    "context_token_type_mask": "",
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
                f"ACL2_V111TF_T2_POLICY_ID={policy_id} "
                f"ACL2_V111TF_T2_TOKEN_MASK={mask_text(mask)}"
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
                        "schema": "acl2_v111tf_t2_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v111tf_t2_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"t2_fullseq_{seq}",
                        "target_kind": "full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": action_mode,
                        "selector": "v110_stage4_B1_semantic_plus_internal_selected_keyframes",
                        "selected_count": len(selected_indices),
                        "force_non_keyframe_indices": selected_string,
                        "context_token_type_mask": mask_text(mask),
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
        "schema": "acl2_v111tf_t2_context_token_config_summary_v1",
        "t2_config_ready": not missing,
        "blocker": "missing_source_selected_rows" if missing else "",
        "missing_sequences": missing,
        "source_policy_id": SOURCE_POLICY_ID,
        "sequences": list(SEQUENCES),
        "policy_count": len(POLICIES),
        "policy_ids": [str(policy["policy_id"]) for policy in POLICIES],
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_manifest_rows": len(prepare_rows_by_seq),
        "run_worker_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "claim_boundary": (
            "Generated full 00/01/02/05 T2 ablation configs plus parity controls. "
            "No semantic-aware promotion is allowed until no-action parity, legacy all-special parity, "
            "geometry gates, and causality controls are evaluated."
        ),
        "outputs": {
            "action_config_rows": rel(T2 / "action_config_rows.csv"),
            "candidate_policy_rows": rel(T2 / "candidate_policy_rows.csv"),
            "run_manifest": rel(T2 / "run_manifest.csv"),
            "summary": rel(T2 / "t2_config_generation_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }

    write_csv(T2 / "action_config_rows.csv", config_rows)
    write_csv(T2 / "candidate_policy_rows.csv", policy_rows)
    write_csv(T2 / "run_manifest.csv", manifest_rows)
    write_json(T2 / "t2_config_generation_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    return 0 if summary["t2_config_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
