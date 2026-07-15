#!/usr/bin/env python3
"""Generate v109TF Stage2 role-specific full-KITTI candidate configs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_role_specific_pilot_configs as rolecfg  # noqa: E402


stage4 = rolecfg.stage4

RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
CORE = RESULT_ROOT / "stage2_f_core_ablation"
OUT = RESULT_ROOT / "stage2_role_specific_full_candidates"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

SEQUENCES = ("00", "01", "02", "05")
SURFACE = "F"
ACTION_MODE = "anchor_special_only"
GPU_CYCLE = ("0", "1", "2", "3", "4")

POLICIES = (
    ("F13_dynamic_boundary_only", "dynamic_boundary_only"),
    ("F14_weak_context_only", "weak_context_only"),
    ("F18_high_risk_high_boundary", "high_risk_high_boundary"),
)


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
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def configure_role_helpers() -> None:
    rolecfg.SEQUENCES = SEQUENCES
    rolecfg.POLICIES = POLICIES


def frame_counts() -> dict[str, int]:
    return {
        row["seq"]: int(float(row["frames"]))
        for row in read_csv(RESULT_ROOT / "stage0_evidence_freeze/full_kitti_baseline_table.csv")
        if row.get("seq") in SEQUENCES
    }


def source_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for row in read_csv(CORE / "action_config_rows.csv"):
        if row.get("policy_id") == "F1_semantic_plus_internal" and row.get("seq") in SEQUENCES:
            out[row["seq"]] = int(float(row["source_selected_count"]))
    return out


def cases_by_seq() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {seq: [] for seq in SEQUENCES}
    for case in rolecfg.stage3.build_surface_cases().get(SURFACE, []):
        seq = case.get("seq_id")
        if seq in out:
            out[seq].append(case)
    for seq in out:
        out[seq].sort(key=lambda row: int(row["frame_id"]))
    return out


def expected_action_field() -> str:
    return "forced_anchor_only"


def build() -> dict[str, Any]:
    configure_role_helpers()
    env = stage4.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frames_by_seq = frame_counts()
    counts = source_counts()
    cases = cases_by_seq()
    blockers: list[str] = []
    for seq in SEQUENCES:
        if seq not in frames_by_seq:
            blockers.append(f"missing_frame_count_{seq}")
        if seq not in counts:
            blockers.append(f"missing_f1_source_count_{seq}")
        if not cases.get(seq):
            blockers.append(f"missing_surface_cases_{seq}")

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    for seq in SEQUENCES:
        dataset = f"kitti_v109tf_stage2_rolefull_fullseq_{seq}"
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

    policy_frames = {
        policy_id: rolecfg.select_policy_frames(policy_id, cases, counts)
        for policy_id, _family in POLICIES
    }
    source_rows = rolecfg.source_frame_rows(policy_frames, cases)
    snap_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows: dict[str, dict[str, Any]] = {}
    reuse_rows: list[dict[str, Any]] = []
    row_index = 0
    for policy_id, policy_family in POLICIES:
        for seq in SEQUENCES:
            source_indices = policy_frames[policy_id][seq]
            frames = frames_by_seq[seq]
            snapped_indices, rows = stage4.snap_to_nearest_base_keyframe(source_indices, frames)
            for row in rows:
                snap_rows.append(
                    {
                        "schema": "acl2_v109tf_stage2_rolefull_keyframe_snap_row_v1",
                        "surface_id": SURFACE,
                        "policy_id": policy_id,
                        "policy_family": policy_family,
                        "seq": seq,
                        **row,
                    }
                )
            dataset = f"kitti_v109tf_stage2_rolefull_fullseq_{seq}"
            method = f"lingbot_map_v109tf_stage2_rolefull_{policy_id}_{seq}"
            action_label = f"v109tf_stage2_rolefull_{policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v109tf_stage2_rolefull_{policy_id}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = GPU_CYCLE[row_index % len(GPU_CYCLE)]
            row_index += 1
            write_text(
                method_path,
                stage4.method_yaml(checkpoint, env_name, use_sdpa, action_label, ACTION_MODE, snapped_indices, None),
            )
            write_text(
                config,
                stage4.run_config_yaml(dataset, method).replace(str(stage4.WORKSPACE.resolve()), str(WORKSPACE.resolve())),
            )
            selected_string = ";".join(str(x) for x in snapped_indices)
            policy = {
                "schema": "acl2_v109tf_stage2_rolefull_policy_row_v1",
                "surface_id": SURFACE,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage2_action_mode": ACTION_MODE,
                "stage4_action_mode": ACTION_MODE,
                "source_selected_count": len(source_indices),
                "source_selected_global_frame_indices": ";".join(str(x) for x in source_indices),
                "selected_count": len(snapped_indices),
                "selected_global_frame_indices": selected_string,
                "frames": frames,
                "full_sequence_keyframe_interval": stage4.keyframe_interval(frames),
                "snap_radius": max(1, stage4.keyframe_interval(frames) // 2),
                "expected_action_field": expected_action_field(),
                "runtime_boundary": "Role-specific F-surface full-candidate action; no GT/post-hoc metric in selector.",
                "candidate_scope": "full KITTI 00/01/02/05 candidate promoted from role-specific 00/02 pilot pre-gate",
            }
            config_rows.append(
                {
                    **policy,
                    "config": str(config.resolve()),
                    "method_config": str(method_path.resolve()),
                    "action_file": str(action_file.resolve()),
                    "gpu": gpu,
                }
            )
            reuse_rows.append(
                {
                    "schema": "acl2_v109tf_stage2_rolefull_reuse_manifest_row_v1",
                    "policy_id": policy_id,
                    "seq": seq,
                    "reuse_used": False,
                    "reason": "new full-candidate run; pilot 00/02 artifacts not reused to avoid output-namespace/provenance mixing",
                    "source_pilot": rel(RESULT_ROOT / "stage2_role_specific_pilot"),
                    "target_out": rel(OUT),
                }
            )
            prefix = stage4.command_prefix(conda_path, pythonpath, gpu)
            action_env = (
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                f"ACL2_V108_STAGE4_POLICY_ID={policy_id} "
                f"ACL2_V108_STAGE4_SURFACE_ID={SURFACE}"
            )
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows.setdefault(
                seq,
                {
                    "schema": "acl2_v109tf_stage2_rolefull_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v109tf_stage2_role_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"rolefull_fullseq_{seq}",
                    "target_kind": "full_sequence_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage2_action_mode": "dataset_prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "rolefull_deduplicated_prepare_once_per_seq",
                    "selected_count": 0,
                    "force_non_keyframe_indices": "",
                    "trace_start_idx": 0,
                    "trace_end_idx_exclusive": frames,
                    "target_frame_start": 0,
                    "target_frame_end": frames - 1,
                    "gpu": gpu,
                    "cwd": str(BENCHMARK.resolve()),
                    "config": str(config.resolve()),
                    "trace_file": "",
                    "action_file": "",
                    "command": prepare_command,
                    "status": "planned",
                },
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
                        "schema": "acl2_v109tf_stage2_rolefull_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v109tf_stage2_role_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"rolefull_fullseq_{seq}",
                        "target_kind": "full_sequence_role_candidate",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy_family,
                        "stage2_action_mode": ACTION_MODE,
                        "stage4_action_mode": ACTION_MODE,
                        "selector": policy_family,
                        "selected_count": len(snapped_indices),
                        "force_non_keyframe_indices": selected_string,
                        "trace_start_idx": 0,
                        "trace_end_idx_exclusive": frames,
                        "target_frame_start": 0,
                        "target_frame_end": frames - 1,
                        "gpu": gpu,
                        "cwd": str(BENCHMARK.resolve()),
                        "config": str(config.resolve()),
                        "trace_file": "",
                        "action_file": str(action_file.resolve()),
                        "command": command,
                        "status": "planned",
                    }
                )
    manifest_rows = [prepare_rows[seq] for seq in SEQUENCES if seq in prepare_rows] + manifest_rows
    write_csv(OUT / "role_source_frame_rows.csv", source_rows)
    write_csv(OUT / "role_keyframe_snap_rows.csv", snap_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_csv(OUT / "stage2_reuse_manifest.csv", reuse_rows)
    summary = {
        "schema": "acl2_v109tf_stage2_rolefull_config_summary_v1",
        "role_full_candidate_config_ready": not blockers,
        "blockers": blockers,
        "surface_id": SURFACE,
        "sequences": list(SEQUENCES),
        "policies": [policy_id for policy_id, _family in POLICIES],
        "policy_count": len(POLICIES),
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_rows": len(prepare_rows),
        "run_worker_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "candidate_basis": "F13/F14/F18 passed the 00/02 role-specific pilot pre-gate; F13/F18 had identical snapped schedule on 00/02 but are retained for 01/05 divergence audit.",
        "source_count_basis": "count-matched to F1_semantic_plus_internal source_selected_count per seq",
        "selector_boundary": "runtime semantic role masses/trust/purity/continuity only; no GT/ATE/post-hoc Sim3",
        "workspace": rel(WORKSPACE),
        "outputs": {
            "role_source_frame_rows": rel(OUT / "role_source_frame_rows.csv"),
            "role_keyframe_snap_rows": rel(OUT / "role_keyframe_snap_rows.csv"),
            "action_config_rows": rel(OUT / "action_config_rows.csv"),
            "run_manifest": rel(OUT / "run_manifest.csv"),
            "stage2_reuse_manifest": rel(OUT / "stage2_reuse_manifest.csv"),
            "config_generation_summary": rel(OUT / "config_generation_summary.json"),
        },
    }
    write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
