#!/usr/bin/env python3
"""Generate v109TF Stage2 F19 exact-count keyframe random control configs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_role_specific_full_candidate_configs as fullcfg  # noqa: E402


stage4 = fullcfg.stage4

RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
F19 = RESULT_ROOT / "stage2_role_specific_safety_candidates"
OUT = RESULT_ROOT / "stage2_f19_keyframe_controls"
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
    ("F20_f19_same_count_keyframe_random_seed0", "f19_same_count_keyframe_random_seed0", "seed0"),
    ("F21_f19_same_count_keyframe_random_seed1", "f19_same_count_keyframe_random_seed1", "seed1"),
    ("F22_f19_same_count_keyframe_random_seed2", "f19_same_count_keyframe_random_seed2", "seed2"),
)


def parse_indices(value: str) -> list[int]:
    return [int(float(part)) for part in str(value).replace(",", ";").split(";") if part.strip()]


def target_counts_and_schedules() -> tuple[dict[str, int], dict[str, list[int]]]:
    counts: dict[str, int] = {}
    schedules: dict[str, list[int]] = {}
    for row in fullcfg.read_csv(F19 / "action_config_rows.csv"):
        if row.get("policy_id") != "F19_dynamic_or_special_admitted_high_risk_else_weak_context":
            continue
        seq = row.get("seq", "")
        if seq in SEQUENCES:
            schedules[seq] = parse_indices(row.get("selected_global_frame_indices", ""))
            counts[seq] = int(float(row.get("selected_count", len(schedules[seq]))))
    return counts, schedules


def random_keyframes(seq: str, frames: int, count: int, seed: str) -> list[int]:
    universe = stage4.base_keyframes(frames)
    ranked = sorted(
        universe,
        key=lambda frame: hashlib.sha256(
            f"v109tf_f19_keyframe_control|{seed}|{seq}|{frame}".encode("utf-8")
        ).hexdigest(),
    )
    return sorted(ranked[:count])


def case_map() -> dict[tuple[str, int], dict[str, Any]]:
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    for case in fullcfg.rolecfg.stage3.build_surface_cases().get(SURFACE, []):
        seq = case.get("seq_id")
        if seq in SEQUENCES:
            cases[(seq, int(case["frame_id"]))] = case
    return cases


def source_frame_rows(policy_frames: dict[str, dict[str, list[int]]]) -> list[dict[str, Any]]:
    cases = case_map()
    family = {policy_id: policy_family for policy_id, policy_family, _seed in POLICIES}
    rows: list[dict[str, Any]] = []
    for policy_id, by_seq in policy_frames.items():
        for seq, frames in by_seq.items():
            for rank, frame_id in enumerate(frames, start=1):
                case = cases.get((seq, frame_id), {})
                rows.append(
                    {
                        "schema": "acl2_v109tf_stage2_f19_keyframe_control_source_frame_row_v1",
                        "surface_id": SURFACE,
                        "policy_id": policy_id,
                        "policy_family": family[policy_id],
                        "seq": seq,
                        "source_frame": frame_id,
                        "source_rank": rank,
                        "role_score": "",
                        "stable_structure_mass": case.get("stable_structure_mass", ""),
                        "dynamic_mass": case.get("dynamic_mass", ""),
                        "boundary_mass": case.get("boundary_mass", ""),
                        "weak_context_mass": case.get("weak_context_mass", ""),
                        "road_ground_mass": case.get("road_ground_mass", ""),
                        "sky_lowobs_mass": case.get("sky_lowobs_mass", ""),
                        "semantic_trust_mean": case.get("semantic_trust_mean", ""),
                        "semantic_purity_mean": case.get("semantic_purity_mean", ""),
                        "semantic_continuity_score": case.get("semantic_continuity_score", ""),
                        "semantic_boundary_risk": case.get("semantic_boundary_risk", ""),
                        "special_token_count": case.get("special_token_count", ""),
                        "cache_append_count": case.get("cache_append_count", ""),
                        "trajectory_write_count": case.get("trajectory_write_count", ""),
                    }
                )
    return rows


def expected_action_field() -> str:
    return "forced_anchor_only"


def build() -> dict[str, Any]:
    env = stage4.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frames_by_seq = fullcfg.frame_counts()
    target_counts, f19_schedules = target_counts_and_schedules()
    blockers: list[str] = []
    for seq in SEQUENCES:
        if seq not in frames_by_seq:
            blockers.append(f"missing_frame_count_{seq}")
        if seq not in target_counts:
            blockers.append(f"missing_f19_selected_count_{seq}")
        if target_counts.get(seq, 0) <= 0:
            blockers.append(f"empty_f19_selected_count_{seq}")

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    for seq in SEQUENCES:
        dataset = f"kitti_v109tf_stage2_f19ctrl_fullseq_{seq}"
        fullcfg.write_text(
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

    policy_frames: dict[str, dict[str, list[int]]] = {}
    overlap_rows: list[dict[str, Any]] = []
    for policy_id, _family, seed in POLICIES:
        policy_frames[policy_id] = {}
        for seq in SEQUENCES:
            frames = frames_by_seq[seq]
            selected = random_keyframes(seq, frames, target_counts.get(seq, 0), seed)
            policy_frames[policy_id][seq] = selected
            f19 = set(f19_schedules.get(seq, []))
            ctrl = set(selected)
            overlap_rows.append(
                {
                    "schema": "acl2_v109tf_stage2_f19_keyframe_control_overlap_row_v1",
                    "policy_id": policy_id,
                    "policy_family": _family,
                    "seq": seq,
                    "control_selected_count": len(ctrl),
                    "f19_selected_count": len(f19),
                    "intersection_count": len(ctrl & f19),
                    "control_only_count": len(ctrl - f19),
                    "f19_only_count": len(f19 - ctrl),
                    "jaccard": (len(ctrl & f19) / len(ctrl | f19)) if (ctrl or f19) else 1.0,
                    "control_selected_global_frame_indices": ";".join(str(x) for x in sorted(ctrl)),
                    "f19_selected_global_frame_indices": ";".join(str(x) for x in sorted(f19)),
                }
            )

    snap_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows: dict[str, dict[str, Any]] = {}
    reuse_rows: list[dict[str, Any]] = []
    row_index = 0
    for policy_id, policy_family, _seed in POLICIES:
        for seq in SEQUENCES:
            source_indices = policy_frames[policy_id][seq]
            frames = frames_by_seq[seq]
            snapped_indices, rows = stage4.snap_to_nearest_base_keyframe(source_indices, frames)
            if len(snapped_indices) != target_counts.get(seq, -1):
                blockers.append(f"selected_count_mismatch_{policy_id}_{seq}_{len(snapped_indices)}_vs_{target_counts.get(seq)}")
            for row in rows:
                snap_rows.append(
                    {
                        "schema": "acl2_v109tf_stage2_f19_keyframe_control_snap_row_v1",
                        "surface_id": SURFACE,
                        "policy_id": policy_id,
                        "policy_family": policy_family,
                        "seq": seq,
                        **row,
                    }
                )
            dataset = f"kitti_v109tf_stage2_f19ctrl_fullseq_{seq}"
            method = f"lingbot_map_v109tf_stage2_f19ctrl_{policy_id}_{seq}"
            action_label = f"v109tf_stage2_f19ctrl_{policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v109tf_stage2_f19ctrl_{policy_id}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = GPU_CYCLE[row_index % len(GPU_CYCLE)]
            row_index += 1
            fullcfg.write_text(
                method_path,
                stage4.method_yaml(checkpoint, env_name, use_sdpa, action_label, ACTION_MODE, snapped_indices, None),
            )
            fullcfg.write_text(
                config,
                stage4.run_config_yaml(dataset, method).replace(str(stage4.WORKSPACE.resolve()), str(WORKSPACE.resolve())),
            )
            selected_string = ";".join(str(x) for x in snapped_indices)
            policy = {
                "schema": "acl2_v109tf_stage2_f19_keyframe_control_policy_row_v1",
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
                "f19_selected_count": target_counts.get(seq, ""),
                "f19_selected_global_frame_indices": ";".join(str(x) for x in f19_schedules.get(seq, [])),
                "frames": frames,
                "full_sequence_keyframe_interval": stage4.keyframe_interval(frames),
                "snap_radius": max(1, stage4.keyframe_interval(frames) // 2),
                "expected_action_field": expected_action_field(),
                "runtime_boundary": "Exact selected-keyframe-count random control for F19; no semantic score or GT metric in selector.",
                "candidate_scope": "full KITTI 00/01/02/05 F19 same-count keyframe-aware control",
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
                    "schema": "acl2_v109tf_stage2_f19_keyframe_control_reuse_manifest_row_v1",
                    "policy_id": policy_id,
                    "seq": seq,
                    "reuse_used": False,
                    "reason": "new same-count keyframe random control; F19 artifacts are reference only",
                    "source_f19": fullcfg.rel(F19),
                    "target_out": fullcfg.rel(OUT),
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
                    "schema": "acl2_v109tf_stage2_f19_keyframe_control_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v109tf_stage2_role_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"f19ctrl_fullseq_{seq}",
                    "target_kind": "full_sequence_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage2_action_mode": "dataset_prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "f19_keyframe_control_deduplicated_prepare_once_per_seq",
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
                        "schema": "acl2_v109tf_stage2_f19_keyframe_control_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v109tf_stage2_role_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"f19ctrl_fullseq_{seq}",
                        "target_kind": "full_sequence_f19_keyframe_control",
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
    fullcfg.write_csv(OUT / "role_source_frame_rows.csv", source_frame_rows(policy_frames))
    fullcfg.write_csv(OUT / "role_keyframe_snap_rows.csv", snap_rows)
    fullcfg.write_csv(OUT / "action_config_rows.csv", config_rows)
    fullcfg.write_csv(OUT / "run_manifest.csv", manifest_rows)
    fullcfg.write_csv(OUT / "stage2_reuse_manifest.csv", reuse_rows)
    fullcfg.write_csv(OUT / "f19_keyframe_overlap_rows.csv", overlap_rows)
    summary = {
        "schema": "acl2_v109tf_stage2_f19_keyframe_control_config_summary_v1",
        "f19_keyframe_control_config_ready": not blockers,
        "blockers": blockers,
        "surface_id": SURFACE,
        "sequences": list(SEQUENCES),
        "policies": [policy_id for policy_id, _family, _seed in POLICIES],
        "policy_count": len(POLICIES),
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_rows": len(prepare_rows),
        "run_worker_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "control_basis": "Exact F19 selected-count random controls sampled from the same LingBot base-keyframe grid per sequence.",
        "f19_reference": fullcfg.rel(F19),
        "workspace": fullcfg.rel(WORKSPACE),
        "outputs": {
            "role_source_frame_rows": fullcfg.rel(OUT / "role_source_frame_rows.csv"),
            "role_keyframe_snap_rows": fullcfg.rel(OUT / "role_keyframe_snap_rows.csv"),
            "action_config_rows": fullcfg.rel(OUT / "action_config_rows.csv"),
            "run_manifest": fullcfg.rel(OUT / "run_manifest.csv"),
            "stage2_reuse_manifest": fullcfg.rel(OUT / "stage2_reuse_manifest.csv"),
            "f19_keyframe_overlap_rows": fullcfg.rel(OUT / "f19_keyframe_overlap_rows.csv"),
            "config_generation_summary": fullcfg.rel(OUT / "config_generation_summary.json"),
        },
    }
    fullcfg.write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
