#!/usr/bin/env python3
"""Generate ACL2 v108TF Stage5 four-sequence validation configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import build_v108tf_stage4_full_kitti_pilot_configs as stage4


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
STAGE3 = RESULT_ROOT / "stage3_operation_cue_screen"
STAGE4 = RESULT_ROOT / "stage4_full_kitti_00_02_action_pilot"
OUT = RESULT_ROOT / "stage5_full_kitti_00_01_02_05_validation"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

SEQUENCES = ("00", "01", "02", "05")
POLICY_FAMILIES = ("semantic_plus_internal", "internal_only", "semantic_shuffle", "same_count_random")
SURFACE_ACTION_MODE = {
    "E": "v106_context_only_with_local_preserve",
    "F": "anchor_special_only",
}
HEADLOCAL_ALL_HEADS = list(range(16))


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
    return path.relative_to(ROOT).as_posix()


def frame_counts() -> dict[str, int]:
    return {
        row["seq"]: int(float(row["frames"]))
        for row in read_csv(STAGE0 / "full_kitti_baseline_table.csv")
        if row.get("seq") in SEQUENCES
    }


def selected_frames() -> dict[tuple[str, str, str], list[int]]:
    selected: dict[tuple[str, str, str], list[int]] = {}
    for row in read_csv(STAGE3 / "surface_policy_frame_rows.csv"):
        surface = row.get("surface_id", "")
        family = row.get("policy_family", "")
        seq = row.get("seq_id", "")
        if surface not in SURFACE_ACTION_MODE or family not in POLICY_FAMILIES or seq not in SEQUENCES:
            continue
        key = (surface, row["policy_id"], seq)
        selected.setdefault(key, []).append(int(float(row["frame_id"])))
    return {key: sorted(set(vals)) for key, vals in selected.items()}


def top_stage4_surfaces() -> list[str]:
    summary_path = STAGE4 / "stage4_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    passing = [s for s in summary.get("passing_surfaces", []) if s in SURFACE_ACTION_MODE]
    rows = summary.get("semantic_control_rows", [])
    score_by_surface = {
        row.get("surface_id"): float(row.get("semantic_plus_median_full_ATE_improvement_vs_no_action", float("nan")))
        for row in rows
        if row.get("surface_id") in passing
    }
    return sorted(passing, key=lambda surface: score_by_surface.get(surface, float("-inf")), reverse=True)[:2]


def expected_action_field(action_mode: str) -> str:
    return {
        "force_non_keyframe": "forced_non_keyframe",
        "anchor_special_only": "forced_anchor_only",
        "context_only_special": "forced_context_only",
        "v106_context_only_with_local_preserve": "headlocal_action_enabled",
    }[action_mode]


def build() -> dict[str, Any]:
    env = stage4.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frames_by_seq = frame_counts()
    selected = selected_frames()
    surfaces = top_stage4_surfaces()
    gpu_cycle = ["0", "1", "2", "3", "4"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)

    snap_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    action_config: dict[str, Any] = {"schema": "acl2_v108tf_stage5_action_config_v1", "policies": []}
    row_index = 0
    blocker = ""

    if len(surfaces) != 2:
        blocker = "stage4_top2_passing_surfaces_not_available"

    for seq in SEQUENCES:
        dataset = f"kitti_v108tf_stage5_fullseq_{seq}"
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

    for surface in surfaces:
        action_mode = SURFACE_ACTION_MODE[surface]
        for policy_family in POLICY_FAMILIES:
            policy_id = f"{surface}1_{policy_family}"
            for seq in SEQUENCES:
                source_indices = selected.get((surface, policy_id, seq), [])
                frames = frames_by_seq[seq]
                snapped_indices, rows = stage4.snap_to_nearest_base_keyframe(source_indices, frames)
                if source_indices and not snapped_indices:
                    blocker = blocker or f"all_selected_frames_failed_keyframe_snap_{surface}_{policy_family}_{seq}"
                for row in rows:
                    snap_rows.append(
                        {
                            "schema": "acl2_v108tf_stage5_keyframe_snap_row_v1",
                            "surface_id": surface,
                            "policy_id": policy_id,
                            "policy_family": policy_family,
                            "seq": seq,
                            **row,
                        }
                    )

                if action_mode == "v106_context_only_with_local_preserve":
                    force_indices: list[int] = []
                    headlocal_map = {idx: HEADLOCAL_ALL_HEADS for idx in snapped_indices}
                else:
                    force_indices = snapped_indices
                    headlocal_map = None

                dataset = f"kitti_v108tf_stage5_fullseq_{seq}"
                action_label = f"v108tf_stage5_{policy_id}"
                method = f"lingbot_map_v108tf_stage5_{policy_id}_{seq}"
                config = CONFIG_ROOT / f"kitti_lingbot_v108tf_stage5_{policy_id}_{seq}.yaml"
                method_path = METHOD_DIR / f"{method}.yaml"
                action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
                gpu = gpu_cycle[row_index % len(gpu_cycle)]
                row_index += 1

                write_text(
                    method_path,
                    stage4.method_yaml(
                        checkpoint,
                        env_name,
                        use_sdpa,
                        action_label,
                        action_mode,
                        force_indices,
                        headlocal_map,
                    ),
                )
                write_text(config, stage4.run_config_yaml(dataset, method).replace(str(stage4.WORKSPACE.resolve()), str(WORKSPACE.resolve())))
                selected_string = ";".join(str(x) for x in snapped_indices)
                policy = {
                    "schema": "acl2_v108tf_stage5_policy_row_v1",
                    "surface_id": surface,
                    "policy_id": policy_id,
                    "policy_family": policy_family,
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": action_label,
                    "stage5_action_mode": action_mode,
                    "stage4_action_mode": action_mode,
                    "source_selected_count": len(source_indices),
                    "source_selected_global_frame_indices": ";".join(str(x) for x in source_indices),
                    "selected_count": len(snapped_indices),
                    "selected_global_frame_indices": selected_string,
                    "headlocal_all_heads": ";".join(str(x) for x in HEADLOCAL_ALL_HEADS) if headlocal_map else "",
                    "frames": frames,
                    "full_sequence_keyframe_interval": stage4.keyframe_interval(frames),
                    "snap_radius": max(1, stage4.keyframe_interval(frames) // 2),
                    "expected_action_field": expected_action_field(action_mode),
                    "stage5_source": "top2_stage4_passing_surface",
                    "runtime_boundary": "LingBot internal memory/cache/context action; no output post-processing.",
                }
                policy_rows.append(policy)
                config_rows.append(
                    {
                        **policy,
                        "config": str(config.resolve()),
                        "method_config": str(method_path.resolve()),
                        "action_file": str(action_file.resolve()),
                        "gpu": gpu,
                    }
                )
                action_config["policies"].append({**policy, "config": rel(config), "method_config": rel(method_path), "action_file": rel(action_file)})

                prefix = stage4.command_prefix(conda_path, pythonpath, gpu)
                action_env = (
                    f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                    f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                    f"ACL2_V108_STAGE4_POLICY_ID={policy_id} "
                    f"ACL2_V108_STAGE4_SURFACE_ID={surface}"
                )
                prepare_command = (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python prepare.py --config {config.resolve()} --force"
                )
                prepare_rows_by_seq.setdefault(
                    seq,
                    {
                        "schema": "acl2_v108tf_stage5_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v108tf_stage5_prepare_{seq}",
                        "phase": "prepare",
                        "target_id": f"fullseq_{seq}",
                        "target_kind": "full_sequence_dataset_prepare",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": "dataset_prepare_once",
                        "action_family": "prepare",
                        "stage5_action_mode": "dataset_prepare",
                        "stage4_action_mode": "dataset_prepare",
                        "selector": "deduplicated_prepare_once_per_dataset_seq_to_avoid_parallel_rmtree_race",
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
                            "schema": "acl2_v108tf_stage5_manifest_row_v1",
                            "run_name": f"kitti_lingbot_v108tf_stage5_{policy_id}_{seq}_{phase}",
                            "phase": phase,
                            "target_id": f"fullseq_{seq}",
                            "target_kind": "full_sequence",
                            "seq": seq,
                            "dataset": dataset,
                            "method": method,
                            "action_name": action_label,
                            "action_family": policy_family,
                            "stage5_action_mode": action_mode,
                            "stage4_action_mode": action_mode,
                            "selector": "stage3_policy_frames_snapped_to_four_sequence_full_validation",
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

    manifest_rows = [prepare_rows_by_seq[seq] for seq in SEQUENCES if seq in prepare_rows_by_seq] + manifest_rows
    write_csv(OUT / "keyframe_snap_rows.csv", snap_rows)
    write_csv(OUT / "full_sequence_policy_rows.csv", policy_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_json(OUT / "action_config.json", action_config)
    summary = {
        "schema": "acl2_v108tf_stage5_config_generation_summary_v1",
        "stage5_config_ready": blocker == "",
        "blocker": blocker,
        "sequences": list(SEQUENCES),
        "surfaces": surfaces,
        "policy_families": list(POLICY_FAMILIES),
        "policy_rows": len(policy_rows),
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "no_action_control_source": "v105 frozen LingBot stream default baseline from stage0/full_kitti_baseline_table.csv and v105 workspace trajectories",
        "low_risk_reverse_status": "not_applicable_no_stage4_low_risk_reverse_policy_defined_for_E_F",
        "workspace": rel(WORKSPACE),
        "raw_action": rel(RAW_ACTION),
        "runtime_boundary": "Full-sequence LingBot internal actions only; no external depth/SLAM/Sim3 post-processing as runtime cue.",
    }
    write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
