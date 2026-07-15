#!/usr/bin/env python3
"""Generate v107R Stage7B keyframe-aware full-sequence policy configs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_STAGE0 = V105 / "stage0_repo_env_audit/stage0_summary.json"
V105_FULL_METRICS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"
STAGE6_POLICY_ROWS = (
    V107R
    / "stage6_runtime_pilot_or_blocked/semantic_wrapper_policy_pilot/semantic_action_policy_rows.csv"
)
OUT = V107R / "stage7b_full_sequence_keyframe_aware_policy"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

SOURCE_ACTION = "semantic_highrisk_early_risk_ge_0p60_force_non_keyframe"
STAGE7B_ACTION = "semantic_threshold0p60_keyframe_snap_fullseq_force_non_keyframe"
SEQUENCES = ["00", "01", "02", "05"]
SCALE_FRAMES = 8
AUTO_KEYFRAME_THRESHOLD = 320


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_indices(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    return sorted({int(float(x)) for x in raw.replace(",", ";").split(";") if x.strip()})


def frame_counts_from_v105() -> dict[str, int]:
    return {row["seq"]: int(float(row["frames"])) for row in read_csv(V105_FULL_METRICS)}


def selected_source_by_seq() -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    by_seq = {seq: [] for seq in SEQUENCES}
    source_rows: list[dict[str, Any]] = []
    for row in read_csv(STAGE6_POLICY_ROWS):
        if row["action_name"] != SOURCE_ACTION:
            continue
        seq = row["seq"]
        indices = parse_indices(row.get("force_global_frame_indices", ""))
        by_seq.setdefault(seq, []).extend(indices)
        source_rows.append(
            {
                "schema": "acl2_v107r_stage7b_keyframe_snap_source_row_v1",
                "seq": seq,
                "source_target_id": row["target_id"],
                "source_target_kind": row["target_kind"],
                "source_action_name": row["action_name"],
                "source_selected_global_frame_indices": ";".join(str(x) for x in indices),
                "source_selected_count": len(indices),
                "selector_scope": row.get("selector_scope", ""),
                "risk_threshold": row.get("risk_threshold", ""),
                "selected_risk_mean": row.get("selected_risk_mean", ""),
            }
        )
    return {seq: sorted(set(vals)) for seq, vals in by_seq.items()}, source_rows


def keyframe_interval(frames: int) -> int:
    if frames <= AUTO_KEYFRAME_THRESHOLD:
        return 1
    return math.ceil(frames / AUTO_KEYFRAME_THRESHOLD)


def base_keyframes(frames: int) -> list[int]:
    interval = keyframe_interval(frames)
    return list(range(SCALE_FRAMES, frames, interval))


def snap_to_nearest_base_keyframe(selected: list[int], frames: int) -> tuple[list[int], list[dict[str, Any]]]:
    bases = base_keyframes(frames)
    interval = keyframe_interval(frames)
    radius = max(1, interval // 2)
    snapped: set[int] = set()
    rows: list[dict[str, Any]] = []
    for idx in selected:
        nearest = min(bases, key=lambda base: (abs(base - idx), base))
        distance = abs(nearest - idx)
        accepted = distance <= radius
        if accepted:
            snapped.add(nearest)
        rows.append(
            {
                "source_frame": idx,
                "snapped_base_keyframe": nearest if accepted else "",
                "distance": distance,
                "accepted": accepted,
                "snap_radius": radius,
                "keyframe_interval": interval,
            }
        )
    return sorted(snapped), rows


def method_yaml(checkpoint: str, env_name: str, use_sdpa: bool, action_name: str, indices: list[int]) -> str:
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
            f"_stage4_action_label: {action_name}",
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


def command_prefix(conda_path: str, pythonpath: str, gpu: str) -> str:
    return f"PATH={Path(conda_path).parent}:$PATH PYTHONPATH={pythonpath} CUDA_VISIBLE_DEVICES={gpu}"


def build() -> dict[str, Any]:
    stage0 = json.loads(V105_STAGE0.read_text(encoding="utf-8"))
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(stage0["kitti"]["resolved_kitti_root"])
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frame_counts = frame_counts_from_v105()
    source_by_seq, source_rows = selected_source_by_seq()

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)

    snap_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    selected_by_seq: dict[str, list[int]] = {}
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    for seq_index, seq in enumerate(SEQUENCES):
        frames = frame_counts[seq]
        source_indices = source_by_seq.get(seq, [])
        snapped_indices, rows = snap_to_nearest_base_keyframe(source_indices, frames)
        selected_by_seq[seq] = snapped_indices
        for row in rows:
            snap_rows.append({"schema": "acl2_v107r_stage7b_keyframe_snap_row_v1", "seq": seq, **row})

        dataset = f"kitti_v107r_stage7b_fullseq_keyframe_snap_{seq}"
        method = f"lingbot_map_v107r_stage7b_{STAGE7B_ACTION}_{seq}"
        config = CONFIG_ROOT / f"kitti_lingbot_v107r_stage7b_{STAGE7B_ACTION}_{seq}.yaml"
        method_path = METHOD_DIR / f"{method}.yaml"
        action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
        gpu = gpu_cycle[seq_index % len(gpu_cycle)]

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
        write_text(method_path, method_yaml(checkpoint, env_name, use_sdpa, STAGE7B_ACTION, snapped_indices))
        write_text(config, run_config_yaml(dataset, method))

        policy = {
            "schema": "acl2_v107r_stage7b_full_sequence_policy_row_v1",
            "seq": seq,
            "dataset": dataset,
            "method": method,
            "action_name": STAGE7B_ACTION,
            "source_stage6_action": SOURCE_ACTION,
            "stage4_action_mode": "force_non_keyframe",
            "source_selected_count": len(source_indices),
            "source_selected_global_frame_indices": ";".join(str(x) for x in source_indices),
            "selected_count": len(snapped_indices),
            "selected_global_frame_indices": ";".join(str(x) for x in snapped_indices),
            "frames": frames,
            "full_sequence_keyframe_interval": keyframe_interval(frames),
            "snap_radius": max(1, keyframe_interval(frames) // 2),
            "policy_scope": "keyframe_aware_snap_of_stage6_semantic_selected_frames",
            "policy_boundary": (
                "Semantic selected frames are snapped to nearby full-sequence base keyframes so "
                "force_non_keyframe changes actual cache writes."
            ),
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

        prefix = command_prefix(conda_path, pythonpath, gpu)
        action_env = (
            f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
            f"ACL2_V105_STAGE4_ACTION_LABEL={STAGE7B_ACTION} "
            "ACL2_V107_STAGE7_POLICY_SCOPE=keyframe_aware_full_sequence_policy"
        )
        commands = {
            "prepare": (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            ),
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
                    "schema": "acl2_v107r_stage7b_full_sequence_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v107r_stage7_{STAGE7B_ACTION}_{seq}_{phase}",
                    "phase": phase,
                    "target_id": f"fullseq_{seq}",
                    "target_kind": "full_sequence",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": STAGE7B_ACTION,
                    "action_family": "semantic_action_threshold_keyframe_aware_full_sequence",
                    "stage4_action_mode": "force_non_keyframe",
                    "selector": "stage6_semantic_risk_ge_0p60_snapped_to_full_sequence_base_keyframes",
                    "selected_count": len(snapped_indices),
                    "force_non_keyframe_indices": ";".join(str(x) for x in snapped_indices),
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

    write_csv(OUT / "stage6_selected_source_rows.csv", source_rows)
    write_csv(OUT / "keyframe_snap_rows.csv", snap_rows)
    write_csv(OUT / "full_sequence_policy_rows.csv", policy_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    summary = {
        "schema": "acl2_v107r_stage7b_keyframe_aware_config_summary_v1",
        "stage6_source_action": SOURCE_ACTION,
        "stage7b_action": STAGE7B_ACTION,
        "sequences": SEQUENCES,
        "selected_indices_by_seq": selected_by_seq,
        "source_indices_by_seq": {seq: source_by_seq.get(seq, []) for seq in SEQUENCES},
        "policy_rows": len(policy_rows),
        "manifest_rows": len(manifest_rows),
        "workspace": rel(WORKSPACE),
        "raw_action": rel(RAW_ACTION),
        "runtime_boundary": "Full-sequence LingBot wrapper force_non_keyframe at semantic-near base keyframes; no output post-processing.",
    }
    write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
