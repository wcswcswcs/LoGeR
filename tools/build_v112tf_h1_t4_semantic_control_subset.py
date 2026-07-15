#!/usr/bin/env python3
"""Build and summarize ACL2 v112TF H1/T4 semantic-control subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402
import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402
import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402
import build_v112tf_h1_t4_lifetime_pilot_configs as h1cfg  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
FULL = RESULT_ROOT / "stage6_h1_t4_full_validation_00_01_02_05"
OUT = RESULT_ROOT / "stage7_h1_t4_semantic_controls_subset_00_01_02_05"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
FULL_CONFIG_ROWS = FULL / "action_config_rows.csv"
FULL_POLICY_SUMMARY = FULL / "policy_summary_rows.csv"
CUE = RESULT_ROOT / "stage2_memory_specific_cue_bank/anchor_memory_cue_rows.csv"
SEQUENCES = ("00", "01", "02", "05")
FULL_GATE_CANDIDATES = (
    "H1_semantic_lifetime_soft_raw",
    "T4_role_adaptive_policy",
)
SAME_BUCKET_SEEDS = (0, 1, 2)
SHUFFLE_SEEDS = (0,)
MATCH_TOL = 0.005


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_indices(raw: str) -> list[int]:
    return [int(float(part)) for part in str(raw).split(";") if str(part).strip()]


def mask_text(mask: list[float]) -> str:
    return ",".join(f"{value:.6g}" for value in mask)


def cue_rows_by_seq() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {seq: [] for seq in SEQUENCES}
    for row in read_csv(CUE):
        seq = row.get("seq", "")
        if seq in out:
            out[seq].append(row)
    for seq in out:
        out[seq].sort(key=lambda row: int(float(row.get("frame_id", 0))))
    return out


def cue_by_key(cues: dict[str, list[dict[str, str]]]) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (seq, int(float(row.get("frame_id", 0)))): row
        for seq, rows in cues.items()
        for row in rows
    }


def frame_bucket(frame: int, frame_count: int) -> str:
    if frame < frame_count / 3:
        return "early"
    if frame < 2 * frame_count / 3:
        return "mid"
    return "late"


def semantic_bucket(row: dict[str, str]) -> str:
    risk = (
        safe_float(row.get("dynamic_mass"), 0.0)
        + safe_float(row.get("boundary_mass"), 0.0)
        + safe_float(row.get("weak_context_mass"), 0.0)
    )
    if risk < 0.20:
        return "low_risk"
    if risk < 0.60:
        return "mid_risk"
    return "high_risk"


def keyframe_bucket(frame: int) -> str:
    return f"kbin_{frame // 300}"


def bucket_tuple(row: dict[str, str], frame_count: int) -> tuple[str, str, str, str]:
    frame = int(float(row.get("frame_id", 0)))
    return (
        frame_bucket(frame, frame_count),
        keyframe_bucket(frame),
        semantic_bucket(row),
        "trajectory",
    )


def select_same_bucket(
    *,
    seq: str,
    selected: list[int],
    cues: list[dict[str, str]],
    cue_lookup: dict[tuple[str, int], dict[str, str]],
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    rng = random.Random(seed)
    frame_count = len(cues)
    selected_set = set(selected)
    picked: list[int] = []
    details: list[dict[str, Any]] = []
    for source_frame in selected:
        source_row = cue_lookup.get((seq, source_frame), {})
        source_bucket = bucket_tuple(source_row, frame_count) if source_row else ("missing", "missing", "missing", "trajectory")
        pools: list[tuple[str, list[dict[str, str]]]] = [
            (
                "exact_frame_key_semantic_operation",
                [
                    row
                    for row in cues
                    if bucket_tuple(row, frame_count) == source_bucket
                    and int(float(row.get("frame_id", 0))) not in selected_set
                    and int(float(row.get("frame_id", 0))) not in picked
                ],
            ),
            (
                "relaxed_frame_semantic_operation",
                [
                    row
                    for row in cues
                    if (
                        bucket_tuple(row, frame_count)[0],
                        bucket_tuple(row, frame_count)[2],
                        bucket_tuple(row, frame_count)[3],
                    )
                    == (source_bucket[0], source_bucket[2], source_bucket[3])
                    and int(float(row.get("frame_id", 0))) not in selected_set
                    and int(float(row.get("frame_id", 0))) not in picked
                ],
            ),
            (
                "relaxed_frame_operation",
                [
                    row
                    for row in cues
                    if (
                        bucket_tuple(row, frame_count)[0],
                        bucket_tuple(row, frame_count)[3],
                    )
                    == (source_bucket[0], source_bucket[3])
                    and int(float(row.get("frame_id", 0))) not in selected_set
                    and int(float(row.get("frame_id", 0))) not in picked
                ],
            ),
            (
                "relaxed_sequence_any",
                [
                    row
                    for row in cues
                    if int(float(row.get("frame_id", 0))) not in selected_set
                    and int(float(row.get("frame_id", 0))) not in picked
                ],
            ),
        ]
        match_level = "none"
        pool: list[dict[str, str]] = []
        for level, candidate_pool in pools:
            if candidate_pool:
                match_level = level
                pool = candidate_pool
                break
        chosen = rng.choice(pool) if pool else source_row
        chosen_frame = int(float(chosen.get("frame_id", source_frame)))
        picked.append(chosen_frame)
        details.append(
            {
                "source_frame": source_frame,
                "control_frame": chosen_frame,
                "source_bucket": "|".join(source_bucket),
                "control_bucket": "|".join(bucket_tuple(chosen, frame_count)) if chosen else "",
                "match_level": match_level,
                "eligible_pool_size": len(pool),
            }
        )
    return picked, details


def make_mask_map(
    *,
    seq: str,
    selected: list[int],
    mask_mode: str,
    cue_lookup: dict[tuple[str, int], dict[str, str]],
) -> dict[int, list[float]]:
    mask_map: dict[int, list[float]] = {}
    for frame in selected:
        mask_map[frame] = h1cfg.mask_for(mask_mode, cue_lookup.get((seq, frame), {}))
    return mask_map


def shuffled_mask_map(mask_map: dict[int, list[float]], seed: int) -> dict[int, list[float]]:
    rng = random.Random(seed)
    frames = list(mask_map)
    masks = [mask_map[frame] for frame in frames]
    rng.shuffle(masks)
    return {frame: mask for frame, mask in zip(frames, masks)}


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


def candidate_config_rows() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(FULL_CONFIG_ROWS)
    return {
        (row.get("policy_id", ""), row.get("seq", "")): row
        for row in rows
        if row.get("policy_id") in FULL_GATE_CANDIDATES and row.get("seq") in SEQUENCES
    }


def build_configs() -> int:
    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    gpu_cycle = ["0", "1", "2", "3"]
    cues = cue_rows_by_seq()
    cue_lookup = cue_by_key(cues)
    candidates = candidate_config_rows()
    missing_candidate_rows = [
        {"candidate_policy_id": policy_id, "seq": seq}
        for policy_id in FULL_GATE_CANDIDATES
        for seq in SEQUENCES
        if (policy_id, seq) not in candidates
    ]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v112tf_h1_t4_ctrl_{seq}"
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

    controls: list[dict[str, Any]] = []
    for candidate_policy_id in FULL_GATE_CANDIDATES:
        for seed in SHUFFLE_SEEDS:
            controls.append(
                {
                    "candidate_policy_id": candidate_policy_id,
                    "control_type": "same_schedule_mask_shuffle",
                    "seed": seed,
                }
            )
        for seed in SAME_BUCKET_SEEDS:
            controls.append(
                {
                    "candidate_policy_id": candidate_policy_id,
                    "control_type": "same_bucket_random",
                    "seed": seed,
                }
            )

    for control in controls:
        candidate_policy_id = str(control["candidate_policy_id"])
        control_type = str(control["control_type"])
        seed = int(control["seed"])
        source_sample = candidates.get((candidate_policy_id, "00"), {})
        mask_mode = source_sample.get("mask_mode", "")
        policy_family = source_sample.get("policy_family", "")
        short = "h1soft" if candidate_policy_id.startswith("H1_") else "t4role"
        control_policy_id = f"CTRL_{short}_{control_type}_seed{seed}"
        for seq in SEQUENCES:
            source = candidates.get((candidate_policy_id, seq), {})
            selected_source = parse_indices(source.get("selected_global_frame_indices", ""))
            if control_type == "same_bucket_random":
                selected, bucket_details = select_same_bucket(
                    seq=seq,
                    selected=selected_source,
                    cues=cues[seq],
                    cue_lookup=cue_lookup,
                    seed=seed,
                )
                mask_map = make_mask_map(seq=seq, selected=selected, mask_mode=mask_mode, cue_lookup=cue_lookup)
            else:
                selected = list(selected_source)
                base_mask_map = make_mask_map(seq=seq, selected=selected, mask_mode=mask_mode, cue_lookup=cue_lookup)
                mask_map = shuffled_mask_map(base_mask_map, seed)
                bucket_details = [
                    {
                        "source_frame": frame,
                        "control_frame": frame,
                        "source_bucket": "",
                        "control_bucket": "",
                        "match_level": "same_schedule_mask_shuffle",
                        "eligible_pool_size": len(selected),
                    }
                    for frame in selected
                ]

            dataset = f"kitti_v112tf_h1_t4_ctrl_{seq}"
            method = f"lingbot_map_v112tf_h1_t4_ctrl_{control_policy_id}_{seq}"
            action_label = f"v112tf_h1_t4_ctrl_{control_policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v112tf_h1_t4_ctrl_{control_policy_id}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = gpu_cycle[row_index % len(gpu_cycle)]
            row_index += 1

            write_text(
                method_path,
                h1cfg.method_yaml(
                    checkpoint=checkpoint,
                    env_name=env_name,
                    use_sdpa=use_sdpa,
                    action_label=action_label,
                    indices=selected,
                    mask_map=mask_map,
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            mask_values = list(mask_map.values())
            cam_mean = sum(mask[0] for mask in mask_values) / len(mask_values) if mask_values else ""
            reg_mean = sum(mask[1] for mask in mask_values) / len(mask_values) if mask_values else ""
            anchor_mean = sum(mask[5] for mask in mask_values) / len(mask_values) if mask_values else ""
            selected_string = ";".join(str(frame) for frame in selected)
            source_string = ";".join(str(frame) for frame in selected_source)
            row = {
                "schema": "acl2_v112tf_h1_t4_semantic_control_policy_row_v1",
                "surface_id": "T",
                "candidate_id": "H1_T4",
                "candidate_policy_id": candidate_policy_id,
                "policy_id": control_policy_id,
                "policy_family": policy_family,
                "control_type": control_type,
                "control_seed": seed,
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage2_action_mode": "context_token_mask",
                "stage4_action_mode": "context_token_mask",
                "selected_count": len(selected),
                "candidate_selected_count": len(selected_source),
                "selected_global_frame_indices": selected_string,
                "candidate_selected_global_frame_indices": source_string,
                "expected_action_field": "forced_context_only",
                "mask_mode": mask_mode,
                "camera_token_weight_mean": cam_mean,
                "register_token_weight_mean": reg_mean,
                "anchor_token_weight_mean": anchor_mean,
                "bucket_match_details_json": json.dumps(bucket_details, sort_keys=True),
                "claim_boundary": "semantic-control subset only; not full random P95 controls.",
                "config": str(config.resolve()),
                "method_config": str(method_path.resolve()),
                "action_file": str(action_file.resolve()),
                "gpu": gpu,
            }
            config_rows.append(row)
            for frame, mask in mask_map.items():
                cue = cue_lookup.get((seq, frame), {})
                frame_rows.append(
                    {
                        "schema": "acl2_v112tf_h1_t4_semantic_control_frame_row_v1",
                        "candidate_policy_id": candidate_policy_id,
                        "control_policy_id": control_policy_id,
                        "control_type": control_type,
                        "control_seed": seed,
                        "seq": seq,
                        "frame": frame,
                        "mask_mode": mask_mode,
                        "dynamic_mass": cue.get("dynamic_mass", ""),
                        "boundary_mass": cue.get("boundary_mass", ""),
                        "weak_context_mass": cue.get("weak_context_mass", ""),
                        "stable_landmark_mass": cue.get("stable_landmark_mass", ""),
                        "token_type_mask": mask_text(mask),
                        "camera_weight": mask[0],
                        "register_weight": mask[1],
                        "anchor_weight": mask[5],
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
                    "schema": "acl2_v112tf_h1_t4_semantic_control_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v112tf_h1_t4_ctrl_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"h1_t4_ctrl_{seq}",
                    "target_kind": "control_dataset_prepare",
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
                f"ACL2_V108_STAGE4_POLICY_ID={control_policy_id} "
                f"ACL2_V108_STAGE4_SURFACE_ID=T "
                f"ACL2_V112TF_H1_T4_POLICY_ID={candidate_policy_id} "
                f"ACL2_V112TF_H1_T4_CONTROL_ID={control_policy_id}"
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
                        "schema": "acl2_v112tf_h1_t4_semantic_control_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v112tf_h1_t4_ctrl_{control_policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"h1_t4_ctrl_{seq}",
                        "target_kind": "control_full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": control_type,
                        "stage4_action_mode": "context_token_mask",
                        "selector": "v112_h1_t4_semantic_control_subset",
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
        "schema": "acl2_v112tf_h1_t4_semantic_control_config_summary_v1",
        "config_ready": not missing_candidate_rows,
        "blocker": "missing_full_gate_candidate_rows" if missing_candidate_rows else "",
        "missing_candidate_rows": missing_candidate_rows,
        "candidate_policy_ids": list(FULL_GATE_CANDIDATES),
        "control_types": ["same_schedule_mask_shuffle", "same_bucket_random"],
        "same_bucket_seeds": list(SAME_BUCKET_SEEDS),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "sequences": list(SEQUENCES),
        "config_rows": len(config_rows),
        "prepare_manifest_rows": len(prepare_rows_by_seq),
        "run_worker_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "gpu_cycle": gpu_cycle,
        "outputs": {
            "action_config_rows": rel(OUT / "action_config_rows.csv"),
            "frame_control_rows": rel(OUT / "frame_control_rows.csv"),
            "run_manifest": rel(OUT / "run_manifest.csv"),
            "summary": rel(OUT / "h1_t4_semantic_control_config_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "frame_control_rows.csv", frame_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_json(OUT / "h1_t4_semantic_control_config_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    return 0 if summary["config_ready"] else 1


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v112tf_h1_t4_ctrl_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v112tf_h1_t4_ctrl_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = stage2m.safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def install_metric_overrides() -> None:
    stage2m.OUT = OUT
    stage2m.CONFIG_ROWS = OUT / "action_config_rows.csv"
    stage2m.RUN_RESULTS = OUT / "run_results.csv"
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = phase_status_for
    original_action_fidelity = stage2m.action_fidelity_row

    def action_fidelity(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
        row = original_action_fidelity(cfg, latest)
        run_name = f"kitti_lingbot_v112tf_h1_t4_ctrl_{cfg['policy_id']}_{cfg['seq']}_run_worker"
        run_row = latest.get((run_name, "run_worker"), {})
        row["schema"] = "acl2_v112tf_h1_t4_semantic_control_action_fidelity_row_v1"
        row["candidate_policy_id"] = cfg.get("candidate_policy_id", "")
        row["control_type"] = cfg.get("control_type", "")
        row["control_seed"] = cfg.get("control_seed", "")
        row["run_worker_returncode"] = run_row.get("returncode", "")
        row["run_worker_duration_sec"] = run_row.get("duration_sec", "")
        row["mask_mode"] = cfg.get("mask_mode", "")
        row["camera_token_weight_mean"] = cfg.get("camera_token_weight_mean", "")
        row["register_token_weight_mean"] = cfg.get("register_token_weight_mean", "")
        row["anchor_token_weight_mean"] = cfg.get("anchor_token_weight_mean", "")
        return row

    stage2m.action_fidelity_row = action_fidelity


def median(values: list[float]) -> float:
    return stage3m.base.median(values)


def mean(values: list[float]) -> float:
    return stage3m.base.mean(values)


def policy_summary_rows(full_rows: list[dict[str, Any]], rolling_rows: list[dict[str, Any]], fidelity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_summary = {row["policy_id"]: row for row in read_csv(FULL_POLICY_SUMMARY)}
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_policy[str(row["policy_id"])].append(row)
    for row in rolling_rows:
        rolling_by_policy[str(row["policy_id"])].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[str(row["policy_id"])].append(row)

    config_by_policy = {row["policy_id"]: row for row in read_csv(OUT / "action_config_rows.csv")}
    out: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        cfg = config_by_policy.get(policy_id, {})
        rels_by_seq = {
            str(row.get("seq", "")): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"), float("nan"))
            for row in rows
        }
        rels = [rels_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        rolling = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"), float("nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        local = [
            safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan"), float("nan"))
            for row in rows
        ]
        candidate_policy_id = cfg.get("candidate_policy_id", "")
        candidate = candidate_summary.get(candidate_policy_id, {})
        candidate_median = safe_float(candidate.get("median_full_rel"), float("nan"))
        this_median = median(rels)
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool_value(row.get("action_fidelity_pass")))
        control_matches = bool(math.isfinite(this_median) and math.isfinite(candidate_median) and this_median >= candidate_median - MATCH_TOL)
        row_out: dict[str, Any] = {
            "schema": "acl2_v112tf_h1_t4_semantic_control_policy_summary_row_v1",
            "candidate_policy_id": candidate_policy_id,
            "control_policy_id": policy_id,
            "control_type": cfg.get("control_type", ""),
            "control_seed": cfg.get("control_seed", ""),
            "sequence_count": len(rows),
            "metric_complete": len(rows) == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows),
            "action_fidelity_pass_count": action_pass_count,
            "all_action_fidelity": action_pass_count == len(SEQUENCES),
            "median_full_rel": this_median,
            "mean_full_rel": mean(rels),
            "min_seq_full_rel": min([value for value in rels if math.isfinite(value)], default=float("nan")),
            "improved_seq_count": sum(1 for value in rels if math.isfinite(value) and value > 0.0),
            "max_harm": stage3m.base.max_rel_harm(rels),
            "rolling_p90_median_rel": median(rolling),
            "local_window_median_harm": stage3m.base.max_rel_harm(local),
            "candidate_median_full_rel": candidate_median,
            "candidate_full_gate_pass": candidate.get("full_gate_pass", ""),
            "median_gap_candidate_minus_control": candidate_median - this_median if math.isfinite(candidate_median) and math.isfinite(this_median) else "",
            "control_matches_candidate_within_tol": control_matches,
            "match_tol": MATCH_TOL,
            "claim_boundary": "control subset only; full random P95 not available from seeds 0..2.",
        }
        for seq in SEQUENCES:
            row_out[f"seq{seq}_full_rel"] = rels_by_seq.get(seq, "")
        out.append(row_out)
    return out


def candidate_control_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        by_candidate[str(row.get("candidate_policy_id", ""))].append(row)
    for candidate_policy_id, rows in sorted(by_candidate.items()):
        same_bucket = [row for row in rows if row.get("control_type") == "same_bucket_random"]
        shuffle = [row for row in rows if row.get("control_type") == "same_schedule_mask_shuffle"]
        all_match = [row for row in rows if bool_value(row.get("control_matches_candidate_within_tol"))]
        best = max(rows, key=lambda row: safe_float(row.get("median_full_rel"), float("nan"))) if rows else {}
        candidate_median = safe_float(best.get("candidate_median_full_rel"), float("nan"))
        out.append(
            {
                "schema": "acl2_v112tf_h1_t4_semantic_control_candidate_summary_row_v1",
                "candidate_policy_id": candidate_policy_id,
                "candidate_median_full_rel": candidate_median,
                "control_policy_count": len(rows),
                "same_bucket_policy_count": len(same_bucket),
                "shuffle_policy_count": len(shuffle),
                "best_control_policy_id": best.get("control_policy_id", ""),
                "best_control_type": best.get("control_type", ""),
                "best_control_seed": best.get("control_seed", ""),
                "best_control_median_full_rel": best.get("median_full_rel", ""),
                "best_gap_candidate_minus_control": best.get("median_gap_candidate_minus_control", ""),
                "any_control_matches_candidate_within_tol": bool(all_match),
                "matching_control_policy_ids": ";".join(str(row.get("control_policy_id", "")) for row in all_match),
                "subset_status": (
                    "CONTROL_MATCH_OR_EXCEED_NO_GO_SUBSET"
                    if all_match
                    else "NO_MATCH_IN_SUBSET_FULL_RANDOM_P95_PENDING"
                ),
            }
        )
    return out


def report_text(summary: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v112TF H1/T4 Semantic Control Subset Report",
        "",
        f"metric_complete: `{summary['metric_complete']}`",
        f"all_action_fidelity: `{summary['all_action_fidelity']}`",
        f"taxonomy: `{summary['taxonomy']}`",
        f"blocker: `{summary['blocker']}`",
        "",
        "## Candidate Summary",
        "",
    ]
    for row in candidate_rows:
        lines.append(
            "- {candidate}: best_control={best} type={ctype} seed={seed} candidate_median={cand} control_median={ctrl} gap={gap} status={status}".format(
                candidate=row.get("candidate_policy_id", ""),
                best=row.get("best_control_policy_id", ""),
                ctype=row.get("best_control_type", ""),
                seed=row.get("best_control_seed", ""),
                cand=row.get("candidate_median_full_rel", ""),
                ctrl=row.get("best_control_median_full_rel", ""),
                gap=row.get("best_gap_candidate_minus_control", ""),
                status=row.get("subset_status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a semantic-control subset using same-schedule mask shuffle seed0 and same-bucket random seeds0..2. It can produce a No-Go if a matched control catches the candidate, but it cannot prove semantic causality without the larger pre-registered random/control set.",
        ]
    )
    return "\n".join(lines)


def build_metrics() -> int:
    install_metric_overrides()
    config_rows = read_csv(OUT / "action_config_rows.csv")
    latest = stage2m.latest_run_results(read_csv(OUT / "run_results.csv"))
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        for row in rows:
            row["schema"] = str(row.get("schema", "")).replace("acl2_v109tf_stage2", "acl2_v112tf_h1_t4_semantic_control")
            cfg = next((cfg for cfg in config_rows if cfg["policy_id"] == row.get("policy_id") and cfg["seq"] == row.get("seq")), {})
            row["candidate_policy_id"] = cfg.get("candidate_policy_id", "")
            row["control_type"] = cfg.get("control_type", "")
            row["control_seed"] = cfg.get("control_seed", "")
            row["mask_mode"] = cfg.get("mask_mode", "")
            row["camera_token_weight_mean"] = cfg.get("camera_token_weight_mean", "")
            row["register_token_weight_mean"] = cfg.get("register_token_weight_mean", "")
            row["anchor_token_weight_mean"] = cfg.get("anchor_token_weight_mean", "")

    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    candidate_rows = candidate_control_rows(policy_rows)
    observed_counts: dict[str, int] = {}
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            phase = str(row.get("phase", ""))
            observed_counts[phase] = observed_counts.get(phase, 0) + 1

    metric_complete = len(full_rows) == len(config_rows) and all(bool_value(row.get("metric_available")) for row in full_rows)
    all_action = len(fidelity_rows) == len(config_rows) and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows)
    any_match = any(bool_value(row.get("any_control_matches_candidate_within_tol")) for row in candidate_rows)
    taxonomy = (
        "H1_T4_SEMANTIC_CONTROL_SUBSET_MATCH_NO_GO"
        if any_match
        else "H1_T4_SEMANTIC_CONTROL_SUBSET_NO_MATCH_FULL_P95_PENDING"
    )
    blocker = (
        "matched_control_reaches_candidate_within_tol"
        if any_match
        else "full_random_p95_and_additional_semantic_controls_pending"
    )
    summary = {
        "schema": "acl2_v112tf_h1_t4_semantic_control_metric_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "taxonomy": taxonomy,
        "blocker": blocker,
        "semantic_causality_claim_allowed": False,
        "semantic_causality_claim_blocker": "Control subset is not the full pre-registered random/same-bucket/semantic-shuffle control set.",
        "observed_prepare_count": observed_counts.get("prepare", 0),
        "observed_run_worker_count": observed_counts.get("run_worker", 0),
        "observed_evaluate_count": observed_counts.get("evaluate", 0),
        "observed_report_count": observed_counts.get("report", 0),
        "expected_run_worker_count": len(config_rows),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_summary_row_count": len(policy_rows),
        "candidate_control_summary_row_count": len(candidate_rows),
        "matching_candidate_policy_ids": [
            row["candidate_policy_id"]
            for row in candidate_rows
            if bool_value(row.get("any_control_matches_candidate_within_tol"))
        ],
        "outputs": {
            "full_metric_rows": rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "policy_summary_rows": rel(OUT / "policy_summary_rows.csv"),
            "candidate_control_summary_rows": rel(OUT / "candidate_control_summary_rows.csv"),
            "report": rel(OUT / "H1_T4_SEMANTIC_CONTROL_SUBSET_REPORT.md"),
            "summary": rel(OUT / "h1_t4_semantic_control_metric_summary.json"),
        },
    }
    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "policy_summary_rows.csv", policy_rows)
    write_csv(OUT / "candidate_control_summary_rows.csv", candidate_rows)
    write_json(OUT / "h1_t4_semantic_control_metric_summary.json", summary)
    write_text(OUT / "H1_T4_SEMANTIC_CONTROL_SUBSET_REPORT.md", report_text(summary, candidate_rows))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="store_true", help="summarize completed control runs")
    args = parser.parse_args()
    return build_metrics() if args.metrics else build_configs()


if __name__ == "__main__":
    raise SystemExit(main())
