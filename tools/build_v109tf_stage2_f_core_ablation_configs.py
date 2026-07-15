#!/usr/bin/env python3
"""Generate ACL2 v109TF Stage2 F-surface full KITTI core ablation configs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage3_operation_cue_screen as stage3  # noqa: E402
import build_v108tf_stage4_full_kitti_pilot_configs as stage4  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
OUT = RESULT_ROOT / "stage2_f_core_ablation"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

V108 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
V108_STAGE3 = V108 / "stage3_operation_cue_screen"
V108_STAGE5 = V108 / "stage5_full_kitti_00_01_02_05_validation"

SEQUENCES = ("00", "01", "02", "05")
SURFACE = "F"
ACTION_MODE = "anchor_special_only"
GPU_CYCLE = ("0", "1", "2", "3", "4")

POLICIES = (
    ("F1_semantic_plus_internal", "semantic_plus_internal", "stage3_existing_semantic_plus_internal"),
    ("F2_internal_only", "internal_only", "stage3_existing_internal_only"),
    ("F3_semantic_only", "semantic_only", "stage3_existing_semantic_only"),
    ("F4_semantic_shuffle_seed0", "semantic_shuffle_seed0", "stage3_existing_semantic_shuffle"),
    ("F5_semantic_shuffle_seed1", "semantic_shuffle_seed1", "seeded_score_shuffle_offset_7"),
    ("F6_semantic_shuffle_seed2", "semantic_shuffle_seed2", "seeded_score_shuffle_offset_17"),
    ("F7_role_rotation", "role_rotation", "role_rotated_balanced_score_count_matched"),
    ("F8_same_count_random_seed0", "same_count_random_seed0", "stage3_existing_same_count_random"),
    ("F9_same_count_random_seed1", "same_count_random_seed1", "deterministic_same_count_seed1"),
    ("F10_same_count_random_seed2", "same_count_random_seed2", "deterministic_same_count_seed2"),
    ("F11_low_risk_reverse", "low_risk_reverse", "stage3_existing_low_risk_reverse"),
)

V108_POLICY_REUSE_MAP = {
    "F1_semantic_plus_internal": "F1_semantic_plus_internal",
    "F2_internal_only": "F1_internal_only",
    "F3_semantic_only": "F1_semantic_only",
    "F4_semantic_shuffle_seed0": "F1_semantic_shuffle",
    "F8_same_count_random_seed0": "F1_same_count_random",
    "F11_low_risk_reverse": "F1_low_risk_reverse",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def frame_counts() -> dict[str, int]:
    return {
        row["seq"]: int(float(row["frames"]))
        for row in read_csv(STAGE0 / "full_kitti_baseline_table.csv")
        if row.get("seq") in SEQUENCES
    }


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_stage3_selection(v108_policy_id: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for row in read_csv(V108_STAGE3 / "surface_policy_frame_rows.csv"):
        if row.get("surface_id") != SURFACE or row.get("policy_id") != v108_policy_id:
            continue
        out[row["seq_id"]].append(int(float(row["frame_id"])))
    return {seq: sorted(set(vals)) for seq, vals in out.items()}


def cases_by_seq() -> dict[str, list[dict[str, Any]]]:
    cases = stage3.build_surface_cases().get(SURFACE, [])
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case["seq_id"] in SEQUENCES:
            out[case["seq_id"]].append(case)
    for seq in out:
        out[seq].sort(key=lambda row: int(row["frame_id"]))
    return out


def internal_threshold(cases: list[dict[str, Any]]) -> float:
    return stage3.quantile([stage3.fnum(case.get("special_token_count", "")) for case in cases], 0.50)


def semantic_thresholds(cases_by_sequence: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for seq, cases in cases_by_sequence.items():
        values = [stage3.fnum(case.get("Q_ref_sem_balanced", "")) for case in cases]
        thresholds[seq] = stage3.quantile(values, 0.25)
    return thresholds


def count_by_seq(selection: dict[str, list[int]]) -> dict[str, int]:
    return {seq: len(frames) for seq, frames in selection.items()}


def select_semantic_shuffle_seed(
    cases_by_sequence: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    thresholds: dict[str, float],
    internal_cut: float,
    offset: int,
    salt: str,
) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for seq, cases in cases_by_sequence.items():
        n = counts.get(seq, 0)
        if n <= 0:
            selected[seq] = []
            continue
        ordered = sorted(cases, key=lambda case: int(case["frame_id"]))
        scores = [stage3.fnum(case.get("Q_ref_sem_balanced", "")) for case in ordered]
        shifted = scores[offset % len(scores):] + scores[: offset % len(scores)] if scores else []
        candidates: list[dict[str, Any]] = []
        fill: list[dict[str, Any]] = []
        for case, shifted_score in zip(ordered, shifted):
            int_ok = stage3.fnum(case.get("special_token_count", "")) >= internal_cut
            if int_ok:
                fill.append(case)
            if int_ok and shifted_score <= thresholds[seq]:
                candidates.append(case)
        if len(candidates) < n:
            existing = {int(case["frame_id"]) for case in candidates}
            candidates.extend(case for case in fill if int(case["frame_id"]) not in existing)
        ranked = sorted(
            candidates,
            key=lambda case: hashlib.sha256(f"{salt}|{seq}|{case['frame_id']}".encode("utf-8")).hexdigest(),
        )
        selected[seq] = sorted({int(case["frame_id"]) for case in ranked[:n]})
    return selected


def role_rotated_score(case: dict[str, Any]) -> float:
    stable_rot = stage3.fnum(case.get("dynamic_mass", "")) + stage3.fnum(case.get("boundary_mass", ""))
    risk_rot = (
        stage3.fnum(case.get("stable_structure_mass", ""))
        + stage3.fnum(case.get("weak_context_mass", ""))
        + stage3.fnum(case.get("sky_lowobs_mass", ""))
    )
    continuity = stage3.fnum(case.get("semantic_continuity_score", ""))
    boundary_risk = stage3.fnum(case.get("semantic_boundary_risk", ""))
    return stable_rot + 0.5 * continuity - risk_rot - 0.5 * boundary_risk


def select_role_rotation(cases_by_sequence: dict[str, list[dict[str, Any]]], counts: dict[str, int], internal_cut: float) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for seq, cases in cases_by_sequence.items():
        n = counts.get(seq, 0)
        eligible = [case for case in cases if stage3.fnum(case.get("special_token_count", "")) >= internal_cut]
        ranked = sorted(
            eligible,
            key=lambda case: (
                role_rotated_score(case),
                hashlib.sha256(f"role_rotation|{seq}|{case['frame_id']}".encode("utf-8")).hexdigest(),
            ),
        )
        selected[seq] = sorted({int(case["frame_id"]) for case in ranked[:n]})
    return selected


def deterministic_same_count(cases_by_sequence: dict[str, list[dict[str, Any]]], counts: dict[str, int], salt: str) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for seq, cases in cases_by_sequence.items():
        n = counts.get(seq, 0)
        ranked = sorted(
            cases,
            key=lambda case: hashlib.sha256(f"{salt}|{seq}|{case['frame_id']}".encode("utf-8")).hexdigest(),
        )
        selected[seq] = sorted({int(case["frame_id"]) for case in ranked[:n]})
    return selected


def build_policy_source_selections() -> dict[str, dict[str, list[int]]]:
    cases = cases_by_seq()
    all_cases = [case for group in cases.values() for case in group]
    semantic_plus = existing_stage3_selection("F1_semantic_plus_internal")
    counts = count_by_seq(semantic_plus)
    thresholds = semantic_thresholds(cases)
    internal_cut = internal_threshold(all_cases)
    selections = {
        "F1_semantic_plus_internal": semantic_plus,
        "F2_internal_only": existing_stage3_selection("F1_internal_only"),
        "F3_semantic_only": existing_stage3_selection("F1_semantic_only"),
        "F4_semantic_shuffle_seed0": existing_stage3_selection("F1_semantic_shuffle"),
        "F5_semantic_shuffle_seed1": select_semantic_shuffle_seed(cases, counts, thresholds, internal_cut, 7, "v109tf_F5_semantic_shuffle_seed1"),
        "F6_semantic_shuffle_seed2": select_semantic_shuffle_seed(cases, counts, thresholds, internal_cut, 17, "v109tf_F6_semantic_shuffle_seed2"),
        "F7_role_rotation": select_role_rotation(cases, counts, internal_cut),
        "F8_same_count_random_seed0": existing_stage3_selection("F1_same_count_random"),
        "F9_same_count_random_seed1": deterministic_same_count(cases, counts, "v109tf_F9_same_count_random_seed1"),
        "F10_same_count_random_seed2": deterministic_same_count(cases, counts, "v109tf_F10_same_count_random_seed2"),
        "F11_low_risk_reverse": existing_stage3_selection("F1_low_risk_reverse"),
    }
    return selections


def source_frame_rows(policy_selections: dict[str, dict[str, list[int]]]) -> list[dict[str, Any]]:
    frame_map = {
        (row["seq_id"], int(float(row["frame_id"]))): row
        for row in read_csv(V108 / "stage2_semantic_cue_bank/frame_semantic_summary.csv")
    }
    rows: list[dict[str, Any]] = []
    family_by_policy = {policy_id: family for policy_id, family, _selector in POLICIES}
    for policy_id, by_seq in policy_selections.items():
        for seq, frames in by_seq.items():
            for rank, frame_id in enumerate(frames, start=1):
                frame = frame_map.get((seq, frame_id), {})
                rows.append(
                    {
                        "schema": "acl2_v109tf_stage2_policy_source_frame_row_v1",
                        "surface_id": SURFACE,
                        "policy_id": policy_id,
                        "policy_family": family_by_policy[policy_id],
                        "seq": seq,
                        "source_frame": frame_id,
                        "source_rank": rank,
                        "Q_ref_sem_balanced": frame.get("Q_ref_sem_balanced", ""),
                        "Q_ref_sem_risk_strict": frame.get("Q_ref_sem_risk_strict", ""),
                        "Q_ref_sem_stable_strict": frame.get("Q_ref_sem_stable_strict", ""),
                        "stable_structure_mass": frame.get("stable_structure_mass", ""),
                        "dynamic_mass": frame.get("dynamic_mass", ""),
                        "boundary_mass": frame.get("boundary_mass", ""),
                        "weak_context_mass": frame.get("weak_context_mass", ""),
                        "road_ground_mass": frame.get("road_ground_mass", ""),
                        "sky_lowobs_mass": frame.get("sky_lowobs_mass", ""),
                        "semantic_trust_mean": frame.get("semantic_trust_mean", ""),
                        "semantic_continuity_score": frame.get("semantic_continuity_score", ""),
                    }
                )
    return rows


def expected_action_field(_action_mode: str) -> str:
    return "forced_anchor_only"


def build() -> dict[str, Any]:
    env = stage4.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frames_by_seq = frame_counts()
    policy_source = build_policy_source_selections()

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v109tf_stage2_fcore_fullseq_{seq}"
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

    snap_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    action_config: dict[str, Any] = {"schema": "acl2_v109tf_stage2_action_config_v1", "policies": []}
    row_index = 0
    blockers: list[str] = []

    current_action_code_hash = file_sha256(ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py")
    current_config_builder_hash = file_sha256(Path(__file__))

    reuse_rows.append(
        {
            "schema": "acl2_v109tf_stage2_reuse_manifest_row_v1",
            "v109_policy_id": "F0_no_action",
            "policy_family": "NO_ACTION",
            "reuse_status": "reused_frozen_v105_baseline",
            "source_artifact": "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv",
            "reason": "No runtime action; v109 Stage0 already exact-checked baseline ATE.",
        }
    )

    for policy_id, policy_family, selector in POLICIES:
        v108_policy = V108_POLICY_REUSE_MAP.get(policy_id, "")
        reuse_rows.append(
            {
                "schema": "acl2_v109tf_stage2_reuse_manifest_row_v1",
                "v109_policy_id": policy_id,
                "policy_family": policy_family,
                "v108_equivalent_policy_id": v108_policy,
                "reuse_status": "not_reused_rerun_for_v109_stage2",
                "source_artifact": rel(V108_STAGE5 / "full_sequence_metric_rows.csv") if v108_policy else "",
                "reason": "Rerun avoids silent config/code-hash mixing; v108 rows remain prior evidence only.",
                "current_lingbot_map_sha256": current_action_code_hash,
                "current_stage2_config_builder_sha256": current_config_builder_hash,
            }
        )

    for policy_id, policy_family, selector in POLICIES:
        for seq in SEQUENCES:
            source_indices = policy_source.get(policy_id, {}).get(seq, [])
            frames = frames_by_seq[seq]
            snapped_indices, rows = stage4.snap_to_nearest_base_keyframe(source_indices, frames)
            if source_indices and not snapped_indices:
                blockers.append(f"all_selected_frames_failed_keyframe_snap_{policy_id}_{seq}")
            for row in rows:
                snap_rows.append(
                    {
                        "schema": "acl2_v109tf_stage2_keyframe_snap_row_v1",
                        "surface_id": SURFACE,
                        "policy_id": policy_id,
                        "policy_family": policy_family,
                        "seq": seq,
                        **row,
                    }
                )

            dataset = f"kitti_v109tf_stage2_fcore_fullseq_{seq}"
            action_label = f"v109tf_stage2_{policy_id}"
            method = f"lingbot_map_v109tf_stage2_{policy_id}_{seq}"
            config = CONFIG_ROOT / f"kitti_lingbot_v109tf_stage2_{policy_id}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = GPU_CYCLE[row_index % len(GPU_CYCLE)]
            row_index += 1

            write_text(
                method_path,
                stage4.method_yaml(
                    checkpoint,
                    env_name,
                    use_sdpa,
                    action_label,
                    ACTION_MODE,
                    snapped_indices,
                    None,
                ),
            )
            write_text(
                config,
                stage4.run_config_yaml(dataset, method).replace(str(stage4.WORKSPACE.resolve()), str(WORKSPACE.resolve())),
            )
            selected_string = ";".join(str(x) for x in snapped_indices)
            policy = {
                "schema": "acl2_v109tf_stage2_policy_row_v1",
                "surface_id": SURFACE,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "selector": selector,
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
                "expected_action_field": expected_action_field(ACTION_MODE),
                "runtime_boundary": "LingBot internal F-surface anchor/special-token action; no output post-processing.",
                "reuse_decision": "rerun_v109_stage2",
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
                f"ACL2_V108_STAGE4_SURFACE_ID={SURFACE}"
            )
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v109tf_stage2_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v109tf_stage2_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"fullseq_{seq}",
                    "target_kind": "full_sequence_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage2_action_mode": "dataset_prepare",
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
                        "schema": "acl2_v109tf_stage2_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v109tf_stage2_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"fullseq_{seq}",
                        "target_kind": "full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy_family,
                        "stage2_action_mode": ACTION_MODE,
                        "stage4_action_mode": ACTION_MODE,
                        "selector": selector,
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
    source_rows = source_frame_rows(policy_source)

    write_csv(OUT / "stage2_reuse_manifest.csv", reuse_rows)
    write_csv(OUT / "policy_source_frame_rows.csv", source_rows)
    write_csv(OUT / "keyframe_snap_rows.csv", snap_rows)
    write_csv(OUT / "full_sequence_policy_rows.csv", policy_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_json(OUT / "action_config.json", action_config)
    summary = {
        "schema": "acl2_v109tf_stage2_config_generation_summary_v1",
        "stage2_config_ready": not blockers,
        "blockers": blockers,
        "surface_id": SURFACE,
        "sequences": list(SEQUENCES),
        "policy_count": len(POLICIES),
        "action_policy_rows": len(policy_rows),
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_rows": len(prepare_rows_by_seq),
        "run_worker_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "no_action_control_source": rel(STAGE0 / "full_kitti_baseline_table.csv"),
        "reuse_policy": "F0 no-action reuses frozen baseline; F1-F11 are rerun in v109 Stage2 to avoid silent v108 hash mixing.",
        "trajectory_only_env_note": "Commands set ACL2_V108_STAGE4_POLICY_ID for LingBot trajectory-only output optimization; this does not change model forward or pose/depth computation.",
        "workspace": rel(WORKSPACE),
        "raw_action": rel(RAW_ACTION),
        "outputs": {
            "stage2_reuse_manifest": rel(OUT / "stage2_reuse_manifest.csv"),
            "policy_source_frame_rows": rel(OUT / "policy_source_frame_rows.csv"),
            "keyframe_snap_rows": rel(OUT / "keyframe_snap_rows.csv"),
            "action_config_rows": rel(OUT / "action_config_rows.csv"),
            "run_manifest": rel(OUT / "run_manifest.csv"),
            "config_generation_summary": rel(OUT / "config_generation_summary.json"),
        },
    }
    write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
