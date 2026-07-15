#!/usr/bin/env python3
"""Generate ACL2 v110R Stage2 candidates and Stage3 00/02 pilot manifest."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import build_v108tf_stage4_full_kitti_pilot_configs as v108_stage4  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
OUT = RESULT_ROOT / "stage2_candidate_generation"
STAGE3 = RESULT_ROOT / "stage3_pilot_00_02"
CONFIG_ROOT = STAGE3 / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = STAGE3 / "workspace"
RAW_ACTION = STAGE3 / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
V108_STAGE3 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage3_operation_cue_screen"

SEQUENCES = ("00", "02")
HEADLOCAL_ALL_HEADS = list(range(16))
POLICY_FAMILIES = (
    "semantic_plus_internal",
    "internal_only",
    "semantic_only",
    "semantic_shuffle",
    "same_count_random",
    "low_risk_reverse",
)

RUNNABLE_CANDIDATES = {
    "A2": {
        "surface_id": "A",
        "candidate_name": "high_risk_reference_skip",
        "source_policy_prefix": "A1",
        "stage3_action_mode": "force_non_keyframe",
        "expected_action_field": "forced_non_keyframe",
        "implementation_boundary": "force high-risk snapped base keyframes non-keyframe; no stable replacement/protect hook claimed",
        "plan_status": "runtime_knob_available_partial",
    },
    "B1": {
        "surface_id": "B",
        "candidate_name": "high_risk_no_append",
        "source_policy_prefix": "B1",
        "stage3_action_mode": "force_non_keyframe",
        "expected_action_field": "forced_non_keyframe",
        "implementation_boundary": "skip cache append by forcing selected high-risk base keyframes non-keyframe",
        "plan_status": "runtime_knob_available",
    },
    "E1": {
        "surface_id": "E",
        "candidate_name": "local_preserve_anchor_reference_block",
        "source_policy_prefix": "E1",
        "stage3_action_mode": "v106_context_only_with_local_preserve",
        "expected_action_field": "headlocal_action_enabled",
        "implementation_boundary": "head-local local-preserve/reference block on selected base keyframes; all 16 heads",
        "plan_status": "runtime_knob_available_high_risk",
    },
    "F2": {
        "surface_id": "F",
        "candidate_name": "semantic_high_risk_special_token_block",
        "source_policy_prefix": "F1",
        "stage3_action_mode": "anchor_special_only",
        "expected_action_field": "forced_anchor_only",
        "implementation_boundary": "force selected high-risk base keyframes to anchor-special-only context append",
        "plan_status": "runtime_knob_available_partial",
    },
}

PLANNED_BUT_NOT_RUNNABLE = [
    ("A1", "A", "stable_reference_preference", "replacement/protect stable keyframe hook not exposed"),
    ("A3", "A", "f19_compatible_stable_protect", "protect existing keyframe hook not exposed"),
    ("A4", "A", "road_ground_weak_context_avoid", "replacement-to-stable hook not exposed"),
    ("B2", "B", "stable_protect_append", "protect/keep-longer append hook not exposed"),
    ("B3", "B", "f19_plus_high_risk_no_append", "composition deferred until single-surface Stage3 evidence"),
    ("B4", "B", "low_risk_reverse_no_append", "represented by low_risk_reverse control family for B1"),
    ("E2", "E", "local_preserve_trajectory_block", "independent trajectory-write block hook not exposed"),
    ("E3", "E", "local_preserve_anchor_trajectory_block", "combined anchor+trajectory block needs trajectory hook"),
    ("E4", "E", "f19_plus_local_preserve", "composition deferred until single-surface Stage3 evidence"),
    ("F1", "F", "anchor_special_only_baseline", "covered as internal/schedule controls under F2 runtime mode"),
    ("F3", "F", "stable_special_token_protect", "special-token protect hook not exposed"),
    ("F4", "F", "f19_plus_special_token_safety", "composition deferred until single-surface Stage3 evidence"),
]

CONTROL_COVERAGE = {
    "NO_ACTION": "covered_by_frozen_v105_baseline",
    "INTERNAL_ONLY": "generated",
    "SEMANTIC_ONLY": "generated",
    "SEMANTIC_PLUS_INTERNAL": "generated",
    "SEMANTIC_SHUFFLE": "generated",
    "ROLE_ROTATION": "not_generated_this_pass_no_existing_v108_stage3_rows",
    "SAME_COUNT_RANDOM": "generated",
    "SAME_BUCKET_RANDOM": "not_generated_this_pass_requires_new_bucketed_selector",
    "LOW_RISK_REVERSE": "generated",
    "SCHEDULE_ONLY": "not_generated_this_pass_same_count_random_is_nearest_available_schedule_control",
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
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frame_counts() -> dict[str, int]:
    rows = read_csv(STAGE0 / "frozen_baseline_table.csv")
    return {row["seq"]: int(float(row["frames"])) for row in rows if row.get("seq") in SEQUENCES}


def allowed_surface_rows() -> dict[str, dict[str, str]]:
    return {row["surface_id"]: row for row in read_csv(STAGE0 / "allowed_action_surfaces.csv")}


def source_selected_frames() -> dict[tuple[str, str, str], list[int]]:
    out: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in read_csv(V108_STAGE3 / "surface_policy_frame_rows.csv"):
        surface = row.get("surface_id", "")
        policy_family = row.get("policy_family", "")
        seq = row.get("seq_id", "")
        if policy_family not in POLICY_FAMILIES or seq not in SEQUENCES:
            continue
        out[(surface, row["policy_id"], seq)].append(int(float(row["frame_id"])))
    return {key: sorted(set(values)) for key, values in out.items()}


def source_policy_rows() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row.get("surface_id", ""), row.get("policy_id", "")): row
        for row in read_csv(V108_STAGE3 / "surface_policy_rows.csv")
    }


def stage3_run_config_yaml(dataset: str, method: str) -> str:
    return v108_stage4.run_config_yaml(dataset, method).replace(
        str(v108_stage4.WORKSPACE.resolve()),
        str(WORKSPACE.resolve()),
    )


def expected_action_field(action_mode: str) -> str:
    return {
        "force_non_keyframe": "forced_non_keyframe",
        "anchor_special_only": "forced_anchor_only",
        "context_only_special": "forced_context_only",
        "v106_context_only_with_local_preserve": "headlocal_action_enabled",
    }[action_mode]


def candidate_surface_configs(allowed: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id, spec in RUNNABLE_CANDIDATES.items():
        surface = spec["surface_id"]
        allowed_row = allowed.get(surface, {})
        rows.append(
            {
                "schema": "acl2_v110r_stage2_candidate_surface_config_row_v1",
                "candidate_id": candidate_id,
                "surface_id": surface,
                "candidate_name": spec["candidate_name"],
                "candidate_status": "generated_for_stage3_full_00_02_pilot",
                "stage3_action_mode": spec["stage3_action_mode"],
                "expected_action_field": spec["expected_action_field"],
                "source_policy_prefix": spec["source_policy_prefix"],
                "implementation_boundary": spec["implementation_boundary"],
                "stage0_v110_status": allowed_row.get("v110_status", ""),
                "stage0_claim_boundary": allowed_row.get("claim_boundary", ""),
                "stage0_note": allowed_row.get("note", ""),
            }
        )
    for candidate_id, surface, name, blocker in PLANNED_BUT_NOT_RUNNABLE:
        allowed_row = allowed.get(surface, {})
        rows.append(
            {
                "schema": "acl2_v110r_stage2_candidate_surface_config_row_v1",
                "candidate_id": candidate_id,
                "surface_id": surface,
                "candidate_name": name,
                "candidate_status": "not_generated_for_stage3_this_pass",
                "stage3_action_mode": "",
                "expected_action_field": "",
                "source_policy_prefix": "",
                "implementation_boundary": blocker,
                "stage0_v110_status": allowed_row.get("v110_status", ""),
                "stage0_claim_boundary": allowed_row.get("claim_boundary", ""),
                "stage0_note": allowed_row.get("note", ""),
            }
        )
    return rows


def cd_hook_audit_rows(allowed: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface in ("C", "D"):
        row = allowed.get(surface, {})
        rows.append(
            {
                "schema": "acl2_v110r_stage2_cd_hook_audit_row_v1",
                "surface_id": surface,
                "operation_type": row.get("operation_type", ""),
                "implementation_status": row.get("implementation_status", ""),
                "has_existing_runtime_knob": row.get("has_existing_runtime_knob", ""),
                "new_hook_needed": row.get("new_hook_needed", ""),
                "full_sequence_pilot_allowed": row.get("full_sequence_pilot_allowed", ""),
                "v110_status": row.get("v110_status", ""),
                "stage2_decision": "hook_audit_only_no_full_pilot_claim",
                "blocker": "minimal_runtime_hook_not_implemented_in_this_stage2_generator",
                "next_required_work": (
                    "implement no-op parity hook plus provenance rows before any C/D geometry claim"
                ),
            }
        )
    return rows


def control_coverage_rows() -> list[dict[str, Any]]:
    return [
        {
            "schema": "acl2_v110r_stage2_control_coverage_row_v1",
            "control_family": family,
            "coverage_status": status,
            "claim_boundary": (
                "required_before_semantic_causality_claim"
                if status.startswith("not_generated")
                else "available_for_stage3_pilot_or_frozen_baseline"
            ),
        }
        for family, status in CONTROL_COVERAGE.items()
    ]


def build() -> dict[str, Any]:
    env = v108_stage4.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frames_by_seq = frame_counts()
    selected = source_selected_frames()
    source_rows = source_policy_rows()
    allowed = allowed_surface_rows()
    gpu_cycle = ["0", "1", "2", "3", "4"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE3.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v110r_stage3_fullseq_{seq}"
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
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    action_config: dict[str, Any] = {"schema": "acl2_v110r_stage2_action_config_v1", "policies": []}
    row_index = 0
    missing_source_rows: list[dict[str, Any]] = []

    for candidate_id, spec in RUNNABLE_CANDIDATES.items():
        surface = spec["surface_id"]
        source_prefix = spec["source_policy_prefix"]
        action_mode = spec["stage3_action_mode"]
        for family in POLICY_FAMILIES:
            source_policy_id = f"{source_prefix}_{family}"
            policy_id = f"{candidate_id}_{family}"
            source_policy = source_rows.get((surface, source_policy_id), {})
            if not source_policy:
                missing_source_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "surface_id": surface,
                        "policy_family": family,
                        "source_policy_id": source_policy_id,
                    }
                )
            for seq in SEQUENCES:
                source_indices = selected.get((surface, source_policy_id, seq), [])
                frames = frames_by_seq[seq]
                snapped_indices, rows = v108_stage4.snap_to_nearest_base_keyframe(source_indices, frames)
                for row in rows:
                    snap_rows.append(
                        {
                            "schema": "acl2_v110r_stage2_keyframe_snap_row_v1",
                            "candidate_id": candidate_id,
                            "surface_id": surface,
                            "source_policy_id": source_policy_id,
                            "policy_id": policy_id,
                            "policy_family": family,
                            "seq": seq,
                            **row,
                        }
                    )
                if action_mode == "v106_context_only_with_local_preserve":
                    force_indices: list[int] = []
                    headlocal_map = {idx: HEADLOCAL_ALL_HEADS for idx in snapped_indices}
                    headlocal_all_heads = ";".join(str(x) for x in HEADLOCAL_ALL_HEADS)
                else:
                    force_indices = snapped_indices
                    headlocal_map = None
                    headlocal_all_heads = ""

                dataset = f"kitti_v110r_stage3_fullseq_{seq}"
                action_label = f"v110r_stage3_{policy_id}"
                method = f"lingbot_map_v110r_stage3_{policy_id}_{seq}"
                config = CONFIG_ROOT / f"kitti_lingbot_v110r_stage3_{policy_id}_{seq}.yaml"
                method_path = METHOD_DIR / f"{method}.yaml"
                action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
                gpu = gpu_cycle[row_index % len(gpu_cycle)]
                row_index += 1

                write_text(
                    method_path,
                    v108_stage4.method_yaml(
                        checkpoint,
                        env_name,
                        use_sdpa,
                        action_label,
                        action_mode,
                        force_indices,
                        headlocal_map,
                    ),
                )
                write_text(config, stage3_run_config_yaml(dataset, method))
                selected_string = ";".join(str(x) for x in snapped_indices)
                policy = {
                    "schema": "acl2_v110r_stage2_candidate_policy_row_v1",
                    "candidate_id": candidate_id,
                    "candidate_name": spec["candidate_name"],
                    "surface_id": surface,
                    "source_policy_id": source_policy_id,
                    "policy_id": policy_id,
                    "policy_family": family,
                    "cue_family": source_policy.get("cue_family", ""),
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": action_label,
                    "stage2_action_mode": action_mode,
                    "stage3_action_mode": action_mode,
                    "stage4_action_mode": action_mode,
                    "source_selected_count": len(source_indices),
                    "source_selected_global_frame_indices": ";".join(str(x) for x in source_indices),
                    "selected_count": len(snapped_indices),
                    "selected_global_frame_indices": selected_string,
                    "headlocal_all_heads": headlocal_all_heads,
                    "frames": frames,
                    "full_sequence_keyframe_interval": v108_stage4.keyframe_interval(frames),
                    "snap_radius": max(1, v108_stage4.keyframe_interval(frames) // 2),
                    "expected_action_field": expected_action_field(action_mode),
                    "runtime_boundary": spec["implementation_boundary"],
                    "pilot_scope": "full KITTI 00/02 Stage3 pilot",
                    "control_gap_note": (
                        "role_rotation/same_bucket_random/schedule_only not generated in this pass; no semantic causality claim allowed from Stage3 alone"
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
                action_config["policies"].append(
                    {**policy, "config": rel(config), "method_config": rel(method_path), "action_file": rel(action_file)}
                )

                prefix = v108_stage4.command_prefix(conda_path, pythonpath, gpu)
                action_env = (
                    f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                    f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                    f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                    f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                    f"ACL2_V105_GCA_TRACE_METHOD={method} "
                    f"ACL2_V108_STAGE4_POLICY_ID={policy_id} "
                    f"ACL2_V108_STAGE4_SURFACE_ID={surface} "
                    f"ACL2_V110R_STAGE3_POLICY_ID={policy_id} "
                    f"ACL2_V110R_STAGE3_CANDIDATE_ID={candidate_id}"
                )
                prepare_command = (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python prepare.py --config {config.resolve()} --force"
                )
                prepare_rows_by_seq.setdefault(
                    seq,
                    {
                        "schema": "acl2_v110r_stage3_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v110r_stage3_prepare_{seq}",
                        "phase": "prepare",
                        "target_id": f"stage3_fullseq_{seq}",
                        "target_kind": "full_sequence_dataset_prepare",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": "dataset_prepare_once",
                        "action_family": "prepare",
                        "stage2_action_mode": "dataset_prepare",
                        "stage3_action_mode": "dataset_prepare",
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
                            "schema": "acl2_v110r_stage3_manifest_row_v1",
                            "run_name": f"kitti_lingbot_v110r_stage3_{policy_id}_{seq}_{phase}",
                            "phase": phase,
                            "target_id": f"stage3_fullseq_{seq}",
                            "target_kind": "full_sequence",
                            "seq": seq,
                            "dataset": dataset,
                            "method": method,
                            "action_name": action_label,
                            "action_family": family,
                            "stage2_action_mode": action_mode,
                            "stage3_action_mode": action_mode,
                            "stage4_action_mode": action_mode,
                            "selector": "v108_stage3_policy_frames_snapped_to_v110_stage3_full_sequence_base_keyframes",
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

    manifest_rows = [prepare_rows_by_seq[seq] for seq in SEQUENCES] + manifest_rows
    surface_rows = candidate_surface_configs(allowed)
    cd_rows = cd_hook_audit_rows(allowed)
    coverage_rows = control_coverage_rows()

    write_csv(OUT / "candidate_surface_configs.csv", surface_rows)
    write_csv(OUT / "candidate_policy_rows.csv", policy_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "keyframe_snap_rows.csv", snap_rows)
    write_csv(OUT / "run_manifest_stage3.csv", manifest_rows)
    write_csv(OUT / "cd_hook_audit.csv", cd_rows)
    write_csv(OUT / "control_coverage_rows.csv", coverage_rows)
    write_json(OUT / "action_config.json", action_config)
    write_csv(STAGE3 / "action_config_rows.csv", config_rows)
    write_csv(STAGE3 / "candidate_policy_rows.csv", policy_rows)
    write_csv(STAGE3 / "run_manifest.csv", manifest_rows)

    missing_source_blocker = bool(missing_source_rows)
    summary = {
        "schema": "acl2_v110r_stage2_candidate_generation_summary_v1",
        "stage2_config_ready": not missing_source_blocker,
        "blocker": "missing_v108_stage3_source_policy_rows" if missing_source_blocker else "",
        "missing_source_rows": missing_source_rows,
        "sequences": list(SEQUENCES),
        "generated_candidates": sorted(RUNNABLE_CANDIDATES),
        "runnable_surface_ids": sorted({spec["surface_id"] for spec in RUNNABLE_CANDIDATES.values()}),
        "policy_families": list(POLICY_FAMILIES),
        "candidate_surface_config_rows": len(surface_rows),
        "candidate_policy_rows": len(policy_rows),
        "action_config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_manifest_rows": len(prepare_rows_by_seq),
        "run_worker_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "cd_hook_audit_status": "C/D not in Stage3 full pilot; hook audit required before claim",
        "control_coverage": CONTROL_COVERAGE,
        "claim_boundary": (
            "Stage3 can assess full ATE/action fidelity for generated A/B/E/F candidates, "
            "but semantic causality requires missing role_rotation/same_bucket/schedule controls before any method claim."
        ),
        "workspace": WORKSPACE,
        "raw_action": RAW_ACTION,
        "outputs": {
            "candidate_surface_configs": OUT / "candidate_surface_configs.csv",
            "candidate_policy_rows": OUT / "candidate_policy_rows.csv",
            "action_config_rows": OUT / "action_config_rows.csv",
            "run_manifest_stage3": OUT / "run_manifest_stage3.csv",
            "stage3_run_manifest": STAGE3 / "run_manifest.csv",
            "surface_design_report": OUT / "surface_design_report.md",
            "cd_hook_audit": OUT / "cd_hook_audit.csv",
            "cd_hook_audit_report": OUT / "cd_hook_audit.md",
            "summary": OUT / "config_generation_summary.json",
        },
    }

    write_text(OUT / "surface_design_report.md", surface_design_report(summary, surface_rows, coverage_rows))
    write_text(OUT / "cd_hook_audit.md", cd_hook_report(cd_rows))
    write_json(OUT / "config_generation_summary.json", summary)
    return summary


def surface_design_report(summary: dict[str, Any], surface_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v110R Stage2 Surface Design Report",
        "",
        "Stage2 generated only runtime-controllable A/B/E/F candidates for the Stage3 00/02 full-sequence pilot.",
        "Rows marked not generated remain plan items or controls that require a new selector/hook before any semantic-causality claim.",
        "",
        "```text",
        f"stage2_config_ready={summary['stage2_config_ready']}",
        f"generated_candidates={','.join(summary['generated_candidates'])}",
        f"action_config_rows={summary['action_config_rows']}",
        f"run_worker_manifest_rows={summary['run_worker_manifest_rows']}",
        f"claim_boundary={summary['claim_boundary']}",
        "```",
        "",
        "## Candidate Status",
        "",
        "| candidate | surface | name | status | runtime mode | boundary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in surface_rows:
        lines.append(
            "| {candidate_id} | {surface_id} | {candidate_name} | {candidate_status} | {stage3_action_mode} | {implementation_boundary} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Control Coverage",
            "",
            "| control | status | claim boundary |",
            "| --- | --- | --- |",
        ]
    )
    for row in coverage_rows:
        lines.append("| {control_family} | {coverage_status} | {claim_boundary} |".format(**row))
    return "\n".join(lines)


def cd_hook_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v110R Stage2 C/D Hook Audit",
        "",
        "C/D are not promoted to Stage3 full pilot in this generator because their independent runtime hooks are not implemented.",
        "This is a hook-implementation blocker, not a scientific No-Go.",
        "",
        "| surface | operation | status | blocker | next work |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {surface_id} | {operation_type} | {v110_status} | {blocker} | {next_required_work} |".format(**row)
        )
    return "\n".join(lines)


def main() -> int:
    summary = build()
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if summary["stage2_config_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
