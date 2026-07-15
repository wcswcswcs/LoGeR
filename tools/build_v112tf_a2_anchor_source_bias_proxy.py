#!/usr/bin/env python3
"""Build and summarize v112TF A2 frame-uniform anchor source bias proxy runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402
import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402
import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
OUT = RESULT_ROOT / "stage9_a2_anchor_source_bias_proxy_pilot_00_02"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
CUE = RESULT_ROOT / "stage2_memory_specific_cue_bank/anchor_memory_cue_rows.csv"
SEQUENCES = ("00", "02")
ANCHOR_FRAMES = tuple(range(8))


POLICIES = [
    {
        "policy_id": "A2_noop_anchor_source_weight_all1",
        "policy_family": "a2_default_off_parity",
        "grid": "all1",
        "lambda": 0.0,
        "mu": 0.0,
        "w_min": 1.0,
        "w_max": 1.0,
    },
    {
        "policy_id": "A2_frame_uniform_anchor_bias_mild",
        "policy_family": "a2_frame_uniform_proxy",
        "grid": "mild",
        "lambda": 0.5,
        "mu": 0.25,
        "w_min": 0.25,
        "w_max": 1.25,
    },
    {
        "policy_id": "A2_frame_uniform_anchor_bias_medium",
        "policy_family": "a2_frame_uniform_proxy",
        "grid": "medium",
        "lambda": 1.0,
        "mu": 0.50,
        "w_min": 0.20,
        "w_max": 1.50,
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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def cue_rows() -> dict[tuple[str, int], dict[str, str]]:
    return {
        (row["seq"], int(float(row["frame_id"]))): row
        for row in read_csv(CUE)
        if row.get("seq") in SEQUENCES
    }


def anchor_weight(policy: dict[str, Any], cue: dict[str, str]) -> float:
    if policy["grid"] == "all1":
        return 1.0
    dynamic = safe_float(cue.get("dynamic_mass"))
    boundary = safe_float(cue.get("boundary_mass"))
    lifetime = safe_float(cue.get("semantic_lifetime_risk"))
    weak = safe_float(cue.get("weak_context_mass"))
    stable_landmark = safe_float(cue.get("stable_landmark_mass"))
    continuity = safe_float(cue.get("semantic_continuity_score"))
    risk = 1.5 * dynamic + boundary + lifetime + 0.3 * weak
    support = stable_landmark + 0.3 * continuity
    raw = math.exp(-float(policy["lambda"]) * risk + float(policy["mu"]) * support)
    return clamp(raw, float(policy["w_min"]), float(policy["w_max"]))


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    weight_map: dict[int, float],
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
            f"_stage4_action_label: {action_label}",
            "_stage4_action_mode: anchor_source_attention_weight",
            f"_stage4_anchor_source_weight_map: {json.dumps({str(k): v for k, v in weight_map.items()}, sort_keys=True)}",
            "_stage4_anchor_source_token_roles: [\"patch\"]",
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


def build_configs() -> dict[str, Any]:
    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    cue_by_key = cue_rows()
    gpu_cycle = ["0", "1", "2", "3"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v112tf_a2_proxy_pilot_{seq}"
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
    weight_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    missing_cue: list[dict[str, Any]] = []
    row_index = 0

    for policy in POLICIES:
        policy_id = str(policy["policy_id"])
        for seq in SEQUENCES:
            weight_map: dict[int, float] = {}
            for frame in ANCHOR_FRAMES:
                cue = cue_by_key.get((seq, frame), {})
                if not cue:
                    missing_cue.append({"seq": seq, "frame": frame, "policy_id": policy_id})
                    continue
                weight = anchor_weight(policy, cue)
                weight_map[frame] = weight
                weight_rows.append(
                    {
                        "schema": "acl2_v112tf_a2_anchor_source_weight_row_v1",
                        "policy_id": policy_id,
                        "policy_family": policy["policy_family"],
                        "seq": seq,
                        "source_frame": frame,
                        "source_context_role": "scale_reference_context",
                        "token_roles": "patch",
                        "weight": weight,
                        "grid": policy["grid"],
                        "lambda": policy["lambda"],
                        "mu": policy["mu"],
                        "w_min": policy["w_min"],
                        "w_max": policy["w_max"],
                        "dynamic_mass": cue.get("dynamic_mass", ""),
                        "boundary_mass": cue.get("boundary_mass", ""),
                        "semantic_lifetime_risk": cue.get("semantic_lifetime_risk", ""),
                        "weak_context_mass": cue.get("weak_context_mass", ""),
                        "stable_landmark_mass": cue.get("stable_landmark_mass", ""),
                        "semantic_continuity_score": cue.get("semantic_continuity_score", ""),
                    }
                )

            dataset = f"kitti_v112tf_a2_proxy_pilot_{seq}"
            method = f"lingbot_map_v112tf_a2_proxy_{policy_id}_{seq}"
            action_label = f"v112tf_a2_proxy_{policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v112tf_a2_proxy_{policy_id}_{seq}.yaml"
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
                    weight_map=weight_map,
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            weights = list(weight_map.values())
            changed = [value for value in weights if abs(value - 1.0) > 1e-12]
            row = {
                "schema": "acl2_v112tf_a2_anchor_source_policy_row_v1",
                "surface_id": "A",
                "candidate_id": "A2_frame_uniform_anchor_source_bias_proxy",
                "policy_id": policy_id,
                "policy_family": policy["policy_family"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage2_action_mode": "anchor_source_attention_weight",
                "stage4_action_mode": "anchor_source_attention_weight",
                "selected_count": len(weight_map),
                "selected_global_frame_indices": ";".join(str(x) for x in sorted(weight_map)),
                "expected_action_field": "anchor_source_attention_weight",
                "weight_min": min(weights) if weights else "",
                "weight_max": max(weights) if weights else "",
                "weight_mean": sum(weights) / len(weights) if weights else "",
                "changed_source_frame_count": len(changed),
                "token_roles": "patch",
                "runtime_boundary": "v112 Stage9 A2 proxy uses frame-uniform read-time attention bias on default scale-reference/anchor patch source tokens.",
                "claim_boundary": "Frame-uniform proxy only; Stage2 lacks individual anchor patch-token semantic cue, so full A2 token-level success claim is not allowed.",
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
                    "schema": "acl2_v112tf_a2_proxy_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v112tf_a2_proxy_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"a2_proxy_pilot_{seq}",
                    "target_kind": "pilot_dataset_prepare",
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
                f"ACL2_V108_STAGE4_SURFACE_ID=A "
                f"ACL2_V112_A2_ACTION_GLOBAL_IDXS=0 "
                f"ACL2_V112TF_A2_POLICY_ID={policy_id}"
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
                        "schema": "acl2_v112tf_a2_proxy_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v112tf_a2_proxy_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"a2_proxy_pilot_{seq}",
                        "target_kind": "pilot_full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": "anchor_source_attention_weight",
                        "selector": "default_first8_scale_reference_patch_source_tokens",
                        "selected_count": len(weight_map),
                        "force_non_keyframe_indices": "",
                        "context_token_type_mask": "",
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
        "schema": "acl2_v112tf_a2_anchor_source_bias_proxy_config_summary_v1",
        "config_ready": not missing_cue,
        "blocker": "missing_stage2_anchor_memory_cue_rows" if missing_cue else "",
        "claim_boundary": "A2 frame-uniform proxy only; individual anchor patch-token semantic cue is absent in Stage2 artifacts.",
        "sequences": list(SEQUENCES),
        "anchor_frames": list(ANCHOR_FRAMES),
        "policy_ids": [policy["policy_id"] for policy in POLICIES],
        "config_rows": len(config_rows),
        "prepare_manifest_rows": len(prepare_rows_by_seq),
        "run_worker_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "missing_cue_rows": missing_cue,
        "outputs": {
            "action_config_rows": rel(OUT / "action_config_rows.csv"),
            "anchor_source_weight_rows": rel(OUT / "anchor_source_weight_rows.csv"),
            "run_manifest": rel(OUT / "run_manifest.csv"),
            "summary": rel(OUT / "a2_proxy_config_generation_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "anchor_source_weight_rows.csv", weight_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_json(OUT / "a2_proxy_config_generation_summary.json", summary)
    return summary


def latest_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v112tf_a2_proxy_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v112tf_a2_proxy_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase), {})
        rc = stage2m.safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def action_fidelity_row(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    rows = [
        row for row in stage2m.base.load_jsonl(action_file)
        if row.get("schema") == "acl2_v112tf_a2_anchor_source_attention_action_row_v1"
    ]
    run_name = f"kitti_lingbot_v112tf_a2_proxy_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    target_rows = [row for row in rows if int(row.get("target_key_count", 0) or 0) > 0]
    changed_rows = [row for row in target_rows if int(row.get("changed_key_count", 0) or 0) > 0]
    expected_changed = int(float(cfg.get("changed_source_frame_count", "0") or 0)) > 0
    action_fidelity_pass = bool(
        action_file.exists()
        and target_rows
        and (bool(changed_rows) == expected_changed)
        and all(str(row.get("action_granularity", "")) == "frame_uniform_source_token_weight" for row in target_rows)
    )
    target_counts = [int(row.get("target_key_count", 0) or 0) for row in target_rows]
    changed_counts = [int(row.get("changed_key_count", 0) or 0) for row in target_rows]
    return {
        "schema": "acl2_v112tf_a2_proxy_action_fidelity_row_v1",
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage2_action_mode": cfg.get("stage2_action_mode", ""),
        "stage4_action_mode": cfg.get("stage4_action_mode", ""),
        "expected_action_field": cfg.get("expected_action_field", ""),
        "expected_action_frame_count": int(float(cfg.get("selected_count", "0") or 0)),
        "observed_action_frame_count": len(target_rows),
        "action_effective_frame_count": len(changed_rows),
        "action_noop_frame_count": len(target_rows) - len(changed_rows),
        "expected_keyframe_count": int(float(cfg.get("selected_count", "0") or 0)),
        "observed_keyframe_count": "",
        "special_token_operation_count": "",
        "trace_error_rows": 0,
        "action_file_exists": action_file.exists(),
        "action_fidelity_pass": action_fidelity_pass,
        "observed_action_indices": cfg.get("selected_global_frame_indices", ""),
        "effective_action_indices": cfg.get("selected_global_frame_indices", "") if expected_changed else "",
        "missing_expected_indices": "",
        "unexpected_observed_indices": "",
        "ineffective_expected_indices": "" if action_fidelity_pass else cfg.get("selected_global_frame_indices", ""),
        "target_key_count_min": min(target_counts) if target_counts else "",
        "target_key_count_max": max(target_counts) if target_counts else "",
        "changed_key_count_min": min(changed_counts) if changed_counts else "",
        "changed_key_count_max": max(changed_counts) if changed_counts else "",
        "action_log_rows": len(rows),
        "action_file": rel(action_file),
        "run_worker_returncode": run_row.get("returncode", ""),
        "run_worker_duration_sec": run_row.get("duration_sec", ""),
    }


def install_metric_overrides() -> None:
    stage2m.OUT = OUT
    stage2m.CONFIG_ROWS = OUT / "action_config_rows.csv"
    stage2m.RUN_RESULTS = OUT / "run_results.csv"
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = phase_status_for
    stage2m.action_fidelity_row = action_fidelity_row


def policy_summary_rows(full_rows: list[dict[str, Any]], rolling_rows: list[dict[str, Any]], fidelity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_policy[str(row["policy_id"])].append(row)
    for row in rolling_rows:
        rolling_by_policy[str(row["policy_id"])].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[str(row["policy_id"])].append(row)
    out: list[dict[str, Any]] = []
    noop_median = float("nan")
    for row in by_policy.get("A2_noop_anchor_source_weight_all1", []):
        pass
    noop_rels = [
        stage2m.safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        for row in by_policy.get("A2_noop_anchor_source_weight_all1", [])
    ]
    if noop_rels:
        noop_median = stage3m.base.median(noop_rels)
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        rels = [
            stage2m.safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
            for row in rows
        ]
        rolling = [
            stage2m.safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool_value(row.get("action_fidelity_pass")))
        all_action = action_pass_count == len(SEQUENCES)
        median_full = stage3m.base.median(rels)
        mean_full = stage3m.base.mean(rels)
        max_harm = stage3m.base.max_rel_harm(rels)
        rolling_median = stage3m.base.median(rolling)
        is_proxy = policy_id != "A2_noop_anchor_source_weight_all1"
        proxy_gate = bool(
            is_proxy
            and len(rows) == len(SEQUENCES)
            and all_action
            and math.isfinite(median_full)
            and math.isfinite(noop_median)
            and median_full - noop_median >= 0.02
            and max_harm <= 0.02
        )
        sample = rows[0]
        row_out: dict[str, Any] = {
            "schema": "acl2_v112tf_a2_proxy_policy_summary_row_v1",
            "candidate_id": "A2_frame_uniform_anchor_source_bias_proxy",
            "surface_id": sample.get("surface_id", ""),
            "policy_id": policy_id,
            "policy_family": sample.get("policy_family", ""),
            "sequence_count": len(rows),
            "metric_complete": len(rows) == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows),
            "action_fidelity_pass_count": action_pass_count,
            "all_action_fidelity": all_action,
            "median_full_rel": median_full,
            "mean_full_rel": mean_full,
            "max_harm": max_harm,
            "rolling_p90_median_rel": rolling_median,
            "median_full_rel_minus_noop": median_full - noop_median if math.isfinite(median_full) and math.isfinite(noop_median) else float("nan"),
            "proxy_gate_pass": proxy_gate,
            "semantic_aware_claim_allowed": False,
            "claim_boundary": "Frame-uniform proxy only; Stage2 lacks individual anchor patch-token semantic cue, so full A2 claim is blocked.",
        }
        for row in rows:
            row_out[f"seq{row['seq']}_full_rel"] = row.get("full_ATE_sim3_relative_improvement_vs_baseline", "")
        out.append(row_out)
    return out


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: stage2m.safe_float(row.get("median_full_rel", "nan")), reverse=True)
    lines = [
        "# ACL2 v112TF A2 Anchor Source Bias Proxy Pilot Report",
        "",
        f"metric_complete: `{summary['metric_complete']}`",
        f"all_action_fidelity: `{summary['all_action_fidelity']}`",
        f"taxonomy: `{summary['taxonomy']}`",
        f"blocker: `{summary['blocker']}`",
        "",
        "## Claim Boundary",
        "",
        "This is a frame-uniform anchor source read-time attention-bias proxy. It is not a full A2 token-level semantic intervention because Stage2 anchor cue artifacts are frame-level, not individual patch-token rows.",
        "",
        "## Ranking",
        "",
    ]
    for row in ranked:
        lines.append(
            "- {policy}: median={median} mean={mean} harm={harm} rolling={rolling} minus_noop={gap} proxy_gate={gate}".format(
                policy=row.get("policy_id", ""),
                median=row.get("median_full_rel", ""),
                mean=row.get("mean_full_rel", ""),
                harm=row.get("max_harm", ""),
                rolling=row.get("rolling_p90_median_rel", ""),
                gap=row.get("median_full_rel_minus_noop", ""),
                gate=row.get("proxy_gate_pass", ""),
            )
        )
    return "\n".join(lines) + "\n"


def build_metrics() -> dict[str, Any]:
    install_metric_overrides()
    latest = latest_rows(OUT / "run_results.csv")
    config_rows = read_csv(OUT / "action_config_rows.csv")
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for row in full_rows:
        row["schema"] = "acl2_v112tf_a2_proxy_full_metric_row_v1"
    for row in rolling_rows:
        row["schema"] = "acl2_v112tf_a2_proxy_rolling_metric_row_v1"
    for row in local_rows:
        row["schema"] = "acl2_v112tf_a2_proxy_local_handoff_metric_row_v1"
    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    metric_complete = bool(full_rows and all(bool_value(row.get("metric_available")) for row in full_rows))
    all_action = bool(fidelity_rows and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows))
    proxy_pass = [row for row in policy_rows if bool_value(row.get("proxy_gate_pass"))]
    taxonomy = (
        "A2_FRAME_UNIFORM_PROXY_PASS_FULL_A2_TOKEN_CUE_BLOCKED"
        if proxy_pass and metric_complete and all_action
        else "A2_FRAME_UNIFORM_PROXY_NO_GO_OR_INCOMPLETE_FULL_A2_TOKEN_CUE_BLOCKED"
    )
    blocker = "" if proxy_pass and metric_complete and all_action else "proxy_gate_not_passed_or_metrics_incomplete"
    summary = {
        "schema": "acl2_v112tf_a2_anchor_source_bias_proxy_metric_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "taxonomy": taxonomy,
        "blocker": blocker,
        "semantic_aware_claim_allowed": False,
        "semantic_aware_claim_blocker": "Stage2 anchor cues are frame-level; full A2 token-level source-token semantic intervention remains blocked.",
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_summary_row_count": len(policy_rows),
        "outputs": {
            "full_metric_rows": rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "policy_summary_rows": rel(OUT / "policy_summary_rows.csv"),
            "summary": rel(OUT / "a2_proxy_metric_summary.json"),
            "report": rel(OUT / "A2_ANCHOR_SOURCE_BIAS_PROXY_PILOT_REPORT.md"),
        },
    }
    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "policy_summary_rows.csv", policy_rows)
    write_json(OUT / "a2_proxy_metric_summary.json", summary)
    write_text(OUT / "A2_ANCHOR_SOURCE_BIAS_PROXY_PILOT_REPORT.md", report_text(summary, policy_rows))
    return summary


def write_blocker_audit() -> None:
    write_text(
        OUT / "A2_TOKEN_LEVEL_CUE_BLOCKED.md",
        """# A2 Token-Level Cue Blocked

Stage2 `anchor_memory_cue_rows.csv` provides one row per `(seq, frame_id)` with frame-level semantic masses and gates. It does not contain individual anchor patch-token rows, spatial patch ids, or per-token dynamic/boundary/stability labels.

This Stage9 run therefore implements only a frame-uniform source-token attention-bias proxy over default scale-reference/anchor patch tokens. It must not be reported as a complete A2 token-level semantic source-token intervention.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    if not args.configs and not args.metrics:
        args.configs = True
        args.metrics = True
    status = 0
    if args.configs:
        summary = build_configs()
        write_blocker_audit()
        print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
        if not summary["config_ready"]:
            status = 1
    if args.metrics:
        summary = build_metrics()
        print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
        if not summary["metric_complete"]:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
