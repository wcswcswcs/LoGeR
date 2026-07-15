#!/usr/bin/env python3
"""Build and summarize v115 LingBot L2 matched full-control runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402
import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402
import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402
import build_v115tf_lingbot_a2_l2_query_hook_smoke as smoke  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
OUT = RESULT_ROOT / "stage5_lingbot_l2_matched_controls_00_02"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
PILOT = RESULT_ROOT / "stage5_lingbot_a2_l2_query_full_pilot_00_02"
SEQUENCES = ("00", "02")
PLAN_GEOMETRY_REL_THRESHOLD = 0.05
L2_SOURCE_FRAMES = tuple(range(8, 16))
SPECIAL_QUERY_ROLES = ["camera_special", "register_special", "scale_special"]
CANDIDATE_POLICY_ID = "LB_L2_special_query_local_risk_suppress_smoke"
BASE_WEIGHT_SOURCE = "computed_all_query_mild"


CONTROL_POLICIES = [
    {
        "policy_id": "LB_L2_special_query_local_noop_all1_control",
        "policy_family": "l2_special_query_noop_all1",
        "query_roles": SPECIAL_QUERY_ROLES,
        "transform": "all1",
    },
    {
        "policy_id": "LB_L2_patch_query_local_control",
        "policy_family": "l2_patch_query_generic_control",
        "query_roles": ["patch"],
        "transform": "base",
    },
    {
        "policy_id": "LB_L2_special_query_local_same_weight_shuffle_seed0",
        "policy_family": "l2_same_weight_histogram_shuffle",
        "query_roles": SPECIAL_QUERY_ROLES,
        "transform": "shuffle_reverse",
    },
    {
        "policy_id": "LB_L2_special_query_local_reverse_risk_control",
        "policy_family": "l2_reverse_risk_direction",
        "query_roles": SPECIAL_QUERY_ROLES,
        "transform": "reverse_risk",
    },
]

MATCHED_CONTROL_POLICIES = CONTROL_POLICIES
SPECIAL_WEIGHT_REPAIR_POLICIES = [
    {
        "policy_id": "LB_L2_special_query_local_special_weight_repair",
        "policy_family": "l2_special_query_special_weight_repair",
        "query_roles": SPECIAL_QUERY_ROLES,
        "transform": "base",
    },
    {
        "policy_id": "LB_L2_patch_query_local_special_weight_control",
        "policy_family": "l2_patch_query_special_weight_control",
        "query_roles": ["patch"],
        "transform": "base",
    },
    {
        "policy_id": "LB_L2_special_query_local_special_weight_noop_all1_control",
        "policy_family": "l2_special_query_special_weight_noop_all1",
        "query_roles": SPECIAL_QUERY_ROLES,
        "transform": "all1",
    },
    {
        "policy_id": "LB_L2_special_query_local_special_weight_shuffle_seed0",
        "policy_family": "l2_special_weight_histogram_shuffle",
        "query_roles": SPECIAL_QUERY_ROLES,
        "transform": "shuffle_reverse",
    },
    {
        "policy_id": "LB_L2_special_query_local_special_weight_reverse_risk",
        "policy_family": "l2_special_weight_reverse_risk_direction",
        "query_roles": SPECIAL_QUERY_ROLES,
        "transform": "reverse_risk",
    },
]


def configure_profile(profile: str) -> None:
    global OUT, CONFIG_ROOT, METHOD_DIR, DATASET_DIR, WORKSPACE, RAW_ACTION
    global CONTROL_POLICIES, CANDIDATE_POLICY_ID, BASE_WEIGHT_SOURCE
    if profile == "matched_controls":
        OUT = RESULT_ROOT / "stage5_lingbot_l2_matched_controls_00_02"
        CONTROL_POLICIES = MATCHED_CONTROL_POLICIES
        CANDIDATE_POLICY_ID = "LB_L2_special_query_local_risk_suppress_smoke"
        BASE_WEIGHT_SOURCE = "computed_all_query_mild"
    elif profile == "special_weight_repair":
        OUT = RESULT_ROOT / "stage5_lingbot_l2_special_weight_repair_00_02"
        CONTROL_POLICIES = SPECIAL_WEIGHT_REPAIR_POLICIES
        CANDIDATE_POLICY_ID = "LB_L2_special_query_local_special_weight_repair"
        BASE_WEIGHT_SOURCE = "cue_w_special_query_mild"
    else:
        raise ValueError(f"unknown profile: {profile}")
    CONFIG_ROOT = OUT / "configs"
    METHOD_DIR = CONFIG_ROOT / "methods"
    DATASET_DIR = CONFIG_ROOT / "datasets"
    WORKSPACE = OUT / "workspace"
    RAW_ACTION = OUT / "raw_action"


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


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def base_local_weight_map(seq: str) -> tuple[dict[int, float], list[dict[str, Any]]]:
    cue_by_local = smoke.local_cues()
    weight_map: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for source_frame in L2_SOURCE_FRAMES:
        cue = cue_by_local.get((seq, int(source_frame)), {})
        if not cue:
            continue
        if BASE_WEIGHT_SOURCE == "cue_w_special_query_mild":
            weight = safe_float(cue.get("w_special_query_mild"))
            if not math.isfinite(weight):
                weight = smoke.local_weight(cue, "local_risk_suppress")
        else:
            weight = smoke.local_weight(cue, "local_risk_suppress")
        weight_map[int(source_frame)] = weight
        rows.append(
            {
                "source_frame": int(source_frame),
                "base_weight": weight,
                "base_weight_source": BASE_WEIGHT_SOURCE,
                "w_special_query_mild": cue.get("w_special_query_mild", ""),
                "w_all_query_mild": cue.get("w_all_query_mild", ""),
                "R_local": cue.get("R_local", ""),
                "S_local": cue.get("S_local", ""),
            }
        )
    return weight_map, rows


def transform_weights(base: dict[int, float], transform: str) -> dict[int, float]:
    if transform == "all1":
        return {frame: 1.0 for frame in base}
    if transform == "base":
        return dict(base)
    frames = sorted(base)
    values = [base[frame] for frame in frames]
    if transform == "shuffle_reverse":
        return {frame: value for frame, value in zip(frames, reversed(values))}
    if transform == "reverse_risk":
        return {frame: clamp(2.0 - value, 0.25, 1.25) for frame, value in base.items()}
    raise ValueError(f"unknown transform: {transform}")


def latest_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def build() -> dict[str, Any]:
    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v115tf_lingbot_l2_controls_{seq}"
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
    missing_cues: list[dict[str, Any]] = []
    row_index = 0

    for policy in CONTROL_POLICIES:
        for seq in SEQUENCES:
            base_weights, base_rows = base_local_weight_map(seq)
            if len(base_weights) != len(L2_SOURCE_FRAMES):
                missing = sorted(set(L2_SOURCE_FRAMES) - set(base_weights))
                for source_frame in missing:
                    missing_cues.append({"policy_id": policy["policy_id"], "seq": seq, "source_frame": source_frame})
            weight_map = transform_weights(base_weights, policy["transform"])
            for row in base_rows:
                source_frame = int(row["source_frame"])
                weight_rows.append(
                    {
                        "schema": "acl2_v115tf_lingbot_l2_control_weight_row_v1",
                        "policy_id": policy["policy_id"],
                        "policy_family": policy["policy_family"],
                        "seq": seq,
                        "source_frame": source_frame,
                        "base_weight": row["base_weight"],
                        "base_weight_source": row.get("base_weight_source", ""),
                        "w_special_query_mild": row.get("w_special_query_mild", ""),
                        "w_all_query_mild": row.get("w_all_query_mild", ""),
                        "control_weight": weight_map.get(source_frame, ""),
                        "transform": policy["transform"],
                        "query_roles": ",".join(policy["query_roles"]) if policy["query_roles"] else "all",
                        "R_local": row.get("R_local", ""),
                        "S_local": row.get("S_local", ""),
                    }
                )

            dataset = f"kitti_v115tf_lingbot_l2_controls_{seq}"
            method = f"lingbot_map_v115tf_l2ctrl_{policy['policy_id']}_{seq}"
            action_label = f"v115tf_l2ctrl_{policy['policy_id']}"
            config = CONFIG_ROOT / f"kitti_lingbot_v115tf_l2ctrl_{policy['policy_id']}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{method}.jsonl"
            gpu = gpu_cycle[row_index % len(gpu_cycle)]
            row_index += 1
            write_text(
                method_path,
                smoke.method_yaml(
                    checkpoint=checkpoint,
                    env_name=env_name,
                    use_sdpa=use_sdpa,
                    action_label=action_label,
                    weight_map=weight_map,
                    token_roles=["patch"],
                    query_roles=list(policy["query_roles"]),
                    source_context_roles=["local_window_context"],
                ),
            )
            write_text(config, smoke.run_config_yaml(dataset, method).replace(str(smoke.WORKSPACE.resolve()), str(WORKSPACE.resolve())))
            weights = list(weight_map.values())
            changed = [value for value in weights if abs(value - 1.0) > 1e-12]
            config_rows.append(
                {
                    "schema": "acl2_v115tf_lingbot_l2_control_config_row_v1",
                    "surface_id": "LB-L2",
                    "policy_id": policy["policy_id"],
                    "policy_family": policy["policy_family"],
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": action_label,
                    "stage2_action_mode": "anchor_source_attention_weight",
                    "stage4_action_mode": "anchor_source_attention_weight",
                    "source": "local",
                    "source_context_roles": "local_window_context",
                    "token_roles": "patch",
                    "query_roles": ",".join(policy["query_roles"]) if policy["query_roles"] else "all",
                    "selected_count": len(weight_map),
                    "selected_global_frame_indices": ";".join(str(x) for x in sorted(weight_map)),
                    "changed_source_frame_count": len(changed),
                    "weight_min": min(weights) if weights else "",
                    "weight_max": max(weights) if weights else "",
                    "weight_mean": sum(weights) / len(weights) if weights else "",
                    "config": str(config.resolve()),
                    "method_config": str(method_path.resolve()),
                    "action_file": str(action_file.resolve()),
                    "gpu": gpu,
                }
            )
            prefix = v108.command_prefix(conda_path, pythonpath, gpu)
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v115tf_lingbot_l2_control_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v115tf_l2ctrl_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"l2ctrl_{seq}",
                    "target_kind": "l2_control_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "full_prepare_once_per_seq",
                    "selected_count": 0,
                    "gpu": gpu,
                    "cwd": str(BENCHMARK.resolve()),
                    "config": str(config.resolve()),
                    "trace_file": "",
                    "action_file": "",
                    "command": f"{prefix} {conda_path} run -n {env_name} python prepare.py --config {config.resolve()} --force",
                    "status": "planned",
                },
            )
            action_env = (
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method} "
                f"ACL2_V108_STAGE4_POLICY_ID={policy['policy_id']} "
                f"ACL2_V108_STAGE4_SURFACE_ID=LB-L2 "
                f"ACL2_V112_A2_ACTION_GLOBAL_IDXS=0 "
                f"ACL2_V112_A2_ACTION_MAX_ROWS=5000"
            )
            commands = {
                "run_worker": (
                    f"{prefix} {action_env} {conda_path} run -n {env_name} "
                    f"python run_worker.py --config {config.resolve()} --method {method} "
                    f"--dataset {dataset} --scene {seq} --force"
                ),
                "evaluate": f"{prefix} {conda_path} run -n {env_name} python evaluate.py --config {config.resolve()} --force",
                "report": f"{prefix} {conda_path} run -n {env_name} python report.py --workspace {WORKSPACE.resolve()} --dataset {dataset}",
            }
            for phase, command in commands.items():
                manifest_rows.append(
                    {
                        "schema": "acl2_v115tf_lingbot_l2_control_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v115tf_l2ctrl_{policy['policy_id']}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"{policy['policy_id']}_{seq}",
                        "target_kind": "l2_control_full",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": "anchor_source_attention_weight",
                        "selector": f"local:{','.join(policy['query_roles']) or 'all'}:{policy['transform']}",
                        "selected_count": len(weight_map),
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
        "schema": "acl2_v115tf_lingbot_l2_control_config_summary_v1",
        "profile": "special_weight_repair" if CANDIDATE_POLICY_ID == "LB_L2_special_query_local_special_weight_repair" else "matched_controls",
        "base_weight_source": BASE_WEIGHT_SOURCE,
        "config_ready": not missing_cues,
        "blocker": "missing_local_cue_rows" if missing_cues else "",
        "sequences": list(SEQUENCES),
        "policy_ids": [policy["policy_id"] for policy in CONTROL_POLICIES],
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "missing_cue_rows": missing_cues,
        "outputs": {
            "action_config_rows": rel(OUT / "action_config_rows.csv"),
            "weight_rows": rel(OUT / "source_weight_rows.csv"),
            "manifest": rel(OUT / "run_manifest.csv"),
            "summary": rel(OUT / "l2_control_config_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "source_weight_rows.csv", weight_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_json(OUT / "l2_control_config_summary.json", summary)
    return summary


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v115tf_l2ctrl_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v115tf_l2ctrl_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase), {})
        rc = stage2m.safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def action_fidelity_row(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    rows = [
        row for row in stage2m.base.load_jsonl(action_file)
        if row.get("row_type") == "anchor_source_attention_weight"
    ]
    target_rows = [row for row in rows if int(row.get("target_key_count", 0) or 0) > 0]
    changed_rows = [
        row for row in target_rows
        if int(row.get("changed_key_count", 0) or 0) > 0
        and int(row.get("target_query_count", 0) or 0) > 0
    ]
    expected_changed = int(float(cfg.get("changed_source_frame_count", "0") or 0)) > 0
    observed_context_roles = sorted({str(row.get("source_context_role", "")) for row in target_rows if row.get("source_context_role", "")})
    observed_query_roles = sorted({str(row.get("query_roles", "")) for row in target_rows if row.get("query_roles", "")})
    target_query_counts = [int(row.get("target_query_count", 0) or 0) for row in target_rows]
    run_name = f"kitti_lingbot_v115tf_l2ctrl_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    action_fidelity_pass = bool(
        action_file.exists()
        and target_rows
        and (bool(changed_rows) == expected_changed)
        and cfg.get("source_context_roles", "") in observed_context_roles
        and cfg.get("query_roles", "") in observed_query_roles
    )
    return {
        "schema": "acl2_v115tf_lingbot_l2_control_action_fidelity_row_v1",
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage2_action_mode": cfg.get("stage2_action_mode", ""),
        "stage4_action_mode": cfg.get("stage4_action_mode", ""),
        "expected_context_roles": cfg.get("source_context_roles", ""),
        "observed_context_roles": ";".join(observed_context_roles),
        "expected_query_roles": cfg.get("query_roles", ""),
        "observed_query_roles": ";".join(observed_query_roles),
        "expected_action_frame_count": int(float(cfg.get("selected_count", "0") or 0)),
        "observed_action_frame_count": len(target_rows),
        "action_effective_frame_count": len(changed_rows),
        "action_noop_frame_count": len(target_rows) - len(changed_rows),
        "target_query_count_min": min(target_query_counts) if target_query_counts else "",
        "target_query_count_max": max(target_query_counts) if target_query_counts else "",
        "trace_error_rows": 0,
        "action_file_exists": action_file.exists(),
        "action_fidelity_pass": action_fidelity_pass,
        "observed_action_indices": cfg.get("selected_global_frame_indices", ""),
        "effective_action_indices": cfg.get("selected_global_frame_indices", "") if expected_changed else "",
        "missing_expected_indices": "",
        "unexpected_observed_indices": "",
        "ineffective_expected_indices": "" if action_fidelity_pass else cfg.get("selected_global_frame_indices", ""),
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
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        rels = [safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan")) for row in rows]
        rolling = [safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan")) for row in rolling_by_policy.get(policy_id, [])]
        local_rels = [safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan")) for row in rows]
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if stage2m.base.boolish(row.get("action_fidelity_pass")))
        all_action = action_pass_count == len(SEQUENCES)
        median_full = stage3m.base.median(rels)
        median_rolling = stage3m.base.median(rolling)
        max_harm = stage3m.base.max_rel_harm(rels)
        max_local_harm = max([max(0.0, -value) for value in local_rels if math.isfinite(value)] or [float("nan")])
        metric_complete = len(rows) == len(SEQUENCES) and all(stage2m.base.boolish(row.get("metric_available")) for row in rows)
        safety_common = bool(
            metric_complete
            and all_action
            and max_harm <= 0.02
            and math.isfinite(max_local_harm)
            and max_local_harm <= 0.02
        )
        positive_safety_gate = bool(safety_common and (median_full > 0 or median_rolling > 0))
        plan_metric_pass = bool(median_full >= PLAN_GEOMETRY_REL_THRESHOLD or median_rolling >= PLAN_GEOMETRY_REL_THRESHOLD)
        gate = bool(safety_common and plan_metric_pass)
        sample = rows[0]
        row_out: dict[str, Any] = {
            "schema": "acl2_v115tf_lingbot_l2_control_policy_summary_row_v1",
            "surface_id": sample.get("surface_id", ""),
            "policy_id": policy_id,
            "policy_family": sample.get("policy_family", ""),
            "sequence_count": len(rows),
            "metric_complete": metric_complete,
            "action_fidelity_pass_count": action_pass_count,
            "all_action_fidelity": all_action,
            "median_full_rel": median_full,
            "mean_full_rel": stage3m.base.mean(rels),
            "max_full_harm": max_harm,
            "rolling_p90_median_rel": median_rolling,
            "max_local_window_harm": max_local_harm,
            "positive_safety_gate_pass": positive_safety_gate,
            "plan_geometry_threshold_rel": PLAN_GEOMETRY_REL_THRESHOLD,
            "plan_geometry_metric_pass": plan_metric_pass,
            "pilot_geometry_gate_pass": gate,
        }
        for row in rows:
            row_out[f"seq{row['seq']}_full_rel"] = row.get("full_ATE_sim3_relative_improvement_vs_baseline", "")
            row_out[f"seq{row['seq']}_local_rel"] = row.get("local_window_ATE_rel_improvement_vs_baseline_median", "")
        out.append(row_out)
    return out


def candidate_reference() -> dict[str, str]:
    for row in read_csv(PILOT / "policy_summary_rows.csv"):
        if row.get("policy_id") == CANDIDATE_POLICY_ID:
            return row
    return {}


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v115TF LingBot L2 Matched Controls Report",
        "",
        f"profile: `{summary['profile']}`",
        f"taxonomy: `{summary['taxonomy']}`",
        f"candidate_policy_id: `{summary['candidate_policy_id']}`",
        f"candidate_median_full_rel: `{summary['candidate_median_full_rel']}`",
        f"best_control_median_full_rel: `{summary['best_control_median_full_rel']}`",
        f"control_matches_or_exceeds_candidate: `{summary['control_matches_or_exceeds_candidate']}`",
        "",
        "## Controls",
        "",
    ]
    for row in sorted(rows, key=lambda item: safe_float(item.get("median_full_rel", "nan")), reverse=True):
        lines.append(
            "- {policy}: median_full={full} rolling_p90={rolling} max_harm={harm} local_harm={local} positive_safety={positive} plan_gate={gate}".format(
                policy=row.get("policy_id", ""),
                full=row.get("median_full_rel", ""),
                rolling=row.get("rolling_p90_median_rel", ""),
                harm=row.get("max_full_harm", ""),
                local=row.get("max_local_window_harm", ""),
                positive=row.get("positive_safety_gate_pass", ""),
                gate=row.get("pilot_geometry_gate_pass", ""),
            )
        )
    return "\n".join(lines) + "\n"


def summarize() -> dict[str, Any]:
    install_metric_overrides()
    latest = latest_rows(OUT / "run_results.csv")
    config_rows = read_csv(OUT / "action_config_rows.csv")
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for row in full_rows:
        row["schema"] = "acl2_v115tf_lingbot_l2_control_full_metric_row_v1"
    for row in rolling_rows:
        row["schema"] = "acl2_v115tf_lingbot_l2_control_rolling_metric_row_v1"
    for row in local_rows:
        row["schema"] = "acl2_v115tf_lingbot_l2_control_local_metric_row_v1"
    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    candidate = next((row for row in policy_rows if row.get("policy_id") == CANDIDATE_POLICY_ID), {})
    if not candidate:
        candidate = candidate_reference()
    candidate_full = safe_float(candidate.get("median_full_rel", "nan"))
    candidate_rolling = safe_float(candidate.get("rolling_p90_median_rel", "nan"))
    candidate_positive_safety = stage2m.base.boolish(candidate.get("positive_safety_gate_pass"))
    candidate_plan_gate = stage2m.base.boolish(candidate.get("pilot_geometry_gate_pass"))
    best_full = max([safe_float(row.get("median_full_rel", "nan")) for row in policy_rows] or [float("nan")])
    best_rolling = max([safe_float(row.get("rolling_p90_median_rel", "nan")) for row in policy_rows] or [float("nan")])
    control_matches = bool(
        (math.isfinite(candidate_full) and math.isfinite(best_full) and best_full >= candidate_full)
        or (math.isfinite(candidate_rolling) and math.isfinite(best_rolling) and best_rolling >= candidate_rolling)
    )
    metric_complete = bool(full_rows and all(stage2m.base.boolish(row.get("metric_available")) for row in full_rows))
    all_action = bool(fidelity_rows and all(stage2m.base.boolish(row.get("action_fidelity_pass")) for row in fidelity_rows))
    summary = {
        "schema": "acl2_v115tf_lingbot_l2_control_metric_summary_v2",
        "profile": "special_weight_repair" if CANDIDATE_POLICY_ID == "LB_L2_special_query_local_special_weight_repair" else "matched_controls",
        "base_weight_source": BASE_WEIGHT_SOURCE,
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "plan_geometry_threshold_rel": PLAN_GEOMETRY_REL_THRESHOLD,
        "candidate_policy_id": CANDIDATE_POLICY_ID,
        "candidate_median_full_rel": candidate_full,
        "candidate_rolling_p90_median_rel": candidate_rolling,
        "candidate_positive_safety_gate_pass": candidate_positive_safety,
        "candidate_plan_geometry_gate_pass": candidate_plan_gate,
        "best_control_median_full_rel": best_full,
        "best_control_rolling_p90_median_rel": best_rolling,
        "control_matches_or_exceeds_candidate": control_matches,
        "taxonomy": "LB_L2_CONTROL_MATCHED_OR_EXCEEDED_CANDIDATE" if control_matches and metric_complete and all_action else "LB_L2_CONTROL_SUBSET_NOT_MATCHING_CANDIDATE",
        "claim_boundary": "This is a 00/02 L2 candidate/control comparison. A candidate is not a plan-level method unless candidate_plan_geometry_gate_pass is true; if the candidate is non-positive, control_matches_or_exceeds_candidate marks the repair as a No-Go.",
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
            "summary": rel(OUT / "l2_control_metric_summary.json"),
            "report": rel(OUT / "LB_L2_MATCHED_CONTROLS_REPORT.md"),
        },
    }
    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "policy_summary_rows.csv", policy_rows)
    write_json(OUT / "l2_control_metric_summary.json", summary)
    write_text(OUT / "LB_L2_MATCHED_CONTROLS_REPORT.md", report_text(summary, policy_rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--profile", choices=["matched_controls", "special_weight_repair"], default="matched_controls")
    args = parser.parse_args()
    configure_profile(args.profile)
    if not args.build and not args.summarize:
        args.build = True
    if args.build:
        print(json.dumps(clean_json(build()), indent=2, sort_keys=True))
    if args.summarize:
        print(json.dumps(clean_json(summarize()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
