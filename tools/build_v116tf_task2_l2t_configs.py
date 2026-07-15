#!/usr/bin/env python3
"""Build and summarize v116 Task2 LingBot L2T token-level 00/02 pilot runs."""

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


RESULT_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"
OUT = RESULT_ROOT / "task2_l2t"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
TOKEN_ROOT = OUT / "token_semantics"
SEQUENCES = ("00", "02")
L2_SOURCE_FRAMES = tuple(range(8, 16))
SPECIAL_QUERY_ROLES = ["camera_special", "register_special", "scale_special"]
PILOT_GEOMETRY_REL_THRESHOLD = 0.03


POLICIES = [
    {
        "policy_id": "L2T0_noop_selected_query_parity",
        "policy_family": "token_hook_parity",
        "query_roles": SPECIAL_QUERY_ROLES,
        "token_weight_mode": "",
    },
    {
        "policy_id": "L2T1_special_query_token_risk_suppress_plus_stable_mild",
        "policy_family": "special_query_token_risk_suppress_plus_stable",
        "query_roles": SPECIAL_QUERY_ROLES,
        "token_weight_mode": "risk_suppress_plus_stable",
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


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def latest_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    weight_map: dict[int, float],
    query_roles: list[str],
    token_weight_mode: str,
) -> str:
    token_root = TOKEN_ROOT.resolve() if token_weight_mode else ""
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
            '_stage4_anchor_source_token_roles: ["patch"]',
            f"_stage4_anchor_source_query_roles: {json.dumps(query_roles)}",
            '_stage4_anchor_source_context_roles: ["local_window_context"]',
            f"_stage4_anchor_source_token_weight_root: {json.dumps(str(token_root))}",
            f"_stage4_anchor_source_token_weight_mode: {json.dumps(token_weight_mode)}",
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
            "    vis: false",
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
        dataset = f"kitti_v116tf_l2t_token_full_{seq}"
        write_text(
            DATASET_DIR / f"{dataset}.yaml",
            "\n".join(
                [
                    "dataset: kitti",
                    f"raw_data_root: {raw_data_root}",
                    "_target_size: [504, 280]",
                    f'_sequences: ["{seq}"]',
                    "",
                ]
            ),
        )

    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    row_index = 0
    weight_map = {int(frame): 1.0 for frame in L2_SOURCE_FRAMES}

    for policy in POLICIES:
        for seq in SEQUENCES:
            dataset = f"kitti_v116tf_l2t_token_full_{seq}"
            method = f"lingbot_map_v116tf_l2t_{policy['policy_id']}_{seq}"
            action_label = f"v116tf_l2t_{policy['policy_id']}"
            config = CONFIG_ROOT / f"kitti_lingbot_v116tf_l2t_{policy['policy_id']}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{method}.jsonl"
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
                    query_roles=list(policy["query_roles"]),
                    token_weight_mode=str(policy["token_weight_mode"]),
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            expected_changed = bool(policy["token_weight_mode"])
            config_rows.append(
                {
                    "schema": "acl2_v116tf_l2t_config_row_v1",
                    "surface_id": "LB-L2T",
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
                    "query_roles": ",".join(policy["query_roles"]),
                    "token_weight_mode": policy["token_weight_mode"],
                    "token_weight_root": str(TOKEN_ROOT.resolve()) if policy["token_weight_mode"] else "",
                    "selected_count": len(weight_map),
                    "selected_global_frame_indices": ";".join(str(x) for x in sorted(weight_map)),
                    "expected_changed": expected_changed,
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
                    "schema": "acl2_v116tf_l2t_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v116tf_l2t_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"l2t_{seq}",
                    "target_kind": "l2t_prepare",
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
                f"ACL2_V108_STAGE4_SURFACE_ID=LB-L2T "
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
                        "schema": "acl2_v116tf_l2t_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v116tf_l2t_{policy['policy_id']}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"{policy['policy_id']}_{seq}",
                        "target_kind": "l2t_pilot",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": "anchor_source_attention_weight",
                        "selector": f"local:local_window_context:{','.join(policy['query_roles'])}:token={policy['token_weight_mode'] or 'none'}",
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
        "schema": "acl2_v116tf_l2t_config_summary_v1",
        "config_ready": True,
        "sequences": list(SEQUENCES),
        "policy_ids": [policy["policy_id"] for policy in POLICIES],
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "outputs": {
            "action_config_rows": rel(OUT / "action_config_rows.csv"),
            "manifest": rel(OUT / "run_manifest.csv"),
            "summary": rel(OUT / "l2t_config_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_json(OUT / "l2t_config_summary.json", summary)
    return summary


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v116tf_l2t_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v116tf_l2t_{cfg['policy_id']}_{seq}_{phase}"
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
        row
        for row in stage2m.base.load_jsonl(action_file)
        if row.get("row_type") == "anchor_source_attention_weight"
    ]
    target_rows = [row for row in rows if int(row.get("target_key_count", 0) or 0) > 0]
    changed_rows = [
        row
        for row in target_rows
        if int(row.get("changed_key_count", 0) or 0) > 0
        and int(row.get("target_query_count", 0) or 0) > 0
    ]
    expected_changed = str(cfg.get("expected_changed", "")).lower() in {"1", "true", "yes"}
    observed_context_roles = sorted({str(row.get("source_context_role", "")) for row in target_rows if row.get("source_context_role", "")})
    observed_query_roles = sorted({str(row.get("query_roles", "")) for row in target_rows if row.get("query_roles", "")})
    observed_granularity = sorted({str(row.get("action_granularity", "")) for row in target_rows if row.get("action_granularity", "")})
    token_key_counts = [int(row.get("token_weight_key_count", 0) or 0) for row in target_rows]
    target_query_counts = [int(row.get("target_query_count", 0) or 0) for row in target_rows]
    changed_query_key_counts = [int(row.get("changed_query_key_count", 0) or 0) for row in target_rows]
    run_name = f"kitti_lingbot_v116tf_l2t_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    token_mode = cfg.get("token_weight_mode", "")
    granularity_ok = (not token_mode) or ("patch_token_source_weight" in observed_granularity)
    action_fidelity_pass = bool(
        action_file.exists()
        and target_rows
        and (bool(changed_rows) == expected_changed)
        and cfg.get("source_context_roles", "") in observed_context_roles
        and cfg.get("query_roles", "") in observed_query_roles
        and granularity_ok
    )
    return {
        "schema": "acl2_v116tf_l2t_action_fidelity_row_v1",
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "token_weight_mode": token_mode,
        "expected_context_roles": cfg.get("source_context_roles", ""),
        "observed_context_roles": ";".join(observed_context_roles),
        "expected_query_roles": cfg.get("query_roles", ""),
        "observed_query_roles": ";".join(observed_query_roles),
        "observed_granularity": ";".join(observed_granularity),
        "token_weight_key_count_min": min(token_key_counts) if token_key_counts else "",
        "token_weight_key_count_max": max(token_key_counts) if token_key_counts else "",
        "expected_changed": expected_changed,
        "observed_action_frame_count": len(target_rows),
        "action_effective_frame_count": len(changed_rows),
        "target_query_count_min": min(target_query_counts) if target_query_counts else "",
        "target_query_count_max": max(target_query_counts) if target_query_counts else "",
        "changed_query_key_count_min": min(changed_query_key_counts) if changed_query_key_counts else "",
        "changed_query_key_count_max": max(changed_query_key_counts) if changed_query_key_counts else "",
        "action_file_exists": action_file.exists(),
        "action_fidelity_pass": action_fidelity_pass,
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
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if stage2m.base.boolish(row.get("action_fidelity_pass")))
        all_action = action_pass_count == len(SEQUENCES)
        median_full = stage3m.base.median(rels)
        median_rolling = stage3m.base.median(rolling)
        max_harm = stage3m.base.max_rel_harm(rels)
        metric_complete = len(rows) == len(SEQUENCES) and all(stage2m.base.boolish(row.get("metric_available")) for row in rows)
        pilot_pass = bool(
            metric_complete
            and all_action
            and max_harm <= 0.01
            and (median_full >= PILOT_GEOMETRY_REL_THRESHOLD or median_rolling >= PILOT_GEOMETRY_REL_THRESHOLD)
        )
        sample = rows[0]
        row_out: dict[str, Any] = {
            "schema": "acl2_v116tf_l2t_policy_summary_row_v1",
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
            "pilot_geometry_threshold_rel": PILOT_GEOMETRY_REL_THRESHOLD,
            "pilot_geometry_gate_pass": pilot_pass,
            "semantic_success_claim_allowed": False,
            "claim_boundary": "Task2 L2T 00/02 parity/candidate pilot only. Controls are required if candidate reaches the 3% pilot gate.",
        }
        for row in rows:
            row_out[f"seq{row['seq']}_full_rel"] = row.get("full_ATE_sim3_relative_improvement_vs_baseline", "")
        out.append(row_out)
    return out


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v116TF Task2 L2T Token Pilot Report",
        "",
        f"metric_complete: `{summary['metric_complete']}`",
        f"all_action_fidelity: `{summary['all_action_fidelity']}`",
        f"taxonomy: `{summary['taxonomy']}`",
        f"blocker: `{summary['blocker']}`",
        "",
        "## Rows",
        "",
    ]
    for row in sorted(rows, key=lambda r: safe_float(r.get("median_full_rel", "nan")), reverse=True):
        lines.append(
            "- {policy}: median_full={full} rolling_p90={rolling} max_harm={harm} action={action} gate={gate}".format(
                policy=row.get("policy_id", ""),
                full=row.get("median_full_rel", ""),
                rolling=row.get("rolling_p90_median_rel", ""),
                harm=row.get("max_full_harm", ""),
                action=row.get("all_action_fidelity", ""),
                gate=row.get("pilot_geometry_gate_pass", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This report is only the Task2 L2T parity/candidate 00/02 pilot. It is not a semantic causality success claim; matched controls must run if the candidate reaches the 3% pilot gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize() -> dict[str, Any]:
    install_metric_overrides()
    latest = latest_rows(OUT / "run_results.csv")
    config_rows = read_csv(OUT / "action_config_rows.csv")
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for row in full_rows:
        row["schema"] = "acl2_v116tf_l2t_full_metric_row_v1"
    for row in rolling_rows:
        row["schema"] = "acl2_v116tf_l2t_rolling_metric_row_v1"
    for row in local_rows:
        row["schema"] = "acl2_v116tf_l2t_local_metric_row_v1"
    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    metric_complete = bool(full_rows and all(stage2m.base.boolish(row.get("metric_available")) for row in full_rows))
    all_action = bool(fidelity_rows and all(stage2m.base.boolish(row.get("action_fidelity_pass")) for row in fidelity_rows))
    passing = [row for row in policy_rows if stage2m.base.boolish(row.get("pilot_geometry_gate_pass"))]
    summary = {
        "schema": "acl2_v116tf_l2t_metric_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "pilot_geometry_threshold_rel": PILOT_GEOMETRY_REL_THRESHOLD,
        "pilot_geometry_gate_pass_count": len(passing),
        "taxonomy": "TASK2_L2T_PILOT_GEOMETRY_GATE_PASS_CONTROLS_REQUIRED" if passing and metric_complete and all_action else "TASK2_L2T_PILOT_NO_GEOMETRY_GATE_OR_INCOMPLETE",
        "blocker": "" if passing and metric_complete and all_action else "candidate_not_over_3pct_or_metrics_incomplete",
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_summary_row_count": len(policy_rows),
        "outputs": {
            "full_metric_rows": rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_metric_rows": rel(OUT / "local_metric_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "policy_summary_rows": rel(OUT / "policy_summary_rows.csv"),
            "summary": rel(OUT / "l2t_metric_summary.json"),
            "report": rel(OUT / "L2T_TOKEN_PILOT_REPORT.md"),
        },
    }
    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_metric_rows.csv", local_rows)
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "policy_summary_rows.csv", policy_rows)
    write_json(OUT / "l2t_metric_summary.json", summary)
    write_text(OUT / "L2T_TOKEN_PILOT_REPORT.md", report_text(summary, policy_rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not args.build and not args.summarize:
        args.build = True
    if args.build:
        print(json.dumps(clean_json(build()), indent=2, sort_keys=True))
    if args.summarize:
        print(json.dumps(clean_json(summarize()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
