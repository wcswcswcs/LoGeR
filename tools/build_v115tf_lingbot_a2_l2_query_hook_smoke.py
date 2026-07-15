#!/usr/bin/env python3
"""Build and summarize v115 LingBot A2/L2 query-role hook smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
OUT = RESULT_ROOT / "stage5_lingbot_a2_l2_query_hook_smoke"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
V112_CUE_ROOT = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented/stage2_memory_specific_cue_bank"
ANCHOR_CUE = V112_CUE_ROOT / "anchor_memory_cue_rows.csv"
LOCAL_CUE = V112_CUE_ROOT / "local_window_token_cue_rows.csv"

SEQUENCES = ("00", "02")
A2_SOURCE_FRAMES = tuple(range(8))
L2_SOURCE_FRAMES = tuple(range(8, 16))
SPECIAL_QUERY_ROLES = ["camera_special", "register_special", "scale_special"]


POLICIES = [
    {
        "policy_id": "LB_A2_noop_all_query_anchor_all1",
        "surface_id": "LB-A2",
        "policy_family": "default_off_parity_smoke",
        "source": "anchor",
        "source_frames": A2_SOURCE_FRAMES,
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "query_roles": [],
        "weight_mode": "all1",
    },
    {
        "policy_id": "LB_A2_special_query_anchor_risk_suppress_smoke",
        "surface_id": "LB-A2",
        "policy_family": "special_query_anchor_risk_suppress",
        "source": "anchor",
        "source_frames": A2_SOURCE_FRAMES,
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "query_roles": SPECIAL_QUERY_ROLES,
        "weight_mode": "anchor_risk_suppress",
    },
    {
        "policy_id": "LB_A2_patch_query_anchor_bias_control_smoke",
        "surface_id": "LB-A2",
        "policy_family": "patch_query_anchor_bias_control",
        "source": "anchor",
        "source_frames": A2_SOURCE_FRAMES,
        "source_context_roles": ["scale_reference_context"],
        "token_roles": ["patch"],
        "query_roles": ["patch"],
        "weight_mode": "anchor_risk_suppress",
    },
    {
        "policy_id": "LB_L2_special_query_local_risk_suppress_smoke",
        "surface_id": "LB-L2",
        "policy_family": "special_query_local_risk_suppress",
        "source": "local",
        "source_frames": L2_SOURCE_FRAMES,
        "source_context_roles": ["local_window_context"],
        "token_roles": ["patch"],
        "query_roles": SPECIAL_QUERY_ROLES,
        "weight_mode": "local_risk_suppress",
    },
    {
        "policy_id": "LB_L2_all_query_local_control_smoke",
        "surface_id": "LB-L2",
        "policy_family": "all_query_local_control",
        "source": "local",
        "source_frames": L2_SOURCE_FRAMES,
        "source_context_roles": ["local_window_context"],
        "token_roles": ["patch"],
        "query_roles": [],
        "weight_mode": "local_risk_suppress",
    },
]


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
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def anchor_cues() -> dict[tuple[str, int], dict[str, str]]:
    return {
        (row.get("seq", ""), int(float(row.get("frame_id", -1)))): row
        for row in read_csv(ANCHOR_CUE)
        if row.get("seq") in SEQUENCES
    }


def local_cues() -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    for row in read_csv(LOCAL_CUE):
        if row.get("seq") not in SEQUENCES:
            continue
        if row.get("context_type") != "local_pose_reference_window":
            continue
        if row.get("token_type") != "image_patch":
            continue
        source_frame = int(float(row.get("source_frame", -1)))
        out.setdefault((row["seq"], source_frame), row)
    return out


def anchor_weight(row: dict[str, str], mode: str) -> float:
    if mode == "all1":
        return 1.0
    dynamic = safe_float(row.get("dynamic_mass"))
    boundary = safe_float(row.get("boundary_mass"))
    lifetime = safe_float(row.get("semantic_lifetime_risk"))
    stable = safe_float(row.get("stable_landmark_mass"))
    continuity = safe_float(row.get("semantic_continuity_score"))
    risk = 1.4 * dynamic + boundary + 0.5 * lifetime
    support = stable + 0.25 * continuity
    return clamp(math.exp(-0.8 * risk + 0.2 * support), 0.25, 1.25)


def local_weight(row: dict[str, str], mode: str) -> float:
    if mode == "all1":
        return 1.0
    risk = safe_float(row.get("R_local"))
    support = safe_float(row.get("S_local"))
    return clamp(math.exp(-0.7 * risk + 0.2 * support), 0.25, 1.25)


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    weight_map: dict[int, float],
    token_roles: list[str],
    query_roles: list[str],
    source_context_roles: list[str],
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
            f"_stage4_anchor_source_token_roles: {json.dumps(token_roles)}",
            f"_stage4_anchor_source_query_roles: {json.dumps(query_roles)}",
            f"_stage4_anchor_source_context_roles: {json.dumps(source_context_roles)}",
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
    cue_by_anchor = anchor_cues()
    cue_by_local = local_cues()
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v115tf_lingbot_hook_smoke_{seq}_trace32"
        write_text(
            DATASET_DIR / f"{dataset}.yaml",
            "\n".join(
                [
                    "dataset: kitti",
                    f"raw_data_root: {raw_data_root}",
                    "_target_size: [504, 280]",
                    f"_sequences: [\"{seq}\"]",
                    "sampling:",
                    "  strategy: sequence",
                    "  num_frames: 32",
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

    for policy in POLICIES:
        for seq in SEQUENCES:
            weight_map: dict[int, float] = {}
            for source_frame in policy["source_frames"]:
                cue = (
                    cue_by_anchor.get((seq, int(source_frame)), {})
                    if policy["source"] == "anchor"
                    else cue_by_local.get((seq, int(source_frame)), {})
                )
                if not cue:
                    missing_cues.append(
                        {
                            "policy_id": policy["policy_id"],
                            "seq": seq,
                            "source": policy["source"],
                            "source_frame": source_frame,
                        }
                    )
                    continue
                weight = (
                    anchor_weight(cue, policy["weight_mode"])
                    if policy["source"] == "anchor"
                    else local_weight(cue, policy["weight_mode"])
                )
                weight_map[int(source_frame)] = weight
                weight_rows.append(
                    {
                        "schema": "acl2_v115tf_lingbot_query_hook_weight_row_v1",
                        "policy_id": policy["policy_id"],
                        "surface_id": policy["surface_id"],
                        "policy_family": policy["policy_family"],
                        "seq": seq,
                        "source": policy["source"],
                        "source_frame": int(source_frame),
                        "source_context_roles": ",".join(policy["source_context_roles"]),
                        "token_roles": ",".join(policy["token_roles"]),
                        "query_roles": ",".join(policy["query_roles"]) if policy["query_roles"] else "all",
                        "weight": weight,
                        "weight_mode": policy["weight_mode"],
                        "dynamic_mass": cue.get("dynamic_mass", ""),
                        "boundary_mass": cue.get("boundary_mass", ""),
                        "stable_landmark_mass": cue.get("stable_landmark_mass", ""),
                        "R_local": cue.get("R_local", ""),
                        "S_local": cue.get("S_local", ""),
                    }
                )

            dataset = f"kitti_v115tf_lingbot_hook_smoke_{seq}_trace32"
            method = f"lingbot_map_v115tf_{policy['policy_id']}_{seq}"
            action_label = f"v115tf_{policy['policy_id']}"
            config = CONFIG_ROOT / f"kitti_lingbot_v115tf_{policy['policy_id']}_{seq}.yaml"
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
                    token_roles=list(policy["token_roles"]),
                    query_roles=list(policy["query_roles"]),
                    source_context_roles=list(policy["source_context_roles"]),
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            weights = list(weight_map.values())
            changed = [value for value in weights if abs(value - 1.0) > 1e-12]
            config_rows.append(
                {
                    "schema": "acl2_v115tf_lingbot_query_hook_config_row_v1",
                    "surface_id": policy["surface_id"],
                    "policy_id": policy["policy_id"],
                    "policy_family": policy["policy_family"],
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": action_label,
                    "stage4_action_mode": "anchor_source_attention_weight",
                    "source": policy["source"],
                    "source_context_roles": ",".join(policy["source_context_roles"]),
                    "token_roles": ",".join(policy["token_roles"]),
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
                    "schema": "acl2_v115tf_lingbot_query_hook_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v115tf_hook_smoke_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"hook_smoke_{seq}",
                    "target_kind": "hook_smoke_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "trace32_prepare_once_per_seq",
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
                f"ACL2_V108_STAGE4_SURFACE_ID={policy['surface_id']} "
                f"ACL2_V112_A2_ACTION_GLOBAL_IDXS=0 "
                f"ACL2_V112_A2_ACTION_MAX_ROWS=5000"
            )
            manifest_rows.append(
                {
                    "schema": "acl2_v115tf_lingbot_query_hook_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v115tf_{policy['policy_id']}_{seq}_run_worker",
                    "phase": "run_worker",
                    "target_id": f"{policy['policy_id']}_{seq}",
                    "target_kind": "hook_smoke",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": action_label,
                    "action_family": policy["policy_family"],
                    "stage4_action_mode": "anchor_source_attention_weight",
                    "selector": f"{policy['source']}:{','.join(policy['source_context_roles'])}:{','.join(policy['query_roles']) or 'all'}",
                    "selected_count": len(weight_map),
                    "gpu": gpu,
                    "cwd": str(BENCHMARK.resolve()),
                    "config": str(config.resolve()),
                    "trace_file": "",
                    "action_file": str(action_file.resolve()),
                    "command": (
                        f"{prefix} {action_env} {conda_path} run -n {env_name} "
                        f"python run_worker.py --config {config.resolve()} --method {method} "
                        f"--dataset {dataset} --scene {seq} --force"
                    ),
                    "status": "planned",
                }
            )

    manifest_rows = [prepare_rows_by_seq[seq] for seq in SEQUENCES] + manifest_rows
    summary = {
        "schema": "acl2_v115tf_lingbot_query_hook_config_summary_v1",
        "config_ready": not missing_cues,
        "blocker": "missing_cue_rows" if missing_cues else "",
        "sequences": list(SEQUENCES),
        "policy_ids": [policy["policy_id"] for policy in POLICIES],
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "missing_cue_rows": missing_cues,
        "outputs": {
            "action_config_rows": rel(OUT / "action_config_rows.csv"),
            "weight_rows": rel(OUT / "source_weight_rows.csv"),
            "manifest": rel(OUT / "run_manifest.csv"),
            "summary": rel(OUT / "query_hook_config_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "source_weight_rows.csv", weight_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    write_json(OUT / "query_hook_config_summary.json", summary)
    return summary


def latest_run_results() -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(OUT / "run_results.csv"):
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize() -> dict[str, Any]:
    config_rows = read_csv(OUT / "action_config_rows.csv")
    latest = latest_run_results()
    fidelity_rows: list[dict[str, Any]] = []
    for cfg in config_rows:
        action_file = Path(cfg["action_file"])
        rows = [
            row for row in read_jsonl(action_file)
            if row.get("row_type") == "anchor_source_attention_weight"
        ]
        target_rows = [row for row in rows if int(row.get("target_key_count", 0) or 0) > 0]
        changed_rows = [
            row for row in target_rows
            if int(row.get("changed_key_count", 0) or 0) > 0
            and int(row.get("target_query_count", 0) or 0) > 0
        ]
        target_query_counts = [int(row.get("target_query_count", 0) or 0) for row in target_rows]
        changed_query_key_counts = [int(row.get("changed_query_key_count", 0) or 0) for row in target_rows]
        expected_changed = int(float(cfg.get("changed_source_frame_count", "0") or 0)) > 0
        observed_context_roles = sorted({str(row.get("source_context_role", "")) for row in target_rows if row.get("source_context_role", "")})
        observed_query_roles = sorted({str(row.get("query_roles", "")) for row in target_rows if row.get("query_roles", "")})
        run_name = f"kitti_lingbot_v115tf_{cfg['policy_id']}_{cfg['seq']}_run_worker"
        run_row = latest.get((run_name, "run_worker"), {})
        action_fidelity_pass = bool(
            action_file.exists()
            and target_rows
            and (bool(changed_rows) == expected_changed)
            and cfg.get("source_context_roles", "") in observed_context_roles
            and cfg.get("query_roles", "") in observed_query_roles
        )
        fidelity_rows.append(
            {
                "schema": "acl2_v115tf_lingbot_query_hook_fidelity_row_v1",
                "surface_id": cfg["surface_id"],
                "policy_id": cfg["policy_id"],
                "policy_family": cfg["policy_family"],
                "seq": cfg["seq"],
                "dataset": cfg["dataset"],
                "method": cfg["method"],
                "source": cfg["source"],
                "expected_context_roles": cfg.get("source_context_roles", ""),
                "observed_context_roles": ";".join(observed_context_roles),
                "expected_query_roles": cfg.get("query_roles", ""),
                "observed_query_roles": ";".join(observed_query_roles),
                "expected_changed": expected_changed,
                "action_file_exists": action_file.exists(),
                "action_log_rows": len(rows),
                "target_rows": len(target_rows),
                "changed_rows": len(changed_rows),
                "target_query_count_min": min(target_query_counts) if target_query_counts else "",
                "target_query_count_max": max(target_query_counts) if target_query_counts else "",
                "changed_query_key_count_min": min(changed_query_key_counts) if changed_query_key_counts else "",
                "changed_query_key_count_max": max(changed_query_key_counts) if changed_query_key_counts else "",
                "action_fidelity_pass": action_fidelity_pass,
                "run_worker_returncode": run_row.get("returncode", ""),
                "run_worker_duration_sec": run_row.get("duration_sec", ""),
                "action_file": rel(action_file),
            }
        )
    passed = [row for row in fidelity_rows if bool_value(row.get("action_fidelity_pass"))]
    a2_pass = any(row["surface_id"] == "LB-A2" and bool_value(row["action_fidelity_pass"]) for row in fidelity_rows)
    l2_pass = any(row["surface_id"] == "LB-L2" and bool_value(row["action_fidelity_pass"]) for row in fidelity_rows)
    all_run_success = bool(fidelity_rows) and all(str(row.get("run_worker_returncode", "")) == "0" for row in fidelity_rows)
    all_action_fidelity = bool(fidelity_rows) and len(passed) == len(fidelity_rows)
    summary = {
        "schema": "acl2_v115tf_lingbot_query_hook_smoke_summary_v1",
        "run_worker_complete": all_run_success,
        "all_action_fidelity": all_action_fidelity,
        "a2_hook_smoke_pass": a2_pass,
        "l2_hook_smoke_pass": l2_pass,
        "fidelity_row_count": len(fidelity_rows),
        "passed_fidelity_row_count": len(passed),
        "taxonomy": (
            "LB_A2_L2_QUERY_HOOK_SMOKE_PASS"
            if all_run_success and all_action_fidelity and a2_pass and l2_pass
            else "LB_A2_L2_QUERY_HOOK_SMOKE_INCOMPLETE_OR_FAIL"
        ),
        "outputs": {
            "fidelity_rows": rel(OUT / "query_hook_fidelity_rows.csv"),
            "summary": rel(OUT / "query_hook_smoke_summary.json"),
            "report": rel(OUT / "LB_A2_L2_QUERY_HOOK_SMOKE_REPORT.md"),
        },
    }
    write_csv(OUT / "query_hook_fidelity_rows.csv", fidelity_rows)
    write_json(OUT / "query_hook_smoke_summary.json", summary)
    write_text(OUT / "LB_A2_L2_QUERY_HOOK_SMOKE_REPORT.md", report_text(summary, fidelity_rows))
    return summary


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v115TF LingBot A2/L2 Query Hook Smoke Report",
        "",
        f"taxonomy: `{summary['taxonomy']}`",
        f"run_worker_complete: `{summary['run_worker_complete']}`",
        f"all_action_fidelity: `{summary['all_action_fidelity']}`",
        f"a2_hook_smoke_pass: `{summary['a2_hook_smoke_pass']}`",
        f"l2_hook_smoke_pass: `{summary['l2_hook_smoke_pass']}`",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            "- {policy} seq{seq}: pass={passed} context={context} query={query} "
            "target_q={qmin}..{qmax} changed_qk={ckmin}..{ckmax} rc={rc}".format(
                policy=row.get("policy_id", ""),
                seq=row.get("seq", ""),
                passed=row.get("action_fidelity_pass", ""),
                context=row.get("observed_context_roles", ""),
                query=row.get("observed_query_roles", ""),
                qmin=row.get("target_query_count_min", ""),
                qmax=row.get("target_query_count_max", ""),
                ckmin=row.get("changed_query_key_count_min", ""),
                ckmax=row.get("changed_query_key_count_max", ""),
                rc=row.get("run_worker_returncode", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This smoke validates source-context and query-role action fidelity in the real LingBot benchmark forward path. It is not a full ATE pilot and does not establish semantic causality.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not args.build and not args.summarize:
        args.build = True
    if args.build:
        summary = build()
        print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    if args.summarize:
        summary = summarize()
        print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
