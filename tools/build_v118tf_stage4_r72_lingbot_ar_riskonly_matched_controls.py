#!/usr/bin/env python3
"""Build and summarize R72 risk-only matched random/permutation controls."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r72_lingbot_ar_fresh_risk_only_matched_controls_08"
CONFIG_DIR = STAGE / "configs"
RUNTIME = STAGE / "runtime_full_thread8"
ACTION_DIR = RUNTIME / "action_traces"
SUMMARY_DIR = STAGE / "summary"
R56 = RESULT_ROOT / "stage4_r56_lingbot_ar_fresh_trace_baseline_08_09"
R57 = RESULT_ROOT / "stage4_r57_lingbot_ar_fresh_support_token_tensors_08_09"
R71 = RESULT_ROOT / "stage4_r71_lingbot_ar_fresh_control_safe_boundary_v3_risk_only_policy_validation_08"
WORKSPACE = R56 / "workspace"
TOKEN_ROOT = R57 / "token_semantics"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
SEQ = "08"
DATASET = f"kitti_v118_r56_fresh_seq{SEQ}"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r56_fresh_trace"
R71_CANDIDATE = "lingbot_map_stream_sdpa_v118_r71_fresh_control_safe_boundary_v3_risk_only_policy_candidate_seq08"
R71_RANDOM = "lingbot_map_stream_sdpa_v118_r71_fresh_control_safe_boundary_v3_risk_only_policy_token_random_control_seq08"
ANCHOR_FRAMES = tuple(range(8))
METHOD_SPECS = [
    {
        "role": "risk_only_permuted_control",
        "method": "lingbot_map_stream_sdpa_v118_r72_fresh_risk_only_permuted_control_seq08",
        "policy": "R72_FRESH_RISK_ONLY_PERMUTED_DISTRIBUTION_CONTROL",
        "token_weight_mode": "risk_suppress_only_permuted_logit_x_frame",
    },
    {
        "role": "risk_only_signshuffle_control",
        "method": "lingbot_map_stream_sdpa_v118_r72_fresh_risk_only_signshuffle_control_seq08",
        "policy": "R72_FRESH_RISK_ONLY_SIGN_SHUFFLE_CONTROL",
        "token_weight_mode": "same_magnitude_random_risk_only_logit_x_frame",
    },
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def method_yaml(policy: str, token_weight_mode: str) -> str:
    weight_map = {str(frame): 1.0 for frame in ANCHOR_FRAMES}
    return "\n".join(
        [
            "model: lingbot_map",
            "env: loger",
            f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
            "_device: cuda",
            "_use_amp: true",
            "_use_sdpa: true",
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
            "_stage4_action_mode: anchor_source_value_scaling",
            f"_stage4_action_label: {policy}",
            f"_stage4_anchor_source_weight_map: {json.dumps(weight_map, sort_keys=True)}",
            '_stage4_anchor_source_token_roles: ["patch"]',
            "_stage4_anchor_source_query_roles: []",
            '_stage4_anchor_source_context_roles: ["scale_reference_context"]',
            f"_stage4_anchor_source_token_weight_root: {json.dumps(str(TOKEN_ROOT.resolve()))}",
            f"_stage4_anchor_source_token_weight_mode: {json.dumps(token_weight_mode)}",
            "",
        ]
    )


def dataset_yaml() -> str:
    return "\n".join(
        [
            "dataset: kitti",
            f"raw_data_root: {ROOT / 'data/kitti/dataset'}",
            "_target_size: [504, 280]",
            f'_sequences: ["{SEQ}"]',
            "",
        ]
    )


def main_config(methods: list[str]) -> str:
    return "\n".join(
        [
            f"workspace: {WORKSPACE}",
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
            f"  - {DATASET}",
            "",
            "methods:",
            *[f"  - {method}" for method in methods],
            "",
        ]
    )


def run_env(method: str, policy: str, gpu: int, action_trace: Path) -> str:
    return (
        f"PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={PYTHONPATH} "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"ACL2_V105_STAGE4_ACTION_FILE={action_trace.resolve()} "
        f"ACL2_V105_STAGE4_ACTION_LABEL={policy} "
        f"ACL2_V105_GCA_TRACE_DATASET={DATASET} "
        f"ACL2_V105_GCA_TRACE_SEQ={SEQ} "
        f"ACL2_V105_GCA_TRACE_METHOD={method} "
        f"ACL2_V118_LB_PROVENANCE_SEQ={SEQ}"
    )


def build_configs() -> list[dict[str, Any]]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "methods").mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "logs").mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    write_text(CONFIG_DIR / "datasets" / f"{DATASET}.yaml", dataset_yaml())
    config = CONFIG_DIR / "kitti_lingbot_sdpa_v118_r72_risk_only_matched_controls_seq08.yaml"
    write_text(config, main_config([str(spec["method"]) for spec in METHOD_SPECS]))

    manifest_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for gpu, spec in enumerate(METHOD_SPECS):
        method = str(spec["method"])
        write_text(CONFIG_DIR / "methods" / f"{method}.yaml", method_yaml(str(spec["policy"]), str(spec["token_weight_mode"])))
        action_trace = ACTION_DIR / f"{method}_seq{SEQ}.jsonl"
        log = RUNTIME / "logs" / f"run_seq{SEQ}_{method}.log"
        cleanup_log = RUNTIME / "logs" / f"cleanup_seq{SEQ}_{method}.log"
        method_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r72_matched_control_method_row_v1",
                **spec,
                "seq": SEQ,
                "dataset": DATASET,
                "config": rel(config),
                "action_trace": rel(action_trace),
                "token_weight_root": rel(TOKEN_ROOT),
            }
        )
        manifest_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r72_matched_control_run_manifest_row_v1",
                "phase": "cleanup_trace",
                "seq": SEQ,
                "dataset": DATASET,
                "method": method,
                "gpu": "",
                "cwd": str(BENCH),
                "config": str(config.resolve()),
                "action_trace": rel(action_trace),
                "log": rel(cleanup_log),
                "command": f"rm -f {action_trace.resolve()} > {cleanup_log} 2>&1",
            }
        )
        manifest_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r72_matched_control_run_manifest_row_v1",
                "phase": "run_worker",
                "seq": SEQ,
                "dataset": DATASET,
                "method": method,
                "gpu": str(gpu),
                "cwd": str(BENCH),
                "config": str(config.resolve()),
                "action_trace": rel(action_trace),
                "log": rel(log),
                "command": (
                    f"{run_env(method, str(spec['policy']), gpu, action_trace)} "
                    f"{CONDA} run -n loger --no-capture-output python run_worker.py "
                    f"--config {config.resolve()} --method {method} --dataset {DATASET} "
                    f"--scene {SEQ} --force > {log} 2>&1"
                ),
            }
        )
    eval_log = RUNTIME / "logs" / f"evaluate_seq{SEQ}.log"
    manifest_rows.append(
        {
            "schema": "acl2_v118tf_stage4_r72_matched_control_run_manifest_row_v1",
            "phase": "evaluate",
            "seq": SEQ,
            "dataset": DATASET,
            "method": ";".join(str(spec["method"]) for spec in METHOD_SPECS),
            "gpu": "",
            "cwd": str(BENCH),
            "config": str(config.resolve()),
            "action_trace": "",
            "log": rel(eval_log),
            "command": (
                f"PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH PYTHONPATH={PYTHONPATH} "
                f"{CONDA} run -n loger --no-capture-output python evaluate.py --config {config.resolve()} "
                f"--force > {eval_log} 2>&1"
            ),
        }
    )
    write_csv(SUMMARY_DIR / "stage4_r72_matched_control_method_rows.csv", method_rows)
    write_csv(STAGE / "run_manifest.csv", manifest_rows)
    return method_rows


def read_metric(method: str) -> dict[str, Any]:
    path = WORKSPACE / DATASET / SEQ / method / "eval/traj.json"
    if not path.is_file():
        return {"method": method, "eval_exists": False, "eval_json": rel(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": method,
        "eval_exists": True,
        "eval_json": rel(path),
        "ate": float(data["ate"]),
        "rpe_rot": float(data["rpe_rot"]),
        "rpe_trans": float(data["rpe_trans"]),
    }


def action_stats(action_trace: Path) -> dict[str, Any]:
    if not action_trace.is_file():
        return {"action_trace_exists": False}
    rows = 0
    applied = 0
    mins: list[float] = []
    maxs: list[float] = []
    modes: set[str] = set()
    with action_trace.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "anchor_source_value_scaling":
                continue
            rows += 1
            if row.get("value_scaling_applied"):
                applied += 1
            if row.get("token_weight_mode"):
                modes.add(str(row["token_weight_mode"]))
            for key, bucket in (("weight_min", mins), ("weight_max", maxs)):
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    bucket.append(value)
    return {
        "action_trace_exists": True,
        "action_rows": rows,
        "value_scaling_applied_rows": applied,
        "observed_token_weight_modes": ";".join(sorted(modes)),
        "weight_min_observed": min(mins) if mins else None,
        "weight_max_observed": max(maxs) if maxs else None,
    }


def summarize(method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = read_metric(BASELINE_METHOD)
    candidate = read_metric(R71_CANDIDATE)
    original_random = read_metric(R71_RANDOM)
    rows: list[dict[str, Any]] = []
    for spec in method_rows:
        method = str(spec["method"])
        metric = read_metric(method)
        trace = ROOT / str(spec["action_trace"])
        row = {
            "schema": "acl2_v118tf_stage4_r72_matched_control_result_row_v1",
            "seq": SEQ,
            "dataset": DATASET,
            "role": spec["role"],
            "method": method,
            "token_weight_mode": spec["token_weight_mode"],
            **metric,
            **action_stats(trace),
        }
        if metric.get("eval_exists") and candidate.get("eval_exists") and baseline.get("eval_exists"):
            row["candidate_ate"] = candidate["ate"]
            row["baseline_ate"] = baseline["ate"]
            row["control_minus_candidate_ate"] = metric["ate"] - candidate["ate"]
            row["candidate_better_control"] = candidate["ate"] < metric["ate"]
            row["control_rel_vs_baseline"] = (baseline["ate"] - metric["ate"]) / baseline["ate"]
        rows.append(row)
    complete = all(bool(row.get("eval_exists")) for row in rows)
    candidate_beats_exact_permutation = next(
        (bool(row.get("candidate_better_control")) for row in rows if row.get("role") == "risk_only_permuted_control"),
        False,
    )
    candidate_beats_signshuffle = next(
        (bool(row.get("candidate_better_control")) for row in rows if row.get("role") == "risk_only_signshuffle_control"),
        False,
    )
    decision = "R72_CONFIG_READY_NOT_RUN"
    if complete:
        if candidate_beats_exact_permutation and candidate_beats_signshuffle:
            decision = "R72_RISK_ONLY_BEATS_MATCHED_CONTROLS_SEQ08_LOCAL_REPAIR"
        elif candidate_beats_exact_permutation:
            decision = "R72_RISK_ONLY_BEATS_DISTRIBUTION_PERMUTATION_BUT_SIGN_SHUFFLE_REMAINS_COMPETITIVE"
        else:
            decision = "R72_RISK_ONLY_MATCHED_CONTROL_STILL_EXPLAINS_SEQ08"
    summary = {
        "schema": "acl2_v118tf_stage4_r72_matched_control_summary_v1",
        "stage4_r72_decision": decision,
        "complete": complete,
        "global_goal_achieved": False,
        "claim_level": "fresh_seq08_matched_control_diagnostic_only",
        "baseline": baseline,
        "r71_candidate": candidate,
        "r71_original_random_control": original_random,
        "candidate_beats_exact_permutation": candidate_beats_exact_permutation,
        "candidate_beats_signshuffle": candidate_beats_signshuffle,
        "result_rows": rows,
        "outputs": {
            "summary": rel(SUMMARY_DIR / "stage4_r72_matched_control_summary.json"),
            "rows": rel(SUMMARY_DIR / "stage4_r72_matched_control_rows.csv"),
            "method_rows": rel(SUMMARY_DIR / "stage4_r72_matched_control_method_rows.csv"),
            "run_manifest": rel(STAGE / "run_manifest.csv"),
        },
    }
    write_csv(SUMMARY_DIR / "stage4_r72_matched_control_rows.csv", rows)
    write_json(SUMMARY_DIR / "stage4_r72_matched_control_summary.json", summary)
    return summary


def main() -> int:
    method_rows = build_configs()
    summary = summarize(method_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
