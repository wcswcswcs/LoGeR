#!/usr/bin/env python3
"""Build and summarize R81 s125 selected-query attention-bias matrix for seq08."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r81_lingbot_ar_fresh_s125_query_bias_matrix_08"
CONFIG_DIR = STAGE / "configs"
RUNTIME = STAGE / "runtime_full_thread8"
ACTION_DIR = RUNTIME / "action_traces"
SUMMARY_DIR = STAGE / "summary"
R56 = RESULT_ROOT / "stage4_r56_lingbot_ar_fresh_trace_baseline_08_09"
R57 = RESULT_ROOT / "stage4_r57_lingbot_ar_fresh_support_token_tensors_08_09"
WORKSPACE = R56 / "workspace"
TOKEN_ROOT = R57 / "token_semantics"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
SEQ = "08"
DATASET = f"kitti_v118_r56_fresh_seq{SEQ}"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r56_fresh_trace"
R79_BEST = "lingbot_map_stream_sdpa_v118_r79_fresh_g1p25_s125_seq08"
R71_ORIGINAL_RANDOM = "lingbot_map_stream_sdpa_v118_r71_fresh_control_safe_boundary_v3_risk_only_policy_token_random_control_seq08"
MIN_REL = 0.03
SOURCE_FRAMES = (1, 2, 5)
SPECIAL_QUERY_ROLES = ["camera_special", "register_special", "scale_special"]


METHOD_SPECS = [
    {
        "role": "s125_special_query_g1p25_candidate",
        "kind": "candidate",
        "method": "lingbot_map_stream_sdpa_v118_r81_fresh_s125_special_query_g1p25_seq08",
        "policy": "R81_FRESH_S125_SPECIAL_QUERY_G1P25",
        "query_roles": SPECIAL_QUERY_ROLES,
        "token_weight_mode": "risk_suppress_only_g1p25_x_frame",
    },
    {
        "role": "s125_patch_query_g1p25_control",
        "kind": "patch_query_control",
        "method": "lingbot_map_stream_sdpa_v118_r81_fresh_s125_patch_query_g1p25_seq08",
        "policy": "R81_FRESH_S125_PATCH_QUERY_G1P25_CONTROL",
        "query_roles": ["patch"],
        "token_weight_mode": "risk_suppress_only_g1p25_x_frame",
    },
    {
        "role": "s125_special_query_g1p25_permuted_control",
        "kind": "permuted_control",
        "method": "lingbot_map_stream_sdpa_v118_r81_fresh_s125_special_query_g1p25_permuted_seq08",
        "policy": "R81_FRESH_S125_SPECIAL_QUERY_G1P25_PERMUTED",
        "query_roles": SPECIAL_QUERY_ROLES,
        "token_weight_mode": "risk_suppress_only_g1p25_permuted_logit_x_frame",
    },
    {
        "role": "s125_special_query_g1p25_random_control",
        "kind": "random_control",
        "method": "lingbot_map_stream_sdpa_v118_r81_fresh_s125_special_query_g1p25_random_seq08",
        "policy": "R81_FRESH_S125_SPECIAL_QUERY_G1P25_RANDOM",
        "query_roles": SPECIAL_QUERY_ROLES,
        "token_weight_mode": "same_magnitude_random_risk_only_g1p25_logit_x_frame",
    },
    {
        "role": "s125_special_query_reverse_control",
        "kind": "reverse_control",
        "method": "lingbot_map_stream_sdpa_v118_r81_fresh_s125_special_query_reverse_seq08",
        "policy": "R81_FRESH_S125_SPECIAL_QUERY_REVERSE_CONTROL",
        "query_roles": SPECIAL_QUERY_ROLES,
        "token_weight_mode": "reverse_risk_x_frame",
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


def method_yaml(policy: str, query_roles: list[str], token_weight_mode: str) -> str:
    weight_map = {str(frame): 1.0 for frame in SOURCE_FRAMES}
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
            "_stage4_action_mode: anchor_source_attention_weight",
            f"_stage4_action_label: {policy}",
            f"_stage4_anchor_source_weight_map: {json.dumps(weight_map, sort_keys=True)}",
            '_stage4_anchor_source_token_roles: ["patch"]',
            f"_stage4_anchor_source_query_roles: {json.dumps(query_roles)}",
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

    methods = [str(spec["method"]) for spec in METHOD_SPECS]
    config = CONFIG_DIR / "kitti_lingbot_sdpa_v118_r81_s125_query_bias_seq08.yaml"
    write_text(CONFIG_DIR / "datasets" / f"{DATASET}.yaml", dataset_yaml())
    write_text(config, main_config(methods))

    manifest_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for gpu, spec in enumerate(METHOD_SPECS):
        method = str(spec["method"])
        token_weight_mode = str(spec["token_weight_mode"])
        query_roles = list(spec["query_roles"])
        write_text(CONFIG_DIR / "methods" / f"{method}.yaml", method_yaml(str(spec["policy"]), query_roles, token_weight_mode))
        action_trace = ACTION_DIR / f"{method}_seq{SEQ}.jsonl"
        log = RUNTIME / "logs" / f"run_seq{SEQ}_{method}.log"
        cleanup_log = RUNTIME / "logs" / f"cleanup_seq{SEQ}_{method}.log"
        method_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r81_s125_query_bias_method_row_v1",
                "seq": SEQ,
                "dataset": DATASET,
                "role": spec["role"],
                "kind": spec["kind"],
                "method": method,
                "policy": spec["policy"],
                "intervention_form": "selected_query_attention_weight",
                "source_frames": ";".join(str(frame) for frame in SOURCE_FRAMES),
                "query_roles": ",".join(query_roles),
                "token_weight_mode": token_weight_mode,
                "config": rel(config),
                "action_trace": rel(action_trace),
                "token_weight_root": rel(TOKEN_ROOT),
                "run_status": "new_r81_method",
            }
        )
        manifest_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r81_s125_query_bias_run_manifest_row_v1",
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
                "schema": "acl2_v118tf_stage4_r81_s125_query_bias_run_manifest_row_v1",
                "phase": "run_worker",
                "seq": SEQ,
                "dataset": DATASET,
                "method": method,
                "gpu": str(gpu % 6),
                "cwd": str(BENCH),
                "config": str(config.resolve()),
                "action_trace": rel(action_trace),
                "log": rel(log),
                "command": (
                    f"{run_env(method, str(spec['policy']), gpu % 6, action_trace)} "
                    f"{CONDA} run -n loger --no-capture-output python run_worker.py "
                    f"--config {config.resolve()} --method {method} --dataset {DATASET} "
                    f"--scene {SEQ} --force > {log} 2>&1"
                ),
            }
        )

    eval_log = RUNTIME / "logs" / f"evaluate_seq{SEQ}.log"
    manifest_rows.append(
        {
            "schema": "acl2_v118tf_stage4_r81_s125_query_bias_run_manifest_row_v1",
            "phase": "evaluate",
            "seq": SEQ,
            "dataset": DATASET,
            "method": ";".join(methods),
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
    write_csv(SUMMARY_DIR / "stage4_r81_s125_query_bias_method_rows.csv", method_rows)
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
    source_sets: set[str] = set()
    modes: set[str] = set()
    query_roles: set[str] = set()
    mins: list[float] = []
    maxs: list[float] = []
    target_queries: list[int] = []
    changed_query_keys: list[int] = []
    with action_trace.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "anchor_source_attention_weight":
                continue
            rows += 1
            if row.get("attention_mask_applied"):
                applied += 1
            if row.get("source_frames"):
                source_sets.add(str(row["source_frames"]))
            if row.get("token_weight_mode"):
                modes.add(str(row["token_weight_mode"]))
            if row.get("query_roles"):
                query_roles.add(str(row["query_roles"]))
            for key, bucket in (("weight_min", mins), ("weight_max", maxs)):
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    bucket.append(value)
            for key, bucket in (("target_query_count", target_queries), ("changed_query_key_count", changed_query_keys)):
                try:
                    value = int(float(row[key]))
                except (KeyError, TypeError, ValueError):
                    continue
                bucket.append(value)
    return {
        "action_trace_exists": True,
        "action_rows": rows,
        "attention_mask_applied_rows": applied,
        "observed_source_frame_sets": ";".join(sorted(source_sets)),
        "observed_token_weight_modes": ";".join(sorted(modes)),
        "observed_query_roles": ";".join(sorted(query_roles)),
        "target_query_count_min": min(target_queries) if target_queries else None,
        "target_query_count_max": max(target_queries) if target_queries else None,
        "changed_query_key_count_min": min(changed_query_keys) if changed_query_keys else None,
        "changed_query_key_count_max": max(changed_query_keys) if changed_query_keys else None,
        "weight_min_observed": min(mins) if mins else None,
        "weight_max_observed": max(maxs) if maxs else None,
    }


def rel_vs_baseline(baseline_ate: float | None, ate: float | None) -> float | None:
    if baseline_ate in (None, 0.0) or ate is None:
        return None
    return (baseline_ate - ate) / baseline_ate


def summarize(method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = read_metric(BASELINE_METHOD)
    r79_best = read_metric(R79_BEST)
    r71_random = read_metric(R71_ORIGINAL_RANDOM)
    baseline_ate = baseline.get("ate") if baseline.get("eval_exists") else None

    rows: list[dict[str, Any]] = []
    for method_row in method_rows:
        method = str(method_row["method"])
        row: dict[str, Any] = {
            **method_row,
            "schema": "acl2_v118tf_stage4_r81_s125_query_bias_result_row_v1",
        }
        row.update(read_metric(method))
        row.update(action_stats(ROOT / str(method_row["action_trace"])))
        if row.get("eval_exists"):
            row["rel_vs_baseline"] = rel_vs_baseline(baseline_ate, row["ate"])
            row["geometry_gate"] = row["rel_vs_baseline"] is not None and row["rel_vs_baseline"] >= MIN_REL
            for label, ref in (("r79_best", r79_best), ("r71_original_random", r71_random)):
                if ref.get("eval_exists"):
                    row[f"delta_ate_vs_{label}"] = row["ate"] - ref["ate"]
                    row[f"better_than_{label}"] = row["ate"] < ref["ate"]
            if baseline_ate is not None:
                row["delta_ate_vs_baseline"] = row["ate"] - baseline_ate
                row["better_than_baseline"] = row["ate"] < baseline_ate
        rows.append(row)

    candidates = [row for row in rows if row.get("kind") == "candidate" and row.get("eval_exists")]
    controls = [row for row in rows if row.get("kind") != "candidate" and row.get("eval_exists")]
    best = min(candidates, key=lambda row: row["ate"], default={})
    complete = all(row.get("eval_exists") and row.get("action_trace_exists") for row in rows)

    for row in candidates:
        row["matched_control_count"] = len(controls)
        row["better_than_all_matched_controls"] = bool(controls) and all(row["ate"] < ctrl["ate"] for ctrl in controls)
        row["best_matched_control_ate"] = min((ctrl["ate"] for ctrl in controls), default=None)
        if row["best_matched_control_ate"] is not None:
            row["delta_ate_vs_best_matched_control"] = row["ate"] - row["best_matched_control_ate"]

    promoted = [
        row
        for row in candidates
        if row.get("geometry_gate")
        and row.get("better_than_all_matched_controls")
        and row.get("better_than_r79_best")
        and row.get("better_than_r71_original_random")
    ]
    if promoted:
        decision = "R81_SPECIAL_QUERY_S125_GEOMETRY_AND_CONTROL_CANDIDATE_FOUND_REQUIRES_FRESH_VALIDATION"
    elif best and best.get("better_than_r79_best"):
        decision = "R81_SPECIAL_QUERY_S125_IMPROVES_R79_BUT_GATE_OR_CONTROLS_FAIL"
    elif complete:
        decision = "R81_SPECIAL_QUERY_S125_NO_R79_REPAIR"
    else:
        decision = "R81_CONFIG_READY_NOT_RUN"

    summary = {
        "schema": "acl2_v118tf_stage4_r81_s125_query_bias_summary_v1",
        "claim_level": "fresh_seq08_s125_selected_query_attention_bias_matrix_only",
        "complete": complete,
        "global_goal_achieved": False,
        "min_rel_gate": MIN_REL,
        "candidate_count": len(candidates),
        "control_count": len(controls),
        "baseline": baseline,
        "r79_best": r79_best,
        "r71_original_random_control": r71_random,
        "best_candidate_row": best,
        "result_rows": rows,
        "stage4_r81_decision": decision,
        "outputs": {
            "summary": rel(SUMMARY_DIR / "stage4_r81_s125_query_bias_summary.json"),
            "rows": rel(SUMMARY_DIR / "stage4_r81_s125_query_bias_rows.csv"),
            "method_rows": rel(SUMMARY_DIR / "stage4_r81_s125_query_bias_method_rows.csv"),
            "run_manifest": rel(STAGE / "run_manifest.csv"),
        },
    }
    write_csv(SUMMARY_DIR / "stage4_r81_s125_query_bias_rows.csv", rows)
    write_json(SUMMARY_DIR / "stage4_r81_s125_query_bias_summary.json", summary)
    return summary


def main() -> None:
    method_rows = build_configs()
    summary = summarize(method_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
