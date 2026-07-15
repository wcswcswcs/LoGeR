#!/usr/bin/env python3
"""Generate v107R Stage6 semantic wrapper-policy runtime action configs.

This branch reopens v107R after the diagnostic-only Stage3 No-Go.  It uses the
already audited semantic cue bank to choose frame-level LingBot memory actions
that change KV/cache write behavior at runtime, without external depth, GT, SLAM,
or post-hoc trajectory edits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_STAGE0 = V105 / "stage0_repo_env_audit/stage0_summary.json"
V107TF = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"
SEMANTIC_ROWS = V107R / "stage1_semantic_cue_bank/frame_semantic_summary.csv"
SOURCE_TARGETS = V107TF / "stage1_cache_operation_instrumentation/target_manifest.csv"
SAFE96_TARGETS = V107TF / "stage3_operation_discovery/length_control_safe96/target_manifest.csv"
OUT = V107R / "stage6_runtime_pilot_or_blocked/semantic_wrapper_policy_pilot"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_TRACE = OUT / "raw_trace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

ACTION_SPECS: dict[str, dict[str, str]] = {
    "no_action": {
        "family": "baseline",
        "mode": "no_action",
        "selector": "none",
    },
    "semantic_highrisk_force_non_keyframe": {
        "family": "semantic_action",
        "mode": "force_non_keyframe",
        "selector": "semantic_highrisk",
    },
    "semantic_highrisk_context_only": {
        "family": "semantic_action",
        "mode": "context_only_special",
        "selector": "semantic_highrisk",
    },
    "semantic_highrisk_anchor_only": {
        "family": "semantic_action",
        "mode": "anchor_special_only",
        "selector": "semantic_highrisk",
    },
    "same_count_random_force_non_keyframe": {
        "family": "required_control",
        "mode": "force_non_keyframe",
        "selector": "same_count_random",
    },
    "semantic_lowrisk_reverse_force_non_keyframe": {
        "family": "required_control",
        "mode": "force_non_keyframe",
        "selector": "semantic_lowrisk_reverse",
    },
    "semantic_highrisk_early_top6_force_non_keyframe": {
        "family": "semantic_action_shrink",
        "mode": "force_non_keyframe",
        "selector": "semantic_highrisk_early_top6",
    },
    "same_count_random_early_top6_force_non_keyframe": {
        "family": "required_control_shrink",
        "mode": "force_non_keyframe",
        "selector": "same_count_random_early_top6",
    },
    "semantic_lowrisk_early_top6_reverse_force_non_keyframe": {
        "family": "required_control_shrink",
        "mode": "force_non_keyframe",
        "selector": "semantic_lowrisk_early_top6_reverse",
    },
    "semantic_highrisk_early_top8_force_non_keyframe": {
        "family": "semantic_action_shrink",
        "mode": "force_non_keyframe",
        "selector": "semantic_highrisk_early_top8",
    },
    "same_count_random_early_top8_force_non_keyframe": {
        "family": "required_control_shrink",
        "mode": "force_non_keyframe",
        "selector": "same_count_random_early_top8",
    },
    "semantic_lowrisk_early_top8_reverse_force_non_keyframe": {
        "family": "required_control_shrink",
        "mode": "force_non_keyframe",
        "selector": "semantic_lowrisk_early_top8_reverse",
    },
    "semantic_highrisk_not_tail_top8_force_non_keyframe": {
        "family": "semantic_action_shrink",
        "mode": "force_non_keyframe",
        "selector": "semantic_highrisk_not_tail_top8",
    },
    "same_count_random_not_tail_top8_force_non_keyframe": {
        "family": "required_control_shrink",
        "mode": "force_non_keyframe",
        "selector": "same_count_random_not_tail_top8",
    },
    "semantic_lowrisk_not_tail_top8_reverse_force_non_keyframe": {
        "family": "required_control_shrink",
        "mode": "force_non_keyframe",
        "selector": "semantic_lowrisk_not_tail_top8_reverse",
    },
    "semantic_highrisk_early_risk_ge_0p60_force_non_keyframe": {
        "family": "semantic_action_threshold",
        "mode": "force_non_keyframe",
        "selector": "semantic_highrisk_early_risk_ge_0p60",
    },
    "same_count_random_early_risk_ge_0p60_force_non_keyframe": {
        "family": "required_control_threshold",
        "mode": "force_non_keyframe",
        "selector": "same_count_random_early_risk_ge_0p60",
    },
    "semantic_lowrisk_early_risk_ge_0p60_reverse_force_non_keyframe": {
        "family": "required_control_threshold",
        "mode": "force_non_keyframe",
        "selector": "semantic_lowrisk_early_risk_ge_0p60_reverse",
    },
    "semantic_highrisk_early_risk_ge_0p65_force_non_keyframe": {
        "family": "semantic_action_threshold",
        "mode": "force_non_keyframe",
        "selector": "semantic_highrisk_early_risk_ge_0p65",
    },
    "same_count_random_early_risk_ge_0p65_force_non_keyframe": {
        "family": "required_control_threshold",
        "mode": "force_non_keyframe",
        "selector": "same_count_random_early_risk_ge_0p65",
    },
    "semantic_lowrisk_early_risk_ge_0p65_reverse_force_non_keyframe": {
        "family": "required_control_threshold",
        "mode": "force_non_keyframe",
        "selector": "semantic_lowrisk_early_risk_ge_0p65_reverse",
    },
}


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


def fnum(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def irow(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def semantic_risk(row: dict[str, str]) -> float:
    """Semantic-only runtime risk score from v107R Stage1 frame cues.

    High score means "do not let this frame update patch KV/reference memory"
    in the wrapper-level pilot.  No GT or external geometry enters this score.
    """

    lowtrust = fnum(row, "unknown_lowtrust_mass") + fnum(row, "sky_or_lowobs_mass")
    unstable = (
        fnum(row, "dynamic_transient_mass")
        + fnum(row, "semantic_boundary_mass")
        + 0.5 * fnum(row, "vegetation_weak_context_mass")
        + 0.5 * fnum(row, "ground_or_road_weak_mass")
    )
    stable = fnum(row, "stable_structure_mass") + fnum(row, "road_boundary_or_layout_mass")
    update_value = fnum(row, "frame_semantic_update_value")
    purity_penalty = max(0.0, 0.75 - fnum(row, "semantic_patch_purity_p10")) * 0.5
    return unstable + lowtrust + purity_penalty - stable - 0.5 * update_value


def selected_targets() -> list[dict[str, Any]]:
    source_rows = read_csv(SOURCE_TARGETS)
    safe96_rows = read_csv(SAFE96_TARGETS)
    high = [row for row in source_rows if row.get("target_kind") == "high_l3"]
    safe96_by_original = {
        row.get("original_target_id", row["target_id"]): row
        for row in safe96_rows
    }
    safe: list[dict[str, str]] = []
    for row in source_rows:
        if row.get("target_kind") != "safe_good_low_drift":
            continue
        replacement = safe96_by_original.get(row["target_id"])
        if replacement is not None:
            safe.append(replacement)
        elif int(float(row.get("trace_frame_count", "0"))) == 96:
            safe.append(row)
    targets = high + safe
    out: list[dict[str, Any]] = []
    for row in targets:
        trace_count = irow(row, "trace_frame_count")
        if trace_count != 96:
            raise ValueError(f"Stage6 expects length-matched 96F targets, got {row['target_id']} count={trace_count}")
        target = dict(row)
        target["schema"] = "acl2_v107r_stage6_semantic_wrapper_target_v1"
        target["stage6_target_kind"] = "high_l3" if row["target_kind"] == "high_l3" else "safe_good_low_drift_96f"
        target["source_manifest"] = rel(SAFE96_TARGETS if "safe96_length_control" in row["target_id"] else SOURCE_TARGETS)
        out.append(target)
    return sorted(out, key=lambda r: (str(r["seq"]), str(r["stage6_target_kind"]), int(float(r["trace_start_idx"]))))


def semantic_rows_by_target() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv(SEMANTIC_ROWS):
        row = dict(row)
        row["risk_score"] = semantic_risk(row)
        for target_id in str(row.get("target_ids", "")).split(";"):
            if not target_id:
                continue
            out.setdefault(target_id, []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(float(r["frame_id"])))
    return out


def seed_for(*parts: str) -> int:
    raw = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def action_indices_for_target(
    target: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
    selector: str,
) -> tuple[list[int], dict[str, Any]]:
    trace_start = int(float(target["trace_start_idx"]))
    trace_count = int(float(target["trace_frame_count"]))
    eligible = [
        {
            **row,
            "local_idx": int(float(row["frame_id"])) - trace_start,
        }
        for row in semantic_rows
        if 8 <= int(float(row["frame_id"])) - trace_start < trace_count
    ]
    selector_scope = "all_after_scale"
    if "_early_" in selector:
        eligible = [row for row in eligible if 8 <= int(row["local_idx"]) <= 31]
        selector_scope = "early_local_8_to_31"
    elif "_not_tail_" in selector:
        eligible = [row for row in eligible if 8 <= int(row["local_idx"]) <= 71]
        selector_scope = "not_tail_local_8_to_71"
    if selector == "none":
        return [], {"eligible_count": len(eligible), "selected_count": 0}
    if not eligible:
        return [], {"eligible_count": 0, "selected_count": 0, "blocker": "no_semantic_rows_for_target"}

    risk_threshold: float | None = None
    if "risk_ge_0p60" in selector:
        risk_threshold = 0.60
    elif "risk_ge_0p65" in selector:
        risk_threshold = 0.65

    if risk_threshold is not None:
        selected_count = min(8, sum(1 for row in eligible if float(row["risk_score"]) >= risk_threshold))
    elif "top6" in selector:
        selected_count = min(6, len(eligible))
    elif "top8" in selector:
        selected_count = min(8, len(eligible))
    else:
        selected_count = min(12, max(4, math.ceil(0.15 * len(eligible))))
    highrisk = sorted(eligible, key=lambda r: (-float(r["risk_score"]), int(r["local_idx"])))
    lowrisk = sorted(eligible, key=lambda r: (float(r["risk_score"]), int(r["local_idx"])))
    if selector in {
        "semantic_highrisk",
        "semantic_highrisk_early_top6",
        "semantic_highrisk_early_top8",
        "semantic_highrisk_not_tail_top8",
        "semantic_highrisk_early_risk_ge_0p60",
        "semantic_highrisk_early_risk_ge_0p65",
    }:
        if risk_threshold is None:
            selected = highrisk[:selected_count]
        else:
            selected = [row for row in highrisk if float(row["risk_score"]) >= risk_threshold][:selected_count]
    elif selector in {
        "semantic_lowrisk_reverse",
        "semantic_lowrisk_early_top6_reverse",
        "semantic_lowrisk_early_top8_reverse",
        "semantic_lowrisk_not_tail_top8_reverse",
        "semantic_lowrisk_early_risk_ge_0p60_reverse",
        "semantic_lowrisk_early_risk_ge_0p65_reverse",
    }:
        selected = lowrisk[:selected_count]
    elif selector in {
        "same_count_random",
        "same_count_random_early_top6",
        "same_count_random_early_top8",
        "same_count_random_not_tail_top8",
        "same_count_random_early_risk_ge_0p60",
        "same_count_random_early_risk_ge_0p65",
    }:
        rng = random.Random(seed_for(str(target["target_id"]), selector, str(trace_start)))
        selected = rng.sample(eligible, k=min(selected_count, len(eligible)))
        selected.sort(key=lambda r: int(r["local_idx"]))
    else:
        raise ValueError(f"unknown selector: {selector}")

    indices = sorted({int(row["local_idx"]) for row in selected})
    scores = [float(row["risk_score"]) for row in selected]
    high_scores = [float(row["risk_score"]) for row in highrisk[:selected_count]]
    return indices, {
        "eligible_count": len(eligible),
        "selected_count": len(indices),
        "selected_global_indices": [trace_start + idx for idx in indices],
        "selected_risk_mean": sum(scores) / len(scores) if scores else "",
        "selected_risk_min": min(scores) if scores else "",
        "selected_risk_max": max(scores) if scores else "",
        "highrisk_reference_mean": sum(high_scores) / len(high_scores) if high_scores else "",
        "selector_scope": selector_scope,
        "risk_threshold": risk_threshold if risk_threshold is not None else "",
        "policy_note": (
            "semantic-only risk from frame_semantic_update_value, role masses, confidence/purity; "
            "GT/L3 labels are not used for frame selection"
        ),
    }


def method_yaml(checkpoint: str, env_name: str, use_sdpa: bool, action_name: str, mode: str, indices: list[int]) -> str:
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
            f"_stage4_action_mode: {mode}",
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

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)

    targets = selected_targets()
    semantic_by_target = semantic_rows_by_target()
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]
    target_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(targets):
        target_id = str(target["target_id"])
        seq = str(target["seq"])
        trace_start = int(float(target["trace_start_idx"]))
        trace_end = int(float(target["trace_end_idx_exclusive"]))
        dataset = f"kitti_v107r_stage6_{target_id}_trace96"
        target_rows.append({**target, "dataset": dataset})
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
                    f"  start_idx: {trace_start}",
                    f"  end_idx: {trace_end}",
                    "  stride: 1",
                    "",
                ]
            ),
        )

        sem_rows = semantic_by_target.get(target_id, [])
        if not sem_rows:
            raise ValueError(f"missing semantic cue rows for target_id={target_id}")

        for action_index, (action_name, spec) in enumerate(ACTION_SPECS.items()):
            mode = spec["mode"]
            selector = spec["selector"]
            indices, policy = action_indices_for_target(target, sem_rows, selector)
            method = f"lingbot_map_v107r_stage6_{action_name}_{target_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v107r_stage6_{action_name}_{target_id}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            trace_file = RAW_TRACE / f"{dataset}_{seq}_{method}.jsonl"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = gpu_cycle[(target_index + action_index) % len(gpu_cycle)]
            method_path.write_text(method_yaml(checkpoint, env_name, use_sdpa, action_name, mode, indices), encoding="utf-8")
            config.write_text(run_config_yaml(dataset, method), encoding="utf-8")

            policy_row = {
                "schema": "acl2_v107r_stage6_semantic_policy_row_v1",
                "target_id": target_id,
                "target_kind": target["stage6_target_kind"],
                "seq": seq,
                "dataset": dataset,
                "action_name": action_name,
                "action_family": spec["family"],
                "stage4_action_mode": mode,
                "selector": selector,
                "trace_start_idx": trace_start,
                "trace_end_idx_exclusive": trace_end,
                "target_frame_start": target["target_frame_start"],
                "target_frame_end": target["target_frame_end"],
                "force_non_keyframe_indices": ";".join(str(x) for x in indices),
                "force_global_frame_indices": ";".join(str(trace_start + x) for x in indices),
                **policy,
            }
            policy_rows.append(policy_row)
            config_rows.append(
                {
                    **policy_row,
                    "config": str(config.resolve()),
                    "method_config": str(method_path.resolve()),
                    "method": method,
                    "trace_file": str(trace_file.resolve()),
                    "action_file": str(action_file.resolve()),
                }
            )

            prefix = command_prefix(conda_path, pythonpath, gpu)
            trace_env = (
                f"ACL2_V107_CACHE_TRACE_FILE={trace_file.resolve()} "
                f"ACL2_V107_CACHE_TRACE_RUN_ID={target_id}_{action_name} "
                f"ACL2_V107_CACHE_TRACE_CASE={dataset}/{seq}/{method} "
                f"ACL2_V107_CACHE_TRACE_DATASET={dataset} "
                f"ACL2_V107_CACHE_TRACE_SEQ={seq} "
                f"ACL2_V107_CACHE_TRACE_METHOD={method} "
                f"ACL2_V107_CACHE_TRACE_WINDOW_ID={target['window_index']} "
                f"ACL2_V107_CACHE_TRACE_FRAME_START_IDX={trace_start} "
                f"ACL2_V107_CACHE_TRACE_GLOBAL_IDXS={target.get('trace_global_idxs', '0,5,11,17,23')} "
                "ACL2_V107_CACHE_TRACE_MAX_ROWS=240000 "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_name}"
            )
            commands = {
                "prepare": (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python prepare.py --config {config.resolve()} --force"
                ),
                "run_worker": (
                    f"{prefix} {trace_env} {conda_path} run -n {env_name} "
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
                        "schema": "acl2_v107r_stage6_semantic_wrapper_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v107r_stage6_{action_name}_{target_id}_{phase}",
                        "phase": phase,
                        "target_id": target_id,
                        "target_kind": target["stage6_target_kind"],
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_name,
                        "action_family": spec["family"],
                        "stage4_action_mode": mode,
                        "selector": selector,
                        "selected_count": len(indices),
                        "force_non_keyframe_indices": ";".join(str(x) for x in indices),
                        "trace_start_idx": trace_start,
                        "trace_end_idx_exclusive": trace_end,
                        "target_frame_start": target["target_frame_start"],
                        "target_frame_end": target["target_frame_end"],
                        "gpu": gpu,
                        "cwd": str(BENCHMARK.resolve()),
                        "config": str(config.resolve()),
                        "trace_file": str(trace_file.resolve()),
                        "action_file": str(action_file.resolve()),
                        "command": command,
                        "status": "planned",
                    }
                )

    write_csv(OUT / "target_manifest.csv", target_rows)
    write_csv(OUT / "semantic_action_policy_rows.csv", policy_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    summary = {
        "schema": "acl2_v107r_stage6_semantic_wrapper_config_summary_v1",
        "target_count": len(target_rows),
        "action_count": len(ACTION_SPECS),
        "method_count": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "actions": sorted(ACTION_SPECS),
        "target_ids": [row["target_id"] for row in target_rows],
        "workspace": rel(WORKSPACE),
        "raw_trace": rel(RAW_TRACE),
        "raw_action": rel(RAW_ACTION),
        "semantic_source": rel(SEMANTIC_ROWS),
        "runtime_boundary": "training-free LingBot wrapper KV/cache write controls only; no external depth, GT selector, SLAM, or post-hoc Sim3 action",
    }
    write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    summary = build()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
