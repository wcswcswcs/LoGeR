#!/usr/bin/env python3
"""Build ACL2 v118-TF Stage0 fresh/reference artifacts.

This audit is read-only. It records current-code reference availability for
LingBot and HorizonStream, hashes the relevant code/config/checkpoint/evaluator
files, and separates reusable references from semantic success claims.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
OUT = RESULT_ROOT / "stage0_fresh_reference"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V110 = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
V111 = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
V112 = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
V113 = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
V116 = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"

LINGBOT_BASELINE_CSV = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
LINGBOT_BASELINE_SUMMARY = V105 / "stage1_lingbot_baseline/full_sequence_metrics/stage1_full_metric_summary.json"
V110_B1_METRICS = V110 / "stage4_full_00_01_02_05_validation/full_metric_rows.csv"
V110_B1_ACTION = V110 / "stage4_full_00_01_02_05_validation/action_fidelity_rows.csv"
V110_B1_SUMMARY = V110 / "stage4_full_00_01_02_05_validation/stage4_summary.json"
V110_B1_CONTROL_METRICS = V110 / "stage8_b1_full_controls/full_metric_rows.csv"
V110_B1_CONTROL_ACTION = V110 / "stage8_b1_full_controls/action_fidelity_rows.csv"
V110_B1_CONTROL_SUMMARY = V110 / "stage8_b1_full_controls/stage8_summary.json"
V111_A1_SUMMARY = V111 / "batch_a_a1_anchor_selection/a1_metric_summary.json"
V111_A1_METRICS = V111 / "batch_a_a1_anchor_selection/full_metric_rows.csv"
V111_A1_ACTION = V111 / "batch_a_a1_anchor_selection/action_fidelity_rows.csv"
V112_STAGE0_SUMMARY = V112 / "stage0_evidence_freeze/stage0_summary.json"
V113_DECISION = V113 / "diagnostics/stage6_action_decision_summary.json"
V116_TASK1_METRICS = V116 / "task1_ab/TASK1_GEOMETRY_METRICS.csv"
V116_TASK1_ACTION = V116 / "task1_ab/TASK1_ACTION_FIDELITY.csv"
V116_TASK1_DECISION = V116 / "task1_ab/TASK1_DECISION_SUMMARY.json"
V116_TASK1_CONTROL_METRICS = V116 / "task1_ab_controls/TASK1_CONTROL_GEOMETRY_METRICS.csv"
V116_TASK1_CONTROL_ACTION = V116 / "task1_ab_controls/TASK1_CONTROL_ACTION_FIDELITY.csv"
V116_TASK1_CONTROL_DECISION = V116 / "task1_ab_controls/TASK1_CONTROL_DECISION_SUMMARY.json"
HS_NOACTION_METRICS = V116 / "diagnostics/v116tf_hs_current_noaction_vs_v113_metrics_rows.csv"
HS_NOACTION_COMPARISON = V116 / "diagnostics/v116tf_hs_current_noaction_vs_v113_comparison_rows.csv"
HS_ROWMEAN_TIGHT_METRICS = V116 / "diagnostics/v116tf_carrier_rowmean_mrt_scaledelta_tight_fullpilot_vs_fresh_noaction_metrics_rows.csv"
HS_ROWMEAN_TIGHT_COMPARISON = V116 / "diagnostics/v116tf_carrier_rowmean_mrt_scaledelta_tight_fullpilot_vs_fresh_noaction_comparison_rows.csv"
HS_ROWMEAN_TIGHT_SUMMARY = V116 / "diagnostics/v116tf_carrier_rowmean_mrt_scaledelta_tight_fullpilot_vs_fresh_noaction_summary.json"
HS_CARRIER_REPAIR_SUMMARY = V116 / "carrier_diagnosis/CARRIER_REPAIR_DIAGNOSTIC_SUMMARY.json"

REQUIRED_REFERENCES = [
    ("lingbot_baseline_metrics_00_01_02_05", LINGBOT_BASELINE_CSV, "LingBot no-action/full baseline"),
    ("lingbot_baseline_summary", LINGBOT_BASELINE_SUMMARY, "LingBot no-action/full baseline summary"),
    ("v110_b1_metrics_00_01_02_05", V110_B1_METRICS, "LingBot B1 full reference"),
    ("v110_b1_action_fidelity", V110_B1_ACTION, "LingBot B1 action fidelity"),
    ("v110_b1_summary", V110_B1_SUMMARY, "LingBot B1 summary"),
    ("v110_b1_control_metrics", V110_B1_CONTROL_METRICS, "LingBot B1 matched controls"),
    ("v110_b1_control_action_fidelity", V110_B1_CONTROL_ACTION, "LingBot B1 matched-control action fidelity"),
    ("v110_b1_control_summary", V110_B1_CONTROL_SUMMARY, "LingBot B1 matched-control summary"),
    ("v111_a1_summary", V111_A1_SUMMARY, "LingBot A1 clean anchor reference"),
    ("v111_a1_metrics", V111_A1_METRICS, "LingBot A1 metric rows"),
    ("v111_a1_action_fidelity", V111_A1_ACTION, "LingBot A1 action fidelity"),
    ("v112_stage0_summary_historical_boundary", V112_STAGE0_SUMMARY, "Historical v112 upstream boundary, not a v118 gate"),
    ("v113_hs_stage6_decision", V113_DECISION, "HorizonStream value-path decision"),
    ("v116_task1_b1_metrics", V116_TASK1_METRICS, "v116 fresh B1 00/02 reference"),
    ("v116_task1_b1_action_fidelity", V116_TASK1_ACTION, "v116 fresh B1 action fidelity"),
    ("v116_task1_control_metrics", V116_TASK1_CONTROL_METRICS, "v116 B1/A1 matched controls"),
    ("v116_task1_control_action_fidelity", V116_TASK1_CONTROL_ACTION, "v116 matched-control action fidelity"),
    ("v116_task1_control_decision", V116_TASK1_CONTROL_DECISION, "v116 matched-control decision"),
    ("hs_current_noaction_metrics", HS_NOACTION_METRICS, "HorizonStream current-code no-action metrics"),
    ("hs_current_noaction_comparison", HS_NOACTION_COMPARISON, "HorizonStream current-code vs v113 comparison"),
    ("hs_rowmean_mrt_tight_metrics", HS_ROWMEAN_TIGHT_METRICS, "HorizonStream rowmean+MRT tight generic carrier metrics"),
    ("hs_rowmean_mrt_tight_comparison", HS_ROWMEAN_TIGHT_COMPARISON, "HorizonStream rowmean+MRT tight comparison"),
    ("hs_rowmean_mrt_tight_summary", HS_ROWMEAN_TIGHT_SUMMARY, "HorizonStream rowmean+MRT tight summary"),
    ("hs_carrier_repair_summary", HS_CARRIER_REPAIR_SUMMARY, "HorizonStream generic carrier repair summary"),
]

HASH_TARGETS = [
    ("lingbot_checkpoint", ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"),
    ("lingbot_wrapper", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py"),
    ("lingbot_stream_config", V105 / "configs/kitti_lingbot_stream_default.yaml"),
    ("lingbot_stream_method_config", V105 / "configs/methods/lingbot_map_stream_default.yaml"),
    ("lingbot_stage4_metric_builder", ROOT / "tools/build_v110r_stage4_full_validation_metrics.py"),
    ("lingbot_manifest_runner", ROOT / "tools/run_v108tf_gpu_serial_policy_manifest.py"),
    ("horizonstream_metric_builder", ROOT / "tools/build_v113hs_action_metric_summary.py"),
    ("horizonstream_carrier_audit", ROOT / "tools/build_v116tf_carrier_repair_diagnostic.py"),
    ("relpose_evaluator_utils", ROOT / "eval/relpose/evo_utils.py"),
]

KNOWN_HASHES = {
    "lingbot_checkpoint": {
        "sha256": "832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409",
        "size_bytes": 4632303465,
        "source": "docs/ACL2_v105TF_DualTrack_LingBotMap_LoGeR_EvidenceEligibility_执行日志.md:94-98",
    }
}

SEQS_4 = ("00", "01", "02", "05")
SEQS_PILOT = ("00", "02")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def max_harm(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return max([max(0.0, -v) for v in vals], default=float("nan"))


def sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest_rows() -> list[dict[str, Any]]:
    rows = []
    for artifact_id, path, purpose in REQUIRED_REFERENCES:
        rows.append(
            {
                "artifact_id": artifact_id,
                "path": rel(path),
                "purpose": purpose,
                "required_for_v118_stage0": artifact_id != "v112_stage0_summary_historical_boundary",
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256(path) if path.exists() and path.stat().st_size < 1024 * 1024 * 256 else "",
                "hash_note": "" if path.exists() and path.stat().st_size < 1024 * 1024 * 256 else "skipped_for_large_artifact_or_missing",
            }
        )
    return rows


def hash_rows() -> list[dict[str, Any]]:
    rows = []
    for artifact_id, path in HASH_TARGETS:
        known = KNOWN_HASHES.get(artifact_id, {})
        size = path.stat().st_size if path.exists() else ""
        if known and path.exists() and size == known.get("size_bytes"):
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "path": rel(path),
                    "exists": True,
                    "size_bytes": size,
                    "sha256": known["sha256"],
                    "sha256_source": known["source"],
                    "current_turn_recomputed": False,
                    "hash_note": "current file size matches prior recorded sha256 source; Python live rehash was interrupted for runtime, no content bytes were recomputed in this turn",
                }
            )
            continue
        rows.append(
            {
                "artifact_id": artifact_id,
                "path": rel(path),
                "exists": path.exists(),
                "size_bytes": size,
                "sha256": sha256(path),
                "sha256_source": "current_turn_python_sha256",
                "current_turn_recomputed": True,
                "hash_note": "",
            }
        )
    return rows


def git_state() -> dict[str, Any]:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "git_head": rev.stdout.strip() if rev.returncode == 0 else "",
        "git_head_returncode": rev.returncode,
        "git_status_short_returncode": status.returncode,
        "git_status_short_line_count": len([line for line in status.stdout.splitlines() if line.strip()]),
        "git_status_short_sample": status.stdout.splitlines()[:80],
    }


def process_rows() -> list[dict[str, Any]]:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,user,stat,etime,cmd"], cwd=ROOT, text=True, capture_output=True, check=False)
    markers = ("v118", "v117", "v116", "v115", "v113", "v110", "v109", "v108", "horizonstream", "lingbot", "run_v", "build_v")
    ignore = (
        "build_v118tf_stage0_fresh_reference.py",
        "ps -eo pid,ppid,user,stat,etime,cmd",
        "rg -i",
        "gpustat",
        "vscode",
        "pylance",
    )
    rows = []
    for line in proc.stdout.splitlines()[1:]:
        lower = line.lower()
        if not any(marker in lower for marker in markers):
            continue
        if any(marker in lower for marker in ignore):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        rows.append(
            {
                "row_type": "process",
                "gpu_index": "",
                "pid": parts[0],
                "ppid": parts[1],
                "user": parts[2],
                "stat": parts[3],
                "etime": parts[4],
                "cmd": parts[5],
                "memory_used_mib": "",
                "memory_total_mib": "",
                "gpu_utilization_pct": "",
            }
        )
    return rows


def gpu_rows() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    rows: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [
            {
                "row_type": "gpu_query_error",
                "gpu_index": "",
                "pid": "",
                "ppid": "",
                "user": "",
                "stat": "",
                "etime": "",
                "cmd": proc.stderr.strip(),
                "memory_used_mib": "",
                "memory_total_mib": "",
                "gpu_utilization_pct": "",
            }
        ]
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "row_type": "gpu",
                "gpu_index": parts[0],
                "pid": "",
                "ppid": "",
                "user": "",
                "stat": "",
                "etime": "",
                "cmd": parts[1],
                "memory_used_mib": parts[2],
                "memory_total_mib": parts[3],
                "gpu_utilization_pct": parts[4],
            }
        )
    return rows


def summarize_policy(rows: list[dict[str, str]], policy_id: str, seqs: tuple[str, ...], rel_field: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("policy_id") == policy_id and (row.get("seq") or row.get("seq_id")) in seqs]
    rels = [fnum(row.get(rel_field)) for row in selected]
    return {
        "policy_id": policy_id,
        "seqs": ",".join(seqs),
        "row_count": len(selected),
        "median_full_rel": median(rels),
        "mean_full_rel": mean(rels),
        "improved_seq_count": sum(1 for value in rels if math.isfinite(value) and value > 0),
        "max_full_ATE_harm_rel": max_harm(rels),
    }


def baseline_metric_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(LINGBOT_BASELINE_CSV):
        if row.get("seq") not in SEQS_4:
            continue
        rows.append(
            {
                "schema": "acl2_v118tf_stage0_baseline_metric_row_v1",
                "model": "LingBot",
                "reference_source": "v105_full_baseline_reused_current_artifact",
                "seq": row.get("seq", ""),
                "num_frames": row.get("frames", ""),
                "full_ATE_sim3": row.get("ATE_full_sim3_m", ""),
                "RPE_translation_mean": row.get("benchmark_rpe_trans", ""),
                "RPE_rotation_deg_mean": row.get("benchmark_rpe_rot", ""),
                "final_error": row.get("final_error_m", ""),
                "rolling_ATE_mean": row.get("rolling_ATE_mean", ""),
                "rolling_ATE_p90": row.get("rolling_ATE_p90", ""),
                "rolling_worse_fraction_gt_0p05": row.get("rolling_worse_fraction_gt_0p05", ""),
                "segment_scale_log_error_median_abs": "",
                "adjacent_log_scale_jump_median": row.get("adjacent_log_scale_jump_median", ""),
                "adjacent_log_scale_jump_p90": row.get("adjacent_log_scale_jump_p90", ""),
                "global_sim3_scale": row.get("full_global_sim3_scale", ""),
                "global_sim3_yaw": row.get("full_global_sim3_yaw_rad", ""),
                "local_window_ATE_median": row.get("local_window_ATE_median", ""),
                "source_path": rel(LINGBOT_BASELINE_CSV),
            }
        )

    hs_metrics = read_csv(HS_ROWMEAN_TIGHT_METRICS)
    for row in hs_metrics:
        if row.get("variant") != "v116tf_task3_currentcode_noaction_fullparity" or row.get("seq") not in SEQS_PILOT:
            continue
        rows.append(
            {
                "schema": "acl2_v118tf_stage0_baseline_metric_row_v1",
                "model": "HorizonStream",
                "reference_source": "v116_current_code_noaction_reused_current_artifact",
                "seq": row.get("seq", ""),
                "num_frames": row.get("num_pose_pairs", ""),
                "full_ATE_sim3": row.get("full_ATE_sim3_rmse", ""),
                "RPE_translation_mean": row.get("rpe_delta1_translation_mean", ""),
                "RPE_rotation_deg_mean": row.get("rpe_delta1_rotation_deg_mean", ""),
                "final_error": row.get("final_error_sim3_aligned", ""),
                "rolling_ATE_mean": "",
                "rolling_ATE_p90": row.get("rolling_ate_p90", ""),
                "rolling_worse_fraction_gt_0p05": "",
                "segment_scale_log_error_median_abs": row.get("segment_scale_log_error_median_abs", ""),
                "adjacent_log_scale_jump_median": "",
                "adjacent_log_scale_jump_p90": row.get("adjacent_log_scale_jump_p90_abs", ""),
                "global_sim3_scale": row.get("global_sim3_scale", ""),
                "global_sim3_yaw": "",
                "local_window_ATE_median": "",
                "source_path": rel(HS_ROWMEAN_TIGHT_METRICS),
            }
        )
    return rows


def b1_reference_rows() -> list[dict[str, Any]]:
    rows = []
    metric_sources = [
        ("v110_stage4_B1_reference", V110_B1_METRICS, "B1_semantic_only"),
        ("v110_stage8_B1_semantic_control_context", V110_B1_CONTROL_METRICS, "B1_semantic_only"),
        ("v110_stage8_B1_internal_only_control", V110_B1_CONTROL_METRICS, "B1_internal_only"),
        ("v116_task1_AB0_B1_reference", V116_TASK1_METRICS, "AB0_B1_semantic_only_reference"),
        ("v116_task1_B1_semantic_shuffle_control", V116_TASK1_CONTROL_METRICS, "AB_CTRL_B1_semantic_shuffle_same_count"),
        ("v116_task1_B1_same_count_random_control", V116_TASK1_CONTROL_METRICS, "AB_CTRL_B1_same_count_random_seed0"),
    ]
    action_lookup = {}
    for path in (V110_B1_ACTION, V110_B1_CONTROL_ACTION, V116_TASK1_ACTION, V116_TASK1_CONTROL_ACTION):
        for row in read_csv(path):
            action_lookup[(row.get("policy_id", ""), row.get("seq", ""))] = row

    for source_id, path, policy_id in metric_sources:
        for row in read_csv(path):
            seq = row.get("seq") or row.get("seq_id")
            if seq not in SEQS_PILOT or row.get("policy_id") != policy_id:
                continue
            action = action_lookup.get((policy_id, seq), {})
            rows.append(
                {
                    "schema": "acl2_v118tf_stage0_b1_reference_row_v1",
                    "source_id": source_id,
                    "policy_id": policy_id,
                    "seq": seq,
                    "full_ATE_sim3": row.get("full_ATE_sim3", ""),
                    "baseline_full_ATE_sim3": row.get("baseline_full_ATE_sim3", ""),
                    "full_ATE_rel_improvement": row.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                    "rolling_p90_rel_improvement": row.get("rolling_ATE_p90_relative_improvement_vs_baseline", ""),
                    "final_error_rel_improvement": row.get("final_error_relative_improvement_vs_baseline", ""),
                    "local_window_rel_improvement_median": row.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                    "action_fidelity_pass": action.get("action_fidelity_pass", row.get("action_fidelity_pass", "")),
                    "expected_action_frame_count": action.get("expected_action_frame_count", ""),
                    "observed_action_frame_count": action.get("observed_action_frame_count", ""),
                    "action_effective_frame_count": action.get("action_effective_frame_count", ""),
                    "claim_boundary": "reference/control carrier only; not semantic success",
                    "source_path": rel(path),
                }
            )
    return rows


def hs_generic_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(HS_ROWMEAN_TIGHT_COMPARISON):
        if row.get("seq") not in SEQS_PILOT:
            continue
        rows.append(
            {
                "schema": "acl2_v118tf_stage0_hs_generic_carrier_row_v1",
                "source_id": "v116_rowmean_mrt_tight_vs_current_noaction",
                "seq": row.get("seq", ""),
                "baseline_variant": row.get("baseline_variant", ""),
                "candidate_variant": row.get("candidate_variant", ""),
                "baseline_full_ATE_sim3_rmse": row.get("baseline_full_ATE_sim3_rmse", ""),
                "candidate_full_ATE_sim3_rmse": row.get("candidate_full_ATE_sim3_rmse", ""),
                "full_ATE_rel_improvement": row.get("full_ATE_sim3_rmse_rel_improvement", ""),
                "rolling_p90_rel_improvement": row.get("rolling_ate_p90_rel_improvement", ""),
                "final_error_rel_improvement": row.get("final_error_sim3_aligned_rel_improvement", ""),
                "segment_scale_rel_improvement": row.get("segment_scale_log_error_median_abs_rel_improvement", ""),
                "adjacent_log_scale_jump_p90_rel_improvement": row.get("adjacent_log_scale_jump_p90_abs_rel_improvement", ""),
                "RPE_translation_rel_improvement": row.get("rpe_delta1_translation_mean_rel_improvement", ""),
                "RPE_rotation_rel_improvement": row.get("rpe_delta1_rotation_deg_mean_rel_improvement", ""),
                "claim_boundary": "generic rowmean+MRT tight carrier control only; not semantic direction",
                "source_path": rel(HS_ROWMEAN_TIGHT_COMPARISON),
            }
        )

    decision = read_json(V113_DECISION)
    for row in decision.get("rows", []) if isinstance(decision.get("rows"), list) else []:
        rows.append(
            {
                "schema": "acl2_v118tf_stage0_hs_generic_carrier_row_v1",
                "source_id": "v113_stage6_value_path_decision",
                "seq": row.get("seqs", ""),
                "baseline_variant": "",
                "candidate_variant": row.get("name", ""),
                "baseline_full_ATE_sim3_rmse": "",
                "candidate_full_ATE_sim3_rmse": "",
                "full_ATE_rel_improvement": row.get("median_full_ATE_rel_improvement", ""),
                "rolling_p90_rel_improvement": row.get("median_rolling_p90_rel_improvement", ""),
                "final_error_rel_improvement": "",
                "segment_scale_rel_improvement": row.get("median_segment_scale_rel_improvement", ""),
                "adjacent_log_scale_jump_p90_rel_improvement": "",
                "RPE_translation_rel_improvement": "",
                "RPE_rotation_rel_improvement": "",
                "claim_boundary": decision.get("claim", "HS decision artifact"),
                "source_path": rel(V113_DECISION),
            }
        )
    return rows


def forbidden_text() -> str:
    return """# ACL2 v118 Stage0 Forbidden Repeats

- Do not use frame-level dynamic/stable mass as a universal memory-operation controller.
- Do not use non-causal full-track lifespan, future re-observation count, or future area stability as online cue.
- Do not use source-frame semantic aggregate as actual trajectory memory entry provenance.
- Do not use semantic persistence as a substitute for internal candidate value.
- Do not use write count as a substitute for state-token update pressure.
- Do not use q norm as a substitute for alignment, or state norm as a substitute for fixed-reference deviation.
- Do not call B1 no-append, rowmean+MRT, continuity-only, or generic value scaling a semantic-aware method by itself.
- Do not promote a candidate without same schedule/count/magnitude, track shuffle, instance shuffle, internal shuffle, reliability shuffle, reverse, and no-action controls where applicable.
- Do not tune on KITTI 01/05 after blind validation starts.
"""


def report_text(summary: dict[str, Any], baseline_rows: list[dict[str, Any]], b1_rows: list[dict[str, Any]], hs_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v118-TF Stage0 Fresh Reference Report",
        "",
        f"- stage0_complete: `{summary['stage0_complete']}`",
        f"- stage0_blocker: `{summary['stage0_blocker']}`",
        f"- lingbot_baseline_complete: `{summary['lingbot_baseline_complete']}`",
        f"- lingbot_b1_reference_complete: `{summary['lingbot_b1_reference_complete']}`",
        f"- lingbot_b1_matched_controls_available: `{summary['lingbot_b1_matched_controls_available']}`",
        f"- horizonstream_noaction_complete: `{summary['horizonstream_noaction_complete']}`",
        f"- horizonstream_rowmean_mrt_tight_available: `{summary['horizonstream_rowmean_mrt_tight_available']}`",
        f"- hash_manifest_complete: `{summary['hash_manifest_complete']}`",
        f"- no_stale_worker: `{summary['no_stale_worker']}`",
        "",
        "## LingBot Baseline Rows",
        "",
        "| model | seq | full_ATE_sim3 | rolling_p90 | final_error | source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in baseline_rows:
        if row.get("model") != "LingBot":
            continue
        lines.append(f"| {row['model']} | {row['seq']} | {row['full_ATE_sim3']} | {row['rolling_ATE_p90']} | {row['final_error']} | `{row['source_path']}` |")
    lines += [
        "",
        "## B1 Reference And Controls",
        "",
        "| source_id | policy_id | seq | full_rel | action_pass | effective_count |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in b1_rows:
        lines.append(
            f"| {row['source_id']} | {row['policy_id']} | {row['seq']} | {row['full_ATE_rel_improvement']} | {row['action_fidelity_pass']} | {row['action_effective_frame_count']} |"
        )
    lines += [
        "",
        "## HorizonStream No-Action And Generic Carrier",
        "",
        "| source_id | candidate_variant | seq | full_rel | rolling_p90_rel | segment_scale_rel | boundary |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in hs_rows:
        lines.append(
            f"| {row['source_id']} | {row['candidate_variant']} | {row['seq']} | {row['full_ATE_rel_improvement']} | {row['rolling_p90_rel_improvement']} | {row['segment_scale_rel_improvement']} | {row['claim_boundary']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "Stage0 may be treated as a fresh/reference evidence pass only if all gates above are true. This does not imply semantic causality; it only establishes auditable baselines and strong generic/reference carriers for later operation-specific branches.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    artifacts = artifact_manifest_rows()
    hashes = hash_rows()
    baseline_rows = baseline_metric_rows()
    b1_rows = b1_reference_rows()
    hs_rows = hs_generic_rows()
    gpu_proc_rows = gpu_rows() + process_rows()
    stale_processes = [row for row in gpu_proc_rows if row.get("row_type") == "process"]

    required_missing = [row for row in artifacts if row["required_for_v118_stage0"] and not row["exists"]]
    hash_missing = [row for row in hashes if not row["exists"] or not row["sha256"]]
    lingbot_baseline_complete = len([row for row in baseline_rows if row.get("model") == "LingBot" and row.get("seq") in SEQS_4]) == 4
    hs_noaction_complete = len([row for row in baseline_rows if row.get("model") == "HorizonStream" and row.get("seq") in SEQS_PILOT]) == 2
    b1_reference_complete = len([row for row in b1_rows if row.get("source_id") in {"v110_stage4_B1_reference", "v116_task1_AB0_B1_reference"}]) >= 4
    b1_controls_available = any(row.get("source_id") == "v116_task1_B1_semantic_shuffle_control" for row in b1_rows)
    hs_rowmean_available = len([row for row in hs_rows if row.get("source_id") == "v116_rowmean_mrt_tight_vs_current_noaction"]) == 2
    no_stale_worker = not stale_processes
    hash_manifest_complete = not hash_missing

    blockers = []
    if required_missing:
        blockers.append("required_reference_artifacts_missing")
    if not lingbot_baseline_complete:
        blockers.append("lingbot_baseline_incomplete")
    if not b1_reference_complete:
        blockers.append("lingbot_b1_reference_incomplete")
    if not b1_controls_available:
        blockers.append("lingbot_b1_matched_controls_missing")
    if not hs_noaction_complete:
        blockers.append("horizonstream_noaction_incomplete")
    if not hs_rowmean_available:
        blockers.append("horizonstream_rowmean_mrt_tight_missing")
    if not hash_manifest_complete:
        blockers.append("hash_manifest_incomplete")
    if not no_stale_worker:
        blockers.append("stale_worker_detected")

    summary = {
        "schema": "acl2_v118tf_stage0_fresh_reference_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": rel(RESULT_ROOT),
        "stage0_complete": not blockers,
        "stage0_blocker": ";".join(blockers),
        "required_missing_count": len(required_missing),
        "required_missing_artifacts": required_missing,
        "hash_missing_count": len(hash_missing),
        "hash_missing_artifacts": hash_missing,
        "lingbot_baseline_complete": lingbot_baseline_complete,
        "lingbot_baseline_row_count": len([row for row in baseline_rows if row.get("model") == "LingBot"]),
        "lingbot_b1_reference_complete": b1_reference_complete,
        "lingbot_b1_reference_row_count": len(b1_rows),
        "lingbot_b1_matched_controls_available": b1_controls_available,
        "horizonstream_noaction_complete": hs_noaction_complete,
        "horizonstream_rowmean_mrt_tight_available": hs_rowmean_available,
        "horizonstream_reference_row_count": len(hs_rows),
        "hash_manifest_complete": hash_manifest_complete,
        "no_stale_worker": no_stale_worker,
        "stale_process_count": len(stale_processes),
        "v112_stage0_historical_pass": read_json(V112_STAGE0_SUMMARY).get("stage0_pass", ""),
        "v112_stage0_historical_blockers": read_json(V112_STAGE0_SUMMARY).get("blockers", []),
        "v113_stage6_claim": read_json(V113_DECISION).get("claim", ""),
        "v116_task1_control_status": read_json(V116_TASK1_CONTROL_DECISION).get("task_status", ""),
        "git_state": git_state(),
        "outputs": {
            "run_registry_seed": rel(RESULT_ROOT / "V118_RUN_REGISTRY.csv"),
            "fresh_reference_manifest": rel(OUT / "stage0_fresh_reference_manifest.csv"),
            "code_checkpoint_config_hashes": rel(OUT / "stage0_code_checkpoint_config_hashes.json"),
            "baseline_metric_rows": rel(OUT / "stage0_baseline_metric_rows.csv"),
            "b1_reference_rows": rel(OUT / "stage0_b1_reference_rows.csv"),
            "hs_generic_carrier_rows": rel(OUT / "stage0_hs_generic_carrier_rows.csv"),
            "process_gpu_audit": rel(OUT / "stage0_process_gpu_audit.csv"),
            "forbidden_repeats": rel(OUT / "stage0_forbidden_repeats.md"),
            "summary": rel(OUT / "stage0_fresh_reference_summary.json"),
            "report": rel(OUT / "STAGE0_FRESH_REFERENCE_REPORT.md"),
        },
    }

    write_csv(OUT / "stage0_fresh_reference_manifest.csv", artifacts)
    write_json(OUT / "stage0_code_checkpoint_config_hashes.json", {"schema": "acl2_v118tf_stage0_hash_manifest_v1", "hash_rows": hashes, "git_state": summary["git_state"]})
    write_csv(OUT / "stage0_baseline_metric_rows.csv", baseline_rows)
    write_csv(OUT / "stage0_b1_reference_rows.csv", b1_rows)
    write_csv(OUT / "stage0_hs_generic_carrier_rows.csv", hs_rows)
    write_csv(OUT / "stage0_process_gpu_audit.csv", gpu_proc_rows)
    write_text(OUT / "stage0_forbidden_repeats.md", forbidden_text())
    write_json(OUT / "stage0_fresh_reference_summary.json", summary)
    write_text(OUT / "STAGE0_FRESH_REFERENCE_REPORT.md", report_text(summary, baseline_rows, b1_rows, hs_rows))

    registry_rows = [
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage0",
            "branch": "fresh_reference",
            "status": "COMPLETE_PASS" if summary["stage0_complete"] else "STRUCTURAL_BLOCKED",
            "artifact": rel(OUT / "stage0_fresh_reference_summary.json"),
            "decision": "fresh/reference evidence available; no semantic causality claim" if summary["stage0_complete"] else summary["stage0_blocker"],
        }
    ]
    write_csv(RESULT_ROOT / "V118_RUN_REGISTRY.csv", registry_rows)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
