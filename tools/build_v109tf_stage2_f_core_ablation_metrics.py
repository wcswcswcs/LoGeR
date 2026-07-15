#!/usr/bin/env python3
"""Build ACL2 v109TF Stage2 F-core ablation metrics and causal report."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_metrics as base  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
OUT = RESULT_ROOT / "stage2_f_core_ablation"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_METRICS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
V105_WORKSPACE = V105 / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05"
V105_METHOD = "lingbot_map_stream_default"

SEQUENCES = ("00", "01", "02", "05")
SEMANTIC_PLUS = "F1_semantic_plus_internal"
INTERNAL_ONLY = "F2_internal_only"
SEMANTIC_ONLY = "F3_semantic_only"
SHUFFLE_POLICIES = ("F4_semantic_shuffle_seed0", "F5_semantic_shuffle_seed1", "F6_semantic_shuffle_seed2")
ROLE_ROTATION = "F7_role_rotation"
RANDOM_POLICIES = ("F8_same_count_random_seed0", "F9_same_count_random_seed1", "F10_same_count_random_seed2")
LOW_RISK_REVERSE = "F11_low_risk_reverse"
CONTROL_POLICIES = (
    INTERNAL_ONLY,
    SEMANTIC_ONLY,
    *SHUFFLE_POLICIES,
    ROLE_ROTATION,
    *RANDOM_POLICIES,
    LOW_RISK_REVERSE,
)

CONTROL_MATCH_TOL = 0.005


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
    path.write_text(json.dumps(base.clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def safe_rc(row: dict[str, str] | None) -> int:
    if not row:
        return 1
    try:
        return int(float(row.get("returncode", 1)))
    except (TypeError, ValueError):
        return 1


def latest_run_results(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def latest_phase_rows(latest: dict[tuple[str, str], dict[str, str]], phase: str) -> list[dict[str, str]]:
    return [row for (_run_name, row_phase), row in latest.items() if row_phase == phase]


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v109tf_stage2_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v109tf_stage2_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def action_fidelity_row(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    action_rows = base.load_jsonl(action_file)
    expected = base.parse_indices(cfg.get("selected_global_frame_indices", ""))
    expected_field = cfg.get("expected_action_field", "")
    mode = cfg.get("stage2_action_mode") or cfg.get("stage4_action_mode", "")
    observed = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get(expected_field, False))
    }
    base_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get("base_is_keyframe", False))
    }
    final_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get("final_is_keyframe", False))
    }
    effective: set[int] = set()
    trace_error_rows = 0
    for row in action_rows:
        try:
            sample = int(float(row.get("sample_position", -1)))
        except ValueError:
            trace_error_rows += 1
            continue
        if mode == "anchor_special_only":
            if (
                base.boolish(row.get("forced_anchor_only", False))
                and base.boolish(row.get("forced_context_only", False))
                and base.boolish(row.get("context_only_append", False))
                and str(row.get("context_only_special_mode", "")) == "scale_only"
            ):
                effective.add(sample)
        elif base.boolish(row.get(expected_field, False)):
            effective.add(sample)

    missing = expected - observed
    unexpected = observed - expected
    ineffective = expected - effective
    action_fidelity_pass = (
        action_file.exists()
        and observed == expected
        and effective == expected
        and trace_error_rows == 0
    )
    run_name = f"kitti_lingbot_v109tf_stage2_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    return {
        "schema": "acl2_v109tf_stage2_action_fidelity_row_v1",
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage2_action_mode": mode,
        "stage4_action_mode": cfg.get("stage4_action_mode", mode),
        "expected_action_field": expected_field,
        "expected_action_frame_count": len(expected),
        "observed_action_frame_count": len(observed),
        "action_effective_frame_count": len(effective),
        "action_noop_frame_count": len(ineffective),
        "expected_keyframe_count": len(expected),
        "observed_keyframe_count": len(observed & base_keyframes),
        "special_token_operation_count": len(effective) if mode == "anchor_special_only" else "",
        "trace_error_rows": trace_error_rows,
        "action_file_exists": action_file.exists(),
        "action_fidelity_pass": action_fidelity_pass,
        "observed_action_indices": ";".join(str(x) for x in sorted(observed)),
        "effective_action_indices": ";".join(str(x) for x in sorted(effective)),
        "missing_expected_indices": ";".join(str(x) for x in sorted(missing)),
        "unexpected_observed_indices": ";".join(str(x) for x in sorted(unexpected)),
        "ineffective_expected_indices": ";".join(str(x) for x in sorted(ineffective)),
        "base_keyframe_count_observed_log": len(base_keyframes),
        "final_keyframe_count_observed_log": len(final_keyframes),
        "action_log_rows": len(action_rows),
        "action_file": rel(action_file),
        "run_worker_returncode": run_row.get("returncode", ""),
        "run_worker_duration_sec": run_row.get("duration_sec", ""),
    }


def no_action_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_by_seq = {row["seq"]: row for row in read_csv(STAGE0 / "full_kitti_baseline_table.csv")}
    for seq in SEQUENCES:
        metric = metric_by_seq.get(seq, {})
        rows.append(
            {
                "schema": "acl2_v109tf_stage2_no_action_control_row_v1",
                "policy_family": "NO_ACTION",
                "policy_id": "F0_no_action",
                "seq": seq,
                "dataset": metric.get("dataset", "kitti_v105_00_01_02_05"),
                "method": metric.get("method", V105_METHOD),
                "full_ATE_sim3": metric.get("ATE_full_sim3_m", ""),
                "final_error_m": metric.get("final_error_m", ""),
                "rolling_ATE_p90": metric.get("rolling_ATE_p90", ""),
                "local_window_ATE_median": metric.get("local_window_ATE_median", ""),
                "source": metric.get("source", rel(V105_METRICS)),
                "source_note": "frozen Stage0/v105 full KITTI baseline; not rerun in v109 Stage2",
            }
        )
    return rows


def build_policy_selected_rows(fidelity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fidelity_by_key = {
        (row["policy_id"], row["seq"]): row
        for row in fidelity_rows
    }
    semantic_by_key = {
        (row["policy_id"], row["seq"], row["source_frame"]): row
        for row in read_csv(OUT / "policy_source_frame_rows.csv")
    }
    rows: list[dict[str, Any]] = []
    for snap in read_csv(OUT / "keyframe_snap_rows.csv"):
        policy_id = snap["policy_id"]
        seq = snap["seq"]
        source = snap["source_frame"]
        snapped = snap["snapped_base_keyframe"]
        sem = semantic_by_key.get((policy_id, seq, source), {})
        fidelity = fidelity_by_key.get((policy_id, seq), {})
        effective = base.parse_indices(str(fidelity.get("effective_action_indices", "")))
        observed = base.parse_indices(str(fidelity.get("observed_action_indices", "")))
        snapped_int = int(float(snapped))
        rows.append(
            {
                "schema": "acl2_v109tf_stage2_policy_selected_frame_row_v1",
                "surface_id": snap["surface_id"],
                "policy_id": policy_id,
                "policy_family": snap["policy_family"],
                "seq": seq,
                "source_frame": source,
                "snapped_base_keyframe": snapped,
                "snap_distance": snap.get("distance", ""),
                "snap_accepted": snap.get("accepted", ""),
                "snapped_observed_action": snapped_int in observed,
                "snapped_effective_action": snapped_int in effective,
                "Q_ref_sem_balanced": sem.get("Q_ref_sem_balanced", ""),
                "Q_ref_sem_risk_strict": sem.get("Q_ref_sem_risk_strict", ""),
                "Q_ref_sem_stable_strict": sem.get("Q_ref_sem_stable_strict", ""),
                "stable_structure_mass": sem.get("stable_structure_mass", ""),
                "dynamic_mass": sem.get("dynamic_mass", ""),
                "boundary_mass": sem.get("boundary_mass", ""),
                "weak_context_mass": sem.get("weak_context_mass", ""),
                "road_ground_mass": sem.get("road_ground_mass", ""),
                "sky_lowobs_mass": sem.get("sky_lowobs_mass", ""),
                "semantic_trust_mean": sem.get("semantic_trust_mean", ""),
                "semantic_continuity_score": sem.get("semantic_continuity_score", ""),
            }
        )
    return rows


def metric_rows(
    config_rows: list[dict[str, str]],
    latest: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    v105_csv = {row["seq"]: row for row in read_csv(V105_METRICS)}
    baseline_cache: dict[str, dict[str, Any]] = {}
    baseline_rolling_cache: dict[str, dict[str, Any]] = {}
    full_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []

    for cfg in config_rows:
        fidelity = action_fidelity_row(cfg, latest)
        fidelity_rows.append(fidelity)
        seq = cfg["seq"]
        dataset = cfg["dataset"]
        method = cfg["method"]
        method_root = WORKSPACE / dataset / seq / method
        action_gt = WORKSPACE / dataset / seq / "gt/traj.txt"
        action_traj = method_root / "traj.txt"
        action_eval = base.load_json(method_root / "eval/traj.json")
        baseline_gt = V105_WORKSPACE / seq / "gt/traj.txt"
        baseline_traj = V105_WORKSPACE / seq / V105_METHOD / "traj.txt"
        baseline_eval = base.load_json(V105_WORKSPACE / seq / V105_METHOD / "eval/traj.json")
        phase_status, all_metric_phase_success, all_phase_success = phase_status_for(cfg, latest)

        action_available = action_gt.exists() and action_traj.exists()
        baseline_available = baseline_gt.exists() and baseline_traj.exists()
        action_blocker = "" if action_available else "missing_action_traj_or_gt"
        baseline_blocker = "" if baseline_available else "missing_v105_baseline_traj_or_gt"
        action_align: dict[str, Any] | None = None
        baseline_align: dict[str, Any] | None = None
        try:
            if action_available:
                action_align = base.align_traj(action_gt, action_traj)
                action_blocker = action_align["frame_blocker"]
        except Exception as exc:  # noqa: BLE001
            action_available = False
            action_blocker = f"{type(exc).__name__}: {exc}"
        try:
            if seq in baseline_cache:
                baseline_align = baseline_cache[seq]
            elif baseline_available:
                baseline_align = base.align_traj(baseline_gt, baseline_traj)
                baseline_cache[seq] = baseline_align
                baseline_blocker = baseline_align["frame_blocker"]
        except Exception as exc:  # noqa: BLE001
            baseline_available = False
            baseline_blocker = f"{type(exc).__name__}: {exc}"

        metric_available = action_available and baseline_available and action_align is not None and baseline_align is not None
        if metric_available:
            action_res = action_align["residual"]
            baseline_res = baseline_align["residual"]
            action_rolling = base.rolling_summary(action_res)
            if seq not in baseline_rolling_cache:
                baseline_rolling_cache[seq] = base.rolling_summary(baseline_res)
            baseline_rolling = baseline_rolling_cache[seq]
            action_ate = base.rmse_values(action_res)
            baseline_ate = base.rmse_values(baseline_res)
            action_final = float(action_res[-1])
            baseline_final = float(baseline_res[-1])
            local_for_cfg = base.local_rows_for(cfg, action_align, baseline_align)
            for row in local_for_cfg:
                row["schema"] = "acl2_v109tf_stage2_local_handoff_metric_row_v1"
            local_rows.extend(local_for_cfg)
            local_summary = base.summarize_local_rows(local_for_cfg)
        else:
            action_rolling = base.rolling_summary(np.asarray([], dtype=np.float64))
            baseline_rolling = base.rolling_summary(np.asarray([], dtype=np.float64))
            action_ate = float("nan")
            baseline_ate = float("nan")
            action_final = float("nan")
            baseline_final = float("nan")
            local_summary = base.summarize_local_rows([])

        rolling_p90_rel = base.rel_improvement(
            safe_float(baseline_rolling["rolling_ATE_p90"]),
            safe_float(action_rolling["rolling_ATE_p90"]),
        )
        rolling_mean_rel = base.rel_improvement(
            safe_float(baseline_rolling["rolling_ATE_mean"]),
            safe_float(action_rolling["rolling_ATE_mean"]),
        )
        final_rel = base.rel_improvement(baseline_final, action_final)
        baseline_csv_ate = safe_float(v105_csv.get(seq, {}).get("ATE_full_sim3_m", "nan"))
        full_row = {
            "schema": "acl2_v109tf_stage2_full_metric_row_v1",
            "surface_id": cfg["surface_id"],
            "policy_id": cfg["policy_id"],
            "policy_family": cfg["policy_family"],
            "seq_id": seq,
            "seq": seq,
            "dataset": dataset,
            "method": method,
            "action_name": cfg["action_name"],
            "num_frames": int(action_align["gt_frame_count"]) if action_align else "",
            "valid_frame_count": int(action_align["common_frame_count"]) if action_align else "",
            "full_ATE_sim3": action_ate,
            "baseline_full_ATE_sim3": baseline_ate,
            "baseline_csv_full_ATE_sim3": baseline_csv_ate,
            "baseline_csv_recompute_abs_diff": abs(baseline_csv_ate - baseline_ate)
            if math.isfinite(baseline_csv_ate) and math.isfinite(baseline_ate)
            else float("nan"),
            "full_ATE_sim3_delta_action_minus_baseline": action_ate - baseline_ate
            if math.isfinite(action_ate) and math.isfinite(baseline_ate)
            else float("nan"),
            "full_ATE_sim3_relative_improvement_vs_baseline": base.rel_improvement(baseline_ate, action_ate),
            "full_RPE_translation": action_eval.get("rpe_trans", ""),
            "full_RPE_rotation": action_eval.get("rpe_rot", ""),
            "baseline_RPE_translation": baseline_eval.get("rpe_trans", ""),
            "baseline_RPE_rotation": baseline_eval.get("rpe_rot", ""),
            "benchmark_ate": action_eval.get("ate", ""),
            "baseline_benchmark_ate": baseline_eval.get("ate", ""),
            "final_error_m": action_final,
            "baseline_final_error_m": baseline_final,
            "final_error_relative_improvement_vs_baseline": final_rel,
            "global_sim3_scale": float(action_align["scale"]) if action_align else float("nan"),
            "baseline_global_sim3_scale": float(baseline_align["scale"]) if baseline_align else float("nan"),
            "global_sim3_yaw_rad": float(action_align["yaw"]) if action_align else float("nan"),
            "baseline_global_sim3_yaw_rad": float(baseline_align["yaw"]) if baseline_align else float("nan"),
            "trajectory_length_m": base.trajectory_length(action_align["gt_pos"]) if action_align else float("nan"),
            "runtime_sec": phase_status.get("run_worker_duration_sec", ""),
            "peak_gpu_memory_mb": "",
            "peak_gpu_memory_mb_note": "not_captured_per_run; run_worker_duration_sec recorded",
            "metric_available": metric_available,
            "all_metric_phase_success": all_metric_phase_success,
            "all_phase_success": all_phase_success,
            "action_fidelity_pass": fidelity["action_fidelity_pass"],
            "no_action_control_source": "v105_frozen_lingbot_stream_default_baseline",
            "action_metric_blocker": action_blocker,
            "baseline_metric_blocker": baseline_blocker,
            "action_traj": rel(action_traj) if action_traj.exists() else "",
            "baseline_traj": rel(baseline_traj) if baseline_traj.exists() else "",
            **phase_status,
            **local_summary,
        }
        full_rows.append(full_row)
        rolling_rows.append(
            {
                "schema": "acl2_v109tf_stage2_rolling_metric_row_v1",
                "surface_id": cfg["surface_id"],
                "policy_id": cfg["policy_id"],
                "policy_family": cfg["policy_family"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": cfg["action_name"],
                **{f"action_{k}": v for k, v in action_rolling.items()},
                **{f"baseline_{k}": v for k, v in baseline_rolling.items()},
                "rolling_ATE_p90_relative_improvement_vs_baseline": rolling_p90_rel,
                "rolling_ATE_mean_relative_improvement_vs_baseline": rolling_mean_rel,
                "rolling_worse_fraction_gt_0p05_delta_action_minus_baseline": (
                    safe_float(action_rolling["rolling_worse_fraction_gt_0p05"])
                    - safe_float(baseline_rolling["rolling_worse_fraction_gt_0p05"])
                    if action_rolling["rolling_worse_fraction_gt_0p05"] != ""
                    and baseline_rolling["rolling_worse_fraction_gt_0p05"] != ""
                    else ""
                ),
                "rolling_worse_fraction_gt_0p10_delta_action_minus_baseline": (
                    safe_float(action_rolling["rolling_worse_fraction_gt_0p10"])
                    - safe_float(baseline_rolling["rolling_worse_fraction_gt_0p10"])
                    if action_rolling["rolling_worse_fraction_gt_0p10"] != ""
                    and baseline_rolling["rolling_worse_fraction_gt_0p10"] != ""
                    else ""
                ),
            }
        )
    return full_rows, rolling_rows, local_rows, fidelity_rows


def rows_by_policy_seq(full_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["policy_id"], row["seq"]): row for row in full_rows}


def row_rel(row: dict[str, Any] | None) -> float:
    if row is None:
        return float("nan")
    return safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))


def mean_action_ate(policy_seq: dict[tuple[str, str], dict[str, Any]], policy_ids: tuple[str, ...], seq: str) -> float:
    vals = [safe_float(policy_seq.get((policy_id, seq), {}).get("full_ATE_sim3", "nan")) for policy_id in policy_ids]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def median_pairwise_vs_group(
    policy_seq: dict[tuple[str, str], dict[str, Any]],
    control_ids: tuple[str, ...],
) -> float:
    vals: list[float] = []
    for seq in SEQUENCES:
        sem = policy_seq.get((SEMANTIC_PLUS, seq))
        sem_ate = safe_float(sem.get("full_ATE_sim3", "nan") if sem else "nan")
        control_ate = mean_action_ate(policy_seq, control_ids, seq)
        vals.append(base.rel_improvement(control_ate, sem_ate))
    return base.median(vals)


def median_pairwise_vs_best_control(
    policy_seq: dict[tuple[str, str], dict[str, Any]],
    control_ids: tuple[str, ...],
) -> float:
    vals: list[float] = []
    for seq in SEQUENCES:
        sem = policy_seq.get((SEMANTIC_PLUS, seq))
        sem_ate = safe_float(sem.get("full_ATE_sim3", "nan") if sem else "nan")
        control_ates = [
            safe_float(policy_seq.get((policy_id, seq), {}).get("full_ATE_sim3", "nan"))
            for policy_id in control_ids
        ]
        control_ates = [value for value in control_ates if math.isfinite(value)]
        best_control_ate = min(control_ates) if control_ates else float("nan")
        vals.append(base.rel_improvement(best_control_ate, sem_ate))
    return base.median(vals)


def semantic_control_rows(full_rows: list[dict[str, Any]], rolling_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy_seq = rows_by_policy_seq(full_rows)
    rels_by_policy: dict[str, list[float]] = defaultdict(list)
    for row in full_rows:
        rels_by_policy[row["policy_id"]].append(row_rel(row))

    policy_median_rel = {
        policy_id: base.median(vals)
        for policy_id, vals in rels_by_policy.items()
    }
    sem_rows = [policy_seq.get((SEMANTIC_PLUS, seq)) for seq in SEQUENCES]
    sem_full_rel = [row_rel(row) for row in sem_rows]
    sem_final_rel = [
        safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan")) if row else float("nan")
        for row in sem_rows
    ]
    sem_local_rel = [
        safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan")) if row else float("nan")
        for row in sem_rows
    ]
    sem_roll = [
        row for row in rolling_rows
        if row["policy_id"] == SEMANTIC_PLUS and row["seq"] in SEQUENCES
    ]
    sem_roll_rel = [safe_float(row["rolling_ATE_p90_relative_improvement_vs_baseline"]) for row in sem_roll]
    roll_delta_05 = [safe_float(row["rolling_worse_fraction_gt_0p05_delta_action_minus_baseline"]) for row in sem_roll]
    roll_delta_10 = [safe_float(row["rolling_worse_fraction_gt_0p10_delta_action_minus_baseline"]) for row in sem_roll]

    sem_med = base.median(sem_full_rel)
    internal_med = policy_median_rel.get(INTERNAL_ONLY, float("nan"))
    semantic_only_med = policy_median_rel.get(SEMANTIC_ONLY, float("nan"))
    shuffle_individual = [policy_median_rel.get(policy_id, float("nan")) for policy_id in SHUFFLE_POLICIES]
    random_individual = [policy_median_rel.get(policy_id, float("nan")) for policy_id in RANDOM_POLICIES]
    shuffle_med = base.median(shuffle_individual)
    random_med = base.median(random_individual)
    role_med = policy_median_rel.get(ROLE_ROTATION, float("nan"))
    low_risk_med = policy_median_rel.get(LOW_RISK_REVERSE, float("nan"))
    sem_vs_internal = median_pairwise_vs_group(policy_seq, (INTERNAL_ONLY,))
    sem_vs_semantic_only = median_pairwise_vs_group(policy_seq, (SEMANTIC_ONLY,))
    sem_vs_shuffle = median_pairwise_vs_group(policy_seq, SHUFFLE_POLICIES)
    sem_vs_shuffle_best = median_pairwise_vs_best_control(policy_seq, SHUFFLE_POLICIES)
    sem_vs_random = median_pairwise_vs_group(policy_seq, RANDOM_POLICIES)
    sem_vs_role = median_pairwise_vs_group(policy_seq, (ROLE_ROTATION,))
    sem_vs_low_risk = median_pairwise_vs_group(policy_seq, (LOW_RISK_REVERSE,))

    strongest_seq_count = 0
    strongest_rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        sem_rel = row_rel(policy_seq.get((SEMANTIC_PLUS, seq)))
        control_rels = {
            policy_id: row_rel(policy_seq.get((policy_id, seq)))
            for policy_id in CONTROL_POLICIES
        }
        finite_controls = {k: v for k, v in control_rels.items() if math.isfinite(v)}
        if finite_controls:
            best_policy = max(finite_controls, key=lambda key: finite_controls[key])
            best_rel = finite_controls[best_policy]
            sem_matches = math.isfinite(sem_rel) and sem_rel >= best_rel - CONTROL_MATCH_TOL
            strongest_seq_count += int(sem_matches)
        else:
            best_policy = ""
            best_rel = float("nan")
            sem_matches = False
        strongest_rows.append(
            {
                "seq": seq,
                "semantic_plus_rel": sem_rel,
                "strongest_control_policy": best_policy,
                "strongest_control_rel": best_rel,
                "semantic_plus_within_tolerance_of_strongest": sem_matches,
            }
        )

    improve_count = sum(1 for value in sem_full_rel if math.isfinite(value) and value > 0.0)
    rolling_worse_max_delta = max([v for v in roll_delta_05 + roll_delta_10 if math.isfinite(v)] or [float("nan")])
    rolling_not_increase = math.isfinite(rolling_worse_max_delta) and rolling_worse_max_delta <= 1e-12
    f1_full_gate_pass = bool(
        math.isfinite(sem_med)
        and sem_med >= 0.05
        and improve_count >= 3
        and base.max_rel_harm(sem_full_rel) <= 0.02
    )
    local_gate_pass = base.max_rel_harm(sem_local_rel) <= 0.02
    semantic_control_gate_pass = bool(
        math.isfinite(sem_vs_internal)
        and sem_vs_internal >= 0.03
        and math.isfinite(sem_vs_shuffle)
        and sem_vs_shuffle >= 0.01
        and math.isfinite(sem_vs_random)
        and sem_vs_random >= 0.03
        and strongest_seq_count >= 3
    )
    row = {
        "schema": "acl2_v109tf_stage2_semantic_control_row_v1",
        "row_type": "semantic_plus_aggregate",
        "surface_id": "F",
        "semantic_plus_policy_id": SEMANTIC_PLUS,
        "policy_id": SEMANTIC_PLUS,
        "policy_family": "semantic_plus_internal",
        "sequence_count": len([r for r in sem_rows if r is not None]),
        "median_full_rel_improvement": sem_med,
        "mean_full_rel_improvement": base.mean(sem_full_rel),
        "num_seq_improved": improve_count,
        "num_seq_worse": sum(1 for value in sem_full_rel if math.isfinite(value) and value < 0.0),
        "max_harm": base.max_rel_harm(sem_full_rel),
        "median_rolling_p90_rel_improvement": base.median(sem_roll_rel),
        "median_final_error_rel_improvement": base.median(sem_final_rel),
        "median_local_window_rel_improvement": base.median(sem_local_rel),
        "semantic_plus_median_full_ATE_improvement_vs_no_action": sem_med,
        "semantic_plus_improved_sequence_count": improve_count,
        "semantic_plus_max_full_ATE_harm": base.max_rel_harm(sem_full_rel),
        "semantic_plus_median_final_error_improvement_vs_no_action": base.median(sem_final_rel),
        "semantic_plus_median_rolling_p90_improvement_vs_no_action": base.median(sem_roll_rel),
        "semantic_plus_max_rolling_worse_fraction_delta": rolling_worse_max_delta,
        "semantic_plus_local_window_ATE_max_harm": base.max_rel_harm(sem_local_rel),
        "internal_only_median_full_ATE_improvement_vs_no_action": internal_med,
        "semantic_only_median_full_ATE_improvement_vs_no_action": semantic_only_med,
        "semantic_shuffle_seed0_median_full_ATE_improvement_vs_no_action": policy_median_rel.get("F4_semantic_shuffle_seed0", float("nan")),
        "semantic_shuffle_seed1_median_full_ATE_improvement_vs_no_action": policy_median_rel.get("F5_semantic_shuffle_seed1", float("nan")),
        "semantic_shuffle_seed2_median_full_ATE_improvement_vs_no_action": policy_median_rel.get("F6_semantic_shuffle_seed2", float("nan")),
        "semantic_shuffle_median_of_seed_medians_full_ATE_improvement_vs_no_action": shuffle_med,
        "role_rotation_median_full_ATE_improvement_vs_no_action": role_med,
        "same_count_random_seed0_median_full_ATE_improvement_vs_no_action": policy_median_rel.get("F8_same_count_random_seed0", float("nan")),
        "same_count_random_seed1_median_full_ATE_improvement_vs_no_action": policy_median_rel.get("F9_same_count_random_seed1", float("nan")),
        "same_count_random_seed2_median_full_ATE_improvement_vs_no_action": policy_median_rel.get("F10_same_count_random_seed2", float("nan")),
        "same_count_random_median_of_seed_medians_full_ATE_improvement_vs_no_action": random_med,
        "low_risk_reverse_median_full_ATE_improvement_vs_no_action": low_risk_med,
        "full_ATE_improvement_vs_internal_only": sem_vs_internal,
        "full_ATE_improvement_vs_semantic_only": sem_vs_semantic_only,
        "full_ATE_improvement_vs_semantic_shuffle_mean": sem_vs_shuffle,
        "full_ATE_improvement_vs_semantic_shuffle_best": sem_vs_shuffle_best,
        "full_ATE_improvement_vs_same_count_random_mean": sem_vs_random,
        "full_ATE_improvement_vs_role_rotation": sem_vs_role,
        "full_ATE_improvement_vs_low_risk_reverse": sem_vs_low_risk,
        "semantic_plus_minus_internal": sem_vs_internal,
        "semantic_plus_minus_shuffle_mean": sem_vs_shuffle,
        "semantic_plus_minus_shuffle_best": sem_vs_shuffle_best,
        "semantic_plus_minus_random_mean": sem_vs_random,
        "semantic_plus_minus_low_risk_reverse": sem_vs_low_risk,
        "semantic_plus_within_0p005_of_strongest_control_sequence_count": strongest_seq_count,
        "f1_full_gate_pass": f1_full_gate_pass,
        "rolling_worse_fraction_not_increased": rolling_not_increase,
        "local_window_gate_pass": local_gate_pass,
        "semantic_control_gate_pass": semantic_control_gate_pass,
        "semantic_causality_pass": f1_full_gate_pass and semantic_control_gate_pass and rolling_not_increase and local_gate_pass,
        "semantic_safety_filter_pass": False,
        "semantic_safety_filter_pass_note": "not_claimed_in_stage2; semantic controls do not prove semantic filter over schedule/content controls",
        "strongest_control_by_seq_json": json.dumps(base.clean_json(strongest_rows), sort_keys=True),
    }
    policy_rows: list[dict[str, Any]] = []
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rolling in rolling_rows:
        rolling_by_policy[rolling["policy_id"]].append(rolling)
    for policy_id in sorted(rels_by_policy):
        rows = [item for item in full_rows if item["policy_id"] == policy_id]
        rels = [row_rel(item) for item in rows]
        finals = [safe_float(item.get("final_error_relative_improvement_vs_baseline", "nan")) for item in rows]
        locals_ = [safe_float(item.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan")) for item in rows]
        roll_rels = [
            safe_float(item.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for item in rolling_by_policy.get(policy_id, [])
        ]
        family = rows[0]["policy_family"] if rows else ""
        policy_rows.append(
            {
                "schema": "acl2_v109tf_stage2_semantic_control_row_v1",
                "row_type": "policy_summary",
                "surface_id": "F",
                "policy_id": policy_id,
                "policy_family": family,
                "sequence_count": len(rows),
                "median_full_rel_improvement": base.median(rels),
                "mean_full_rel_improvement": base.mean(rels),
                "num_seq_improved": sum(1 for value in rels if math.isfinite(value) and value > 0.0),
                "num_seq_worse": sum(1 for value in rels if math.isfinite(value) and value < 0.0),
                "max_harm": base.max_rel_harm(rels),
                "median_rolling_p90_rel_improvement": base.median(roll_rels),
                "median_final_error_rel_improvement": base.median(finals),
                "median_local_window_rel_improvement": base.median(locals_),
                "semantic_plus_minus_internal": sem_vs_internal if policy_id == SEMANTIC_PLUS else "",
                "semantic_plus_minus_shuffle_mean": sem_vs_shuffle if policy_id == SEMANTIC_PLUS else "",
                "semantic_plus_minus_shuffle_best": sem_vs_shuffle_best if policy_id == SEMANTIC_PLUS else "",
                "semantic_plus_minus_random_mean": sem_vs_random if policy_id == SEMANTIC_PLUS else "",
                "semantic_plus_minus_low_risk_reverse": sem_vs_low_risk if policy_id == SEMANTIC_PLUS else "",
                "semantic_causality_pass": row["semantic_causality_pass"] if policy_id == SEMANTIC_PLUS else "",
                "semantic_safety_filter_pass": row["semantic_safety_filter_pass"] if policy_id == SEMANTIC_PLUS else "",
            }
        )
    return [row] + policy_rows, row


def taxonomy(metric_complete: bool, all_action_fidelity: bool, control: dict[str, Any]) -> tuple[str, bool, str]:
    if not metric_complete:
        return "STAGE2_F_CORE_METRICS_NOT_COMPLETE", False, "stage2_f_core_metrics_not_complete"
    if not all_action_fidelity:
        return "STAGE2_ACTION_FIDELITY_FAIL", False, "stage2_action_fidelity_fail"
    f1_full = bool(control.get("f1_full_gate_pass", False))
    semantic_control = bool(control.get("semantic_control_gate_pass", False))
    rolling_ok = bool(control.get("rolling_worse_fraction_not_increased", False))
    local_ok = bool(control.get("local_window_gate_pass", False))
    stage2_pass = f1_full and semantic_control and rolling_ok and local_ok
    if stage2_pass:
        return "F_SEMANTIC_CAUSAL_CORE_PASS", True, ""
    if f1_full and not semantic_control:
        return "F_FULL_GEOMETRY_PASS_SEMANTIC_CAUSALITY_FAIL", False, "semantic_controls_do_not_support_causality"
    strongest_count = int(control.get("semantic_plus_within_0p005_of_strongest_control_sequence_count", 0))
    sem_med = safe_float(control.get("semantic_plus_median_full_ATE_improvement_vs_no_action", "nan"))
    internal_med = safe_float(control.get("internal_only_median_full_ATE_improvement_vs_no_action", "nan"))
    if math.isfinite(internal_med) and math.isfinite(sem_med) and internal_med >= sem_med:
        return "F_INTERNAL_ONLY_DOMINATES", False, "internal_only_matches_or_beats_semantic_plus"
    if strongest_count < 3:
        return "F_SELECTION_COUNT_OR_SCHEDULE_EFFECT", False, "strongest_control_matches_semantic_plus_in_fewer_than_3_sequences"
    return "F_SEMANTIC_CORE_NO_GO", False, "stage2_gate_failed"


def build_report(summary: dict[str, Any], full_rows: list[dict[str, Any]], control: dict[str, Any]) -> str:
    policy_seq = rows_by_policy_seq(full_rows)
    lines = [
        "# ACL2 v109TF Stage2 F Semantic Causality Report",
        "",
        f"metric_complete: {summary['metric_complete']}",
        f"stage2_pass: {summary['stage2_pass']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        f"observed_run_workers: {summary['observed_run_worker_count']}/{summary['expected_run_worker_count']}",
        f"observed_evaluate_rows: {summary['observed_evaluate_count']}/{summary['expected_run_worker_count']}",
        f"observed_report_rows: {summary['observed_report_count']}/{summary['expected_run_worker_count']}",
        "",
        "## F1 semantic+ per-sequence full ATE",
        "",
    ]
    for seq in SEQUENCES:
        row = policy_seq.get((SEMANTIC_PLUS, seq))
        if row is None:
            lines.append(f"- {seq}: missing")
            continue
        lines.append(
            "- {seq}: baseline={base_ate} action={action_ate} rel={rel_improve} "
            "final_rel={final_rel} local_rel_median={local_rel} rolling_p90_rel={rolling_rel}".format(
                seq=seq,
                base_ate=row.get("baseline_full_ATE_sim3", ""),
                action_ate=row.get("full_ATE_sim3", ""),
                rel_improve=row.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                final_rel=row.get("final_error_relative_improvement_vs_baseline", ""),
                local_rel=row.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                rolling_rel=next(
                    (
                        rr.get("rolling_ATE_p90_relative_improvement_vs_baseline", "")
                        for rr in read_csv(OUT / "rolling_metric_rows.csv")
                        if rr.get("policy_id") == SEMANTIC_PLUS and rr.get("seq") == seq
                    ),
                    "",
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Semantic control summary",
            "",
            f"semantic_plus_median_full_ATE_improvement_vs_no_action: {control.get('semantic_plus_median_full_ATE_improvement_vs_no_action')}",
            f"semantic_plus_improved_sequence_count: {control.get('semantic_plus_improved_sequence_count')}",
            f"semantic_plus_max_full_ATE_harm: {control.get('semantic_plus_max_full_ATE_harm')}",
            f"full_ATE_improvement_vs_internal_only: {control.get('full_ATE_improvement_vs_internal_only')}",
            f"full_ATE_improvement_vs_semantic_shuffle_mean: {control.get('full_ATE_improvement_vs_semantic_shuffle_mean')}",
            f"full_ATE_improvement_vs_same_count_random_mean: {control.get('full_ATE_improvement_vs_same_count_random_mean')}",
            f"semantic_plus_within_0p005_of_strongest_control_sequence_count: {control.get('semantic_plus_within_0p005_of_strongest_control_sequence_count')}",
            "",
            "## Interpretation",
            "",
        ]
    )
    if not summary["metric_complete"]:
        lines.append("Stage2 metric evidence is incomplete. Do not interpret F semantic causality until run_worker and evaluate finish or fail with logged blocker evidence.")
    elif summary["stage2_pass"]:
        lines.append("F semantic+ passed full KITTI, rolling/local safety, and semantic control gates under the v109 criteria.")
    else:
        lines.append("F semantic+ did not pass the full v109 causal gate. Use semantic_control_rows.csv and per-sequence full_metric_rows.csv for the exact failing clauses.")
    return "\n".join(lines)


def build() -> dict[str, Any]:
    config_rows = read_csv(CONFIG_ROWS)
    run_result_rows = read_csv(RUN_RESULTS)
    latest = latest_run_results(run_result_rows)
    full_rows, rolling_rows, local_rows, fidelity_rows = metric_rows(config_rows, latest)
    selected_rows = build_policy_selected_rows(fidelity_rows)
    semantic_rows, control = semantic_control_rows(full_rows, rolling_rows)

    expected_run_workers = len(config_rows)
    run_worker_rows = latest_phase_rows(latest, "run_worker")
    evaluate_rows = latest_phase_rows(latest, "evaluate")
    report_rows = latest_phase_rows(latest, "report")
    observed_run_workers = len(run_worker_rows)
    all_run_worker_success = (
        observed_run_workers >= expected_run_workers
        and all(safe_rc(row) == 0 for row in run_worker_rows)
    )
    all_evaluate_success = (
        len(evaluate_rows) >= expected_run_workers
        and all(safe_rc(row) == 0 for row in evaluate_rows)
    )
    all_action_fidelity = (
        len(fidelity_rows) == expected_run_workers
        and all(bool(row["action_fidelity_pass"]) for row in fidelity_rows)
    )
    metric_complete = (
        len(full_rows) == expected_run_workers
        and all(bool(row["metric_available"]) for row in full_rows)
        and all(bool(row["all_metric_phase_success"]) for row in full_rows)
        and all_run_worker_success
        and all_evaluate_success
    )
    tax, stage2_pass, blocker = taxonomy(metric_complete, all_action_fidelity, control)

    write_csv(OUT / "no_action_control_rows.csv", no_action_rows())
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "semantic_control_rows.csv", semantic_rows)
    write_csv(OUT / "policy_selected_frame_rows.csv", selected_rows)

    summary = {
        "schema": "acl2_v109tf_stage2_f_core_ablation_summary_v1",
        "stage2_pass": stage2_pass,
        "metric_complete": metric_complete,
        "taxonomy": tax,
        "blocker": blocker,
        "expected_run_worker_count": expected_run_workers,
        "observed_run_worker_count": observed_run_workers,
        "observed_evaluate_count": len(evaluate_rows),
        "observed_report_count": len(report_rows),
        "all_run_worker_success": all_run_worker_success,
        "all_evaluate_success": all_evaluate_success,
        "all_action_fidelity": all_action_fidelity,
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_selected_frame_row_count": len(selected_rows),
        "semantic_control_row": control,
        "gate_definition": {
            "median_full_ATE_relative_improvement_min": 0.05,
            "improved_sequence_count_min": 3,
            "max_sequence_full_ATE_harm": 0.02,
            "rolling_worse_fraction_delta_max": 0.0,
            "local_window_ATE_median_max_harm": 0.02,
            "semantic_plus_internal_vs_internal_only_min": 0.03,
            "semantic_plus_vs_shuffle_mean_min": 0.01,
            "semantic_plus_vs_random_mean_min": 0.03,
            "semantic_plus_within_strongest_control_tolerance": CONTROL_MATCH_TOL,
            "semantic_plus_within_strongest_control_sequence_count_min": 3,
        },
        "outputs": {
            "no_action_control_rows": rel(OUT / "no_action_control_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "full_metric_rows": rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "semantic_control_rows": rel(OUT / "semantic_control_rows.csv"),
            "policy_selected_frame_rows": rel(OUT / "policy_selected_frame_rows.csv"),
            "stage2_summary": rel(OUT / "stage2_summary.json"),
            "f_semantic_causality_report": rel(OUT / "f_semantic_causality_report.md"),
        },
    }
    write_json(OUT / "stage2_summary.json", summary)
    write_text(OUT / "f_semantic_causality_report.md", build_report(summary, full_rows, control))
    return summary


def main() -> None:
    print(json.dumps(base.clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
