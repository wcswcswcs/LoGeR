#!/usr/bin/env python3
"""Build v108TF Stage5 four-sequence validation metrics and report."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import build_v108tf_stage4_full_kitti_metrics as base


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
OUT = RESULT_ROOT / "stage5_full_kitti_00_01_02_05_validation"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_METRICS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
V105_WORKSPACE = V105 / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05"
V105_METHOD = "lingbot_map_stream_default"

CONTROL_MATCH_TOL = 0.005
SEQUENCES = ("00", "01", "02", "05")


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


def clean_json(value: Any) -> Any:
    return base.clean_json(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return base.load_json(path)


def latest_run_results(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def latest_phase_rows(latest: dict[tuple[str, str], dict[str, str]], phase: str) -> list[dict[str, str]]:
    return [row for (_run_name, row_phase), row in latest.items() if row_phase == phase]


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def action_fidelity_row(cfg: dict[str, str], run_rows: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    action_rows = base.load_jsonl(action_file)
    expected = base.parse_indices(cfg.get("selected_global_frame_indices", ""))
    expected_field = cfg.get("expected_action_field", "")
    mode = cfg.get("stage4_action_mode", cfg.get("stage5_action_mode", ""))
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
        if mode == "force_non_keyframe":
            if (
                base.boolish(row.get("forced_non_keyframe", False))
                and base.boolish(row.get("base_is_keyframe", False))
                and not base.boolish(row.get("final_is_keyframe", True))
            ):
                effective.add(sample)
        elif mode == "anchor_special_only":
            if (
                base.boolish(row.get("forced_anchor_only", False))
                and base.boolish(row.get("forced_context_only", False))
                and base.boolish(row.get("context_only_append", False))
                and str(row.get("context_only_special_mode", "")) == "scale_only"
            ):
                effective.add(sample)
        elif mode == "v106_context_only_with_local_preserve":
            heads = {int(x) for x in str(row.get("headlocal_action_heads", "")).split(",") if x.strip().isdigit()}
            expected_heads = set(range(16))
            if (
                base.boolish(row.get("headlocal_action_enabled", False))
                and str(row.get("headlocal_action_mode", "")) == mode
                and heads == expected_heads
            ):
                effective.add(sample)
        elif base.boolish(row.get(expected_field, False)):
            effective.add(sample)

    missing = expected - observed
    unexpected = observed - expected
    ineffective = expected - effective
    action_fidelity_pass = (observed == expected) and (effective == expected) and trace_error_rows == 0
    run_name = f"kitti_lingbot_v108tf_stage5_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = run_rows.get((run_name, "run_worker"), {})
    return {
        "schema": "acl2_v108tf_stage5_action_fidelity_row_v1",
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage5_action_mode": cfg.get("stage5_action_mode", mode),
        "stage4_action_mode": mode,
        "expected_action_field": expected_field,
        "expected_action_frame_count": len(expected),
        "observed_action_frame_count": len(observed),
        "action_effective_frame_count": len(effective),
        "action_noop_frame_count": len(ineffective),
        "expected_keyframe_count": len(expected),
        "observed_keyframe_count": len(observed & base_keyframes),
        "special_token_operation_count": len(effective) if mode == "anchor_special_only" else "",
        "headlocal_operation_count": len(effective) if mode == "v106_context_only_with_local_preserve" else "",
        "trace_error_rows": trace_error_rows,
        "no_action_control_source": "v105_frozen_lingbot_stream_default_baseline",
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
    metric_by_seq = {row["seq"]: row for row in read_csv(V105_METRICS)}
    for seq in SEQUENCES:
        metric = metric_by_seq.get(seq, {})
        rows.append(
            {
                "schema": "acl2_v108tf_stage5_no_action_control_row_v1",
                "policy_family": "NO_ACTION",
                "policy_id": "v105_lingbot_stream_default_no_action",
                "seq": seq,
                "dataset": "kitti_v105_00_01_02_05",
                "method": V105_METHOD,
                "full_ATE_sim3": metric.get("ATE_full_sim3_m", ""),
                "final_error_m": metric.get("final_error_m", ""),
                "rolling_ATE_p90": metric.get("rolling_ATE_p90", ""),
                "local_window_ATE_median": metric.get("local_window_ATE_median", ""),
                "source": rel(V105_METRICS),
                "source_note": "frozen Stage0/v105 full KITTI baseline, not rerun in Stage5",
            }
        )
    return rows


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool]:
    status: dict[str, Any] = {}
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        run_name = f"kitti_lingbot_v108tf_stage5_prepare_{seq}" if phase == "prepare" else f"kitti_lingbot_v108tf_stage5_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = int(float(row.get("returncode", 1))) if row else 1
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_phase_success = all_phase_success and rc == 0
    return status, all_phase_success


def build() -> dict[str, Any]:
    config_rows = read_csv(CONFIG_ROWS)
    run_result_rows = read_csv(RUN_RESULTS)
    latest = latest_run_results(run_result_rows)
    v105_csv = {row["seq"]: row for row in read_csv(V105_METRICS)}

    baseline_cache: dict[str, dict[str, Any]] = {}
    baseline_rolling_cache: dict[str, dict[str, Any]] = {}
    action_fidelity_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []

    for cfg in config_rows:
        fidelity = action_fidelity_row(cfg, latest)
        action_fidelity_rows.append(fidelity)
        seq = cfg["seq"]
        dataset = cfg["dataset"]
        method = cfg["method"]
        method_root = WORKSPACE / dataset / seq / method
        action_gt = WORKSPACE / dataset / seq / "gt/traj.txt"
        action_traj = method_root / "traj.txt"
        action_eval = load_json(method_root / "eval/traj.json")
        baseline_gt = V105_WORKSPACE / seq / "gt/traj.txt"
        baseline_traj = V105_WORKSPACE / seq / V105_METHOD / "traj.txt"
        baseline_eval = load_json(V105_WORKSPACE / seq / V105_METHOD / "eval/traj.json")
        phase_status, all_phase_success = phase_status_for(cfg, latest)

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
                row["schema"] = "acl2_v108tf_stage5_local_handoff_metric_row_v1"
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
            "schema": "acl2_v108tf_stage5_full_sequence_metric_row_v1",
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
            "peak_gpu_memory_mb_note": "not_captured_per_run; GPU allocation monitored during batch via nvidia-smi",
            "metric_available": metric_available,
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
                "schema": "acl2_v108tf_stage5_rolling_metric_row_v1",
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
                    if action_rolling["rolling_worse_fraction_gt_0p05"] != "" and baseline_rolling["rolling_worse_fraction_gt_0p05"] != ""
                    else ""
                ),
                "rolling_worse_fraction_gt_0p10_delta_action_minus_baseline": (
                    safe_float(action_rolling["rolling_worse_fraction_gt_0p10"])
                    - safe_float(baseline_rolling["rolling_worse_fraction_gt_0p10"])
                    if action_rolling["rolling_worse_fraction_gt_0p10"] != "" and baseline_rolling["rolling_worse_fraction_gt_0p10"] != ""
                    else ""
                ),
            }
        )

    rows_by_surface_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        rows_by_surface_policy[(row["surface_id"], row["policy_family"])].append(row)

    semantic_rows: list[dict[str, Any]] = []
    passing_surfaces: list[str] = []
    surfaces = sorted({row["surface_id"] for row in full_rows})
    for surface in surfaces:
        sem = rows_by_surface_policy.get((surface, "semantic_plus_internal"), [])
        internal = rows_by_surface_policy.get((surface, "internal_only"), [])
        shuffle = rows_by_surface_policy.get((surface, "semantic_shuffle"), [])
        random_rows = rows_by_surface_policy.get((surface, "same_count_random"), [])
        internal_by_seq = {r["seq"]: r for r in internal}
        shuffle_by_seq = {r["seq"]: r for r in shuffle}
        random_by_seq = {r["seq"]: r for r in random_rows}

        def median_pairwise_vs(other_by_seq: dict[str, dict[str, Any]], field: str) -> float:
            vals: list[float] = []
            for row in sem:
                other = other_by_seq.get(row["seq"])
                if other is not None:
                    vals.append(base.rel_improvement(float(other[field]), float(row[field])))
            return base.median(vals)

        sem_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in sem]
        sem_final_rel = [float(r["final_error_relative_improvement_vs_baseline"]) for r in sem]
        sem_local_rel = [float(r["local_window_ATE_rel_improvement_vs_baseline_median"]) for r in sem]
        sem_roll_rows = [
            r for r in rolling_rows if r["surface_id"] == surface and r["policy_family"] == "semantic_plus_internal"
        ]
        sem_roll_rel = [float(r["rolling_ATE_p90_relative_improvement_vs_baseline"]) for r in sem_roll_rows]
        roll_delta_05 = [safe_float(r["rolling_worse_fraction_gt_0p05_delta_action_minus_baseline"]) for r in sem_roll_rows]
        roll_delta_10 = [safe_float(r["rolling_worse_fraction_gt_0p10_delta_action_minus_baseline"]) for r in sem_roll_rows]
        internal_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in internal]
        shuffle_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in shuffle]
        random_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in random_rows]
        sem_med = base.median(sem_full_rel)
        internal_med = base.median(internal_full_rel)
        shuffle_med = base.median(shuffle_full_rel)
        random_med = base.median(random_full_rel)
        sem_vs_internal = median_pairwise_vs(internal_by_seq, "full_ATE_sim3")
        sem_vs_shuffle = median_pairwise_vs(shuffle_by_seq, "full_ATE_sim3")
        sem_vs_random = median_pairwise_vs(random_by_seq, "full_ATE_sim3")
        shuffle_matches = math.isfinite(shuffle_med) and math.isfinite(sem_med) and shuffle_med >= sem_med - CONTROL_MATCH_TOL
        random_matches = math.isfinite(random_med) and math.isfinite(sem_med) and random_med >= sem_med - CONTROL_MATCH_TOL
        controls_do_not_match = not shuffle_matches and not random_matches
        sem_beats_internal = math.isfinite(sem_vs_internal) and sem_vs_internal >= 0.03
        improve_count = sum(1 for v in sem_full_rel if math.isfinite(v) and v > 0.0)
        rolling_not_increase = all(v <= 1e-12 for v in roll_delta_05 + roll_delta_10 if math.isfinite(v))
        candidate_complete = (
            len(sem) == 4
            and all(bool(r["metric_available"]) for r in sem)
            and all(bool(r["all_phase_success"]) for r in sem)
            and all(bool(r["action_fidelity_pass"]) for r in sem)
            and len(internal) == 4
            and len(shuffle) == 4
            and len(random_rows) == 4
        )
        stage5_candidate_pass = bool(
            candidate_complete
            and math.isfinite(sem_med)
            and sem_med >= 0.05
            and improve_count >= 3
            and base.max_rel_harm(sem_full_rel) <= 0.02
            and base.median(sem_final_rel) >= 0.03
            and rolling_not_increase
            and base.max_rel_harm(sem_local_rel) <= 0.02
            and sem_beats_internal
            and controls_do_not_match
        )
        if stage5_candidate_pass:
            passing_surfaces.append(surface)
        semantic_rows.append(
            {
                "schema": "acl2_v108tf_stage5_semantic_control_row_v1",
                "surface_id": surface,
                "semantic_plus_policy_id": f"{surface}1_semantic_plus_internal",
                "sequence_count": len(sem),
                "candidate_complete": candidate_complete,
                "semantic_plus_median_full_ATE_improvement_vs_no_action": sem_med,
                "semantic_plus_improved_sequence_count": improve_count,
                "semantic_plus_max_full_ATE_harm": base.max_rel_harm(sem_full_rel),
                "semantic_plus_median_final_error_improvement_vs_no_action": base.median(sem_final_rel),
                "semantic_plus_median_rolling_p90_improvement_vs_no_action": base.median(sem_roll_rel),
                "semantic_plus_max_rolling_worse_fraction_delta": max(
                    [v for v in roll_delta_05 + roll_delta_10 if math.isfinite(v)] or [float("nan")]
                ),
                "semantic_plus_local_window_ATE_max_harm": base.max_rel_harm(sem_local_rel),
                "internal_only_median_full_ATE_improvement_vs_no_action": internal_med,
                "semantic_shuffle_median_full_ATE_improvement_vs_no_action": shuffle_med,
                "same_count_random_median_full_ATE_improvement_vs_no_action": random_med,
                "full_ATE_improvement_vs_internal_only": sem_vs_internal,
                "full_ATE_improvement_vs_semantic_shuffle": sem_vs_shuffle,
                "full_ATE_improvement_vs_same_count_random": sem_vs_random,
                "semantic_shuffle_gap": sem_med - shuffle_med if math.isfinite(sem_med) and math.isfinite(shuffle_med) else float("nan"),
                "same_count_random_margin": sem_med - random_med if math.isfinite(sem_med) and math.isfinite(random_med) else float("nan"),
                "semantic_shuffle_matches_same_improvement": shuffle_matches,
                "same_count_random_matches_same_improvement": random_matches,
                "semantic_plus_beats_internal_on_full_ATE": sem_beats_internal,
                "controls_do_not_match_same_improvement": controls_do_not_match,
                "runtime_memory_overhead_recorded": True,
                "runtime_memory_note": "run_worker_duration_sec recorded for every row; peak GPU memory not captured per run",
                "stage5_candidate_pass": stage5_candidate_pass,
            }
        )

    expected_run_workers = len(config_rows)
    historical_run_worker_rows = [row for row in run_result_rows if row.get("phase") == "run_worker"]
    run_worker_rows = latest_phase_rows(latest, "run_worker")
    evaluate_rows = latest_phase_rows(latest, "evaluate")
    report_rows = latest_phase_rows(latest, "report")
    observed_run_workers = len(run_worker_rows)
    all_run_worker_success = (
        observed_run_workers >= expected_run_workers
        and all(int(float(row.get("returncode", 1))) == 0 for row in run_worker_rows)
    )
    observed_run_worker_historical_rows = len(historical_run_worker_rows)
    observed_run_worker_historical_failure_rows = sum(
        1 for row in historical_run_worker_rows if int(float(row.get("returncode", 1))) != 0
    )
    metric_complete = (
        len(full_rows) == len(config_rows)
        and all(bool(row["metric_available"]) for row in full_rows)
        and observed_run_workers >= expected_run_workers
        and len(evaluate_rows) >= expected_run_workers
    )
    stage5_pass = bool(passing_surfaces)
    if not metric_complete:
        blocker = "stage5_four_sequence_metrics_not_complete"
    elif not stage5_pass:
        blocker = "NO_STAGE4_PASSING_SURFACE_PASSES_STAGE5_FOUR_SEQUENCE_GATE"
    else:
        blocker = ""

    no_action = no_action_rows()
    write_csv(OUT / "no_action_control_rows.csv", no_action)
    write_csv(OUT / "action_fidelity_rows.csv", action_fidelity_rows)
    write_csv(OUT / "full_sequence_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "semantic_control_rows.csv", semantic_rows)

    report_lines = [
        "# ACL2 v108TF Stage5 Semantic Contribution Report",
        "",
        f"metric_complete: {metric_complete}",
        f"stage5_pass: {stage5_pass}",
        f"blocker: {blocker}",
        f"observed_run_workers: {observed_run_workers}/{expected_run_workers}",
        f"observed_run_worker_historical_rows: {observed_run_worker_historical_rows}",
        f"observed_run_worker_historical_failure_rows: {observed_run_worker_historical_failure_rows}",
        f"observed_evaluate_rows: {len(evaluate_rows)}/{expected_run_workers}",
        f"observed_report_rows: {len(report_rows)}/{expected_run_workers}",
        "",
        "## No-action control",
        "",
        f"NO_ACTION uses frozen v105 LingBot stream default baseline: `{rel(V105_METRICS)}`.",
        "It is not rerun in Stage5; Stage0 already froze complete KITTI 00/01/02/05 baseline trajectories and metrics.",
        "",
        "## Candidate summary",
        "",
    ]
    for row in semantic_rows:
        report_lines.append(
            "- {surface}: pass={passed} sem_median_full_rel={sem} improved_seq={improved}/4 "
            "max_harm={harm} final_median_rel={final} sem_vs_internal={sem_vs_internal} "
            "shuffle_matches={shuffle_match} random_matches={random_match}".format(
                surface=row["surface_id"],
                passed=row["stage5_candidate_pass"],
                sem=row["semantic_plus_median_full_ATE_improvement_vs_no_action"],
                improved=row["semantic_plus_improved_sequence_count"],
                harm=row["semantic_plus_max_full_ATE_harm"],
                final=row["semantic_plus_median_final_error_improvement_vs_no_action"],
                sem_vs_internal=row["full_ATE_improvement_vs_internal_only"],
                shuffle_match=row["semantic_shuffle_matches_same_improvement"],
                random_match=row["same_count_random_matches_same_improvement"],
            )
        )
    if not metric_complete:
        report_lines.extend(
            [
                "",
                "## Current blocker",
                "",
                "Stage5 evidence is incomplete. Do not interpret four-sequence geometry until run_worker and evaluate finish or fail with logged blocker evidence.",
            ]
        )
    elif not stage5_pass:
        report_lines.extend(
            [
                "",
                "## No-Go taxonomy",
                "",
                "No Stage4-passing surface satisfied the stricter four-sequence gate. Use semantic_control_rows.csv and per-sequence full_sequence_metric_rows.csv to distinguish sequence-specific failure, control-match failure, local-geometry harm, or final-error tradeoff.",
            ]
        )
    write_text(OUT / "semantic_contribution_report.md", "\n".join(report_lines))

    summary = {
        "schema": "acl2_v108tf_stage5_four_sequence_metric_summary_v1",
        "stage5_pass": stage5_pass,
        "metric_complete": metric_complete,
        "blocker": blocker,
        "passing_surfaces": passing_surfaces,
        "expected_run_worker_count": expected_run_workers,
        "observed_run_worker_count": observed_run_workers,
        "observed_run_worker_historical_count": observed_run_worker_historical_rows,
        "observed_run_worker_historical_failure_count": observed_run_worker_historical_failure_rows,
        "observed_evaluate_count": len(evaluate_rows),
        "observed_report_count": len(report_rows),
        "all_run_worker_success": all_run_worker_success,
        "full_metric_row_count": len(full_rows),
        "action_fidelity_row_count": len(action_fidelity_rows),
        "semantic_control_rows": semantic_rows,
        "no_action_control_source": rel(V105_METRICS),
        "low_risk_reverse_status": "not_applicable_no_stage4_low_risk_reverse_policy_defined_for_E_F",
        "gate_definition": {
            "median_full_ATE_relative_improvement_min": 0.05,
            "improved_sequence_count_min": 3,
            "max_sequence_full_ATE_harm": 0.02,
            "median_final_error_relative_improvement_min": 0.03,
            "rolling_worse_fraction_delta_max": 0.0,
            "local_window_ATE_median_max_harm": 0.02,
            "semantic_plus_internal_vs_internal_only_min": 0.03,
            "control_match_tolerance": CONTROL_MATCH_TOL,
        },
        "outputs": {
            "no_action_control_rows": rel(OUT / "no_action_control_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "full_sequence_metric_rows": rel(OUT / "full_sequence_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "semantic_control_rows": rel(OUT / "semantic_control_rows.csv"),
            "semantic_contribution_report": rel(OUT / "semantic_contribution_report.md"),
        },
    }
    write_json(OUT / "stage5_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
