#!/usr/bin/env python3
"""Summarize v107R Stage6 semantic wrapper-policy runtime action results."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE6_ROOT = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control/stage6_runtime_pilot_or_blocked"
OUT = STAGE6_ROOT / "semantic_wrapper_policy_pilot"
CONFIG_ROWS = OUT / "action_config_rows.csv"
TARGET_ROWS = OUT / "target_manifest.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"
PRIMARY_SEMANTIC_ACTIONS = {
    "semantic_highrisk_force_non_keyframe",
    "semantic_highrisk_context_only",
    "semantic_highrisk_anchor_only",
    "semantic_highrisk_early_top6_force_non_keyframe",
    "semantic_highrisk_early_top8_force_non_keyframe",
    "semantic_highrisk_not_tail_top8_force_non_keyframe",
    "semantic_highrisk_early_risk_ge_0p60_force_non_keyframe",
    "semantic_highrisk_early_risk_ge_0p65_force_non_keyframe",
}
REQUIRED_CONTROLS = {
    "same_count_random_force_non_keyframe",
    "semantic_lowrisk_reverse_force_non_keyframe",
    "same_count_random_early_top6_force_non_keyframe",
    "semantic_lowrisk_early_top6_reverse_force_non_keyframe",
    "same_count_random_early_top8_force_non_keyframe",
    "semantic_lowrisk_early_top8_reverse_force_non_keyframe",
    "same_count_random_not_tail_top8_force_non_keyframe",
    "semantic_lowrisk_not_tail_top8_reverse_force_non_keyframe",
    "same_count_random_early_risk_ge_0p60_force_non_keyframe",
    "semantic_lowrisk_early_risk_ge_0p60_reverse_force_non_keyframe",
    "same_count_random_early_risk_ge_0p65_force_non_keyframe",
    "semantic_lowrisk_early_risk_ge_0p65_reverse_force_non_keyframe",
}


def relevant_controls_for_action(action: str) -> set[str]:
    if action == "semantic_highrisk_force_non_keyframe":
        return {"same_count_random_force_non_keyframe", "semantic_lowrisk_reverse_force_non_keyframe"}
    if action in {"semantic_highrisk_context_only", "semantic_highrisk_anchor_only"}:
        return {"same_count_random_force_non_keyframe", "semantic_lowrisk_reverse_force_non_keyframe"}
    if action == "semantic_highrisk_early_top6_force_non_keyframe":
        return {"same_count_random_early_top6_force_non_keyframe", "semantic_lowrisk_early_top6_reverse_force_non_keyframe"}
    if action == "semantic_highrisk_early_top8_force_non_keyframe":
        return {"same_count_random_early_top8_force_non_keyframe", "semantic_lowrisk_early_top8_reverse_force_non_keyframe"}
    if action == "semantic_highrisk_not_tail_top8_force_non_keyframe":
        return {"same_count_random_not_tail_top8_force_non_keyframe", "semantic_lowrisk_not_tail_top8_reverse_force_non_keyframe"}
    if action == "semantic_highrisk_early_risk_ge_0p60_force_non_keyframe":
        return {
            "same_count_random_early_risk_ge_0p60_force_non_keyframe",
            "semantic_lowrisk_early_risk_ge_0p60_reverse_force_non_keyframe",
        }
    if action == "semantic_highrisk_early_risk_ge_0p65_force_non_keyframe":
        return {
            "same_count_random_early_risk_ge_0p65_force_non_keyframe",
            "semantic_lowrisk_early_risk_ge_0p65_reverse_force_non_keyframe",
        }
    return set(REQUIRED_CONTROLS)


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


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_indices(raw: str | None) -> set[int]:
    if raw is None or raw.strip() == "":
        return set()
    return {int(float(x)) for x in raw.replace(",", ";").split(";") if x.strip()}


def latest_run_results(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def load_traj_positions(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            frame = int(vals[0])
            mat = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            poses[frame] = mat[:3, 3].copy()
    return poses


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(src) < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    var_src = float(np.mean(np.sum(x * x, axis=1)))
    if var_src <= 1e-12:
        return 1.0, np.eye(3), mu_dst - mu_src
    cov = (y.T @ x) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def sim3_residuals(gt_path: Path, pred_path: Path) -> tuple[dict[int, float], float, str]:
    gt = load_traj_positions(gt_path)
    pred = load_traj_positions(pred_path)
    frames = sorted(set(gt) & set(pred))
    if len(frames) < 3:
        return {}, float("nan"), "fewer_than_3_common_frames"
    gt_pos = np.stack([gt[frame] for frame in frames], axis=0)
    pred_pos = np.stack([pred[frame] for frame in frames], axis=0)
    scale, rot, trans = umeyama(pred_pos, gt_pos)
    aligned = scale * (pred_pos @ rot.T) + trans
    residual = np.linalg.norm(aligned - gt_pos, axis=1)
    blocker = "" if len(frames) == len(gt) == len(pred) else "trajectory_frame_intersection_used"
    return {int(frame): float(err) for frame, err in zip(frames, residual)}, scale, blocker


def rmse(values: list[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def median(values: list[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def max_or_nan(values: list[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.max(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def relative_delta(base: float, value: float) -> float:
    if not math.isfinite(base) or abs(base) <= 1e-12 or not math.isfinite(value):
        return float("nan")
    return (base - value) / abs(base)


def operation_features(trace_rows: list[dict[str, Any]], selected_global_frames: set[int]) -> dict[str, Any]:
    cache_rows = [row for row in trace_rows if row.get("row_type") == "cache_operation"]
    op_counts = Counter(str(row.get("operation_type", "")) for row in cache_rows)
    cache_append_token_sum = sum(int(float(row.get("token_count", 0) or 0)) for row in cache_rows if row.get("operation_type") == "cache_append")
    patch_cache_append_token_sum = sum(
        int(float(row.get("token_count", 0) or 0))
        for row in cache_rows
        if row.get("operation_type") == "cache_append" and row.get("token_type") == "image_patch"
    )
    selected_patch_cache_append_token_sum = sum(
        int(float(row.get("token_count", 0) or 0))
        for row in cache_rows
        if row.get("operation_type") == "cache_append"
        and row.get("token_type") == "image_patch"
        and int(float(row.get("frame_id", -1) or -1)) in selected_global_frames
    )
    return {
        "trace_rows": len(trace_rows),
        "cache_operation_rows": len(cache_rows),
        "operation_counts_json": json.dumps(dict(sorted(op_counts.items())), sort_keys=True),
        "cache_append_token_sum": cache_append_token_sum,
        "patch_cache_append_token_sum": patch_cache_append_token_sum,
        "selected_patch_cache_append_token_sum": selected_patch_cache_append_token_sum,
        "skip_append_operation_rows": sum(1 for row in cache_rows if str(row.get("skip_append", "")).lower() == "true"),
        "context_only_operation_rows": sum(1 for row in cache_rows if str(row.get("context_only_append", "")).lower() == "true"),
        "trace_error_rows": sum(1 for row in trace_rows if row.get("row_type") == "trace_error"),
    }


def action_fidelity(action_rows: list[dict[str, Any]], mode: str, expected_indices: set[int]) -> tuple[bool, set[int], str]:
    if mode == "no_action":
        observed = {
            int(row["sample_position"])
            for row in action_rows
            if str(row.get("forced_non_keyframe", "")).lower() == "true"
            or str(row.get("forced_context_only", "")).lower() == "true"
            or str(row.get("forced_anchor_only", "")).lower() == "true"
        }
        return (not expected_indices and not observed), observed, "" if not observed else "unexpected_forced_action_rows"
    if mode == "force_non_keyframe":
        observed = {int(row["sample_position"]) for row in action_rows if str(row.get("forced_non_keyframe", "")).lower() == "true"}
    elif mode == "context_only_special":
        observed = {
            int(row["sample_position"])
            for row in action_rows
            if str(row.get("forced_context_only", "")).lower() == "true"
            and str(row.get("forced_anchor_only", "")).lower() != "true"
        }
    elif mode == "anchor_special_only":
        observed = {int(row["sample_position"]) for row in action_rows if str(row.get("forced_anchor_only", "")).lower() == "true"}
    else:
        observed = set()
    blocker = "" if observed == expected_indices else f"expected_{sorted(expected_indices)}_observed_{sorted(observed)}"
    return observed == expected_indices, observed, blocker


def build() -> dict[str, Any]:
    config_rows = read_csv(CONFIG_ROWS)
    target_rows = {row["target_id"]: row for row in read_csv(TARGET_ROWS)}
    run_rows = read_csv(RUN_RESULTS)
    latest = latest_run_results(run_rows)
    metric_rows: list[dict[str, Any]] = []
    trace_rows_out: list[dict[str, Any]] = []

    for cfg in config_rows:
        target_id = cfg["target_id"]
        seq = cfg["seq"]
        dataset = cfg["dataset"]
        method = cfg["method"]
        action_name = cfg["action_name"]
        mode = cfg["stage4_action_mode"]
        expected_indices = parse_indices(cfg.get("force_non_keyframe_indices", ""))
        expected_global = parse_indices(cfg.get("force_global_frame_indices", ""))
        traj = WORKSPACE / dataset / seq / method / "traj.txt"
        gt_traj = WORKSPACE / dataset / seq / "gt/traj.txt"
        traj_exists = traj.exists()
        gt_exists = gt_traj.exists()
        metric_available = traj_exists and gt_exists
        metric_blocker = ""
        residuals: dict[int, float] = {}
        sim3_scale = float("nan")
        if metric_available:
            try:
                residuals, sim3_scale, metric_blocker = sim3_residuals(gt_traj, traj)
            except Exception as exc:  # noqa: BLE001 - audit script should record blocker instead of hiding it
                metric_available = False
                metric_blocker = f"{type(exc).__name__}: {exc}"
        else:
            metric_blocker = "missing_traj_or_gt"

        trace_start = int(float(cfg["trace_start_idx"]))
        target = target_rows.get(target_id, {})
        target_frame_start = cfg.get("target_frame_start") or target.get("target_frame_start")
        target_frame_end = cfg.get("target_frame_end") or target.get("target_frame_end")
        if target_frame_start is None or target_frame_end is None:
            raise KeyError(f"missing target_frame_start/end for target_id={target_id}")
        target_start = int(float(target_frame_start)) - trace_start
        target_end = int(float(target_frame_end)) - trace_start
        target_values = [err for frame, err in residuals.items() if target_start <= frame <= target_end]
        trace_values = list(residuals.values())
        action_jsonl = load_jsonl(Path(cfg["action_file"]))
        cache_jsonl = load_jsonl(Path(cfg["trace_file"]))
        fidelity_pass, observed_indices, fidelity_blocker = action_fidelity(action_jsonl, mode, expected_indices)
        ops = operation_features(cache_jsonl, expected_global)

        phase_status: dict[str, Any] = {}
        all_phase_success = True
        for phase in ("prepare", "run_worker", "evaluate", "report"):
            run_name = f"kitti_lingbot_v107r_stage6_{action_name}_{target_id}_{phase}"
            row = latest.get((run_name, phase))
            rc = int(float(row.get("returncode", 1))) if row else 1
            phase_status[f"{phase}_returncode"] = rc
            all_phase_success = all_phase_success and rc == 0

        trace_fidelity_pass = bool(
            all_phase_success
            and metric_available
            and fidelity_pass
            and ops["cache_operation_rows"] > 0
            and ops["trace_error_rows"] == 0
        )
        row = {
            "schema": "acl2_v107r_stage6_semantic_wrapper_metric_row_v1",
            "target_id": target_id,
            "target_kind": cfg["target_kind"],
            "seq": seq,
            "dataset": dataset,
            "method": method,
            "action_name": action_name,
            "action_family": cfg["action_family"],
            "stage4_action_mode": mode,
            "selector": cfg["selector"],
            "selected_count": len(expected_indices),
            "expected_local_indices": ";".join(str(x) for x in sorted(expected_indices)),
            "observed_local_indices": ";".join(str(x) for x in sorted(observed_indices)),
            "metric_available": metric_available,
            "traj_exists": traj_exists,
            "gt_exists": gt_exists,
            "target_local_start": target_start,
            "target_local_end": target_end,
            "trace_ate_sim3_rmse": rmse(trace_values),
            "target_window_ate_sim3_rmse": rmse(target_values),
            "final_aligned_error": residuals.get(max(residuals), float("nan")) if residuals else float("nan"),
            "sim3_scale_to_gt": sim3_scale,
            "action_rows": len(action_jsonl),
            "action_fidelity_pass": fidelity_pass,
            "action_fidelity_blocker": fidelity_blocker,
            "trace_fidelity_pass": trace_fidelity_pass,
            "all_phase_success": all_phase_success,
            "metric_blocker": metric_blocker,
            **phase_status,
            **ops,
        }
        metric_rows.append(row)
        trace_rows_out.append(
            {
                "schema": "acl2_v107r_stage6_semantic_wrapper_trace_fidelity_row_v1",
                "target_id": target_id,
                "target_kind": cfg["target_kind"],
                "seq": seq,
                "action_name": action_name,
                "stage4_action_mode": mode,
                "selected_count": len(expected_indices),
                "action_rows": len(action_jsonl),
                "expected_local_indices": row["expected_local_indices"],
                "observed_local_indices": row["observed_local_indices"],
                "cache_operation_rows": ops["cache_operation_rows"],
                "trace_error_rows": ops["trace_error_rows"],
                "action_fidelity_pass": fidelity_pass,
                "trace_fidelity_pass": trace_fidelity_pass,
                "operation_counts_json": ops["operation_counts_json"],
            }
        )

    by_target_action = {(row["target_id"], row["action_name"]): row for row in metric_rows}
    for row in metric_rows:
        baseline = by_target_action.get((row["target_id"], "no_action"))
        if baseline and row["metric_available"] and baseline["metric_available"]:
            base_trace = float(baseline["trace_ate_sim3_rmse"])
            base_target = float(baseline["target_window_ate_sim3_rmse"])
            row["trace_ate_rel_improvement_vs_no_action"] = relative_delta(base_trace, float(row["trace_ate_sim3_rmse"]))
            row["target_window_ate_rel_improvement_vs_no_action"] = relative_delta(base_target, float(row["target_window_ate_sim3_rmse"]))
            row["target_window_ate_rel_harm_vs_no_action"] = -float(row["target_window_ate_rel_improvement_vs_no_action"])
            for key in (
                "cache_append_token_sum",
                "patch_cache_append_token_sum",
                "selected_patch_cache_append_token_sum",
                "skip_append_operation_rows",
                "context_only_operation_rows",
            ):
                row[f"{key}_delta_vs_no_action"] = float(row[key]) - float(baseline[key])
        else:
            row["trace_ate_rel_improvement_vs_no_action"] = float("nan")
            row["target_window_ate_rel_improvement_vs_no_action"] = float("nan")
            row["target_window_ate_rel_harm_vs_no_action"] = float("nan")

    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    action_target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in metric_rows:
        action = row["action_name"]
        if action == "no_action":
            continue
        target_kind = str(row["target_kind"])
        action_target_counts[action][target_kind] += 1
        if target_kind == "high_l3":
            aggregates[action]["high_improvements"].append(float(row.get("target_window_ate_rel_improvement_vs_no_action", float("nan"))))
        if target_kind.startswith("safe_good"):
            aggregates[action]["safe_harms"].append(float(row.get("target_window_ate_rel_harm_vs_no_action", float("nan"))))
        aggregates[action]["trace_pass"].append(1.0 if row["trace_fidelity_pass"] else 0.0)
        aggregates[action]["trace_effect_delta"].append(abs(float(row.get("patch_cache_append_token_sum_delta_vs_no_action", 0.0) or 0.0)))

    aggregate_rows: list[dict[str, Any]] = []
    for action in sorted(aggregates):
        vals = aggregates[action]
        high_med = median(vals["high_improvements"])
        safe_med = median(vals["safe_harms"])
        safe_max = max_or_nan(vals["safe_harms"])
        trace_all = bool(vals["trace_pass"]) and all(bool(x) for x in vals["trace_pass"])
        aggregate_rows.append(
            {
                "schema": "acl2_v107r_stage6_semantic_wrapper_aggregate_row_v1",
                "action_name": action,
                "action_family": next((row["action_family"] for row in metric_rows if row["action_name"] == action), ""),
                "high_target_count": action_target_counts[action]["high_l3"],
                "safe_target_count": sum(v for k, v in action_target_counts[action].items() if k.startswith("safe_good")),
                "high_target_window_median_improvement": high_med,
                "safe_target_window_median_harm": safe_med,
                "safe_target_window_max_harm": safe_max,
                "trace_fidelity_pass": trace_all,
                "trace_effect_observed": bool(vals["trace_effect_delta"] and max(vals["trace_effect_delta"]) > 0.0),
            }
        )

    aggregate_by_action = {row["action_name"]: row for row in aggregate_rows}
    semantic_pass_rows: list[dict[str, Any]] = []
    for action in PRIMARY_SEMANTIC_ACTIONS:
        row = aggregate_by_action.get(action)
        if not row:
            continue
        relevant_controls = relevant_controls_for_action(action)
        control_rows = [aggregate_by_action[name] for name in sorted(relevant_controls) if name in aggregate_by_action]
        control_high = [
            float(ctrl.get("high_target_window_median_improvement", float("nan")))
            for ctrl in control_rows
        ]
        control_ceiling = max([x for x in control_high if math.isfinite(x)], default=float("nan"))
        control_details = [
            {
                "action_name": ctrl["action_name"],
                "high_target_window_median_improvement": ctrl.get("high_target_window_median_improvement"),
                "safe_target_window_median_harm": ctrl.get("safe_target_window_median_harm"),
                "safe_target_window_max_harm": ctrl.get("safe_target_window_max_harm"),
                "trace_fidelity_pass": ctrl.get("trace_fidelity_pass"),
            }
            for ctrl in control_rows
        ]
        high_med = float(row["high_target_window_median_improvement"])
        safe_med = float(row["safe_target_window_median_harm"])
        safe_max = float(row["safe_target_window_max_harm"])
        coverage_pass = int(row["high_target_count"]) >= 4 and int(row["safe_target_count"]) >= 4
        control_pass = math.isfinite(control_ceiling) and high_med > control_ceiling
        action_pass = bool(
            coverage_pass
            and row["trace_fidelity_pass"]
            and row["trace_effect_observed"]
            and math.isfinite(high_med)
            and high_med >= 0.05
            and math.isfinite(safe_med)
            and safe_med <= 0.02
            and math.isfinite(safe_max)
            and safe_max <= 0.05
            and control_pass
        )
        semantic_pass_rows.append(
            {
                **row,
                "coverage_pass": coverage_pass,
                "relevant_controls": ";".join(sorted(relevant_controls)),
                "control_ceiling_high_improvement": control_ceiling,
                "control_details_json": json.dumps(clean_json(control_details), sort_keys=True),
                "semantic_beats_controls": control_pass,
                "stage6_action_pass": action_pass,
            }
        )

    stage6_pass = any(bool(row["stage6_action_pass"]) for row in semantic_pass_rows)
    best_semantic = max(
        semantic_pass_rows,
        key=lambda row: (
            float(row.get("stage6_action_pass", False)),
            float(row.get("high_target_window_median_improvement", -999.0) or -999.0),
            -float(row.get("safe_target_window_median_harm", 999.0) or 999.0),
        ),
        default={},
    )

    write_csv(OUT / "action_metric_rows.csv", metric_rows)
    write_csv(OUT / "action_trace_fidelity.csv", trace_rows_out)
    write_csv(OUT / "action_aggregate_metrics.csv", aggregate_rows)
    write_csv(OUT / "semantic_action_gate_rows.csv", semantic_pass_rows)
    write_csv(STAGE6_ROOT / "runtime_pilot_metric_rows.csv", metric_rows)
    write_csv(STAGE6_ROOT / "runtime_pilot_aggregate_metrics.csv", aggregate_rows)

    summary = {
        "schema": "acl2_v107r_stage6_semantic_wrapper_summary_v1",
        "stage6_pass": stage6_pass,
        "stage6_runtime_pilot_run": bool(metric_rows),
        "target_count": len({row["target_id"] for row in metric_rows}),
        "metric_row_count": len(metric_rows),
        "run_result_rows": len(run_rows),
        "run_failures": sum(1 for row in latest.values() if int(float(row.get("returncode", 1))) != 0),
        "primary_semantic_actions": sorted(PRIMARY_SEMANTIC_ACTIONS),
        "required_controls": sorted(REQUIRED_CONTROLS),
        "best_semantic_action": best_semantic,
        "semantic_action_gate_rows": semantic_pass_rows,
        "blocker": "" if stage6_pass else "no_semantic_wrapper_action_met_high_improvement_good_safety_trace_and_control_gates",
        "outputs": {
            "action_metric_rows": rel(OUT / "action_metric_rows.csv"),
            "action_aggregate_metrics": rel(OUT / "action_aggregate_metrics.csv"),
            "action_trace_fidelity": rel(OUT / "action_trace_fidelity.csv"),
            "semantic_action_gate_rows": rel(OUT / "semantic_action_gate_rows.csv"),
        },
        "runtime_boundary": "Actions change LingBot streaming KV/cache write behavior via force_non_keyframe/context_only/anchor_only hooks; no output post-processing action is used.",
    }
    write_json(OUT / "stage6_summary.json", summary)
    write_json(STAGE6_ROOT / "stage6_summary.json", {**summary, "supersedes_prior_blocked_placeholder": True})

    harm_report = [
        "# v107R Stage6 Semantic Wrapper Policy Good Harm Attribution",
        "",
        f"- stage6_pass: `{stage6_pass}`",
        f"- blocker: `{summary['blocker']}`",
        f"- best_semantic_action: `{best_semantic.get('action_name', '')}`",
        f"- best_high_target_window_median_improvement: `{best_semantic.get('high_target_window_median_improvement', '')}`",
        f"- best_safe_target_window_median_harm: `{best_semantic.get('safe_target_window_median_harm', '')}`",
        f"- best_safe_target_window_max_harm: `{best_semantic.get('safe_target_window_max_harm', '')}`",
        "",
        "Evidence interpretation:",
        "- A positive high improvement means lower target-window Sim3 ATE than no-action for high-L3 windows.",
        "- A positive safe harm means higher target-window Sim3 ATE than no-action for safe-good windows.",
        "- A semantic action is not promoted unless it also beats same-count/random and low-risk reverse controls.",
    ]
    if not stage6_pass:
        harm_report.append(
            "- No success is claimed here; failed gates remain blockers and require another repair branch before any full-KITTI promotion."
        )
    (OUT / "good_harm_attribution.md").write_text("\n".join(harm_report) + "\n", encoding="utf-8")

    trace_report = [
        "# v107R Stage6 Trace Effect Report",
        "",
        f"- metric_rows: `{len(metric_rows)}`",
        f"- aggregate_rows: `{len(aggregate_rows)}`",
        f"- all_trace_fidelity_pass_rows: `{all(bool(row['trace_fidelity_pass']) for row in metric_rows) if metric_rows else False}`",
        "",
        "The trace gate checks both action-row fidelity and v107 cache-operation rows. A trajectory-only metric without trace fidelity is not accepted.",
    ]
    if aggregate_rows and not stage6_pass:
        trace_report.append("Trace movement alone is insufficient; the current branch remains No-Go if geometry/safety/control gates fail.")
    (OUT / "trace_effect_report.md").write_text("\n".join(trace_report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = build()
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
