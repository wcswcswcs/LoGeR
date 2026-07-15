#!/usr/bin/env python3
"""Build v108TF Stage4 full KITTI 00/02 action pilot metrics."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
OUT = RESULT_ROOT / "stage4_full_kitti_00_02_action_pilot"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_METRICS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
V105_WORKSPACE = V105 / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05"
V105_METHOD = "lingbot_map_stream_default"

ROLLING_WINDOW = 64
LOCAL_WINDOWS = (32, 64)
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


def clean_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


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
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_indices(raw: str | None) -> set[int]:
    if raw is None or raw.strip() == "":
        return set()
    return {int(float(x)) for x in raw.replace(",", ";").split(";") if x.strip()}


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            frames.append(int(vals[0]))
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            mats.append(mat)
    if not mats:
        raise ValueError(f"empty trajectory: {path}")
    return np.asarray(frames, dtype=np.int64), np.stack(mats, axis=0)


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
    rot = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ rot.T) + trans


def rmse_values(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))


def rmse_points(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def yaw_from_rotation(rot: np.ndarray) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def trajectory_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def percentile(values: list[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(vals, q)) if vals else float("nan")


def median(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def rel_improvement(base: float, value: float) -> float:
    if not math.isfinite(base) or not math.isfinite(value) or abs(base) <= 1e-12:
        return float("nan")
    return (base - value) / abs(base)


def max_rel_harm(rel_values: list[float]) -> float:
    harms = [-float(v) for v in rel_values if math.isfinite(float(v)) and float(v) < 0.0]
    return max(harms) if harms else 0.0


def align_traj(gt_path: Path, pred_path: Path) -> dict[str, Any]:
    gt_frames, gt = load_traj(gt_path)
    pred_frames, pred = load_traj(pred_path)
    gt_by_frame = {int(frame): mat for frame, mat in zip(gt_frames, gt)}
    pred_by_frame = {int(frame): mat for frame, mat in zip(pred_frames, pred)}
    frames = sorted(set(gt_by_frame) & set(pred_by_frame))
    if len(frames) < 3:
        raise ValueError(f"fewer than 3 common frames: {gt_path} {pred_path}")
    gt_common = np.stack([gt_by_frame[frame] for frame in frames], axis=0)
    pred_common = np.stack([pred_by_frame[frame] for frame in frames], axis=0)
    gt_pos = gt_common[:, :3, 3]
    pred_pos = pred_common[:, :3, 3]
    scale, rot, trans = umeyama(pred_pos, gt_pos)
    pred_aligned = apply_sim3(pred_pos, scale, rot, trans)
    residual = np.linalg.norm(pred_aligned - gt_pos, axis=1)
    frame_blocker = "" if len(frames) == len(gt_frames) == len(pred_frames) else "trajectory_frame_intersection_used"
    return {
        "frames": np.asarray(frames, dtype=np.int64),
        "gt_pos": gt_pos,
        "pred_pos": pred_pos,
        "pred_aligned": pred_aligned,
        "residual": residual,
        "scale": scale,
        "yaw": yaw_from_rotation(rot),
        "rot": rot,
        "trans": trans,
        "frame_blocker": frame_blocker,
        "gt_frame_count": int(len(gt_frames)),
        "pred_frame_count": int(len(pred_frames)),
        "common_frame_count": int(len(frames)),
    }


def rolling_summary(residual: np.ndarray, window_size: int = ROLLING_WINDOW) -> dict[str, Any]:
    if len(residual) < 3:
        return {
            "rolling_window_size": "",
            "rolling_ATE_mean": "",
            "rolling_ATE_p50": "",
            "rolling_ATE_p90": "",
            "rolling_ATE_max": "",
            "rolling_worse_fraction_gt_0p05": "",
            "rolling_worse_fraction_gt_0p10": "",
            "tail_rolling_ATE_mean": "",
            "tail_rolling_ATE_p90": "",
            "long_window_drift_slope": "",
        }
    win = min(window_size, len(residual))
    rolling = [
        float(np.sqrt(np.mean(residual[i : i + win] ** 2)))
        for i in range(0, len(residual) - win + 1)
    ]
    tail_n = max(1, int(math.ceil(0.1 * len(rolling))))
    tail = rolling[-tail_n:]
    slope = float(np.polyfit(np.arange(len(rolling), dtype=np.float64), np.asarray(rolling), 1)[0]) if len(rolling) >= 2 else 0.0
    return {
        "rolling_window_size": win,
        "rolling_ATE_mean": mean(rolling),
        "rolling_ATE_p50": percentile(rolling, 50),
        "rolling_ATE_p90": percentile(rolling, 90),
        "rolling_ATE_max": max(rolling) if rolling else float("nan"),
        "rolling_worse_fraction_gt_0p05": float(np.mean(np.asarray(rolling) > 0.05)) if rolling else float("nan"),
        "rolling_worse_fraction_gt_0p10": float(np.mean(np.asarray(rolling) > 0.10)) if rolling else float("nan"),
        "tail_rolling_ATE_mean": mean(tail),
        "tail_rolling_ATE_p90": percentile(tail, 90),
        "long_window_drift_slope": slope,
    }


def window_slices(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size) if min(i + size, n) - i >= 3]


def local_rows_for(
    cfg: dict[str, str],
    align: dict[str, Any],
    baseline_align: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt = align["gt_pos"]
    pred = align["pred_pos"]
    frames = align["frames"]
    baseline_gt = baseline_align["gt_pos"]
    baseline_pred = baseline_align["pred_pos"]
    baseline_frames = baseline_align["frames"]
    for win_size in LOCAL_WINDOWS:
        action_window_ates: list[float] = []
        baseline_window_ates: list[float] = []
        prev_scale: float | None = None
        prev_rot: np.ndarray | None = None
        prev_trans: np.ndarray | None = None
        prev_baseline_scale: float | None = None
        prev_baseline_rot: np.ndarray | None = None
        prev_baseline_trans: np.ndarray | None = None
        prev_action_slice: slice | None = None
        prev_baseline_slice: slice | None = None
        action_slices = window_slices(len(gt), win_size)
        baseline_slices = window_slices(len(baseline_gt), win_size)
        for idx, sl in enumerate(action_slices):
            scale, rot, trans = umeyama(pred[sl], gt[sl])
            aligned = apply_sim3(pred[sl], scale, rot, trans)
            local_ate = rmse_points(aligned, gt[sl])
            action_window_ates.append(local_ate)
            baseline_ate = float("nan")
            baseline_scale = float("nan")
            baseline_rot = np.eye(3)
            baseline_trans = np.zeros(3)
            if idx < len(baseline_slices):
                bsl = baseline_slices[idx]
                baseline_scale, baseline_rot, baseline_trans = umeyama(baseline_pred[bsl], baseline_gt[bsl])
                baseline_aligned = apply_sim3(baseline_pred[bsl], baseline_scale, baseline_rot, baseline_trans)
                baseline_ate = rmse_points(baseline_aligned, baseline_gt[bsl])
                baseline_window_ates.append(baseline_ate)
            transfer_penalty = ""
            baseline_transfer_penalty = ""
            adjacent_jump = ""
            baseline_adjacent_jump = ""
            if prev_scale is not None and prev_rot is not None and prev_trans is not None and prev_action_slice is not None:
                transfer = rmse_points(apply_sim3(pred[sl], prev_scale, prev_rot, prev_trans), gt[sl])
                transfer_penalty = transfer - local_ate
                adjacent_jump = abs(math.log(max(scale, 1e-12)) - math.log(max(prev_scale, 1e-12)))
            if (
                idx < len(baseline_slices)
                and prev_baseline_scale is not None
                and prev_baseline_rot is not None
                and prev_baseline_trans is not None
                and prev_baseline_slice is not None
            ):
                bsl = baseline_slices[idx]
                btransfer = rmse_points(
                    apply_sim3(baseline_pred[bsl], prev_baseline_scale, prev_baseline_rot, prev_baseline_trans),
                    baseline_gt[bsl],
                )
                baseline_transfer_penalty = btransfer - baseline_ate
                baseline_adjacent_jump = abs(
                    math.log(max(baseline_scale, 1e-12)) - math.log(max(prev_baseline_scale, 1e-12))
                )
            rows.append(
                {
                    "schema": "acl2_v108tf_stage4_local_handoff_metric_row_v1",
                    "surface_id": cfg["surface_id"],
                    "policy_id": cfg["policy_id"],
                    "policy_family": cfg["policy_family"],
                    "seq": cfg["seq"],
                    "dataset": cfg["dataset"],
                    "method": cfg["method"],
                    "action_name": cfg["action_name"],
                    "window_size": win_size,
                    "window_index": idx,
                    "frame_start": int(frames[sl][0]),
                    "frame_end": int(frames[sl][-1]),
                    "frames": int(len(frames[sl])),
                    "local_window_ATE": local_ate,
                    "baseline_local_window_ATE": baseline_ate,
                    "local_window_ATE_rel_improvement_vs_baseline": rel_improvement(baseline_ate, local_ate),
                    "local_scale": scale,
                    "baseline_local_scale": baseline_scale,
                    "local_to_global_scale_tradeoff": abs(math.log(max(scale, 1e-12)) - math.log(max(float(align["scale"]), 1e-12))),
                    "adjacent_log_scale_jump": adjacent_jump,
                    "baseline_adjacent_log_scale_jump": baseline_adjacent_jump,
                    "handoff_transfer_penalty": transfer_penalty,
                    "baseline_handoff_transfer_penalty": baseline_transfer_penalty,
                    "overlap_to_future_rmse": "",
                    "overlap_to_future_rmse_note": "not_computed_nonoverlap_v105_window_protocol",
                }
            )
            prev_scale, prev_rot, prev_trans, prev_action_slice = scale, rot, trans, sl
            if idx < len(baseline_slices):
                prev_baseline_scale = baseline_scale
                prev_baseline_rot = baseline_rot
                prev_baseline_trans = baseline_trans
                prev_baseline_slice = baseline_slices[idx]
    return rows


def summarize_local_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_ates = [float(r["local_window_ATE"]) for r in rows if r.get("local_window_ATE") != ""]
    baseline_ates = [float(r["baseline_local_window_ATE"]) for r in rows if r.get("baseline_local_window_ATE") != ""]
    rels = [float(r["local_window_ATE_rel_improvement_vs_baseline"]) for r in rows if r.get("local_window_ATE_rel_improvement_vs_baseline") != ""]
    jumps = [float(r["adjacent_log_scale_jump"]) for r in rows if r.get("adjacent_log_scale_jump") != ""]
    penalties = [float(r["handoff_transfer_penalty"]) for r in rows if r.get("handoff_transfer_penalty") != ""]
    return {
        "local_window_ATE_median": median(action_ates),
        "baseline_local_window_ATE_median": median(baseline_ates),
        "local_window_ATE_p90": percentile(action_ates, 90),
        "local_window_ATE_worse_fraction": float(np.mean(np.asarray(rels) < 0.0)) if rels else float("nan"),
        "local_window_ATE_rel_improvement_vs_baseline_median": median(rels),
        "adjacent_log_scale_jump_median": median(jumps),
        "adjacent_log_scale_jump_p90": percentile(jumps, 90),
        "handoff_transfer_penalty_median": median(penalties),
        "handoff_transfer_penalty_p90": percentile(penalties, 90),
    }


def latest_run_results(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        latest[(row.get("run_name", ""), row.get("phase", ""))] = row
    return latest


def action_fidelity_row(cfg: dict[str, str], run_rows: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    action_rows = load_jsonl(action_file)
    expected = parse_indices(cfg.get("selected_global_frame_indices", ""))
    expected_field = cfg.get("expected_action_field", "")
    mode = cfg.get("stage4_action_mode", "")
    observed = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if boolish(row.get(expected_field, False))
    }
    base_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if boolish(row.get("base_is_keyframe", False))
    }
    final_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if boolish(row.get("final_is_keyframe", False))
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
                boolish(row.get("forced_non_keyframe", False))
                and boolish(row.get("base_is_keyframe", False))
                and not boolish(row.get("final_is_keyframe", True))
            ):
                effective.add(sample)
        elif mode == "anchor_special_only":
            if (
                boolish(row.get("forced_anchor_only", False))
                and boolish(row.get("forced_context_only", False))
                and boolish(row.get("context_only_append", False))
                and str(row.get("context_only_special_mode", "")) == "scale_only"
            ):
                effective.add(sample)
        elif mode == "v106_context_only_with_local_preserve":
            heads = {int(x) for x in str(row.get("headlocal_action_heads", "")).split(",") if x.strip().isdigit()}
            expected_heads = set(range(16))
            if (
                boolish(row.get("headlocal_action_enabled", False))
                and str(row.get("headlocal_action_mode", "")) == mode
                and heads == expected_heads
            ):
                effective.add(sample)
        elif boolish(row.get(expected_field, False)):
            effective.add(sample)

    missing = expected - observed
    unexpected = observed - expected
    ineffective = expected - effective
    action_fidelity_pass = (observed == expected) and (effective == expected) and trace_error_rows == 0
    run_name = f"kitti_lingbot_v108tf_stage4_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = run_rows.get((run_name, "run_worker"), {})
    return {
        "schema": "acl2_v108tf_stage4_action_fidelity_row_v1",
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage4_action_mode": mode,
        "expected_action_field": expected_field,
        "expected_action_frame_count": len(expected),
        "observed_action_frame_count": len(observed),
        "action_effective_frame_count": len(effective),
        "action_noop_frame_count": len(ineffective),
        "expected_keyframe_count": len(expected),
        "observed_keyframe_count": len(observed & base_keyframes),
        "expected_cache_append_count": len(expected) if cfg["surface_id"] == "B" else "",
        "observed_cache_append_count": len(effective) if cfg["surface_id"] == "B" else "",
        "expected_retention_change_count": "",
        "observed_retention_change_count": "",
        "expected_eviction_change_count": "",
        "observed_eviction_change_count": "",
        "expected_trajectory_write_change_count": "",
        "observed_trajectory_write_change_count": "",
        "special_token_operation_count": len(effective) if mode == "anchor_special_only" else "",
        "trace_error_rows": trace_error_rows,
        "no_action_parity_pass": "stage1_noop_parity_checked",
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


def baseline_metric_by_seq() -> dict[str, dict[str, str]]:
    return {row["seq"]: row for row in read_csv(V105_METRICS)}


def build() -> dict[str, Any]:
    config_rows = read_csv(CONFIG_ROWS)
    run_result_rows = read_csv(RUN_RESULTS)
    latest = latest_run_results(run_result_rows)
    v105_csv = baseline_metric_by_seq()

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

        phase_status: dict[str, Any] = {}
        all_phase_success = True
        for phase in ("prepare", "run_worker", "evaluate", "report"):
            if phase == "prepare":
                run_name = f"kitti_lingbot_v108tf_stage4_prepare_{seq}"
            else:
                run_name = f"kitti_lingbot_v108tf_stage4_{cfg['policy_id']}_{seq}_{phase}"
            row = latest.get((run_name, phase))
            rc = int(float(row.get("returncode", 1))) if row else 1
            phase_status[f"{phase}_returncode"] = rc
            phase_status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
            if phase in {"prepare", "run_worker", "evaluate"}:
                all_phase_success = all_phase_success and rc == 0

        action_available = action_gt.exists() and action_traj.exists()
        baseline_available = baseline_gt.exists() and baseline_traj.exists()
        action_blocker = "" if action_available else "missing_action_traj_or_gt"
        baseline_blocker = "" if baseline_available else "missing_v105_baseline_traj_or_gt"
        action_align: dict[str, Any] | None = None
        baseline_align: dict[str, Any] | None = None
        try:
            if action_available:
                action_align = align_traj(action_gt, action_traj)
                action_blocker = action_align["frame_blocker"]
        except Exception as exc:  # noqa: BLE001
            action_available = False
            action_blocker = f"{type(exc).__name__}: {exc}"
        try:
            if seq in baseline_cache:
                baseline_align = baseline_cache[seq]
            elif baseline_available:
                baseline_align = align_traj(baseline_gt, baseline_traj)
                baseline_cache[seq] = baseline_align
                baseline_blocker = baseline_align["frame_blocker"]
        except Exception as exc:  # noqa: BLE001
            baseline_available = False
            baseline_blocker = f"{type(exc).__name__}: {exc}"

        metric_available = action_available and baseline_available and action_align is not None and baseline_align is not None
        if metric_available:
            action_res = action_align["residual"]
            baseline_res = baseline_align["residual"]
            action_rolling = rolling_summary(action_res)
            if seq not in baseline_rolling_cache:
                baseline_rolling_cache[seq] = rolling_summary(baseline_res)
            baseline_rolling = baseline_rolling_cache[seq]
            action_ate = rmse_values(action_res)
            baseline_ate = rmse_values(baseline_res)
            action_final = float(action_res[-1])
            baseline_final = float(baseline_res[-1])
            local_for_cfg = local_rows_for(cfg, action_align, baseline_align)
            local_rows.extend(local_for_cfg)
            local_summary = summarize_local_rows(local_for_cfg)
        else:
            action_rolling = rolling_summary(np.asarray([], dtype=np.float64))
            baseline_rolling = rolling_summary(np.asarray([], dtype=np.float64))
            action_ate = float("nan")
            baseline_ate = float("nan")
            action_final = float("nan")
            baseline_final = float("nan")
            local_summary = summarize_local_rows([])

        rolling_p90_rel = rel_improvement(
            float(baseline_rolling["rolling_ATE_p90"]) if baseline_rolling["rolling_ATE_p90"] != "" else float("nan"),
            float(action_rolling["rolling_ATE_p90"]) if action_rolling["rolling_ATE_p90"] != "" else float("nan"),
        )
        final_rel = rel_improvement(baseline_final, action_final)
        baseline_csv_ate = float(v105_csv.get(seq, {}).get("ATE_full_sim3_m", "nan"))
        full_row = {
            "schema": "acl2_v108tf_stage4_full_sequence_metric_row_v1",
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
            "full_ATE_sim3_relative_improvement_vs_baseline": rel_improvement(baseline_ate, action_ate),
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
            "trajectory_length_m": trajectory_length(action_align["gt_pos"]) if action_align else float("nan"),
            "runtime_sec": phase_status.get("run_worker_duration_sec", ""),
            "peak_gpu_memory_mb": "",
            "peak_gpu_memory_mb_note": "not_captured_per_run",
            "metric_available": metric_available,
            "all_phase_success": all_phase_success,
            "action_fidelity_pass": fidelity["action_fidelity_pass"],
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
                "schema": "acl2_v108tf_stage4_rolling_metric_row_v1",
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
                "rolling_ATE_mean_relative_improvement_vs_baseline": rel_improvement(
                    float(baseline_rolling["rolling_ATE_mean"]) if baseline_rolling["rolling_ATE_mean"] != "" else float("nan"),
                    float(action_rolling["rolling_ATE_mean"]) if action_rolling["rolling_ATE_mean"] != "" else float("nan"),
                ),
                "rolling_worse_fraction_gt_0p05_delta_action_minus_baseline": (
                    float(action_rolling["rolling_worse_fraction_gt_0p05"]) - float(baseline_rolling["rolling_worse_fraction_gt_0p05"])
                    if action_rolling["rolling_worse_fraction_gt_0p05"] != "" and baseline_rolling["rolling_worse_fraction_gt_0p05"] != ""
                    else ""
                ),
                "rolling_worse_fraction_gt_0p10_delta_action_minus_baseline": (
                    float(action_rolling["rolling_worse_fraction_gt_0p10"]) - float(baseline_rolling["rolling_worse_fraction_gt_0p10"])
                    if action_rolling["rolling_worse_fraction_gt_0p10"] != "" and baseline_rolling["rolling_worse_fraction_gt_0p10"] != ""
                    else ""
                ),
            }
        )

    semantic_rows: list[dict[str, Any]] = []
    rows_by_surface_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        rows_by_surface_policy[(row["surface_id"], row["policy_family"])].append(row)

    surfaces = sorted({row["surface_id"] for row in full_rows})
    passing_surfaces: list[str] = []
    candidate_gate_rows: list[dict[str, Any]] = []
    for surface in surfaces:
        sem = rows_by_surface_policy.get((surface, "semantic_plus_internal"), [])
        internal = rows_by_surface_policy.get((surface, "internal_only"), [])
        shuffle = rows_by_surface_policy.get((surface, "semantic_shuffle"), [])
        random_rows = rows_by_surface_policy.get((surface, "same_count_random"), [])
        sem_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in sem]
        sem_roll_rel = [
            float(r["rolling_ATE_p90_relative_improvement_vs_baseline"])
            for r in rolling_rows
            if r["surface_id"] == surface and r["policy_family"] == "semantic_plus_internal"
        ]
        sem_final_rel = [float(r["final_error_relative_improvement_vs_baseline"]) for r in sem]
        sem_local_rel = [float(r["local_window_ATE_rel_improvement_vs_baseline_median"]) for r in sem]
        internal_by_seq = {r["seq"]: r for r in internal}
        shuffle_by_seq = {r["seq"]: r for r in shuffle}
        random_by_seq = {r["seq"]: r for r in random_rows}

        def median_pairwise_vs(other_by_seq: dict[str, dict[str, Any]], field: str) -> float:
            vals: list[float] = []
            for row in sem:
                other = other_by_seq.get(row["seq"])
                if other is None:
                    continue
                vals.append(rel_improvement(float(other[field]), float(row[field])))
            return median(vals)

        sem_vs_internal = median_pairwise_vs(internal_by_seq, "full_ATE_sim3")
        sem_vs_shuffle = median_pairwise_vs(shuffle_by_seq, "full_ATE_sim3")
        sem_vs_random = median_pairwise_vs(random_by_seq, "full_ATE_sim3")
        internal_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in internal]
        shuffle_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in shuffle]
        random_full_rel = [float(r["full_ATE_sim3_relative_improvement_vs_baseline"]) for r in random_rows]
        sem_med = median(sem_full_rel)
        internal_med = median(internal_full_rel)
        shuffle_med = median(shuffle_full_rel)
        random_med = median(random_full_rel)
        shuffle_matches = math.isfinite(shuffle_med) and math.isfinite(sem_med) and shuffle_med >= sem_med - CONTROL_MATCH_TOL
        random_matches = math.isfinite(random_med) and math.isfinite(sem_med) and random_med >= sem_med - CONTROL_MATCH_TOL
        controls_do_not_match = not shuffle_matches and not random_matches
        sem_beats_internal = math.isfinite(sem_vs_internal) and sem_vs_internal >= 0.03
        candidate_complete = (
            len(sem) == 2
            and all(bool(r["metric_available"]) for r in sem)
            and all(bool(r["all_phase_success"]) for r in sem)
            and all(bool(r["action_fidelity_pass"]) for r in sem)
        )
        stage4_candidate_pass = bool(
            candidate_complete
            and math.isfinite(sem_med)
            and sem_med >= 0.05
            and max_rel_harm(sem_full_rel) <= 0.02
            and median(sem_roll_rel) >= 0.03
            and median(sem_final_rel) >= 0.0
            and max_rel_harm(sem_local_rel) <= 0.02
            and sem_beats_internal
            and controls_do_not_match
        )
        if stage4_candidate_pass:
            passing_surfaces.append(surface)
        semantic_row = {
            "schema": "acl2_v108tf_stage4_semantic_control_row_v1",
            "surface_id": surface,
            "semantic_plus_policy_id": f"{surface}1_semantic_plus_internal",
            "sequence_count": len(sem),
            "candidate_complete": candidate_complete,
            "semantic_plus_median_full_ATE_improvement_vs_no_action": sem_med,
            "internal_only_median_full_ATE_improvement_vs_no_action": internal_med,
            "semantic_shuffle_median_full_ATE_improvement_vs_no_action": shuffle_med,
            "same_count_random_median_full_ATE_improvement_vs_no_action": random_med,
            "full_ATE_improvement_vs_internal_only": sem_vs_internal,
            "full_ATE_improvement_vs_semantic_shuffle": sem_vs_shuffle,
            "full_ATE_improvement_vs_same_count_random": sem_vs_random,
            "rolling_p90_improvement_vs_internal_only": "",
            "good_harm_vs_internal_only": "",
            "good_harm_note": "not_available_in_full_sequence_metric_protocol",
            "semantic_shuffle_gap": sem_med - shuffle_med if math.isfinite(sem_med) and math.isfinite(shuffle_med) else float("nan"),
            "role_rotation_gap": "",
            "same_count_random_margin": sem_med - random_med if math.isfinite(sem_med) and math.isfinite(random_med) else float("nan"),
            "semantic_shuffle_matches_same_improvement": shuffle_matches,
            "same_count_random_matches_same_improvement": random_matches,
            "semantic_plus_beats_internal_on_full_ATE": sem_beats_internal,
            "controls_do_not_match_same_improvement": controls_do_not_match,
            "stage4_candidate_pass": stage4_candidate_pass,
        }
        semantic_rows.append(semantic_row)
        candidate_gate_rows.append(semantic_row)

    expected_run_workers = len(config_rows)
    run_worker_rows = [row for row in run_result_rows if row.get("phase") == "run_worker"]
    latest_run_worker_rows = [row for (_run_name, phase), row in latest.items() if phase == "run_worker"]
    historical_run_worker_failure_count = sum(
        1 for row in run_worker_rows if int(float(row.get("returncode", 1))) != 0
    )
    observed_run_workers = len(latest_run_worker_rows)
    all_run_worker_success = (
        observed_run_workers >= expected_run_workers
        and all(int(float(row.get("returncode", 1))) == 0 for row in latest_run_worker_rows)
    )
    evaluate_rows = [row for (_run_name, phase), row in latest.items() if phase == "evaluate"]
    report_rows = [row for (_run_name, phase), row in latest.items() if phase == "report"]
    metric_complete = (
        len(full_rows) == len(config_rows)
        and all(bool(row["metric_available"]) for row in full_rows)
        and observed_run_workers >= expected_run_workers
        and len(evaluate_rows) >= expected_run_workers
    )
    stage4_pass = bool(passing_surfaces)
    if not metric_complete:
        blocker = "stage4_full_kitti_metrics_not_complete"
    elif not stage4_pass:
        blocker = "NO_SURFACE_PASSES_FULL_KITTI_GATE"
    else:
        blocker = ""

    write_csv(OUT / "action_fidelity_rows.csv", action_fidelity_rows)
    write_csv(OUT / "full_sequence_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "semantic_control_rows.csv", semantic_rows)

    failure_lines = [
        "# ACL2 v108TF Stage4 Failure / Gate Analysis",
        "",
        f"metric_complete: {metric_complete}",
        f"stage4_pass: {stage4_pass}",
        f"blocker: {blocker}",
        f"observed_run_workers: {observed_run_workers}/{expected_run_workers}",
        f"historical_run_worker_rows: {len(run_worker_rows)}",
        f"historical_run_worker_failure_count: {historical_run_worker_failure_count}",
        f"observed_evaluate_rows: {len(evaluate_rows)}/{expected_run_workers}",
        f"observed_report_rows: {len(report_rows)}/{expected_run_workers}",
        "",
        "## Surface control summary",
        "",
    ]
    for row in semantic_rows:
        failure_lines.append(
            "- {surface}: pass={passed} sem_median_full_rel={sem} internal_median_full_rel={internal} "
            "shuffle_median_full_rel={shuffle} random_median_full_rel={random} sem_vs_internal={sem_vs_internal} "
            "controls_do_not_match={controls}".format(
                surface=row["surface_id"],
                passed=row["stage4_candidate_pass"],
                sem=row["semantic_plus_median_full_ATE_improvement_vs_no_action"],
                internal=row["internal_only_median_full_ATE_improvement_vs_no_action"],
                shuffle=row["semantic_shuffle_median_full_ATE_improvement_vs_no_action"],
                random=row["same_count_random_median_full_ATE_improvement_vs_no_action"],
                sem_vs_internal=row["full_ATE_improvement_vs_internal_only"],
                controls=row["controls_do_not_match_same_improvement"],
            )
        )
    if not metric_complete:
        failure_lines.extend(
            [
                "",
                "## Current blocker",
                "",
                "Stage4 full KITTI metric evidence is incomplete. Do not interpret geometry until run_worker and evaluate finish or fail with logged blocker evidence.",
            ]
        )
    elif not stage4_pass:
        failure_lines.extend(
            [
                "",
                "## No-Go taxonomy",
                "",
                "No semantic_plus_internal surface satisfied the full Stage4 gate. Use semantic_control_rows.csv and full_sequence_metric_rows.csv to distinguish full_ate_no_improvement, semantic_control_not_causal, local_geometry_degraded, or action_fidelity_blocked.",
            ]
        )
    write_text(OUT / "failure_analysis.md", "\n".join(failure_lines))

    summary = {
        "schema": "acl2_v108tf_stage4_full_kitti_metric_summary_v1",
        "stage4_pass": stage4_pass,
        "metric_complete": metric_complete,
        "blocker": blocker,
        "passing_surfaces": passing_surfaces,
        "expected_run_worker_count": expected_run_workers,
        "observed_run_worker_count": observed_run_workers,
        "observed_run_worker_historical_count": len(run_worker_rows),
        "observed_run_worker_historical_failure_count": historical_run_worker_failure_count,
        "observed_evaluate_count": len(evaluate_rows),
        "observed_report_count": len(report_rows),
        "all_run_worker_success": all_run_worker_success,
        "full_metric_row_count": len(full_rows),
        "action_fidelity_row_count": len(action_fidelity_rows),
        "semantic_control_rows": semantic_rows,
        "gate_definition": {
            "median_full_ATE_relative_improvement_min": 0.05,
            "max_sequence_full_ATE_harm": 0.02,
            "median_rolling_ATE_p90_relative_improvement_min": 0.03,
            "final_error_median_not_worse": True,
            "local_window_ATE_median_max_harm": 0.02,
            "semantic_plus_internal_vs_internal_only_min": 0.03,
            "control_match_tolerance": CONTROL_MATCH_TOL,
        },
        "outputs": {
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "full_sequence_metric_rows": rel(OUT / "full_sequence_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "semantic_control_rows": rel(OUT / "semantic_control_rows.csv"),
            "failure_analysis": rel(OUT / "failure_analysis.md"),
        },
    }
    write_json(OUT / "stage4_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
