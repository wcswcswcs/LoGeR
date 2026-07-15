#!/usr/bin/env python3
"""Summarize v107R Stage7 full-sequence selected-policy ATE."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_METRICS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
V105_WORKSPACE = V105 / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05"
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"
OUT = V107R / "stage7_full_sequence_selected_policy"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"
STAGE7_COMPAT = V107R / "stage7_full_validation_or_blocked"
SEQUENCES = ["00", "01", "02", "05"]


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
    rot = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


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


def relative_improvement(base: float, value: float) -> float:
    if not math.isfinite(base) or abs(base) <= 1e-12 or not math.isfinite(value):
        return float("nan")
    return (base - value) / abs(base)


def action_fidelity(action_rows: list[dict[str, Any]], expected_indices: set[int]) -> tuple[bool, set[int], str]:
    observed = {
        int(float(row["sample_position"]))
        for row in action_rows
        if str(row.get("forced_non_keyframe", "")).lower() == "true"
    }
    if observed == expected_indices:
        return True, observed, ""
    return False, observed, f"expected_{sorted(expected_indices)}_observed_{sorted(observed)}"


def baseline_csv_by_seq() -> dict[str, dict[str, str]]:
    return {row["seq"]: row for row in read_csv(V105_METRICS)}


def mean(values: list[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return float(np.mean(vals)) if vals else float("nan")


def median(values: list[float]) -> float:
    vals = [x for x in values if math.isfinite(x)]
    return float(np.median(vals)) if vals else float("nan")


def max_rel_harm(rel_improvements: list[float]) -> float:
    harms = [-x for x in rel_improvements if math.isfinite(x) and x < 0.0]
    return max(harms) if harms else 0.0


def build() -> dict[str, Any]:
    config_rows = read_csv(CONFIG_ROWS)
    run_rows = read_csv(RUN_RESULTS)
    latest = latest_run_results(run_rows)
    baseline_csv = baseline_csv_by_seq()
    metric_rows: list[dict[str, Any]] = []

    for cfg in config_rows:
        seq = cfg["seq"]
        dataset = cfg["dataset"]
        method = cfg["method"]
        expected_indices = parse_indices(cfg.get("selected_global_frame_indices", ""))
        action_traj = WORKSPACE / dataset / seq / method / "traj.txt"
        action_gt = WORKSPACE / dataset / seq / "gt/traj.txt"
        baseline_traj = V105_WORKSPACE / seq / "lingbot_map_stream_default/traj.txt"
        baseline_gt = V105_WORKSPACE / seq / "gt/traj.txt"
        action_file = Path(cfg["action_file"])

        phase_status: dict[str, Any] = {}
        all_phase_success = True
        for phase in ("prepare", "run_worker", "evaluate", "report"):
            run_name = f"kitti_lingbot_v107r_stage7_{cfg['action_name']}_{seq}_{phase}"
            row = latest.get((run_name, phase))
            rc = int(float(row.get("returncode", 1))) if row else 1
            phase_status[f"{phase}_returncode"] = rc
            all_phase_success = all_phase_success and rc == 0

        action_metric_available = action_traj.exists() and action_gt.exists()
        action_blocker = ""
        action_residuals: dict[int, float] = {}
        action_scale = float("nan")
        if action_metric_available:
            try:
                action_residuals, action_scale, action_blocker = sim3_residuals(action_gt, action_traj)
            except Exception as exc:  # noqa: BLE001
                action_metric_available = False
                action_blocker = f"{type(exc).__name__}: {exc}"
        else:
            action_blocker = "missing_action_traj_or_gt"

        baseline_metric_available = baseline_traj.exists() and baseline_gt.exists()
        baseline_blocker = ""
        baseline_residuals: dict[int, float] = {}
        baseline_scale = float("nan")
        if baseline_metric_available:
            try:
                baseline_residuals, baseline_scale, baseline_blocker = sim3_residuals(baseline_gt, baseline_traj)
            except Exception as exc:  # noqa: BLE001
                baseline_metric_available = False
                baseline_blocker = f"{type(exc).__name__}: {exc}"
        else:
            baseline_blocker = "missing_v105_baseline_traj_or_gt"

        action_ate = rmse(list(action_residuals.values()))
        baseline_ate = rmse(list(baseline_residuals.values()))
        baseline_csv_ate = float(baseline_csv.get(seq, {}).get("ATE_full_sim3_m", "nan"))
        action_rows = load_jsonl(action_file)
        fidelity_pass, observed_indices, fidelity_blocker = action_fidelity(action_rows, expected_indices)
        common_action_frames = sorted(action_residuals)
        common_baseline_frames = sorted(baseline_residuals)

        row = {
            "schema": "acl2_v107r_stage7_full_sequence_metric_row_v1",
            "seq": seq,
            "dataset": dataset,
            "method": method,
            "action_name": cfg["action_name"],
            "source_stage6_action": cfg.get("source_stage6_action", ""),
            "selected_count": len(expected_indices),
            "expected_force_non_keyframe_indices": ";".join(str(x) for x in sorted(expected_indices)),
            "observed_force_non_keyframe_indices": ";".join(str(x) for x in sorted(observed_indices)),
            "action_rows": len(action_rows),
            "action_fidelity_pass": fidelity_pass,
            "action_fidelity_blocker": fidelity_blocker,
            "all_phase_success": all_phase_success,
            "action_metric_available": action_metric_available,
            "baseline_metric_available": baseline_metric_available,
            "metric_available": action_metric_available and baseline_metric_available,
            "frames_expected_v105": cfg.get("frames", ""),
            "action_common_frames": len(common_action_frames),
            "baseline_common_frames": len(common_baseline_frames),
            "action_ATE_full_sim3_m": action_ate,
            "baseline_recomputed_ATE_full_sim3_m": baseline_ate,
            "baseline_csv_ATE_full_sim3_m": baseline_csv_ate,
            "baseline_csv_recompute_abs_diff_m": abs(baseline_csv_ate - baseline_ate)
            if math.isfinite(baseline_csv_ate) and math.isfinite(baseline_ate)
            else float("nan"),
            "ATE_full_delta_m_action_minus_baseline": action_ate - baseline_ate
            if math.isfinite(action_ate) and math.isfinite(baseline_ate)
            else float("nan"),
            "ATE_full_rel_improvement_vs_baseline": relative_improvement(baseline_ate, action_ate),
            "action_final_aligned_error_m": action_residuals.get(max(action_residuals), float("nan")) if action_residuals else float("nan"),
            "baseline_final_aligned_error_m": baseline_residuals.get(max(baseline_residuals), float("nan")) if baseline_residuals else float("nan"),
            "action_sim3_scale_to_gt": action_scale,
            "baseline_sim3_scale_to_gt": baseline_scale,
            "action_metric_blocker": action_blocker,
            "baseline_metric_blocker": baseline_blocker,
            "action_traj": rel(action_traj) if action_traj.exists() else "",
            "baseline_traj": rel(baseline_traj) if baseline_traj.exists() else "",
            **phase_status,
        }
        metric_rows.append(row)

    rel_improvements = [float(row["ATE_full_rel_improvement_vs_baseline"]) for row in metric_rows]
    ate_complete = (
        len(metric_rows) == len(SEQUENCES)
        and all(bool(row["metric_available"]) for row in metric_rows)
        and all(bool(row["all_phase_success"]) for row in metric_rows)
        and all(bool(row["action_fidelity_pass"]) for row in metric_rows)
    )
    max_baseline_recompute_diff = max(
        [
            float(row["baseline_csv_recompute_abs_diff_m"])
            for row in metric_rows
            if math.isfinite(float(row["baseline_csv_recompute_abs_diff_m"]))
        ],
        default=float("nan"),
    )
    full_sequence_promote_pass = bool(
        ate_complete
        and math.isfinite(median(rel_improvements))
        and median(rel_improvements) > 0.0
        and max_rel_harm(rel_improvements) <= 0.02
        and (not math.isfinite(max_baseline_recompute_diff) or max_baseline_recompute_diff <= 1e-6)
    )

    write_csv(OUT / "full_sequence_metric_rows.csv", metric_rows)
    summary = {
        "schema": "acl2_v107r_stage7_full_sequence_summary_v1",
        "stage7_ate_complete": ate_complete,
        "stage7_full_sequence_promote_pass": full_sequence_promote_pass,
        "stage7_pass": full_sequence_promote_pass,
        "sequence_count": len(metric_rows),
        "sequences": [row["seq"] for row in metric_rows],
        "mean_rel_improvement_vs_baseline": mean(rel_improvements),
        "median_rel_improvement_vs_baseline": median(rel_improvements),
        "max_rel_harm_vs_baseline": max_rel_harm(rel_improvements),
        "max_baseline_csv_recompute_abs_diff_m": max_baseline_recompute_diff,
        "metric_rows": metric_rows,
        "blocker": ""
        if full_sequence_promote_pass
        else (
            "full_sequence_ate_not_complete_or_did_not_improve_without_material_harm"
            if metric_rows
            else "no_stage7_metric_rows"
        ),
        "outputs": {
            "full_sequence_metric_rows": rel(OUT / "full_sequence_metric_rows.csv"),
            "run_results": rel(RUN_RESULTS),
        },
        "policy_boundary": (
            "Stage7 embeds Stage6-selected global intervention frames into full sequences. "
            "It measures complete-sequence ATE but does not by itself prove a general recurrent semantic policy."
        ),
    }
    write_json(OUT / "stage7_summary.json", summary)
    write_json(STAGE7_COMPAT / "stage7_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
