#!/usr/bin/env python3
"""Summarize ACL2 v105-TF LingBot Stage 1 debug trajectory metrics.

The script intentionally discovers completed debug96 LingBot benchmark outputs
from the workspace instead of assuming a single smoke run.  It computes the
same L0-L4-style trajectory summaries for every completed method output and
leaves unsupported fields blank rather than fabricating runtime or GPU data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE1 = RESULT_ROOT / "stage1_lingbot_baseline"
WORKSPACE = STAGE1 / "workspace"
OUT_DIR = STAGE1 / "debug96_metrics"


@dataclass(frozen=True)
class Case:
    dataset: str
    scene: str
    method: str
    scene_root: Path
    method_root: Path


def load_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
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
    r = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def rmse(points_a: np.ndarray, points_b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((points_a - points_b) ** 2, axis=1))))


def yaw_from_rotation(rotation: np.ndarray) -> float:
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def window_slices(n: int, size: int) -> list[slice]:
    return [slice(start, min(start + size, n)) for start in range(0, n, size) if min(start + size, n) - start >= 3]


def frame_sha(frames: np.ndarray) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in frames).encode("utf-8")).hexdigest()


def method_mode(method: str) -> str:
    return "windowed" if "window" in method else "streaming"


def method_setting(method: str) -> str:
    if method.endswith("_kf1"):
        return "kf1"
    if method.endswith("_kf4"):
        return "kf4"
    if "window64" in method:
        return "window64"
    return "default"


def discover_cases() -> list[Case]:
    cases: list[Case] = []
    if not WORKSPACE.exists():
        return cases
    for dataset_dir in sorted(p for p in WORKSPACE.iterdir() if p.is_dir() and p.name.endswith("debug96")):
        for scene_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            if not (scene_dir / "gt/traj.txt").is_file():
                continue
            for method_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
                if method_dir.name == "gt":
                    continue
                if (method_dir / "traj.txt").is_file():
                    cases.append(
                        Case(
                            dataset=dataset_dir.name,
                            scene=scene_dir.name,
                            method=method_dir.name,
                            scene_root=scene_dir,
                            method_root=method_dir,
                        )
                    )
    return cases


def load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_case(case: Case) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_frames, gt = load_traj(case.scene_root / "gt/traj.txt")
    pred_frames, pred = load_traj(case.method_root / "traj.txt")
    if not np.array_equal(gt_frames, pred_frames):
        raise ValueError(f"GT and prediction frame indices differ for {case.dataset}/{case.scene}/{case.method}")

    gt_pos = gt[:, :3, 3]
    pred_pos = pred[:, :3, 3]
    full_scale, full_rot, full_t = umeyama(pred_pos, gt_pos)
    pred_aligned = apply_sim3(pred_pos, full_scale, full_rot, full_t)
    residual = np.linalg.norm(pred_aligned - gt_pos, axis=1)

    rolling_window = min(16, len(residual))
    rolling = [
        float(np.sqrt(np.mean(residual[i : i + rolling_window] ** 2)))
        for i in range(0, len(residual) - rolling_window + 1)
    ]

    local_rows: list[dict[str, Any]] = []
    for win_size in [32, 64]:
        prev_scale: float | None = None
        prev_slice: slice | None = None
        for idx, sl in enumerate(window_slices(len(gt_pos), win_size)):
            scale, rot, trans = umeyama(pred_pos[sl], gt_pos[sl])
            aligned = apply_sim3(pred_pos[sl], scale, rot, trans)
            local_ate = rmse(aligned, gt_pos[sl])
            row: dict[str, Any] = {
                "dataset": case.dataset,
                "seq": case.scene,
                "method": case.method,
                "mode": method_mode(case.method),
                "setting": method_setting(case.method),
                "window_size": win_size,
                "window_index": idx,
                "frame_start": int(gt_frames[sl][0]),
                "frame_end": int(gt_frames[sl][-1]),
                "frames": int(len(gt_frames[sl])),
                "local_sim3_ate_rmse_m": local_ate,
                "local_scale": scale,
                "local_yaw_rad": yaw_from_rotation(rot),
                "adjacent_log_scale_jump": "",
                "handoff_transfer_penalty": "",
            }
            if prev_scale is not None and prev_slice is not None:
                row["adjacent_log_scale_jump"] = abs(math.log(max(scale, 1e-12)) - math.log(max(prev_scale, 1e-12)))
                prev_s, prev_r, prev_t = umeyama(pred_pos[prev_slice], gt_pos[prev_slice])
                transfer = rmse(apply_sim3(pred_pos[sl], prev_s, prev_r, prev_t), gt_pos[sl])
                row["handoff_transfer_penalty"] = transfer - local_ate
            local_rows.append(row)
            prev_scale = scale
            prev_slice = sl

    jumps = [float(r["adjacent_log_scale_jump"]) for r in local_rows if r["adjacent_log_scale_jump"] != ""]
    penalties = [float(r["handoff_transfer_penalty"]) for r in local_rows if r["handoff_transfer_penalty"] != ""]
    local_ates = [float(r["local_sim3_ate_rmse_m"]) for r in local_rows]
    scales = [float(r["local_scale"]) for r in local_rows if r["window_size"] == 32]
    benchmark = load_json_if_present(case.method_root / "eval/traj.json")
    complete = load_json_if_present(case.method_root / ".complete.json")
    complete_meta = complete.get("metadata", {}) if isinstance(complete.get("metadata", {}), dict) else {}

    row: dict[str, Any] = {
        "schema": "acl2_v105tf_lingbot_stage1_debug96_metric_row_v2",
        "dataset": case.dataset,
        "seq": case.scene,
        "model": "LingBot",
        "method": case.method,
        "mode": method_mode(case.method),
        "setting": method_setting(case.method),
        "frames": int(len(gt_frames)),
        "frame_start": int(gt_frames[0]),
        "frame_end": int(gt_frames[-1]),
        "frame_index_sha256": frame_sha(gt_frames),
        "pose_depth_available": bool((case.method_root / ".complete.json").is_file() and (case.method_root / "traj.txt").is_file()),
        "eval_available": bool((case.method_root / "eval/traj.json").is_file()),
        "full_ATE": rmse(pred_aligned, gt_pos),
        "ATE_full_sim3_debug96_m": rmse(pred_aligned, gt_pos),
        "benchmark_ate": benchmark.get("ate", ""),
        "benchmark_rpe_rot": benchmark.get("rpe_rot", ""),
        "benchmark_rpe_trans": benchmark.get("rpe_trans", ""),
        "final_error": float(residual[-1]),
        "final_error_m": float(residual[-1]),
        "rolling_window": rolling_window,
        "rolling_ATE_mean": float(np.mean(rolling)) if rolling else "",
        "rolling_ATE_p90": float(np.percentile(rolling, 90)) if rolling else "",
        "rolling_ATE_max": float(np.max(rolling)) if rolling else "",
        "rolling_worse_fraction_gt_0p05": float(np.mean(np.asarray(rolling) > 0.05)) if rolling else "",
        "full_global_sim3_scale": full_scale,
        "full_global_sim3_yaw_rad": yaw_from_rotation(full_rot),
        "local_window_ATE_median": float(np.median(local_ates)) if local_ates else "",
        "adjacent_log_scale_jump_median": float(np.median(jumps)) if jumps else "",
        "adjacent_log_scale_jump_p90": float(np.percentile(jumps, 90)) if jumps else "",
        "handoff_transfer_penalty_median": float(np.median(penalties)) if penalties else "",
        "overlap_to_future_rmse_median": "",
        "cumulative_log_scale_drift_abs": abs(math.log(max(scales[-1], 1e-12)) - math.log(max(scales[0], 1e-12))) if len(scales) >= 2 else "",
        "runtime_fps": "",
        "max_gpu_mem": "",
        "image_width": complete.get("image_width", complete_meta.get("image_width", "")),
        "image_height": complete.get("image_height", complete_meta.get("image_height", "")),
        "metric_scope_note": "debug96 smoke/diagnostic only; not full KITTI 00/01/02/05 Stage1 baseline",
        "method_root": case.method_root.relative_to(ROOT).as_posix(),
    }
    return row, local_rows


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


def make_job_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "run_name": "stage0_repo_env_audit",
            "phase": "stage0",
            "returncode": 0,
            "status": "pass",
            "evidence": "stage0_summary.json stage1_baseline_allowed=true",
        },
        {
            "run_name": "stage1_config_generation",
            "phase": "config_generation",
            "returncode": 0,
            "status": "pass",
            "evidence": "config_generation_summary.json",
        },
        {
            "run_name": "kitti_lingbot_stream_default_debug96",
            "phase": "run.py",
            "returncode": 1,
            "status": "failed_then_bypassed",
            "evidence": "nested dispatch failed because run.py called bare conda inside conda-run env",
        },
        {
            "run_name": "kitti_lingbot_stream_default_debug96",
            "phase": "run_worker_first",
            "returncode": 1,
            "status": "failed_repaired",
            "evidence": "OpenEXR missing while saving confidence/depth EXR",
        },
        {
            "run_name": "OpenEXR_dependency_repair",
            "phase": "pip_install",
            "returncode": 0,
            "status": "repair_applied",
            "evidence": "conda run -n loger python -m pip install OpenEXR; OpenEXR 3.4.13 import ok",
        },
    ]
    for row in metric_rows:
        run_name = f"{row['dataset']}/{row['seq']}/{row['method']}"
        rows.extend(
            [
                {
                    "run_name": run_name,
                    "phase": "run_worker_output",
                    "returncode": 0,
                    "status": "pass",
                    "evidence": f"{row['method_root']}/.complete.json",
                },
                {
                    "run_name": run_name,
                    "phase": "evaluate",
                    "returncode": 0 if row["eval_available"] else "",
                    "status": "pass" if row["eval_available"] else "missing",
                    "evidence": f"{row['method_root']}/eval/traj.json",
                },
                {
                    "run_name": run_name,
                    "phase": "metric_summary",
                    "returncode": 0,
                    "status": "pass",
                    "evidence": "debug96_metrics/stage1_debug96_all_metric_rows.csv",
                },
            ]
        )
    return rows


def make_gate_probe(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stream_default_seqs = sorted(
        {str(r["seq"]) for r in metric_rows if r["mode"] == "streaming" and r["setting"] == "default"}
    )
    eval_rows = [r for r in metric_rows if r["eval_available"] and r["pose_depth_available"]]
    high_rolling = [
        f"{r['dataset']}/{r['seq']}/{r['method']}"
        for r in metric_rows
        if r["rolling_ATE_p90"] != "" and float(r["rolling_ATE_p90"]) > 0.05
    ]
    seq00_rows = [r for r in metric_rows if r["seq"] == "00"]
    sensitivity = {}
    if len(seq00_rows) >= 2:
        sensitivity = {
            r["method"]: {
                "dataset": r["dataset"],
                "frame_index_sha256": r["frame_index_sha256"],
                "rolling_ATE_p90": r["rolling_ATE_p90"],
                "adjacent_log_scale_jump_p90": r["adjacent_log_scale_jump_p90"],
                "handoff_transfer_penalty_median": r["handoff_transfer_penalty_median"],
            }
            for r in seq00_rows
        }
    return {
        "schema": "acl2_v105tf_lingbot_stage1_debug96_gate_probe_v1",
        "scope": "debug96 only; does not satisfy full Stage1 00/01/02/05 closeout",
        "completed_metric_rows": len(metric_rows),
        "stream_default_sequences": stream_default_seqs,
        "baseline_runs_complete_for_ge_2_debug_sequences": len(stream_default_seqs) >= 2,
        "pose_depth_outputs_available_for_metric_rows": len(eval_rows) == len(metric_rows) and bool(metric_rows),
        "l0_l4_debug_metrics_computed": bool(metric_rows),
        "analyzable_debug_failure_rows_gt_0p05_rolling_p90": high_rolling,
        "seq00_keyframe_window_sensitivity_probe": sensitivity,
        "full_stage1_complete": False,
        "missing_for_full_stage1": [
            "full or broader KITTI 00/01/02/05 baseline",
            "LoGeR comparison metrics",
            "Stage2 GCA trace no-action parity",
        ],
    }


def write_report(metric_rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    lines = [
        "# Stage1 Debug96 Metric Alignment Report",
        "",
        "Scope: LingBot debug96 diagnostics only. This report does not close full Stage1.",
        "",
        "| dataset | seq | method | mode | frames | full_ATE | rolling_ATE_p90 | adjacent_log_scale_jump_p90 | handoff_transfer_penalty_median | benchmark_ate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            "| {dataset} | {seq} | {method} | {mode} | {frames} | {full_ATE} | {rolling_ATE_p90} | {adjacent_log_scale_jump_p90} | {handoff_transfer_penalty_median} | {benchmark_ate} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Gate probe:",
            f"- stream_default_sequences: {gate['stream_default_sequences']}",
            f"- baseline_runs_complete_for_ge_2_debug_sequences: {gate['baseline_runs_complete_for_ge_2_debug_sequences']}",
            f"- pose_depth_outputs_available_for_metric_rows: {gate['pose_depth_outputs_available_for_metric_rows']}",
            f"- l0_l4_debug_metrics_computed: {gate['l0_l4_debug_metrics_computed']}",
            f"- full_stage1_complete: {gate['full_stage1_complete']}",
            "",
            "Unsupported fields intentionally left blank in CSV: `runtime_fps`, `max_gpu_mem`, `overlap_to_future_rmse_median`.",
            "Reason: current benchmark logs do not provide direct audited values for these fields.",
        ]
    )
    (STAGE1 / "metric_alignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_cards(metric_rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        metric_rows,
        key=lambda r: float(r["rolling_ATE_p90"]) if r["rolling_ATE_p90"] != "" else -1.0,
        reverse=True,
    )
    lines = [
        "# Failure Case Cards",
        "",
        "Debug96 diagnostic cards. These are not final full-sequence failure cards.",
        "",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.extend(
            [
                f"## Card {idx}: {row['dataset']} / seq {row['seq']} / {row['method']}",
                "",
                f"- mode: `{row['mode']}`",
                f"- frames: `{row['frames']}` (`{row['frame_start']}` to `{row['frame_end']}`)",
                f"- full_ATE: `{row['full_ATE']}`",
                f"- rolling_ATE_p90: `{row['rolling_ATE_p90']}`",
                f"- adjacent_log_scale_jump_p90: `{row['adjacent_log_scale_jump_p90']}`",
                f"- handoff_transfer_penalty_median: `{row['handoff_transfer_penalty_median']}`",
                f"- benchmark_ate: `{row['benchmark_ate']}`",
                "",
            ]
        )
    (STAGE1 / "failure_case_cards.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cases = discover_cases()
    metric_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for case in cases:
        row, rows = summarize_case(case)
        metric_rows.append(row)
        local_rows.extend(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gate = make_gate_probe(metric_rows)
    summary = {
        "schema": "acl2_v105tf_lingbot_stage1_debug96_metrics_v2",
        "metric_row_count": len(metric_rows),
        "metric_rows": metric_rows,
        "gate_probe": gate,
    }
    (OUT_DIR / "stage1_debug96_metric_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "stage1_debug96_gate_probe.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(OUT_DIR / "stage1_debug96_all_metric_rows.csv", metric_rows)
    write_csv(OUT_DIR / "local_window_rows.csv", local_rows)
    write_csv(STAGE1 / "lingbot_streaming_metrics.csv", [r for r in metric_rows if r["mode"] == "streaming"])
    write_csv(STAGE1 / "lingbot_windowed_metrics.csv", [r for r in metric_rows if r["mode"] == "windowed"])
    write_csv(
        STAGE1 / "loger_comparison_metrics.csv",
        [{"status": "not_run", "reason": "LoGeR comparison not computed yet in current Stage1 debug96 expansion"}],
    )
    write_csv(STAGE1 / "job_results.csv", make_job_rows(metric_rows))
    write_report(metric_rows, gate)
    write_failure_cards(metric_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
