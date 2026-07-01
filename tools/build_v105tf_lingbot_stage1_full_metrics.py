#!/usr/bin/env python3
"""Build full-sequence LingBot KITTI Stage1 metrics for ACL2 v105-TF."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE1 = RESULT_ROOT / "stage1_lingbot_baseline"
WORKSPACE = STAGE1 / "workspace"
DATASET = "kitti_v105_00_01_02_05"
METHOD = "lingbot_map_stream_default"
OUT_DIR = STAGE1 / "full_sequence_metrics"
SEQ_ORDER = ["00", "01", "02", "05"]
ROLLING_WINDOW = 64
LOCAL_WINDOWS = [32, 64]


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def yaw_from_rotation(rot: np.ndarray) -> float:
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def frame_sha(frames: np.ndarray) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in frames).encode("utf-8")).hexdigest()


def window_slices(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size) if min(i + size, n) - i >= 3]


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


def summarize_seq(seq: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scene_root = WORKSPACE / DATASET / seq
    method_root = scene_root / METHOD
    gt_frames, gt = load_traj(scene_root / "gt/traj.txt")
    pred_frames, pred = load_traj(method_root / "traj.txt")
    if not np.array_equal(gt_frames, pred_frames):
        raise ValueError(f"frame index mismatch for {DATASET}/{seq}/{METHOD}")

    gt_pos = gt[:, :3, 3]
    pred_pos = pred[:, :3, 3]
    scale, rot, trans = umeyama(pred_pos, gt_pos)
    pred_aligned = apply_sim3(pred_pos, scale, rot, trans)
    residual = np.linalg.norm(pred_aligned - gt_pos, axis=1)
    rolling_window = min(ROLLING_WINDOW, len(residual))
    rolling = [
        float(np.sqrt(np.mean(residual[i : i + rolling_window] ** 2)))
        for i in range(0, len(residual) - rolling_window + 1)
    ]

    local_rows: list[dict[str, Any]] = []
    for win_size in LOCAL_WINDOWS:
        prev_scale: float | None = None
        prev_slice: slice | None = None
        for idx, sl in enumerate(window_slices(len(gt_pos), win_size)):
            local_scale, local_rot, local_trans = umeyama(pred_pos[sl], gt_pos[sl])
            local_aligned = apply_sim3(pred_pos[sl], local_scale, local_rot, local_trans)
            local_ate = rmse(local_aligned, gt_pos[sl])
            row: dict[str, Any] = {
                "schema": "acl2_v105tf_lingbot_stage1_full_local_window_row_v1",
                "dataset": DATASET,
                "seq": seq,
                "method": METHOD,
                "window_size": win_size,
                "window_index": idx,
                "frame_start": int(gt_frames[sl][0]),
                "frame_end": int(gt_frames[sl][-1]),
                "frames": int(len(gt_frames[sl])),
                "local_sim3_ate_rmse_m": local_ate,
                "local_scale": local_scale,
                "local_yaw_rad": yaw_from_rotation(local_rot),
                "adjacent_log_scale_jump": "",
                "handoff_transfer_penalty": "",
            }
            if prev_scale is not None and prev_slice is not None:
                row["adjacent_log_scale_jump"] = abs(
                    math.log(max(local_scale, 1e-12)) - math.log(max(prev_scale, 1e-12))
                )
                prev_s, prev_r, prev_t = umeyama(pred_pos[prev_slice], gt_pos[prev_slice])
                transfer = rmse(apply_sim3(pred_pos[sl], prev_s, prev_r, prev_t), gt_pos[sl])
                row["handoff_transfer_penalty"] = transfer - local_ate
            local_rows.append(row)
            prev_scale = local_scale
            prev_slice = sl

    local_ates = [float(r["local_sim3_ate_rmse_m"]) for r in local_rows]
    jumps = [float(r["adjacent_log_scale_jump"]) for r in local_rows if r["adjacent_log_scale_jump"] != ""]
    penalties = [float(r["handoff_transfer_penalty"]) for r in local_rows if r["handoff_transfer_penalty"] != ""]
    scales32 = [float(r["local_scale"]) for r in local_rows if r["window_size"] == 32]
    benchmark = load_json(method_root / "eval/traj.json")
    complete = load_json(method_root / ".complete.json")
    meta = complete.get("metadata", {}) if isinstance(complete.get("metadata"), dict) else {}

    row = {
        "schema": "acl2_v105tf_lingbot_stage1_full_metric_row_v1",
        "dataset": DATASET,
        "seq": seq,
        "model": "LingBot",
        "method": METHOD,
        "mode": "streaming",
        "setting": "default_auto_keyframe",
        "frames": int(len(gt_frames)),
        "frame_start": int(gt_frames[0]),
        "frame_end": int(gt_frames[-1]),
        "frame_index_sha256": frame_sha(gt_frames),
        "pose_depth_available": bool((method_root / ".complete.json").is_file() and (method_root / "traj.txt").is_file()),
        "eval_available": bool((method_root / "eval/traj.json").is_file()),
        "ATE_full_sim3_m": rmse(pred_aligned, gt_pos),
        "benchmark_ate": benchmark.get("ate", ""),
        "benchmark_rpe_rot": benchmark.get("rpe_rot", ""),
        "benchmark_rpe_trans": benchmark.get("rpe_trans", ""),
        "final_error_m": float(residual[-1]),
        "rolling_window": rolling_window,
        "rolling_ATE_mean": float(np.mean(rolling)) if rolling else "",
        "rolling_ATE_p90": float(np.percentile(rolling, 90)) if rolling else "",
        "rolling_ATE_max": float(np.max(rolling)) if rolling else "",
        "rolling_worse_fraction_gt_0p05": float(np.mean(np.asarray(rolling) > 0.05)) if rolling else "",
        "full_global_sim3_scale": scale,
        "full_global_sim3_yaw_rad": yaw_from_rotation(rot),
        "local_window_ATE_median": float(np.median(local_ates)) if local_ates else "",
        "adjacent_log_scale_jump_median": float(np.median(jumps)) if jumps else "",
        "adjacent_log_scale_jump_p90": float(np.percentile(jumps, 90)) if jumps else "",
        "handoff_transfer_penalty_median": float(np.median(penalties)) if penalties else "",
        "cumulative_log_scale_drift_abs": abs(math.log(max(scales32[-1], 1e-12)) - math.log(max(scales32[0], 1e-12))) if len(scales32) >= 2 else "",
        "image_width": meta.get("image_width", ""),
        "image_height": meta.get("image_height", ""),
        "metric_scope_note": "full KITTI sequence; LBM_STREAM_DEFAULT; complete 00/01/02/05 baseline",
        "method_root": method_root.relative_to(ROOT).as_posix(),
    }
    return row, local_rows


def main() -> None:
    rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for seq in SEQ_ORDER:
        row, local = summarize_seq(seq)
        rows.append(row)
        local_rows.extend(local)

    write_csv(OUT_DIR / "lingbot_stream_default_full_metrics.csv", rows)
    write_csv(OUT_DIR / "local_window_rows.csv", local_rows)
    write_csv(STAGE1 / "lingbot_streaming_full_metrics.csv", rows)

    summary = {
        "schema": "acl2_v105tf_lingbot_stage1_full_metric_summary_v1",
        "dataset": DATASET,
        "method": METHOD,
        "sequences": SEQ_ORDER,
        "metric_row_count": len(rows),
        "completed_sequences": [r["seq"] for r in rows if r["pose_depth_available"] and r["eval_available"]],
        "full_stage1_stream_default_complete": len(rows) == len(SEQ_ORDER)
        and all(r["pose_depth_available"] and r["eval_available"] for r in rows),
        "rolling_window": ROLLING_WINDOW,
        "local_windows": LOCAL_WINDOWS,
        "metric_rows": rows,
    }
    (OUT_DIR / "stage1_full_metric_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
