#!/usr/bin/env python3
"""Build ACL2 v119-TF LingBot Stage0 fresh FlashInfer baseline metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage0_lingbot_fresh_baselines"
WORKSPACE = RUN_ROOT / "workspace"
OUT_DIR = RUN_ROOT / "full_sequence_metrics"
DATASET = "kitti_v119_stage0_00_01_02_05"
METHOD = "lingbot_map_stream_default_flashinfer_v119_stage0"
SEQ_ORDER = ["00", "01", "02", "05"]
ROLLING_WINDOW = 64
LOCAL_WINDOWS = [32, 64]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def summarize_seq(seq: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
                "schema": "acl2_v119tf_lingbot_stage0_fresh_local_window_row_v1",
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

    full_row = {
        "schema": "acl2_v119tf_lingbot_stage0_fresh_full_metric_row_v1",
        "dataset": DATASET,
        "seq": seq,
        "model": "LingBot",
        "method": METHOD,
        "mode": "streaming",
        "setting": "default_auto_keyframe_flashinfer",
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
        "metric_scope_note": "fresh v119 KITTI sequence; LingBot default FlashInfer; complete 00/01/02/05 baseline",
        "method_root": rel(method_root),
    }
    stage0_row = {
        "schema": "acl2_v119tf_stage0_lingbot_fresh_baseline_metric_row_v1",
        "model": "LingBot",
        "reference_source": "v119_fresh_flashinfer_baseline_rerun",
        "seq": seq,
        "num_frames": int(len(gt_frames)),
        "full_ATE_sim3": full_row["ATE_full_sim3_m"],
        "RPE_translation_mean": full_row["benchmark_rpe_trans"],
        "RPE_rotation_deg_mean": full_row["benchmark_rpe_rot"],
        "final_error": full_row["final_error_m"],
        "rolling_ATE_mean": full_row["rolling_ATE_mean"],
        "rolling_ATE_p90": full_row["rolling_ATE_p90"],
        "rolling_worse_fraction_gt_0p05": full_row["rolling_worse_fraction_gt_0p05"],
        "segment_scale_log_error_median_abs": "",
        "adjacent_log_scale_jump_median": full_row["adjacent_log_scale_jump_median"],
        "adjacent_log_scale_jump_p90": full_row["adjacent_log_scale_jump_p90"],
        "global_sim3_scale": full_row["full_global_sim3_scale"],
        "global_sim3_yaw": full_row["full_global_sim3_yaw_rad"],
        "local_window_ATE_median": full_row["local_window_ATE_median"],
        "source_path": rel(OUT_DIR / "stage0_lingbot_flashinfer_baseline_rows.csv"),
        "method_root": full_row["method_root"],
    }
    return full_row, stage0_row, local_rows


def main() -> None:
    full_rows: list[dict[str, Any]] = []
    stage0_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for seq in SEQ_ORDER:
        full, stage0, local = summarize_seq(seq)
        full_rows.append(full)
        stage0_rows.append(stage0)
        local_rows.extend(local)

    write_csv(OUT_DIR / "lingbot_flashinfer_default_full_metrics.csv", full_rows)
    write_csv(OUT_DIR / "stage0_lingbot_flashinfer_baseline_rows.csv", stage0_rows)
    write_csv(OUT_DIR / "local_window_rows.csv", local_rows)
    summary = {
        "schema": "acl2_v119tf_lingbot_stage0_fresh_metric_summary_v1",
        "dataset": DATASET,
        "method": METHOD,
        "sequences": SEQ_ORDER,
        "metric_row_count": len(full_rows),
        "completed_sequences": [r["seq"] for r in full_rows if r["pose_depth_available"] and r["eval_available"]],
        "fresh_lingbot_flashinfer_baseline_complete": len(full_rows) == len(SEQ_ORDER)
        and all(r["pose_depth_available"] and r["eval_available"] for r in full_rows),
        "rolling_window": ROLLING_WINDOW,
        "local_windows": LOCAL_WINDOWS,
        "stage0_rows": stage0_rows,
    }
    (OUT_DIR / "stage0_lingbot_flashinfer_metric_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
