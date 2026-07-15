#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_w2c_txt(path: Path) -> tuple[list[int], np.ndarray]:
    frames: list[int] = []
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = np.asarray(vals[1:10], dtype=np.float64).reshape(3, 3)
            mat[:3, 3] = np.asarray(vals[10:13], dtype=np.float64)
            frames.append(int(vals[0]))
            mats.append(mat)
    return frames, np.asarray(mats, dtype=np.float64)


def similarity_align(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / max(len(src), 1)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s[-1, -1] = -1.0
    r = u @ s @ vt
    if with_scale:
        var = np.mean(np.sum(src_centered**2, axis=1))
        scale = float(np.trace(np.diag(d) @ s) / max(var, 1e-12))
    else:
        scale = 1.0
    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def rot_angle_deg(r: np.ndarray) -> float:
    cos = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def build_aligned_c2w(pred_c2w: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pred_c2w.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", r, pred_c2w[:, :3, :3])
    out[:, :3, 3] = (scale * (r @ pred_c2w[:, :3, 3].T)).T + t[None]
    return out


def rpe_delta(pred_c2w: np.ndarray, gt_c2w: np.ndarray, delta: int) -> dict[str, Any]:
    rte = []
    rre = []
    for i in range(0, len(gt_c2w) - delta):
        j = i + delta
        gt_rel = np.linalg.inv(gt_c2w[i]) @ gt_c2w[j]
        pred_rel = np.linalg.inv(pred_c2w[i]) @ pred_c2w[j]
        err = np.linalg.inv(gt_rel) @ pred_rel
        rte.append(float(np.linalg.norm(err[:3, 3])))
        rre.append(rot_angle_deg(err[:3, :3]))
    if not rte:
        return {
            f"rpe_delta{delta}_translation_mean": None,
            f"rpe_delta{delta}_translation_median": None,
            f"rpe_delta{delta}_rotation_deg_mean": None,
            f"rpe_delta{delta}_rotation_deg_median": None,
            f"rpe_delta{delta}_count": 0,
        }
    return {
        f"rpe_delta{delta}_translation_mean": float(np.mean(rte)),
        f"rpe_delta{delta}_translation_median": float(np.median(rte)),
        f"rpe_delta{delta}_rotation_deg_mean": float(np.mean(rre)),
        f"rpe_delta{delta}_rotation_deg_median": float(np.median(rre)),
        f"rpe_delta{delta}_count": int(len(rte)),
    }


def rolling_ate_stats(errors: np.ndarray, window: int = 100, stride: int = 10) -> dict[str, Any]:
    if len(errors) == 0:
        return {"rolling_ate_window": window, "rolling_ate_p90": None, "rolling_ate_count": 0}
    if len(errors) < window:
        rmse = math.sqrt(float(np.mean(errors**2)))
        return {"rolling_ate_window": len(errors), "rolling_ate_p90": rmse, "rolling_ate_count": 1}
    rmses = []
    for start in range(0, len(errors) - window + 1, stride):
        rmses.append(math.sqrt(float(np.mean(errors[start : start + window] ** 2))))
    return {
        "rolling_ate_window": int(window),
        "rolling_ate_stride": int(stride),
        "rolling_ate_p90": float(np.percentile(rmses, 90)),
        "rolling_ate_count": int(len(rmses)),
    }


def segment_scale_stats(pred_xyz: np.ndarray, gt_xyz: np.ndarray, window: int = 100, stride: int = 50) -> dict[str, Any]:
    vals = []
    for start in range(0, len(gt_xyz) - window, stride):
        end = start + window
        gt_dist = float(np.linalg.norm(gt_xyz[end] - gt_xyz[start]))
        pred_dist = float(np.linalg.norm(pred_xyz[end] - pred_xyz[start]))
        if gt_dist <= 1e-9 or pred_dist <= 1e-9:
            continue
        vals.append(math.log(pred_dist / gt_dist))
    if not vals:
        return {
            "segment_scale_window": window,
            "segment_scale_stride": stride,
            "segment_scale_log_error_mean_abs": None,
            "segment_scale_log_error_median_abs": None,
            "segment_scale_log_error_p90_abs": None,
            "segment_scale_count": 0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    abs_arr = np.abs(arr)
    return {
        "segment_scale_window": int(window),
        "segment_scale_stride": int(stride),
        "segment_scale_log_error_mean_abs": float(np.mean(abs_arr)),
        "segment_scale_log_error_median_abs": float(np.median(abs_arr)),
        "segment_scale_log_error_p90_abs": float(np.percentile(abs_arr, 90)),
        "segment_scale_log_error_signed_mean": float(np.mean(arr)),
        "segment_scale_count": int(len(arr)),
    }


def adjacent_log_scale_jump(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, Any]:
    gt_disp = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    pred_disp = np.linalg.norm(np.diff(pred_xyz, axis=0), axis=1)
    valid = (gt_disp > 1e-6) & (pred_disp > 1e-6)
    ratios = np.log(pred_disp[valid] / gt_disp[valid])
    if len(ratios) < 2:
        return {"adjacent_log_scale_jump_mean_abs": None, "adjacent_log_scale_jump_p90_abs": None, "adjacent_log_scale_jump_count": 0}
    jumps = np.abs(np.diff(ratios))
    return {
        "adjacent_log_scale_jump_mean_abs": float(np.mean(jumps)),
        "adjacent_log_scale_jump_p90_abs": float(np.percentile(jumps, 90)),
        "adjacent_log_scale_jump_count": int(len(jumps)),
    }


def summarize_sequence(output_root: Path, seq: str) -> dict[str, Any]:
    eval_summary = json.loads((output_root / "eval_summary.json").read_text(encoding="utf-8"))
    seq_key = f"{seq}/02"
    seq_summary = eval_summary["sequences"][seq_key]
    metrics = seq_summary["metrics"]["main"]
    pose_dir = output_root / seq / "02" / "poses"
    pred_frames, pred_w2c = read_w2c_txt(pose_dir / "abs_pose.txt")
    gt_frames, gt_w2c = read_w2c_txt(pose_dir / "gt_abs_pose.txt")
    if pred_frames != gt_frames:
        raise ValueError(f"frame mismatch for {seq}: pred={len(pred_frames)} gt={len(gt_frames)}")

    pred_c2w = np.linalg.inv(pred_w2c)
    gt_c2w = np.linalg.inv(gt_w2c)
    pred_xyz = pred_c2w[:, :3, 3]
    gt_xyz = gt_c2w[:, :3, 3]
    scale, r, t = similarity_align(pred_xyz, gt_xyz, with_scale=True)
    pred_aligned_xyz = (scale * (r @ pred_xyz.T)).T + t[None]
    per_frame_error = np.linalg.norm(pred_aligned_xyz - gt_xyz, axis=1)
    pred_aligned_c2w = build_aligned_c2w(pred_c2w, scale, r, t)

    row: dict[str, Any] = {
        "seq": seq,
        "output_root": str(output_root),
        "num_pose_pairs": int(metrics["num_pose_pairs"]),
        "full_ATE_sim3_rmse": float(metrics["ate_rmse"]),
        "full_ATE_sim3_mean": float(metrics["ate_mean"]),
        "full_ATE_sim3_median": float(metrics["ate_median"]),
        "global_sim3_scale": float(metrics["sim3_scale"]),
        "recomputed_sim3_scale": float(scale),
        "final_error_sim3_aligned": float(per_frame_error[-1]),
        "per_frame_error_p90_sim3_aligned": float(np.percentile(per_frame_error, 90)),
    }
    row.update(rolling_ate_stats(per_frame_error, window=100, stride=10))
    row.update(rpe_delta(pred_aligned_c2w, gt_c2w, delta=1))
    row.update(rpe_delta(pred_aligned_c2w, gt_c2w, delta=10))
    row.update(segment_scale_stats(pred_xyz, gt_xyz, window=100, stride=50))
    row.update(adjacent_log_scale_jump(pred_xyz, gt_xyz))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence")
    parser.add_argument("--seqs", default="00,02,05")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    rows = []
    for seq in [s.strip() for s in args.seqs.split(",") if s.strip()]:
        rows.append(summarize_sequence(root / "outputs" / f"baseline_kitti_{seq}", seq))

    aggregate = {
        "seqs": [row["seq"] for row in rows],
        "median_full_ATE_sim3_rmse": float(np.median([row["full_ATE_sim3_rmse"] for row in rows])),
        "mean_full_ATE_sim3_rmse": float(np.mean([row["full_ATE_sim3_rmse"] for row in rows])),
        "median_rolling_ate_p90": float(np.median([row["rolling_ate_p90"] for row in rows])),
        "median_segment_scale_log_error_abs": float(np.median([row["segment_scale_log_error_median_abs"] for row in rows])),
        "metric_definitions": {
            "full_ATE_sim3": "HorizonStream evaluator Sim3-aligned ATE over full sequence.",
            "rolling_ate_p90": "p90 of 100-frame rolling RMSE of per-frame position errors after one full-sequence Sim3 alignment.",
            "RPE": "relative pose error after applying the same full-sequence Sim3 transform to predicted c2w poses.",
            "segment_scale_log_error": "raw trajectory displacement log(pred_dist/gt_dist) over 100-frame segments with stride 50; no GT used at runtime.",
            "adjacent_log_scale_jump": "absolute adjacent difference of raw one-step displacement log scale ratios.",
        },
        "not_available_until_stage3_trace": [
            "state_norm_stats",
            "gamma_stats",
            "MRT_scale_stats",
            "head_gate_distribution",
        ],
    }
    summary = {"rows": rows, "aggregate": aggregate}
    out_dir = root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "stage1_hs_baseline_metrics_rows.csv", rows)
    (out_dir / "stage1_hs_baseline_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
