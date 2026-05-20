#!/usr/bin/env python3
"""Audit short-rollout sandbox runs against a full online reference.

This tool intentionally does not rewrite trajectories or choose candidates. It
summarizes landed sandbox artifacts for a fixed future frame window, writing GT
audit metrics separately from no-GT proxy features so the deploy boundary stays
clear.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _angle_diff_deg,
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
    _yaw_from_pose,
)


def _parse_named_path(item: str) -> Tuple[str, Path]:
    if "=" not in item:
        raise SystemExit(f"Expected NAME=path, got {item!r}")
    name, path = item.split("=", 1)
    return name, Path(path)


def _parse_status_runtime(path: Path) -> float:
    status = path / "run_status.txt"
    if not status.exists():
        return float("nan")
    start: Optional[datetime] = None
    done: Optional[datetime] = None
    pattern = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(?P<kind>START|DONE|FAIL)\b")
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
        if match.group("kind") == "START":
            start = ts
        elif match.group("kind") in {"DONE", "FAIL"}:
            done = ts
    if start is None or done is None:
        return float("nan")
    return max(0.0, (done - start).total_seconds())


def _hmc_summary(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "hmc_state_hash.jsonl"
    out: Dict[str, object] = {
        "hmc_rows": 0,
        "state_changed_count": 0,
        "commit_mode_ok": False,
        "first_input_hash": "",
        "last_output_hash": "",
    }
    if not path.exists():
        return out
    commit_modes: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out["hmc_rows"] = int(out["hmc_rows"]) + 1
            commit_modes.append(str(rec.get("hmc_commit_mode", "")))
            input_hash = str(rec.get("controlled_input_state_hash") or "")
            output_hash = str(rec.get("hash_H_m_after_commit") or rec.get("hash_H_next") or "")
            if not out["first_input_hash"]:
                out["first_input_hash"] = input_hash
            if output_hash:
                out["last_output_hash"] = output_hash
            if input_hash and output_hash and input_hash != output_hash:
                out["state_changed_count"] = int(out["state_changed_count"]) + 1
    out["commit_mode_ok"] = bool(commit_modes) and all(mode == "probe_ttt_write" for mode in commit_modes)
    return out


def _load_window_metrics(
    pred_txt: Path,
    gt_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    frame_offset: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    frames, poses, raw_pos = _load_tum_prediction(pred_txt, gt_pos.shape[0])
    frames = frames.astype(np.int64) + int(frame_offset)
    mask = (frames >= int(start_frame)) & (frames < int(end_frame))
    if int(mask.sum()) < 3:
        raise ValueError(
            f"{pred_txt} has only {int(mask.sum())} valid frames in [{start_frame},{end_frame})"
        )
    frames_w = frames[mask]
    poses_w = poses[mask]
    raw_w = raw_pos[mask]
    matched_gt = gt_pos[frames_w]
    scale, R, t = _umeyama_sim3(raw_w, matched_gt, with_scale=True)
    aligned = _apply_alignment(poses_w, scale, R, t)
    trans_err = np.linalg.norm(aligned[:, :3, 3] - matched_gt, axis=1)
    yaw_err = _angle_diff_deg(_yaw_from_pose(aligned, "xz"), _yaw_from_pose(gt_poses[frames_w], "xz"))

    raw_steps = np.linalg.norm(np.diff(raw_w, axis=0), axis=1)
    gt_steps = np.linalg.norm(np.diff(matched_gt, axis=0), axis=1)
    step_ratio = raw_steps / np.maximum(gt_steps, 1e-12)
    proxy = {
        "step_length_median": float(np.nanmedian(raw_steps)) if raw_steps.size else float("nan"),
        "step_length_p90": float(np.nanpercentile(raw_steps, 90)) if raw_steps.size else float("nan"),
        "step_length_median_ratio_gt_audit": float(np.nanmedian(step_ratio)) if step_ratio.size else float("nan"),
        "step_length_p90_ratio_gt_audit": float(np.nanpercentile(step_ratio, 90)) if step_ratio.size else float("nan"),
        "raw_path_length": float(np.nansum(raw_steps)) if raw_steps.size else 0.0,
        "frame_count": float(frames_w.size),
    }
    metrics = {
        "start_frame": float(start_frame),
        "end_frame": float(end_frame),
        "frame_count": float(frames_w.size),
        "future_window_ate": _rmse(trans_err),
        "future_window_yaw_rmse": _rmse(yaw_err),
        "future_window_final_err": float(trans_err[-1]),
        "future_window_sim3_scale": float(scale),
    }
    return metrics, proxy


def _raw_pose_diff_vs_reference(
    ref_txt: Path,
    candidate_txt: Path,
    gt_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    candidate_frame_offset: int = 0,
) -> Dict[str, float]:
    gt_frames, _gt_poses, gt_pos = _load_kitti_gt(gt_path)
    _ = gt_frames
    ref_frames, ref_poses, _ref_raw = _load_tum_prediction(ref_txt, gt_pos.shape[0])
    cand_frames, cand_poses, _cand_raw = _load_tum_prediction(candidate_txt, gt_pos.shape[0])
    ref_frames = ref_frames.astype(np.int64)
    cand_frames = cand_frames.astype(np.int64) + int(candidate_frame_offset)

    ref_by_frame = {
        int(frame): pose
        for frame, pose in zip(ref_frames, ref_poses)
        if int(start_frame) <= int(frame) < int(end_frame)
    }
    ref_window_frames = {
        int(frame)
        for frame in ref_frames
        if int(start_frame) <= int(frame) < int(end_frame)
    }
    cand_window_frames = {
        int(frame)
        for frame in cand_frames
        if int(start_frame) <= int(frame) < int(end_frame)
    }
    max_abs = 0.0
    max_trans = 0.0
    trans_diffs: List[float] = []
    matched = 0
    for frame, pose in zip(cand_frames, cand_poses):
        frame_i = int(frame)
        if frame_i < int(start_frame) or frame_i >= int(end_frame):
            continue
        ref_pose = ref_by_frame.get(frame_i)
        if ref_pose is None:
            continue
        diff = np.asarray(pose, dtype=np.float64) - np.asarray(ref_pose, dtype=np.float64)
        max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))
        trans_diff = float(np.linalg.norm(diff[:3, 3]))
        max_trans = max(max_trans, trans_diff)
        trans_diffs.append(trans_diff)
        matched += 1
    return {
        "raw_pose_matched_frames_vs_full": float(matched),
        "raw_pose_max_abs_diff_vs_full": float(max_abs) if matched else float("nan"),
        "raw_pose_max_trans_diff_vs_full": float(max_trans) if matched else float("nan"),
        "raw_pose_mean_trans_diff_vs_full": float(np.nanmean(trans_diffs)) if trans_diffs else float("nan"),
        "timestamp_mapping_equal": ref_window_frames == cand_window_frames,
        "ref_window_frame_count": float(len(ref_window_frames)),
        "candidate_window_frame_count": float(len(cand_window_frames)),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    parser.add_argument("--full-run", required=True, help="NAME=run_dir for the full online reference")
    parser.add_argument("--sandbox-run", action="append", default=[], help="NAME=run_dir, repeatable")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--sandbox-frame-offset", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    full_name, full_dir = _parse_named_path(args.full_run)
    out_dir = Path(args.out_dir)
    gt_path = Path(args.gt)

    full_metrics, full_proxy = _load_window_metrics(
        full_dir / "01.txt",
        gt_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    full_runtime = _parse_status_runtime(full_dir)
    full_hmc = _hmc_summary(full_dir)

    metric_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    proxy_rows: List[Dict[str, object]] = []
    debug_rows: List[Dict[str, object]] = []

    ref_row: Dict[str, object] = {
        "candidate": full_name,
        "run_dir": str(full_dir),
        "is_full_reference": True,
        "runtime_sec": full_runtime,
        "runtime_ratio_vs_full": 1.0 if math.isfinite(full_runtime) and full_runtime > 0 else float("nan"),
        **full_metrics,
        **full_hmc,
    }
    metric_rows.append(ref_row)
    audit_rows.append(ref_row)
    proxy_rows.append({"candidate": full_name, "run_dir": str(full_dir), **full_proxy})
    debug_rows.append({"candidate": full_name, "run_dir": str(full_dir), **full_hmc})

    for item in args.sandbox_run:
        name, run_dir = _parse_named_path(item)
        metrics, proxy = _load_window_metrics(
            run_dir / "01.txt",
            gt_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_offset=args.sandbox_frame_offset,
        )
        raw_diff = _raw_pose_diff_vs_reference(
            full_dir / "01.txt",
            run_dir / "01.txt",
            gt_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            candidate_frame_offset=args.sandbox_frame_offset,
        )
        runtime = _parse_status_runtime(run_dir)
        hmc = _hmc_summary(run_dir)
        ate_diff = float(metrics["future_window_ate"]) - float(full_metrics["future_window_ate"])
        row: Dict[str, object] = {
            "candidate": name,
            "run_dir": str(run_dir),
            "is_full_reference": False,
            "runtime_sec": runtime,
            "runtime_ratio_vs_full": (
                runtime / full_runtime
                if math.isfinite(runtime) and math.isfinite(full_runtime) and full_runtime > 0
                else float("nan")
            ),
            "ate_diff_vs_full_reference": ate_diff,
            "phase1_parity_pass_ate_0p05": abs(ate_diff) <= 0.05,
            **metrics,
            **raw_diff,
            **hmc,
        }
        metric_rows.append(row)
        audit_rows.append(row)
        proxy_rows.append({"candidate": name, "run_dir": str(run_dir), **proxy})
        debug_rows.append({"candidate": name, "run_dir": str(run_dir), **hmc})

    _write_csv(out_dir / "short_rollout_metrics.csv", metric_rows)
    _write_csv(out_dir / "short_rollout_gt_audit.csv", audit_rows)
    _write_jsonl(out_dir / "short_rollout_proxy.jsonl", proxy_rows)
    _write_jsonl(out_dir / "candidate_commit_debug.jsonl", debug_rows)
    print(f"Wrote sandbox audit to {out_dir}")


if __name__ == "__main__":
    main()
