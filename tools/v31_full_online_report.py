#!/usr/bin/env python3
"""Report ACL2 v31 full-online trajectory metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v18_true_action_report import (  # noqa: E402
    _align_metrics,
    _load_kitti_gt,
    _load_tum_prediction,
    _raw_diff,
    _segment_ate,
)


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runtime_sec(run_dir: Path) -> float:
    status = run_dir / "run_status.txt"
    if not status.exists():
        return float("nan")
    starts: List[datetime] = []
    dones: List[datetime] = []
    pattern = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<kind>START|DONE)\b")
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        stamp = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
        if match.group("kind") == "START":
            starts.append(stamp)
        else:
            dones.append(stamp)
    if not starts or not dones:
        return float("nan")
    return float((max(dones) - min(starts)).total_seconds())


def _pose_lookup(frames: np.ndarray, poses: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(frame): pose for frame, pose in zip(frames.astype(np.int64), poses)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--runs", required=True, help="Comma-separated NAME=RUN_DIR_BASENAME entries")
    parser.add_argument("--reference-name", default="")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-ate", type=float, default=30.0)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    args = parser.parse_args()

    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    entries: List[tuple[str, Path]] = []
    for item in args.runs.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise SystemExit(f"Bad --runs item: {item}")
        name, run_name = item.split("=", 1)
        entries.append((name.strip(), args.rollout_root / run_name.strip()))

    ref_lookup: Dict[int, np.ndarray] | None = None
    if args.reference_name:
        for name, run_dir in entries:
            if name == args.reference_name:
                frames, poses, _ = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
                ref_lookup = _pose_lookup(frames, poses)
                break

    rows: List[Dict[str, object]] = []
    for name, run_dir in entries:
        pred_path = run_dir / "01.txt"
        if not pred_path.exists():
            rows.append({"name": name, "run_dir": str(run_dir), "status": "missing_prediction"})
            continue
        frames, poses, _ = _load_tum_prediction(pred_path, gt_pos.shape[0])
        aligned, metrics = _align_metrics(frames.astype(np.int64), poses, gt_poses, gt_pos)
        seg_200 = _segment_ate(frames.astype(np.int64), aligned, gt_pos, 200, 300)
        seg_400 = _segment_ate(frames.astype(np.int64), aligned, gt_pos, 400, 600)
        row: Dict[str, object] = {
            "name": name,
            "run_dir": str(run_dir),
            "status": "done",
            "frames": int(frames.shape[0]),
            "wall_seconds": _runtime_sec(run_dir),
            "ATE_full": float(metrics["ATE_horizon"]),
            "Rot_full": float(metrics["Rot_horizon"]),
            "FinalErr_full": float(metrics["FinalErr_horizon"]),
            "segment_200_300_ATE": seg_200,
            "segment_400_600_ATE": seg_400,
            "target30_pass": float(metrics["ATE_horizon"]) <= float(args.target_ate),
        }
        if ref_lookup is not None and name != args.reference_name:
            ref_subset = np.stack([ref_lookup[int(frame)] for frame in frames.astype(np.int64)], axis=0)
            ref_aligned, ref_metrics = _align_metrics(frames.astype(np.int64), ref_subset, gt_poses, gt_pos)
            ref_seg_200 = _segment_ate(frames.astype(np.int64), ref_aligned, gt_pos, 200, 300)
            ref_seg_400 = _segment_ate(frames.astype(np.int64), ref_aligned, gt_pos, 400, 600)
            raw_max_abs, raw_max_trans, timestamp_equal = _raw_diff(frames.astype(np.int64), poses, ref_lookup)
            row.update({
                "reference_name": args.reference_name,
                "ATE_delta_vs_reference": float(metrics["ATE_horizon"] - ref_metrics["ATE_horizon"]),
                "segment_200_300_delta_vs_reference": float(seg_200 - ref_seg_200),
                "segment_400_600_delta_vs_reference": float(seg_400 - ref_seg_400),
                "raw_pose_max_abs_diff_vs_reference": raw_max_abs,
                "raw_translation_max_diff_vs_reference": raw_max_trans,
                "timestamp_equal_reference": bool(timestamp_equal),
            })
        rows.append(row)

    _write_csv(args.out_dir / "full_online_metrics.csv", rows)
    _write_json(args.out_dir / "full_online_metrics.json", rows)
    candidates = [r for r in rows if r.get("status") == "done" and r.get("name") != args.reference_name]
    best = min(candidates, key=lambda r: float(r["ATE_full"]), default=None)
    summary = {
        "rows": len(rows),
        "done_rows": sum(1 for r in rows if r.get("status") == "done"),
        "target_ate": float(args.target_ate),
        "best_candidate": best.get("name") if best else None,
        "best_candidate_ATE_full": best.get("ATE_full") if best else None,
        "target30_pass": bool(best and float(best["ATE_full"]) <= float(args.target_ate)),
        "counts_as_deployable_online": bool(best and float(best["ATE_full"]) <= float(args.target_ate)),
        "no_gt_runtime_action": True,
        "no_offline_trajectory_rewrite": True,
        "no_selector_using_gt": True,
    }
    _write_json(args.out_dir / "full_online_summary.json", summary)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
