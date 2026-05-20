#!/usr/bin/env python3
"""Build a v11 result registry with strict TTT-vs-diagnostic flags."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

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


SEGMENTS = (
    (0, 100),
    (100, 200),
    (200, 300),
    (200, 400),
    (300, 400),
    (400, 500),
    (400, 600),
    (600, 800),
)


def _fixed_segment(frames: np.ndarray, aligned: np.ndarray, gt_pos_by_frame: np.ndarray, start: int, end: int) -> float:
    mask = (frames >= start) & (frames < end)
    if int(mask.sum()) < 3:
        return float("nan")
    err = aligned[mask, :3, 3] - gt_pos_by_frame[frames[mask]]
    return _rmse(np.linalg.norm(err, axis=1))


def _parse_kitti_log(path: Path) -> Dict[str, float]:
    out = {"rpe_t": float("nan"), "rpe_r": float("nan"), "ate_log": float("nan"), "rot_log": float("nan")}
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    mode = None
    for line in lines:
        if line.startswith("RPE stats"):
            mode = "rpe"
            continue
        if line.startswith("ATE RMSE stats"):
            mode = "ate"
            continue
        match = re.match(r"^01\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", line.strip())
        if not match:
            continue
        a, b = float(match.group(1)), float(match.group(2))
        if mode == "rpe":
            out["rpe_t"], out["rpe_r"] = a, b
        elif mode == "ate":
            out["ate_log"], out["rot_log"] = a, b
    return out


def _hmc_debug_summary(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "hmc_state_hash.jsonl"
    out: Dict[str, object] = {
        "hmc_state_hash_exists": path.exists(),
        "hmc_state_rows": 0,
        "commit_mode_ok": False,
        "memory_ttt_mean_rel_diff_max": float("nan"),
        "controlled_state_changed_count": 0,
        "probe_no_commit_hash_equal_all": None,
    }
    if not path.exists():
        return out
    max_diff = 0.0
    changed = 0
    probe_equal_values: List[bool] = []
    commit_modes: List[str] = []
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            commit_modes.append(str(rec.get("hmc_commit_mode", "")))
            if rec.get("probe_no_commit_hash_equal") is not None:
                probe_equal_values.append(bool(rec.get("probe_no_commit_hash_equal")))
            val = rec.get("memory_ttt_mean_rel_diff")
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                max_diff = max(max_diff, float(val))
            if rec.get("controlled_input_state_hash") != rec.get("hash_H_m_after_commit"):
                if rec.get("controlled_input_state_hash") is not None and rec.get("hash_H_m_after_commit") is not None:
                    changed += 1
    out["hmc_state_rows"] = rows
    out["commit_mode_ok"] = bool(commit_modes) and all(mode == "probe_ttt_write" for mode in commit_modes)
    out["memory_ttt_mean_rel_diff_max"] = max_diff if rows else float("nan")
    out["controlled_state_changed_count"] = changed
    out["probe_no_commit_hash_equal_all"] = all(probe_equal_values) if probe_equal_values else None
    return out


def _online_row(name: str, run_dir: Path, gt_path: Path, *, note: str = "") -> Dict[str, object]:
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    frames, raw_poses, raw_pos = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
    frames = frames.astype(np.int64)
    matched_gt = gt_pos[frames]
    scale, R, t = _umeyama_sim3(raw_pos, matched_gt, with_scale=True)
    aligned = _apply_alignment(raw_poses, scale, R, t)
    err = aligned[:, :3, 3] - matched_gt
    norm = np.linalg.norm(err, axis=1)
    yaw_err = _angle_diff_deg(_yaw_from_pose(aligned, "xz"), _yaw_from_pose(gt_poses[frames], "xz"))
    row: Dict[str, object] = {
        "run": name,
        "result_class": "online_hmc_ttt_write_candidate",
        "counts_as_ttt_write": True,
        "diagnostic_only": False,
        "source_path": str(run_dir),
        "output_from_online_hmc": True,
        "no_external_trajectory_rewrite": True,
        "no_postprocess_flag": True,
        "no_gt_runtime_action": True,
        "ate_rmse": _rmse(norm),
        "rot_rmse": float("nan"),
        "rpe_t": float("nan"),
        "rpe_r": float("nan"),
        "final_err": float(norm[-1]),
        "yaw_rmse": _rmse(yaw_err),
        "sim3_scale": float(scale),
        "note": note,
    }
    for start, end in SEGMENTS:
        row[f"seg_{start}_{end}"] = _fixed_segment(frames, aligned, gt_pos, start, end)
    row.update(_parse_kitti_log(run_dir / "kitti_benchmark.log"))
    if math.isfinite(float(row.get("rot_log", float("nan")))):
        row["rot_rmse"] = row["rot_log"]
    row.update(_hmc_debug_summary(run_dir))
    return row


def _diagnostic_rows(summary_csv: Path, selected: Iterable[str]) -> List[Dict[str, object]]:
    selected_set = set(selected)
    rows: List[Dict[str, object]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for rec in reader:
            name = rec.get("run", "")
            if selected_set and name not in selected_set:
                continue
            row: Dict[str, object] = {
                "run": name,
                "result_class": "offline_nogt_pose_proxy_diagnostic",
                "counts_as_ttt_write": False,
                "diagnostic_only": True,
                "source_path": str(summary_csv),
                "output_from_online_hmc": False,
                "no_external_trajectory_rewrite": False,
                "no_postprocess_flag": False,
                "no_gt_runtime_action": True,
                "ate_rmse": rec.get("ate_rmse"),
                "rot_rmse": "",
                "rpe_t": "",
                "rpe_r": "",
                "final_err": rec.get("final_err"),
                "yaw_rmse": rec.get("yaw_rmse"),
                "sim3_scale": rec.get("sim3_scale"),
                "seg_200_300": rec.get("seg_200_300"),
                "seg_200_400": rec.get("seg_200_400"),
                "seg_400_500": rec.get("seg_400_500"),
                "seg_400_600": rec.get("seg_400_600"),
                "note": "diagnostic only; trajectory was rewritten offline by no-GT scale proxy",
            }
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = [
        "run", "result_class", "counts_as_ttt_write", "diagnostic_only", "source_path",
        "output_from_online_hmc", "no_external_trajectory_rewrite", "no_postprocess_flag",
        "no_gt_runtime_action", "ate_rmse", "rot_rmse", "rpe_t", "rpe_r", "final_err",
        "yaw_rmse", "sim3_scale",
    ]
    for start, end in SEGMENTS:
        keys.append(f"seg_{start}_{end}")
    keys.extend([
        "hmc_state_hash_exists", "hmc_state_rows", "commit_mode_ok",
        "memory_ttt_mean_rel_diff_max", "controlled_state_changed_count",
        "probe_no_commit_hash_equal_all", "note",
    ])
    extras = sorted({key for row in rows for key in row.keys()} - set(keys))
    fields = keys + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    parser.add_argument("--online-run", action="append", default=[], help="NAME=run_dir")
    parser.add_argument("--diagnostic-summary-csv", default="")
    parser.add_argument("--diagnostic-run", action="append", default=[])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for item in args.online_run:
        if "=" not in item:
            raise SystemExit(f"--online-run must be NAME=run_dir, got {item!r}")
        name, path_s = item.split("=", 1)
        rows.append(_online_row(name, Path(path_s), Path(args.gt)))
    if args.diagnostic_summary_csv:
        rows.extend(_diagnostic_rows(Path(args.diagnostic_summary_csv), args.diagnostic_run))
    _write_csv(Path(args.out_csv), rows)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({"runs": rows}, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
