#!/usr/bin/env python3
"""Metric guard for ACL2 v67 dense-semantic experiments.

The guard makes the baseline contract explicit: lower ATE is better, and a
704-frame candidate cannot claim H35 progress by only beating an internal
dense-ignore baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction, _umeyama_sim3
except ImportError:  # pragma: no cover - supports execution from repo root.
    from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction, _umeyama_sim3


DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")


def _maybe_float(text: str) -> Optional[float]:
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        raise ValueError(f"Non-finite ATE value: {text}")
    return value


def _parse_results_ate(path: Path) -> float:
    if path.is_dir():
        path = path / "results_sim3" / "results_ate.txt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find results_ate.txt or ATE file at {path}")
    avg_re = re.compile(r"^Average:\s+([0-9eE+\-.]+)")
    first_re = re.compile(r"^\S+\s+([0-9eE+\-.]+)\s+")
    first_value: Optional[float] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = avg_re.match(line)
        if m:
            return float(m.group(1))
        if first_value is None:
            m = first_re.match(line)
            if m:
                first_value = float(m.group(1))
    if first_value is not None:
        return first_value
    raise ValueError(f"Could not parse ATE from {path}")


def _trajectory_path(run_or_file: Path) -> Path:
    if run_or_file.is_dir():
        return run_or_file / "01.txt"
    if run_or_file.name == "01.txt":
        return run_or_file
    candidate = run_or_file.parent.parent / "01.txt" if run_or_file.name == "results_ate.txt" else run_or_file
    if candidate.exists() and candidate.name == "01.txt":
        return candidate
    raise FileNotFoundError(f"Cannot infer 01.txt trajectory from {run_or_file}")


def _frame_count(run_or_file: Path) -> Optional[int]:
    try:
        traj = _trajectory_path(run_or_file)
    except FileNotFoundError:
        return None
    count = 0
    for line in traj.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            count += 1
    return count


def _sim3_ate_for_prefix(run_or_file: Path, gt_poses: Path, frame_limit: int) -> Tuple[float, int]:
    traj = _trajectory_path(run_or_file)
    _, _, gt_pos = _load_kitti_gt(gt_poses)
    frames, _, raw_pos = _load_tum_prediction(traj, gt_pos.shape[0])
    keep = frames < frame_limit
    frames = frames[keep]
    raw_pos = raw_pos[keep]
    if frames.shape[0] < 3:
        raise ValueError(f"Need at least 3 frames under frame_limit={frame_limit}, got {frames.shape[0]}")
    dst = gt_pos[frames]
    scale, rot, trans = _umeyama_sim3(raw_pos, dst, with_scale=True)
    aligned = (scale * (rot @ raw_pos.T)).T + trans
    err = np.linalg.norm(aligned - dst, axis=1)
    return float(np.sqrt(np.mean(err * err))), int(frames.shape[0])


def _resolve_ate(
    value_or_path: str,
    *,
    name: str,
    gt_poses: Path,
    first_n_frames: Optional[int] = None,
) -> Dict[str, Any]:
    direct = _maybe_float(value_or_path)
    if direct is not None:
        return {
            "name": name,
            "ate": direct,
            "source": "literal_value",
            "input": value_or_path,
            "frame_count": None,
        }
    path = Path(value_or_path)
    if first_n_frames is not None:
        ate, count = _sim3_ate_for_prefix(path, gt_poses, first_n_frames)
        return {
            "name": name,
            "ate": ate,
            "source": "sim3_recomputed_from_trajectory_prefix",
            "input": str(path),
            "frame_count": count,
            "frame_limit": first_n_frames,
            "gt_poses": str(gt_poses),
        }
    return {
        "name": name,
        "ate": _parse_results_ate(path),
        "source": "results_ate",
        "input": str(path),
        "frame_count": _frame_count(path),
    }


def _claim_error(delta_704: float, delta_full: float, threshold: float) -> str:
    if delta_704 > 0:
        return "candidate worse than official H35 first704"
    if delta_704 > -threshold:
        return f"candidate does not improve official H35 first704 by >= {threshold:g}m"
    if delta_full > 0:
        return "candidate worse than official H35 full"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-run",
        help="Candidate run or ATE value. Omit together with --internal-base-run for Phase O0 baseline-only output.",
    )
    parser.add_argument(
        "--internal-base-run",
        help="Internal baseline run or ATE value. Omit together with --candidate-run for Phase O0 baseline-only output.",
    )
    parser.add_argument(
        "--official-h35-first704-run-or-value",
        "--official-h35-first704-value",
        dest="official_h35_first704_run_or_value",
        required=True,
    )
    parser.add_argument(
        "--official-h35-full-run-or-value",
        "--official-h35-full-value",
        dest="official_h35_full_run_or_value",
        required=True,
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--gt-poses", default=str(DEFAULT_GT), type=Path)
    parser.add_argument("--first704-frame-limit", default=704, type=int)
    parser.add_argument("--progress-threshold-m", default=0.5, type=float)
    parser.add_argument(
        "--recompute-official-first704-from-run",
        action="store_true",
        help="Treat --official-h35-first704-run-or-value as a trajectory/run path and recompute Sim3 ATE for first N frames.",
    )
    args = parser.parse_args()

    h35_first704 = _resolve_ate(
        args.official_h35_first704_run_or_value,
        name="official_h35_first704",
        gt_poses=args.gt_poses,
        first_n_frames=args.first704_frame_limit if args.recompute_official_first704_from_run else None,
    )
    h35_full = _resolve_ate(args.official_h35_full_run_or_value, name="official_h35_full", gt_poses=args.gt_poses)

    h35_first704_ate = float(h35_first704["ate"])
    h35_full_ate = float(h35_full["ate"])

    has_candidate = args.candidate_run is not None
    has_internal = args.internal_base_run is not None
    if has_candidate != has_internal:
        raise SystemExit("--candidate-run and --internal-base-run must be provided together or both omitted")

    if not has_candidate:
        output: Dict[str, Any] = {
            "lower_ate_is_better": True,
            "mode": "baseline_only",
            "official_h35_first704_ate": h35_first704_ate,
            "official_h35_full_ate": h35_full_ate,
            "can_claim_success_without_candidate": False,
            "inputs": {
                "official_h35_first704": h35_first704,
                "official_h35_full": h35_full,
            },
            "warnings": [],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2, sort_keys=True))
        return

    candidate = _resolve_ate(args.candidate_run, name="candidate", gt_poses=args.gt_poses)
    internal = _resolve_ate(args.internal_base_run, name="internal_base", gt_poses=args.gt_poses)

    candidate_ate = float(candidate["ate"])
    internal_ate = float(internal["ate"])

    delta_internal = candidate_ate - internal_ate
    delta_h35_first704 = candidate_ate - h35_first704_ate
    delta_h35_full = candidate_ate - h35_full_ate
    threshold = float(args.progress_threshold_m)

    can_claim_704_progress = delta_h35_first704 < 0.0
    can_claim_full_progress = delta_h35_full < 0.0
    can_claim_704_gate = delta_h35_first704 <= -threshold
    can_claim_full_gate = delta_h35_full <= -threshold

    warnings = []
    if candidate.get("frame_count") not in (None, args.first704_frame_limit):
        warnings.append(
            f"candidate frame_count={candidate.get('frame_count')} is not first704; compare frame scope before claiming 704 progress"
        )
    if h35_first704.get("frame_count") not in (None, args.first704_frame_limit):
        warnings.append(
            f"official_h35_first704 frame_count={h35_first704.get('frame_count')} is not {args.first704_frame_limit}"
        )

    output: Dict[str, Any] = {
        "lower_ate_is_better": True,
        "candidate_ate": candidate_ate,
        "internal_base_ate": internal_ate,
        "official_h35_first704_ate": h35_first704_ate,
        "official_h35_full_ate": h35_full_ate,
        "delta_vs_internal_base": delta_internal,
        "delta_vs_official_h35_first704": delta_h35_first704,
        "delta_vs_official_h35_full": delta_h35_full,
        "can_claim_704_progress_vs_h35": can_claim_704_progress,
        "can_claim_full_progress_vs_h35": can_claim_full_progress,
        "can_claim_704_gate_vs_h35_by_0p5m": can_claim_704_gate,
        "can_claim_full_gate_vs_h35_by_0p5m": can_claim_full_gate,
        "error_if_claiming_success": _claim_error(delta_h35_first704, delta_h35_full, threshold),
        "progress_threshold_m": threshold,
        "inputs": {
            "candidate": candidate,
            "internal_base": internal,
            "official_h35_first704": h35_first704,
            "official_h35_full": h35_full,
        },
        "warnings": warnings,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
