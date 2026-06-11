#!/usr/bin/env python3
"""Aggregate ACL2 v31 Track A cue-reconditioning rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v18_true_action_report import (  # noqa: E402
    CHUNK_START_FRAME,
    _align_metrics,
    _load_kitti_gt,
    _load_tum_prediction,
    _raw_diff,
    _segment_ate,
)


CANDIDATES = [
    "V31_A0_ORIG_C23",
    "V31_A1_SEM_Z_FINE",
    "V31_A1B_SEM_Z_COARSE",
    "V31_A5_SEM_RESID_FINE_L025",
    "V31_A5B_SEM_RESID_COARSE_L025",
    "V31_B0_STATIC_RESCUE_EXISTING",
]


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


def _clean_json(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_json(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pose_lookup(frames: np.ndarray, poses: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(frame): pose for frame, pose in zip(frames.astype(np.int64), poses)}


def _eval_window(chunk: int, horizon: int) -> Tuple[int, int]:
    start = CHUNK_START_FRAME[int(chunk)]
    return start, start + 32 + (int(horizon) - 1) * 29


def _run_name(prefix: str, candidate: str, chunk: int, horizon: int) -> str:
    return f"{prefix}_{candidate}_chunk{chunk}_h{horizon}_globalgate_H9parent_SWKS3"


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


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _last_debug(run_dir: Path) -> Dict[str, object]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    return rows[-1] if rows else {}


def _semantic_summary(run_dir: Path) -> Dict[str, object]:
    rows = _read_jsonl(run_dir / "semantic_group_summary.jsonl")
    available = [r for r in rows if r.get("fine_label_available") is True]
    if not rows:
        return {}
    first = available[0] if available else rows[0]
    return {
        "semantic_rows": len(rows),
        "semantic_rows_available": len(available),
        "fine_label_available": bool(first.get("fine_label_available", False)),
        "fine_label_count": first.get("fine_label_count"),
        "fine_label_name_counts": first.get("fine_label_name_counts"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--run-prefix", default="V31_TRACKA_H10_R1")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--report-prefix", default="track_a_h10")
    parser.add_argument("--ate-threshold", type=float, default=-1.5)
    parser.add_argument("--segment-threshold", type=float, default=-3.0)
    parser.add_argument("--downstream-threshold", type=float, default=1.0)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    args = parser.parse_args()

    rollout_root = Path(args.rollout_root)
    out_dir = Path(args.out_dir)
    chunks = [int(x) for x in str(args.chunks).split(",") if x.strip()]
    candidates = [x.strip() for x in str(args.candidates).split(",") if x.strip()]
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))

    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for chunk in chunks:
        ref_dir = rollout_root / _run_name(args.run_prefix, "V31_BASE_H9_REFERENCE", chunk, args.horizon)
        if not (ref_dir / "01.txt").exists():
            missing.append({"chunk": chunk, "candidate": "V31_BASE_H9_REFERENCE", "reason": "missing_reference"})
            continue
        h9_frames, h9_poses, _ = _load_tum_prediction(ref_dir / "01.txt", gt_pos.shape[0])
        h9_lookup = _pose_lookup(h9_frames, h9_poses)
        eval_start, eval_end = _eval_window(chunk, args.horizon)
        for candidate in candidates:
            run_dir = rollout_root / _run_name(args.run_prefix, candidate, chunk, args.horizon)
            if not (run_dir / "01.txt").exists():
                missing.append({"chunk": chunk, "candidate": candidate, "reason": "missing_prediction"})
                continue
            frames, poses, _ = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
            frames = frames.astype(np.int64)
            eval_mask = (frames >= eval_start) & (frames < eval_end)
            if int(eval_mask.sum()) < 3:
                missing.append({"chunk": chunk, "candidate": candidate, "reason": "too_few_eval_frames"})
                continue
            frames_eval = frames[eval_mask]
            poses_eval = poses[eval_mask]
            h9_subset = np.stack([h9_lookup[int(frame)] for frame in frames_eval], axis=0)
            aligned, metrics = _align_metrics(frames_eval, poses_eval, gt_poses, gt_pos)
            h9_aligned, h9_metrics = _align_metrics(frames_eval, h9_subset, gt_poses, gt_pos)
            raw_max_abs, raw_max_trans, timestamp_equal = _raw_diff(frames_eval, poses_eval, h9_lookup)
            seg_200 = _segment_ate(frames_eval, aligned, gt_pos, 200, 300)
            h9_200 = _segment_ate(frames_eval, h9_aligned, gt_pos, 200, 300)
            seg_400 = _segment_ate(frames_eval, aligned, gt_pos, 400, 600)
            h9_400 = _segment_ate(frames_eval, h9_aligned, gt_pos, 400, 600)
            ate_delta = float(metrics["ATE_horizon"] - h9_metrics["ATE_horizon"])
            seg_200_delta = float(seg_200 - h9_200) if math.isfinite(seg_200) and math.isfinite(h9_200) else float("nan")
            seg_400_delta = float(seg_400 - h9_400) if math.isfinite(seg_400) and math.isfinite(h9_400) else float("nan")
            gate = (
                (
                    ate_delta <= float(args.ate_threshold)
                    or (math.isfinite(seg_200_delta) and seg_200_delta <= float(args.segment_threshold))
                )
                and (not math.isfinite(seg_400_delta) or seg_400_delta <= float(args.downstream_threshold))
            )
            debug = _last_debug(run_dir)
            sem = _semantic_summary(run_dir)
            rows.append({
                "chunk": int(chunk),
                "horizon": int(args.horizon),
                "candidate": candidate,
                "run_dir": str(run_dir),
                "wall_seconds": _runtime_sec(run_dir),
                "ATE_horizon": float(metrics["ATE_horizon"]),
                "H9_ATE_horizon": float(h9_metrics["ATE_horizon"]),
                "ATE_delta_vs_H9": ate_delta,
                "intersection_200_300_ATE": seg_200,
                "H9_intersection_200_300_ATE": h9_200,
                "intersection_200_300_delta_vs_H9": seg_200_delta,
                "intersection_400_600_ATE": seg_400,
                "H9_intersection_400_600_ATE": h9_400,
                "intersection_400_600_delta_vs_H9": seg_400_delta,
                "raw_pose_max_abs_diff_vs_H9": raw_max_abs,
                "raw_translation_max_diff_vs_H9": raw_max_trans,
                "timestamp_equal": bool(timestamp_equal),
                "gate_pass": bool(gate),
                "semantic_rows_available": sem.get("semantic_rows_available"),
                "fine_label_available": sem.get("fine_label_available"),
                "fine_label_count": sem.get("fine_label_count"),
                "prior_cue_source_effective": debug.get("prior_cue_source_effective"),
                "prior_mean_D_patch": debug.get("prior_mean_D_patch"),
                "prior_q90_D_patch": debug.get("prior_q90_D_patch"),
                "prior_fallback_rate": debug.get("prior_fallback_rate"),
                "prior_v31_semantic_recondition_applied": debug.get("prior_v31_semantic_recondition_applied"),
                "prior_v31_semantic_recondition_mode": debug.get("prior_v31_semantic_recondition_mode"),
                "prior_v31_semantic_label_field": debug.get("prior_v31_semantic_label_field"),
                "prior_v31_semantic_label_count": debug.get("prior_v31_semantic_label_count"),
                "prior_v31_semantic_label_fallback_ratio": debug.get("prior_v31_semantic_label_fallback_ratio"),
            })

    report_prefix = str(args.report_prefix)
    _write_csv(out_dir / f"{report_prefix}_effects.csv", rows)
    _write_json(out_dir / f"{report_prefix}_effects.json", rows)
    _write_csv(out_dir / f"{report_prefix}_missing_rows.csv", missing)

    by_chunk: List[Dict[str, object]] = []
    for chunk in chunks:
        subset = [r for r in rows if int(r["chunk"]) == int(chunk)]
        if not subset:
            continue
        best_ate = min(subset, key=lambda r: float(r["ATE_delta_vs_H9"]))
        finite_seg = [r for r in subset if math.isfinite(float(r["intersection_200_300_delta_vs_H9"]))]
        best_seg = min(finite_seg, key=lambda r: float(r["intersection_200_300_delta_vs_H9"])) if finite_seg else None
        by_chunk.append({
            "chunk": int(chunk),
            "rows": len(subset),
            "best_ATE_candidate": best_ate["candidate"],
            "best_ATE_delta_vs_H9": best_ate["ATE_delta_vs_H9"],
            "best_200_300_candidate": best_seg["candidate"] if best_seg else None,
            "best_200_300_delta_vs_H9": best_seg["intersection_200_300_delta_vs_H9"] if best_seg else None,
            "best_400_600_delta_for_best_ATE": best_ate["intersection_400_600_delta_vs_H9"],
            "gate_pass_candidates": [r["candidate"] for r in subset if r["gate_pass"]],
        })
    _write_csv(out_dir / f"{report_prefix}_by_chunk.csv", by_chunk)

    best_ate = min(rows, key=lambda r: float(r["ATE_delta_vs_H9"]), default=None)
    finite_seg_rows = [r for r in rows if math.isfinite(float(r["intersection_200_300_delta_vs_H9"]))]
    best_seg = min(finite_seg_rows, key=lambda r: float(r["intersection_200_300_delta_vs_H9"]), default=None)
    summary = {
        "rows": len(rows),
        "missing_rows": len(missing),
        "all_rows_done": len(rows) == len(chunks) * len(candidates) and not missing,
        "gate_ate_threshold": float(args.ate_threshold),
        "gate_segment_200_300_threshold": float(args.segment_threshold),
        "gate_downstream_400_600_threshold": float(args.downstream_threshold),
        "gate_pass": any(bool(r["gate_pass"]) for r in rows),
        "gate_pass_candidates": [
            {"chunk": r["chunk"], "candidate": r["candidate"]}
            for r in rows
            if bool(r["gate_pass"])
        ],
        "best_ATE_candidate": best_ate["candidate"] if best_ate else None,
        "best_ATE_chunk": best_ate["chunk"] if best_ate else None,
        "best_ATE_delta_vs_H9": best_ate["ATE_delta_vs_H9"] if best_ate else None,
        "best_200_300_candidate": best_seg["candidate"] if best_seg else None,
        "best_200_300_chunk": best_seg["chunk"] if best_seg else None,
        "best_200_300_delta_vs_H9": best_seg["intersection_200_300_delta_vs_H9"] if best_seg else None,
        "best_400_600_delta_for_best_ATE": best_ate["intersection_400_600_delta_vs_H9"] if best_ate else None,
        "by_chunk": by_chunk,
    }
    _write_json(out_dir / f"{report_prefix}_summary.json", summary)
    print(json.dumps(_clean_json(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
