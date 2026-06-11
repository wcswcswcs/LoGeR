#!/usr/bin/env python3
"""Report ACL2 v32 full-online transfer and runtime activation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _resolve_run(root: Path, value: str) -> Path:
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return root / path


def _parse_runs(root: Path, text: str) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise SystemExit(f"Bad --runs item: {item}")
        name, run_path = item.split("=", 1)
        entries.append((name.strip(), _resolve_run(root, run_path)))
    return entries


def _activation_stats(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "hmc_state_hash.jsonl"
    if not path.exists():
        return {}
    rows = 0
    active_rows = 0
    inactive_rows = 0
    active_chunks: List[int] = []
    high_mass: List[float] = []
    trigger_metric: List[float] = []
    gate_modes: Dict[str, int] = {}
    gate_reasons: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        mode = item.get("prior_v32_semantic_cue_gate_mode")
        if mode is None:
            continue
        rows += 1
        gate_modes[str(mode)] = gate_modes.get(str(mode), 0) + 1
        reason = item.get("prior_v32_semantic_cue_gate_reason")
        gate_reasons[str(reason)] = gate_reasons.get(str(reason), 0) + 1
        active = bool(item.get("prior_v32_semantic_cue_active"))
        if active:
            active_rows += 1
            chunk = item.get("chunk_idx", item.get("prior_v32_semantic_cue_chunk_idx"))
            if chunk is not None:
                active_chunks.append(int(chunk))
        else:
            inactive_rows += 1
        mass = item.get("prior_v32_semantic_z_high_mass")
        if mass is not None:
            high_mass.append(float(mass))
        metric = item.get("prior_v32_semantic_trigger_metric")
        if metric is not None:
            trigger_metric.append(float(metric))
    return {
        "v32_gate_rows": rows,
        "v32_active_rows": active_rows,
        "v32_inactive_rows": inactive_rows,
        "v32_active_chunks": sorted(set(active_chunks)),
        "v32_gate_modes": gate_modes,
        "v32_gate_reasons": gate_reasons,
        "v32_semantic_z_high_mass_mean": float(np.mean(high_mass)) if high_mass else None,
        "v32_semantic_z_high_mass_p90": float(np.quantile(high_mass, 0.9)) if high_mass else None,
        "v32_trigger_metric_mean": float(np.mean(trigger_metric)) if trigger_metric else None,
        "v32_trigger_metric_p90": float(np.quantile(trigger_metric, 0.9)) if trigger_metric else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--runs", required=True, help="Comma-separated NAME=RUN_DIR_BASENAME_OR_ABS_PATH entries")
    parser.add_argument("--reference-name", default="")
    parser.add_argument("--c9-reference-ate", type=float, default=33.7629421029)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-ate", type=float, default=30.0)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    args = parser.parse_args()

    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    entries = _parse_runs(args.rollout_root, args.runs)

    ref_lookup: Optional[Dict[int, np.ndarray]] = None
    ref_metrics_full: Optional[float] = None
    ref_seg_200: Optional[float] = None
    ref_seg_400: Optional[float] = None
    if args.reference_name:
        for name, run_dir in entries:
            if name == args.reference_name and (run_dir / "01.txt").exists():
                frames, poses, _ = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
                aligned, metrics = _align_metrics(frames.astype(np.int64), poses, gt_poses, gt_pos)
                ref_lookup = _pose_lookup(frames, poses)
                ref_metrics_full = float(metrics["ATE_horizon"])
                ref_seg_200 = _segment_ate(frames.astype(np.int64), aligned, gt_pos, 200, 300)
                ref_seg_400 = _segment_ate(frames.astype(np.int64), aligned, gt_pos, 400, 600)
                break

    rows: List[Dict[str, object]] = []
    for name, run_dir in entries:
        pred_path = run_dir / "01.txt"
        row: Dict[str, object] = {"name": name, "run_dir": str(run_dir)}
        if not pred_path.exists():
            row["status"] = "missing_prediction"
            rows.append(row)
            continue
        frames, poses, _ = _load_tum_prediction(pred_path, gt_pos.shape[0])
        aligned, metrics = _align_metrics(frames.astype(np.int64), poses, gt_poses, gt_pos)
        seg_200 = _segment_ate(frames.astype(np.int64), aligned, gt_pos, 200, 300)
        seg_400 = _segment_ate(frames.astype(np.int64), aligned, gt_pos, 400, 600)
        row.update({
            "status": "done",
            "frames": int(frames.shape[0]),
            "wall_seconds": _runtime_sec(run_dir),
            "ATE_full": float(metrics["ATE_horizon"]),
            "Rot_full": float(metrics["Rot_horizon"]),
            "FinalErr_full": float(metrics["FinalErr_horizon"]),
            "segment_200_300_ATE": seg_200,
            "segment_400_600_ATE": seg_400,
            "target30_pass": float(metrics["ATE_horizon"]) <= float(args.target_ate),
            "delta_vs_C9_reference_ATE": float(metrics["ATE_horizon"]) - float(args.c9_reference_ate),
        })
        if ref_lookup is not None and name != args.reference_name:
            ref_subset = np.stack([ref_lookup[int(frame)] for frame in frames.astype(np.int64)], axis=0)
            ref_aligned, ref_metrics = _align_metrics(frames.astype(np.int64), ref_subset, gt_poses, gt_pos)
            ref_seg_200_run = _segment_ate(frames.astype(np.int64), ref_aligned, gt_pos, 200, 300)
            ref_seg_400_run = _segment_ate(frames.astype(np.int64), ref_aligned, gt_pos, 400, 600)
            raw_max_abs, raw_max_trans, timestamp_equal = _raw_diff(frames.astype(np.int64), poses, ref_lookup)
            row.update({
                "reference_name": args.reference_name,
                "ATE_delta_vs_reference": float(metrics["ATE_horizon"] - ref_metrics["ATE_horizon"]),
                "segment_200_300_delta_vs_reference": float(seg_200 - ref_seg_200_run),
                "segment_400_600_delta_vs_reference": float(seg_400 - ref_seg_400_run),
                "raw_pose_max_abs_diff_vs_reference": raw_max_abs,
                "raw_translation_max_diff_vs_reference": raw_max_trans,
                "timestamp_equal_reference": bool(timestamp_equal),
            })
        elif name == args.reference_name and ref_metrics_full is not None:
            row.update({
                "ATE_delta_vs_reference": 0.0,
                "segment_200_300_delta_vs_reference": 0.0 if ref_seg_200 is not None else None,
                "segment_400_600_delta_vs_reference": 0.0 if ref_seg_400 is not None else None,
            })
        row.update(_activation_stats(run_dir))
        rows.append(row)

    _write_csv(args.out_dir / "v32_full_metrics.csv", rows)
    _write_json(args.out_dir / "v32_full_metrics.json", rows)
    candidates = [r for r in rows if r.get("status") == "done" and r.get("name") != args.reference_name]
    best = min(candidates, key=lambda r: float(r["ATE_full"]), default=None)
    best_vs_c9 = min(candidates, key=lambda r: float(r["delta_vs_C9_reference_ATE"]), default=None)
    summary = {
        "rows": len(rows),
        "done_rows": sum(1 for r in rows if r.get("status") == "done"),
        "target_ate": float(args.target_ate),
        "c9_reference_ate": float(args.c9_reference_ate),
        "reference_name": args.reference_name or None,
        "best_candidate": best.get("name") if best else None,
        "best_candidate_ATE_full": best.get("ATE_full") if best else None,
        "best_vs_c9_candidate": best_vs_c9.get("name") if best_vs_c9 else None,
        "best_vs_c9_delta": best_vs_c9.get("delta_vs_C9_reference_ATE") if best_vs_c9 else None,
        "target30_pass": bool(best and float(best["ATE_full"]) <= float(args.target_ate)),
        "counts_as_deployable_online": bool(best and float(best["ATE_full"]) <= float(args.target_ate)),
        "no_gt_runtime_action": True,
        "no_offline_trajectory_rewrite": True,
        "fixed_chunk_activation_is_diagnostic_only": True,
    }
    _write_json(args.out_dir / "v32_full_summary.json", summary)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
