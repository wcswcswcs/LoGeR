#!/usr/bin/env python3
"""Aggregate v30 diverse masklet causal-bank rollouts.

Unlike the older candidate-bank reports, v30 uses one run prefix per selected
masklet/action. This report scans the landed R2 rollout directories, joins them
with the planned intervention table, and compares every candidate against the
same clean H9 reference on the same frame intersection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_json(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pose_lookup(frames: np.ndarray, poses: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(frame): pose for frame, pose in zip(frames.astype(np.int64), poses)}


def _eval_window(chunk_id: int, horizon: int) -> Tuple[int, int]:
    start = CHUNK_START_FRAME[chunk_id]
    return start, start + 32 + (horizon - 1) * 29


def _runtime_sec(run_dir: Path) -> float:
    path = run_dir / "run_status.txt"
    if not path.exists():
        return float("nan")
    starts: List[datetime] = []
    dones: List[datetime] = []
    pattern = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<kind>START|DONE)\b")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _intervention_stats(run_dir: Path) -> Dict[str, object]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    selected = [r.get("v29c_masklet_intervention", {}) for r in rows]
    enabled = [r for r in selected if isinstance(r, dict) and r.get("enabled") is True]
    out: Dict[str, object] = {
        "runtime_intervention_enabled_chunks": len(enabled),
        "runtime_intervention_rows": len(selected),
        "runtime_selected_token_count_mean": float("nan"),
        "runtime_selected_patch_count_mean": float("nan"),
        "context_empty_source_events": 0,
        "swa_enabled_layers_sum": 0,
        "swa_source_gate_applied_sum": 0,
        "swa_source_replace_applied_sum": 0,
        "chunk_context_source_skip_applied_sum": 0,
    }
    token_counts = []
    patch_counts = []
    for item in enabled:
        try:
            token_counts.append(float(item.get("token_count", 0)))
            patch_counts.append(float(item.get("patch_count", 0)))
        except Exception:
            pass
    if token_counts:
        out["runtime_selected_token_count_mean"] = sum(token_counts) / len(token_counts)
    if patch_counts:
        out["runtime_selected_patch_count_mean"] = sum(patch_counts) / len(patch_counts)
    for row in rows:
        trace = row.get("control_trace", {})
        hooks = trace.get("hook_effect_summary", {}) if isinstance(trace, dict) else {}
        for hook in hooks.values():
            if not isinstance(hook, dict):
                continue
            out["context_empty_source_events"] = int(out["context_empty_source_events"]) + int(hook.get("num_context_empty_source_events", 0) or 0)
        swa = hooks.get("swa_read", {}) if isinstance(hooks, dict) else {}
        if isinstance(swa, dict):
            out["swa_enabled_layers_sum"] = int(out["swa_enabled_layers_sum"]) + int(swa.get("num_enabled_layers", 0) or 0)
            out["swa_source_gate_applied_sum"] = int(out["swa_source_gate_applied_sum"]) + int(swa.get("num_source_gate_applied", 0) or 0)
            out["swa_source_replace_applied_sum"] = int(out["swa_source_replace_applied_sum"]) + int(swa.get("num_swa_overlap_source_replace_applied", 0) or 0)
        chunk = hooks.get("chunk_attention", {}) if isinstance(hooks, dict) else {}
        if isinstance(chunk, dict):
            out["chunk_context_source_skip_applied_sum"] = int(out["chunk_context_source_skip_applied_sum"]) + int(chunk.get("num_context_source_skip_applied", 0) or 0)
    return out


def _run_done(run_dir: Path) -> bool:
    status = run_dir / "run_status.txt"
    return status.exists() and "DONE" in status.read_text(encoding="utf-8", errors="replace")


def _planned_key(row: Mapping[str, str], r2_prefix: str) -> str:
    prefix = row.get("run_prefix", "").replace("V30_H4_WAVE1_R1", r2_prefix)
    return f"{prefix}_{row.get('candidate_alias')}_chunk10_h10_globalgate_H9parent_SWKS3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--planned-interventions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--r2-prefix", default="V30_H4_WAVE1_R2")
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    args = parser.parse_args()

    rollout_root = Path(args.rollout_root)
    ref_run = Path(args.reference_run)
    planned = _read_csv(Path(args.planned_interventions))
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    h9_frames, h9_poses, _ = _load_tum_prediction(ref_run / "01.txt", gt_pos.shape[0])
    h9_lookup = _pose_lookup(h9_frames, h9_poses)

    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    eval_start, eval_end = _eval_window(args.chunk, args.horizon)

    for plan in planned:
        run_name = _planned_key(plan, args.r2_prefix)
        run_dir = rollout_root / run_name
        if not (run_dir / "01.txt").exists():
            missing.append({"run_name": run_name, **plan})
            continue
        frames, poses, _ = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
        frames = frames.astype(np.int64)
        rollout_frame_start = int(frames[0])
        rollout_frame_end_inclusive = int(frames[-1])
        eval_mask = (frames >= eval_start) & (frames < eval_end)
        if int(eval_mask.sum()) < 3:
            missing.append({"run_name": run_name, "reason": "too_few_eval_frames", **plan})
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
        stats = _intervention_stats(run_dir)
        ate_delta = metrics["ATE_horizon"] - h9_metrics["ATE_horizon"]
        seg_200_delta = seg_200 - h9_200 if math.isfinite(seg_200) and math.isfinite(h9_200) else float("nan")
        seg_400_delta = seg_400 - h9_400 if math.isfinite(seg_400) and math.isfinite(h9_400) else float("nan")
        oracle_pass = (
            ((math.isfinite(ate_delta) and ate_delta <= -3.0) or (math.isfinite(seg_200_delta) and seg_200_delta <= -5.0))
            and (not math.isfinite(seg_400_delta) or seg_400_delta <= 1.0)
            and int(stats["context_empty_source_events"]) == 0
        )
        rows.append({
            **plan,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "run_done": _run_done(run_dir),
            "runtime_sec": _runtime_sec(run_dir),
            "chunk_id": args.chunk,
            "horizon": args.horizon,
            "rollout_frame_start": rollout_frame_start,
            "rollout_frame_end_inclusive": rollout_frame_end_inclusive,
            "eval_frame_start": int(frames_eval[0]),
            "eval_frame_end_inclusive": int(frames_eval[-1]),
            "eval_frame_count": int(frames_eval.shape[0]),
            "ATE_horizon": metrics["ATE_horizon"],
            "ATE_H9_intersection": h9_metrics["ATE_horizon"],
            "ATE_delta_vs_H9": ate_delta,
            "intersection_200_300_ATE": seg_200,
            "intersection_200_300_H9_ATE": h9_200,
            "intersection_200_300_delta_vs_H9": seg_200_delta,
            "intersection_400_600_ATE": seg_400,
            "intersection_400_600_H9_ATE": h9_400,
            "intersection_400_600_delta_vs_H9": seg_400_delta,
            "raw_pose_max_abs_diff": raw_max_abs,
            "raw_trans_max_diff": raw_max_trans,
            "timestamp_mapping_equal": timestamp_equal,
            "oracle_gate_pass": oracle_pass,
            "diagnostic_only_short_rollout": True,
            "counts_as_deployable_online_success": False,
            **stats,
        })

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "causal_effects.csv", rows)
    _write_csv(out_dir / "missing_or_invalid_runs.csv", missing)
    by_path: List[Dict[str, object]] = []
    by_category: List[Dict[str, object]] = []
    for key_name, out_rows in [("intervention_path", by_path), ("selection_category", by_category)]:
        groups = sorted({str(row.get(key_name, "")) for row in rows})
        for group in groups:
            subset = [row for row in rows if str(row.get(key_name, "")) == group]
            best_ate = min(subset, key=lambda r: float(r["ATE_delta_vs_H9"]))
            best_seg = min(subset, key=lambda r: float(r["intersection_200_300_delta_vs_H9"]) if math.isfinite(float(r["intersection_200_300_delta_vs_H9"])) else float("inf"))
            out_rows.append({
                key_name: group,
                "rows": len(subset),
                "best_ATE_delta_vs_H9": best_ate["ATE_delta_vs_H9"],
                "best_ATE_run_name": best_ate["run_name"],
                "best_200_300_delta_vs_H9": best_seg["intersection_200_300_delta_vs_H9"],
                "best_200_300_run_name": best_seg["run_name"],
            })
    _write_csv(out_dir / "causal_effects_by_path.csv", by_path)
    _write_csv(out_dir / "causal_effects_by_category.csv", by_category)
    best_ate = min(rows, key=lambda r: float(r["ATE_delta_vs_H9"]), default=None)
    finite_seg_rows = [r for r in rows if math.isfinite(float(r["intersection_200_300_delta_vs_H9"]))]
    best_seg = min(finite_seg_rows, key=lambda r: float(r["intersection_200_300_delta_vs_H9"]), default=None)
    summary = {
        "rows_expected": len(planned),
        "rows_reported": len(rows),
        "missing_or_invalid_rows": len(missing),
        "all_rows_done": all(bool(row["run_done"]) for row in rows) and not missing,
        "oracle_gate_pass_candidates": sorted(str(row["run_name"]) for row in rows if row["oracle_gate_pass"]),
        "oracle_gate_pass": any(bool(row["oracle_gate_pass"]) for row in rows),
        "best_ATE_delta_vs_H9": best_ate["ATE_delta_vs_H9"] if best_ate else None,
        "best_ATE_run_name": best_ate["run_name"] if best_ate else None,
        "best_200_300_delta_vs_H9": best_seg["intersection_200_300_delta_vs_H9"] if best_seg else None,
        "best_200_300_run_name": best_seg["run_name"] if best_seg else None,
        "best_400_600_delta_for_best_ATE": best_ate["intersection_400_600_delta_vs_H9"] if best_ate else None,
        "context_empty_source_events_total": sum(int(row["context_empty_source_events"]) for row in rows),
        "selector_allowed": False,
        "full_online_validation_allowed": False,
        "counts_as_deployable_online_success": False,
    }
    _write_json(out_dir / "causal_bank_summary.json", summary)


if __name__ == "__main__":
    main()
