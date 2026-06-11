#!/usr/bin/env python3
"""Aggregate ACL2 v38 short-rollout durability evidence.

This script only reads landed rollout artifacts. It does not reconstruct
missing tensor-state, attention, pixel, or semantic overlay evidence.
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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


DEFAULT_GT = "/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt"
BASE_CANDIDATE = "V31_BASE_H9_REFERENCE"


def _parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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


def _write_md(path: Path, title: str, summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for key in [
        "rows",
        "missing_rows",
        "all_rows_done",
        "gate_pass",
        "best_ATE_candidate",
        "best_ATE_chunk",
        "best_ATE_delta_vs_base",
        "best_rolling_100f_candidate",
        "best_rolling_100f_chunk",
        "best_rolling_100f_best_delta",
        "best_downstream_400_600_delta_for_best_ATE",
    ]:
        if key in summary:
            lines.append(f"- `{key}`: `{_clean(summary[key])}`")
    lines.extend(["", "| Chunk | Candidate | ATE delta | Roll100 best | Roll100 high-error best | [200,300) delta | [400,600) delta | Gate |", "|---:|---|---:|---:|---:|---:|---:|---|"])
    for row in sorted(rows, key=lambda r: (int(r["chunk"]), str(r["candidate"]))):
        lines.append(
            "| {chunk} | `{candidate}` | {ate:.10f} | {r100:.10f} | {hr100:.10f} | {s200:.10f} | {s400:.10f} | `{gate}` |".format(
                chunk=int(row["chunk"]),
                candidate=row["candidate"],
                ate=float(row["ATE_delta_vs_base"]),
                r100=float(row.get("rolling_100f_best_delta_vs_base", float("nan"))),
                hr100=float(row.get("rolling_100f_high_error_best_delta_vs_base", float("nan"))),
                s200=float(row.get("intersection_200_300_delta_vs_base", float("nan"))),
                s400=float(row.get("intersection_400_600_delta_vs_base", float("nan"))),
                gate=bool(row.get("gate_pass", False)),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _status_done(run_dir: Path, run_name: str) -> bool:
    status = run_dir / "run_status.txt"
    if not status.exists():
        return False
    return f"DONE {run_name}" in status.read_text(encoding="utf-8", errors="replace")


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


def _eval_window(chunk: int, horizon: int) -> Tuple[int, int]:
    start = CHUNK_START_FRAME[int(chunk)]
    return start, start + 32 + (int(horizon) - 1) * 29


def _run_name(prefix: str, parent: str, candidate: str, chunk: int, horizon: int) -> str:
    return f"{prefix}_{parent}_{candidate}_chunk{chunk}_h{horizon}_globalgate_H9parent_SWKS3"


def _ate(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    if int(pos_a.shape[0]) < 3 or int(pos_b.shape[0]) < 3:
        return float("nan")
    diff = pos_a - pos_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _as_positions(poses_or_pos: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses_or_pos)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr[:, :3, 3]
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3]
    raise ValueError(f"Unsupported pose/position array shape for rolling ATE: {arr.shape}")


def _rolling_windows(
    frames: np.ndarray,
    pred_poses_or_pos: np.ndarray,
    gt_pos: np.ndarray,
    width: int,
) -> Dict[int, float]:
    if int(frames.size) < 3:
        return {}
    pred_pos = _as_positions(pred_poses_or_pos)
    frame_min = int(frames.min())
    frame_max = int(frames.max())
    out: Dict[int, float] = {}
    for start in range(frame_min, frame_max - int(width) + 2):
        end = start + int(width)
        mask = (frames >= start) & (frames < end)
        if int(mask.sum()) < 3:
            continue
        gt_subset = gt_pos[frames[mask].astype(np.int64)]
        out[start] = _ate(pred_pos[mask], gt_subset)
    return out


def _rolling_stats(
    candidate: Dict[int, float],
    base: Dict[int, float],
    *,
    high_error_quantile: float = 0.90,
) -> Dict[str, Any]:
    starts = sorted(set(candidate) & set(base))
    deltas = [float(candidate[s] - base[s]) for s in starts if math.isfinite(candidate[s]) and math.isfinite(base[s])]
    cand_vals = [float(candidate[s]) for s in starts if math.isfinite(candidate[s])]
    base_vals = [float(base[s]) for s in starts if math.isfinite(base[s])]
    if not deltas:
        return {
            "count": 0,
            "best_delta": float("nan"),
            "worst_delta": float("nan"),
            "mean_delta": float("nan"),
            "p90_delta": float("nan"),
            "candidate_worst": float("nan"),
            "base_worst": float("nan"),
            "worst_ate_delta": float("nan"),
            "high_error_best_delta": float("nan"),
            "high_error_count": 0,
        }
    base_threshold = float(np.quantile(np.asarray(base_vals, dtype=np.float64), high_error_quantile)) if base_vals else float("nan")
    high_deltas = [
        float(candidate[s] - base[s])
        for s in starts
        if math.isfinite(base_threshold) and math.isfinite(base[s]) and base[s] >= base_threshold
    ]
    return {
        "count": len(deltas),
        "best_delta": float(min(deltas)),
        "worst_delta": float(max(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "p90_delta": float(np.quantile(np.asarray(deltas, dtype=np.float64), 0.90)),
        "candidate_worst": float(max(cand_vals)) if cand_vals else float("nan"),
        "base_worst": float(max(base_vals)) if base_vals else float("nan"),
        "worst_ate_delta": float(max(cand_vals) - max(base_vals)) if cand_vals and base_vals else float("nan"),
        "high_error_best_delta": float(min(high_deltas)) if high_deltas else float("nan"),
        "high_error_count": len(high_deltas),
    }


def _last_hmc(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    return rows[-1] if rows else {}


def _hook_summary(run_dir: Path) -> Dict[str, Any]:
    hmc = _last_hmc(run_dir)
    summary = hmc.get("hook_effect_summary")
    if isinstance(summary, dict):
        return summary
    trace = hmc.get("control_trace")
    if isinstance(trace, dict) and isinstance(trace.get("hook_effect_summary"), dict):
        return trace["hook_effect_summary"]
    return {}


def _path_metric(summary: Dict[str, Any], path: str, key: str) -> Any:
    value = summary.get(path)
    if isinstance(value, dict):
        return value.get(key)
    return None


def _gate(row: Dict[str, Any], mode: str, args: argparse.Namespace) -> bool:
    ate_delta = float(row.get("ATE_delta_vs_base", float("nan")))
    roll100_best = float(row.get("rolling_100f_best_delta_vs_base", float("nan")))
    roll100_high = float(row.get("rolling_100f_high_error_best_delta_vs_base", float("nan")))
    seg400 = float(row.get("intersection_400_600_delta_vs_base", float("nan")))
    roll200_worst = float(row.get("rolling_200f_worst_delta_vs_base", float("nan")))
    if mode == "h15":
        signal = (
            (math.isfinite(ate_delta) and ate_delta <= float(args.h15_ate_threshold))
            or (math.isfinite(roll100_best) and roll100_best <= float(args.h15_rolling100_threshold))
            or (math.isfinite(roll100_high) and roll100_high <= float(args.h15_high_error_threshold))
        )
    else:
        signal = (
            (math.isfinite(ate_delta) and ate_delta <= float(args.h10_ate_threshold))
            or (math.isfinite(roll100_best) and roll100_best <= float(args.h10_rolling100_threshold))
            or (math.isfinite(roll100_high) and roll100_high <= float(args.h10_high_error_threshold))
        )
    downstream = True
    if math.isfinite(seg400):
        downstream = downstream and seg400 <= float(args.downstream_400_600_threshold)
    if math.isfinite(roll200_worst):
        downstream = downstream and roll200_worst <= float(args.downstream_rolling200_threshold)
    return bool(signal and downstream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--parents", default="H9,C9")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--mode", choices=("h10", "h15"), default="h10")
    parser.add_argument("--base-candidate", default=BASE_CANDIDATE)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--h10-ate-threshold", type=float, default=-1.5)
    parser.add_argument("--h10-rolling100-threshold", type=float, default=-3.0)
    parser.add_argument("--h10-high-error-threshold", type=float, default=-5.0)
    parser.add_argument("--h15-ate-threshold", type=float, default=-2.0)
    parser.add_argument("--h15-rolling100-threshold", type=float, default=-4.0)
    parser.add_argument("--h15-high-error-threshold", type=float, default=-5.0)
    parser.add_argument("--downstream-400-600-threshold", type=float, default=1.0)
    parser.add_argument("--downstream-rolling200-threshold", type=float, default=1.0)
    args = parser.parse_args()

    parents = _parse_csv_strs(args.parents)
    chunks = _parse_csv_ints(args.chunks)
    candidates = _parse_csv_strs(args.candidates)
    rollout_root = args.rollout_root
    gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    del gt_frames

    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for parent in parents:
        for chunk in chunks:
            base_name = _run_name(args.run_prefix, parent, args.base_candidate, chunk, args.horizon)
            base_dir = rollout_root / base_name
            if not _status_done(base_dir, base_name) or not (base_dir / "01.txt").exists():
                missing.append({
                    "parent": parent,
                    "chunk": chunk,
                    "candidate": args.base_candidate,
                    "run_name": base_name,
                    "run_dir": str(base_dir),
                    "reason": "missing_or_not_done_reference",
                })
                continue
            base_frames, base_poses, _ = _load_tum_prediction(base_dir / "01.txt", gt_pos.shape[0])
            base_lookup = _pose_lookup(base_frames, base_poses)
            eval_start, eval_end = _eval_window(chunk, args.horizon)

            for candidate in candidates:
                run_name = _run_name(args.run_prefix, parent, candidate, chunk, args.horizon)
                run_dir = rollout_root / run_name
                if not _status_done(run_dir, run_name) or not (run_dir / "01.txt").exists():
                    missing.append({
                        "parent": parent,
                        "chunk": chunk,
                        "candidate": candidate,
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "reason": "missing_or_not_done_prediction",
                    })
                    continue
                frames, poses, _ = _load_tum_prediction(run_dir / "01.txt", gt_pos.shape[0])
                frames = frames.astype(np.int64)
                eval_mask = (frames >= eval_start) & (frames < eval_end)
                if int(eval_mask.sum()) < 3:
                    missing.append({
                        "parent": parent,
                        "chunk": chunk,
                        "candidate": candidate,
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "reason": "too_few_eval_frames",
                    })
                    continue
                frames_eval = frames[eval_mask]
                poses_eval = poses[eval_mask]
                try:
                    base_subset = np.stack([base_lookup[int(frame)] for frame in frames_eval], axis=0)
                except KeyError:
                    missing.append({
                        "parent": parent,
                        "chunk": chunk,
                        "candidate": candidate,
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "reason": "reference_missing_matching_frames",
                    })
                    continue
                aligned, metrics = _align_metrics(frames_eval, poses_eval, gt_poses, gt_pos)
                base_aligned, base_metrics = _align_metrics(frames_eval, base_subset, gt_poses, gt_pos)
                raw_max_abs, raw_max_trans, timestamp_equal = _raw_diff(frames_eval, poses_eval, base_lookup)
                seg_200 = _segment_ate(frames_eval, aligned, gt_pos, 200, 300)
                base_200 = _segment_ate(frames_eval, base_aligned, gt_pos, 200, 300)
                seg_400 = _segment_ate(frames_eval, aligned, gt_pos, 400, 600)
                base_400 = _segment_ate(frames_eval, base_aligned, gt_pos, 400, 600)
                row: Dict[str, Any] = {
                    "parent": parent,
                    "chunk": int(chunk),
                    "horizon": int(args.horizon),
                    "candidate": candidate,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "wall_seconds": _runtime_sec(run_dir),
                    "eval_start_frame": int(eval_start),
                    "eval_end_frame_exclusive": int(eval_end),
                    "eval_frames": int(frames_eval.size),
                    "ATE_horizon": float(metrics["ATE_horizon"]),
                    "base_ATE_horizon": float(base_metrics["ATE_horizon"]),
                    "ATE_delta_vs_base": float(metrics["ATE_horizon"] - base_metrics["ATE_horizon"]),
                    "intersection_200_300_ATE": seg_200,
                    "base_intersection_200_300_ATE": base_200,
                    "intersection_200_300_delta_vs_base": (
                        float(seg_200 - base_200) if math.isfinite(seg_200) and math.isfinite(base_200) else float("nan")
                    ),
                    "intersection_400_600_ATE": seg_400,
                    "base_intersection_400_600_ATE": base_400,
                    "intersection_400_600_delta_vs_base": (
                        float(seg_400 - base_400) if math.isfinite(seg_400) and math.isfinite(base_400) else float("nan")
                    ),
                    "raw_pose_max_abs_diff_vs_base": raw_max_abs,
                    "raw_translation_max_diff_vs_base": raw_max_trans,
                    "timestamp_equal": bool(timestamp_equal),
                }
                for width in (50, 100, 200):
                    cand_roll = _rolling_windows(frames_eval, aligned, gt_pos, width)
                    base_roll = _rolling_windows(frames_eval, base_aligned, gt_pos, width)
                    stats = _rolling_stats(cand_roll, base_roll)
                    for key, value in stats.items():
                        row[f"rolling_{width}f_{key}_vs_base"] = value
                hook = _hook_summary(run_dir)
                for path in ("frame_attention", "chunk_attention", "swa_read", "ttt_apply"):
                    for key in (
                        "num_context_source_skip_applied",
                        "max_context_source_skip_tokens",
                        "mean_context_source_keep_ratio",
                        "num_source_gate_applied",
                        "num_swa_overlap_source_gate_applied",
                        "mean_swa_overlap_source_gate",
                        "max_swa_overlap_source_gate_delta",
                        "action_delta_mean_rel_diff",
                        "state_mean_rel_diff",
                        "w0_mean_rel_diff",
                    ):
                        row[f"{path}_{key}"] = _path_metric(hook, path, key)
                row["gate_pass"] = _gate(row, args.mode, args)
                rows.append(row)

    report_prefix = args.report_prefix
    out_dir = args.out_dir
    _write_csv(out_dir / f"{report_prefix}_effects.csv", rows)
    _write_json(out_dir / f"{report_prefix}_effects.json", rows)
    _write_csv(out_dir / f"{report_prefix}_missing_rows.csv", missing)

    by_parent: List[Dict[str, Any]] = []
    for parent in parents:
        subset = [r for r in rows if r["parent"] == parent]
        if not subset:
            continue
        best_ate = min(subset, key=lambda r: float(r["ATE_delta_vs_base"]))
        best_roll = min(subset, key=lambda r: float(r.get("rolling_100f_best_delta_vs_base", float("inf"))))
        finite_200 = [r for r in subset if math.isfinite(float(r.get("intersection_200_300_delta_vs_base", float("nan"))))]
        best_200 = min(finite_200, key=lambda r: float(r["intersection_200_300_delta_vs_base"])) if finite_200 else None
        by_parent.append({
            "parent": parent,
            "rows": len(subset),
            "gate_pass": any(bool(r.get("gate_pass")) for r in subset),
            "gate_pass_candidates": [
                {"chunk": r["chunk"], "candidate": r["candidate"]}
                for r in subset
                if bool(r.get("gate_pass"))
            ],
            "best_ATE_candidate": best_ate["candidate"],
            "best_ATE_chunk": best_ate["chunk"],
            "best_ATE_delta_vs_base": best_ate["ATE_delta_vs_base"],
            "best_rolling_100f_candidate": best_roll["candidate"],
            "best_rolling_100f_chunk": best_roll["chunk"],
            "best_rolling_100f_best_delta": best_roll.get("rolling_100f_best_delta_vs_base"),
            "best_200_300_candidate": best_200["candidate"] if best_200 else None,
            "best_200_300_chunk": best_200["chunk"] if best_200 else None,
            "best_200_300_delta_vs_base": best_200["intersection_200_300_delta_vs_base"] if best_200 else None,
            "best_downstream_400_600_delta_for_best_ATE": best_ate["intersection_400_600_delta_vs_base"],
        })
    _write_csv(out_dir / f"{report_prefix}_by_parent.csv", by_parent)

    best_ate = min(rows, key=lambda r: float(r["ATE_delta_vs_base"]), default=None)
    best_roll = min(rows, key=lambda r: float(r.get("rolling_100f_best_delta_vs_base", float("inf"))), default=None)
    summary = {
        "rows": len(rows),
        "missing_rows": len(missing),
        "all_rows_done": len(rows) == len(parents) * len(chunks) * len(candidates) and not missing,
        "parents": parents,
        "chunks": chunks,
        "horizon": int(args.horizon),
        "mode": args.mode,
        "gate_pass": any(bool(r.get("gate_pass")) for r in rows),
        "gate_pass_candidates": [
            {"parent": r["parent"], "chunk": r["chunk"], "candidate": r["candidate"]}
            for r in rows
            if bool(r.get("gate_pass"))
        ],
        "best_ATE_candidate": best_ate["candidate"] if best_ate else None,
        "best_ATE_parent": best_ate["parent"] if best_ate else None,
        "best_ATE_chunk": best_ate["chunk"] if best_ate else None,
        "best_ATE_delta_vs_base": best_ate["ATE_delta_vs_base"] if best_ate else None,
        "best_rolling_100f_candidate": best_roll["candidate"] if best_roll else None,
        "best_rolling_100f_parent": best_roll["parent"] if best_roll else None,
        "best_rolling_100f_chunk": best_roll["chunk"] if best_roll else None,
        "best_rolling_100f_best_delta": best_roll.get("rolling_100f_best_delta_vs_base") if best_roll else None,
        "best_downstream_400_600_delta_for_best_ATE": best_ate["intersection_400_600_delta_vs_base"] if best_ate else None,
        "by_parent": by_parent,
        "thresholds": {
            "h10_ate": args.h10_ate_threshold,
            "h10_rolling100": args.h10_rolling100_threshold,
            "h10_high_error": args.h10_high_error_threshold,
            "h15_ate": args.h15_ate_threshold,
            "h15_rolling100": args.h15_rolling100_threshold,
            "h15_high_error": args.h15_high_error_threshold,
            "downstream_400_600": args.downstream_400_600_threshold,
            "downstream_rolling200": args.downstream_rolling200_threshold,
        },
    }
    _write_json(out_dir / f"{report_prefix}_summary.json", summary)
    _write_md(out_dir / f"{report_prefix}_report.md", f"{report_prefix} durability report", summary, rows)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
