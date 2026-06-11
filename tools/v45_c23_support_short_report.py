#!/usr/bin/env python3
"""Report v45 C23 support short-fork diagnostics without treating them as full success."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

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

CHUNK_START_FRAME = {6: 174, 10: 290, 16: 464}
RUN_RE = re.compile(
    r"^V45_P3SHORT_(?P<parent>C9|C9CLEAN)_(?P<candidate>S\d_.+)_CH(?P<chunk>\d+)_H(?P<horizon>\d+)_READONLY$"
)


def _float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
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


def _json_clean(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_done(run_dir: Path) -> bool:
    status = run_dir / "run_status.txt"
    return status.exists() and "DONE" in status.read_text(encoding="utf-8", errors="replace")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _eval_window(chunk_id: int, horizon: int) -> Tuple[int, int]:
    start = CHUNK_START_FRAME[chunk_id]
    return start, start + 32 + (horizon - 1) * 29


def _align_metrics(frames: np.ndarray, raw_poses: np.ndarray, gt_poses: np.ndarray, gt_pos: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    frames = frames.astype(np.int64)
    raw_pos = raw_poses[:, :3, 3]
    matched_gt_pos = gt_pos[frames]
    scale, rot, trans = _umeyama_sim3(raw_pos, matched_gt_pos, with_scale=True)
    aligned = _apply_alignment(raw_poses, scale, rot, trans)
    err_norm = np.linalg.norm(aligned[:, :3, 3] - matched_gt_pos, axis=1)
    yaw_err = _angle_diff_deg(_yaw_from_pose(aligned), _yaw_from_pose(gt_poses[frames]))
    return aligned, {
        "ATE_h10": _rmse(err_norm),
        "Rot_h10": _rmse(yaw_err),
        "FinalErr_h10": float(err_norm[-1]) if err_norm.size else float("nan"),
        "alignment_scale": float(scale),
    }


def _segment_ate(frames: np.ndarray, aligned: np.ndarray, gt_pos: np.ndarray, start: int, end: int) -> float:
    mask = (frames >= start) & (frames < end)
    if int(mask.sum()) < 3:
        return float("nan")
    err = aligned[mask, :3, 3] - gt_pos[frames[mask]]
    return _rmse(np.linalg.norm(err, axis=1))


def _rolling_stats(frames: np.ndarray, aligned: np.ndarray, gt_pos: np.ndarray, window: int = 100) -> Dict[str, float]:
    if frames.size < window:
        return {"rolling100_mean": float("nan"), "rolling100_p90": float("nan"), "rolling100_worst": float("nan")}
    err = np.linalg.norm(aligned[:, :3, 3] - gt_pos[frames], axis=1)
    vals = np.asarray([_rmse(err[i : i + window]) for i in range(0, err.size - window + 1)], dtype=float)
    return {
        "rolling100_mean": float(np.mean(vals)),
        "rolling100_p90": float(np.percentile(vals, 90)),
        "rolling100_worst": float(np.max(vals)),
    }


def _load_eval(run_dir: Path, gt_poses: np.ndarray, gt_pos: np.ndarray, *, chunk_id: int, horizon: int) -> Optional[Dict[str, object]]:
    traj = run_dir / "01.txt"
    if not traj.exists():
        return None
    frames, poses, _ = _load_tum_prediction(traj, gt_pos.shape[0])
    frames = frames.astype(np.int64)
    eval_start, eval_end = _eval_window(chunk_id, horizon)
    mask = (frames >= eval_start) & (frames < eval_end)
    if int(mask.sum()) < 3:
        return None
    frames = frames[mask]
    poses = poses[mask]
    aligned, metrics = _align_metrics(frames, poses, gt_poses, gt_pos)
    metrics.update(_rolling_stats(frames, aligned, gt_pos))
    metrics.update({
        "segment_200_300_ATE": _segment_ate(frames, aligned, gt_pos, 200, 300),
        "segment_400_600_ATE": _segment_ate(frames, aligned, gt_pos, 400, 600),
        "eval_frame_start": int(frames[0]),
        "eval_frame_end_inclusive": int(frames[-1]),
        "eval_frame_count": int(frames.size),
    })
    return {"frames": frames, "aligned": aligned, "metrics": metrics}


def _discover(rollout_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in sorted(p for p in rollout_root.iterdir() if p.is_dir()):
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        rows.append({
            "parent": m.group("parent"),
            "candidate": m.group("candidate"),
            "chunk_id": int(m.group("chunk")),
            "horizon": int(m.group("horizon")),
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gt", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt", type=Path)
    args = parser.parse_args()

    gt_frames, gt_poses, gt_pos = _load_kitti_gt(args.gt)
    if gt_frames.size != gt_pos.shape[0]:
        raise ValueError("Unexpected KITTI GT layout")

    discovered = _discover(args.rollout_root)
    by_key = {(r["parent"], r["chunk_id"], r["horizon"], r["candidate"]): r for r in discovered}
    rows: List[Dict[str, object]] = []
    for meta in discovered:
        chunk_id = int(meta["chunk_id"])
        horizon = int(meta["horizon"])
        parent = str(meta["parent"])
        ref_meta = by_key.get((parent, chunk_id, horizon, "S0_C23_PAST"))
        if ref_meta is None:
            continue
        run_eval = _load_eval(Path(str(meta["run_dir"])), gt_poses, gt_pos, chunk_id=chunk_id, horizon=horizon)
        ref_eval = _load_eval(Path(str(ref_meta["run_dir"])), gt_poses, gt_pos, chunk_id=chunk_id, horizon=horizon)
        if run_eval is None or ref_eval is None:
            continue
        metrics = dict(run_eval["metrics"])  # type: ignore[arg-type]
        ref_metrics = dict(ref_eval["metrics"])  # type: ignore[arg-type]
        row: Dict[str, object] = {
            **meta,
            "run_done": _run_done(Path(str(meta["run_dir"]))),
            "diagnostic_only_short_rollout": True,
            "counts_as_full_online_success": False,
            "hmc_rows": _line_count(Path(str(meta["run_dir"])) / "hmc_state_hash.jsonl"),
        }
        for key, value in metrics.items():
            row[key] = value
            ref_value = _float(ref_metrics.get(key))
            cur_value = _float(value)
            if ref_value is not None and cur_value is not None and key.endswith(("_ATE", "_h10", "_mean", "_p90", "_worst")):
                row[f"{key}_delta_vs_S0"] = cur_value - ref_value
        rows.append(row)

    _write_csv(args.out_dir / "v45_c23_support_short_rows.csv", rows)
    candidate_rows = [r for r in rows if r.get("candidate") != "S0_C23_PAST" and r.get("run_done") is True]
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in candidate_rows:
        grouped.setdefault((str(row["parent"]), str(row["candidate"])), []).append(row)
    agg: List[Dict[str, object]] = []
    for (parent, candidate), group in sorted(grouped.items()):
        deltas = [_float(r.get("ATE_h10_delta_vs_S0")) for r in group]
        deltas = [v for v in deltas if v is not None]
        seg = [_float(r.get("segment_200_300_ATE_delta_vs_S0")) for r in group]
        seg = [v for v in seg if v is not None]
        roll = [_float(r.get("rolling100_mean_delta_vs_S0")) for r in group]
        roll = [v for v in roll if v is not None]
        down = [_float(r.get("segment_400_600_ATE_delta_vs_S0")) for r in group]
        down = [v for v in down if v is not None]
        agg.append({
            "parent": parent,
            "candidate": candidate,
            "done_chunks": len(group),
            "mean_ATE_h10_delta_vs_S0": sum(deltas) / len(deltas) if deltas else None,
            "best_ATE_h10_delta_vs_S0": min(deltas) if deltas else None,
            "best_segment_200_300_delta_vs_S0": min(seg) if seg else None,
            "best_rolling100_mean_delta_vs_S0": min(roll) if roll else None,
            "max_segment_400_600_delta_vs_S0": max(down) if down else None,
        })
    _write_csv(args.out_dir / "v45_c23_support_short_aggregate.csv", agg)

    def pass_row(row: Mapping[str, object]) -> bool:
        ate = _float(row.get("ATE_h10_delta_vs_S0"))
        roll = _float(row.get("rolling100_mean_delta_vs_S0"))
        seg = _float(row.get("segment_200_300_ATE_delta_vs_S0"))
        down = _float(row.get("segment_400_600_ATE_delta_vs_S0"))
        return bool(
            (ate is not None and ate <= -1.0)
            or (roll is not None and roll <= -3.0)
            or (seg is not None and seg <= -5.0 and (down is None or down <= 1.0))
        )

    passing = [r for r in candidate_rows if pass_row(r)]
    top2 = sorted(
        [r for r in agg if _float(r.get("mean_ATE_h10_delta_vs_S0")) is not None],
        key=lambda r: _float(r.get("mean_ATE_h10_delta_vs_S0")) or float("inf"),
    )[:2]
    summary = {
        "rows": len(rows),
        "done_candidate_rows": len(candidate_rows),
        "short_gate_pass": bool(passing),
        "passing_rows": len(passing),
        "top2_by_mean_ATE_delta": top2,
        "diagnostic_only_short_rollout": True,
        "counts_as_full_online_success": False,
        "gate_rule": "h10 ATE delta <= -1.0m, or rolling100 mean delta <= -3m, or [200,300) delta <= -5m with [400,600) regression <= +1m",
    }
    _write_json(args.out_dir / "v45_c23_support_short_summary.json", summary)

    lines = ["# v45 C23 Support Short-Fork Report", "", "Short rows are diagnostic only and never count as full-online success.", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Aggregate", "", "| Parent | Candidate | Done Chunks | Mean h10 Delta | Best h10 Delta | Best [200,300) Delta | Best Rolling100 Delta | Max [400,600) Delta |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in agg:
        lines.append(
            f"| {row.get('parent')} | `{row.get('candidate')}` | {row.get('done_chunks')} | "
            f"{_float(row.get('mean_ATE_h10_delta_vs_S0')) if _float(row.get('mean_ATE_h10_delta_vs_S0')) is not None else ''} | "
            f"{_float(row.get('best_ATE_h10_delta_vs_S0')) if _float(row.get('best_ATE_h10_delta_vs_S0')) is not None else ''} | "
            f"{_float(row.get('best_segment_200_300_delta_vs_S0')) if _float(row.get('best_segment_200_300_delta_vs_S0')) is not None else ''} | "
            f"{_float(row.get('best_rolling100_mean_delta_vs_S0')) if _float(row.get('best_rolling100_mean_delta_vs_S0')) is not None else ''} | "
            f"{_float(row.get('max_segment_400_600_delta_vs_S0')) if _float(row.get('max_segment_400_600_delta_vs_S0')) is not None else ''} |"
        )
    lines.append("")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "v45_c23_support_short_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
