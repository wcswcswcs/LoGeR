#!/usr/bin/env python3
"""Landed-artifact-only v42 full-online report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


DEFAULT_GT = "/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt"
HISTORICAL_C9_ATE = 33.7629421029


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def _status_done(run_dir: Path, run_name: str) -> bool:
    path = run_dir / "run_status.txt"
    return path.exists() and f"DONE {run_name}" in path.read_text(encoding="utf-8", errors="replace")


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
        elif match.group("kind") == "DONE":
            dones.append(stamp)
    if not starts or not dones:
        return float("nan")
    return float((max(dones) - min(starts)).total_seconds())


def _pose_lookup(frames: np.ndarray, poses: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(frame): pose for frame, pose in zip(frames.astype(np.int64), poses)}


def _as_positions(poses_or_pos: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses_or_pos)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr[:, :3, 3]
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3]
    raise ValueError(f"Unsupported pose/position array shape: {arr.shape}")


def _ate(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    if int(pos_a.shape[0]) < 3 or int(pos_b.shape[0]) < 3:
        return float("nan")
    diff = pos_a - pos_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _rolling_windows(frames: np.ndarray, pred_poses_or_pos: np.ndarray, gt_pos: np.ndarray, width: int) -> Dict[int, float]:
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


def _rolling_stats(values: Dict[int, float]) -> Dict[str, Any]:
    vals = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "mean": None, "p90": None, "worst": None, "best_start": None, "worst_start": None}
    best_start = min(values, key=lambda k: values[k])
    worst_start = max(values, key=lambda k: values[k])
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": len(vals),
        "mean": float(np.mean(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "worst": float(np.max(arr)),
        "best_start": int(best_start),
        "worst_start": int(worst_start),
    }


def _rolling_delta_stats(candidate: Dict[int, float], reference: Dict[int, float]) -> Dict[str, Any]:
    starts = sorted(set(candidate) & set(reference))
    deltas = [float(candidate[s] - reference[s]) for s in starts if math.isfinite(candidate[s]) and math.isfinite(reference[s])]
    if not deltas:
        return {"count": 0, "best_delta": None, "mean_delta": None, "p90_delta": None, "worst_delta": None}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "count": len(deltas),
        "best_delta": float(np.min(arr)),
        "mean_delta": float(np.mean(arr)),
        "p90_delta": float(np.quantile(arr, 0.90)),
        "worst_delta": float(np.max(arr)),
    }


def _action_stats(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    configured_active_chunks: List[int] = []
    config_path = run_dir / "hmc_config.yaml"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("semantic_action_active_chunks:"):
                value = line.split(":", 1)[1].strip().strip("'\"")
                for token in value.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    if "-" in token:
                        left, right = token.split("-", 1)
                        start, end = int(left), int(right)
                        if end < start:
                            start, end = end, start
                        configured_active_chunks.extend(range(start, end + 1))
                    else:
                        configured_active_chunks.append(int(token))
    chunks: List[int] = []
    active_chunks: List[int] = []
    context_empty = 0.0
    attention_mass_rows = 0
    for item in rows:
        if "chunk_idx" in item:
            chunks.append(int(item["chunk_idx"]))
        chunk = int(item["chunk_idx"]) if "chunk_idx" in item else None
        if chunk is not None and chunk in set(configured_active_chunks):
            active_chunks.append(chunk)
        trace = item.get("control_trace")
        summaries = {}
        if isinstance(trace, dict):
            summaries = trace.get("hook_effect_summary") or {}
        for summary in summaries.values() if isinstance(summaries, dict) else []:
            if not isinstance(summary, dict):
                continue
            try:
                context_empty += float(summary.get("num_context_empty_source_events") or 0.0)
            except (TypeError, ValueError):
                pass
            if summary.get("attention_mass_available"):
                attention_mass_rows += 1
    unique_chunks = sorted(set(chunks))
    unique_active = sorted(set(active_chunks))
    return {
        "hmc_rows": len(rows),
        "hmc_unique_chunks": unique_chunks,
        "hmc_unique_chunk_count": len(unique_chunks),
        "action_active_chunks": unique_active,
        "action_active_chunk_count": len(unique_active),
        "action_active_ratio": float(len(unique_active) / len(unique_chunks)) if unique_chunks else None,
        "context_empty_source_events_total": float(context_empty),
        "scalar_attention_mass_rows": int(attention_mass_rows),
    }


def _parse_runs(root: Path, text: str) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Bad --runs item: {item}")
        name, rel = item.split("=", 1)
        path = Path(rel.strip())
        if not path.is_absolute():
            path = root / path
        entries.append((name.strip(), path))
    return entries


def _write_md(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ACL2 v42 Full Online Report",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "rows",
        "done_rows",
        "reference_name",
        "reference_ATE_full",
        "historical_c9_ate",
        "best_read_candidate",
        "best_read_ATE_full",
        "best_read_delta_vs_reference",
        "best_read_delta_vs_historical_c9",
        "minimum_progress_pass",
        "stage_success_pass",
        "strong_success_pass",
        "target30_success",
        "phase4_allowed",
    ]:
        lines.append(f"- `{key}`: `{_clean(summary.get(key))}`")
    lines.extend([
        "",
        "## Rows",
        "",
        "| Name | Status | ATE | Delta vs F0 | Delta vs historical C9 | [200,300) | [400,600) | Action chunks | Target30 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in rows:
        lines.append(
            "| `{name}` | `{status}` | {ate} | {dref} | {dhist} | {s200} | {s400} | `{chunks}` | `{target}` |".format(
                name=row.get("name"),
                status=row.get("status"),
                ate=_fmt(row.get("ATE_full")),
                dref=_fmt(row.get("ATE_delta_vs_reference")),
                dhist=_fmt(row.get("delta_vs_historical_c9_ATE")),
                s200=_fmt(row.get("segment_200_300_ATE")),
                s400=_fmt(row.get("segment_400_600_ATE")),
                chunks=row.get("action_active_chunks"),
                target=row.get("target30_pass"),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return ""
        return f"{float(value):.10f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True, type=Path)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--reference-name", default="F0")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--historical-c9-ate", type=float, default=HISTORICAL_C9_ATE)
    args = parser.parse_args()

    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    entries = _parse_runs(args.rollout_root, args.runs)

    reference_lookup: Optional[Dict[int, np.ndarray]] = None
    reference_roll: Dict[int, Dict[int, float]] = {}
    reference_ate: Optional[float] = None
    rows: List[Dict[str, Any]] = []

    loaded: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    for name, run_dir in entries:
        pred_path = run_dir / "01.txt"
        row: Dict[str, Any] = {
            "name": name,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "run_status_done": _status_done(run_dir, run_dir.name),
        }
        if not pred_path.exists():
            row["status"] = "missing_prediction"
            row.update(_action_stats(run_dir))
            rows.append(row)
            continue
        frames, poses, _ = _load_tum_prediction(pred_path, gt_pos.shape[0])
        aligned, metrics = _align_metrics(frames.astype(np.int64), poses, gt_poses, gt_pos)
        row.update({
            "status": "done",
            "frames": int(frames.shape[0]),
            "wall_seconds": _runtime_sec(run_dir),
            "ATE_full": float(metrics["ATE_horizon"]),
            "Rot_full": float(metrics["Rot_horizon"]),
            "FinalErr_full": float(metrics["FinalErr_horizon"]),
            "RPE_t_full": float(metrics.get("RPE_t_horizon", float("nan"))),
            "RPE_r_full": float(metrics.get("RPE_r_horizon", float("nan"))),
            "segment_200_300_ATE": _segment_ate(frames.astype(np.int64), aligned, gt_pos, 200, 300),
            "segment_400_600_ATE": _segment_ate(frames.astype(np.int64), aligned, gt_pos, 400, 600),
            "delta_vs_historical_c9_ATE": float(metrics["ATE_horizon"]) - float(args.historical_c9_ate),
            "target30_pass": float(metrics["ATE_horizon"]) <= 30.0,
        })
        for width in (50, 100, 200):
            rw = _rolling_windows(frames.astype(np.int64), aligned, gt_pos, width)
            stats = _rolling_stats(rw)
            row.update({f"rolling{width}_{k}": v for k, v in stats.items()})
            if name == args.reference_name:
                reference_roll[width] = rw
        row.update(_action_stats(run_dir))
        loaded[name] = (frames, poses, aligned, row)
        if name == args.reference_name:
            reference_lookup = _pose_lookup(frames, poses)
            reference_ate = float(metrics["ATE_horizon"])
        rows.append(row)

    if reference_lookup is not None and reference_ate is not None:
        for name, (frames, poses, aligned, row) in loaded.items():
            if name == args.reference_name:
                row["ATE_delta_vs_reference"] = 0.0
                row["segment_200_300_delta_vs_reference"] = 0.0
                row["segment_400_600_delta_vs_reference"] = 0.0
                continue
            ref_subset = np.stack([reference_lookup[int(frame)] for frame in frames.astype(np.int64)], axis=0)
            ref_aligned, ref_metrics = _align_metrics(frames.astype(np.int64), ref_subset, gt_poses, gt_pos)
            row["ATE_delta_vs_reference"] = float(row["ATE_full"] - reference_ate)
            row["segment_200_300_delta_vs_reference"] = float(
                row["segment_200_300_ATE"] - _segment_ate(frames.astype(np.int64), ref_aligned, gt_pos, 200, 300)
            )
            row["segment_400_600_delta_vs_reference"] = float(
                row["segment_400_600_ATE"] - _segment_ate(frames.astype(np.int64), ref_aligned, gt_pos, 400, 600)
            )
            raw_abs, raw_trans, ts_equal = _raw_diff(frames.astype(np.int64), poses, reference_lookup)
            row["raw_pose_max_abs_diff_vs_reference"] = raw_abs
            row["raw_translation_max_diff_vs_reference"] = raw_trans
            row["timestamp_equal_reference"] = bool(ts_equal)
            for width in (50, 100, 200):
                cand_rw = _rolling_windows(frames.astype(np.int64), aligned, gt_pos, width)
                delta_stats = _rolling_delta_stats(cand_rw, reference_roll.get(width, {}))
                row.update({f"rolling{width}_{k}_vs_reference": v for k, v in delta_stats.items()})

    done_rows = [r for r in rows if r.get("status") == "done"]
    read_rows = [r for r in done_rows if r.get("name") not in {args.reference_name, "F5"}]
    best = min(read_rows, key=lambda r: float(r.get("ATE_full", float("inf"))), default=None)
    best_delta_ref = best.get("ATE_delta_vs_reference") if best else None
    best_ate = best.get("ATE_full") if best else None
    downstream_ok = bool(best and float(best.get("segment_400_600_delta_vs_reference", float("inf"))) <= 1.0)
    minimum_progress = bool(
        best
        and (
            float(best.get("ATE_full", float("inf"))) <= 33.3
            or float(best.get("delta_vs_historical_c9_ATE", float("inf"))) <= -0.5
        )
        and downstream_ok
    )
    summary = {
        "rows": len(rows),
        "done_rows": len(done_rows),
        "reference_name": args.reference_name,
        "reference_ATE_full": reference_ate,
        "historical_c9_ate": float(args.historical_c9_ate),
        "best_read_candidate": best.get("name") if best else None,
        "best_read_ATE_full": best_ate,
        "best_read_delta_vs_reference": best_delta_ref,
        "best_read_delta_vs_historical_c9": best.get("delta_vs_historical_c9_ATE") if best else None,
        "minimum_progress_pass": minimum_progress,
        "stage_success_pass": bool(best and float(best.get("ATE_full", float("inf"))) <= 33.0),
        "strong_success_pass": bool(best and float(best.get("ATE_full", float("inf"))) <= 32.0),
        "target30_success": bool(best and float(best.get("ATE_full", float("inf"))) <= 30.0),
        "phase4_allowed": bool(best_delta_ref is not None and float(best_delta_ref) <= -0.3 and float(best_ate) > 33.0),
        "no_gt_runtime_action": True,
        "training_free": True,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "full_online_registry.csv", rows)
    _write_json(args.out_dir / "v42_full_online_summary.json", summary)
    _write_json(args.out_dir / "v42_full_online_rows.json", rows)
    _write_md(args.out_dir / "v42_full_online_report.md", summary, rows)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
