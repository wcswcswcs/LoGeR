#!/usr/bin/env python3
"""ACL2 v19 scale/trajectory-state proxy audit.

This is an offline diagnostic.  The proxy values are computed only from
predicted trajectories and rollout bookkeeping.  Ground truth is used only
afterwards to score whether a proxy correlates with future trajectory error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
    _yaw_from_pose,
)


STEP = 29
CHUNK_SIZE = 32


def _safe_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return out.strip("_") or "run"


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _jsonable(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    serial = [{key: _jsonable(value) for key, value in row.items()} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _parse_named_path(text: str) -> Tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
    elif ":" in text and not text.startswith("/"):
        name, path = text.split(":", 1)
    else:
        path = text
        name = Path(path).parent.name
    return _safe_name(name), Path(path)


def _collect_from_csv(path: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_dir = row.get("run_dir")
            if not run_dir:
                continue
            candidate = row.get("candidate_id") or Path(run_dir).name
            chunk = row.get("chunk_id", "")
            horizon = row.get("horizon", "")
            name = _safe_name(f"{candidate}_chunk{chunk}_h{horizon}")
            out.append((name, Path(run_dir) / "01.txt"))
    return out


def _collect_from_scan_dir(path: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for pred in sorted(path.rglob("01.txt")):
        status = pred.parent / "run_status.txt"
        if status.exists():
            text = status.read_text(encoding="utf-8", errors="replace")
            if "DONE" not in text:
                continue
        out.append((_safe_name(pred.parent.name), pred))
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, int]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    n = int(xx.shape[0])
    if n < 4:
        return float("nan"), n
    rx = _rankdata(xx)
    ry = _rankdata(yy)
    if float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return float("nan"), n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def _path_length(pos: np.ndarray) -> float:
    if pos.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def _step_lengths(pos: np.ndarray) -> np.ndarray:
    if pos.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(pos, axis=0), axis=1)


def _slice_by_frame(frames: np.ndarray, start: int, end_exclusive: int) -> np.ndarray:
    return (frames >= int(start)) & (frames < int(end_exclusive))


def _chunk_specs(run_dir: Path, frames: np.ndarray, horizon: int) -> List[Tuple[int, int, int]]:
    raw_rows = _read_jsonl(run_dir / "raw_prediction_buffer_summary.jsonl")
    specs: List[Tuple[int, int, int]] = []
    if raw_rows:
        for row in raw_rows:
            if "global_start" not in row:
                continue
            start = int(row.get("global_start", 0) or 0)
            chunk = int(row.get("chunk_idx", round(start / STEP)) or 0)
            end = int(row.get("global_end", start + CHUNK_SIZE) or (start + CHUNK_SIZE))
            specs.append((chunk, start, end))
    if not specs:
        first = int(frames.min())
        last = int(frames.max())
        start = max(0, int(round(first / STEP)) * STEP)
        while start <= last:
            chunk = int(round(start / STEP))
            specs.append((chunk, start, start + CHUNK_SIZE))
            start += STEP
    # Keep chunks where the requested future horizon has at least some support.
    out: List[Tuple[int, int, int]] = []
    max_frame = int(frames.max())
    for chunk, start, end in specs:
        future_end = start + CHUNK_SIZE + int(horizon) * STEP
        if start <= max_frame and end > int(frames.min()) and future_end > int(frames.min()):
            out.append((chunk, start, end))
    return out


def _align_run_metrics(
    frames: np.ndarray,
    poses: np.ndarray,
    gt_poses: np.ndarray,
    gt_pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    raw_pos = poses[:, :3, 3]
    matched_gt = gt_pos[frames]
    scale, rot, trans = _umeyama_sim3(raw_pos, matched_gt, with_scale=True)
    aligned = _apply_alignment(poses, scale, rot, trans)
    err = np.linalg.norm(aligned[:, :3, 3] - matched_gt, axis=1)
    return aligned, err, float(scale)


def _build_rows_for_run(
    name: str,
    pred_path: Path,
    *,
    gt_poses: np.ndarray,
    gt_pos: np.ndarray,
    horizon: int,
) -> List[Dict[str, object]]:
    frames, poses, _ = _load_tum_prediction(pred_path, gt_poses.shape[0])
    if frames.shape[0] < 5:
        return []
    aligned, err, global_scale = _align_run_metrics(frames, poses, gt_poses, gt_pos)
    yaw = _yaw_from_pose(poses)
    run_dir = pred_path.parent
    rows: List[Dict[str, object]] = []
    previous_step_median = float("nan")
    previous_path_length = float("nan")
    ema_step_median = float("nan")
    ema_alpha = 0.30

    for chunk, start, end in _chunk_specs(run_dir, frames, horizon):
        mask = _slice_by_frame(frames, start, end)
        future_end = start + CHUNK_SIZE + int(horizon) * STEP
        future_mask = _slice_by_frame(frames, start, future_end)
        if int(mask.sum()) < 4 or int(future_mask.sum()) < 8:
            continue
        local_pos = poses[mask, :3, 3]
        aligned_future = aligned[future_mask]
        future_frames = frames[future_mask]
        future_err = err[future_mask]
        steps = _step_lengths(local_pos)
        step_median = float(np.median(steps)) if steps.size else float("nan")
        step_mean = float(np.mean(steps)) if steps.size else float("nan")
        path_len = _path_length(local_pos)
        chord = float(np.linalg.norm(local_pos[-1] - local_pos[0])) if local_pos.shape[0] >= 2 else float("nan")
        path_chord_ratio = path_len / max(chord, 1e-9) if _finite(path_len) and _finite(chord) else float("nan")
        local_yaw = yaw[mask]
        yaw_steps = np.abs(np.diff(local_yaw))
        yaw_steps = np.minimum(yaw_steps, 2.0 * np.pi - yaw_steps)
        yaw_median = float(np.median(yaw_steps)) if yaw_steps.size else float("nan")
        trans_yaw_ratio = yaw_median / max(step_median, 1e-9) if _finite(yaw_median) and _finite(step_median) else float("nan")

        head_steps = steps[: max(1, min(3, steps.size))]
        tail_steps = steps[-max(1, min(3, steps.size)) :]
        head_tail_ratio = (
            float(np.median(head_steps) / max(float(np.median(tail_steps)), 1e-9))
            if head_steps.size and tail_steps.size
            else float("nan")
        )
        prev_step_ratio = (
            math.log(step_median / max(previous_step_median, 1e-9))
            if _finite(step_median) and _finite(previous_step_median)
            else float("nan")
        )
        prev_path_ratio = (
            math.log(path_len / max(previous_path_length, 1e-9))
            if _finite(path_len) and _finite(previous_path_length)
            else float("nan")
        )
        ema_step_ratio = (
            math.log(step_median / max(ema_step_median, 1e-9))
            if _finite(step_median) and _finite(ema_step_median)
            else float("nan")
        )

        seg_200_300 = _segment_rmse(frames[future_mask], aligned_future, gt_pos, 200, 300)
        seg_400_600 = _segment_rmse(frames[future_mask], aligned_future, gt_pos, 400, 600)
        rows.append({
            "run_name": name,
            "run_dir": str(run_dir),
            "pred_path": str(pred_path),
            "chunk_id": int(chunk),
            "chunk_start": int(start),
            "chunk_end_exclusive": int(end),
            "horizon": int(horizon),
            "future_end_exclusive": int(future_end),
            "future_frame_count": int(future_frames.shape[0]),
            "future_ate": _rmse(future_err),
            "future_final_err": float(future_err[-1]) if future_err.size else float("nan"),
            "future_alignment_scale_full_run": float(global_scale),
            "future_alignment_scale_log_abs": abs(math.log(max(float(global_scale), 1e-12))),
            "segment_200_300_ate": seg_200_300,
            "segment_400_600_ate": seg_400_600,
            "scale_proxy_pose_step_median_log_ratio_prev": prev_step_ratio,
            "scale_proxy_pose_path_log_ratio_prev": prev_path_ratio,
            "scale_proxy_pose_step_median_log_ratio_ema": ema_step_ratio,
            "scale_proxy_overlap_head_tail_step_log_ratio": (
                math.log(head_tail_ratio) if _finite(head_tail_ratio) else float("nan")
            ),
            "scale_proxy_path_chord_excess": path_chord_ratio - 1.0 if _finite(path_chord_ratio) else float("nan"),
            "scale_proxy_turn_abs_median": yaw_median,
            "scale_proxy_yaw_per_meter": trans_yaw_ratio,
            "scale_proxy_median_fwd_motion": step_median,
            "scale_proxy_mean_fwd_motion": step_mean,
            "proxy_note": "computed_from_prediction_trajectory_only;gt_used_for_audit_metrics_only",
        })

        if _finite(step_median):
            if _finite(ema_step_median):
                ema_step_median = (1.0 - ema_alpha) * ema_step_median + ema_alpha * step_median
            else:
                ema_step_median = step_median
            previous_step_median = step_median
        if _finite(path_len):
            previous_path_length = path_len
    return rows


def _segment_rmse(
    frames: np.ndarray,
    aligned: np.ndarray,
    gt_pos: np.ndarray,
    start: int,
    end: int,
) -> float:
    mask = (frames >= int(start)) & (frames < int(end))
    if int(mask.sum()) < 3:
        return float("nan")
    err = aligned[mask, :3, 3] - gt_pos[frames[mask]]
    return _rmse(np.linalg.norm(err, axis=1))


def _correlation_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    proxy_cols = [
        key for key in rows[0].keys()
        if str(key).startswith("scale_proxy_")
    ] if rows else []
    targets = [
        "future_ate",
        "future_final_err",
        "future_alignment_scale_log_abs",
        "segment_200_300_ate",
        "segment_400_600_ate",
    ]
    out: List[Dict[str, object]] = []
    for proxy in proxy_cols:
        x = [_to_float(row.get(proxy)) for row in rows]
        finite_x = int(np.isfinite(np.asarray(x, dtype=np.float64)).sum())
        for target in targets:
            y = [_to_float(row.get(target)) for row in rows]
            rho, n = _spearman(x, y)
            out.append({
                "proxy": proxy,
                "target": target,
                "spearman": rho,
                "abs_spearman": abs(rho) if _finite(rho) else float("nan"),
                "orientation": 1 if _finite(rho) and rho >= 0.0 else (-1 if _finite(rho) else 0),
                "n": int(n),
                "finite_proxy_values": int(finite_x),
                "gate_pass_abs_ge_0p35": bool(_finite(rho) and abs(rho) >= 0.35),
                "uses_gt_runtime_action": False,
                "gt_use": "offline_audit_target_only",
            })
    out.sort(key=lambda r: (
        -(_to_float(r.get("abs_spearman")) if _finite(_to_float(r.get("abs_spearman"))) else -1.0),
        str(r.get("proxy")),
        str(r.get("target")),
    ))
    return out


def _write_debug_maps(out_dir: Path, rows: Sequence[Mapping[str, object]], corr_rows: Sequence[Mapping[str, object]]) -> None:
    debug_dir = out_dir / "scale_proxy_debug_maps"
    debug_dir.mkdir(parents=True, exist_ok=True)
    best = [row for row in corr_rows if bool(row.get("gate_pass_abs_ge_0p35"))]
    best = best[:6] if best else list(corr_rows[:6])
    if not best:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (debug_dir / "plot_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return
    for corr in best:
        proxy = str(corr.get("proxy"))
        target = str(corr.get("target"))
        x = np.asarray([_to_float(row.get(proxy)) for row in rows], dtype=np.float64)
        y = np.asarray([_to_float(row.get(target)) for row in rows], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 4:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=140)
        ax.scatter(x[mask], y[mask], s=14, alpha=0.70)
        ax.set_xlabel(proxy)
        ax.set_ylabel(target)
        ax.set_title(f"Spearman={_to_float(corr.get('spearman')):.3f}, n={int(corr.get('n', 0) or 0)}")
        fig.tight_layout()
        fig.savefig(debug_dir / f"{_safe_name(proxy)}__{_safe_name(target)}.png")
        plt.close(fig)


def _write_markdown(
    out_dir: Path,
    rows: Sequence[Mapping[str, object]],
    corr_rows: Sequence[Mapping[str, object]],
    *,
    horizon: int,
) -> None:
    best = corr_rows[0] if corr_rows else {}
    passed = [row for row in corr_rows if bool(row.get("gate_pass_abs_ge_0p35"))]
    lines = [
        "# ACL2 v19 Scale Proxy Audit",
        "",
        "This file is generated from actual trajectory artifacts. Proxy values are no-GT trajectory-only signals; GT is used only for offline audit targets.",
        "",
        f"- horizon: h{horizon}",
        f"- proxy rows: {len(rows)}",
        f"- correlation rows: {len(corr_rows)}",
        f"- proxy gate pass count: {len(passed)}",
    ]
    if best:
        lines.extend([
            "",
            "Best absolute Spearman:",
            "",
            f"- proxy: `{best.get('proxy')}`",
            f"- target: `{best.get('target')}`",
            f"- spearman: `{_to_float(best.get('spearman')):.6f}`",
            f"- n: `{int(best.get('n', 0) or 0)}`",
            f"- gate pass abs>=0.35: `{bool(best.get('gate_pass_abs_ge_0p35'))}`",
        ])
    lines.extend([
        "",
        "Top correlations:",
        "",
        "| Proxy | Target | Spearman | n | Gate |",
        "|---|---|---:|---:|---|",
    ])
    for row in corr_rows[:12]:
        lines.append(
            f"| `{row.get('proxy')}` | `{row.get('target')}` | "
            f"`{_to_float(row.get('spearman')):.6f}` | `{int(row.get('n', 0) or 0)}` | "
            f"`{bool(row.get('gate_pass_abs_ge_0p35'))}` |"
        )
    (out_dir / "scale_proxy_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, help="KITTI pose file used only for offline audit.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--trajectory", action="append", default=[], help="NAME=path/to/01.txt or path/to/01.txt")
    p.add_argument("--scan-run-dir", action="append", default=[], help="Recursively scan for DONE run directories containing 01.txt.")
    p.add_argument("--scan-csv", action="append", default=[], help="CSV with run_dir column, e.g. v18 short_rollout_metrics.")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--max-runs", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))

    sources: List[Tuple[str, Path]] = []
    for item in args.trajectory:
        sources.append(_parse_named_path(item))
    for item in args.scan_csv:
        sources.extend(_collect_from_csv(Path(item)))
    for item in args.scan_run_dir:
        sources.extend(_collect_from_scan_dir(Path(item)))

    dedup: Dict[Path, str] = {}
    for name, path in sources:
        pred = path.resolve()
        if pred.exists() and pred not in dedup:
            dedup[pred] = name
    items = [(name, path) for path, name in dedup.items()]
    if int(args.max_runs) > 0:
        items = items[: int(args.max_runs)]
    if not items:
        raise SystemExit("No trajectory artifacts found for audit.")

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for name, pred in items:
        try:
            rows.extend(
                _build_rows_for_run(
                    name,
                    pred,
                    gt_poses=gt_poses,
                    gt_pos=gt_pos,
                    horizon=int(args.horizon),
                )
            )
        except Exception as exc:
            failures.append({"run_name": name, "pred_path": str(pred), "error": str(exc)})

    corr_rows = _correlation_rows(rows)
    _write_csv(out_dir / "scale_proxy_by_chunk.csv", rows)
    _write_json(out_dir / "scale_proxy_by_chunk.json", rows)
    _write_csv(out_dir / "scale_proxy_correlation_summary.csv", corr_rows)
    _write_json(out_dir / "scale_proxy_correlation_summary.json", corr_rows)
    _write_json(out_dir / "scale_proxy_failures.json", failures)
    _write_debug_maps(out_dir, rows, corr_rows)
    _write_markdown(out_dir, rows, corr_rows, horizon=int(args.horizon))
    summary = {
        "trajectory_artifacts_seen": len(items),
        "proxy_rows": len(rows),
        "failed_artifacts": len(failures),
        "horizon": int(args.horizon),
        "best_abs_spearman": _jsonable(corr_rows[0].get("abs_spearman")) if corr_rows else None,
        "best_proxy": corr_rows[0].get("proxy") if corr_rows else None,
        "best_target": corr_rows[0].get("target") if corr_rows else None,
        "gate_pass_any_abs_ge_0p35": any(bool(row.get("gate_pass_abs_ge_0p35")) for row in corr_rows),
        "uses_gt_runtime_action": False,
        "gt_use": "offline_audit_target_only",
    }
    (out_dir / "scale_proxy_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
