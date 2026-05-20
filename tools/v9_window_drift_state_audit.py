#!/usr/bin/env python3
"""Build the v9 window-drift audit table from existing trajectory diagnostics.

The v8 diagnostics were emitted in several per-batch directories.  This tool
deduplicates runs across those directories, joins available per-chunk trajectory
signals with top-level HMC debug summaries, and writes the v9 Batch-A CSVs.
It intentionally does not invent missing low-level TTT fields; unavailable
projection/tri-mass fields are left absent and reported as gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_WINDOWS = (
    ("pre", 0, 4),
    ("body", 5, 9),
    ("exit", 10, 12),
    ("mid", 13, 15),
    ("c16", 16, 16),
    ("handoff", 17, 18),
    ("post", 19, 24),
    ("tail", 25, 37),
)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return mean(vals) if vals else float("nan")


def _rmse(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 4:
        return float("nan")
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1e-18 or vy <= 1e-18:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 4:
        return float("nan")

    def ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and vals[order[j]] == vals[order[i]]:
                j += 1
            rank = (i + j - 1) / 2.0
            for k in range(i, j):
                out[order[k]] = rank
            i = j
        return out

    rx = ranks([x for x, _ in pairs])
    ry = ranks([y for _, y in pairs])
    return _pearson(rx, ry)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fixed_segment(run: Mapping[str, Any], start: int, end: int) -> float:
    for item in run.get("segment_summary", {}).get("fixed_segments", []):
        if _int(item.get("start")) == start and _int(item.get("end")) == end:
            return _float(item.get("ate_rmse_m"))
    return float("nan")


def _load_runs(root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    runs: Dict[str, Dict[str, Any]] = {}
    chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(root.glob("trajectory_diagnostics*/summary.json")):
        diag_dir = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for run in summary.get("runs", []):
            name = str(run.get("name", ""))
            if not name:
                continue
            record = {
                "run": name,
                "diag_dir": str(diag_dir),
                "pred_path": str(run.get("path", "")),
                "run_dir": str(Path(str(run.get("path", ""))).parent) if run.get("path") else "",
                "ate_rmse": _float(run.get("aligned_ate_rmse_m")),
                "final_error": _float(run.get("final_error_m")),
                "yaw_rmse": _float(run.get("yaw_rmse_deg")),
                "sim3_scale": _float(run.get("sim3_scale")),
                "rot_rmse": float("nan"),
                "rpe_t": float("nan"),
                "rpe_r": float("nan"),
                "seg_200_300": _fixed_segment(run, 200, 300),
                "seg_200_400": _fixed_segment(run, 200, 400),
                "seg_400_500": _fixed_segment(run, 400, 500),
                "seg_400_600": _fixed_segment(run, 400, 600),
            }
            # Keep the lowest ATE duplicate when a run appears in several dashboards.
            if name not in runs or record["ate_rmse"] < runs[name].get("ate_rmse", float("inf")):
                runs[name] = record
        for row in _read_csv(diag_dir / "segment_errors.csv"):
            name = row.get("run", "")
            if name in runs:
                start, end = _int(row.get("start")), _int(row.get("end"))
                if (start, end) in {(200, 300), (200, 400), (400, 500), (400, 600)}:
                    runs[name][f"seg_{start}_{end}"] = _float(row.get("ate_rmse_m"))
        frame_by_run_chunk: Dict[Tuple[str, int], List[Dict[str, float]]] = defaultdict(list)
        for row in _read_csv(diag_dir / "per_frame_errors.csv"):
            run = row.get("run", "")
            ci = _int(row.get("chunk_idx"))
            if run and ci >= 0:
                frame_by_run_chunk[(run, ci)].append(
                    {
                        "frame": _float(row.get("frame")),
                        "err": _float(row.get("aligned_error_m")),
                        "x": _float(row.get("aligned_error_x_m")),
                        "y": _float(row.get("aligned_error_y_m")),
                        "z": _float(row.get("aligned_error_z_m")),
                        "yaw": _float(row.get("yaw_error_deg")),
                    }
                )
        for values in frame_by_run_chunk.values():
            values.sort(key=lambda item: item["frame"])
        for row in _read_csv(diag_dir / "chunk_errors.csv"):
            run = row.get("run", "")
            ci = _int(row.get("chunk_idx"))
            if run not in runs or ci < 0:
                continue
            out: Dict[str, Any] = {
                "run": run,
                "chunk_idx": ci,
                "start_frame": _int(row.get("start")),
                "end_frame": _int(row.get("end")),
                "chunk_rmse": _float(row.get("rmse_m")),
                "chunk_mean": _float(row.get("mean_m")),
                "chunk_start_error": _float(row.get("start_error_m")),
                "chunk_end_error": _float(row.get("end_error_m")),
            }
            frames = frame_by_run_chunk.get((run, ci), [])
            if frames:
                first, last = frames[0], frames[-1]
                dx, dy, dz = last["x"] - first["x"], last["y"] - first["y"], last["z"] - first["z"]
                out.update(
                    {
                        "drift_x": dx,
                        "drift_y": dy,
                        "drift_z": dz,
                        "drift_norm": math.sqrt(dx * dx + dy * dy + dz * dz),
                        "yaw_delta": last["yaw"] - first["yaw"],
                        "error_slope": (last["err"] - first["err"]) / max(last["frame"] - first["frame"], 1.0),
                    }
                )
            chunks[run].append(out)
    for rows in chunks.values():
        rows.sort(key=lambda row: _int(row.get("chunk_idx")))
    return runs, chunks


def _load_hmc_top(run_dir: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    path = run_dir / "hmc_control_summary.jsonl"
    if not path.exists():
        path = run_dir / "hmc_state_hash.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ci = _int(row.get("chunk_idx"))
        if ci < 0:
            continue
        feats = {
            key: _float(value)
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        out[ci] = feats
    return out


def _window_rows(runs: Mapping[str, Mapping[str, Any]], chunks: Mapping[str, List[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run, meta in sorted(runs.items()):
        by_chunk = {_int(row.get("chunk_idx")): row for row in chunks.get(run, [])}
        hmc = _load_hmc_top(Path(str(meta.get("run_dir", "")))) if meta.get("run_dir") else {}
        for window, start, end in DEFAULT_WINDOWS:
            selected = [by_chunk[i] for i in range(start, end + 1) if i in by_chunk]
            if not selected:
                continue
            dx = sum(_float(row.get("drift_x"), 0.0) for row in selected)
            dy = sum(_float(row.get("drift_y"), 0.0) for row in selected)
            dz = sum(_float(row.get("drift_z"), 0.0) for row in selected)
            hmc_selected = [hmc.get(_int(row.get("chunk_idx")), {}) for row in selected]
            row = dict(meta)
            row.update(
                {
                    "window": window,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_count": len(selected),
                    "mean_chunk_rmse": _mean(_float(item.get("chunk_rmse")) for item in selected),
                    "rmse_chunk_rmse": _rmse(_float(item.get("chunk_rmse")) for item in selected),
                    "mean_chunk_delta_error": _mean(
                        _float(item.get("chunk_end_error")) - _float(item.get("chunk_start_error")) for item in selected
                    ),
                    "sum_drift_x": dx,
                    "sum_drift_y": dy,
                    "sum_drift_z": dz,
                    "sum_drift_norm": math.sqrt(dx * dx + dy * dy + dz * dz),
                    "mean_drift_norm": _mean(_float(item.get("drift_norm")) for item in selected),
                    "mean_yaw_delta": _mean(_float(item.get("yaw_delta")) for item in selected),
                    "mean_error_slope": _mean(_float(item.get("error_slope")) for item in selected),
                    "hmc_memory_ttt_w0_mean_rel_diff": _mean(
                        _float(item.get("memory_ttt_w0_mean_rel_diff")) for item in hmc_selected
                    ),
                    "hmc_pass1_pass2_pose_t_mean": _mean(
                        _float(item.get("pass1_pass2_pose_t_mean")) for item in hmc_selected
                    ),
                    "hmc_pass1_pass2_world_points_l1_mean": _mean(
                        _float(item.get("pass1_pass2_world_points_l1_mean")) for item in hmc_selected
                    ),
                }
            )
            rows.append(row)
    return rows


def _tradeoff(rows: List[Mapping[str, Any]], reference_run: str) -> List[Dict[str, Any]]:
    body_ref = next((r for r in rows if r.get("run") == reference_run and r.get("window") == "body"), None)
    if not body_ref:
        return []
    ref_local = _float(body_ref.get("seg_200_300"))
    ref_down = _float(body_ref.get("seg_400_600"))
    ref_ate = _float(body_ref.get("ate_rmse"))
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        run = str(row.get("run"))
        if run in seen or row.get("window") != "body":
            continue
        seen.add(run)
        local = _float(row.get("seg_200_300"))
        down = _float(row.get("seg_400_600"))
        ate = _float(row.get("ate_rmse"))
        out.append(
            {
                "run": run,
                "reference_run": reference_run,
                "ate_rmse": ate,
                "local_gain_200_300": ref_local - local,
                "downstream_cost_400_600": down - ref_down,
                "overall_gain_ate": ref_ate - ate,
                "safe_vs_reference": int((ref_ate - ate) > 0.0 and (down - ref_down) < 0.5),
                "pareto_x_local_gain": ref_local - local,
                "pareto_y_downstream_cost": down - ref_down,
            }
        )
    out.sort(key=lambda item: _float(item.get("ate_rmse")))
    return out


def _correlation_report(rows: List[Mapping[str, Any]], trade: List[Mapping[str, Any]], target: str) -> List[Dict[str, Any]]:
    by_run = {str(row.get("run")): row for row in trade}
    candidates = [
        "mean_chunk_rmse",
        "rmse_chunk_rmse",
        "sum_drift_norm",
        "mean_drift_norm",
        "mean_yaw_delta",
        "mean_error_slope",
        "hmc_memory_ttt_w0_mean_rel_diff",
        "hmc_pass1_pass2_pose_t_mean",
        "hmc_pass1_pass2_world_points_l1_mean",
        "yaw_rmse",
        "sim3_scale",
        "final_error",
    ]
    out: List[Dict[str, Any]] = []
    for window in sorted({str(row.get("window")) for row in rows}):
        vals = [row for row in rows if row.get("window") == window and str(row.get("run")) in by_run]
        ys = [_float(by_run[str(row.get("run"))].get(target)) for row in vals]
        for feature in candidates:
            xs = [_float(row.get(feature)) for row in vals]
            r = _pearson(xs, ys)
            sr = _spearman(xs, ys)
            n = sum(1 for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y))
            if n >= 4 and (math.isfinite(r) or math.isfinite(sr)):
                out.append({"window": window, "target": target, "feature": feature, "pearson_r": r, "spearman_r": sr, "n": n})
    out.sort(key=lambda item: (item["window"], -abs(_float(item.get("spearman_r"), 0.0)), item["feature"]))
    return out


def _write_markdown(path: Path, runs: Mapping[str, Mapping[str, Any]], trade: List[Mapping[str, Any]], corr_local: List[Mapping[str, Any]], corr_down: List[Mapping[str, Any]], reference_run: str) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        val = _float(value)
        return f"{val:.{digits}f}" if math.isfinite(val) else "nan"

    lines = [
        "# ACL2 v9 Window Drift-State Audit",
        "",
        f"reference_run = `{reference_run}`",
        "",
        "## Run Ranking",
        "",
        "| Run | ATE | FinalErr | Yaw | [200,300) | [400,600) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(runs.values(), key=lambda item: _float(item.get("ate_rmse")))[:20]:
        lines.append(
            f"| `{row.get('run')}` | {fmt(row.get('ate_rmse'))} | {fmt(row.get('final_error'))} | "
            f"{fmt(row.get('yaw_rmse'))} | {fmt(row.get('seg_200_300'))} | {fmt(row.get('seg_400_600'))} |"
        )
    lines += ["", "## Trade-Off vs Reference", "", "| Run | local gain [200,300) | downstream cost [400,600) | ATE gain | safe |", "|---|---:|---:|---:|---:|"]
    for row in trade[:20]:
        lines.append(
            f"| `{row.get('run')}` | {fmt(row.get('local_gain_200_300'))} | "
            f"{fmt(row.get('downstream_cost_400_600'))} | {fmt(row.get('overall_gain_ate'))} | {row.get('safe_vs_reference')} |"
        )
    lines += ["", "## Strongest Offline Correlations", ""]
    for title, corr in [("local_gain_200_300", corr_local), ("downstream_cost_400_600", corr_down)]:
        lines += [f"### {title}", "", "| Window | Feature | Spearman | Pearson | n |", "|---|---|---:|---:|---:|"]
        for row in corr[:20]:
            lines.append(
                f"| `{row.get('window')}` | `{row.get('feature')}` | {fmt(row.get('spearman_r'), 3)} | "
                f"{fmt(row.get('pearson_r'), 3)} | {row.get('n')} |"
            )
        lines.append("")
    lines += [
        "## Data Gaps",
        "",
        "- Existing JSONL summaries do not contain per-token/per-layer projection groups.",
        "- Existing compact HMC summaries do not expose tri-replay pos/neg/neutral mass by chunk; those values are visible in verbose logs only and are not reliably machine-readable across all runs.",
        "- Batch B must therefore add explicit projection debug fields before any projection full run is considered valid.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-run", default="WINGAM_03_repeat")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    runs, chunks = _load_runs(root)
    if not runs:
        raise SystemExit(f"No trajectory diagnostic summaries found under {root}")
    rows = _window_rows(runs, chunks)
    trade = _tradeoff(rows, args.reference_run)
    corr_local = _correlation_report(rows, trade, "local_gain_200_300")
    corr_down = _correlation_report(rows, trade, "downstream_cost_400_600")

    _write_csv(out_dir / "window_drift_state_raw.csv", rows)
    _write_csv(out_dir / "window_drift_state_summary.csv", rows)
    _write_csv(out_dir / "run_pair_tradeoff.csv", trade)
    _write_csv(out_dir / "feature_correlation_to_local_gain.csv", corr_local)
    _write_csv(out_dir / "feature_correlation_to_downstream_cost.csv", corr_down)
    _write_markdown(out_dir / "v9_window_drift_state_audit.md", runs, trade, corr_local, corr_down, args.reference_run)
    print(f"Wrote v9 window drift audit for {len(runs)} runs to {out_dir}")


if __name__ == "__main__":
    main()
