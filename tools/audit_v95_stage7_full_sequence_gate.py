#!/usr/bin/env python3
"""Audit v95 Stage7 full-sequence READ gate results.

This script is intentionally small and evidence-oriented: it consumes the
trajectory diagnostics emitted by tools/kitti_trajectory_diagnostics.py plus a
candidate HMC jsonl trace, then writes rolling-window and gate-vs-chunk deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def _safe_median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def _safe_p90(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    idx = int(math.ceil(0.9 * len(xs))) - 1
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / float(len(values)))


def _read_per_frame_errors(path: Path) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"]
            frame = int(row["frame"])
            err = _float(row.get("aligned_error_m"))
            if err is None or not math.isfinite(err):
                continue
            out.setdefault(run, {})[frame] = err
    return out


def _rolling(rows_by_run: Mapping[str, Mapping[int, float]], windows: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run, frame_errs in rows_by_run.items():
        if not frame_errs:
            continue
        max_frame = max(frame_errs)
        for window in windows:
            for start in range(0, max_frame - window + 2):
                frame_ids = list(range(start, start + window))
                vals = [frame_errs[i] for i in frame_ids if i in frame_errs]
                if len(vals) != window:
                    continue
                rows.append(
                    {
                        "run": run,
                        "window": window,
                        "start": start,
                        "end": start + window,
                        "rmse_m": _rmse(vals),
                    }
                )
    return rows


def _summarize_rolling(
    rolling_rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
) -> Dict[str, Any]:
    by_run_window: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    by_run_window_start: Dict[Tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rolling_rows:
        run = str(row["run"])
        window = int(row["window"])
        start = int(row["start"])
        by_run_window.setdefault((run, window), []).append(row)
        by_run_window_start[(run, window, start)] = row

    summary: Dict[str, Any] = {}
    for (run, window), rows in sorted(by_run_window.items()):
        vals = [float(r["rmse_m"]) for r in rows]
        worst = max(rows, key=lambda r: float(r["rmse_m"]))
        best = min(rows, key=lambda r: float(r["rmse_m"]))
        summary.setdefault(str(window), {})[run] = {
            "count": len(vals),
            "mean_rmse_m": _safe_mean(vals),
            "median_rmse_m": _safe_median(vals),
            "p90_rmse_m": _safe_p90(vals),
            "worst": dict(worst),
            "best": dict(best),
        }

    for window_text, per_run in summary.items():
        window = int(window_text)
        deltas: List[Dict[str, Any]] = []
        starts = sorted(
            {
                int(row["start"])
                for row in by_run_window.get((baseline, window), [])
            }
            & {
                int(row["start"])
                for row in by_run_window.get((candidate, window), [])
            }
        )
        for start in starts:
            b = by_run_window_start[(baseline, window, start)]
            c = by_run_window_start[(candidate, window, start)]
            delta = float(c["rmse_m"]) - float(b["rmse_m"])
            deltas.append(
                {
                    "window": window,
                    "start": start,
                    "end": int(c["end"]),
                    "baseline_rmse_m": float(b["rmse_m"]),
                    "candidate_rmse_m": float(c["rmse_m"]),
                    "delta_candidate_minus_baseline_m": delta,
                }
            )
        delta_vals = [d["delta_candidate_minus_baseline_m"] for d in deltas]
        per_run["candidate_minus_baseline"] = {
            "count": len(deltas),
            "mean_delta_m": _safe_mean(delta_vals),
            "median_delta_m": _safe_median(delta_vals),
            "p90_delta_m": _safe_p90(delta_vals),
            "worse_count": sum(1 for v in delta_vals if v > 0),
            "worse_fraction": (sum(1 for v in delta_vals if v > 0) / len(delta_vals)) if delta_vals else None,
            "worst_delta": max(deltas, key=lambda d: d["delta_candidate_minus_baseline_m"]) if deltas else None,
            "best_delta": min(deltas, key=lambda d: d["delta_candidate_minus_baseline_m"]) if deltas else None,
        }
    return summary


def _read_hmc_gate(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_idx = int(row["chunk_idx"])
            frame_summary = (
                row.get("control_trace", {})
                .get("hook_effect_summary", {})
                .get("frame_attention", {})
            )
            out[chunk_idx] = {
                "chunk_idx": chunk_idx,
                "start_frame": row.get("start_frame"),
                "end_frame": row.get("end_frame"),
                "gate_active": bool(row.get("prior_v95_trackH_gate_active")),
                "gate_score": row.get("prior_v95_trackH_gate_score"),
                "gate_threshold": row.get("prior_v95_trackH_gate_threshold"),
                "gate_source": row.get("prior_v95_trackH_gate_source"),
                "gate_effective": row.get("prior_cue_source_effective"),
                "mean_abs_bias": frame_summary.get("mean_abs_bias"),
                "max_abs_bias": frame_summary.get("max_abs_bias"),
            }
    return out


def _read_chunk_errors(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[(row["run"], int(row["chunk_idx"]))] = dict(row)
    return out


def _chunk_gate_deltas(
    chunk_errors: Mapping[Tuple[str, int], Mapping[str, Any]],
    gate: Mapping[int, Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    chunk_ids = sorted({k[1] for k in chunk_errors if k[0] == candidate})
    for chunk_idx in chunk_ids:
        b = chunk_errors.get((baseline, chunk_idx))
        c = chunk_errors.get((candidate, chunk_idx))
        if not b or not c:
            continue
        b_rmse = _float(b.get("rmse_m"))
        c_rmse = _float(c.get("rmse_m"))
        if b_rmse is None or c_rmse is None:
            continue
        g = gate.get(chunk_idx, {})
        rows.append(
            {
                "chunk_idx": chunk_idx,
                "start": c.get("start"),
                "end": c.get("end"),
                "gate_active": g.get("gate_active"),
                "gate_score": g.get("gate_score"),
                "gate_threshold": g.get("gate_threshold"),
                "gate_effective": g.get("gate_effective"),
                "mean_abs_bias": g.get("mean_abs_bias"),
                "baseline_rmse_m": b_rmse,
                "candidate_rmse_m": c_rmse,
                "delta_candidate_minus_baseline_m": c_rmse - b_rmse,
                "baseline_end_error_m": _float(b.get("end_error_m")),
                "candidate_end_error_m": _float(c.get("end_error_m")),
                "delta_end_error_m": (_float(c.get("end_error_m")) or 0.0) - (_float(b.get("end_error_m")) or 0.0),
            }
        )
    return rows


def _summarize_gate_chunk(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "chunk_count": len(rows),
        "active_count": sum(1 for r in rows if r.get("gate_active")),
        "inactive_count": sum(1 for r in rows if not r.get("gate_active")),
    }
    for label, pred in (
        ("active", lambda r: bool(r.get("gate_active"))),
        ("inactive", lambda r: not bool(r.get("gate_active"))),
        ("all", lambda r: True),
    ):
        subset = [r for r in rows if pred(r)]
        deltas = [float(r["delta_candidate_minus_baseline_m"]) for r in subset]
        out[label] = {
            "count": len(subset),
            "mean_delta_m": _safe_mean(deltas),
            "median_delta_m": _safe_median(deltas),
            "worse_count": sum(1 for v in deltas if v > 0),
            "worse_fraction": (sum(1 for v in deltas if v > 0) / len(deltas)) if deltas else None,
            "worst_delta": max(subset, key=lambda r: float(r["delta_candidate_minus_baseline_m"])) if subset else None,
            "best_delta": min(subset, key=lambda r: float(r["delta_candidate_minus_baseline_m"])) if subset else None,
        }
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics-dir", required=True)
    ap.add_argument("--candidate-hmc-jsonl", required=True)
    ap.add_argument("--baseline", default="READ0_NATIVE")
    ap.add_argument("--candidate", default="READ15_GATE_FA_KEY_ALL_Q60_THEN_L07")
    ap.add_argument("--rolling-windows", type=int, nargs="*", default=[50, 100, 200])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-rolling-csv", required=True)
    ap.add_argument("--out-gate-chunk-csv", required=True)
    args = ap.parse_args(argv)

    diagnostics_dir = Path(args.diagnostics_dir)
    summary = json.loads((diagnostics_dir / "summary.json").read_text())
    frame_errors = _read_per_frame_errors(diagnostics_dir / "per_frame_errors.csv")
    rolling_rows = _rolling(frame_errors, args.rolling_windows)
    rolling_summary = _summarize_rolling(
        rolling_rows,
        baseline=args.baseline,
        candidate=args.candidate,
    )
    gate = _read_hmc_gate(Path(args.candidate_hmc_jsonl))
    chunk_errors = _read_chunk_errors(diagnostics_dir / "chunk_errors.csv")
    gate_chunk_rows = _chunk_gate_deltas(
        chunk_errors,
        gate,
        baseline=args.baseline,
        candidate=args.candidate,
    )
    gate_chunk_summary = _summarize_gate_chunk(gate_chunk_rows)

    runs = {run["name"]: run for run in summary.get("runs", [])}
    base_run = runs[args.baseline]
    cand_run = runs[args.candidate]
    out = {
        "diagnostic_only": False,
        "stage7_full_sequence_validation": True,
        "method_success": False,
        "method_success_rule": "candidate full-sequence aligned ATE should improve vs baseline without segment/rolling regression",
        "method_success_reason": "computed after loading metrics",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "full_sequence": {
            "baseline_aligned_ate_rmse_m": base_run["aligned_ate_rmse_m"],
            "candidate_aligned_ate_rmse_m": cand_run["aligned_ate_rmse_m"],
            "delta_aligned_ate_rmse_m": cand_run["aligned_ate_rmse_m"] - base_run["aligned_ate_rmse_m"],
            "baseline_final_error_m": base_run["final_error_m"],
            "candidate_final_error_m": cand_run["final_error_m"],
            "delta_final_error_m": cand_run["final_error_m"] - base_run["final_error_m"],
            "baseline_error_slope_m_per_100f": base_run["error_slope_m_per_100f"],
            "candidate_error_slope_m_per_100f": cand_run["error_slope_m_per_100f"],
            "delta_error_slope_m_per_100f": cand_run["error_slope_m_per_100f"] - base_run["error_slope_m_per_100f"],
            "baseline_yaw_rmse_deg": base_run["yaw_rmse_deg"],
            "candidate_yaw_rmse_deg": cand_run["yaw_rmse_deg"],
            "delta_yaw_rmse_deg": cand_run["yaw_rmse_deg"] - base_run["yaw_rmse_deg"],
        },
        "segment_summary": {
            window: {
                "baseline_mean_ate_rmse_m": base_run["segment_summary"][window]["mean_ate_rmse_m"],
                "candidate_mean_ate_rmse_m": cand_run["segment_summary"][window]["mean_ate_rmse_m"],
                "delta_mean_ate_rmse_m": cand_run["segment_summary"][window]["mean_ate_rmse_m"] - base_run["segment_summary"][window]["mean_ate_rmse_m"],
                "baseline_worst_ate_rmse_m": base_run["segment_summary"][window]["worst"]["ate_rmse_m"],
                "candidate_worst_ate_rmse_m": cand_run["segment_summary"][window]["worst"]["ate_rmse_m"],
                "delta_worst_ate_rmse_m": cand_run["segment_summary"][window]["worst"]["ate_rmse_m"] - base_run["segment_summary"][window]["worst"]["ate_rmse_m"],
            }
            for window in sorted(base_run.get("segment_summary", {}).keys(), key=int)
        },
        "rolling_summary": rolling_summary,
        "gate_chunk_summary": gate_chunk_summary,
    }
    out["method_success"] = bool(out["full_sequence"]["delta_aligned_ate_rmse_m"] < 0.0)
    out["method_success_reason"] = (
        "candidate aligned ATE improved vs baseline"
        if out["method_success"]
        else "candidate aligned ATE did not improve vs baseline"
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    _write_csv(Path(args.out_rolling_csv), rolling_rows)
    _write_csv(Path(args.out_gate_chunk_csv), gate_chunk_rows)

    print(f"method_success={out['method_success']}")
    print(f"delta_aligned_ate_rmse_m={out['full_sequence']['delta_aligned_ate_rmse_m']}")
    print(f"gate_active_count={gate_chunk_summary['active_count']}")
    print(f"gate_inactive_count={gate_chunk_summary['inactive_count']}")
    print(f"wrote={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
