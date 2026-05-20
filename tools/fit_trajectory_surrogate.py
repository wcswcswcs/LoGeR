#!/usr/bin/env python3
"""Fit a small trajectory surrogate from existing diagnostics.

This is a diagnostic/ranking helper, not a source of final metrics.  It scans
trajectory diagnostics summaries, fits a simple ridge-regression score for full
ATE from segment/yaw/scale/final-error features, and reports rank quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


FEATURES = (
    "seg_200_300",
    "seg_200_400",
    "seg_400_600",
    "final_error",
    "yaw_rmse",
    "abs_scale_delta",
)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _fixed_segment(run: Mapping[str, Any], start: int, end: int) -> float:
    for item in run.get("segment_summary", {}).get("fixed_segments", []):
        if int(item.get("start", -1)) == start and int(item.get("end", -1)) == end:
            return _float(item.get("ate_rmse_m"))
    return float("nan")


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float))
    ranks = np.empty(len(order), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 4:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    if float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_rows(root: Path, reference_scale: float | None) -> List[Dict[str, Any]]:
    rows_by_name: Dict[str, Dict[str, Any]] = {}
    for summary_path in sorted(root.glob("trajectory_diagnostics*/summary.json")):
        diag_dir = summary_path.parent
        fixed_segments: Dict[tuple[str, int, int], float] = {}
        segment_path = diag_dir / "segment_errors.csv"
        if segment_path.exists():
            with segment_path.open("r", encoding="utf-8", newline="") as handle:
                for seg in csv.DictReader(handle):
                    run_name = str(seg.get("run", ""))
                    start = int(float(seg.get("start", -1)))
                    end = int(float(seg.get("end", -1)))
                    if (start, end) in {(200, 300), (200, 400), (400, 600)}:
                        fixed_segments[(run_name, start, end)] = _float(seg.get("ate_rmse_m"))
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for run in data.get("runs", []):
            name = str(run.get("name", ""))
            if not name:
                continue
            scale = _float(run.get("sim3_scale"))
            row = {
                "run": name,
                "diag_dir": str(summary_path.parent),
                "ate_rmse": _float(run.get("aligned_ate_rmse_m")),
                "final_error": _float(run.get("final_error_m")),
                "yaw_rmse": _float(run.get("yaw_rmse_deg")),
                "sim3_scale": scale,
                "seg_200_300": fixed_segments.get((name, 200, 300), _fixed_segment(run, 200, 300)),
                "seg_200_400": fixed_segments.get((name, 200, 400), _fixed_segment(run, 200, 400)),
                "seg_400_600": fixed_segments.get((name, 400, 600), _fixed_segment(run, 400, 600)),
            }
            target_scale = reference_scale if reference_scale is not None else scale
            row["abs_scale_delta"] = abs(scale - target_scale) if math.isfinite(scale) else float("nan")
            if name not in rows_by_name or row["ate_rmse"] < rows_by_name[name]["ate_rmse"]:
                rows_by_name[name] = row
    return sorted(rows_by_name.values(), key=lambda item: item["ate_rmse"])


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-run", default="C16ROLE_01")
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    first_rows = _load_rows(root, None)
    reference_scale = None
    for row in first_rows:
        if row["run"] == args.reference_run:
            reference_scale = _float(row.get("sim3_scale"))
            break
    rows = _load_rows(root, reference_scale)
    usable = [
        row for row in rows
        if math.isfinite(_float(row.get("ate_rmse")))
        and all(math.isfinite(_float(row.get(feature))) for feature in FEATURES)
    ]
    if len(usable) < len(FEATURES) + 2:
        raise SystemExit(f"Not enough usable rows for surrogate fit: {len(usable)}")

    x = np.asarray([[_float(row[feature]) for feature in FEATURES] for row in usable], dtype=float)
    y = np.asarray([_float(row["ate_rmse"]) for row in usable], dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    xs = (x - mean) / std
    design = np.concatenate([np.ones((xs.shape[0], 1), dtype=float), xs], axis=1)
    reg = np.eye(design.shape[1], dtype=float) * float(args.ridge)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ coef
    spearman = _spearman(pred.tolist(), y.tolist())
    top = min(5, len(usable))
    true_top = {row["run"] for row in sorted(usable, key=lambda item: item["ate_rmse"])[:top]}
    pred_top = {usable[i]["run"] for i in np.argsort(pred)[:top]}
    top_recall = len(true_top & pred_top) / float(top)

    pred_rows: List[Dict[str, Any]] = []
    for row, score in zip(usable, pred.tolist()):
        out = dict(row)
        out["surrogate_score"] = float(score)
        out["surrogate_error"] = float(score - _float(row["ate_rmse"]))
        pred_rows.append(out)
    pred_rows.sort(key=lambda item: item["surrogate_score"])
    _write_csv(out_dir / "surrogate_predictions.csv", pred_rows)

    coeff = {
        "intercept": float(coef[0]),
        "features": {
            feature: {
                "standardized_coefficient": float(value),
                "mean": float(mu),
                "std": float(sigma),
            }
            for feature, value, mu, sigma in zip(FEATURES, coef[1:], mean, std)
        },
        "n": len(usable),
        "spearman_score_vs_ate": spearman,
        "top5_recall": top_recall,
        "reference_run": args.reference_run,
        "reference_scale": reference_scale,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surrogate_coefficients.json").write_text(json.dumps(coeff, indent=2, sort_keys=True), encoding="utf-8")

    def fmt(value: Any, digits: int = 4) -> str:
        val = _float(value)
        return f"{val:.{digits}f}" if math.isfinite(val) else "nan"

    lines = [
        "# V9 Trajectory Surrogate Fit",
        "",
        f"usable_runs = `{len(usable)}`",
        f"reference_run = `{args.reference_run}`",
        f"Spearman(score, ATE) = `{fmt(spearman, 3)}`",
        f"Top-5 recall = `{fmt(top_recall, 3)}`",
        "",
        "## Coefficients",
        "",
        "| Feature | standardized coefficient | mean | std |",
        "|---|---:|---:|---:|",
    ]
    for feature in FEATURES:
        item = coeff["features"][feature]
        lines.append(
            f"| `{feature}` | {fmt(item['standardized_coefficient'])} | {fmt(item['mean'])} | {fmt(item['std'])} |"
        )
    lines += ["", "## Top Predicted Runs", "", "| Run | surrogate | ATE | [200,300) | [400,600) |", "|---|---:|---:|---:|---:|"]
    for row in pred_rows[:12]:
        lines.append(
            f"| `{row['run']}` | {fmt(row['surrogate_score'])} | {fmt(row['ate_rmse'])} | "
            f"{fmt(row['seg_200_300'])} | {fmt(row['seg_400_600'])} |"
        )
    if not (math.isfinite(spearman) and spearman >= 0.60 and top_recall >= 0.60):
        lines += [
            "",
            "## Gate",
            "",
            "Surrogate does not meet the v9 gating target; use it for visualization/ranking context only, not as a hard full-run filter.",
        ]
    (out_dir / "surrogate_fit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote surrogate fit for {len(usable)} runs to {out_dir}")


if __name__ == "__main__":
    main()
