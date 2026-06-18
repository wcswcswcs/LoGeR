#!/usr/bin/env python3
"""Boundary-level predictiveness diagnostic for ACL2 v67 O3.

This tool consumes single-boundary oracle CSVs and tests whether existing
semantic/geometric boundary features predict which reset-boundary hold improves
ATE. It is diagnostic-only: it does not define an online controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


FEATURES = [
    "boundary_Q_scale",
    "boundary_Q_scale_smoothed",
    "boundary_geometry_confidence_mean",
    "boundary_condition_mean",
    "boundary_vertical_mean",
    "boundary_road_mean",
    "boundary_vegetation_mean",
    "boundary_dynamic_mean",
    "boundary_future_err_mean",
    "boundary_headtail_mean",
    "boundary_H35_gap_mean",
    "boundary_source_scale",
]


ALIASES = {
    "source_label": ["source_label", "source"],
    "candidate": ["candidate", "policy"],
    "held_boundary_chunk": ["held_boundary_chunk", "held_boundary"],
    "global_ate": ["global_ate", "candidate_ate"],
    "delta_vs_baseline_global_ate": ["delta_vs_baseline_global_ate", "delta_ate"],
    "boundary_Q_scale": ["boundary_Q_scale", "Q_scale"],
    "boundary_Q_scale_smoothed": ["boundary_Q_scale_smoothed", "Q_scale_smoothed"],
    "boundary_geometry_confidence_mean": ["boundary_geometry_confidence_mean", "geometry_confidence_mean"],
    "boundary_condition_mean": ["boundary_condition_mean", "condition_score"],
    "boundary_vertical_mean": ["boundary_vertical_mean", "vertical_static_total_ratio"],
    "boundary_road_mean": ["boundary_road_mean", "road_plane_dominance"],
    "boundary_vegetation_mean": ["boundary_vegetation_mean", "vegetation_ratio"],
    "boundary_dynamic_mean": ["boundary_dynamic_mean", "dynamic_ratio"],
    "boundary_future_err_mean": ["boundary_future_err_mean", "future_after_overlap_error"],
    "boundary_headtail_mean": ["boundary_headtail_mean", "head_to_tail_transfer_ratio"],
    "boundary_H35_gap_mean": ["boundary_H35_gap_mean", "H35_minus_C9_gap"],
    "boundary_source_scale": ["boundary_source_scale", "boundary_scale"],
}


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_row(row: Dict[str, Any], source_csv: Path) -> Dict[str, Any]:
    out = dict(row)
    for canonical, aliases in ALIASES.items():
        out[canonical] = _first_present(row, aliases)
    out["oracle_csv"] = str(source_csv)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    positives = [(s, y) for s, y in zip(scores, labels) if y]
    negatives = [(s, y) for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for ps, _ in positives:
        for ns, _ in negatives:
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / total) if total else None


def _collect_feature_rows(rows: Iterable[Dict[str, Any]], feature: str) -> Tuple[List[float], List[float], List[bool]]:
    xs: List[float] = []
    improvements: List[float] = []
    labels: List[bool] = []
    for row in rows:
        x = _float(row.get(feature))
        delta_ate = _float(row.get("delta_vs_baseline_global_ate"))
        if not math.isfinite(x) or not math.isfinite(delta_ate):
            continue
        xs.append(x)
        improvements.append(-delta_ate)
        labels.append(delta_ate < 0.0)
    return xs, improvements, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-csv", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--high-q", type=float, default=0.60)
    parser.add_argument("--low-q", type=float, default=0.35)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in args.oracle_csv:
        for row in _read_csv(path):
            rows.append(_normalize_row(dict(row), path))

    usable = [
        row for row in rows
        if row.get("candidate") != "MISSING_INPUT"
        and str(row.get("candidate", "")).startswith("hold_only_")
        and math.isfinite(_float(row.get("delta_vs_baseline_global_ate")))
    ]
    positives = [row for row in usable if _float(row.get("delta_vs_baseline_global_ate")) < 0.0]
    negatives = [row for row in usable if _float(row.get("delta_vs_baseline_global_ate")) >= 0.0]

    feature_rows: List[Dict[str, Any]] = []
    for feature in FEATURES:
        xs, improvements, labels = _collect_feature_rows(usable, feature)
        auc_high = _auc(xs, labels)
        auc_low = _auc([-x for x in xs], labels) if xs else None
        best_auc = None
        best_direction = None
        if auc_high is not None and auc_low is not None:
            if auc_high >= auc_low:
                best_auc = auc_high
                best_direction = "higher_predicts_improvement"
            else:
                best_auc = auc_low
                best_direction = "lower_predicts_improvement"
        feature_rows.append({
            "feature": feature,
            "valid_n": len(xs),
            "positive_n": sum(labels),
            "negative_n": len(labels) - sum(labels),
            "pearson_vs_ate_improvement": _pearson(xs, improvements),
            "spearman_vs_ate_improvement": _spearman(xs, improvements),
            "auc_higher_predicts_ate_improvement": auc_high,
            "auc_lower_predicts_ate_improvement": auc_low,
            "best_auc": best_auc,
            "best_direction": best_direction,
        })

    mismatch_rows: List[Dict[str, Any]] = []
    for row in usable:
        q = _float(row.get("boundary_Q_scale"))
        delta_ate = _float(row.get("delta_vs_baseline_global_ate"))
        if not math.isfinite(q):
            continue
        mismatch_type = None
        if q >= args.high_q and delta_ate >= 0.0:
            mismatch_type = "high_Q_but_hold_worsens_or_no_improve"
        elif q <= args.low_q and delta_ate < 0.0:
            mismatch_type = "low_Q_but_hold_improves"
        if mismatch_type is None:
            continue
        mismatch_rows.append({
            "mismatch_type": mismatch_type,
            "source_label": row.get("source_label"),
            "origin_mode": row.get("origin_mode"),
            "candidate": row.get("candidate"),
            "held_boundary_chunk": row.get("held_boundary_chunk"),
            "delta_vs_baseline_global_ate": delta_ate,
            "global_ate": row.get("global_ate"),
            "boundary_Q_scale": q,
            "boundary_Q_scale_smoothed": row.get("boundary_Q_scale_smoothed"),
            "boundary_geometry_confidence_mean": row.get("boundary_geometry_confidence_mean"),
            "boundary_source_scale": row.get("boundary_source_scale"),
            "oracle_csv": row.get("oracle_csv"),
        })

    mismatch_rows.sort(key=lambda r: (str(r["mismatch_type"]), float(r["delta_vs_baseline_global_ate"])))
    feature_rows.sort(key=lambda r: (
        1 if r["best_auc"] is None else 0,
        0.0 if r["best_auc"] is None else -float(r["best_auc"]),
        str(r["feature"]),
    ))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "feature_importance.csv", feature_rows)
    _write_csv(out_dir / "mismatch_chunks.csv", mismatch_rows)
    summary = {
        "oracle_csvs": [str(path) for path in args.oracle_csv],
        "usable_rows": len(usable),
        "ate_improved_rows": len(positives),
        "ate_not_improved_rows": len(negatives),
        "high_q": float(args.high_q),
        "low_q": float(args.low_q),
        "high_q_worsen_or_no_improve_count": sum(
            1 for row in mismatch_rows if row["mismatch_type"] == "high_Q_but_hold_worsens_or_no_improve"
        ),
        "low_q_improve_count": sum(
            1 for row in mismatch_rows if row["mismatch_type"] == "low_Q_but_hold_improves"
        ),
        "top_features": feature_rows[:5],
        "note": (
            "Diagnostic-only posthoc oracle analysis. AUC/correlation here cannot be used as method success "
            "without a pre-registered selector and held-out validation."
        ),
    }
    (out_dir / "boundary_predictiveness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
