#!/usr/bin/env python3
"""Phase D diagnostic selector for v69 S5-local-positive labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    pos = [float(s) for s, y in zip(scores, labels) if y and math.isfinite(float(s))]
    neg = [float(s) for s, y in zip(scores, labels) if not y and math.isfinite(float(s))]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for ps in pos:
        for ns in neg:
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / total) if total else None


def _norm(values: Sequence[float], value: float) -> float:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    if not xs or not math.isfinite(float(value)):
        return 0.0
    lo = min(xs)
    hi = max(xs)
    if abs(hi - lo) < 1e-9:
        return 0.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-rows", type=Path, required=True)
    parser.add_argument("--s5-label-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--min-positive-count", type=int, default=3)
    parser.add_argument("--min-auc", type=float, default=0.70)
    parser.add_argument("--min-top5-precision", type=float, default=0.40)
    parser.add_argument("--semantic-margin", type=float, default=0.05)
    args = parser.parse_args()

    labels_by_chunk = {int(float(row["chunk_id"])): _bool(row.get("s5_positive")) for row in _read_csv(args.s5_label_table)}
    rows = [row for row in _read_csv(args.anchor_rows) if int(float(row.get("chunk_id", -1))) in labels_by_chunk]
    if not rows:
        raise ValueError("no rows after joining anchor rows with S5 labels")

    harm_values = [_float(row.get("dynamic_risk_mass"), 0.0) + _float(row.get("sky_risk_mass"), 0.0) for row in rows]
    scale_values = [_float(row.get("valid_scale_anchor_count"), 0.0) for row in rows]
    coverage_values = [_float(row.get("anchor_grid_coverage"), 0.0) for row in rows]
    cond_values = [_float(row.get("anchor_condition_score"), 0.0) for row in rows]
    gram_values = [_float(row.get("anchor_gram_motion_mean"), 0.0) for row in rows]
    harm_q60 = _quantile(harm_values, 0.60)
    feature_rows: List[Dict[str, Any]] = []
    labels: List[bool] = []
    semantic_scores: List[float] = []
    geometry_scores: List[float] = []
    for row, harm in zip(rows, harm_values):
        chunk = int(float(row["chunk_id"]))
        y = bool(labels_by_chunk[chunk])
        labels.append(y)
        valid_scale = _float(row.get("valid_scale_anchor_count"), 0.0)
        coverage = _float(row.get("anchor_grid_coverage"), 0.0)
        cond = _float(row.get("anchor_condition_score"), 0.0)
        gram = _float(row.get("anchor_gram_motion_mean"), 0.0)
        harm_norm = _norm(harm_values, harm)
        scale_norm = min(1.0, valid_scale / 3.0)
        coverage_norm = _norm(coverage_values, coverage)
        cond_norm = _norm(cond_values, cond)
        gram_stable = 1.0 - _norm(gram_values, gram)
        semantic_score = (
            0.35 * harm_norm
            + 0.25 * scale_norm
            + 0.15 * coverage_norm
            + 0.15 * cond_norm
            + 0.10 * gram_stable
        )
        geometry_score = 0.40 * scale_norm + 0.30 * coverage_norm + 0.30 * cond_norm
        rule_positive = bool(
            harm_q60 is not None
            and harm >= harm_q60
            and valid_scale >= 2.0
            and coverage >= 0.20
            and cond >= 0.10
            and gram <= 0.50
        )
        semantic_scores.append(float(semantic_score))
        geometry_scores.append(float(geometry_score))
        feature_rows.append({
            "chunk_id": chunk,
            "s5_positive": y,
            "semantic_score": semantic_score,
            "geometry_score": geometry_score,
            "rule_positive": rule_positive,
            "harm_mass": harm,
            "harm_q60": harm_q60,
            "valid_scale_anchor_count": valid_scale,
            "anchor_grid_coverage": coverage,
            "anchor_condition_score": cond,
            "anchor_gram_motion_mean": gram,
            "dynamic_risk_mass": _float(row.get("dynamic_risk_mass"), 0.0),
            "sky_risk_mass": _float(row.get("sky_risk_mass"), 0.0),
        })

    semantic_auc = _auc(semantic_scores, labels)
    geometry_auc = _auc(geometry_scores, labels)
    order = sorted(range(len(feature_rows)), key=lambda i: semantic_scores[i], reverse=True)
    top5 = order[: min(5, len(order))]
    top5_precision = float(sum(1 for i in top5 if labels[i]) / max(1, len(top5)))
    rng = random.Random(int(args.random_seed))
    null_aucs: List[float] = []
    for _ in range(int(args.permutations)):
        shuffled = list(semantic_scores)
        rng.shuffle(shuffled)
        auc = _auc(shuffled, labels)
        if auc is not None:
            null_aucs.append(float(auc))
    shuffled_p95 = _quantile(null_aucs, 0.95)
    positive_count = int(sum(labels))
    semantic_margin_vs_geometry = None if semantic_auc is None or geometry_auc is None else float(semantic_auc - geometry_auc)
    semantic_margin_vs_shuffle = None if semantic_auc is None or shuffled_p95 is None else float(semantic_auc - shuffled_p95)
    gate_reasons = []
    if positive_count < int(args.min_positive_count):
        gate_reasons.append(f"positive_count {positive_count} < {int(args.min_positive_count)}")
    if semantic_auc is None or semantic_auc < float(args.min_auc):
        gate_reasons.append(f"semantic_auc {semantic_auc} < {float(args.min_auc)}")
    if top5_precision < float(args.min_top5_precision):
        gate_reasons.append(f"top5_precision {top5_precision} < {float(args.min_top5_precision)}")
    if semantic_margin_vs_geometry is None or semantic_margin_vs_geometry < float(args.semantic_margin):
        gate_reasons.append(f"semantic margin vs geometry {semantic_margin_vs_geometry} < {float(args.semantic_margin)}")
    if semantic_margin_vs_shuffle is None or semantic_margin_vs_shuffle < float(args.semantic_margin):
        gate_reasons.append(f"semantic margin vs shuffled-p95 {semantic_margin_vs_shuffle} < {float(args.semantic_margin)}")
    summary = {
        "schema": "acl2_v69_s5_selector_summary_v1",
        "anchor_rows": str(args.anchor_rows),
        "s5_label_table": str(args.s5_label_table),
        "rows": len(feature_rows),
        "positive_chunks": [row["chunk_id"] for row in feature_rows if row["s5_positive"]],
        "positive_count": positive_count,
        "semantic_auc": semantic_auc,
        "geometry_auc": geometry_auc,
        "top5_precision": top5_precision,
        "semantic_shuffled_auc_p95": shuffled_p95,
        "semantic_margin_vs_geometry": semantic_margin_vs_geometry,
        "semantic_margin_vs_shuffled_p95": semantic_margin_vs_shuffle,
        "rule_positive_chunks": [row["chunk_id"] for row in feature_rows if row["rule_positive"]],
        "phaseD_gate_pass": not gate_reasons,
        "phaseD_gate_fail_reasons": gate_reasons,
        "note": "Rule/logistic-free diagnostic only; no online S5 use unless gate passes.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "s5_selector_rows.csv", sorted(feature_rows, key=lambda r: float(r["semantic_score"]), reverse=True))
    (args.out_dir / "s5_selector_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
