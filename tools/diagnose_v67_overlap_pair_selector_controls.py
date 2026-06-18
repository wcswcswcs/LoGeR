#!/usr/bin/env python3
"""Strict selector/control validation for v67 overlap-pair oracle positives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GEOMETRY_FEATURES = [
    "raw_residual_rmse",
    "raw_residual_mean",
    "valid_pair_count",
    "saved_pair_count",
    "prev_conf_mean",
    "curr_conf_mean",
]

SEMANTIC_FEATURES = [
    "semantic_nonvoid_ratio",
    "semantic_conf_mean",
    "dynamic_ratio",
    "sky_context_ratio",
    "vegetation_farstuff_ratio",
    "vertical_static_ratio",
    "ground_static_ratio",
    "void_lowtrust_ratio",
    "label_road_ratio",
    "label_building_ratio",
    "label_car_ratio",
    "label_vegetation_ratio",
    "label_sky_ratio",
    "label_void_ratio",
]

ALL_FEATURES = GEOMETRY_FEATURES + SEMANTIC_FEATURES


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    positives = [float(s) for s, label in zip(scores, labels) if label and math.isfinite(float(s))]
    negatives = [float(s) for s, label in zip(scores, labels) if not label and math.isfinite(float(s))]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for ps in positives:
        for ns in negatives:
            total += 1
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / total) if total else None


def _rank_of_positive(scores: Sequence[float], labels: Sequence[bool], direction: str) -> Optional[int]:
    positives = [idx for idx, label in enumerate(labels) if label]
    if len(positives) != 1:
        return None
    pos = positives[0]
    values = [float(s) if math.isfinite(float(s)) else float("-inf") for s in scores]
    reverse = direction == "higher"
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=reverse)
    return int(order.index(pos) + 1)


def _evaluate_feature(rows: Sequence[Dict[str, Any]], labels: Sequence[bool], feature: str) -> Dict[str, Any]:
    values = [_float(row.get(feature)) for row in rows]
    auc_high = _auc(values, labels)
    auc_low = _auc([-v for v in values], labels)
    if auc_high is None or auc_low is None:
        best_auc = None
        direction = ""
    elif auc_high >= auc_low:
        best_auc = auc_high
        direction = "higher"
    else:
        best_auc = auc_low
        direction = "lower"
    rank = _rank_of_positive(values, labels, direction) if direction else None
    return {
        "feature": feature,
        "auc_higher": auc_high,
        "auc_lower": auc_low,
        "best_auc": best_auc,
        "best_direction": direction,
        "positive_rank": rank,
        "top1_hit": rank == 1 if rank is not None else None,
        "top3_hit": rank <= 3 if rank is not None else None,
        "top5_hit": rank <= 5 if rank is not None else None,
    }


def _best_group(rows: Sequence[Dict[str, Any]], labels: Sequence[bool], features: Sequence[str], group: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    eval_rows = []
    for feature in features:
        if feature not in rows[0]:
            continue
        item = _evaluate_feature(rows, labels, feature)
        item["feature_group"] = group
        eval_rows.append(item)
    eval_rows.sort(key=lambda row: (
        -1.0 if row.get("best_auc") is None else -float(row["best_auc"]),
        999 if row.get("positive_rank") is None else int(row["positive_rank"]),
        row["feature"],
    ))
    return (eval_rows[0] if eval_rows else {}), eval_rows


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
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


def _semantic_row_shuffle_null(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[bool],
    *,
    features: Sequence[str],
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    null_best_auc: List[float] = []
    null_best_rank: List[int] = []
    base_values = {feature: [_float(row.get(feature)) for row in rows] for feature in features if feature in rows[0]}
    for _ in range(int(permutations)):
        perm = list(range(len(rows)))
        rng.shuffle(perm)
        best_auc = float("nan")
        best_rank = None
        for feature, values in base_values.items():
            shuffled = [values[idx] for idx in perm]
            auc_high = _auc(shuffled, labels)
            auc_low = _auc([-v for v in shuffled], labels)
            if auc_high is None or auc_low is None:
                continue
            if auc_high >= auc_low:
                auc = auc_high
                direction = "higher"
            else:
                auc = auc_low
                direction = "lower"
            rank = _rank_of_positive(shuffled, labels, direction)
            if not math.isfinite(best_auc) or auc > best_auc:
                best_auc = auc
                best_rank = rank
        if math.isfinite(best_auc):
            null_best_auc.append(float(best_auc))
            if best_rank is not None:
                null_best_rank.append(int(best_rank))
    return {
        "permutations": int(permutations),
        "best_auc_p50": _percentile(null_best_auc, 0.50),
        "best_auc_p95": _percentile(null_best_auc, 0.95),
        "best_auc_p99": _percentile(null_best_auc, 0.99),
        "best_rank_p50": _percentile(null_best_rank, 0.50),
        "best_rank_p05": _percentile(null_best_rank, 0.05),
        "best_rank_p01": _percentile(null_best_rank, 0.01),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-features-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--min-positive-count", type=int, default=2)
    parser.add_argument("--semantic-margin", type=float, default=0.05)
    args = parser.parse_args()

    rows = _read_csv(args.selector_features_csv)
    if not rows:
        raise ValueError(f"{args.selector_features_csv}: no rows")
    labels = [_bool(row.get("oracle_positive")) for row in rows]
    positive_count = int(sum(labels))
    positive_chunks = [int(float(str(row.get("curr_chunk", "-1")))) for row, label in zip(rows, labels) if label]

    best_geometry, geometry_rows = _best_group(rows, labels, GEOMETRY_FEATURES, "geometry")
    best_semantic, semantic_rows = _best_group(rows, labels, SEMANTIC_FEATURES, "semantic")
    best_all, all_rows = _best_group(rows, labels, ALL_FEATURES, "all")
    null = _semantic_row_shuffle_null(
        rows,
        labels,
        features=SEMANTIC_FEATURES,
        permutations=int(args.permutations),
        seed=int(args.random_seed),
    )
    semantic_auc = _float(best_semantic.get("best_auc"))
    geometry_auc = _float(best_geometry.get("best_auc"))
    null_p95 = _float(null.get("best_auc_p95"))
    semantic_margin_vs_geometry = (
        None if not math.isfinite(semantic_auc) or not math.isfinite(geometry_auc)
        else float(semantic_auc - geometry_auc)
    )
    semantic_margin_vs_null_p95 = (
        None if not math.isfinite(semantic_auc) or not math.isfinite(null_p95)
        else float(semantic_auc - null_p95)
    )
    positive_support_pass = positive_count >= int(args.min_positive_count)
    semantic_selector_gate_pass = bool(
        positive_support_pass
        and semantic_margin_vs_geometry is not None
        and semantic_margin_vs_geometry >= float(args.semantic_margin)
        and semantic_margin_vs_null_p95 is not None
        and semantic_margin_vs_null_p95 >= float(args.semantic_margin)
        and bool(best_semantic.get("top5_hit"))
    )
    reason = []
    if not positive_support_pass:
        reason.append(f"positive_count {positive_count} < {int(args.min_positive_count)}")
    if semantic_margin_vs_geometry is None or semantic_margin_vs_geometry < float(args.semantic_margin):
        reason.append(f"semantic margin vs geometry {semantic_margin_vs_geometry} < {float(args.semantic_margin)}")
    if semantic_margin_vs_null_p95 is None or semantic_margin_vs_null_p95 < float(args.semantic_margin):
        reason.append(f"semantic margin vs shuffled-p95 {semantic_margin_vs_null_p95} < {float(args.semantic_margin)}")
    if not bool(best_semantic.get("top5_hit")):
        reason.append("best semantic feature does not hit top5")

    summary = {
        "schema": "acl2_v67_overlap_pair_selector_controls_summary_v1",
        "selector_features_csv": str(args.selector_features_csv),
        "rows": len(rows),
        "positive_count": positive_count,
        "positive_chunks": positive_chunks,
        "min_positive_count": int(args.min_positive_count),
        "best_geometry": best_geometry,
        "best_semantic": best_semantic,
        "best_all": best_all,
        "semantic_row_shuffle_null": null,
        "semantic_margin_vs_geometry": semantic_margin_vs_geometry,
        "semantic_margin_vs_shuffled_p95": semantic_margin_vs_null_p95,
        "semantic_selector_gate_pass": semantic_selector_gate_pass,
        "semantic_selector_gate_fail_reasons": reason,
        "note": "Strict diagnostic. One oracle-positive chunk is insufficient for robust semantic selector claims.",
    }
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "selector_control_feature_auc.csv", geometry_rows + semantic_rows + all_rows)
    (out_dir / "selector_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
