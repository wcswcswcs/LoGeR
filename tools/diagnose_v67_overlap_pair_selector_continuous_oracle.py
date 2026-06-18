#!/usr/bin/env python3
"""Continuous oracle-target selector diagnostic for v67 overlap-pair features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from diagnose_v67_overlap_pair_selector_controls import (
    ALL_FEATURES,
    GEOMETRY_FEATURES,
    SEMANTIC_FEATURES,
    _auc,
    _bool,
    _float,
    _percentile,
    _read_csv,
    _write_csv,
)


def _parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be LABEL=selector_features.csv")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--source label is empty")
    return label, Path(path)


def _mean(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [float("nan")] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0.0 or dy == 0.0:
        return None
    return float(num / (dx * dy))


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return None
    rx = _rankdata([x for x, _ in pairs])
    ry = _rankdata([y for _, y in pairs])
    return _pearson(rx, ry)


def _load_chunk_rows(sources: Sequence[Tuple[str, Path]], target_column: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for label, path in sources:
        for row in _read_csv(path):
            item = dict(row)
            item["source_label"] = label
            item["selector_features_csv"] = str(path)
            key = (
                int(float(str(row.get("prev_chunk", "-1")))),
                int(float(str(row.get("curr_chunk", "-1")))),
            )
            grouped[key].append(item)

    out: List[Dict[str, Any]] = []
    for (prev_chunk, curr_chunk), group in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "prev_chunk": prev_chunk,
            "curr_chunk": curr_chunk,
            "source_count": len(group),
            "source_labels": ";".join(str(item.get("source_label", "")) for item in group),
            "oracle_positive_any": any(_bool(item.get("oracle_positive")) for item in group),
            "target_column": target_column,
            "target_mean": _mean([_float(item.get(target_column)) for item in group]),
            "target_max": max(
                [_float(item.get(target_column)) for item in group if math.isfinite(_float(item.get(target_column)))]
                or [float("nan")]
            ),
        }
        for feature in ALL_FEATURES:
            row[feature] = _mean([_float(item.get(feature)) for item in group])
        out.append(row)
    return out


def _topk_labels(targets: Sequence[float], top_k: int) -> List[bool]:
    finite = [idx for idx, value in enumerate(targets) if math.isfinite(float(value))]
    finite.sort(key=lambda idx: float(targets[idx]), reverse=True)
    labels = [False] * len(targets)
    for idx in finite[: int(top_k)]:
        labels[idx] = True
    return labels


def _evaluate_feature(
    rows: Sequence[Dict[str, Any]],
    targets: Sequence[float],
    labels: Sequence[bool],
    feature: str,
) -> Dict[str, Any]:
    values = [_float(row.get(feature)) for row in rows]
    auc_high = _auc(values, labels)
    auc_low = _auc([-v for v in values], labels)
    if auc_high is None or auc_low is None:
        best_auc = None
        auc_direction = ""
    elif auc_high >= auc_low:
        best_auc = auc_high
        auc_direction = "higher"
    else:
        best_auc = auc_low
        auc_direction = "lower"

    rho = _spearman(values, targets)
    if rho is None:
        abs_spearman = None
        spearman_direction = ""
    else:
        abs_spearman = abs(float(rho))
        spearman_direction = "higher" if rho >= 0.0 else "lower"

    return {
        "feature": feature,
        "topk_auc_higher": auc_high,
        "topk_auc_lower": auc_low,
        "best_topk_auc": best_auc,
        "topk_auc_direction": auc_direction,
        "spearman_rho": rho,
        "abs_spearman_rho": abs_spearman,
        "spearman_direction": spearman_direction,
    }


def _best_group(
    rows: Sequence[Dict[str, Any]],
    targets: Sequence[float],
    labels: Sequence[bool],
    features: Sequence[str],
    group: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    eval_rows = []
    for feature in features:
        if feature not in rows[0]:
            continue
        item = _evaluate_feature(rows, targets, labels, feature)
        item["feature_group"] = group
        eval_rows.append(item)
    eval_rows.sort(
        key=lambda row: (
            -1.0 if row.get("best_topk_auc") is None else -float(row["best_topk_auc"]),
            -1.0 if row.get("abs_spearman_rho") is None else -float(row["abs_spearman_rho"]),
            row["feature"],
        )
    )
    return (eval_rows[0] if eval_rows else {}), eval_rows


def _semantic_shuffle_null(
    rows: Sequence[Dict[str, Any]],
    targets: Sequence[float],
    labels: Sequence[bool],
    *,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    base_values = {feature: [_float(row.get(feature)) for row in rows] for feature in SEMANTIC_FEATURES if feature in rows[0]}
    best_auc_values: List[float] = []
    best_abs_spearman_values: List[float] = []
    for _ in range(int(permutations)):
        perm = list(range(len(rows)))
        rng.shuffle(perm)
        best_auc = float("nan")
        best_abs_spearman = float("nan")
        for values in base_values.values():
            shuffled = [values[idx] for idx in perm]
            auc_high = _auc(shuffled, labels)
            auc_low = _auc([-v for v in shuffled], labels)
            if auc_high is not None and auc_low is not None:
                auc = max(float(auc_high), float(auc_low))
                if not math.isfinite(best_auc) or auc > best_auc:
                    best_auc = auc
            rho = _spearman(shuffled, targets)
            if rho is not None:
                arho = abs(float(rho))
                if not math.isfinite(best_abs_spearman) or arho > best_abs_spearman:
                    best_abs_spearman = arho
        if math.isfinite(best_auc):
            best_auc_values.append(float(best_auc))
        if math.isfinite(best_abs_spearman):
            best_abs_spearman_values.append(float(best_abs_spearman))
    return {
        "permutations": int(permutations),
        "best_topk_auc_p50": _percentile(best_auc_values, 0.50),
        "best_topk_auc_p95": _percentile(best_auc_values, 0.95),
        "best_topk_auc_p99": _percentile(best_auc_values, 0.99),
        "best_abs_spearman_p50": _percentile(best_abs_spearman_values, 0.50),
        "best_abs_spearman_p95": _percentile(best_abs_spearman_values, 0.95),
        "best_abs_spearman_p99": _percentile(best_abs_spearman_values, 0.99),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-column", default="oracle_best_mechanism_improvement")
    parser.add_argument("--target-aggregate", choices=["mean", "max"], default="mean")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--semantic-margin", type=float, default=0.05)
    args = parser.parse_args()

    rows = _load_chunk_rows(args.source, args.target_column)
    if not rows:
        raise ValueError("no chunk rows")
    target_key = "target_mean" if args.target_aggregate == "mean" else "target_max"
    targets = [_float(row.get(target_key)) for row in rows]
    labels = _topk_labels(targets, int(args.top_k))

    best_geometry, geometry_rows = _best_group(rows, targets, labels, GEOMETRY_FEATURES, "geometry")
    best_semantic, semantic_rows = _best_group(rows, targets, labels, SEMANTIC_FEATURES, "semantic")
    best_all, all_rows = _best_group(rows, targets, labels, ALL_FEATURES, "all")
    null = _semantic_shuffle_null(
        rows,
        targets,
        labels,
        permutations=int(args.permutations),
        seed=int(args.random_seed),
    )

    semantic_auc = _float(best_semantic.get("best_topk_auc"))
    geometry_auc = _float(best_geometry.get("best_topk_auc"))
    null_auc_p95 = _float(null.get("best_topk_auc_p95"))
    semantic_margin_vs_geometry = (
        None
        if not math.isfinite(semantic_auc) or not math.isfinite(geometry_auc)
        else float(semantic_auc - geometry_auc)
    )
    semantic_margin_vs_null_p95 = (
        None
        if not math.isfinite(semantic_auc) or not math.isfinite(null_auc_p95)
        else float(semantic_auc - null_auc_p95)
    )
    gate_pass = bool(
        semantic_margin_vs_geometry is not None
        and semantic_margin_vs_geometry >= float(args.semantic_margin)
        and semantic_margin_vs_null_p95 is not None
        and semantic_margin_vs_null_p95 >= float(args.semantic_margin)
    )
    reasons: List[str] = []
    if semantic_margin_vs_geometry is None or semantic_margin_vs_geometry < float(args.semantic_margin):
        reasons.append(f"semantic top-k AUC margin vs geometry {semantic_margin_vs_geometry} < {float(args.semantic_margin)}")
    if semantic_margin_vs_null_p95 is None or semantic_margin_vs_null_p95 < float(args.semantic_margin):
        reasons.append(f"semantic top-k AUC margin vs shuffled-p95 {semantic_margin_vs_null_p95} < {float(args.semantic_margin)}")

    target_order = sorted(range(len(rows)), key=lambda idx: targets[idx], reverse=True)
    top_targets = [
        {
            "prev_chunk": int(rows[idx]["prev_chunk"]),
            "curr_chunk": int(rows[idx]["curr_chunk"]),
            target_key: targets[idx],
            "oracle_positive_any": bool(rows[idx].get("oracle_positive_any")),
        }
        for idx in target_order[: int(args.top_k)]
    ]
    for item in geometry_rows + semantic_rows + all_rows:
        item["target_key"] = target_key
        item["top_k"] = int(args.top_k)

    summary = {
        "schema": "acl2_v67_overlap_pair_selector_continuous_oracle_summary_v1",
        "sources": [{"label": label, "selector_features_csv": str(path)} for label, path in args.source],
        "rows": len(rows),
        "target_column": args.target_column,
        "target_aggregate": args.target_aggregate,
        "target_key": target_key,
        "top_k": int(args.top_k),
        "top_target_chunks": top_targets,
        "best_geometry": best_geometry,
        "best_semantic": best_semantic,
        "best_all": best_all,
        "semantic_shuffle_null": null,
        "semantic_margin_vs_geometry": semantic_margin_vs_geometry,
        "semantic_margin_vs_shuffled_p95": semantic_margin_vs_null_p95,
        "continuous_semantic_selector_gate_pass": gate_pass,
        "continuous_semantic_selector_gate_fail_reasons": reasons,
        "note": (
            "Continuous diagnostic only. It tests whether semantic features predict the oracle mechanism "
            "score across chunks; it does not by itself authorize O4 or full-method claims."
        ),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "continuous_chunk_features.csv", rows)
    _write_csv(out_dir / "continuous_feature_scores.csv", geometry_rows + semantic_rows + all_rows)
    (out_dir / "continuous_oracle_selector_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
