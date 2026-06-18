#!/usr/bin/env python3
"""Pooled/source-aware selector controls for v67 overlap-pair oracle positives."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from diagnose_v67_overlap_pair_selector_controls import (
    ALL_FEATURES,
    GEOMETRY_FEATURES,
    SEMANTIC_FEATURES,
    _best_group,
    _bool,
    _float,
    _read_csv,
    _semantic_row_shuffle_null,
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


def _load_sources(sources: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, path in sources:
        source_rows = _read_csv(path)
        for row in source_rows:
            item = dict(row)
            item["source_label"] = label
            item["selector_features_csv"] = str(path)
            rows.append(item)
    return rows


def _chunk_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return (
        int(float(str(row.get("prev_chunk", "-1")))),
        int(float(str(row.get("curr_chunk", "-1")))),
    )


def _collapse_by_chunk(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_chunk_key(row)].append(row)

    out: List[Dict[str, Any]] = []
    for (prev_chunk, curr_chunk), group in sorted(grouped.items()):
        positives = [row for row in group if _bool(row.get("oracle_positive"))]
        source_labels = [str(row.get("source_label", "")) for row in group]
        positive_source_labels = [str(row.get("source_label", "")) for row in positives]
        item: Dict[str, Any] = {
            "prev_chunk": prev_chunk,
            "curr_chunk": curr_chunk,
            "source_count": len(group),
            "source_labels": ";".join(source_labels),
            "positive_source_count": len(positives),
            "positive_source_labels": ";".join(positive_source_labels),
            "oracle_positive": bool(positives),
            "oracle_best_candidate": ";".join(str(row.get("oracle_best_candidate", "")) for row in positives),
            "oracle_best_mechanism_improvement_mean": _mean(
                [_float(row.get("oracle_best_mechanism_improvement")) for row in positives]
            ),
            "oracle_best_delta_ate_mean": _mean([_float(row.get("oracle_best_delta_ate")) for row in positives]),
            "oracle_best_raw_overlap_improvement_ratio_mean": _mean(
                [_float(row.get("oracle_best_raw_overlap_improvement_ratio")) for row in positives]
            ),
        }
        for feature in ALL_FEATURES:
            item[feature] = _mean([_float(row.get(feature)) for row in group])
        out.append(item)
    return out


def _evaluate(rows: Sequence[Dict[str, Any]], *, permutations: int, seed: int, label: str) -> Dict[str, Any]:
    labels = [_bool(row.get("oracle_positive")) for row in rows]
    best_geometry, geometry_rows = _best_group(rows, labels, GEOMETRY_FEATURES, "geometry")
    best_semantic, semantic_rows = _best_group(rows, labels, SEMANTIC_FEATURES, "semantic")
    best_all, all_rows = _best_group(rows, labels, ALL_FEATURES, "all")
    null = _semantic_row_shuffle_null(
        rows,
        labels,
        features=SEMANTIC_FEATURES,
        permutations=int(permutations),
        seed=int(seed),
    )
    semantic_auc = _float(best_semantic.get("best_auc"))
    geometry_auc = _float(best_geometry.get("best_auc"))
    null_p95 = _float(null.get("best_auc_p95"))
    semantic_margin_vs_geometry = (
        None
        if not math.isfinite(semantic_auc) or not math.isfinite(geometry_auc)
        else float(semantic_auc - geometry_auc)
    )
    semantic_margin_vs_null_p95 = (
        None
        if not math.isfinite(semantic_auc) or not math.isfinite(null_p95)
        else float(semantic_auc - null_p95)
    )
    positive_chunks = sorted({int(float(str(row.get("curr_chunk", "-1")))) for row, y in zip(rows, labels) if y})
    return {
        "label": label,
        "rows": len(rows),
        "positive_rows": int(sum(labels)),
        "positive_chunks": positive_chunks,
        "distinct_positive_chunk_count": len(positive_chunks),
        "best_geometry": best_geometry,
        "best_semantic": best_semantic,
        "best_all": best_all,
        "semantic_row_shuffle_null": null,
        "semantic_margin_vs_geometry": semantic_margin_vs_geometry,
        "semantic_margin_vs_shuffled_p95": semantic_margin_vs_null_p95,
        "feature_auc_rows": geometry_rows + semantic_rows + all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--min-distinct-positive-chunks", type=int, default=2)
    parser.add_argument("--semantic-margin", type=float, default=0.05)
    args = parser.parse_args()

    rows = _load_sources(args.source)
    if not rows:
        raise ValueError("no pooled selector rows")
    chunk_rows = _collapse_by_chunk(rows)
    if not chunk_rows:
        raise ValueError("no chunk-collapsed selector rows")

    row_eval = _evaluate(rows, permutations=int(args.permutations), seed=int(args.random_seed), label="row_level")
    chunk_eval = _evaluate(
        chunk_rows,
        permutations=int(args.permutations),
        seed=int(args.random_seed),
        label="chunk_level_source_collapsed",
    )

    chunk_distinct_pass = chunk_eval["distinct_positive_chunk_count"] >= int(args.min_distinct_positive_chunks)
    chunk_semantic_margin_geometry = chunk_eval["semantic_margin_vs_geometry"]
    chunk_semantic_margin_null = chunk_eval["semantic_margin_vs_shuffled_p95"]
    gate_pass = bool(
        chunk_distinct_pass
        and chunk_semantic_margin_geometry is not None
        and chunk_semantic_margin_geometry >= float(args.semantic_margin)
        and chunk_semantic_margin_null is not None
        and chunk_semantic_margin_null >= float(args.semantic_margin)
        and bool(chunk_eval["best_semantic"].get("top5_hit"))
    )
    reasons: List[str] = []
    if not chunk_distinct_pass:
        reasons.append(
            "distinct_positive_chunk_count "
            f"{chunk_eval['distinct_positive_chunk_count']} < {int(args.min_distinct_positive_chunks)}"
        )
    if chunk_semantic_margin_geometry is None or chunk_semantic_margin_geometry < float(args.semantic_margin):
        reasons.append(
            f"chunk semantic margin vs geometry {chunk_semantic_margin_geometry} < {float(args.semantic_margin)}"
        )
    if chunk_semantic_margin_null is None or chunk_semantic_margin_null < float(args.semantic_margin):
        reasons.append(
            f"chunk semantic margin vs shuffled-p95 {chunk_semantic_margin_null} < {float(args.semantic_margin)}"
        )
    if not bool(chunk_eval["best_semantic"].get("top5_hit")):
        reasons.append("chunk best semantic feature does not hit top5")

    summary = {
        "schema": "acl2_v67_overlap_pair_selector_pooled_controls_summary_v1",
        "sources": [{"label": label, "selector_features_csv": str(path)} for label, path in args.source],
        "row_level": {k: v for k, v in row_eval.items() if k != "feature_auc_rows"},
        "chunk_level_source_collapsed": {k: v for k, v in chunk_eval.items() if k != "feature_auc_rows"},
        "min_distinct_positive_chunks": int(args.min_distinct_positive_chunks),
        "semantic_margin": float(args.semantic_margin),
        "pooled_semantic_selector_gate_pass": gate_pass,
        "pooled_semantic_selector_gate_fail_reasons": reasons,
        "note": (
            "Gate uses source-collapsed chunk-level positives so repeated source variants of the same "
            "chunk cannot be counted as independent selector evidence."
        ),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "pooled_selector_rows.csv", rows)
    _write_csv(out_dir / "pooled_chunk_features.csv", chunk_rows)
    for eval_obj in (row_eval, chunk_eval):
        for item in eval_obj["feature_auc_rows"]:
            item["eval_level"] = eval_obj["label"]
    _write_csv(out_dir / "pooled_feature_auc.csv", row_eval["feature_auc_rows"] + chunk_eval["feature_auc_rows"])
    (out_dir / "pooled_selector_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
