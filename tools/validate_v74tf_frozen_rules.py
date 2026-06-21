#!/usr/bin/env python3
"""Phase 2 frozen diagnostic-rule validation for ACL2 v74-TF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v73_semantic_memory_common import TARGET_CHUNKS, auc_binary, parse_chunks, read_csv, safe_float, topk_precision, utc_now, write_csv, write_json
from v74tf_common import REPORT_ROOT, V73_REPORT_ROOT, parse_seqs


FORMULAS = [
    {
        "formula": "F_short",
        "target_label": "Y_short",
        "terms": [("global_k_layer5_gram_motion", 1.0), ("sky_context_ratio", 1.0), ("radio_static_mean", -1.0)],
        "geometry_feature": "global_k_layer5_gram_motion",
        "semantic_features": ["sky_context_ratio"],
        "radio_features": ["radio_static_mean"],
    },
    {
        "formula": "F_mid",
        "target_label": "Y_mid",
        "terms": [("D_geo_mean_patch", 1.0), ("dynamic_thing_ratio", 1.0), ("stable_structure_ratio", -1.0)],
        "geometry_feature": "D_geo_mean_patch",
        "semantic_features": ["dynamic_thing_ratio", "stable_structure_ratio"],
        "radio_features": [],
    },
    {
        "formula": "F_scale",
        "target_label": "Y_scale_drift",
        "terms": [("raw_overlap_residual_rmse", 1.0), ("dynamic_thing_ratio", 1.0), ("radio_static_mean", -1.0)],
        "geometry_feature": "raw_overlap_residual_rmse",
        "semantic_features": ["dynamic_thing_ratio"],
        "radio_features": ["radio_static_mean"],
    },
]


def _std(values: list[float]) -> float:
    std = float(np.std(np.asarray(values, dtype=np.float64)))
    return std if std > 1e-12 else 1.0


def _score_row(row: dict[str, Any], stats: dict[str, tuple[float, float]], terms: Sequence[tuple[str, float]]) -> float | None:
    score = 0.0
    for feature, coef in terms:
        val = safe_float(row.get(feature))
        if val is None or feature not in stats:
            return None
        mean, std = stats[feature]
        score += float(coef) * ((val - mean) / std)
    return float(score)


def _fold_scores(rows: list[dict[str, Any]], terms: Sequence[tuple[str, float]], folds: list[list[int]]) -> list[float | None]:
    scores: list[float | None] = [None for _ in rows]
    for test_idx in folds:
        test_set = set(test_idx)
        train_idx = [i for i in range(len(rows)) if i not in test_set]
        stats: dict[str, tuple[float, float]] = {}
        for feature, _coef in terms:
            vals = [safe_float(rows[i].get(feature)) for i in train_idx]
            finite = [float(v) for v in vals if v is not None]
            if finite:
                stats[feature] = (float(np.mean(finite)), _std(finite))
        for i in test_idx:
            scores[i] = _score_row(rows[i], stats, terms)
    return scores


def _loocv_scores(rows: list[dict[str, Any]], terms: Sequence[tuple[str, float]]) -> list[float | None]:
    return _fold_scores(rows, terms, [[i] for i in range(len(rows))])


def _best_auc(auc: float | None) -> float | None:
    return None if auc is None else float(max(auc, 1.0 - auc))


def _shuffle_p95(
    rows: list[dict[str, Any]],
    formula: dict[str, Any],
    labels: Sequence[Any],
    features: Sequence[str],
    seed: int,
    repeats: int,
) -> float | None:
    if not features:
        return None
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(repeats):
        shuffled = [dict(row) for row in rows]
        for feature in features:
            vals = [row.get(feature) for row in shuffled]
            rng.shuffle(vals)
            for row, val in zip(shuffled, vals):
                row[feature] = val
        auc = auc_binary(_loocv_scores(shuffled, formula["terms"]), labels)
        if auc is not None:
            aucs.append(float(auc))
    return float(np.quantile(np.asarray(aucs, dtype=np.float64), 0.95)) if aucs else None


def _label_permutation_p95(scores: Sequence[Any], labels: Sequence[Any], seed: int, repeats: int) -> float | None:
    rng = np.random.default_rng(seed)
    valid = [(safe_float(s), int(str(y))) for s, y in zip(scores, labels) if safe_float(s) is not None and str(y) in {"0", "1"}]
    if not valid:
        return None
    score_vals = [float(s) for s, _ in valid]
    label_vals = np.asarray([y for _, y in valid], dtype=np.int64)
    aucs: list[float] = []
    for _ in range(repeats):
        auc = auc_binary(score_vals, rng.permutation(label_vals).tolist())
        if auc is not None:
            aucs.append(float(auc))
    return float(np.quantile(np.asarray(aucs, dtype=np.float64), 0.95)) if aucs else None


def _read_seq_rows(seq: str, path: Path, target_chunks: list[int]) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        return [], "blocked_missing_feature_csv"
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    target_set = set(target_chunks)
    for row in rows:
        try:
            chunk = int(row.get("chunk_id", -1))
        except (TypeError, ValueError):
            continue
        if seq == "01" and chunk not in target_set:
            continue
        row = dict(row)
        row["seq"] = seq
        row["chunk_id"] = chunk
        out.append(row)
    out.sort(key=lambda item: int(item["chunk_id"]))
    return out, "ok" if out else "blocked_no_matching_rows"


def _validate_seq(seq: str, rows: list[dict[str, Any]], seed: int, repeats: int, shuffles: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for fidx, formula in enumerate(FORMULAS):
        labels = [row.get(formula["target_label"]) for row in rows]
        scores = _loocv_scores(rows, formula["terms"])
        geom_scores = _loocv_scores(rows, [(formula["geometry_feature"], 1.0)])
        auc = auc_binary(scores, labels)
        geom_auc = auc_binary(geom_scores, labels)
        geom_best = _best_auc(geom_auc)
        top5 = topk_precision(scores, labels, k=5, higher_is_positive=True)
        label_p95 = _label_permutation_p95(scores, labels, seed + fidx * 17, repeats)
        semantic_p95 = _shuffle_p95(rows, formula, labels, formula["semantic_features"], seed + fidx * 31, shuffles)
        radio_p95 = _shuffle_p95(rows, formula, labels, formula["radio_features"], seed + fidx * 43, shuffles)
        confidence_p95 = _shuffle_p95(rows, formula, labels, ["semantic_confidence_mean"], seed + fidx * 53, shuffles)
        shuffle_controls = [x for x in (label_p95, semantic_p95, radio_p95, confidence_p95) if x is not None]
        max_shuffle = max(shuffle_controls) if shuffle_controls else None
        margin_vs_geometry = None if auc is None or geom_best is None else float(auc - geom_best)
        margin_vs_shuffle = None if auc is None or max_shuffle is None else float(auc - max_shuffle)
        positive_count = sum(1 for value in labels if str(value) == "1")
        deployable_01 = bool(
            seq == "01"
            and auc is not None
            and top5 is not None
            and margin_vs_geometry is not None
            and margin_vs_shuffle is not None
            and auc >= 0.70
            and top5 >= 0.40
            and margin_vs_geometry >= 0.05
            and margin_vs_shuffle >= 0.05
        )
        deployable_09 = bool(seq == "09" and auc is not None and auc >= 0.60)
        gate_row = {
            "seq": seq,
            "formula": formula["formula"],
            "target_label": formula["target_label"],
            "terms": " + ".join(feature if coef > 0 else f"- {feature}" for feature, coef in formula["terms"]).replace("+ -", "-"),
            "row_count": len(rows),
            "positive_count": positive_count,
            "LOOCV_AUC": auc,
            "top5_precision": top5,
            "geometry_only_AUC": geom_auc,
            "geometry_only_best_AUC": geom_best,
            "label_shuffle_p95": label_p95,
            "semantic_shuffle_p95": semantic_p95,
            "radio_shuffle_p95": radio_p95,
            "confidence_shuffle_p95": confidence_p95,
            "max_shuffle_p95": max_shuffle,
            "margin_vs_geometry": margin_vs_geometry,
            "margin_vs_shuffle": margin_vs_shuffle,
            "deployable_as_01_rule_component": deployable_01,
            "nonreversing_09_validation": deployable_09,
            "training_free_note": "LOOCV/scoring only; no fitted deployment weights or thresholds.",
        }
        gate_rows.append(gate_row)
        for row, score, geom_score, label in zip(rows, scores, geom_scores, labels):
            score_rows.append(
                {
                    "seq": seq,
                    "formula": formula["formula"],
                    "target_label": formula["target_label"],
                    "chunk_id": row.get("chunk_id"),
                    "label": label,
                    "loocv_score": score,
                    "geometry_only_score": geom_score,
                }
            )
    return gate_rows, score_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default="01,09")
    parser.add_argument("--target-chunks", default=",".join(map(str, TARGET_CHUNKS)))
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "phase2_frozen_rule_validation")
    parser.add_argument("--seq01-features-csv", type=Path, default=V73_REPORT_ROOT / "phase3_semantic_explanation" / "semantic_geometry_features_by_chunk.csv")
    parser.add_argument("--seq09-features-csv", type=Path, default=REPORT_ROOT / "phase1_multiseq_scale_drift_ledger" / "seq09_semantic_geometry_features_by_chunk.csv")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--shuffles", type=int, default=400)
    parser.add_argument("--seed", type=int, default=74)
    args = parser.parse_args()

    target_chunks = parse_chunks(args.target_chunks)
    seq_paths = {"01": args.seq01_features_csv, "09": args.seq09_features_csv}
    all_gate_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    seq_status: dict[str, Any] = {}
    for seq in parse_seqs(args.seqs):
        rows, status = _read_seq_rows(seq, seq_paths.get(seq, Path("__missing__")), target_chunks)
        seq_status[seq] = {"status": status, "features_csv": str(seq_paths.get(seq, "")), "rows": len(rows)}
        if status != "ok":
            continue
        gate_rows, score_rows = _validate_seq(seq, rows, int(args.seed), int(args.permutations), int(args.shuffles))
        all_gate_rows.extend(gate_rows)
        all_score_rows.extend(score_rows)
    deployable_01 = [row["formula"] for row in all_gate_rows if row.get("seq") == "01" and row.get("deployable_as_01_rule_component")]
    validated_09 = [row["formula"] for row in all_gate_rows if row.get("seq") == "09" and row.get("nonreversing_09_validation")]
    summary = {
        "schema": "acl2_v74tf_phase2_frozen_rule_validation_v1",
        "created_at": utc_now(),
        "seq_status": seq_status,
        "deployable_01_formulas": deployable_01,
        "validated_09_formulas": validated_09,
        "phase2_cross_seq_gate_pass": bool(deployable_01 and validated_09 and set(deployable_01).intersection(validated_09)),
        "blocked_reason": "" if deployable_01 and validated_09 else "Missing or failing KITTI09 fixed-formula validation prevents cross-sequence rule promotion.",
        "training_free_compliance": "No selector/classifier/calibrator is trained; outputs are diagnostic rule-component validation only.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "frozen_rule_validation_rows.csv", all_gate_rows)
    write_csv(args.out_dir / "frozen_rule_scores_by_chunk.csv", all_score_rows)
    write_json(args.out_dir / "frozen_rule_validation_summary.json", summary)
    print(
        {
            "out_dir": str(args.out_dir),
            "deployable_01_formulas": deployable_01,
            "validated_09_formulas": validated_09,
            "phase2_cross_seq_gate_pass": summary["phase2_cross_seq_gate_pass"],
        }
    )


if __name__ == "__main__":
    main()
