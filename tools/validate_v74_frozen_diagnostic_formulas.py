#!/usr/bin/env python3
"""Phase 1 frozen diagnostic formula validation for ACL2 v74."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v73_semantic_memory_common import (
    TARGET_CHUNKS,
    auc_binary,
    parse_chunks,
    read_csv,
    safe_float,
    topk_precision,
    utc_now,
    write_csv,
    write_json,
)


DEFAULT_FEATURES = Path(
    "results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final/"
    "phase3_semantic_explanation/semantic_geometry_features_by_chunk.csv"
)
DEFAULT_OUT = Path(
    "results/kitti01_hmc_v2/acl2_v74_diagnostic_to_action_semantic_memory_control/"
    "report_final/phase1_frozen_formula_validation"
)


FORMULAS = [
    {
        "formula": "F_short",
        "target_label": "Y_short",
        "terms": [
            ("global_k_layer5_gram_motion", 1.0),
            ("sky_context_ratio", 1.0),
            ("radio_static_mean", -1.0),
        ],
        "geometry_feature": "global_k_layer5_gram_motion",
        "semantic_features": ["sky_context_ratio", "radio_static_mean"],
    },
    {
        "formula": "F_mid",
        "target_label": "Y_mid",
        "terms": [
            ("D_geo_mean_patch", 1.0),
            ("dynamic_thing_ratio", 1.0),
            ("stable_structure_ratio", -1.0),
        ],
        "geometry_feature": "D_geo_mean_patch",
        "semantic_features": ["dynamic_thing_ratio", "stable_structure_ratio"],
    },
    {
        "formula": "F_scale",
        "target_label": "Y_scale_drift",
        "terms": [
            ("raw_overlap_residual_rmse", 1.0),
            ("dynamic_thing_ratio", 1.0),
            ("radio_static_mean", -1.0),
        ],
        "geometry_feature": "raw_overlap_residual_rmse",
        "semantic_features": ["dynamic_thing_ratio", "radio_static_mean"],
    },
]


def _is_finite(value: Any) -> bool:
    return safe_float(value) is not None


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
        train_idx = [i for i in range(len(rows)) if i not in set(test_idx)]
        stats: dict[str, tuple[float, float]] = {}
        for feature, _coef in terms:
            vals = [safe_float(rows[i].get(feature)) for i in train_idx]
            finite = [float(v) for v in vals if v is not None]
            if not finite:
                continue
            stats[feature] = (float(np.mean(finite)), _std(finite))
        for i in test_idx:
            scores[i] = _score_row(rows[i], stats, terms)
    return scores


def _loocv_scores(rows: list[dict[str, Any]], terms: Sequence[tuple[str, float]]) -> list[float | None]:
    return _fold_scores(rows, terms, [[i] for i in range(len(rows))])


def _reset_block_scores(rows: list[dict[str, Any]], terms: Sequence[tuple[str, float]]) -> list[float | None]:
    blocks: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key = str(row.get("reset_relative_index", "unknown"))
        blocks.setdefault(key, []).append(i)
    return _fold_scores(rows, terms, list(blocks.values()))


def _auc(scores: Sequence[Any], labels: Sequence[Any]) -> float | None:
    return auc_binary(scores, labels)


def _best_auc_from_auc(auc: float | None) -> float | None:
    if auc is None:
        return None
    return float(max(auc, 1.0 - auc))


def _p95(values: list[float]) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95)) if values else None


def _label_permutation_p95(scores: Sequence[Any], labels: Sequence[Any], seed: int, repeats: int) -> float | None:
    rng = np.random.default_rng(seed)
    valid = [(safe_float(s), int(str(y))) for s, y in zip(scores, labels) if safe_float(s) is not None and str(y) in {"0", "1"}]
    if not valid:
        return None
    score_vals = [s for s, _ in valid]
    label_vals = np.asarray([y for _, y in valid], dtype=np.int64)
    aucs: list[float] = []
    for _ in range(repeats):
        perm = rng.permutation(label_vals)
        auc = auc_binary(score_vals, perm.tolist())
        if auc is not None:
            aucs.append(float(auc))
    return _p95(aucs)


def _semantic_shuffle_p95(
    rows: list[dict[str, Any]],
    formula: dict[str, Any],
    labels: Sequence[Any],
    seed: int,
    repeats: int,
) -> float | None:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    terms = formula["terms"]
    semantic_features = formula["semantic_features"]
    for _ in range(repeats):
        shuffled = [dict(row) for row in rows]
        for feature in semantic_features:
            vals = [row.get(feature) for row in shuffled]
            rng.shuffle(vals)
            for row, val in zip(shuffled, vals):
                row[feature] = val
        scores = _loocv_scores(shuffled, terms)
        auc = auc_binary(scores, labels)
        if auc is not None:
            aucs.append(float(auc))
    return _p95(aucs)


def _single_sign_flip_best_auc(rows: list[dict[str, Any]], terms: Sequence[tuple[str, float]], labels: Sequence[Any]) -> float | None:
    best: float | None = None
    for pos in range(len(terms)):
        flipped = [(feature, -coef if i == pos else coef) for i, (feature, coef) in enumerate(terms)]
        auc = auc_binary(_loocv_scores(rows, flipped), labels)
        best_auc = _best_auc_from_auc(auc)
        if best_auc is not None and (best is None or best_auc > best):
            best = best_auc
    return best


def _target_rows(path: Path, chunks: list[int]) -> list[dict[str, Any]]:
    rows = read_csv(path)
    chunk_set = set(chunks)
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            chunk = int(row.get("chunk_id", -1))
        except (TypeError, ValueError):
            continue
        if chunk in chunk_set:
            row = dict(row)
            row["chunk_id"] = chunk
            out.append(row)
    out.sort(key=lambda item: int(item["chunk_id"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--target-chunks", default=",".join(map(str, TARGET_CHUNKS)))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--semantic-shuffles", type=int, default=400)
    parser.add_argument("--seed", type=int, default=74)
    args = parser.parse_args()

    chunks = parse_chunks(args.target_chunks)
    rows = _target_rows(args.features_csv, chunks)
    score_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for fidx, formula in enumerate(FORMULAS):
        label_key = str(formula["target_label"])
        labels = [row.get(label_key) for row in rows]
        loocv_scores = _loocv_scores(rows, formula["terms"])
        reset_scores = _reset_block_scores(rows, formula["terms"])
        loocv_auc = _auc(loocv_scores, labels)
        reset_auc = _auc(reset_scores, labels)
        top5 = topk_precision(loocv_scores, labels, k=5, higher_is_positive=True)
        perm_p95 = _label_permutation_p95(loocv_scores, labels, args.seed + fidx * 17, args.permutations)
        sem_shuffle_p95 = _semantic_shuffle_p95(rows, formula, labels, args.seed + fidx * 31, args.semantic_shuffles)
        geom_scores = _loocv_scores(rows, [(formula["geometry_feature"], 1.0)])
        geom_auc = _auc(geom_scores, labels)
        geom_best_auc = _best_auc_from_auc(geom_auc)
        all_sign_flip_auc = _auc([-s if s is not None else None for s in loocv_scores], labels)
        single_flip_best_auc = _single_sign_flip_best_auc(rows, formula["terms"], labels)
        positive_count = sum(1 for value in labels if str(value) == "1")
        margin_vs_geometry = None if loocv_auc is None or geom_best_auc is None else float(loocv_auc - geom_best_auc)
        margin_vs_sem_shuffle = None if loocv_auc is None or sem_shuffle_p95 is None else float(loocv_auc - sem_shuffle_p95)
        deployable = bool(
            loocv_auc is not None
            and top5 is not None
            and margin_vs_geometry is not None
            and margin_vs_sem_shuffle is not None
            and loocv_auc >= 0.70
            and top5 >= 0.40
            and margin_vs_geometry >= 0.05
            and margin_vs_sem_shuffle >= 0.05
        )
        gate_row = {
            "formula": formula["formula"],
            "target_label": label_key,
            "terms": " + ".join(
                feature if coef > 0 else f"- {feature}" for feature, coef in formula["terms"]
            ).replace("+ -", "-"),
            "LOOCV_AUC": loocv_auc,
            "reset_block_AUC": reset_auc,
            "top5_precision": top5,
            "permutation_p95": perm_p95,
            "geometry_only_AUC": geom_auc,
            "geometry_only_best_AUC": geom_best_auc,
            "semantic_shuffle_p95": sem_shuffle_p95,
            "margin_vs_geometry": margin_vs_geometry,
            "margin_vs_semantic_shuffle": margin_vs_sem_shuffle,
            "positive_count": positive_count,
            "target_count": len(labels),
            "sign_flip_all_AUC": all_sign_flip_auc,
            "single_feature_sign_flip_best_AUC": single_flip_best_auc,
            "deployable_as_diagnostic": deployable,
            "gate_rule": "LOOCV_AUC>=0.70 and top5>=0.40 and margin_vs_geometry>=0.05 and margin_vs_semantic_shuffle>=0.05",
        }
        gate_rows.append(gate_row)
        for row, score, reset_score, geom_score, label in zip(rows, loocv_scores, reset_scores, geom_scores, labels):
            score_rows.append(
                {
                    "formula": formula["formula"],
                    "target_label": label_key,
                    "chunk_id": row.get("chunk_id"),
                    "label": label,
                    "loocv_score": score,
                    "reset_block_score": reset_score,
                    "geometry_only_score": geom_score,
                    "reset_relative_index": row.get("reset_relative_index"),
                }
            )

    summary = {
        "schema": "acl2_v74_phase1_frozen_formula_validation_v1",
        "created_at": utc_now(),
        "features_csv": str(args.features_csv),
        "target_chunks": chunks,
        "target_rows": len(rows),
        "permutations": args.permutations,
        "semantic_shuffles": args.semantic_shuffles,
        "seed": args.seed,
        "any_deployable_as_diagnostic": any(bool(row.get("deployable_as_diagnostic")) for row in gate_rows),
        "deployable_formulas": [row["formula"] for row in gate_rows if row.get("deployable_as_diagnostic")],
        "gate_rows": gate_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "frozen_formula_validation_rows.csv", gate_rows)
    write_csv(args.out_dir / "frozen_formula_scores_by_chunk.csv", score_rows)
    write_json(args.out_dir / "frozen_formula_validation_summary.json", summary)
    print(
        {
            "out_dir": str(args.out_dir),
            "target_rows": len(rows),
            "deployable_formulas": summary["deployable_formulas"],
            "any_deployable_as_diagnostic": summary["any_deployable_as_diagnostic"],
        }
    )


if __name__ == "__main__":
    main()
