from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v28_proposal_selection import _labels_from_selected, _load_gt_labels, _rows_for_variant
from tools.run_v34_3d_identity_routes import (
    DIAGNOSTIC_POLICY,
    LOCAL_GATE,
    _gate_status,
    _mean,
    _proposal_score,
    _read_json,
    _row_core_set,
    _row_risk,
    _rows_by_scene,
    _safe_float,
    _score_kwargs,
    _write_json,
)


PAIR_FEATURE_NAMES = [
    "log_co_count",
    "score_mean",
    "score_max",
    "inv_size_sum",
    "risk_min",
    "mask_ratio",
    "d4rt_ratio",
    "temporal_ratio",
    "cannot_min",
    "visible_negative_min",
]

FEATURE_ABLATIONS = {
    "full": set(),
    "no_d4rt": {"d4rt_ratio"},
    "no_mask": {"mask_ratio"},
    "no_temporal": {"temporal_ratio"},
    "no_negative": {"cannot_min", "visible_negative_min"},
}


def _pair_row_type(proposal_type: str) -> str:
    if proposal_type.startswith(("R8_", "R9_", "R10_", "R12_")):
        return "temporal"
    if proposal_type.startswith(("R3_", "R5_")):
        return "d4rt"
    return "mask"


def _build_pair_features(
    scene_rows: list[dict[str, Any]],
    *,
    max_core_tubes: int,
    max_proposal_rows: int,
    min_proposal_score: float,
) -> dict[tuple[int, int], list[float]]:
    candidates: list[tuple[float, dict[str, Any], list[int]]] = []
    for row in _rows_for_variant(scene_rows, "P4_greedy_set_packing"):
        core = sorted(_row_core_set(row))
        if len(core) < 2 or len(core) > int(max_core_tubes):
            continue
        score = _proposal_score(row, **_score_kwargs())
        if score < float(min_proposal_score):
            continue
        candidates.append((float(score), row, core))
    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    candidates = candidates[: int(max_proposal_rows)]

    pair_stats: dict[tuple[int, int], list[float]] = {}
    for score, row, core in candidates:
        size = float(len(core))
        risk = float(_row_risk(row))
        proposal_type = str(row.get("proposal_type") or "")
        kind = _pair_row_type(proposal_type)
        cannot = _safe_float(row.get("same_frame_cannot_link_rate"), 999.0) or 999.0
        visible_negative = _safe_float(row.get("visible_outside_negative_rate"), 999.0) or 999.0
        for a, b in itertools.combinations(core, 2):
            key = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            stats = pair_stats.get(key)
            if stats is None:
                stats = [0.0, 0.0, -999.0, 0.0, 999.0, 0.0, 0.0, 0.0, 999.0, 999.0]
                pair_stats[key] = stats
            stats[0] += 1.0
            stats[1] += score
            stats[2] = max(stats[2], score)
            stats[3] += 1.0 / max(size, 1.0)
            stats[4] = min(stats[4], risk)
            if kind == "mask":
                stats[5] += 1.0
            elif kind == "d4rt":
                stats[6] += 1.0
            else:
                stats[7] += 1.0
            stats[8] = min(stats[8], cannot)
            stats[9] = min(stats[9], visible_negative)

    out: dict[tuple[int, int], list[float]] = {}
    for key, stats in pair_stats.items():
        count = max(stats[0], 1.0)
        out[key] = [
            float(np.log1p(count)),
            float(stats[1] / count),
            float(stats[2]),
            float(stats[3]),
            float(stats[4] if stats[4] < 999.0 else 0.0),
            float(stats[5] / count),
            float(stats[6] / count),
            float(stats[7] / count),
            float(stats[8] if stats[8] < 999.0 else 0.0),
            float(stats[9] if stats[9] < 999.0 else 0.0),
        ]
    return out


def _pair_arrays(pair_features: dict[tuple[int, int], list[float]], gt_labels: dict[int, int]) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    pairs: list[tuple[int, int]] = []
    features: list[list[float]] = []
    labels: list[int] = []
    for pair, vec in pair_features.items():
        a, b = pair
        ga = int(gt_labels.get(int(a), 0))
        gb = int(gt_labels.get(int(b), 0))
        if ga <= 0 or gb <= 0:
            continue
        pairs.append(pair)
        features.append(vec)
        labels.append(int(ga == gb))
    if not pairs:
        return [], np.zeros((0, len(PAIR_FEATURE_NAMES)), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return pairs, np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _balanced_sample_indices(y: np.ndarray, max_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        size = min(int(max_pairs), len(y))
        return np.sort(rng.choice(np.arange(len(y)), size=size, replace=False))
    per_class = max(1, min(len(pos), len(neg), int(max_pairs) // 2))
    pos_sel = rng.choice(pos, size=per_class, replace=False)
    neg_sel = rng.choice(neg, size=per_class, replace=False)
    return np.sort(np.concatenate([pos_sel, neg_sel]))


def _parse_csv_tokens(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_int_tokens(value: str) -> list[int]:
    return [int(part) for part in _parse_csv_tokens(value)]


def _feature_indices_for_ablation(ablation: str) -> tuple[list[int], list[str]]:
    dropped = FEATURE_ABLATIONS.get(ablation)
    if dropped is None:
        raise ValueError(f"unknown feature ablation {ablation!r}; known={sorted(FEATURE_ABLATIONS)}")
    names = [name for name in PAIR_FEATURE_NAMES if name not in dropped]
    indices = [PAIR_FEATURE_NAMES.index(name) for name in names]
    return indices, names


def _variant_name_for_ablation(ablation: str) -> str:
    return "D8_pair_graph_rf" if ablation == "full" else f"D8_pair_graph_rf_{ablation}_ablation"


def _threshold_grid(scores: np.ndarray) -> list[float]:
    if len(scores) == 0:
        return [0.5]
    quantiles = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
    values = np.quantile(scores, quantiles).tolist()
    values.extend([0.20, 0.35, 0.50, 0.65, 0.80])
    return sorted({float(np.clip(v, 0.0, 1.0)) for v in values if np.isfinite(float(v))})


def _binary_ece(y_true: np.ndarray, scores: np.ndarray, *, bins: int = 10) -> float | None:
    if len(y_true) == 0:
        return None
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(len(y_true))
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not bool(mask.any()):
            continue
        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(mask.sum()) / total * abs(confidence - accuracy)
    return float(ece)


def _identity_selection_tuple(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    ari = _safe_float(row.get("local_ARI"), 0.0) or 0.0
    purity = _safe_float(row.get("local_purity"), 0.0) or 0.0
    completeness = _safe_float(row.get("local_completeness"), 0.0) or 0.0
    unknown = _safe_float(row.get("unknown_tube_ratio"), 1.0) or 1.0
    gateish = (
        min(ari / LOCAL_GATE["local_ARI"], 1.0)
        + min(purity / LOCAL_GATE["local_purity"], 1.0)
        + min(completeness / LOCAL_GATE["local_completeness"], 1.0)
        + min(max(LOCAL_GATE["unknown_tube_ratio_max"] - unknown, 0.0) / LOCAL_GATE["unknown_tube_ratio_max"], 1.0)
    )
    return (float(gateish), float(ari), float(purity), float(completeness), float(-unknown))


def _labels_from_pair_scores(
    *,
    pairs: list[tuple[int, int]],
    scores: np.ndarray,
    threshold: float,
    labeled_tubes: list[int],
    min_component_tubes: int,
) -> tuple[dict[int, int], int, int]:
    parent = {int(tid): int(tid) for tid in labeled_tubes}

    def find(tid: int) -> int:
        cur = int(tid)
        while parent[cur] != cur:
            parent[cur] = parent[parent[cur]]
            cur = parent[cur]
        return cur

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_count = 0
    for (a, b), score in zip(pairs, scores.tolist()):
        if float(score) >= float(threshold):
            union(int(a), int(b))
            edge_count += 1

    comps: dict[int, list[int]] = defaultdict(list)
    for tid in labeled_tubes:
        comps[find(int(tid))].append(int(tid))

    labels_pred: dict[int, int] = {}
    next_label = 0
    unknown_count = 0
    for comp in sorted(comps.values(), key=lambda values: (len(values), values[0]), reverse=True):
        if len(comp) >= int(min_component_tubes):
            for tid in comp:
                labels_pred[int(tid)] = int(next_label)
            next_label += 1
        else:
            for tid in comp:
                labels_pred[int(tid)] = int(next_label)
                next_label += 1
                unknown_count += 1
    return labels_pred, unknown_count, edge_count


def _evaluate_pair_graph(
    *,
    scene: str,
    pairs: list[tuple[int, int]],
    scores: np.ndarray,
    threshold: float,
    min_component_tubes: int,
    gt_labels: dict[int, int],
) -> dict[str, Any]:
    labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
    labels_pred, unknown_count, edge_count = _labels_from_pair_scores(
        pairs=pairs,
        scores=scores,
        threshold=threshold,
        labeled_tubes=labeled_tubes,
        min_component_tubes=min_component_tubes,
    )
    metrics = _cluster_metrics(labels_pred, gt_labels)
    return {
        "scene": scene,
        "edge_count": int(edge_count),
        "labeled_tube_count": int(len(labeled_tubes)),
        "unknown_tube_count": int(unknown_count),
        "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
        "local_ARI": metrics["ari"],
        "local_purity": metrics["purity"],
        "local_completeness": metrics["completeness"],
        "local_overmerge": metrics["overmerge"],
        "local_oversplit": metrics["oversplit"],
    }


def _best_oracle_threshold(
    *,
    scene: str,
    pairs: list[tuple[int, int]],
    scores: np.ndarray,
    min_component_options: list[int],
    gt_labels: dict[int, int],
) -> dict[str, Any]:
    best_tuple: tuple[float, float, float, float, float] | None = None
    best_row: dict[str, Any] = {}
    best_threshold = None
    best_min_component = None
    for threshold in _threshold_grid(scores):
        for min_component in min_component_options:
            row = _evaluate_pair_graph(
                scene=scene,
                pairs=pairs,
                scores=scores,
                threshold=float(threshold),
                min_component_tubes=int(min_component),
                gt_labels=gt_labels,
            )
            candidate_tuple = _identity_selection_tuple(row)
            if best_tuple is None or candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_row = row
                best_threshold = float(threshold)
                best_min_component = int(min_component)
    return {
        "oracle_threshold": best_threshold,
        "oracle_min_component_tubes": best_min_component,
        "oracle_local_ARI": best_row.get("local_ARI"),
        "oracle_local_purity": best_row.get("local_purity"),
        "oracle_local_completeness": best_row.get("local_completeness"),
        "oracle_unknown_tube_ratio": best_row.get("unknown_tube_ratio"),
        "oracle_edge_count": best_row.get("edge_count"),
        "oracle_selection_score_tuple": json.dumps(_json_safe(best_tuple), sort_keys=True),
    }


def _aggregate_rows(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    aggregate = {
        "scene": "ALL",
        "variant": variant,
        "edge_count": int(sum(int(row["edge_count"]) for row in rows)),
        "labeled_tube_count": int(sum(int(row["labeled_tube_count"]) for row in rows)),
        "unknown_tube_count": int(sum(int(row["unknown_tube_count"]) for row in rows)),
        "unknown_tube_ratio": _mean([_safe_float(row["unknown_tube_ratio"]) for row in rows]),
        "local_ARI": _mean([_safe_float(row["local_ARI"]) for row in rows]),
        "local_purity": _mean([_safe_float(row["local_purity"]) for row in rows]),
        "local_completeness": _mean([_safe_float(row["local_completeness"]) for row in rows]),
        "local_overmerge": _mean([_safe_float(row["local_overmerge"]) for row in rows]),
        "local_oversplit": _mean([_safe_float(row["local_oversplit"]) for row in rows]),
        "scene0081_local_ARI": next((row["local_ARI"] for row in rows if str(row["scene"]) == "scene0081_01"), None),
        **DIAGNOSTIC_POLICY,
    }
    return {**aggregate, **_gate_status(aggregate)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_split(Path(args.split))
    proposal_rows = _read_json(Path(args.proposal_root) / f"{args.proposal_label}_proposal_rows.json")
    rows_by_scene = _rows_by_scene(proposal_rows, scenes)
    gt_by_scene = {
        scene: _load_gt_labels(
            Path(args.cache_root),
            scene,
            int(args.max_tubes_per_window),
            int(args.image_width),
            int(args.image_height),
        )
        for scene in scenes
        if (Path(args.cache_root) / scene).exists()
    }

    pair_data: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        features = _build_pair_features(
            rows_by_scene.get(scene, []),
            max_core_tubes=int(args.max_core_tubes),
            max_proposal_rows=int(args.max_proposal_rows),
            min_proposal_score=float(args.min_proposal_score),
        )
        pairs, x, y = _pair_arrays(features, gt_by_scene.get(scene, {}))
        pair_data[scene] = {"pairs": pairs, "x": x, "y": y, "pair_feature_count": len(features)}

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as exc:  # pragma: no cover
        status = {
            "status": "not_run",
            "not_run_reason": f"scikit-learn import failed: {exc}",
            **DIAGNOSTIC_POLICY,
        }
        _write_json(output_root / "routeD_pair_graph_status.json", status)
        return status

    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    feature_importance_rows: list[dict[str, Any]] = []
    rng_seed = int(args.seed)
    min_component_options = _parse_int_tokens(args.min_component_options)
    ablations = _parse_csv_tokens(args.feature_ablations)
    for ablation in ablations:
        feature_indices, active_feature_names = _feature_indices_for_ablation(ablation)
        variant = _variant_name_for_ablation(ablation)
        variant_rows: list[dict[str, Any]] = []
        for heldout in scenes:
            train_scenes = [scene for scene in scenes if scene != heldout and len(pair_data.get(scene, {}).get("y", [])) > 0]
            if not train_scenes or len(pair_data.get(heldout, {}).get("y", [])) == 0:
                continue
            x_train_all_full = np.concatenate([pair_data[scene]["x"] for scene in train_scenes], axis=0)
            x_train_all = x_train_all_full[:, feature_indices]
            y_train_all = np.concatenate([pair_data[scene]["y"] for scene in train_scenes], axis=0)
            idx = _balanced_sample_indices(y_train_all, int(args.max_train_pairs), rng_seed)
            x_train = x_train_all[idx]
            y_train = y_train_all[idx]
            model = RandomForestClassifier(
                n_estimators=int(args.n_estimators),
                max_depth=int(args.max_depth),
                min_samples_leaf=int(args.min_samples_leaf),
                class_weight="balanced_subsample",
                random_state=rng_seed,
                n_jobs=int(args.n_jobs),
            )
            model.fit(x_train, y_train)
            for feature_name, importance in zip(active_feature_names, model.feature_importances_.tolist()):
                feature_importance_rows.append(
                    {
                        "variant": variant,
                        "heldout_scene": heldout,
                        "feature": feature_name,
                        "importance": float(importance),
                    }
                )
            train_scores_by_scene = {
                scene: model.predict_proba(pair_data[scene]["x"][:, feature_indices])[:, 1]
                for scene in train_scenes
            }
            thresholds = _threshold_grid(np.concatenate(list(train_scores_by_scene.values()), axis=0))
            best_tuple: tuple[float, float, float, float, float] | None = None
            best_threshold = thresholds[0]
            best_min_component = min_component_options[0]
            best_train_agg: dict[str, Any] = {}
            for threshold in thresholds:
                for min_component in min_component_options:
                    train_rows = [
                        _evaluate_pair_graph(
                            scene=scene,
                            pairs=pair_data[scene]["pairs"],
                            scores=train_scores_by_scene[scene],
                            threshold=float(threshold),
                            min_component_tubes=int(min_component),
                            gt_labels=gt_by_scene.get(scene, {}),
                        )
                        for scene in train_scenes
                    ]
                    aggregate = _aggregate_rows(train_rows, f"{variant}_train")
                    candidate_tuple = _identity_selection_tuple(aggregate)
                    calibration_rows.append(
                        {
                            "variant": variant,
                            "heldout_scene": heldout,
                            "threshold": float(threshold),
                            "min_component_tubes": int(min_component),
                            **{f"train_{k}": v for k, v in aggregate.items() if k in {"local_ARI", "local_purity", "local_completeness", "unknown_tube_ratio", "scene0081_local_ARI"}},
                            "train_selection_score_tuple": json.dumps(_json_safe(candidate_tuple), sort_keys=True),
                        }
                    )
                    if best_tuple is None or candidate_tuple > best_tuple:
                        best_tuple = candidate_tuple
                        best_threshold = float(threshold)
                        best_min_component = int(min_component)
                        best_train_agg = aggregate

            x_test = pair_data[heldout]["x"][:, feature_indices]
            y_test = pair_data[heldout]["y"]
            test_scores = model.predict_proba(x_test)[:, 1]
            row = _evaluate_pair_graph(
                scene=heldout,
                pairs=pair_data[heldout]["pairs"],
                scores=test_scores,
                threshold=best_threshold,
                min_component_tubes=best_min_component,
                gt_labels=gt_by_scene.get(heldout, {}),
            )
            auc = None
            ap = None
            f1 = None
            ece = _binary_ece(y_test, test_scores)
            brier = None
            if len(set(y_test.tolist())) >= 2:
                from sklearn.metrics import brier_score_loss, f1_score

                auc = float(roc_auc_score(y_test, test_scores))
                ap = float(average_precision_score(y_test, test_scores))
                f1 = float(f1_score(y_test, (test_scores >= float(best_threshold)).astype(np.int64), zero_division=0))
                brier = float(brier_score_loss(y_test, test_scores))
            oracle = _best_oracle_threshold(
                scene=heldout,
                pairs=pair_data[heldout]["pairs"],
                scores=test_scores,
                min_component_options=min_component_options,
                gt_labels=gt_by_scene.get(heldout, {}),
            )
            variant_rows.append(
                {
                    **row,
                    **oracle,
                    "variant": variant,
                    "feature_ablation": ablation,
                    "active_feature_names": json.dumps(active_feature_names, sort_keys=True),
                    "heldout_scene": heldout,
                    "pair_count": int(len(y_test)),
                    "pair_positive_count": int(y_test.sum()),
                    "pair_feature_count": int(pair_data[heldout]["pair_feature_count"]),
                    "train_pair_count": int(len(y_train_all)),
                    "train_sample_count": int(len(y_train)),
                    "train_positive_sample_count": int(y_train.sum()),
                    "threshold": float(best_threshold),
                    "min_component_tubes": int(best_min_component),
                    "diagnostic_auc": auc,
                    "diagnostic_average_precision": ap,
                    "diagnostic_f1_at_threshold": f1,
                    "diagnostic_calibration_ece": ece,
                    "diagnostic_brier_score": brier,
                    "calibration_info": json.dumps(_json_safe({"train_identity_metrics": best_train_agg, "selection_score_tuple": best_tuple}), sort_keys=True),
                    **DIAGNOSTIC_POLICY,
                }
            )

        aggregate = _aggregate_rows(variant_rows, variant) if variant_rows else {}
        if aggregate:
            aggregate.update(
                {
                    "diagnostic_auc": _mean([_safe_float(row.get("diagnostic_auc")) for row in variant_rows]),
                    "diagnostic_average_precision": _mean([_safe_float(row.get("diagnostic_average_precision")) for row in variant_rows]),
                    "diagnostic_f1_at_threshold": _mean([_safe_float(row.get("diagnostic_f1_at_threshold")) for row in variant_rows]),
                    "diagnostic_calibration_ece": _mean([_safe_float(row.get("diagnostic_calibration_ece")) for row in variant_rows]),
                    "diagnostic_brier_score": _mean([_safe_float(row.get("diagnostic_brier_score")) for row in variant_rows]),
                    "oracle_local_ARI": _mean([_safe_float(row.get("oracle_local_ARI")) for row in variant_rows]),
                    "oracle_local_purity": _mean([_safe_float(row.get("oracle_local_purity")) for row in variant_rows]),
                    "oracle_local_completeness": _mean([_safe_float(row.get("oracle_local_completeness")) for row in variant_rows]),
                    "oracle_unknown_tube_ratio": _mean([_safe_float(row.get("oracle_unknown_tube_ratio")) for row in variant_rows]),
                    "oracle_scene0081_local_ARI": next((row.get("oracle_local_ARI") for row in variant_rows if str(row.get("scene")) == "scene0081_01"), None),
                    "pair_count": int(sum(int(row.get("pair_count", 0)) for row in variant_rows)),
                    "pair_positive_count": int(sum(int(row.get("pair_positive_count", 0)) for row in variant_rows)),
                    "train_sample_count": int(sum(int(row.get("train_sample_count", 0)) for row in variant_rows)),
                }
            )
            oracle_checks = {
                "oracle_ari_pass": (_safe_float(aggregate.get("oracle_local_ARI")) or 0.0) >= LOCAL_GATE["local_ARI"],
                "oracle_purity_pass": (_safe_float(aggregate.get("oracle_local_purity")) or 0.0) >= LOCAL_GATE["local_purity"],
                "oracle_completeness_pass": (_safe_float(aggregate.get("oracle_local_completeness")) or 0.0) >= LOCAL_GATE["local_completeness"],
                "oracle_unknown_pass": (_safe_float(aggregate.get("oracle_unknown_tube_ratio"), 1.0) or 1.0) <= LOCAL_GATE["unknown_tube_ratio_max"],
                "oracle_scene0081_pass": (_safe_float(aggregate.get("oracle_scene0081_local_ARI")) or 0.0) >= LOCAL_GATE["scene0081_local_ARI"],
            }
            aggregate.update({**oracle_checks, "oracle_local_gate_pass": bool(all(oracle_checks.values()))})
            summary_rows.extend(variant_rows)
            summary_rows.append(aggregate)
        else:
            summary_rows.extend(variant_rows)

    aggregate_feature_rows: list[dict[str, Any]] = []
    by_variant_feature: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in feature_importance_rows:
        by_variant_feature[(str(row["variant"]), str(row["feature"]))].append(float(row["importance"]))
    for (variant, feature), values in sorted(by_variant_feature.items()):
        aggregate_feature_rows.append(
            {
                "variant": variant,
                "heldout_scene": "ALL",
                "feature": feature,
                "importance": float(np.mean(values)),
                "importance_std": float(np.std(values)),
            }
        )
    rows_out = summary_rows
    _write_csv(output_root / "routeD_pair_graph_summary.csv", rows_out)
    _write_json(
        output_root / "routeD_pair_graph_summary.json",
        {
            "summary_rows": rows_out,
            "calibration_rows": calibration_rows,
            "feature_importance_rows": feature_importance_rows + aggregate_feature_rows,
            "ablation_status": {
                "no_visual": "not_applicable_no_frozen_visual_embedding_features_in_current_pair_graph_inputs",
            },
            "pair_feature_names": PAIR_FEATURE_NAMES,
            "config": vars(args),
            "policy": dict(DIAGNOSTIC_POLICY),
        },
    )
    _write_csv(output_root / "routeD_pair_graph_calibration.csv", calibration_rows)
    _write_csv(output_root / "routeD_pair_graph_feature_importance.csv", feature_importance_rows + aggregate_feature_rows)
    return {"summary_rows": rows_out}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v34 diagnostic tube-pair graph scorer.")
    parser.add_argument("--proposal-root", default="outputs/audit/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--proposal-label", default="v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-root", default="outputs/audit/v34_3d_object_identity/v34_routeD_pair_graph")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--max-core-tubes", type=int, default=120)
    parser.add_argument("--max-proposal-rows", type=int, default=1200)
    parser.add_argument("--min-proposal-score", type=float, default=-0.25)
    parser.add_argument("--max-train-pairs", type=int, default=220000)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3408)
    parser.add_argument("--feature-ablations", default="full,no_d4rt,no_mask,no_temporal,no_negative")
    parser.add_argument("--min-component-options", default="1,2,3,4")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    payload = run(parsed)
    print(json.dumps(_json_safe({"output_root": str(parsed.output_root), "summary_count": len(payload.get("summary_rows", []))}), indent=2))
