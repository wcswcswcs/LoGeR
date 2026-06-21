#!/usr/bin/env python3
"""Phase 8 diagnostic reliability calibrator for v73 semantic memory control.

The script compares geometry-only features against semantic+geometry features
with leave-one-chunk-out threshold rules. It is diagnostic unless the plan gate
is met and enough action-positive chunks exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GEOMETRY_FEATURES = [
    "D_geo_mean_patch",
    "D_geo_q90_patch",
    "raw_overlap_residual_rmse",
    "raw_overlap_residual_mean",
    "merge_transform_translation_norm",
    "merge_transform_rotation_deg",
    "global_k_layer5_gram_motion",
    "global_k_layer7_gram_motion",
    "reset_relative_index",
    "qscale_factor",
    "overlap_residual",
    "effective_alpha",
]

SEMANTIC_FEATURES = [
    "semantic_confidence_mean",
    "semantic_nonvoid_ratio",
    "stable_structure_ratio",
    "dynamic_thing_ratio",
    "lowtrust_stuff_ratio",
    "road_context_ratio",
    "sky_context_ratio",
    "thing_source_ratio",
    "stuff_source_ratio",
    "radio_static_mean",
    "radio_dynamic_mean",
    "radio_lowtrust_mean",
    "radio_sky_mean",
    "radio_boundary_mean",
    "radio_interior_mean",
    "radio_temporal_stability_mean",
    "radio_component_count_mean",
    "q_handoff",
    "stable_mean",
    "risk_mean",
    "remaining_valid_ratio",
    "component_consistency_proxy",
    "component_top_mass_ratio",
    "component_count_norm",
    "component_stable_variance",
    "component_risk_variance",
]

LABELS = [
    "action_head_tail_pass",
    "action_overlap_pass",
    "Y_mid",
    "Y_scale_drift",
]


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _truthy(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value)
    if text in {"1", "1.0", "True", "true"}:
        return True
    if text in {"0", "0.0", "False", "false"}:
        return False
    return None


def _load_rows(feature_csv: Path, action_rows_csv: Path) -> List[Dict[str, Any]]:
    feature_rows = {int(row["chunk_id"]): dict(row) for row in _read_csv(feature_csv)}
    action_rows = {int(row["chunk"]): dict(row) for row in _read_csv(action_rows_csv)}
    out: List[Dict[str, Any]] = []
    for chunk, action in sorted(action_rows.items()):
        if chunk not in feature_rows:
            continue
        row: Dict[str, Any] = {"chunk": chunk}
        row.update(feature_rows[chunk])
        row.update(action)
        row["action_head_tail_pass"] = action.get("head_tail_pass")
        row["action_overlap_pass"] = action.get("overlap_pass")
        out.append(row)
    return out


def _auc_score(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    pos = [score for score, label in zip(scores, labels) if label]
    neg = [score for score, label in zip(scores, labels) if not label]
    if not pos or not neg:
        return None
    wins = 0
    ties = 0
    for p_score in pos:
        for n_score in neg:
            if p_score > n_score:
                wins += 1
            elif p_score == n_score:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _candidate_thresholds(values: Sequence[float]) -> List[float]:
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique
    thresholds = []
    for left, right in zip(unique[:-1], unique[1:]):
        thresholds.append((left + right) * 0.5)
    thresholds.insert(0, unique[0] - 1.0e-9)
    thresholds.append(unique[-1] + 1.0e-9)
    return thresholds


def _fit_rule(rows: Sequence[Dict[str, Any]], features: Sequence[str], label: str) -> Optional[Tuple[str, str, float, float]]:
    best: Optional[Tuple[float, float, str, str, float]] = None
    y_all = [_truthy(row.get(label)) for row in rows]
    if any(value is None for value in y_all):
        return None
    y = [bool(value) for value in y_all]
    for feature in features:
        values_raw = [_safe_float(row.get(feature)) for row in rows]
        if any(value is None for value in values_raw):
            continue
        values = [float(value) for value in values_raw]
        for direction in ("ge", "le"):
            for threshold in _candidate_thresholds(values):
                pred = [value >= threshold if direction == "ge" else value <= threshold for value in values]
                tp = sum(1 for p, yy in zip(pred, y) if p and yy)
                fp = sum(1 for p, yy in zip(pred, y) if p and not yy)
                fn = sum(1 for p, yy in zip(pred, y) if not p and yy)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                accuracy = sum(1 for p, yy in zip(pred, y) if p == yy) / len(y)
                score = (f1, accuracy)
                if best is None or score > (best[0], best[1]):
                    best = (f1, accuracy, feature, direction, threshold)
    if best is None:
        return None
    return best[2], best[3], best[4], best[0]


def _apply_rule(row: Dict[str, Any], rule: Tuple[str, str, float, float]) -> float:
    feature, direction, threshold, _ = rule
    value = _safe_float(row.get(feature))
    if value is None:
        return 0.0
    if direction == "ge":
        return 1.0 if value >= threshold else 0.0
    return 1.0 if value <= threshold else 0.0


def _loocv(rows: List[Dict[str, Any]], features: Sequence[str], label: str) -> Dict[str, Any]:
    labels_opt = [_truthy(row.get(label)) for row in rows]
    valid_rows = [row for row, yy in zip(rows, labels_opt) if yy is not None]
    labels = [bool(_truthy(row.get(label))) for row in valid_rows]
    if not labels or len(set(labels)) < 2:
        return {
            "label": label,
            "row_count": len(valid_rows),
            "positive_count": sum(labels),
            "auc": None,
            "top5_precision": None,
            "best_full_rule": None,
            "loocv_scores": [],
        }
    scores: List[float] = []
    folds: List[Dict[str, Any]] = []
    for i, row in enumerate(valid_rows):
        train_rows = [r for j, r in enumerate(valid_rows) if j != i]
        rule = _fit_rule(train_rows, features, label)
        score = _apply_rule(row, rule) if rule is not None else 0.0
        scores.append(score)
        folds.append({"chunk": row["chunk"], "label": labels[i], "score": score, "rule": rule})
    auc = _auc_score(scores, labels)
    order = sorted(range(len(valid_rows)), key=lambda idx: scores[idx], reverse=True)
    top5 = order[:5]
    top5_precision = sum(1 for idx in top5 if labels[idx]) / 5.0
    full_rule = _fit_rule(valid_rows, features, label)
    return {
        "label": label,
        "row_count": len(valid_rows),
        "positive_count": sum(labels),
        "auc": auc,
        "top5_precision": top5_precision,
        "top5_chunks": ",".join(str(valid_rows[idx]["chunk"]) for idx in top5),
        "best_full_rule": {
            "feature": full_rule[0],
            "direction": full_rule[1],
            "threshold": full_rule[2],
            "train_f1": full_rule[3],
        }
        if full_rule is not None
        else None,
        "loocv_scores": folds,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--action-rows-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.features_csv), Path(args.action_rows_csv))
    if not rows:
        raise ValueError("no joined rows")

    geometry_rows: List[Dict[str, Any]] = []
    semantic_rows: List[Dict[str, Any]] = []
    payload_labels: Dict[str, Any] = {}
    for label in LABELS:
        geometry = _loocv(rows, GEOMETRY_FEATURES, label)
        semantic = _loocv(rows, GEOMETRY_FEATURES + SEMANTIC_FEATURES, label)
        geom_auc = geometry.get("auc")
        sem_auc = semantic.get("auc")
        auc_gain = None if geom_auc is None or sem_auc is None else float(sem_auc) - float(geom_auc)
        gate_pass = bool(
            sem_auc is not None
            and sem_auc >= 0.70
            and semantic.get("top5_precision") is not None
            and float(semantic["top5_precision"]) >= 0.40
            and auc_gain is not None
            and auc_gain >= 0.05
            and int(semantic["positive_count"]) >= 4
        )
        row_common = {
            "label": label,
            "geometry_auc": geom_auc,
            "semantic_geometry_auc": sem_auc,
            "auc_gain": auc_gain,
            "geometry_top5_precision": geometry.get("top5_precision"),
            "semantic_geometry_top5_precision": semantic.get("top5_precision"),
            "positive_count": semantic.get("positive_count"),
            "phase8_gate_pass": gate_pass,
            "geometry_rule": json.dumps(geometry.get("best_full_rule"), sort_keys=True),
            "semantic_geometry_rule": json.dumps(semantic.get("best_full_rule"), sort_keys=True),
        }
        geometry_rows.append(row_common)
        semantic_rows.append(row_common)
        payload_labels[label] = {
            "geometry": geometry,
            "semantic_geometry": semantic,
            "auc_gain": auc_gain,
            "phase8_gate_pass": gate_pass,
        }

    _write_csv(out_dir / "calibrator_gate_rows.csv", semantic_rows)
    joined_rows = []
    for row in rows:
        out = {"chunk": row["chunk"]}
        for key in sorted(set(GEOMETRY_FEATURES + SEMANTIC_FEATURES + LABELS)):
            if key in row:
                out[key] = row[key]
        joined_rows.append(out)
    _write_csv(out_dir / "calibrator_joined_rows.csv", joined_rows)
    payload = {
        "schema": "v73_phase8_reliability_calibrator_diagnostic_v1",
        "diagnostic_only": True,
        "features_csv": str(args.features_csv),
        "action_rows_csv": str(args.action_rows_csv),
        "row_count": len(rows),
        "labels": payload_labels,
        "phase8_any_gate_pass": any(item["phase8_gate_pass"] for item in payload_labels.values()),
        "deployable": False,
        "reason": "Diagnostic calibrator is not deployed unless semantic+geometry beats geometry-only by AUC >=0.05 with AUC >=0.70, top5 precision >=0.40, and >=4 positives.",
    }
    (out_dir / "calibrator_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote_rows={out_dir / 'calibrator_gate_rows.csv'}")
    print(f"wrote_joined={out_dir / 'calibrator_joined_rows.csv'}")
    print(f"wrote_summary={out_dir / 'calibrator_summary.json'}")


if __name__ == "__main__":
    main()
