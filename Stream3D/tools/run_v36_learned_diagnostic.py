from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v36_masklet_first_identity import (
    _assign_tubes,
    _load_gt_labels,
    _make_masklets,
    _parse_core_tube_ids,
    _source_group,
    _write_json,
)


FEATURES = [
    "num_core_tubes",
    "num_boundary_tubes",
    "region_area",
    "proposal_area_ratio",
    "eroded_interior_ratio",
    "boundary_contact_ratio",
    "visibility_mean",
    "confidence_mean",
    "tube_temporal_length_mean",
    "tube_canonical_compactness",
    "tube_xy_compactness",
    "appearance_variance",
    "image_gradient_boundary_score",
    "mask_distance_mean",
    "mask_distance_p10",
    "mask_distance_p50",
    "mask_distance_p90",
    "visible_outside_negative_rate",
    "same_frame_cannot_link_rate",
    "mask_temporal_repeat_score",
]

SOURCE_FLAGS = [
    "R0_current_cropformer",
    "R1_boundary_watershed",
    "R3_d4rt_tube_seeded_split",
    "R4_hybrid_split",
    "R6_hybrid_union",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _overlap_counts(row: dict[str, Any]) -> dict[int, int]:
    raw = row.get("_gt_overlap_counts") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return {int(k): int(v) for k, v in dict(raw).items() if int(v) > 0}


def _target_mixed(row: dict[str, Any]) -> int:
    return int(len(_overlap_counts(row)) > 1)


def _feature_row(row: dict[str, Any]) -> list[float]:
    vals = [_float(row.get(name), 0.0) for name in FEATURES]
    source = _source_group(row)
    vals.extend([1.0 if source == flag else 0.0 for flag in SOURCE_FLAGS])
    ptype = str(row.get("proposal_type") or "")
    vals.extend(
        [
            1.0 if ptype.startswith(("R8_", "R9_", "R10_", "R11_", "R12_")) else 0.0,
            1.0 if "d4rt" in ptype.lower() or "canonical" in ptype.lower() else 0.0,
        ]
    )
    return vals


def _mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            if value is not None and math.isfinite(float(value)):
                vals.append(float(value))
        except (TypeError, ValueError):
            pass
    return float(np.mean(vals)) if vals else None


def _model(name: str, seed: int) -> Any:
    if name == "logistic":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=int(seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
    raise ValueError(name)


def _calibration_ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float | None:
    if y_true.size == 0:
        return None
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & (prob < right if right < 1.0 else prob <= right)
        if not np.any(mask):
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(conf - acc)
    return float(ece)


def _assignment_from_predictions(
    *,
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    gt_labels: dict[int, int],
    threshold: float,
) -> dict[str, Any]:
    selected = []
    for row, prob in zip(rows, probs.tolist()):
        if float(prob) <= float(threshold) and len(_parse_core_tube_ids(row)) >= 3:
            selected.append(row)
    masklets = _make_masklets(selected, "D3_hybrid_unknown_R6")
    labels_pred, unknown_ratio = _assign_tubes(masklets, gt_labels, unknown_min_support=1, unknown_min_fraction=0.0)
    metrics = _cluster_metrics(labels_pred, gt_labels)
    return {
        "selected_region_count": int(len(selected)),
        "masklet_count": int(len(masklets)),
        "ARI": metrics.get("ari"),
        "purity": metrics.get("purity"),
        "completeness": metrics.get("completeness"),
        "unknown_tube_ratio": unknown_ratio,
        "labeled_tube_count": metrics.get("labeled_tube_count"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_json(Path(args.proposal_root) / f"{args.proposal_label}_proposal_rows.json")
    scenes = _read_split(Path(args.split))
    gt_labels = _load_gt_labels(args, scenes)
    feature_names = FEATURES + [f"source::{name}" for name in SOURCE_FLAGS] + ["temporal_source", "d4rt_source"]
    X = np.asarray([_feature_row(row) for row in rows], dtype=np.float32)
    y = np.asarray([_target_mixed(row) for row in rows], dtype=np.int64)
    scene_arr = np.asarray([str(row.get("scene") or "") for row in rows])
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    importance: Counter[str] = Counter()
    for model_name in ["logistic", "random_forest"]:
        for heldout in scenes:
            train_mask = scene_arr != heldout
            test_mask = scene_arr == heldout
            if not np.any(test_mask) or len(set(y[test_mask].tolist())) < 2:
                summary_rows.append(
                    {
                        "fold": heldout,
                        "target": "mixed_region_classifier",
                        "model": model_name,
                        "AUC": None,
                        "F1": None,
                        "calibration_error": None,
                        "ARI": None,
                        "purity": None,
                        "completeness": None,
                        "unknown_tube_ratio": None,
                        "status": "not_enough_test_labels",
                    }
                )
                continue
            clf = _model(model_name, int(args.seed))
            clf.fit(X[train_mask], y[train_mask])
            prob = clf.predict_proba(X[test_mask])[:, 1]
            pred = (prob >= float(args.mixed_threshold)).astype(np.int64)
            auc = float(roc_auc_score(y[test_mask], prob))
            f1 = float(f1_score(y[test_mask], pred))
            assign = _assignment_from_predictions(
                rows=[rows[int(idx)] for idx in np.flatnonzero(test_mask).tolist()],
                probs=prob,
                gt_labels=gt_labels.get(heldout, {}),
                threshold=float(args.mixed_threshold),
            )
            row = {
                "fold": heldout,
                "target": "mixed_region_to_masklet_assignment",
                "model": model_name,
                "AUC": auc,
                "F1": f1,
                "calibration_error": _calibration_ece(y[test_mask], prob),
                **assign,
                "status": "ok",
            }
            summary_rows.append(row)
            if hasattr(clf, "feature_importances_"):
                for name, value in zip(feature_names, clf.feature_importances_.tolist()):
                    importance[name] += float(value)

    all_rows = [
        row
        for row in summary_rows
        if row.get("status") == "ok" and row.get("model") == "random_forest"
    ]
    aggregate = {
        "fold": "ALL",
        "target": "mixed_region_to_masklet_assignment",
        "model": "random_forest",
        "AUC": _mean([row.get("AUC") for row in all_rows]),
        "F1": _mean([row.get("F1") for row in all_rows]),
        "calibration_error": _mean([row.get("calibration_error") for row in all_rows]),
        "ARI": _mean([row.get("ARI") for row in all_rows]),
        "purity": _mean([row.get("purity") for row in all_rows]),
        "completeness": _mean([row.get("completeness") for row in all_rows]),
        "unknown_tube_ratio": _mean([row.get("unknown_tube_ratio") for row in all_rows]),
        "status": "aggregate",
    }
    summary_rows.append(aggregate)
    aggregate["phaseG_pass"] = bool(
        aggregate["ARI"] is not None
        and aggregate["ARI"] >= 0.40
        and aggregate["purity"] is not None
        and aggregate["purity"] >= 0.85
        and aggregate["completeness"] is not None
        and aggregate["completeness"] >= 0.50
        and next((row.get("ARI") for row in all_rows if row.get("fold") == "scene0081_01"), -1.0) >= 0.20
    )

    importance_rows = [
        {"feature": name, "importance_sum": value}
        for name, value in sorted(importance.items(), key=lambda item: item[1], reverse=True)
    ]
    _write_csv(out_dir / "learned_region_summary.csv", summary_rows)
    _write_json(out_dir / "learned_region_summary.json", summary_rows)
    _write_csv(out_dir / "learned_region_feature_importance.csv", importance_rows)
    manifest = {
        "phase": "v36_phaseG",
        "proposal_root": str(args.proposal_root),
        "proposal_label": str(args.proposal_label),
        "uses_gt_for_prediction": True,
        "uses_gt_for_diagnostic_labels": True,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "target": "mixed region classifier plus held-out masklet assignment",
        "aggregate": aggregate,
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-root", default="outputs/audit/v35_mask_source_audit/proposal_rebuild_conda")
    parser.add_argument("--proposal-label", default="v35_mask_source_rebuild_conda")
    parser.add_argument("--output-root", default="outputs/audit/v36_learned_diagnostic")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--mixed-threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=3617)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
