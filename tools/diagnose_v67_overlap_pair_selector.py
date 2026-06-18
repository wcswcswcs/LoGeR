#!/usr/bin/env python3
"""Selector diagnostic for v67 overlap-pair action oracle rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


GROUP_LABELS: Mapping[str, Sequence[str]] = {
    "dynamic": ("person", "car", "truck", "bus", "van", "rider", "cyclist", "bicycle", "motorcycle", "animal"),
    "sky_context": ("sky", "cloud", "horizon"),
    "vegetation_farstuff": ("grass", "tree", "vegetation", "plant", "terrain", "mountain"),
    "vertical_static": (
        "building",
        "house",
        "wall",
        "handrail_or_fence",
        "fence",
        "pole",
        "traffic sign",
        "traffic light",
        "billboard_or_bulletin_board",
        "bridge",
    ),
    "ground_static": ("road", "ground", "sidewalk", "crosswalk", "floor"),
    "void_lowtrust": ("void", "unknown", "unlabeled"),
}


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


def _normalise_label_names(label_names: Any) -> Dict[int, str]:
    if isinstance(label_names, Mapping):
        return {int(k): str(v) for k, v in label_names.items()}
    return {int(i): str(v) for i, v in enumerate(label_names)}


def _ids_for(names: Iterable[str], label_to_id: Mapping[str, int]) -> List[int]:
    return [int(label_to_id[name]) for name in names if name in label_to_id]


def _ratio(labels: torch.Tensor, ids: Sequence[int]) -> float:
    if labels.numel() == 0:
        return float("nan")
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label_id in ids:
        mask |= labels == int(label_id)
    return float(mask.float().mean().item())


def _mean_tensor(value: Any) -> Optional[float]:
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    return float(value.detach().cpu().float().mean().item())


def _pair_features(path: Path, label_to_id: Mapping[str, int]) -> Dict[str, Any]:
    pair = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(pair, dict):
        raise ValueError(f"{path}: expected dict")
    labels = pair.get("curr_semantic_labels")
    labels = labels.detach().cpu().long() if torch.is_tensor(labels) else torch.empty((0,), dtype=torch.long)
    row: Dict[str, Any] = {
        "overlap_pair_file": str(path),
        "prev_chunk": int(pair.get("prev_chunk")),
        "curr_chunk": int(pair.get("curr_chunk")),
        "saved_pair_count": int(pair.get("saved_pair_count", labels.numel())),
        "valid_pair_count": int(pair.get("valid_pair_count", labels.numel())),
        "raw_residual_rmse": pair.get("raw_residual_rmse"),
        "raw_residual_mean": pair.get("raw_residual_mean"),
        "semantic_label_projected_ratio": pair.get("semantic_label_projected_ratio"),
        "semantic_nonvoid_ratio": pair.get("semantic_nonvoid_ratio"),
        "prev_conf_mean": _mean_tensor(pair.get("prev_conf")),
        "curr_conf_mean": _mean_tensor(pair.get("curr_conf")),
        "semantic_conf_mean": _mean_tensor(pair.get("curr_semantic_conf")),
    }
    for group, names in GROUP_LABELS.items():
        row[f"{group}_ratio"] = _ratio(labels, _ids_for(names, label_to_id))
    for name in ("road", "building", "car", "vegetation", "sky", "void"):
        row[f"label_{name}_ratio"] = _ratio(labels, _ids_for((name,), label_to_id))
    return row


def _best_oracle_by_chunk(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        chunk = int(float(str(row.get("curr_chunk", "-1"))))
        old = out.get(chunk)
        if old is None:
            out[chunk] = row
            continue
        key = (
            _bool(row.get("oracle_action_gate_pass")),
            _float(row.get("best_mechanism_improvement")),
            _float(row.get("raw_overlap_improvement_ratio")),
            -_float(row.get("delta_vs_baseline_global_ate")),
        )
        old_key = (
            _bool(old.get("oracle_action_gate_pass")),
            _float(old.get("best_mechanism_improvement")),
            _float(old.get("raw_overlap_improvement_ratio")),
            -_float(old.get("delta_vs_baseline_global_ate")),
        )
        if key > old_key:
            out[chunk] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-pairs-dir", type=Path, required=True)
    parser.add_argument("--oracle-results-csv", type=Path, required=True)
    parser.add_argument("--semantic-full-pt", type=Path, default=Path("results/kitti_preprocess/01/sparse_masklets_with_semantic.pt"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    sem_payload = torch.load(args.semantic_full_pt, map_location="cpu", weights_only=False)
    sem = sem_payload.get("semantic_segmentation", sem_payload) if isinstance(sem_payload, dict) else {}
    label_names = _normalise_label_names(sem.get("label_names", []))
    label_to_id = {name: idx for idx, name in label_names.items()}
    pair_rows = [_pair_features(path, label_to_id) for path in sorted(args.overlap_pairs_dir.glob("chunk_*_*.pt"))]
    oracle_rows = _read_csv(args.oracle_results_csv)
    oracle_by_chunk = _best_oracle_by_chunk(oracle_rows)
    rows: List[Dict[str, Any]] = []
    for row in pair_rows:
        chunk = int(row["curr_chunk"])
        oracle = oracle_by_chunk.get(chunk, {})
        item = dict(row)
        item.update({
            "oracle_positive": _bool(oracle.get("oracle_action_gate_pass")),
            "oracle_best_candidate": oracle.get("candidate", ""),
            "oracle_best_mechanism_improvement": oracle.get("best_mechanism_improvement", ""),
            "oracle_best_delta_ate": oracle.get("delta_vs_baseline_global_ate", ""),
            "oracle_best_raw_overlap_improvement_ratio": oracle.get("raw_overlap_improvement_ratio", ""),
            "oracle_best_safe_correction_pass": oracle.get("safe_correction_pass", ""),
        })
        rows.append(item)

    feature_names = [
        "raw_residual_rmse",
        "raw_residual_mean",
        "valid_pair_count",
        "saved_pair_count",
        "semantic_nonvoid_ratio",
        "prev_conf_mean",
        "curr_conf_mean",
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
    labels = [bool(row["oracle_positive"]) for row in rows]
    feature_rows: List[Dict[str, Any]] = []
    for feature in feature_names:
        values = [_float(row.get(feature)) for row in rows]
        auc_high = _auc(values, labels)
        auc_low = _auc([-v for v in values], labels)
        if auc_high is None or auc_low is None:
            best_direction = ""
            best_auc = None
        elif auc_high >= auc_low:
            best_direction = "higher"
            best_auc = auc_high
        else:
            best_direction = "lower"
            best_auc = auc_low
        positive_values = [v for v, label in zip(values, labels) if label and math.isfinite(v)]
        feature_rows.append({
            "feature": feature,
            "auc_higher": auc_high,
            "auc_lower": auc_low,
            "best_auc": best_auc,
            "best_direction": best_direction,
            "positive_value": positive_values[0] if positive_values else None,
        })
    feature_rows.sort(key=lambda row: (-1.0 if row["best_auc"] is None else -float(row["best_auc"]), row["feature"]))

    positives = [row for row in rows if row["oracle_positive"]]
    summary = {
        "schema": "acl2_v67_overlap_pair_selector_summary_v1",
        "overlap_pairs_dir": str(args.overlap_pairs_dir),
        "oracle_results_csv": str(args.oracle_results_csv),
        "semantic_full_pt": str(args.semantic_full_pt),
        "rows": len(rows),
        "positive_chunks": [int(row["curr_chunk"]) for row in positives],
        "positive_count": len(positives),
        "top_features": feature_rows[:10],
        "note": "Diagnostic only. With one positive chunk, AUC/rank is weak evidence and cannot establish a robust semantic selector.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "overlap_pair_selector_features.csv", rows)
    _write_csv(args.out_dir / "overlap_pair_selector_feature_auc.csv", feature_rows)
    (args.out_dir / "overlap_pair_selector_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
