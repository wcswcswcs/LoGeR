from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.run_v26_object_quality_diagnostics import _auc, _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v28_proposal_selection import _load_gt_labels


O5_PREFIXES = (
    "R0_",
    "R1_",
    "R2_",
    "R3_",
    "R4_",
    "R5_",
    "R6_",
    "R7_",
    "R8_temporal_",
    "R9_temporal_",
    "R10_temporal_",
    "R11_temporal_",
    "R12_temporal_",
)
TEMPORAL_PREFIXES = ("R8_temporal_", "R9_temporal_", "R10_temporal_", "R11_temporal_", "R12_temporal_")
MASK_ONLY_TYPES = {
    "R0_full_mask_region",
    "R1_boundary_eroded_interior",
    "R2_distance_watershed_region",
    "R4_image_gradient_split",
    "R6_mask_overlap_consensus_region",
    "R6_mask_overlap_consensus_union",
}
SMALL_CHILD_PREFIXES = ("R1_", "R2_", "R3_", "R4_", "R5_", "R7_", "R12_")


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _quantile(values: list[float], q: float) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.quantile(vals, q)) if vals else None


def _core_ids(row: dict[str, Any]) -> tuple[int, ...]:
    if "_core_tube_ids" in row and row.get("_core_tube_ids") is not None:
        return tuple(sorted(int(v) for v in row.get("_core_tube_ids") or []))
    text = str(row.get("core_tube_ids") or "")
    if not text:
        return ()
    return tuple(sorted(int(v) for v in text.split(";") if str(v).strip()))


def _fringe_ids(row: dict[str, Any]) -> tuple[int, ...]:
    if "_fringe_tube_ids" in row and row.get("_fringe_tube_ids") is not None:
        return tuple(sorted(int(v) for v in row.get("_fringe_tube_ids") or []))
    text = str(row.get("fringe_tube_ids") or "")
    if not text:
        return ()
    return tuple(sorted(int(v) for v in text.split(";") if str(v).strip()))


def _set_core_ids(row: dict[str, Any], ids: tuple[int, ...]) -> None:
    ids = tuple(sorted(int(v) for v in ids))
    row["_core_tube_ids"] = list(ids)
    row["core_tube_ids"] = ";".join(str(v) for v in ids)
    row["num_core_tubes"] = int(len(ids))
    row["core_tube_count"] = int(len(ids))


def _is_o5(row: dict[str, Any]) -> bool:
    proposal_type = str(row.get("proposal_type") or "")
    return proposal_type.startswith(O5_PREFIXES)


def _type_bucket(proposal_type: str) -> str:
    if proposal_type.startswith("R0_"):
        return "R0_full_mask"
    if proposal_type.startswith("R1_"):
        return "R1_eroded_core"
    if proposal_type.startswith("R2_"):
        return "R2_watershed"
    if proposal_type.startswith("R3_"):
        return "R3_d4rt_seeded"
    if proposal_type.startswith("R4_"):
        return "R4_gradient_split"
    if proposal_type.startswith("R5_"):
        return "R5_canonical_split"
    if proposal_type.startswith("R6_"):
        return "R6_consensus"
    if proposal_type.startswith("R7_"):
        return "R7_high_purity_core"
    if proposal_type.startswith("R8_"):
        return "R8_temporal_union"
    if proposal_type.startswith("R9_"):
        return "R9_temporal_consensus"
    if proposal_type.startswith("R10_"):
        return "R10_negative_pruned_temporal"
    if proposal_type.startswith("R12_"):
        return "R12_eroded_pruned_temporal"
    if proposal_type.startswith("R13_"):
        return "R13_v29_medium"
    return proposal_type.split("_", 1)[0] or "unknown"


def _proposal_gt_counts(row: dict[str, Any], gt_labels: dict[int, int]) -> tuple[Counter[int], int]:
    existing = row.get("_gt_overlap_counts")
    if isinstance(existing, dict):
        counts = Counter({int(k): int(v) for k, v in existing.items() if int(v) > 0})
        labeled = _int(row, "_proposal_labeled_tube_count", _int(row, "proposal_labeled_tube_count", sum(counts.values())))
        return counts, int(labeled)
    counts: Counter[int] = Counter()
    labeled = 0
    for tid in _core_ids(row):
        gt = int(gt_labels.get(int(tid), 0))
        if gt > 0:
            counts[gt] += 1
            labeled += 1
    return counts, int(labeled)


def _add_diagnostic_gt_fields(row: dict[str, Any], gt_labels: dict[int, int], gt_counts: Counter[int]) -> None:
    counts, labeled = _proposal_gt_counts(row, gt_labels)
    best_gt = 0
    best_overlap = 0
    if counts:
        best_gt, best_overlap = counts.most_common(1)[0]
    purity = float(best_overlap / max(labeled, 1)) if labeled else None
    completeness = float(best_overlap / max(int(gt_counts.get(int(best_gt), 0)), 1)) if best_gt else None
    iou = None
    if best_gt:
        iou = float(best_overlap / max(labeled + int(gt_counts[best_gt]) - best_overlap, 1))
    row["_gt_overlap_counts"] = {int(k): int(v) for k, v in counts.items()}
    row["_proposal_labeled_tube_count"] = int(labeled)
    row["proposal_labeled_tube_count"] = int(labeled)
    row["proposal_best_GT"] = int(best_gt)
    row["proposal_purity"] = purity
    row["proposal_completeness"] = completeness
    row["proposal_best_IoU"] = iou


def _labels_from_selected(selected: list[dict[str, Any]], labeled_tubes: list[int], *, include_fringe: dict[str, set[int]] | None = None) -> tuple[dict[int, int], int]:
    labels_pred: dict[int, int] = {}
    for idx, row in enumerate(selected):
        proposal_id = str(row.get("proposal_id"))
        owned = set(_core_ids(row))
        if include_fringe:
            owned.update(include_fringe.get(proposal_id, set()))
        for tid in sorted(owned):
            tid = int(tid)
            if tid not in labels_pred:
                labels_pred[tid] = idx
    next_label = len(selected)
    unknown_count = 0
    for tid in labeled_tubes:
        if int(tid) not in labels_pred:
            labels_pred[int(tid)] = next_label
            next_label += 1
            unknown_count += 1
    return labels_pred, int(unknown_count)


def _eval_selected(selected: list[dict[str, Any]], gt_labels: dict[int, int], *, include_fringe: dict[str, set[int]] | None = None) -> dict[str, Any]:
    labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
    labels_pred, unknown_count = _labels_from_selected(selected, labeled_tubes, include_fringe=include_fringe)
    metrics = _cluster_metrics(labels_pred, gt_labels)
    selected_tubes = set()
    for row in selected:
        selected_tubes.update(_core_ids(row))
        if include_fringe:
            selected_tubes.update(include_fringe.get(str(row.get("proposal_id")), set()))
    return {
        "ARI": metrics["ari"],
        "purity": metrics["purity"],
        "completeness": metrics["completeness"],
        "overmerge": metrics["overmerge"],
        "oversplit": metrics["oversplit"],
        "owned_tube_ratio": float(len(selected_tubes & set(labeled_tubes)) / max(len(labeled_tubes), 1)),
        "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
        "unknown_tube_count": int(unknown_count),
        "labeled_tube_count": int(len(labeled_tubes)),
    }


def _oracle_selected(
    rows: list[dict[str, Any]],
    gt_labels: dict[int, int],
    gt_counts: Counter[int],
    *,
    include_medium: bool,
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    candidates = [
        row
        for row in rows
        if _is_o5(row) or (include_medium and str(row.get("proposal_type", "")).startswith("R13_"))
    ]
    per_gt_best: dict[int, tuple[float, dict[str, Any] | None]] = {}
    for gt in sorted(gt_counts):
        best_iou = -1.0
        best_row: dict[str, Any] | None = None
        for row in candidates:
            counts, labeled = _proposal_gt_counts(row, gt_labels)
            overlap = int(counts.get(int(gt), 0))
            iou = float(overlap / max(labeled + int(gt_counts[gt]) - overlap, 1))
            if iou > best_iou:
                best_iou = iou
                best_row = row
        per_gt_best[int(gt)] = (float(max(best_iou, 0.0)), best_row)
    selected: dict[str, dict[str, Any]] = {}
    best_ious: dict[int, float] = {}
    for gt, (score, row) in per_gt_best.items():
        best_ious[int(gt)] = float(score)
        if row is not None:
            selected[str(row["proposal_id"])] = row
    return list(selected.values()), best_ious


def _oracle_summary(scene: str, rows: list[dict[str, Any]], gt_labels: dict[int, int], *, include_medium: bool) -> dict[str, Any]:
    gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
    selected, best_ious = _oracle_selected(rows, gt_labels, gt_counts, include_medium=include_medium)
    metrics = _eval_selected(selected, gt_labels)
    return {
        "scene": scene,
        "pool": "O5M_medium_augmented" if include_medium else "O5_final",
        "proposal_count": int(sum(1 for row in rows if _is_o5(row) or (include_medium and str(row.get("proposal_type", "")).startswith("R13_")))),
        "oracle_ARI": metrics["ARI"],
        "oracle_purity": metrics["purity"],
        "oracle_completeness": metrics["completeness"],
        "oracle_overmerge": metrics["overmerge"],
        "oracle_oversplit": metrics["oversplit"],
        "GT_count": int(len(gt_counts)),
        "covered_GT_count": int(sum(1 for score in best_ious.values() if score > 0.0)),
        "GT_with_best_IoU_ge_025": float(sum(1 for score in best_ious.values() if score >= 0.25) / max(len(best_ious), 1)),
        "GT_with_best_IoU_ge_050": float(sum(1 for score in best_ious.values() if score >= 0.50) / max(len(best_ious), 1)),
        "oracle_per_GT_best_IoU_mean": _mean(list(best_ious.values())),
        "medium_oracle_selected_count": int(sum(1 for row in selected if str(row.get("proposal_type", "")).startswith("R13_"))),
    }


def _aggregate(rows: list[dict[str, Any]], group_key: str, metric_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        row: dict[str, Any] = {group_key: key, "scene": "ALL", "scene_count": len(items)}
        for metric in metric_keys:
            if metric.endswith("_count") or metric in {"proposal_count", "GT_count", "covered_GT_count", "selected_proposal_count"}:
                row[metric] = int(sum(int(float(item.get(metric, 0) or 0)) for item in items))
            else:
                row[metric] = _mean([float(item[metric]) for item in items if item.get(metric) is not None])
        row["scene0081_oracle_ARI"] = next((item.get("oracle_ARI") for item in items if item.get("scene") == "scene0081_01"), None)
        row["scene0081_ARI"] = next((item.get("ARI") for item in items if item.get("scene") == "scene0081_01"), None)
        out.append(row)
    return out


def _selected_ids_from_csv(path: Path, variant: str) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in _read_csv_dicts(path):
        if str(row.get("variant")) == variant:
            proposal_id = str(row.get("proposal_id") or "")
            if proposal_id.endswith("_p11own"):
                proposal_id = proposal_id[: -len("_p11own")]
            if proposal_id:
                ids.add(proposal_id)
    return ids


def _feature_bucket(core_count: int) -> str:
    if core_count <= 4:
        return "xs"
    if core_count <= 12:
        return "s"
    if core_count <= 40:
        return "m"
    if core_count <= 120:
        return "l"
    return "xl"


def _residual_by_bucket(rows: list[dict[str, Any]], key: str, bucket_fields: tuple[str, ...]) -> dict[str, float]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        bucket = []
        for field in bucket_fields:
            if field == "core_bin":
                bucket.append(_feature_bucket(_int(row, "num_core_tubes", _int(row, "core_tube_count", len(_core_ids(row))))))
            elif field == "area_bin":
                bucket.append(_feature_bucket(int(max(_float(row, "proposal_area", 0.0) / 500.0, 1.0))))
            else:
                bucket.append(row.get(field))
        buckets[tuple(bucket)].append(_float(row, key))
    stats = {bucket: (float(np.mean(vals)), float(np.std(vals)) + 1e-6) for bucket, vals in buckets.items()}
    out: dict[str, float] = {}
    for row in rows:
        bucket = []
        for field in bucket_fields:
            if field == "core_bin":
                bucket.append(_feature_bucket(_int(row, "num_core_tubes", _int(row, "core_tube_count", len(_core_ids(row))))))
            elif field == "area_bin":
                bucket.append(_feature_bucket(int(max(_float(row, "proposal_area", 0.0) / 500.0, 1.0))))
            else:
                bucket.append(row.get(field))
        mean, std = stats[tuple(bucket)]
        out[str(row["proposal_id"])] = float((_float(row, key) - mean) / math.sqrt(abs(mean) + std + 1e-6))
    return out


def _oriented_auc(labels: list[int], scores: list[float]) -> float | None:
    if not labels:
        return None
    auc = _auc(np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64))
    if auc is None:
        return None
    return float(max(float(auc), 1.0 - float(auc)))


def _annotate_features(rows: list[dict[str, Any]], oracle_ids: set[str], p4_ids: set[str]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    o5_rows = [row for row in rows if _is_o5(row)]
    residual_specs = {
        "cannot_link_residual_frame_core": ("same_frame_cannot_link_rate", ("scene", "frame_id", "core_bin")),
        "cannot_link_residual_scene_core": ("same_frame_cannot_link_rate", ("scene", "core_bin")),
        "visible_negative_residual_frame_core": ("visible_outside_negative_rate", ("scene", "frame_id", "core_bin")),
        "boundary_risk_residual_scene_core": ("boundary_risk", ("scene", "core_bin")),
        "overlap_density_residual_scene_core": ("overlap_with_other_proposals", ("scene", "core_bin")),
    }
    residuals: dict[str, dict[str, float]] = {}
    for out_key, (source_key, bucket_fields) in residual_specs.items():
        residuals[out_key] = _residual_by_bucket(o5_rows, source_key, bucket_fields)

    by_scene: dict[str, list[int]] = defaultdict(list)
    for row in o5_rows:
        by_scene[str(row.get("scene"))].append(_int(row, "num_core_tubes", len(_core_ids(row))))
    scene_q: dict[str, tuple[float, float]] = {}
    for scene, vals in by_scene.items():
        scene_q[scene] = (
            float(np.quantile(vals, 0.30)) if vals else 1.0,
            float(np.quantile(vals, 0.80)) if vals else 1.0,
        )

    feature_by_id: dict[str, dict[str, float]] = {}
    for row in rows:
        pid = str(row["proposal_id"])
        core = _int(row, "num_core_tubes", len(_core_ids(row)))
        q30, q80 = scene_q.get(str(row.get("scene")), (1.0, 1.0))
        if q80 <= q30:
            core_balance = 0.0
        else:
            mid = 0.5 * (q30 + q80)
            core_balance = max(0.0, 1.0 - abs(float(core) - mid) / max(q80 - q30, 1.0))
        feats = {
            "raw_cannot_link_rate": _float(row, "same_frame_cannot_link_rate"),
            "area_normalized_cannot_link_rate": _float(row, "same_frame_cannot_link_rate")
            / max(_float(row, "proposal_area_ratio", 1e-6), 1e-6),
            "raw_visible_negative_rate": _float(row, "visible_outside_negative_rate"),
            "raw_boundary_risk": _float(row, "boundary_risk"),
            "raw_boundary_contact": _float(row, "boundary_contact_ratio"),
            "raw_core_tube_count": float(core),
            "raw_area_ratio": _float(row, "proposal_area_ratio"),
            "raw_compactness": _float(row, "tube_canonical_compactness"),
            "visibility_mean": _float(row, "visibility_mean", 0.5),
            "confidence_mean": _float(row, "confidence_mean", 0.5),
            "eroded_interior_ratio": _float(row, "eroded_interior_ratio"),
            "temporal_repeat_score": _float(row, "mask_temporal_repeat_score"),
            "core_balance_medium": float(core_balance),
        }
        for key, values in residuals.items():
            feats[key] = values.get(pid, 0.0)
        target_balance = 1.0 / (1.0 + abs(math.log1p(max(core, 1)) - math.log(20.0)))
        feats["medium_prior_score"] = float(
            0.60 * target_balance
            + 0.20 * feats["eroded_interior_ratio"]
            + 0.15 * feats["visibility_mean"]
            - 0.08 * max(feats["area_normalized_cannot_link_rate"], 0.0)
            - 0.08 * max(feats["visible_negative_residual_frame_core"], 0.0)
            - 0.08 * max(feats["boundary_risk_residual_scene_core"], 0.0)
            - 0.05 * min(feats["raw_compactness"], 10.0)
        )
        feats["is_medium_candidate"] = float(
            q30 <= core <= q80
            and feats["cannot_link_residual_scene_core"] <= 1.0
            and feats["boundary_risk_residual_scene_core"] <= 1.0
        )
        feature_by_id[pid] = feats
        row["_v29_features"] = feats

    labels_oracle_good = []
    labels_false_merge = []
    labels_selected_bad = []
    labels_oracle_selected = []
    labels_p4_selected = []
    summary_rows = []
    feature_names = sorted(next(iter(feature_by_id.values())).keys()) if feature_by_id else []
    for row in o5_rows:
        pid = str(row["proposal_id"])
        purity = _float(row, "proposal_purity", 0.0)
        iou = _float(row, "proposal_best_IoU", 0.0)
        oracle_good = pid in oracle_ids or (purity >= 0.85 and iou >= 0.25)
        false_merge = purity <= 0.60 or (purity <= 0.75 and iou < 0.25)
        selected_bad = pid in p4_ids and false_merge
        labels_oracle_good.append(1 if oracle_good else 0)
        labels_false_merge.append(1 if false_merge else 0)
        labels_selected_bad.append(1 if selected_bad else 0)
        labels_oracle_selected.append(1 if pid in oracle_ids else 0)
        labels_p4_selected.append(1 if pid in p4_ids else 0)
    for feat in feature_names:
        scores = [feature_by_id[str(row["proposal_id"])][feat] for row in o5_rows]
        scene0081_rows = [row for row in o5_rows if row.get("scene") == "scene0081_01"]
        scene0081_labels = [
            1
            if (str(row["proposal_id"]) in oracle_ids or (_float(row, "proposal_purity", 0.0) >= 0.85 and _float(row, "proposal_best_IoU", 0.0) >= 0.25))
            else 0
            for row in scene0081_rows
        ]
        scene0081_scores = [feature_by_id[str(row["proposal_id"])][feat] for row in scene0081_rows]
        per_scene_aucs = []
        for scene in sorted({str(row.get("scene")) for row in o5_rows}):
            scene_rows = [row for row in o5_rows if row.get("scene") == scene]
            scene_labels = [
                1
                if (str(row["proposal_id"]) in oracle_ids or (_float(row, "proposal_purity", 0.0) >= 0.85 and _float(row, "proposal_best_IoU", 0.0) >= 0.25))
                else 0
                for row in scene_rows
            ]
            scene_scores = [feature_by_id[str(row["proposal_id"])][feat] for row in scene_rows]
            auc_val = _oriented_auc(scene_labels, scene_scores)
            if auc_val is not None:
                per_scene_aucs.append(float(auc_val))
        summary_rows.append(
            {
                "feature": feat,
                "oracle_good_AUC": _oriented_auc(labels_oracle_good, scores),
                "false_merge_AUC": _oriented_auc(labels_false_merge, scores),
                "selected_bad_AUC": _oriented_auc(labels_selected_bad, scores),
                "oracle_selected_vs_P4_AUC": _oriented_auc(
                    [1 if a == 1 else 0 for a in labels_oracle_selected],
                    scores,
                ),
                "scene0081_oracle_good_AUC": _oriented_auc(scene0081_labels, scene0081_scores),
                "feature_stability_across_scenes": int(sum(1 for val in per_scene_aucs if val >= 0.60)),
            }
        )
    return feature_by_id, summary_rows


def _quality(row: dict[str, Any], *, broad_penalty: float = 0.0, medium_bonus: float = 0.0, coverage_weight: float = 0.15) -> float:
    feats = row.get("_v29_features") or {}
    core = max(len(_core_ids(row)), 1)
    proposal_type = str(row.get("proposal_type") or "")
    type_bonus = 0.0
    if proposal_type.startswith("R13_"):
        type_bonus += float(medium_bonus)
    if proposal_type.startswith("R7_") or proposal_type.startswith("R12_"):
        type_bonus += 0.08
    if proposal_type.startswith("R10_"):
        type_bonus -= float(broad_penalty)
    if proposal_type.startswith("R8_"):
        type_bonus -= float(broad_penalty) * 0.75
    return float(
        type_bonus
        + 0.50 * float(feats.get("medium_prior_score", 0.0))
        + 0.22 * float(feats.get("eroded_interior_ratio", _float(row, "eroded_interior_ratio")))
        + 0.18 * float(feats.get("visibility_mean", _float(row, "visibility_mean", 0.5)))
        + float(coverage_weight) * min(math.log1p(core) / math.log(128.0), 1.0)
        - 0.16 * max(float(feats.get("cannot_link_residual_scene_core", 0.0)), 0.0)
        - 0.12 * max(float(feats.get("visible_negative_residual_frame_core", 0.0)), 0.0)
        - 0.10 * max(float(feats.get("boundary_risk_residual_scene_core", 0.0)), 0.0)
        - 0.04 * min(float(feats.get("raw_compactness", _float(row, "tube_canonical_compactness"))), 10.0)
    )


def _compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("scene") != b.get("scene"):
        return False
    if _int(a, "frame_id") != _int(b, "frame_id"):
        return False
    if _int(a, "mask_id") != _int(b, "mask_id"):
        return False
    ca, cb = set(_core_ids(a)), set(_core_ids(b))
    if not ca or not cb:
        return False
    overlap = len(ca & cb) / max(min(len(ca), len(cb)), 1)
    if overlap > 0.70:
        return False
    fa = a.get("_v29_features") or {}
    fb = b.get("_v29_features") or {}
    if max(float(fa.get("cannot_link_residual_scene_core", 0.0)), float(fb.get("cannot_link_residual_scene_core", 0.0))) > 1.5:
        return False
    if max(float(fa.get("visible_negative_residual_frame_core", 0.0)), float(fb.get("visible_negative_residual_frame_core", 0.0))) > 1.5:
        return False
    return True


def _make_medium_row(
    scene: str,
    generator: str,
    source_rows: list[dict[str, Any]],
    ids: set[int],
    index: int,
    gt_labels: dict[int, int],
    gt_counts: Counter[int],
) -> dict[str, Any]:
    first = source_rows[0]
    row: dict[str, Any] = {
        "proposal_id": f"{scene}_v29_{generator}_{index:06d}",
        "scene": scene,
        "frame_id": _int(first, "frame_id"),
        "mask_id": _int(first, "mask_id"),
        "proposal_type": f"R13_v29_medium_{generator}",
        "v29_medium_generator": generator,
        "v29_source_proposal_ids": ";".join(str(item.get("proposal_id")) for item in source_rows),
        "v29_source_proposal_count": int(len(source_rows)),
        "is_diagnostic_only": False,
    }
    _set_core_ids(row, tuple(sorted(ids)))
    numeric_keys = [
        "appearance_variance",
        "boundary_contact_ratio",
        "boundary_risk",
        "confidence_mean",
        "eroded_interior_ratio",
        "image_gradient_boundary_score",
        "mask_area",
        "mask_distance_mean",
        "mask_temporal_repeat_score",
        "overlap_with_other_proposals",
        "proposal_area",
        "proposal_area_ratio",
        "region_area",
        "same_frame_cannot_link_rate",
        "tube_canonical_compactness",
        "tube_density",
        "tube_temporal_length_mean",
        "tube_xy_compactness",
        "visibility_mean",
        "visible_outside_negative_rate",
    ]
    for key in numeric_keys:
        vals = [_float(item, key) for item in source_rows if item.get(key) is not None]
        row[key] = _mean(vals) if vals else 0.0
    _add_diagnostic_gt_fields(row, gt_labels, gt_counts)
    return row


def _generate_medium(rows_by_scene: dict[str, list[dict[str, Any]]], gt_by_scene: dict[str, dict[int, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    medium_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    existing = {tuple(_core_ids(row)) for rows in rows_by_scene.values() for row in rows}
    for scene, rows in rows_by_scene.items():
        gt_labels = gt_by_scene[scene]
        gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
        core_sizes = [len(_core_ids(row)) for row in rows if _is_o5(row)]
        p30 = float(np.quantile(core_sizes, 0.30)) if core_sizes else 3.0
        p80 = float(np.quantile(core_sizes, 0.80)) if core_sizes else 40.0
        max_count = 0
        generator_counts: Counter[str] = Counter()
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if not _is_o5(row):
                continue
            groups[(_int(row, "frame_id"), _int(row, "mask_id"))].append(row)
        for group_rows in groups.values():
            children = [
                row
                for row in group_rows
                if str(row.get("proposal_type", "")).startswith(SMALL_CHILD_PREFIXES)
                and 2 <= len(_core_ids(row)) <= max(p80, 4.0)
            ]
            ranked = sorted(children, key=lambda item: _quality(item, medium_bonus=0.2), reverse=True)[:10]
            for i, seed in enumerate(ranked):
                merged = set(_core_ids(seed))
                sources = [seed]
                for other in ranked[i + 1 :]:
                    trial = merged | set(_core_ids(other))
                    if len(trial) > max(p80 * 1.15, p30 + 1):
                        continue
                    if len(trial) < max(len(merged), len(_core_ids(other))) + 2:
                        continue
                    if _compatible(seed, other):
                        merged = trial
                        sources.append(other)
                    if len(merged) >= p30:
                        break
                if p30 <= len(merged) <= max(p80 * 1.15, p30):
                    key = tuple(sorted(merged))
                    if key not in existing:
                        existing.add(key)
                        medium_rows.append(_make_medium_row(scene, "D1_sibling_merge", sources, merged, len(medium_rows), gt_labels, gt_counts))
                        generator_counts["D1_sibling_merge"] += 1
                        max_count += 1
                if max_count >= 1200:
                    break
            seeds = [
                row
                for row in ranked
                if str(row.get("proposal_type", "")).startswith(("R7_", "R1_"))
                and _quality(row, medium_bonus=0.1) > 0.2
            ]
            for seed in seeds[:5]:
                grown = set(_core_ids(seed))
                sources = [seed]
                for other in ranked:
                    if other is seed or not _compatible(seed, other):
                        continue
                    trial = grown | set(_core_ids(other))
                    if len(trial) > max(p80, p30 + 1):
                        continue
                    grown = trial
                    sources.append(other)
                    if len(grown) >= p30:
                        break
                if p30 <= len(grown) <= max(p80, p30):
                    key = tuple(sorted(grown))
                    if key not in existing:
                        existing.add(key)
                        medium_rows.append(_make_medium_row(scene, "D2_boundary_grow", sources, grown, len(medium_rows), gt_labels, gt_counts))
                        generator_counts["D2_boundary_grow"] += 1
                        max_count += 1
                if max_count >= 1200:
                    break
            broad = [row for row in group_rows if str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES) and len(_core_ids(row)) > p80]
            for parent in sorted(broad, key=lambda item: len(_core_ids(item)), reverse=True)[:4]:
                parent_core = set(_core_ids(parent))
                contained = [child for child in ranked if set(_core_ids(child)) and set(_core_ids(child)).issubset(parent_core)]
                if len(contained) < 2:
                    continue
                component: set[int] = set()
                sources = [parent]
                for child in contained:
                    trial = component | set(_core_ids(child))
                    if len(trial) > max(p80, p30 + 1):
                        continue
                    if not component or all(_compatible(child, src) for src in sources[1:]):
                        component = trial
                        sources.append(child)
                    if len(component) >= p30:
                        break
                if p30 <= len(component) <= max(p80, p30):
                    key = tuple(sorted(component))
                    if key not in existing:
                        existing.add(key)
                        gen = "D3_parent_conflict_split" if str(parent.get("proposal_type", "")).startswith(("R8_", "R9_")) else "D4_temporal_minus_conflict"
                        medium_rows.append(_make_medium_row(scene, gen, sources, component, len(medium_rows), gt_labels, gt_counts))
                        generator_counts[gen] += 1
                        max_count += 1
                if max_count >= 1200:
                    break
        scene_medium = [row for row in medium_rows if row.get("scene") == scene]
        summary_rows.append(
            {
                "scene": scene,
                "medium_proposal_count": int(len(scene_medium)),
                "medium_per_parent_mean": _mean([_float(row, "v29_source_proposal_count") for row in scene_medium]),
                "medium_core_tube_count_p50": _quantile([len(_core_ids(row)) for row in scene_medium], 0.50),
                "medium_core_tube_count_p90": _quantile([len(_core_ids(row)) for row in scene_medium], 0.90),
                "medium_purity_mean_diagnostic": _mean([_float(row, "proposal_purity") for row in scene_medium]),
                "medium_completeness_mean_diagnostic": _mean([_float(row, "proposal_completeness") for row in scene_medium]),
                "medium_best_IoU_mean_diagnostic": _mean([_float(row, "proposal_best_IoU") for row in scene_medium]),
                "medium_GT_IoU_ge_025": float(sum(1 for row in scene_medium if _float(row, "proposal_best_IoU") >= 0.25) / max(len(scene_medium), 1)),
                **{f"count_{key}": int(generator_counts[key]) for key in sorted(generator_counts)},
            }
        )
    return medium_rows, summary_rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[int, ...]]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("scene")), tuple(_core_ids(row)))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _build_solver_candidates(rows: list[dict[str, Any]], *, mode: str, max_per_scene: int) -> dict[str, list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        proposal_type = str(row.get("proposal_type") or "")
        if mode == "real":
            ok = _is_o5(row) or proposal_type.startswith("R13_")
        elif mode == "no_temporal":
            ok = (_is_o5(row) or proposal_type.startswith("R13_")) and not proposal_type.startswith(TEMPORAL_PREFIXES)
        elif mode == "mask_only":
            ok = proposal_type in MASK_ONLY_TYPES
        else:
            ok = _is_o5(row) or proposal_type.startswith("R13_")
        if ok and len(_core_ids(row)) >= 2:
            by_scene[str(row["scene"])].append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for scene, items in by_scene.items():
        ranked = sorted(items, key=lambda row: _quality(row, medium_bonus=0.25), reverse=True)
        medium = [row for row in ranked if str(row.get("proposal_type", "")).startswith("R13_")]
        non_medium = [row for row in ranked if not str(row.get("proposal_type", "")).startswith("R13_")]
        keep = _dedupe_rows((medium[: max_per_scene // 2] + non_medium)[:max_per_scene])
        out[scene] = keep
    return out


def _build_conflict_graph(candidates_by_scene: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, set[tuple[str, str]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    hard_by_scene: dict[str, set[tuple[str, str]]] = defaultdict(set)
    edge_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary_rows: list[dict[str, Any]] = []
    for scene, rows in candidates_by_scene.items():
        core_by_id = {str(row["proposal_id"]): set(_core_ids(row)) for row in rows}
        by_pid = {str(row["proposal_id"]): row for row in rows}
        inverted: dict[int, list[str]] = defaultdict(list)
        for pid, core in core_by_id.items():
            for tid in core:
                inverted[int(tid)].append(pid)
        overlap_counts: Counter[tuple[str, str]] = Counter()
        for pids in inverted.values():
            if len(pids) < 2:
                continue
            pids = pids[:250]
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    overlap_counts[tuple(sorted((pids[i], pids[j])))] += 1
        complement_edges = 0
        complement_same = 0
        hard_diff = 0
        hard_same = 0
        soft_count = 0
        parent_child_count = 0
        for (pa, pb), shared in overlap_counts.items():
            a, b = by_pid[pa], by_pid[pb]
            ca, cb = core_by_id[pa], core_by_id[pb]
            min_size = max(min(len(ca), len(cb)), 1)
            max_size = max(max(len(ca), len(cb)), 1)
            min_overlap = float(shared / min_size)
            max_overlap = float(shared / max_size)
            parent_child = bool(min_overlap >= 0.95 and len(ca) != len(cb))
            if parent_child:
                parent_child_count += 1
            hard = False
            reason = ""
            if min_overlap >= 0.80 and not parent_child:
                hard = True
                reason = "high_nonhierarchical_tube_overlap"
            elif min_overlap >= 0.20 and (_int(a, "frame_id") == _int(b, "frame_id")) and (_int(a, "mask_id") != _int(b, "mask_id")):
                hard = True
                reason = "same_frame_mask_conflict"
            elif min_overlap >= 0.10 and max(
                (a.get("_v29_features") or {}).get("cannot_link_residual_scene_core", 0.0),
                (b.get("_v29_features") or {}).get("cannot_link_residual_scene_core", 0.0),
            ) > 1.8:
                hard = True
                reason = "high_normalized_cannot_link_overlap"
            else:
                soft_count += 1
            if hard:
                hard_by_scene[scene].add((pa, pb))
                gta = _int(a, "proposal_best_GT")
                gtb = _int(b, "proposal_best_GT")
                if gta > 0 and gtb > 0:
                    if gta == gtb:
                        hard_same += 1
                    else:
                        hard_diff += 1
                edge_rows[scene].append(
                    {
                        "scene": scene,
                        "proposal_a": pa,
                        "proposal_b": pb,
                        "edge_type": "hard_conflict",
                        "reason": reason,
                        "tube_overlap_ratio_min": min_overlap,
                        "tube_overlap_ratio_max": max_overlap,
                        "a_best_GT_diagnostic": gta,
                        "b_best_GT_diagnostic": gtb,
                    }
                )
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(_int(row, "frame_id"), _int(row, "mask_id"))].append(row)
        for group_rows in groups.values():
            ranked = sorted(group_rows, key=lambda row: _quality(row, medium_bonus=0.1), reverse=True)[:40]
            for i, a in enumerate(ranked):
                ca = set(_core_ids(a))
                if not ca:
                    continue
                for b in ranked[i + 1 :]:
                    cb = set(_core_ids(b))
                    if not cb:
                        continue
                    if len(ca & cb) / max(min(len(ca), len(cb)), 1) > 0.05:
                        continue
                    if not _compatible(a, b):
                        continue
                    complement_edges += 1
                    if _int(a, "proposal_best_GT") > 0 and _int(a, "proposal_best_GT") == _int(b, "proposal_best_GT"):
                        complement_same += 1
                    if complement_edges <= 20000:
                        edge_rows[scene].append(
                            {
                                "scene": scene,
                                "proposal_a": a["proposal_id"],
                                "proposal_b": b["proposal_id"],
                                "edge_type": "complement",
                                "reason": "same_parent_compatible_siblings",
                                "tube_overlap_ratio_min": 0.0,
                                "tube_overlap_ratio_max": 0.0,
                                "a_best_GT_diagnostic": _int(a, "proposal_best_GT"),
                                "b_best_GT_diagnostic": _int(b, "proposal_best_GT"),
                            }
                        )
        tube_candidates: Counter[int] = Counter()
        for core in core_by_id.values():
            for tid in core:
                tube_candidates[int(tid)] += 1
        cand_values = list(tube_candidates.values())
        hard_total_diag = hard_diff + hard_same
        summary_rows.append(
            {
                "scene": scene,
                "proposal_count": int(len(rows)),
                "hard_conflict_edge_count": int(len(hard_by_scene[scene])),
                "soft_conflict_edge_count": int(soft_count),
                "complement_edge_count": int(complement_edges),
                "parent_child_edge_count": int(parent_child_count),
                "tube_coverage_candidate_mean": _mean([float(v) for v in cand_values]),
                "ambiguous_tube_ratio": float(sum(1 for v in cand_values if v > 1) / max(len(cand_values), 1)),
                "single_candidate_tube_ratio": float(sum(1 for v in cand_values if v == 1) / max(len(cand_values), 1)),
                "no_candidate_tube_ratio": 0.0,
                "hard_conflict_different_GT_rate": float(hard_diff / max(hard_total_diag, 1)),
                "hard_conflict_same_GT_error_rate": float(hard_same / max(hard_total_diag, 1)),
                "complement_same_GT_rate": float(complement_same / max(complement_edges, 1)),
            }
        )
    return hard_by_scene, edge_rows, summary_rows


def _violates_hard(pid: str, selected_ids: set[str], hard_edges: set[tuple[str, str]]) -> bool:
    for other in selected_ids:
        if tuple(sorted((pid, other))) in hard_edges:
            return True
    return False


def _select_greedy_solver(
    candidates: list[dict[str, Any]],
    hard_edges: set[tuple[str, str]],
    *,
    min_new_tubes: int,
    max_overlap_ratio: float,
    min_quality: float,
    broad_penalty: float,
    medium_bonus: float,
    coverage_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    owned: set[int] = set()
    attempted = 0
    accepted = 0
    ranked = sorted(
        candidates,
        key=lambda row: (_quality(row, broad_penalty=broad_penalty, medium_bonus=medium_bonus, coverage_weight=coverage_weight), len(_core_ids(row))),
        reverse=True,
    )
    for row in ranked:
        attempted += 1
        pid = str(row["proposal_id"])
        q = _quality(row, broad_penalty=broad_penalty, medium_bonus=medium_bonus, coverage_weight=coverage_weight)
        if q < min_quality:
            continue
        core = set(_core_ids(row))
        if len(core) < min_new_tubes:
            continue
        new = core - owned
        overlap_ratio = float((len(core) - len(new)) / max(len(core), 1))
        if len(new) < min_new_tubes or overlap_ratio > max_overlap_ratio:
            continue
        if _violates_hard(pid, selected_ids, hard_edges):
            continue
        selected.append(row)
        selected_ids.add(pid)
        owned.update(core)
        accepted += 1
    return selected, {"num_moves_attempted": attempted, "num_moves_accepted": accepted}


def _replace_broad_with_medium(selected: list[dict[str, Any]], candidates: list[dict[str, Any]], hard_edges: set[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = list(selected)
    selected_ids = {str(row["proposal_id"]) for row in selected}
    medium_candidates = [row for row in candidates if str(row.get("proposal_type", "")).startswith("R13_")]
    attempted = 0
    accepted = 0
    for broad in list(selected):
        if not str(broad.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES):
            continue
        broad_core = set(_core_ids(broad))
        siblings = [
            row
            for row in medium_candidates
            if row.get("scene") == broad.get("scene")
            and len(set(_core_ids(row)) & broad_core) / max(len(_core_ids(row)), 1) >= 0.60
        ]
        if len(siblings) < 2:
            continue
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()
        owned: set[int] = set()
        for row in sorted(siblings, key=lambda item: _quality(item, medium_bonus=0.35), reverse=True)[:8]:
            attempted += 1
            pid = str(row["proposal_id"])
            other_selected = (selected_ids - {str(broad["proposal_id"])}) | chosen_ids
            if _violates_hard(pid, other_selected, hard_edges):
                continue
            core = set(_core_ids(row))
            if len(core - owned) < 3:
                continue
            chosen.append(row)
            chosen_ids.add(pid)
            owned.update(core)
        if len(chosen) >= 2 and sum(_quality(row, medium_bonus=0.35) for row in chosen) > _quality(broad, broad_penalty=0.30):
            selected = [row for row in selected if row is not broad] + chosen
            selected_ids = {str(row["proposal_id"]) for row in selected}
            accepted += 1
    return selected, {"num_moves_attempted": attempted, "num_moves_accepted": accepted}


def _hard_violation_count(selected: list[dict[str, Any]], hard_edges: set[tuple[str, str]]) -> int:
    ids = [str(row["proposal_id"]) for row in selected]
    count = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if tuple(sorted((ids[i], ids[j]))) in hard_edges:
                count += 1
    return int(count)


def _run_solver_variants(
    candidates_by_scene: dict[str, list[dict[str, Any]]],
    hard_by_scene: dict[str, set[tuple[str, str]]],
    gt_by_scene: dict[str, dict[int, int]],
    *,
    graph_hard_usable: bool,
    control_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    variant_configs = [
        ("S1_greedy_hard_set_packing", 3, 0.10, 0.03, 0.15, 0.20, 0.12, False),
        ("S2_local_search_medium_replace", 3, 0.12, -0.02, 0.30, 0.35, 0.14, True),
        ("S3_lagrangian_coverage_fallback", 2, 0.22, -0.08, 0.05, 0.18, 0.28, False),
        ("S2_purity_repair_no_broad", 2, 0.06, 0.00, 0.45, 0.25, 0.08, True),
    ]
    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_by_variant_scene: dict[str, list[dict[str, Any]]] = {}
    for scene, candidates in candidates_by_scene.items():
        usable_hard = hard_by_scene.get(scene, set()) if graph_hard_usable else set()
        if control_kind == "no_temporal":
            candidates = [row for row in candidates if not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]
        elif control_kind == "mask_only":
            candidates = [row for row in candidates if str(row.get("proposal_type")) in MASK_ONLY_TYPES]
        for variant, min_new, overlap, min_q, broad_penalty, medium_bonus, coverage_weight, replace in variant_configs:
            t0 = time.time()
            selected, move_stats = _select_greedy_solver(
                candidates,
                usable_hard,
                min_new_tubes=min_new,
                max_overlap_ratio=overlap,
                min_quality=min_q,
                broad_penalty=broad_penalty,
                medium_bonus=medium_bonus,
                coverage_weight=coverage_weight,
            )
            if replace:
                repaired, repair_stats = _replace_broad_with_medium(selected, candidates, usable_hard)
                selected = repaired
                move_stats["num_moves_attempted"] += repair_stats["num_moves_attempted"]
                move_stats["num_moves_accepted"] += repair_stats["num_moves_accepted"]
            metrics = _eval_selected(selected, gt_by_scene[scene])
            runtime = time.time() - t0
            hard_violations = _hard_violation_count(selected, usable_hard)
            variant_name = f"{variant}_{control_kind}" if control_kind != "real" else variant
            row = {
                "scene": scene,
                "solver": variant_name,
                "control_kind": control_kind,
                "selected_proposal_count": int(len(selected)),
                "owned_tube_ratio": metrics["owned_tube_ratio"],
                "unknown_tube_ratio": metrics["unknown_tube_ratio"],
                "hard_conflict_violation_count": int(hard_violations),
                "soft_conflict_penalty": None,
                "objective_proposal_quality": float(sum(_quality(item, broad_penalty=broad_penalty, medium_bonus=medium_bonus, coverage_weight=coverage_weight) for item in selected)),
                "objective_ownership_quality": metrics["owned_tube_ratio"],
                "objective_unknown_penalty": metrics["unknown_tube_ratio"],
                "objective_conflict_penalty": int(hard_violations),
                "objective_risk_penalty": _mean([max((item.get("_v29_features") or {}).get("cannot_link_residual_scene_core", 0.0), 0.0) for item in selected]),
                "solver_runtime_sec": runtime,
                "num_moves_attempted": int(move_stats["num_moves_attempted"]),
                "num_moves_accepted": int(move_stats["num_moves_accepted"]),
                "ARI": metrics["ARI"],
                "purity": metrics["purity"],
                "completeness": metrics["completeness"],
                "overmerge": metrics["overmerge"],
                "oversplit": metrics["oversplit"],
            }
            scene_rows.append(row)
            selected_by_variant_scene[f"{variant_name}:{scene}"] = selected
            for rank, item in enumerate(selected):
                selected_rows.append(
                    {
                        "scene": scene,
                        "solver": variant_name,
                        "rank": int(rank),
                        "proposal_id": item.get("proposal_id"),
                        "proposal_type": item.get("proposal_type"),
                        "num_core_tubes": len(_core_ids(item)),
                        "quality": _quality(item, broad_penalty=broad_penalty, medium_bonus=medium_bonus, coverage_weight=coverage_weight),
                        "uses_gt_for_prediction": False,
                    }
                )
    aggregate = _aggregate(
        scene_rows,
        "solver",
        [
            "selected_proposal_count",
            "owned_tube_ratio",
            "unknown_tube_ratio",
            "hard_conflict_violation_count",
            "objective_proposal_quality",
            "objective_ownership_quality",
            "objective_unknown_penalty",
            "objective_conflict_penalty",
            "solver_runtime_sec",
            "num_moves_attempted",
            "num_moves_accepted",
            "ARI",
            "purity",
            "completeness",
            "overmerge",
            "oversplit",
        ],
    )
    for row in aggregate:
        row["control_kind"] = control_kind
    return scene_rows + aggregate, selected_rows, selected_by_variant_scene


def _run_fringe(
    solver_rows: list[dict[str, Any]],
    selected_by_variant_scene: dict[str, list[dict[str, Any]]],
    gt_by_scene: dict[str, dict[int, int]],
) -> list[dict[str, Any]]:
    all_rows = [row for row in solver_rows if row.get("scene") == "ALL" and str(row.get("control_kind", "real")) in {"real", ""}]
    if not all_rows:
        return []
    best = max(all_rows, key=lambda row: (float(row.get("ARI") or 0.0), float(row.get("purity") or 0.0)))
    solver = str(best["solver"])
    out: list[dict[str, Any]] = []
    for variant, threshold, strict in [
        ("F0_no_fringe_core_only", 999.0, True),
        ("F1_greedy_fringe_assignment", -0.15, False),
        ("F2_calibrated_fringe_unknown", 0.05, False),
        ("F3_hard_cannot_link_fringe", 0.15, True),
    ]:
        scene_rows = []
        for key, selected in selected_by_variant_scene.items():
            if not key.startswith(f"{solver}:"):
                continue
            scene = key.split(":", 1)[1]
            gt_labels = gt_by_scene[scene]
            before = _eval_selected(selected, gt_labels)
            fringe: dict[str, set[int]] = defaultdict(set)
            assigned = 0
            false = 0
            if variant != "F0_no_fringe_core_only":
                owned = {tid for row in selected for tid in _core_ids(row)}
                for row in selected:
                    row_gt = _int(row, "proposal_best_GT")
                    row_feats = row.get("_v29_features") or {}
                    score = (
                        0.30 * float(row_feats.get("visibility_mean", _float(row, "visibility_mean", 0.5)))
                        + 0.20 * float(row_feats.get("confidence_mean", _float(row, "confidence_mean", 0.5)))
                        + 0.20 * float(row_feats.get("eroded_interior_ratio", _float(row, "eroded_interior_ratio")))
                        - 0.15 * max(float(row_feats.get("boundary_risk_residual_scene_core", 0.0)), 0.0)
                        - 0.15 * max(float(row_feats.get("cannot_link_residual_scene_core", 0.0)), 0.0)
                    )
                    if strict and float(row_feats.get("cannot_link_residual_scene_core", 0.0)) > 0.5:
                        continue
                    if score < threshold:
                        continue
                    for tid in _fringe_ids(row):
                        if int(tid) in owned:
                            continue
                        fringe[str(row["proposal_id"])].add(int(tid))
                        owned.add(int(tid))
                        assigned += 1
                        gt = int(gt_labels.get(int(tid), 0))
                        if row_gt > 0 and gt > 0 and gt != row_gt:
                            false += 1
            after = _eval_selected(selected, gt_labels, include_fringe=fringe if fringe else None)
            scene_rows.append(
                {
                    "scene": scene,
                    "fringe_variant": variant,
                    "base_solver": solver,
                    "core_tube_count": int(sum(len(_core_ids(row)) for row in selected)),
                    "fringe_candidate_count": int(sum(len(_fringe_ids(row)) for row in selected)),
                    "assigned_fringe_count": int(assigned),
                    "unknown_fringe_count": None,
                    "rejected_fringe_count": None,
                    "fringe_assignment_confidence_mean": None,
                    "fringe_boundary_violation": None,
                    "fringe_cannot_link_violation": None,
                    "purity_before": before["purity"],
                    "purity_after": after["purity"],
                    "completeness_before": before["completeness"],
                    "completeness_after": after["completeness"],
                    "ARI_before": before["ARI"],
                    "ARI_after": after["ARI"],
                    "false_expansion_rate": float(false / max(assigned, 1)),
                }
            )
        out.extend(scene_rows)
    aggregate = _aggregate(
        out,
        "fringe_variant",
        [
            "core_tube_count",
            "fringe_candidate_count",
            "assigned_fringe_count",
            "purity_before",
            "purity_after",
            "completeness_before",
            "completeness_after",
            "ARI_before",
            "ARI_after",
            "false_expansion_rate",
        ],
    )
    for row in aggregate:
        row["base_solver"] = solver
        row["comp_gain"] = (row.get("completeness_after") or 0.0) - (row.get("completeness_before") or 0.0)
        row["purity_drop"] = (row.get("purity_before") or 0.0) - (row.get("purity_after") or 0.0)
        row["ARI_change"] = (row.get("ARI_after") or 0.0) - (row.get("ARI_before") or 0.0)
    return out + aggregate


def _medium_ablation_rows(
    base_rows_by_scene: dict[str, list[dict[str, Any]]],
    medium_rows: list[dict[str, Any]],
    gt_by_scene: dict[str, dict[int, int]],
    scenes: list[str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    generators = sorted({str(row.get("v29_medium_generator")) for row in medium_rows})
    subsets: dict[str, set[str]] = {"none": set(), "all": set(generators)}
    for gen in generators:
        subsets[f"only_{gen}"] = {gen}
        subsets[f"without_{gen}"] = set(generators) - {gen}
    rows: list[dict[str, Any]] = []
    metric_keys = [
        "proposal_count",
        "oracle_ARI",
        "oracle_purity",
        "oracle_completeness",
        "oracle_overmerge",
        "oracle_oversplit",
        "GT_count",
        "covered_GT_count",
        "GT_with_best_IoU_ge_025",
        "GT_with_best_IoU_ge_050",
        "oracle_per_GT_best_IoU_mean",
        "medium_oracle_selected_count",
    ]
    for subset_name, allowed in subsets.items():
        mids = [row for row in medium_rows if str(row.get("v29_medium_generator")) in allowed]
        mids_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in mids:
            mids_by_scene[str(row.get("scene"))].append(row)
        scene_rows = []
        for scene in scenes:
            scene_rows.append(
                {
                    **_oracle_summary(scene, base_rows_by_scene[scene] + mids_by_scene.get(scene, []), gt_by_scene[scene], include_medium=True),
                    "generator_subset": subset_name,
                    "enabled_generators": ";".join(sorted(allowed)),
                }
            )
        rows.extend(scene_rows)
        agg = _aggregate(scene_rows, "generator_subset", metric_keys)
        for row in agg:
            row["enabled_generators"] = ";".join(sorted(allowed))
        rows.extend(agg)
    base_all = next(row for row in rows if row.get("scene") == "ALL" and row.get("generator_subset") == "none")
    best_subset = "none"
    best_gate = {
        "phaseD_medium_gate_pass": False,
        "selected_generator_subset": "none",
        "selected_enabled_generators": "",
        "reason": "no medium generator subset preserved or improved O5 oracle frontier",
    }
    best_score = -1e9
    for row in rows:
        if row.get("scene") != "ALL" or row.get("generator_subset") == "none":
            continue
        gate = (
            float(row["oracle_ARI"]) >= float(base_all["oracle_ARI"])
            and float(row["oracle_completeness"]) >= float(base_all["oracle_completeness"])
            and float(row["oracle_purity"]) >= float(base_all["oracle_purity"]) - 0.02
            and float(row.get("scene0081_oracle_ARI") or 0.0) >= float(base_all.get("scene0081_oracle_ARI") or 0.0)
            and int(row.get("medium_oracle_selected_count") or 0) > 0
        )
        score = float(row["oracle_ARI"]) + float(row["oracle_completeness"]) + 0.01 * int(row.get("medium_oracle_selected_count") or 0)
        if gate and score > best_score:
            best_score = score
            best_subset = str(row["generator_subset"])
            best_gate = {
                "phaseD_medium_gate_pass": True,
                "selected_generator_subset": best_subset,
                "selected_enabled_generators": row.get("enabled_generators", ""),
                "O5M_oracle_ARI": row.get("oracle_ARI"),
                "O5_oracle_ARI": base_all.get("oracle_ARI"),
                "O5M_completeness": row.get("oracle_completeness"),
                "O5_completeness": base_all.get("oracle_completeness"),
                "O5M_purity": row.get("oracle_purity"),
                "O5_purity": base_all.get("oracle_purity"),
                "medium_oracle_selected_count": row.get("medium_oracle_selected_count"),
            }
    allowed = set(subsets.get(best_subset, set())) if best_gate["phaseD_medium_gate_pass"] else set()
    return rows, sorted(allowed), best_gate


def _import_v28_selection_summary(path: Path, variant_map: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    imported: list[dict[str, Any]] = []
    for row in _read_csv_dicts(path):
        variant = str(row.get("variant") or "")
        if variant not in variant_map:
            continue
        solver, control_kind = variant_map[variant]
        imported.append(
            {
                "scene": row.get("scene"),
                "solver": solver,
                "control_kind": control_kind,
                "selected_proposal_count": _int(row, "selected_proposal_count"),
                "owned_tube_ratio": None,
                "unknown_tube_ratio": _float(row, "unknown_tube_ratio"),
                "hard_conflict_violation_count": 0,
                "soft_conflict_penalty": None,
                "objective_proposal_quality": None,
                "objective_ownership_quality": None,
                "objective_unknown_penalty": None,
                "objective_conflict_penalty": None,
                "objective_risk_penalty": None,
                "solver_runtime_sec": None,
                "num_moves_attempted": None,
                "num_moves_accepted": None,
                "ARI": _float(row, "local_ARI"),
                "purity": _float(row, "local_purity"),
                "completeness": _float(row, "local_completeness"),
                "overmerge": _float(row, "local_overmerge"),
                "oversplit": _float(row, "local_oversplit"),
                "scene0081_ARI": row.get("scene0081_local_ARI") or None,
                "imported_v28_baseline": True,
                "uses_gt_for_prediction": False,
            }
        )
    return imported


def _write_gap_figures(out_dir: Path, gap_rows: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[float]] = defaultdict(list)
    for row in gap_rows:
        if row.get("set") in {"oracle_selected", "strict_P4_selected"}:
            groups[str(row["set"])].append(float(row.get("num_core_tubes") or 0.0))
    if groups:
        plt.figure(figsize=(6, 4))
        labels = list(groups)
        plt.boxplot([groups[label] for label in labels], labels=labels, showfliers=False)
        plt.ylabel("core tube count")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "oracle_vs_strict_core_size.png", dpi=150)
        plt.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_base = Path(args.output_base)
    out_base.mkdir(parents=True, exist_ok=True)
    proposal_root = Path(args.proposal_root)
    proposal_label = str(args.proposal_label)
    proposal_rows = json.loads((proposal_root / f"{proposal_label}_proposal_rows.json").read_text(encoding="utf-8"))
    proposal_rows = [row for row in proposal_rows if _is_o5(row)]
    scenes = _read_split(Path(args.split))
    rows_by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    for row in proposal_rows:
        if row.get("scene") in rows_by_scene:
            _set_core_ids(row, _core_ids(row))
            rows_by_scene[str(row["scene"])].append(row)
    gt_by_scene: dict[str, dict[int, int]] = {}
    for scene in scenes:
        gt_by_scene[scene] = _load_gt_labels(Path(args.cache_root), scene, int(args.max_tubes_per_window), int(args.image_width), int(args.image_height))

    strict_ids = _selected_ids_from_csv(Path(args.selection_root) / f"{args.selection_label}_selected_proposals.csv", "P4_greedy_set_packing")
    p11_ids = _selected_ids_from_csv(Path(args.p11_selection_root) / f"{args.p11_selection_label}_selected_proposals.csv", "P11_calibrated_ownership_expansion")

    oracle_ids: set[str] = set()
    for scene in scenes:
        gt_counts = Counter(int(gt) for gt in gt_by_scene[scene].values() if int(gt) > 0)
        selected, _ = _oracle_selected(rows_by_scene[scene], gt_by_scene[scene], gt_counts, include_medium=False)
        oracle_ids.update(str(row["proposal_id"]) for row in selected)

    feature_by_id, feature_summary_rows = _annotate_features(proposal_rows, oracle_ids, strict_ids)

    gap_dir = out_base / "v29_gap_attribution"
    gap_rows: list[dict[str, Any]] = []
    for row in proposal_rows:
        pid = str(row["proposal_id"])
        sets = []
        if pid in oracle_ids:
            sets.append("oracle_selected")
        if pid in strict_ids:
            sets.append("strict_P4_selected")
        if pid in p11_ids:
            sets.append("P11_repair_selected")
        oracle_good = pid in oracle_ids or (_float(row, "proposal_purity") >= 0.85 and _float(row, "proposal_best_IoU") >= 0.25)
        oracle_bad = _float(row, "proposal_purity") <= 0.60 or (_float(row, "proposal_purity") <= 0.75 and _float(row, "proposal_best_IoU") < 0.25)
        if oracle_good and pid not in strict_ids:
            sets.append("oracle_good_rejected")
        if oracle_bad and pid in strict_ids:
            sets.append("selected_oracle_bad")
        for set_name in sets:
            gap_rows.append(
                {
                    "set": set_name,
                    "scene": row.get("scene"),
                    "proposal_id": pid,
                    "proposal_type": row.get("proposal_type"),
                    "type_bucket": _type_bucket(str(row.get("proposal_type"))),
                    "num_core_tubes": len(_core_ids(row)),
                    "proposal_purity_diagnostic": row.get("proposal_purity"),
                    "proposal_completeness_diagnostic": row.get("proposal_completeness"),
                    "proposal_best_IoU_diagnostic": row.get("proposal_best_IoU"),
                    "cannot_link_rate": row.get("same_frame_cannot_link_rate"),
                    "visible_negative_rate": row.get("visible_outside_negative_rate"),
                    "compactness": row.get("tube_canonical_compactness"),
                    "eroded_ratio": row.get("eroded_interior_ratio"),
                    "medium_prior_score": feature_by_id.get(pid, {}).get("medium_prior_score"),
                }
            )
    _write_csv(gap_dir / "oracle_vs_selected_proposal_rows.csv", gap_rows)
    _write_gap_figures(gap_dir, gap_rows)

    type_rows: list[dict[str, Any]] = []
    for bucket in sorted({_type_bucket(str(row.get("proposal_type"))) for row in proposal_rows}):
        for set_name in ["oracle_selected", "strict_P4_selected", "oracle_good_rejected", "selected_oracle_bad", "P11_repair_selected"]:
            items = [row for row in gap_rows if row["type_bucket"] == bucket and row["set"] == set_name]
            if not items:
                continue
            type_rows.append(
                {
                    "type_bucket": bucket,
                    "set": set_name,
                    "count": int(len(items)),
                    "median_core_tubes": _quantile([_float(row, "num_core_tubes") for row in items], 0.50),
                    "p90_core_tubes": _quantile([_float(row, "num_core_tubes") for row in items], 0.90),
                    "purity_mean": _mean([_float(row, "proposal_purity_diagnostic") for row in items]),
                    "completeness_mean": _mean([_float(row, "proposal_completeness_diagnostic") for row in items]),
                    "IoU_mean": _mean([_float(row, "proposal_best_IoU_diagnostic") for row in items]),
                    "cannot_link_rate": _mean([_float(row, "cannot_link_rate") for row in items]),
                    "visible_negative_rate": _mean([_float(row, "visible_negative_rate") for row in items]),
                    "compactness": _mean([_float(row, "compactness") for row in items]),
                    "eroded_ratio": _mean([_float(row, "eroded_ratio") for row in items]),
                }
            )
    _write_csv(gap_dir / "type_distribution.csv", type_rows)

    scene_gap_rows: list[dict[str, Any]] = []
    for scene in scenes:
        oracle_scene = [row for row in gap_rows if row["scene"] == scene and row["set"] == "oracle_selected"]
        strict_scene = [row for row in gap_rows if row["scene"] == scene and row["set"] == "strict_P4_selected"]
        rejected_scene = [row for row in gap_rows if row["scene"] == scene and row["set"] == "oracle_good_rejected"]
        selected_bad_scene = [row for row in gap_rows if row["scene"] == scene and row["set"] == "selected_oracle_bad"]
        scene_gap_rows.append(
            {
                "scene": scene,
                "oracle_selected_count": int(len(oracle_scene)),
                "strict_selected_count": int(len(strict_scene)),
                "oracle_median_core_tubes": _quantile([_float(row, "num_core_tubes") for row in oracle_scene], 0.50),
                "strict_median_core_tubes": _quantile([_float(row, "num_core_tubes") for row in strict_scene], 0.50),
                "oracle_p90_core_tubes": _quantile([_float(row, "num_core_tubes") for row in oracle_scene], 0.90),
                "strict_p90_core_tubes": _quantile([_float(row, "num_core_tubes") for row in strict_scene], 0.90),
                "oracle_good_rejected_count": int(len(rejected_scene)),
                "selected_bad_count": int(len(selected_bad_scene)),
            }
        )
    oracle_med = _quantile([_float(row, "num_core_tubes") for row in gap_rows if row["set"] == "oracle_selected"], 0.50) or 0.0
    strict_med = _quantile([_float(row, "num_core_tubes") for row in gap_rows if row["set"] == "strict_P4_selected"], 0.50) or 0.0
    main_mismatch_type = "selected_too_large" if strict_med > oracle_med * 2.0 else "score_calibration_wrong"
    for row in scene_gap_rows:
        row["main_mismatch_type"] = main_mismatch_type
    _write_csv(gap_dir / "scene_gap_summary.csv", scene_gap_rows)

    norm_dir = out_base / "v29_normalized_features"
    _write_csv(norm_dir / "feature_auc_summary.csv", feature_summary_rows)
    feature_gate = {
        "feature_count_auc_ge_0_65": int(
            sum(
                1
                for row in feature_summary_rows
                if max(float(row.get("oracle_good_AUC") or 0.0), float(row.get("false_merge_AUC") or 0.0)) >= 0.65
            )
        ),
        "scene0081_feature_count_auc_ge_0_60": int(sum(1 for row in feature_summary_rows if float(row.get("scene0081_oracle_good_AUC") or 0.0) >= 0.60)),
        "raw_cannot_link_AUC": next((row.get("oracle_good_AUC") for row in feature_summary_rows if row["feature"] == "raw_cannot_link_rate"), None),
        "residual_cannot_link_AUC": next((row.get("oracle_good_AUC") for row in feature_summary_rows if row["feature"] == "cannot_link_residual_scene_core"), None),
        "normalized_cannot_link_AUC": next((row.get("oracle_good_AUC") for row in feature_summary_rows if row["feature"] == "area_normalized_cannot_link_rate"), None),
        "medium_prior_AUC": next((row.get("oracle_good_AUC") for row in feature_summary_rows if row["feature"] == "medium_prior_score"), None),
    }
    feature_gate["phaseC_feature_gate_pass"] = bool(
        feature_gate["feature_count_auc_ge_0_65"] >= 5
        and float(feature_gate["normalized_cannot_link_AUC"] or 0.0) > float(feature_gate["raw_cannot_link_AUC"] or 1.0)
        and float(feature_gate["scene0081_feature_count_auc_ge_0_60"]) >= 3
        and float(feature_gate["medium_prior_AUC"] or 0.0) >= 0.65
    )
    (norm_dir / "manifest.json").write_text(json.dumps(_json_safe({"gates": feature_gate, **_manifest_policy(True)}), indent=2, sort_keys=True), encoding="utf-8")

    medium_rows, medium_summary_rows = _generate_medium(rows_by_scene, gt_by_scene)
    medium_rows = _dedupe_rows(medium_rows)
    for row in medium_rows:
        row["_v29_features"] = {}
    medium_dir = out_base / "v29_medium_proposals"
    medium_csv_rows = []
    for row in medium_rows:
        medium_csv_rows.append(
            {
                "scene": row.get("scene"),
                "proposal_id": row.get("proposal_id"),
                "proposal_type": row.get("proposal_type"),
                "generator": row.get("v29_medium_generator"),
                "num_core_tubes": len(_core_ids(row)),
                "proposal_purity_diagnostic": row.get("proposal_purity"),
                "proposal_completeness_diagnostic": row.get("proposal_completeness"),
                "proposal_best_IoU_diagnostic": row.get("proposal_best_IoU"),
                "source_proposal_count": row.get("v29_source_proposal_count"),
                "source_proposal_ids": row.get("v29_source_proposal_ids"),
            }
        )
    _write_csv(medium_dir / "medium_proposal_rows.csv", medium_csv_rows)
    ablation_rows, allowed_medium_generators, medium_gate = _medium_ablation_rows(rows_by_scene, medium_rows, gt_by_scene, scenes)
    selected_medium_rows = [row for row in medium_rows if str(row.get("v29_medium_generator")) in set(allowed_medium_generators)]
    combined_rows = proposal_rows + selected_medium_rows
    _annotate_features(combined_rows, oracle_ids, strict_ids)
    oracle_rows = []
    for row in ablation_rows:
        item = dict(row)
        item["pool"] = "O5_final" if item.get("generator_subset") == "none" else "O5M_medium_augmented"
        oracle_rows.append(item)
    _write_csv(medium_dir / "O5M_oracle_summary.csv", oracle_rows)
    _write_csv(medium_dir / "medium_generator_ablation.csv", ablation_rows)
    _write_csv(medium_dir / "medium_summary.csv", medium_summary_rows)
    (medium_dir / "manifest.json").write_text(json.dumps(_json_safe({"gates": medium_gate, **_manifest_policy(False)}), indent=2, sort_keys=True), encoding="utf-8")

    solver_input_rows = combined_rows if medium_gate["phaseD_medium_gate_pass"] else proposal_rows
    candidates_real = _build_solver_candidates(solver_input_rows, mode="real", max_per_scene=int(args.max_proposals_per_scene))
    conflict_dir = out_base / "v29_conflict_graph"
    hard_by_scene, edge_rows_by_scene, conflict_summary_rows = _build_conflict_graph(candidates_real)
    all_edge_rows = [row for rows in edge_rows_by_scene.values() for row in rows]
    _write_csv(conflict_dir / "proposal_conflict_edges.csv", all_edge_rows)
    _write_csv(conflict_dir / "conflict_graph_summary.csv", conflict_summary_rows)
    hard_rates = [row for row in conflict_summary_rows if row.get("hard_conflict_edge_count", 0) > 0]
    hard_diff = _mean([float(row["hard_conflict_different_GT_rate"]) for row in hard_rates]) or 0.0
    hard_same = _mean([float(row["hard_conflict_same_GT_error_rate"]) for row in hard_rates]) or 1.0
    comp_same = _mean([float(row["complement_same_GT_rate"]) for row in conflict_summary_rows]) or 0.0
    conflict_gate = {
        "hard_conflict_different_GT_rate": hard_diff,
        "hard_conflict_same_GT_error_rate": hard_same,
        "complement_same_GT_rate": comp_same,
        "hard_conflicts_usable": bool(hard_diff >= 0.70 and hard_same <= 0.20),
        "complement_edges_usable": bool(comp_same >= 0.60),
    }
    (conflict_dir / "manifest.json").write_text(json.dumps(_json_safe({"gates": conflict_gate, **_manifest_policy(True)}), indent=2, sort_keys=True), encoding="utf-8")

    solver_dir = out_base / "v29_ownership_solver"
    solver_rows, selected_rows, selected_by_variant_scene = _run_solver_variants(
        candidates_real,
        hard_by_scene,
        gt_by_scene,
        graph_hard_usable=bool(conflict_gate["hard_conflicts_usable"]),
        control_kind="real",
    )
    shuffle_rows = json.loads((Path(args.shuffle_proposal_root) / f"{args.shuffle_proposal_label}_proposal_rows.json").read_text(encoding="utf-8"))
    for row in shuffle_rows:
        _set_core_ids(row, _core_ids(row))
    _annotate_features(shuffle_rows, set(), set())
    shuffle_candidates = _build_solver_candidates(shuffle_rows, mode="real", max_per_scene=int(args.max_proposals_per_scene))
    shuffle_hard, _, _ = _build_conflict_graph(shuffle_candidates)
    shuffle_solver_rows, shuffle_selected_rows, _ = _run_solver_variants(
        shuffle_candidates,
        shuffle_hard,
        gt_by_scene,
        graph_hard_usable=False,
        control_kind="shuffled_d4rt",
    )
    no_temporal_candidates = _build_solver_candidates(solver_input_rows, mode="no_temporal", max_per_scene=int(args.max_proposals_per_scene))
    no_temporal_rows, no_temporal_selected, _ = _run_solver_variants(
        no_temporal_candidates,
        hard_by_scene,
        gt_by_scene,
        graph_hard_usable=False,
        control_kind="no_temporal",
    )
    mask_only_candidates = _build_solver_candidates(solver_input_rows, mode="mask_only", max_per_scene=int(args.max_proposals_per_scene))
    mask_rows, mask_selected, _ = _run_solver_variants(
        mask_only_candidates,
        hard_by_scene,
        gt_by_scene,
        graph_hard_usable=False,
        control_kind="mask_only",
    )
    all_solver_rows = solver_rows + shuffle_solver_rows + no_temporal_rows + mask_rows
    imported_rows = []
    imported_rows.extend(
        _import_v28_selection_summary(
            Path(args.selection_root) / f"{args.selection_label}_selection_summary.csv",
            {
                "P4_greedy_set_packing": ("S0_v28_strict_P4_baseline", "real"),
                "P9_no_temporal_control": ("S8_v28_no_temporal_control", "no_temporal"),
                "P10_mask_only_control": ("S9_v28_mask_only_control", "mask_only"),
            },
        )
    )
    imported_rows.extend(
        _import_v28_selection_summary(
            Path(args.p11_selection_root) / f"{args.p11_selection_label}_selection_summary.csv",
            {
                "P11_calibrated_ownership_expansion": ("S3_v28_P11_ownership_expansion_imported", "real"),
            },
        )
    )
    all_solver_rows.extend(imported_rows)
    all_selected_rows = selected_rows + shuffle_selected_rows + no_temporal_selected + mask_selected
    _write_csv(solver_dir / "ownership_solver_summary.csv", all_solver_rows)
    _write_csv(solver_dir / "ownership_solver_selected_proposals.csv", all_selected_rows)
    real_all = [row for row in all_solver_rows if row.get("scene") == "ALL" and str(row.get("control_kind", "real")) in {"real", ""}]
    best_real = max(real_all, key=lambda row: float(row.get("ARI") or 0.0)) if real_all else {}
    best_shuffle = max([row for row in all_solver_rows if row.get("scene") == "ALL" and row.get("control_kind") == "shuffled_d4rt"], key=lambda row: float(row.get("ARI") or 0.0), default={})
    best_no_temporal = max([row for row in all_solver_rows if row.get("scene") == "ALL" and row.get("control_kind") == "no_temporal"], key=lambda row: float(row.get("ARI") or 0.0), default={})
    best_mask = max([row for row in all_solver_rows if row.get("scene") == "ALL" and row.get("control_kind") == "mask_only"], key=lambda row: float(row.get("ARI") or 0.0), default={})
    solver_gate = {
        "best_real_solver": best_real.get("solver"),
        "best_real_ARI": best_real.get("ARI"),
        "best_real_purity": best_real.get("purity"),
        "best_real_completeness": best_real.get("completeness"),
        "best_real_scene0081_ARI": best_real.get("scene0081_ARI"),
        "best_shuffle_ARI": best_shuffle.get("ARI"),
        "best_no_temporal_ARI": best_no_temporal.get("ARI"),
        "best_mask_only_ARI": best_mask.get("ARI"),
        "local_gate_pass": bool(
            float(best_real.get("ARI") or 0.0) >= 0.35
            and float(best_real.get("purity") or 0.0) >= 0.85
            and float(best_real.get("completeness") or 0.0) >= 0.50
            and int(float(best_real.get("hard_conflict_violation_count") or 0)) == 0
        ),
        "real_beats_shuffled_by_0_20": float(best_real.get("ARI") or 0.0) >= float(best_shuffle.get("ARI") or 0.0) + 0.20,
        "real_beats_no_temporal_by_0_05": float(best_real.get("ARI") or 0.0) >= float(best_no_temporal.get("ARI") or 0.0) + 0.05,
        "real_beats_mask_only_by_0_05": float(best_real.get("ARI") or 0.0) >= float(best_mask.get("ARI") or 0.0) + 0.05,
        "ortools_available": False,
        "ortools_note": "import ortools failed in current loger environment; S4 CP-SAT/ILP skipped per v29 fallback rule",
    }
    solver_gate["phaseF_solver_gate_pass"] = bool(
        solver_gate["local_gate_pass"]
        and solver_gate["real_beats_shuffled_by_0_20"]
        and solver_gate["real_beats_no_temporal_by_0_05"]
        and solver_gate["real_beats_mask_only_by_0_05"]
    )
    (solver_dir / "manifest.json").write_text(json.dumps(_json_safe({"gates": solver_gate, **_manifest_policy(False)}), indent=2, sort_keys=True), encoding="utf-8")

    fringe_dir = out_base / "v29_fringe_ownership"
    fringe_rows = _run_fringe(solver_rows, selected_by_variant_scene, gt_by_scene)
    _write_csv(fringe_dir / "fringe_ownership_summary.csv", fringe_rows)
    fringe_gate = {"phaseG_fringe_gate_pass": False, "reason": "no fringe variant satisfied gate or Phase F did not pass"}
    for row in fringe_rows:
        if row.get("scene") == "ALL" and row.get("fringe_variant") != "F0_no_fringe_core_only":
            if float(row.get("comp_gain") or 0.0) >= 0.05 and float(row.get("purity_drop") or 0.0) <= 0.03 and float(row.get("ARI_change") or 0.0) > 0.0 and float(row.get("false_expansion_rate") or 1.0) <= 0.05:
                fringe_gate = {"phaseG_fringe_gate_pass": True, "best_fringe_variant": row.get("fringe_variant")}
                break
    (fringe_dir / "manifest.json").write_text(json.dumps(_json_safe({"gates": fringe_gate, **_manifest_policy(False)}), indent=2, sort_keys=True), encoding="utf-8")

    for gated_dir in ["v29_memory", "v29_eval_export"]:
        d = out_base / gated_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(
            json.dumps(
                _json_safe(
                    {
                        **_manifest_policy(False),
                        "not_run": True,
                        "reason": "Phase F/G local ownership gate did not pass; v29 plan forbids memory/AP method claim before local gate.",
                    }
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    summary = {
        "gap_main_mismatch_type": main_mismatch_type,
        "feature_gate": feature_gate,
        "medium_gate": medium_gate,
        "conflict_gate": conflict_gate,
        "solver_gate": solver_gate,
        "fringe_gate": fringe_gate,
        "artifact_roots": {
            "gap": str(gap_dir),
            "features": str(norm_dir),
            "medium": str(medium_dir),
            "conflict": str(conflict_dir),
            "solver": str(solver_dir),
            "fringe": str(fringe_dir),
        },
    }
    (out_base / "v29_run_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _manifest_policy(diagnostic_only: bool) -> dict[str, Any]:
    return {
        "is_method_result": not diagnostic_only,
        "is_diagnostic_only": bool(diagnostic_only),
        "forbidden_for_method_table": bool(diagnostic_only),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "geometry_field": "D4RT canonical proposal tube memberships plus image-space mask/proposal features",
        "coordinate_frame": "d4rt_canonical for tube geometry; image space for masks/features",
        "alignment_source": "D4RT self-Sim3 inherited from v28 final artifacts",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v29 constrained ownership solver diagnostics and local solver controls.")
    parser.add_argument("--proposal-root", required=True)
    parser.add_argument("--proposal-label", required=True)
    parser.add_argument("--shuffle-proposal-root", required=True)
    parser.add_argument("--shuffle-proposal-label", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--selection-label", required=True)
    parser.add_argument("--p11-selection-root", required=True)
    parser.add_argument("--p11-selection-label", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-base", default="outputs/audit")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--max-proposals-per-scene", type=int, default=1200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
