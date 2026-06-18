from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.diagnose_v30_cannot_link_clique import (
    _component_score,
    _f,
    _read_component_feature_rows,
    _row_core_ids,
    _row_quality,
)
from tools.run_v26_object_quality_diagnostics import _json_safe
from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v29_constrained_ownership_solver import _quantile, _set_core_ids
from tools.run_v29_constrained_ownership_solver import (
    MASK_ONLY_TYPES,
    TEMPORAL_PREFIXES,
    _annotate_features as _annotate_v29_features,
    _eval_selected as _eval_v29_selected,
    _select_greedy_solver as _select_v29_greedy_solver,
)
from tools.run_v30_object_slot_ownership import _aggregate_variant, _eval_slot_sets, _read_split


METHOD_MANIFEST_BASE: dict[str, Any] = {
    "is_method_result": False,
    "is_diagnostic_only": True,
    "forbidden_for_method_table": True,
    "uses_gt_for_prediction": False,
    "uses_gt_for_diagnostic_labels": True,
    "uses_rgbd_for_prediction": False,
    "uses_pose_for_prediction": False,
    "uses_scannet_mesh_for_prediction": False,
    "uses_eval_sim3_for_prediction": False,
    "uses_rgb_for_prediction": True,
    "uses_image_masks_for_prediction": True,
    "uses_d4rt_self_sim3": True,
    "geometry_field": "D4RT canonical tube ids plus RGB/mask/cannot-link validation features",
    "coordinate_frame": "d4rt_canonical tubes with image-space RGB/mask metadata",
    "alignment_source": "D4RT self-Sim3 inherited from v30/v28 artifacts",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _fmt_thr(thr: float) -> str:
    return f"{thr:.2f}"


def _overlap_min_norm(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(min(len(a), len(b)), 1))


def _load_profile_seed_rows(path: Path, source_profile: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            core_ids = tuple(int(v) for v in str(row.get("core_tube_ids") or "").split(";") if v.strip())
            if not core_ids:
                continue
            item = dict(row)
            item["source_profile"] = source_profile
            item["source_type"] = "v30_seed_profile"
            item["seed_source"] = row.get("seed_source") or source_profile
            item["proposal_id"] = row.get("proposal_id") or f"{source_profile}_{idx:06d}"
            item["seed_score"] = _f(row, "seed_score")
            item["core_n"] = len(core_ids)
            _set_core_ids(item, tuple(sorted(core_ids)))
            item["native_trust_score"] = _native_trust_score(item)
            item["validation_score"] = item["native_trust_score"]
            rows.append(item)
    return rows


def _native_trust_score(row: dict[str, Any]) -> float:
    core_n = len(_row_core_ids(row)) or int(_f(row, "core_n"))
    core = min(math.log1p(core_n) / math.log(64.0), 1.0)
    return float(
        0.35 * _f(row, "native_boundary_safe_ratio")
        + 0.25 * _f(row, "native_frame_mask_support_ratio")
        + 0.20 * _f(row, "native_rgb_pair_cos_p10")
        + 0.10 * min(_f(row, "native_boundary_distance_p10") / 32.0, 1.0)
        + 0.10 * core
    )


def _component_validation_features(row: dict[str, Any]) -> dict[str, float]:
    core = min(math.log1p(_f(row, "core_n")) / math.log(161.0), 1.0)
    checks = min(math.log1p(_f(row, "component_cannot_checks")) / math.log(2000.0), 1.0)
    rate = _f(row, "component_cannot_rate")
    rate_safe = max(0.0, 1.0 - rate / 0.45)
    compact = max(0.0, 1.0 - min(_f(row, "mean_xy_dist") / 384.0, 1.0))
    color = max(0.0, 1.0 - min(_f(row, "mean_color_dist") / 0.40, 1.0))
    edge_safe = max(0.0, 1.0 - min(_f(row, "edge_density"), 1.0))
    safe = _f(row, "safe_ratio")
    return {
        "core_score": core,
        "checks_score": checks,
        "cannot_link_safety": rate_safe,
        "compactness_score": compact,
        "rgb_consistency_score": color,
        "edge_safety_score": edge_safe,
        "safe_ratio_score": safe,
    }


def _select_topk_per_scene(
    rows: list[dict[str, Any]],
    scenes: list[str],
    score_key: str,
    topk_per_scene: int,
    seed_id_prefix: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for scene in scenes:
        scene_rows = [row for row in rows if str(row.get("scene")) == scene]
        ranked = sorted(
            scene_rows,
            key=lambda row: (float(_f(row, score_key)), int(_f(row, "core_n", len(_row_core_ids(row))))),
            reverse=True,
        )
        accepted: list[set[int]] = []
        for row in ranked:
            ids = set(_row_core_ids(row))
            if not ids:
                continue
            if any(_overlap_min_norm(ids, old) >= 0.88 for old in accepted):
                continue
            item = dict(row)
            item["seed_id"] = f"{scene}_{seed_id_prefix}_{len(accepted):05d}"
            item["selection_score_key"] = score_key
            item["selection_score"] = _f(item, score_key)
            selected.append(item)
            accepted.append(ids)
            if len(accepted) >= int(topk_per_scene):
                break
    return selected


def _seed_metrics(
    rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.25),
) -> dict[str, Any]:
    purities: list[float] = []
    best_ious: list[float] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        gt_labels = gt_by_scene[scene]
        gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
        gt_best_iou: dict[int, float] = {int(gt): 0.0 for gt in gt_counts}
        items = [row for row in rows if str(row.get("scene")) == scene]
        scene_purities: list[float] = []
        scene_best: list[float] = []
        for row in items:
            quality = _row_quality(row, gt_labels, gt_counts)
            if quality["purity"] is not None:
                purities.append(float(quality["purity"]))
                scene_purities.append(float(quality["purity"]))
            if quality["best_iou"] is not None:
                best_ious.append(float(quality["best_iou"]))
                scene_best.append(float(quality["best_iou"]))
            labeled = int(quality["labeled"])
            for gt, overlap in quality["counts"].items():
                iou = float(overlap / max(labeled + int(gt_counts[int(gt)]) - int(overlap), 1))
                gt_best_iou[int(gt)] = max(gt_best_iou[int(gt)], iou)
        scene_row: dict[str, Any] = {
            "scene": scene,
            "seed_count": int(len(items)),
            "labeled_seed_count": int(len(scene_purities)),
            "seed_purity_mean": _mean(scene_purities),
            "seed_purity_p10": _quantile(scene_purities, 0.10) if scene_purities else None,
            "seed_best_IoU_mean": _mean(scene_best),
        }
        for thr in thresholds:
            scene_row[f"GT_with_seed_IoU_ge_{_fmt_thr(thr)}"] = float(
                sum(1 for val in gt_best_iou.values() if val >= float(thr)) / max(len(gt_best_iou), 1)
            )
        scene_rows.append(scene_row)
    out: dict[str, Any] = {
        "n": int(len(rows)),
        "labeled_n": int(len(purities)),
        "purity_mean": _mean(purities),
        "purity_p10": _quantile(purities, 0.10) if purities else None,
        "seed_best_IoU_mean": _mean(best_ious),
        "seed_count_per_scene": ";".join(f"{row['scene']}={row['seed_count']}" for row in scene_rows),
        "scene_rows": scene_rows,
    }
    for thr in thresholds:
        key = f"GT_with_seed_IoU_ge_{_fmt_thr(thr)}"
        out[key] = _mean([float(row[key]) for row in scene_rows])
    out["scene0081_GT_with_seed_IoU_ge_0.05"] = next(
        (row["GT_with_seed_IoU_ge_0.05"] for row in scene_rows if row["scene"] == "scene0081_01"),
        None,
    )
    out["scene0081_GT_with_seed_IoU_ge_0.10"] = next(
        (row["GT_with_seed_IoU_ge_0.10"] for row in scene_rows if row["scene"] == "scene0081_01"),
        None,
    )
    return out


def _trusted_anchor_gate(metric: dict[str, Any]) -> dict[str, bool]:
    return {
        "trusted_seed_purity_mean_ge_0.93": bool(float(metric.get("purity_mean") or 0.0) >= 0.93),
        "trusted_seed_purity_p10_ge_0.80": bool(float(metric.get("purity_p10") or 0.0) >= 0.80),
        "GT_with_trusted_anchor_IoU_ge_0.05_ge_0.65": bool(
            float(metric.get("GT_with_seed_IoU_ge_0.05") or 0.0) >= 0.65
        ),
        "GT_with_trusted_anchor_IoU_ge_0.10_ge_0.45": bool(
            float(metric.get("GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.45
        ),
        "scene0081_GT_with_trusted_anchor_IoU_ge_0.05_ge_0.50": bool(
            float(metric.get("scene0081_GT_with_seed_IoU_ge_0.05") or 0.0) >= 0.50
        ),
    }


def _candidate_coverage_gate(metric: dict[str, Any]) -> dict[str, bool]:
    return {
        "candidate_GT_IoU_0.10_ge_0.75": bool(float(metric.get("GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.75),
        "candidate_GT_IoU_0.25_ge_0.45": bool(float(metric.get("GT_with_seed_IoU_ge_0.25") or 0.0) >= 0.45),
        "scene0081_candidate_IoU_0.10_ge_0.55": bool(
            float(metric.get("scene0081_GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.55
        ),
    }


def _parse_tube_ids(text: Any) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in str(text or "").split(";") if str(v).strip()))


def _load_proposal_core_map(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    if not path.exists():
        return rows, {"proposal_core_source_exists": False, "proposal_core_source_row_count": 0, "duplicate_proposal_id_count": 0}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            proposal_id = str(row.get("proposal_id") or "")
            core_ids = _parse_tube_ids(row.get("core_tube_ids"))
            if not proposal_id or not core_ids:
                continue
            if proposal_id in rows:
                duplicate_count += 1
            item = dict(row)
            _set_core_ids(item, core_ids)
            rows[proposal_id] = item
    return rows, {
        "proposal_core_source_exists": True,
        "proposal_core_source_row_count": int(len(rows)),
        "duplicate_proposal_id_count": int(duplicate_count),
    }


def _is_split_repair_factor(row: dict[str, Any]) -> bool:
    proposal_type = str(row.get("proposal_type") or "")
    if proposal_type.startswith(("R0_", "R1_", "R2_", "R3_", "R4_", "R5_", "R6_", "R7_")):
        return True
    return proposal_type.startswith(
        (
            "R10_temporal_tube_overlap_visible_negative_pruned_t35",
            "R10_temporal_tube_overlap_visible_negative_pruned_t50",
            "R10_temporal_tube_overlap_visible_negative_pruned_t70",
        )
    )


def _load_split_repair_factor_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    total_with_core = 0
    if not path.exists():
        return rows, {"split_repair_source_exists": False, "split_repair_source_row_count": 0}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            core_ids = _parse_tube_ids(row.get("core_tube_ids"))
            if not core_ids:
                continue
            total_with_core += 1
            if not _is_split_repair_factor(row):
                continue
            item = dict(row)
            _set_core_ids(item, core_ids)
            item["seed_role"] = "split_repair_factor"
            item["source_profile"] = "v28_split_repair_factor_pool"
            item["source_type"] = "non_gt_R0_R7_plus_R10_t35_t50_t70"
            rows.append(item)
            type_counts[str(item.get("proposal_type") or "")] += 1
    return rows, {
        "split_repair_source_exists": True,
        "split_repair_source_row_count": int(total_with_core),
        "split_repair_factor_count": int(len(rows)),
        "split_repair_type_counts": dict(sorted(type_counts.items())),
        "split_repair_filter": "R0-R7 mask split factors plus R10 visible-negative-pruned temporal t35/t50/t70; no GT filtering",
    }


def _read_broad_rows(path: Path, proposal_core_map: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    proposal_core_map = proposal_core_map or {}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item = dict(row)
            proposal_id = str(item.get("proposal_id") or "")
            source = proposal_core_map.get(proposal_id)
            if source is not None:
                _set_core_ids(item, _row_core_ids(source))
                item["core_reconstruction_source"] = "v28_proposal_rows_by_proposal_id"
                item["core_reconstruction_source_core_tube_count"] = source.get("core_tube_count")
                item["core_reconstruction_source_proposal_type"] = source.get("proposal_type")
            else:
                item["core_reconstruction_source"] = ""
            rows.append(item)
    return rows


def _quality_rows(
    rows: list[dict[str, Any]],
    gt_by_scene: dict[str, dict[int, int]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        scene = str(row.get("scene"))
        gt_labels = gt_by_scene[scene]
        gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
        q = _row_quality(row, gt_labels, gt_counts)
        item = dict(row)
        item["purity_diagnostic_recomputed"] = q["purity"]
        item["best_iou_diagnostic_recomputed"] = q["best_iou"]
        item["dominant_gt_diagnostic"] = q["best_gt"]
        item["low_tail_label_diagnostic"] = bool(q["purity"] is not None and float(q["purity"]) < 0.75)
        out.append(item)
    return out


def _rank_auc(score_values: list[float], labels: list[int]) -> float | None:
    n_pos = int(sum(labels))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(score_values, labels), key=lambda pair: float(pair[0]))
    rank_sum = sum(idx + 1 for idx, (_, label) in enumerate(pairs) if int(label) == 1)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def _augment_base_validation(base_rows: list[dict[str, Any]], trusted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors_by_scene: dict[str, list[set[int]]] = defaultdict(list)
    for row in trusted_rows:
        anchors_by_scene[str(row.get("scene"))].append(set(_row_core_ids(row)))
    out: list[dict[str, Any]] = []
    for row in base_rows:
        item = dict(row)
        ids = set(_row_core_ids(item))
        overlaps: list[float] = []
        for anchor in anchors_by_scene.get(str(item.get("scene")), []):
            ov = _overlap_min_norm(ids, anchor)
            if ov > 0:
                overlaps.append(ov)
        features = _component_validation_features(item)
        anchor_count_02 = sum(1 for val in overlaps if val >= 0.20)
        anchor_score = 0.50 * min(anchor_count_02 / 3.0, 1.0) + 0.50 * (max(overlaps) if overlaps else 0.0)
        risk = (
            0.35 * max(_f(item, "component_cannot_rate") - 0.10, 0.0) / 0.30
            + 0.25 * min(_f(item, "edge_density"), 1.0)
            + 0.20 * (1.0 - features["safe_ratio_score"])
            + 0.20 * (1.0 - features["rgb_consistency_score"])
        )
        item.update(features)
        item["anchor_overlap_max_min_norm"] = max(overlaps) if overlaps else 0.0
        item["anchor_overlap_count_0.20"] = int(anchor_count_02)
        item["anchor_overlap_count_0.50"] = int(sum(1 for val in overlaps if val >= 0.50))
        item["broad_support_score"] = 0.0
        item["multi_frame_support_score"] = 0.0
        item["validation_score"] = float(
            0.25 * features["safe_ratio_score"]
            + 0.15 * features["core_score"]
            + 0.10 * features["checks_score"]
            + 0.20 * features["compactness_score"]
            + 0.15 * features["rgb_consistency_score"]
            + 0.25 * anchor_score
            - risk
        )
        item["validation_risk_score"] = float(
            risk
            + 0.30 * (1.0 - min(anchor_score, 1.0))
            + 0.20 * (1.0 - features["checks_score"])
            + 0.10 * (1.0 - features["core_score"])
        )
        out.append(item)
    return out


def _feature_auc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [1 if bool(row.get("low_tail_label_diagnostic")) else 0 for row in rows]
    feature_names = [
        "seed_score",
        "validation_score",
        "validation_risk_score",
        "anchor_overlap_max_min_norm",
        "anchor_overlap_count_0.20",
        "component_cannot_rate",
        "component_cannot_checks",
        "safe_ratio",
        "mean_xy_dist",
        "mean_color_dist",
        "edge_density",
    ]
    out: list[dict[str, Any]] = []
    for name in feature_names:
        vals = [_f(row, name) for row in rows]
        auc = _rank_auc(vals, labels)
        out.append(
            {
                "feature": name,
                "low_tail_AUC_high_score_positive": auc,
                "low_tail_AUC_low_score_positive": None if auc is None else 1.0 - float(auc),
                "low_tail_count": int(sum(labels)),
                "row_count": int(len(rows)),
            }
        )
    return out


def _cleaner_metrics(
    keep_rows: list[dict[str, Any]],
    base_metric: dict[str, Any],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
    score_name: str,
    cleaner: str,
    dropped: int,
) -> dict[str, Any]:
    metric = _seed_metrics(keep_rows, scenes, gt_by_scene)
    coverage_loss = float(base_metric.get("GT_with_seed_IoU_ge_0.10") or 0.0) - float(
        metric.get("GT_with_seed_IoU_ge_0.10") or 0.0
    )
    gate = {
        "seed_purity_mean_ge_0.90": bool(float(metric.get("purity_mean") or 0.0) >= 0.90),
        "seed_purity_p10_ge_0.75": bool(float(metric.get("purity_p10") or 0.0) >= 0.75),
        "GT_IoU_0.10_ge_0.75": bool(float(metric.get("GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.75),
        "GT_IoU_0.25_ge_0.45": bool(float(metric.get("GT_with_seed_IoU_ge_0.25") or 0.0) >= 0.45),
        "scene0081_IoU_0.10_ge_0.55": bool(
            float(metric.get("scene0081_GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.55
        ),
        "coverage_loss_le_0.05": bool(coverage_loss <= 0.05),
    }
    metric.update(
        {
            "cleaner": cleaner,
            "score_name": score_name,
            "seed_count_before": int(base_metric.get("n") or 0),
            "seed_count_after": int(len(keep_rows)),
            "deleted_seed_count": int(dropped),
            "coverage_loss_after_cleaning": coverage_loss,
            "gate": gate,
            "gate_pass_count": int(sum(1 for ok in gate.values() if ok)),
            "cleaner_gate_pass_without_precision": bool(all(gate.values())),
        }
    )
    return metric


def _evaluate_cleaners(
    base_rows: list[dict[str, Any]],
    trusted_rows: list[dict[str, Any]],
    candidate_metric: dict[str, Any],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
) -> dict[str, Any]:
    augmented = _quality_rows(_augment_base_validation(base_rows, trusted_rows), gt_by_scene)
    feature_auc = _feature_auc_rows(augmented)
    labels = [bool(row.get("low_tail_label_diagnostic")) for row in augmented]

    cleaner_rows: list[dict[str, Any]] = []
    for score_name, reverse in [
        ("validation_risk_score", True),
        ("validation_score", False),
        ("anchor_overlap_max_min_norm", False),
        ("seed_score", False),
    ]:
        for drop_k in [20, 40, 60, 80, 85, 100, 120, 160, 200, 240, 300]:
            ranked = sorted(augmented, key=lambda row: _f(row, score_name), reverse=reverse)
            drop_ids = {id(row) for row in ranked[:drop_k]}
            keep_rows = [row for row in augmented if id(row) not in drop_ids]
            metric = _cleaner_metrics(
                keep_rows,
                candidate_metric,
                scenes,
                gt_by_scene,
                score_name=score_name,
                cleaner="C1_delete_by_score",
                dropped=drop_k,
            )
            dropped_labels = [bool(row.get("low_tail_label_diagnostic")) for row in ranked[:drop_k]]
            metric["low_tail_precision_diagnostic"] = float(sum(dropped_labels) / max(len(dropped_labels), 1))
            metric["low_tail_recall_diagnostic"] = float(sum(dropped_labels) / max(sum(labels), 1))
            metric["cleaner_gate_pass"] = bool(
                metric["cleaner_gate_pass_without_precision"]
                and float(metric["low_tail_precision_diagnostic"]) >= 0.60
            )
            cleaner_rows.append(metric)

    # C5 is the v31 role change: the v30 base route stays available only as coverage factors,
    # while high-trust native anchors initialize slots. This is not a precise low-tail detector,
    # so it is reported separately from the C1 precision gate.
    trusted_metric = _seed_metrics(trusted_rows, scenes, gt_by_scene)
    c5_gate = {
        "trusted_seed_purity_mean_ge_0.90": bool(float(trusted_metric.get("purity_mean") or 0.0) >= 0.90),
        "trusted_seed_purity_p10_ge_0.75": bool(float(trusted_metric.get("purity_p10") or 0.0) >= 0.75),
        "candidate_GT_IoU_0.10_ge_0.75": bool(
            float(candidate_metric.get("GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.75
        ),
        "candidate_GT_IoU_0.25_ge_0.45": bool(
            float(candidate_metric.get("GT_with_seed_IoU_ge_0.25") or 0.0) >= 0.45
        ),
        "scene0081_candidate_IoU_0.10_ge_0.55": bool(
            float(candidate_metric.get("scene0081_GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.55
        ),
    }
    c5 = {
        "cleaner": "C5_role_change_to_coverage_candidate",
        "description": "v30 all_score_top300 seeds are retained as coverage_candidate factors and not used as trusted slot-initializing anchors",
        "seed_count_before": int(candidate_metric.get("n") or 0),
        "trusted_anchor_count_after": int(len(trusted_rows)),
        "role_changed_to_coverage_candidate_count": int(len(base_rows)),
        "deleted_seed_count": 0,
        "downweighted_seed_count": 0,
        "trusted_anchor_metric": trusted_metric,
        "candidate_coverage_metric": candidate_metric,
        "gate": c5_gate,
        "gate_pass": bool(all(c5_gate.values())),
        "low_tail_precision_diagnostic": float(sum(labels) / max(len(labels), 1)),
        "low_tail_recall_diagnostic": 1.0,
        "precision_note": "C5 changes role for the whole v30 base route; it preserves coverage but is not a targeted low-tail detector.",
    }
    return {"validation_rows": augmented, "feature_auc": feature_auc, "cleaner_grid": cleaner_rows, "c5_role_change": c5}


def _oracle_select_slot_sets(rows: list[dict[str, Any]], gt_labels: dict[int, int]) -> tuple[list[set[int]], dict[str, Any]]:
    gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
    best_by_gt: dict[int, tuple[float, dict[str, Any]]] = {}
    for row in rows:
        ids = set(_row_core_ids(row))
        if not ids:
            continue
        counts: Counter[int] = Counter()
        labeled = 0
        for tid in ids:
            gt = int(gt_labels.get(int(tid), 0))
            if gt > 0:
                counts[gt] += 1
                labeled += 1
        for gt, overlap in counts.items():
            iou = float(overlap / max(labeled + int(gt_counts[int(gt)]) - int(overlap), 1))
            old = best_by_gt.get(int(gt))
            if old is None or iou > old[0]:
                best_by_gt[int(gt)] = (iou, row)
    chosen: list[set[int]] = []
    seen_keys: set[tuple[int, ...]] = set()
    for gt in sorted(best_by_gt):
        ids = tuple(sorted(_row_core_ids(best_by_gt[gt][1])))
        if ids and ids not in seen_keys:
            chosen.append(set(ids))
            seen_keys.add(ids)
    return chosen, {"GT_with_factor_count": int(len(best_by_gt)), "selected_slot_count": int(len(chosen))}


def _decomposition_oracle(
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
    trusted_rows: list[dict[str, Any]],
    candidate_anchor_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    broad_rows: list[dict[str, Any]],
    split_repair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    broad_core_rows = [row for row in broad_rows if _row_core_ids(row)]
    variants = {
        "D1_trusted_anchors_only": trusted_rows,
        "D2_trusted_plus_candidate_anchors": trusted_rows + candidate_anchor_rows,
        "D3_trusted_plus_coverage_factors": trusted_rows + coverage_rows,
        "D4_trusted_plus_broad_observations": trusted_rows + broad_core_rows,
        "D5_trusted_plus_all_available_core_factors": trusted_rows + candidate_anchor_rows + coverage_rows + broad_core_rows,
        "D7_trusted_plus_split_repair_factors": trusted_rows + split_repair_rows,
    }
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        gt_labels = gt_by_scene[scene]
        for variant, rows in variants.items():
            scene_items = [row for row in rows if str(row.get("scene")) == scene]
            slots, diag = _oracle_select_slot_sets(scene_items, gt_labels)
            metrics = _eval_slot_sets(slots, gt_labels)
            scene_rows.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "ARI": metrics["ARI"],
                    "purity": metrics["purity"],
                    "completeness": metrics["completeness"],
                    "unknown_tube_ratio": metrics.get("unknown_tube_ratio"),
                    "owned_tube_ratio": metrics.get("owned_tube_ratio"),
                    "overmerge": metrics.get("overmerge"),
                    "oversplit": metrics.get("oversplit"),
                    "trusted_anchor_usage": sum(1 for row in scene_items if row.get("seed_role") == "trusted_anchor"),
                    "candidate_anchor_usage": sum(1 for row in scene_items if row.get("seed_role") == "candidate_anchor"),
                    "coverage_factor_usage": sum(1 for row in scene_items if row.get("seed_role") == "coverage_candidate"),
                    "broad_observation_usage": sum(1 for row in scene_items if row.get("seed_role") == "broad_observation"),
                    "split_repair_factor_usage": sum(1 for row in scene_items if row.get("seed_role") == "split_repair_factor"),
                    "broad_observation_unavailable_reason": (
                        "some broad rows still missing core_tube_ids"
                        if broad_rows and len(broad_core_rows) < len(broad_rows)
                        else None
                    ),
                    **diag,
                }
            )
        gt_slots = [{int(tid) for tid, gt in gt_labels.items() if int(gt) == label} for label in sorted(set(gt_labels.values())) if int(label) > 0]
        metrics = _eval_slot_sets(gt_slots, gt_labels)
        scene_rows.append(
            {
                "scene": scene,
                "variant": "D6_GT_full_oracle_forbidden",
                "ARI": metrics["ARI"],
                "purity": metrics["purity"],
                "completeness": metrics["completeness"],
                "unknown_tube_ratio": metrics.get("unknown_tube_ratio"),
                "owned_tube_ratio": metrics.get("owned_tube_ratio"),
                "overmerge": metrics.get("overmerge"),
                "oversplit": metrics.get("oversplit"),
                "trusted_anchor_usage": None,
                "candidate_anchor_usage": None,
                "coverage_factor_usage": None,
                "broad_observation_usage": None,
                "GT_with_factor_count": None,
                "selected_slot_count": len(gt_slots),
            }
        )
    aggregate = _aggregate_variant(scene_rows)
    d5 = next((row for row in aggregate if row["variant"] == "D5_trusted_plus_all_available_core_factors"), {})
    d7 = next((row for row in aggregate if row["variant"] == "D7_trusted_plus_split_repair_factors"), {})
    d5_gate = {
        "D5_ARI_ge_0.40": bool(float(d5.get("ARI") or 0.0) >= 0.40),
        "D5_purity_ge_0.85": bool(float(d5.get("purity") or 0.0) >= 0.85),
        "D5_completeness_ge_0.50": bool(float(d5.get("completeness") or 0.0) >= 0.50),
        "D5_scene0081_ARI_ge_0.20": bool(float(d5.get("scene0081_ARI") or 0.0) >= 0.20),
    }
    d7_gate = {
        "D7_ARI_ge_0.40": bool(float(d7.get("ARI") or 0.0) >= 0.40),
        "D7_purity_ge_0.85": bool(float(d7.get("purity") or 0.0) >= 0.85),
        "D7_completeness_ge_0.50": bool(float(d7.get("completeness") or 0.0) >= 0.50),
        "D7_scene0081_ARI_ge_0.20": bool(float(d7.get("scene0081_ARI") or 0.0) >= 0.20),
    }
    return {
        "scene_rows": scene_rows,
        "aggregate_rows": aggregate,
        "gate": {
            **d5_gate,
            "phaseD_gate_pass": bool(all(d5_gate.values())),
            "D7_split_repair_gate": {**d7_gate, "phaseD_split_repair_gate_pass": bool(all(d7_gate.values()))},
            "phaseD_any_oracle_gate_pass": bool(all(d5_gate.values()) or all(d7_gate.values())),
        },
        "broad_rows_count": int(len(broad_rows)),
        "broad_rows_with_core_ids": int(len(broad_core_rows)),
        "broad_rows_missing_core_ids": int(len(broad_rows) - len(broad_core_rows)),
        "split_repair_factor_count": int(len(split_repair_rows)),
    }


def _clone_with_core_ids(row: dict[str, Any], ids: tuple[int, ...]) -> dict[str, Any]:
    item = dict(row)
    _set_core_ids(item, ids)
    return item


def _make_shuffled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shuffled: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row.get("scene"))].append(row)
    for scene, items in by_scene.items():
        ordered = sorted(items, key=lambda row: str(row.get("proposal_id")))
        if len(ordered) < 2:
            shuffled.extend(dict(row) for row in ordered)
            continue
        shift = max(1, len(ordered) // 3)
        core_sets = [_row_core_ids(row) for row in ordered]
        for idx, row in enumerate(ordered):
            item = _clone_with_core_ids(row, core_sets[(idx + shift) % len(core_sets)])
            item["proposal_id"] = f"{row.get('proposal_id')}_v31shuffle"
            item["control_source_proposal_id"] = row.get("proposal_id")
            item["control_kind"] = "shuffled_d4rt"
            shuffled.append(item)
    return shuffled


def _aggregate_solver_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    solvers = sorted({str(row.get("solver")) for row in rows})
    numeric_keys = [
        "slot_count",
        "active_slot_count",
        "owned_tube_ratio",
        "unknown_tube_ratio",
        "coverage_factor_explained_ratio",
        "broad_observation_explained_ratio",
        "cannot_link_violation_count",
        "boundary_violation_rate",
        "appearance_consistency",
        "motion_consistency",
        "solver_runtime_sec",
        "solver_iterations",
        "ARI",
        "purity",
        "completeness",
        "overmerge",
        "oversplit",
    ]
    for solver in solvers:
        items = [row for row in rows if str(row.get("solver")) == solver]
        row: dict[str, Any] = {
            "scene": "ALL",
            "solver": solver,
            "scene_count": int(len(items)),
            "control_kind": items[0].get("control_kind") if items else "",
        }
        for key in numeric_keys:
            vals = [float(item[key]) for item in items if item.get(key) not in (None, "")]
            row[key] = _mean(vals)
        row["scene0081_ARI"] = next((item.get("ARI") for item in items if item.get("scene") == "scene0081_01"), None)
        out.append(row)
    return out


def _clip_norm(value: Any, scale: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    if not np.isfinite(out):
        out = 0.0
    return float(min(max(out / max(float(scale), 1e-6), 0.0), 3.0))


def _v31_type_prior_score(row: dict[str, Any]) -> float:
    """Non-GT score that deliberately gives temporal/consensus factors a chance to own coverage."""
    proposal_type = str(row.get("proposal_type") or "")
    type_bonus = -0.20
    for prefix, bonus in [
        ("R10_temporal_tube_overlap_visible_negative_pruned_t50", 1.00),
        ("R10_temporal_tube_overlap_visible_negative_pruned_t70", 0.90),
        ("R10_temporal_tube_overlap_visible_negative_pruned_t35", 0.65),
        ("R6_", 0.35),
        ("R7_", 0.25),
        ("R1_", 0.05),
        ("R3_", 0.05),
        ("R4_", 0.05),
        ("R5_", 0.05),
        ("R2_", -0.15),
    ]:
        if proposal_type.startswith(prefix):
            type_bonus = float(bonus)
            break
    core_count = max(len(_row_core_ids(row)), 1)
    size_score = math.log1p(core_count) / math.log(256.0)
    medium_score = 1.0 / (1.0 + abs(math.log1p(core_count) - math.log(24.0)))
    risk = (
        0.45 * _clip_norm(row.get("boundary_risk"), 0.25)
        + 0.35 * _clip_norm(row.get("same_frame_cannot_link_rate"), 80.0)
        + 0.20 * _clip_norm(row.get("visible_outside_negative_rate"), 70.0)
    )
    return float(
        type_bonus
        + 0.80 * size_score
        + 0.20 * medium_score
        - 0.20 * _f(row, "eroded_interior_ratio")
        + 0.05 * _f(row, "visibility_mean")
        - 0.18 * risk
        - 0.04 * _clip_norm(row.get("tube_canonical_compactness"), 2.0)
    )


def _v31_type_prior_risk(row: dict[str, Any]) -> float:
    return float(
        0.45 * _clip_norm(row.get("boundary_risk"), 0.25)
        + 0.35 * _clip_norm(row.get("same_frame_cannot_link_rate"), 80.0)
        + 0.20 * _clip_norm(row.get("visible_outside_negative_rate"), 70.0)
    )


def _select_v31_type_prior_solver(
    candidates: list[dict[str, Any]],
    *,
    min_new_tubes: int,
    max_overlap_ratio: float,
    max_slots: int,
    min_score: float = -1e9,
    max_temporal_fraction: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    owned: set[int] = set()
    attempted = 0
    accepted = 0
    temporal_count = 0
    ranked = sorted(
        candidates,
        key=lambda row: (_v31_type_prior_score(row), len(_row_core_ids(row))),
        reverse=True,
    )
    for row in ranked:
        attempted += 1
        score = _v31_type_prior_score(row)
        if score < float(min_score):
            continue
        is_temporal = str(row.get("proposal_type") or "").startswith(TEMPORAL_PREFIXES)
        if is_temporal and (temporal_count + 1) > float(max_temporal_fraction) * max(len(selected) + 1, 1):
            continue
        core = set(_row_core_ids(row))
        if len(core) < int(min_new_tubes):
            continue
        new = core - owned
        if len(new) < int(min_new_tubes):
            continue
        overlap_ratio = float((len(core) - len(new)) / max(len(core), 1))
        if overlap_ratio > float(max_overlap_ratio):
            continue
        selected.append(row)
        owned.update(core)
        accepted += 1
        if is_temporal:
            temporal_count += 1
        if len(selected) >= int(max_slots):
            break
    return selected, {
        "num_moves_attempted": attempted,
        "num_moves_accepted": accepted,
        "temporal_selected_count": int(temporal_count),
    }


def _run_phaseE_solver(
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
    split_repair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in split_repair_rows]
    _annotate_v29_features(rows, set(), set())
    shuffled_rows = _make_shuffled_rows(rows)
    _annotate_v29_features(shuffled_rows, set(), set())
    configs = [
        ("E1_greedy_hard_set_packing", 3, 0.10, 0.03, 0.15, 0.20, 0.12),
        ("E2_local_search_medium_replace", 3, 0.12, -0.02, 0.30, 0.35, 0.14),
        ("E3_lagrangian_coverage_fallback", 2, 0.22, -0.08, 0.05, 0.18, 0.28),
        ("E4_purity_repair_no_broad", 2, 0.06, 0.00, 0.45, 0.25, 0.08),
        ("E5_unknown_low_overlap", 5, 0.06, 0.05, 0.30, 0.20, 0.10),
        ("E6_unknown_high_coverage", 2, 0.35, -0.12, 0.05, 0.15, 0.40),
        ("E7_split_local_balanced", 3, 0.18, -0.04, 0.18, 0.25, 0.22),
    ]
    type_prior_configs = [
        ("E8_temporal_type_prior_high_coverage", 3, 0.82, 56, -1e9, 1.00),
        ("E9_temporal_type_prior_temporal_cap", 3, 0.82, 56, -1e9, 0.75),
    ]
    controls: list[tuple[str, list[dict[str, Any]]]] = [
        ("real", rows),
        ("shuffled_d4rt", shuffled_rows),
        ("no_temporal", [row for row in rows if not str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]),
        ("mask_only", [row for row in rows if str(row.get("proposal_type")) in MASK_ONLY_TYPES]),
    ]
    scene_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for control_kind, control_rows in controls:
        for solver_name, min_new, overlap, min_q, broad_penalty, medium_bonus, coverage_weight in configs:
            solver = f"{solver_name}_{control_kind}" if control_kind != "real" else solver_name
            for scene in scenes:
                candidates = [row for row in control_rows if str(row.get("scene")) == scene]
                t0 = datetime.now(timezone.utc)
                selected, stats = _select_v29_greedy_solver(
                    candidates,
                    set(),
                    min_new_tubes=int(min_new),
                    max_overlap_ratio=float(overlap),
                    min_quality=float(min_q),
                    broad_penalty=float(broad_penalty),
                    medium_bonus=float(medium_bonus),
                    coverage_weight=float(coverage_weight),
                )
                runtime = (datetime.now(timezone.utc) - t0).total_seconds()
                metrics = _eval_v29_selected(selected, gt_by_scene[scene])
                selected_ids = set().union(*(set(_row_core_ids(row)) for row in selected)) if selected else set()
                split_ids = set().union(*(set(_row_core_ids(row)) for row in candidates)) if candidates else set()
                coverage_factor_explained_ratio = float(len(selected_ids & split_ids) / max(len(split_ids), 1))
                temporal_selected = [row for row in selected if str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]
                row = {
                    "scene": scene,
                    "solver": solver,
                    "control_kind": control_kind,
                    "slot_count": int(len(candidates)),
                    "active_slot_count": int(len(selected)),
                    "owned_tube_ratio": metrics["owned_tube_ratio"],
                    "unknown_tube_ratio": metrics["unknown_tube_ratio"],
                    "coverage_factor_explained_ratio": coverage_factor_explained_ratio,
                    "broad_observation_explained_ratio": float(len(temporal_selected) / max(len(selected), 1)),
                    "cannot_link_violation_count": 0,
                    "boundary_violation_rate": _mean([_f(item, "boundary_risk") for item in selected]) or 0.0,
                    "appearance_consistency": 1.0 - min((_mean([_f(item, "appearance_variance") for item in selected]) or 0.0) / 0.12, 1.0),
                    "motion_consistency": _mean([_f(item, "mask_temporal_repeat_score") for item in selected]) or 0.0,
                    "energy_anchor": None,
                    "energy_coverage": 1.0 - metrics["owned_tube_ratio"],
                    "energy_cannot": 0,
                    "energy_boundary": _mean([_f(item, "boundary_risk") for item in selected]) or 0.0,
                    "energy_unknown": metrics["unknown_tube_ratio"],
                    "energy_size": _mean([len(_row_core_ids(item)) for item in selected]) or 0.0,
                    "solver_iterations": int(stats["num_moves_attempted"]),
                    "solver_runtime_sec": runtime,
                    "ARI": metrics["ARI"],
                    "purity": metrics["purity"],
                    "completeness": metrics["completeness"],
                    "overmerge": metrics["overmerge"],
                    "oversplit": metrics["oversplit"],
                    "num_moves_attempted": int(stats["num_moves_attempted"]),
                    "num_moves_accepted": int(stats["num_moves_accepted"]),
                }
                scene_rows.append(row)
                for rank, item in enumerate(selected):
                    selected_rows.append(
                        {
                            "scene": scene,
                            "solver": solver,
                            "control_kind": control_kind,
                            "rank": int(rank),
                            "proposal_id": item.get("proposal_id"),
                            "proposal_type": item.get("proposal_type"),
                            "core_tube_count": len(_row_core_ids(item)),
                            "core_tube_ids": ";".join(str(tid) for tid in _row_core_ids(item)),
                            "uses_gt_for_prediction": False,
                        }
                    )
        for solver_name, min_new, overlap, max_slots, min_score, max_temporal_fraction in type_prior_configs:
            solver = f"{solver_name}_{control_kind}" if control_kind != "real" else solver_name
            for scene in scenes:
                candidates = [row for row in control_rows if str(row.get("scene")) == scene]
                t0 = datetime.now(timezone.utc)
                selected, stats = _select_v31_type_prior_solver(
                    candidates,
                    min_new_tubes=int(min_new),
                    max_overlap_ratio=float(overlap),
                    max_slots=int(max_slots),
                    min_score=float(min_score),
                    max_temporal_fraction=float(max_temporal_fraction),
                )
                runtime = (datetime.now(timezone.utc) - t0).total_seconds()
                metrics = _eval_v29_selected(selected, gt_by_scene[scene])
                selected_ids = set().union(*(set(_row_core_ids(row)) for row in selected)) if selected else set()
                split_ids = set().union(*(set(_row_core_ids(row)) for row in candidates)) if candidates else set()
                coverage_factor_explained_ratio = float(len(selected_ids & split_ids) / max(len(split_ids), 1))
                temporal_selected = [row for row in selected if str(row.get("proposal_type", "")).startswith(TEMPORAL_PREFIXES)]
                row = {
                    "scene": scene,
                    "solver": solver,
                    "control_kind": control_kind,
                    "slot_count": int(len(candidates)),
                    "active_slot_count": int(len(selected)),
                    "owned_tube_ratio": metrics["owned_tube_ratio"],
                    "unknown_tube_ratio": metrics["unknown_tube_ratio"],
                    "coverage_factor_explained_ratio": coverage_factor_explained_ratio,
                    "broad_observation_explained_ratio": float(len(temporal_selected) / max(len(selected), 1)),
                    "cannot_link_violation_count": 0,
                    "boundary_violation_rate": _mean([_f(item, "boundary_risk") for item in selected]) or 0.0,
                    "appearance_consistency": 1.0 - min((_mean([_f(item, "appearance_variance") for item in selected]) or 0.0) / 0.12, 1.0),
                    "motion_consistency": _mean([_f(item, "mask_temporal_repeat_score") for item in selected]) or 0.0,
                    "energy_anchor": None,
                    "energy_coverage": 1.0 - metrics["owned_tube_ratio"],
                    "energy_cannot": _mean([_v31_type_prior_risk(item) for item in selected]) or 0.0,
                    "energy_boundary": _mean([_f(item, "boundary_risk") for item in selected]) or 0.0,
                    "energy_unknown": metrics["unknown_tube_ratio"],
                    "energy_size": _mean([len(_row_core_ids(item)) for item in selected]) or 0.0,
                    "solver_iterations": int(stats["num_moves_attempted"]),
                    "solver_runtime_sec": runtime,
                    "ARI": metrics["ARI"],
                    "purity": metrics["purity"],
                    "completeness": metrics["completeness"],
                    "overmerge": metrics["overmerge"],
                    "oversplit": metrics["oversplit"],
                    "num_moves_attempted": int(stats["num_moves_attempted"]),
                    "num_moves_accepted": int(stats["num_moves_accepted"]),
                }
                scene_rows.append(row)
                for rank, item in enumerate(selected):
                    selected_rows.append(
                        {
                            "scene": scene,
                            "solver": solver,
                            "control_kind": control_kind,
                            "rank": int(rank),
                            "proposal_id": item.get("proposal_id"),
                            "proposal_type": item.get("proposal_type"),
                            "core_tube_count": len(_row_core_ids(item)),
                            "core_tube_ids": ";".join(str(tid) for tid in _row_core_ids(item)),
                            "uses_gt_for_prediction": False,
                        }
                    )
    aggregate = _aggregate_solver_rows(scene_rows)
    real_rows = [row for row in aggregate if row.get("control_kind") == "real"]
    no_temporal_rows = [row for row in aggregate if row.get("control_kind") == "no_temporal"]
    mask_rows = [row for row in aggregate if row.get("control_kind") == "mask_only"]
    shuffled_rows_agg = [row for row in aggregate if row.get("control_kind") == "shuffled_d4rt"]
    best_real = max(real_rows, key=lambda row: (float(row.get("ARI") or 0.0), float(row.get("purity") or 0.0))) if real_rows else {}
    best_no_temporal = max(no_temporal_rows, key=lambda row: float(row.get("ARI") or 0.0)) if no_temporal_rows else {}
    best_mask = max(mask_rows, key=lambda row: float(row.get("ARI") or 0.0)) if mask_rows else {}
    best_shuffled = max(shuffled_rows_agg, key=lambda row: float(row.get("ARI") or 0.0)) if shuffled_rows_agg else {}
    real_ari = float(best_real.get("ARI") or 0.0)
    gate = {
        "E_real_ARI_ge_0.35": bool(real_ari >= 0.35),
        "E_real_purity_ge_0.85": bool(float(best_real.get("purity") or 0.0) >= 0.85),
        "E_real_completeness_ge_0.50": bool(float(best_real.get("completeness") or 0.0) >= 0.50),
        "E_scene0081_ARI_ge_0.20": bool(float(best_real.get("scene0081_ARI") or 0.0) >= 0.20),
        "E_real_vs_shuffled_margin_ge_0.20": bool(real_ari - float(best_shuffled.get("ARI") or 0.0) >= 0.20),
        "E_real_vs_no_temporal_margin_ge_0.05": bool(real_ari - float(best_no_temporal.get("ARI") or 0.0) >= 0.05),
        "E_real_vs_mask_only_margin_ge_0.05": bool(real_ari - float(best_mask.get("ARI") or 0.0) >= 0.05),
    }
    return {
        "scene_rows": scene_rows,
        "aggregate_rows": aggregate,
        "selected_rows": selected_rows,
        "gate": {
            **gate,
            "phaseE_gate_pass": bool(all(gate.values())),
            "best_real_solver": best_real,
            "best_shuffled_solver": best_shuffled,
            "best_no_temporal_solver": best_no_temporal,
            "best_mask_only_solver": best_mask,
        },
    }


def _plot_hist(path: Path, values_good: list[float], values_bad: list[float], title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.0))
    plt.hist(values_good, bins=30, alpha=0.65, label="purity>=0.75")
    plt.hist(values_bad, bins=30, alpha=0.65, label="purity<0.75")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("seed count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = Path(args.audit_root)
    out_root = Path(args.out_root)
    scenes = _read_split(Path(args.split))
    gt_by_scene = {
        scene: _load_gt_labels(
            Path(args.cache_root),
            scene,
            int(args.max_tubes_per_window),
            int(args.image_width),
            int(args.image_height),
        )
        for scene in scenes
    }

    component_rows = _read_component_feature_rows(Path(args.component_feature_csv))
    base_rows = _select_topk_per_scene(component_rows, scenes, "seed_score", 300, "v31_base_all_score_top300")
    for row in base_rows:
        row["source_profile"] = "v30_coverage_proxy_base"
        row["source_type"] = "component_cannot_link_base_route"
        row["seed_role"] = "coverage_candidate"

    native_profile_path = audit_root / "v30_profiles/native_rgb_boundary_repair/v30_seed_slots/seed_slot_rows.csv"
    native_rows = _load_profile_seed_rows(native_profile_path, "native_rgb_boundary_repair")
    trusted_rows = _select_topk_per_scene(native_rows, scenes, "native_trust_score", 240, "v31_trusted_native_top240")
    for row in trusted_rows:
        row["seed_role"] = "trusted_anchor"
        row["validation_score"] = row["native_trust_score"]

    candidate_anchor_rows = _select_topk_per_scene(native_rows, scenes, "seed_score", 300, "v31_candidate_native_top300")
    for row in candidate_anchor_rows:
        row["seed_role"] = "candidate_anchor"
        row["validation_score"] = row.get("native_trust_score")

    proposal_core_map, proposal_core_stats = _load_proposal_core_map(Path(args.proposal_row_csv))
    split_repair_rows, split_repair_stats = _load_split_repair_factor_rows(Path(args.proposal_row_csv))
    broad_rows = _read_broad_rows(
        audit_root / "v30_profiles/native_rgb_boundary_repair/v30_broad_observation_diagnostic/broad_proposal_rows.csv",
        proposal_core_map,
    )
    for row in broad_rows:
        row["seed_role"] = "broad_observation"
        row["source_profile"] = "native_rgb_boundary_repair"
        row["source_type"] = "v30_broad_observation_reconstructed_core_from_v28"

    trusted_metric = _seed_metrics(trusted_rows, scenes, gt_by_scene)
    candidate_anchor_metric = _seed_metrics(candidate_anchor_rows, scenes, gt_by_scene)
    candidate_coverage_metric = _seed_metrics(base_rows, scenes, gt_by_scene)
    trusted_gate = _trusted_anchor_gate(trusted_metric)
    candidate_gate = _candidate_coverage_gate(candidate_coverage_metric)

    role_rows: list[dict[str, Any]] = []
    for row in trusted_rows + candidate_anchor_rows + base_rows:
        q = _row_quality(row, gt_by_scene[str(row["scene"])], Counter(int(gt) for gt in gt_by_scene[str(row["scene"])].values() if int(gt) > 0))
        role_rows.append(
            {
                "seed_id": row.get("seed_id"),
                "seed_role": row.get("seed_role"),
                "source_profile": row.get("source_profile"),
                "source_type": row.get("source_type"),
                "scene": row.get("scene"),
                "proposal_id": row.get("proposal_id"),
                "core_tube_ids": ";".join(str(tid) for tid in _row_core_ids(row)),
                "core_tube_count": len(_row_core_ids(row)),
                "supporting_mask_ids": row.get("mask") or row.get("mask_id") or "",
                "frame_ids": row.get("frame") or row.get("frame_id") or "",
                "submap_id": "",
                "rgb_embedding_summary": row.get("native_rgb_pair_cos_p10") or row.get("mean_color_dist") or "",
                "boundary_score": row.get("native_boundary_distance_p10") or row.get("safe_ratio") or "",
                "cannot_link_score": row.get("component_cannot_rate") or "",
                "multi_frame_support_score": row.get("native_frame_mask_support_ratio") or 0.0,
                "broad_support_score": row.get("broad_support_score") or 0.0,
                "uniqueness_score": row.get("anchor_overlap_max_min_norm") or "",
                "redundancy_score": row.get("anchor_overlap_count_0.20") or "",
                "validation_score": row.get("validation_score") or row.get("seed_score") or "",
                "purity_diagnostic": q["purity"],
                "best_iou_diagnostic": q["best_iou"],
                "dominant_gt_diagnostic": q["best_gt"],
            }
        )

    for idx, row in enumerate(broad_rows):
        core_ids = _row_core_ids(row)
        q = None
        if core_ids:
            scene = str(row.get("scene"))
            q = _row_quality(
                row,
                gt_by_scene[scene],
                Counter(int(gt) for gt in gt_by_scene[scene].values() if int(gt) > 0),
            )
        role_rows.append(
            {
                "seed_id": f"broad_observation_{idx:05d}",
                "seed_role": "broad_observation",
                "source_profile": "native_rgb_boundary_repair",
                "source_type": row.get("source_type") or "v30_broad_observation_reconstructed_core_from_v28",
                "scene": row.get("scene"),
                "proposal_id": row.get("proposal_id"),
                "core_tube_ids": ";".join(str(tid) for tid in core_ids),
                "core_tube_count": len(core_ids) if core_ids else row.get("core_tube_count"),
                "supporting_mask_ids": "",
                "frame_ids": "",
                "submap_id": "",
                "rgb_embedding_summary": "",
                "boundary_score": "",
                "cannot_link_score": "",
                "multi_frame_support_score": "",
                "broad_support_score": row.get("overlap_seed_count"),
                "uniqueness_score": "",
                "redundancy_score": row.get("overlap_child_count"),
                "validation_score": "",
                "purity_diagnostic": q["purity"] if q else row.get("proposal_purity"),
                "best_iou_diagnostic": q["best_iou"] if q else row.get("proposal_best_IoU"),
                "dominant_gt_diagnostic": q["best_gt"] if q else "",
                "core_reconstruction_source": row.get("core_reconstruction_source"),
            }
        )

    seed_roles_dir = out_root / "v31_seed_roles"
    _write_csv(seed_roles_dir / "seed_role_rows.csv", role_rows)
    role_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trusted_anchor_count": len(trusted_rows),
        "candidate_anchor_count": len(candidate_anchor_rows),
        "coverage_candidate_count": len(base_rows),
        "broad_observation_count": len(broad_rows),
        "broad_observation_with_core_ids": int(sum(1 for row in broad_rows if _row_core_ids(row))),
        "broad_core_reconstruction": {
            **proposal_core_stats,
            "proposal_core_source": str(Path(args.proposal_row_csv)),
            "matched_broad_rows": int(sum(1 for row in broad_rows if _row_core_ids(row))),
            "missing_broad_rows": int(sum(1 for row in broad_rows if not _row_core_ids(row))),
        },
        "split_repair_factor_pool": {
            **split_repair_stats,
            "proposal_core_source": str(Path(args.proposal_row_csv)),
        },
        "reject_count": 0,
        "trusted_anchor_metric": trusted_metric,
        "candidate_anchor_metric": candidate_anchor_metric,
        "candidate_coverage_metric": candidate_coverage_metric,
        "trusted_anchor_gate": {**trusted_gate, "trusted_anchor_gate_pass": bool(all(trusted_gate.values()))},
        "candidate_coverage_gate": {**candidate_gate, "candidate_coverage_gate_pass": bool(all(candidate_gate.values()))},
        "phaseB_strict_gate_pass": bool(all(trusted_gate.values()) and all(candidate_gate.values())),
        "phaseB_fallback_allowed_by_plan": bool(
            float(trusted_metric.get("purity_mean") or 0.0) >= 0.93
            and float(trusted_metric.get("purity_p10") or 0.0) >= 0.80
            and all(candidate_gate.values())
        ),
        "fallback_reason": "trusted anchors are high-purity but do not satisfy strict anchor coverage; plan 5.6 allows moving to Phase C low-tail cleaner without using the same set for coverage",
    }
    _write_json(seed_roles_dir / "role_summary.json", role_summary)
    _write_json(
        seed_roles_dir / "manifest.json",
        {
            **METHOD_MANIFEST_BASE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "v31_phaseB_seed_roles",
            "artifact_files": ["seed_role_rows.csv", "role_summary.json"],
            "role_selection_uses_gt": False,
            "diagnostic_gt_fields": ["purity_diagnostic", "best_iou_diagnostic", "role_summary gate metrics"],
            "broad_core_reconstruction_source": str(Path(args.proposal_row_csv)),
            "broad_observation_with_core_ids": int(sum(1 for row in broad_rows if _row_core_ids(row))),
            "split_repair_factor_count": int(len(split_repair_rows)),
            "phaseB_strict_gate_pass": role_summary["phaseB_strict_gate_pass"],
            "phaseB_fallback_allowed_by_plan": role_summary["phaseB_fallback_allowed_by_plan"],
        },
    )

    cleaner = _evaluate_cleaners(base_rows, trusted_rows, candidate_coverage_metric, scenes, gt_by_scene)
    validation_dir = out_root / "v31_lowtail_validation"
    _write_csv(validation_dir / "validation_feature_rows.csv", cleaner["validation_rows"])
    _write_csv(validation_dir / "feature_auc.csv", cleaner["feature_auc"])
    good = [
        _f(row, "validation_risk_score")
        for row in cleaner["validation_rows"]
        if not bool(row.get("low_tail_label_diagnostic"))
    ]
    bad = [
        _f(row, "validation_risk_score")
        for row in cleaner["validation_rows"]
        if bool(row.get("low_tail_label_diagnostic"))
    ]
    _plot_hist(
        validation_dir / "figures/validation_risk_hist.png",
        good,
        bad,
        "v31 validation risk by diagnostic low-tail label",
        "validation_risk_score",
    )
    _write_json(
        validation_dir / "figures/figure_manifest.json",
        {"figures": ["validation_risk_hist.png"], "uses_gt_for_visualization": True, "uses_gt_for_prediction": False},
    )
    _write_json(
        validation_dir / "manifest.json",
        {
            **METHOD_MANIFEST_BASE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "v31_phaseC_lowtail_validation",
            "artifact_files": ["validation_feature_rows.csv", "feature_auc.csv", "figures/validation_risk_hist.png"],
            "validation_uses_gt_for_scoring": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    )

    cleaner_dir = out_root / "v31_lowtail_cleaner"
    cleaner_grid = sorted(
        cleaner["cleaner_grid"],
        key=lambda row: (
            bool(row.get("cleaner_gate_pass")),
            float(row.get("purity_p10") or 0.0),
            float(row.get("low_tail_precision_diagnostic") or 0.0),
            float(row.get("GT_with_seed_IoU_ge_0.10") or 0.0),
        ),
        reverse=True,
    )
    _write_json(
        cleaner_dir / "cleaner_grid.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_metric": candidate_coverage_metric,
            "top_cleaner_rows": cleaner_grid[:100],
            "c5_role_change": cleaner["c5_role_change"],
            "c1_any_gate_pass": any(bool(row.get("cleaner_gate_pass")) for row in cleaner_grid),
            "c5_role_gate_pass": cleaner["c5_role_change"]["gate_pass"],
            "phaseC_targeted_lowtail_cleaner_pass": any(bool(row.get("cleaner_gate_pass")) for row in cleaner_grid),
            "phaseC_role_redefinition_pass": cleaner["c5_role_change"]["gate_pass"],
        },
    )
    _write_json(
        cleaner_dir / "manifest.json",
        {
            **METHOD_MANIFEST_BASE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "v31_phaseC_lowtail_cleaner",
            "artifact_files": ["cleaner_grid.json"],
            "cleaner_uses_gt_for_selection": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    )

    decomp = _decomposition_oracle(
        scenes,
        gt_by_scene,
        trusted_rows,
        candidate_anchor_rows,
        base_rows,
        broad_rows,
        split_repair_rows,
    )
    decomp_dir = out_root / "v31_decomposition_oracle"
    _write_csv(decomp_dir / "decomposition_summary.csv", decomp["scene_rows"] + decomp["aggregate_rows"])
    split_summary_rows: list[dict[str, Any]] = []
    by_scene_type: Counter[tuple[str, str]] = Counter()
    for row in split_repair_rows:
        by_scene_type[(str(row.get("scene")), str(row.get("proposal_type")))] += 1
    for (scene, proposal_type), count in sorted(by_scene_type.items()):
        split_summary_rows.append({"scene": scene, "proposal_type": proposal_type, "split_repair_factor_count": int(count)})
    _write_csv(decomp_dir / "split_repair_factor_summary.csv", split_summary_rows)
    _write_json(
        decomp_dir / "manifest.json",
        {
            **METHOD_MANIFEST_BASE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "v31_phaseD_decomposition_oracle",
            "artifact_files": ["decomposition_summary.csv", "split_repair_factor_summary.csv"],
            "decomposition_uses_gt_for_assignment": True,
            "decomposition_oracle_is_method_result": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "decomposition_gate": decomp["gate"],
            "broad_rows_count": decomp["broad_rows_count"],
            "broad_rows_with_core_ids": decomp["broad_rows_with_core_ids"],
            "broad_rows_missing_core_ids": decomp["broad_rows_missing_core_ids"],
            "broad_core_reconstruction_source": str(Path(args.proposal_row_csv)),
            "split_repair_factor_count": decomp["split_repair_factor_count"],
            "split_repair_factor_source": str(Path(args.proposal_row_csv)),
            "split_repair_filter": split_repair_stats.get("split_repair_filter"),
        },
    )

    phaseE_can_run = bool(decomp["gate"].get("phaseD_any_oracle_gate_pass"))
    phaseE_gate: dict[str, Any]
    slot_dir = out_root / "v31_slot_ownership"
    if phaseE_can_run:
        phaseE = _run_phaseE_solver(scenes, gt_by_scene, split_repair_rows)
        _write_csv(slot_dir / "solver_summary.csv", phaseE["scene_rows"] + phaseE["aggregate_rows"])
        _write_csv(slot_dir / "selected_slot_rows.csv", phaseE["selected_rows"])
        phaseE_gate = phaseE["gate"]
        _write_json(
            slot_dir / "manifest.json",
            {
                **METHOD_MANIFEST_BASE,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phase": "v31_phaseE_slot_ownership",
                "artifact_files": ["solver_summary.csv", "selected_slot_rows.csv"],
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
                "slot_ownership_uses_gt_for_selection": False,
                "phaseE_gate": phaseE_gate,
                "phaseE_gate_pass": phaseE_gate["phaseE_gate_pass"],
                "controls": ["real", "shuffled_d4rt", "no_temporal", "mask_only"],
                "split_repair_factor_source": str(Path(args.proposal_row_csv)),
            },
        )
    else:
        phaseE_gate = {"phaseE_gate_pass": False, "not_run_reason": "Phase D decomposition oracle did not pass"}
        _write_json(
            slot_dir / "manifest.json",
            {
                **METHOD_MANIFEST_BASE,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phase": "v31_phaseE_slot_ownership_not_run",
                "status": "not_run",
                "not_run_reason": phaseE_gate["not_run_reason"],
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
            },
        )

    phaseF_reason = (
        "Phase E slot ownership gate passed"
        if phaseE_gate.get("phaseE_gate_pass")
        else "Phase E slot ownership gate failed; memory/densification/eval export not run"
    )
    for dirname, phase in [
        ("v31_memory", "v31_phaseF_memory"),
        ("v31_densification", "v31_phaseF_densification"),
        ("v31_eval_export", "v31_phaseF_eval_export"),
    ]:
        out = out_root / dirname
        out.mkdir(parents=True, exist_ok=True)
        if phaseE_gate.get("phaseE_gate_pass"):
            status_payload = {
                **METHOD_MANIFEST_BASE,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phase": f"{phase}_pending",
                "status": "pending",
                "not_run_reason": "Phase F implementation not executed in this v31 run",
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        else:
            status_payload = {
                **METHOD_MANIFEST_BASE,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phase": f"{phase}_not_run",
                "status": "not_run",
                "not_run_reason": phaseF_reason,
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        _write_json(out / "manifest.json", status_payload)

    gate_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phaseA_gate_pass_required": True,
        "phaseB_strict_gate_pass": role_summary["phaseB_strict_gate_pass"],
        "phaseB_fallback_allowed_by_plan": role_summary["phaseB_fallback_allowed_by_plan"],
        "phaseC_targeted_lowtail_cleaner_pass": any(bool(row.get("cleaner_gate_pass")) for row in cleaner_grid),
        "phaseC_role_redefinition_pass": cleaner["c5_role_change"]["gate_pass"],
        "phaseD_decomposition_gate": decomp["gate"],
        "phaseE_ownership_gate": phaseE_gate,
        "can_run_phaseE_non_gt_solver": bool(
            (
                any(bool(row.get("cleaner_gate_pass")) for row in cleaner_grid)
                or decomp["gate"].get("phaseD_gate_pass")
                or decomp["gate"].get("phaseD_any_oracle_gate_pass")
            )
        ),
        "can_run_phaseF_memory_ap": bool(phaseE_gate.get("phaseE_gate_pass")),
        "stop_rule": None,
    }
    if not role_summary["phaseB_fallback_allowed_by_plan"]:
        gate_summary["stop_rule"] = "No-Go B seed-anchor blocker"
    elif not any(bool(row.get("cleaner_gate_pass")) for row in cleaner_grid) and not decomp["gate"].get("phaseD_any_oracle_gate_pass"):
        gate_summary["stop_rule"] = "No-Go C/D low-tail targeted cleaner and decomposition blocker"
    elif not phaseE_gate.get("phaseE_gate_pass"):
        gate_summary["stop_rule"] = "No-Go E solver blocker"
    elif not any(bool(row.get("cleaner_gate_pass")) for row in cleaner_grid):
        gate_summary["stop_rule"] = "Phase C targeted low-tail cleaner blocker; Phase D split-repair oracle can be used for next solver gate"
    _write_json(out_root / "v31_phaseBCD_gate_summary.json", gate_summary)

    return {
        "phaseB": role_summary,
        "phaseC": {
            "feature_auc": cleaner["feature_auc"],
            "top_cleaner_rows": cleaner_grid[:10],
            "c5_role_change": cleaner["c5_role_change"],
        },
        "phaseD": decomp["gate"],
        "phaseE": phaseE_gate,
        "gate_summary": gate_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stream4D v31 seed-anchor and low-tail validation diagnostics.")
    parser.add_argument("--audit-root", default="outputs/audit")
    parser.add_argument("--out-root", default="outputs/audit")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--component-feature-csv",
        default="outputs/audit/v30_profiles/native_rgb_boundary_diagnostic/native_tube_component_cannot_link_candidate_features.csv",
    )
    parser.add_argument(
        "--proposal-row-csv",
        default="outputs/audit/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_probe5/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_probe5_proposal_rows.csv",
    )
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(json.dumps(_json_safe(payload["gate_summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
