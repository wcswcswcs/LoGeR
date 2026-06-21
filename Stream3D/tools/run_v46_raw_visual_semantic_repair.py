from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.run_v46_raw_carrier_incidence_repair import (
    ROOT,
    MaskNode,
    WindowTrace,
    _build_nodes,
    _deterministic_permutation,
    _json_safe,
    _load_mask_label,
    _load_scene_windows,
    _rank_auc,
    _safe_mean,
    _safe_median,
    _safe_quantile,
    _shared_jaccard,
)
from tools.run_v46_temporal_positive_edge_repair import (
    _edge_rows_for_scene as _temporal_edge_rows_for_scene,
    _frame_rank_by_id,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _precision_at_k(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    ranked = sorted(rows, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)[: min(int(k), len(rows))]
    if not ranked:
        return None
    return float(sum(1 for row in ranked if row.get("diagnostic_same_gt") is True) / len(ranked))


def _recall_at_threshold(rows: list[dict[str, Any]], score_key: str, threshold: float) -> float | None:
    positives = [row for row in rows if row.get("diagnostic_same_gt") is True]
    if not positives:
        return None
    return float(sum(1 for row in positives if float(row.get(score_key) or 0.0) >= float(threshold)) / len(positives))


def _eligible_nodes(nodes: list[MaskNode], *, max_edge_nodes: int, min_node_carriers: int) -> list[MaskNode]:
    eligible = [
        node
        for node in nodes
        if node.support_count >= int(min_node_carriers) and node.dominant_gt is not None and node.dominant_gt_purity is not None
    ]
    eligible.sort(key=lambda node: (node.support_count, node.area), reverse=True)
    return eligible[: int(max_edge_nodes)]


def _load_color(scene: str, frame_id: int) -> np.ndarray | None:
    path = ROOT / "data/scannet/processed" / scene / "color" / f"{int(frame_id)}.jpg"
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def _node_color_histogram(node: MaskNode, *, min_pixels: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    color = _load_color(node.scene, node.frame_id)
    label = _load_mask_label(node.scene, node.frame_id)
    diag = {
        "scene": node.scene,
        "node_id": node.node_id,
        "frame_id": node.frame_id,
        "mask_id": node.mask_id,
        "descriptor_available": False,
        "descriptor_pixel_count": 0,
        "missing_reason": None,
        "uses_rgb_for_prediction": True,
        "uses_gt_for_prediction": False,
    }
    if color is None:
        diag["missing_reason"] = "missing_color_jpg"
        return None, diag
    if label is None:
        diag["missing_reason"] = "missing_cropformer_mask_png"
        return None, diag
    if color.shape[:2] != label.shape[:2]:
        color = cv2.resize(color, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = label == int(node.mask_id)
    pixel_count = int(mask.sum())
    diag["descriptor_pixel_count"] = pixel_count
    if pixel_count < int(min_pixels):
        diag["missing_reason"] = "too_few_mask_pixels"
        return None, diag
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask]
    h_hist = np.histogram(pixels[:, 0], bins=12, range=(0, 180))[0].astype(np.float32)
    s_hist = np.histogram(pixels[:, 1], bins=8, range=(0, 256))[0].astype(np.float32)
    v_hist = np.histogram(pixels[:, 2], bins=8, range=(0, 256))[0].astype(np.float32)
    hist = np.concatenate([h_hist, s_hist, v_hist])
    total = float(hist.sum())
    if total <= 0.0:
        diag["missing_reason"] = "empty_histogram"
        return None, diag
    hist /= total
    diag["descriptor_available"] = True
    return hist, diag


def _hist_intersection(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.minimum(left, right).sum())


def _cosine01(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, 0.5 * (float(np.dot(a, b) / denom) + 1.0))))


def _descriptor_similarity(left: np.ndarray | None, right: np.ndarray | None, *, feature_backend: str) -> float:
    if str(feature_backend) == "colorhist":
        return _hist_intersection(left, right)
    return _cosine01(left, right)


def _is_frozen_dense_backend(feature_backend: str) -> bool:
    return str(feature_backend) in {"dinov2_timm", "radio_radseg"}


def _semantic_source(feature_backend: str) -> str:
    if str(feature_backend) == "colorhist":
        return "rgb_hsv_color_histogram_proxy"
    return str(feature_backend)


def _node_feature_descriptor(
    node: MaskNode,
    *,
    min_pixels: int,
    feature_backend: str,
    feature_adapter: Any | None,
    feature_map_cache: dict[int, Any],
    feature_checkpoint: str | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if str(feature_backend) == "colorhist":
        descriptor, diag = _node_color_histogram(node, min_pixels=min_pixels)
        diag["feature_backend"] = "colorhist"
        diag["feature_checkpoint"] = None
        diag["semantic_source"] = _semantic_source(feature_backend)
        diag["uses_frozen_dense_features"] = False
        return descriptor, diag

    color = _load_color(node.scene, node.frame_id)
    label = _load_mask_label(node.scene, node.frame_id)
    diag = {
        "scene": node.scene,
        "node_id": node.node_id,
        "frame_id": node.frame_id,
        "mask_id": node.mask_id,
        "descriptor_available": False,
        "descriptor_pixel_count": 0,
        "descriptor_dim": None,
        "feature_map_shape": None,
        "feature_backend": str(feature_backend),
        "feature_checkpoint": feature_checkpoint,
        "semantic_source": _semantic_source(feature_backend),
        "uses_rgb_for_prediction": True,
        "uses_frozen_dense_features": _is_frozen_dense_backend(feature_backend),
        "uses_gt_for_prediction": False,
        "missing_reason": None,
    }
    if feature_adapter is None:
        diag["missing_reason"] = "missing_feature_adapter"
        return None, diag
    if color is None:
        diag["missing_reason"] = "missing_color_jpg"
        return None, diag
    if label is None:
        diag["missing_reason"] = "missing_cropformer_mask_png"
        return None, diag
    if color.shape[:2] != label.shape[:2]:
        color = cv2.resize(color, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = label == int(node.mask_id)
    pixel_count = int(mask.sum())
    diag["descriptor_pixel_count"] = pixel_count
    if pixel_count < int(min_pixels):
        diag["missing_reason"] = "too_few_mask_pixels"
        return None, diag
    try:
        feature_map = feature_map_cache.get(int(node.frame_id))
        if feature_map is None:
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            feature_map = feature_adapter.extract_dense_features(rgb)
            feature_map_cache[int(node.frame_id)] = feature_map
        pooled = np.asarray(feature_adapter.pool_mask_feature(feature_map, mask), dtype=np.float32)
    except Exception as exc:
        diag["missing_reason"] = f"feature_extract_failed:{type(exc).__name__}:{exc}"
        return None, diag
    norm = float(np.linalg.norm(pooled))
    if pooled.size == 0 or norm <= 1e-8:
        diag["missing_reason"] = "empty_feature"
        return None, diag
    diag["descriptor_available"] = True
    diag["descriptor_dim"] = int(pooled.reshape(-1).shape[0])
    diag["feature_map_shape"] = list(np.asarray(feature_map.features).shape)
    return pooled / norm, diag


def _semantic_capped_boost(
    *,
    p4_score: float,
    p3_score: float,
    temporal_score: float,
    semantic_score: float,
    semantic_weight: float,
    semantic_floor: float,
    geometry_support_min: float,
    low_geometry_semantic_cap: float,
) -> float:
    if p3_score < float(geometry_support_min) and temporal_score < float(geometry_support_min):
        semantic_score = min(float(semantic_score), float(low_geometry_semantic_cap))
    boost = max(0.0, float(semantic_score) - float(semantic_floor))
    return float(min(1.0, float(p4_score) + float(semantic_weight) * boost))


def _semantic_linear(
    *,
    p4_score: float,
    p3_score: float,
    temporal_score: float,
    semantic_score: float,
    semantic_weight: float,
    geometry_support_min: float,
    low_geometry_semantic_cap: float,
) -> float:
    if p3_score < float(geometry_support_min) and temporal_score < float(geometry_support_min):
        semantic_score = min(float(semantic_score), float(low_geometry_semantic_cap))
    return float((1.0 - float(semantic_weight)) * float(p4_score) + float(semantic_weight) * float(semantic_score))


def _semantic_product_rescore(
    *,
    p4_score: float,
    p3_score: float,
    temporal_score: float,
    semantic_score: float,
    semantic_rescore_weight: float,
    geometry_support_min: float,
    low_geometry_semantic_cap: float,
) -> float:
    if p3_score < float(geometry_support_min) and temporal_score < float(geometry_support_min):
        semantic_score = min(float(semantic_score), float(low_geometry_semantic_cap))
    semantic_factor = (1.0 - float(semantic_rescore_weight)) + float(semantic_rescore_weight) * float(semantic_score)
    return float(float(p4_score) * max(0.0, min(1.0, semantic_factor)))


def _negative_precision(rows: list[dict[str, Any]], flag_key: str) -> float | None:
    flagged = [row for row in rows if bool(row.get(flag_key))]
    if not flagged:
        return None
    return float(sum(1 for row in flagged if row.get("diagnostic_same_gt") is False) / len(flagged))


def _false_merge_reduction(rows: list[dict[str, Any]], flag_key: str, positive_key: str, threshold: float) -> float | None:
    false_merges = [
        row
        for row in rows
        if row.get("diagnostic_same_gt") is False and float(row.get(positive_key) or 0.0) >= float(threshold)
    ]
    if not false_merges:
        return None
    vetoed = [row for row in false_merges if bool(row.get(flag_key))]
    return float(len(vetoed) / len(false_merges))


def _augment_edges_with_visual_semantics(
    *,
    scene: str,
    edge_rows: list[dict[str, Any]],
    capped_nodes: list[MaskNode],
    descriptors_by_node_id: dict[int, np.ndarray | None],
    semantic_weight: float,
    semantic_rescore_weight: float,
    semantic_floor: float,
    geometry_support_min: float,
    low_geometry_semantic_cap: float,
    positive_threshold: float,
    semantic_negative_thresholds: list[float],
    semantic_negative_protect_geometry: float,
    feature_backend: str,
) -> list[dict[str, Any]]:
    nodes_by_id = {node.node_id: node for node in capped_nodes}
    permuted_indices = _deterministic_permutation(capped_nodes, seed_text=f"v46_visual_semantic_shuffle:{scene}") if capped_nodes else []
    shuffled_by_node_id: dict[int, MaskNode] = {}
    for idx, node in enumerate(capped_nodes):
        shuffled_by_node_id[node.node_id] = capped_nodes[permuted_indices[idx % len(permuted_indices)]]

    augmented: list[dict[str, Any]] = []
    for row in edge_rows:
        left = nodes_by_id.get(int(row["left_node_id"]))
        right = nodes_by_id.get(int(row["right_node_id"]))
        if left is None or right is None:
            continue
        shuffled_right = shuffled_by_node_id.get(right.node_id, right)
        left_desc = descriptors_by_node_id.get(left.node_id)
        right_desc = descriptors_by_node_id.get(right.node_id)
        shuffled_right_desc = descriptors_by_node_id.get(shuffled_right.node_id)
        p4 = float(row.get("P4_vc_q_temporal") or 0.0)
        p4_shuffled = float(row.get("P4_shuffled_vc_q_temporal") or 0.0)
        p3 = float(row.get("P3_view_consensus_q") or 0.0)
        temporal = float(row.get("P1_adjacent_temporal") or 0.0)
        semantic = _descriptor_similarity(left_desc, right_desc, feature_backend=feature_backend)
        semantic_shuffled = _descriptor_similarity(left_desc, shuffled_right_desc, feature_backend=feature_backend)
        p5_boost = _semantic_capped_boost(
            p4_score=p4,
            p3_score=p3,
            temporal_score=temporal,
            semantic_score=semantic,
            semantic_weight=semantic_weight,
            semantic_floor=semantic_floor,
            geometry_support_min=geometry_support_min,
            low_geometry_semantic_cap=low_geometry_semantic_cap,
        )
        p5_linear = _semantic_linear(
            p4_score=p4,
            p3_score=p3,
            temporal_score=temporal,
            semantic_score=semantic,
            semantic_weight=semantic_weight,
            geometry_support_min=geometry_support_min,
            low_geometry_semantic_cap=low_geometry_semantic_cap,
        )
        p5_product = _semantic_product_rescore(
            p4_score=p4,
            p3_score=p3,
            temporal_score=temporal,
            semantic_score=semantic,
            semantic_rescore_weight=semantic_rescore_weight,
            geometry_support_min=geometry_support_min,
            low_geometry_semantic_cap=low_geometry_semantic_cap,
        )
        p5_no_temporal_boost = _semantic_capped_boost(
            p4_score=p3,
            p3_score=p3,
            temporal_score=0.0,
            semantic_score=semantic,
            semantic_weight=semantic_weight,
            semantic_floor=semantic_floor,
            geometry_support_min=geometry_support_min,
            low_geometry_semantic_cap=low_geometry_semantic_cap,
        )
        p5_shuffled_boost = _semantic_capped_boost(
            p4_score=p4_shuffled,
            p3_score=float(row.get("P3_view_consensus_q") or 0.0),
            temporal_score=float(row.get("P4_shuffled_temporal") or 0.0),
            semantic_score=semantic_shuffled,
            semantic_weight=semantic_weight,
            semantic_floor=semantic_floor,
            geometry_support_min=geometry_support_min,
            low_geometry_semantic_cap=low_geometry_semantic_cap,
        )
        p5_no_temporal_product = _semantic_product_rescore(
            p4_score=p3,
            p3_score=p3,
            temporal_score=0.0,
            semantic_score=semantic,
            semantic_rescore_weight=semantic_rescore_weight,
            geometry_support_min=geometry_support_min,
            low_geometry_semantic_cap=low_geometry_semantic_cap,
        )
        p5_shuffled_product = _semantic_product_rescore(
            p4_score=p4_shuffled,
            p3_score=float(row.get("P3_view_consensus_q") or 0.0),
            temporal_score=float(row.get("P4_shuffled_temporal") or 0.0),
            semantic_score=semantic_shuffled,
            semantic_rescore_weight=semantic_rescore_weight,
            geometry_support_min=geometry_support_min,
            low_geometry_semantic_cap=low_geometry_semantic_cap,
        )
        out = {
            **row,
            "P5_p4_semantic_boost_capped": p5_boost,
            "P5_p4_semantic_linear_capped": p5_linear,
            "P5_p4_semantic_product_rescore_capped": p5_product,
            "P5_no_temporal_semantic_boost_capped": p5_no_temporal_boost,
            "P5_no_temporal_semantic_product_rescore_capped": p5_no_temporal_product,
            "P5_shuffled_semantic_boost_capped": p5_shuffled_boost,
            "P5_shuffled_semantic_product_rescore_capped": p5_shuffled_product,
            "P6_feature_only": semantic,
            "P6_feature_only_shuffled": semantic_shuffled,
            "left_descriptor_available": left_desc is not None,
            "right_descriptor_available": right_desc is not None,
            "semantic_weight": float(semantic_weight),
            "semantic_rescore_weight": float(semantic_rescore_weight),
            "semantic_floor": float(semantic_floor),
            "geometry_support_min": float(geometry_support_min),
            "low_geometry_semantic_cap": float(low_geometry_semantic_cap),
            "semantic_source": _semantic_source(feature_backend),
            "feature_backend": str(feature_backend),
            "uses_frozen_dense_features": _is_frozen_dense_backend(feature_backend),
            "uses_rgb_for_prediction": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for threshold in semantic_negative_thresholds:
            suffix = f"{threshold:g}".replace(".", "p")
            hard_key = f"N4_semantic_contradiction_le_{suffix}"
            guarded_key = f"N4_semantic_contradiction_guarded_le_{suffix}"
            out[hard_key] = bool(semantic <= float(threshold))
            out[guarded_key] = bool(semantic <= float(threshold) and p4 < float(semantic_negative_protect_geometry))
        out["positive_threshold_for_reduction"] = float(positive_threshold)
        out["semantic_negative_protect_geometry"] = float(semantic_negative_protect_geometry)
        augmented.append(out)
    return augmented


def _summarize_positive_edges(
    *,
    scene: str,
    rows: list[dict[str, Any]],
    positive_threshold: float,
    feature_backend: str,
) -> list[dict[str, Any]]:
    labels = [bool(row["diagnostic_same_gt"]) for row in rows]
    metric_keys = [
        "shared_carrier_jaccard",
        "P4_vc_q_temporal",
        "P5_p4_semantic_boost_capped",
        "P5_p4_semantic_linear_capped",
        "P5_p4_semantic_product_rescore_capped",
        "P5_no_temporal_semantic_boost_capped",
        "P5_no_temporal_semantic_product_rescore_capped",
        "P5_shuffled_semantic_boost_capped",
        "P5_shuffled_semantic_product_rescore_capped",
        "P6_feature_only",
    ]
    summary_by_key: dict[str, dict[str, Any]] = {}
    for key in metric_keys:
        scores = [float(row.get(key) or 0.0) for row in rows]
        summary_by_key[key] = {
            "scene": scene,
            "variant": key,
            "edge_count": len(rows),
            "positive_edge_density@threshold": float(sum(1 for score in scores if score >= float(positive_threshold)) / max(len(scores), 1)),
            "mean_observer_count": _safe_mean(row.get("P3_observer_count") for row in rows),
            "median_observer_count": _safe_median(row.get("P3_observer_count") for row in rows),
            "view_consensus_mean": _safe_mean(row.get("P3_view_consensus_q") for row in rows),
            "view_consensus_p90": _safe_quantile([row.get("P3_view_consensus_q") for row in rows], 0.90),
            "edge_same_gt_AUC": _rank_auc(labels, scores),
            "edge_precision@top1k": _precision_at_k(rows, key, 1000),
            "edge_precision@top5k": _precision_at_k(rows, key, 5000),
            "edge_recall@threshold": _recall_at_threshold(rows, key, positive_threshold),
            "score_mean": _safe_mean(scores),
            "score_p90": _safe_quantile(scores, 0.90),
            "descriptor_pair_coverage": float(
                sum(1 for row in rows if bool(row.get("left_descriptor_available")) and bool(row.get("right_descriptor_available")))
                / max(len(rows), 1)
            ),
            "semantic_source": _semantic_source(feature_backend),
            "feature_backend": str(feature_backend),
            "uses_rgb_for_prediction": True,
            "uses_frozen_dense_features": _is_frozen_dense_backend(feature_backend),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    shared = summary_by_key["shared_carrier_jaccard"]
    feature_only = summary_by_key["P6_feature_only"]
    controls = {
        "P5_p4_semantic_boost_capped": (
            summary_by_key["P5_shuffled_semantic_boost_capped"],
            summary_by_key["P5_no_temporal_semantic_boost_capped"],
        ),
        "P5_p4_semantic_linear_capped": (
            summary_by_key["P5_shuffled_semantic_boost_capped"],
            summary_by_key["P5_no_temporal_semantic_boost_capped"],
        ),
        "P5_p4_semantic_product_rescore_capped": (
            summary_by_key["P5_shuffled_semantic_product_rescore_capped"],
            summary_by_key["P5_no_temporal_semantic_product_rescore_capped"],
        ),
    }
    for key, (shuffled, no_temporal) in controls.items():
        row = summary_by_key[key]
        auc = row.get("edge_same_gt_AUC")
        p5 = row.get("edge_precision@top5k")
        row["real_minus_shared_edge_AUC"] = None if auc is None or shared.get("edge_same_gt_AUC") is None else float(auc - shared["edge_same_gt_AUC"])
        row["precision_top5k_minus_shared"] = None if p5 is None or shared.get("edge_precision@top5k") is None else float(p5 - shared["edge_precision@top5k"])
        row["real_minus_shuffled_edge_AUC"] = None if auc is None or shuffled.get("edge_same_gt_AUC") is None else float(auc - shuffled["edge_same_gt_AUC"])
        row["real_minus_no_temporal_edge_AUC"] = (
            None if auc is None or no_temporal.get("edge_same_gt_AUC") is None else float(auc - no_temporal["edge_same_gt_AUC"])
        )
        row["P6_feature_only_minus_P5_edge_AUC"] = (
            None if auc is None or feature_only.get("edge_same_gt_AUC") is None else float(feature_only["edge_same_gt_AUC"] - auc)
        )
        row["P6_feature_only_beats_full_P5"] = bool(
            row["P6_feature_only_minus_P5_edge_AUC"] is not None and row["P6_feature_only_minus_P5_edge_AUC"] > 0.0
        )
        row["gate_pass"] = bool(
            row["real_minus_shared_edge_AUC"] is not None
            and row["real_minus_shared_edge_AUC"] >= 0.08
            and row["precision_top5k_minus_shared"] is not None
            and row["precision_top5k_minus_shared"] >= 0.10
            and row["real_minus_shuffled_edge_AUC"] is not None
            and row["real_minus_shuffled_edge_AUC"] >= 0.10
            and row["real_minus_no_temporal_edge_AUC"] is not None
            and row["real_minus_no_temporal_edge_AUC"] >= 0.08
            and not row["P6_feature_only_beats_full_P5"]
        )
    return list(summary_by_key.values())


def _summarize_negative_edges(
    *,
    scene: str,
    rows: list[dict[str, Any]],
    semantic_negative_thresholds: list[float],
    positive_key: str,
    positive_threshold: float,
    feature_backend: str,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for threshold in semantic_negative_thresholds:
        suffix = f"{threshold:g}".replace(".", "p")
        for key in [f"N4_semantic_contradiction_le_{suffix}", f"N4_semantic_contradiction_guarded_le_{suffix}"]:
            hard_count = int(sum(1 for row in rows if bool(row.get(key))))
            precision = _negative_precision(rows, key)
            reduction = _false_merge_reduction(rows, key, positive_key, positive_threshold)
            summary_rows.append(
                {
                    "scene": scene,
                    "variant": key,
                    "negative_edge_count": hard_count,
                    "hard_negative_count": hard_count,
                    "semantic_negative_count": hard_count,
                    "negative_edge_precision": precision,
                    "same_frame_false_merge_reduction": reduction,
                    "positive_negative_conflict_ratio": float(
                        sum(1 for row in rows if bool(row.get(key)) and float(row.get(positive_key) or 0.0) >= float(positive_threshold))
                        / max(hard_count, 1)
                    )
                    if hard_count
                    else None,
                    "gate_pass": bool(precision is not None and precision >= 0.75),
                    "semantic_source": _semantic_source(feature_backend),
                    "feature_backend": str(feature_backend),
                    "uses_rgb_for_prediction": True,
                    "uses_frozen_dense_features": _is_frozen_dense_backend(feature_backend),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    return summary_rows


def _scene_payload(
    *,
    scene: str,
    carrier_cache_root: Path,
    visibility_threshold: float,
    confidence_threshold: float,
    min_mask_area: int,
    max_edge_nodes: int,
    min_node_carriers: int,
    min_visible_carriers: int,
    max_temporal_frame_gap: int,
    temporal_weight: float,
    visible_outside_threshold: float,
    quality_variant: str,
    descriptor_min_pixels: int,
    semantic_weight: float,
    semantic_rescore_weight: float,
    semantic_floor: float,
    geometry_support_min: float,
    low_geometry_semantic_cap: float,
    positive_threshold: float,
    semantic_negative_thresholds: list[float],
    semantic_negative_protect_geometry: float,
    feature_backend: str,
    feature_adapter: Any | None,
    feature_checkpoint: str | None,
) -> dict[str, Any]:
    windows, window_rows, manifest_diag = _load_scene_windows(
        scene=scene,
        carrier_cache_root=carrier_cache_root,
        visibility_threshold=float(visibility_threshold),
        confidence_threshold=float(confidence_threshold),
        min_mask_area=int(min_mask_area),
    )
    nodes, frame_rows, node_diag = _build_nodes(scene, windows, min_mask_area=int(min_mask_area))
    frame_rank_by_id = _frame_rank_by_id(windows)
    edge_rows, quality_rows = _temporal_edge_rows_for_scene(
        scene=scene,
        nodes=nodes,
        windows=windows,
        frame_rank_by_id=frame_rank_by_id,
        quality_variant=quality_variant,
        max_edge_nodes=int(max_edge_nodes),
        min_node_carriers=int(min_node_carriers),
        min_visible_carriers=int(min_visible_carriers),
        max_temporal_observation_gap=int(max_temporal_frame_gap),
        temporal_weight=float(temporal_weight),
        visible_outside_threshold=float(visible_outside_threshold),
    )
    capped_nodes = _eligible_nodes(nodes, max_edge_nodes=max_edge_nodes, min_node_carriers=min_node_carriers)
    descriptors_by_node_id: dict[int, np.ndarray | None] = {}
    descriptor_rows: list[dict[str, Any]] = []
    feature_map_cache: dict[int, Any] = {}
    for node in capped_nodes:
        descriptor, descriptor_row = _node_feature_descriptor(
            node,
            min_pixels=int(descriptor_min_pixels),
            feature_backend=str(feature_backend),
            feature_adapter=feature_adapter,
            feature_map_cache=feature_map_cache,
            feature_checkpoint=feature_checkpoint,
        )
        descriptors_by_node_id[node.node_id] = descriptor
        descriptor_rows.append(descriptor_row)
    augmented_edges = _augment_edges_with_visual_semantics(
        scene=scene,
        edge_rows=edge_rows,
        capped_nodes=capped_nodes,
        descriptors_by_node_id=descriptors_by_node_id,
        semantic_weight=float(semantic_weight),
        semantic_rescore_weight=float(semantic_rescore_weight),
        semantic_floor=float(semantic_floor),
        geometry_support_min=float(geometry_support_min),
        low_geometry_semantic_cap=float(low_geometry_semantic_cap),
        positive_threshold=float(positive_threshold),
        semantic_negative_thresholds=semantic_negative_thresholds,
        semantic_negative_protect_geometry=float(semantic_negative_protect_geometry),
        feature_backend=str(feature_backend),
    )
    positive_summary_rows = _summarize_positive_edges(
        scene=scene,
        rows=augmented_edges,
        positive_threshold=float(positive_threshold),
        feature_backend=str(feature_backend),
    )
    negative_summary_rows = _summarize_negative_edges(
        scene=scene,
        rows=augmented_edges,
        semantic_negative_thresholds=semantic_negative_thresholds,
        positive_key="P5_p4_semantic_boost_capped",
        positive_threshold=float(positive_threshold),
        feature_backend=str(feature_backend),
    )
    diag = {
        **manifest_diag,
        **node_diag,
        "edge_eval_node_count": len(capped_nodes),
        "descriptor_available_count": int(sum(1 for row in descriptor_rows if bool(row["descriptor_available"]))),
        "descriptor_node_count": int(len(descriptor_rows)),
        "descriptor_coverage": float(
            sum(1 for row in descriptor_rows if bool(row["descriptor_available"])) / max(len(descriptor_rows), 1)
        ),
        "semantic_source": _semantic_source(feature_backend),
        "feature_backend": str(feature_backend),
        "feature_checkpoint": feature_checkpoint,
        "feature_map_cache_frame_count": int(len(feature_map_cache)),
        "uses_rgb_for_prediction": True,
        "uses_frozen_dense_features": _is_frozen_dense_backend(feature_backend),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    node_rows = [
        {
            "scene": scene,
            "node_id": node.node_id,
            "frame_id": node.frame_id,
            "mask_id": node.mask_id,
            "area": node.area,
            "carrier_support_count": node.support_count,
            "support_density": node.support_density,
            "diagnostic_gt_instance": node.dominant_gt,
            "diagnostic_gt_purity": node.dominant_gt_purity,
            "selected_for_edge_eval": node in capped_nodes,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for node in nodes
    ]
    return {
        "edge_rows": augmented_edges,
        "positive_summary_rows": positive_summary_rows,
        "negative_summary_rows": negative_summary_rows,
        "descriptor_rows": descriptor_rows,
        "node_rows": node_rows,
        "frame_rows": frame_rows,
        "window_rows": window_rows,
        "quality_rows": quality_rows,
        "diag": diag,
    }


def _gate(summary_rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in summary_rows if str(row.get("variant")) == variant]
    return {
        "variant": variant,
        "any_scene_variant_gate_pass": any(bool(row.get("gate_pass")) for row in selected),
        "all_scene_variant_gate_pass": bool(selected and all(bool(row.get("gate_pass")) for row in selected)),
        "scene_count": len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-cache v46 P5/P6/N4 RGB color-hist semantic proxy repair.")
    parser.add_argument("--carrier-cache-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--max-edge-nodes", type=int, default=120)
    parser.add_argument("--min-node-carriers", type=int, default=5)
    parser.add_argument("--min-visible-carriers", type=int, default=3)
    parser.add_argument("--max-temporal-frame-gap", type=int, default=2)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--visible-outside-threshold", type=float, default=0.80)
    parser.add_argument("--quality-variant", default="Q5_split_outside_fragment_soft")
    parser.add_argument("--descriptor-min-pixels", type=int, default=64)
    parser.add_argument("--semantic-weight", type=float, default=0.15)
    parser.add_argument("--semantic-rescore-weight", type=float, default=0.50)
    parser.add_argument("--semantic-floor", type=float, default=0.45)
    parser.add_argument("--geometry-support-min", type=float, default=0.04)
    parser.add_argument("--low-geometry-semantic-cap", type=float, default=0.10)
    parser.add_argument("--positive-threshold", type=float, default=0.50)
    parser.add_argument("--semantic-negative-thresholds", default="0.20,0.30,0.40")
    parser.add_argument("--semantic-negative-protect-geometry", type=float, default=0.25)
    parser.add_argument("--feature-backend", default="colorhist", choices=["colorhist", "rgb_stats", "dinov2_timm", "radio_radseg"])
    parser.add_argument("--feature-device", default="cpu")
    parser.add_argument("--feature-short-side", type=int, default=518)
    parser.add_argument("--feature-checkpoint", default="")
    parser.add_argument("--radio-lang-model", default="siglip2")
    parser.add_argument("--radio-lang-align", action="store_true")
    parser.add_argument("--radio-slide-crop", type=int, default=0)
    parser.add_argument("--radio-slide-stride", type=int, default=224)
    parser.add_argument("--output-root", default="outputs/audit/v46_raw_visual_semantic_repair")
    args = parser.parse_args()

    carrier_cache_root = ROOT / str(args.carrier_cache_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    semantic_negative_thresholds = [
        float(item.strip()) for item in str(args.semantic_negative_thresholds).split(",") if item.strip()
    ]
    feature_checkpoint = str(args.feature_checkpoint).strip() or None
    feature_adapter: Any | None = None
    if str(args.feature_backend) != "colorhist":
        from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_dinov2_checkpoint, locate_default_radio_checkpoint

        if feature_checkpoint is None and str(args.feature_backend) == "dinov2_timm":
            feature_checkpoint = locate_default_dinov2_checkpoint()
        if feature_checkpoint is None and str(args.feature_backend) == "radio_radseg":
            feature_checkpoint = locate_default_radio_checkpoint()
        feature_adapter = FrozenFeatureAdapter(
            backend=str(args.feature_backend),
            device=str(args.feature_device),
            checkpoint=feature_checkpoint,
            short_side=int(args.feature_short_side),
            radio_lang_model=str(args.radio_lang_model),
            radio_lang_align=bool(args.radio_lang_align),
            radio_slide_crop=int(args.radio_slide_crop),
            radio_slide_stride=int(args.radio_slide_stride),
        )

    all_edge_rows: list[dict[str, Any]] = []
    all_positive_summary_rows: list[dict[str, Any]] = []
    all_negative_summary_rows: list[dict[str, Any]] = []
    all_descriptor_rows: list[dict[str, Any]] = []
    all_node_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_quality_rows: list[dict[str, Any]] = []
    diags: dict[str, Any] = {}

    for scene in scenes:
        payload = _scene_payload(
            scene=scene,
            carrier_cache_root=carrier_cache_root,
            visibility_threshold=float(args.visibility_threshold),
            confidence_threshold=float(args.confidence_threshold),
            min_mask_area=int(args.min_mask_area),
            max_edge_nodes=int(args.max_edge_nodes),
            min_node_carriers=int(args.min_node_carriers),
            min_visible_carriers=int(args.min_visible_carriers),
            max_temporal_frame_gap=int(args.max_temporal_frame_gap),
            temporal_weight=float(args.temporal_weight),
            visible_outside_threshold=float(args.visible_outside_threshold),
            quality_variant=str(args.quality_variant),
            descriptor_min_pixels=int(args.descriptor_min_pixels),
            semantic_weight=float(args.semantic_weight),
            semantic_rescore_weight=float(args.semantic_rescore_weight),
            semantic_floor=float(args.semantic_floor),
            geometry_support_min=float(args.geometry_support_min),
            low_geometry_semantic_cap=float(args.low_geometry_semantic_cap),
            positive_threshold=float(args.positive_threshold),
            semantic_negative_thresholds=semantic_negative_thresholds,
            semantic_negative_protect_geometry=float(args.semantic_negative_protect_geometry),
            feature_backend=str(args.feature_backend),
            feature_adapter=feature_adapter,
            feature_checkpoint=feature_checkpoint,
        )
        all_edge_rows.extend(payload["edge_rows"])
        all_positive_summary_rows.extend(payload["positive_summary_rows"])
        all_negative_summary_rows.extend(payload["negative_summary_rows"])
        all_descriptor_rows.extend(payload["descriptor_rows"])
        all_node_rows.extend(payload["node_rows"])
        all_frame_rows.extend(payload["frame_rows"])
        all_window_rows.extend(payload["window_rows"])
        all_quality_rows.extend(payload["quality_rows"])
        diags[scene] = payload["diag"]

    p5_boost_gate = _gate(all_positive_summary_rows, "P5_p4_semantic_boost_capped")
    p5_linear_gate = _gate(all_positive_summary_rows, "P5_p4_semantic_linear_capped")
    p5_product_gate = _gate(all_positive_summary_rows, "P5_p4_semantic_product_rescore_capped")
    n4_rows = [row for row in all_negative_summary_rows if str(row.get("variant")).startswith("N4_semantic")]
    payload = {
        "phase": "v46_raw_visual_semantic_repair",
        "created_at": _utc_now(),
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scenes": scenes,
        "visibility_threshold": float(args.visibility_threshold),
        "confidence_threshold": float(args.confidence_threshold),
        "min_mask_area": int(args.min_mask_area),
        "max_edge_nodes": int(args.max_edge_nodes),
        "min_node_carriers": int(args.min_node_carriers),
        "min_visible_carriers": int(args.min_visible_carriers),
        "max_temporal_frame_gap": int(args.max_temporal_frame_gap),
        "max_temporal_gap_uses_observation_rank": True,
        "temporal_weight": float(args.temporal_weight),
        "quality_variant": str(args.quality_variant),
        "descriptor_min_pixels": int(args.descriptor_min_pixels),
        "semantic_weight": float(args.semantic_weight),
        "semantic_rescore_weight": float(args.semantic_rescore_weight),
        "semantic_floor": float(args.semantic_floor),
        "geometry_support_min": float(args.geometry_support_min),
        "low_geometry_semantic_cap": float(args.low_geometry_semantic_cap),
        "positive_threshold": float(args.positive_threshold),
        "semantic_negative_thresholds": semantic_negative_thresholds,
        "semantic_negative_protect_geometry": float(args.semantic_negative_protect_geometry),
        "semantic_source": _semantic_source(str(args.feature_backend)),
        "feature_backend": str(args.feature_backend),
        "feature_device": str(args.feature_device),
        "feature_short_side": int(args.feature_short_side),
        "feature_checkpoint": feature_checkpoint,
        "radio_lang_model": str(args.radio_lang_model),
        "radio_lang_align": bool(args.radio_lang_align),
        "radio_slide_crop": int(args.radio_slide_crop),
        "radio_slide_stride": int(args.radio_slide_stride),
        "uses_frozen_dense_features": _is_frozen_dense_backend(str(args.feature_backend)),
        "positive_summary_rows": all_positive_summary_rows,
        "negative_summary_rows": all_negative_summary_rows,
        "diag": diags,
        "gate": {
            "P5_p4_semantic_boost_capped": p5_boost_gate,
            "P5_p4_semantic_linear_capped": p5_linear_gate,
            "P5_p4_semantic_product_rescore_capped": p5_product_gate,
            "N4_any_scene_variant_gate_pass": any(bool(row.get("gate_pass")) for row in n4_rows),
            "N4_all_scene_variant_gate_pass": False,
            "pass": bool(
                p5_boost_gate["all_scene_variant_gate_pass"]
                or p5_linear_gate["all_scene_variant_gate_pass"]
                or p5_product_gate["all_scene_variant_gate_pass"]
            ),
            "uses_rgb_for_prediction": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    }
    for variant in sorted({str(row.get("variant")) for row in n4_rows}):
        selected = [row for row in n4_rows if str(row.get("variant")) == variant]
        if selected and all(bool(row.get("gate_pass")) for row in selected):
            payload["gate"]["N4_all_scene_variant_gate_pass"] = True

    out = ROOT / str(args.output_root)
    _write_json(out / "raw_visual_semantic_repair.json", payload)
    _write_csv(out / "raw_visual_semantic_edge_rows.csv", all_edge_rows)
    _write_csv(out / "raw_visual_semantic_positive_summary_rows.csv", all_positive_summary_rows)
    _write_csv(out / "raw_visual_semantic_negative_summary_rows.csv", all_negative_summary_rows)
    _write_csv(out / "raw_visual_semantic_descriptor_rows.csv", all_descriptor_rows)
    _write_csv(out / "raw_visual_semantic_node_rows.csv", all_node_rows)
    _write_csv(out / "raw_visual_semantic_quality_rows.csv", all_quality_rows)
    _write_csv(out / "raw_visual_semantic_frame_rows.csv", all_frame_rows)
    _write_csv(out / "raw_visual_semantic_window_rows.csv", all_window_rows)
    print(json.dumps({"summary": str(out / "raw_visual_semantic_repair.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
