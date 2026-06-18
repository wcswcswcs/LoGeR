from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .frozen_feature_adapter import FrozenFeatureAdapter, FeatureMap


@dataclass(frozen=True)
class SemanticPartToken:
    token_id: int
    frame_id: int
    mask_id: int
    area: int
    feature: np.ndarray
    boundary_contrast: float
    centroid_y: float
    centroid_x: float
    diagnostic_gt_instance: int | None = None
    diagnostic_gt_purity: float | None = None
    diagnostic_gt_iou: float | None = None


def label_map_to_masks(label_map: np.ndarray, *, min_area: int = 32) -> list[tuple[int, np.ndarray]]:
    labels = np.asarray(label_map)
    masks: list[tuple[int, np.ndarray]] = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == int(label)
        if int(mask.sum()) >= int(min_area):
            masks.append((int(label), mask))
    return masks


def stack_to_masks(stack: np.ndarray, *, min_area: int = 32) -> list[tuple[int, np.ndarray]]:
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return label_map_to_masks(arr, min_area=min_area)
    if arr.ndim != 3:
        raise ValueError("mask stack must have shape [N,H,W] or label map [H,W]")
    masks: list[tuple[int, np.ndarray]] = []
    for index, mask in enumerate(arr):
        mask_bool = np.asarray(mask, dtype=bool)
        if int(mask_bool.sum()) >= int(min_area):
            masks.append((int(index + 1), mask_bool))
    return masks


def split_masks_by_feature_clusters(
    masks: list[tuple[int, np.ndarray]],
    feature_map: FeatureMap,
    *,
    image_shape: tuple[int, int],
    min_area: int = 32,
    max_splits: int = 3,
    spatial_weight: float = 0.15,
) -> list[tuple[int, np.ndarray]]:
    """Split large masks into frozen-feature-consistent fragments.

    This is a deterministic v42 repair path for overmerged masks. It uses only
    frozen dense features and mask geometry; any GT labels remain diagnostic.
    """
    import cv2

    feat = np.asarray(feature_map.features, dtype=np.float32)
    feat_h, feat_w = int(feat.shape[0]), int(feat.shape[1])
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    out: list[tuple[int, np.ndarray]] = []
    for mask_id, mask in masks:
        mask_bool = np.asarray(mask, dtype=bool)
        if int(mask_bool.sum()) < int(min_area) * 2:
            out.append((int(mask_id), mask_bool))
            continue
        small = cv2.resize(mask_bool.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        coords = np.argwhere(small)
        if int(coords.shape[0]) < 6:
            out.append((int(mask_id), mask_bool))
            continue
        vectors = feat[small].astype(np.float32)
        if spatial_weight > 0:
            yy = (coords[:, 0:1].astype(np.float32) + 0.5) / max(float(feat_h), 1.0)
            xx = (coords[:, 1:2].astype(np.float32) + 0.5) / max(float(feat_w), 1.0)
            vectors = np.concatenate([vectors, float(spatial_weight) * yy, float(spatial_weight) * xx], axis=1)
        k = 2 if int(coords.shape[0]) < 24 else int(max_splits)
        k = max(2, min(int(k), int(max_splits), int(coords.shape[0])))
        labels = _deterministic_kmeans(vectors, k=k, iters=10)
        label_grid = np.full((feat_h, feat_w), -1, dtype=np.int16)
        label_grid[small] = labels.astype(np.int16)
        full_labels = cv2.resize((label_grid + 1).astype(np.int16), (img_w, img_h), interpolation=cv2.INTER_NEAREST) - 1
        fragments: list[tuple[int, np.ndarray]] = []
        serial = 1
        covered = np.zeros_like(mask_bool, dtype=bool)
        for label in range(k):
            candidate = (full_labels == int(label)) & mask_bool
            if not np.any(candidate):
                continue
            num_components, comp = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
            for comp_id in range(1, int(num_components)):
                piece = comp == int(comp_id)
                if int(piece.sum()) < int(min_area):
                    continue
                fragments.append((int(mask_id) * 1000 + serial, piece))
                covered |= piece
                serial += 1
        residual = mask_bool & ~covered
        if int(residual.sum()) >= int(min_area):
            fragments.append((int(mask_id) * 1000 + serial, residual))
            covered |= residual
        if len(fragments) >= 2 and int(covered.sum()) >= int(min_area):
            out.extend(fragments)
        else:
            out.append((int(mask_id), mask_bool))
    return out


def merge_masks_by_feature_affinity(
    masks: list[tuple[int, np.ndarray]],
    feature_map: FeatureMap,
    *,
    image_shape: tuple[int, int],
    min_area: int = 32,
    affinity_threshold: float = 0.95,
    max_center_distance: float = 0.35,
    max_group_size: int = 6,
) -> list[tuple[int, np.ndarray]]:
    """Merge oversegmented masks using frozen-feature affinity and layout only."""
    import cv2

    if len(masks) <= 1:
        return [(int(mask_id), np.asarray(mask, dtype=bool)) for mask_id, mask in masks]
    feat = np.asarray(feature_map.features, dtype=np.float32)
    feat_h, feat_w = int(feat.shape[0]), int(feat.shape[1])
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    records: list[dict[str, Any]] = []
    for index, (mask_id, mask) in enumerate(masks):
        mask_bool = np.asarray(mask, dtype=bool)
        if int(mask_bool.sum()) < int(min_area):
            continue
        small = cv2.resize(mask_bool.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not np.any(small):
            continue
        yy, xx = np.nonzero(mask_bool)
        feature = _normalize_vector(feat[small].mean(axis=0).astype(np.float32))
        records.append(
            {
                "index": int(index),
                "mask_id": int(mask_id),
                "mask": mask_bool,
                "feature": feature,
                "cy": float(np.mean(yy) / max(float(img_h), 1.0)),
                "cx": float(np.mean(xx) / max(float(img_w), 1.0)),
            }
        )
    if len(records) <= 1:
        return [(int(mask_id), np.asarray(mask, dtype=bool)) for mask_id, mask in masks]

    parent = list(range(len(records)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, a in enumerate(records):
        for right in range(left + 1, len(records)):
            b = records[right]
            distance = float(np.hypot(float(a["cy"]) - float(b["cy"]), float(a["cx"]) - float(b["cx"])))
            if distance > float(max_center_distance):
                continue
            affinity = float(np.dot(np.asarray(a["feature"]), np.asarray(b["feature"])))
            if affinity >= float(affinity_threshold):
                union(left, right)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)

    merged: list[tuple[int, np.ndarray]] = []
    serial = 1
    used_indices: set[int] = set()
    for group in groups.values():
        if len(group) <= 1 or len(group) > int(max_group_size):
            continue
        union_mask = np.zeros((img_h, img_w), dtype=bool)
        for record in group:
            union_mask |= np.asarray(record["mask"], dtype=bool)
            used_indices.add(int(record["index"]))
        if int(union_mask.sum()) >= int(min_area):
            merged.append((int(min(record["mask_id"] for record in group)) * 1000 + serial, union_mask))
            serial += 1
    for index, (mask_id, mask) in enumerate(masks):
        if int(index) not in used_indices:
            merged.append((int(mask_id), np.asarray(mask, dtype=bool)))
    return merged


def _deterministic_kmeans(vectors: np.ndarray, *, k: int, iters: int) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if int(k) <= 1 or x.shape[0] == 1:
        return np.zeros((x.shape[0],), dtype=np.int64)
    centered = x - x.mean(axis=0, keepdims=True)
    axis = int(np.argmax(np.var(centered, axis=0)))
    proj = centered[:, axis]
    seeds: list[int] = []
    for q in np.linspace(0.0, 1.0, num=int(k), dtype=np.float32):
        target = float(np.quantile(proj, float(q)))
        index = int(np.argmin(np.abs(proj - target)))
        if index not in seeds:
            seeds.append(index)
    while len(seeds) < int(k):
        remaining = [idx for idx in range(x.shape[0]) if idx not in seeds]
        if not remaining:
            break
        if not seeds:
            seeds.append(int(remaining[0]))
            continue
        dist = ((x[remaining, None, :] - x[seeds][None, :, :]) ** 2).sum(axis=2).min(axis=1)
        seeds.append(int(remaining[int(np.argmax(dist))]))
    centers = x[seeds[: int(k)]].copy()
    labels = np.zeros((x.shape[0],), dtype=np.int64)
    for _ in range(int(iters)):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1).astype(np.int64)
        for label in range(centers.shape[0]):
            selected = x[labels == label]
            if selected.shape[0] > 0:
                centers[label] = selected.mean(axis=0)
    return labels


def build_semantic_part_tokens(
    *,
    frame_id: int,
    frame: Any,
    masks: list[tuple[int, np.ndarray]],
    adapter: FrozenFeatureAdapter,
    feature_map: FeatureMap | None = None,
    gt_instance: np.ndarray | None = None,
    start_token_id: int = 0,
    feature_mode: str = "pooled",
) -> list[SemanticPartToken]:
    fmap = feature_map or adapter.extract_dense_features(frame)
    tokens: list[SemanticPartToken] = []
    for offset, (mask_id, mask) in enumerate(masks):
        mask_bool = np.asarray(mask, dtype=bool)
        if not np.any(mask_bool):
            continue
        yy, xx = np.nonzero(mask_bool)
        dominant_gt, dominant_purity, dominant_iou = _dominant_gt(mask_bool, gt_instance)
        if str(feature_mode) == "pooled_local_contrast":
            feature = pool_mask_local_contrast_feature(fmap, mask_bool)
        elif str(feature_mode) == "pooled":
            feature = adapter.pool_mask_feature(fmap, mask_bool)
        else:
            raise ValueError(f"unsupported semantic part token feature mode: {feature_mode}")
        tokens.append(
            SemanticPartToken(
                token_id=int(start_token_id + offset),
                frame_id=int(frame_id),
                mask_id=int(mask_id),
                area=int(mask_bool.sum()),
                feature=feature,
                boundary_contrast=float(adapter.compute_boundary_contrast(fmap, mask_bool)),
                centroid_y=float(np.mean(yy)),
                centroid_x=float(np.mean(xx)),
                diagnostic_gt_instance=dominant_gt,
                diagnostic_gt_purity=dominant_purity,
                diagnostic_gt_iou=dominant_iou,
            )
        )
    return tokens


def pool_mask_local_contrast_feature(feature_map: FeatureMap, mask: np.ndarray) -> np.ndarray:
    """Pool a frozen-feature token with a local ring contrast component."""
    import cv2

    features = np.asarray(feature_map.features, dtype=np.float32)
    feat_h, feat_w = int(features.shape[0]), int(features.shape[1])
    mask_small = cv2.resize(np.asarray(mask, dtype=np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not np.any(mask_small):
        return np.zeros((features.shape[-1] * 2,), dtype=np.float32)
    inner = features[mask_small].mean(axis=0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask_small.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~mask_small
    if not np.any(ring):
        ring = ~mask_small
    outer = features[ring].mean(axis=0).astype(np.float32) if np.any(ring) else np.zeros_like(inner)
    contrast = (inner - outer).astype(np.float32)
    inner_norm = _normalize_vector(inner)
    contrast_norm = _normalize_vector(contrast)
    merged = np.concatenate([inner_norm, contrast_norm], axis=0).astype(np.float32)
    return _normalize_vector(merged)


def _normalize_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > float(eps) else arr


def _dominant_gt(mask: np.ndarray, gt_instance: np.ndarray | None) -> tuple[int | None, float | None, float | None]:
    if gt_instance is None:
        return None, None, None
    gt = np.asarray(gt_instance)
    if gt.shape != mask.shape:
        import cv2

        gt = cv2.resize(gt.astype(np.int32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    values, counts = np.unique(gt[np.asarray(mask, dtype=bool)], return_counts=True)
    positive = [(int(v), int(c)) for v, c in zip(values.tolist(), counts.tolist()) if int(v) > 0]
    if not positive:
        return None, 0.0, 0.0
    label, count = max(positive, key=lambda item: item[1])
    gt_mask = gt == int(label)
    union = int(mask.sum()) + int(gt_mask.sum()) - int(count)
    return int(label), float(count / max(int(mask.sum()), 1)), float(count / max(union, 1))
