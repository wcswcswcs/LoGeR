from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AppearanceSnapshot:
    global_object_id: int
    frame_id: int
    feature_source: str
    feature_dim: int
    quality: float
    mask_core_area_px: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppearanceViewSet:
    global_object_id: int
    snapshots: list[AppearanceSnapshot] = field(default_factory=list)

    def add(self, snapshot: AppearanceSnapshot) -> None:
        if snapshot.global_object_id != self.global_object_id:
            raise ValueError("snapshot global_object_id does not match view-set")
        self.snapshots.append(snapshot)

    def high_quality(self, min_quality: float) -> list[AppearanceSnapshot]:
        return [snapshot for snapshot in self.snapshots if snapshot.quality >= min_quality]


@dataclass(frozen=True)
class AppearanceDescriptor:
    scene_id: str
    frame_id: int
    object_id: int
    variant: str
    feature_source: str
    vector: np.ndarray = field(repr=False, compare=False)
    mask_area_px: int = 0
    core_area_px: int = 0
    bbox_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    quality: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return int(self.vector.size)


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64, copy=False) > 0


def bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    mask_bool = as_bool_mask(mask)
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def l2_normalize(vector: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = l2_normalize(a)
    bb = l2_normalize(b)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    return float(np.dot(aa, bb))


def interior_core_mask(
    mask: np.ndarray,
    *,
    min_core_area_px: int = 24,
    distance_percentile: float = 55.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a mask core biased away from object boundaries.

    The fallback keeps the deepest pixels when the eroded/interior set would be
    too small. This is intentionally generic; it avoids margin sweeps.
    """

    mask_bool = as_bool_mask(mask)
    area = int(np.count_nonzero(mask_bool))
    empty = np.zeros_like(mask_bool, dtype=bool)
    if area == 0:
        return empty, {
            "mask_area_px": 0,
            "core_area_px": 0,
            "core_strategy": "empty_mask",
            "distance_threshold_px": 0.0,
        }

    dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    vals = dist[mask_bool]
    if vals.size == 0:
        return mask_bool, {
            "mask_area_px": int(area),
            "core_area_px": int(area),
            "core_strategy": "distance_transform_empty_fallback",
            "distance_threshold_px": 0.0,
        }

    threshold = max(1.0, float(np.percentile(vals, float(distance_percentile))))
    core = (dist >= threshold) & mask_bool
    target = min(int(min_core_area_px), int(area))
    strategy = "distance_percentile"
    if int(np.count_nonzero(core)) < target:
        coords = np.column_stack(np.nonzero(mask_bool))
        order = np.argsort(vals)
        keep = coords[order[-target:]] if target > 0 else coords[:0]
        core = np.zeros_like(mask_bool, dtype=bool)
        if keep.size:
            core[keep[:, 0], keep[:, 1]] = True
        strategy = "deepest_pixels_fallback"

    return core.astype(bool), {
        "mask_area_px": int(area),
        "core_area_px": int(np.count_nonzero(core)),
        "core_strategy": strategy,
        "distance_threshold_px": float(threshold),
        "distance_percentile": float(distance_percentile),
    }


def rgb_shape_descriptor(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    core_mask: np.ndarray | None = None,
    hist_bins: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    mask_bool = as_bool_mask(mask)
    if core_mask is None:
        core_mask, core_meta = interior_core_mask(mask_bool)
    else:
        core_mask = as_bool_mask(core_mask)
        core_meta = {
            "mask_area_px": int(np.count_nonzero(mask_bool)),
            "core_area_px": int(np.count_nonzero(core_mask)),
            "core_strategy": "provided",
        }
    sample_mask = core_mask if np.any(core_mask) else mask_bool
    pixels = rgb_arr[sample_mask]
    if pixels.size == 0:
        return np.zeros(6 + hist_bins * 3, dtype=np.float32), {
            **core_meta,
            "descriptor_status": "empty_mask",
        }

    pix = pixels.astype(np.float32) / 255.0
    mean = pix.mean(axis=0)
    std = pix.std(axis=0)
    hists = []
    for channel in range(3):
        hist, _ = np.histogram(pix[:, channel], bins=int(hist_bins), range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        hists.append(hist)

    h, w = mask_bool.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy(mask_bool)
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    area = int(np.count_nonzero(mask_bool))
    shape = np.asarray(
        [
            np.sqrt(float(area) / float(max(h * w, 1))),
            float(bw) / float(max(w, 1)),
            float(bh) / float(max(h, 1)),
            float(x0 + x1 + 1) / (2.0 * float(max(w, 1))),
            float(y0 + y1 + 1) / (2.0 * float(max(h, 1))),
            float(np.clip(np.log((bw + 1.0) / (bh + 1.0)), -4.0, 4.0)) / 4.0,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([mean, std, *hists, shape], axis=0)
    return l2_normalize(vector), {
        **core_meta,
        "descriptor_status": "ok",
        "hist_bins": int(hist_bins),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
    }


def _feature_to_chw(feature: Any) -> np.ndarray:
    if hasattr(feature, "detach"):
        feature = feature.detach().float().cpu().numpy()
    arr = np.asarray(feature, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"batched feature must have batch=1, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"feature must be CHW/HWC/1CHW tensor, got shape {arr.shape}")
    # SAM2 predictor features are CHW. HWC support is kept for cached numpy arrays.
    if arr.shape[0] <= 4096 and arr.shape[1] <= 4096 and arr.shape[2] <= 4096:
        if arr.shape[0] <= arr.shape[-1] and arr.shape[0] not in {arr.shape[1], arr.shape[2]}:
            return arr.astype(np.float32, copy=False)
        if arr.shape[-1] < arr.shape[0] and arr.shape[0] == arr.shape[1]:
            return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def pool_feature_descriptor(
    feature: Any,
    mask: np.ndarray,
    *,
    core_mask: np.ndarray | None = None,
    use_core: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    feat = _feature_to_chw(feature)
    channels, feat_h, feat_w = [int(v) for v in feat.shape]
    mask_bool = as_bool_mask(mask)
    if use_core:
        sample_mask = as_bool_mask(core_mask) if core_mask is not None else interior_core_mask(mask_bool)[0]
        source = "core"
    else:
        sample_mask = mask_bool
        source = "full_mask"
    small = cv2.resize(sample_mask.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST) > 0
    if not bool(np.any(small)):
        small = cv2.resize(mask_bool.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST) > 0
        source = f"{source}_empty_fallback_full_mask"
    if not bool(np.any(small)):
        return np.zeros(channels, dtype=np.float32), {
            "descriptor_status": "empty_mask",
            "feature_shape_chw": [channels, feat_h, feat_w],
            "feature_sample_px": 0,
            "sample_source": source,
        }
    pooled = feat[:, small].mean(axis=1)
    return l2_normalize(pooled), {
        "descriptor_status": "ok",
        "feature_shape_chw": [channels, feat_h, feat_w],
        "feature_sample_px": int(np.count_nonzero(small)),
        "sample_source": source,
    }


def descriptor_memory_bytes(descriptors: list[AppearanceDescriptor], *, dtype_bytes: int = 2) -> int:
    return int(sum(int(desc.feature_dim) * int(dtype_bytes) for desc in descriptors))
