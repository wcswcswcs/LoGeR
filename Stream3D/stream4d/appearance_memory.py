from __future__ import annotations

import numpy as np

from .local_4d_filter import LocalProposal


def normalize_feature(feature: np.ndarray | None) -> np.ndarray | None:
    if feature is None:
        return None
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    a_norm = normalize_feature(a)
    b_norm = normalize_feature(b)
    if a_norm is None or b_norm is None or a_norm.shape != b_norm.shape:
        return 0.0
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))


def cosine_similarity_01(a: np.ndarray | None, b: np.ndarray | None) -> float:
    score, _ = cosine_similarity_01_valid(a, b)
    return score


def cosine_similarity_01_valid(a: np.ndarray | None, b: np.ndarray | None) -> tuple[float, bool]:
    a_norm = normalize_feature(a)
    b_norm = normalize_feature(b)
    if a_norm is None or b_norm is None or a_norm.shape != b_norm.shape:
        return 0.0, False
    return float(0.5 * (np.clip(np.dot(a_norm, b_norm), -1.0, 1.0) + 1.0)), True


def _rgb_histogram(
    rgb: np.ndarray,
    mask: np.ndarray,
    mask_id: int,
    bins: int,
    max_pixels: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    ys, xs = np.where(mask == int(mask_id))
    if ys.size == 0:
        return None, None, 0
    if max_pixels > 0 and ys.size > max_pixels:
        keep = np.linspace(0, ys.size - 1, num=max_pixels, dtype=np.int64)
        ys = ys[keep]
        xs = xs[keep]
    pixels = rgb[ys, xs].astype(np.float32)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=bins, range=(0.0, 255.0))
        hist_parts.append(hist.astype(np.float32))
    hist_vec = np.concatenate(hist_parts, axis=0)
    hist_vec = normalize_feature(hist_vec)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    centroid = np.asarray(
        [
            float(xs.mean() / max(width - 1, 1)),
            float(ys.mean() / max(height - 1, 1)),
        ],
        dtype=np.float32,
    )
    return hist_vec, centroid, int(ys.size)


def attach_proposal_features(
    proposals: list[LocalProposal],
    rgb_window: np.ndarray,
    masks_window: np.ndarray,
    frame_ids: list[int],
    bins: int = 8,
    max_pixels_per_mask: int = 2048,
    max_masks_per_proposal: int = 8,
) -> dict[str, float]:
    if rgb_window.ndim != 4:
        raise ValueError(f"rgb_window must be [T,H,W,3], got {rgb_window.shape}")
    if masks_window.ndim != 3:
        raise ValueError(f"masks_window must be [T,H,W], got {masks_window.shape}")
    frame_to_local = {int(frame_id): idx for idx, frame_id in enumerate(frame_ids)}
    feature_count = 0
    centroid_count = 0
    mask_count = 0
    pixel_count = 0

    for proposal in proposals:
        unique_masks: dict[tuple[int, int], float] = {}
        for frame_id, mask_id, coverage in proposal.mask_observations:
            key = (int(frame_id), int(mask_id))
            unique_masks[key] = max(float(coverage), unique_masks.get(key, 0.0))
        ranked_masks = sorted(unique_masks.items(), key=lambda item: item[1], reverse=True)
        if max_masks_per_proposal > 0:
            ranked_masks = ranked_masks[: int(max_masks_per_proposal)]

        features: list[np.ndarray] = []
        centroids: list[np.ndarray] = []
        weights: list[float] = []
        for (frame_id, mask_id), coverage in ranked_masks:
            local_idx = frame_to_local.get(int(frame_id))
            if local_idx is None:
                continue
            hist, centroid_xy, used_pixels = _rgb_histogram(
                rgb_window[local_idx],
                masks_window[local_idx],
                int(mask_id),
                bins=max(2, int(bins)),
                max_pixels=max_pixels_per_mask,
            )
            if hist is None or centroid_xy is None:
                continue
            frame_term = float(local_idx / max(len(frame_ids) - 1, 1))
            centroid = np.asarray([centroid_xy[0], centroid_xy[1], frame_term], dtype=np.float32)
            weight = max(float(coverage), 1e-6)
            features.append(hist)
            centroids.append(centroid)
            weights.append(weight)
            mask_count += 1
            pixel_count += int(used_pixels)

        if features:
            weights_arr = np.asarray(weights, dtype=np.float32)
            weights_arr = weights_arr / max(float(weights_arr.sum()), 1e-8)
            appearance = np.sum(np.stack(features, axis=0) * weights_arr[:, None], axis=0)
            centroid = np.sum(np.stack(centroids, axis=0) * weights_arr[:, None], axis=0)
            proposal.appearance_feature = normalize_feature(appearance)
            proposal.centroid_feature = centroid.astype(np.float32)
            proposal.feature_type = "rgb_histogram_2d_centroid"
            feature_count += 1
            centroid_count += 1
        else:
            proposal.appearance_feature = None
            proposal.centroid_feature = None
            proposal.feature_type = "rgb_histogram_2d_centroid_missing"

    return {
        "memory_v2_feature_type": "rgb_histogram_2d_centroid",
        "memory_v2_appearance_bins": float(max(2, int(bins))),
        "memory_v2_featured_proposals": float(feature_count),
        "memory_v2_centroided_proposals": float(centroid_count),
        "memory_v2_feature_masks": float(mask_count),
        "memory_v2_feature_pixels": float(pixel_count),
    }
