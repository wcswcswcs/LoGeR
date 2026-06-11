from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .measurement_bank import MeasurementBank


@dataclass
class ExplanationParams:
    birth_min_surfels: int = 16
    birth_min_boundary_safe_ratio: float = 0.65
    birth_max_ambiguous_ratio: float = 0.25
    core_posterior_threshold: float = 0.70
    fringe_posterior_threshold: float = 0.45
    reject_negative_threshold: float = 0.40
    visible_outside_negative_weight: float = 1.0
    boundary_risk_weight: float = 0.5
    appearance_weight: float = 0.3
    d4rt_temporal_weight: float = 0.5
    max_slots_per_frame_mask: int = 3
    min_core_surfels_per_object: int = 12
    min_export_points_per_object: int = 100
    boundary_safe_px: float = 3.0
    measurement_min_surfels: int = 4
    measurement_min_core_ratio: float = 0.08
    unknown_export: bool = False
    enable_target_births: bool = False


def birth_groups(
    bank: MeasurementBank,
    params: ExplanationParams,
    *,
    shuffled_source: bool = False,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    src_frame = bank.src_frame_global.copy()
    src_mask = bank.src_mask_id.copy()
    if shuffled_source and src_mask.shape[0] > 1:
        local_rng = rng or np.random.default_rng(0)
        src_mask = src_mask[local_rng.permutation(src_mask.shape[0])]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (frame_id, mask_id) in enumerate(zip(src_frame.tolist(), src_mask.tolist())):
        if int(mask_id) <= 0:
            continue
        groups[(int(frame_id), int(mask_id))].append(int(idx))
    if params.enable_target_births:
        for local_idx, frame_id in enumerate(bank.frame_ids.tolist()):
            if not bool(bank.mask_frame_available[local_idx]):
                continue
            positive = bank.positive_observation[local_idx]
            ids = bank.target_mask_id[local_idx]
            for mask_id in np.unique(ids[positive]).tolist():
                if int(mask_id) <= 0:
                    continue
                indices = np.flatnonzero(positive & (ids == int(mask_id))).astype(np.int64)
                if indices.size:
                    existing = groups[(int(frame_id), int(mask_id))]
                    existing_set = set(existing)
                    existing.extend(int(v) for v in indices.tolist() if int(v) not in existing_set)
    out: list[dict[str, Any]] = []
    for (frame_id, mask_id), indices_raw in groups.items():
        indices = np.asarray(indices_raw, dtype=np.int64)
        if indices.shape[0] < int(params.birth_min_surfels):
            continue
        boundary_safe = bank.source_boundary_distance[indices] >= float(params.boundary_safe_px)
        boundary_safe_ratio = float(np.count_nonzero(boundary_safe) / max(indices.shape[0], 1))
        negative_counts = bank.negative_observation[:, indices].sum(axis=0)
        ambiguous_ratio = float(np.count_nonzero(negative_counts > 0) / max(indices.shape[0], 1))
        out.append(
            {
                "birth_frame": int(frame_id),
                "birth_mask_id": int(mask_id),
                "surfel_indices": indices,
                "num_birth_surfels": int(indices.shape[0]),
                "boundary_safe_ratio": boundary_safe_ratio,
                "ambiguous_ratio": ambiguous_ratio,
                "passes_birth_gate": bool(
                    boundary_safe_ratio >= float(params.birth_min_boundary_safe_ratio)
                    and ambiguous_ratio <= float(params.birth_max_ambiguous_ratio)
                ),
            }
        )
    out.sort(
        key=lambda item: (
            bool(item["passes_birth_gate"]),
            float(item["boundary_safe_ratio"]),
            int(item["num_birth_surfels"]),
        ),
        reverse=True,
    )
    return out


def posterior_for_group(
    bank: MeasurementBank,
    surfels: np.ndarray,
    params: ExplanationParams,
    *,
    use_negative: bool,
    use_temporal: bool,
) -> dict[str, np.ndarray | float]:
    if surfels.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return {
            "core": empty,
            "fringe": empty,
            "unknown": empty,
            "reject": empty,
            "positive_score": 0.0,
            "negative_ratio": 0.0,
            "boundary_risk": 0.0,
            "temporal_consistency": 0.0,
            "appearance_consistency": 0.0,
        }
    visible_counts = bank.visible_ok[:, surfels].sum(axis=0).astype(np.float32)
    propagated_counts = bank.source_positive_propagated[:, surfels].sum(axis=0).astype(np.float32)
    target_positive_counts = bank.positive_observation[:, surfels].sum(axis=0).astype(np.float32)
    negative_counts = bank.negative_observation[:, surfels].sum(axis=0).astype(np.float32)
    boundary_safe = bank.source_boundary_distance[surfels] >= float(params.boundary_safe_px)
    temporal = propagated_counts / np.maximum(visible_counts, 1.0)
    positive = target_positive_counts / np.maximum(bank.mask_frame_available.sum(), 1.0)
    if use_temporal:
        positive = 0.5 * positive + 0.5 * temporal
    negative = negative_counts / np.maximum(visible_counts, 1.0)
    boundary_risk = 1.0 - boundary_safe.astype(np.float32)
    rgb = bank.src_rgb[surfels]
    if rgb.shape[0] >= 2:
        center = rgb.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(rgb - center, axis=1)
        appearance = 1.0 / (1.0 + dist)
    else:
        appearance = np.ones((surfels.shape[0],), dtype=np.float32)
    logits = (
        positive
        + float(params.d4rt_temporal_weight) * temporal
        + float(params.appearance_weight) * appearance
        - float(params.boundary_risk_weight) * boundary_risk
    )
    if use_negative:
        logits = logits - float(params.visible_outside_negative_weight) * negative
    posterior = 1.0 / (1.0 + np.exp(-4.0 * (logits - 0.5)))
    reject_local = negative >= float(params.reject_negative_threshold) if use_negative else np.zeros_like(negative, dtype=bool)
    core_local = (posterior >= float(params.core_posterior_threshold)) & ~reject_local
    fringe_local = (
        (posterior >= float(params.fringe_posterior_threshold))
        & (posterior < float(params.core_posterior_threshold))
        & ~reject_local
    )
    unknown_local = ~(core_local | fringe_local | reject_local)
    return {
        "core": surfels[core_local],
        "fringe": surfels[fringe_local],
        "unknown": surfels[unknown_local],
        "reject": surfels[reject_local],
        "positive_score": float(np.mean(positive)) if positive.size else 0.0,
        "negative_ratio": float(np.mean(negative)) if negative.size else 0.0,
        "boundary_risk": float(np.mean(boundary_risk)) if boundary_risk.size else 0.0,
        "temporal_consistency": float(np.mean(temporal)) if temporal.size else 0.0,
        "appearance_consistency": float(np.mean(appearance)) if appearance.size else 0.0,
        "posterior_mean": float(np.mean(posterior)) if posterior.size else 0.0,
    }


def measurement_votes(
    bank: MeasurementBank,
    core_surfels: np.ndarray,
    params: ExplanationParams,
    *,
    include_temporal_targets: bool,
) -> list[tuple[int, int, float]]:
    if core_surfels.size == 0:
        return []
    observations: list[tuple[int, int, float]] = []
    min_count = max(int(params.measurement_min_surfels), int(np.ceil(float(params.measurement_min_core_ratio) * core_surfels.size)))
    for local_idx, frame_id in enumerate(bank.frame_ids.tolist()):
        if not bool(bank.mask_frame_available[local_idx]):
            continue
        ids = bank.target_mask_id[local_idx, core_surfels]
        ids = ids[ids > 0]
        if ids.size == 0:
            continue
        counts = Counter(int(v) for v in ids.tolist())
        for mask_id, count in counts.most_common(int(params.max_slots_per_frame_mask)):
            if count >= min_count:
                observations.append((int(frame_id), int(mask_id), float(count)))
    if include_temporal_targets:
        src_counts = Counter(
            (int(frame), int(mask))
            for frame, mask in zip(bank.src_frame_global[core_surfels].tolist(), bank.src_mask_id[core_surfels].tolist())
            if int(mask) > 0
        )
        for (frame_id, mask_id), count in src_counts.most_common(int(params.max_slots_per_frame_mask)):
            observations.append((int(frame_id), int(mask_id), float(count)))
    dedup: dict[tuple[int, int], float] = {}
    for frame_id, mask_id, score in observations:
        key = (int(frame_id), int(mask_id))
        dedup[key] = max(float(score), dedup.get(key, 0.0))
    return [(frame, mask, score) for (frame, mask), score in sorted(dedup.items())]
