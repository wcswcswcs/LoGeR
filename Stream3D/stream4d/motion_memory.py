from __future__ import annotations

from math import exp

import numpy as np


def centroid_similarity(
    object_centroid: np.ndarray | None,
    proposal_centroid: np.ndarray | None,
    sigma: float = 0.35,
) -> float:
    if object_centroid is None or proposal_centroid is None:
        return 0.0
    a = np.asarray(object_centroid, dtype=np.float32).reshape(-1)
    b = np.asarray(proposal_centroid, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    dist = float(np.linalg.norm(a - b))
    return float(exp(-dist / max(float(sigma), 1e-6)))


def velocity_continuity_score(
    centroid_history: list[tuple[int, list[float]]] | list[tuple[int, tuple[float, ...]]],
    proposal_centroid: np.ndarray | None,
    window_index: int,
    sigma: float = 0.35,
) -> float:
    if proposal_centroid is None or len(centroid_history) < 2:
        return 0.0
    try:
        t0, c0_raw = centroid_history[-2]
        t1, c1_raw = centroid_history[-1]
    except (TypeError, ValueError):
        return 0.0
    c0 = np.asarray(c0_raw, dtype=np.float32)
    c1 = np.asarray(c1_raw, dtype=np.float32)
    if c0.shape != c1.shape:
        return 0.0
    dt = max(int(t1) - int(t0), 1)
    predicted = c1 + (c1 - c0) * (max(int(window_index) - int(t1), 1) / float(dt))
    return centroid_similarity(predicted, proposal_centroid, sigma=sigma)
