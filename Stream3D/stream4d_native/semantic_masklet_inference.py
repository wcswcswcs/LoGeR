from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaskletMeasurement:
    measurement_id: int
    frame_rank: int
    mask_id: int
    feature: tuple[float, ...]
    d4rt_support_key: str
    diagnostic_object_id: int | None = None


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


class _UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {int(i): int(i) for i in ids}

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def infer_semantic_masklets(
    measurements: list[MaskletMeasurement],
    *,
    use_visual: bool = True,
    use_d4rt: bool = True,
    max_rank_delta: int = 1,
    visual_threshold: float = 0.90,
) -> dict[int, int]:
    uf = _UnionFind([m.measurement_id for m in measurements])
    by_id = {m.measurement_id: m for m in measurements}
    ids = sorted(by_id)
    for i, mid_i in enumerate(ids):
        a = by_id[mid_i]
        for mid_j in ids[i + 1 :]:
            b = by_id[mid_j]
            delta = abs(int(a.frame_rank) - int(b.frame_rank))
            if delta == 0 or delta > int(max_rank_delta):
                continue
            visual_ok = use_visual and _cosine(a.feature, b.feature) >= float(visual_threshold)
            d4rt_ok = use_d4rt and a.d4rt_support_key == b.d4rt_support_key
            mask_id_ok = (not use_visual and not use_d4rt) and int(a.mask_id) == int(b.mask_id)
            if visual_ok or d4rt_ok or mask_id_ok:
                uf.union(a.measurement_id, b.measurement_id)
    root_to_label: dict[int, int] = {}
    out: dict[int, int] = {}
    for mid in ids:
        root = uf.find(mid)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        out[mid] = root_to_label[root]
    return out


def evaluate_masklet_assignments(
    measurements: list[MaskletMeasurement],
    assignments: dict[int, int],
) -> dict[str, Any]:
    true = {m.measurement_id: m.diagnostic_object_id for m in measurements if m.diagnostic_object_id is not None}
    pred_clusters: dict[int, list[int]] = {}
    true_clusters: dict[int, list[int]] = {}
    same_frame_conflicts = 0
    for m in measurements:
        pred = int(assignments[int(m.measurement_id)])
        pred_clusters.setdefault(pred, []).append(int(m.measurement_id))
        if m.diagnostic_object_id is not None:
            true_clusters.setdefault(int(m.diagnostic_object_id), []).append(int(m.measurement_id))
    for mids in pred_clusters.values():
        frames = [next(m.frame_rank for m in measurements if m.measurement_id == mid) for mid in mids]
        if len(frames) != len(set(frames)):
            same_frame_conflicts += 1
    if not true:
        return {
            "masklet_purity": None,
            "masklet_completeness": None,
            "same_frame_conflict_violation": int(same_frame_conflicts),
            "temporal_span_mean": _temporal_span_mean(measurements, assignments),
        }
    n = len(true)
    purity_sum = 0
    for mids in pred_clusters.values():
        counts: dict[int, int] = {}
        for mid in mids:
            if mid in true:
                counts[int(true[mid])] = counts.get(int(true[mid]), 0) + 1
        purity_sum += max(counts.values(), default=0)
    completeness_sum = 0
    for mids in true_clusters.values():
        counts: dict[int, int] = {}
        for mid in mids:
            counts[int(assignments[mid])] = counts.get(int(assignments[mid]), 0) + 1
        completeness_sum += max(counts.values(), default=0)
    return {
        "masklet_purity": float(purity_sum / max(n, 1)),
        "masklet_completeness": float(completeness_sum / max(n, 1)),
        "same_frame_conflict_violation": int(same_frame_conflicts),
        "temporal_span_mean": _temporal_span_mean(measurements, assignments),
    }


def _temporal_span_mean(measurements: list[MaskletMeasurement], assignments: dict[int, int]) -> float:
    frames_by_cluster: dict[int, list[int]] = {}
    for measurement in measurements:
        frames_by_cluster.setdefault(int(assignments[int(measurement.measurement_id)]), []).append(int(measurement.frame_rank))
    spans = [max(frames) - min(frames) + 1 for frames in frames_by_cluster.values() if frames]
    return float(np.mean(spans)) if spans else 0.0

