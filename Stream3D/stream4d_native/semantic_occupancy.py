from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaterialQuery:
    frame_rank: int
    y: int
    x: int
    reason: str
    score: float


def _split_interior_boundary(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(masks)
    foreground = labels > 0
    boundary = np.zeros_like(foreground, dtype=bool)
    boundary[:, 1:, :] |= foreground[:, 1:, :] & (labels[:, 1:, :] != labels[:, :-1, :])
    boundary[:, :-1, :] |= foreground[:, :-1, :] & (labels[:, :-1, :] != labels[:, 1:, :])
    boundary[:, :, 1:] |= foreground[:, :, 1:] & (labels[:, :, 1:] != labels[:, :, :-1])
    boundary[:, :, :-1] |= foreground[:, :, :-1] & (labels[:, :, :-1] != labels[:, :, 1:])
    interior = foreground & ~boundary
    return interior, boundary


def _take_even(candidates: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or candidates.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if candidates.shape[0] <= count:
        return candidates
    idx = np.linspace(0, candidates.shape[0] - 1, num=count, dtype=np.int64)
    return candidates[idx]


class SemanticOccupancyState:
    def __init__(
        self,
        masks: np.ndarray,
        *,
        overlap_frame_ranks: list[int] | None = None,
        disagreement: np.ndarray | None = None,
    ) -> None:
        self.masks = np.asarray(masks, dtype=np.int32)
        if self.masks.ndim != 3:
            raise ValueError("masks must have shape [T,H,W]")
        self.num_frames, self.height, self.width = self.masks.shape
        self.foreground = self.masks > 0
        self.interior, self.boundary = _split_interior_boundary(self.masks)
        self.disagreement = np.zeros_like(self.foreground, dtype=bool) if disagreement is None else np.asarray(disagreement, dtype=bool)
        if self.disagreement.shape != self.foreground.shape:
            raise ValueError("disagreement shape must match masks")
        self.overlap = np.zeros_like(self.foreground, dtype=bool)
        for frame_rank in overlap_frame_ranks or []:
            if 0 <= int(frame_rank) < self.num_frames:
                self.overlap[int(frame_rank), :, :] = True
        self.exploration = ~self.foreground

    def _region(self, reason: str) -> np.ndarray:
        if reason == "fixed_grid":
            return np.ones_like(self.foreground, dtype=bool)
        if reason == "pixel_occupancy":
            return np.ones_like(self.foreground, dtype=bool)
        if reason == "mask_interior":
            return self.interior
        if reason == "mask_boundary":
            return self.boundary
        if reason == "overlap_anchor":
            return self.overlap
        if reason == "disagreement":
            return self.disagreement
        if reason == "exploration":
            return self.exploration
        raise ValueError(f"unknown occupancy reason: {reason}")

    def _queries_for_reason(self, reason: str, count: int, score: float, used: set[tuple[int, int, int]]) -> list[MaterialQuery]:
        candidates = np.argwhere(self._region(reason))
        if used:
            keep = [tuple(int(v) for v in row) not in used for row in candidates]
            candidates = candidates[np.asarray(keep, dtype=bool)]
        picked = _take_even(candidates, int(count))
        out = []
        for t, y, x in picked:
            key = (int(t), int(y), int(x))
            used.add(key)
            out.append(MaterialQuery(frame_rank=key[0], y=key[1], x=key[2], reason=reason, score=float(score)))
        return out

    def schedule(self, *, variant: str, budget: int) -> list[MaterialQuery]:
        budget = max(0, int(budget))
        quotas_by_variant = {
            "B0": [("fixed_grid", 1.0, 0.20)],
            "B1": [("pixel_occupancy", 1.0, 0.25)],
            "B2": [("mask_interior", 0.80, 0.70), ("fixed_grid", 0.20, 0.10)],
            "B3": [("mask_boundary", 0.75, 0.85), ("mask_interior", 0.25, 0.40)],
            "B4": [("disagreement", 0.70, 0.90), ("mask_boundary", 0.20, 0.60), ("fixed_grid", 0.10, 0.10)],
            "B5": [
                ("mask_boundary", 0.30, 0.90),
                ("overlap_anchor", 0.35, 0.82),
                ("disagreement", 0.15, 0.78),
                ("mask_interior", 0.15, 0.60),
                ("fixed_grid", 0.05, 0.20),
            ],
            "B6": [
                ("mask_boundary", 0.30, 0.90),
                ("overlap_anchor", 0.20, 0.82),
                ("disagreement", 0.15, 0.78),
                ("mask_interior", 0.20, 0.60),
                ("exploration", 0.15, 0.35),
            ],
        }
        if variant not in quotas_by_variant:
            raise ValueError(f"unknown semantic occupancy variant: {variant}")
        used: set[tuple[int, int, int]] = set()
        queries: list[MaterialQuery] = []
        quotas = quotas_by_variant[variant]
        remaining = budget
        for idx, (reason, frac, score) in enumerate(quotas):
            if remaining <= 0:
                break
            count = remaining if idx == len(quotas) - 1 else int(round(budget * float(frac)))
            count = min(max(count, 0), remaining)
            new_queries = self._queries_for_reason(reason, count, score, used)
            queries.extend(new_queries)
            remaining = budget - len(queries)
        if len(queries) < budget and variant != "B6":
            queries.extend(self._queries_for_reason("fixed_grid", budget - len(queries), 0.05, used))
        return queries[:budget]

    def coverage_metrics(self, queries: list[MaterialQuery]) -> dict[str, Any]:
        selected = np.zeros_like(self.foreground, dtype=bool)
        for query in queries:
            selected[int(query.frame_rank), int(query.y), int(query.x)] = True

        def coverage(region: np.ndarray) -> float:
            denom = int(np.count_nonzero(region))
            if denom == 0:
                return 0.0
            return float(np.count_nonzero(selected & region) / denom)

        accepted = [q for q in queries if q.score >= 0.5]
        reason_counts: dict[str, int] = {}
        for query in queries:
            reason_counts[query.reason] = reason_counts.get(query.reason, 0) + 1
        outside_mask = int(np.count_nonzero(selected & self.exploration))
        return {
            "query_count": int(len(queries)),
            "accepted_tube_count": int(len(accepted)),
            "accepted_tube_ratio": float(len(accepted) / max(len(queries), 1)),
            "mask_interior_coverage": coverage(self.interior),
            "mask_boundary_coverage": coverage(self.boundary),
            "overlap_anchor_coverage": coverage(self.overlap),
            "disagreement_coverage": coverage(self.disagreement),
            "queries_per_accepted_tube": float(len(queries) / max(len(accepted), 1)),
            "exploration_outside_mask_ratio": float(outside_mask / max(len(queries), 1)),
            "reason_counts": reason_counts,
        }


def run_semantic_occupancy_variants(
    masks: np.ndarray,
    *,
    budget: int,
    overlap_frame_ranks: list[int] | None = None,
    disagreement: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    state = SemanticOccupancyState(masks, overlap_frame_ranks=overlap_frame_ranks, disagreement=disagreement)
    rows = []
    for variant in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
        queries = state.schedule(variant=variant, budget=budget)
        row = {"variant": variant, **state.coverage_metrics(queries)}
        rows.append(row)
    return rows
