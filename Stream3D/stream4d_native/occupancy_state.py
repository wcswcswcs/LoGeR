from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OccupancyCoverageTargets:
    min_visibility: float = 0.5
    min_confidence: float = 0.5
    mark_radius_px: int = 2
    pixel_coverage_target: float = 0.20
    mask_interior_coverage_target: float = 0.20
    boundary_coverage_target: float = 0.10


class SpatioTemporalOccupancyState:
    def __init__(
        self,
        *,
        num_frames: int,
        image_height: int,
        image_width: int,
        masks: np.ndarray | None = None,
    ) -> None:
        self.num_frames = int(num_frames)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.visited = np.zeros((self.num_frames, self.image_height, self.image_width), dtype=bool)
        self.tube_id = np.full((self.num_frames, self.image_height, self.image_width), -1, dtype=np.int64)
        self.masks = None if masks is None else np.asarray(masks)
        if self.masks is not None and self.masks.shape[:3] != self.visited.shape:
            raise ValueError(f"masks shape {self.masks.shape} does not match occupancy grid {self.visited.shape}")
        self.mask_interior = None
        self.mask_boundary = None
        if self.masks is not None:
            self.mask_interior, self.mask_boundary = self._split_mask_interior_boundary(self.masks)
        self.round_index = 0
        self.source_queries = 0
        self.output_tubes = 0

    @property
    def naive_source_query_count(self) -> int:
        return int(self.num_frames * self.image_height * self.image_width)

    def _xy_from_uv(self, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        uv = np.asarray(uv_norm, dtype=np.float32)
        x = np.rint(uv[:, 0] * float(max(self.image_width - 1, 1))).astype(np.int64)
        y = np.rint(uv[:, 1] * float(max(self.image_height - 1, 1))).astype(np.int64)
        ok = (x >= 0) & (x < self.image_width) & (y >= 0) & (y < self.image_height)
        return x, y, ok

    @staticmethod
    def _split_mask_interior_boundary(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        masks = np.asarray(masks)
        foreground = masks > 0
        boundary = np.zeros_like(foreground, dtype=bool)
        if masks.size == 0:
            return foreground, boundary
        boundary[:, 1:, :] |= foreground[:, 1:, :] & (masks[:, 1:, :] != masks[:, :-1, :])
        boundary[:, :-1, :] |= foreground[:, :-1, :] & (masks[:, :-1, :] != masks[:, 1:, :])
        boundary[:, :, 1:] |= foreground[:, :, 1:] & (masks[:, :, 1:] != masks[:, :, :-1])
        boundary[:, :, :-1] |= foreground[:, :, :-1] & (masks[:, :, :-1] != masks[:, :, 1:])
        interior = foreground & ~boundary
        return interior, boundary

    def mark_visible_track_as_visited(self, *, track: dict[str, Any], tube_id: int, mark_radius_px: int = 2) -> None:
        uv = np.asarray(track["uv_norm"], dtype=np.float32).reshape(self.num_frames, 2)
        visibility = np.asarray(track.get("visibility", np.ones((self.num_frames,), dtype=np.float32)), dtype=np.float32)
        confidence = np.asarray(track.get("confidence", np.ones((self.num_frames,), dtype=np.float32)), dtype=np.float32)
        visible = np.asarray(track.get("valid", np.ones((self.num_frames,), dtype=bool)), dtype=bool) & (visibility > 0.0) & (confidence > 0.0)
        x, y, in_bounds = self._xy_from_uv(uv)
        radius = max(0, int(mark_radius_px))
        for t in range(self.num_frames):
            if not (visible[t] and in_bounds[t]):
                continue
            x0 = max(0, int(x[t]) - radius)
            x1 = min(self.image_width, int(x[t]) + radius + 1)
            y0 = max(0, int(y[t]) - radius)
            y1 = min(self.image_height, int(y[t]) + radius + 1)
            self.visited[t, y0:y1, x0:x1] = True
            self.tube_id[t, y0:y1, x0:x1] = int(tube_id)
        self.output_tubes = max(self.output_tubes, int(tube_id) + 1)

    def _priority_candidates(self, priority: str) -> np.ndarray:
        unvisited = ~self.visited
        if priority == "large_mask_interior_uncovered" and self.mask_interior is not None:
            return np.argwhere(unvisited & self.mask_interior)
        if priority == "mask_boundary_uncovered" and self.mask_boundary is not None:
            return np.argwhere(unvisited & self.mask_boundary)
        if priority == "uniform_unvisited":
            return np.argwhere(unvisited)
        return np.empty((0, 3), dtype=np.int64)

    @staticmethod
    def _take_strided(candidates: np.ndarray, count: int) -> np.ndarray:
        if candidates.shape[0] <= count:
            return candidates
        keep = np.linspace(0, candidates.shape[0] - 1, num=count, dtype=np.int64)
        return candidates[keep]

    def sample_unvisited_source_points(self, *, batch_size: int, priority_order: list[str] | None = None) -> np.ndarray:
        batch_size = max(0, int(batch_size))
        if batch_size == 0:
            return np.empty((0, 3), dtype=np.float32)
        priorities = priority_order or ["uniform_unvisited"]
        selected: list[np.ndarray] = []
        selected_keys: set[tuple[int, int, int]] = set()
        candidate_by_priority: list[tuple[str, np.ndarray]] = []
        for priority in priorities:
            if priority == "uniform_unvisited":
                continue
            candidates = self._priority_candidates(priority)
            if candidates.size > 0:
                candidate_by_priority.append((priority, candidates))
        remaining = batch_size
        active_count = len(candidate_by_priority)
        base_quota = max(1, int(np.ceil(batch_size / active_count))) if active_count else batch_size
        for priority, candidates in candidate_by_priority:
            if remaining <= 0:
                break
            if selected_keys:
                keep = [tuple(int(v) for v in row) not in selected_keys for row in candidates]
                candidates = candidates[np.asarray(keep, dtype=bool)]
            if candidates.size == 0:
                continue
            quota = min(remaining, base_quota)
            picked = self._take_strided(candidates, quota)
            selected.append(picked)
            for row in picked:
                selected_keys.add(tuple(int(v) for v in row))
            remaining -= int(picked.shape[0])
        if remaining > 0 and "uniform_unvisited" in priorities:
            candidates = np.argwhere(~self.visited)
            if selected_keys:
                keep = [tuple(int(v) for v in row) not in selected_keys for row in candidates]
                candidates = candidates[np.asarray(keep, dtype=bool)]
            if candidates.size > 0:
                selected.append(self._take_strided(candidates, remaining))
        if not selected:
            return np.empty((0, 3), dtype=np.float32)
        candidates = np.concatenate(selected, axis=0)
        t = candidates[:, 0].astype(np.float32)
        y = candidates[:, 1].astype(np.float32)
        x = candidates[:, 2].astype(np.float32)
        uv = np.stack(
            [
                x / float(max(self.image_width - 1, 1)),
                y / float(max(self.image_height - 1, 1)),
            ],
            axis=1,
        )
        self.source_queries += int(candidates.shape[0])
        return np.concatenate([t[:, None], uv.astype(np.float32)], axis=1)

    def coverage_satisfied(self, targets: OccupancyCoverageTargets) -> bool:
        if self.masks is not None:
            interior = self.mask_interior_coverage_mean()
            boundary = self.mask_boundary_coverage_mean()
            interior_ok = interior is not None and interior >= float(targets.mask_interior_coverage_target)
            boundary_ok = boundary is not None and boundary >= float(targets.boundary_coverage_target)
            return bool(interior_ok and boundary_ok)
        return self.pixel_occupancy_coverage_mean() >= float(targets.pixel_coverage_target)

    def pixel_occupancy_coverage_mean(self) -> float:
        return float(np.mean(self.visited)) if self.visited.size else 0.0

    def pixel_occupancy_coverage_p10(self) -> float:
        if self.visited.size == 0:
            return 0.0
        per_frame = self.visited.reshape(self.num_frames, -1).mean(axis=1)
        return float(np.percentile(per_frame, 10))

    def _scoped_coverage_per_frame(self, scope: np.ndarray | None) -> np.ndarray | None:
        if scope is None:
            return None
        values: list[float] = []
        for t in range(self.num_frames):
            denom = int(np.count_nonzero(scope[t]))
            if denom == 0:
                continue
            values.append(float(np.count_nonzero(self.visited[t] & scope[t]) / denom))
        if not values:
            return None
        return np.asarray(values, dtype=np.float32)

    def mask_interior_coverage_mean(self) -> float | None:
        values = self._scoped_coverage_per_frame(self.mask_interior)
        return None if values is None else float(np.mean(values))

    def mask_interior_coverage_p10(self) -> float | None:
        values = self._scoped_coverage_per_frame(self.mask_interior)
        return None if values is None else float(np.percentile(values, 10))

    def mask_boundary_coverage_mean(self) -> float | None:
        values = self._scoped_coverage_per_frame(self.mask_boundary)
        return None if values is None else float(np.mean(values))

    def mask_boundary_coverage_p10(self) -> float | None:
        values = self._scoped_coverage_per_frame(self.mask_boundary)
        return None if values is None else float(np.percentile(values, 10))

    def _undercovered_mask_count(self, scope: np.ndarray | None, target: float) -> int | None:
        if self.masks is None or scope is None:
            return None
        count = 0
        for t in range(self.num_frames):
            ids = np.unique(self.masks[t][scope[t]])
            ids = ids[ids > 0]
            for mask_id in ids:
                cur = scope[t] & (self.masks[t] == mask_id)
                denom = int(np.count_nonzero(cur))
                if denom == 0:
                    continue
                cov = float(np.count_nonzero(self.visited[t] & cur) / denom)
                if cov < float(target):
                    count += 1
        return int(count)

    def duplicate_track_rate(self) -> float:
        ids = self.tube_id[self.tube_id >= 0]
        if ids.size == 0:
            return 0.0
        unique = np.unique(ids)
        return float(1.0 - unique.shape[0] / max(ids.shape[0], 1))

    def summarize(self, *, query_budget_hit: bool = False, total_time_sec: float | None = None) -> dict[str, Any]:
        actual = max(int(self.source_queries), 1)
        return {
            "uses_spatiotemporal_occupancy": True,
            "naive_source_query_count": int(self.naive_source_query_count),
            "actual_source_query_count": int(self.source_queries),
            "adaptive_speedup_vs_naive": float(self.naive_source_query_count / actual),
            "num_output_tubes": int(self.output_tubes),
            "pixel_occupancy_coverage_mean": self.pixel_occupancy_coverage_mean(),
            "pixel_occupancy_coverage_p10": self.pixel_occupancy_coverage_p10(),
            "mask_interior_coverage_mean": self.mask_interior_coverage_mean(),
            "mask_interior_coverage_p10": self.mask_interior_coverage_p10(),
            "mask_boundary_coverage_mean": self.mask_boundary_coverage_mean(),
            "mask_boundary_coverage_p10": self.mask_boundary_coverage_p10(),
            "overlap_anchor_coverage": None,
            "unvisited_large_mask_count": self._undercovered_mask_count(
                self.mask_interior,
                target=OccupancyCoverageTargets().mask_interior_coverage_target,
            ),
            "unvisited_boundary_count": self._undercovered_mask_count(
                self.mask_boundary,
                target=OccupancyCoverageTargets().boundary_coverage_target,
            ),
            "duplicate_track_rate": self.duplicate_track_rate(),
            "redundant_query_rate": 0.0,
            "coverage_saturation_round": int(self.round_index),
            "query_budget_hit": bool(query_budget_hit),
            "total_d4rt_time_sec": total_time_sec,
        }
