from __future__ import annotations

import numpy as np

from .carrier_store import CarrierSources


def _stable_carrier_id(frame_id: int, x: np.ndarray, y: np.ndarray, width: int) -> np.ndarray:
    return (
        np.int64(frame_id) * np.int64(10_000_000_000)
        + y.astype(np.int64) * np.int64(max(width, 1))
        + x.astype(np.int64)
    )


class CarrierSampler:
    def __init__(
        self,
        max_points_per_mask: int = 32,
        min_points_per_mask: int = 4,
        strategy: str = "uniform_mask_pixels",
        seed: int = 13,
        min_mask_area: int = 8,
    ) -> None:
        self.max_points_per_mask = int(max_points_per_mask)
        self.min_points_per_mask = int(min_points_per_mask)
        self.strategy = strategy
        self.seed = int(seed)
        self.min_mask_area = int(min_mask_area)

    def _sample_indices(self, ys: np.ndarray, xs: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        if self.strategy == "grid_inside_mask":
            order = np.lexsort((xs, ys))
            if order.shape[0] <= count:
                return order
            keep = np.linspace(0, order.shape[0] - 1, num=count, dtype=np.int64)
            return order[keep]
        if self.strategy != "uniform_mask_pixels":
            raise ValueError(f"Unsupported carrier sampling strategy: {self.strategy}")
        if ys.shape[0] <= count:
            return np.arange(ys.shape[0], dtype=np.int64)
        return np.sort(rng.choice(ys.shape[0], size=count, replace=False))

    def sample(self, masks: np.ndarray, frame_ids: list[int]) -> CarrierSources:
        if masks.ndim != 3:
            raise ValueError(f"masks must be [T,H,W], got {masks.shape}")
        carrier_ids: list[np.ndarray] = []
        src_frames: list[np.ndarray] = []
        src_globals: list[np.ndarray] = []
        src_xys: list[np.ndarray] = []
        src_uvs: list[np.ndarray] = []
        src_masks: list[np.ndarray] = []

        _, height, width = masks.shape
        for local_idx, frame_id in enumerate(frame_ids):
            mask = masks[local_idx]
            ids = np.unique(mask)
            ids = ids[ids > 0]
            for mask_id in ids:
                ys, xs = np.where(mask == mask_id)
                area = int(ys.shape[0])
                if area < self.min_mask_area:
                    continue
                sample_count = min(self.max_points_per_mask, max(self.min_points_per_mask, min(area, self.max_points_per_mask)))
                seed = self.seed + int(frame_id) * 1009 + int(mask_id) * 9176
                rng = np.random.default_rng(seed)
                keep = self._sample_indices(ys, xs, sample_count, rng)
                xs_keep = xs[keep].astype(np.int64)
                ys_keep = ys[keep].astype(np.int64)
                actual_count = int(keep.shape[0])
                ids_keep = _stable_carrier_id(frame_id, xs_keep, ys_keep, width)
                uv = np.stack(
                    [
                        xs_keep.astype(np.float32) / float(max(width - 1, 1)),
                        ys_keep.astype(np.float32) / float(max(height - 1, 1)),
                    ],
                    axis=1,
                )
                carrier_ids.append(ids_keep)
                src_frames.append(np.full(actual_count, local_idx, dtype=np.int64))
                src_globals.append(np.full(actual_count, int(frame_id), dtype=np.int64))
                src_xys.append(np.stack([xs_keep, ys_keep], axis=1))
                src_uvs.append(uv.astype(np.float32))
                src_masks.append(np.full(actual_count, int(mask_id), dtype=np.int64))

        if not carrier_ids:
            return CarrierSources(
                carrier_id=np.empty((0,), dtype=np.int64),
                src_frame=np.empty((0,), dtype=np.int64),
                src_frame_global=np.empty((0,), dtype=np.int64),
                src_xy=np.empty((0, 2), dtype=np.int64),
                src_uv=np.empty((0, 2), dtype=np.float32),
                src_mask_id=np.empty((0,), dtype=np.int64),
            )
        return CarrierSources(
            carrier_id=np.concatenate(carrier_ids, axis=0),
            src_frame=np.concatenate(src_frames, axis=0),
            src_frame_global=np.concatenate(src_globals, axis=0),
            src_xy=np.concatenate(src_xys, axis=0),
            src_uv=np.concatenate(src_uvs, axis=0),
            src_mask_id=np.concatenate(src_masks, axis=0),
        )
