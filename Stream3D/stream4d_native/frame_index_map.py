from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FrameIndexRecord:
    raw_frame_id: int
    dense_rgb_rank: int
    d4rt_clip_local_index: int
    mask_observation_rank: int | None = None


def _as_int_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _delta_histogram(values: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for a, b in zip(values, values[1:]):
        delta = int(b) - int(a)
        hist[str(delta)] = hist.get(str(delta), 0) + 1
    return hist


def _is_contiguous(values: list[int]) -> bool:
    return len(values) <= 1 or all((b - a) == 1 for a, b in zip(values, values[1:]))


class FrameIndexMap:
    """Map raw RGB ids, dense D4RT ranks, and sparse mask observation ranks."""

    def __init__(self, records: list[FrameIndexRecord]) -> None:
        self.records = sorted(records, key=lambda r: r.dense_rgb_rank)
        raw_ids = [r.raw_frame_id for r in self.records]
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError("raw_frame_id values must be unique")
        dense_ranks = [r.dense_rgb_rank for r in self.records]
        if dense_ranks != list(range(len(dense_ranks))):
            raise ValueError("dense_rgb_rank must be contiguous from zero")
        d4rt_local = [r.d4rt_clip_local_index for r in self.records]
        if d4rt_local != list(range(len(d4rt_local))):
            raise ValueError("d4rt_clip_local_index must be contiguous from zero")
        self._by_raw = {r.raw_frame_id: r for r in self.records}
        self._mask_records = [r for r in self.records if r.mask_observation_rank is not None]
        mask_ranks = [int(r.mask_observation_rank) for r in self._mask_records]
        if mask_ranks != list(range(len(mask_ranks))):
            raise ValueError("mask_observation_rank must be contiguous from zero over mask frames")

    @classmethod
    def from_frame_ids(cls, rgb_frame_ids: Iterable[int], mask_frame_ids: Iterable[int]) -> "FrameIndexMap":
        rgb_ids = sorted(_as_int_list(rgb_frame_ids))
        mask_ids = sorted(_as_int_list(mask_frame_ids))
        rgb_set = set(rgb_ids)
        missing = sorted(set(mask_ids) - rgb_set)
        if missing:
            raise ValueError(f"mask_frame_ids missing from rgb_frame_ids: {missing[:10]}")
        mask_rank_by_raw = {raw_id: rank for rank, raw_id in enumerate(mask_ids)}
        records = [
            FrameIndexRecord(
                raw_frame_id=raw_id,
                dense_rgb_rank=rank,
                d4rt_clip_local_index=rank,
                mask_observation_rank=mask_rank_by_raw.get(raw_id),
            )
            for rank, raw_id in enumerate(rgb_ids)
        ]
        return cls(records)

    @property
    def rgb_frame_ids(self) -> list[int]:
        return [r.raw_frame_id for r in self.records]

    @property
    def mask_frame_ids(self) -> list[int]:
        return [r.raw_frame_id for r in self._mask_records]

    def temporal_rank_delta(self, raw_a: int, raw_b: int, *, prefer_mask_rank: bool = True) -> int:
        rec_a = self._by_raw[int(raw_a)]
        rec_b = self._by_raw[int(raw_b)]
        if prefer_mask_rank and rec_a.mask_observation_rank is not None and rec_b.mask_observation_rank is not None:
            return int(rec_b.mask_observation_rank) - int(rec_a.mask_observation_rank)
        return int(rec_b.dense_rgb_rank) - int(rec_a.dense_rgb_rank)

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "raw_frame_id": r.raw_frame_id,
                "dense_rgb_rank": r.dense_rgb_rank,
                "d4rt_clip_local_index": r.d4rt_clip_local_index,
                "mask_observation_rank": "" if r.mask_observation_rank is None else int(r.mask_observation_rank),
            }
            for r in self.records
        ]

    def audit_summary(self, *, first_n: int = 20) -> dict[str, Any]:
        rgb_ids = self.rgb_frame_ids
        mask_ids = self.mask_frame_ids
        mask_ranks = [int(r.mask_observation_rank) for r in self._mask_records if r.mask_observation_rank is not None]
        raw_rgb_deltas = _delta_histogram(rgb_ids)
        mask_deltas = _delta_histogram(mask_ids)
        rank_deltas = _delta_histogram(mask_ranks)
        return {
            "first_20_d4rt_rgb_frame_ids": rgb_ids[:first_n],
            "first_20_mask_frame_ids": mask_ids[:first_n],
            "raw_rgb_delta_distribution": raw_rgb_deltas,
            "mask_delta_distribution": mask_deltas,
            "rank_delta_distribution": rank_deltas,
            "d4rt_encoder_stride": 1 if _is_contiguous(rgb_ids) else "non_contiguous",
            "mask_observation_stride": None if len(mask_deltas) == 0 else sorted(int(k) for k in mask_deltas),
            "uses_contiguous_rgb_for_d4rt": _is_contiguous(rgb_ids),
            "uses_sparse_masks_as_measurements": len(mask_ids) <= len(rgb_ids) and set(mask_ids).issubset(set(rgb_ids)),
            "temporal_curriculum_uses_rank_delta": True,
        }


def build_frame_index_audit(rgb_frame_ids: Iterable[int], mask_frame_ids: Iterable[int]) -> tuple[FrameIndexMap, dict[str, Any]]:
    fmap = FrameIndexMap.from_frame_ids(rgb_frame_ids, mask_frame_ids)
    return fmap, fmap.audit_summary()

