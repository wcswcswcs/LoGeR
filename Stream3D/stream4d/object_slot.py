from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObjectSlot:
    object_id: int
    birth_frame: int
    birth_mask_id: int
    core_surfels: np.ndarray
    fringe_surfels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    unknown_surfels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    reject_surfels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    mask_observations: list[tuple[int, int, float]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_object_dict_record(self) -> dict[str, Any]:
        return {
            "mask_list": [(int(frame), int(mask), float(score)) for frame, mask, score in self.mask_observations],
            "carrier_ids": self.core_surfels.astype(np.int64, copy=False),
            "v12_slot": {
                "object_id": int(self.object_id),
                "birth_frame": int(self.birth_frame),
                "birth_mask_id": int(self.birth_mask_id),
                "num_core_surfels": int(self.core_surfels.shape[0]),
                "num_fringe_surfels": int(self.fringe_surfels.shape[0]),
                "num_unknown_surfels": int(self.unknown_surfels.shape[0]),
                "num_reject_surfels": int(self.reject_surfels.shape[0]),
                **self.diagnostics,
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            "object_id": int(self.object_id),
            "birth_frame": int(self.birth_frame),
            "birth_mask_id": int(self.birth_mask_id),
            "num_core_surfels": int(self.core_surfels.shape[0]),
            "num_fringe_surfels": int(self.fringe_surfels.shape[0]),
            "num_unknown_surfels": int(self.unknown_surfels.shape[0]),
            "num_reject_surfels": int(self.reject_surfels.shape[0]),
            "num_mask_observations": int(len(self.mask_observations)),
            "mask_observations": [(int(frame), int(mask), float(score)) for frame, mask, score in self.mask_observations],
            **self.diagnostics,
        }
