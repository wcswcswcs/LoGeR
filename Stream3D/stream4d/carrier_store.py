from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CarrierBatch:
    carrier_id: np.ndarray
    src_frame: np.ndarray
    src_uv: np.ndarray
    xyz_ref: np.ndarray
    uv_pred: np.ndarray
    visibility_prob: np.ndarray
    confidence_prob: np.ndarray
    valid: np.ndarray
    xyz_local: np.ndarray | None = None
    src_frame_global: np.ndarray | None = None
    src_xy: np.ndarray | None = None
    src_mask_id: np.ndarray | None = None
    persistent_tube_id: np.ndarray | None = None
    parent_tube_id: np.ndarray | None = None
    warmstart_source_chunk: np.ndarray | None = None
    warmstart_source_frame: np.ndarray | None = None
    is_warmstarted: np.ndarray | None = None

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "carrier_id": self.carrier_id,
            "src_frame": self.src_frame,
            "src_uv": self.src_uv,
            "xyz_ref": self.xyz_ref,
            "uv_pred": self.uv_pred,
            "visibility_prob": self.visibility_prob,
            "confidence_prob": self.confidence_prob,
            "valid": self.valid,
        }
        if self.xyz_local is not None:
            payload["xyz_local"] = self.xyz_local
        if self.src_frame_global is not None:
            payload["src_frame_global"] = self.src_frame_global
        if self.src_xy is not None:
            payload["src_xy"] = self.src_xy
        if self.src_mask_id is not None:
            payload["src_mask_id"] = self.src_mask_id
        if self.persistent_tube_id is not None:
            payload["persistent_tube_id"] = self.persistent_tube_id
        if self.parent_tube_id is not None:
            payload["parent_tube_id"] = self.parent_tube_id
        if self.warmstart_source_chunk is not None:
            payload["warmstart_source_chunk"] = self.warmstart_source_chunk
        if self.warmstart_source_frame is not None:
            payload["warmstart_source_frame"] = self.warmstart_source_frame
        if self.is_warmstarted is not None:
            payload["is_warmstarted"] = self.is_warmstarted
        np.savez_compressed(path, **payload)


@dataclass
class CarrierSources:
    carrier_id: np.ndarray
    src_frame: np.ndarray
    src_frame_global: np.ndarray
    src_xy: np.ndarray
    src_uv: np.ndarray
    src_mask_id: np.ndarray

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            carrier_id=self.carrier_id,
            src_frame=self.src_frame,
            src_frame_global=self.src_frame_global,
            src_xy=self.src_xy,
            src_uv=self.src_uv,
            src_mask_id=self.src_mask_id,
        )
