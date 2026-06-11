from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .carrier_store import CarrierBatch


@dataclass
class MaskObservation:
    frame_id: int
    frame_local: int
    mask_id: int
    carrier_ids: np.ndarray
    weights: np.ndarray
    uv_norm: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area: int

    def carrier_set(self) -> set[int]:
        return set(int(v) for v in self.carrier_ids.tolist())


class MaskEvidenceBuilder:
    def __init__(self, rho_min: float = 0.35) -> None:
        self.rho_min = float(rho_min)

    @staticmethod
    def _bbox(mask: np.ndarray, mask_id: int) -> tuple[int, int, int, int, int]:
        ys, xs = np.where(mask == mask_id)
        if ys.size == 0:
            return (0, 0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(ys.size))

    def build(
        self,
        carrier_batch: CarrierBatch,
        masks: np.ndarray,
        frame_ids: list[int],
    ) -> tuple[list[MaskObservation], dict[str, float]]:
        if masks.ndim != 3:
            raise ValueError(f"masks must be [T,H,W], got {masks.shape}")
        target_count, carrier_count = carrier_batch.valid.shape
        if target_count != masks.shape[0]:
            raise ValueError("Carrier target frame count must equal masks frame count")

        height, width = int(masks.shape[1]), int(masks.shape[2])
        rho = carrier_batch.visibility_prob * carrier_batch.confidence_prob
        observations: list[MaskObservation] = []
        total_valid = 0
        total_assigned = 0
        mask_carrier_counts: list[int] = []

        for local_idx in range(target_count):
            uv = carrier_batch.uv_pred[local_idx]
            valid = (
                carrier_batch.valid[local_idx]
                & np.isfinite(uv).all(axis=1)
                & (rho[local_idx] >= self.rho_min)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
            )
            total_valid += int(valid.sum())
            if not np.any(valid):
                continue
            xs = np.rint(uv[:, 0] * float(max(width - 1, 1))).astype(np.int64)
            ys = np.rint(uv[:, 1] * float(max(height - 1, 1))).astype(np.int64)
            valid_idx = np.flatnonzero(valid)
            sampled_ids = masks[local_idx, ys[valid_idx], xs[valid_idx]]
            positive = sampled_ids > 0
            total_assigned += int(positive.sum())
            if not np.any(positive):
                continue

            valid_idx = valid_idx[positive]
            sampled_ids = sampled_ids[positive]
            for mask_id in np.unique(sampled_ids):
                obs_idx = valid_idx[sampled_ids == mask_id]
                bbox = self._bbox(masks[local_idx], int(mask_id))
                observations.append(
                    MaskObservation(
                        frame_id=int(frame_ids[local_idx]),
                        frame_local=int(local_idx),
                        mask_id=int(mask_id),
                        carrier_ids=carrier_batch.carrier_id[obs_idx].astype(np.int64),
                        weights=rho[local_idx, obs_idx].astype(np.float32),
                        uv_norm=uv[obs_idx].astype(np.float32),
                        bbox_xyxy=bbox[:4],
                        area=bbox[4],
                    )
                )
                mask_carrier_counts.append(int(obs_idx.shape[0]))

        raw_mask_count = int(sum(max(0, np.unique(masks[i]).shape[0] - (1 if 0 in np.unique(masks[i]) else 0)) for i in range(masks.shape[0])))
        diagnostics = {
            "num_raw_mask_observations": float(raw_mask_count),
            "num_mask_observations_with_carriers": float(len(observations)),
            "carrier_visibility_rate": float(total_valid / max(target_count * carrier_count, 1)),
            "carrier_assignment_rate": float(total_assigned / max(total_valid, 1)),
            "mean_mask_carrier_count": float(np.mean(mask_carrier_counts)) if mask_carrier_counts else 0.0,
        }
        return observations, diagnostics
