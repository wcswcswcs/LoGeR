from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .carrier_store import CarrierBatch
from .local_4d_filter import LocalProposal
from .mask_evidence import MaskObservation


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_overlay(
    path: str | Path,
    rgb: np.ndarray,
    uv_norm: np.ndarray,
    color: tuple[int, int, int] = (255, 32, 32),
    max_points: int = 2000,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = rgb.copy()
    if image.ndim != 3:
        return
    h, w = image.shape[:2]
    pts = np.asarray(uv_norm, dtype=np.float32).reshape(-1, 2)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 0] >= 0) & (pts[:, 0] <= 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= 1)
    pts = pts[valid]
    if pts.shape[0] > max_points:
        keep = np.linspace(0, pts.shape[0] - 1, num=max_points, dtype=np.int64)
        pts = pts[keep]
    for u, v in pts:
        x = int(round(float(u) * float(max(w - 1, 1))))
        y = int(round(float(v) * float(max(h - 1, 1))))
        cv2.circle(image, (x, y), 2, color, -1)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def local_props_payload(
    observations: list[MaskObservation],
    proposals: list[LocalProposal],
    diagnostics: dict[str, float],
) -> dict:
    return {
        "diagnostics": diagnostics,
        "proposals": [
            {
                "proposal_id": int(prop.proposal_id),
                "num_observations": int(len(prop.observation_indices)),
                "num_carriers": int(len(prop.carrier_ids)),
                "frames": sorted(int(v) for v in prop.frame_support.keys()),
                "mask_observations": [
                    {"frame_id": int(f), "mask_id": int(m), "coverage": float(c)}
                    for f, m, c in prop.mask_observations[:20]
                ],
            }
            for prop in proposals
        ],
        "num_observations": int(len(observations)),
    }


def carrier_diagnostics(batch: CarrierBatch) -> dict[str, float]:
    uv = batch.uv_pred
    uv_finite = np.isfinite(uv).all(axis=-1)
    uv_in01 = uv_finite & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    rho = batch.visibility_prob * batch.confidence_prob
    return {
        "num_carriers": float(batch.carrier_id.shape[0]),
        "num_target_frames": float(batch.valid.shape[0]),
        "uv_min": float(np.nanmin(uv)) if uv.size else float("nan"),
        "uv_max": float(np.nanmax(uv)) if uv.size else float("nan"),
        "uv_in01_rate": float(np.mean(uv_in01)) if uv.size else 0.0,
        "visibility_prob_mean": float(np.nanmean(batch.visibility_prob)) if batch.visibility_prob.size else 0.0,
        "confidence_prob_mean": float(np.nanmean(batch.confidence_prob)) if batch.confidence_prob.size else 0.0,
        "rho_mean": float(np.nanmean(rho)) if rho.size else 0.0,
    }
