from __future__ import annotations

import numpy as np

from .local_4d_filter import LocalProposal
from .object_memory import Object4D


def finite_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def finite_max(values: list[float]) -> float:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        return 0.0
    return float(np.max(clean))


def mask_ids_by_frame(mask_observations: list[tuple[int, int, float]]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for frame_id, mask_id, _ in mask_observations:
        out.setdefault(int(frame_id), set()).add(int(mask_id))
    return out


def same_frame_conflict_rate(obj: Object4D, proposal: LocalProposal) -> float:
    object_masks = mask_ids_by_frame(obj.mask_observations)
    proposal_masks = mask_ids_by_frame(proposal.mask_observations)
    shared_frames = set(object_masks).intersection(proposal_masks)
    if not shared_frames:
        return 0.0
    conflicts = 0
    for frame_id in shared_frames:
        if object_masks[frame_id].isdisjoint(proposal_masks[frame_id]):
            conflicts += 1
    return float(conflicts / max(len(shared_frames), 1))


def carrier_ioc(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a.intersection(b)) / max(1, min(len(a), len(b))))
