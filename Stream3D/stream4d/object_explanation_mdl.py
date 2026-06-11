from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .evidence_terms import ExplanationParams, birth_groups, measurement_votes, posterior_for_group
from .measurement_bank import MeasurementBank, json_safe
from .object_slot import ObjectSlot
from .video_masklet import VideoMaskletBank


@dataclass
class MDLParams:
    birth_min_surfels: int = 12
    birth_min_boundary_safe_ratio: float = 0.50
    birth_max_ambiguous_ratio: float = 0.60
    core_posterior_threshold: float = 0.62
    fringe_posterior_threshold: float = 0.40
    reject_negative_threshold: float = 0.45
    visible_outside_negative_weight: float = 1.0
    boundary_risk_weight: float = 0.45
    appearance_weight: float = 0.25
    d4rt_temporal_weight: float = 0.60
    min_core_surfels_per_object: int = 8
    min_export_points_per_object: int = 60
    measurement_min_surfels: int = 3
    measurement_min_core_ratio: float = 0.05
    max_slots_per_frame_mask: int = 4
    boundary_safe_px: float = 3.0
    model_cost: float = 0.80
    overlap_penalty: float = 2.5
    unexplained_penalty: float = 0.30
    measurement_reward: float = 0.65
    surfel_reward: float = 0.35
    negative_cost: float = 1.1
    boundary_cost: float = 0.45
    motion_reward: float = 0.25
    appearance_reward: float = 0.15
    max_core_overlap_ratio: float = 0.20
    max_slots: int = 96

    def to_explanation_params(self) -> ExplanationParams:
        return ExplanationParams(
            birth_min_surfels=int(self.birth_min_surfels),
            birth_min_boundary_safe_ratio=float(self.birth_min_boundary_safe_ratio),
            birth_max_ambiguous_ratio=float(self.birth_max_ambiguous_ratio),
            core_posterior_threshold=float(self.core_posterior_threshold),
            fringe_posterior_threshold=float(self.fringe_posterior_threshold),
            reject_negative_threshold=float(self.reject_negative_threshold),
            visible_outside_negative_weight=float(self.visible_outside_negative_weight),
            boundary_risk_weight=float(self.boundary_risk_weight),
            appearance_weight=float(self.appearance_weight),
            d4rt_temporal_weight=float(self.d4rt_temporal_weight),
            max_slots_per_frame_mask=int(self.max_slots_per_frame_mask),
            min_core_surfels_per_object=int(self.min_core_surfels_per_object),
            min_export_points_per_object=int(self.min_export_points_per_object),
            boundary_safe_px=float(self.boundary_safe_px),
            measurement_min_surfels=int(self.measurement_min_surfels),
            measurement_min_core_ratio=float(self.measurement_min_core_ratio),
        )


def _masklet_votes(
    bank: MeasurementBank,
    masklets: VideoMaskletBank | None,
    birth_frame: int,
    birth_mask_id: int,
) -> list[tuple[int, int, float]]:
    if masklets is None:
        return []
    out: dict[tuple[int, int], float] = {}
    for row_idx in masklets.rows_by_birth().get((int(birth_frame), int(birth_mask_id)), []):
        frame_id = int(masklets.frame_id[row_idx])
        local = np.flatnonzero(bank.frame_ids == frame_id)
        if local.size == 0:
            continue
        frame_idx = int(local[0])
        if not bool(bank.mask_frame_available[frame_idx]):
            continue
        surfels = masklets.surfels_for_row(row_idx)
        ids = bank.target_mask_id[frame_idx, surfels]
        ids = ids[ids > 0]
        if ids.size == 0:
            continue
        counts = Counter(int(v) for v in ids.tolist())
        mask_id, count = counts.most_common(1)[0]
        score = float(count) * float(masklets.confidence[row_idx])
        key = (frame_id, int(mask_id))
        out[key] = max(out.get(key, 0.0), score)
    return [(frame, mask, score) for (frame, mask), score in sorted(out.items())]


def _dedup_votes(votes: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    best: dict[tuple[int, int], float] = {}
    for frame_id, mask_id, score in votes:
        key = (int(frame_id), int(mask_id))
        best[key] = max(best.get(key, 0.0), float(score))
    return [(frame, mask, score) for (frame, mask), score in sorted(best.items())]


def _slot_energy(slot: ObjectSlot, params: MDLParams) -> float:
    core = max(int(slot.core_surfels.shape[0]), 1)
    fringe = int(slot.fringe_surfels.shape[0])
    observations = max(int(len(slot.mask_observations)), 0)
    negative = float(slot.diagnostics.get("negative_ratio", 0.0))
    boundary = float(slot.diagnostics.get("boundary_risk", 0.0))
    temporal = float(slot.diagnostics.get("temporal_consistency", 0.0))
    appearance = float(slot.diagnostics.get("appearance_consistency", 0.0))
    return float(
        params.model_cost
        + params.negative_cost * negative
        + params.boundary_cost * boundary
        + 0.05 * fringe / max(core, 1)
        - params.measurement_reward * np.log1p(observations)
        - params.surfel_reward * np.log1p(core)
        - params.motion_reward * temporal
        - params.appearance_reward * appearance
    )


def _candidate_slots(
    bank: MeasurementBank,
    masklets: VideoMaskletBank | None,
    params: MDLParams,
) -> tuple[list[ObjectSlot], int]:
    explanation_params = params.to_explanation_params()
    groups = birth_groups(bank, explanation_params)
    slots: list[ObjectSlot] = []
    rejected = 0
    for group in groups:
        if not bool(group["passes_birth_gate"]):
            rejected += 1
            continue
        surfels = np.asarray(group["surfel_indices"], dtype=np.int64)
        posterior = posterior_for_group(
            bank,
            surfels,
            explanation_params,
            use_negative=True,
            use_temporal=True,
        )
        core = np.asarray(posterior["core"], dtype=np.int64)
        fringe = np.asarray(posterior["fringe"], dtype=np.int64)
        unknown = np.asarray(posterior["unknown"], dtype=np.int64)
        reject = np.asarray(posterior["reject"], dtype=np.int64)
        if core.shape[0] < int(params.min_core_surfels_per_object):
            rejected += 1
            continue
        votes = measurement_votes(bank, core, explanation_params, include_temporal_targets=True)
        votes.extend(_masklet_votes(bank, masklets, int(group["birth_frame"]), int(group["birth_mask_id"])))
        votes.append((int(group["birth_frame"]), int(group["birth_mask_id"]), float(core.shape[0])))
        votes = _dedup_votes(votes)
        if not votes:
            rejected += 1
            continue
        slot = ObjectSlot(
            object_id=len(slots),
            birth_frame=int(group["birth_frame"]),
            birth_mask_id=int(group["birth_mask_id"]),
            core_surfels=core,
            fringe_surfels=fringe,
            unknown_surfels=unknown,
            reject_surfels=reject,
            mask_observations=votes,
            diagnostics={
                "mode": "mdl",
                "num_birth_surfels": int(group["num_birth_surfels"]),
                "birth_boundary_safe_ratio": float(group["boundary_safe_ratio"]),
                "birth_ambiguous_ratio": float(group["ambiguous_ratio"]),
                "positive_score": float(posterior["positive_score"]),
                "negative_ratio": float(posterior["negative_ratio"]),
                "boundary_risk": float(posterior["boundary_risk"]),
                "temporal_consistency": float(posterior["temporal_consistency"]),
                "appearance_consistency": float(posterior["appearance_consistency"]),
                "posterior_mean": float(posterior["posterior_mean"]),
            },
        )
        slot.diagnostics["mdl_energy"] = _slot_energy(slot, params)
        slots.append(slot)
    slots.sort(key=lambda item: (float(item.diagnostics["mdl_energy"]), -int(item.core_surfels.shape[0])))
    return slots, rejected


def _surfel_overlap_ratio(candidate: np.ndarray, owned: set[int]) -> float:
    if candidate.size == 0 or not owned:
        return 0.0
    return float(sum(1 for v in candidate.tolist() if int(v) in owned) / max(int(candidate.size), 1))


def _apply_measurement_wta(slots: list[ObjectSlot]) -> int:
    winners: dict[tuple[int, int], tuple[int, float]] = {}
    for slot_idx, slot in enumerate(slots):
        for frame_id, mask_id, score in slot.mask_observations:
            key = (int(frame_id), int(mask_id))
            current = winners.get(key)
            if current is None or float(score) > current[1]:
                winners[key] = (slot_idx, float(score))
    removed = 0
    for slot_idx, slot in enumerate(slots):
        kept = []
        for frame_id, mask_id, score in slot.mask_observations:
            if winners.get((int(frame_id), int(mask_id)), (-1, 0.0))[0] == slot_idx:
                kept.append((int(frame_id), int(mask_id), float(score)))
            else:
                removed += 1
        slot.mask_observations = kept
    return removed


def explain_objects_mdl(
    bank: MeasurementBank,
    *,
    masklets: VideoMaskletBank | None = None,
    params: MDLParams | None = None,
) -> tuple[list[ObjectSlot], dict[str, Any]]:
    params = params or MDLParams()
    candidates, rejected_births = _candidate_slots(bank, masklets, params)
    selected: list[ObjectSlot] = []
    owned_core: set[int] = set()
    rejected_by_overlap = 0
    trim_events = 0
    for candidate in candidates:
        if len(selected) >= int(params.max_slots):
            break
        overlap = _surfel_overlap_ratio(candidate.core_surfels, owned_core)
        if overlap > float(params.max_core_overlap_ratio):
            rejected_by_overlap += 1
            continue
        if owned_core and candidate.fringe_surfels.size:
            keep = np.asarray([int(v) not in owned_core for v in candidate.fringe_surfels.tolist()], dtype=bool)
            if np.count_nonzero(~keep):
                trim_events += 1
                candidate.fringe_surfels = candidate.fringe_surfels[keep]
        selected.append(candidate)
        owned_core.update(int(v) for v in candidate.core_surfels.tolist())
    wta_removed = _apply_measurement_wta(selected)
    selected = [slot for slot in selected if slot.mask_observations]
    for idx, slot in enumerate(selected):
        slot.object_id = idx

    explained: set[tuple[int, int]] = set()
    multi: defaultdict[tuple[int, int], int] = defaultdict(int)
    for slot in selected:
        for frame_id, mask_id, _score in slot.mask_observations:
            key = (int(frame_id), int(mask_id))
            explained.add(key)
            multi[key] += 1
    available_measurements: set[tuple[int, int]] = set()
    for frame_idx, frame_id in enumerate(bank.frame_ids.tolist()):
        ids = np.unique(bank.target_mask_id[frame_idx][bank.positive_observation[frame_idx]])
        for mask_id in ids.tolist():
            if int(mask_id) > 0:
                available_measurements.add((int(frame_id), int(mask_id)))
    total_core = sum(int(slot.core_surfels.shape[0]) for slot in selected)
    total_fringe = sum(int(slot.fringe_surfels.shape[0]) for slot in selected)
    total_unknown = sum(int(slot.unknown_surfels.shape[0]) for slot in selected)
    total_reject = sum(int(slot.reject_surfels.shape[0]) for slot in selected)
    total_assigned = total_core + total_fringe
    energies = [float(slot.diagnostics.get("mdl_energy", 0.0)) for slot in selected]
    diag = {
        "mode": "mdl",
        "masklet_mode": None if masklets is None else masklets.mode,
        "num_candidate_slots": int(len(candidates)),
        "num_selected_slots": int(len(selected)),
        "selected_unselected_ratio": float(len(selected) / max(len(candidates) - len(selected), 1)),
        "num_birth_slots": int(len(candidates) + rejected_births),
        "num_rejected_slots": int(rejected_births + rejected_by_overlap),
        "num_rejected_by_overlap": int(rejected_by_overlap),
        "num_split_events": 0,
        "num_merge_events": 0,
        "num_swap_accepted": 0,
        "num_fringe_trim_events": int(trim_events),
        "num_measurement_wta_removed": int(wta_removed),
        "energy_total": float(np.sum(energies)) if energies else 0.0,
        "energy_mean": float(np.mean(energies)) if energies else 0.0,
        "energy_model_cost": float(params.model_cost * len(selected)),
        "energy_overlap_penalty": float(params.overlap_penalty * rejected_by_overlap),
        "energy_unexplained_penalty": float(
            params.unexplained_penalty * max(len(available_measurements) - len(explained), 0)
        ),
        "explained_measurement_ratio": float(len(explained) / max(len(available_measurements), 1)),
        "unexplained_measurement_ratio": float((len(available_measurements) - len(explained)) / max(len(available_measurements), 1)),
        "multi_explained_measurement_ratio": float(sum(1 for count in multi.values() if count > 1) / max(len(available_measurements), 1)),
        "surfel_overlap_conflict": float(0.0),
        "assigned_surfel_ratio": float(total_assigned / max(bank.num_surfels, 1)),
        "core_surfel_ratio": float(total_core / max(bank.num_surfels, 1)),
        "fringe_surfel_ratio": float(total_fringe / max(bank.num_surfels, 1)),
        "unknown_surfel_ratio": float(total_unknown / max(bank.num_surfels, 1)),
        "reject_surfel_ratio": float(total_reject / max(bank.num_surfels, 1)),
        "mean_measurements_per_selected_slot": float(np.mean([len(slot.mask_observations) for slot in selected])) if selected else 0.0,
        "mean_surfels_per_selected_slot": float(np.mean([slot.core_surfels.shape[0] + slot.fringe_surfels.shape[0] for slot in selected])) if selected else 0.0,
        "available_measurement_count": int(len(available_measurements)),
        "explained_measurement_count": int(len(explained)),
    }
    return selected, json_safe(diag)


def slot_to_posterior_record(slot: ObjectSlot) -> dict[str, Any]:
    return {
        "mask_list": [(int(frame), int(mask), float(score)) for frame, mask, score in slot.mask_observations],
        "carrier_ids": slot.core_surfels.astype(np.int64, copy=False),
        "core_surfels": slot.core_surfels.astype(np.int64, copy=False),
        "fringe_surfels": slot.fringe_surfels.astype(np.int64, copy=False),
        "unknown_surfels": slot.unknown_surfels.astype(np.int64, copy=False),
        "reject_surfels": slot.reject_surfels.astype(np.int64, copy=False),
        "v13_slot": {
            "object_id": int(slot.object_id),
            "birth_frame": int(slot.birth_frame),
            "birth_mask_id": int(slot.birth_mask_id),
            "num_core_surfels": int(slot.core_surfels.shape[0]),
            "num_fringe_surfels": int(slot.fringe_surfels.shape[0]),
            "num_unknown_surfels": int(slot.unknown_surfels.shape[0]),
            "num_reject_surfels": int(slot.reject_surfels.shape[0]),
            **slot.diagnostics,
        },
    }
