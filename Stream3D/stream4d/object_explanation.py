from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .evidence_terms import ExplanationParams, birth_groups, measurement_votes, posterior_for_group
from .measurement_bank import MeasurementBank, json_safe
from .object_slot import ObjectSlot


def _apply_measurement_wta(slots: list[ObjectSlot]) -> None:
    winners: dict[tuple[int, int], tuple[int, float]] = {}
    for slot_idx, slot in enumerate(slots):
        for frame_id, mask_id, score in slot.mask_observations:
            key = (int(frame_id), int(mask_id))
            current = winners.get(key)
            if current is None or float(score) > current[1]:
                winners[key] = (slot_idx, float(score))
    for slot_idx, slot in enumerate(slots):
        slot.mask_observations = [
            (int(frame_id), int(mask_id), float(score))
            for frame_id, mask_id, score in slot.mask_observations
            if winners.get((int(frame_id), int(mask_id)), (-1, 0.0))[0] == slot_idx
        ]


def explain_objects(
    bank: MeasurementBank,
    *,
    params: ExplanationParams | None = None,
    mode: str = "with_negative",
    seed: int = 0,
) -> tuple[list[ObjectSlot], dict[str, Any]]:
    params = params or ExplanationParams()
    if mode not in {
        "no_negative",
        "with_negative",
        "shuffled_d4rt",
        "no_d4rt_temporal",
        "surfel_cluster_candidate",
    }:
        raise ValueError(f"Unsupported v12 object explanation mode: {mode}")
    rng = np.random.default_rng(int(seed))
    use_negative = mode in {"with_negative", "surfel_cluster_candidate"}
    use_temporal = mode != "no_d4rt_temporal"
    shuffled = mode == "shuffled_d4rt"
    groups = birth_groups(bank, params, shuffled_source=shuffled, rng=rng)
    slots: list[ObjectSlot] = []
    rejected_births = 0
    for group in groups:
        if mode != "surfel_cluster_candidate" and not bool(group["passes_birth_gate"]):
            rejected_births += 1
            continue
        posterior = posterior_for_group(
            bank,
            np.asarray(group["surfel_indices"], dtype=np.int64),
            params,
            use_negative=use_negative,
            use_temporal=use_temporal,
        )
        core = np.asarray(posterior["core"], dtype=np.int64)
        fringe = np.asarray(posterior["fringe"], dtype=np.int64)
        unknown = np.asarray(posterior["unknown"], dtype=np.int64)
        reject = np.asarray(posterior["reject"], dtype=np.int64)
        if mode == "surfel_cluster_candidate":
            core = np.asarray(group["surfel_indices"], dtype=np.int64)
            fringe = np.empty((0,), dtype=np.int64)
            unknown = np.empty((0,), dtype=np.int64)
            reject = np.empty((0,), dtype=np.int64)
        if core.shape[0] < int(params.min_core_surfels_per_object):
            rejected_births += 1
            continue
        observations = measurement_votes(
            bank,
            core,
            params,
            include_temporal_targets=mode != "no_d4rt_temporal",
        )
        if not observations:
            rejected_births += 1
            continue
        slot = ObjectSlot(
            object_id=len(slots),
            birth_frame=int(group["birth_frame"]),
            birth_mask_id=int(group["birth_mask_id"]),
            core_surfels=core,
            fringe_surfels=fringe,
            unknown_surfels=unknown,
            reject_surfels=reject,
            mask_observations=observations,
            diagnostics={
                "mode": mode,
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
        slots.append(slot)
    if mode in {"with_negative", "surfel_cluster_candidate"}:
        _apply_measurement_wta(slots)
        slots = [slot for slot in slots if slot.mask_observations]
        for idx, slot in enumerate(slots):
            slot.object_id = idx

    explained: set[tuple[int, int]] = set()
    multi: defaultdict[tuple[int, int], int] = defaultdict(int)
    for slot in slots:
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
    total_slot_surfels = sum(int(slot.core_surfels.shape[0] + slot.fringe_surfels.shape[0]) for slot in slots)
    total_core = sum(int(slot.core_surfels.shape[0]) for slot in slots)
    total_fringe = sum(int(slot.fringe_surfels.shape[0]) for slot in slots)
    total_unknown = sum(int(slot.unknown_surfels.shape[0]) for slot in slots)
    total_reject = sum(int(slot.reject_surfels.shape[0]) for slot in slots)
    diag = {
        "mode": mode,
        "num_birth_slots": int(len(groups)),
        "num_active_slots": int(len(slots)),
        "num_rejected_slots": int(rejected_births),
        "num_split_events": 0,
        "num_merge_events": 0,
        "num_unknown_surfels": int(total_unknown),
        "assigned_surfel_ratio": float(total_slot_surfels / max(bank.num_surfels, 1)),
        "core_surfel_ratio": float(total_core / max(bank.num_surfels, 1)),
        "fringe_surfel_ratio": float(total_fringe / max(bank.num_surfels, 1)),
        "reject_surfel_ratio": float(total_reject / max(bank.num_surfels, 1)),
        "visible_outside_negative_ratio": float(np.count_nonzero(bank.negative_observation) / max(np.count_nonzero(bank.visible_ok), 1)),
        "same_frame_cannot_link_violations": int(sum(1 for count in multi.values() if count > 1)),
        "mean_object_positive_evidence": _mean_slot(slots, "positive_score"),
        "mean_object_negative_evidence": _mean_slot(slots, "negative_ratio"),
        "mean_object_boundary_risk": _mean_slot(slots, "boundary_risk"),
        "mean_object_appearance_consistency": _mean_slot(slots, "appearance_consistency"),
        "mean_object_temporal_consistency": _mean_slot(slots, "temporal_consistency"),
        "measurement_explained_ratio": float(len(explained) / max(len(available_measurements), 1)),
        "measurement_multi_explained_ratio": float(sum(1 for count in multi.values() if count > 1) / max(len(available_measurements), 1)),
        "measurement_unexplained_ratio": float((len(available_measurements) - len(explained)) / max(len(available_measurements), 1)),
        "available_measurement_count": int(len(available_measurements)),
        "explained_measurement_count": int(len(explained)),
    }
    return slots, json_safe(diag)


def _mean_slot(slots: list[ObjectSlot], key: str) -> float:
    vals = [float(slot.diagnostics[key]) for slot in slots if slot.diagnostics.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0
