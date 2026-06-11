from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .appearance_memory import cosine_similarity_01_valid, normalize_feature
from .local_4d_filter import LocalProposal
from .memory_diagnostics import carrier_ioc, finite_max, finite_mean, same_frame_conflict_rate
from .motion_memory import centroid_similarity, velocity_continuity_score
from .object_memory import Object4D

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - exercised only when scipy is unavailable.
    linear_sum_assignment = None


@dataclass
class MatchComponents:
    total: float
    carrier: float
    appearance: float
    geometry: float
    motion: float
    conflict: float


class ObjectMemory4DV2:
    def __init__(
        self,
        history_match_threshold: float = 0.30,
        lost_tolerance_windows: int = 3,
        carrier_weight: float = 0.55,
        appearance_weight: float = 0.25,
        geometry_weight: float = 0.20,
        motion_weight: float = 0.0,
        conflict_weight: float = 0.30,
        geometry_sigma: float = 0.35,
        motion_sigma: float = 0.35,
        min_carrier_score: float = 0.0,
    ) -> None:
        self.history_match_threshold = float(history_match_threshold)
        self.lost_tolerance_windows = int(lost_tolerance_windows)
        self.carrier_weight = float(carrier_weight)
        self.appearance_weight = float(appearance_weight)
        self.geometry_weight = float(geometry_weight)
        self.motion_weight = float(motion_weight)
        self.conflict_weight = float(conflict_weight)
        self.geometry_sigma = float(geometry_sigma)
        self.motion_sigma = float(motion_sigma)
        self.min_carrier_score = float(min_carrier_score)
        self.objects: dict[int, Object4D] = {}
        self.next_object_id = 0
        self.reactivation_count = 0
        self.match_history: list[dict[str, float | int | str]] = []

    def _candidate_ids(self, window_index: int) -> list[int]:
        ids: list[int] = []
        max_gap = max(self.lost_tolerance_windows * 2, self.lost_tolerance_windows + 1)
        for object_id, obj in self.objects.items():
            if obj.state == "merged":
                continue
            if int(window_index) - int(obj.last_seen) <= max_gap:
                ids.append(int(object_id))
        return ids

    def _score(self, obj: Object4D, proposal: LocalProposal, window_index: int) -> MatchComponents:
        carrier = carrier_ioc(obj.carrier_ids, proposal.carrier_ids)
        appearance, appearance_valid = cosine_similarity_01_valid(
            getattr(obj, "appearance_feature", None),
            getattr(proposal, "appearance_feature", None),
        )
        geometry = centroid_similarity(
            getattr(obj, "centroid_feature", None),
            getattr(proposal, "centroid_feature", None),
            sigma=self.geometry_sigma,
        )
        motion = velocity_continuity_score(
            getattr(obj, "centroid_history", []),
            getattr(proposal, "centroid_feature", None),
            window_index=window_index,
            sigma=self.motion_sigma,
        )
        conflict = same_frame_conflict_rate(obj, proposal)
        appearance_weight = self.appearance_weight if appearance_valid else 0.0
        total = (
            self.carrier_weight * carrier
            + appearance_weight * appearance
            + self.geometry_weight * geometry
            + self.motion_weight * motion
            - self.conflict_weight * conflict
        )
        return MatchComponents(
            total=float(total),
            carrier=float(carrier),
            appearance=float(appearance),
            geometry=float(geometry),
            motion=float(motion),
            conflict=float(conflict),
        )

    @staticmethod
    def _proposal_weight(proposal: LocalProposal) -> float:
        return float(max(len(proposal.mask_observations), len(proposal.carrier_ids), 1))

    def _update_features(self, obj: Object4D, proposal: LocalProposal, window_index: int) -> None:
        proposal_feature = normalize_feature(getattr(proposal, "appearance_feature", None))
        proposal_weight = self._proposal_weight(proposal)
        if proposal_feature is not None:
            old_sum = getattr(obj, "feature_sum", None)
            old_weight = float(getattr(obj, "feature_weight", 0.0))
            if old_sum is None or old_weight <= 0.0:
                feature_sum = proposal_feature.astype(np.float32) * proposal_weight
                feature_weight = proposal_weight
            else:
                feature_sum = np.asarray(old_sum, dtype=np.float32) + proposal_feature.astype(np.float32) * proposal_weight
                feature_weight = old_weight + proposal_weight
            obj.feature_sum = feature_sum
            obj.feature_weight = float(feature_weight)
            obj.appearance_feature = normalize_feature(feature_sum)

        proposal_centroid = getattr(proposal, "centroid_feature", None)
        if proposal_centroid is not None:
            centroid = np.asarray(proposal_centroid, dtype=np.float32).reshape(-1)
            if np.isfinite(centroid).all():
                obj.centroid_feature = centroid
                history = list(getattr(obj, "centroid_history", []))
                history.append((int(window_index), tuple(float(v) for v in centroid.tolist())))
                obj.centroid_history = history[-8:]
                if len(history) >= 2:
                    prev_t, prev_centroid_raw = history[-2]
                    prev_centroid = np.asarray(prev_centroid_raw, dtype=np.float32)
                    dt = max(int(window_index) - int(prev_t), 1)
                    velocity = (centroid - prev_centroid) / float(dt)
                    velocity_history = list(getattr(obj, "velocity_history", []))
                    velocity_history.append(tuple(float(v) for v in velocity.tolist()))
                    obj.velocity_history = velocity_history[-8:]

        obj.feature_type = getattr(proposal, "feature_type", "unknown")

    def _merge_proposal(self, obj: Object4D, proposal: LocalProposal, window_index: int) -> None:
        obj.last_seen = int(window_index)
        obj.carrier_ids.update(proposal.carrier_ids)
        for frame_id, carriers in proposal.frame_support.items():
            obj.frame_support.setdefault(int(frame_id), set()).update(carriers)
        obj.mask_observations.extend(proposal.mask_observations)
        self._update_features(obj, proposal, window_index)

    def _create_object(self, proposal: LocalProposal, window_index: int) -> Object4D:
        object_id = self.next_object_id
        self.next_object_id += 1
        obj = Object4D(
            object_id=object_id,
            carrier_ids=set(proposal.carrier_ids),
            frame_support={int(k): set(v) for k, v in proposal.frame_support.items()},
            mask_observations=list(proposal.mask_observations),
            last_seen=int(window_index),
            birth_frame=min(proposal.frame_support.keys()) if proposal.frame_support else 0,
            state="active",
        )
        obj.birth_window = int(window_index)
        obj.match_history = []
        obj.centroid_history = []
        obj.velocity_history = []
        self._update_features(obj, proposal, window_index)
        self.objects[object_id] = obj
        return obj

    def _hungarian_assignments(
        self,
        score_matrix: np.ndarray,
        candidate_ids: list[int],
    ) -> list[tuple[int, int, float]]:
        if score_matrix.size == 0:
            return []
        if linear_sum_assignment is None:
            pairs: list[tuple[int, int, float]] = []
            used_props: set[int] = set()
            used_objects: set[int] = set()
            flat = [
                (float(score_matrix[p_idx, o_idx]), p_idx, o_idx)
                for p_idx in range(score_matrix.shape[0])
                for o_idx in range(score_matrix.shape[1])
            ]
            for score, p_idx, o_idx in sorted(flat, reverse=True):
                if p_idx in used_props or o_idx in used_objects:
                    continue
                used_props.add(p_idx)
                used_objects.add(o_idx)
                pairs.append((p_idx, int(candidate_ids[o_idx]), score))
            return pairs
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        return [(int(row), int(candidate_ids[col]), float(score_matrix[row, col])) for row, col in zip(row_ind, col_ind)]

    def update(self, proposals: list[LocalProposal], window_index: int) -> dict[str, float]:
        candidate_ids = self._candidate_ids(window_index)
        score_components: dict[tuple[int, int], MatchComponents] = {}
        score_matrix = np.zeros((len(proposals), len(candidate_ids)), dtype=np.float32)
        for p_idx, proposal in enumerate(proposals):
            for o_idx, object_id in enumerate(candidate_ids):
                components = self._score(self.objects[object_id], proposal, window_index)
                score_components[(p_idx, object_id)] = components
                score_matrix[p_idx, o_idx] = components.total

        created = 0
        matched = 0
        reactivated = 0
        accepted_scores: list[float] = []
        accepted_carrier: list[float] = []
        accepted_appearance: list[float] = []
        accepted_geometry: list[float] = []
        accepted_motion: list[float] = []
        accepted_conflict: list[float] = []
        matched_proposals: set[int] = set()
        matched_objects: set[int] = set()

        for proposal_idx, object_id, score in self._hungarian_assignments(score_matrix, candidate_ids):
            if score < self.history_match_threshold:
                continue
            components = score_components[(proposal_idx, object_id)]
            if components.carrier < self.min_carrier_score:
                continue
            if proposal_idx in matched_proposals or object_id in matched_objects:
                continue
            proposal = proposals[proposal_idx]
            obj = self.objects[object_id]
            previous_state = obj.state
            if previous_state == "lost":
                reactivated += 1
                self.reactivation_count += 1
                obj.state = "reactivated"
            else:
                obj.state = "active"
            self._merge_proposal(obj, proposal, window_index)
            history_item = {
                "window_index": int(window_index),
                "object_id": int(object_id),
                "proposal_id": int(proposal.proposal_id),
                "score": float(components.total),
                "carrier": float(components.carrier),
                "appearance": float(components.appearance),
                "geometry": float(components.geometry),
                "motion": float(components.motion),
                "conflict": float(components.conflict),
                "previous_state": str(previous_state),
            }
            obj_history = list(getattr(obj, "match_history", []))
            obj_history.append(history_item)
            obj.match_history = obj_history
            self.match_history.append(history_item)
            matched_proposals.add(proposal_idx)
            matched_objects.add(object_id)
            matched += 1
            accepted_scores.append(float(components.total))
            accepted_carrier.append(float(components.carrier))
            accepted_appearance.append(float(components.appearance))
            accepted_geometry.append(float(components.geometry))
            accepted_motion.append(float(components.motion))
            accepted_conflict.append(float(components.conflict))

        for proposal_idx, proposal in enumerate(proposals):
            if proposal_idx in matched_proposals:
                continue
            self._create_object(proposal, window_index)
            created += 1

        lost_now = 0
        for obj in self.objects.values():
            if int(window_index) - obj.last_seen > self.lost_tolerance_windows and obj.state in {"active", "reactivated"}:
                obj.state = "lost"
                lost_now += 1

        active_count = sum(1 for obj in self.objects.values() if obj.state in {"active", "reactivated"})
        lost_count = sum(1 for obj in self.objects.values() if obj.state == "lost")
        return {
            "num_objects": float(len(self.objects)),
            "num_active_objects": float(active_count),
            "num_lost_objects": float(lost_count),
            "num_created": float(created),
            "num_matched": float(matched),
            "num_reactivated": float(reactivated),
            "num_lost_new": float(lost_now),
            "object_reactivation_count": float(self.reactivation_count),
            "memory_v2_num_candidate_pairs": float(score_matrix.size),
            "memory_v2_match_score_mean": finite_mean(accepted_scores),
            "memory_v2_match_score_max": finite_max(accepted_scores),
            "memory_v2_carrier_score_mean": finite_mean(accepted_carrier),
            "memory_v2_appearance_score_mean": finite_mean(accepted_appearance),
            "memory_v2_geometry_score_mean": finite_mean(accepted_geometry),
            "memory_v2_motion_score_mean": finite_mean(accepted_motion),
            "memory_v2_same_frame_conflict_rate": finite_mean(accepted_conflict),
            "memory_v2_hungarian_assignments": float(len(matched_proposals)),
            "memory_v2_match_threshold": float(self.history_match_threshold),
            "memory_v2_min_carrier_score": float(self.min_carrier_score),
        }

    def to_jsonable(self) -> dict:
        return {
            "memory_version": "v2",
            "memory_params": {
                "history_match_threshold": float(self.history_match_threshold),
                "lost_tolerance_windows": int(self.lost_tolerance_windows),
                "carrier_weight": float(self.carrier_weight),
                "appearance_weight": float(self.appearance_weight),
                "geometry_weight": float(self.geometry_weight),
                "motion_weight": float(self.motion_weight),
                "conflict_weight": float(self.conflict_weight),
                "geometry_sigma": float(self.geometry_sigma),
                "motion_sigma": float(self.motion_sigma),
                "min_carrier_score": float(self.min_carrier_score),
            },
            "objects": [
                {
                    "object_id": int(obj.object_id),
                    "state": obj.state,
                    "num_carriers": int(len(obj.carrier_ids)),
                    "frames": sorted(int(k) for k in obj.frame_support.keys()),
                    "num_mask_observations": int(len(obj.mask_observations)),
                    "last_seen_window": int(obj.last_seen),
                    "birth_window": int(getattr(obj, "birth_window", 0)),
                    "birth_frame": int(obj.birth_frame),
                    "feature_type": str(getattr(obj, "feature_type", "")),
                    "feature_weight": float(getattr(obj, "feature_weight", 0.0)),
                    "centroid_history": list(getattr(obj, "centroid_history", [])),
                    "velocity_history": list(getattr(obj, "velocity_history", [])),
                    "match_history": list(getattr(obj, "match_history", [])),
                }
                for obj in sorted(self.objects.values(), key=lambda item: item.object_id)
            ],
            "object_reactivation_count": int(self.reactivation_count),
            "match_history": self.match_history,
        }
