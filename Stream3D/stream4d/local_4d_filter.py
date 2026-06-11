from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from .mask_evidence import MaskObservation


@dataclass
class LocalProposal:
    proposal_id: int
    observation_indices: list[int]
    carrier_ids: set[int]
    frame_support: dict[int, set[int]]
    mask_observations: list[tuple[int, int, float]]
    appearance_feature: np.ndarray | None = None
    centroid_feature: np.ndarray | None = None
    feature_type: str = ""


class Local4DFilter:
    def __init__(
        self,
        local_ioc_threshold: float = 0.25,
        temporal_eta: float = 12.0,
        lambda_carrier: float = 0.8,
        lambda_temporal: float = 0.2,
    ) -> None:
        self.local_ioc_threshold = float(local_ioc_threshold)
        self.temporal_eta = float(temporal_eta)
        self.lambda_carrier = float(lambda_carrier)
        self.lambda_temporal = float(lambda_temporal)

    @staticmethod
    def _support(obs: MaskObservation) -> set[tuple[int, int]]:
        return {(int(obs.frame_id), int(cid)) for cid in obs.carrier_ids.tolist()}

    @staticmethod
    def _carrier_weights(observations: list[MaskObservation]) -> dict[tuple[int, int], float]:
        weights: dict[tuple[int, int], float] = {}
        for obs in observations:
            for cid, weight in zip(obs.carrier_ids.tolist(), obs.weights.tolist()):
                key = (int(obs.frame_id), int(cid))
                weights[key] = max(weights.get(key, 0.0), float(weight))
        return weights

    def select(self, observations: list[MaskObservation]) -> list[int]:
        if not observations:
            return []
        supports = [self._support(obs) for obs in observations]
        weights = self._carrier_weights(observations)
        uncovered = set(weights.keys())
        selected: list[int] = []
        remaining = set(range(len(observations)))
        while uncovered and remaining:
            best_idx = -1
            best_gain = 0.0
            for idx in remaining:
                gain = sum(weights[key] for key in supports[idx].intersection(uncovered))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx < 0 or best_gain <= 0.0:
                break
            selected.append(best_idx)
            uncovered.difference_update(supports[best_idx])
            remaining.remove(best_idx)
        return selected

    @staticmethod
    def _ioc(a: set[int], b: set[int]) -> float:
        denom = max(1, min(len(a), len(b)))
        return float(len(a.intersection(b)) / denom)

    def merge(self, observations: list[MaskObservation], selected_indices: list[int]) -> list[LocalProposal]:
        if not selected_indices:
            return []
        carrier_sets = [observations[idx].carrier_set() for idx in selected_indices]
        parent = list(range(len(selected_indices)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(selected_indices)):
            obs_i = observations[selected_indices[i]]
            for j in range(i + 1, len(selected_indices)):
                obs_j = observations[selected_indices[j]]
                ioc = self._ioc(carrier_sets[i], carrier_sets[j])
                temporal = exp(-abs(obs_i.frame_id - obs_j.frame_id) / max(self.temporal_eta, 1e-6))
                score = self.lambda_carrier * ioc + self.lambda_temporal * temporal
                if score >= self.local_ioc_threshold:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for local_idx, obs_idx in enumerate(selected_indices):
            groups.setdefault(find(local_idx), []).append(obs_idx)

        proposals: list[LocalProposal] = []
        for proposal_id, obs_indices in enumerate(groups.values()):
            carriers: set[int] = set()
            frame_support: dict[int, set[int]] = {}
            mask_infos: list[tuple[int, int, float]] = []
            for obs_idx in obs_indices:
                obs = observations[obs_idx]
                obs_carriers = obs.carrier_set()
                carriers.update(obs_carriers)
                frame_support.setdefault(int(obs.frame_id), set()).update(obs_carriers)
                coverage = float(len(obs.carrier_ids) / max(obs.area, 1))
                mask_infos.append((int(obs.frame_id), int(obs.mask_id), coverage))
            proposals.append(
                LocalProposal(
                    proposal_id=proposal_id,
                    observation_indices=list(obs_indices),
                    carrier_ids=carriers,
                    frame_support=frame_support,
                    mask_observations=mask_infos,
                )
            )
        return proposals

    def run(self, observations: list[MaskObservation]) -> tuple[list[LocalProposal], dict[str, float]]:
        selected = self.select(observations)
        proposals = self.merge(observations, selected)
        all_carriers = set()
        selected_carriers = set()
        for obs in observations:
            all_carriers.update(obs.carrier_set())
        for idx in selected:
            selected_carriers.update(observations[idx].carrier_set())
        diagnostics = {
            "num_raw_mask_observations": float(len(observations)),
            "num_selected_mask_observations": float(len(selected)),
            "num_local_proposals": float(len(proposals)),
            "mean_carriers_per_proposal": float(np.mean([len(p.carrier_ids) for p in proposals])) if proposals else 0.0,
            "carrier_coverage": float(len(selected_carriers) / max(len(all_carriers), 1)),
            "local_selected_mask_ratio": float(len(selected) / max(len(observations), 1)),
        }
        return proposals, diagnostics
