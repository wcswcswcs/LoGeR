from __future__ import annotations

from dataclasses import dataclass, field

from .local_4d_filter import LocalProposal


@dataclass
class Object4D:
    object_id: int
    carrier_ids: set[int] = field(default_factory=set)
    frame_support: dict[int, set[int]] = field(default_factory=dict)
    mask_observations: list[tuple[int, int, float]] = field(default_factory=list)
    feature_sum: object | None = None
    feature_weight: float = 0.0
    last_seen: int = 0
    birth_frame: int = 0
    state: str = "active"


class ObjectMemory4D:
    def __init__(self, history_match_threshold: float = 0.30, lost_tolerance_windows: int = 3) -> None:
        self.history_match_threshold = float(history_match_threshold)
        self.lost_tolerance_windows = int(lost_tolerance_windows)
        self.objects: dict[int, Object4D] = {}
        self.next_object_id = 0
        self.reactivation_count = 0

    @staticmethod
    def _carrier_overlap(a: set[int], b: set[int]) -> float:
        if not a or not b:
            return 0.0
        return float(len(a.intersection(b)) / max(1, min(len(a), len(b))))

    def _best_match(self, proposal: LocalProposal, candidate_ids: set[int] | None = None) -> tuple[int | None, float]:
        best_id: int | None = None
        best_score = 0.0
        for object_id, obj in self.objects.items():
            if candidate_ids is not None and object_id not in candidate_ids:
                continue
            if obj.state == "merged":
                continue
            score = self._carrier_overlap(proposal.carrier_ids, obj.carrier_ids)
            if score > best_score:
                best_score = score
                best_id = object_id
        return best_id, best_score

    def update(self, proposals: list[LocalProposal], window_index: int) -> dict[str, float]:
        created = 0
        matched = 0
        historical_ids = set(self.objects.keys())
        for proposal in proposals:
            best_id, best_score = self._best_match(proposal, candidate_ids=historical_ids)
            if best_id is not None and best_score >= self.history_match_threshold:
                obj = self.objects[best_id]
                if obj.state == "lost":
                    self.reactivation_count += 1
                obj.state = "active"
                obj.last_seen = int(window_index)
                obj.carrier_ids.update(proposal.carrier_ids)
                for frame_id, carriers in proposal.frame_support.items():
                    obj.frame_support.setdefault(int(frame_id), set()).update(carriers)
                obj.mask_observations.extend(proposal.mask_observations)
                matched += 1
                continue

            object_id = self.next_object_id
            self.next_object_id += 1
            self.objects[object_id] = Object4D(
                object_id=object_id,
                carrier_ids=set(proposal.carrier_ids),
                frame_support={int(k): set(v) for k, v in proposal.frame_support.items()},
                mask_observations=list(proposal.mask_observations),
                last_seen=int(window_index),
                birth_frame=min(proposal.frame_support.keys()) if proposal.frame_support else 0,
                state="active",
            )
            created += 1

        for obj in self.objects.values():
            if int(window_index) - obj.last_seen > self.lost_tolerance_windows and obj.state == "active":
                obj.state = "lost"

        return {
            "num_objects": float(len(self.objects)),
            "num_active_objects": float(sum(1 for obj in self.objects.values() if obj.state == "active")),
            "num_lost_objects": float(sum(1 for obj in self.objects.values() if obj.state == "lost")),
            "num_created": float(created),
            "num_matched": float(matched),
            "object_reactivation_count": float(self.reactivation_count),
        }

    def to_jsonable(self) -> dict:
        return {
            "objects": [
                {
                    "object_id": int(obj.object_id),
                    "num_carriers": int(len(obj.carrier_ids)),
                    "frames": sorted(int(k) for k in obj.frame_support.keys()),
                    "num_mask_observations": int(len(obj.mask_observations)),
                    "last_seen_window": int(obj.last_seen),
                    "birth_frame": int(obj.birth_frame),
                    "state": obj.state,
                }
                for obj in sorted(self.objects.values(), key=lambda item: item.object_id)
            ],
            "object_reactivation_count": int(self.reactivation_count),
        }
