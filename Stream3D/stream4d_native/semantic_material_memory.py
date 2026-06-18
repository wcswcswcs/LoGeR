from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryObjectState:
    object_id: int
    active: bool = True
    temporal_span: int = 1
    reactivation_count: int = 0


@dataclass(frozen=True)
class MemoryObservation:
    object_id: int
    has_semantic_support: bool
    material_consistency: float
    frame_rank: int


@dataclass
class SemanticMaterialMemoryUpdate:
    objects: dict[int, MemoryObjectState] = field(default_factory=dict)
    blocked_events: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SemanticMaterialMemory:
    def __init__(self, *, min_material_consistency: float = 0.50) -> None:
        self.objects: dict[int, MemoryObjectState] = {}
        self.min_material_consistency = float(min_material_consistency)

    def update(self, observations: list[MemoryObservation]) -> SemanticMaterialMemoryUpdate:
        blocked: list[dict[str, Any]] = []
        reactivated = 0
        for obs in observations:
            object_id = int(obs.object_id)
            if not bool(obs.has_semantic_support):
                blocked.append(
                    {
                        "object_id": object_id,
                        "frame_rank": int(obs.frame_rank),
                        "reason": "memory_cannot_birth_or_reactivate_without_semantic_support",
                    }
                )
                continue
            if float(obs.material_consistency) < self.min_material_consistency:
                blocked.append(
                    {
                        "object_id": object_id,
                        "frame_rank": int(obs.frame_rank),
                        "reason": "material_consistency_below_threshold",
                    }
                )
                continue
            state = self.objects.get(object_id)
            if state is None:
                state = MemoryObjectState(object_id=object_id)
                self.objects[object_id] = state
            elif not state.active:
                state.active = True
                state.reactivation_count += 1
                reactivated += 1
            state.temporal_span += 1
        diagnostics = {
            "object_count": int(len(self.objects)),
            "reactivation_success": int(reactivated),
            "blocked_event_count": int(len(blocked)),
            "memory_birth_without_semantic_support_count": int(
                sum(1 for event in blocked if event["reason"] == "memory_cannot_birth_or_reactivate_without_semantic_support")
            ),
            "temporal_span_mean": float(
                sum(state.temporal_span for state in self.objects.values()) / max(len(self.objects), 1)
            ),
        }
        return SemanticMaterialMemoryUpdate(objects=dict(self.objects), blocked_events=blocked, diagnostics=diagnostics)

