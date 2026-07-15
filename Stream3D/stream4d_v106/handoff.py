from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class HandoffObject:
    global_id: int
    source_chunk_index: int
    target_chunk_index: int
    prompt_kind: str
    quality_score: float


@dataclass
class HandoffPacket:
    scene_id: str
    source_chunk_index: int
    target_chunk_index: int
    history_version: int
    objects: List[HandoffObject] = field(default_factory=list)

    def validate(self) -> None:
        if self.target_chunk_index != self.source_chunk_index + 1:
            raise AssertionError("handoff target must be the next sequential chunk")
        seen: Dict[int, int] = {}
        for obj in self.objects:
            if obj.target_chunk_index != self.target_chunk_index:
                raise AssertionError("handoff object target chunk mismatch")
            if obj.global_id in seen:
                raise AssertionError(f"duplicate global id in handoff: {obj.global_id}")
            seen[obj.global_id] = 1

