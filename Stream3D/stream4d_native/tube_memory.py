from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .object_tube_io import MergeGeometryError, TubeRecord


@dataclass
class TubeMemoryObject:
    object_id: int
    tube_ids: list[int] = field(default_factory=list)


@dataclass
class TubeMemoryUpdateResult:
    objects: list[TubeMemoryObject]
    diagnostics: dict[str, Any]
    blocked_events: list[dict[str, Any]]


class TubeMemory:
    def __init__(self) -> None:
        self.objects: list[TubeMemoryObject] = []
        self._next_object_id = 0

    def update(
        self,
        components: list[list[int]],
        tubes_by_id: dict[int, TubeRecord],
        *,
        context: str = "v25_tube_memory",
    ) -> TubeMemoryUpdateResult:
        blocked: list[dict[str, Any]] = []
        memory_match_count = 0
        for component in components:
            component = [int(v) for v in component]
            matched: TubeMemoryObject | None = None
            for obj in self.objects:
                rep_existing = tubes_by_id.get(obj.tube_ids[0]) if obj.tube_ids else None
                rep_new = tubes_by_id.get(component[0]) if component else None
                if rep_existing is None or rep_new is None:
                    continue
                try:
                    rep_existing.get_geometry_for_merge(rep_new, context, merge_type="memory_match")
                except MergeGeometryError as exc:
                    blocked.append({"event_type": "memory_match_blocked", "error": str(exc)})
                    continue
                matched = obj
                memory_match_count += 1
                break
            if matched is None:
                matched = TubeMemoryObject(object_id=self._next_object_id)
                self._next_object_id += 1
                self.objects.append(matched)
            matched.tube_ids = sorted(set(matched.tube_ids) | set(component))
        return TubeMemoryUpdateResult(
            objects=list(self.objects),
            diagnostics={
                "object_count": int(len(self.objects)),
                "memory_match_count": int(memory_match_count),
                "memory_match_blocked_count": int(len(blocked)),
            },
            blocked_events=blocked,
        )
