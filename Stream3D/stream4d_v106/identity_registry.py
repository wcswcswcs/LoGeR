from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set


@dataclass
class GlobalIdentityRegistry:
    local_to_global: Dict[tuple[int, int], int] = field(default_factory=dict)
    assigned_current_ids: Set[int] = field(default_factory=set)

    def bind(self, chunk_index: int, local_id: int, global_id: int) -> None:
        key = (chunk_index, local_id)
        if key in self.local_to_global and self.local_to_global[key] != global_id:
            raise AssertionError(f"local id {key} already bound to a different global id")
        if global_id in self.assigned_current_ids:
            raise AssertionError(f"global id {global_id} assigned twice in the current chunk")
        self.local_to_global[key] = global_id
        self.assigned_current_ids.add(global_id)

    def reset_current_assignment(self) -> None:
        self.assigned_current_ids.clear()

    def global_ids_for_chunk(self, chunk_index: int) -> Iterable[int]:
        for (idx, _local_id), global_id in self.local_to_global.items():
            if idx == chunk_index:
                yield global_id

