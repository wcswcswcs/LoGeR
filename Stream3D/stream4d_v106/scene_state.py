from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ObjectletState:
    global_id: int
    local_id: int
    birth_chunk: int
    last_seen_chunk: int
    status: str = "active"
    mask_area: int = 0
    parent_global_id: Optional[int] = None


@dataclass
class ChunkState:
    scene_id: str
    chunk_index: int
    history_version_in: int
    history_version_out: int
    initialization_mode: str
    inherited_global_ids: List[int] = field(default_factory=list)
    born_global_ids: List[int] = field(default_factory=list)
    active_global_ids: List[int] = field(default_factory=list)


@dataclass
class SceneStreamState:
    scene_id: str
    next_global_id: int = 1
    history_version: int = 0
    objectlets: Dict[int, ObjectletState] = field(default_factory=dict)
    chunks: List[ChunkState] = field(default_factory=list)

    def plan_chunk_initialization(self, chunk_index: int) -> str:
        if chunk_index == 0:
            return "chunk0_full_baseline_x_initialization"
        if not self.chunks:
            raise RuntimeError("later chunk cannot be initialized before chunk0 state exists")
        return "inherited_scene_state_plus_exact_gap_birth"

    def allocate_objectlet(self, local_id: int, chunk_index: int, mask_area: int = 0) -> ObjectletState:
        global_id = self.next_global_id
        self.next_global_id += 1
        obj = ObjectletState(
            global_id=global_id,
            local_id=local_id,
            birth_chunk=chunk_index,
            last_seen_chunk=chunk_index,
            mask_area=mask_area,
        )
        self.objectlets[global_id] = obj
        return obj

    def begin_chunk(self, chunk_index: int) -> ChunkState:
        mode = self.plan_chunk_initialization(chunk_index)
        inherited = sorted(
            gid for gid, obj in self.objectlets.items() if obj.status in {"active", "occluded"}
        )
        return ChunkState(
            scene_id=self.scene_id,
            chunk_index=chunk_index,
            history_version_in=self.history_version,
            history_version_out=self.history_version,
            initialization_mode=mode,
            inherited_global_ids=inherited if chunk_index > 0 else [],
            active_global_ids=inherited,
        )

    def finish_chunk(self, chunk: ChunkState) -> ChunkState:
        self.history_version += 1
        chunk.history_version_out = self.history_version
        self.chunks.append(chunk)
        return chunk


def assert_same_scene_sequential(chunk_states: List[ChunkState]) -> None:
    expected = list(range(len(chunk_states)))
    actual = [c.chunk_index for c in chunk_states]
    if actual != expected:
        raise AssertionError(f"same-scene chunks are not sequential: expected {expected}, got {actual}")
    for prev, cur in zip(chunk_states, chunk_states[1:]):
        if cur.history_version_in != prev.history_version_out:
            raise AssertionError(
                "chunk handoff is not history-versioned: "
                f"chunk {prev.chunk_index}->{cur.chunk_index} "
                f"{prev.history_version_out}!={cur.history_version_in}"
            )

