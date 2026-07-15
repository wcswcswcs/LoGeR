from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LifecycleDecision:
    global_id: int
    from_status: str
    to_status: str
    reason: str


def decide_lifecycle(
    global_id: int,
    was_visible: bool,
    is_visible: bool,
    missing_chunks: int,
    occlusion_memory_max_chunks: int,
) -> LifecycleDecision:
    if is_visible:
        return LifecycleDecision(global_id, "unknown", "active", "visible in current chunk")
    if was_visible and missing_chunks <= occlusion_memory_max_chunks:
        return LifecycleDecision(global_id, "active", "occluded", "within occlusion memory")
    return LifecycleDecision(global_id, "occluded", "retired", "occlusion memory expired")

