from __future__ import annotations

from dataclasses import dataclass

from .config import V106Config


@dataclass(frozen=True)
class ChunkInitializationDecision:
    chunk_index: int
    mode: str
    full_reinitialize_allowed: bool
    reason: str


def decide_chunk_initialization(config: V106Config, chunk_index: int) -> ChunkInitializationDecision:
    if chunk_index == 0:
        return ChunkInitializationDecision(
            chunk_index=chunk_index,
            mode="baseline_x_full_stage1_stage2",
            full_reinitialize_allowed=True,
            reason="chunk0 is the only chunk allowed to build objectlets from scratch",
        )
    if config.local_exact.later_chunk_full_reinitialize:
        raise RuntimeError("later_chunk_full_reinitialize is forbidden by v106")
    return ChunkInitializationDecision(
        chunk_index=chunk_index,
        mode="inherited_masks_plus_exact_gap_birth",
        full_reinitialize_allowed=False,
        reason="later chunks must inherit scene state and only use exact residual birth",
    )

