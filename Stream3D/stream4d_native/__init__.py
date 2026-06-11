"""D4RT-native scene building utilities for Stream4D v21.3.

This package is intentionally separate from the older ``stream4d`` package:
native prediction code here must not import ScanNet RGB-D, pose, mesh, or GT
helpers. Evaluation-only adapters live outside this package.
"""

from .chunk_alignment import (
    ChunkPolicy,
    build_checkpoint_chunk_policy,
    make_sliding_window_clip_ranges,
    read_checkpoint_clip_frames,
)
from .d4rt_scene_builder import D4RTNativeSceneBuilder
from .sim3 import (
    Sim3Transform,
    apply_sim3_to_xyz,
    compose_sim3,
    estimate_overlap_sim3,
    invert_sim3,
)

__all__ = [
    "ChunkPolicy",
    "D4RTNativeSceneBuilder",
    "Sim3Transform",
    "apply_sim3_to_xyz",
    "build_checkpoint_chunk_policy",
    "compose_sim3",
    "estimate_overlap_sim3",
    "invert_sim3",
    "make_sliding_window_clip_ranges",
    "read_checkpoint_clip_frames",
]
