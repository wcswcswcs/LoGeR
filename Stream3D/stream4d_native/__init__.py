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
from .d4rt_scene_builder import D4RTNativeSceneBuilder, stable_source_carrier_id
from .measurement_bank import MaskMeasurement, build_measurement_bank, count_pair_measurement_evidence
from .object_tube_io import MergeGeometryError, TubeRecord, assert_merge_geometry_valid
from .sim3 import (
    Sim3Transform,
    apply_sim3_to_xyz,
    compose_sim3,
    estimate_overlap_sim3,
    invert_sim3,
)
from .signed_tube_graph import TubeGraphEdge, build_signed_tube_graph
from .tube_cover import TubeCoverResult, select_tube_cover
from .tube_memory import TubeMemory
from .tube_partition import (
    TubePartitionResult,
    filter_edges_by_mutual_topk,
    filter_edges_by_min_score,
    filter_edges_by_pair_evidence,
    partition_tube_graph,
)

__all__ = [
    "ChunkPolicy",
    "D4RTNativeSceneBuilder",
    "MaskMeasurement",
    "MergeGeometryError",
    "Sim3Transform",
    "TubeCoverResult",
    "TubeGraphEdge",
    "TubeMemory",
    "TubePartitionResult",
    "TubeRecord",
    "apply_sim3_to_xyz",
    "assert_merge_geometry_valid",
    "build_measurement_bank",
    "build_checkpoint_chunk_policy",
    "build_signed_tube_graph",
    "compose_sim3",
    "estimate_overlap_sim3",
    "count_pair_measurement_evidence",
    "filter_edges_by_pair_evidence",
    "filter_edges_by_mutual_topk",
    "filter_edges_by_min_score",
    "invert_sim3",
    "make_sliding_window_clip_ranges",
    "partition_tube_graph",
    "read_checkpoint_clip_frames",
    "select_tube_cover",
    "stable_source_carrier_id",
]
