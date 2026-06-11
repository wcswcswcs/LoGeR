from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .chunk_alignment import ChunkPolicy, build_checkpoint_chunk_policy, make_sliding_window_clip_ranges
from .occupancy_dense_tracker import QueryBudget, query_d4rt_tubes_with_spatiotemporal_occupancy
from .occupancy_state import OccupancyCoverageTargets
from .sim3 import estimate_overlap_sim3


class D4RTNativeSceneBuilder:
    """Checkpoint-aware D4RT-native tube scene builder.

    This class deliberately accepts RGB frames, prepared 2D masks and a frozen
    D4RT runner. It does not import ScanNet depth, pose, mesh or GT helpers.
    """

    def __init__(
        self,
        d4rt_model: Any,
        checkpoint_config: str | dict[str, Any],
        *,
        temporal_chunk_size: int | None = None,
        temporal_chunk_stride: int | None = None,
        query_batch_size: int = 2048,
    ) -> None:
        self.d4rt_model = d4rt_model
        self.policy: ChunkPolicy = build_checkpoint_chunk_policy(
            checkpoint_config,
            temporal_chunk_size=temporal_chunk_size,
            temporal_chunk_stride=temporal_chunk_stride,
            full_scene_method=True,
        )
        self.clip_frames = int(self.policy.clip_frames)
        self.temporal_chunk_size = int(self.policy.temporal_chunk_size)
        self.temporal_chunk_stride = int(self.policy.temporal_chunk_stride)
        self.temporal_chunk_overlap = int(self.policy.temporal_chunk_overlap)
        self.query_batch_size = int(query_batch_size)
        if self.temporal_chunk_size > self.clip_frames:
            raise ValueError("temporal_chunk_size must not exceed checkpoint clip_frames")
        if self.temporal_chunk_overlap <= 0:
            raise ValueError("full-scene method requires overlapping chunks")

    def build_chunks(self, rgb_video: np.ndarray) -> list[dict[str, int]]:
        rgb_video = np.asarray(rgb_video)
        ranges = make_sliding_window_clip_ranges(
            num_frames=int(rgb_video.shape[0]),
            clip_frames=self.temporal_chunk_size,
            stride=self.temporal_chunk_stride,
        )
        return [
            {"chunk_id": int(idx), "start": int(start), "end": int(end), "num_frames": int(end - start)}
            for idx, (start, end) in enumerate(ranges)
        ]

    def _decode_source_points(self, frames: np.ndarray, source_points: np.ndarray) -> list[dict[str, Any]]:
        if not hasattr(self.d4rt_model, "infer_carriers"):
            raise RuntimeError("d4rt_model must expose infer_carriers for native tube extraction")
        src_frame = source_points[:, 0].astype(np.int64)
        src_uv = source_points[:, 1:3].astype(np.float32)
        batch = self.d4rt_model.infer_carriers(
            video_rgb_uint8=frames,
            src_uv_norm=src_uv,
            src_frame_local=src_frame,
            query_chunk_size=self.query_batch_size,
        )
        tubes: list[dict[str, Any]] = []
        for idx in range(src_uv.shape[0]):
            tubes.append(
                {
                    "uv_norm": np.asarray(batch.uv_pred[:, idx, :], dtype=np.float32),
                    "xyz": np.asarray(batch.xyz_ref[:, idx, :], dtype=np.float32),
                    "visibility": np.asarray(batch.visibility_prob[:, idx], dtype=np.float32),
                    "confidence": np.asarray(batch.confidence_prob[:, idx], dtype=np.float32),
                    "valid": np.asarray(batch.valid[:, idx], dtype=bool),
                    "source_frame_local": int(src_frame[idx]),
                }
            )
        return tubes

    def extract_local_tubes_with_occupancy(
        self,
        rgb_video: np.ndarray,
        masks_by_frame: np.ndarray | None,
        chunk: dict[str, int],
        *,
        coverage_targets: OccupancyCoverageTargets | None = None,
        query_budget: QueryBudget | None = None,
    ) -> dict[str, Any]:
        start = int(chunk["start"])
        end = int(chunk["end"])
        frames = np.asarray(rgb_video[start:end])
        masks = None if masks_by_frame is None else np.asarray(masks_by_frame[start:end])
        tubes, diagnostics = query_d4rt_tubes_with_spatiotemporal_occupancy(
            frames=frames,
            masks=masks,
            decode_source_points=lambda source_points: self._decode_source_points(frames, source_points),
            coverage_targets=coverage_targets,
            query_budget=query_budget,
        )
        return {"chunk": dict(chunk), "tubes": tubes, "diagnostics": diagnostics}

    def estimate_overlap_self_sim3(self, previous_chunk: dict[str, Any], current_chunk: dict[str, Any]) -> dict[str, Any] | None:
        prev_tubes = previous_chunk.get("tubes", [])
        curr_tubes = current_chunk.get("tubes", [])
        if not prev_tubes or not curr_tubes:
            return None
        count = min(len(prev_tubes), len(curr_tubes))
        prev_xyz = np.stack([np.asarray(prev_tubes[i]["xyz"], dtype=np.float32) for i in range(count)], axis=0)
        curr_xyz = np.stack([np.asarray(curr_tubes[i]["xyz"], dtype=np.float32) for i in range(count)], axis=0)
        prev_vis = np.stack([np.asarray(prev_tubes[i].get("valid", np.ones(prev_xyz.shape[1], dtype=bool)), dtype=bool) for i in range(count)], axis=0)
        curr_vis = np.stack([np.asarray(curr_tubes[i].get("valid", np.ones(curr_xyz.shape[1], dtype=bool)), dtype=bool) for i in range(count)], axis=0)
        return estimate_overlap_sim3(prev_xyz, curr_xyz, prev_vis, curr_vis)

    def stitch_to_canonical(self, local_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        stitched: list[dict[str, Any]] = []
        weak_alignment = 0
        pairwise: list[dict[str, Any] | None] = []
        for idx, chunk in enumerate(local_chunks):
            if idx == 0:
                stitched.append(chunk)
                continue
            fit = self.estimate_overlap_self_sim3(stitched[-1], chunk)
            pairwise.append(fit)
            if fit is None:
                weak_alignment += 1
            stitched.append({**chunk, "self_sim3_to_previous": fit, "weak_alignment": fit is None})
        return {
            "chunks": stitched,
            "diagnostics": {
                "num_chunks": int(len(local_chunks)),
                "weak_alignment_chunk_count": int(weak_alignment),
                "chunk_policy": asdict(self.policy),
                "pairwise_self_sim3": pairwise,
            },
        }

    def build_scene_tubes(
        self,
        rgb_video: np.ndarray,
        masks_by_frame: np.ndarray | None,
        *,
        coverage_targets: OccupancyCoverageTargets | None = None,
        query_budget: QueryBudget | None = None,
    ) -> dict[str, Any]:
        chunks = self.build_chunks(rgb_video)
        local_chunks = [
            self.extract_local_tubes_with_occupancy(
                rgb_video,
                masks_by_frame,
                chunk,
                coverage_targets=coverage_targets,
                query_budget=query_budget,
            )
            for chunk in chunks
        ]
        return self.stitch_to_canonical(local_chunks)
