from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .chunk_alignment import ChunkPolicy, build_checkpoint_chunk_policy, make_sliding_window_clip_ranges
from .occupancy_dense_tracker import QueryBudget, query_d4rt_tubes_with_spatiotemporal_occupancy
from .occupancy_state import OccupancyCoverageTargets
from .self_stitch import fit_sim3_with_diagnostics, match_overlap_carriers
from .sim3 import Sim3Transform, apply_sim3_to_xyz


def stable_source_carrier_id(frame_global: int, x: int, y: int, image_width: int) -> int:
    width = max(int(image_width), 1)
    return int(frame_global) * 10_000_000_000 + int(y) * width + int(x)


def source_xy_from_uv(source_uv: np.ndarray, *, image_width: int, image_height: int) -> tuple[int, int]:
    uv = np.asarray(source_uv, dtype=np.float64).reshape(2)
    width = max(int(image_width), 1)
    height = max(int(image_height), 1)
    x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, height - 1))
    return x, y


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

    def _decode_source_points(
        self,
        frames: np.ndarray,
        source_points: np.ndarray,
        *,
        frame_start: int = 0,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[dict[str, Any]]:
        if not hasattr(self.d4rt_model, "infer_carriers"):
            raise RuntimeError("d4rt_model must expose infer_carriers for native tube extraction")
        frames = np.asarray(frames)
        if frames.ndim >= 3:
            inferred_height = int(frames.shape[1])
            inferred_width = int(frames.shape[2])
        else:
            inferred_height = 1
            inferred_width = 1
        width = int(image_width if image_width is not None else inferred_width)
        height = int(image_height if image_height is not None else inferred_height)
        src_frame = source_points[:, 0].astype(np.int64)
        src_uv = source_points[:, 1:3].astype(np.float32)
        batch = self.d4rt_model.infer_carriers(
            video_rgb_uint8=frames,
            src_uv_norm=src_uv,
            src_frame_local=src_frame,
            query_chunk_size=self.query_batch_size,
        )
        tubes: list[dict[str, Any]] = []
        carrier_id = getattr(batch, "carrier_id", None)
        src_frame_global = getattr(batch, "src_frame_global", None)
        src_xy = getattr(batch, "src_xy", None)
        persistent_tube_id = getattr(batch, "persistent_tube_id", None)
        xyz_local = getattr(batch, "xyz_local", None)
        for idx in range(src_uv.shape[0]):
            global_frame = (
                int(frame_start) + int(src_frame[idx])
                if src_frame_global is None
                else int(np.asarray(src_frame_global)[idx])
            )
            source_xy = (
                source_xy_from_uv(src_uv[idx], image_width=width, image_height=height)
                if src_xy is None
                else tuple(int(v) for v in np.asarray(src_xy)[idx].tolist())
            )
            stable_id = stable_source_carrier_id(global_frame, int(source_xy[0]), int(source_xy[1]), width)
            tube_carrier_id = stable_id if carrier_id is None else int(np.asarray(carrier_id)[idx])
            tube_persistent_id = stable_id if persistent_tube_id is None else int(np.asarray(persistent_tube_id)[idx])
            tubes.append(
                {
                    "carrier_id": int(tube_carrier_id),
                    "persistent_tube_id": int(tube_persistent_id),
                    "uv_norm": np.asarray(batch.uv_pred[:, idx, :], dtype=np.float32),
                    "xyz": np.asarray(batch.xyz_ref[:, idx, :], dtype=np.float32),
                    "xyz_ref0": np.asarray(batch.xyz_ref[:, idx, :], dtype=np.float32),
                    "xyz_local": None if xyz_local is None else np.asarray(xyz_local[:, idx, :], dtype=np.float32),
                    "visibility": np.asarray(batch.visibility_prob[:, idx], dtype=np.float32),
                    "confidence": np.asarray(batch.confidence_prob[:, idx], dtype=np.float32),
                    "valid": np.asarray(batch.valid[:, idx], dtype=bool),
                    "source_frame_local": int(src_frame[idx]),
                    "source_frame_global": global_frame,
                    "source_xy": source_xy,
                    "source_uv": tuple(float(v) for v in src_uv[idx].tolist()),
                    "source_pixel_key": f"{global_frame}:{int(source_xy[0])}:{int(source_xy[1])}",
                    "source_identity_from_fallback": bool(carrier_id is None or persistent_tube_id is None or src_xy is None),
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
            decode_source_points=lambda source_points: self._decode_source_points(
                frames,
                source_points,
                frame_start=start,
            ),
            coverage_targets=coverage_targets,
            query_budget=query_budget,
        )
        return {"chunk": dict(chunk), "tubes": tubes, "diagnostics": diagnostics}

    def _chunk_frame_ids(self, chunk: dict[str, Any], tube_frames: int) -> list[int]:
        meta = chunk.get("chunk", chunk)
        if "frame_ids" in meta:
            values = [int(v) for v in meta["frame_ids"]]
            if len(values) == int(tube_frames):
                return values
        start = int(meta.get("start", 0))
        end = int(meta.get("end", start + int(tube_frames)))
        values = list(range(start, end))
        if len(values) != int(tube_frames):
            values = [start + idx for idx in range(int(tube_frames))]
        return values

    def _chunk_to_window_data(self, chunk: dict[str, Any], *, use_canonical_xyz: bool) -> dict[str, np.ndarray]:
        tubes = list(chunk.get("tubes", []))
        if not tubes:
            return {
                "frame_ids": np.asarray([], dtype=np.int64),
                "xyz": np.empty((0, 0, 3), dtype=np.float32),
                "uv": np.empty((0, 0, 2), dtype=np.float32),
                "valid": np.empty((0, 0), dtype=bool),
                "visibility": np.empty((0, 0), dtype=np.float32),
                "confidence": np.empty((0, 0), dtype=np.float32),
                "carrier_id": np.empty((0,), dtype=np.int64),
                "persistent_tube_id": np.empty((0,), dtype=np.int64),
                "src_frame_global": np.empty((0,), dtype=np.int64),
                "src_xy": np.empty((0, 2), dtype=np.int64),
            }
        xyz_items: list[np.ndarray] = []
        uv_items: list[np.ndarray] = []
        valid_items: list[np.ndarray] = []
        visibility_items: list[np.ndarray] = []
        confidence_items: list[np.ndarray] = []
        carrier_ids: list[int] = []
        persistent_ids: list[int] = []
        src_frames: list[int] = []
        src_xy_values: list[tuple[int, int]] = []
        for idx, tube in enumerate(tubes):
            xyz_key = "xyz_canonical" if use_canonical_xyz and tube.get("xyz_canonical") is not None else "xyz"
            xyz = np.asarray(tube.get(xyz_key, tube.get("xyz_ref0", tube.get("xyz_local"))), dtype=np.float32)
            if xyz.ndim != 2 or xyz.shape[-1] != 3:
                raise ValueError(f"tube {idx} xyz must have shape [T,3], got {xyz.shape}")
            uv = np.asarray(tube.get("uv_norm", tube.get("uv")), dtype=np.float32)
            if uv.ndim != 2 or uv.shape[-1] != 2:
                raise ValueError(f"tube {idx} uv must have shape [T,2], got {uv.shape}")
            valid = np.asarray(tube.get("valid", np.ones((xyz.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
            visibility = np.asarray(tube.get("visibility", np.ones((xyz.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
            confidence = np.asarray(tube.get("confidence", np.ones((xyz.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if not (uv.shape[0] == xyz.shape[0] == valid.shape[0] == visibility.shape[0] == confidence.shape[0]):
                raise ValueError(f"tube {idx} temporal fields have inconsistent lengths")
            xyz_items.append(xyz)
            uv_items.append(uv)
            valid_items.append(valid)
            visibility_items.append(visibility)
            confidence_items.append(confidence)
            local_source = int(tube.get("source_frame_local", 0))
            meta = chunk.get("chunk", chunk)
            global_source = int(tube.get("source_frame_global", int(meta.get("start", 0)) + local_source))
            src_frames.append(global_source)
            xy = tube.get("source_xy", (-1, -1))
            source_xy = (int(xy[0]), int(xy[1]))
            src_xy_values.append(source_xy)
            if "carrier_id" in tube:
                carrier_ids.append(int(tube["carrier_id"]))
            elif source_xy[0] >= 0 and source_xy[1] >= 0:
                carrier_ids.append(stable_source_carrier_id(global_source, source_xy[0], source_xy[1], 1_000_000))
            else:
                carrier_ids.append(-1)
            if "persistent_tube_id" in tube:
                persistent_ids.append(int(tube["persistent_tube_id"]))
            elif source_xy[0] >= 0 and source_xy[1] >= 0:
                persistent_ids.append(stable_source_carrier_id(global_source, source_xy[0], source_xy[1], 1_000_000))
            else:
                persistent_ids.append(-1)
        first_t = int(xyz_items[0].shape[0])
        return {
            "frame_ids": np.asarray(self._chunk_frame_ids(chunk, first_t), dtype=np.int64),
            "xyz": np.stack(xyz_items, axis=1).astype(np.float32),
            "uv": np.stack(uv_items, axis=1).astype(np.float32),
            "valid": np.stack(valid_items, axis=1).astype(bool),
            "visibility": np.stack(visibility_items, axis=1).astype(np.float32),
            "confidence": np.stack(confidence_items, axis=1).astype(np.float32),
            "carrier_id": np.asarray(carrier_ids, dtype=np.int64),
            "persistent_tube_id": np.asarray(persistent_ids, dtype=np.int64),
            "src_frame_global": np.asarray(src_frames, dtype=np.int64),
            "src_xy": np.asarray(src_xy_values, dtype=np.int64),
        }

    def estimate_overlap_self_sim3(
        self,
        previous_chunk: dict[str, Any],
        current_chunk: dict[str, Any],
        *,
        min_points: int = 4,
        min_inlier_abs010: float = 0.50,
    ) -> dict[str, Any] | None:
        if not previous_chunk.get("tubes") or not current_chunk.get("tubes"):
            return None
        prev_window = self._chunk_to_window_data(previous_chunk, use_canonical_xyz=True)
        curr_window = self._chunk_to_window_data(current_chunk, use_canonical_xyz=False)
        match = match_overlap_carriers(prev_window, curr_window)
        if int(match.stats.get("used_anchor_count", 0)) < int(min_points):
            return {
                "pass_gate": False,
                "fail_reason": "insufficient_overlap_anchors",
                "anchor_count": int(match.stats.get("used_anchor_count", 0)),
                "match_stats": match.stats,
                "alignment_source": "d4rt_self_sim3",
            }
        try:
            fit = fit_sim3_with_diagnostics(
                match.curr_xyz.reshape(-1, 3),
                match.prev_xyz.reshape(-1, 3),
            )
        except ValueError as exc:
            return {
                "pass_gate": False,
                "fail_reason": f"sim3_fit_failed:{exc}",
                "anchor_count": int(match.stats.get("used_anchor_count", 0)),
                "match_stats": match.stats,
                "alignment_source": "d4rt_self_sim3",
            }
        inlier = float(fit.get("inlier_ratio_abs010") or 0.0)
        fit["pass_gate"] = bool(int(fit.get("anchor_count", 0)) >= int(min_points) and inlier >= float(min_inlier_abs010))
        fit["fail_reason"] = None if fit["pass_gate"] else "alignment_quality_gate_failed"
        fit["match_stats"] = match.stats
        fit["alignment_source"] = "d4rt_self_sim3"
        return fit

    def _policy_dict(self) -> dict[str, Any]:
        if hasattr(self, "policy"):
            try:
                return asdict(self.policy)
            except TypeError:
                return dict(getattr(self.policy, "__dict__", {}))
        return {}

    def _apply_chunk_canonical_transform(
        self,
        chunk: dict[str, Any],
        *,
        transform: Sim3Transform,
        transform_id: str,
        submap_id: int,
        alignment_quality: dict[str, Any],
        alignment_source: str,
        allow_metric_merge: bool,
        weak_alignment: bool,
    ) -> dict[str, Any]:
        out = {**chunk}
        transform_payload = {
            "scale": float(transform.scale),
            "rot": np.asarray(transform.rot, dtype=np.float64),
            "trans": np.asarray(transform.trans, dtype=np.float64),
            "transform_id": str(transform_id),
            "submap_id": int(submap_id),
        }
        tubes: list[dict[str, Any]] = []
        for tube in chunk.get("tubes", []):
            item = dict(tube)
            xyz = np.asarray(item.get("xyz", item.get("xyz_ref0", item.get("xyz_local"))), dtype=np.float32)
            item["xyz_canonical"] = apply_sim3_to_xyz(xyz, transform=transform)
            item["T_chunk_to_canonical"] = transform_payload
            item["submap_id"] = int(submap_id)
            item["alignment_quality"] = dict(alignment_quality)
            item["coordinate_frame"] = "d4rt_canonical"
            item["scale_status"] = "canonical" if allow_metric_merge else "submap_unaligned"
            item["allow_metric_merge"] = bool(allow_metric_merge)
            item["alignment_source"] = str(alignment_source)
            item["transform_id"] = str(transform_id)
            tubes.append(item)
        out["tubes"] = tubes
        out["submap_id"] = int(submap_id)
        out["T_chunk_to_canonical"] = transform_payload
        out["alignment_quality"] = dict(alignment_quality)
        out["alignment_source"] = str(alignment_source)
        out["allow_metric_merge"] = bool(allow_metric_merge)
        out["weak_alignment"] = bool(weak_alignment)
        out["transform_id"] = str(transform_id)
        return out

    def stitch_to_canonical(self, local_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        stitched: list[dict[str, Any]] = []
        weak_alignment = 0
        pairwise: list[dict[str, Any] | None] = []
        for idx, chunk in enumerate(local_chunks):
            identity = Sim3Transform(scale=1.0, rot=np.eye(3, dtype=np.float64), trans=np.zeros((3,), dtype=np.float64))
            if idx == 0:
                stitched.append(
                    self._apply_chunk_canonical_transform(
                        chunk,
                        transform=identity,
                        transform_id="chunk000_identity",
                        submap_id=0,
                        alignment_quality={"pass_gate": True, "anchor_count": None, "fail_reason": None},
                        alignment_source="same_chunk_identity",
                        allow_metric_merge=True,
                        weak_alignment=False,
                    )
                )
                continue
            fit = self.estimate_overlap_self_sim3(stitched[-1], chunk)
            pairwise.append(fit)
            if fit is None or not bool(fit.get("pass_gate", False)):
                weak_alignment += 1
                submap_id = int(stitched[-1].get("submap_id", -1)) + 1
                stitched.append(
                    self._apply_chunk_canonical_transform(
                        {**chunk, "self_sim3_to_previous": fit},
                        transform=identity,
                        transform_id=f"chunk{idx:03d}_submap_identity",
                        submap_id=submap_id,
                        alignment_quality={
                            "pass_gate": False,
                            "anchor_count": None if fit is None else int(fit.get("anchor_count", 0)),
                            "fail_reason": "missing_fit" if fit is None else fit.get("fail_reason"),
                        },
                        alignment_source="submap_identity_after_failed_self_sim3",
                        allow_metric_merge=False,
                        weak_alignment=True,
                    )
                )
                continue
            transform = Sim3Transform(
                scale=float(fit["scale"]),
                rot=np.asarray(fit["rot"], dtype=np.float64),
                trans=np.asarray(fit["trans"], dtype=np.float64),
            )
            alignment_quality = {
                "pass_gate": True,
                "anchor_count": int(fit.get("anchor_count", 0)),
                "used_anchor_count": int(fit.get("match_stats", {}).get("used_anchor_count", fit.get("anchor_count", 0))),
                "residual_median": fit.get("residual_median"),
                "residual_p90": fit.get("residual_p90"),
                "inlier_ratio_abs010": fit.get("inlier_ratio_abs010"),
                "fail_reason": None,
            }
            stitched.append(
                self._apply_chunk_canonical_transform(
                    {**chunk, "self_sim3_to_previous": fit},
                    transform=transform,
                    transform_id=f"chunk{idx:03d}_self_sim3",
                    submap_id=int(stitched[-1].get("submap_id", 0)),
                    alignment_quality=alignment_quality,
                    alignment_source="d4rt_self_sim3",
                    allow_metric_merge=True,
                    weak_alignment=False,
                )
            )
        return {
            "chunks": stitched,
            "diagnostics": {
                "num_chunks": int(len(local_chunks)),
                "weak_alignment_chunk_count": int(weak_alignment),
                "submap_count": int(len({int(chunk.get("submap_id", 0)) for chunk in stitched})),
                "canonicalized_chunk_count": int(sum(1 for chunk in stitched if chunk.get("tubes"))),
                "chunk_policy": self._policy_dict(),
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
