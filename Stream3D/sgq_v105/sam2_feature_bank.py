"""SAM2 frame feature bank utilities for Stream4D v105.

The bank owns immutable per-frame SAM2 backbone outputs and binds them back into
the official SAM2 image/video predictor APIs. It does not change prompts, masks,
object IDs, logits, or SAM2 decoder code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def _tensor_nbytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _tree_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_nbytes(value)
    if isinstance(value, dict):
        return sum(_tree_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(v) for v in value)
    return 0


def _tree_metadata(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "nbytes": _tensor_nbytes(value),
        }
    if isinstance(value, dict):
        return {str(k): _tree_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tree_metadata(v) for v in value]
    return str(type(value).__name__)


def _move_tree(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(v, device) for v in value)
    return value


def _clone_detached_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_detached_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_detached_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_detached_tree(v) for v in value)
    return value


@dataclass
class Sam2FrameFeatureRecord:
    frame_id: int
    chunk_index: int
    orig_hw: tuple[int, int]
    input_image: torch.Tensor
    backbone_out: dict[str, Any]
    source: str
    build_runtime_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "chunk_index": int(self.chunk_index),
            "orig_hw": [int(self.orig_hw[0]), int(self.orig_hw[1])],
            "source": self.source,
            "build_runtime_sec": float(self.build_runtime_sec),
            "input_image": _tree_metadata(self.input_image),
            "backbone_out": _tree_metadata(self.backbone_out),
            "metadata": self.metadata,
        }


class Sam2FrameFeatureBank:
    """Owns one SAM2 backbone output per frame and exposes official API bindings."""

    def __init__(self, *, storage_device: str = "cuda", clone_tensors: bool = True) -> None:
        self.storage_device = torch.device(storage_device)
        self.clone_tensors = bool(clone_tensors)
        self.records_by_frame_id: dict[int, Sam2FrameFeatureRecord] = {}
        self.records_by_chunk_index: dict[int, Sam2FrameFeatureRecord] = {}
        self.sam2_backbone_forward_count = 0
        self.expected_backbone_forward_count = 0
        self.feature_bank_hit_count = 0
        self.feature_bank_miss_count = 0
        self.feature_bank_prefetch_wait_sec = 0.0
        self.feature_bank_h2d_bytes = 0
        self.build_runtime_sec = 0.0

    def _store_record(self, record: Sam2FrameFeatureRecord) -> None:
        self.records_by_frame_id[int(record.frame_id)] = record
        self.records_by_chunk_index[int(record.chunk_index)] = record

    def build_for_image_predictor(
        self,
        predictor: Any,
        *,
        frame_ids: Iterable[int],
        rgb_frames: Iterable[np.ndarray],
    ) -> None:
        """Build features using SAM2ImagePredictor's transform path."""
        t_bank = time.time()
        frame_list = list(frame_ids)
        rgb_list = list(rgb_frames)
        self.expected_backbone_forward_count = len(frame_list)
        for chunk_index, (frame_id, rgb) in enumerate(zip(frame_list, rgb_list, strict=True)):
            if not isinstance(rgb, np.ndarray):
                raise TypeError(f"rgb frame must be numpy ndarray, got {type(rgb)!r}")
            orig_hw = tuple(int(v) for v in rgb.shape[:2])
            input_image = predictor._transforms(rgb)[None, ...].to(predictor.device)
            t0 = time.time()
            with torch.no_grad():
                backbone_out = predictor.model.forward_image(input_image)
            self.sam2_backbone_forward_count += 1
            if self.clone_tensors:
                input_store = input_image.detach().clone().to(self.storage_device)
                backbone_store = _move_tree(_clone_detached_tree(backbone_out), self.storage_device)
            else:
                input_store = input_image.detach().to(self.storage_device)
                backbone_store = _move_tree(backbone_out, self.storage_device)
            record = Sam2FrameFeatureRecord(
                frame_id=int(frame_id),
                chunk_index=int(chunk_index),
                orig_hw=(int(orig_hw[0]), int(orig_hw[1])),
                input_image=input_store,
                backbone_out=backbone_store,
                source="image_predictor_transform",
                build_runtime_sec=float(time.time() - t0),
                metadata={
                    "model_class": type(predictor.model).__name__,
                    "transform_class": type(predictor._transforms).__name__,
                    "image_size": int(predictor.model.image_size),
                },
            )
            self._store_record(record)
        self.build_runtime_sec += time.time() - t_bank

    def build_for_video_paths(
        self,
        predictor: Any,
        *,
        frame_ids: Iterable[int],
        frame_paths: Iterable[Path],
    ) -> None:
        """Build features using SAM2VideoPredictor's JPEG loader transform path."""
        from sam2.utils.misc import _load_img_as_tensor

        t_bank = time.time()
        frame_list = list(frame_ids)
        path_list = [Path(p) for p in frame_paths]
        self.expected_backbone_forward_count = len(frame_list)
        for chunk_index, (frame_id, frame_path) in enumerate(zip(frame_list, path_list, strict=True)):
            img, video_height, video_width = _load_img_as_tensor(frame_path, predictor.image_size)
            img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
            img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
            img = (img - img_mean) / img_std
            input_image = img.to(predictor.device).float().unsqueeze(0)
            t0 = time.time()
            with torch.no_grad():
                backbone_out = predictor.forward_image(input_image)
            self.sam2_backbone_forward_count += 1
            if self.clone_tensors:
                input_store = input_image.detach().clone().to(self.storage_device)
                backbone_store = _move_tree(_clone_detached_tree(backbone_out), self.storage_device)
            else:
                input_store = input_image.detach().to(self.storage_device)
                backbone_store = _move_tree(backbone_out, self.storage_device)
            record = Sam2FrameFeatureRecord(
                frame_id=int(frame_id),
                chunk_index=int(chunk_index),
                orig_hw=(int(video_height), int(video_width)),
                input_image=input_store,
                backbone_out=backbone_store,
                source="video_predictor_jpeg_loader",
                build_runtime_sec=float(time.time() - t0),
                metadata={
                    "model_class": type(predictor).__name__,
                    "image_size": int(predictor.image_size),
                    "frame_path": str(frame_path),
                },
            )
            self._store_record(record)
        self.build_runtime_sec += time.time() - t_bank

    def get_image_features(self, frame_id: int) -> Sam2FrameFeatureRecord:
        record = self.records_by_frame_id.get(int(frame_id))
        if record is None:
            self.feature_bank_miss_count += 1
            raise KeyError(f"feature bank missing frame_id={frame_id}")
        self.feature_bank_hit_count += 1
        return record

    def get_chunk_features(self, chunk_index: int) -> Sam2FrameFeatureRecord:
        record = self.records_by_chunk_index.get(int(chunk_index))
        if record is None:
            self.feature_bank_miss_count += 1
            raise KeyError(f"feature bank missing chunk_index={chunk_index}")
        self.feature_bank_hit_count += 1
        return record

    def bind_image_predictor(self, predictor: Any, frame_id: int) -> None:
        """Bind bank features into SAM2ImagePredictor without running forward_image."""
        record = self.get_image_features(int(frame_id))
        predictor.reset_predictor()
        predictor._orig_hw = [tuple(record.orig_hw)]
        backbone_out = _move_tree(record.backbone_out, predictor.device)
        _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
        if predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
        ][::-1]
        predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        predictor._is_image_set = True
        predictor._is_batch = False

    def bind_video_state(self, inference_state: dict[str, Any]) -> None:
        """Populate a SAM2VideoPredictor state cache with bank features by chunk index."""
        device = inference_state.get("device", self.storage_device)
        cache: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
        for chunk_index, record in self.records_by_chunk_index.items():
            image = record.input_image.to(device)
            backbone_out = _move_tree(record.backbone_out, device)
            self.feature_bank_h2d_bytes += _tree_nbytes(image) + _tree_nbytes(backbone_out)
            cache[int(chunk_index)] = (image, backbone_out)
        inference_state["cached_features"] = cache

    def summary(self) -> dict[str, Any]:
        cpu_bytes = 0
        gpu_bytes = 0
        for record in self.records_by_chunk_index.values():
            nbytes = _tree_nbytes(record.input_image) + _tree_nbytes(record.backbone_out)
            device_text = str(record.input_image.device)
            if device_text.startswith("cuda"):
                gpu_bytes += nbytes
            else:
                cpu_bytes += nbytes
        return {
            "schema_version": "stream4d_v105_sam2_frame_feature_bank_summary_v1",
            "record_count": int(len(self.records_by_chunk_index)),
            "sam2_backbone_forward_count": int(self.sam2_backbone_forward_count),
            "expected_backbone_forward_count": int(self.expected_backbone_forward_count),
            "feature_bank_hit_count": int(self.feature_bank_hit_count),
            "feature_bank_miss_count": int(self.feature_bank_miss_count),
            "feature_bank_prefetch_wait_sec": float(self.feature_bank_prefetch_wait_sec),
            "feature_bank_cpu_bytes": int(cpu_bytes),
            "feature_bank_gpu_bytes": int(gpu_bytes),
            "feature_bank_h2d_bytes": int(self.feature_bank_h2d_bytes),
            "build_runtime_sec": float(self.build_runtime_sec),
            "storage_device": str(self.storage_device),
            "clone_tensors": bool(self.clone_tensors),
            "records": [record.to_json() for _, record in sorted(self.records_by_chunk_index.items())],
        }
