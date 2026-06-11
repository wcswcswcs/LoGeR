from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evidence_terms import ExplanationParams, birth_groups
from .measurement_bank import MeasurementBank, json_safe


@dataclass
class VideoMaskletBank:
    scene: str
    mode: str
    frame_ids: np.ndarray
    masklet_id: np.ndarray
    birth_frame: np.ndarray
    birth_mask_id: np.ndarray
    frame_id: np.ndarray
    offset: np.ndarray
    surfel_indices_flat: np.ndarray
    confidence: np.ndarray
    compactness: np.ndarray
    area_growth_ratio: np.ndarray
    cycle_error_p50: np.ndarray
    cycle_error_p90: np.ndarray
    appearance_drift: np.ndarray
    available_mask_agreement_iou: np.ndarray
    negative_visible_outside_ratio: np.ndarray
    meta: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene=np.asarray(self.scene),
            mode=np.asarray(self.mode),
            frame_ids=self.frame_ids,
            masklet_id=self.masklet_id,
            birth_frame=self.birth_frame,
            birth_mask_id=self.birth_mask_id,
            frame_id=self.frame_id,
            offset=self.offset,
            surfel_indices_flat=self.surfel_indices_flat,
            confidence=self.confidence,
            compactness=self.compactness,
            area_growth_ratio=self.area_growth_ratio,
            cycle_error_p50=self.cycle_error_p50,
            cycle_error_p90=self.cycle_error_p90,
            appearance_drift=self.appearance_drift,
            available_mask_agreement_iou=self.available_mask_agreement_iou,
            negative_visible_outside_ratio=self.negative_visible_outside_ratio,
            meta_json=np.asarray(json.dumps(json_safe(self.meta), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> "VideoMaskletBank":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                scene=str(data["scene"].item()),
                mode=str(data["mode"].item()),
                frame_ids=np.asarray(data["frame_ids"], dtype=np.int64),
                masklet_id=np.asarray(data["masklet_id"], dtype=np.int64),
                birth_frame=np.asarray(data["birth_frame"], dtype=np.int64),
                birth_mask_id=np.asarray(data["birth_mask_id"], dtype=np.int64),
                frame_id=np.asarray(data["frame_id"], dtype=np.int64),
                offset=np.asarray(data["offset"], dtype=np.int64),
                surfel_indices_flat=np.asarray(data["surfel_indices_flat"], dtype=np.int64),
                confidence=np.asarray(data["confidence"], dtype=np.float32),
                compactness=np.asarray(data["compactness"], dtype=np.float32),
                area_growth_ratio=np.asarray(data["area_growth_ratio"], dtype=np.float32),
                cycle_error_p50=np.asarray(data["cycle_error_p50"], dtype=np.float32),
                cycle_error_p90=np.asarray(data["cycle_error_p90"], dtype=np.float32),
                appearance_drift=np.asarray(data["appearance_drift"], dtype=np.float32),
                available_mask_agreement_iou=np.asarray(data["available_mask_agreement_iou"], dtype=np.float32),
                negative_visible_outside_ratio=np.asarray(data["negative_visible_outside_ratio"], dtype=np.float32),
                meta=json.loads(str(data["meta_json"].item())),
            )

    def surfels_for_row(self, row_idx: int) -> np.ndarray:
        start = int(self.offset[row_idx])
        end = int(self.offset[row_idx + 1])
        return self.surfel_indices_flat[start:end]

    def rows_by_birth(self) -> dict[tuple[int, int], list[int]]:
        out: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (frame_id, mask_id) in enumerate(zip(self.birth_frame.tolist(), self.birth_mask_id.tolist())):
            out[(int(frame_id), int(mask_id))].append(int(idx))
        return dict(out)


@dataclass
class VideoMaskletParams:
    min_birth_surfels: int = 12
    min_frame_surfels: int = 6
    boundary_safe_px: float = 3.0
    c2_min_available_mask_agreement: float = 0.30
    c3_min_available_mask_agreement: float = 0.40
    c3_min_boundary_safe_ratio: float = 0.35
    c3_min_confidence: float = 0.45
    c3_max_negative_visible_outside_ratio: float = 0.75


def _dominant_positive_ratio(ids: np.ndarray) -> float:
    ids = ids[ids > 0]
    if ids.size == 0:
        return 0.0
    unique, counts = np.unique(ids, return_counts=True)
    _ = unique
    return float(np.max(counts) / max(int(ids.size), 1))


def _uv_compactness(uv: np.ndarray) -> float:
    if uv.shape[0] <= 1:
        return 1.0
    extent = np.ptp(uv, axis=0)
    diag = float(np.linalg.norm(extent))
    return float(1.0 / (1.0 + diag))


def _appearance_drift(bank: MeasurementBank, surfels: np.ndarray) -> float:
    if surfels.size <= 1:
        return 0.0
    rgb = bank.src_rgb[surfels].astype(np.float32)
    center = rgb.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(rgb - center, axis=1).mean())


def _bank_cycle_error(bank: MeasurementBank, key: str) -> float:
    value = bank.meta.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _passes_mode_gate(
    bank: MeasurementBank,
    mode: str,
    frame_idx: int,
    surfels: np.ndarray,
    params: VideoMaskletParams,
) -> tuple[bool, dict[str, float]]:
    target_ids = bank.target_mask_id[frame_idx, surfels]
    mask_available = bool(bank.mask_frame_available[frame_idx])
    target_positive = target_ids > 0
    agreement = _dominant_positive_ratio(target_ids) if mask_available else 1.0
    negative_ratio = float(np.count_nonzero(target_ids <= 0) / max(int(target_ids.shape[0]), 1)) if mask_available else 0.0
    boundary = bank.boundary_distance[frame_idx, surfels]
    positive_boundary = boundary[target_positive]
    boundary_safe_ratio = (
        float(np.count_nonzero(positive_boundary >= float(params.boundary_safe_px)) / max(int(positive_boundary.size), 1))
        if positive_boundary.size
        else 1.0
    )
    confidence = float(np.mean(bank.confidence[frame_idx, surfels])) if surfels.size else 0.0
    metrics = {
        "available_mask_agreement_iou": float(agreement),
        "negative_visible_outside_ratio": float(negative_ratio),
        "boundary_safe_ratio": float(boundary_safe_ratio),
        "confidence": float(confidence),
    }
    if mode == "C1":
        return True, metrics
    if mode == "C2":
        if mask_available and target_positive.size and agreement < float(params.c2_min_available_mask_agreement):
            return False, metrics
        return True, metrics
    if mode == "C3":
        if confidence < float(params.c3_min_confidence):
            return False, metrics
        if mask_available and agreement < float(params.c3_min_available_mask_agreement):
            return False, metrics
        if boundary_safe_ratio < float(params.c3_min_boundary_safe_ratio):
            return False, metrics
        if negative_ratio > float(params.c3_max_negative_visible_outside_ratio):
            return False, metrics
        return True, metrics
    raise ValueError(f"Unsupported video masklet mode: {mode}")


def build_video_masklet_bank(
    bank: MeasurementBank,
    *,
    mode: str,
    params: VideoMaskletParams | None = None,
) -> tuple[VideoMaskletBank, dict[str, Any]]:
    if mode not in {"C1", "C2", "C3"}:
        raise ValueError(f"mode must be C1/C2/C3, got {mode}")
    params = params or VideoMaskletParams()
    birth_params = ExplanationParams(
        birth_min_surfels=int(params.min_birth_surfels),
        birth_min_boundary_safe_ratio=0.0,
        birth_max_ambiguous_ratio=1.0,
        boundary_safe_px=float(params.boundary_safe_px),
    )
    groups = birth_groups(bank, birth_params)
    row_masklet_id: list[int] = []
    row_birth_frame: list[int] = []
    row_birth_mask: list[int] = []
    row_frame: list[int] = []
    row_confidence: list[float] = []
    row_compactness: list[float] = []
    row_area_growth: list[float] = []
    row_cycle_p50: list[float] = []
    row_cycle_p90: list[float] = []
    row_appearance: list[float] = []
    row_agreement: list[float] = []
    row_negative: list[float] = []
    offsets = [0]
    flat: list[int] = []
    next_masklet_id = 0

    for group in groups:
        birth_surfels = np.asarray(group["surfel_indices"], dtype=np.int64)
        if birth_surfels.size < int(params.min_birth_surfels):
            continue
        birth_size = max(int(birth_surfels.size), 1)
        for frame_idx, frame_id in enumerate(bank.frame_ids.tolist()):
            visible = bank.visible_ok[frame_idx, birth_surfels]
            surfels = birth_surfels[visible]
            if surfels.size < int(params.min_frame_surfels):
                continue
            keep, gate_metrics = _passes_mode_gate(bank, mode, frame_idx, surfels, params)
            if not keep:
                continue
            uv = bank.uv_pred[frame_idx, surfels]
            row_masklet_id.append(next_masklet_id)
            row_birth_frame.append(int(group["birth_frame"]))
            row_birth_mask.append(int(group["birth_mask_id"]))
            row_frame.append(int(frame_id))
            row_confidence.append(gate_metrics["confidence"])
            row_compactness.append(_uv_compactness(uv))
            row_area_growth.append(float(surfels.size / birth_size))
            row_cycle_p50.append(_bank_cycle_error(bank, "cycle_uv_error_p50_mean"))
            row_cycle_p90.append(_bank_cycle_error(bank, "cycle_uv_error_p90_mean"))
            row_appearance.append(_appearance_drift(bank, surfels))
            row_agreement.append(gate_metrics["available_mask_agreement_iou"])
            row_negative.append(gate_metrics["negative_visible_outside_ratio"])
            flat.extend(int(v) for v in surfels.tolist())
            offsets.append(len(flat))
            next_masklet_id += 1

    masklets = VideoMaskletBank(
        scene=bank.scene,
        mode=mode,
        frame_ids=bank.frame_ids.astype(np.int64, copy=False),
        masklet_id=np.asarray(row_masklet_id, dtype=np.int64),
        birth_frame=np.asarray(row_birth_frame, dtype=np.int64),
        birth_mask_id=np.asarray(row_birth_mask, dtype=np.int64),
        frame_id=np.asarray(row_frame, dtype=np.int64),
        offset=np.asarray(offsets, dtype=np.int64),
        surfel_indices_flat=np.asarray(flat, dtype=np.int64),
        confidence=np.asarray(row_confidence, dtype=np.float32),
        compactness=np.asarray(row_compactness, dtype=np.float32),
        area_growth_ratio=np.asarray(row_area_growth, dtype=np.float32),
        cycle_error_p50=np.asarray(row_cycle_p50, dtype=np.float32),
        cycle_error_p90=np.asarray(row_cycle_p90, dtype=np.float32),
        appearance_drift=np.asarray(row_appearance, dtype=np.float32),
        available_mask_agreement_iou=np.asarray(row_agreement, dtype=np.float32),
        negative_visible_outside_ratio=np.asarray(row_negative, dtype=np.float32),
        meta={"params": params.__dict__, "num_birth_groups": int(len(groups))},
    )
    return masklets, summarize_masklets(bank, masklets)


def summarize_original_sparse(bank: MeasurementBank) -> dict[str, Any]:
    positive_counts = bank.positive_observation.sum(axis=0).astype(np.float32)
    return {
        "scene": bank.scene,
        "mode": "C0",
        "num_mask_frames_available": int(np.count_nonzero(bank.mask_frame_available)),
        "num_effective_semantic_frames_per_surfel": float(np.mean(positive_counts)) if positive_counts.size else 0.0,
        "positive_observations_per_surfel": float(np.mean(positive_counts)) if positive_counts.size else 0.0,
        "unobserved_surfel_ratio": float(np.count_nonzero(positive_counts == 0) / max(bank.num_surfels, 1)),
        "masklet_count": 0,
        "masklet_frames_per_object_birth": 0.0,
        "masklet_compactness": 0.0,
        "masklet_area_growth_ratio": 0.0,
        "cycle_error_p50_for_masklet_surfels": 0.0,
        "cycle_error_p90_for_masklet_surfels": 0.0,
        "rgb_appearance_drift": 0.0,
        "available_mask_agreement_iou": 0.0,
        "negative_visible_outside_ratio": 0.0,
        "ambiguous_surfel_ratio": float(
            np.count_nonzero((bank.src_mask_id > 0) & (bank.negative_observation.sum(axis=0) > 0))
            / max(int(np.count_nonzero(bank.src_mask_id > 0)), 1)
        ),
    }


def summarize_masklets(bank: MeasurementBank, masklets: VideoMaskletBank) -> dict[str, Any]:
    frame_membership: dict[int, set[int]] = defaultdict(set)
    birth_counts: dict[tuple[int, int], int] = defaultdict(int)
    surfel_frame_pairs: set[tuple[int, int]] = set()
    surfel_positive_counts = np.zeros((bank.num_surfels,), dtype=np.int32)
    for row_idx in range(masklets.masklet_id.shape[0]):
        surfels = masklets.surfels_for_row(row_idx)
        frame = int(masklets.frame_id[row_idx])
        birth = (int(masklets.birth_frame[row_idx]), int(masklets.birth_mask_id[row_idx]))
        birth_counts[birth] += 1
        for surfel in surfels.tolist():
            surfel_frame_pairs.add((int(surfel), frame))
            frame_membership[int(surfel)].add(frame)
    for surfel, frames in frame_membership.items():
        surfel_positive_counts[int(surfel)] = len(frames)
    def mean_arr(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else 0.0
    return {
        "scene": bank.scene,
        "mode": masklets.mode,
        "num_mask_frames_available": int(np.count_nonzero(bank.mask_frame_available)),
        "num_effective_semantic_frames_per_surfel": float(np.mean(surfel_positive_counts))
        if surfel_positive_counts.size
        else 0.0,
        "positive_observations_per_surfel": float(len(surfel_frame_pairs) / max(bank.num_surfels, 1)),
        "unobserved_surfel_ratio": float(np.count_nonzero(surfel_positive_counts == 0) / max(bank.num_surfels, 1)),
        "masklet_count": int(masklets.masklet_id.shape[0]),
        "masklet_frames_per_object_birth": float(np.mean(list(birth_counts.values()))) if birth_counts else 0.0,
        "masklet_compactness": mean_arr(masklets.compactness),
        "masklet_area_growth_ratio": mean_arr(masklets.area_growth_ratio),
        "cycle_error_p50_for_masklet_surfels": mean_arr(masklets.cycle_error_p50),
        "cycle_error_p90_for_masklet_surfels": mean_arr(masklets.cycle_error_p90),
        "rgb_appearance_drift": mean_arr(masklets.appearance_drift),
        "available_mask_agreement_iou": mean_arr(masklets.available_mask_agreement_iou),
        "negative_visible_outside_ratio": mean_arr(masklets.negative_visible_outside_ratio),
        "ambiguous_surfel_ratio": float(
            np.count_nonzero((bank.src_mask_id > 0) & (bank.negative_observation.sum(axis=0) > 0))
            / max(int(np.count_nonzero(bank.src_mask_id > 0)), 1)
        ),
    }
