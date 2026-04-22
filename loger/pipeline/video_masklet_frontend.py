"""
Stage C: Video Masklet Front-end  (v4)

Correct division of responsibility:
  * **Detector** (GDINO / YOLOE)  →  *what* is in the scene (labels)
  * **SAM 2.1 Video Predictor**   →  *where* each object is across time

The detector runs on one or more discovery frames inside a chunk. SAM 2.1
handles temporal tracking once those per-frame seeds have been registered.

Thing objects (chair, monitor …) are registered with SAM 2.1 for
video-level tracking.  Stuff regions (wall, floor …) are detected
per-keyframe independently – they have no sharp boundaries and SAM 2.1
is not designed to track amorphous regions.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

# ---------------------------------------------------------------------------
# Grounded-SAM-2 imports
# ---------------------------------------------------------------------------
GSAM2_ROOT = os.environ.get(
    "GSAM2_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "Grounded-SAM-2"),
)
GSAM2_ROOT = os.path.abspath(GSAM2_ROOT)
if GSAM2_ROOT not in sys.path:
    sys.path.insert(0, GSAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------------------------
# Semantic taxonomy
# ---------------------------------------------------------------------------
SEMANTIC_GROUP_STRUCTURE_ANCHOR = 0
SEMANTIC_GROUP_STATIC_THING = 1
SEMANTIC_GROUP_MOVABLE_THING = 2
SEMANTIC_GROUP_LOW_VALUE_STUFF = 3
SEMANTIC_GROUP_UNCERTAIN_REGION = 4

SEMANTIC_GROUP_NAMES = {
    SEMANTIC_GROUP_STRUCTURE_ANCHOR: "STRUCTURE_ANCHOR",
    SEMANTIC_GROUP_STATIC_THING: "STATIC_THING",
    SEMANTIC_GROUP_MOVABLE_THING: "MOVABLE_THING",
    SEMANTIC_GROUP_LOW_VALUE_STUFF: "LOW_VALUE_STUFF",
    SEMANTIC_GROUP_UNCERTAIN_REGION: "UNCERTAIN_REGION",
}

_LABEL_TO_GROUP: Dict[str, int] = {}

_STRUCTURE_LABELS = [
    "wall", "floor", "ceiling", "building", "road", "stair",
    "sidewalk", "bridge", "fence", "railing",
]
_STATIC_THING_LABELS = [
    "cabinet", "shelf", "desk", "table", "sofa", "bed", "refrigerator",
    "oven", "sink", "toilet", "bathtub", "counter", "bookshelf",
    "fixed furniture", "large appliance", "door", "window",
]
_MOVABLE_THING_LABELS = [
    "person", "people", "rider", "bicycle", "motorcycle", "car", "bus",
    "truck", "train", "animal", "dog", "cat", "bird", "horse",
    "chair", "box", "bag", "suitcase", "backpack", "bottle",
    "cup", "book", "phone", "laptop", "keyboard", "mouse", "monitor",
]
_LOW_VALUE_STUFF_LABELS = [
    "sky", "tree", "vegetation", "grass", "plant", "flower", "water",
    "river", "sea", "ocean", "lake", "pool", "reflection", "glass",
    "mirror", "screen", "smoke", "fog", "cloud",
]

for _lbl in _STRUCTURE_LABELS:
    _LABEL_TO_GROUP[_lbl] = SEMANTIC_GROUP_STRUCTURE_ANCHOR
for _lbl in _STATIC_THING_LABELS:
    _LABEL_TO_GROUP[_lbl] = SEMANTIC_GROUP_STATIC_THING
for _lbl in _MOVABLE_THING_LABELS:
    _LABEL_TO_GROUP[_lbl] = SEMANTIC_GROUP_MOVABLE_THING
for _lbl in _LOW_VALUE_STUFF_LABELS:
    _LABEL_TO_GROUP[_lbl] = SEMANTIC_GROUP_LOW_VALUE_STUFF

_STUFF_GROUPS = {SEMANTIC_GROUP_STRUCTURE_ANCHOR, SEMANTIC_GROUP_LOW_VALUE_STUFF}
_GROUP_TO_LABELS: Dict[int, List[str]] = {
    SEMANTIC_GROUP_STRUCTURE_ANCHOR: _STRUCTURE_LABELS,
    SEMANTIC_GROUP_STATIC_THING: _STATIC_THING_LABELS,
    SEMANTIC_GROUP_MOVABLE_THING: _MOVABLE_THING_LABELS,
    SEMANTIC_GROUP_LOW_VALUE_STUFF: _LOW_VALUE_STUFF_LABELS,
}
_GROUP_PRIORITY: Dict[int, int] = {
    SEMANTIC_GROUP_MOVABLE_THING: 4,
    SEMANTIC_GROUP_STATIC_THING: 3,
    SEMANTIC_GROUP_STRUCTURE_ANCHOR: 2,
    SEMANTIC_GROUP_LOW_VALUE_STUFF: 1,
}


@lru_cache(maxsize=512)
def _classify_label(label: str) -> Tuple[int, str]:
    label_lower = label.strip().lower()
    if label_lower in _LABEL_TO_GROUP:
        return _LABEL_TO_GROUP[label_lower], label_lower

    tokens = [tok for tok in re.split(r"[^a-z0-9]+", label_lower) if tok]
    best_group = SEMANTIC_GROUP_UNCERTAIN_REGION
    best_key = label_lower
    best_score = 0
    best_priority = -1

    for group, labels in _GROUP_TO_LABELS.items():
        score = 0
        key_for_group = None
        for key in labels:
            if key == label_lower:
                score = max(score, 100)
                key_for_group = key
                continue
            if key in tokens:
                score = max(score, 10 + len(key))
                if key_for_group is None or (10 + len(key)) >= score:
                    key_for_group = key
                continue
            if key in label_lower or label_lower in key:
                score = max(score, len(key))
                if key_for_group is None or len(key) >= score:
                    key_for_group = key
        priority = _GROUP_PRIORITY.get(group, 0)
        if score > best_score or (score == best_score and priority > best_priority):
            best_group = group
            best_key = key_for_group or label_lower
            best_score = score
            best_priority = priority

    if best_score > 0:
        return best_group, best_key
    return SEMANTIC_GROUP_UNCERTAIN_REGION, label_lower


def label_to_group(label: str) -> int:
    return _classify_label(label)[0]


def canonicalize_label(label: str) -> str:
    return _classify_label(label)[1]


def _is_stuff(sem_group: int) -> bool:
    return sem_group in _STUFF_GROUPS


def _labels_compatible(label_a: str, label_b: str) -> bool:
    a = label_a.strip().lower()
    b = label_b.strip().lower()
    return a == b or a in b or b in a


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


DEFAULT_SEMANTIC_WEIGHTS: Dict[int, float] = {
    SEMANTIC_GROUP_STRUCTURE_ANCHOR: 1.00,
    SEMANTIC_GROUP_STATIC_THING: 0.75,
    SEMANTIC_GROUP_MOVABLE_THING: 0.25,
    SEMANTIC_GROUP_LOW_VALUE_STUFF: 0.10,
    SEMANTIC_GROUP_UNCERTAIN_REGION: 0.15,
}

THING_PROMPTS = [
    "person", "chair", "table", "sofa", "box", "bag", "cabinet",
    "shelf", "desk", "monitor", "laptop", "keyboard", "mouse",
    "bottle", "cup", "book", "phone", "car", "bicycle", "door", "window",
]
STUFF_PROMPTS = [
    "wall", "floor", "ceiling", "building",
    "sky", "tree", "vegetation", "water", "road",
]


# ===================================================================
# Detector abstraction
# ===================================================================
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_path, thing_prompts, stuff_prompts,
               box_threshold, text_threshold) -> List[Dict[str, Any]]: ...
    def release_gpu(self): pass
    def to_device(self): pass


class GroundingDINODetector(BaseDetector):
    def __init__(
        self,
        grounding_model,
        image_predictor: Optional[SAM2ImagePredictor],
        device: str,
    ):
        from grounding_dino.groundingdino.util.inference import (
            load_image as _li, predict as _pred,
        )
        self._load_image, self._predict = _li, _pred
        self.model = grounding_model
        self.image_predictor = image_predictor
        self.device = device

    def detect(self, image_path, thing_prompts, stuff_prompts, box_threshold, text_threshold):
        image_source, image_transformed = self._load_image(image_path)
        h_src, w_src = image_source.shape[:2]
        prompt_text = " . ".join(thing_prompts + stuff_prompts) + " ."

        boxes, confidences, labels = self._predict(
            model=self.model, image=image_transformed,
            caption=prompt_text, box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if len(boxes) == 0:
            return []

        boxes_pixel = boxes * torch.Tensor([w_src, h_src, w_src, h_src])
        input_boxes = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        if self.image_predictor is not None:
            self.image_predictor.set_image(image_source)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                masks, scores, _ = self.image_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=input_boxes, multimask_output=False,
                )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
        else:
            masks = []
            for box_xyxy in input_boxes:
                x1, y1, x2, y2 = box_xyxy.astype(int)
                mask = np.zeros((h_src, w_src), dtype=np.uint8)
                mask[max(0, y1):min(h_src, y2), max(0, x1):min(w_src, x2)] = 1
                masks.append(mask)
            masks = (
                np.stack(masks, axis=0)
                if masks
                else np.zeros((0, h_src, w_src), dtype=np.uint8)
            )

        out = []
        for i in range(masks.shape[0]):
            lbl = labels[i] if i < len(labels) else "unknown"
            conf = confidences[i].item() if torch.is_tensor(confidences[i]) else float(confidences[i])
            out.append({"mask": masks[i], "box": input_boxes[i],
                        "confidence": conf, "label": lbl})
        return out

    def release_gpu(self):
        if hasattr(self.model, "cpu"):
            self.model.cpu()
        if self.image_predictor is not None and hasattr(self.image_predictor, "model"):
            self.image_predictor.model.cpu()

    def to_device(self):
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if (
            self.image_predictor is not None
            and hasattr(self.image_predictor, "model")
            and self.image_predictor.model is not None
        ):
            self.image_predictor.model.to(self.device)


class YOLOEDetector(BaseDetector):
    def __init__(self, model_path: str = "yoloe-11l-seg.pt", device: str = "cuda"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self._prompts_set: Optional[str] = None
        self.model.to(device)

    def detect(self, image_path, thing_prompts, stuff_prompts, box_threshold, text_threshold):
        all_prompts = thing_prompts + stuff_prompts
        key = ",".join(sorted(all_prompts))
        if self._prompts_set != key:
            self.model.set_classes(all_prompts)
            self._prompts_set = key

        results = self.model.predict(image_path, conf=box_threshold, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        r = results[0]
        names = r.names
        has_masks = r.masks is not None and r.masks.data is not None
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        masks_data = r.masks.data.cpu().numpy() if has_masks else None

        out = []
        for i in range(len(boxes_xyxy)):
            label = names.get(cls_ids[i], "unknown")
            if masks_data is not None:
                mask = masks_data[i]
                if mask.shape[0] != r.orig_shape[0] or mask.shape[1] != r.orig_shape[1]:
                    mask = cv2.resize(mask, (r.orig_shape[1], r.orig_shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
            else:
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                h, w = r.orig_shape
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
            out.append({"mask": mask, "box": boxes_xyxy[i],
                        "confidence": float(confs[i]), "label": label})
        return out

    def release_gpu(self):
        if hasattr(self.model, "model") and self.model.model is not None:
            self.model.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def to_device(self):
        self.model.to(self.device)


# ===================================================================
# Output container
# ===================================================================
@dataclass
class MaskletOutput:
    M_mask: torch.Tensor       # [J, T, H, W]
    V_mask: torch.Tensor       # [J, T] bool
    B_mask: torch.Tensor       # [J, T, 4]
    Q_mask: torch.Tensor       # [J, T]
    L_sem: List[str]
    G_sem: torch.Tensor
    W_sem: torch.Tensor
    A_ratio: torch.Tensor      # [J, T]

    num_masklets: int = 0
    num_frames: int = 0
    frame_height: int = 0
    frame_width: int = 0

    source_type: List[str] = field(default_factory=list)  # "thing_tracked" | "structure_tracked" | "stuff_static"
    birth_frame: List[int] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
# Core front-end
# ===================================================================
class VideoMaskletFrontend:
    """Stage C – correct architecture:

    1. Detector finds objects on **one** annotation frame → labels + masks
    2. Thing objects are registered with SAM 2.1 Video Predictor → tracked
    3. Stuff regions are **not** tracked; they are detected per-keyframe
    """

    def __init__(
        self,
        video_predictor,
        detector: BaseDetector,
        *,
        sam_backend: str = "sam2",
        device: str = "cuda",
        thing_prompts: Optional[List[str]] = None,
        stuff_prompts: Optional[List[str]] = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        ann_frame_idx: int = 0,
        discovery_frame_stride: int = 8,
        stuff_keyframe_stride: int = 10,
        min_mask_area_ratio: float = 0.001,
        max_mask_area_ratio: float = 0.95,
        nms_iou_threshold: float = 0.70,
        discovery_match_iou_threshold: float = 0.50,
        max_thing_objects: int = 15,
        prompt_type: str = "mask",
        sam31_offload_video_to_cpu: bool = True,
    ):
        self.video_predictor = video_predictor
        self.detector = detector
        self.sam_backend = sam_backend
        self.device = device

        self.thing_prompts = thing_prompts or THING_PROMPTS
        self.stuff_prompts = stuff_prompts or STUFF_PROMPTS
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.ann_frame_idx = ann_frame_idx
        self.discovery_frame_stride = max(int(discovery_frame_stride), 1)
        self.stuff_keyframe_stride = stuff_keyframe_stride
        self.min_mask_area_ratio = min_mask_area_ratio
        self.max_mask_area_ratio = max_mask_area_ratio
        self.nms_iou_threshold = nms_iou_threshold
        self.discovery_match_iou_threshold = discovery_match_iou_threshold
        self.max_thing_objects = max_thing_objects
        self.prompt_type = prompt_type
        self.sam31_offload_video_to_cpu = sam31_offload_video_to_cpu

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        sam2_checkpoint: Optional[str] = None,
        sam2_model_cfg: Optional[str] = None,
        device: str = "cuda",
        *,
        sam_backend: str = "sam2",
        sam3_checkpoint: Optional[str] = None,
        sam31_checkpoint: Optional[str] = None,
        detector_type: str = "gdino",
        gdino_config: Optional[str] = None,
        gdino_checkpoint: Optional[str] = None,
        yoloe_model: str = "yoloe-11l-seg.pt",
        **kwargs,
    ) -> "VideoMaskletFrontend":
        sam31_offload_video_to_cpu = bool(kwargs.pop("sam31_offload_video_to_cpu", True))
        sam31_offload_outputs_to_cpu = bool(kwargs.pop("sam31_offload_outputs_to_cpu", True))
        sam31_postprocess_batch_size = max(int(kwargs.pop("sam31_postprocess_batch_size", 1)), 1)
        sam31_batched_grounding_batch_size = max(
            int(kwargs.pop("sam31_batched_grounding_batch_size", 1)), 1
        )
        if sam_backend == "sam2":
            assert sam2_checkpoint and sam2_model_cfg, (
                "sam2_checkpoint and sam2_model_cfg are required when sam_backend='sam2'"
            )
            video_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint)
        elif sam_backend == "sam3":
            if not device.startswith("cuda"):
                raise ValueError("SAM3 currently requires a CUDA device.")
            if not sam3_checkpoint:
                raise ValueError("sam3_checkpoint is required when sam_backend='sam3'")
            from sam3.model_builder import build_sam3_video_predictor
            video_predictor = build_sam3_video_predictor(
                checkpoint_path=sam3_checkpoint,
                gpus_to_use=[torch.cuda.current_device()],
                async_loading_frames=False,
            )
        elif sam_backend == "sam31_multiplex":
            if not device.startswith("cuda"):
                raise ValueError("SAM3.1 multiplex currently requires a CUDA device.")
            from sam3.model_builder import build_sam3_multiplex_video_predictor
            try:
                video_predictor = build_sam3_multiplex_video_predictor(
                    checkpoint_path=sam31_checkpoint,
                    async_loading_frames=False,
                    compile=False,
                    warm_up=False,
                    use_fa3=False,
                    max_num_objects=max(int(kwargs.get("max_thing_objects", 15)), 16),
                    multiplex_count=max(int(kwargs.get("max_thing_objects", 15)), 16),
                )
                if hasattr(video_predictor, "model"):
                    if hasattr(video_predictor.model, "postprocess_batch_size"):
                        video_predictor.model.postprocess_batch_size = sam31_postprocess_batch_size
                    if hasattr(video_predictor.model, "batched_grounding_batch_size"):
                        video_predictor.model.batched_grounding_batch_size = (
                            sam31_batched_grounding_batch_size
                        )
                    if (
                        hasattr(video_predictor.model, "detector")
                        and hasattr(
                            video_predictor.model.detector,
                            "offload_outputs_to_cpu_for_eval",
                        )
                    ):
                        video_predictor.model.detector.offload_outputs_to_cpu_for_eval = (
                            sam31_offload_outputs_to_cpu
                        )
            except Exception as exc:
                msg = str(exc)
                if "GatedRepoError" in msg or "Cannot access gated repo" in msg or "401 Client Error" in msg:
                    raise RuntimeError(
                        "SAM3.1 multiplex checkpoint is not available locally and HuggingFace auto-download is gated. "
                        "Please provide a local --sam31_checkpoint (e.g. sam3.1_multiplex.pt) or log in to HuggingFace."
                    ) from exc
                raise
        else:
            raise ValueError(f"Unknown sam_backend: {sam_backend}")

        if detector_type == "gdino":
            from grounding_dino.groundingdino.util.inference import (
                load_model as _gdino_load_model,
            )
            assert gdino_config and gdino_checkpoint
            grounding_model = _gdino_load_model(
                model_config_path=gdino_config,
                model_checkpoint_path=gdino_checkpoint,
                device="cpu",
            )
            image_predictor = None
            if sam2_checkpoint and sam2_model_cfg:
                sam2_image_model = build_sam2(sam2_model_cfg, sam2_checkpoint)
                image_predictor = SAM2ImagePredictor(sam2_image_model)
            detector = GroundingDINODetector(grounding_model, image_predictor, device)
        elif detector_type == "yoloe":
            detector = YOLOEDetector(model_path=yoloe_model, device=device)
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        return cls(video_predictor=video_predictor, detector=detector,
                   sam_backend=sam_backend,
                   device=device,
                   sam31_offload_video_to_cpu=sam31_offload_video_to_cpu,
                   **kwargs)

    # -- Public API --------------------------------------------------------

    def run(
        self,
        images: torch.Tensor,
        ann_frame_idx: Optional[int] = None,
        discovery_frame_indices: Optional[List[int]] = None,
        seed_detections_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> MaskletOutput:
        """Run Stage C.

        Parameters
        ----------
        images : [T, 3, H, W] float32 in [0, 1]
        ann_frame_idx : override annotation frame (default: self.ann_frame_idx)
        discovery_frame_indices : optional list of local frame indices on which
            the detector should run to discover late-appearing objects.
        """
        T, C, H, W = images.shape
        assert C == 3
        ann_idx = ann_frame_idx if ann_frame_idx is not None else self.ann_frame_idx
        discovery_indices = self._build_discovery_frame_indices(
            T, ann_idx, discovery_frame_indices,
        )

        self.detector.to_device()

        frame_dir = tempfile.mkdtemp(prefix="masklet_frames_")
        try:
            self._write_frames(images, frame_dir)
            return self._run_pipeline(
                frame_dir,
                T,
                H,
                W,
                discovery_indices,
                seed_detections_by_frame=seed_detections_by_frame,
            )
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    # -- Pipeline ----------------------------------------------------------

    def _build_discovery_frame_indices(
        self,
        T: int,
        ann_idx: int,
        discovery_frame_indices: Optional[List[int]],
    ) -> List[int]:
        if discovery_frame_indices:
            valid = sorted({int(i) for i in discovery_frame_indices if 0 <= int(i) < T})
            if valid:
                return valid

        ann_idx = int(np.clip(ann_idx, 0, max(T - 1, 0)))
        indices = list(range(ann_idx, T, self.discovery_frame_stride))
        if ann_idx not in indices:
            indices.insert(0, ann_idx)
        if 0 not in indices:
            indices.insert(0, 0)
        return sorted({i for i in indices if 0 <= i < T})

    def _start_sam31_session(self, frame_dir: str) -> str:
        request = dict(
            type="start_session",
            resource_path=frame_dir,
            offload_video_to_cpu=self.sam31_offload_video_to_cpu,
        )
        return self.video_predictor.handle_request(request)["session_id"]

    def _select_multiframe_things(
        self,
        frame_dir: str,
        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        if self.sam_backend == "sam3":
            return self._select_multiframe_things_sam3(
                frame_dir, thing_dets_by_frame, discovery_indices, T, H, W, area,
            )
        if self.sam_backend == "sam31_multiplex":
            return self._select_multiframe_things_sam31_multiplex(
                frame_dir, thing_dets_by_frame, discovery_indices, T, H, W, area,
            )

        selected: List[Dict[str, Any]] = []
        thing_segments: Dict[int, Dict[int, np.ndarray]] = {}
        thing_oid_map: Dict[int, Dict] = {}

        for frame_idx in discovery_indices:
            candidates = list(thing_dets_by_frame.get(frame_idx, []))
            if not candidates:
                continue

            candidates.sort(
                key=lambda d: (d["mask"].astype(bool).sum() / max(area, 1), d["confidence"]),
                reverse=True,
            )

            added = False
            for det in candidates:
                if len(selected) >= self.max_thing_objects:
                    break
                if self._is_duplicate_thing_detection(det, frame_idx, thing_segments, thing_oid_map):
                    continue
                selected.append(det)
                added = True

            if added:
                thing_segments, thing_oid_map = self._propagate_things(
                    frame_dir, selected, T, H, W,
                )

            if len(selected) >= self.max_thing_objects:
                break

        return selected, thing_segments, thing_oid_map

    def _select_multiframe_things_sam31_multiplex(
        self,
        frame_dir: str,
        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        selected: List[Dict[str, Any]] = []
        thing_segments: Dict[int, Dict[int, np.ndarray]] = {}
        thing_oid_map: Dict[int, Dict] = {}
        next_oid = 1

        label_to_dets: Dict[str, List[Dict[str, Any]]] = {}
        for frame_idx in discovery_indices:
            candidates = list(thing_dets_by_frame.get(frame_idx, []))
            if not candidates:
                continue
            candidates.sort(
                key=lambda d: (d["mask"].astype(bool).sum() / max(area, 1), d["confidence"]),
                reverse=True,
            )
            for det in candidates:
                label_to_dets.setdefault(str(det["label"]), []).append(dict(det))

        if not label_to_dets:
            return selected, thing_segments, thing_oid_map

        session_id = self._start_sam31_session(frame_dir)

        try:
            label_order = sorted(
                label_to_dets.keys(),
                key=lambda lbl: min(int(det.get("frame_idx", 0)) for det in label_to_dets[lbl]),
            )
            for label in label_order:
                if len(selected) >= self.max_thing_objects:
                    break

                # SAM3.1 multiplex long runs can leave tracker-side state behind even
                # after reset_session. We still try the cheap reset first, but on the
                # specific "No points are provided" failure we reopen a fresh session.
                self.video_predictor.handle_request(
                    dict(type="reset_session", session_id=session_id)
                )

                raw_dets = sorted(
                    label_to_dets[label],
                    key=lambda d: (
                        int(d.get("frame_idx", 0)),
                        -(d["mask"].astype(bool).sum() / max(area, 1)),
                        -float(d["confidence"]),
                    ),
                )
                candidate_dets: List[Dict[str, Any]] = []
                for det in raw_dets:
                    if len(selected) + len(candidate_dets) >= self.max_thing_objects:
                        break
                    if self._is_duplicate_thing_detection(
                        det, int(det.get("frame_idx", 0)), thing_segments, thing_oid_map
                    ):
                        continue
                    candidate_dets.append(det)

                if not candidate_dets:
                    continue

                prompt_frame = min(int(det.get("frame_idx", 0)) for det in candidate_dets)
                obj_segments: Dict[int, Dict[int, np.ndarray]] = {}
                obj_scores: Dict[int, float] = {}
                try:
                    obj_segments, obj_scores = self._track_label_with_sam31_multiplex_session(
                        session_id=session_id,
                        label=label,
                        frame_idx=prompt_frame,
                    )
                except RuntimeError as exc:
                    if "No points are provided; please add points first" not in str(exc):
                        raise
                    # Retry from a brand-new session; in long videos reset_session is not
                    # always enough to clear the multiplex tracker state.
                    try:
                        self.video_predictor.handle_request(
                            dict(type="close_session", session_id=session_id)
                        )
                    except Exception:
                        pass
                    session_id = self._start_sam31_session(frame_dir)
                    try:
                        obj_segments, obj_scores = self._track_label_with_sam31_multiplex_session(
                            session_id=session_id,
                            label=label,
                            frame_idx=prompt_frame,
                        )
                    except RuntimeError as retry_exc:
                        if "No points are provided; please add points first" not in str(retry_exc):
                            raise
                        print(
                            f"[sam31_multiplex] skipping label={label!r} prompt_frame={prompt_frame} "
                            "after repeated 'No points are provided' failures"
                        )
                        continue
                if not obj_segments:
                    continue

                used_prompt_obj_ids = set()
                for det in candidate_dets:
                    if len(selected) >= self.max_thing_objects:
                        break
                    match = self._match_detection_to_object_track(
                        det=det,
                        obj_segments=obj_segments,
                    )
                    if match is None:
                        continue
                    prompt_obj_id, match_score = match
                    if prompt_obj_id in used_prompt_obj_ids:
                        continue
                    used_prompt_obj_ids.add(prompt_obj_id)

                    det_copy = dict(det)
                    det_copy["confidence"] = max(
                        float(det_copy["confidence"]),
                        float(obj_scores.get(prompt_obj_id, 0.0)),
                    )
                    oid = next_oid
                    next_oid += 1
                    thing_oid_map[oid] = det_copy
                    selected.append(det_copy)

                    segments = obj_segments[prompt_obj_id]
                    for t, mask in segments.items():
                        thing_segments.setdefault(int(t), {})[oid] = mask

        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return selected, thing_segments, thing_oid_map

    def _select_multiframe_structure_sam31_multiplex(
        self,
        frame_dir: str,
        structure_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        selected: List[Dict[str, Any]] = []
        structure_segments: Dict[int, Dict[int, np.ndarray]] = {}
        structure_oid_map: Dict[int, Dict] = {}
        next_oid = 1

        label_to_dets: Dict[str, List[Dict[str, Any]]] = {}
        for frame_idx in discovery_indices:
            candidates = list(structure_dets_by_frame.get(frame_idx, []))
            if not candidates:
                continue
            candidates.sort(
                key=lambda d: (d["mask"].astype(bool).sum() / max(area, 1), d["confidence"]),
                reverse=True,
            )
            for det in candidates:
                label_to_dets.setdefault(str(det["label"]), []).append(dict(det))

        if not label_to_dets:
            return selected, structure_segments, structure_oid_map

        session_id = self._start_sam31_session(frame_dir)

        try:
            label_order = sorted(
                label_to_dets.keys(),
                key=lambda lbl: min(int(det.get("frame_idx", 0)) for det in label_to_dets[lbl]),
            )
            for label in label_order:
                self.video_predictor.handle_request(
                    dict(type="reset_session", session_id=session_id)
                )

                raw_dets = sorted(
                    label_to_dets[label],
                    key=lambda d: (
                        int(d.get("frame_idx", 0)),
                        -(d["mask"].astype(bool).sum() / max(area, 1)),
                        -float(d["confidence"]),
                    ),
                )
                candidate_dets: List[Dict[str, Any]] = []
                for det in raw_dets:
                    if self._is_duplicate_detection(
                        det,
                        int(det.get("frame_idx", 0)),
                        structure_segments,
                        structure_oid_map,
                    ):
                        continue
                    candidate_dets.append(det)

                if not candidate_dets:
                    continue

                prompt_frame = min(int(det.get("frame_idx", 0)) for det in candidate_dets)
                obj_segments: Dict[int, Dict[int, np.ndarray]] = {}
                obj_scores: Dict[int, float] = {}
                try:
                    obj_segments, obj_scores = self._track_label_with_sam31_multiplex_session(
                        session_id=session_id,
                        label=label,
                        frame_idx=prompt_frame,
                    )
                except RuntimeError as exc:
                    if "No points are provided; please add points first" not in str(exc):
                        raise
                    try:
                        self.video_predictor.handle_request(
                            dict(type="close_session", session_id=session_id)
                        )
                    except Exception:
                        pass
                    session_id = self._start_sam31_session(frame_dir)
                    try:
                        obj_segments, obj_scores = self._track_label_with_sam31_multiplex_session(
                            session_id=session_id,
                            label=label,
                            frame_idx=prompt_frame,
                        )
                    except RuntimeError as retry_exc:
                        if "No points are provided; please add points first" not in str(retry_exc):
                            raise
                        print(
                            f"[sam31_multiplex] skipping structure label={label!r} prompt_frame={prompt_frame} "
                            "after repeated 'No points are provided' failures"
                        )
                        continue
                if not obj_segments:
                    continue

                best_obj_id = None
                best_obj_score = -1.0
                best_det = None
                for det in candidate_dets:
                    match = self._match_detection_to_object_track(
                        det=det,
                        obj_segments=obj_segments,
                    )
                    if match is None:
                        continue
                    prompt_obj_id, match_score = match
                    score = float(match_score) + 0.10 * float(obj_scores.get(prompt_obj_id, 0.0))
                    if score > best_obj_score:
                        best_obj_id = int(prompt_obj_id)
                        best_obj_score = score
                        best_det = det

                if best_obj_id is None or best_det is None:
                    continue

                det_copy = dict(best_det)
                det_copy["confidence"] = max(
                    float(det_copy["confidence"]),
                    float(obj_scores.get(best_obj_id, 0.0)),
                )
                oid = next_oid
                next_oid += 1
                structure_oid_map[oid] = det_copy
                selected.append(det_copy)

                for t, mask in obj_segments[best_obj_id].items():
                    structure_segments.setdefault(int(t), {})[oid] = mask
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return selected, structure_segments, structure_oid_map

    def _select_multiframe_things_sam3(
        self,
        frame_dir: str,
        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        selected: List[Dict[str, Any]] = []
        thing_segments: Dict[int, Dict[int, np.ndarray]] = {}
        thing_oid_map: Dict[int, Dict] = {}
        next_oid = 1
        request = dict(type="start_session", resource_path=frame_dir)
        session_id = self.video_predictor.handle_request(request)["session_id"]

        try:
            for frame_idx in discovery_indices:
                candidates = list(thing_dets_by_frame.get(frame_idx, []))
                if not candidates:
                    continue

                candidates.sort(
                    key=lambda d: (d["mask"].astype(bool).sum() / max(area, 1), d["confidence"]),
                    reverse=True,
                )

                filtered_candidates: List[Dict[str, Any]] = []
                for det in candidates:
                    if len(selected) + len(filtered_candidates) >= self.max_thing_objects:
                        break
                    if self._is_duplicate_thing_detection(det, frame_idx, thing_segments, thing_oid_map):
                        continue
                    filtered_candidates.append(dict(det))

                if not filtered_candidates:
                    continue

                for det_copy in filtered_candidates:
                    if len(selected) >= self.max_thing_objects:
                        break

                    track_segments, prompt_meta = self._track_single_thing_with_sam3_session(
                        session_id=session_id,
                        det=det_copy,
                        H=H,
                        W=W,
                    )
                    if not track_segments:
                        continue

                    det_copy["confidence"] = max(
                        float(det_copy["confidence"]),
                        float(prompt_meta.get("prompt_prob", 0.0)),
                    )
                    det_copy["frame_idx"] = frame_idx
                    oid = next_oid
                    next_oid += 1
                    thing_oid_map[oid] = det_copy
                    selected.append(det_copy)

                    for t, mask in track_segments.items():
                        thing_segments.setdefault(t, {})[oid] = mask

                if len(selected) >= self.max_thing_objects:
                    break
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return selected, thing_segments, thing_oid_map

    def _is_duplicate_detection(
        self,
        det: Dict[str, Any],
        frame_idx: int,
        segments_by_frame: Dict[int, Dict[int, np.ndarray]],
        oid_map: Dict[int, Dict],
    ) -> bool:
        frame_segments = segments_by_frame.get(frame_idx, {})
        if not frame_segments:
            return False

        det_mask = det["mask"].astype(bool)
        det_area = float(det_mask.sum())
        if det_area <= 0:
            return True

        for oid, tracked_mask in frame_segments.items():
            tracked_det = oid_map.get(oid)
            if tracked_det is None:
                continue
            if tracked_det.get("sem_group") != det.get("sem_group"):
                continue
            if not _labels_compatible(tracked_det.get("label", ""), det.get("label", "")):
                continue

            tracked_mask = tracked_mask.astype(bool)
            inter = float((det_mask & tracked_mask).sum())
            iou = _mask_iou(det_mask, tracked_mask)
            cover = inter / max(det_area, 1.0)
            if iou >= self.discovery_match_iou_threshold or cover >= 0.75:
                return True

        return False

    def _is_duplicate_thing_detection(
        self,
        det: Dict[str, Any],
        frame_idx: int,
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict],
    ) -> bool:
        return self._is_duplicate_detection(
            det, frame_idx, thing_segments, thing_oid_map,
        )

    def _run_pipeline(
        self,
        frame_dir: str,
        T: int,
        H: int,
        W: int,
        discovery_indices: List[int],
        seed_detections_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> MaskletOutput:
        area = H * W

        # ======== Step 1: detect on discovery frames ========
        detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for frame_idx in discovery_indices:
            frame_dets = self._detect_and_filter(frame_dir, frame_idx, H, W)
            for det in frame_dets:
                det["frame_idx"] = frame_idx
            detections_by_frame[frame_idx] = frame_dets

        if seed_detections_by_frame:
            for frame_idx, seed_dets in seed_detections_by_frame.items():
                frame_idx = int(frame_idx)
                if not (0 <= frame_idx < T):
                    continue
                merged = list(detections_by_frame.get(frame_idx, []))
                for det in seed_dets:
                    det_copy = dict(det)
                    det_copy["frame_idx"] = frame_idx
                    merged.append(det_copy)
                detections_by_frame[frame_idx] = self._nms(merged)

        if not any(detections_by_frame.values()):
            return self._empty_output(T, H, W)

        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        structure_dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        stuff_dets: List[Dict[str, Any]] = []
        for frame_idx, dets in detections_by_frame.items():
            thing_dets_by_frame[frame_idx] = []
            structure_dets_by_frame[frame_idx] = []
            for det in dets:
                sem_group = int(det["sem_group"])
                if (
                    self.sam_backend == "sam31_multiplex"
                    and sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR
                ):
                    structure_dets_by_frame[frame_idx].append(det)
                elif _is_stuff(sem_group):
                    stuff_dets.append(det)
                else:
                    thing_dets_by_frame[frame_idx].append(det)

        # ======== Step 2: free detector GPU → give memory to SAM2 ========
        self.detector.release_gpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ======== Step 3: SAM2 Video Predictor — track thing objects ========
        thing_segments, thing_oid_map = {}, {}
        selected_thing_dets, thing_segments, thing_oid_map = \
            self._select_multiframe_things(
                frame_dir, thing_dets_by_frame, discovery_indices, T, H, W, area,
            )
        if not thing_segments and selected_thing_dets:
            thing_segments, thing_oid_map = self._propagate_things(
                frame_dir, selected_thing_dets, T, H, W,
            )

        structure_segments, structure_oid_map = {}, {}
        selected_structure_dets: List[Dict[str, Any]] = []
        if self.sam_backend == "sam31_multiplex":
            (
                selected_structure_dets,
                structure_segments,
                structure_oid_map,
            ) = self._select_multiframe_structure_sam31_multiplex(
                frame_dir,
                structure_dets_by_frame,
                discovery_indices,
                T,
                H,
                W,
                area,
            )

        tracked_structure_labels = {str(det["label"]) for det in selected_structure_dets}
        if tracked_structure_labels:
            stuff_dets.extend(
                d
                for frame_idx in discovery_indices
                for d in structure_dets_by_frame.get(frame_idx, [])
                if str(d["label"]) not in tracked_structure_labels
            )
        else:
            stuff_dets.extend(
                d
                for frame_idx in discovery_indices
                for d in structure_dets_by_frame.get(frame_idx, [])
            )

        # ======== Step 4: stuff — per-keyframe static masks ========
        stuff_entries = self._build_stuff_entries(
            stuff_dets, frame_dir, T, H, W,
        )

        # ======== Step 5: assemble ========
        output = self._assemble(
            thing_segments,
            thing_oid_map,
            structure_segments,
            structure_oid_map,
            stuff_entries,
            T,
            H,
            W,
        )
        output.debug.update({
            "sam_backend": self.sam_backend,
            "discovery_frames": list(discovery_indices),
            "num_discovery_frames": len(discovery_indices),
        })
        return output

    # -- Detection & filtering --------------------------------------------

    def _detect_and_filter(self, frame_dir: str, frame_idx: int,
                           H: int, W: int) -> List[Dict]:
        img_path = os.path.join(frame_dir, f"{frame_idx:05d}.jpg")
        raw = self.detector.detect(
            img_path, self.thing_prompts, self.stuff_prompts,
            self.box_threshold, self.text_threshold,
        )

        area = H * W
        filtered = []
        for det in raw:
            mask = det["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask_bool = mask.astype(bool) if mask.dtype != bool else mask
            if mask_bool.shape[0] != H or mask_bool.shape[1] != W:
                mask_f = mask.astype(np.float32)
                mask_f = cv2.resize(mask_f, (W, H), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_f > 0.5

            ratio = mask_bool.sum() / area
            if ratio < self.min_mask_area_ratio or ratio > self.max_mask_area_ratio:
                continue

            raw_label = str(det.get("label", "unknown"))
            det["raw_label"] = raw_label
            det["label"] = canonicalize_label(raw_label)
            det["mask"] = mask_bool.astype(np.uint8)
            det["sem_group"] = label_to_group(det["label"])
            det["area_ratio"] = ratio
            filtered.append(det)

        # NMS within frame
        return self._nms(filtered)

    def _nms(self, dets: List[Dict]) -> List[Dict]:
        if len(dets) <= 1:
            return dets
        masks = np.stack([d["mask"].astype(bool) for d in dets])
        scores = np.array([d["confidence"] for d in dets])
        order = scores.argsort()[::-1]
        keep, suppressed = [], set()
        for i in order:
            if i in suppressed:
                continue
            keep.append(i)
            ai = masks[i].sum()
            for j in order:
                if j in suppressed or j == i:
                    continue
                inter = (masks[i] & masks[j]).sum()
                union = ai + masks[j].sum() - inter
                if union > 0 and inter / union > self.nms_iou_threshold:
                    suppressed.add(j)
        return [dets[i] for i in keep]

    # -- Thing propagation -------------------------------------------------

    def _xyxy_to_normalized_xywh(
        self,
        box_xyxy: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        return np.array(
            [
                x1 / max(W, 1),
                y1 / max(H, 1),
                max(0.0, x2 - x1) / max(W, 1),
                max(0.0, y2 - y1) / max(H, 1),
            ],
            dtype=np.float32,
        )

    def _sam3_xywh_to_xyxy(
        self,
        box_xywh_norm: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        x, y, w, h = [float(v) for v in box_xywh_norm]
        return np.array([x * W, y * H, (x + w) * W, (y + h) * H], dtype=np.float32)

    def _match_sam3_output_to_detection(
        self,
        det: Dict[str, Any],
        outputs: Dict[str, Any],
        H: int,
        W: int,
    ) -> Optional[Dict[str, Any]]:
        obj_ids = outputs.get("out_obj_ids")
        masks = outputs.get("out_binary_masks")
        boxes_xywh = outputs.get("out_boxes_xywh")
        probs = outputs.get("out_probs")
        if obj_ids is None or masks is None or len(obj_ids) == 0:
            return None

        det_mask = det["mask"].astype(bool)
        det_box = np.asarray(det["box"], dtype=np.float32)
        best = None

        for i in range(len(obj_ids)):
            sam_mask = masks[i].astype(bool)
            if sam_mask.sum() <= 0:
                continue
            sam_box = self._sam3_xywh_to_xyxy(boxes_xywh[i], H, W)
            mask_iou = _mask_iou(det_mask, sam_mask)
            det_cover = float((det_mask & sam_mask).sum()) / max(float(det_mask.sum()), 1.0)
            box_iou = _box_iou_xyxy(det_box, sam_box)
            score = 0.50 * mask_iou + 0.30 * det_cover + 0.20 * box_iou
            if best is None or score > best["score"]:
                best = {
                    "index": i,
                    "obj_id": int(obj_ids[i]),
                    "score": float(score),
                    "prompt_prob": float(probs[i]) if probs is not None else 0.0,
                }

        if best is None or best["score"] < 0.10:
            return None
        return best

    def _match_detection_to_object_track(
        self,
        det: Dict[str, Any],
        obj_segments: Dict[int, Dict[int, np.ndarray]],
    ) -> Optional[Tuple[int, float]]:
        det_frame = int(det.get("frame_idx", 0))
        det_mask = det["mask"].astype(bool)
        det_area = float(det_mask.sum())
        if det_area <= 0:
            return None

        best_obj_id = None
        best_score = -1.0
        for obj_id, frame_to_mask in obj_segments.items():
            track_mask = frame_to_mask.get(det_frame)
            if track_mask is None:
                continue
            track_mask = track_mask.astype(bool)
            inter = float((det_mask & track_mask).sum())
            iou = _mask_iou(det_mask, track_mask)
            cover = inter / max(det_area, 1.0)
            score = 0.70 * iou + 0.30 * cover
            if score > best_score:
                best_score = score
                best_obj_id = int(obj_id)

        if best_obj_id is None:
            return None
        if best_score < 0.10:
            return None
        return best_obj_id, float(best_score)

    @torch.inference_mode()
    def _track_single_thing_with_sam3_session(
        self,
        session_id: str,
        det: Dict[str, Any],
        H: int,
        W: int,
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
        ann_idx = int(det.get("frame_idx", 0))
        box_xywh = self._xyxy_to_normalized_xywh(np.asarray(det["box"]), H, W)
        response = self.video_predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=ann_idx,
                text=det["label"],
                bounding_boxes=[box_xywh.tolist()],
                bounding_box_labels=[1],
            )
        )
        prompt_outputs = response["outputs"]
        match = self._match_sam3_output_to_detection(det, prompt_outputs, H, W)
        if match is None:
            return {}, {}

        selected_obj_id = int(match["obj_id"])
        segments: Dict[int, np.ndarray] = {}
        for prop_response in self.video_predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            out_frame_idx = int(prop_response["frame_index"])
            outputs = prop_response["outputs"]
            curr_ids = outputs.get("out_obj_ids", [])
            if len(curr_ids) == 0:
                continue
            curr_ids = np.asarray(curr_ids)
            hit = np.where(curr_ids == selected_obj_id)[0]
            if len(hit) == 0:
                continue
            mask = outputs["out_binary_masks"][int(hit[0])]
            segments[out_frame_idx] = mask.astype(np.uint8)

        return segments, {
            "prompt_score": float(match["score"]),
            "prompt_prob": float(match["prompt_prob"]),
        }

    @torch.inference_mode()
    def _track_label_with_sam31_multiplex_session(
        self,
        session_id: str,
        label: str,
        frame_idx: int,
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, float]]:
        response = self.video_predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=int(frame_idx),
                text=str(label),
            )
        )
        prompt_outputs = response["outputs"]
        prompt_obj_ids = prompt_outputs.get("out_obj_ids", [])
        if len(prompt_obj_ids) == 0:
            return {}, {}

        obj_segments: Dict[int, Dict[int, np.ndarray]] = {
            int(obj_id): {} for obj_id in prompt_obj_ids
        }
        obj_scores: Dict[int, float] = {}

        prompt_scores = prompt_outputs.get("out_probs", [])
        prompt_masks = prompt_outputs.get("out_binary_masks", [])
        for idx, obj_id in enumerate(prompt_obj_ids):
            obj_id = int(obj_id)
            if idx < len(prompt_scores):
                obj_scores[obj_id] = max(obj_scores.get(obj_id, 0.0), float(prompt_scores[idx]))
            if idx < len(prompt_masks):
                obj_segments[obj_id][int(frame_idx)] = prompt_masks[idx].astype(np.uint8)

        for prop_response in self.video_predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=int(frame_idx),
            )
        ):
            out_frame_idx = int(prop_response["frame_index"])
            outputs = prop_response["outputs"]
            out_obj_ids = outputs.get("out_obj_ids", [])
            out_masks = outputs.get("out_binary_masks", [])
            out_probs = outputs.get("out_probs", [])
            for idx, obj_id in enumerate(out_obj_ids):
                obj_id = int(obj_id)
                if idx < len(out_masks):
                    obj_segments.setdefault(obj_id, {})[out_frame_idx] = out_masks[idx].astype(np.uint8)
                if idx < len(out_probs):
                    obj_scores[obj_id] = max(obj_scores.get(obj_id, 0.0), float(out_probs[idx]))

        obj_segments = {
            obj_id: frame_to_mask
            for obj_id, frame_to_mask in obj_segments.items()
            if frame_to_mask
        }
        return obj_segments, obj_scores

    @torch.inference_mode()
    def _propagate_things(
        self, frame_dir: str, thing_dets: List[Dict], T: int, H: int, W: int,
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        if self.sam_backend == "sam3":
            dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
            for det in thing_dets:
                dets_by_frame.setdefault(int(det.get("frame_idx", 0)), []).append(det)
            selected, segments, oid_map = self._select_multiframe_things_sam3(
                frame_dir,
                dets_by_frame,
                sorted(dets_by_frame.keys()),
                T,
                H,
                W,
                H * W,
            )
            return segments, oid_map
        if self.sam_backend == "sam31_multiplex":
            dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
            for det in thing_dets:
                dets_by_frame.setdefault(int(det.get("frame_idx", 0)), []).append(det)
            selected, segments, oid_map = self._select_multiframe_things_sam31_multiplex(
                frame_dir,
                dets_by_frame,
                sorted(dets_by_frame.keys()),
                T,
                H,
                W,
                H * W,
            )
            return segments, oid_map

        inference_state = self.video_predictor.init_state(video_path=frame_dir)
        oid_map: Dict[int, Dict] = {}

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            for idx, det in enumerate(thing_dets):
                obj_id = idx + 1
                oid_map[obj_id] = det
                ann_idx = det.get("frame_idx", 0)

                if self.prompt_type == "mask":
                    self.video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=ann_idx, obj_id=obj_id,
                        mask=det["mask"],
                    )
                else:
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_idx, obj_id=obj_id,
                        box=det["box"],
                    )

            # Propagate — use logits > 0.0 (following official demo)
            segments: Dict[int, Dict[int, np.ndarray]] = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in \
                    self.video_predictor.propagate_in_video(inference_state):
                segments[out_frame_idx] = {}
                for i, oid in enumerate(out_obj_ids):
                    mask_binary = (out_mask_logits[i] > 0.0).cpu().numpy()
                    segments[out_frame_idx][oid] = mask_binary

        self.video_predictor.reset_state(inference_state)
        return segments, oid_map

    # -- Stuff: chunk-level stable anchors ----------------------------------

    def _build_stuff_entries(
        self, stuff_dets: List[Dict],
        frame_dir: str, T: int, H: int, W: int,
    ) -> List[Dict]:
        """Build stable stuff masklet entries for the whole chunk.

        We first collapse repeated structure/stuff detections across discovery
        frames into a small set of chunk-level tracks. Then each track is made
        visible on every frame in the chunk by copying the nearest discovered
        mask. This greatly reduces per-frame flicker and gives cross-chunk ID
        matching actual overlap to work with.
        """
        if not stuff_dets:
            return []

        clusters: List[Dict[str, Any]] = []
        for det in sorted(stuff_dets, key=lambda d: (int(d.get("frame_idx", 0)), -float(d["confidence"]))):
            det_frame = int(det.get("frame_idx", 0))
            best_cluster = None
            best_score = -1.0

            for cluster in clusters:
                if cluster["sem_group"] != det["sem_group"]:
                    continue
                if not _labels_compatible(cluster["label"], det["label"]):
                    continue
                ref_det = cluster["detections"][-1]
                mask_iou = _mask_iou(det["mask"], ref_det["mask"])
                box_iou = _box_iou_xyxy(np.asarray(det["box"]), np.asarray(ref_det["box"]))
                area_sim = min(det["area_ratio"], ref_det["area_ratio"]) / max(
                    max(det["area_ratio"], ref_det["area_ratio"]), 1e-6
                )
                frame_gap = abs(det_frame - int(ref_det.get("frame_idx", 0)))
                gap_bonus = 0.10 if frame_gap <= max(self.discovery_frame_stride, 1) else 0.0
                score = 0.55 * mask_iou + 0.25 * box_iou + 0.10 * area_sim + gap_bonus
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if best_cluster is not None and best_score >= 0.20:
                best_cluster["detections"].append(det)
            else:
                clusters.append({
                    "label": det["label"],
                    "sem_group": det["sem_group"],
                    "detections": [det],
                })

        entries: List[Dict[str, Any]] = []
        for cluster in clusters:
            frame_to_det: Dict[int, Dict[str, Any]] = {}
            for det in cluster["detections"]:
                frame_idx = int(det.get("frame_idx", 0))
                prev = frame_to_det.get(frame_idx)
                if prev is None or float(det["confidence"]) > float(prev["confidence"]):
                    frame_to_det[frame_idx] = det

            detected_frames = sorted(frame_to_det.keys())
            if not detected_frames:
                continue

            frame_to_mask: Dict[int, np.ndarray] = {}
            frame_to_confidence: Dict[int, float] = {}
            for t in range(T):
                nearest_frame = min(detected_frames, key=lambda k: abs(k - t))
                chosen = frame_to_det[nearest_frame]
                frame_to_mask[t] = chosen["mask"].astype(np.uint8)
                frame_to_confidence[t] = float(chosen["confidence"])

            entries.append({
                "label": cluster["label"],
                "sem_group": cluster["sem_group"],
                "frame_to_mask": frame_to_mask,
                "frame_to_confidence": frame_to_confidence,
                "birth_frame": int(detected_frames[0]),
                "support_frames": detected_frames,
            })

        merged_entries: List[Dict[str, Any]] = []
        structure_entries: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for entry in entries:
            key = (int(entry["sem_group"]), str(entry["label"]))
            if int(entry["sem_group"]) != SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                merged_entries.append(entry)
                continue

            merged = structure_entries.get(key)
            if merged is None:
                structure_entries[key] = {
                    "label": entry["label"],
                    "sem_group": entry["sem_group"],
                    "frame_to_mask": {
                        int(t): mask.copy() for t, mask in entry["frame_to_mask"].items()
                    },
                    "frame_to_confidence": {
                        int(t): float(v) for t, v in entry["frame_to_confidence"].items()
                    },
                    "birth_frame": int(entry["birth_frame"]),
                    "support_frames": list(entry["support_frames"]),
                }
                continue

            for t, mask_np in entry["frame_to_mask"].items():
                t = int(t)
                prev_mask = merged["frame_to_mask"].get(t)
                if prev_mask is None:
                    merged["frame_to_mask"][t] = mask_np.copy()
                else:
                    merged["frame_to_mask"][t] = np.logical_or(
                        prev_mask.astype(bool), mask_np.astype(bool)
                    ).astype(np.uint8)
                merged["frame_to_confidence"][t] = max(
                    float(merged["frame_to_confidence"].get(t, 0.0)),
                    float(entry["frame_to_confidence"].get(t, 0.0)),
                )

            merged["birth_frame"] = min(int(merged["birth_frame"]), int(entry["birth_frame"]))
            merged["support_frames"] = sorted(
                {int(v) for v in merged["support_frames"] + list(entry["support_frames"])}
            )

        merged_entries.extend(structure_entries.values())
        merged_entries.sort(key=lambda item: (int(item["birth_frame"]), int(item["sem_group"]), str(item["label"])))
        return merged_entries

    # -- Assembly ----------------------------------------------------------

    def _assemble(
        self,
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict],
        structure_segments: Dict[int, Dict[int, np.ndarray]],
        structure_oid_map: Dict[int, Dict],
        stuff_entries: List[Dict],
        T: int, H: int, W: int,
    ) -> MaskletOutput:
        thing_ids = sorted(thing_oid_map.keys())
        structure_ids = sorted(structure_oid_map.keys())
        J_thing = len(thing_ids)
        J_structure = len(structure_ids)
        J_stuff = len(stuff_entries)
        J = J_thing + J_structure + J_stuff

        if J == 0:
            return self._empty_output(T, H, W)

        area = H * W
        M_mask = torch.zeros(J, T, H, W, dtype=torch.float32)
        Q_mask = torch.zeros(J, T, dtype=torch.float32)
        V_mask = torch.zeros(J, T, dtype=torch.bool)
        B_mask = torch.zeros(J, T, 4, dtype=torch.float32)
        A_ratio = torch.zeros(J, T, dtype=torch.float32)

        L_sem, source_type, birth_frame = [], [], []
        G_sem = torch.zeros(J, dtype=torch.long)
        W_sem = torch.zeros(J, dtype=torch.float32)

        # ---- thing masklets (SAM2-tracked) ----
        for j, oid in enumerate(thing_ids):
            det = thing_oid_map[oid]
            L_sem.append(det["label"])
            g = det["sem_group"]
            G_sem[j] = g
            W_sem[j] = DEFAULT_SEMANTIC_WEIGHTS.get(g, 0.15)
            source_type.append("thing_tracked")
            birth_frame.append(det.get("frame_idx", 0))

            for t in range(T):
                if t not in thing_segments or oid not in thing_segments[t]:
                    continue
                prob = thing_segments[t][oid]
                if prob.ndim == 3:
                    prob = prob.squeeze(0)
                if prob.shape[0] != H or prob.shape[1] != W:
                    prob = cv2.resize(prob.astype(np.float32), (W, H),
                                      interpolation=cv2.INTER_NEAREST)

                binary = prob.astype(bool)
                mask_area = binary.sum()
                if mask_area < 1:
                    continue

                M_mask[j, t] = torch.from_numpy(binary.astype(np.float32))
                Q_mask[j, t] = 1.0   # logit-thresholded, high confidence
                V_mask[j, t] = True
                A_ratio[j, t] = mask_area / area

                ys, xs = np.where(binary)
                B_mask[j, t] = torch.tensor(
                    [xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32,
                )

        # ---- structure anchors tracked with semantic video prompts ----
        for s, oid in enumerate(structure_ids):
            j = J_thing + s
            det = structure_oid_map[oid]
            L_sem.append(det["label"])
            g = det["sem_group"]
            G_sem[j] = g
            W_sem[j] = DEFAULT_SEMANTIC_WEIGHTS.get(g, 0.15)
            source_type.append("structure_tracked")
            birth_frame.append(det.get("frame_idx", 0))

            for t in range(T):
                if t not in structure_segments or oid not in structure_segments[t]:
                    continue
                prob = structure_segments[t][oid]
                if prob.ndim == 3:
                    prob = prob.squeeze(0)
                if prob.shape[0] != H or prob.shape[1] != W:
                    prob = cv2.resize(
                        prob.astype(np.float32),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    )

                binary = prob.astype(bool)
                mask_area = binary.sum()
                if mask_area < 1:
                    continue

                M_mask[j, t] = torch.from_numpy(binary.astype(np.float32))
                Q_mask[j, t] = 1.0
                V_mask[j, t] = True
                A_ratio[j, t] = mask_area / area

                ys, xs = np.where(binary)
                B_mask[j, t] = torch.tensor(
                    [xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32,
                )

        # ---- stuff masklets (stable chunk-level anchors) ----
        for s, entry in enumerate(stuff_entries):
            j = J_thing + J_structure + s
            L_sem.append(entry["label"])
            g = entry["sem_group"]
            G_sem[j] = g
            W_sem[j] = DEFAULT_SEMANTIC_WEIGHTS.get(g, 0.15)
            source_type.append("stuff_static")
            birth_frame.append(entry["birth_frame"])

            frame_to_mask = entry.get("frame_to_mask")
            if frame_to_mask is None:
                static_mask = entry["mask_static"].astype(bool)
                mask_area = static_mask.sum()
                if mask_area < 1:
                    continue

                t0 = entry["birth_frame"]
                M_mask[j, t0] = torch.from_numpy(static_mask.astype(np.float32))
                Q_mask[j, t0] = entry["confidence"]
                V_mask[j, t0] = True
                A_ratio[j, t0] = mask_area / area
                ys, xs = np.where(static_mask)
                B_mask[j, t0] = torch.tensor(
                    [xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32,
                )
                continue

            frame_to_confidence = entry.get("frame_to_confidence", {})
            for t, mask_np in frame_to_mask.items():
                if not (0 <= int(t) < T):
                    continue
                binary = mask_np.astype(bool)
                mask_area = binary.sum()
                if mask_area < 1:
                    continue

                M_mask[j, t] = torch.from_numpy(binary.astype(np.float32))
                Q_mask[j, t] = float(frame_to_confidence.get(t, 0.0))
                V_mask[j, t] = True
                A_ratio[j, t] = mask_area / area
                ys, xs = np.where(binary)
                B_mask[j, t] = torch.tensor(
                    [xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32,
                )

        return MaskletOutput(
            M_mask=M_mask, V_mask=V_mask, B_mask=B_mask, Q_mask=Q_mask,
            L_sem=L_sem, G_sem=G_sem, W_sem=W_sem, A_ratio=A_ratio,
            num_masklets=J, num_frames=T, frame_height=H, frame_width=W,
            source_type=source_type, birth_frame=birth_frame,
            debug={"J_thing": J_thing, "J_structure": J_structure, "J_stuff": J_stuff},
        )

    # -- Helpers -----------------------------------------------------------

    def _write_frames(self, images: torch.Tensor, out_dir: str) -> None:
        for t in range(images.shape[0]):
            img_np = (images[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"{t:05d}.jpg"), img_bgr)

    def _empty_output(self, T: int, H: int, W: int) -> MaskletOutput:
        return MaskletOutput(
            M_mask=torch.zeros(0, T, H, W), V_mask=torch.zeros(0, T, dtype=torch.bool),
            B_mask=torch.zeros(0, T, 4), Q_mask=torch.zeros(0, T),
            L_sem=[], G_sem=torch.zeros(0, dtype=torch.long),
            W_sem=torch.zeros(0), A_ratio=torch.zeros(0, T),
            num_masklets=0, num_frames=T, frame_height=H, frame_width=W,
        )
