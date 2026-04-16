"""
Stage C: Video Masklet Front-end  (v3)

Correct division of responsibility:
  * **Detector** (GDINO / YOLOE)  →  *what* is in the scene (labels)
  * **SAM 2.1 Video Predictor**   →  *where* each object is across time

The detector runs **once** on a single annotation frame.  SAM 2.1
handles all temporal tracking via its memory-attention mechanism.

Thing objects (chair, monitor …) are registered with SAM 2.1 for
video-level tracking.  Stuff regions (wall, floor …) are detected
per-keyframe independently – they have no sharp boundaries and SAM 2.1
is not designed to track amorphous regions.
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    "wall", "floor", "ceiling", "building", "road", "stair", "door",
    "window", "sidewalk", "bridge", "fence", "railing",
]
_STATIC_THING_LABELS = [
    "cabinet", "shelf", "desk", "table", "sofa", "bed", "refrigerator",
    "oven", "sink", "toilet", "bathtub", "counter", "bookshelf",
    "fixed furniture", "large appliance",
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


def label_to_group(label: str) -> int:
    label_lower = label.strip().lower()
    if label_lower in _LABEL_TO_GROUP:
        return _LABEL_TO_GROUP[label_lower]
    for key, group in _LABEL_TO_GROUP.items():
        if key in label_lower or label_lower in key:
            return group
    return SEMANTIC_GROUP_UNCERTAIN_REGION


def _is_stuff(sem_group: int) -> bool:
    return sem_group in _STUFF_GROUPS


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
    "bottle", "cup", "book", "phone", "car", "bicycle",
]
STUFF_PROMPTS = [
    "wall", "floor", "ceiling", "building", "door", "window",
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
    def __init__(self, grounding_model, image_predictor: SAM2ImagePredictor, device: str):
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

        self.image_predictor.set_image(image_source)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks, scores, _ = self.image_predictor.predict(
                point_coords=None, point_labels=None,
                box=input_boxes, multimask_output=False,
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        out = []
        for i in range(masks.shape[0]):
            lbl = labels[i] if i < len(labels) else "unknown"
            conf = confidences[i].item() if torch.is_tensor(confidences[i]) else float(confidences[i])
            out.append({"mask": masks[i], "box": input_boxes[i],
                        "confidence": conf, "label": lbl})
        return out

    def release_gpu(self):
        if hasattr(self.image_predictor, "model"):
            self.image_predictor.model.cpu()


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

    source_type: List[str] = field(default_factory=list)  # "thing_tracked" | "stuff_static"
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
        device: str = "cuda",
        thing_prompts: Optional[List[str]] = None,
        stuff_prompts: Optional[List[str]] = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        ann_frame_idx: int = 0,
        stuff_keyframe_stride: int = 10,
        min_mask_area_ratio: float = 0.001,
        max_mask_area_ratio: float = 0.95,
        nms_iou_threshold: float = 0.70,
        max_thing_objects: int = 15,
        prompt_type: str = "mask",
    ):
        self.video_predictor = video_predictor
        self.detector = detector
        self.device = device

        self.thing_prompts = thing_prompts or THING_PROMPTS
        self.stuff_prompts = stuff_prompts or STUFF_PROMPTS
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.ann_frame_idx = ann_frame_idx
        self.stuff_keyframe_stride = stuff_keyframe_stride
        self.min_mask_area_ratio = min_mask_area_ratio
        self.max_mask_area_ratio = max_mask_area_ratio
        self.nms_iou_threshold = nms_iou_threshold
        self.max_thing_objects = max_thing_objects
        self.prompt_type = prompt_type

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        device: str = "cuda",
        *,
        detector_type: str = "gdino",
        gdino_config: Optional[str] = None,
        gdino_checkpoint: Optional[str] = None,
        yoloe_model: str = "yoloe-11l-seg.pt",
        **kwargs,
    ) -> "VideoMaskletFrontend":
        video_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint)

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
            sam2_image_model = build_sam2(sam2_model_cfg, sam2_checkpoint)
            image_predictor = SAM2ImagePredictor(sam2_image_model)
            detector = GroundingDINODetector(grounding_model, image_predictor, device)
        elif detector_type == "yoloe":
            detector = YOLOEDetector(model_path=yoloe_model, device=device)
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        return cls(video_predictor=video_predictor, detector=detector,
                   device=device, **kwargs)

    # -- Public API --------------------------------------------------------

    def run(
        self,
        images: torch.Tensor,
        ann_frame_idx: Optional[int] = None,
    ) -> MaskletOutput:
        """Run Stage C.

        Parameters
        ----------
        images : [T, 3, H, W] float32 in [0, 1]
        ann_frame_idx : override annotation frame (default: self.ann_frame_idx)
        """
        T, C, H, W = images.shape
        assert C == 3
        ann_idx = ann_frame_idx if ann_frame_idx is not None else self.ann_frame_idx

        frame_dir = tempfile.mkdtemp(prefix="masklet_frames_")
        try:
            self._write_frames(images, frame_dir)
            return self._run_pipeline(frame_dir, T, H, W, ann_idx)
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    # -- Pipeline ----------------------------------------------------------

    def _run_pipeline(self, frame_dir: str, T: int, H: int, W: int,
                      ann_idx: int) -> MaskletOutput:
        area = H * W

        # ======== Step 1: detect on annotation frame ========
        detections = self._detect_and_filter(frame_dir, ann_idx, H, W)
        if not detections:
            return self._empty_output(T, H, W)

        # Split thing vs stuff
        thing_dets = [d for d in detections if not _is_stuff(d["sem_group"])]
        stuff_dets = [d for d in detections if _is_stuff(d["sem_group"])]

        # Prioritise thing objects: sort by area (large first) then confidence
        thing_dets.sort(key=lambda d: (d["mask"].astype(bool).sum() / area,
                                       d["confidence"]), reverse=True)
        if len(thing_dets) > self.max_thing_objects:
            thing_dets = thing_dets[: self.max_thing_objects]

        # ======== Step 2: free detector GPU → give memory to SAM2 ========
        self.detector.release_gpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ======== Step 3: SAM2 Video Predictor — track thing objects ========
        thing_segments, thing_oid_map = {}, {}
        if thing_dets:
            thing_segments, thing_oid_map = self._propagate_things(
                frame_dir, thing_dets, T, H, W,
            )

        # ======== Step 4: stuff — per-keyframe static masks ========
        stuff_entries = self._build_stuff_entries(
            stuff_dets, ann_idx, frame_dir, T, H, W,
        )

        # ======== Step 5: assemble ========
        return self._assemble(
            thing_segments, thing_oid_map, stuff_entries, T, H, W,
        )

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

    # -- Thing propagation (SAM2 Video Predictor) --------------------------

    @torch.inference_mode()
    def _propagate_things(
        self, frame_dir: str, thing_dets: List[Dict], T: int, H: int, W: int,
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
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

    # -- Stuff: static per-keyframe masks ----------------------------------

    def _build_stuff_entries(
        self, stuff_dets: List[Dict], ann_idx: int,
        frame_dir: str, T: int, H: int, W: int,
    ) -> List[Dict]:
        """Build stuff masklet entries.

        For stuff regions we replicate the annotation-frame mask across
        all frames – these regions (wall, floor) are spatially stable.
        """
        entries = []
        for det in stuff_dets:
            entries.append({
                "label": det["label"],
                "sem_group": det["sem_group"],
                "mask_static": det["mask"],    # [H, W] uint8
                "confidence": det["confidence"],
                "birth_frame": ann_idx,
            })
        return entries

    # -- Assembly ----------------------------------------------------------

    def _assemble(
        self,
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict],
        stuff_entries: List[Dict],
        T: int, H: int, W: int,
    ) -> MaskletOutput:
        thing_ids = sorted(thing_oid_map.keys())
        J_thing = len(thing_ids)
        J_stuff = len(stuff_entries)
        J = J_thing + J_stuff

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

        # ---- stuff masklets (annotation frame only) ----
        for s, entry in enumerate(stuff_entries):
            j = J_thing + s
            L_sem.append(entry["label"])
            g = entry["sem_group"]
            G_sem[j] = g
            W_sem[j] = DEFAULT_SEMANTIC_WEIGHTS.get(g, 0.15)
            source_type.append("stuff_static")
            birth_frame.append(entry["birth_frame"])

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

        return MaskletOutput(
            M_mask=M_mask, V_mask=V_mask, B_mask=B_mask, Q_mask=Q_mask,
            L_sem=L_sem, G_sem=G_sem, W_sem=W_sem, A_ratio=A_ratio,
            num_masklets=J, num_frames=T, frame_height=H, frame_width=W,
            source_type=source_type, birth_frame=birth_frame,
            debug={"J_thing": J_thing, "J_stuff": J_stuff},
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
