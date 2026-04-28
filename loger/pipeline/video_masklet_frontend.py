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
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
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
_PLANAR_STRUCTURE_LABELS = {"wall", "floor", "ceiling", "building", "road", "sidewalk"}
_GROUND_STRUCTURE_LABELS = {"floor", "road", "sidewalk", "stair"}
_OVERHEAD_STRUCTURE_LABELS = {"ceiling"}
_VERTICAL_STRUCTURE_LABELS = {"wall", "building"}
_THIN_STRUCTURE_LABELS = {"fence", "railing"}
_AMBIGUOUS_STRUCTURE_LABELS = {"bridge"}
_STRUCTURE_PROMPT_TEXT = {
    "wall": "wall surface",
    "floor": "floor surface",
    "ceiling": "ceiling surface",
    "building": "building facade",
    "road": "road surface",
    "stair": "stairs",
    "sidewalk": "sidewalk pavement",
    "bridge": "bridge structure",
    "fence": "fence barrier",
    "railing": "railing handrail",
}
_STATIC_THING_LABELS = [
    "cabinet", "shelf", "desk", "table", "sofa", "bed", "refrigerator",
    "oven", "sink", "toilet", "bathtub", "counter", "bookshelf",
    "fixed furniture", "large appliance", "door", "window",
]
_MOVABLE_THING_LABELS = [
    "person", "people", "rider", "bicycle", "motorcycle", "car", "bus",
    "truck", "train", "animal", "dog", "cat", "bird", "horse",
    "man", "woman", "singer", "dancer", "performer",
    "chair",
    "guitar", "musical instrument", "microphone", "microphone stand",
    # "box", "bag", "suitcase", "backpack", "bottle",
    # "cup", "book", "phone", "laptop", "keyboard", "mouse",
    "monitor",
]
_LOW_VALUE_STUFF_LABELS = [
    "sky", "tree", "vegetation", "grass", "plant", "flower", "water",
    "river", "sea", "ocean", "lake", "pool", "reflection", "glass",
    "mirror", "screen", "smoke", "fog", "cloud",
    "curtain",
]
_PERSON_ALIASES = {
    "person", "people", "man", "woman", "singer", "dancer", "performer", "rider"
}

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
    if a in _PERSON_ALIASES and b in _PERSON_ALIASES:
        return True
    return a == b or a in b or b in a


def _structure_prompt_text(label: str) -> str:
    canonical = canonicalize_label(label)
    return _STRUCTURE_PROMPT_TEXT.get(canonical, canonical)


def _structure_prompt_texts(label: str) -> Tuple[str, ...]:
    canonical = canonicalize_label(label)
    preferred = _STRUCTURE_PROMPT_TEXT.get(canonical, canonical)
    variants = [preferred, canonical]
    if canonical == "floor":
        variants.append("ground floor")
    elif canonical == "ceiling":
        variants.append("ceiling")
    elif canonical == "wall":
        variants.append("wall")
    elif canonical == "stair":
        variants.append("staircase")
    elif canonical == "sidewalk":
        variants.append("pavement sidewalk")
    elif canonical == "road":
        variants.append("street road")

    deduped: List[str] = []
    seen: Set[str] = set()
    for text in variants:
        text = str(text).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return tuple(deduped)


def _structure_geometry_stats(
    mask: np.ndarray,
    box: Optional[np.ndarray],
    H: int,
    W: int,
) -> Optional[Dict[str, float]]:
    mask_bool = np.asarray(mask).astype(bool)
    area = float(mask_bool.sum())
    if area <= 0.0 or H <= 0 or W <= 0:
        return None

    if box is None:
        ys, xs = np.where(mask_bool)
        if xs.size == 0 or ys.size == 0:
            return None
        x1 = float(xs.min())
        x2 = float(xs.max())
        y1 = float(ys.min())
        y2 = float(ys.max())
    else:
        box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
        if box_arr.size < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in box_arr[:4]]

    x1 = max(0.0, min(x1, float(W - 1)))
    x2 = max(0.0, min(x2, float(W - 1)))
    y1 = max(0.0, min(y1, float(H - 1)))
    y2 = max(0.0, min(y2, float(H - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw_px = max(x2 - x1 + 1.0, 1.0)
    bh_px = max(y2 - y1 + 1.0, 1.0)
    box_area = max(bw_px * bh_px, 1.0)
    Hf = float(max(H, 1))
    Wf = float(max(W, 1))
    ys = np.where(mask_bool)[0]
    top_mass = float((ys < 0.35 * Hf).sum()) / area
    mid_mass = float(((ys >= 0.30 * Hf) & (ys <= 0.70 * Hf)).sum()) / area
    bottom_mass = float((ys > 0.55 * Hf).sum()) / area

    return {
        "area_ratio": area / max(Hf * Wf, 1.0),
        "cx": ((x1 + x2) * 0.5) / Wf,
        "cy": ((y1 + y2) * 0.5) / Hf,
        "bw": bw_px / Wf,
        "bh": bh_px / Hf,
        "aspect": bw_px / max(bh_px, 1.0),
        "extent": area / box_area,
        "touch_top": float(y1 <= 0.03 * Hf),
        "touch_bottom": float(y2 >= 0.97 * Hf),
        "touch_left": float(x1 <= 0.03 * Wf),
        "touch_right": float(x2 >= 0.97 * Wf),
        "top_mass": top_mass,
        "mid_mass": mid_mass,
        "bottom_mass": bottom_mass,
    }


def passes_structure_mask_quality(
    label: str,
    mask: np.ndarray,
    box: Optional[np.ndarray],
    confidence: float,
    area_ratio: Optional[float],
    H: int,
    W: int,
) -> Tuple[bool, str]:
    """Reject structure masks whose geometry disagrees with the semantic label."""
    canonical = canonicalize_label(label)
    if label_to_group(canonical) != SEMANTIC_GROUP_STRUCTURE_ANCHOR:
        return True, "not_structure"

    stats = _structure_geometry_stats(mask, box, H, W)
    if stats is None:
        return False, "empty"
    ratio = float(area_ratio) if area_ratio is not None else float(stats["area_ratio"])
    conf = float(confidence)
    cy = float(stats["cy"])
    bw = float(stats["bw"])
    bh = float(stats["bh"])
    aspect = float(stats["aspect"])
    extent = float(stats["extent"])
    touch_top = bool(stats["touch_top"])
    touch_bottom = bool(stats["touch_bottom"])
    touch_left = bool(stats["touch_left"])
    touch_right = bool(stats["touch_right"])
    top_mass = float(stats["top_mass"])
    bottom_mass = float(stats["bottom_mass"])

    if ratio <= 0.0:
        return False, "empty"
    if ratio > 0.72:
        return False, "full_frame_structure_wash"

    if canonical in _THIN_STRUCTURE_LABELS:
        if ratio < 0.0008 or ratio > 0.22:
            return False, "thin_structure_area"
        elongated = aspect >= 1.8 or bh <= 0.22 or extent <= 0.42
        if not elongated:
            return False, "thin_structure_not_thin"
        if conf < 0.30 and ratio < 0.006:
            return False, "thin_structure_weak"
        return True, "ok"

    if canonical in _AMBIGUOUS_STRUCTURE_LABELS:
        if ratio < 0.004 or ratio > 0.35:
            return False, "ambiguous_structure_area"
        if conf < 0.42 and ratio < 0.02:
            return False, "ambiguous_structure_weak"
        return True, "ok"

    if canonical in _GROUND_STRUCTURE_LABELS:
        min_ratio = 0.006 if canonical == "stair" else 0.010
        if ratio < min_ratio:
            return False, "ground_structure_tiny"
        if ratio > 0.58 and touch_top and touch_bottom:
            return False, "ground_structure_full_height"
        if cy < 0.42 and not touch_bottom:
            return False, "ground_structure_too_high"
        if top_mass > 0.55 and bottom_mass < 0.28:
            return False, "ground_structure_top_heavy"
        return True, "ok"

    if canonical in _OVERHEAD_STRUCTURE_LABELS:
        if ratio < 0.008:
            return False, "ceiling_tiny"
        if cy > 0.60 and not touch_top:
            return False, "ceiling_too_low"
        if bottom_mass > 0.58 and top_mass < 0.28:
            return False, "ceiling_bottom_heavy"
        if ratio > 0.58 and touch_top and touch_bottom:
            return False, "ceiling_full_height"
        return True, "ok"

    if canonical in _VERTICAL_STRUCTURE_LABELS:
        if ratio < 0.010:
            return False, "vertical_structure_tiny"
        if ratio > 0.64 and touch_top and touch_bottom and touch_left and touch_right:
            return False, "vertical_structure_full_frame"
        if canonical == "wall" and ratio > 0.50 and cy > 0.68 and bottom_mass > 0.65:
            return False, "wall_looks_like_floor"
        return True, "ok"

    if canonical in _PLANAR_STRUCTURE_LABELS and ratio < 0.008:
        return False, "planar_structure_tiny"

    if bw < 0.03 or bh < 0.03:
        return False, "structure_box_tiny"
    return True, "ok"


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def _clean_instance_mask_components(
    mask: np.ndarray,
    sem_group: int,
    label: str,
    *,
    image_area: int,
) -> np.ndarray:
    """Keep instance masks from carrying far-away SAM fragments.

    SAM video propagation can occasionally attach small disconnected islands
    from other people/objects. Those islands make boxes jump across the frame
    and weaken cross-chunk matching, so tracked instance masks are reduced to
    their dominant connected support. Stuff masks are intentionally not passed
    through this helper.
    """
    binary = mask.astype(bool)
    if int(sem_group) not in {
        SEMANTIC_GROUP_MOVABLE_THING,
        SEMANTIC_GROUP_STATIC_THING,
    }:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8,
    )
    if num_labels <= 2:
        return binary

    comp_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    if comp_areas.size == 0:
        return binary

    order = np.argsort(comp_areas)[::-1]
    largest = float(comp_areas[order[0]])
    if largest <= 0.0:
        return binary

    label_l = str(label).strip().lower()
    is_person = _labels_compatible(label_l, "person") or _labels_compatible(label_l, "people")
    single_instance = int(sem_group) == SEMANTIC_GROUP_MOVABLE_THING
    if is_person:
        single_instance = True

    keep = np.zeros_like(binary, dtype=bool)
    max_components = 1 if single_instance else 3
    min_component_area = max(12.0, 0.00002 * float(max(image_area, 1)))
    if is_person:
        # SAM can split a person into torso/legs/hands when the body is partly
        # occluded. Keep nearby secondary components, but still reject far-away
        # islands that usually belong to another person.
        max_components = 3
        min_component_area = max(min_component_area, 0.035 * largest)
    if not single_instance:
        min_component_area = max(min_component_area, 0.08 * largest)

    kept = 0
    main_rank = int(order[0])
    mx = float(stats[main_rank + 1, cv2.CC_STAT_LEFT])
    my = float(stats[main_rank + 1, cv2.CC_STAT_TOP])
    mw = float(stats[main_rank + 1, cv2.CC_STAT_WIDTH])
    mh = float(stats[main_rank + 1, cv2.CC_STAT_HEIGHT])
    main_box = (mx, my, mx + mw, my + mh)
    main_cx = mx + 0.5 * mw
    main_cy = my + 0.5 * mh

    for comp_rank in order:
        comp_area = float(comp_areas[comp_rank])
        if kept > 0 and comp_area < min_component_area:
            continue
        if is_person and kept > 0:
            x = float(stats[int(comp_rank) + 1, cv2.CC_STAT_LEFT])
            y = float(stats[int(comp_rank) + 1, cv2.CC_STAT_TOP])
            w = float(stats[int(comp_rank) + 1, cv2.CC_STAT_WIDTH])
            h = float(stats[int(comp_rank) + 1, cv2.CC_STAT_HEIGHT])
            cx = x + 0.5 * w
            cy = y + 0.5 * h

            expand_x = max(14.0, 0.45 * max(mw, 1.0))
            expand_y = max(18.0, 0.60 * max(mh, 1.0))
            expanded_main = (
                main_box[0] - expand_x,
                main_box[1] - expand_y,
                main_box[2] + expand_x,
                main_box[3] + expand_y,
            )
            overlaps_expanded = not (
                x + w < expanded_main[0]
                or x > expanded_main[2]
                or y + h < expanded_main[1]
                or y > expanded_main[3]
            )
            center_dx = abs(cx - main_cx) / max(0.5 * (mw + w), 1.0)
            center_dy = abs(cy - main_cy) / max(0.5 * (mh + h), 1.0)
            close_to_main = center_dx <= 1.25 and center_dy <= 1.35
            sizeable_component = comp_area >= 0.16 * largest
            if not (overlaps_expanded or close_to_main or sizeable_component):
                continue
        keep |= labels == int(comp_rank + 1)
        kept += 1
        if kept >= max_components:
            break

    return keep if keep.any() else binary


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

THING_PROMPTS = list(dict.fromkeys(_MOVABLE_THING_LABELS + _STATIC_THING_LABELS))
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
    def detect_batch(
        self,
        image_paths,
        thing_prompts,
        stuff_prompts,
        box_threshold,
        text_threshold,
    ) -> List[List[Dict[str, Any]]]:
        return [
            self.detect(path, thing_prompts, stuff_prompts, box_threshold, text_threshold)
            for path in image_paths
        ]
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


def _resolve_yoloe_model_path(model_path: str) -> str:
    model_path = str(model_path or "").strip()
    hf_aliases = {
        "yoloe26-l-seg": "openvision/yoloe26-l-seg",
        "yoloe26-l-seg.pt": "openvision/yoloe26-l-seg",
        "openvision/yoloe26-l-seg": "openvision/yoloe26-l-seg",
    }
    repo_id = None
    if model_path.startswith("hf://"):
        repo_id = model_path[len("hf://") :]
    elif model_path in hf_aliases:
        repo_id = hf_aliases[model_path]

    if repo_id is None:
        return model_path

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            f"YOLOE model {model_path!r} requires huggingface_hub to download {repo_id!r}"
        ) from exc

    # The OpenVision YOLOE-26 segmentation repos publish the Ultralytics
    # checkpoint as model.pt.
    return hf_hub_download(repo_id=repo_id, filename="model.pt")


class YOLOEDetector(BaseDetector):
    def __init__(
        self,
        model_path: str = "yoloe-11l-seg.pt",
        device: str = "cuda",
        batch_size: int = 8,
        imgsz: int = 0,
    ):
        from ultralytics import YOLO
        self.model_path = _resolve_yoloe_model_path(model_path)
        self.model = YOLO(self.model_path)
        self.device = device
        self.batch_size = max(int(batch_size), 1)
        self.imgsz = max(int(imgsz), 0)
        self._prompts_set: Optional[str] = None
        self.model.to(device)

    def _set_prompts(self, thing_prompts, stuff_prompts) -> None:
        all_prompts = thing_prompts + stuff_prompts
        key = ",".join(sorted(all_prompts))
        if self._prompts_set != key:
            if hasattr(self.model, "get_text_pe"):
                self.model.set_classes(all_prompts, self.model.get_text_pe(all_prompts))
            else:
                self.model.set_classes(all_prompts)
            self._prompts_set = key

    def _result_to_detections(self, result) -> List[Dict[str, Any]]:
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names
        has_masks = result.masks is not None and result.masks.data is not None
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        masks_data = result.masks.data.cpu().numpy() if has_masks else None

        out = []
        for i in range(len(boxes_xyxy)):
            label = names.get(cls_ids[i], "unknown")
            if masks_data is not None:
                mask = masks_data[i]
                if mask.shape[0] != result.orig_shape[0] or mask.shape[1] != result.orig_shape[1]:
                    mask = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
            else:
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                h, w = result.orig_shape
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
            out.append({"mask": mask, "box": boxes_xyxy[i],
                        "confidence": float(confs[i]), "label": label})
        return out

    def detect(self, image_path, thing_prompts, stuff_prompts, box_threshold, text_threshold):
        self._set_prompts(thing_prompts, stuff_prompts)
        # Use YOLOE primarily as a high-recall seed generator; SAM3 does the
        # temporally-consistent filtering/tracking afterwards.
        yolo_conf = min(float(box_threshold), 0.18)
        predict_kwargs = {
            "conf": yolo_conf,
            "verbose": False,
            "batch": self.batch_size,
        }
        if self.imgsz > 0:
            predict_kwargs["imgsz"] = self.imgsz
        results = self.model.predict(image_path, **predict_kwargs)
        if not results:
            return []
        return self._result_to_detections(results[0])

    def detect_batch(self, image_paths, thing_prompts, stuff_prompts, box_threshold, text_threshold):
        if not image_paths:
            return []
        self._set_prompts(thing_prompts, stuff_prompts)
        yolo_conf = min(float(box_threshold), 0.18)
        predict_kwargs = {
            "conf": yolo_conf,
            "verbose": False,
            "batch": self.batch_size,
        }
        if self.imgsz > 0:
            predict_kwargs["imgsz"] = self.imgsz
        results = self.model.predict(list(image_paths), **predict_kwargs)
        return [self._result_to_detections(result) for result in results]

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
    seed_global_track_idx: List[Optional[int]] = field(default_factory=list)
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
        discovery_frame_stride: int = 4,
        stuff_keyframe_stride: int = 10,
        min_mask_area_ratio: float = 0.001,
        max_mask_area_ratio: float = 0.95,
        nms_iou_threshold: float = 0.70,
        discovery_match_iou_threshold: float = 0.50,
        max_thing_objects: int = 18,
        prompt_type: str = "mask",
        sam31_offload_video_to_cpu: bool = True,
        sam31_offload_state_to_cpu: bool = True,
        sam31_offload_sam_during_detection: bool = False,
        sam31_text_track_labels: Optional[List[str]] = None,
        sam31_structure_prompt_labels: Optional[List[str]] = None,
        sam31_structure_prompt_frame_count: int = 1,
        sam31_structure_prompt_chunk_stride: int = 1,
        sam31_person_refresh_prompt_frames: int = 0,
        sam31_nontext_object_prompt_budget: int = 0,
        sam31_nontext_object_prompt_min_support: int = 2,
        sam31_nontext_sparse_support: bool = False,
        sam31_max_text_prompt_objects: int = 0,
        sam31_max_movable_objects: int = 4,
        sam31_max_static_objects: int = 1,
        sam31_max_structure_objects: int = 3,
        sam31_enable_backward: bool = False,
    ):
        self.video_predictor = video_predictor
        self.detector = detector
        self.sam_backend = sam_backend
        self.device = device

        self.thing_prompts = THING_PROMPTS if thing_prompts is None else list(thing_prompts)
        self.stuff_prompts = STUFF_PROMPTS if stuff_prompts is None else list(stuff_prompts)
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
        self.sam31_offload_state_to_cpu = sam31_offload_state_to_cpu
        self.sam31_offload_sam_during_detection = bool(sam31_offload_sam_during_detection)
        self.sam31_text_track_labels = (
            [str(lbl).strip().lower() for lbl in sam31_text_track_labels if str(lbl).strip()]
            if sam31_text_track_labels is not None
            else ["person"]
        )
        self.sam31_structure_prompt_labels = (
            [str(lbl).strip().lower() for lbl in sam31_structure_prompt_labels if str(lbl).strip()]
            if sam31_structure_prompt_labels is not None
            else ["wall", "floor", "ceiling"]
        )
        self.sam31_structure_prompt_frame_count = max(int(sam31_structure_prompt_frame_count), 0)
        self.sam31_structure_prompt_chunk_stride = max(
            int(sam31_structure_prompt_chunk_stride), 0
        )
        self.sam31_person_refresh_prompt_frames = max(
            int(sam31_person_refresh_prompt_frames), 0
        )
        self.sam31_nontext_object_prompt_budget = max(
            int(sam31_nontext_object_prompt_budget), 0
        )
        self.sam31_nontext_object_prompt_min_support = max(
            int(sam31_nontext_object_prompt_min_support), 1
        )
        self.sam31_nontext_sparse_support = bool(sam31_nontext_sparse_support)
        self.sam31_max_text_prompt_objects = max(int(sam31_max_text_prompt_objects), 0)
        self.sam31_max_movable_objects = max(int(sam31_max_movable_objects), 0)
        self.sam31_max_static_objects = max(int(sam31_max_static_objects), 0)
        self.sam31_max_structure_objects = max(int(sam31_max_structure_objects), 0)
        self.sam31_enable_backward = bool(sam31_enable_backward)
        self._sam31_run_call_index = 0

    def _sam31_prompt_only_budget(self, label: str, sem_group: int) -> int:
        """Allow SAM3.1 text-prompt instances that YOLOE failed to seed.

        YOLOE detections are still used as high-confidence anchors, but for
        categories like person the SAM3.1 text prompt can discover additional
        instances in the same query. Keep this narrow and capped to avoid the
        noisy prompt-only explosion seen when every label is admitted.
        """
        if sem_group != SEMANTIC_GROUP_MOVABLE_THING:
            return 0
        if _labels_compatible(label, "person") or _labels_compatible(label, "people"):
            return min(8, max(int(self.max_thing_objects), 0))
        return min(1, max(int(self.max_thing_objects), 0))

    def _sam31_prompt_only_filters(
        self,
        label: str,
        sem_group: int,
        T: int,
    ) -> Tuple[int, float, float, float]:
        if sem_group == SEMANTIC_GROUP_MOVABLE_THING and (
            _labels_compatible(label, "person") or _labels_compatible(label, "people")
        ):
            # Keep prompt-only person recovery for detector misses, but reject
            # tiny body-part fragments that otherwise survive as extra IDs.
            return max(6, min(int(T), int(round(T * 0.18)))), 0.22, 0.0100, 0.60
        if sem_group == SEMANTIC_GROUP_MOVABLE_THING:
            return max(6, min(int(T), int(round(T * 0.25)))), 0.45, 0.0010, 0.35
        return max(6, min(int(T), int(round(T * 0.25)))), 0.50, 0.0020, 0.50

    def _uses_sam31_text_tracking(self, label: str) -> bool:
        if self.sam31_text_track_labels is None:
            return True
        # Keep this intentionally stricter than semantic compatibility.
        # "performer"/"singer"/"dancer" are person-like and useful detector
        # supports, but running a separate SAM3.1 text propagation for each
        # alias duplicates work and often fragments the same people into many
        # IDs. Non-exact aliases still enter the sparse support path below.
        label_l = str(label).strip().lower()
        return label_l in {str(allowed).strip().lower() for allowed in self.sam31_text_track_labels}

    def _select_sam31_structure_prompt_frames(
        self,
        discovery_indices: List[int],
        T: int,
    ) -> List[int]:
        if self.sam31_structure_prompt_frame_count <= 0 or not discovery_indices:
            return []
        available = sorted({int(idx) for idx in discovery_indices if 0 <= int(idx) < T})
        if len(available) <= self.sam31_structure_prompt_frame_count:
            return available

        if self.sam31_structure_prompt_frame_count == 1:
            return [available[len(available) // 2]]

        chosen: List[int] = []
        for pos in np.linspace(0, len(available) - 1, self.sam31_structure_prompt_frame_count):
            chosen.append(available[int(round(float(pos)))])
        return sorted(set(chosen))

    def _resolve_sam31_run_index(self, chunk_index: Optional[int]) -> int:
        if chunk_index is not None:
            return max(int(chunk_index), 0)
        run_index = self._sam31_run_call_index
        self._sam31_run_call_index += 1
        return run_index

    def _should_run_sam31_structure_prompts(self, sam31_run_index: int) -> bool:
        stride = int(self.sam31_structure_prompt_chunk_stride)
        if stride <= 0:
            return False
        return int(sam31_run_index) % stride == 0

    def _select_sam31_primary_prompt_frame(
        self,
        label: str,
        candidate_dets: List[Dict[str, Any]],
        T: int,
    ) -> int:
        """Pick a text-prompt frame that balances recall and handoff stability."""
        if not candidate_dets:
            return 0

        min_frame = min(int(det.get("frame_idx", 0)) for det in candidate_dets)
        if not self.sam31_enable_backward:
            # In forward-only mode, a later prompt cannot recover masks for
            # earlier overlap/seed frames. Start from the earliest available
            # support and rely on refresh prompts for newly appearing objects.
            return min_frame

        label_l = str(label).strip().lower()
        is_person = (
            _labels_compatible(label_l, "person")
            or _labels_compatible(label_l, "people")
        )
        if not is_person:
            return min_frame

        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        last_safe_frame = max(int(T) - 2, 0)
        for det in candidate_dets:
            frame_idx = int(det.get("frame_idx", 0))
            frame_idx = min(max(frame_idx, 0), last_safe_frame)
            by_frame.setdefault(frame_idx, []).append(det)

        if not by_frame:
            return min_frame

        # A person text prompt only discovers people visible on the prompt
        # frame. Prefer frames with more non-seed/person support, but keep a
        # center bias so reverse propagation is not asked to recover an entire
        # chunk from the last frame.
        best_frame = min(by_frame.keys())
        best_score = -1e9
        center = 0.5 * float(max(int(T) - 1, 1))
        for frame_idx, dets in by_frame.items():
            cluster_ids = {
                int(det["_cluster_id"])
                for det in dets
                if det.get("_cluster_id", None) is not None
            }
            unique_count = len(cluster_ids) if cluster_ids else len(dets)
            seed_count = sum(1 for det in dets if bool(det.get("is_seed_track", False)))
            non_seed_count = max(len(dets) - seed_count, 0)
            total_area = sum(float(det.get("area_ratio", 0.0)) for det in dets)
            mean_conf = sum(float(det.get("confidence", 0.0)) for det in dets) / max(len(dets), 1)
            center_score = 1.0 - min(abs(float(frame_idx) - center) / max(center, 1.0), 1.0)
            later_score = float(frame_idx) / max(float(max(int(T) - 1, 1)), 1.0)

            score = (
                3.0 * float(non_seed_count)
                + 1.4 * float(unique_count)
                + 1.0 * min(float(seed_count), 2.0)
                + 4.0 * min(float(total_area), 0.45)
                + 0.8 * float(mean_conf)
                + 0.35 * center_score
                + 0.15 * later_score
            )
            if score > best_score or (score == best_score and frame_idx < best_frame):
                best_score = score
                best_frame = int(frame_idx)

        return int(best_frame)

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
        yoloe_imgsz: int = 0,
        **kwargs,
    ) -> "VideoMaskletFrontend":
        sam31_offload_video_to_cpu = bool(kwargs.pop("sam31_offload_video_to_cpu", True))
        sam31_offload_outputs_to_cpu = bool(kwargs.pop("sam31_offload_outputs_to_cpu", True))
        sam31_offload_sam_during_detection = bool(
            kwargs.pop("sam31_offload_sam_during_detection", False)
        )
        sam31_text_track_labels_arg = kwargs.pop("sam31_text_track_labels", None)
        if isinstance(sam31_text_track_labels_arg, str):
            if sam31_text_track_labels_arg.strip().lower() == "all":
                sam31_text_track_labels = None
            else:
                sam31_text_track_labels = [
                    s.strip() for s in sam31_text_track_labels_arg.split(",") if s.strip()
                ]
        else:
            sam31_text_track_labels = sam31_text_track_labels_arg
        sam31_structure_prompt_labels_arg = kwargs.pop("sam31_structure_prompt_labels", None)
        if isinstance(sam31_structure_prompt_labels_arg, str):
            sam31_structure_prompt_labels = [
                s.strip() for s in sam31_structure_prompt_labels_arg.split(",") if s.strip()
            ]
        else:
            sam31_structure_prompt_labels = sam31_structure_prompt_labels_arg
        sam31_structure_prompt_frame_count = int(
            kwargs.pop("sam31_structure_prompt_frame_count", 1)
        )
        sam31_structure_prompt_chunk_stride = int(
            kwargs.pop("sam31_structure_prompt_chunk_stride", 1)
        )
        sam31_person_refresh_prompt_frames = int(
            kwargs.pop("sam31_person_refresh_prompt_frames", 0)
        )
        sam31_enable_backward = bool(kwargs.pop("sam31_enable_backward", False))
        sam31_nontext_object_prompt_budget = int(
            kwargs.pop("sam31_nontext_object_prompt_budget", 0)
        )
        sam31_nontext_object_prompt_min_support = int(
            kwargs.pop("sam31_nontext_object_prompt_min_support", 2)
        )
        sam31_nontext_sparse_support = bool(
            kwargs.pop("sam31_nontext_sparse_support", True)
        )
        sam31_max_text_prompt_objects = int(
            kwargs.pop("sam31_max_text_prompt_objects", 0)
        )
        sam31_max_internal_objects = max(
            int(kwargs.pop("sam31_max_internal_objects", 12)), 1
        )
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
                    use_fa3=False,
                    use_rope_real=False,
                    # The public SAM3.1 multiplex checkpoint is trained with
                    # 16 object slots. Keep the model slot count checkpoint-
                    # compatible; the frontend may still aggregate more than
                    # 16 final tracks across multiple text-prompt refreshes.
                    max_num_objects=sam31_max_internal_objects,
                    multiplex_count=16,
                    compile=False,
                    warm_up=False,
                    async_loading_frames=False,
                )
                if hasattr(video_predictor, "model"):
                    if hasattr(video_predictor.model, "postprocess_batch_size"):
                        video_predictor.model.postprocess_batch_size = sam31_postprocess_batch_size
                    if hasattr(video_predictor.model, "batched_grounding_batch_size"):
                        video_predictor.model.batched_grounding_batch_size = (
                            sam31_batched_grounding_batch_size
                        )
                    detector = getattr(video_predictor.model, "detector", None)
                    if (
                        detector is not None
                        and hasattr(detector, "offload_outputs_to_cpu_for_eval")
                    ):
                        detector.offload_outputs_to_cpu_for_eval = (
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
            yoloe_batch_size = int(kwargs.pop("yoloe_batch_size", 8))
            detector = YOLOEDetector(
                model_path=yoloe_model,
                device=device,
                batch_size=yoloe_batch_size,
                imgsz=yoloe_imgsz,
            )
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        return cls(video_predictor=video_predictor, detector=detector,
                   sam_backend=sam_backend,
                   device=device,
                   sam31_offload_video_to_cpu=sam31_offload_video_to_cpu,
                   sam31_offload_state_to_cpu=sam31_offload_outputs_to_cpu,
                   sam31_offload_sam_during_detection=sam31_offload_sam_during_detection,
                   sam31_text_track_labels=sam31_text_track_labels,
                   sam31_structure_prompt_labels=sam31_structure_prompt_labels,
                   sam31_structure_prompt_frame_count=sam31_structure_prompt_frame_count,
                   sam31_structure_prompt_chunk_stride=sam31_structure_prompt_chunk_stride,
                   sam31_person_refresh_prompt_frames=sam31_person_refresh_prompt_frames,
                   sam31_nontext_object_prompt_budget=sam31_nontext_object_prompt_budget,
                   sam31_nontext_object_prompt_min_support=sam31_nontext_object_prompt_min_support,
                   sam31_nontext_sparse_support=sam31_nontext_sparse_support,
                   sam31_max_text_prompt_objects=sam31_max_text_prompt_objects,
                   sam31_enable_backward=sam31_enable_backward,
                   **kwargs)

    # -- Public API --------------------------------------------------------

    def run(
        self,
        images: torch.Tensor,
        ann_frame_idx: Optional[int] = None,
        discovery_frame_indices: Optional[List[int]] = None,
        seed_detections_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        chunk_index: Optional[int] = None,
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
        sam31_run_index = self._resolve_sam31_run_index(chunk_index)

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
                sam31_run_index=sam31_run_index,
            )
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def run_from_paths(
        self,
        image_paths: List[str],
        ann_frame_idx: Optional[int] = None,
        discovery_frame_indices: Optional[List[int]] = None,
        seed_detections_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        chunk_index: Optional[int] = None,
        detector_image_paths: Optional[List[str]] = None,
    ) -> MaskletOutput:
        """Run Stage C directly from existing frame files.

        This avoids decoding the chunk into a tensor and then re-encoding it
        back to temporary jpg files before detector / SAM3.1 consume it.
        """
        if not image_paths:
            raise ValueError("image_paths must be non-empty")
        if detector_image_paths is not None and len(detector_image_paths) != len(image_paths):
            raise ValueError("detector_image_paths must have the same length as image_paths")

        sample = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
        if sample is None:
            raise FileNotFoundError(f"Failed to read frame: {image_paths[0]}")

        T = len(image_paths)
        H, W = sample.shape[:2]
        ann_idx = ann_frame_idx if ann_frame_idx is not None else self.ann_frame_idx
        discovery_indices = self._build_discovery_frame_indices(
            T, ann_idx, discovery_frame_indices,
        )
        sam31_run_index = self._resolve_sam31_run_index(chunk_index)

        self.detector.to_device()

        frame_dir = tempfile.mkdtemp(prefix="masklet_links_")
        try:
            self._link_frames(image_paths, frame_dir)
            return self._run_pipeline(
                frame_dir,
                T,
                H,
                W,
                discovery_indices,
                seed_detections_by_frame=seed_detections_by_frame,
                detector_image_paths=detector_image_paths or image_paths,
                sam31_run_index=sam31_run_index,
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

    def _start_sam31_session(self, frame_resource: Any) -> str:
        request = dict(
            type="start_session",
            resource_path=frame_resource,
            offload_video_to_cpu=self.sam31_offload_video_to_cpu,
            offload_state_to_cpu=self.sam31_offload_state_to_cpu,
        )
        return self.video_predictor.handle_request(request)["session_id"]

    def _move_sam31_predictor_model(self, device: str) -> None:
        predictor_model = getattr(self.video_predictor, "model", None)
        if predictor_model is not None and hasattr(predictor_model, "to"):
            predictor_model.to(device)

    def _query_sam31_label_tracks(
        self,
        frame_resource: Any,
        session_id: str,
        label: str,
        prompt_frame: int,
        *,
        log_prefix: str = "",
    ) -> Tuple[str, Dict[int, Dict[int, np.ndarray]], Dict[int, float]]:
        # SAM3.1 multiplex long runs can leave tracker-side state behind even
        # after reset_session. We still try the cheap reset first, but on the
        # known state/cache failures we reopen a fresh session.
        def _is_retriable_failure(exc: BaseException) -> bool:
            msg = str(exc)
            return (
                isinstance(exc, KeyError)
                or "No points are provided; please add points first" in msg
                or "cached_frame_outputs" in msg
            )

        self.video_predictor.handle_request(
            dict(type="reset_session", session_id=session_id)
        )
        try:
            obj_segments, obj_scores = self._track_label_with_sam31_multiplex_session(
                session_id=session_id,
                label=label,
                frame_idx=prompt_frame,
            )
            return session_id, obj_segments, obj_scores
        except (RuntimeError, KeyError) as exc:
            if not _is_retriable_failure(exc):
                raise
            print(
                f"[sam31_multiplex] warning: retrying {log_prefix}label={label!r} "
                f"prompt_frame={prompt_frame} after session/cache failure: {exc!r}"
            )

        try:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )
        except Exception:
            pass

        session_id = self._start_sam31_session(frame_resource)
        try:
            obj_segments, obj_scores = self._track_label_with_sam31_multiplex_session(
                session_id=session_id,
                label=label,
                frame_idx=prompt_frame,
            )
            return session_id, obj_segments, obj_scores
        except (RuntimeError, KeyError) as retry_exc:
            if not _is_retriable_failure(retry_exc):
                raise
            print(
                f"[sam31_multiplex] skipping {log_prefix}label={label!r} "
                f"prompt_frame={prompt_frame} after repeated "
                f"SAM3.1 session/cache failures: {retry_exc!r}"
            )
            return session_id, {}, {}

    def _query_sam31_prompt_frame_detections(
        self,
        frame_resource: Any,
        session_id: str,
        label: str,
        frame_idx: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.video_predictor.handle_request(
            dict(type="reset_session", session_id=session_id)
        )
        try:
            response = self.video_predictor.handle_request(
                dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=int(frame_idx),
                    text=str(label),
                )
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
            session_id = self._start_sam31_session(frame_resource)
            try:
                response = self.video_predictor.handle_request(
                    dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=int(frame_idx),
                        text=str(label),
                    )
                )
            except RuntimeError as retry_exc:
                if "No points are provided; please add points first" not in str(retry_exc):
                    raise
                return session_id, []

        outputs = response.get("outputs", {})
        masks = outputs.get("out_binary_masks", [])
        probs = outputs.get("out_probs", [])
        dets: List[Dict[str, Any]] = []
        for idx, mask in enumerate(masks):
            mask_bool = np.asarray(mask).astype(bool)
            mask_area = int(mask_bool.sum())
            if mask_area <= 0:
                continue
            area_ratio = float(mask_area) / max(int(area), 1)
            # SAM text prompts for structure may return tiny fragments; keep
            # only support regions large enough to be useful for write priors.
            if area_ratio < 0.012 or area_ratio > self.max_mask_area_ratio:
                continue
            confidence = float(probs[idx]) if idx < len(probs) else 0.0
            box = self._mask_to_box_xyxy(mask_bool)
            if float(box[2] - box[0]) <= 1.0 or float(box[3] - box[1]) <= 1.0:
                continue
            canonical_label = canonicalize_label(label)
            if confidence < 0.20 and area_ratio < 0.04:
                continue
            keep, _ = passes_structure_mask_quality(
                canonical_label, mask_bool, box, confidence, area_ratio, H, W,
            )
            if not keep:
                continue
            dets.append(
                {
                    "mask": mask_bool.astype(np.uint8),
                    "box": box.astype(np.float32),
                    "confidence": confidence,
                    "label": canonical_label,
                    "raw_label": str(label),
                    "sem_group": label_to_group(canonical_label),
                    "area_ratio": area_ratio,
                    "frame_idx": int(frame_idx),
                    "detector_source": "sam31_prompt_frame",
                    "is_prompt_only": True,
                }
            )

        dets.sort(
            key=lambda d: (
                -float(d.get("confidence", 0.0)),
                -float(d.get("area_ratio", 0.0)),
            )
        )
        return session_id, dets[:2]

    def _select_multiframe_things(
        self,
        frame_resource: Any,
        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        if self.sam_backend == "sam3":
            return self._select_multiframe_things_sam3(
                frame_resource, thing_dets_by_frame, discovery_indices, T, H, W, area,
            )
        if self.sam_backend == "sam31_multiplex":
            return self._select_multiframe_things_sam31_multiplex(
                frame_resource, thing_dets_by_frame, discovery_indices, T, H, W, area,
            )

        selected: List[Dict[str, Any]] = []
        thing_segments: Dict[int, Dict[int, np.ndarray]] = {}
        thing_oid_map: Dict[int, Dict] = {}

        candidate_frame_indices = sorted(
            {int(k) for k in discovery_indices}.union(int(k) for k in thing_dets_by_frame.keys())
        )
        for frame_idx in candidate_frame_indices:
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
                    frame_resource, selected, T, H, W,
                )

            if len(selected) >= self.max_thing_objects:
                break

        return selected, thing_segments, thing_oid_map

    def _select_multiframe_things_sam31_multiplex(
        self,
        frame_resource: Any,
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
        candidate_frame_indices = sorted(
            {int(k) for k in discovery_indices}.union(int(k) for k in thing_dets_by_frame.keys())
        )
        for frame_idx in candidate_frame_indices:
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

        session_id = self._start_sam31_session(frame_resource)
        object_prompt_dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}

        try:
            label_order = sorted(
                label_to_dets.keys(),
                key=lambda lbl: min(int(det.get("frame_idx", 0)) for det in label_to_dets[lbl]),
            )
            for label in label_order:
                if len(selected) >= self.max_thing_objects:
                    break

                raw_dets = sorted(
                    label_to_dets[label],
                    key=lambda d: (
                        -int(bool(d.get("is_seed_track", False))),
                        int(d.get("frame_idx", 0)),
                        -(d["mask"].astype(bool).sum() / max(area, 1)),
                        -float(d["confidence"]),
                    ),
                )
                candidate_dets: List[Dict[str, Any]] = []
                remaining_track_budget = max(int(self.max_thing_objects) - len(selected), 1)
                # Keep a wider temporal candidate pool than the final object
                # budget. Otherwise early-frame duplicates can consume the
                # pre-SAM quota and late-entering objects never reach the
                # clustering / refresh logic at all.
                candidate_pool_limit = max(
                    remaining_track_budget * max(4, min(len(discovery_indices), 8)),
                    remaining_track_budget + 8,
                )
                for det in raw_dets:
                    if len(candidate_dets) >= candidate_pool_limit:
                        break
                    if self._is_duplicate_thing_detection(
                        det, int(det.get("frame_idx", 0)), thing_segments, thing_oid_map
                    ):
                        continue
                    candidate_dets.append(det)

                if not candidate_dets:
                    continue
                candidate_dets = self._assign_sam31_detection_cluster_ids(
                    candidate_dets,
                    area=area,
                )
                if not self._uses_sam31_text_tracking(label):
                    for det in candidate_dets:
                        object_prompt_dets_by_frame.setdefault(
                            int(det.get("frame_idx", 0)),
                            [],
                        ).append(dict(det))
                    continue

                prompt_frame = self._select_sam31_primary_prompt_frame(
                    label,
                    candidate_dets,
                    T,
                )
                session_id, obj_segments, obj_scores = self._query_sam31_label_tracks(
                    frame_resource,
                    session_id,
                    label,
                    int(prompt_frame),
                )
                if not obj_segments:
                    continue

                matched_by_prompt_obj_id: Dict[int, List[Tuple[Dict[str, Any], float]]] = {}
                for det in candidate_dets:
                    match = self._match_detection_to_object_track(
                        det=det,
                        obj_segments=obj_segments,
                    )
                    if match is None:
                        continue
                    prompt_obj_id, match_score = match
                    matched_by_prompt_obj_id.setdefault(int(prompt_obj_id), []).append(
                        (det, float(match_score))
                    )

                sem_group = int(
                    candidate_dets[0].get(
                        "sem_group",
                        SEMANTIC_GROUP_UNCERTAIN_REGION,
                    )
                )
                prompt_only_budget = self._sam31_prompt_only_budget(label, sem_group)
                (
                    prompt_only_min_track_length,
                    prompt_only_min_obj_score,
                    prompt_only_min_area_ratio,
                    prompt_only_max_area_ratio,
                ) = self._sam31_prompt_only_filters(label, sem_group, T)

                prompt_candidates = self._build_sam31_prompt_track_candidates(
                    label=label,
                    matched_by_prompt_obj_id=matched_by_prompt_obj_id,
                    obj_segments=obj_segments,
                    obj_scores=obj_scores,
                    prompt_frame_idx=int(prompt_frame),
                    include_unmatched_prompt_tracks=prompt_only_budget > 0,
                    area=area,
                    support_dets=candidate_dets,
                    duplicate_threshold=0.78,
                    min_support_frames=2,
                    min_track_length=max(6, self.discovery_frame_stride + 2),
                    min_obj_score=0.45,
                    max_prompt_only_tracks=prompt_only_budget,
                    prompt_only_min_track_length=prompt_only_min_track_length,
                    prompt_only_min_obj_score=prompt_only_min_obj_score,
                    prompt_only_min_area_ratio=prompt_only_min_area_ratio,
                    prompt_only_max_area_ratio=prompt_only_max_area_ratio,
                )
                remaining_budget = max(
                    0,
                    int(self.max_thing_objects) - len(selected) - len(prompt_candidates),
                )
                if (
                    sem_group == SEMANTIC_GROUP_MOVABLE_THING
                    and remaining_budget > 0
                ):
                    used_cluster_ids = {
                        int(cluster_id)
                        for candidate in prompt_candidates
                        for cluster_id in candidate.get("cluster_ids", [])
                    }
                    used_seed_ids = {
                        int(seed_idx)
                        for candidate in prompt_candidates
                        for seed_idx in candidate.get("seed_global_ids", [])
                    }
                    late_unmatched_dets: List[Dict[str, Any]] = []
                    for det in candidate_dets:
                        det_frame = int(det.get("frame_idx", 0))
                        if det_frame <= prompt_frame:
                            continue
                        det_seed_idx = det.get("seed_global_track_idx", None)
                        if det_seed_idx is not None and int(det_seed_idx) in used_seed_ids:
                            continue
                        det_cluster_id = det.get("_cluster_id", None)
                        if det_cluster_id is not None and int(det_cluster_id) in used_cluster_ids:
                            continue
                        late_unmatched_dets.append(det)

                    if late_unmatched_dets:
                        is_person_label = (
                            _labels_compatible(label, "person")
                            or _labels_compatible(label, "people")
                        )
                        if is_person_label:
                            max_refresh_frames = min(
                                self.sam31_person_refresh_prompt_frames,
                                remaining_budget,
                            )
                        else:
                            max_refresh_frames = 1
                        fallback_frames = self._select_sam31_label_prompt_frames(
                            late_unmatched_dets,
                            max_prompt_frames=max_refresh_frames,
                        )
                        queried_refresh_frames: Set[int] = set()
                        for fallback_frame in fallback_frames:
                            if len(prompt_candidates) >= int(self.max_thing_objects) - len(selected):
                                break
                            fallback_frame = min(int(fallback_frame), max(int(T) - 2, 0))
                            if fallback_frame in queried_refresh_frames:
                                continue
                            queried_refresh_frames.add(fallback_frame)
                            min_late_gap = max(self.discovery_frame_stride, 3)
                            if fallback_frame < prompt_frame + min_late_gap:
                                continue

                            used_cluster_ids = {
                                int(cluster_id)
                                for candidate in prompt_candidates
                                for cluster_id in candidate.get("cluster_ids", [])
                            }
                            used_seed_ids = {
                                int(seed_idx)
                                for candidate in prompt_candidates
                                for seed_idx in candidate.get("seed_global_ids", [])
                            }
                            refresh_unmatched_dets: List[Dict[str, Any]] = []
                            for det in late_unmatched_dets:
                                det_seed_idx = det.get("seed_global_track_idx", None)
                                if det_seed_idx is not None and int(det_seed_idx) in used_seed_ids:
                                    continue
                                det_cluster_id = det.get("_cluster_id", None)
                                if det_cluster_id is not None and int(det_cluster_id) in used_cluster_ids:
                                    continue
                                refresh_unmatched_dets.append(det)
                            if not refresh_unmatched_dets:
                                break

                            session_id, fallback_segments, fallback_scores = (
                                self._query_sam31_label_tracks(
                                    frame_resource,
                                    session_id,
                                    label,
                                    fallback_frame,
                                )
                            )
                            if not fallback_segments:
                                continue

                            fallback_matched: Dict[
                                int,
                                List[Tuple[Dict[str, Any], float]],
                            ] = {}
                            for det in refresh_unmatched_dets:
                                match = self._match_detection_to_object_track(
                                    det=det,
                                    obj_segments=fallback_segments,
                                )
                                if match is None:
                                    continue
                                prompt_obj_id, match_score = match
                                fallback_matched.setdefault(
                                    int(prompt_obj_id),
                                    [],
                                ).append((det, float(match_score)))

                            if fallback_matched:
                                fallback_candidates = (
                                    self._build_sam31_prompt_track_candidates(
                                        label=label,
                                        matched_by_prompt_obj_id=fallback_matched,
                                        obj_segments=fallback_segments,
                                        obj_scores=fallback_scores,
                                        support_dets=refresh_unmatched_dets,
                                        duplicate_threshold=0.78,
                                        min_support_frames=2,
                                        min_track_length=max(
                                            6,
                                            self.discovery_frame_stride + 2,
                                        ),
                                        min_obj_score=0.50,
                                        max_prompt_only_tracks=0,
                                    )
                                )
                                if fallback_candidates:
                                    prompt_candidates = (
                                        self._select_sam31_prompt_track_candidates(
                                            prompt_candidates + fallback_candidates,
                                            duplicate_threshold=0.76,
                                            min_support_frames=2,
                                            min_track_length=max(
                                                6,
                                                self.discovery_frame_stride + 2,
                                            ),
                                            min_obj_score=0.50,
                                            max_prompt_only_tracks=prompt_only_budget,
                                            prompt_only_min_track_length=prompt_only_min_track_length,
                                            prompt_only_min_obj_score=prompt_only_min_obj_score,
                                            prompt_only_min_area_ratio=prompt_only_min_area_ratio,
                                            prompt_only_max_area_ratio=prompt_only_max_area_ratio,
                                        )
                                    )

                object_fallback_budget = 0
                # Object-prompt fallback is much slower than the text-track
                # branch and often duplicates prompt-only person tracks. Keep it
                # disabled by default; YOLOE detections still feed late text
                # refresh above, which is cheaper and easier to deduplicate.

                remaining_object_budget = max(
                    0,
                    min(
                        int(object_fallback_budget),
                        int(self.max_thing_objects) - len(selected) - len(prompt_candidates),
                    ),
                )
                if remaining_object_budget > 0:
                    used_cluster_ids = {
                        int(cluster_id)
                        for candidate in prompt_candidates
                        for cluster_id in candidate.get("cluster_ids", [])
                    }
                    used_seed_ids = {
                        int(seed_idx)
                        for candidate in prompt_candidates
                        for seed_idx in candidate.get("seed_global_ids", [])
                    }
                    object_prompt_dets: List[Dict[str, Any]] = []
                    for det in sorted(
                        candidate_dets,
                        key=lambda d: (
                            -int(bool(d.get("is_seed_track", False))),
                            -float(d.get("confidence", 0.0)),
                            -(d["mask"].astype(bool).sum() / max(area, 1)),
                            int(d.get("frame_idx", 0)),
                        ),
                    ):
                        det_seed_idx = det.get("seed_global_track_idx", None)
                        if det_seed_idx is not None and int(det_seed_idx) in used_seed_ids:
                            continue
                        det_cluster_id = det.get("_cluster_id", None)
                        if det_cluster_id is not None and int(det_cluster_id) in used_cluster_ids:
                            continue
                        object_prompt_dets.append(dict(det))
                        if len(object_prompt_dets) >= remaining_object_budget:
                            break

                    if object_prompt_dets:
                        self.video_predictor.handle_request(
                            dict(type="reset_session", session_id=session_id)
                        )
                        object_segments, object_scores, object_to_det = (
                            self._track_objects_once_sam31_multiplex_session(
                                session_id=session_id,
                                detections=object_prompt_dets,
                                H=H,
                                W=W,
                            )
                        )
                        object_candidates = self._build_sam31_object_prompt_track_candidates(
                            label=label,
                            detections=object_prompt_dets,
                            obj_segments=object_segments,
                            obj_scores=object_scores,
                            obj_to_det=object_to_det,
                        )
                        if object_candidates:
                            prompt_candidates = (
                                self._select_sam31_prompt_track_candidates(
                                    prompt_candidates + object_candidates,
                                    duplicate_threshold=0.76,
                                    min_support_frames=1,
                                    min_track_length=max(5, self.discovery_frame_stride + 1),
                                    min_obj_score=0.35,
                                    max_prompt_only_tracks=prompt_only_budget,
                                    prompt_only_min_track_length=prompt_only_min_track_length,
                                    prompt_only_min_obj_score=prompt_only_min_obj_score,
                                    prompt_only_min_area_ratio=prompt_only_min_area_ratio,
                                    prompt_only_max_area_ratio=prompt_only_max_area_ratio,
                                )
                            )

                for candidate in prompt_candidates:
                    if len(selected) >= self.max_thing_objects:
                        break
                    segment_frames = sorted(int(t) for t in candidate["segments"].keys())
                    if not segment_frames:
                        continue
                    support_start = int(segment_frames[0])
                    det_copy = dict(candidate["best_det"])
                    det_copy["frame_idx"] = support_start
                    det_copy["confidence"] = max(
                        float(det_copy["confidence"]),
                        float(candidate.get("obj_score", 0.0)),
                    )
                    if candidate["primary_seed_global_idx"] is not None:
                        det_copy["seed_global_track_idx"] = int(
                            candidate["primary_seed_global_idx"]
                        )
                    oid = next_oid
                    next_oid += 1
                    thing_oid_map[oid] = det_copy
                    selected.append(det_copy)

                    segments = candidate["segments"]
                    for t, mask in segments.items():
                        thing_segments.setdefault(int(t), {})[oid] = mask

            if (
                object_prompt_dets_by_frame
                and self.sam31_nontext_object_prompt_budget > 0
                and len(selected) < self.max_thing_objects
            ):
                object_candidates = self._cluster_sam31_track_candidates(
                    object_prompt_dets_by_frame,
                    discovery_indices,
                    area,
                )
                object_candidates = self._filter_sam31_track_candidates(object_candidates)
                object_prompt_dets: List[Dict[str, Any]] = []
                remaining_slots = max(
                    0,
                    min(
                        16,
                        int(self.max_thing_objects) - len(selected),
                        int(self.sam31_nontext_object_prompt_budget),
                    ),
                )
                for det in object_candidates:
                    if len(object_prompt_dets) >= remaining_slots:
                        break
                    if (
                        int(det.get("seed_count", 0)) <= 0
                        and int(det.get("cluster_size", 1))
                        < self.sam31_nontext_object_prompt_min_support
                    ):
                        continue
                    frame_idx = int(det.get("frame_idx", 0))
                    if self._is_duplicate_thing_detection(
                        det, frame_idx, thing_segments, thing_oid_map,
                    ):
                        continue
                    object_prompt_dets.append(dict(det))

                if object_prompt_dets:
                    self.video_predictor.handle_request(
                        dict(type="reset_session", session_id=session_id)
                    )
                    object_segments, object_scores, object_to_det = (
                        self._track_objects_once_sam31_multiplex_session(
                            session_id=session_id,
                            detections=object_prompt_dets,
                            H=H,
                            W=W,
                        )
                    )
                    object_track_candidates: List[Dict[str, Any]] = []
                    object_labels = sorted(
                        {str(det.get("label", "unknown")) for det in object_to_det.values()}
                    )
                    for object_label in object_labels:
                        label_obj_to_det = {
                            int(obj_id): det
                            for obj_id, det in object_to_det.items()
                            if _labels_compatible(str(det.get("label", "")), object_label)
                        }
                        label_segments = {
                            int(obj_id): object_segments[int(obj_id)]
                            for obj_id in label_obj_to_det
                            if int(obj_id) in object_segments
                        }
                        if not label_segments:
                            continue
                        object_track_candidates.extend(
                            self._build_sam31_object_prompt_track_candidates(
                                label=object_label,
                                detections=list(label_obj_to_det.values()),
                                obj_segments=label_segments,
                                obj_scores=object_scores,
                                obj_to_det=label_obj_to_det,
                            )
                        )

                    object_track_candidates = self._select_sam31_prompt_track_candidates(
                        object_track_candidates,
                        duplicate_threshold=0.76,
                        min_support_frames=1,
                        min_track_length=max(5, self.discovery_frame_stride + 1),
                        min_obj_score=0.35,
                        max_prompt_only_tracks=0,
                    )

                    for candidate in object_track_candidates:
                        if len(selected) >= self.max_thing_objects:
                            break
                        det_copy = dict(candidate["best_det"])
                        duplicate_existing = False
                        for oid_existing, det_existing in thing_oid_map.items():
                            if not _labels_compatible(
                                str(det_existing.get("label", "")),
                                str(det_copy.get("label", "")),
                            ):
                                continue
                            existing_segments = {
                                int(t): frame_objs[int(oid_existing)]
                                for t, frame_objs in thing_segments.items()
                                if int(oid_existing) in frame_objs
                            }
                            similarity = self._compute_sam31_prompt_track_similarity(
                                existing_segments,
                                candidate["segments"],
                            )
                            if similarity >= 0.76:
                                duplicate_existing = True
                                break
                        if duplicate_existing:
                            continue

                        segment_frames = sorted(int(t) for t in candidate["segments"].keys())
                        if not segment_frames:
                            continue
                        det_copy["frame_idx"] = int(segment_frames[0])
                        det_copy["confidence"] = max(
                            float(det_copy["confidence"]),
                            float(candidate.get("obj_score", 0.0)),
                        )
                        oid = next_oid
                        next_oid += 1
                        thing_oid_map[oid] = det_copy
                        selected.append(det_copy)
                        for t, mask in candidate["segments"].items():
                            thing_segments.setdefault(int(t), {})[oid] = mask

        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return selected, thing_segments, thing_oid_map

    def _detect_structure_prompts_sam31_multiplex(
        self,
        frame_resource: Any,
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        if (
            self.sam_backend != "sam31_multiplex"
            or not self.sam31_structure_prompt_labels
            or self.sam31_structure_prompt_frame_count <= 0
        ):
            return {}

        prompt_frames = self._select_sam31_structure_prompt_frames(
            discovery_indices, T,
        )
        if not prompt_frames:
            return {}

        session_id = self._start_sam31_session(frame_resource)
        detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        try:
            for frame_idx in prompt_frames:
                for label in self.sam31_structure_prompt_labels:
                    canonical_label = canonicalize_label(label)
                    if label_to_group(canonical_label) != SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                        continue
                    label_dets: List[Dict[str, Any]] = []
                    for prompt_text in _structure_prompt_texts(canonical_label):
                        session_id, dets = self._query_sam31_prompt_frame_detections(
                            frame_resource,
                            session_id,
                            prompt_text,
                            int(frame_idx),
                            H,
                            W,
                            area,
                        )
                        if dets:
                            label_dets.extend(dets)
                        # Avoid multiplying SAM calls once a reliable prompt
                        # already returned usable structure masks.
                        if dets and canonical_label not in _GROUND_STRUCTURE_LABELS:
                            break
                    if label_dets:
                        detections_by_frame.setdefault(int(frame_idx), []).extend(label_dets)
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return {
            int(frame_idx): self._nms(dets)
            for frame_idx, dets in detections_by_frame.items()
            if dets
        }

    def _select_multiframe_structure_sam31_multiplex(
        self,
        frame_resource: Any,
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

        if self.sam31_max_structure_objects <= 0:
            return selected, structure_segments, structure_oid_map

        structure_candidates = self._cluster_sam31_track_candidates(
            structure_dets_by_frame,
            discovery_indices,
            area,
        )
        structure_candidates = self._filter_sam31_track_candidates(structure_candidates)
        selected_structure_dets = self._take_sam31_track_candidates(
            structure_candidates,
            budget=self.sam31_max_structure_objects,
            allowed_groups={SEMANTIC_GROUP_STRUCTURE_ANCHOR},
            per_label_limit=1,
        )

        if not selected_structure_dets:
            return selected, structure_segments, structure_oid_map

        session_id = self._start_sam31_session(frame_resource)
        try:
            obj_segments, obj_scores, obj_to_det = self._track_objects_once_sam31_multiplex_session(
                session_id=session_id,
                detections=selected_structure_dets,
                H=H,
                W=W,
            )
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        min_track_length = min(max(4, self.discovery_frame_stride + 1), max(int(T), 1))
        for obj_id in sorted(obj_segments.keys()):
            frame_to_mask = {
                int(t): mask.astype(np.uint8)
                for t, mask in obj_segments.get(int(obj_id), {}).items()
                if mask is not None
            }
            if not frame_to_mask:
                continue
            det_src = obj_to_det.get(int(obj_id))
            if det_src is None:
                continue
            if (
                len(frame_to_mask) < min_track_length
                and det_src.get("seed_global_track_idx", None) is None
            ):
                continue

            mask_areas = [float(mask.astype(bool).sum()) for mask in frame_to_mask.values()]
            mean_area_ratio = float(sum(mask_areas) / max(len(mask_areas), 1)) / max(float(area), 1.0)
            if mean_area_ratio < self.min_mask_area_ratio or mean_area_ratio > self.max_mask_area_ratio:
                continue

            det_copy = dict(det_src)
            segment_frames = sorted(frame_to_mask.keys())
            det_copy["frame_idx"] = int(segment_frames[0])
            det_copy["confidence"] = max(
                float(det_copy.get("confidence", 0.0)),
                float(obj_scores.get(int(obj_id), 0.0)),
            )
            oid = next_oid
            next_oid += 1
            structure_oid_map[oid] = det_copy
            selected.append(det_copy)
            for t, mask in frame_to_mask.items():
                structure_segments.setdefault(int(t), {})[oid] = mask

        tracked_labels = {str(det.get("label", "")) for det in selected}
        if len(selected) < self.sam31_max_structure_objects:
            label_to_dets: Dict[str, List[Dict[str, Any]]] = {}
            candidate_frame_indices = sorted(
                {int(k) for k in discovery_indices}.union(
                    int(k) for k in structure_dets_by_frame.keys()
                )
            )
            for frame_idx in candidate_frame_indices:
                candidates = list(structure_dets_by_frame.get(int(frame_idx), []))
                if not candidates:
                    continue
                candidates.sort(
                    key=lambda d: (
                        -int(bool(d.get("is_seed_track", False))),
                        -(d["mask"].astype(bool).sum() / max(area, 1)),
                        -float(d.get("confidence", 0.0)),
                    ),
                )
                for det in candidates:
                    label_to_dets.setdefault(str(det["label"]), []).append(dict(det))

            fallback_labels = [
                label for label in sorted(
                    label_to_dets.keys(),
                    key=lambda lbl: min(int(det.get("frame_idx", 0)) for det in label_to_dets[lbl]),
                )
                if label not in tracked_labels
            ]
            if fallback_labels:
                session_id = self._start_sam31_session(frame_resource)
                try:
                    for label in fallback_labels:
                        if len(selected) >= self.sam31_max_structure_objects:
                            break

                        raw_dets = sorted(
                            label_to_dets.get(label, []),
                            key=lambda d: (
                                -int(bool(d.get("is_seed_track", False))),
                                int(d.get("frame_idx", 0)),
                                -(d["mask"].astype(bool).sum() / max(area, 1)),
                                -float(d.get("confidence", 0.0)),
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

                        candidate_dets = self._assign_sam31_detection_cluster_ids(
                            candidate_dets,
                            area=area,
                        )
                        prompt_frame = min(
                            int(det.get("frame_idx", 0)) for det in candidate_dets
                        )
                        session_id, text_segments, text_scores = self._query_sam31_label_tracks(
                            frame_resource,
                            session_id,
                            label,
                            int(prompt_frame),
                            log_prefix="structure fallback ",
                        )
                        if not text_segments:
                            continue

                        matched_by_prompt_obj_id: Dict[int, List[Tuple[Dict[str, Any], float]]] = {}
                        for det in candidate_dets:
                            match = self._match_detection_to_object_track(
                                det=det,
                                obj_segments=text_segments,
                            )
                            if match is None:
                                continue
                            prompt_obj_id, match_score = match
                            matched_by_prompt_obj_id.setdefault(int(prompt_obj_id), []).append(
                                (det, float(match_score))
                            )

                        text_candidates = self._build_sam31_prompt_track_candidates(
                            label=label,
                            matched_by_prompt_obj_id=matched_by_prompt_obj_id,
                            obj_segments=text_segments,
                            obj_scores=text_scores,
                            prompt_frame_idx=int(prompt_frame),
                            include_unmatched_prompt_tracks=True,
                            area=area,
                            duplicate_threshold=0.74,
                            min_support_frames=1,
                            min_track_length=max(4, self.discovery_frame_stride + 1),
                            min_obj_score=0.30,
                            max_prompt_only_tracks=1,
                            prompt_only_min_track_length=max(4, self.discovery_frame_stride + 1),
                            prompt_only_min_obj_score=0.20,
                            prompt_only_min_area_ratio=max(self.min_mask_area_ratio, 0.006),
                            prompt_only_max_area_ratio=min(self.max_mask_area_ratio, 0.90),
                        )
                        if not text_candidates:
                            continue

                        for candidate in text_candidates:
                            if len(selected) >= self.sam31_max_structure_objects:
                                break
                            duplicate_existing = False
                            for oid_existing, det_existing in structure_oid_map.items():
                                if not _labels_compatible(
                                    str(det_existing.get("label", "")),
                                    str(candidate["label"]),
                                ):
                                    continue
                                existing_segments = {
                                    int(t): frame_objs[int(oid_existing)]
                                    for t, frame_objs in structure_segments.items()
                                    if int(oid_existing) in frame_objs
                                }
                                similarity = self._compute_sam31_prompt_track_similarity(
                                    existing_segments,
                                    candidate["segments"],
                                )
                                if similarity >= 0.74:
                                    duplicate_existing = True
                                    break
                            if duplicate_existing:
                                continue

                            segment_frames = sorted(int(t) for t in candidate["segments"].keys())
                            if not segment_frames:
                                continue
                            det_copy = dict(candidate["best_det"])
                            det_copy["frame_idx"] = int(segment_frames[0])
                            det_copy["confidence"] = max(
                                float(det_copy.get("confidence", 0.0)),
                                float(candidate.get("obj_score", 0.0)),
                            )
                            if candidate["primary_seed_global_idx"] is not None:
                                det_copy["seed_global_track_idx"] = int(
                                    candidate["primary_seed_global_idx"]
                                )
                            oid = next_oid
                            next_oid += 1
                            structure_oid_map[oid] = det_copy
                            selected.append(det_copy)
                            tracked_labels.add(str(det_copy.get("label", label)))
                            for t, mask in candidate["segments"].items():
                                structure_segments.setdefault(int(t), {})[oid] = mask
                            break
                finally:
                    self.video_predictor.handle_request(
                        dict(type="close_session", session_id=session_id)
                    )

        return selected, structure_segments, structure_oid_map

    def _cluster_sam31_track_candidates(
        self,
        dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        area: int,
    ) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for frame_idx in discovery_indices:
            candidates = list(dets_by_frame.get(frame_idx, []))
            if not candidates:
                continue
            candidates.sort(
                key=lambda d: (
                    int(d.get("frame_idx", frame_idx)),
                    -(d["mask"].astype(bool).sum() / max(area, 1)),
                    -float(d["confidence"]),
                ),
            )
            for det in candidates:
                det_frame = int(det.get("frame_idx", frame_idx))
                best_cluster = None
                best_score = -1.0
                det_group = int(det["sem_group"])
                det_mask = det["mask"].astype(bool)
                det_box = np.asarray(det["box"], dtype=np.float32)
                det_area = max(float(det_mask.sum()), 1.0)

                for cluster in clusters:
                    ref_det = cluster["detections"][-1]
                    if int(ref_det["sem_group"]) != det_group:
                        continue
                    if not _labels_compatible(str(ref_det["label"]), str(det["label"])):
                        continue

                    ref_mask = ref_det["mask"].astype(bool)
                    ref_box = np.asarray(ref_det["box"], dtype=np.float32)
                    ref_area = max(float(ref_mask.sum()), 1.0)
                    mask_iou = _mask_iou(det_mask, ref_mask)
                    box_iou = _box_iou_xyxy(det_box, ref_box)
                    area_sim = min(det_area, ref_area) / max(det_area, ref_area)
                    frame_gap = abs(det_frame - int(ref_det.get("frame_idx", det_frame)))
                    gap_bonus = 0.10 if frame_gap <= max(self.discovery_frame_stride, 1) else 0.0

                    if det_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                        score = 0.45 * mask_iou + 0.25 * box_iou + 0.20 * area_sim + gap_bonus
                        threshold = 0.18
                    elif det_group == SEMANTIC_GROUP_STATIC_THING:
                        score = 0.25 * mask_iou + 0.45 * box_iou + 0.20 * area_sim + gap_bonus
                        threshold = 0.18
                    else:
                        score = 0.15 * mask_iou + 0.55 * box_iou + 0.20 * area_sim + gap_bonus
                        threshold = 0.16

                    if frame_gap == 0:
                        if det_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                            score = 0.60 * mask_iou + 0.40 * box_iou
                            threshold = 0.30
                        elif det_group == SEMANTIC_GROUP_STATIC_THING:
                            score = 0.45 * mask_iou + 0.55 * box_iou
                            threshold = 0.35
                        else:
                            score = 0.40 * mask_iou + 0.60 * box_iou
                            threshold = 0.35

                    if score >= threshold and score > best_score:
                        best_score = score
                        best_cluster = cluster

                if best_cluster is None:
                    clusters.append({
                        "label": str(det["label"]),
                        "sem_group": det_group,
                        "detections": [dict(det)],
                    })
                else:
                    best_cluster["detections"].append(dict(det))

        candidates: List[Dict[str, Any]] = []
        for cluster in clusters:
            dets = sorted(
                cluster["detections"],
                key=lambda d: (
                    -int(bool(d.get("is_seed_track", False))),
                    int(d.get("frame_idx", 0)),
                    -float(d["confidence"]),
                    -(d["mask"].astype(bool).sum() / max(area, 1)),
                ),
            )
            seed_count = sum(1 for d in dets if bool(d.get("is_seed_track", False)))
            prompt_det = dict(dets[0])
            prompt_det["cluster_size"] = len(dets)
            prompt_det["seed_count"] = int(seed_count)
            prompt_det["support_frames"] = sorted({int(d.get("frame_idx", 0)) for d in dets})
            prompt_det["cluster_confidence"] = max(float(d["confidence"]) for d in dets)
            candidates.append(prompt_det)

        candidates.sort(
            key=lambda d: (
                -int(d.get("seed_count", 0)),
                -int(d.get("cluster_size", 1)),
                -float(d.get("cluster_confidence", d["confidence"])),
                -(d["mask"].astype(bool).sum() / max(area, 1)),
                int(d.get("frame_idx", 0)),
            )
        )
        return candidates

    def _filter_sam31_track_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for det in candidates:
            seed_count = int(det.get("seed_count", 0))
            cluster_size = int(det.get("cluster_size", 1))
            confidence = float(det.get("cluster_confidence", det.get("confidence", 0.0)))
            area_ratio = float(det.get("area_ratio", 0.0))
            sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
            label = canonicalize_label(str(det.get("label", "")))

            if seed_count > 0 and sem_group != SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                filtered.append(det)
                continue

            if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                if label in _THIN_STRUCTURE_LABELS:
                    keep = (
                        (cluster_size >= 2 and confidence >= 0.42 and area_ratio <= 0.18)
                        or (confidence >= 0.70 and area_ratio <= 0.20)
                        or (seed_count > 0 and area_ratio <= 0.18)
                    )
                elif label in _AMBIGUOUS_STRUCTURE_LABELS:
                    keep = (
                        (cluster_size >= 2 and area_ratio <= 0.30)
                        or (confidence >= 0.68 and 0.004 <= area_ratio <= 0.30)
                        or (seed_count > 0 and area_ratio <= 0.32)
                    )
                elif label in _GROUND_STRUCTURE_LABELS:
                    keep = (
                        cluster_size >= 2
                        or confidence >= 0.55
                        or 0.025 <= area_ratio <= 0.48
                        or seed_count > 0
                    )
                elif label in _OVERHEAD_STRUCTURE_LABELS:
                    keep = (
                        cluster_size >= 2
                        or confidence >= 0.56
                        or 0.020 <= area_ratio <= 0.45
                        or seed_count > 0
                    )
                elif label in _VERTICAL_STRUCTURE_LABELS:
                    keep = (
                        cluster_size >= 2
                        or confidence >= 0.58
                        or 0.035 <= area_ratio <= 0.52
                        or seed_count > 0
                    )
                else:
                    keep = cluster_size >= 2 or confidence >= 0.62
            elif sem_group == SEMANTIC_GROUP_STATIC_THING:
                keep = cluster_size >= 2 or confidence >= 0.58 or area_ratio >= 0.03
            elif sem_group == SEMANTIC_GROUP_MOVABLE_THING:
                keep = cluster_size >= 2 or confidence >= 0.68
            else:
                keep = cluster_size >= 2 or confidence >= 0.72

            if keep:
                filtered.append(det)
        return filtered

    def _take_sam31_track_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        budget: int,
        allowed_groups: Optional[Set[int]] = None,
        per_label_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if budget <= 0:
            return []

        selected: List[Dict[str, Any]] = []
        label_counts: Dict[str, int] = {}
        for det in candidates:
            sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
            if allowed_groups is not None and sem_group not in allowed_groups:
                continue
            label = str(det.get("label", "unknown"))
            if per_label_limit is not None and label_counts.get(label, 0) >= per_label_limit:
                continue
            selected.append(dict(det))
            label_counts[label] = label_counts.get(label, 0) + 1
            if len(selected) >= budget:
                break
        return selected

    def _build_sam31_object_prompt(
        self,
        det: Dict[str, Any],
        H: int,
        W: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = det["mask"].astype(bool)
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx = float(xs.mean())
            cy = float(ys.mean())
        else:
            x1, y1, x2, y2 = [float(v) for v in np.asarray(det["box"], dtype=np.float32)]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

        x1, y1, x2, y2 = [float(v) for v in np.asarray(det["box"], dtype=np.float32)]
        x1 = np.clip(x1, 0.0, max(W - 1, 0))
        x2 = np.clip(x2, 0.0, max(W - 1, 0))
        y1 = np.clip(y1, 0.0, max(H - 1, 0))
        y2 = np.clip(y2, 0.0, max(H - 1, 0))
        cx = np.clip(cx, 0.0, max(W - 1, 0))
        cy = np.clip(cy, 0.0, max(H - 1, 0))

        points = np.array(
            [
                [x1 / max(W, 1), y1 / max(H, 1)],
                [x2 / max(W, 1), y2 / max(H, 1)],
                [cx / max(W, 1), cy / max(H, 1)],
            ],
            dtype=np.float32,
        )
        labels = np.array([2, 3, 1], dtype=np.int32)
        return points, labels

    @torch.inference_mode()
    def _track_objects_once_sam31_multiplex_session(
        self,
        session_id: str,
        detections: List[Dict[str, Any]],
        H: int,
        W: int,
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, float], Dict[int, Dict[str, Any]]]:
        if not detections:
            return {}, {}, {}

        obj_to_det: Dict[int, Dict[str, Any]] = {}
        obj_segments: Dict[int, Dict[int, np.ndarray]] = {}
        obj_scores: Dict[int, float] = {}
        start_frame_idx = min(int(det.get("frame_idx", 0)) for det in detections)
        last_prompt_frame_idx = max(int(det.get("frame_idx", 0)) for det in detections)

        dets_by_prompt_frame: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        for obj_id, det in enumerate(detections, start=1):
            obj_id = int(obj_id)
            prompt_frame = int(det.get("frame_idx", 0))
            obj_to_det[obj_id] = det
            obj_segments[obj_id] = {}
            obj_scores[obj_id] = max(obj_scores.get(obj_id, 0.0), float(det.get("confidence", 0.0)))
            dets_by_prompt_frame.setdefault(prompt_frame, []).append((obj_id, det))

        for prompt_frame in sorted(dets_by_prompt_frame.keys()):
            frame_entries = dets_by_prompt_frame[prompt_frame]
            for obj_id, det in frame_entries:
                points, labels = self._build_sam31_object_prompt(det, H=H, W=W)
                response = self.video_predictor.handle_request(
                    dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=int(prompt_frame),
                        points=points,
                        point_labels=labels,
                        obj_id=int(obj_id),
                    )
                )
                outputs = response["outputs"]
                out_obj_ids = outputs.get("out_obj_ids", [])
                out_masks = outputs.get("out_binary_masks", [])
                out_probs = outputs.get("out_probs", [])
                for idx, out_obj_id in enumerate(out_obj_ids):
                    out_obj_id = int(out_obj_id)
                    if out_obj_id not in obj_to_det:
                        continue
                    if idx < len(out_masks):
                        obj_segments.setdefault(out_obj_id, {})[int(prompt_frame)] = (
                            out_masks[idx].astype(np.uint8)
                        )
                    if idx < len(out_probs):
                        obj_scores[out_obj_id] = max(obj_scores.get(out_obj_id, 0.0), float(out_probs[idx]))

        def _consume(direction: str, start_frame: int) -> None:
            for prop_response in self.video_predictor.handle_stream_request(
                dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=direction,
                    start_frame_index=int(start_frame),
                )
            ):
                out_frame_idx = int(prop_response["frame_index"])
                outputs = prop_response["outputs"]
                out_obj_ids = outputs.get("out_obj_ids", [])
                out_masks = outputs.get("out_binary_masks", [])
                out_probs = outputs.get("out_probs", [])
                for idx, out_obj_id in enumerate(out_obj_ids):
                    out_obj_id = int(out_obj_id)
                    if out_obj_id not in obj_to_det:
                        continue
                    if idx < len(out_masks):
                        obj_segments.setdefault(out_obj_id, {})[out_frame_idx] = (
                            out_masks[idx].astype(np.uint8)
                        )
                    if idx < len(out_probs):
                        obj_scores[out_obj_id] = max(obj_scores.get(out_obj_id, 0.0), float(out_probs[idx]))

        _consume("forward", int(start_frame_idx))
        if self.sam31_enable_backward and int(last_prompt_frame_idx) > 0:
            try:
                _consume("backward", int(last_prompt_frame_idx))
            except KeyError as exc:
                print(
                    f"[sam31_multiplex] warning: object reverse propagation failed "
                    f"at frame_idx={last_prompt_frame_idx} with {exc!r}; "
                    "keeping forward-only result"
                )

        obj_segments = {
            obj_id: frame_to_mask
            for obj_id, frame_to_mask in obj_segments.items()
            if frame_to_mask
        }
        return obj_segments, obj_scores, obj_to_det

    def _track_all_objects_sam31_multiplex(
        self,
        frame_resource: Any,
        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        structure_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[
        List[Dict[str, Any]],
        Dict[int, Dict[int, np.ndarray]],
        Dict[int, Dict],
        List[Dict[str, Any]],
        Dict[int, Dict[int, np.ndarray]],
        Dict[int, Dict],
    ]:
        selected_thing_dets: List[Dict[str, Any]] = []
        thing_segments: Dict[int, Dict[int, np.ndarray]] = {}
        thing_oid_map: Dict[int, Dict] = {}
        selected_structure_dets: List[Dict[str, Any]] = []
        structure_segments: Dict[int, Dict[int, np.ndarray]] = {}
        structure_oid_map: Dict[int, Dict] = {}

        thing_candidates = self._cluster_sam31_track_candidates(
            thing_dets_by_frame, discovery_indices, area,
        )
        structure_candidates = self._cluster_sam31_track_candidates(
            structure_dets_by_frame, discovery_indices, area,
        )
        thing_candidates = self._filter_sam31_track_candidates(thing_candidates)
        structure_candidates = self._filter_sam31_track_candidates(structure_candidates)

        thing_budget = max(int(self.max_thing_objects), 0)
        selected_movable = self._take_sam31_track_candidates(
            thing_candidates,
            budget=min(thing_budget, self.sam31_max_movable_objects),
            allowed_groups={SEMANTIC_GROUP_MOVABLE_THING},
        )
        remaining_thing_budget = max(0, thing_budget - len(selected_movable))
        selected_static = self._take_sam31_track_candidates(
            thing_candidates,
            budget=min(remaining_thing_budget, self.sam31_max_static_objects),
            allowed_groups={SEMANTIC_GROUP_STATIC_THING},
            per_label_limit=1,
        )
        selected_structure_dets = self._take_sam31_track_candidates(
            structure_candidates,
            budget=self.sam31_max_structure_objects,
            allowed_groups={SEMANTIC_GROUP_STRUCTURE_ANCHOR},
            per_label_limit=1,
        )
        selected_thing_dets = selected_movable + selected_static

        all_candidates = selected_thing_dets + selected_structure_dets
        if not all_candidates:
            return (
                selected_thing_dets,
                thing_segments,
                thing_oid_map,
                selected_structure_dets,
                structure_segments,
                structure_oid_map,
            )

        session_id = self._start_sam31_session(frame_resource)
        try:
            obj_segments, obj_scores, obj_to_det = self._track_objects_once_sam31_multiplex_session(
                session_id=session_id,
                detections=all_candidates,
                H=H,
                W=W,
            )
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        next_thing_oid = 1
        next_structure_oid = 1
        selected_thing_final: List[Dict[str, Any]] = []
        selected_structure_final: List[Dict[str, Any]] = []
        thing_set = {id(det) for det in selected_thing_dets}

        for obj_id in sorted(obj_segments.keys()):
            det = dict(obj_to_det[obj_id])
            det["confidence"] = max(
                float(det["confidence"]),
                float(obj_scores.get(obj_id, 0.0)),
            )
            if id(obj_to_det[obj_id]) in thing_set:
                oid = next_thing_oid
                next_thing_oid += 1
                thing_oid_map[oid] = det
                selected_thing_final.append(det)
                for t, mask in obj_segments[obj_id].items():
                    thing_segments.setdefault(int(t), {})[oid] = mask
            else:
                oid = next_structure_oid
                next_structure_oid += 1
                structure_oid_map[oid] = det
                selected_structure_final.append(det)
                for t, mask in obj_segments[obj_id].items():
                    structure_segments.setdefault(int(t), {})[oid] = mask

        return (
            selected_thing_final,
            thing_segments,
            thing_oid_map,
            selected_structure_final,
            structure_segments,
            structure_oid_map,
        )

    def _select_multiframe_things_sam3(
        self,
        frame_resource: Any,
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
        request = dict(type="start_session", resource_path=frame_resource)
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
        frame_resource: Any,
        T: int,
        H: int,
        W: int,
        discovery_indices: List[int],
        seed_detections_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        detector_image_paths: Optional[List[str]] = None,
        sam31_run_index: int = 0,
    ) -> MaskletOutput:
        area = H * W

        # ======== Step 1: detect on discovery frames ========
        sam31_model_offloaded_for_detection = False
        if (
            self.sam_backend == "sam31_multiplex"
            and self.sam31_offload_sam_during_detection
            and str(self.device).startswith("cuda")
        ):
            # YOLOE and SAM3.1 are not needed simultaneously. Keeping SAM on
            # CPU during discovery avoids detector+tracker VRAM stacking in
            # the full A/B/C/D/E pipeline.
            self._move_sam31_predictor_model("cpu")
            sam31_model_offloaded_for_detection = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.detector.to_device()
        detections_by_frame = self._detect_and_filter_batch(
            frame_resource,
            discovery_indices,
            H,
            W,
            detector_image_paths=detector_image_paths,
        )

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
        sparse_thing_support_dets: List[Dict[str, Any]] = []
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
                    det_label = str(det.get("label", ""))
                    is_person_det = (
                        _labels_compatible(det_label, "person")
                        or _labels_compatible(det_label, "people")
                    )
                    # Full open-vocabulary sparse support produced noisy indoor
                    # false positives (e.g. train/bicycle on walls). Keep this
                    # path focused on the failure mode we need it for: person
                    # detections that SAM3.1 text tracking drops for a few
                    # frames.
                    keep_sparse_support = (
                        is_person_det
                        and float(det.get("confidence", 0.0)) >= 0.35
                    )
                    if (
                        self.sam_backend == "sam31_multiplex"
                        and self.sam31_nontext_sparse_support
                        and keep_sparse_support
                    ):
                        det_copy = dict(det)
                        # Keep open-vocabulary person aliases useful for
                        # discovery, but canonicalize the emitted track label.
                        det_copy["label"] = "person"
                        det_copy["detector_source"] = str(
                            det_copy.get("detector_source", "detector_sparse_support")
                        )
                        sparse_thing_support_dets.append(det_copy)

        # ======== Step 2: free detector GPU → give memory to SAM2 ========
        self.detector.release_gpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if sam31_model_offloaded_for_detection:
            self._move_sam31_predictor_model(self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        structure_prompt_active = (
            self.sam_backend == "sam31_multiplex"
            and self._should_run_sam31_structure_prompts(sam31_run_index)
        )
        if structure_prompt_active:
            prompt_structure_dets = self._detect_structure_prompts_sam31_multiplex(
                frame_resource,
                discovery_indices,
                T,
                H,
                W,
                area,
            )
            for frame_idx, dets in prompt_structure_dets.items():
                merged = list(structure_dets_by_frame.get(int(frame_idx), []))
                merged.extend(dets)
                structure_dets_by_frame[int(frame_idx)] = self._nms(merged)

        # ======== Step 3: track thing / structure objects ========
        thing_segments, thing_oid_map = {}, {}
        structure_segments, structure_oid_map = {}, {}
        selected_structure_dets: List[Dict[str, Any]] = []
        if self.sam_backend == "sam31_multiplex":
            selected_thing_dets, thing_segments, thing_oid_map = (
                self._select_multiframe_things_sam31_multiplex(
                    frame_resource,
                    thing_dets_by_frame,
                    discovery_indices,
                    T,
                    H,
                    W,
                    area,
                )
            )
            if sparse_thing_support_dets:
                self._append_sparse_detector_thing_tracks(
                    sparse_thing_support_dets,
                    discovery_indices,
                    thing_segments,
                    thing_oid_map,
                    area,
                )
            (
                selected_structure_dets,
                structure_segments,
                structure_oid_map,
            ) = self._select_multiframe_structure_sam31_multiplex(
                frame_resource,
                structure_dets_by_frame,
                discovery_indices,
                T,
                H,
                W,
                area,
            )
        else:
            selected_thing_dets, thing_segments, thing_oid_map = \
                self._select_multiframe_things(
                    frame_resource, thing_dets_by_frame, discovery_indices, T, H, W, area,
                )
            if not thing_segments and selected_thing_dets:
                thing_segments, thing_oid_map = self._propagate_things(
                    frame_resource, selected_thing_dets, T, H, W,
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
        if sparse_thing_support_dets:
            # Non-person detector support has already been folded into sparse
            # thing tracks above. Do not also add it as stuff, or the same
            # physical object will be rendered twice with different IDs.
            pass

        # ======== Step 4: stuff — sparse support masks only ========
        stuff_entries = self._build_stuff_entries(
            stuff_dets, frame_resource, T, H, W,
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
            "sam31_run_index": int(sam31_run_index),
            "sam31_structure_prompt_active": bool(structure_prompt_active),
            "sam31_structure_prompt_chunk_stride": int(
                self.sam31_structure_prompt_chunk_stride
            ),
        })
        return output

    # -- Detection & filtering --------------------------------------------

    def _detect_and_filter(
        self,
        frame_resource: Any,
        frame_idx: int,
        H: int,
        W: int,
        detector_image_paths: Optional[List[str]] = None,
    ) -> List[Dict]:
        if detector_image_paths is not None:
            img_path = detector_image_paths[frame_idx]
        else:
            img_path = os.path.join(frame_resource, f"{frame_idx:05d}.jpg")
        raw = self.detector.detect(
            img_path, self.thing_prompts, self.stuff_prompts,
            self.box_threshold, self.text_threshold,
        )
        filtered = self._filter_raw_detections(raw, H, W)
        for det in filtered:
            det["frame_idx"] = int(frame_idx)
        return filtered

    def _detect_and_filter_batch(
        self,
        frame_resource: Any,
        frame_indices: List[int],
        H: int,
        W: int,
        detector_image_paths: Optional[List[str]] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        valid_indices = [int(idx) for idx in frame_indices if 0 <= int(idx)]
        if not valid_indices:
            return {}
        image_paths = [
            detector_image_paths[idx]
            if detector_image_paths is not None
            else os.path.join(frame_resource, f"{idx:05d}.jpg")
            for idx in valid_indices
        ]
        raw_batches = self.detector.detect_batch(
            image_paths,
            self.thing_prompts,
            self.stuff_prompts,
            self.box_threshold,
            self.text_threshold,
        )
        detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for frame_idx, raw in zip(valid_indices, raw_batches):
            filtered = self._filter_raw_detections(raw, H, W)
            for det in filtered:
                det["frame_idx"] = int(frame_idx)
            detections_by_frame[int(frame_idx)] = filtered
        return detections_by_frame

    def _filter_raw_detections(self, raw: List[Dict[str, Any]], H: int, W: int) -> List[Dict]:
        area = H * W
        filtered = []
        for det in raw:
            det = dict(det)
            mask = det["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            src_h, src_w = int(mask.shape[0]), int(mask.shape[1])
            mask_bool = mask.astype(bool) if mask.dtype != bool else mask
            if mask_bool.shape[0] != H or mask_bool.shape[1] != W:
                mask_f = mask.astype(np.float32)
                mask_f = cv2.resize(mask_f, (W, H), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_f > 0.5

            det_box = det.get("box", None)
            if det_box is None:
                det_box_arr = self._mask_to_box_xyxy(mask_bool)
            else:
                if isinstance(det_box, torch.Tensor):
                    det_box_arr = det_box.detach().cpu().numpy()
                else:
                    det_box_arr = np.asarray(det_box, dtype=np.float32)
                det_box_arr = det_box_arr.reshape(-1)[:4].astype(np.float32, copy=True)
                if src_h > 0 and src_w > 0 and (src_h != H or src_w != W):
                    det_box_arr[[0, 2]] *= float(W) / float(src_w)
                    det_box_arr[[1, 3]] *= float(H) / float(src_h)
                det_box_arr[[0, 2]] = np.clip(det_box_arr[[0, 2]], 0.0, float(max(W - 1, 0)))
                det_box_arr[[1, 3]] = np.clip(det_box_arr[[1, 3]], 0.0, float(max(H - 1, 0)))
                if det_box_arr[2] < det_box_arr[0]:
                    det_box_arr[0], det_box_arr[2] = det_box_arr[2], det_box_arr[0]
                if det_box_arr[3] < det_box_arr[1]:
                    det_box_arr[1], det_box_arr[3] = det_box_arr[3], det_box_arr[1]

            raw_label = str(det.get("label", "unknown"))
            label = canonicalize_label(raw_label)
            sem_group = label_to_group(label)
            confidence = float(det.get("confidence", 0.0))
            if sem_group != SEMANTIC_GROUP_MOVABLE_THING and confidence < self.box_threshold:
                continue
            ratio = mask_bool.sum() / area
            min_area_ratio = self.min_mask_area_ratio
            if sem_group == SEMANTIC_GROUP_MOVABLE_THING:
                min_area_ratio = min(min_area_ratio, 0.0004)
            elif sem_group == SEMANTIC_GROUP_STATIC_THING:
                min_area_ratio = min(min_area_ratio, 0.0007)
            if ratio < min_area_ratio or ratio > self.max_mask_area_ratio:
                continue

            if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                keep, _ = passes_structure_mask_quality(
                    label,
                    mask_bool,
                    det_box_arr,
                    confidence,
                    float(ratio),
                    H,
                    W,
                )
                if not keep:
                    continue

            det["raw_label"] = raw_label
            det["label"] = label
            det["mask"] = mask_bool.astype(np.uint8)
            det["box"] = det_box_arr.astype(np.float32)
            det["sem_group"] = sem_group
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

    def _mask_to_box_xyxy(self, mask: np.ndarray) -> np.ndarray:
        mask_bool = mask.astype(bool)
        ys, xs = np.where(mask_bool)
        if len(xs) == 0 or len(ys) == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [xs.min(), ys.min(), xs.max(), ys.max()],
            dtype=np.float32,
        )

    def _compute_sam31_prompt_track_similarity(
        self,
        frame_to_mask_a: Dict[int, np.ndarray],
        frame_to_mask_b: Dict[int, np.ndarray],
    ) -> float:
        shared_frames = sorted(set(frame_to_mask_a).intersection(frame_to_mask_b))
        if not shared_frames:
            return -1.0

        frame_ious: List[float] = []
        frame_covers: List[float] = []
        frame_box_ious: List[float] = []

        for frame_idx in shared_frames:
            mask_a = frame_to_mask_a[frame_idx].astype(bool)
            mask_b = frame_to_mask_b[frame_idx].astype(bool)
            area_a = float(mask_a.sum())
            area_b = float(mask_b.sum())
            if area_a <= 0.0 or area_b <= 0.0:
                continue
            inter = float((mask_a & mask_b).sum())
            union = area_a + area_b - inter
            frame_ious.append(inter / union if union > 0.0 else 0.0)
            frame_covers.append(max(inter / area_a, inter / area_b))
            frame_box_ious.append(
                _box_iou_xyxy(
                    self._mask_to_box_xyxy(mask_a),
                    self._mask_to_box_xyxy(mask_b),
                )
            )

        if not frame_ious:
            return -1.0

        mean_iou = sum(frame_ious) / len(frame_ious)
        mean_cover = sum(frame_covers) / len(frame_covers)
        mean_box_iou = sum(frame_box_ious) / len(frame_box_ious)
        return 0.35 * mean_iou + 0.45 * mean_cover + 0.20 * mean_box_iou

    def _select_sam31_label_prompt_frames(
        self,
        detections: List[Dict[str, Any]],
        *,
        max_prompt_frames: int,
    ) -> List[int]:
        if not detections or max_prompt_frames <= 0:
            return []

        frame_priorities: Dict[int, float] = {}
        for det in detections:
            frame_idx = int(det.get("frame_idx", 0))
            priority = float(det.get("confidence", 0.0))
            priority += min(float(det.get("area_ratio", 0.0)) * 6.0, 1.0)
            if bool(det.get("is_seed_track", False)):
                priority += 2.0
            if det.get("seed_global_track_idx", None) is not None:
                priority += 1.5
            prev = frame_priorities.get(frame_idx, None)
            if prev is None or priority > prev:
                frame_priorities[frame_idx] = priority

        if not frame_priorities:
            return []

        chosen = [min(frame_priorities.keys())]
        remaining = {frame_idx for frame_idx in frame_priorities if frame_idx not in chosen}
        min_gap = max(self.discovery_frame_stride // 2, 1)

        while remaining and len(chosen) < max_prompt_frames:
            best_frame = None
            best_score = -1e9
            for frame_idx in sorted(remaining):
                gap = min(abs(frame_idx - ref_frame) for ref_frame in chosen)
                score = frame_priorities[frame_idx] + 0.20 * float(gap)
                if frame_idx > max(chosen):
                    score += 0.35
                if gap < min_gap:
                    score -= 0.60
                if score > best_score:
                    best_score = score
                    best_frame = frame_idx
            if best_frame is None:
                break
            chosen.append(int(best_frame))
            remaining.remove(best_frame)

        return sorted({int(frame_idx) for frame_idx in chosen})

    def _assign_sam31_detection_cluster_ids(
        self,
        detections: List[Dict[str, Any]],
        area: int,
    ) -> List[Dict[str, Any]]:
        if not detections:
            return []

        clusters: List[List[Dict[str, Any]]] = []
        dets_sorted = sorted(
            detections,
            key=lambda d: (
                -int(bool(d.get("is_seed_track", False))),
                int(d.get("frame_idx", 0)),
                -(d["mask"].astype(bool).sum() / max(area, 1)),
                -float(d.get("confidence", 0.0)),
            ),
        )

        clustered_dets: List[Dict[str, Any]] = []
        for det in dets_sorted:
            det_copy = dict(det)
            det_mask = det_copy["mask"].astype(bool)
            det_box = np.asarray(det_copy["box"], dtype=np.float32)
            det_area = max(float(det_mask.sum()), 1.0)
            det_group = int(det_copy.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
            det_seed = det_copy.get("seed_global_track_idx", None)

            best_cluster_idx = None
            best_score = -1.0
            best_threshold = 1.0

            for cluster_idx, cluster_dets in enumerate(clusters):
                ref_det = cluster_dets[-1]
                ref_mask = ref_det["mask"].astype(bool)
                ref_box = np.asarray(ref_det["box"], dtype=np.float32)
                ref_area = max(float(ref_mask.sum()), 1.0)
                mask_iou = _mask_iou(det_mask, ref_mask)
                box_iou = _box_iou_xyxy(det_box, ref_box)
                area_sim = min(det_area, ref_area) / max(det_area, ref_area)
                frame_gap = abs(
                    int(det_copy.get("frame_idx", 0)) - int(ref_det.get("frame_idx", 0))
                )
                gap_bonus = 0.10 if frame_gap <= max(self.discovery_frame_stride, 1) else 0.0

                if det_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                    score = 0.45 * mask_iou + 0.25 * box_iou + 0.20 * area_sim + gap_bonus
                    threshold = 0.18
                elif det_group == SEMANTIC_GROUP_STATIC_THING:
                    score = 0.25 * mask_iou + 0.45 * box_iou + 0.20 * area_sim + gap_bonus
                    threshold = 0.18
                else:
                    score = 0.15 * mask_iou + 0.55 * box_iou + 0.20 * area_sim + gap_bonus
                    threshold = 0.16

                ref_seed = ref_det.get("seed_global_track_idx", None)
                if det_seed is not None and ref_seed is not None and int(det_seed) == int(ref_seed):
                    score += 0.50
                    threshold = min(threshold, 0.05)
                elif frame_gap == 0:
                    # Same-frame detections are different object hypotheses.
                    # Do not merge separate people/objects just because their
                    # areas are similar; require real spatial overlap.
                    if det_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                        score = 0.60 * mask_iou + 0.40 * box_iou
                        threshold = 0.30
                    elif det_group == SEMANTIC_GROUP_STATIC_THING:
                        score = 0.45 * mask_iou + 0.55 * box_iou
                        threshold = 0.35
                    else:
                        score = 0.40 * mask_iou + 0.60 * box_iou
                        threshold = 0.35

                if score >= threshold and score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_cluster_idx = cluster_idx

            if best_cluster_idx is None:
                det_copy["_cluster_id"] = len(clusters)
                clusters.append([det_copy])
            else:
                det_copy["_cluster_id"] = int(best_cluster_idx)
                clusters[best_cluster_idx].append(det_copy)

            clustered_dets.append(det_copy)

        return clustered_dets

    def _build_sam31_prompt_only_detection(
        self,
        label: str,
        prompt_obj_id: int,
        prompt_frame_idx: int,
        obj_segments: Dict[int, Dict[int, np.ndarray]],
        obj_scores: Dict[int, float],
        *,
        area: int,
    ) -> Optional[Dict[str, Any]]:
        frame_to_mask = obj_segments.get(int(prompt_obj_id), {})
        if not frame_to_mask:
            return None

        use_frame_idx = (
            int(prompt_frame_idx)
            if int(prompt_frame_idx) in frame_to_mask
            else int(min(frame_to_mask.keys()))
        )
        mask = frame_to_mask.get(use_frame_idx)
        if mask is None:
            return None

        mask_bool = mask.astype(bool)
        mask_area = int(mask_bool.sum())
        if mask_area <= 0:
            return None

        area_ratio = float(mask_area) / max(area, 1)
        if area_ratio < self.min_mask_area_ratio or area_ratio > self.max_mask_area_ratio:
            return None

        box = self._mask_to_box_xyxy(mask_bool)
        if float(box[2] - box[0]) <= 1.0 or float(box[3] - box[1]) <= 1.0:
            return None

        return {
            "mask": mask_bool.astype(np.uint8),
            "box": box.astype(np.float32),
            "confidence": float(obj_scores.get(int(prompt_obj_id), 0.0)),
            "label": str(label),
            "raw_label": str(label),
            "sem_group": label_to_group(label),
            "area_ratio": area_ratio,
            "frame_idx": int(use_frame_idx),
            "detector_source": "sam31_prompt",
            "is_prompt_only": True,
        }

    def _person_support_should_repair(
        self,
        current_mask: np.ndarray,
        support_mask: np.ndarray,
    ) -> bool:
        current = current_mask.astype(bool)
        support = support_mask.astype(bool)
        current_area = float(current.sum())
        support_area = float(support.sum())
        if current_area <= 0.0 or support_area <= 0.0:
            return False

        inter = float((current & support).sum())
        if inter <= 0.0:
            return False
        current_cover = inter / max(current_area, 1.0)
        support_cover = inter / max(support_area, 1.0)
        if current_cover < 0.18 and support_cover < 0.06:
            return False

        c_box = self._mask_to_box_xyxy(current)
        s_box = self._mask_to_box_xyxy(support)
        c_w = float(c_box[2] - c_box[0] + 1.0)
        c_h = float(c_box[3] - c_box[1] + 1.0)
        s_w = float(s_box[2] - s_box[0] + 1.0)
        s_h = float(s_box[3] - s_box[1] + 1.0)
        if c_w <= 1.0 or c_h <= 1.0 or s_w <= 1.0 or s_h <= 1.0:
            return False

        c_cx = 0.5 * float(c_box[0] + c_box[2])
        c_cy = 0.5 * float(c_box[1] + c_box[3])
        s_cx = 0.5 * float(s_box[0] + s_box[2])
        s_cy = 0.5 * float(s_box[1] + s_box[3])
        center_dx = abs(c_cx - s_cx) / max(0.5 * (c_w + s_w), 1.0)
        center_dy = abs(c_cy - s_cy) / max(0.5 * (c_h + s_h), 1.0)
        if center_dx > 0.80 or center_dy > 0.95:
            return False

        bottom_gap = float(s_box[3] - c_box[3])
        height_gain = s_h / max(c_h, 1.0)
        area_gain = support_area / max(current_area, 1.0)
        return (
            area_gain >= 1.18
            or height_gain >= 1.18
            or bottom_gap >= 0.14 * max(s_h, 1.0)
        )

    def _repair_person_track_with_detector_support(
        self,
        frame_to_mask: Dict[int, np.ndarray],
        matched_dets: List[Tuple[Dict[str, Any], float]],
    ) -> Dict[int, np.ndarray]:
        if not frame_to_mask or not matched_dets:
            return frame_to_mask

        support_by_frame: Dict[int, Tuple[float, float, np.ndarray]] = {}
        for det, match_score in matched_dets:
            if float(match_score) < 0.10:
                continue
            det_mask = np.asarray(det.get("mask")).astype(bool)
            if det_mask.sum() <= 0:
                continue
            frame_idx = int(det.get("frame_idx", 0))
            det_area = float(det_mask.sum())
            prev = support_by_frame.get(frame_idx)
            candidate_rank = (float(match_score), det_area)
            prev_rank = (prev[0], prev[1]) if prev is not None else (-1.0, -1.0)
            if prev is None or candidate_rank > prev_rank:
                support_by_frame[frame_idx] = (
                    float(match_score),
                    det_area,
                    det_mask.astype(np.uint8),
                )

        if not support_by_frame:
            return frame_to_mask

        repaired: Dict[int, np.ndarray] = {
            int(frame_idx): np.asarray(mask).astype(np.uint8)
            for frame_idx, mask in frame_to_mask.items()
        }

        for frame_idx, (_, _, support_mask) in support_by_frame.items():
            current = repaired.get(int(frame_idx))
            if current is None:
                repaired[int(frame_idx)] = support_mask.astype(np.uint8)
                continue
            if self._person_support_should_repair(current, support_mask):
                repaired[int(frame_idx)] = (
                    current.astype(bool) | support_mask.astype(bool)
                ).astype(np.uint8)

        support_frames = sorted(support_by_frame.keys())
        support_radius = max(int(self.discovery_frame_stride) + 1, 5)
        for frame_idx, current in list(repaired.items()):
            if int(frame_idx) in support_by_frame:
                continue
            nearest = min(
                support_frames,
                key=lambda support_frame: abs(int(support_frame) - int(frame_idx)),
            )
            if abs(int(nearest) - int(frame_idx)) > support_radius:
                continue
            support_mask = support_by_frame[int(nearest)][2]
            if self._person_support_should_repair(current, support_mask):
                repaired[int(frame_idx)] = (
                    current.astype(bool) | support_mask.astype(bool)
                ).astype(np.uint8)

        return repaired

    def _match_person_support_dets_to_track(
        self,
        frame_to_mask: Dict[int, np.ndarray],
        support_dets: Optional[List[Dict[str, Any]]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not frame_to_mask or not support_dets:
            return []

        matched: List[Tuple[Dict[str, Any], float]] = []
        track_frames = sorted(int(frame_idx) for frame_idx in frame_to_mask.keys())
        support_radius = max(int(self.discovery_frame_stride) * 6, 12)
        for det in support_dets:
            label = str(det.get("label", ""))
            if not (_labels_compatible(label, "person") or _labels_compatible(label, "people")):
                continue
            frame_idx = int(det.get("frame_idx", 0))
            current = frame_to_mask.get(frame_idx)
            det_mask = np.asarray(det.get("mask")).astype(bool)
            if det_mask.sum() <= 0:
                continue
            if current is None:
                det_box = self._mask_to_box_xyxy(det_mask)
                det_area = float(det_mask.sum())
                best_score = -1.0
                for ref_frame in track_frames:
                    frame_gap = abs(int(ref_frame) - frame_idx)
                    if frame_gap > support_radius:
                        continue
                    ref_mask = np.asarray(frame_to_mask.get(ref_frame)).astype(bool)
                    ref_area = float(ref_mask.sum())
                    if ref_area <= 0.0:
                        continue
                    ref_box = self._mask_to_box_xyxy(ref_mask)
                    ref_w = max(float(ref_box[2] - ref_box[0] + 1.0), 1.0)
                    ref_h = max(float(ref_box[3] - ref_box[1] + 1.0), 1.0)
                    det_w = max(float(det_box[2] - det_box[0] + 1.0), 1.0)
                    det_h = max(float(det_box[3] - det_box[1] + 1.0), 1.0)
                    ref_cx = 0.5 * float(ref_box[0] + ref_box[2])
                    ref_cy = 0.5 * float(ref_box[1] + ref_box[3])
                    det_cx = 0.5 * float(det_box[0] + det_box[2])
                    det_cy = 0.5 * float(det_box[1] + det_box[3])
                    center_norm = float(
                        np.hypot(det_cx - ref_cx, det_cy - ref_cy)
                        / max(0.5 * (np.hypot(ref_w, ref_h) + np.hypot(det_w, det_h)), 1.0)
                    )
                    box_iou = _box_iou_xyxy(det_box, ref_box)
                    area_sim = min(det_area, ref_area) / max(det_area, ref_area)
                    size_sim = 0.5 * (
                        min(det_w, ref_w) / max(det_w, ref_w)
                        + min(det_h, ref_h) / max(det_h, ref_h)
                    )
                    partial_ref_recovery = (
                        ref_area < 0.28 * det_area
                        and det_area > 0.0
                        and frame_gap <= support_radius
                        and center_norm <= 0.88
                        and det_h >= 1.35 * ref_h
                    )

                    if box_iou < 0.02 and center_norm > 1.20 and not partial_ref_recovery:
                        continue
                    if (
                        area_sim < 0.10
                        and box_iou < 0.08
                        and size_sim < 0.30
                        and not partial_ref_recovery
                    ):
                        continue
                    if frame_gap > max(int(self.discovery_frame_stride) * 3, 6):
                        if (
                            center_norm > 0.90
                            or area_sim < 0.18
                        ) and not partial_ref_recovery:
                            continue

                    time_score = 1.0 - min(float(frame_gap) / max(float(support_radius), 1.0), 1.0)
                    center_score = 1.0 - min(center_norm / 1.40, 1.0)
                    score = (
                        0.34 * center_score
                        + 0.26 * box_iou
                        + 0.22 * area_sim
                        + 0.10 * size_sim
                        + 0.08 * time_score
                    )
                    if partial_ref_recovery:
                        score += 0.08
                    best_score = max(best_score, float(score))
                if best_score < 0.34:
                    continue
                matched.append((det, float(best_score)))
                continue

            if not self._person_support_should_repair(current, det_mask):
                continue

            current_bool = current.astype(bool)
            inter = float((current_bool & det_mask).sum())
            current_cover = inter / max(float(current_bool.sum()), 1.0)
            det_cover = inter / max(float(det_mask.sum()), 1.0)
            score = 0.70 * current_cover + 0.30 * det_cover
            matched.append((det, float(score)))

        return matched

    def _build_sam31_prompt_track_candidate_records(
        self,
        label: str,
        matched_by_prompt_obj_id: Dict[int, List[Tuple[Dict[str, Any], float]]],
        obj_segments: Dict[int, Dict[int, np.ndarray]],
        obj_scores: Dict[int, float],
        *,
        prompt_frame_idx: Optional[int] = None,
        include_unmatched_prompt_tracks: bool = False,
        area: Optional[int] = None,
        support_dets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        all_prompt_obj_ids = (
            sorted(int(obj_id) for obj_id in obj_segments.keys())
            if include_unmatched_prompt_tracks
            else sorted(int(obj_id) for obj_id in matched_by_prompt_obj_id.keys())
        )

        candidates: List[Dict[str, Any]] = []
        for prompt_obj_id in all_prompt_obj_ids:
            matched_dets = list(matched_by_prompt_obj_id.get(int(prompt_obj_id), []))
            prompt_only = len(matched_dets) == 0
            if matched_dets:
                matched_dets.sort(
                    key=lambda item: (
                        -int(bool(item[0].get("seed_global_track_idx") is not None)),
                        -float(item[1]),
                        int(item[0].get("frame_idx", 0)),
                        -float(item[0].get("confidence", 0.0)),
                    ),
                )
                best_det = dict(matched_dets[0][0])
                support_frames = sorted(
                    {int(det.get("frame_idx", 0)) for det, _ in matched_dets}
                )
                cluster_ids = sorted(
                    {
                        int(det["_cluster_id"])
                        for det, _ in matched_dets
                        if det.get("_cluster_id") is not None
                    }
                )
                seed_global_ids = sorted(
                    {
                        int(det.get("seed_global_track_idx"))
                        for det, _ in matched_dets
                        if det.get("seed_global_track_idx") is not None
                    }
                )
                best_match_score = float(max(score for _, score in matched_dets))
            else:
                if prompt_frame_idx is None or area is None:
                    continue
                best_det = self._build_sam31_prompt_only_detection(
                    label=label,
                    prompt_obj_id=int(prompt_obj_id),
                    prompt_frame_idx=int(prompt_frame_idx),
                    obj_segments=obj_segments,
                    obj_scores=obj_scores,
                    area=int(area),
                )
                if best_det is None:
                    continue
                support_frames = [int(best_det.get("frame_idx", prompt_frame_idx))]
                cluster_ids = []
                seed_global_ids = []
                best_match_score = float(best_det.get("confidence", 0.0))

            frame_to_mask = {
                int(frame_idx): mask
                for frame_idx, mask in obj_segments.get(int(prompt_obj_id), {}).items()
            }
            if not frame_to_mask:
                continue
            if (
                _labels_compatible(label, "person")
                or _labels_compatible(label, "people")
            ):
                repair_dets = list(matched_dets)
                loose_support = self._match_person_support_dets_to_track(
                    frame_to_mask,
                    support_dets,
                )
                if loose_support:
                    seen_support = {
                        (
                            int(det.get("frame_idx", 0)),
                            int(det.get("_cluster_id", -1)),
                            int(det.get("seed_global_track_idx", -1))
                            if det.get("seed_global_track_idx", None) is not None
                            else -1,
                        )
                        for det, _ in repair_dets
                    }
                    for det, score in loose_support:
                        key = (
                            int(det.get("frame_idx", 0)),
                            int(det.get("_cluster_id", -1)),
                            int(det.get("seed_global_track_idx", -1))
                            if det.get("seed_global_track_idx", None) is not None
                            else -1,
                        )
                        if key in seen_support:
                            continue
                        repair_dets.append((det, score))
                        seen_support.add(key)
                # SAM3.1 text prompts sometimes lock onto torso+guitar and miss
                # visible legs. YOLOE person masks are only used as local support
                # for the same matched prompt object, not as independent tracks.
                if repair_dets:
                    frame_to_mask = self._repair_person_track_with_detector_support(
                        frame_to_mask,
                        repair_dets,
                    )

            mask_areas = [
                float(mask.astype(bool).sum()) for mask in frame_to_mask.values()
            ]
            area_denom = None
            if area is not None:
                area_denom = max(float(area), 1.0)
            elif best_det.get("area_ratio", None) is not None:
                best_area = float(best_det.get("area_ratio", 0.0))
                best_mask_area = float(best_det["mask"].astype(bool).sum())
                if best_area > 0.0 and best_mask_area > 0.0:
                    area_denom = best_mask_area / best_area
            mean_mask_area = float(sum(mask_areas) / len(mask_areas)) if mask_areas else 0.0
            mean_mask_area_ratio = (
                float(mean_mask_area / area_denom)
                if area_denom is not None and area_denom > 0.0
                else float(best_det.get("area_ratio", 0.0))
            )
            support_start = int(best_det.get("frame_idx", support_frames[0]))
            candidate_prompt_obj_id = (
                int(prompt_frame_idx) * 10000 + int(prompt_obj_id)
                if prompt_frame_idx is not None
                else int(prompt_obj_id)
            )
            candidates.append(
                {
                    "label": str(label),
                    "prompt_obj_id": int(candidate_prompt_obj_id),
                    "best_det": best_det,
                    "segments": frame_to_mask,
                    "support_frames": support_frames,
                    "support_start": support_start,
                    "track_length": int(len(frame_to_mask)),
                    "obj_score": float(obj_scores.get(prompt_obj_id, 0.0)),
                    "best_match_score": float(best_match_score),
                    "seed_global_ids": seed_global_ids,
                    "cluster_ids": cluster_ids,
                    "primary_cluster_id": (
                        int(cluster_ids[0]) if cluster_ids else None
                    ),
                    "primary_seed_global_idx": (
                        int(seed_global_ids[0]) if seed_global_ids else None
                    ),
                    "mean_mask_area": mean_mask_area,
                    "mean_mask_area_ratio": mean_mask_area_ratio,
                    "prompt_only": bool(prompt_only),
                    "prompt_frame_idx": (
                        int(prompt_frame_idx)
                        if prompt_frame_idx is not None
                        else support_start
                    ),
                }
            )

        return candidates

    def _build_sam31_object_prompt_track_candidates(
        self,
        label: str,
        detections: List[Dict[str, Any]],
        obj_segments: Dict[int, Dict[int, np.ndarray]],
        obj_scores: Dict[int, float],
        obj_to_det: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for obj_id, frame_to_mask_raw in obj_segments.items():
            obj_id = int(obj_id)
            det_src = obj_to_det.get(obj_id)
            if det_src is None:
                continue
            best_det = dict(det_src)
            frame_to_mask = {
                int(frame_idx): mask
                for frame_idx, mask in frame_to_mask_raw.items()
                if mask is not None
            }
            if not frame_to_mask:
                continue

            support_frame = int(best_det.get("frame_idx", min(frame_to_mask.keys())))
            support_frames = [support_frame]
            cluster_ids = (
                [int(best_det["_cluster_id"])]
                if best_det.get("_cluster_id", None) is not None
                else []
            )
            seed_global_ids = (
                [int(best_det.get("seed_global_track_idx"))]
                if best_det.get("seed_global_track_idx", None) is not None
                else []
            )
            mask_areas = [
                float(mask.astype(bool).sum()) for mask in frame_to_mask.values()
            ]
            mean_mask_area = float(sum(mask_areas) / len(mask_areas)) if mask_areas else 0.0
            best_area_ratio = float(best_det.get("area_ratio", 0.0))
            best_mask_area = float(best_det["mask"].astype(bool).sum())
            area_denom = (
                best_mask_area / best_area_ratio
                if best_area_ratio > 0.0 and best_mask_area > 0.0
                else None
            )
            mean_mask_area_ratio = (
                float(mean_mask_area / area_denom)
                if area_denom is not None and area_denom > 0.0
                else best_area_ratio
            )
            track_length = int(len(frame_to_mask))
            obj_score = float(obj_scores.get(obj_id, best_det.get("confidence", 0.0)))
            min_track_length = max(5, self.discovery_frame_stride + 1)
            if track_length < min_track_length or obj_score < 0.35:
                continue

            if _labels_compatible(label, "person") or _labels_compatible(label, "people"):
                min_area_ratio = 0.0008
                max_area_ratio = 0.45
            else:
                min_area_ratio = 0.0010
                max_area_ratio = 0.35
            if mean_mask_area_ratio < min_area_ratio or mean_mask_area_ratio > max_area_ratio:
                continue

            candidates.append(
                {
                    "label": str(label),
                    "prompt_obj_id": int(900000000 + obj_id),
                    "best_det": best_det,
                    "segments": frame_to_mask,
                    "support_frames": support_frames,
                    "support_start": int(min(frame_to_mask.keys())),
                    "track_length": track_length,
                    "obj_score": obj_score,
                    "best_match_score": 1.0,
                    "seed_global_ids": seed_global_ids,
                    "cluster_ids": cluster_ids,
                    "primary_cluster_id": int(cluster_ids[0]) if cluster_ids else None,
                    "primary_seed_global_idx": int(seed_global_ids[0]) if seed_global_ids else None,
                    "mean_mask_area": mean_mask_area,
                    "mean_mask_area_ratio": mean_mask_area_ratio,
                    "prompt_only": False,
                    "prompt_frame_idx": support_frame,
                }
            )

        return candidates

    def _select_sam31_prompt_track_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        duplicate_threshold: float,
        min_support_frames: int,
        min_track_length: int,
        min_obj_score: float,
        max_prompt_only_tracks: Optional[int] = None,
        prompt_only_min_track_length: Optional[int] = None,
        prompt_only_min_obj_score: Optional[float] = None,
        prompt_only_min_area_ratio: float = 0.0,
        prompt_only_max_area_ratio: float = 1.0,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        candidates.sort(
            key=lambda item: (
                int(item["primary_seed_global_idx"] is not None),
                int(not item.get("prompt_only", False)),
                int(len(item["seed_global_ids"])),
                int(len(item["support_frames"])),
                float(item["best_match_score"]),
                float(item["obj_score"]),
                int(item["track_length"]),
                float(item["mean_mask_area"]),
                -int(item["support_start"]),
            ),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []
        used_seed_ids: Set[int] = set()
        used_cluster_ids: Set[int] = set()
        selected_prompt_only = 0
        for candidate in candidates:
            is_prompt_only = bool(candidate.get("prompt_only", False))
            if is_prompt_only:
                if max_prompt_only_tracks is not None and selected_prompt_only >= max_prompt_only_tracks:
                    continue
                if (
                    prompt_only_min_track_length is not None
                    and int(candidate["track_length"]) < int(prompt_only_min_track_length)
                ):
                    continue
                if (
                    prompt_only_min_obj_score is not None
                    and float(candidate["obj_score"]) < float(prompt_only_min_obj_score)
                ):
                    continue
                mean_area_ratio = float(candidate.get("mean_mask_area_ratio", 0.0))
                if mean_area_ratio < prompt_only_min_area_ratio:
                    continue
                if mean_area_ratio > prompt_only_max_area_ratio:
                    continue

            candidate_seed_ids = {
                int(seed_idx) for seed_idx in candidate["seed_global_ids"]
            }
            if candidate_seed_ids & used_seed_ids:
                # A handoff seed should continue as only one local track.
                continue
            candidate_cluster_ids = {
                int(cluster_idx) for cluster_idx in candidate["cluster_ids"]
            }
            candidate_group = int(
                candidate["best_det"].get(
                    "sem_group",
                    SEMANTIC_GROUP_UNCERTAIN_REGION,
                )
            )
            if (
                candidate["primary_seed_global_idx"] is None
                and candidate_group != SEMANTIC_GROUP_MOVABLE_THING
                and candidate_cluster_ids & used_cluster_ids
            ):
                continue

            is_duplicate = False
            for chosen in selected:
                if not _labels_compatible(chosen["label"], candidate["label"]):
                    continue
                similarity = self._compute_sam31_prompt_track_similarity(
                    chosen["segments"],
                    candidate["segments"],
                )
                if similarity >= duplicate_threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            if candidate["primary_seed_global_idx"] is None:
                if (
                    len(candidate["support_frames"]) < min_support_frames
                    and int(candidate["track_length"]) < min_track_length
                    and float(candidate["obj_score"]) < min_obj_score
                ):
                    continue

            selected.append(candidate)
            used_seed_ids.update(candidate_seed_ids)
            used_cluster_ids.update(candidate_cluster_ids)
            if is_prompt_only:
                selected_prompt_only += 1

        return selected

    def _build_sam31_prompt_track_candidates(
        self,
        label: str,
        matched_by_prompt_obj_id: Dict[int, List[Tuple[Dict[str, Any], float]]],
        obj_segments: Dict[int, Dict[int, np.ndarray]],
        obj_scores: Dict[int, float],
        *,
        prompt_frame_idx: Optional[int] = None,
        include_unmatched_prompt_tracks: bool = False,
        area: Optional[int] = None,
        duplicate_threshold: float,
        min_support_frames: int,
        min_track_length: int,
        min_obj_score: float,
        support_dets: Optional[List[Dict[str, Any]]] = None,
        max_prompt_only_tracks: Optional[int] = None,
        prompt_only_min_track_length: Optional[int] = None,
        prompt_only_min_obj_score: Optional[float] = None,
        prompt_only_min_area_ratio: float = 0.0,
        prompt_only_max_area_ratio: float = 1.0,
    ) -> List[Dict[str, Any]]:
        candidates = self._build_sam31_prompt_track_candidate_records(
            label=label,
            matched_by_prompt_obj_id=matched_by_prompt_obj_id,
            obj_segments=obj_segments,
            obj_scores=obj_scores,
            prompt_frame_idx=prompt_frame_idx,
            include_unmatched_prompt_tracks=include_unmatched_prompt_tracks,
            area=area,
            support_dets=support_dets,
        )
        return self._select_sam31_prompt_track_candidates(
            candidates,
            duplicate_threshold=duplicate_threshold,
            min_support_frames=min_support_frames,
            min_track_length=min_track_length,
            min_obj_score=min_obj_score,
            max_prompt_only_tracks=max_prompt_only_tracks,
            prompt_only_min_track_length=prompt_only_min_track_length,
            prompt_only_min_obj_score=prompt_only_min_obj_score,
            prompt_only_min_area_ratio=prompt_only_min_area_ratio,
            prompt_only_max_area_ratio=prompt_only_max_area_ratio,
        )

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
        prompt_obj_ids = [int(obj_id) for obj_id in prompt_outputs.get("out_obj_ids", [])]
        if len(prompt_obj_ids) == 0:
            return {}, {}
        prompt_scores = prompt_outputs.get("out_probs", [])
        prompt_masks = prompt_outputs.get("out_binary_masks", [])

        max_prompt_objects = int(self.sam31_max_text_prompt_objects)
        if max_prompt_objects > 0 and len(prompt_obj_ids) > max_prompt_objects:
            ranked: List[Tuple[float, float, float, int, int]] = []
            for idx, obj_id in enumerate(prompt_obj_ids):
                score = float(prompt_scores[idx]) if idx < len(prompt_scores) else 0.0
                if idx < len(prompt_masks):
                    mask_np = np.asarray(prompt_masks[idx]).astype(bool)
                    area_ratio = float(mask_np.sum()) / max(float(mask_np.size), 1.0)
                else:
                    area_ratio = 0.0
                priority = score + 0.08 * min(area_ratio * 30.0, 1.0)
                ranked.append((priority, score, area_ratio, -idx, int(obj_id)))
            keep_obj_ids = {
                int(item[-1])
                for item in sorted(ranked, reverse=True)[:max_prompt_objects]
            }
            # Do not remove extra SAM3.1 text-prompt objects from the session
            # before propagation. Removing here mutates SAM3.1's interactive
            # action / hotstart state and can turn a full propagation into a
            # cache-only fetch or violate bucket-size assertions. We instead
            # let SAM3.1 propagate its full prompt set and only collect the
            # ranked subset below.
            keep_indices = [
                idx for idx, obj_id in enumerate(prompt_obj_ids)
                if int(obj_id) in keep_obj_ids
            ]
            prompt_obj_ids = [prompt_obj_ids[idx] for idx in keep_indices]
        else:
            keep_indices = list(range(len(prompt_obj_ids)))

        obj_segments: Dict[int, Dict[int, np.ndarray]] = {
            int(obj_id): {} for obj_id in prompt_obj_ids
        }
        obj_scores: Dict[int, float] = {}
        allowed_obj_ids = {int(obj_id) for obj_id in prompt_obj_ids}

        for idx in keep_indices:
            obj_id = int(prompt_outputs.get("out_obj_ids", [])[idx])
            obj_id = int(obj_id)
            if idx < len(prompt_scores):
                obj_scores[obj_id] = max(obj_scores.get(obj_id, 0.0), float(prompt_scores[idx]))
            if idx < len(prompt_masks):
                obj_segments[obj_id][int(frame_idx)] = prompt_masks[idx].astype(np.uint8)

        def _consume(direction: str) -> None:
            for prop_response in self.video_predictor.handle_stream_request(
                dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=direction,
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
                    if obj_id not in allowed_obj_ids:
                        continue
                    if idx < len(out_masks):
                        obj_segments.setdefault(obj_id, {})[out_frame_idx] = out_masks[idx].astype(np.uint8)
                    if idx < len(out_probs):
                        obj_scores[obj_id] = max(obj_scores.get(obj_id, 0.0), float(out_probs[idx]))

        _consume("forward")
        if self.sam31_enable_backward and int(frame_idx) > 0:
            try:
                _consume("backward")
            except KeyError as exc:
                # SAM3.1 multiplex occasionally misses reverse-cache entries on long runs.
                # Keep the forward chunk result rather than crashing the whole video.
                print(
                    f"[sam31_multiplex] warning: reverse propagation failed for "
                    f"label={label!r} frame_idx={frame_idx} with {exc!r}; "
                    "keeping forward-only result"
                )

        obj_segments = {
            obj_id: frame_to_mask
            for obj_id, frame_to_mask in obj_segments.items()
            if frame_to_mask
        }
        return obj_segments, obj_scores

    @torch.inference_mode()
    def _propagate_things(
        self, frame_resource: Any, thing_dets: List[Dict], T: int, H: int, W: int,
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        if self.sam_backend == "sam3":
            dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
            for det in thing_dets:
                dets_by_frame.setdefault(int(det.get("frame_idx", 0)), []).append(det)
            selected, segments, oid_map = self._select_multiframe_things_sam3(
                frame_resource,
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
                frame_resource,
                dets_by_frame,
                sorted(dets_by_frame.keys()),
                T,
                H,
                W,
                H * W,
            )
            return segments, oid_map

        inference_state = self.video_predictor.init_state(video_path=frame_resource)
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

    def _append_sparse_detector_thing_tracks(
        self,
        support_dets: List[Dict[str, Any]],
        discovery_indices: List[int],
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict],
        area: int,
    ) -> None:
        """Turn repeated detector hits into cheap sparse object tracks.

        SAM3.1 object-prompt propagation is the expensive path. For non-person
        objects, YOLOE detections across discovery frames are often enough to
        provide stable support and cross-chunk IDs. We only write masks on
        supported frames; the global short-gap bridge may fill tiny gaps later.
        """
        if not support_dets:
            return

        ordered_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for det in support_dets:
            det_label = str(det.get("label", ""))
            if not (
                _labels_compatible(det_label, "person")
                or _labels_compatible(det_label, "people")
            ):
                continue
            if int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) != SEMANTIC_GROUP_MOVABLE_THING:
                continue
            if float(det.get("confidence", 0.0)) < 0.38:
                continue
            det_frame = int(det.get("frame_idx", 0))
            if self._is_duplicate_thing_detection(
                det,
                det_frame,
                thing_segments,
                thing_oid_map,
            ):
                continue
            ordered_by_frame.setdefault(det_frame, []).append(dict(det))

        ordered_dets: List[Dict[str, Any]] = []
        for det_frame in sorted(ordered_by_frame):
            frame_dets = sorted(
                ordered_by_frame[det_frame],
                key=lambda d: (
                    -float(d.get("confidence", 0.0)),
                    -float(d.get("area_ratio", 0.0)),
                ),
            )
            ordered_dets.extend(frame_dets[:4])
        max_sparse_support = max(24, min(128, 2 * max(len(discovery_indices), 1)))
        if len(ordered_dets) > max_sparse_support:
            keep_indices = {
                int(round(float(pos)))
                for pos in np.linspace(0, len(ordered_dets) - 1, max_sparse_support)
            }
            ordered_dets = [
                det for idx, det in enumerate(ordered_dets) if idx in keep_indices
            ][:max_sparse_support]
        if not ordered_dets:
            return

        discovery_set = {int(v) for v in discovery_indices}
        ordered_dets.sort(
            key=lambda d: (
                int(d.get("frame_idx", 0)) not in discovery_set,
                int(d.get("frame_idx", 0)),
                -int(bool(d.get("is_seed_track", False))),
                -float(d.get("confidence", 0.0)),
                -float(d.get("area_ratio", 0.0)),
            )
        )

        clusters: List[Dict[str, Any]] = []
        for det in ordered_dets:
            det_frame = int(det.get("frame_idx", 0))
            det_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
            det_label = str(det.get("label", "unknown"))
            det_mask = det["mask"].astype(bool)
            det_box = np.asarray(det["box"], dtype=np.float32)
            det_area = max(float(det_mask.sum()), 1.0)

            best_cluster: Optional[Dict[str, Any]] = None
            best_score = -1.0
            best_threshold = 1.0
            for cluster in clusters:
                if int(cluster["sem_group"]) != det_group:
                    continue
                if not _labels_compatible(str(cluster["label"]), det_label):
                    continue
                ref_det = cluster["detections"][-1]
                ref_mask = ref_det["mask"].astype(bool)
                ref_box = np.asarray(ref_det["box"], dtype=np.float32)
                ref_area = max(float(ref_mask.sum()), 1.0)
                mask_iou = _mask_iou(det_mask, ref_mask)
                box_iou = _box_iou_xyxy(det_box, ref_box)
                area_sim = min(det_area, ref_area) / max(det_area, ref_area)
                frame_gap = abs(det_frame - int(ref_det.get("frame_idx", det_frame)))
                gap_bonus = 0.12 if frame_gap <= max(self.discovery_frame_stride, 1) else 0.0

                if det_group == SEMANTIC_GROUP_STATIC_THING:
                    score = 0.25 * mask_iou + 0.45 * box_iou + 0.20 * area_sim + gap_bonus
                    threshold = 0.18
                else:
                    score = 0.15 * mask_iou + 0.55 * box_iou + 0.20 * area_sim + gap_bonus
                    threshold = 0.16

                if frame_gap == 0:
                    if det_group == SEMANTIC_GROUP_STATIC_THING:
                        score = 0.45 * mask_iou + 0.55 * box_iou
                        threshold = 0.35
                    else:
                        score = 0.40 * mask_iou + 0.60 * box_iou
                        threshold = 0.35

                if score >= threshold and score > best_score:
                    best_score = float(score)
                    best_threshold = float(threshold)
                    best_cluster = cluster

            if best_cluster is not None and best_score >= best_threshold:
                best_cluster["detections"].append(det)
            else:
                clusters.append({
                    "label": det_label,
                    "sem_group": det_group,
                    "detections": [det],
                })

        next_oid = max(thing_oid_map.keys(), default=0) + 1
        for cluster in clusters:
            dets = list(cluster["detections"])
            if not dets:
                continue

            frame_to_det: Dict[int, Dict[str, Any]] = {}
            for det in dets:
                frame_idx = int(det.get("frame_idx", 0))
                prev = frame_to_det.get(frame_idx)
                if prev is None or float(det.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    frame_to_det[frame_idx] = det
            support_frames = sorted(frame_to_det.keys())
            if not support_frames:
                continue

            seed_dets = [d for d in dets if d.get("seed_global_track_idx", None) is not None]
            max_conf = max(float(d.get("confidence", 0.0)) for d in dets)
            max_area_ratio = max(float(d.get("area_ratio", 0.0)) for d in dets)
            sem_group = int(cluster["sem_group"])

            if not seed_dets:
                if sem_group == SEMANTIC_GROUP_STATIC_THING:
                    keep = len(support_frames) >= 2 or max_conf >= 0.58 or max_area_ratio >= 0.03
                else:
                    keep = len(support_frames) >= 2 or max_conf >= 0.68
                if not keep:
                    continue

            best_det = dict(
                sorted(
                    dets,
                    key=lambda d: (
                        -int(d.get("seed_global_track_idx", None) is not None),
                        int(d.get("frame_idx", 0)),
                        -float(d.get("confidence", 0.0)),
                        -float(d.get("area_ratio", 0.0)),
                    ),
                )[0]
            )
            best_det["frame_idx"] = int(support_frames[0])
            best_det["confidence"] = float(max_conf)
            best_det["cluster_size"] = int(len(support_frames))
            best_det["support_frames"] = list(support_frames)
            best_det["detector_source"] = "detector_sparse_track"
            if seed_dets:
                best_det["seed_global_track_idx"] = int(seed_dets[0]["seed_global_track_idx"])

            if self._is_duplicate_thing_detection(
                best_det,
                int(best_det.get("frame_idx", 0)),
                thing_segments,
                thing_oid_map,
            ):
                continue

            oid = next_oid
            next_oid += 1
            thing_oid_map[oid] = best_det
            for frame_idx in support_frames:
                chosen = frame_to_det[int(frame_idx)]
                thing_segments.setdefault(int(frame_idx), {})[oid] = (
                    chosen["mask"].astype(np.uint8)
                )

    def _build_stuff_entries(
        self, stuff_dets: List[Dict],
        frame_resource: Any, T: int, H: int, W: int,
    ) -> List[Dict]:
        """Build sparse stuff entries from detector support frames.

        Copying a single stuff mask to all later frames caused severe temporal
        artifacts, so we only keep masks on the frames where the detector
        actually supported them. Structure-style stuff is handled separately by
        the tracked SAM3.1 path when possible.
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
            for t in detected_frames:
                chosen = frame_to_det[t]
                frame_to_mask[int(t)] = chosen["mask"].astype(np.uint8)
                frame_to_confidence[int(t)] = float(chosen["confidence"])

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

        L_sem, source_type, birth_frame, seed_global_track_idx = [], [], [], []
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
            seed_global_track_idx.append(det.get("seed_global_track_idx", None))

            for t in range(T):
                if t not in thing_segments or oid not in thing_segments[t]:
                    continue
                prob = thing_segments[t][oid]
                if prob.ndim == 3:
                    prob = prob.squeeze(0)
                if prob.shape[0] != H or prob.shape[1] != W:
                    prob = cv2.resize(prob.astype(np.float32), (W, H),
                                      interpolation=cv2.INTER_NEAREST)

                binary = _clean_instance_mask_components(
                    prob,
                    int(g),
                    str(det["label"]),
                    image_area=area,
                )
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
            seed_global_track_idx.append(det.get("seed_global_track_idx", None))

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

                binary = _clean_instance_mask_components(
                    prob,
                    int(g),
                    str(det["label"]),
                    image_area=area,
                )
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
            seed_global_track_idx.append(entry.get("seed_global_track_idx", None))

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
            seed_global_track_idx=seed_global_track_idx,
            debug={"J_thing": J_thing, "J_structure": J_structure, "J_stuff": J_stuff},
        )

    # -- Helpers -----------------------------------------------------------

    def _write_frames(self, images: torch.Tensor, out_dir: str) -> None:
        for t in range(images.shape[0]):
            img_np = (images[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"{t:05d}.jpg"), img_bgr)

    def _tensor_to_pil_images(self, images: torch.Tensor) -> List[Image.Image]:
        out: List[Image.Image] = []
        for t in range(images.shape[0]):
            img_np = (images[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            out.append(Image.fromarray(img_np, mode="RGB"))
        return out

    def _load_pil_images(self, image_paths: List[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())
        return images

    def _link_frames(self, image_paths: List[str], out_dir: str) -> None:
        for t, src_path in enumerate(image_paths):
            dst_path = os.path.join(out_dir, f"{t:05d}.jpg")
            abs_src = os.path.abspath(src_path)
            try:
                os.symlink(abs_src, dst_path)
            except OSError:
                try:
                    os.link(abs_src, dst_path)
                except OSError:
                    shutil.copy2(abs_src, dst_path)

    def _empty_output(self, T: int, H: int, W: int) -> MaskletOutput:
        return MaskletOutput(
            M_mask=torch.zeros(0, T, H, W), V_mask=torch.zeros(0, T, dtype=torch.bool),
            B_mask=torch.zeros(0, T, 4), Q_mask=torch.zeros(0, T),
            L_sem=[], G_sem=torch.zeros(0, dtype=torch.long),
            W_sem=torch.zeros(0), A_ratio=torch.zeros(0, T),
            num_masklets=0, num_frames=T, frame_height=H, frame_width=W,
        )
