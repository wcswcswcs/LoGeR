from __future__ import annotations

import os
import sys
import gc
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from huggingface_hub import hf_hub_download

from .video_masklet_frontend import (
    MaskletOutput,
    SEMANTIC_GROUP_MOVABLE_THING,
    SEMANTIC_GROUP_STRUCTURE_ANCHOR,
    SEMANTIC_GROUP_NAMES,
    SEMANTIC_GROUP_STATIC_THING,
    SEMANTIC_GROUP_UNCERTAIN_REGION,
    THING_PROMPTS,
    STUFF_PROMPTS,
    VideoMaskletFrontend,
    YOLOEDetector,
    _box_iou_xyxy,
    _is_stuff,
    _mask_iou,
    _labels_compatible,
    canonicalize_label,
    label_to_group,
    passes_structure_mask_quality,
)

DEFAULT_EFFICIENTSAM3_REPO_ID = "Simon7108528/EfficientSAM3"
DEFAULT_EFFICIENTSAM3_FILENAME = "stage1_all_converted/efficient_sam3_efficientvit_s.pt"
DEFAULT_SAM3_CHECKPOINT = "ckpts/SAM3/sam3.pt"
DEFAULT_SAM2_CHECKPOINT = "/home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt"
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_EDGETAM_CHECKPOINT = "third_party/EdgeTAM/checkpoints/edgetam.pt"
DEFAULT_EDGETAM_MODEL_CFG = "configs/edgetam.yaml"
DISABLED_EFFICIENT_PROMPT_LABELS = {"book"}


class _NoOpDetector:
    def detect(self, image_path, thing_prompts, stuff_prompts, box_threshold, text_threshold):
        return []

    def detect_batch(self, image_paths, thing_prompts, stuff_prompts, box_threshold, text_threshold):
        return [[] for _ in image_paths]

    def release_gpu(self):
        return None

    def to_device(self):
        return None


def _ensure_sam3_path() -> None:
    repo_root = Path(__file__).resolve().parents[2] / "third_party" / "sam3"
    repo_root_str = str(repo_root)
    if repo_root.exists() and repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _import_cutie_components():
    repo_root = Path(__file__).resolve().parents[2] / "third_party" / "Cutie"
    repo_root_str = str(repo_root)
    if not repo_root.exists():
        raise RuntimeError(
            "Cutie is not available. Expected third_party/Cutie; clone "
            "https://github.com/hkchengrex/Cutie.git first."
        )
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from cutie.inference.inference_core import InferenceCore
    from cutie.utils.get_default_model import get_default_model
    from hydra.core.global_hydra import GlobalHydra

    return InferenceCore, get_default_model, GlobalHydra


def _import_efficientsam3_components():
    repo_root = Path(__file__).resolve().parents[2] / "third_party" / "efficientsam3" / "sam3"
    package_root = repo_root / "sam3"

    existing = sys.modules.get("sam3")
    if existing is not None:
        mod_file = str(getattr(existing, "__file__", "") or "")
        if str(package_root) not in mod_file:
            raise RuntimeError(
                "A different 'sam3' package is already imported in this process. "
                "Please run EfficientVideoMaskletFrontend in a fresh Python process."
            )

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from sam3.model_builder import build_efficientsam3_video_model
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    return build_efficientsam3_video_model, Sam3VideoPredictor


def _resolve_efficientsam3_checkpoint(
    checkpoint_path: Optional[str],
    *,
    repo_id: str,
    filename: str,
    local_dir: str,
) -> str:
    if checkpoint_path:
        resolved = os.path.expanduser(checkpoint_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"EfficientSAM3 checkpoint not found: {resolved}")
        return resolved

    local_dir_resolved = os.path.expanduser(local_dir)
    os.makedirs(local_dir_resolved, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir_resolved,
        local_dir_use_symlinks=False,
    )


def _resolve_repo_relative_path(path: str) -> str:
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded) or os.path.exists(expanded):
        return expanded
    repo_path = Path(__file__).resolve().parents[2] / expanded
    return str(repo_path)


def _assert_sam2_package_source(expected_fragment: str, backend_name: str) -> None:
    import sam2

    package_root = str(getattr(sam2, "__path__", [""])[0])
    if expected_fragment and expected_fragment not in package_root:
        raise RuntimeError(
            f"{backend_name} needs the '{expected_fragment}' sam2 package, but the "
            f"already imported sam2 package is from: {package_root}. Start a fresh "
            f"process with GSAM2_ROOT pointing at the desired tracker repo."
        )


class EfficientVideoMaskletFrontend(VideoMaskletFrontend):
    """Stage C front-end backed by SAM3 / EfficientSAM3 + YOLOE.

    The default Cutie backend uses SAM3 text-prompt discovery plus YOLOE as the
    high-recall semantic seed generator, then lets Cutie handle cheap temporal
    propagation. The legacy EfficientSAM3 tracker path is still available for
    comparison, but its masks are weaker on Taylor.
    """

    def __init__(
        self,
        *args,
        efficient_model: Optional[Any] = None,
        efficient_tracker: Optional[Any] = None,
        tracker_backend: str = "efficientsam3",
        sam3_image_processor: Optional[Any] = None,
        cutie_model: Optional[Any] = None,
        cutie_inference_core_cls: Optional[Any] = None,
        cutie_max_internal_size: int = 480,
        sam3_cutie_sam_confidence_threshold: float = 0.10,
        sam3_cutie_detection_frame_count: int = 6,
        sam3_cutie_max_prompt_dets_per_label: int = 4,
        sam3_cutie_use_yoloe: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tracker_backend = str(tracker_backend)
        self.efficient_model = efficient_model
        self.efficient_tracker = (
            efficient_tracker
            if efficient_tracker is not None
            else getattr(efficient_model, "tracker", None)
        )
        if self.efficient_model is not None and self.efficient_tracker is not None:
            self.efficient_tracker.backbone = self.efficient_model.detector.backbone
        self.sam3_image_processor = sam3_image_processor
        self.cutie_model = cutie_model
        self.cutie_inference_core_cls = cutie_inference_core_cls
        self.cutie_max_internal_size = int(cutie_max_internal_size)
        self.sam3_cutie_sam_confidence_threshold = float(sam3_cutie_sam_confidence_threshold)
        self.sam3_cutie_detection_frame_count = max(int(sam3_cutie_detection_frame_count), 1)
        self.sam3_cutie_max_prompt_dets_per_label = max(int(sam3_cutie_max_prompt_dets_per_label), 1)
        self.sam3_cutie_use_yoloe = bool(sam3_cutie_use_yoloe)

    def _sam3_cutie_detections_compete(
        self,
        det_a: Dict[str, Any],
        det_b: Dict[str, Any],
    ) -> bool:
        """Return True when two detections should suppress one another.

        The base NMS is label-agnostic. That is dangerous for this backend
        because large wall/floor masks can heavily contain a person seed and
        prevent Cutie from ever receiving the person object. Keep suppression
        semantic-aware so background structures cannot erase THING recall.
        """
        group_a = int(det_a.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
        group_b = int(det_b.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
        label_a = canonicalize_label(str(det_a.get("label", "")))
        label_b = canonicalize_label(str(det_b.get("label", "")))

        if group_a == group_b:
            if group_a in {SEMANTIC_GROUP_MOVABLE_THING, SEMANTIC_GROUP_STATIC_THING}:
                return self._sam3_cutie_labels_compete(label_a, label_b)
            return True
        if (
            (group_a == SEMANTIC_GROUP_STRUCTURE_ANCHOR or _is_stuff(group_a))
            and (group_b == SEMANTIC_GROUP_STRUCTURE_ANCHOR or _is_stuff(group_b))
        ):
            return True
        return False

    def _sam3_cutie_labels_compete(self, label_a: str, label_b: str) -> bool:
        label_a = canonicalize_label(str(label_a))
        label_b = canonicalize_label(str(label_b))
        a_person = _labels_compatible(label_a, "person") or _labels_compatible(label_a, "people")
        b_person = _labels_compatible(label_b, "person") or _labels_compatible(label_b, "people")
        if a_person and b_person:
            return True
        return _labels_compatible(label_a, label_b)

    def _nms(self, dets: List[Dict]) -> List[Dict]:
        if len(dets) <= 1:
            return dets

        masks = [np.asarray(d["mask"]).astype(bool) for d in dets]

        def _priority(idx: int) -> float:
            det = dets[idx]
            confidence = float(det.get("confidence", 0.0))
            area_ratio = float(det.get("area_ratio", 0.0))
            if area_ratio <= 0.0 and masks[idx].size > 0:
                area_ratio = float(masks[idx].sum()) / max(float(masks[idx].size), 1.0)
            label = canonicalize_label(str(det.get("label", "")))
            sem_group = int(det.get("sem_group", label_to_group(label)))
            source = str(det.get("detector_source", "")).lower()

            if (
                sem_group == SEMANTIC_GROUP_MOVABLE_THING
                and (_labels_compatible(label, "person") or _labels_compatible(label, "people"))
            ):
                # Full-body people are more valuable than high-confidence torso
                # shards because Cutie will propagate the seed shape faithfully.
                return confidence + min(area_ratio * 14.0, 2.0) + (0.12 if "yolo" in source else 0.0)
            if sem_group == SEMANTIC_GROUP_MOVABLE_THING:
                return confidence + min(area_ratio * 5.0, 0.8)
            if sem_group == SEMANTIC_GROUP_STATIC_THING:
                return confidence + min(area_ratio * 2.5, 0.5)
            if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR or _is_stuff(sem_group):
                return confidence + min(area_ratio * 1.2, 0.5)
            return confidence

        order = sorted(range(len(dets)), key=_priority, reverse=True)
        keep: List[int] = []
        suppressed: set[int] = set()
        for i in order:
            if i in suppressed:
                continue
            keep.append(i)
            ai = float(masks[i].sum())
            for j in order:
                if j in suppressed or j == i:
                    continue
                if not self._sam3_cutie_detections_compete(dets[i], dets[j]):
                    continue
                inter = float((masks[i] & masks[j]).sum())
                union = ai + float(masks[j].sum()) - inter
                if union > 0.0 and inter / union > float(self.nms_iou_threshold):
                    suppressed.add(j)
        return [dets[i] for i in keep]

    @classmethod
    def from_config(
        cls,
        *,
        device: str = "cuda",
        tracker_backend: str = "sam2",
        sam3_checkpoint: Optional[str] = None,
        sam2_checkpoint: Optional[str] = None,
        sam2_model_cfg: Optional[str] = None,
        edgetam_checkpoint: Optional[str] = None,
        edgetam_model_cfg: Optional[str] = None,
        efficientsam3_checkpoint: Optional[str] = None,
        efficientsam3_repo_id: str = DEFAULT_EFFICIENTSAM3_REPO_ID,
        efficientsam3_filename: str = DEFAULT_EFFICIENTSAM3_FILENAME,
        efficientsam3_cache_dir: str = "ckpts/EfficientSAM3",
        efficientsam3_backbone_type: str = "efficientvit",
        efficientsam3_model_name: str = "b0",
        efficientsam3_text_encoder_type: Optional[str] = None,
        efficientsam3_text_context_length: int = 77,
        yoloe_model: str = "yoloe-11l-seg.pt",
        thing_prompts: Optional[List[str]] = None,
        stuff_prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> "EfficientVideoMaskletFrontend":
        if not device.startswith("cuda"):
            raise ValueError("The efficient video masklet frontend currently requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        tracker_backend = str(tracker_backend).strip().lower()
        if tracker_backend in {"sam3_cutie", "cutie", "sam2", "sam2.1", "edgetam", "edge_tam"}:
            _ensure_sam3_path()
            checkpoint_path = os.path.expanduser(sam3_checkpoint or DEFAULT_SAM3_CHECKPOINT)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            sam3_build_device = "cuda" if str(device).startswith("cuda") else device
            sam3_image_model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                device=sam3_build_device,
            )
            sam3_conf_threshold = float(
                kwargs.pop("sam3_cutie_sam_confidence_threshold", 0.10)
            )
            sam3_image_processor = Sam3Processor(
                sam3_image_model,
                device=device,
                confidence_threshold=sam3_conf_threshold,
            )
            if tracker_backend in {"sam2", "sam2.1", "edgetam", "edge_tam"}:
                # The image model is only needed during discovery. Offload it
                # before building the video tracker to avoid a high transient
                # load peak when Stage C is used inside the full LoGeR pipeline.
                if hasattr(sam3_image_model, "to"):
                    sam3_image_model.to("cpu")
                if hasattr(sam3_image_processor, "device"):
                    sam3_image_processor.device = "cpu"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            sam3_cutie_use_yoloe = bool(kwargs.pop("sam3_cutie_use_yoloe", True))

            for list_arg in ("sam31_text_track_labels", "sam31_structure_prompt_labels"):
                value = kwargs.get(list_arg)
                if isinstance(value, str):
                    kwargs[list_arg] = [s.strip() for s in value.split(",") if s.strip()]

            if tracker_backend in {"sam2", "sam2.1", "edgetam", "edge_tam"}:
                if tracker_backend in {"edgetam", "edge_tam"}:
                    tracker_label = "edgetam"
                    tracker_display = "EdgeTAM"
                    _assert_sam2_package_source("EdgeTAM", tracker_display)
                    tracker_checkpoint_path = _resolve_repo_relative_path(
                        edgetam_checkpoint or DEFAULT_EDGETAM_CHECKPOINT
                    )
                    tracker_model_cfg = edgetam_model_cfg or DEFAULT_EDGETAM_MODEL_CFG
                else:
                    tracker_label = "sam2"
                    tracker_display = "SAM2.1"
                    _assert_sam2_package_source("Grounded-SAM-2", tracker_display)
                    tracker_checkpoint_path = _resolve_repo_relative_path(
                        sam2_checkpoint or DEFAULT_SAM2_CHECKPOINT
                    )
                    tracker_model_cfg = sam2_model_cfg or DEFAULT_SAM2_MODEL_CFG

                if not os.path.exists(tracker_checkpoint_path):
                    raise FileNotFoundError(
                        f"{tracker_display} checkpoint not found: {tracker_checkpoint_path}"
                    )
                from sam2.build_sam import build_sam2_video_predictor

                video_predictor = build_sam2_video_predictor(
                    tracker_model_cfg,
                    tracker_checkpoint_path,
                    device=device,
                )
                if hasattr(video_predictor, "to"):
                    video_predictor.to(device)
                return cls(
                    video_predictor=video_predictor,
                    detector=YOLOEDetector(model_path=yoloe_model, device=device)
                    if sam3_cutie_use_yoloe
                    else _NoOpDetector(),
                    sam_backend=f"sam3_{tracker_label}",
                    device=device,
                    thing_prompts=thing_prompts or THING_PROMPTS,
                    stuff_prompts=stuff_prompts or STUFF_PROMPTS,
                    tracker_backend=tracker_label,
                    sam3_image_processor=sam3_image_processor,
                    sam3_cutie_sam_confidence_threshold=sam3_conf_threshold,
                    sam3_cutie_detection_frame_count=int(
                        kwargs.pop("sam3_cutie_detection_frame_count", 6)
                    ),
                    sam3_cutie_max_prompt_dets_per_label=int(
                        kwargs.pop("sam3_cutie_max_prompt_dets_per_label", 4)
                    ),
                    sam3_cutie_use_yoloe=sam3_cutie_use_yoloe,
                    **kwargs,
                )

            InferenceCore, get_default_model, GlobalHydra = _import_cutie_components()
            # Cutie uses Hydra's global singleton in get_default_model(); clear
            # it so repeated frontend construction in tests or notebooks is safe.
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            cutie_model = get_default_model()

            return cls(
                video_predictor=None,
                detector=YOLOEDetector(model_path=yoloe_model, device=device)
                if sam3_cutie_use_yoloe
                else _NoOpDetector(),
                sam_backend="sam3_cutie",
                device=device,
                thing_prompts=thing_prompts or THING_PROMPTS,
                stuff_prompts=stuff_prompts or STUFF_PROMPTS,
                tracker_backend="cutie",
                sam3_image_processor=sam3_image_processor,
                cutie_model=cutie_model,
                cutie_inference_core_cls=InferenceCore,
                cutie_max_internal_size=int(kwargs.pop("cutie_max_internal_size", 480)),
                sam3_cutie_sam_confidence_threshold=sam3_conf_threshold,
                sam3_cutie_detection_frame_count=int(
                    kwargs.pop("sam3_cutie_detection_frame_count", 6)
                ),
                sam3_cutie_max_prompt_dets_per_label=int(
                    kwargs.pop("sam3_cutie_max_prompt_dets_per_label", 4)
                ),
                sam3_cutie_use_yoloe=sam3_cutie_use_yoloe,
                **kwargs,
            )

        if tracker_backend not in {"efficientsam3", "efficient"}:
            raise ValueError(f"Unknown tracker_backend: {tracker_backend}")

        build_efficientsam3_video_model, Sam3VideoPredictor = _import_efficientsam3_components()
        checkpoint_path = _resolve_efficientsam3_checkpoint(
            efficientsam3_checkpoint,
            repo_id=efficientsam3_repo_id,
            filename=efficientsam3_filename,
            local_dir=efficientsam3_cache_dir,
        )

        model = build_efficientsam3_video_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            strict_state_dict_loading=False,
            backbone_type=efficientsam3_backbone_type,
            model_name=efficientsam3_model_name,
            text_encoder_type=efficientsam3_text_encoder_type,
            text_encoder_context_length=efficientsam3_text_context_length,
            device=device,
        )

        video_predictor = Sam3VideoPredictor(
            async_loading_frames=False,
            model_builder=lambda **_: model,
        )
        efficient_tracker = model.tracker
        efficient_tracker.backbone = model.detector.backbone
        detector = YOLOEDetector(model_path=yoloe_model, device=device)

        for list_arg in ("sam31_text_track_labels", "sam31_structure_prompt_labels"):
            value = kwargs.get(list_arg)
            if isinstance(value, str):
                kwargs[list_arg] = [s.strip() for s in value.split(",") if s.strip()]
        # The default EfficientSAM3 checkpoint used here does not expose reliable
        # text grounding: text-only prompts return empty masks on Taylor. Keep
        # semantic discovery in YOLOE, then use EfficientSAM3 object prompts for
        # temporal propagation.
        kwargs.setdefault("sam31_text_track_labels", [])
        kwargs.setdefault("sam31_max_static_objects", 0)
        kwargs.setdefault("sam31_max_structure_objects", 1)

        return cls(
            video_predictor=video_predictor,
            detector=detector,
            sam_backend="efficientsam3",
            device=device,
            thing_prompts=thing_prompts or THING_PROMPTS,
            stuff_prompts=stuff_prompts or STUFF_PROMPTS,
            efficient_model=model,
            efficient_tracker=efficient_tracker,
            **kwargs,
        )

    def _select_multiframe_things(
        self,
        frame_resource: Any,
        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, Any]], Dict[int, Dict]]:
        return self._select_multiframe_things_sam3(
            frame_resource,
            thing_dets_by_frame,
            discovery_indices,
            T,
            H,
            W,
            area,
        )

    @torch.inference_mode()
    def _propagate_things(
        self,
        frame_resource: Any,
        thing_dets: List[Dict],
        T: int,
        H: int,
        W: int,
    ) -> Tuple[Dict[int, Dict[int, Any]], Dict[int, Dict]]:
        dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for det in thing_dets:
            dets_by_frame.setdefault(int(det.get("frame_idx", 0)), []).append(det)
        _selected, segments, oid_map = self._select_multiframe_things_sam3(
            frame_resource,
            dets_by_frame,
            sorted(dets_by_frame.keys()),
            T,
            H,
            W,
            H * W,
        )
        return segments, oid_map

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
            not self.sam31_structure_prompt_labels
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
                    session_id, dets = self._query_sam31_prompt_frame_detections(
                        frame_resource,
                        session_id,
                        canonical_label,
                        int(frame_idx),
                        H,
                        W,
                        area,
                    )
                    if dets:
                        detections_by_frame.setdefault(int(frame_idx), []).extend(dets)
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return {
            int(frame_idx): self._nms(dets)
            for frame_idx, dets in detections_by_frame.items()
            if dets
        }

    def _select_prompt_frames(
        self,
        discovery_indices: List[int],
        T: int,
        count: int,
    ) -> List[int]:
        if count <= 0 or not discovery_indices:
            return []
        available = sorted({int(idx) for idx in discovery_indices if 0 <= int(idx) < T})
        if len(available) <= count:
            return available
        if count == 1:
            return [available[len(available) // 2]]
        return sorted({
            available[int(round(float(pos)))]
            for pos in np.linspace(0, len(available) - 1, count)
        })

    def _detect_text_track_prompt_dets(
        self,
        frame_resource: Any,
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        labels = [
            str(label).strip().lower()
            for label in self.sam31_text_track_labels
            if str(label).strip() and str(label).strip().lower() != "all"
        ]
        if not labels:
            return {}

        prompt_frames = self._select_prompt_frames(
            discovery_indices,
            T,
            max(1, int(self.sam31_person_refresh_prompt_frames)),
        )
        if not prompt_frames:
            return {}

        detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        session_id = self._start_sam31_session(frame_resource)
        try:
            for frame_idx in prompt_frames:
                for label in labels:
                    canonical_label = canonicalize_label(label)
                    sem_group = label_to_group(canonical_label)
                    if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR or _is_stuff(sem_group):
                        continue
                    session_id, dets = self._query_sam31_prompt_frame_detections(
                        frame_resource,
                        session_id,
                        canonical_label,
                        int(frame_idx),
                        H,
                        W,
                        area,
                    )
                    for det in dets:
                        det["detector_source"] = "efficientsam3_text_prompt_frame"
                        det["is_prompt_only"] = True
                    if dets:
                        detections_by_frame.setdefault(int(frame_idx), []).extend(dets)
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return {
            int(frame_idx): self._nms(dets)
            for frame_idx, dets in detections_by_frame.items()
            if dets
        }

    def _select_structure_text_tracks_efficientsam3(
        self,
        frame_resource: Any,
        structure_dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[int, np.ndarray]], Dict[int, Dict]]:
        if self.sam31_max_structure_objects <= 0:
            return [], {}, {}

        labels = {
            canonicalize_label(label)
            for label in self.sam31_structure_prompt_labels
            if str(label).strip()
        }
        for dets in structure_dets_by_frame.values():
            labels.update(canonicalize_label(str(det.get("label", ""))) for det in dets)
        labels = {
            label
            for label in labels
            if label and label_to_group(label) == SEMANTIC_GROUP_STRUCTURE_ANCHOR
        }
        if not labels:
            return [], {}, {}

        prompt_frames = self._select_sam31_structure_prompt_frames(discovery_indices, T)
        if not prompt_frames:
            return [], {}, {}

        label_to_dets: Dict[str, List[Dict[str, Any]]] = {}
        for dets in structure_dets_by_frame.values():
            for det in dets:
                label_to_dets.setdefault(
                    canonicalize_label(str(det.get("label", ""))),
                    [],
                ).append(det)

        selected: List[Dict[str, Any]] = []
        structure_segments: Dict[int, Dict[int, np.ndarray]] = {}
        structure_oid_map: Dict[int, Dict] = {}
        next_oid = 1

        session_id = self._start_sam31_session(frame_resource)
        try:
            for label in sorted(labels):
                if len(selected) >= self.sam31_max_structure_objects:
                    break
                dets_for_label = label_to_dets.get(label, [])
                if dets_for_label:
                    prompt_frame = int(
                        sorted(
                            dets_for_label,
                            key=lambda d: (
                                -int(bool(d.get("is_seed_track", False))),
                                -float(d.get("area_ratio", 0.0)),
                                -float(d.get("confidence", 0.0)),
                                int(d.get("frame_idx", 0)),
                            ),
                        )[0].get("frame_idx", prompt_frames[0])
                    )
                else:
                    prompt_frame = int(prompt_frames[0])

                session_id, obj_segments, obj_scores = self._query_sam31_label_tracks(
                    frame_resource,
                    session_id,
                    label,
                    prompt_frame,
                    log_prefix="efficient structure text ",
                )
                if not obj_segments:
                    continue

                candidates: List[Tuple[float, int, Dict[int, np.ndarray], Optional[Dict[str, Any]]]] = []
                for obj_id, frame_to_mask_raw in obj_segments.items():
                    frame_to_mask = {
                        int(t): np.asarray(mask).astype(np.uint8)
                        for t, mask in frame_to_mask_raw.items()
                        if mask is not None and np.asarray(mask).astype(bool).any()
                    }
                    if len(frame_to_mask) < min(max(3, self.discovery_frame_stride), max(int(T), 1)):
                        continue
                    mask_areas = [float(mask.astype(bool).sum()) for mask in frame_to_mask.values()]
                    mean_area_ratio = float(sum(mask_areas) / max(len(mask_areas), 1)) / max(float(area), 1.0)
                    if mean_area_ratio < max(self.min_mask_area_ratio, 0.004):
                        continue
                    if mean_area_ratio > min(self.max_mask_area_ratio, 0.92):
                        continue

                    best_det = None
                    best_match = 0.0
                    for det in dets_for_label:
                        match = self._match_detection_to_object_track(
                            det=det,
                            obj_segments={int(obj_id): frame_to_mask},
                        )
                        if match is None:
                            continue
                        _matched_obj, score = match
                        if float(score) > best_match:
                            best_match = float(score)
                            best_det = det

                    score = float(obj_scores.get(int(obj_id), 0.0))
                    rank_score = score + 0.25 * mean_area_ratio + 0.15 * min(len(frame_to_mask) / max(T, 1), 1.0)
                    if best_det is not None:
                        rank_score += 0.25 * best_match
                    if score < 0.12 and best_det is None:
                        continue
                    candidates.append((rank_score, int(obj_id), frame_to_mask, best_det))

                candidates.sort(key=lambda item: item[0], reverse=True)
                for _rank_score, obj_id, frame_to_mask, best_det in candidates:
                    if len(selected) >= self.sam31_max_structure_objects:
                        break

                    duplicate_existing = False
                    for existing_oid in structure_oid_map:
                        existing_segments = {
                            int(t): frame_objs[int(existing_oid)]
                            for t, frame_objs in structure_segments.items()
                            if int(existing_oid) in frame_objs
                        }
                        similarity = self._compute_sam31_prompt_track_similarity(
                            existing_segments,
                            frame_to_mask,
                        )
                        if similarity >= 0.72:
                            duplicate_existing = True
                            break
                    if duplicate_existing:
                        continue

                    segment_frames = sorted(frame_to_mask.keys())
                    ref_frame = int(prompt_frame) if int(prompt_frame) in frame_to_mask else int(segment_frames[0])
                    ref_mask = frame_to_mask[ref_frame].astype(bool)
                    if best_det is None:
                        box = self._mask_to_box_xyxy(ref_mask)
                        det_copy: Dict[str, Any] = {
                            "mask": ref_mask.astype(np.uint8),
                            "box": box.astype(np.float32),
                            "confidence": float(obj_scores.get(int(obj_id), 0.0)),
                            "label": label,
                            "raw_label": label,
                            "sem_group": SEMANTIC_GROUP_STRUCTURE_ANCHOR,
                            "area_ratio": float(ref_mask.sum()) / max(area, 1),
                            "frame_idx": ref_frame,
                            "detector_source": "efficientsam3_structure_text_track",
                            "is_prompt_only": True,
                        }
                    else:
                        det_copy = dict(best_det)
                        det_copy["detector_source"] = "efficientsam3_structure_text_track"
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
                        structure_segments.setdefault(int(t), {})[oid] = mask.astype(np.uint8)
                    break
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )

        return selected, structure_segments, structure_oid_map

    def _xyxy_to_efficientsam3_rel_box(
        self,
        det: Dict[str, Any],
        H: int,
        W: int,
    ) -> Optional[np.ndarray]:
        box = np.asarray(det.get("box", np.zeros(4)), dtype=np.float32).reshape(-1)
        if box.size < 4 or float(box[2] - box[0]) <= 1.0 or float(box[3] - box[1]) <= 1.0:
            mask = np.asarray(det.get("mask", np.zeros((H, W), dtype=np.uint8))).astype(bool)
            if mask.any():
                box = self._mask_to_box_xyxy(mask)
            else:
                return None

        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        label = str(det.get("label", "")).lower()
        if _labels_compatible(label, "person") or _labels_compatible(label, "people"):
            # Use a deliberately loose box for people. YOLOE masks sometimes
            # cover only the torso, and mask prompts can make EfficientSAM3
            # faithfully propagate that incomplete shape.
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            x1 -= 0.08 * bw
            x2 += 0.08 * bw
            y1 -= 0.08 * bh
            y2 += 0.22 * bh

        x1 = float(np.clip(x1, 0.0, max(W - 1, 0)))
        y1 = float(np.clip(y1, 0.0, max(H - 1, 0)))
        x2 = float(np.clip(max(x2, x1 + 1.0), 0.0, max(W - 1, 0)))
        y2 = float(np.clip(max(y2, y1 + 1.0), 0.0, max(H - 1, 0)))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1 / max(W, 1), y1 / max(H, 1), x2 / max(W, 1), y2 / max(H, 1)], dtype=np.float32)

    def _mask_tensor_to_numpy(self, mask: Any) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().float().cpu().numpy()
        else:
            arr = np.asarray(mask)
        arr = np.squeeze(arr)
        if arr.ndim < 2:
            return None
        if arr.ndim > 2:
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]
        return (arr > 0.0).astype(np.uint8)

    def _score_to_confidence(self, scores: Any, idx: int) -> float:
        if scores is None:
            return 0.0
        try:
            value = scores[idx]
        except Exception:
            return 0.0
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0.0
            value_f = float(value.detach().reshape(-1)[0].cpu().item())
            return float(torch.sigmoid(torch.tensor(value_f)).item())
        try:
            value_f = float(np.asarray(value).reshape(-1)[0])
        except Exception:
            return 0.0
        return float(1.0 / (1.0 + np.exp(-value_f)))

    def _record_tracker_outputs(
        self,
        frame_idx: int,
        out_obj_ids: Any,
        video_res_masks: Any,
        obj_segments: Dict[int, Dict[int, np.ndarray]],
        obj_scores: Dict[int, float],
        tracker_scores: Any = None,
    ) -> None:
        if out_obj_ids is None or video_res_masks is None:
            return
        for idx, out_obj_id in enumerate(list(out_obj_ids)):
            obj_id = int(out_obj_id)
            if idx >= len(video_res_masks):
                continue
            mask_np = self._mask_tensor_to_numpy(video_res_masks[idx])
            if mask_np is None or not mask_np.astype(bool).any():
                continue
            obj_segments.setdefault(obj_id, {})[int(frame_idx)] = mask_np.astype(np.uint8)
            score = self._score_to_confidence(tracker_scores, idx)
            if score > 0.0:
                obj_scores[obj_id] = max(float(obj_scores.get(obj_id, 0.0)), float(score))

    def _det_to_prompt_mask_tensor(
        self,
        det: Dict[str, Any],
        H: int,
        W: int,
    ) -> Optional[torch.Tensor]:
        mask = det.get("mask")
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().float().cpu().numpy()
        else:
            arr = np.asarray(mask)
        arr = np.squeeze(arr)
        if arr.ndim < 2:
            return None
        if arr.ndim > 2:
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]
        if arr.shape[0] != H or arr.shape[1] != W:
            arr = cv2.resize(arr.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mask_bool = arr.astype(np.float32) > 0.5
        if int(mask_bool.sum()) < 8:
            return None
        return torch.from_numpy(mask_bool.astype(np.float32))

    def _person_prompt_points_from_detector_mask(
        self,
        det: Dict[str, Any],
        H: int,
        W: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        label = str(det.get("label", "")).lower()
        if not (_labels_compatible(label, "person") or _labels_compatible(label, "people")):
            return None, None

        mask = det.get("mask")
        if mask is None:
            return None, None
        arr = np.asarray(mask)
        arr = np.squeeze(arr)
        if arr.ndim < 2:
            return None, None
        if arr.ndim > 2:
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]
        if arr.shape[0] != H or arr.shape[1] != W:
            arr = cv2.resize(arr.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mask_bool = arr.astype(np.float32) > 0.5
        ys, xs = np.nonzero(mask_bool)
        if xs.size < 24:
            return None, None

        def _median_point(indices: np.ndarray) -> Optional[Tuple[float, float]]:
            if indices.size < 8:
                return None
            return float(np.median(xs[indices])), float(np.median(ys[indices]))

        ordered = np.argsort(ys)
        thirds = np.array_split(ordered, 3)
        candidates: List[Tuple[float, float]] = []
        center = _median_point(np.arange(xs.size))
        if center is not None:
            candidates.append(center)
        # A few positive clicks make the loose person box less likely to grab
        # railings/background while still letting EfficientSAM3 complete limbs.
        for part in (thirds[0], thirds[-1], thirds[1]):
            point = _median_point(part)
            if point is not None:
                candidates.append(point)

        point_labels: List[int] = []
        points: List[Tuple[float, float]] = []
        min_dist = max(10.0, 0.035 * float(max(H, W)))
        for x, y in candidates:
            if any((x - px) ** 2 + (y - py) ** 2 < min_dist ** 2 for px, py in points):
                continue
            points.append((
                float(np.clip(x / max(W, 1), 0.0, 1.0)),
                float(np.clip(y / max(H, 1), 0.0, 1.0)),
            ))
            point_labels.append(1)
            if len(points) >= 3:
                break

        if not points:
            return None, None
        box = np.asarray(det.get("box", self._mask_to_box_xyxy(mask_bool)), dtype=np.float32).reshape(-1)
        if box.size >= 4:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            x1 = float(np.clip(x1 - 0.05 * bw, 0.0, max(W - 1, 0)))
            x2 = float(np.clip(x2 + 0.05 * bw, 0.0, max(W - 1, 0)))
            y1 = float(np.clip(y1 - 0.05 * bh, 0.0, max(H - 1, 0)))
            y2 = float(np.clip(y2 + 0.14 * bh, 0.0, max(H - 1, 0)))
            inverse = (~mask_bool).astype(np.uint8)
            dist_to_person = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
            neg_candidates = [
                (x1 + 0.12 * bw, y1 + 0.12 * bh),
                (x2 - 0.12 * bw, y1 + 0.12 * bh),
                (x1 + 0.12 * bw, y2 - 0.12 * bh),
                (x2 - 0.12 * bw, y2 - 0.12 * bh),
            ]
            for x, y in neg_candidates:
                px = int(np.clip(round(x), 0, max(W - 1, 0)))
                py = int(np.clip(round(y), 0, max(H - 1, 0)))
                if mask_bool[py, px] or float(dist_to_person[py, px]) < max(8.0, 0.018 * max(H, W)):
                    continue
                if any((px / max(W, 1) - qx) ** 2 + (py / max(H, 1) - qy) ** 2 < 0.0025 for qx, qy in points):
                    continue
                points.append((float(px / max(W, 1)), float(py / max(H, 1))))
                point_labels.append(0)
                if len(point_labels) >= 5:
                    break

        point_tensor = torch.tensor(points, dtype=torch.float32)
        label_tensor = torch.tensor(point_labels, dtype=torch.int32)
        return point_tensor, label_tensor

    def _cluster_sam31_track_candidates(
        self,
        dets_by_frame: Dict[int, List[Dict[str, Any]]],
        discovery_indices: List[int],
        area: int,
    ) -> List[Dict[str, Any]]:
        """Cluster detector hits, keeping support detections for multi-prompt tracking.

        The SAM3.1 frontend can often survive from one prompt per chunk. The
        EfficientSAM3 student tracker drifts more easily, so each selected
        object needs a small set of detector-supported anchors from the same
        chunk. This mirrors the base clustering logic but preserves the
        per-frame detections instead of returning only one representative.
        """
        clusters: List[Dict[str, Any]] = []
        for frame_idx in discovery_indices:
            candidates = list(dets_by_frame.get(int(frame_idx), []))
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
                det_label = canonicalize_label(str(det.get("label", "")))
                det_is_person = (
                    det_group == SEMANTIC_GROUP_MOVABLE_THING
                    and (
                        _labels_compatible(det_label, "person")
                        or _labels_compatible(det_label, "people")
                    )
                )
                det_mask = det["mask"].astype(bool)
                det_box = np.asarray(det["box"], dtype=np.float32)
                det_area = max(float(det_mask.sum()), 1.0)

                for cluster in clusters:
                    if int(cluster.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) != det_group:
                        continue
                    if not _labels_compatible(str(cluster.get("label", "")), str(det.get("label", ""))):
                        continue

                    # Compare moving people against several anchors in the
                    # cluster, not just the latest one. Taylor has long dark
                    # doorway intervals where the latest detection can be a
                    # poor partial shard even though an earlier full-body
                    # anchor is still the right identity.
                    ref_dets = cluster["detections"] if det_is_person else cluster["detections"][-1:]
                    for ref_det in ref_dets[-8:]:
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
                        elif det_is_person:
                            frame_h, frame_w = det_mask.shape[:2]
                            det_cx = 0.5 * float(det_box[0] + det_box[2]) / max(float(frame_w), 1.0)
                            det_cy = 0.5 * float(det_box[1] + det_box[3]) / max(float(frame_h), 1.0)
                            ref_cx = 0.5 * float(ref_box[0] + ref_box[2]) / max(float(frame_w), 1.0)
                            ref_cy = 0.5 * float(ref_box[1] + ref_box[3]) / max(float(frame_h), 1.0)
                            center_norm = float(np.hypot(det_cx - ref_cx, det_cy - ref_cy))
                            center_score = max(0.0, 1.0 - center_norm / 0.45)
                            if frame_gap > 0:
                                # Keep person clusters purer than generic THINGs.
                                # Multi-person Taylor shots often have different
                                # people passing through similar screen positions;
                                # overly loose clustering then prompts SAM2/EdgeTAM
                                # with several identities under one object id.
                                far_without_overlap = (
                                    center_norm > 0.34
                                    and box_iou < 0.06
                                    and mask_iou < 0.03
                                )
                                weak_long_match = (
                                    frame_gap > max(self.discovery_frame_stride + 1, 5)
                                    and center_norm > 0.26
                                    and box_iou < 0.10
                                    and mask_iou < 0.05
                                    and area_sim < 0.45
                                )
                                if far_without_overlap or weak_long_match:
                                    continue
                            score = (
                                0.34 * box_iou
                                + 0.18 * mask_iou
                                + 0.26 * area_sim
                                + 0.22 * center_score
                                + gap_bonus
                            )
                            threshold = 0.24
                            if frame_gap == 0:
                                threshold = 0.42
                            elif area_sim >= 0.35 and center_norm <= 0.24:
                                score += 0.12
                        else:
                            score = 0.15 * mask_iou + 0.55 * box_iou + 0.20 * area_sim + gap_bonus
                            threshold = 0.16

                        if frame_gap == 0 and not det_is_person:
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
            prompt_det["support_detections"] = [dict(d) for d in dets]
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

    def _select_prompt_support_detections(
        self,
        det: Dict[str, Any],
        *,
        T: int,
    ) -> List[Dict[str, Any]]:
        support_raw = det.get("support_detections")
        support = [dict(d) for d in support_raw] if isinstance(support_raw, list) else []
        support.append(dict(det))

        frame_best: Dict[int, Dict[str, Any]] = {}
        label = str(det.get("label", ""))
        sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
        for item in support:
            frame_idx = int(np.clip(int(item.get("frame_idx", 0)), 0, max(int(T) - 1, 0)))
            if int(item.get("sem_group", sem_group)) != sem_group:
                continue
            if not _labels_compatible(label, str(item.get("label", ""))):
                continue
            item["frame_idx"] = frame_idx
            prev = frame_best.get(frame_idx)
            if prev is None or float(item.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                frame_best[frame_idx] = item

        if not frame_best:
            return [dict(det)]

        ordered = [frame_best[t] for t in sorted(frame_best)]
        is_person = _labels_compatible(label, "person") or _labels_compatible(label, "people")
        if is_person:
            max_prompts = 6 if self.tracker_backend in {"sam2", "edgetam"} else 4
        elif sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
            max_prompts = 2
        else:
            max_prompts = 3
        if len(ordered) <= max_prompts:
            return ordered

        # Keep temporal endpoints and the strongest intermediate anchors. This
        # is more stable than "top confidence only" when the camera/subject
        # motion is large, while staying cheap.
        selected_by_frame: Dict[int, Dict[str, Any]] = {
            int(ordered[0]["frame_idx"]): ordered[0],
            int(ordered[-1]["frame_idx"]): ordered[-1],
        }
        remaining = sorted(
            ordered[1:-1],
            key=lambda d: (
                -float(d.get("confidence", 0.0)),
                -float(d.get("area_ratio", 0.0)),
                int(d.get("frame_idx", 0)),
            ),
        )
        for item in remaining:
            if len(selected_by_frame) >= max_prompts:
                break
            selected_by_frame[int(item["frame_idx"])] = item
        return [selected_by_frame[t] for t in sorted(selected_by_frame)]

    def _add_efficientsam3_object_prompt(
        self,
        inference_state: Dict[str, Any],
        det: Dict[str, Any],
        obj_id: int,
        prompt_frame: int,
        H: int,
        W: int,
    ) -> Tuple[int, Any, Any, Any]:
        label = str(det.get("label", "")).lower()
        prefer_box_prompt = _labels_compatible(label, "person") or _labels_compatible(label, "people")
        if prefer_box_prompt:
            rel_box = self._xyxy_to_efficientsam3_rel_box(det, H=H, W=W)
            if rel_box is not None:
                try:
                    return self.efficient_tracker.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=int(prompt_frame),
                        obj_id=int(obj_id),
                        box=rel_box,
                        rel_coordinates=True,
                        normalize_coords=True,
                    )
                except Exception as exc:
                    print(
                        f"[efficientsam3] warning: box prompt failed for "
                        f"person obj_id={obj_id} frame={prompt_frame}: {exc!r}; falling back to mask prompt"
                    )

        prompt_mask = self._det_to_prompt_mask_tensor(det, H=H, W=W)
        if prompt_mask is not None:
            try:
                return self.efficient_tracker.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=int(prompt_frame),
                    obj_id=int(obj_id),
                    mask=prompt_mask,
                )
            except Exception as exc:
                print(
                    f"[efficientsam3] warning: mask prompt failed for "
                    f"obj_id={obj_id} frame={prompt_frame}: {exc!r}; falling back to box prompt"
                )

        rel_box = self._xyxy_to_efficientsam3_rel_box(det, H=H, W=W)
        if rel_box is None:
            raise ValueError("detection has neither a usable mask nor a usable box")
        return self.efficient_tracker.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(prompt_frame),
            obj_id=int(obj_id),
            box=rel_box,
            rel_coordinates=True,
            normalize_coords=True,
        )

    @torch.inference_mode()
    def _track_objects_once_efficientsam3_tracker(
        self,
        frame_resource: Any,
        detections: List[Dict[str, Any]],
        T: int,
        H: int,
        W: int,
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, float], Dict[int, Dict[str, Any]]]:
        if not detections:
            return {}, {}, {}
        if self.efficient_tracker is None:
            raise RuntimeError("EfficientSAM3 tracker is not initialized.")

        inference_state = self.efficient_tracker.init_state(
            video_path=frame_resource,
            offload_video_to_cpu=bool(self.sam31_offload_video_to_cpu),
            offload_state_to_cpu=bool(self.sam31_offload_state_to_cpu),
            async_loading_frames=False,
        )

        obj_to_det: Dict[int, Dict[str, Any]] = {}
        obj_segments: Dict[int, Dict[int, np.ndarray]] = {}
        obj_scores: Dict[int, float] = {}
        prompt_frames: List[int] = []

        try:
            for obj_id, det in enumerate(detections, start=1):
                prompt_dets = self._select_prompt_support_detections(det, T=T)
                prompt_frame = int(np.clip(int(prompt_dets[0].get("frame_idx", 0)), 0, max(T - 1, 0)))
                obj_to_det[int(obj_id)] = det
                obj_scores[int(obj_id)] = float(det.get("confidence", 0.0))
                added_any = False
                for prompt_det in prompt_dets:
                    prompt_frame = int(np.clip(int(prompt_det.get("frame_idx", 0)), 0, max(T - 1, 0)))
                    try:
                        out_frame_idx, out_obj_ids, _low_res_masks, video_res_masks = (
                            self._add_efficientsam3_object_prompt(
                                inference_state,
                                prompt_det,
                                obj_id=int(obj_id),
                                prompt_frame=prompt_frame,
                                H=H,
                                W=W,
                            )
                        )
                    except Exception as exc:
                        print(
                            f"[efficientsam3] warning: skipping unusable prompt "
                            f"obj_id={obj_id} frame={prompt_frame}: {exc!r}"
                        )
                        continue
                    added_any = True
                    prompt_frames.append(prompt_frame)
                    self._record_tracker_outputs(
                        int(out_frame_idx),
                        out_obj_ids,
                        video_res_masks,
                        obj_segments,
                        obj_scores,
                    )
                if not added_any:
                    print(f"[efficientsam3] warning: dropping object with no usable prompts obj_id={obj_id}")
                    obj_to_det.pop(int(obj_id), None)
                    obj_scores.pop(int(obj_id), None)
                    continue

            if not obj_to_det:
                return {}, {}, {}

            min_prompt_frame = min(prompt_frames) if prompt_frames else 0
            max_prompt_frame = max(prompt_frames) if prompt_frames else 0

            def _consume(start_frame: int, reverse: bool, max_frames: int) -> None:
                for (
                    out_frame_idx,
                    out_obj_ids,
                    _low_res_masks,
                    video_res_masks,
                    tracker_scores,
                ) in self.efficient_tracker.propagate_in_video(
                    inference_state,
                    start_frame_idx=int(start_frame),
                    max_frame_num_to_track=max(int(max_frames), 1),
                    reverse=bool(reverse),
                    tqdm_disable=True,
                    propagate_preflight=True,
                ):
                    self._record_tracker_outputs(
                        int(out_frame_idx),
                        out_obj_ids,
                        video_res_masks,
                        obj_segments,
                        obj_scores,
                        tracker_scores=tracker_scores,
                    )

            _consume(
                start_frame=int(min_prompt_frame),
                reverse=False,
                max_frames=max(int(T) - int(min_prompt_frame), 1),
            )
            if int(max_prompt_frame) > 0:
                _consume(
                    start_frame=int(max_prompt_frame),
                    reverse=True,
                    max_frames=int(max_prompt_frame) + 1,
                )
        finally:
            try:
                if hasattr(self.efficient_tracker, "reset_state"):
                    self.efficient_tracker.reset_state(inference_state)
                elif hasattr(self.efficient_tracker, "clear_all_points_in_video"):
                    self.efficient_tracker.clear_all_points_in_video(inference_state)
            except Exception as exc:
                print(f"[efficientsam3] warning: failed to reset tracker state: {exc!r}")
            try:
                inference_state.clear()
            except Exception:
                pass
            del inference_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (
            {obj_id: frame_to_mask for obj_id, frame_to_mask in obj_segments.items() if frame_to_mask},
            obj_scores,
            obj_to_det,
        )

    def _track_all_objects_efficientsam3_tracker(
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
        thing_segments: Dict[int, Dict[int, np.ndarray]] = {}
        thing_oid_map: Dict[int, Dict] = {}
        structure_segments: Dict[int, Dict[int, np.ndarray]] = {}
        structure_oid_map: Dict[int, Dict] = {}

        thing_candidates = self._filter_sam31_track_candidates(
            self._cluster_sam31_track_candidates(thing_dets_by_frame, discovery_indices, area)
        )
        structure_candidates = self._filter_sam31_track_candidates(
            self._cluster_sam31_track_candidates(structure_dets_by_frame, discovery_indices, area)
        )
        if not structure_candidates:
            # EfficientSAM3 text grounding is weak with the local checkpoint, so
            # structure anchors must come from YOLOE. Accept single-frame large
            # floor/wall/ceiling detections instead of falling back to static
            # one-frame stuff entries.
            structure_candidates = [
                det
                for det in self._cluster_sam31_track_candidates(
                    structure_dets_by_frame, discovery_indices, area,
                )
                if (
                    int(det.get("seed_count", 0)) > 0
                    or int(det.get("cluster_size", 1)) >= 1
                    and (
                        float(det.get("area_ratio", 0.0)) >= 0.025
                        or float(det.get("cluster_confidence", det.get("confidence", 0.0))) >= 0.24
                    )
                )
            ]

        thing_budget = max(int(self.max_thing_objects), 0)
        movable_budget = min(thing_budget, self.sam31_max_movable_objects)
        person_candidates = [
            det
            for det in thing_candidates
            if (
                int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) == SEMANTIC_GROUP_MOVABLE_THING
                and (
                    _labels_compatible(str(det.get("label", "")), "person")
                    or _labels_compatible(str(det.get("label", "")), "people")
                )
            )
        ]
        selected_movable = self._select_efficient_person_candidates(
            person_candidates,
            budget=movable_budget,
        )
        if not selected_movable:
            selected_movable = self._take_sam31_track_candidates(
                thing_candidates,
                budget=movable_budget,
                allowed_groups={SEMANTIC_GROUP_MOVABLE_THING},
            )
        else:
            remaining_movable_budget = max(0, movable_budget - len(selected_movable))
            strong_other_movable = [
                det
                for det in thing_candidates
                if (
                    int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) == SEMANTIC_GROUP_MOVABLE_THING
                    and not (
                        _labels_compatible(str(det.get("label", "")), "person")
                        or _labels_compatible(str(det.get("label", "")), "people")
                    )
                    and float(det.get("cluster_confidence", det.get("confidence", 0.0))) >= 0.65
                    and float(det.get("area_ratio", 0.0)) >= 0.008
                )
            ]
            selected_movable.extend(
                self._take_sam31_track_candidates(
                    strong_other_movable,
                    budget=remaining_movable_budget,
                    allowed_groups={SEMANTIC_GROUP_MOVABLE_THING},
                )
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
        selected_thing_dets = self._refresh_person_prompt_detections(
            selected_thing_dets,
            [
                det
                for dets in thing_dets_by_frame.values()
                for det in dets
            ],
        )
        all_candidates = selected_thing_dets + selected_structure_dets
        if not all_candidates:
            return selected_thing_dets, thing_segments, thing_oid_map, selected_structure_dets, structure_segments, structure_oid_map

        if self.tracker_backend in {"sam2", "edgetam"}:
            obj_segments, obj_scores, obj_to_det = self._track_objects_once_sam2_tracker(
                frame_resource=frame_resource,
                detections=all_candidates,
                T=T,
                H=H,
                W=W,
            )
        else:
            obj_segments, obj_scores, obj_to_det = self._track_objects_once_efficientsam3_tracker(
                frame_resource=frame_resource,
                detections=all_candidates,
                T=T,
                H=H,
                W=W,
            )

        support_dets = [
            det
            for dets in thing_dets_by_frame.values()
            for det in dets
            if int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) == SEMANTIC_GROUP_MOVABLE_THING
        ]
        all_thing_support_dets = [
            det
            for dets in thing_dets_by_frame.values()
            for det in dets
        ]
        all_structure_support_dets = [
            det
            for dets in structure_dets_by_frame.values()
            for det in dets
        ]
        thing_obj_ids = set(range(1, len(selected_thing_dets) + 1))
        next_thing_oid = 1
        next_structure_oid = 1
        selected_thing_final: List[Dict[str, Any]] = []
        selected_structure_final: List[Dict[str, Any]] = []

        for obj_id in sorted(obj_segments.keys()):
            det = dict(obj_to_det.get(int(obj_id), {}))
            if not det:
                continue
            frame_to_mask = {
                int(t): np.asarray(mask).astype(np.uint8)
                for t, mask in obj_segments[int(obj_id)].items()
                if mask is not None and np.asarray(mask).astype(bool).any()
            }
            if not frame_to_mask:
                continue
            label = str(det.get("label", ""))
            is_person_track = _labels_compatible(label, "person") or _labels_compatible(label, "people")
            person_track_support_dets: List[Dict[str, Any]] = []
            if _labels_compatible(label, "person") or _labels_compatible(label, "people"):
                matched_support = self._match_person_support_dets_to_track(frame_to_mask, support_dets)
                if matched_support:
                    frame_to_mask = self._repair_person_track_with_detector_support(frame_to_mask, matched_support)
                    person_track_support_dets.extend(dict(support) for support, _score in matched_support)
                own_support = det.get("support_detections")
                if isinstance(own_support, list):
                    person_track_support_dets.extend(
                        dict(support)
                        for support in own_support
                        if (
                            _labels_compatible(str(support.get("label", "")), "person")
                            or _labels_compatible(str(support.get("label", "")), "people")
                        )
                    )
            det["confidence"] = max(
                float(det.get("confidence", 0.0)),
                float(obj_scores.get(int(obj_id), 0.0)),
            )
            if int(obj_id) in thing_obj_ids:
                overwrite_support = (
                    person_track_support_dets if is_person_track else all_thing_support_dets
                )
                frame_to_mask = self._overwrite_track_with_detector_support(
                    frame_to_mask,
                    det,
                    overwrite_support,
                )
                if is_person_track and self.tracker_backend not in {"sam2", "edgetam"}:
                    frame_to_mask = self._clip_person_track_on_detector_contradictions(
                        frame_to_mask,
                        det,
                        all_thing_support_dets,
                    )
                if is_person_track and self.tracker_backend in {"sam2", "edgetam"}:
                    frame_to_mask = self._fill_person_track_gaps(
                        frame_to_mask,
                        H=H,
                        W=W,
                    )
                if not frame_to_mask:
                    continue
                oid = next_thing_oid
                next_thing_oid += 1
                thing_oid_map[oid] = det
                selected_thing_final.append(det)
                for t, mask in frame_to_mask.items():
                    thing_segments.setdefault(int(t), {})[oid] = mask.astype(np.uint8)
            else:
                frame_to_mask = self._overwrite_track_with_detector_support(
                    frame_to_mask,
                    det,
                    all_structure_support_dets,
                )
                mean_area_ratio = (
                    sum(float(np.asarray(mask).astype(bool).sum()) for mask in frame_to_mask.values())
                    / max(len(frame_to_mask), 1)
                    / max(float(area), 1.0)
                )
                prompt_area_ratio = float(det.get("area_ratio", 0.0))
                min_structure_area = max(0.012, min(0.040, 0.25 * max(prompt_area_ratio, 0.0)))
                if mean_area_ratio < min_structure_area:
                    continue
                oid = next_structure_oid
                next_structure_oid += 1
                structure_oid_map[oid] = det
                selected_structure_final.append(det)
                for t, mask in frame_to_mask.items():
                    structure_segments.setdefault(int(t), {})[oid] = mask.astype(np.uint8)

        removed_person_oids = self._prune_tiny_person_shadow_segments(
            thing_segments,
            thing_oid_map,
            T=T,
            area=area,
        )
        if removed_person_oids:
            selected_thing_final = [
                det for oid, det in thing_oid_map.items()
                if int(oid) not in removed_person_oids
            ]
        removed_aux_things, removed_aux_structures = self._prune_unreliable_auxiliary_segments(
            thing_segments,
            thing_oid_map,
            structure_segments,
            structure_oid_map,
            T=T,
        )
        if removed_aux_things:
            selected_thing_final = [
                det for oid, det in thing_oid_map.items()
                if int(oid) not in removed_aux_things
            ]
        if removed_aux_structures:
            selected_structure_final = [
                det for oid, det in structure_oid_map.items()
                if int(oid) not in removed_aux_structures
            ]

        return (
            selected_thing_final,
            thing_segments,
            thing_oid_map,
            selected_structure_final,
            structure_segments,
            structure_oid_map,
        )

    def _is_seed_track_candidate(self, det: Dict[str, Any]) -> bool:
        return (
            bool(det.get("is_seed_track", False))
            or int(det.get("seed_count", 0)) > 0
            or det.get("seed_global_track_idx", None) is not None
        )

    def _select_efficient_person_candidates(
        self,
        person_candidates: List[Dict[str, Any]],
        *,
        budget: int,
    ) -> List[Dict[str, Any]]:
        if budget <= 0 or not person_candidates:
            return []

        seeded = [dict(det) for det in person_candidates if self._is_seed_track_candidate(det)]
        fresh = [dict(det) for det in person_candidates if not self._is_seed_track_candidate(det)]

        seeded.sort(
            key=lambda d: (
                -int(d.get("seed_count", 0)),
                -int(d.get("cluster_size", 1)),
                -float(d.get("cluster_confidence", d.get("confidence", 0.0))),
                -float(d.get("area_ratio", 0.0)),
                int(d.get("frame_idx", 0)),
            )
        )
        fresh.sort(
            key=lambda d: (
                -float(d.get("cluster_confidence", d.get("confidence", 0.0))),
                -float(d.get("area_ratio", 0.0)),
                -int(d.get("cluster_size", 1)),
                int(d.get("frame_idx", 0)),
            )
        )

        selected: List[Dict[str, Any]] = []
        # Keep one continuity slot, but do not let old seed tracks consume all
        # person budget. Late-entering people must still get a prompt.
        if seeded:
            selected.append(seeded[0])

        for det in fresh:
            if len(selected) >= budget:
                break
            duplicate = False
            det_mask = np.asarray(det.get("mask")).astype(bool)
            for chosen in selected:
                if int(chosen.get("frame_idx", -1)) != int(det.get("frame_idx", -2)):
                    continue
                chosen_mask = np.asarray(chosen.get("mask")).astype(bool)
                if _mask_iou(det_mask, chosen_mask) >= 0.35:
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(det)

        for det in seeded[1:]:
            if len(selected) >= budget:
                break
            selected.append(det)

        return selected[:budget]

    def _refresh_person_prompt_detections(
        self,
        selected_dets: List[Dict[str, Any]],
        support_dets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        refreshed: List[Dict[str, Any]] = []
        used_support: List[Dict[str, Any]] = []
        person_support = [
            det
            for det in support_dets
            if (
                not bool(det.get("is_seed_track", False))
                and (
                    _labels_compatible(str(det.get("label", "")), "person")
                    or _labels_compatible(str(det.get("label", "")), "people")
                )
                and float(det.get("confidence", 0.0)) >= 0.45
            )
        ]

        for det in selected_dets:
            label = str(det.get("label", ""))
            if not (_labels_compatible(label, "person") or _labels_compatible(label, "people")):
                refreshed.append(det)
                continue
            has_seed = (
                bool(det.get("is_seed_track", False))
                or int(det.get("seed_count", 0)) > 0
                or det.get("seed_global_track_idx", None) is not None
            )
            if not has_seed:
                refreshed.append(det)
                continue

            support_frames = {int(v) for v in det.get("support_frames", [])}
            # Seed tracks arrive from the overlap region. If we only refresh
            # from those overlap frames, a bad carry-over prompt can keep
            # propagating a torso/guitar fragment even when YOLOE sees a full
            # person a few frames later in the same chunk. Keep overlap frames
            # as a preference, not a hard gate.
            candidates = list(person_support)
            if not candidates:
                refreshed.append(det)
                continue

            candidates.sort(
                key=lambda support: (
                    -(
                        float(support.get("confidence", 0.0))
                        + min(float(support.get("area_ratio", 0.0)) * 8.0, 1.0)
                        + (0.18 if int(support.get("frame_idx", 0)) in support_frames else 0.0)
                    ),
                    -float(support.get("confidence", 0.0)),
                    -float(support.get("area_ratio", 0.0)),
                    int(support.get("frame_idx", 0)),
                )
            )
            replacement = None
            for support in candidates:
                duplicate = False
                support_mask = np.asarray(support.get("mask")).astype(bool)
                for used in used_support:
                    if int(used.get("frame_idx", -1)) != int(support.get("frame_idx", -2)):
                        continue
                    if _mask_iou(support_mask, np.asarray(used.get("mask")).astype(bool)) >= 0.55:
                        duplicate = True
                        break
                if not duplicate:
                    replacement = dict(support)
                    break
            if replacement is None:
                refreshed.append(det)
                continue

            for key in (
                "seed_global_track_idx",
                "seed_count",
                "cluster_size",
                "support_frames",
                "support_detections",
            ):
                if key in det and key not in replacement:
                    replacement[key] = det[key]
            replacement_support = [dict(v) for v in det.get("support_detections", [])]
            replacement_support.append(dict(replacement))
            replacement["support_detections"] = replacement_support
            if det.get("seed_global_track_idx", None) is not None:
                replacement["seed_global_track_idx"] = det["seed_global_track_idx"]
            replacement["detector_source"] = "efficient_person_refreshed_prompt"
            refreshed.append(replacement)
            used_support.append(replacement)

        return refreshed

    def _mask_box_center_norm(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        box_a = self._mask_to_box_xyxy(mask_a.astype(bool))
        box_b = self._mask_to_box_xyxy(mask_b.astype(bool))
        aw = max(float(box_a[2] - box_a[0] + 1.0), 1.0)
        ah = max(float(box_a[3] - box_a[1] + 1.0), 1.0)
        bw = max(float(box_b[2] - box_b[0] + 1.0), 1.0)
        bh = max(float(box_b[3] - box_b[1] + 1.0), 1.0)
        acx = 0.5 * float(box_a[0] + box_a[2])
        acy = 0.5 * float(box_a[1] + box_a[3])
        bcx = 0.5 * float(box_b[0] + box_b[2])
        bcy = 0.5 * float(box_b[1] + box_b[3])
        dx = abs(acx - bcx) / max(0.5 * (aw + bw), 1.0)
        dy = abs(acy - bcy) / max(0.5 * (ah + bh), 1.0)
        return float(max(dx, dy))

    def _person_support_should_repair(
        self,
        current_mask: np.ndarray,
        support_mask: np.ndarray,
    ) -> bool:
        if super()._person_support_should_repair(current_mask, support_mask):
            return True

        current = np.asarray(current_mask).astype(bool)
        support = np.asarray(support_mask).astype(bool)
        current_area = float(current.sum())
        support_area = float(support.sum())
        if current_area <= 0.0 or support_area <= 0.0:
            return False

        current_box = self._mask_to_box_xyxy(current)
        support_box = self._mask_to_box_xyxy(support)
        current_w = max(float(current_box[2] - current_box[0] + 1.0), 1.0)
        current_h = max(float(current_box[3] - current_box[1] + 1.0), 1.0)
        support_w = max(float(support_box[2] - support_box[0] + 1.0), 1.0)
        support_h = max(float(support_box[3] - support_box[1] + 1.0), 1.0)
        current_cx = 0.5 * float(current_box[0] + current_box[2])
        current_cy = 0.5 * float(current_box[1] + current_box[3])
        support_cx = 0.5 * float(support_box[0] + support_box[2])
        support_cy = 0.5 * float(support_box[1] + support_box[3])
        center_dx = abs(current_cx - support_cx) / max(0.5 * (current_w + support_w), 1.0)
        center_dy = abs(current_cy - support_cy) / max(0.5 * (current_h + support_h), 1.0)
        area_gain = support_area / max(current_area, 1.0)
        height_gain = support_h / max(current_h, 1.0)
        width_gain = support_w / max(current_w, 1.0)

        # EfficientSAM3 often locks onto a torso/guitar-sized part for person.
        # Prefer the detector mask when it is clearly a larger person-shaped
        # support near the same object, even if the partial mask has low IoU.
        if center_dx <= 1.35 and center_dy <= 1.45:
            return area_gain >= 1.35 or height_gain >= 1.25 or width_gain >= 1.35
        if center_dx <= 1.70 and center_dy <= 1.75:
            return area_gain >= 2.10 and height_gain >= 1.20
        return False

    def _seed_person_has_detector_support(
        self,
        seed_det: Dict[str, Any],
        detector_dets: List[Dict[str, Any]],
    ) -> bool:
        label = str(seed_det.get("label", ""))
        if not (_labels_compatible(label, "person") or _labels_compatible(label, "people")):
            return True

        seed_mask = np.asarray(seed_det.get("mask")).astype(bool)
        if seed_mask.sum() <= 0:
            return False
        seed_box = np.asarray(seed_det.get("box", self._mask_to_box_xyxy(seed_mask)), dtype=np.float32)
        seed_area = max(float(seed_mask.sum()), 1.0)

        person_support = [
            det
            for det in detector_dets
            if (
                int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) == SEMANTIC_GROUP_MOVABLE_THING
                and (
                    _labels_compatible(str(det.get("label", "")), "person")
                    or _labels_compatible(str(det.get("label", "")), "people")
                )
                and float(det.get("confidence", 0.0)) >= 0.30
            )
        ]
        if not person_support:
            return False

        for det in person_support:
            support_mask = np.asarray(det.get("mask")).astype(bool)
            if support_mask.sum() <= 0:
                continue
            support_box = np.asarray(det.get("box", self._mask_to_box_xyxy(support_mask)), dtype=np.float32)
            support_area = max(float(support_mask.sum()), 1.0)
            mask_iou = _mask_iou(seed_mask, support_mask)
            box_iou = _box_iou_xyxy(seed_box, support_box)
            center_norm = self._mask_box_center_norm(seed_mask, support_mask)
            area_sim = min(seed_area, support_area) / max(seed_area, support_area, 1.0)
            if (
                mask_iou >= 0.04
                or box_iou >= 0.10
                or (center_norm <= 1.10 and area_sim >= 0.08)
            ):
                return True
        return False

    def _overwrite_track_with_detector_support(
        self,
        frame_to_mask: Dict[int, np.ndarray],
        track_det: Dict[str, Any],
        support_dets: List[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        if not frame_to_mask or not support_dets:
            return frame_to_mask

        label = str(track_det.get("label", ""))
        sem_group = int(track_det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
        repaired = {
            int(frame_idx): np.asarray(mask).astype(np.uint8)
            for frame_idx, mask in frame_to_mask.items()
        }
        support_by_frame: Dict[int, Tuple[np.ndarray, float, float]] = {}
        for det in support_dets:
            if int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) != sem_group:
                continue
            if not _labels_compatible(label, str(det.get("label", ""))):
                continue
            support_mask = np.asarray(det.get("mask")).astype(bool)
            if support_mask.sum() <= 0:
                continue
            frame_idx = int(det.get("frame_idx", 0))
            current = repaired.get(frame_idx)
            if current is None:
                continue
            current_bool = current.astype(bool)
            if current_bool.sum() <= 0:
                continue

            inter = float((current_bool & support_mask).sum())
            current_area = float(current_bool.sum())
            support_area = float(support_mask.sum())
            current_cover = inter / max(float(current_bool.sum()), 1.0)
            support_cover = inter / max(float(support_mask.sum()), 1.0)
            center_norm = self._mask_box_center_norm(current_bool, support_mask)
            is_person = _labels_compatible(label, "person") or _labels_compatible(label, "people")
            det_conf = float(det.get("confidence", 0.0))
            support_fuller_person = (
                is_person
                and support_area >= 1.18 * max(current_area, 1.0)
                and center_norm <= 1.55
            )
            close_enough = center_norm <= (1.35 if is_person else 0.85)
            overlaps_enough = current_cover >= 0.10 or support_cover >= 0.10
            if not (overlaps_enough or close_enough or support_fuller_person):
                continue

            if is_person:
                repair_with_union = self._person_support_should_repair(current_bool, support_mask)
                current_box = self._mask_to_box_xyxy(current_bool)
                support_box = self._mask_to_box_xyxy(support_mask)
                current_w = max(float(current_box[2] - current_box[0] + 1.0), 1.0)
                current_h = max(float(current_box[3] - current_box[1] + 1.0), 1.0)
                support_w = max(float(support_box[2] - support_box[0] + 1.0), 1.0)
                support_h = max(float(support_box[3] - support_box[1] + 1.0), 1.0)
                height_ratio = support_h / max(current_h, 1.0)
                width_ratio = support_w / max(current_w, 1.0)
                likely_half_body_detector = (
                    support_area < 0.78 * max(current_area, 1.0)
                    and height_ratio < 0.78
                    and not (
                        current_w >= 1.45 * support_w
                        and height_ratio >= 0.66
                    )
                )
                demerge_detector = (
                    support_area >= 0.35 * max(current_area, 1.0)
                    and height_ratio >= 0.66
                    and current_w >= 1.25 * support_w
                    and det_conf >= 0.45
                    and (current_cover >= 0.14 or support_cover >= 0.18 or center_norm <= 1.10)
                )
                comparable_detector = (
                    support_area >= 0.86 * max(current_area, 1.0)
                    and det_conf >= 0.55
                    and center_norm <= 1.15
                    and not likely_half_body_detector
                )
                tracker_collapsed = (
                    current_area < 0.012 * support_area
                    or (
                        current_area < 0.010 * float(current_bool.size)
                        and support_area >= 1.35 * max(current_area, 1.0)
                    )
                )
                if support_fuller_person or tracker_collapsed:
                    candidate_mask = support_mask.astype(np.uint8)
                elif demerge_detector:
                    candidate_mask = support_mask.astype(np.uint8)
                elif repair_with_union:
                    candidate_mask = (current_bool | support_mask).astype(np.uint8)
                elif comparable_detector:
                    candidate_mask = support_mask.astype(np.uint8)
                else:
                    # Avoid regressing a propagated full-body person into a
                    # YOLOE torso/half-body support mask on detector frames.
                    continue
            else:
                candidate_mask = support_mask.astype(np.uint8)

            candidate_area = float(candidate_mask.astype(bool).sum())
            prev = support_by_frame.get(frame_idx)
            prev_score = -1.0 if prev is None else float(prev[1]) + 0.10 * min(float(prev[2]), 1.0)
            candidate_score = det_conf + 0.10 * min(candidate_area / max(current_area, 1.0), 1.0)
            if prev is None or candidate_score > prev_score:
                support_by_frame[frame_idx] = (candidate_mask, det_conf, candidate_area / max(current_area, 1.0))

        for frame_idx, (mask, _conf, _area_ratio) in support_by_frame.items():
            repaired[int(frame_idx)] = np.asarray(mask).astype(np.uint8)
        return repaired

    def _clip_person_track_on_detector_contradictions(
        self,
        frame_to_mask: Dict[int, np.ndarray],
        track_det: Dict[str, Any],
        support_dets: List[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        if not frame_to_mask or not support_dets:
            return frame_to_mask

        label = str(track_det.get("label", ""))
        if not (_labels_compatible(label, "person") or _labels_compatible(label, "people")):
            return frame_to_mask

        same_label_support: Dict[int, List[Dict[str, Any]]] = {}
        for det in support_dets:
            if int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) != SEMANTIC_GROUP_MOVABLE_THING:
                continue
            if not (
                _labels_compatible(str(det.get("label", "")), "person")
                or _labels_compatible(str(det.get("label", "")), "people")
            ):
                continue
            if float(det.get("confidence", 0.0)) < 0.45:
                continue
            same_label_support.setdefault(int(det.get("frame_idx", 0)), []).append(det)
        if not same_label_support:
            return frame_to_mask

        own_support_frames = {int(v) for v in track_det.get("support_frames", [])}
        own_support_dets = track_det.get("support_detections", [])
        if isinstance(own_support_dets, list):
            own_support_frames.update(int(d.get("frame_idx", 0)) for d in own_support_dets)

        contradiction_frames: List[int] = []
        for frame_idx in sorted(same_label_support):
            current = frame_to_mask.get(int(frame_idx))
            if current is None:
                continue
            current_bool = np.asarray(current).astype(bool)
            if current_bool.sum() <= 0:
                continue

            best_iou = 0.0
            best_cover = 0.0
            best_center_norm = 999.0
            for support in same_label_support[int(frame_idx)]:
                support_mask = np.asarray(support.get("mask")).astype(bool)
                if support_mask.sum() <= 0:
                    continue
                inter = float((current_bool & support_mask).sum())
                union = float((current_bool | support_mask).sum())
                iou = inter / union if union > 0.0 else 0.0
                cover = max(
                    inter / max(float(current_bool.sum()), 1.0),
                    inter / max(float(support_mask.sum()), 1.0),
                )
                center_norm = self._mask_box_center_norm(current_bool, support_mask)
                best_iou = max(best_iou, iou)
                best_cover = max(best_cover, cover)
                best_center_norm = min(best_center_norm, center_norm)

            matched_detector = (
                best_iou >= 0.08
                or best_cover >= 0.18
                or best_center_norm <= 1.15
            )
            if matched_detector:
                continue

            has_own_future_anchor = any(
                abs(int(anchor_t) - int(frame_idx)) <= max(self.discovery_frame_stride, 1)
                for anchor_t in own_support_frames
            )
            if has_own_future_anchor:
                continue

            contradiction_frames.append(int(frame_idx))

        if not contradiction_frames:
            return frame_to_mask

        stride = max(self.discovery_frame_stride, 1)
        if own_support_frames:
            own_min = min(own_support_frames)
            own_max = max(own_support_frames)
            leading_clip: Optional[int] = None
            trailing_clip: Optional[int] = None
            drop_frames = set()
            for frame_idx in contradiction_frames:
                if frame_idx < own_min - stride:
                    leading_clip = frame_idx if leading_clip is None else max(leading_clip, frame_idx)
                elif frame_idx > own_max + stride:
                    trailing_clip = frame_idx if trailing_clip is None else min(trailing_clip, frame_idx)
                else:
                    drop_frames.add(frame_idx)

            clipped = {}
            for t, mask in frame_to_mask.items():
                t = int(t)
                if leading_clip is not None and t <= leading_clip:
                    continue
                if trailing_clip is not None and t >= trailing_clip:
                    continue
                if t in drop_frames:
                    continue
                clipped[t] = np.asarray(mask).astype(np.uint8)
            return clipped

        first_contradiction = min(contradiction_frames)
        clipped = {
            int(t): np.asarray(mask).astype(np.uint8)
            for t, mask in frame_to_mask.items()
            if int(t) < first_contradiction
        }
        return clipped

    def _filter_sparse_support_against_tracked(
        self,
        support_dets: List[Dict[str, Any]],
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict],
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for det in support_dets:
            label = str(det.get("label", ""))
            frame_idx = int(det.get("frame_idx", 0))
            det_mask = np.asarray(det.get("mask")).astype(bool)
            if det_mask.sum() <= 0:
                continue
            duplicate = False
            for oid, tracked_det in thing_oid_map.items():
                if not _labels_compatible(label, str(tracked_det.get("label", ""))):
                    continue
                tracked_mask = thing_segments.get(frame_idx, {}).get(int(oid))
                if tracked_mask is None:
                    continue
                tracked_bool = np.asarray(tracked_mask).astype(bool)
                iou = _mask_iou(det_mask, tracked_bool)
                center_norm = self._mask_box_center_norm(det_mask, tracked_bool)
                det_area = float(det_mask.sum())
                tracked_area = float(tracked_bool.sum())
                area_sim = min(det_area, tracked_area) / max(det_area, tracked_area, 1.0)
                detector_is_fuller_person = (
                    (_labels_compatible(label, "person") or _labels_compatible(label, "people"))
                    and det_area >= 1.45 * max(tracked_area, 1.0)
                    and center_norm <= 1.35
                )
                if (
                    iou >= 0.20
                    or (
                        (_labels_compatible(label, "person") or _labels_compatible(label, "people"))
                        and center_norm <= 0.78
                        and area_sim >= 0.28
                    )
                ) and not detector_is_fuller_person:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(det)
        return filtered

    def _build_stuff_entries(
        self,
        stuff_dets: List[Dict[str, Any]],
        frame_resource: Any,
        T: int,
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        entries = super()._build_stuff_entries(stuff_dets, frame_resource, T, H, W)
        if not entries:
            return entries

        # Stuff/structure regions are not propagated by the EfficientSAM3
        # object tracker.  Use cheap local optical flow to carry each detected
        # or cross-chunk seeded anchor through the current chunk instead of
        # letting STUFF blink on sparse detector frames.
        max_hold = min(max(T - 1, 0), max(8, max(int(self.discovery_frame_stride), 1) * 16))
        max_bridge = max(max_hold + 2, max(int(self.discovery_frame_stride), 1) * 8)
        flow_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

        def _read_gray(frame_idx: int) -> Optional[np.ndarray]:
            path = os.path.join(str(frame_resource), f"{int(frame_idx):05d}.jpg")
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return gray

        def _flow(src_t: int, dst_t: int) -> Optional[np.ndarray]:
            key = (int(src_t), int(dst_t))
            if key in flow_cache:
                return flow_cache[key]
            src = _read_gray(src_t)
            dst = _read_gray(dst_t)
            if src is None or dst is None:
                flow_cache[key] = None
                return None
            scale = 0.5 if max(H, W) >= 1000 else 1.0
            if scale != 1.0:
                size = (max(16, int(round(W * scale))), max(16, int(round(H * scale))))
                src_eval = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
                dst_eval = cv2.resize(dst, size, interpolation=cv2.INTER_AREA)
            else:
                src_eval = src
                dst_eval = dst
            flow_eval = cv2.calcOpticalFlowFarneback(
                src_eval,
                dst_eval,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            if scale != 1.0:
                flow_full = cv2.resize(flow_eval, (W, H), interpolation=cv2.INTER_LINEAR)
                flow_full[..., 0] /= scale
                flow_full[..., 1] /= scale
            else:
                flow_full = flow_eval
            flow_cache[key] = flow_full.astype(np.float32)
            return flow_cache[key]

        grid_x, grid_y = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )

        def _clean_warped_mask(mask: np.ndarray, ref_area: float) -> np.ndarray:
            mask_bool = np.asarray(mask).astype(bool)
            if mask_bool.sum() <= 0:
                return mask_bool.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
            if num_labels <= 1:
                return cleaned.astype(np.uint8)
            min_component = max(128, int(0.015 * max(float(ref_area), 1.0)))
            kept = np.zeros_like(cleaned, dtype=np.uint8)
            for comp_idx in range(1, num_labels):
                if int(stats[comp_idx, cv2.CC_STAT_AREA]) >= min_component:
                    kept[labels == comp_idx] = 1
            if kept.sum() <= 0:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                kept[labels == largest] = 1
            return kept.astype(np.uint8)

        def _warp_mask_once(mask: np.ndarray, src_t: int, dst_t: int) -> Optional[np.ndarray]:
            flow = _flow(src_t, dst_t)
            if flow is None:
                return None
            mask_f = np.asarray(mask).astype(np.float32)
            # Approximate inverse warp. It is deliberately local and confidence
            # decayed, so STUFF moves with the camera instead of freezing.
            warped = cv2.remap(
                mask_f,
                grid_x - flow[..., 0],
                grid_y - flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return (warped > 0.45).astype(np.uint8)

        def _propagate_from_anchor(
            filled_masks: Dict[int, np.ndarray],
            filled_conf: Dict[int, float],
            anchor_t: int,
            direction: int,
            steps: int,
            base_conf: float,
        ) -> None:
            src_t = int(anchor_t)
            current = np.asarray(filled_masks[src_t]).astype(np.uint8)
            ref_area = float(current.astype(bool).sum())
            if ref_area <= 0.0:
                return
            for step in range(1, int(steps) + 1):
                dst_t = src_t + int(direction)
                if not (0 <= dst_t < T):
                    break
                warped = _warp_mask_once(current, src_t, dst_t)
                if warped is None:
                    break
                warped = _clean_warped_mask(warped, ref_area)
                warped_area = float(warped.astype(bool).sum())
                if warped_area <= 0.0:
                    break
                area_sim = min(warped_area, ref_area) / max(warped_area, ref_area, 1.0)
                if area_sim < 0.20:
                    break
                conf = max(0.18, float(base_conf) * (0.92 ** step))
                if conf > float(filled_conf.get(dst_t, -1.0)):
                    filled_masks[dst_t] = warped.astype(np.uint8)
                    filled_conf[dst_t] = conf
                current = warped
                src_t = dst_t

        for entry in entries:
            frame_to_mask = entry.get("frame_to_mask")
            frame_to_conf = entry.get("frame_to_confidence", {})
            if not isinstance(frame_to_mask, dict) or not frame_to_mask:
                continue

            detected_frames = sorted(int(t) for t in frame_to_mask.keys())
            filled_masks = {
                int(t): np.asarray(mask).astype(np.uint8)
                for t, mask in frame_to_mask.items()
            }
            filled_conf = {int(t): float(v) for t, v in frame_to_conf.items()}

            for left, right in zip(detected_frames[:-1], detected_frames[1:]):
                gap = int(right) - int(left)
                if gap <= 1 or gap > max_bridge:
                    continue
                left_mask = np.asarray(filled_masks[left]).astype(bool)
                right_mask = np.asarray(filled_masks[right]).astype(bool)
                if left_mask.sum() <= 0 or right_mask.sum() <= 0:
                    continue
                area_sim = min(float(left_mask.sum()), float(right_mask.sum())) / max(
                    float(left_mask.sum()),
                    float(right_mask.sum()),
                    1.0,
                )
                box_iou = _box_iou_xyxy(
                    np.asarray(self._mask_to_box_xyxy(left_mask), dtype=np.float32),
                    np.asarray(self._mask_to_box_xyxy(right_mask), dtype=np.float32),
                )
                if area_sim < 0.35 and box_iou < 0.10:
                    continue
                _propagate_from_anchor(
                    filled_masks,
                    filled_conf,
                    int(left),
                    direction=1,
                    steps=max(0, gap - 1),
                    base_conf=min(filled_conf.get(left, 0.0), 0.45),
                )
                _propagate_from_anchor(
                    filled_masks,
                    filled_conf,
                    int(right),
                    direction=-1,
                    steps=max(0, gap - 1),
                    base_conf=min(filled_conf.get(right, 0.0), 0.45),
                )

            for anchor in detected_frames:
                _propagate_from_anchor(
                    filled_masks,
                    filled_conf,
                    int(anchor),
                    direction=-1,
                    steps=max_hold,
                    base_conf=min(filled_conf.get(anchor, 0.0), 0.38),
                )
                _propagate_from_anchor(
                    filled_masks,
                    filled_conf,
                    int(anchor),
                    direction=1,
                    steps=max_hold,
                    base_conf=min(filled_conf.get(anchor, 0.0), 0.38),
                )

            entry["frame_to_mask"] = dict(sorted(filled_masks.items()))
            entry["frame_to_confidence"] = dict(sorted(filled_conf.items()))
            entry["support_frames"] = sorted(filled_masks.keys())
        return entries

    def _append_sparse_detector_thing_tracks(
        self,
        support_dets: List[Dict[str, Any]],
        discovery_indices: List[int],
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict],
        area: int,
    ) -> None:
        if not support_dets:
            return

        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for det in support_dets:
            det_copy = dict(det)
            frame_idx = int(det_copy.get("frame_idx", 0))
            by_frame.setdefault(frame_idx, []).append(det_copy)
        clusters = self._cluster_sam31_track_candidates(
            by_frame,
            sorted(by_frame.keys()),
            area,
        )
        if not clusters:
            return

        tiny_labels = {"book", "bottle", "cup", "phone"}
        selected: List[Dict[str, Any]] = []
        counts = {
            "person": 0,
            "static": 0,
            "tiny": 0,
            "other": 0,
        }

        def _bucket(det: Dict[str, Any]) -> str:
            label = str(det.get("label", "")).strip().lower()
            sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
            if _labels_compatible(label, "person") or _labels_compatible(label, "people"):
                return "person"
            if sem_group == SEMANTIC_GROUP_STATIC_THING:
                return "static"
            if label in tiny_labels:
                return "tiny"
            return "other"

        limits = {
            "person": 4,
            "static": 3,
            "tiny": 1,
            "other": 2,
        }

        clusters.sort(
            key=lambda d: (
                -int(d.get("seed_count", 0)),
                -int(_bucket(d) == "person"),
                -int(d.get("cluster_size", 1)),
                -float(d.get("cluster_confidence", d.get("confidence", 0.0))),
                -float(d.get("area_ratio", 0.0)),
                int(d.get("frame_idx", 0)),
            )
        )

        for cluster in clusters:
            bucket = _bucket(cluster)
            if counts[bucket] >= limits[bucket] or len(selected) >= 10:
                continue

            cluster_size = int(cluster.get("cluster_size", 1))
            conf = float(cluster.get("cluster_confidence", cluster.get("confidence", 0.0)))
            area_ratio = float(cluster.get("area_ratio", 0.0))
            has_seed = int(cluster.get("seed_count", 0)) > 0 or cluster.get("seed_global_track_idx", None) is not None

            if not has_seed:
                if bucket == "tiny" and not (cluster_size >= 2 and (conf >= 0.72 or area_ratio >= 0.003)):
                    continue
                if bucket == "person" and not (cluster_size >= 2 or conf >= 0.58 or area_ratio >= 0.012):
                    continue
                if bucket in {"static", "other"} and not (cluster_size >= 2 or conf >= 0.65 or area_ratio >= 0.012):
                    continue

            selected.append(cluster)
            counts[bucket] += 1

        if not selected:
            return

        selected_support: List[Dict[str, Any]] = []
        for cluster in selected:
            support = cluster.get("support_detections", [])
            if isinstance(support, list) and support:
                selected_support.extend(dict(det) for det in support)
            else:
                selected_support.append(dict(cluster))

        super()._append_sparse_detector_thing_tracks(
            selected_support,
            discovery_indices,
            thing_segments,
            thing_oid_map,
            area,
        )

    def _subtract_thing_masks_from_stuff_entries(
        self,
        stuff_entries: List[Dict[str, Any]],
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict],
    ) -> None:
        if not stuff_entries or not thing_segments:
            return

        kernel = np.ones((5, 5), np.uint8)
        thing_union_by_frame: Dict[int, np.ndarray] = {}
        for frame_idx, frame_objs in thing_segments.items():
            union_mask = None
            for oid, mask in frame_objs.items():
                det = thing_oid_map.get(int(oid), {})
                if int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                    continue
                mask_bool = np.asarray(mask).astype(bool)
                if mask_bool.sum() <= 0:
                    continue
                union_mask = mask_bool if union_mask is None else (union_mask | mask_bool)
            if union_mask is None:
                continue
            thing_union_by_frame[int(frame_idx)] = cv2.dilate(
                union_mask.astype(np.uint8),
                kernel,
                iterations=1,
            ).astype(bool)

        if not thing_union_by_frame:
            return

        for entry in stuff_entries:
            frame_to_mask = entry.get("frame_to_mask")
            if not isinstance(frame_to_mask, dict):
                continue
            for frame_idx, mask in list(frame_to_mask.items()):
                thing_union = thing_union_by_frame.get(int(frame_idx))
                if thing_union is None:
                    continue
                mask_bool = np.asarray(mask).astype(bool)
                original_area = float(mask_bool.sum())
                if original_area <= 0.0:
                    continue
                cleaned = mask_bool & (~thing_union)
                cleaned_area = float(cleaned.sum())
                if cleaned_area < max(64.0, 0.18 * original_area):
                    continue
                frame_to_mask[int(frame_idx)] = cleaned.astype(np.uint8)

    def _sam3_cutie_prompt_frames(self, discovery_indices: List[int], T: int) -> List[int]:
        frames = self._select_prompt_frames(
            discovery_indices,
            T,
            max(1, int(self.sam3_cutie_detection_frame_count)),
        )
        if frames:
            return frames
        return [0] if T > 0 else []

    def _sam3_cutie_expand_prompt_labels(self, labels: List[str]) -> List[Tuple[str, str]]:
        expanded: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for label in labels:
            canonical = canonicalize_label(str(label))
            aliases: List[Tuple[str, str]] = [(canonical, canonical)]
            if _labels_compatible(canonical, "person") or _labels_compatible(canonical, "people"):
                aliases.extend([
                    ("full body person", "person"),
                    ("walking person", "person"),
                    ("standing person", "person"),
                    ("black-clothed person", "person"),
                    ("person in black clothes", "person"),
                    ("woman", "person"),
                    ("human figure", "person"),
                ])
            for prompt_text, alias_canonical in aliases:
                alias_canonical = canonicalize_label(alias_canonical)
                if alias_canonical in DISABLED_EFFICIENT_PROMPT_LABELS:
                    continue
                key = (str(prompt_text), alias_canonical)
                if key not in seen:
                    expanded.append(key)
                    seen.add(key)
        return expanded

    def _sam3_cutie_min_area_ratio(self, sem_group: int, label: str) -> float:
        label_l = str(label).strip().lower()
        is_person = _labels_compatible(label_l, "person") or _labels_compatible(label_l, "people")
        if is_person:
            return 0.0020
        if int(sem_group) == SEMANTIC_GROUP_MOVABLE_THING:
            return 0.0012
        if int(sem_group) == SEMANTIC_GROUP_STATIC_THING:
            return 0.0030
        if int(sem_group) == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
            return 0.0100
        return 0.0060

    def _query_sam3_frame_prompt_detections(
        self,
        session_id: str,
        label: str,
        frame_idx: int,
        H: int,
        W: int,
        area: int,
        canonical_label_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
            if "No points are provided; please add points first" in str(exc):
                return []
            raise

        outputs = response.get("outputs", {})
        masks = outputs.get("out_binary_masks", [])
        probs = outputs.get("out_probs", [])
        canonical_label = canonicalize_label(canonical_label_override or label)
        sem_group = label_to_group(canonical_label)
        min_area_ratio = self._sam3_cutie_min_area_ratio(sem_group, canonical_label)
        max_area_ratio = float(self.max_mask_area_ratio)
        if int(sem_group) == SEMANTIC_GROUP_STATIC_THING:
            max_area_ratio = min(max_area_ratio, 0.34)
        dets: List[Dict[str, Any]] = []
        for idx, mask in enumerate(masks):
            mask_bool = np.asarray(mask).astype(bool)
            mask_area = int(mask_bool.sum())
            if mask_area <= 0:
                continue
            area_ratio = float(mask_area) / max(int(area), 1)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            confidence = float(probs[idx]) if idx < len(probs) else 0.0
            if confidence < 0.12 and area_ratio < max(0.02, 2.0 * min_area_ratio):
                continue
            box = self._mask_to_box_xyxy(mask_bool)
            if float(box[2] - box[0]) <= 1.0 or float(box[3] - box[1]) <= 1.0:
                continue
            if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                keep, _reason = passes_structure_mask_quality(
                    canonical_label,
                    mask_bool,
                    box,
                    confidence,
                    area_ratio,
                    H,
                    W,
                )
                if not keep:
                    continue
            dets.append({
                "mask": mask_bool.astype(np.uint8),
                "box": box.astype(np.float32),
                "confidence": confidence,
                "label": canonical_label,
                "raw_label": str(label),
                "sem_group": int(sem_group),
                "area_ratio": area_ratio,
                "frame_idx": int(frame_idx),
                "detector_source": "sam3_prompt_frame",
                "is_prompt_only": True,
            })

        is_person_prompt = _labels_compatible(canonical_label, "person") or _labels_compatible(canonical_label, "people")
        if int(sem_group) == SEMANTIC_GROUP_STRUCTURE_ANCHOR or _is_stuff(int(sem_group)):
            dets.sort(
                key=lambda d: (
                    -float(d.get("area_ratio", 0.0)),
                    -float(d.get("confidence", 0.0)),
                )
            )
        elif is_person_prompt:
            # SAM3 text prompts may rank a confident torso higher than a full
            # silhouette. For tracking seeds, completeness is more valuable.
            dets.sort(
                key=lambda d: (
                    -float(d.get("area_ratio", 0.0)),
                    -float(d.get("confidence", 0.0)),
                )
            )
        else:
            dets.sort(
                key=lambda d: (
                    -int(int(d.get("sem_group", -1)) == SEMANTIC_GROUP_MOVABLE_THING),
                    -float(d.get("confidence", 0.0)),
                    -float(d.get("area_ratio", 0.0)),
                )
            )
        return dets[: max(1, int(self.sam3_cutie_max_prompt_dets_per_label))]

    def _query_sam3_image_prompt_detections(
        self,
        image_state: Dict[str, Any],
        label: str,
        frame_idx: int,
        H: int,
        W: int,
        area: int,
        canonical_label_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.sam3_image_processor is None:
            return []

        self.sam3_image_processor.reset_all_prompts(image_state)
        try:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(self.device).startswith("cuda"),
            ):
                image_state = self.sam3_image_processor.set_text_prompt(
                    prompt=str(label),
                    state=image_state,
                )
        except RuntimeError as exc:
            if "No points are provided; please add points first" in str(exc):
                return []
            raise

        masks = image_state.get("masks", [])
        boxes = image_state.get("boxes", [])
        scores = image_state.get("scores", [])
        if isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.detach().float().cpu().numpy()
        else:
            boxes_np = np.asarray(boxes)
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().float().cpu().numpy()
        else:
            scores_np = np.asarray(scores)

        canonical_label = canonicalize_label(canonical_label_override or label)
        sem_group = label_to_group(canonical_label)
        min_area_ratio = self._sam3_cutie_min_area_ratio(sem_group, canonical_label)
        max_area_ratio = float(self.max_mask_area_ratio)
        if int(sem_group) == SEMANTIC_GROUP_STATIC_THING:
            max_area_ratio = min(max_area_ratio, 0.34)
        dets: List[Dict[str, Any]] = []
        for idx, mask in enumerate(masks_np):
            mask_bool = np.asarray(mask).squeeze().astype(bool)
            if mask_bool.shape != (H, W):
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mask_area = int(mask_bool.sum())
            if mask_area <= 0:
                continue
            area_ratio = float(mask_area) / max(int(area), 1)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            confidence = float(scores_np[idx]) if idx < len(scores_np) else 0.0
            if confidence < 0.12 and area_ratio < max(0.02, 2.0 * min_area_ratio):
                continue
            if idx < len(boxes_np):
                box = np.asarray(boxes_np[idx], dtype=np.float32).reshape(-1)[:4]
            else:
                box = self._mask_to_box_xyxy(mask_bool).astype(np.float32)
            if box.size < 4 or float(box[2] - box[0]) <= 1.0 or float(box[3] - box[1]) <= 1.0:
                box = self._mask_to_box_xyxy(mask_bool).astype(np.float32)
            if float(box[2] - box[0]) <= 1.0 or float(box[3] - box[1]) <= 1.0:
                continue
            if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                keep, _reason = passes_structure_mask_quality(
                    canonical_label,
                    mask_bool,
                    box,
                    confidence,
                    area_ratio,
                    H,
                    W,
                )
                if not keep:
                    continue
            dets.append({
                "mask": mask_bool.astype(np.uint8),
                "box": box.astype(np.float32),
                "confidence": confidence,
                "label": canonical_label,
                "raw_label": str(label),
                "sem_group": int(sem_group),
                "area_ratio": area_ratio,
                "frame_idx": int(frame_idx),
                "detector_source": "sam3_image_prompt_frame",
                "is_prompt_only": True,
            })

        is_person_prompt = _labels_compatible(canonical_label, "person") or _labels_compatible(canonical_label, "people")
        if int(sem_group) == SEMANTIC_GROUP_STRUCTURE_ANCHOR or _is_stuff(int(sem_group)):
            dets.sort(
                key=lambda d: (
                    -float(d.get("area_ratio", 0.0)),
                    -float(d.get("confidence", 0.0)),
                )
            )
        elif is_person_prompt:
            # Prefer a full-body seed over a high-confidence partial person
            # mask; Cutie will faithfully propagate whichever seed we give it.
            dets.sort(
                key=lambda d: (
                    -float(d.get("area_ratio", 0.0)),
                    -float(d.get("confidence", 0.0)),
                )
            )
        else:
            dets.sort(
                key=lambda d: (
                    -int(int(d.get("sem_group", -1)) == SEMANTIC_GROUP_MOVABLE_THING),
                    -float(d.get("confidence", 0.0)),
                    -float(d.get("area_ratio", 0.0)),
                )
            )
        return dets[: max(1, int(self.sam3_cutie_max_prompt_dets_per_label))]

    def _detect_sam3_cutie_keyframe_detections(
        self,
        frame_resource: Any,
        discovery_indices: List[int],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        prompt_frames = self._sam3_cutie_prompt_frames(discovery_indices, T)
        labels = []
        for label in list(self.thing_prompts) + list(self.stuff_prompts):
            canonical = canonicalize_label(str(label))
            if canonical in DISABLED_EFFICIENT_PROMPT_LABELS:
                continue
            if canonical and canonical not in labels:
                labels.append(canonical)
        if self.sam3_cutie_use_yoloe:
            # YOLOE already provides dense semantic seeds for non-person THINGs.
            # Keep SAM3 text prompts focused on people and large structures,
            # where SAM3 adds the most value and avoids many slow/noisy prompts.
            labels = [
                label for label in labels
                if (
                    (
                        label_to_group(label) == SEMANTIC_GROUP_MOVABLE_THING
                        and (
                            _labels_compatible(label, "person")
                            or _labels_compatible(label, "people")
                        )
                    )
                    or label_to_group(label) == SEMANTIC_GROUP_STRUCTURE_ANCHOR
                    or _is_stuff(label_to_group(label))
                )
            ]
        prompt_specs = self._sam3_cutie_expand_prompt_labels(labels)
        if not prompt_frames or not prompt_specs:
            return {}

        detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        if self.sam3_image_processor is not None:
            refresh_labels = [
                spec
                for spec in prompt_specs
                if (
                    label_to_group(spec[1]) == SEMANTIC_GROUP_MOVABLE_THING
                    and (
                        not self.sam3_cutie_use_yoloe
                        or _labels_compatible(spec[1], "person")
                        or _labels_compatible(spec[1], "people")
                    )
                )
            ]
            for frame_rank, frame_idx in enumerate(prompt_frames):
                frame_path = os.path.join(str(frame_resource), f"{int(frame_idx):05d}.jpg")
                image = Image.open(frame_path).convert("RGB")
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=str(self.device).startswith("cuda"),
                ):
                    image_state = self.sam3_image_processor.set_image(image)
                frame_dets: List[Dict[str, Any]] = []
                try:
                    labels_for_frame = prompt_specs if frame_rank == 0 else refresh_labels
                    for prompt_text, canonical_label in labels_for_frame:
                        frame_dets.extend(
                            self._query_sam3_image_prompt_detections(
                                image_state,
                                prompt_text,
                                int(frame_idx),
                                H,
                                W,
                                area,
                                canonical_label_override=canonical_label,
                            )
                        )
                finally:
                    self.sam3_image_processor.reset_all_prompts(image_state)
                    image.close()
                if frame_dets:
                    detections_by_frame[int(frame_idx)] = self._nms(frame_dets)
            return detections_by_frame

        session_id = self._start_sam31_session(frame_resource)
        try:
            refresh_labels = [
                spec
                for spec in prompt_specs
                if (
                    label_to_group(spec[1]) == SEMANTIC_GROUP_MOVABLE_THING
                    and (
                        not self.sam3_cutie_use_yoloe
                        or _labels_compatible(spec[1], "person")
                        or _labels_compatible(spec[1], "people")
                    )
                )
            ]
            for frame_rank, frame_idx in enumerate(prompt_frames):
                frame_dets: List[Dict[str, Any]] = []
                labels_for_frame = prompt_specs if frame_rank == 0 else refresh_labels
                for prompt_text, canonical_label in labels_for_frame:
                    frame_dets.extend(
                        self._query_sam3_frame_prompt_detections(
                            session_id,
                            prompt_text,
                            int(frame_idx),
                            H,
                            W,
                            area,
                            canonical_label_override=canonical_label,
                        )
                    )
                if frame_dets:
                    detections_by_frame[int(frame_idx)] = self._nms(frame_dets)
        finally:
            self.video_predictor.handle_request(
                dict(type="close_session", session_id=session_id)
            )
        return detections_by_frame

    def _sam3_cutie_detection_priority(self, det: Dict[str, Any]) -> Tuple[int, float, float]:
        sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
        label = canonicalize_label(str(det.get("label", "")))
        if (
            sem_group == SEMANTIC_GROUP_MOVABLE_THING
            and (_labels_compatible(label, "person") or _labels_compatible(label, "people"))
        ):
            group_priority = 0
        elif sem_group == SEMANTIC_GROUP_MOVABLE_THING:
            group_priority = 1
        elif sem_group == SEMANTIC_GROUP_STATIC_THING:
            group_priority = 2
        elif sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
            group_priority = 3
        else:
            group_priority = 4
        return (
            int(group_priority),
            -float(det.get("area_ratio", 0.0)),
            -float(det.get("confidence", 0.0)),
        )

    def _sam3_cutie_is_duplicate_new_det(
        self,
        det: Dict[str, Any],
        active_masks: Dict[int, np.ndarray],
        oid_map: Dict[int, Dict],
        same_frame_dets: List[Dict[str, Any]],
    ) -> bool:
        det_mask = np.asarray(det.get("mask")).astype(bool)
        if det_mask.sum() <= 0:
            return True
        for prev_det in same_frame_dets:
            if not self._sam3_cutie_detections_compete(det, prev_det):
                continue
            prev = np.asarray(prev_det.get("mask")).astype(bool)
            prev_bool = np.asarray(prev).astype(bool)
            if prev_bool.sum() <= 0:
                continue
            inter = float((det_mask & prev_bool).sum())
            prev_box = self._mask_to_box_xyxy(prev_bool)
            det_box = np.asarray(det.get("box", self._mask_to_box_xyxy(det_mask)), dtype=np.float32)
            box_iou = _box_iou_xyxy(det_box, prev_box)
            contained = inter / max(float(det_mask.sum()), 1.0)
            if (
                _mask_iou(det_mask, prev_bool) >= 0.42
                or contained >= 0.72
                or (box_iou >= 0.35 and contained >= 0.45)
            ):
                return True

        det_label = str(det.get("label", ""))
        det_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
        for oid, prev_mask in active_masks.items():
            meta = oid_map.get(int(oid), {})
            if int(meta.get("sem_group", -999)) != det_group:
                continue
            if not self._sam3_cutie_labels_compete(det_label, str(meta.get("label", ""))):
                continue
            prev_bool = np.asarray(prev_mask).astype(bool)
            if prev_bool.sum() <= 0:
                continue
            inter = float((det_mask & prev_bool).sum())
            prev_box = self._mask_to_box_xyxy(prev_bool)
            det_box = np.asarray(det.get("box", self._mask_to_box_xyxy(det_mask)), dtype=np.float32)
            box_iou = _box_iou_xyxy(det_box, prev_box)
            contained = inter / max(float(det_mask.sum()), 1.0)
            if (
                _mask_iou(det_mask, prev_bool) >= 0.22
                or contained >= 0.52
                or (box_iou >= 0.18 and contained >= 0.35)
            ):
                return True
        return False

    def _sam3_cutie_seed_budget(
        self,
        sem_group: int,
        label: str,
    ) -> Tuple[int, int]:
        """Return per-group and per-label seed caps for Cutie initialization."""
        if int(sem_group) == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
            group_budget = max(int(self.sam31_max_structure_objects), 0)
            return group_budget, 1
        if _is_stuff(int(sem_group)):
            group_budget = max(min(int(self.sam31_max_structure_objects), 2), 0)
            return group_budget, 1
        if int(sem_group) == SEMANTIC_GROUP_STATIC_THING:
            group_budget = max(int(self.sam31_max_static_objects), 0)
            return group_budget, 1
        if int(sem_group) == SEMANTIC_GROUP_MOVABLE_THING:
            group_budget = max(int(self.max_thing_objects), 1)
            label_l = str(label).strip().lower()
            if _labels_compatible(label_l, "person") or _labels_compatible(label_l, "people"):
                return group_budget, max(5, int(self.sam31_max_movable_objects))
            return group_budget, max(1, min(group_budget, 2))
        return 0, 0

    def _read_frame_tensor_for_cutie(self, frame_resource: Any, frame_idx: int) -> torch.Tensor:
        path = os.path.join(str(frame_resource), f"{int(frame_idx):05d}.jpg")
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read frame for Cutie: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div_(255.0)
        return tensor.to(self.device, non_blocking=True)

    def _warp_mask_between_boxes(
        self,
        mask: np.ndarray,
        src_box: np.ndarray,
        dst_box: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        src = np.asarray(src_box, dtype=np.float32).reshape(-1)[:4]
        dst = np.asarray(dst_box, dtype=np.float32).reshape(-1)[:4]
        src_w = max(float(src[2] - src[0]), 1.0)
        src_h = max(float(src[3] - src[1]), 1.0)
        dst_w = max(float(dst[2] - dst[0]), 1.0)
        dst_h = max(float(dst[3] - dst[1]), 1.0)
        sx = float(np.clip(dst_w / src_w, 0.72, 1.38))
        sy = float(np.clip(dst_h / src_h, 0.72, 1.38))
        src_cx = 0.5 * float(src[0] + src[2])
        src_cy = 0.5 * float(src[1] + src[3])
        dst_cx = 0.5 * float(dst[0] + dst[2])
        dst_cy = 0.5 * float(dst[1] + dst[3])
        matrix = np.asarray(
            [
                [sx, 0.0, dst_cx - sx * src_cx],
                [0.0, sy, dst_cy - sy * src_cy],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            np.asarray(mask).astype(np.uint8),
            matrix,
            (int(W), int(H)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return (warped > 0).astype(np.uint8)

    def _fill_person_track_gaps(
        self,
        frame_to_mask: Dict[int, np.ndarray],
        *,
        H: int,
        W: int,
        max_gap: int = 36,
    ) -> Dict[int, np.ndarray]:
        if len(frame_to_mask) < 2:
            return frame_to_mask
        frames = sorted(int(t) for t in frame_to_mask.keys())
        filled = dict(frame_to_mask)
        area = max(float(H * W), 1.0)
        for left, right in zip(frames[:-1], frames[1:]):
            gap = int(right) - int(left)
            if gap <= 1 or gap > int(max_gap):
                continue
            left_mask = np.asarray(filled[left]).astype(bool)
            right_mask = np.asarray(filled[right]).astype(bool)
            left_area = float(left_mask.sum()) / area
            right_area = float(right_mask.sum()) / area
            if min(left_area, right_area) < 0.020:
                continue
            left_box = self._mask_to_box_xyxy(left_mask)
            right_box = self._mask_to_box_xyxy(right_mask)
            center_dist = float(
                np.hypot(
                    ((left_box[0] + left_box[2]) - (right_box[0] + right_box[2])) * 0.5 / max(float(W), 1.0),
                    ((left_box[1] + left_box[3]) - (right_box[1] + right_box[3])) * 0.5 / max(float(H), 1.0),
                )
            )
            area_sim = min(left_area, right_area) / max(left_area, right_area)
            long_gap = gap > 18
            max_center_dist = 0.46 if long_gap else 0.36
            min_area_sim = 0.28 if long_gap else 0.34
            if center_dist > max_center_dist or area_sim < min_area_sim:
                continue
            for t in range(left + 1, right):
                alpha = float(t - left) / float(gap)
                target_box = (1.0 - alpha) * left_box + alpha * right_box
                if alpha <= 0.5:
                    src_mask = left_mask
                    src_box = left_box
                else:
                    src_mask = right_mask
                    src_box = right_box
                filled[int(t)] = self._warp_mask_between_boxes(
                    src_mask,
                    src_box,
                    target_box,
                    H,
                    W,
                )
        return filled

    def _prune_tiny_person_shadow_segments(
        self,
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict[str, Any]],
        *,
        T: int,
        area: int,
    ) -> set[int]:
        person_oids = [
            int(oid)
            for oid, det in thing_oid_map.items()
            if (
                _labels_compatible(str(det.get("label", "")), "person")
                or _labels_compatible(str(det.get("label", "")), "people")
            )
        ]
        if len(person_oids) <= 1:
            return set()

        stats: Dict[int, Dict[str, float]] = {}
        for oid in person_oids:
            areas: List[float] = []
            frames: List[int] = []
            for t, objs in thing_segments.items():
                mask = objs.get(int(oid))
                if mask is None:
                    continue
                frames.append(int(t))
                areas.append(float(np.asarray(mask).astype(bool).sum()) / max(float(area), 1.0))
            if areas:
                stats[int(oid)] = {
                    "visible": float(len(areas)),
                    "mean_area": float(sum(areas) / max(len(areas), 1)),
                    "max_area": float(max(areas)),
                    "birth": float(min(frames) if frames else 0),
                }
        if len(stats) <= 1:
            return set()

        dominant = max(
            stats,
            key=lambda oid: (
                stats[oid]["visible"],
                stats[oid]["mean_area"],
            ),
        )
        dom = stats[dominant]
        if dom["visible"] < max(12.0, 0.85 * float(T)) or dom["mean_area"] < 0.055:
            return set()

        removed: set[int] = set()
        for oid, st in stats.items():
            if oid == dominant:
                continue
            if (
                st["visible"] <= 0.60 * float(T)
                and st["mean_area"] < 0.012
                and st["max_area"] < 0.022
                and abs(st["birth"] - dom["birth"]) <= 4.0
                and dom["mean_area"] >= 6.0 * max(st["mean_area"], 1e-6)
            ):
                removed.add(int(oid))
        if not removed:
            return removed
        for t in list(thing_segments.keys()):
            for oid in removed:
                thing_segments[t].pop(int(oid), None)
            if not thing_segments[t]:
                thing_segments.pop(t, None)
        for oid in removed:
            thing_oid_map.pop(int(oid), None)
        return removed

    def _prune_unreliable_auxiliary_segments(
        self,
        thing_segments: Dict[int, Dict[int, np.ndarray]],
        thing_oid_map: Dict[int, Dict[str, Any]],
        structure_segments: Dict[int, Dict[int, np.ndarray]],
        structure_oid_map: Dict[int, Dict[str, Any]],
        *,
        T: int,
    ) -> Tuple[set[int], set[int]]:
        removed_things: set[int] = set()
        removed_structures: set[int] = set()

        for oid, det in list(thing_oid_map.items()):
            label = str(det.get("label", ""))
            if _labels_compatible(label, "person") or _labels_compatible(label, "people"):
                continue
            sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
            visible = sum(1 for objs in thing_segments.values() if int(oid) in objs)
            if sem_group == SEMANTIC_GROUP_STATIC_THING and visible <= 1:
                removed_things.add(int(oid))
            elif visible <= 1 and sem_group != SEMANTIC_GROUP_MOVABLE_THING:
                removed_things.add(int(oid))

        min_structure_visible = max(8, int(round(0.15 * max(int(T), 1))))
        for oid, det in list(structure_oid_map.items()):
            visible = sum(1 for objs in structure_segments.values() if int(oid) in objs)
            if visible < min_structure_visible:
                removed_structures.add(int(oid))

        for t in list(thing_segments.keys()):
            for oid in removed_things:
                thing_segments[t].pop(int(oid), None)
            if not thing_segments[t]:
                thing_segments.pop(t, None)
        for oid in removed_things:
            thing_oid_map.pop(int(oid), None)

        for t in list(structure_segments.keys()):
            for oid in removed_structures:
                structure_segments[t].pop(int(oid), None)
            if not structure_segments[t]:
                structure_segments.pop(t, None)
        for oid in removed_structures:
            structure_oid_map.pop(int(oid), None)

        return removed_things, removed_structures

    def _move_tensor_tree_to_device(self, obj: Any, device: str) -> Any:
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, list):
            return [self._move_tensor_tree_to_device(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_tensor_tree_to_device(v, device) for v in obj)
        if isinstance(obj, dict):
            return {k: self._move_tensor_tree_to_device(v, device) for k, v in obj.items()}
        if is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                try:
                    value = self._move_tensor_tree_to_device(getattr(obj, field.name), device)
                    setattr(obj, field.name, value)
                except Exception:
                    pass
            return obj
        return obj

    def _move_sam3_image_processor_to_device(self, device: str) -> None:
        processor = getattr(self, "sam3_image_processor", None)
        if processor is None:
            return
        model = getattr(processor, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to(device)
        if hasattr(processor, "device"):
            processor.device = device
        if hasattr(processor, "find_stage"):
            processor.find_stage = self._move_tensor_tree_to_device(
                processor.find_stage,
                device,
            )

    def _sam2_device_type(self) -> str:
        return "cuda" if str(self.device).startswith("cuda") else "cpu"

    def _add_sam2_object_prompt(
        self,
        inference_state: Dict[str, Any],
        det: Dict[str, Any],
        obj_id: int,
        prompt_frame: int,
        H: int,
        W: int,
    ) -> Tuple[int, Any, Any]:
        label = str(det.get("label", "")).strip().lower()
        is_person = _labels_compatible(label, "person") or _labels_compatible(label, "people")
        if is_person:
            rel_box = self._xyxy_to_efficientsam3_rel_box(det, H=H, W=W)
            if rel_box is not None:
                abs_box = np.asarray(
                    [
                        float(rel_box[0]) * max(W, 1),
                        float(rel_box[1]) * max(H, 1),
                        float(rel_box[2]) * max(W, 1),
                        float(rel_box[3]) * max(H, 1),
                    ],
                    dtype=np.float32,
                )
                try:
                    return self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=int(prompt_frame),
                        obj_id=int(obj_id),
                        box=abs_box,
                    )
                except Exception as exc:
                    print(
                        f"[sam2] warning: box prompt failed for person "
                        f"obj_id={obj_id} frame={prompt_frame}: {exc!r}; falling back to mask prompt"
                    )

        prompt_mask = self._det_to_prompt_mask_tensor(det, H=H, W=W)
        if prompt_mask is not None:
            return self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=int(prompt_frame),
                obj_id=int(obj_id),
                mask=prompt_mask,
            )
        box = det.get("box", None)
        if box is None:
            raise ValueError("detection has neither a usable mask nor a usable box")
        return self.video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(prompt_frame),
            obj_id=int(obj_id),
            box=np.asarray(box, dtype=np.float32),
        )

    @torch.inference_mode()
    def _track_objects_once_sam2_tracker(
        self,
        frame_resource: Any,
        detections: List[Dict[str, Any]],
        T: int,
        H: int,
        W: int,
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, float], Dict[int, Dict[str, Any]]]:
        if not detections:
            return {}, {}, {}
        if self.video_predictor is None:
            raise RuntimeError("SAM2 tracker is not initialized.")

        inference_state = self.video_predictor.init_state(
            video_path=frame_resource,
            offload_video_to_cpu=bool(self.sam31_offload_video_to_cpu),
            offload_state_to_cpu=bool(self.sam31_offload_state_to_cpu),
            async_loading_frames=False,
        )

        obj_to_det: Dict[int, Dict[str, Any]] = {}
        obj_segments: Dict[int, Dict[int, np.ndarray]] = {}
        obj_scores: Dict[int, float] = {}
        prompt_frames: List[int] = []

        try:
            with torch.autocast(
                device_type=self._sam2_device_type(),
                dtype=torch.bfloat16,
                enabled=str(self.device).startswith("cuda"),
            ):
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                for obj_id, det in enumerate(detections, start=1):
                    prompt_dets = self._select_prompt_support_detections(det, T=T)
                    added_any = False
                    for prompt_det in prompt_dets:
                        prompt_frame = int(np.clip(int(prompt_det.get("frame_idx", 0)), 0, max(T - 1, 0)))
                        try:
                            out_frame_idx, out_obj_ids, video_res_masks = self._add_sam2_object_prompt(
                                inference_state,
                                prompt_det,
                                obj_id=int(obj_id),
                                prompt_frame=prompt_frame,
                                H=H,
                                W=W,
                            )
                        except Exception as exc:
                            print(
                                f"[sam2] warning: skipping unusable prompt "
                                f"obj_id={obj_id} frame={prompt_frame}: {exc!r}"
                            )
                            continue
                        added_any = True
                        prompt_frames.append(prompt_frame)
                        self._record_tracker_outputs(
                            int(out_frame_idx),
                            out_obj_ids,
                            video_res_masks,
                            obj_segments,
                            obj_scores,
                        )
                    if added_any:
                        obj_to_det[int(obj_id)] = dict(det)
                        obj_scores[int(obj_id)] = float(det.get("confidence", 0.0))
                    else:
                        print(f"[sam2] warning: dropping object with no usable prompts obj_id={obj_id}")

                if not obj_to_det:
                    return {}, {}, {}

                min_prompt_frame = min(prompt_frames) if prompt_frames else 0
                max_prompt_frame = max(prompt_frames) if prompt_frames else 0

                def _consume(start_frame: int, reverse: bool, max_frames: int) -> None:
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=int(start_frame),
                        max_frame_num_to_track=max(int(max_frames), 1),
                        reverse=bool(reverse),
                    ):
                        self._record_tracker_outputs(
                            int(out_frame_idx),
                            out_obj_ids,
                            out_mask_logits,
                            obj_segments,
                            obj_scores,
                        )

                _consume(
                    start_frame=int(min_prompt_frame),
                    reverse=False,
                    max_frames=max(int(T) - int(min_prompt_frame), 1),
                )
                if int(max_prompt_frame) > 0:
                    _consume(
                        start_frame=int(max_prompt_frame),
                        reverse=True,
                        max_frames=int(max_prompt_frame) + 1,
                    )
        finally:
            try:
                self.video_predictor.reset_state(inference_state)
            except Exception as exc:
                print(f"[sam2] warning: failed to reset tracker state: {exc!r}")
            try:
                inference_state.clear()
            except Exception:
                pass
            del inference_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (
            {obj_id: frame_to_mask for obj_id, frame_to_mask in obj_segments.items() if frame_to_mask},
            obj_scores,
            obj_to_det,
        )

    @torch.inference_mode()
    def _track_sam3_detections_with_cutie(
        self,
        frame_resource: Any,
        detections_by_frame: Dict[int, List[Dict[str, Any]]],
        T: int,
        H: int,
        W: int,
        area: int,
    ) -> Tuple[
        Dict[int, Dict[int, np.ndarray]],
        Dict[int, Dict],
        Dict[int, Dict[int, np.ndarray]],
        Dict[int, Dict],
        List[Dict[str, Any]],
        Dict[str, Any],
    ]:
        if self.cutie_model is None or self.cutie_inference_core_cls is None:
            raise RuntimeError("Cutie backend is not initialized.")

        processor = self.cutie_inference_core_cls(self.cutie_model, cfg=self.cutie_model.cfg)
        processor.max_internal_size = int(self.cutie_max_internal_size)

        next_oid = 1
        oid_map: Dict[int, Dict] = {}
        active_masks: Dict[int, np.ndarray] = {}
        thing_segments: Dict[int, Dict[int, np.ndarray]] = {}
        structure_segments: Dict[int, Dict[int, np.ndarray]] = {}
        seeded_group_counts: Dict[int, int] = {}
        seeded_label_counts: Dict[Tuple[int, str], int] = {}
        debug = {
            "sam3_cutie_seed_frames": {},
            "sam3_cutie_num_seeded": 0,
            "sam3_cutie_max_internal_size": int(self.cutie_max_internal_size),
        }

        for t in range(T):
            frame_dets = list(detections_by_frame.get(int(t), []))
            frame_dets.sort(key=self._sam3_cutie_detection_priority)
            seed_mask = np.zeros((H, W), dtype=np.int64)
            new_objects: List[int] = []
            same_frame_dets: List[Dict[str, Any]] = []

            for det in frame_dets:
                if len(oid_map) >= self.max_thing_objects + self.sam31_max_structure_objects + 8:
                    break
                sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
                label = canonicalize_label(str(det.get("label", "")))
                group_budget, label_budget = self._sam3_cutie_seed_budget(sem_group, label)
                if group_budget <= 0 or label_budget <= 0:
                    continue
                if int(seeded_group_counts.get(sem_group, 0)) >= group_budget:
                    continue
                label_key = (sem_group, label)
                if int(seeded_label_counts.get(label_key, 0)) >= label_budget:
                    continue
                if self._sam3_cutie_is_duplicate_new_det(
                    det,
                    active_masks,
                    oid_map,
                    same_frame_dets,
                ):
                    continue
                mask_bool = np.asarray(det["mask"]).astype(bool)
                if mask_bool.sum() <= 0:
                    continue
                oid = next_oid
                next_oid += 1
                det_copy = dict(det)
                det_copy["frame_idx"] = int(t)
                oid_map[oid] = det_copy
                seeded_group_counts[sem_group] = int(seeded_group_counts.get(sem_group, 0)) + 1
                seeded_label_counts[label_key] = int(seeded_label_counts.get(label_key, 0)) + 1
                new_objects.append(int(oid))
                same_frame_dets.append(det_copy)
                seed_mask[mask_bool] = int(oid)

            image_t = self._read_frame_tensor_for_cutie(frame_resource, int(t))
            if new_objects:
                mask_t = torch.from_numpy(seed_mask).to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                    output_prob = processor.step(
                        image_t,
                        mask_t,
                        objects=new_objects,
                        force_permanent=True,
                    )
                debug["sam3_cutie_seed_frames"][int(t)] = list(new_objects)
                debug["sam3_cutie_num_seeded"] = int(debug["sam3_cutie_num_seeded"]) + len(new_objects)
            elif oid_map:
                with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                    output_prob = processor.step(image_t)
            else:
                continue

            obj_mask = processor.output_prob_to_mask(output_prob).detach().cpu().numpy()
            active_masks = {}
            for oid, meta in oid_map.items():
                mask = (obj_mask == int(oid))
                if mask.sum() < 32:
                    continue
                current_area_ratio = float(mask.sum()) / max(float(area), 1.0)
                active_masks[int(oid)] = mask.astype(np.uint8)
                sem_group = int(meta.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
                if sem_group == SEMANTIC_GROUP_STATIC_THING:
                    seed_area_ratio = float(meta.get("area_ratio", current_area_ratio))
                    max_static_area = min(0.34, max(0.12, 3.2 * seed_area_ratio))
                    if current_area_ratio > max_static_area:
                        active_masks.pop(int(oid), None)
                        continue
                if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR or _is_stuff(sem_group):
                    structure_segments.setdefault(int(t), {})[int(oid)] = mask.astype(np.uint8)
                else:
                    thing_segments.setdefault(int(t), {})[int(oid)] = mask.astype(np.uint8)

        thing_oid_map = {
            int(oid): meta
            for oid, meta in oid_map.items()
            if not (int(meta.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) == SEMANTIC_GROUP_STRUCTURE_ANCHOR
                    or _is_stuff(int(meta.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))))
        }
        structure_oid_map = {
            int(oid): meta
            for oid, meta in oid_map.items()
            if int(meta.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)) == SEMANTIC_GROUP_STRUCTURE_ANCHOR
            or _is_stuff(int(meta.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION)))
        }
        return thing_segments, thing_oid_map, structure_segments, structure_oid_map, list(oid_map.values()), debug

    def _run_sam3_cutie_pipeline(
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
        detections_by_frame = self._detect_sam3_cutie_keyframe_detections(
            frame_resource,
            discovery_indices,
            T,
            H,
            W,
            area,
        )
        if seed_detections_by_frame:
            for frame_idx, seed_dets in seed_detections_by_frame.items():
                merged = list(detections_by_frame.get(int(frame_idx), []))
                for det in seed_dets:
                    det_copy = dict(det)
                    det_copy["frame_idx"] = int(frame_idx)
                    det_copy["detector_source"] = str(
                        det_copy.get("detector_source", "overlap_seed")
                    )
                    merged.append(det_copy)
                detections_by_frame[int(frame_idx)] = self._nms(merged)

        if self.sam3_cutie_use_yoloe:
            self.detector.to_device()
            yoloe_dets = self._detect_and_filter_batch(
                frame_resource,
                discovery_indices,
                H,
                W,
                detector_image_paths=detector_image_paths,
            )
            self.detector.release_gpu()
            for frame_idx, dets in yoloe_dets.items():
                merged = list(detections_by_frame.get(int(frame_idx), []))
                merged.extend(dets)
                detections_by_frame[int(frame_idx)] = self._nms(merged)

        if not any(detections_by_frame.values()):
            return self._empty_output(T, H, W)

        (
            thing_segments,
            thing_oid_map,
            structure_segments,
            structure_oid_map,
            selected_dets,
            cutie_debug,
        ) = self._track_sam3_detections_with_cutie(
            frame_resource,
            detections_by_frame,
            T,
            H,
            W,
            area,
        )
        output = self._assemble(
            thing_segments,
            thing_oid_map,
            structure_segments,
            structure_oid_map,
            [],
            T,
            H,
            W,
        )
        output.debug.update({
            "sam_backend": "sam3_cutie",
            "tracker_backend": "cutie",
            "discovery_frames": list(discovery_indices),
            "sam3_cutie_prompt_frames": self._sam3_cutie_prompt_frames(discovery_indices, T),
            "sam3_cutie_detection_frames": {
                int(frame_idx): len(dets)
                for frame_idx, dets in detections_by_frame.items()
            },
            "sam3_cutie_selected_detections": len(selected_dets),
            "sam31_run_index": int(sam31_run_index),
            **cutie_debug,
        })
        return output

    def _run_sam3_sam2_pipeline(
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
        self._move_sam3_image_processor_to_device(self.device)
        detections_by_frame = self._detect_sam3_cutie_keyframe_detections(
            frame_resource,
            discovery_indices,
            T,
            H,
            W,
            area,
        )
        if seed_detections_by_frame:
            for frame_idx, seed_dets in seed_detections_by_frame.items():
                merged = list(detections_by_frame.get(int(frame_idx), []))
                for det in seed_dets:
                    det_copy = dict(det)
                    det_copy["frame_idx"] = int(frame_idx)
                    det_copy["detector_source"] = str(
                        det_copy.get("detector_source", "overlap_seed")
                    )
                    merged.append(det_copy)
                detections_by_frame[int(frame_idx)] = self._nms(merged)

        if self.sam3_cutie_use_yoloe:
            self.detector.to_device()
            yoloe_dets = self._detect_and_filter_batch(
                frame_resource,
                discovery_indices,
                H,
                W,
                detector_image_paths=detector_image_paths,
            )
            self.detector.release_gpu()
            for frame_idx, dets in yoloe_dets.items():
                merged = list(detections_by_frame.get(int(frame_idx), []))
                merged.extend(dets)
                detections_by_frame[int(frame_idx)] = self._nms(merged)

        if not any(detections_by_frame.values()):
            return self._empty_output(T, H, W)

        # SAM3 was only needed for keyframe discovery. Move it away before
        # SAM2.1 propagation so this backend can coexist with LoGeR in the
        # full pipeline without stacking two large video models on GPU.
        self._move_sam3_image_processor_to_device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        candidate_indices = sorted(
            {
                int(frame_idx)
                for frame_idx in detections_by_frame.keys()
                if 0 <= int(frame_idx) < T
            }
        )
        thing_dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        structure_dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        stuff_dets: List[Dict[str, Any]] = []
        for frame_idx, dets in detections_by_frame.items():
            thing_dets_by_frame[int(frame_idx)] = []
            structure_dets_by_frame[int(frame_idx)] = []
            for det in dets:
                sem_group = int(det.get("sem_group", SEMANTIC_GROUP_UNCERTAIN_REGION))
                if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                    structure_dets_by_frame[int(frame_idx)].append(det)
                elif _is_stuff(sem_group):
                    stuff_dets.append(det)
                else:
                    thing_dets_by_frame[int(frame_idx)].append(det)

        (
            selected_thing_dets,
            thing_segments,
            thing_oid_map,
            selected_structure_dets,
            structure_segments,
            structure_oid_map,
        ) = self._track_all_objects_efficientsam3_tracker(
            frame_resource,
            thing_dets_by_frame,
            structure_dets_by_frame,
            candidate_indices,
            T,
            H,
            W,
            area,
        )

        tracked_structure_labels = {str(det["label"]) for det in selected_structure_dets}
        if self.tracker_backend in {"sam2", "edgetam"}:
            # Do not video-track structures with EdgeTAM: planar regions drift
            # into people.  Still keep reliable keyframe structure detections
            # as flow-propagated STUFF so floor/wall/ceiling are not silently
            # dropped from the efficient path.
            reliable_structure_labels = {"floor", "wall", "ceiling"}
            for frame_idx in candidate_indices:
                for det in structure_dets_by_frame.get(frame_idx, []):
                    label_l = str(det.get("label", "")).strip().lower()
                    if label_l not in reliable_structure_labels:
                        continue
                    area_ratio = float(det.get("area_ratio", 0.0))
                    confidence = float(det.get("confidence", det.get("cluster_confidence", 0.0)))
                    if area_ratio < 0.010 or area_ratio > 0.62:
                        continue
                    if confidence < 0.16 and area_ratio < 0.045:
                        continue
                    stuff_dets.append(det)
        elif tracked_structure_labels:
            stuff_dets.extend(
                d
                for frame_idx in candidate_indices
                for d in structure_dets_by_frame.get(frame_idx, [])
                if str(d["label"]) not in tracked_structure_labels
            )
        else:
            stuff_dets.extend(
                d
                for frame_idx in candidate_indices
                for d in structure_dets_by_frame.get(frame_idx, [])
            )

        stuff_entries = self._build_stuff_entries(
            stuff_dets,
            frame_resource,
            T,
            H,
            W,
        )
        self._subtract_thing_masks_from_stuff_entries(
            stuff_entries,
            thing_segments,
            thing_oid_map,
        )
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
            "sam_backend": f"sam3_{self.tracker_backend}",
            "tracker_backend": str(self.tracker_backend),
            "discovery_frames": list(discovery_indices),
            "candidate_frames": list(candidate_indices),
            "sam3_prompt_frames": self._sam3_cutie_prompt_frames(discovery_indices, T),
            "sam3_yoloe_detection_frames": {
                int(frame_idx): len(dets)
                for frame_idx, dets in detections_by_frame.items()
            },
            "selected_thing_detections": len(selected_thing_dets),
            "selected_structure_detections": len(selected_structure_dets),
            "sam31_run_index": int(sam31_run_index),
        })
        return output

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
        if self.tracker_backend in {"sam2", "edgetam"}:
            return self._run_sam3_sam2_pipeline(
                frame_resource,
                T,
                H,
                W,
                discovery_indices,
                seed_detections_by_frame=seed_detections_by_frame,
                detector_image_paths=detector_image_paths,
                sam31_run_index=sam31_run_index,
            )
        if self.tracker_backend == "cutie":
            return self._run_sam3_cutie_pipeline(
                frame_resource,
                T,
                H,
                W,
                discovery_indices,
                seed_detections_by_frame=seed_detections_by_frame,
                detector_image_paths=detector_image_paths,
                sam31_run_index=sam31_run_index,
            )

        area = H * W

        self.detector.to_device()
        detections_by_frame = self._detect_and_filter_batch(
            frame_resource,
            discovery_indices,
            H,
            W,
            detector_image_paths=detector_image_paths,
        )
        for frame_idx in discovery_indices:
            detections_by_frame.setdefault(int(frame_idx), [])

        seed_drop_debug: List[Dict[str, Any]] = []
        if seed_detections_by_frame:
            for frame_idx, seed_dets in seed_detections_by_frame.items():
                frame_idx = int(frame_idx)
                if not (0 <= frame_idx < T):
                    continue
                detector_dets = list(detections_by_frame.get(frame_idx, []))
                merged = list(detector_dets)
                for det in seed_dets:
                    det_copy = dict(det)
                    det_copy["frame_idx"] = frame_idx
                    if not self._seed_person_has_detector_support(det_copy, detector_dets):
                        if len(seed_drop_debug) < 100:
                            seed_drop_debug.append({
                                "frame_idx": int(frame_idx),
                                "label": str(det_copy.get("label", "")),
                                "seed_global_track_idx": det_copy.get("seed_global_track_idx", None),
                                "area_ratio": float(det_copy.get("area_ratio", 0.0)),
                            })
                        continue
                    merged.append(det_copy)
                detections_by_frame[frame_idx] = self._nms(merged)

        candidate_indices = sorted(
            {
                int(frame_idx)
                for frame_idx in detections_by_frame.keys()
                if 0 <= int(frame_idx) < T
            }
        )
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
                if sem_group == SEMANTIC_GROUP_STRUCTURE_ANCHOR:
                    structure_dets_by_frame[frame_idx].append(det)
                elif _is_stuff(sem_group):
                    stuff_dets.append(det)
                else:
                    thing_dets_by_frame[frame_idx].append(det)
                    if (
                        self.sam31_nontext_sparse_support
                        and not self._uses_sam31_text_tracking(str(det.get("label", "")))
                    ):
                        det_copy = dict(det)
                        det_copy["detector_source"] = str(
                            det_copy.get("detector_source", "detector_sparse_support")
                        )
                        sparse_thing_support_dets.append(det_copy)

        self.detector.release_gpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prompt_thing_dets: Dict[int, List[Dict[str, Any]]] = {}

        structure_prompt_active = False
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

        (
            selected_thing_dets,
            thing_segments,
            thing_oid_map,
            selected_structure_dets,
            structure_segments,
            structure_oid_map,
        ) = self._track_all_objects_efficientsam3_tracker(
            frame_resource,
            thing_dets_by_frame,
            structure_dets_by_frame,
            candidate_indices,
            T,
            H,
            W,
            area,
        )
        if sparse_thing_support_dets:
            sparse_thing_support_dets = self._filter_sparse_support_against_tracked(
                sparse_thing_support_dets,
                thing_segments,
                thing_oid_map,
            )
            self._append_sparse_detector_thing_tracks(
                sparse_thing_support_dets,
                discovery_indices,
                thing_segments,
                thing_oid_map,
                area,
            )

        tracked_structure_labels = {str(det["label"]) for det in selected_structure_dets}
        if tracked_structure_labels:
            stuff_dets.extend(
                d
                for frame_idx in candidate_indices
                for d in structure_dets_by_frame.get(frame_idx, [])
                if str(d["label"]) not in tracked_structure_labels
            )
        else:
            stuff_dets.extend(
                d
                for frame_idx in candidate_indices
                for d in structure_dets_by_frame.get(frame_idx, [])
            )

        stuff_entries = self._build_stuff_entries(
            stuff_dets, frame_resource, T, H, W,
        )
        self._subtract_thing_masks_from_stuff_entries(
            stuff_entries,
            thing_segments,
            thing_oid_map,
        )
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
            "candidate_frames": list(candidate_indices),
            "num_discovery_frames": len(discovery_indices),
            "sam31_run_index": int(sam31_run_index),
            "sam31_structure_prompt_active": bool(structure_prompt_active),
            "sam31_structure_prompt_chunk_stride": int(
                self.sam31_structure_prompt_chunk_stride
            ),
            "efficient_tracker_api": "sam2_box_prompt",
            "efficient_detector_batch_discovery": True,
            "selected_thing_detections": len(selected_thing_dets),
            "selected_structure_detections": len(selected_structure_dets),
            "efficient_text_prompt_detection_frames": {
                int(frame_idx): len(dets)
                for frame_idx, dets in prompt_thing_dets.items()
            },
            "efficient_dropped_person_seeds": seed_drop_debug,
        })
        return output


__all__ = [
    "DEFAULT_EFFICIENTSAM3_FILENAME",
    "DEFAULT_EFFICIENTSAM3_REPO_ID",
    "DEFAULT_EDGETAM_CHECKPOINT",
    "DEFAULT_EDGETAM_MODEL_CFG",
    "DEFAULT_SAM2_CHECKPOINT",
    "DEFAULT_SAM2_MODEL_CFG",
    "DEFAULT_SAM3_CHECKPOINT",
    "EfficientVideoMaskletFrontend",
    "MaskletOutput",
    "SEMANTIC_GROUP_NAMES",
]
