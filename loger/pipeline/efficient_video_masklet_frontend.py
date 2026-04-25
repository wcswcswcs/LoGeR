from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download

from .video_masklet_frontend import (
    MaskletOutput,
    SEMANTIC_GROUP_NAMES,
    THING_PROMPTS,
    STUFF_PROMPTS,
    VideoMaskletFrontend,
    YOLOEDetector,
)

DEFAULT_EFFICIENTSAM3_REPO_ID = "Simon7108528/EfficientSAM3"
DEFAULT_EFFICIENTSAM3_FILENAME = "stage1_all_converted/efficient_sam3_efficientvit_s.pt"


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


class EfficientVideoMaskletFrontend(VideoMaskletFrontend):
    """Stage C front-end backed by EfficientSAM3 + YOLOE.

    This class intentionally reuses the existing SAM3-style prompt / propagation
    logic from :class:`VideoMaskletFrontend`. The only architectural difference
    is that the video predictor is built from ``third_party/efficientsam3``.
    """

    @classmethod
    def from_config(
        cls,
        *,
        device: str = "cuda",
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
            raise ValueError("EfficientSAM3 video predictor currently requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but EfficientSAM3 requires CUDA.")

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
        detector = YOLOEDetector(model_path=yoloe_model, device=device)

        return cls(
            video_predictor=video_predictor,
            detector=detector,
            sam_backend="efficientsam3",
            device=device,
            thing_prompts=thing_prompts or THING_PROMPTS,
            stuff_prompts=stuff_prompts or STUFF_PROMPTS,
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


__all__ = [
    "DEFAULT_EFFICIENTSAM3_FILENAME",
    "DEFAULT_EFFICIENTSAM3_REPO_ID",
    "EfficientVideoMaskletFrontend",
    "MaskletOutput",
    "SEMANTIC_GROUP_NAMES",
]
