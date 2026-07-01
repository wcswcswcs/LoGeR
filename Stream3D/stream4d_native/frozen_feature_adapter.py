from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureMap:
    features: np.ndarray
    image_height: int
    image_width: int
    backend: str
    patch_size: int | None = None


def _as_rgb_array(frame: Any) -> np.ndarray:
    if hasattr(frame, "convert"):
        frame = np.asarray(frame.convert("RGB"))
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("frame must be HxWx3 RGB-like data")
    arr = arr[:, :, :3].astype(np.float32)
    if arr.max(initial=0.0) > 2.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, eps)


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.shape == (height, width):
        return mask_u8.astype(bool)
    resized = cv2.resize(mask_u8, (int(width), int(height)), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not np.any(resized) and np.any(mask_u8):
        ys, xs = np.nonzero(mask_u8)
        cy = int(np.clip(round(float(ys.mean()) * float(height) / max(mask_u8.shape[0], 1)), 0, int(height) - 1))
        cx = int(np.clip(round(float(xs.mean()) * float(width) / max(mask_u8.shape[1], 1)), 0, int(width) - 1))
        resized[cy, cx] = True
    return resized


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary & mask


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class FrozenFeatureAdapter:
    """Thin frozen-feature interface used by v42 audits.

    The default `rgb_stats` backend is deterministic and dependency-light, so
    unit tests can validate the pooling/affinity contract. `dinov2_timm`
    performs a real frozen DINOv2 forward pass from local checkpoints.
    """

    def __init__(
        self,
        *,
        backend: str = "rgb_stats",
        device: str = "cpu",
        checkpoint: str | None = None,
        short_side: int = 518,
        radio_lang_model: str = "siglip2",
        radio_lang_align: bool = False,
        radio_slide_crop: int = 0,
        radio_slide_stride: int = 224,
    ) -> None:
        self.backend = str(backend)
        self.device = str(device)
        self.checkpoint = checkpoint
        self.short_side = int(short_side)
        self.radio_lang_model = str(radio_lang_model)
        self.radio_lang_align = bool(radio_lang_align)
        self.radio_slide_crop = int(radio_slide_crop)
        self.radio_slide_stride = int(radio_slide_stride)
        self._model: Any | None = None
        self._torch: Any | None = None

    def extract_dense_features(self, frame: Any) -> FeatureMap:
        if self.backend == "rgb_stats":
            return self._extract_rgb_stats(frame)
        if self.backend == "dinov2_timm":
            return self._extract_dinov2_timm(frame)
        if self.backend == "radio_radseg":
            return self._extract_radio_radseg(frame)
        raise ValueError(f"unsupported frozen feature backend: {self.backend}")

    def pool_mask_feature(self, feature_map: FeatureMap, mask: np.ndarray) -> np.ndarray:
        features = np.asarray(feature_map.features, dtype=np.float32)
        mask_small = _resize_mask(mask, features.shape[0], features.shape[1])
        if not np.any(mask_small):
            return np.zeros((features.shape[-1],), dtype=np.float32)
        pooled = features[mask_small].mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(pooled))
        return pooled / norm if norm > 1e-8 else pooled

    def compute_boundary_contrast(self, feature_map: FeatureMap, mask: np.ndarray) -> float:
        features = np.asarray(feature_map.features, dtype=np.float32)
        mask_small = _resize_mask(mask, features.shape[0], features.shape[1])
        boundary = _mask_boundary(mask_small)
        if not np.any(boundary) or not np.any(~mask_small):
            return 0.0
        inner = features[boundary].mean(axis=0)
        outer = features[~mask_small].mean(axis=0)
        return float(1.0 - _cosine(inner, outer))

    def compute_token_affinity(self, token_i: Any, token_j: Any) -> float:
        return _cosine(_token_feature(token_i), _token_feature(token_j))

    def _extract_rgb_stats(self, frame: Any) -> FeatureMap:
        rgb = _as_rgb_array(frame)
        h, w = rgb.shape[:2]
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])[:, :, None]
        rg = (rgb[:, :, 0:1] - rgb[:, :, 1:2])
        by = (rgb[:, :, 2:3] - 0.5 * (rgb[:, :, 0:1] + rgb[:, :, 1:2]))
        features = np.concatenate([rgb, luma, rg, by, xx[:, :, None], yy[:, :, None]], axis=-1).astype(np.float32)
        return FeatureMap(_l2_normalize(features), h, w, backend=self.backend, patch_size=1)

    def _load_dinov2(self) -> Any:
        if self._model is not None:
            return self._model
        import timm
        import torch

        model = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0)
        if self.checkpoint:
            state = torch.load(str(self.checkpoint), map_location="cpu")
            model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        self._torch = torch
        self._model = model
        return model

    def _extract_dinov2_timm(self, frame: Any) -> FeatureMap:
        import cv2

        rgb = _as_rgb_array(frame)
        h, w = rgb.shape[:2]
        model = self._load_dinov2()
        torch = self._torch
        assert torch is not None
        resized = cv2.resize(rgb, (self.short_side, self.short_side), interpolation=cv2.INTER_AREA).astype(np.float32)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)).float()[None].to(self.device)
        with torch.inference_mode():
            out = model.forward_features(tensor)
            tokens = out["x_norm_patchtokens"] if isinstance(out, dict) and "x_norm_patchtokens" in out else out[:, 1:, :]
            tokens = torch.nn.functional.normalize(tokens.float(), dim=-1).squeeze(0).detach().cpu().numpy()
        grid = int(round(np.sqrt(tokens.shape[0])))
        if grid * grid != int(tokens.shape[0]):
            raise ValueError(f"unexpected DINO token count: {tokens.shape[0]}")
        features = tokens.reshape(grid, grid, tokens.shape[-1]).astype(np.float32)
        return FeatureMap(features, h, w, backend=self.backend, patch_size=max(1, int(round(self.short_side / grid))))

    def _load_radio_radseg(self) -> Any:
        if self._model is not None:
            return self._model
        radio_root = Path(__file__).resolve().parents[2] / "third_party" / "RADIO-ViPE"
        if str(radio_root) not in sys.path:
            sys.path.insert(0, str(radio_root))
        # RADIO-ViPE's JIT extension reads CONDA_PREFIX/include for Eigen.
        # Shell sessions may keep CONDA_PREFIX pointed at base conda even when
        # this adapter runs inside the project env, so use the live interpreter.
        os.environ["CONDA_PREFIX"] = sys.prefix
        import torch
        from vipe.priors.embedding.radseg_encoder import RADSegEncoder

        checkpoint = self.checkpoint or locate_default_radio_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("no local RADIO/RADSeg checkpoint found")

        original_load = torch.load

        def _compat_load(*args: Any, **kwargs: Any) -> Any:
            if args and str(args[0]) == str(checkpoint) and "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = _compat_load
        try:
            model = RADSegEncoder(
                device=self.device,
                model_version=str(checkpoint),
                lang_model=self.radio_lang_model,
                return_radio_features=True,
                compile=False,
                amp=False,
                predict=False,
                slide_crop=self.radio_slide_crop,
                slide_stride=self.radio_slide_stride,
            )
        finally:
            torch.load = original_load
        self._torch = torch
        self._model = model
        self.checkpoint = str(checkpoint)
        return model

    def _extract_radio_radseg(self, frame: Any) -> FeatureMap:
        rgb = _as_rgb_array(frame)
        h, w = rgb.shape[:2]
        model = self._load_radio_radseg()
        torch = self._torch
        assert torch is not None
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()[None].to(self.device)
        with torch.inference_mode():
            feat = model.encode_image_to_feat_map(tensor)
            if self.radio_lang_align:
                feat = model.align_spatial_features_with_language(feat, onehot=False)
            features = feat.squeeze(0).permute(1, 2, 0).contiguous().float().detach().cpu().numpy()
        features = _l2_normalize(features.astype(np.float32))
        return FeatureMap(features, h, w, backend=self.backend, patch_size=int(model.model.patch_size))


def _token_feature(token: Any) -> np.ndarray:
    if isinstance(token, dict):
        if "feature" in token:
            return np.asarray(token["feature"], dtype=np.float32)
        if "pooled_feature" in token:
            return np.asarray(token["pooled_feature"], dtype=np.float32)
    if hasattr(token, "feature"):
        return np.asarray(getattr(token, "feature"), dtype=np.float32)
    return np.asarray(token, dtype=np.float32)


def locate_default_dinov2_checkpoint() -> str | None:
    for path in [
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"),
        Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"),
    ]:
        if path.exists():
            return str(path)
    return None


def locate_default_cradio_checkpoint() -> str | None:
    for path in [
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/c-radio_v3-b_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/c-radio-v3_l_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/c-radio_v3-h_half.pth.tar"),
    ]:
        if path.exists():
            return str(path)
    return None


def locate_default_radio_checkpoint() -> str | None:
    for path in [
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/c-radio_v3-b_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/radio-v2.5-l_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/radio-v2.5-b_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/radio_v2.5-h.pth.tar"),
    ]:
        if path.exists():
            return str(path)
    return None
