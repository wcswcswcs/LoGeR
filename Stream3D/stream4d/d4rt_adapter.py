from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .carrier_store import CarrierBatch


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


class D4RTAdapter:
    def __init__(
        self,
        d4rt_root: str | Path,
        model_config: str | Path,
        ckpt_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.d4rt_root = Path(d4rt_root).resolve()
        self.model_config = Path(model_config).resolve()
        self.ckpt_path = Path(ckpt_path).resolve()
        self.last_infer_diagnostics: dict[str, float | int | str] = {}
        if not self.d4rt_root.exists():
            raise FileNotFoundError(f"D4RT root does not exist: {self.d4rt_root}")
        if not self.model_config.exists():
            raise FileNotFoundError(f"D4RT model config does not exist: {self.model_config}")
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                "D4RT checkpoint does not exist: "
                f"{self.ckpt_path}. Download/provide the matching .ckpt and retry."
            )

        sys.path.insert(0, str(self.d4rt_root))
        from src.core.config import load_yaml_config
        from src.model.builder import build_model

        self.cfg = load_yaml_config(self.model_config)
        self.image_hw = tuple(int(v) for v in self.cfg.get_path("model.input.image_size", [256, 256]))
        if len(self.image_hw) != 2:
            raise ValueError(f"model.input.image_size must be [H,W], got {self.image_hw}")
        self.clip_frames = int(self.cfg.get_path("model.input.clip_frames", 48))
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA for D4RTAdapter, but torch.cuda.is_available() is False")
        self.device = torch.device(device)

        self.model = build_model(self.cfg["model"])
        self.checkpoint_size_bytes = int(self.ckpt_path.stat().st_size)
        payload = torch.load(self.ckpt_path, map_location="cpu")
        state = self._unwrap_state_dict(payload)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"D4RT checkpoint state mismatch: missing={len(missing)} unexpected={len(unexpected)}; "
                f"first_missing={missing[:5]} first_unexpected={unexpected[:5]}"
            )
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("state_dict", "model", "module", "network", "net"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            if payload and all(torch.is_tensor(v) for v in payload.values()):
                return payload
        raise ValueError("No usable D4RT state_dict found in checkpoint payload")

    def _resize_video(self, video_rgb_uint8: np.ndarray) -> np.ndarray:
        h, w = self.image_hw
        resized = [
            cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[0] >= h else cv2.INTER_LINEAR)
            for frame in video_rgb_uint8
        ]
        return np.stack(resized, axis=0).astype(np.uint8)

    def _decode_chunks(
        self,
        video_tensor: torch.Tensor,
        aspect_tensor: torch.Tensor,
        memory: torch.Tensor,
        query_base: dict[str, np.ndarray],
        query_chunk_size: int,
    ) -> dict[str, np.ndarray]:
        out: dict[str, list[np.ndarray]] = {}
        num_queries = int(query_base["u"].shape[0])
        step = max(1, int(query_chunk_size))
        for start in range(0, num_queries, step):
            end = min(num_queries, start + step)
            query = {
                key: torch.from_numpy(value[start:end]).to(
                    device=self.device,
                    dtype=torch.float32 if key in ("u", "v") else torch.long,
                ).view(1, -1)
                for key, value in query_base.items()
            }
            pred = self.model.decode_queries(video=video_tensor, query=query, memory=memory)
            for key, value in pred.items():
                arr = value[0].detach().float().cpu().numpy()
                out.setdefault(key, []).append(arr)
        return {key: np.concatenate(chunks, axis=0) for key, chunks in out.items()}

    def infer_carriers(
        self,
        video_rgb_uint8: np.ndarray,
        src_uv_norm: np.ndarray,
        src_frame_local: np.ndarray,
        target_frames_local: np.ndarray | None = None,
        t_cam_local: int | None = None,
        query_chunk_size: int = 2048,
        carrier_id: np.ndarray | None = None,
        src_frame_global: np.ndarray | None = None,
        src_xy: np.ndarray | None = None,
        src_mask_id: np.ndarray | None = None,
    ) -> CarrierBatch:
        video_rgb_uint8 = np.asarray(video_rgb_uint8)
        if video_rgb_uint8.ndim != 4 or video_rgb_uint8.shape[-1] != 3:
            raise ValueError(f"video_rgb_uint8 must have shape [T,H,W,3], got {video_rgb_uint8.shape}")
        src_uv_norm = np.asarray(src_uv_norm, dtype=np.float32).reshape(-1, 2)
        src_frame_local = np.asarray(src_frame_local, dtype=np.int64).reshape(-1)
        num_carriers = int(src_uv_norm.shape[0])
        if src_frame_local.shape[0] != num_carriers:
            raise ValueError("src_frame_local length must equal src_uv_norm length")
        if target_frames_local is None:
            target_frames_local = np.arange(video_rgb_uint8.shape[0], dtype=np.int64)
        target_frames_local = np.asarray(target_frames_local, dtype=np.int64).reshape(-1)
        if carrier_id is None:
            carrier_id = np.arange(num_carriers, dtype=np.int64)
        carrier_id = np.asarray(carrier_id, dtype=np.int64).reshape(-1)

        target_count = int(target_frames_local.shape[0])
        empty_xyz = np.empty((target_count, num_carriers, 3), dtype=np.float32)
        empty_uv = np.empty((target_count, num_carriers, 2), dtype=np.float32)
        if num_carriers == 0 or target_count == 0:
            empty_xyz.fill(np.nan)
            empty_uv.fill(np.nan)
            return CarrierBatch(
                carrier_id=carrier_id,
                src_frame=src_frame_local,
                src_uv=src_uv_norm,
                xyz_ref=empty_xyz,
                uv_pred=empty_uv,
                visibility_prob=np.zeros((target_count, num_carriers), dtype=np.float32),
                confidence_prob=np.zeros((target_count, num_carriers), dtype=np.float32),
                valid=np.zeros((target_count, num_carriers), dtype=bool),
                xyz_local=empty_xyz.copy(),
                src_frame_global=src_frame_global,
                src_xy=src_xy,
                src_mask_id=src_mask_id,
            )

        if int(video_rgb_uint8.shape[0]) > self.clip_frames:
            raise ValueError(
                f"D4RTAdapter window has {video_rgb_uint8.shape[0]} frames, "
                f"but checkpoint supports clip_frames={self.clip_frames}."
            )

        resize_t0 = time.time()
        video_model = self._resize_video(video_rgb_uint8)
        seconds_resize = float(time.time() - resize_t0)
        native_h, native_w = int(video_rgb_uint8.shape[1]), int(video_rgb_uint8.shape[2])
        aspect = np.asarray([[float(native_w) / float(max(1, native_h))]], dtype=np.float32)
        video_tensor = (
            torch.from_numpy(video_model)
            .to(device=self.device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            / 255.0
        )
        aspect_tensor = torch.from_numpy(aspect).to(device=self.device, dtype=torch.float32)

        repeated_uv = np.tile(src_uv_norm, (target_count, 1)).astype(np.float32)
        t_src = np.tile(src_frame_local, target_count).astype(np.int64)
        t_tgt = np.repeat(target_frames_local, num_carriers).astype(np.int64)
        t_cam = t_tgt.copy() if t_cam_local is None else np.full_like(t_tgt, int(t_cam_local))
        query_local = {
            "u": repeated_uv[:, 0],
            "v": repeated_uv[:, 1],
            "t_src": t_src,
            "t_tgt": t_tgt,
            "t_cam": t_cam,
        }
        query_ref = dict(query_local)
        query_ref["t_cam"] = np.zeros_like(t_tgt)

        with torch.inference_mode():
            encode_t0 = time.time()
            memory = self.model.encode_video(video=video_tensor, aspect_ratio=aspect_tensor)
            seconds_encode = float(time.time() - encode_t0)
            decode_local_t0 = time.time()
            pred_local = self._decode_chunks(video_tensor, aspect_tensor, memory, query_local, query_chunk_size)
            seconds_decode_local = float(time.time() - decode_local_t0)
            decode_ref_t0 = time.time()
            pred_ref = self._decode_chunks(video_tensor, aspect_tensor, memory, query_ref, query_chunk_size)
            seconds_decode_ref = float(time.time() - decode_ref_t0)

        def reshape(name: str, pred: dict[str, np.ndarray], tail: tuple[int, ...]) -> np.ndarray:
            arr = pred[name].astype(np.float32, copy=False)
            return arr.reshape(target_count, num_carriers, *tail)

        uv_pred = reshape("uv_2d", pred_local, (2,))
        xyz_local = reshape("xyz_3d", pred_local, (3,))
        xyz_ref = reshape("xyz_3d", pred_ref, (3,))
        visibility_prob = _sigmoid_np(reshape("visibility", pred_local, ()) if pred_local["visibility"].ndim == 1 else pred_local["visibility"].reshape(target_count, num_carriers))
        confidence_prob = _sigmoid_np(reshape("confidence", pred_local, ()) if pred_local["confidence"].ndim == 1 else pred_local["confidence"].reshape(target_count, num_carriers))
        valid = np.isfinite(uv_pred).all(axis=-1) & np.isfinite(xyz_ref).all(axis=-1) & np.isfinite(xyz_local).all(axis=-1)
        self.last_infer_diagnostics = {
            "checkpoint_size_bytes": int(self.checkpoint_size_bytes),
            "clip_frames": int(self.clip_frames),
            "num_input_frames": int(video_rgb_uint8.shape[0]),
            "num_target_frames": int(target_count),
            "num_carriers": int(num_carriers),
            "num_queries_per_decode": int(target_count * num_carriers),
            "query_chunk_size": int(query_chunk_size),
            "seconds_resize": seconds_resize,
            "seconds_d4rt_encode": seconds_encode,
            "seconds_d4rt_decode_local": seconds_decode_local,
            "seconds_d4rt_decode_ref": seconds_decode_ref,
            "seconds_d4rt_decode": float(seconds_decode_local + seconds_decode_ref),
        }

        return CarrierBatch(
            carrier_id=carrier_id,
            src_frame=src_frame_local,
            src_uv=src_uv_norm,
            xyz_ref=xyz_ref,
            uv_pred=uv_pred,
            visibility_prob=visibility_prob,
            confidence_prob=confidence_prob,
            valid=valid,
            xyz_local=xyz_local,
            src_frame_global=src_frame_global,
            src_xy=src_xy,
            src_mask_id=src_mask_id,
        )
