#!/usr/bin/env python3
"""Shared utilities for ACL2 v70 RADIO sidecar tools."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_chunks(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


@dataclass(frozen=True)
class StageChunk:
    chunk_id: int
    start_frame: int
    end_frame: int
    chunk_dir: Path
    masklet_path: Path
    manifest_path: Path
    num_frames: int


def find_stage_chunk(stage_c_cache: Path, chunk_id: int) -> StageChunk:
    candidates = sorted(stage_c_cache.glob(f"chunk_{int(chunk_id):03d}_*"))
    if not candidates:
        raise FileNotFoundError(f"no stage-c chunk directory for chunk {chunk_id}: {stage_c_cache}")
    chunk_dir = candidates[0]
    manifest_path = chunk_dir / "manifest.json"
    masklet_path = chunk_dir / "masklet.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not masklet_path.exists():
        raise FileNotFoundError(f"missing masklet: {masklet_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = int(manifest.get("start_frame"))
    end = int(manifest.get("end_frame"))
    return StageChunk(
        chunk_id=int(manifest.get("chunk_idx", chunk_id)),
        start_frame=start,
        end_frame=end,
        chunk_dir=chunk_dir,
        masklet_path=masklet_path,
        manifest_path=manifest_path,
        num_frames=max(0, end - start),
    )


def image_path_for_frame(image_dir: Path, frame_id: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = image_dir / f"{int(frame_id):06d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for frame {frame_id:06d} under {image_dir}")


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_stage_semantic(masklet_path: Path) -> dict[str, Any]:
    payload = torch_load(masklet_path)
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict payload in {masklet_path}, got {type(payload)}")
    sem = payload.get("semantic_segmentation")
    if not isinstance(sem, dict):
        raise KeyError(f"missing semantic_segmentation in {masklet_path}")
    return sem


def ids_containing(label_names: Sequence[str], words: Iterable[str]) -> list[int]:
    keys = [str(w).lower() for w in words]
    out: list[int] = []
    for idx, label in enumerate(label_names):
        lowered = str(label).lower()
        if any(key in lowered for key in keys):
            out.append(idx)
    return out


def resize_nearest(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(shape_hw[0]), int(shape_hw[1])
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


def resize_linear(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(shape_hw[0]), int(shape_hw[1])
    if arr.shape[:2] == (h, w):
        return arr
    src = arr.astype(np.float32)
    if src.ndim == 3 and src.shape[2] > 4:
        chans = [cv2.resize(src[:, :, c], (w, h), interpolation=cv2.INTER_AREA) for c in range(src.shape[2])]
        return np.stack(chans, axis=2).astype(np.float32)
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)


def pool_semantic_to_grid(labels: Any, confidence: Any | None, label_names: Sequence[str], grid_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    labels_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    conf_np = None
    if confidence is not None:
        conf_np = confidence.detach().cpu().numpy() if hasattr(confidence, "detach") else np.asarray(confidence)
    t = int(labels_np.shape[0])
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    pooled = np.zeros((t, gh, gw), dtype=np.int32)
    purity = np.zeros((t, gh, gw), dtype=np.float32)
    conf_out = np.ones((t, gh, gw), dtype=np.float32)
    for frame_idx in range(t):
        small_label = resize_nearest(labels_np[frame_idx].astype(np.int32), (gh, gw)).astype(np.int32)
        pooled[frame_idx] = small_label
        # Nearest pooling is used for label identity; purity is a local agreement proxy.
        expanded = resize_nearest(small_label, labels_np[frame_idx].shape[:2]).astype(np.int32)
        local_match = (expanded == labels_np[frame_idx]).astype(np.float32)
        purity[frame_idx] = resize_linear(local_match, (gh, gw)).astype(np.float32)
        if conf_np is not None:
            conf_out[frame_idx] = resize_linear(conf_np[frame_idx].astype(np.float32), (gh, gw)).astype(np.float32)
    dynamic_ids = ids_containing(label_names, ("person", "car", "truck", "bus", "bicycle", "motorcycle"))
    sky_ids = ids_containing(label_names, ("sky",))
    lowtrust_ids = ids_containing(label_names, ("grass", "tree", "vegetation", "mountain"))
    dynamic = np.isin(pooled, np.asarray(dynamic_ids, dtype=np.int32)).astype(np.float32) if dynamic_ids else np.zeros_like(conf_out)
    sky = np.isin(pooled, np.asarray(sky_ids, dtype=np.int32)).astype(np.float32) if sky_ids else np.zeros_like(conf_out)
    lowtrust_label = np.isin(pooled, np.asarray(lowtrust_ids, dtype=np.int32)).astype(np.float32) if lowtrust_ids else np.zeros_like(conf_out)
    return {
        "label": pooled,
        "purity": np.clip(purity, 0.0, 1.0),
        "confidence": np.clip(conf_out, 0.0, 1.0),
        "dynamic": dynamic,
        "sky": sky,
        "lowtrust_label": lowtrust_label,
    }


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def fit_pca(samples: np.ndarray, pca_dim: int, max_fit_tokens: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(samples, dtype=np.float32).reshape(-1, int(samples.shape[-1]))
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    if x.shape[0] < 3:
        raise ValueError(f"need >=3 finite feature rows for PCA, got {x.shape[0]}")
    if x.shape[0] > int(max_fit_tokens):
        idx = rng.choice(x.shape[0], size=int(max_fit_tokens), replace=False)
        x = x[idx]
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, svals, vh = np.linalg.svd(xc, full_matrices=False)
    k = min(int(pca_dim), int(vh.shape[0]))
    basis = vh[:k].astype(np.float32)
    var = (svals.astype(np.float64) ** 2)
    ratio = (var / max(float(var.sum()), 1e-12))[:k].astype(np.float32)
    return {"mean": mean.reshape(-1).astype(np.float32), "basis": basis, "explained_variance_ratio": ratio}


def apply_pca(features: np.ndarray, pca: Mapping[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    flat = x.reshape(-1, int(x.shape[-1]))
    mean = np.asarray(pca["mean"], dtype=np.float32).reshape(1, -1)
    basis = np.asarray(pca["basis"], dtype=np.float32)
    out = (flat - mean) @ basis.T
    return out.reshape(*x.shape[:-1], basis.shape[0]).astype(np.float32)


def robust01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo, hi = np.percentile(vals, [5.0, 95.0])
    if abs(float(hi - lo)) < 1e-8:
        lo, hi = float(vals.min()), float(vals.max())
    if abs(float(hi - lo)) < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (float(hi - lo) + 1e-6), 0.0, 1.0).astype(np.float32)


def connected_components_from_features(
    features: np.ndarray,
    *,
    seed: int,
    min_clusters: int = 8,
    max_clusters: int = 64,
    max_components: int = 200,
) -> np.ndarray:
    import cv2
    from sklearn.cluster import MiniBatchKMeans

    feat = l2_normalize(features)
    h, w, d = feat.shape
    n_tokens = h * w
    n_clusters = int(np.clip(round(n_tokens / 80.0), min_clusters, max_clusters))
    n_clusters = max(2, min(n_clusters, n_tokens))
    flat = feat.reshape(n_tokens, d)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=int(seed), n_init=3, batch_size=4096)
    cluster = km.fit_predict(flat).reshape(h, w).astype(np.int32)
    comp = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for label in sorted(np.unique(cluster).tolist()):
        mask = (cluster == int(label)).astype(np.uint8)
        ncc, cc = cv2.connectedComponents(mask, connectivity=4)
        for cid in range(1, int(ncc)):
            comp[cc == cid] = next_id
            next_id += 1
    if component_count(comp) > int(max_components):
        return (cluster + 1).astype(np.int32)
    return comp.astype(np.int32)


def component_boundary(component_id: np.ndarray) -> np.ndarray:
    comp = np.asarray(component_id)
    boundary = np.zeros(comp.shape, dtype=bool)
    boundary[1:, :] |= comp[1:, :] != comp[:-1, :]
    boundary[:-1, :] |= comp[:-1, :] != comp[1:, :]
    boundary[:, 1:] |= comp[:, 1:] != comp[:, :-1]
    boundary[:, :-1] |= comp[:, :-1] != comp[:, 1:]
    return boundary.astype(np.float32)


def temporal_stability(features_t: np.ndarray) -> dict[str, np.ndarray]:
    feat = l2_normalize(features_t)
    t = int(feat.shape[0])
    sims = np.ones(feat.shape[:-1], dtype=np.float32)
    if t > 1:
        sims = np.zeros(feat.shape[:-1], dtype=np.float32)
        counts = np.zeros(feat.shape[:-1], dtype=np.float32)
        for i in range(t - 1):
            sim = np.sum(feat[i] * feat[i + 1], axis=-1).astype(np.float32)
            sims[i] += sim
            sims[i + 1] += sim
            counts[i] += 1.0
            counts[i + 1] += 1.0
        sims = sims / np.maximum(counts, 1.0)
    mean_feat = l2_normalize(feat.mean(axis=0, keepdims=True))[0]
    mean_sim = np.sum(feat * mean_feat[None], axis=-1).astype(np.float32)
    var = np.var(feat, axis=0).mean(axis=-1, keepdims=False).astype(np.float32)
    var_t = np.repeat(var[None], t, axis=0)
    return {
        "temporal_stability": np.clip((sims + 1.0) * 0.5, 0.0, 1.0).astype(np.float32),
        "temporal_embedding_mean_sim": np.clip((mean_sim + 1.0) * 0.5, 0.0, 1.0).astype(np.float32),
        "temporal_embedding_var": robust01(var_t),
    }


def neighbor_contrast(features: np.ndarray, component: np.ndarray) -> dict[str, float | None]:
    feat = l2_normalize(features)
    same_vals: list[np.ndarray] = []
    cross_vals: list[np.ndarray] = []
    for dy, dx in ((1, 0), (0, 1)):
        a = feat[:-dy or None, :-dx or None] if dx else feat[:-dy, :]
        b = feat[dy:, dx:] if dx else feat[dy:, :]
        ca = component[:-dy or None, :-dx or None] if dx else component[:-dy, :]
        cb = component[dy:, dx:] if dx else component[dy:, :]
        sim = np.sum(a * b, axis=-1)
        same = ca == cb
        if np.any(same):
            same_vals.append(sim[same])
        if np.any(~same):
            cross_vals.append(sim[~same])
    same_all = np.concatenate(same_vals) if same_vals else np.asarray([], dtype=np.float32)
    cross_all = np.concatenate(cross_vals) if cross_vals else np.asarray([], dtype=np.float32)
    same_mean = float(np.mean(same_all)) if same_all.size else None
    cross_mean = float(np.mean(cross_all)) if cross_all.size else None
    contrast = None if same_mean is None or cross_mean is None else float(same_mean - cross_mean)
    return {"inside_neighbor_cosine": same_mean, "across_boundary_cosine": cross_mean, "boundary_contrast": contrast}


def component_count(component: np.ndarray) -> int:
    vals = np.unique(np.asarray(component))
    return int(vals[vals > 0].size)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return None
    x = x[finite] - float(x[finite].mean())
    y = y[finite] - float(y[finite].mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-8:
        return None
    return float(np.dot(x, y) / denom)


def locate_default_radio_checkpoint() -> str | None:
    for path in [
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/c-radio_v3-b_half.pth.tar"),
        Path.home() / ".cache/torch/hub/checkpoints/c-radio_v3-b_half.pth.tar",
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/c-radio-v3_l_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/radio-v2.5-l_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/radio-v2.5-b_half.pth.tar"),
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/radio_v2.5-h.pth.tar"),
    ]:
        if path.exists():
            return str(path)
    return None


def import_status(module: str, extra_path: Path | None = None) -> dict[str, Any]:
    old_path = list(sys.path)
    if extra_path is not None:
        sys.path.insert(0, str(extra_path))
    try:
        __import__(module)
        return {"module": module, "ok": True, "error_type": "", "error_message": ""}
    except Exception as exc:
        return {"module": module, "ok": False, "error_type": type(exc).__name__, "error_message": str(exc)}
    finally:
        sys.path[:] = old_path


def load_radseg_encoder(
    radio_root: Path,
    checkpoint: str | None,
    device: str,
    lang_model: str,
    *,
    amp: bool = False,
    slide_crop: int = 336,
    slide_stride: int = 224,
) -> Any:
    if str(radio_root) not in sys.path:
        sys.path.insert(0, str(radio_root))
    os.environ["CONDA_PREFIX"] = sys.prefix
    import torch
    from vipe.priors.embedding.radseg_encoder import RADSegEncoder

    ckpt = checkpoint or locate_default_radio_checkpoint()
    if ckpt is None:
        raise FileNotFoundError("no local RADIO/RADSeg checkpoint found")
    original_load = torch.load

    def compat_load(*args: Any, **kwargs: Any) -> Any:
        if args and str(args[0]) == str(ckpt) and "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = compat_load
    try:
        model = RADSegEncoder(
            device=device,
            model_version=str(ckpt),
            lang_model=str(lang_model),
            return_radio_features=True,
            compile=False,
            amp=bool(amp),
            predict=False,
            slide_crop=int(slide_crop),
            slide_stride=int(slide_stride),
            sam_refinement=False,
        )
    finally:
        torch.load = original_load
    return model


def extract_radio_feature(model: Any, image_path: Path, device: str) -> np.ndarray:
    import torch
    from PIL import Image

    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()[None].to(device)
    with torch.inference_mode():
        feat = model.encode_image_to_feat_map(tensor)
        feat = torch.nn.functional.normalize(feat.float(), dim=1, eps=1e-6)
    return feat.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy().astype(np.float32)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
