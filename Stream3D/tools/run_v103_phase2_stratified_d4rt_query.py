#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream


SOURCE_CODEBOOK = {
    1: "uniform_grid",
    2: "mask_interior",
    3: "mask_boundary_band",
    4: "competing_mask_boundary",
    5: "semantic_gradient",
    6: "high_risk_broad_mask_interior",
    7: "overlap_frame_anchor",
}

REQUIRED_STRATA = list(SOURCE_CODEBOOK.values())
SEMANTIC_FEATURE_ROOTS = {
    "scene0011_00": "outputs/audit/v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    "scene0050_00": "outputs/audit/v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
}
MASK_ROOT_CANDIDATES = {
    "scene0011_00": [
        "outputs/cache/v66_cropformer_chunk_masks/scene0011_00/stride_5/cropformer_conf_0p500/mask2former_hornet_3x/final_processed/scene0011_00/output_Cropformer/mask",
        "outputs/cache/v65_cropformer_chunk_masks/scene0011_00/stride_5/cropformer_conf_0p500/mask2former_hornet_3x/final_processed/scene0011_00/output_Cropformer/mask",
    ],
    "scene0050_00": [
        "outputs/cache/v66_cropformer_chunk_masks/scene0050_00/stride_5/cropformer_conf_0p500/mask2former_hornet_3x/final_processed/scene0050_00/output_Cropformer/mask",
        "outputs/cache/v65_cropformer_chunk_masks/scene0050_00/stride_5/cropformer_conf_0p500/mask2former_hornet_3x/final_processed/scene0050_00/output_Cropformer/mask",
    ],
}


@dataclass
class SourceArrays:
    carrier_id: np.ndarray
    src_frame: np.ndarray
    src_frame_global: np.ndarray
    src_xy: np.ndarray
    src_uv: np.ndarray
    src_mask_id: np.ndarray
    query_source_code: np.ndarray


class SourceBuilder:
    def __init__(self, height: int, width: int) -> None:
        self.height = int(height)
        self.width = int(width)
        self.carrier_id: list[np.ndarray] = []
        self.src_frame: list[np.ndarray] = []
        self.src_frame_global: list[np.ndarray] = []
        self.src_xy: list[np.ndarray] = []
        self.src_uv: list[np.ndarray] = []
        self.src_mask_id: list[np.ndarray] = []
        self.query_source_code: list[np.ndarray] = []
        self.used: dict[int, set[tuple[int, int]]] = {}

    def add_points(
        self,
        *,
        frame_id: int,
        local_idx: int,
        source_code: int,
        xs: np.ndarray,
        ys: np.ndarray,
        mask_values: np.ndarray,
    ) -> int:
        xs = np.asarray(xs, dtype=np.int64).reshape(-1)
        ys = np.asarray(ys, dtype=np.int64).reshape(-1)
        mask_values = np.asarray(mask_values, dtype=np.int64).reshape(-1)
        if xs.size == 0:
            return 0
        xs = np.clip(xs, 0, max(self.width - 1, 0))
        ys = np.clip(ys, 0, max(self.height - 1, 0))
        if mask_values.shape[0] != xs.shape[0]:
            mask_values = np.full(xs.shape[0], int(mask_values[0]) if mask_values.size else 0, dtype=np.int64)
        used = self.used.setdefault(int(frame_id), set())
        keep_x: list[int] = []
        keep_y: list[int] = []
        keep_mask: list[int] = []
        for x, y, m in zip(xs.tolist(), ys.tolist(), mask_values.tolist()):
            key = (int(x), int(y))
            if key in used:
                continue
            used.add(key)
            keep_x.append(int(x))
            keep_y.append(int(y))
            keep_mask.append(int(m))
        if not keep_x:
            return 0
        x_arr = np.asarray(keep_x, dtype=np.int64)
        y_arr = np.asarray(keep_y, dtype=np.int64)
        m_arr = np.asarray(keep_mask, dtype=np.int64)
        source_arr = np.full(x_arr.shape[0], int(source_code), dtype=np.int64)
        carrier = (
            np.int64(frame_id) * np.int64(1_000_000_000_000)
            + np.int64(source_code) * np.int64(1_000_000_000)
            + y_arr.astype(np.int64) * np.int64(max(self.width, 1))
            + x_arr.astype(np.int64)
        )
        self.carrier_id.append(carrier.astype(np.int64))
        self.src_frame.append(np.full(x_arr.shape[0], int(local_idx), dtype=np.int64))
        self.src_frame_global.append(np.full(x_arr.shape[0], int(frame_id), dtype=np.int64))
        self.src_xy.append(np.stack([x_arr, y_arr], axis=1).astype(np.int64))
        self.src_uv.append(
            np.stack(
                [
                    x_arr.astype(np.float32) / float(max(self.width - 1, 1)),
                    y_arr.astype(np.float32) / float(max(self.height - 1, 1)),
                ],
                axis=1,
            ).astype(np.float32)
        )
        self.src_mask_id.append(m_arr.astype(np.int64))
        self.query_source_code.append(source_arr)
        return int(x_arr.shape[0])

    def build(self) -> SourceArrays:
        if not self.carrier_id:
            return SourceArrays(
                carrier_id=np.empty((0,), dtype=np.int64),
                src_frame=np.empty((0,), dtype=np.int64),
                src_frame_global=np.empty((0,), dtype=np.int64),
                src_xy=np.empty((0, 2), dtype=np.int64),
                src_uv=np.empty((0, 2), dtype=np.float32),
                src_mask_id=np.empty((0,), dtype=np.int64),
                query_source_code=np.empty((0,), dtype=np.int64),
            )
        return SourceArrays(
            carrier_id=np.concatenate(self.carrier_id, axis=0),
            src_frame=np.concatenate(self.src_frame, axis=0),
            src_frame_global=np.concatenate(self.src_frame_global, axis=0),
            src_xy=np.concatenate(self.src_xy, axis=0),
            src_uv=np.concatenate(self.src_uv, axis=0),
            src_mask_id=np.concatenate(self.src_mask_id, axis=0),
            query_source_code=np.concatenate(self.query_source_code, axis=0),
        )


def project(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO_ROOT / path
    return STREAM3D_ROOT / path


def resolve_repo(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def array_digest(arr: np.ndarray) -> str:
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr_c.shape).encode("utf-8"))
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def sources_digest(sources: SourceArrays) -> str:
    h = hashlib.sha256()
    for name in ["carrier_id", "src_frame", "src_frame_global", "src_xy", "src_uv", "src_mask_id", "query_source_code"]:
        digest = array_digest(getattr(sources, name))
        h.update(name.encode("utf-8"))
        h.update(digest.encode("utf-8"))
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["schema_version"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_semantic_rows(scene: str) -> dict[tuple[int, int], dict[str, Any]]:
    rel_path = SEMANTIC_FEATURE_ROOTS.get(scene)
    if not rel_path:
        return {}
    path = project(rel_path)
    if not path.exists():
        return {}
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                mask_id = int(row["mask_id"])
            except Exception:
                continue
            out[(frame_id, mask_id)] = {
                "broad_background_risk": parse_bool(row.get("broad_background_risk")),
                "semantic_boundary_variance": float(row.get("semantic_boundary_variance") or 0.0),
                "semantic_entropy": float(row.get("semantic_entropy") or 0.0),
                "semantic_background_score_proxy": parse_bool(row.get("semantic_background_score_proxy")),
                "used_pixel_count": int(float(row.get("used_pixel_count") or 0)),
            }
    return out


def default_mask_root(scene: str, stream: ScanNetStream, explicit: str | None) -> Path:
    if explicit:
        path = resolve_repo(explicit)
        if not path.exists():
            raise FileNotFoundError(f"explicit mask root does not exist: {path}")
        return path
    candidates = [project(p) for p in MASK_ROOT_CANDIDATES.get(scene, [])]
    candidates.append(stream.mask_dir)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No mask root found for "
        f"{scene}; tried " + "; ".join(str(p) for p in candidates)
    )


def load_mask_from_root(mask_root: Path, frame_id: int) -> np.ndarray:
    path = mask_root / f"{int(frame_id)}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask frame: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def base_grid_xy(height: int, width: int, grid_size: int, margin_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    margin_ratio = float(np.clip(margin_ratio, 0.0, 0.49))
    xs = np.rint(
        np.linspace(
            margin_ratio * float(max(width - 1, 1)),
            float(max(width - 1, 0)) - margin_ratio * float(max(width - 1, 1)),
            num=max(1, int(grid_size)),
        )
    ).astype(np.int64)
    ys = np.rint(
        np.linspace(
            margin_ratio * float(max(height - 1, 1)),
            float(max(height - 1, 0)) - margin_ratio * float(max(height - 1, 1)),
            num=max(1, int(grid_size)),
        )
    ).astype(np.int64)
    gx, gy = np.meshgrid(xs, ys)
    return gx.reshape(-1), gy.reshape(-1)


def take_even(xs: np.ndarray, ys: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=np.int64).reshape(-1)
    ys = np.asarray(ys, dtype=np.int64).reshape(-1)
    if xs.size == 0 or count <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    order = np.lexsort((xs, ys))
    if order.shape[0] <= count:
        selected = order
    else:
        selected = order[np.linspace(0, order.shape[0] - 1, num=int(count), dtype=np.int64)]
    return xs[selected], ys[selected]


def mask_pixels(mask: np.ndarray, mask_id: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask == int(mask_id))
    return xs.astype(np.int64), ys.astype(np.int64)


def eroded_or_all(mask_binary: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    eroded = cv2.erode(mask_binary.astype(np.uint8), kernel, iterations=1)
    if int(np.count_nonzero(eroded)) > 0:
        return eroded.astype(bool)
    return mask_binary.astype(bool)


def boundary_band(mask_binary: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    grad = cv2.morphologyEx(mask_binary.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    if int(np.count_nonzero(grad)) > 0:
        return grad.astype(bool)
    return mask_binary.astype(bool)


def competing_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    out = np.zeros(mask.shape, dtype=bool)
    right = (mask[:, :-1] != mask[:, 1:]) & (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
    down = (mask[:-1, :] != mask[1:, :]) & (mask[:-1, :] > 0) & (mask[1:, :] > 0)
    out[:, :-1] |= right
    out[:, 1:] |= right
    out[:-1, :] |= down
    out[1:, :] |= down
    return out


def generate_sources(
    *,
    masks: np.ndarray,
    frame_ids: list[int],
    semantic_rows: dict[tuple[int, int], dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[SourceArrays, list[dict[str, Any]]]:
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [T,H,W], got {masks.shape}")
    num_frames, height, width = masks.shape
    builder = SourceBuilder(height=height, width=width)
    frame_rows: list[dict[str, Any]] = []

    for local_idx, frame_id in enumerate(frame_ids):
        frame_counts = {name: 0 for name in SOURCE_CODEBOOK.values()}
        mask = np.asarray(masks[local_idx])
        mask_ids = [int(v) for v in np.unique(mask).tolist() if int(v) > 0]

        # High-risk broad masks first so they are represented even if they overlap other strata.
        area_ratio_by_mask: dict[int, float] = {}
        broad_ids: list[int] = []
        for mask_id in mask_ids:
            row = semantic_rows.get((int(frame_id), int(mask_id)), {})
            area_ratio = float(np.count_nonzero(mask == int(mask_id))) / float(max(height * width, 1))
            area_ratio_by_mask[int(mask_id)] = area_ratio
            if bool(row.get("broad_background_risk")) or bool(row.get("semantic_background_score_proxy")) or area_ratio >= float(args.broad_area_ratio):
                broad_ids.append(int(mask_id))
        broad_id_set = set(broad_ids)
        mask_strata_ids: list[int] = []
        for mask_id in mask_ids:
            area_ratio = float(area_ratio_by_mask.get(int(mask_id), 0.0))
            if bool(args.object_like_mask_strata_only):
                if int(mask_id) in broad_id_set:
                    continue
                if area_ratio < float(args.object_like_min_area_ratio):
                    continue
                if float(args.object_like_max_area_ratio) > 0.0 and area_ratio > float(args.object_like_max_area_ratio):
                    continue
            mask_strata_ids.append(int(mask_id))

        for mask_id in broad_ids:
            binary = mask == int(mask_id)
            pts = eroded_or_all(binary, kernel_size=int(args.interior_kernel_size))
            ys, xs = np.where(pts)
            xs, ys = take_even(xs, ys, int(args.broad_interior_points_per_mask))
            added = builder.add_points(
                frame_id=int(frame_id),
                local_idx=int(local_idx),
                source_code=6,
                xs=xs,
                ys=ys,
                mask_values=np.full(xs.shape[0], int(mask_id), dtype=np.int64),
            )
            frame_counts["high_risk_broad_mask_interior"] += added

        # Semantic-gradient masks: GT-free high boundary variance / entropy rows.
        scored: list[tuple[float, int]] = []
        for mask_id in mask_strata_ids:
            row = semantic_rows.get((int(frame_id), int(mask_id)), {})
            score = float(row.get("semantic_boundary_variance", 0.0)) + 0.05 * float(row.get("semantic_entropy", 0.0))
            if score > 0.0:
                scored.append((score, int(mask_id)))
        scored.sort(reverse=True)
        for _score, mask_id in scored[: int(args.semantic_top_masks_per_frame)]:
            band = boundary_band(mask == int(mask_id), kernel_size=int(args.boundary_kernel_size))
            ys, xs = np.where(band)
            xs, ys = take_even(xs, ys, int(args.semantic_points_per_mask))
            added = builder.add_points(
                frame_id=int(frame_id),
                local_idx=int(local_idx),
                source_code=5,
                xs=xs,
                ys=ys,
                mask_values=np.full(xs.shape[0], int(mask_id), dtype=np.int64),
            )
            frame_counts["semantic_gradient"] += added

        comp = competing_boundary(mask)
        ys, xs = np.where(comp)
        xs, ys = take_even(xs, ys, int(args.competing_boundary_points_per_frame))
        added = builder.add_points(
            frame_id=int(frame_id),
            local_idx=int(local_idx),
            source_code=4,
            xs=xs,
            ys=ys,
            mask_values=mask[ys, xs] if xs.size else np.empty((0,), dtype=np.int64),
        )
        frame_counts["competing_mask_boundary"] += added

        for mask_id in mask_strata_ids:
            binary = mask == int(mask_id)
            band = boundary_band(binary, kernel_size=int(args.boundary_kernel_size))
            ys, xs = np.where(band)
            xs, ys = take_even(xs, ys, int(args.boundary_points_per_mask))
            added = builder.add_points(
                frame_id=int(frame_id),
                local_idx=int(local_idx),
                source_code=3,
                xs=xs,
                ys=ys,
                mask_values=np.full(xs.shape[0], int(mask_id), dtype=np.int64),
            )
            frame_counts["mask_boundary_band"] += added

        for mask_id in mask_strata_ids:
            binary = mask == int(mask_id)
            interior = eroded_or_all(binary, kernel_size=int(args.interior_kernel_size))
            ys, xs = np.where(interior)
            xs, ys = take_even(xs, ys, int(args.interior_points_per_mask))
            added = builder.add_points(
                frame_id=int(frame_id),
                local_idx=int(local_idx),
                source_code=2,
                xs=xs,
                ys=ys,
                mask_values=np.full(xs.shape[0], int(mask_id), dtype=np.int64),
            )
            frame_counts["mask_interior"] += added

        if local_idx < int(args.overlap_frames) or local_idx >= max(0, num_frames - int(args.overlap_frames)):
            xs, ys = base_grid_xy(height, width, int(args.overlap_anchor_grid_size), float(args.overlap_anchor_margin_ratio))
            added = builder.add_points(
                frame_id=int(frame_id),
                local_idx=int(local_idx),
                source_code=7,
                xs=xs,
                ys=ys,
                mask_values=mask[ys, xs] if xs.size else np.empty((0,), dtype=np.int64),
            )
            frame_counts["overlap_frame_anchor"] += added

        xs, ys = base_grid_xy(height, width, int(args.uniform_grid_size), float(args.grid_margin_ratio))
        added = builder.add_points(
            frame_id=int(frame_id),
            local_idx=int(local_idx),
            source_code=1,
            xs=xs,
            ys=ys,
            mask_values=mask[ys, xs] if xs.size else np.empty((0,), dtype=np.int64),
        )
        frame_counts["uniform_grid"] += added

        for source_name, count in frame_counts.items():
            frame_rows.append(
                {
                    "schema_version": "stream4d_v103_phase2_frame_query_row_v1",
                    "phase_id": "v103_phase2_stratified_d4rt_query",
                    "scene_id": args.scene,
                    "frame_id": int(frame_id),
                    "frame_local_index": int(local_idx),
                    "query_source": source_name,
                    "source_count": int(count),
                    "mask_count_in_frame": int(len(mask_ids)),
                    "broad_mask_count_in_frame": int(len(broad_ids)),
                    "mask_strata_count_in_frame": int(len(mask_strata_ids)),
                    "object_like_mask_strata_only": bool(args.object_like_mask_strata_only),
                }
            )

    return builder.build(), frame_rows


def rate(num: int | float, den: int | float) -> float:
    den_f = float(den)
    if den_f <= 0:
        return 0.0
    return float(num) / den_f


def batch_metrics(batch: Any, visible_threshold: float, confidence_threshold: float) -> dict[str, Any]:
    uv = np.asarray(batch.uv_pred, dtype=np.float32)
    xyz = np.asarray(batch.xyz_ref, dtype=np.float32)
    valid = np.asarray(batch.valid, dtype=bool)
    visibility = np.asarray(batch.visibility_prob, dtype=np.float32)
    confidence = np.asarray(batch.confidence_prob, dtype=np.float32)
    finite = np.isfinite(uv).all(axis=-1) & np.isfinite(xyz).all(axis=-1)
    in01 = finite & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    valid_in01 = valid & in01
    visible = valid & finite & (visibility >= float(visible_threshold)) & (confidence >= float(confidence_threshold))
    visible_in01 = visible & in01
    total = int(valid.size)
    carrier_rates = np.mean(valid_in01, axis=0) if valid.shape[1] else np.empty((0,), dtype=np.float32)
    return {
        "projection_valid_rate_model_valid": rate(int(np.count_nonzero(valid)), total),
        "uv_in01_rate_all_observations": rate(int(np.count_nonzero(valid_in01)), total),
        "visible_confident_observation_rate": rate(int(np.count_nonzero(visible)), total),
        "uv_in01_rate_visible_confident_observations": rate(int(np.count_nonzero(visible_in01)), int(np.count_nonzero(visible))),
        "in_image_rate": rate(int(np.count_nonzero(valid_in01)), int(np.count_nonzero(valid))),
        "visibility_mean_in_image": float(np.mean(visibility[valid_in01])) if int(np.count_nonzero(valid_in01)) else 0.0,
        "confidence_mean_in_image": float(np.mean(confidence[valid_in01])) if int(np.count_nonzero(valid_in01)) else 0.0,
        "carrier_track_in_image_rate_p10": float(np.percentile(carrier_rates, 10)) if carrier_rates.size else 0.0,
        "carrier_track_in_image_rate_p50": float(np.percentile(carrier_rates, 50)) if carrier_rates.size else 0.0,
        "carrier_track_in_image_rate_p90": float(np.percentile(carrier_rates, 90)) if carrier_rates.size else 0.0,
    }


def resolve_reuse_carrier_batch(value: str) -> Path | None:
    value = str(value).strip()
    if not value:
        return None
    path = resolve_repo(value)
    if path.is_dir():
        path = path / "carrier_batch.npz"
    if not path.exists():
        raise FileNotFoundError(f"requested carrier batch cache does not exist: {path}")
    return path


def cache_source_parity(sources: SourceArrays, cache_npz: Path) -> tuple[bool, dict[str, Any]]:
    cached = np.load(cache_npz, allow_pickle=False)
    checks: dict[str, bool] = {}
    max_abs_src_uv = 0.0
    for name in ["carrier_id", "src_frame", "src_frame_global", "src_xy", "src_mask_id", "query_source_code"]:
        if name not in cached:
            checks[name] = False
            continue
        checks[name] = bool(np.array_equal(getattr(sources, name), cached[name]))
    if "src_uv" in cached:
        diff = np.abs(np.asarray(sources.src_uv, dtype=np.float32) - np.asarray(cached["src_uv"], dtype=np.float32))
        max_abs_src_uv = float(np.max(diff)) if diff.size else 0.0
        checks["src_uv"] = bool(max_abs_src_uv <= 1e-7 and tuple(sources.src_uv.shape) == tuple(cached["src_uv"].shape))
    else:
        checks["src_uv"] = False
    ok = all(bool(v) for v in checks.values())
    return ok, {
        "cache_npz": rel(cache_npz),
        "source_parity_pass": ok,
        "source_parity_checks": checks,
        "source_digest_current": sources_digest(sources),
        "source_digest_cache_equivalent": sources_digest(
            SourceArrays(
                carrier_id=np.asarray(cached["carrier_id"], dtype=np.int64),
                src_frame=np.asarray(cached["src_frame"], dtype=np.int64),
                src_frame_global=np.asarray(cached["src_frame_global"], dtype=np.int64),
                src_xy=np.asarray(cached["src_xy"], dtype=np.int64),
                src_uv=np.asarray(cached["src_uv"], dtype=np.float32),
                src_mask_id=np.asarray(cached["src_mask_id"], dtype=np.int64),
                query_source_code=np.asarray(cached["query_source_code"], dtype=np.int64),
            )
        )
        if all(name in cached for name in ["carrier_id", "src_frame", "src_frame_global", "src_xy", "src_uv", "src_mask_id", "query_source_code"])
        else "",
        "src_uv_max_abs_diff": max_abs_src_uv,
    }


def carrier_batch_namespace(npz_path: Path) -> Any:
    data = np.load(npz_path, allow_pickle=False)
    return SimpleNamespace(
        carrier_id=data["carrier_id"],
        src_frame=data["src_frame"],
        src_uv=data["src_uv"],
        xyz_ref=data["xyz_ref"],
        uv_pred=data["uv_pred"],
        visibility_prob=data["visibility_prob"],
        confidence_prob=data["confidence_prob"],
        valid=data["valid"],
        xyz_local=data["xyz_local"],
        src_frame_global=data["src_frame_global"],
        src_xy=data["src_xy"],
        src_mask_id=data["src_mask_id"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 stratified D4RT query generator and decoder.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scannet-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--d4rt-root", default="Open-d4rt")
    parser.add_argument("--d4rt-config", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--d4rt-ckpt", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--uniform-grid-size", type=int, default=24)
    parser.add_argument("--grid-margin-ratio", type=float, default=0.02)
    parser.add_argument("--interior-points-per-mask", type=int, default=2)
    parser.add_argument("--boundary-points-per-mask", type=int, default=2)
    parser.add_argument("--semantic-top-masks-per-frame", type=int, default=4)
    parser.add_argument("--semantic-points-per-mask", type=int, default=2)
    parser.add_argument("--broad-interior-points-per-mask", type=int, default=2)
    parser.add_argument("--broad-area-ratio", type=float, default=0.12)
    parser.add_argument("--object-like-mask-strata-only", action="store_true")
    parser.add_argument("--object-like-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--object-like-max-area-ratio", type=float, default=0.0)
    parser.add_argument("--competing-boundary-points-per-frame", type=int, default=96)
    parser.add_argument("--overlap-anchor-grid-size", type=int, default=10)
    parser.add_argument("--overlap-anchor-margin-ratio", type=float, default=0.05)
    parser.add_argument("--boundary-kernel-size", type=int, default=5)
    parser.add_argument("--interior-kernel-size", type=int, default=7)
    parser.add_argument("--visible-threshold", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--min-query-count-per-frame",
        type=float,
        default=16384.0,
        help="Minimum generated query density required by the current v103 run contract.",
    )
    parser.add_argument("--dry-run-sources-only", action="store_true")
    parser.add_argument(
        "--reuse-carrier-batch",
        default="",
        help="Optional path to a previous carrier_batch.npz or Phase2 output dir. Source arrays must match exactly.",
    )
    return parser


def main() -> int:
    started = time.time()
    args = build_parser().parse_args()
    out = project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    stream = ScanNetStream(seq_name=args.scene, root=resolve_repo(args.scannet_root))
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    frame_ids = stream.frame_ids(stride=int(args.stride), max_frames=int(args.max_frames))
    if len(frame_ids) != int(args.chunk_size):
        raise RuntimeError(f"Expected chunk_size={args.chunk_size} frames, got {len(frame_ids)}")
    mask_root = default_mask_root(args.scene, stream, str(args.mask_root).strip() or None)
    rgbs = [stream.load_rgb(fid) for fid in frame_ids]
    masks = [load_mask_from_root(mask_root, fid) for fid in frame_ids]
    rgb = np.stack(rgbs, axis=0)
    mask = np.stack(masks, axis=0)
    semantic_rows = load_semantic_rows(args.scene)
    sources, frame_rows = generate_sources(masks=mask, frame_ids=frame_ids, semantic_rows=semantic_rows, args=args)
    source_count = int(sources.carrier_id.shape[0])
    if source_count <= 0:
        raise RuntimeError("No query sources generated")

    source_counts: list[dict[str, Any]] = []
    for code, name in SOURCE_CODEBOOK.items():
        count = int(np.count_nonzero(sources.query_source_code == int(code)))
        source_counts.append(
            {
                "schema_version": "stream4d_v103_phase2_query_source_count_row_v1",
                "phase_id": "v103_phase2_stratified_d4rt_query",
                "scene_id": args.scene,
                "query_source": name,
                "source_code": int(code),
                "source_count": count,
                "source_rate": rate(count, source_count),
            }
        )

    np.savez_compressed(
        out / "carrier_sources.npz",
        carrier_id=sources.carrier_id,
        src_frame=sources.src_frame,
        src_frame_global=sources.src_frame_global,
        src_xy=sources.src_xy,
        src_uv=sources.src_uv,
        src_mask_id=sources.src_mask_id,
        query_source_code=sources.query_source_code,
    )
    write_json(out / "query_source_codebook.json", SOURCE_CODEBOOK)
    write_csv(out / "query_source_count_rows.csv", source_counts)
    write_csv(out / "frame_query_rows.csv", frame_rows)

    decode_error_count = 0
    batch = None
    gpu_memory_peak_mb: float | None = None
    reuse_cache = resolve_reuse_carrier_batch(str(args.reuse_carrier_batch))
    cache_meta: dict[str, Any] = {
        "carrier_batch_cache_requested": reuse_cache is not None,
        "carrier_batch_cache_used": False,
        "fresh_d4rt_decode": not bool(args.dry_run_sources_only) and reuse_cache is None,
        "source_parity_pass": "",
    }
    if not args.dry_run_sources_only:
        if reuse_cache is not None:
            parity_ok, parity_meta = cache_source_parity(sources, reuse_cache)
            cache_meta.update(parity_meta)
            if not parity_ok:
                raise RuntimeError(f"carrier batch cache source parity failed: {json.dumps(jsonable(parity_meta), sort_keys=True)}")
            dst_cache = out / "carrier_batch.npz"
            if reuse_cache.resolve() != dst_cache.resolve():
                shutil.copy2(reuse_cache, dst_cache)
            batch = carrier_batch_namespace(dst_cache)
            infer_diagnostics = {
                "cache_reuse": True,
                "cache_npz": rel(reuse_cache),
                "fresh_d4rt_decode": False,
                "note": "D4RT decode was reused after exact source-array parity; metrics were recomputed from cached carrier_batch.npz.",
            }
            cache_meta["carrier_batch_cache_used"] = True
            cache_meta["fresh_d4rt_decode"] = False
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                torch = None  # type: ignore[assignment]
            adapter = D4RTAdapter(
                d4rt_root=resolve_repo(args.d4rt_root),
                model_config=resolve_repo(args.d4rt_config),
                ckpt_path=resolve_repo(args.d4rt_ckpt),
                device=args.device,
            )
            batch = adapter.infer_carriers(
                video_rgb_uint8=rgb,
                src_uv_norm=sources.src_uv,
                src_frame_local=sources.src_frame,
                carrier_id=sources.carrier_id,
                src_frame_global=sources.src_frame_global,
                src_xy=sources.src_xy,
                src_mask_id=sources.src_mask_id,
                query_chunk_size=int(args.query_chunk_size),
            )
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_memory_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                gpu_memory_peak_mb = None
            np.savez_compressed(
                out / "carrier_batch.npz",
                carrier_id=batch.carrier_id,
                src_frame=batch.src_frame,
                src_uv=batch.src_uv,
                xyz_ref=batch.xyz_ref,
                uv_pred=batch.uv_pred,
                visibility_prob=batch.visibility_prob,
                confidence_prob=batch.confidence_prob,
                valid=batch.valid,
                xyz_local=batch.xyz_local,
                src_frame_global=batch.src_frame_global,
                src_xy=batch.src_xy,
                src_mask_id=batch.src_mask_id,
                query_source_code=sources.query_source_code,
            )
            infer_diagnostics = dict(adapter.last_infer_diagnostics)
        metric = batch_metrics(batch, float(args.visible_threshold), float(args.confidence_threshold))
    else:
        infer_diagnostics = {}
        metric = {
            "projection_valid_rate_model_valid": "",
            "uv_in01_rate_all_observations": "",
            "visible_confident_observation_rate": "",
            "uv_in01_rate_visible_confident_observations": "",
            "in_image_rate": "",
            "visibility_mean_in_image": "",
            "confidence_mean_in_image": "",
            "carrier_track_in_image_rate_p10": "",
            "carrier_track_in_image_rate_p50": "",
            "carrier_track_in_image_rate_p90": "",
        }

    present_strata = sorted(
        SOURCE_CODEBOOK[int(code)]
        for code in np.unique(sources.query_source_code).tolist()
        if int(code) in SOURCE_CODEBOOK and int(np.count_nonzero(sources.query_source_code == int(code))) > 0
    )
    all_required_present = all(s in present_strata for s in REQUIRED_STRATA)
    avg_query_count_per_frame = float(source_count) / float(max(len(frame_ids), 1))
    gate_specs = [
        ("decode_error_count_eq_0", decode_error_count == 0, decode_error_count, 0),
        ("all_required_query_strata_present", all_required_present, ";".join(present_strata), ";".join(REQUIRED_STRATA)),
        (
            "query_count_per_frame_ge_min_required",
            avg_query_count_per_frame >= float(args.min_query_count_per_frame),
            avg_query_count_per_frame,
            float(args.min_query_count_per_frame),
        ),
        ("projection_valid_rate_model_valid_ge_0p80", metric["projection_valid_rate_model_valid"] == "" or float(metric["projection_valid_rate_model_valid"]) >= 0.80, metric["projection_valid_rate_model_valid"], 0.80),
        ("uv_in01_rate_visible_confident_ge_0p80", metric["uv_in01_rate_visible_confident_observations"] == "" or float(metric["uv_in01_rate_visible_confident_observations"]) >= 0.80, metric["uv_in01_rate_visible_confident_observations"], 0.80),
        ("uses_gt_for_query_selection_false", True, False, False),
        ("uses_future_false", True, False, False),
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v103_phase2_stratified_gate_row_v1",
            "phase_id": "v103_phase2_stratified_d4rt_query",
            "scene_id": args.scene,
            "gate_name": name,
            "pass": bool(ok),
            "observed": observed,
            "required": required,
        }
        for name, ok, observed, required in gate_specs
    ]
    failure_rows: list[dict[str, Any]] = []
    for row in gate_rows:
        if not bool(row["pass"]):
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase2_stratified_failure_row_v1",
                    "phase_id": "v103_phase2_stratified_d4rt_query",
                    "scene_id": args.scene,
                    "failure_id": row["gate_name"],
                    "severity": "blocking",
                    "evidence": f"observed={row['observed']} required={row['required']}",
                    "repair_direction": "Increase query density or adjust stratified source generation; if decode/runtime fails, use query batching/streaming cache.",
                }
            )
    write_csv(out / "gate_rows.csv", gate_rows)
    write_csv(out / "failure_rows.csv", failure_rows)

    summary = {
        "schema_version": "stream4d_v103_phase2_stratified_d4rt_query_summary_v1",
        "phase_id": "v103_phase2_stratified_d4rt_query",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "scene_id": args.scene,
        "frame_stride": int(args.stride),
        "chunk_size": int(args.chunk_size),
        "overlap_frames": int(args.overlap_frames),
        "frame_count": len(frame_ids),
        "frame_ids": [int(v) for v in frame_ids],
        "mask_root": rel(mask_root),
        "source_count": source_count,
        "query_count_per_frame": avg_query_count_per_frame,
        "min_query_count_per_frame_required": float(args.min_query_count_per_frame),
        "present_strata": present_strata,
        "all_required_query_strata_present": all_required_present,
        "query_generation_policy": {
            "object_like_mask_strata_only": bool(args.object_like_mask_strata_only),
            "object_like_min_area_ratio": float(args.object_like_min_area_ratio),
            "object_like_max_area_ratio": float(args.object_like_max_area_ratio),
            "broad_area_ratio": float(args.broad_area_ratio),
            "mask_strata_policy": (
                "mask_interior/mask_boundary_band/semantic_gradient skip GT-free broad masks"
                if bool(args.object_like_mask_strata_only)
                else "all positive CropFormer masks"
            ),
            "broad_mask_diagnostic_stratum_retained": True,
            "uses_gt": False,
        },
        "decode_error_count": decode_error_count,
        "dry_run_sources_only": bool(args.dry_run_sources_only),
        "uses_gt_for_query_selection": False,
        "uses_future": False,
        "gpu_memory_peak_MB": gpu_memory_peak_mb,
        "gpu_memory_peak_recorded": gpu_memory_peak_mb is not None,
        "carrier_batch_cache": cache_meta,
        "d4rt_infer_diagnostics": infer_diagnostics,
        "metrics": metric,
        "source_counts": source_counts,
        "failure_count": len(failure_rows),
        "stratified_phase2_pass_like": all(bool(row["pass"]) for row in gate_rows),
        "truthfulness_note": "This runner generates GT-free stratified D4RT query sources and decodes carriers; AP is not computed here.",
        "outputs": {
            "summary": rel(out / "summary.json"),
            "carrier_sources": rel(out / "carrier_sources.npz"),
            "carrier_batch": rel(out / "carrier_batch.npz") if not args.dry_run_sources_only else "",
            "query_source_codebook": rel(out / "query_source_codebook.json"),
            "query_source_count_rows": rel(out / "query_source_count_rows.csv"),
            "frame_query_rows": rel(out / "frame_query_rows.csv"),
            "gate_rows": rel(out / "gate_rows.csv"),
            "failure_rows": rel(out / "failure_rows.csv"),
            "last_command": rel(out / "last_command.txt"),
        },
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0 if len(failure_rows) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
