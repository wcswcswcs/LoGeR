from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from tools.materialize_d4rt_aligned_geometry_for_stream3d import (
    _apply_fit,
    _collect_anchors,
    _fit_summary,
    _fit_transform,
    _frame_ids_for_carrier_file,
    _load_carrier,
    _spacing_stats,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class D4RTStream3DGeometryAdapter:
    """Materialize D4RT geometry into Stream3D-like per-frame geometry files.

    This adapter intentionally stops before original Stream3D set-cover/manifold
    inference. It creates the data bridge needed for that refactor: aligned
    local point clouds, per-frame mask-to-D4RT point mappings, and geometry
    diagnostics.
    """

    def __init__(
        self,
        *,
        debug_root: str | Path,
        output_root: str | Path,
        mode: str,
        backbone: str = "Cropformer",
        min_visibility: float = 0.5,
        min_confidence: float = 0.5,
        max_anchors: int = 8000,
        robust_trim_percentile: float = 90.0,
    ) -> None:
        if mode not in {"raw", "scene_sim3", "window_sim3"}:
            raise ValueError(f"Unsupported adapter mode: {mode}")
        self.debug_root = Path(debug_root)
        self.output_root = Path(output_root)
        self.mode = mode
        self.backbone = backbone
        self.min_visibility = float(min_visibility)
        self.min_confidence = float(min_confidence)
        self.max_anchors = int(max_anchors)
        self.robust_trim_percentile = float(robust_trim_percentile)

    def _sample_mask_mapping(
        self,
        *,
        stream: ScanNetStream,
        frame_id: int,
        uv: np.ndarray,
        ok: np.ndarray,
    ) -> dict[str, Any]:
        try:
            mask = stream.load_mask(int(frame_id))
        except FileNotFoundError:
            return {
                "frame_id": int(frame_id),
                "mask_available": False,
                "num_mask_ids": 0,
                "num_mask_ids_with_d4rt_points": 0,
                "empty_mask_ratio": None,
                "local_point_indices": np.empty((0,), dtype=np.int64),
                "mask_ids_for_points": np.empty((0,), dtype=np.int64),
                "unique_mask_ids_with_points": np.empty((0,), dtype=np.int64),
            }
        h, w = mask.shape[:2]
        x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
        y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
        in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        use = ok & in_bounds
        point_indices = np.flatnonzero(use).astype(np.int64)
        sampled = np.zeros((uv.shape[0],), dtype=np.int64)
        if point_indices.size:
            sampled[point_indices] = mask[y[point_indices], x[point_indices]].astype(np.int64)
        positive = point_indices[sampled[point_indices] > 0]
        positive_mask_ids = sampled[positive].astype(np.int64)
        unique_all = np.unique(mask[mask > 0].astype(np.int64))
        unique_hit = np.unique(positive_mask_ids)
        empty_ratio = float(1.0 - unique_hit.shape[0] / max(unique_all.shape[0], 1)) if unique_all.size else 0.0
        return {
            "frame_id": int(frame_id),
            "mask_available": True,
            "num_mask_ids": int(unique_all.shape[0]),
            "num_mask_ids_with_d4rt_points": int(unique_hit.shape[0]),
            "empty_mask_ratio": empty_ratio,
            "local_point_indices": positive.astype(np.int64),
            "mask_ids_for_points": positive_mask_ids.astype(np.int64),
            "unique_mask_ids_with_points": unique_hit.astype(np.int64),
        }

    def materialize_scene(self, scene: str) -> dict[str, Any]:
        stream = ScanNetStream(seq_name=scene, backbone=self.backbone)
        errors = stream.validate(require_masks=True)
        if errors:
            raise RuntimeError("; ".join(errors))
        scene_debug = self.debug_root / scene
        carrier_paths = sorted(scene_debug.glob("carriers_window*.npz"))
        if not carrier_paths:
            raise FileNotFoundError(f"No carrier files under {scene_debug}")

        source, target, anchor_diag = _collect_anchors(
            stream,
            carrier_paths,
            min_visibility=self.min_visibility,
            min_confidence=self.min_confidence,
            max_anchors=self.max_anchors,
        )
        scene_fit = None
        if self.mode != "raw":
            scene_fit = _fit_transform(
                source,
                target,
                robust_trim_percentile=self.robust_trim_percentile,
            )

        out_scene = self.output_root / scene
        local_dir = out_scene / "local_pointcloud"
        mapping_dir = out_scene / "mask_point_mapping"
        local_dir.mkdir(parents=True, exist_ok=True)
        mapping_dir.mkdir(parents=True, exist_ok=True)

        all_points: list[np.ndarray] = []
        frame_rows: list[dict[str, Any]] = []
        raw = self.mode == "raw"
        for window_idx, carrier_path in enumerate(carrier_paths):
            data = _load_carrier(carrier_path)
            window_fit = scene_fit
            if self.mode == "window_sim3":
                src, tgt, _ = _collect_anchors(
                    stream,
                    [carrier_path],
                    min_visibility=self.min_visibility,
                    min_confidence=self.min_confidence,
                    max_anchors=min(self.max_anchors, 4096),
                )
                window_fit = _fit_transform(
                    src,
                    tgt,
                    robust_trim_percentile=self.robust_trim_percentile,
                )
            xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
            uv = np.asarray(data["uv_pred"], dtype=np.float32)
            visibility = np.asarray(data["visibility"], dtype=np.float32)
            confidence = np.asarray(data["confidence"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            frame_ids = _frame_ids_for_carrier_file(carrier_path, uv.shape[0])
            aligned = _apply_fit(xyz.reshape(-1, 3), window_fit, raw=raw).reshape(xyz.shape)
            all_points.append(aligned.reshape(-1, 3))
            for local_idx, frame_id in enumerate(frame_ids):
                ok = (
                    valid[local_idx]
                    & np.isfinite(aligned[local_idx]).all(axis=1)
                    & np.isfinite(uv[local_idx]).all(axis=1)
                    & (uv[local_idx, :, 0] >= 0.0)
                    & (uv[local_idx, :, 0] <= 1.0)
                    & (uv[local_idx, :, 1] >= 0.0)
                    & (uv[local_idx, :, 1] <= 1.0)
                    & (visibility[local_idx] >= self.min_visibility)
                    & (confidence[local_idx] >= self.min_confidence)
                )
                points = aligned[local_idx][ok].astype(np.float32)
                point_path = local_dir / f"window{window_idx:03d}_frame{int(frame_id):06d}_points.npy"
                np.save(point_path, points)
                mapping = self._sample_mask_mapping(
                    stream=stream,
                    frame_id=int(frame_id),
                    uv=uv[local_idx],
                    ok=ok,
                )
                mapping_path = mapping_dir / f"window{window_idx:03d}_frame{int(frame_id):06d}_mapping.npz"
                np.savez_compressed(
                    mapping_path,
                    local_point_indices=mapping["local_point_indices"],
                    mask_ids_for_points=mapping["mask_ids_for_points"],
                    unique_mask_ids_with_points=mapping["unique_mask_ids_with_points"],
                )
                frame_rows.append(
                    {
                        "window": int(window_idx),
                        "frame_id": int(frame_id),
                        "num_points": int(points.shape[0]),
                        "point_path": str(point_path),
                        "mapping_path": str(mapping_path),
                        "mask_available": bool(mapping["mask_available"]),
                        "num_mask_ids": int(mapping["num_mask_ids"]),
                        "num_mask_ids_with_d4rt_points": int(mapping["num_mask_ids_with_d4rt_points"]),
                        "empty_mask_ratio": mapping["empty_mask_ratio"],
                    }
                )

        all_aligned = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32)
        empty_values = [
            float(row["empty_mask_ratio"])
            for row in frame_rows
            if row.get("mask_available") and row.get("empty_mask_ratio") is not None
        ]
        manifest = {
            "scene": scene,
            "adapter": "D4RTStream3DGeometryAdapter",
            "mode": self.mode,
            "diagnostic_alignment_uses_scannet_depth_pose": self.mode != "raw",
            "is_complete_stream3d_replacement": False,
            "replacement_boundary": (
                "Materializes Stream3D-like geometry inputs; original Stream3D local proposal, "
                "set-cover and manifold stages are not rerun by this adapter."
            ),
            **anchor_diag,
            **_fit_summary(scene_fit),
            **_spacing_stats(all_aligned),
            "num_windows": int(len(carrier_paths)),
            "num_frames_materialized": int(len(frame_rows)),
            "empty_mask_ratio_mean": float(np.mean(empty_values)) if empty_values else None,
            "empty_mask_ratio_max": float(np.max(empty_values)) if empty_values else None,
            "frames": frame_rows,
        }
        out_scene.mkdir(parents=True, exist_ok=True)
        (out_scene / "geometry_manifest.json").write_text(
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest
