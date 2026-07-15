from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class LingBotMapStream3DGeometryAdapter:
    """Materialize saved LingBot-Map frame geometry for Stream4D support steps."""

    def __init__(
        self,
        *,
        lingbot_root: str | Path,
        output_root: str | Path,
        max_points_per_frame: int = 20000,
        min_depth: float = 1e-4,
        max_depth: float = 200.0,
        min_confidence: float | None = None,
    ) -> None:
        self.lingbot_root = Path(lingbot_root)
        self.output_root = Path(output_root)
        self.max_points_per_frame = int(max_points_per_frame)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_confidence = None if min_confidence is None else float(min_confidence)

    def materialize_frames(self, frame_ids: Iterable[int]) -> dict[str, Any]:
        provider = LingBotMapGeometryProvider(
            geometry_root=self.lingbot_root,
            max_points_per_frame=self.max_points_per_frame,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            min_confidence=self.min_confidence,
        )
        local_dir = self.output_root / "local_pointcloud"
        pixel_dir = self.output_root / "pixel_samples"
        local_dir.mkdir(parents=True, exist_ok=True)
        pixel_dir.mkdir(parents=True, exist_ok=True)

        frame_rows: list[dict[str, Any]] = []
        all_points: list[np.ndarray] = []
        for frame_id in [int(v) for v in frame_ids]:
            samples = provider.load_frame_samples(frame_id)
            points = np.asarray(samples.points, dtype=np.float32).reshape(-1, 3)
            xy = samples.xy
            finite = np.isfinite(points).all(axis=1)
            if xy is not None and xy.shape[0] == points.shape[0]:
                xy = np.asarray(xy, dtype=np.float32)[finite]
            points = points[finite]
            point_path = local_dir / f"frame{frame_id:06d}_points.npy"
            np.save(point_path, points)
            pixel_path = pixel_dir / f"frame{frame_id:06d}_xy.npy"
            if xy is not None:
                np.save(pixel_path, xy.astype(np.float32))
            else:
                np.save(pixel_path, np.empty((0, 2), dtype=np.float32))
            if points.size:
                all_points.append(points)
            row = {
                "frame_id": frame_id,
                "source_frame_id": samples.source_frame_id if samples.source_frame_id is not None else frame_id,
                "source": samples.source,
                "num_points": int(points.shape[0]),
                "point_path": point_path.as_posix(),
                "pixel_path": pixel_path.as_posix(),
                "has_pixel_samples": bool(xy is not None and xy.shape[0] > 0),
                "num_pixel_samples": int(xy.shape[0]) if xy is not None else 0,
                "image_shape": list(samples.image_shape) if samples.image_shape is not None else [],
                "finite_point_ratio": 1.0 if points.shape[0] else 0.0,
                "bbox_min": points.min(axis=0).tolist() if points.shape[0] else [],
                "bbox_max": points.max(axis=0).tolist() if points.shape[0] else [],
            }
            frame_rows.append(row)

        merged = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32)
        manifest = {
            "adapter": "LingBotMapStream3DGeometryAdapter",
            "lingbot_root": self.lingbot_root.as_posix(),
            "output_root": self.output_root.as_posix(),
            "sampling_json": provider.sampling_json.as_posix() if provider.sampling_json is not None else "",
            "num_frames_materialized": len(frame_rows),
            "total_points": int(merged.shape[0]),
            "max_points_per_frame": self.max_points_per_frame,
            "uses_d4rt_for_prediction": False,
            "uses_da3_for_prediction": False,
            "uses_gt_sim3_for_prediction": False,
            "is_complete_stream3d_replacement": False,
            "stream4d_metric_ready": False,
            "replacement_boundary": (
                "Materializes LingBot frame geometry only; mask support, affinity "
                "readout, and AP/MV_AP evaluation are not run by this adapter."
            ),
            "frames": frame_rows,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "geometry_manifest.json").write_text(
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
