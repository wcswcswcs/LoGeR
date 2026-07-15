from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .base import FrameProjection


@dataclass(frozen=True)
class LingBotFrameSamples:
    points: np.ndarray
    source: str
    xy: np.ndarray | None = None
    image_shape: tuple[int, int] | None = None
    source_frame_id: int | None = None


def _parse_bss_intrinsics(path: Path) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            rows[int(float(parts[0]))] = np.asarray([float(v) for v in parts[1:7]], dtype=np.float32)
        except ValueError:
            continue
    return rows


def _parse_bss_traj(path: Path) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            frame_id = int(float(parts[0]))
            vals = [float(v) for v in parts[1:13]]
        except ValueError:
            continue
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :4] = np.asarray(vals, dtype=np.float32).reshape(3, 4)
        rows[frame_id] = pose
    return rows


def _load_sampling_frame_ids(path: Path | None) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    frames = payload.get("frames", [])
    out: dict[int, int] = {}
    for idx, frame_id in enumerate(frames):
        try:
            out[int(idx)] = int(frame_id)
        except Exception:
            continue
    return out


def _autodetect_sampling_json(geometry_root: Path) -> Path | None:
    candidates = [
        geometry_root.parent / "gt" / "sampling.json",
        geometry_root / "sampling.json",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


class LingBotMapGeometryProvider:
    name = "lingbot_map_provider"
    uses_rgbd_for_prediction = False
    uses_pose_for_prediction = False
    uses_scannet_mesh_for_prediction = False
    uses_gt_sim3_for_prediction = False
    is_diagnostic_only = False

    def __init__(
        self,
        *,
        geometry_root: str | Path,
        nn_radius: float = 0.05,
        min_depth: float = 1e-4,
        max_depth: float = 200.0,
        max_points_per_frame: int = 20000,
        confidence_root: str | Path | None = None,
        min_confidence: float | None = None,
        sampling_json: str | Path | None = None,
    ) -> None:
        self.geometry_root = Path(geometry_root)
        self.nn_radius = float(nn_radius)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.max_points_per_frame = int(max_points_per_frame)
        self.confidence_root = Path(confidence_root) if confidence_root is not None else None
        self.min_confidence = None if min_confidence is None else float(min_confidence)
        self._intrinsics = _parse_bss_intrinsics(self.geometry_root / "intrinsics.txt")
        self._poses = _parse_bss_traj(self.geometry_root / "traj.txt")
        self.sampling_json = Path(sampling_json) if sampling_json is not None else _autodetect_sampling_json(self.geometry_root)
        self._source_frame_ids = _load_sampling_frame_ids(self.sampling_json)

    def source_frame_id(self, frame_id: int) -> int:
        return int(self._source_frame_ids.get(int(frame_id), int(frame_id)))

    def _candidate_paths(
        self,
        subdir: str,
        frame_id: int,
        suffixes: tuple[str, ...],
        *,
        root: Path | None = None,
    ) -> list[Path]:
        frame_key = f"{int(frame_id):06d}"
        paths: list[Path] = []
        base = root or self.geometry_root
        roots = [base, base / subdir]
        for item in roots:
            for suffix in suffixes:
                paths.append(item / f"{frame_key}{suffix}")
                paths.extend(sorted(item.glob(f"**/*frame{frame_key}*{suffix}")))
        seen: set[Path] = set()
        unique: list[Path] = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if path.exists() and path.is_file():
                unique.append(path)
        return unique

    def _load_npy_points(self, frame_id: int) -> LingBotFrameSamples:
        parts = []
        for path in self._candidate_paths("points", frame_id, (".npy",)):
            arr = np.asarray(np.load(path), dtype=np.float32)
            if arr.size and arr.shape[-1] == 3:
                parts.append(arr.reshape(-1, 3))
        points = np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=np.float32)
        return LingBotFrameSamples(points=points, source="npy_points", source_frame_id=self.source_frame_id(frame_id))

    def _read_exr(self, path: Path) -> np.ndarray:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"failed to read EXR frame: {path}")
        return np.asarray(arr, dtype=np.float32)

    def _load_depth(self, frame_id: int) -> np.ndarray:
        for path in self._candidate_paths("depth", frame_id, (".npy", ".exr")):
            if path.suffix.lower() == ".npy":
                depth = np.asarray(np.load(path), dtype=np.float32)
            else:
                depth = self._read_exr(path)
            if depth.ndim == 3:
                depth = depth[..., 0]
            return np.asarray(depth, dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)

    def _load_confidence(self, frame_id: int) -> np.ndarray:
        if self.min_confidence is None:
            return np.empty((0, 0), dtype=np.float32)
        root = self.confidence_root or self.geometry_root
        for path in self._candidate_paths("confidence", frame_id, (".npy", ".exr"), root=root):
            if path.suffix.lower() == ".npy":
                conf = np.asarray(np.load(path), dtype=np.float32)
            else:
                conf = self._read_exr(path)
            if conf.ndim == 3:
                conf = conf[..., 0]
            return np.asarray(conf, dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)

    def _depth_to_world_samples(self, frame_id: int) -> LingBotFrameSamples:
        depth = self._load_depth(frame_id)
        pose = self._poses.get(int(frame_id))
        intr = self._intrinsics.get(int(frame_id))
        if depth.size == 0 or pose is None or intr is None:
            return LingBotFrameSamples(
                points=np.empty((0, 3), dtype=np.float32),
                source="depth_pose_intrinsics",
                source_frame_id=self.source_frame_id(frame_id),
            )

        valid = np.isfinite(depth) & (depth > self.min_depth) & (depth <= self.max_depth)
        conf = self._load_confidence(frame_id)
        if conf.shape == depth.shape and self.min_confidence is not None:
            valid &= np.isfinite(conf) & (conf >= self.min_confidence)
        y, x = np.nonzero(valid)
        if x.size == 0:
            return LingBotFrameSamples(
                points=np.empty((0, 3), dtype=np.float32),
                source="depth_pose_intrinsics",
                xy=np.empty((0, 2), dtype=np.float32),
                image_shape=tuple(int(v) for v in depth.shape[:2]),
                source_frame_id=self.source_frame_id(frame_id),
            )
        if x.size > self.max_points_per_frame:
            keep = np.linspace(0, x.size - 1, self.max_points_per_frame, dtype=np.int64)
            x = x[keep]
            y = y[keep]
        z = depth[y, x].astype(np.float32)
        fx, fy, cx, cy = [float(v) for v in intr[:4]]
        cam = np.stack(
            [
                (x.astype(np.float32) - cx) * z / fx,
                (y.astype(np.float32) - cy) * z / fy,
                z,
                np.ones_like(z),
            ],
            axis=1,
        )
        world = (pose @ cam.T).T[:, :3]
        xy = np.stack([x.astype(np.float32), y.astype(np.float32)], axis=1)
        return LingBotFrameSamples(
            points=np.asarray(world, dtype=np.float32),
            source="depth_pose_intrinsics",
            xy=xy.astype(np.float32),
            image_shape=tuple(int(v) for v in depth.shape[:2]),
            source_frame_id=self.source_frame_id(frame_id),
        )

    def _load_samples(self, frame_id: int) -> LingBotFrameSamples:
        samples = self._load_npy_points(frame_id)
        if samples.points.size:
            return samples
        return self._depth_to_world_samples(frame_id)

    def load_frame_points(self, frame_id: int) -> tuple[np.ndarray, str]:
        samples = self._load_samples(frame_id)
        return samples.points, samples.source

    def load_frame_samples(self, frame_id: int) -> LingBotFrameSamples:
        return self._load_samples(frame_id)

    @staticmethod
    def _mask_ids_for_xy(mask_image: np.ndarray, xy: np.ndarray, image_shape: tuple[int, int] | None) -> np.ndarray:
        mask = np.asarray(mask_image)
        if mask.ndim < 2 or xy.size == 0:
            return np.zeros((xy.shape[0],), dtype=np.int64)
        mh, mw = mask.shape[:2]
        if image_shape is None:
            sh, sw = mh, mw
        else:
            sh, sw = image_shape
        scale_x = float(max(mw - 1, 1)) / float(max(sw - 1, 1))
        scale_y = float(max(mh - 1, 1)) / float(max(sh - 1, 1))
        mx = np.rint(xy[:, 0] * scale_x).astype(np.int64)
        my = np.rint(xy[:, 1] * scale_y).astype(np.int64)
        in_bounds = (mx >= 0) & (mx < mw) & (my >= 0) & (my < mh)
        ids = np.zeros((xy.shape[0],), dtype=np.int64)
        if np.any(in_bounds):
            ids[in_bounds] = mask[my[in_bounds], mx[in_bounds]].astype(np.int64)
        return ids

    def project_frame_masks(
        self,
        *,
        dataset: object,
        scene_points: np.ndarray,
        mask_image: np.ndarray,
        frame_id: int,
        depth_max_pre: float,
    ) -> FrameProjection:
        del dataset, depth_max_pre
        samples = self._load_samples(frame_id)
        points = np.asarray(samples.points, dtype=np.float32).reshape(-1, 3)
        xy = samples.xy
        finite = np.isfinite(points).all(axis=1)
        if xy is not None and xy.shape[0] == points.shape[0]:
            xy = np.asarray(xy, dtype=np.float32)[finite]
        points = points[finite]
        if points.size == 0 or scene_points.size == 0:
            return FrameProjection(
                mask_info={},
                frame_point_ids=[],
                depth_max=0.0,
                diagnostics={
                    "provider": self.name,
                    "source": samples.source,
                    "bss_frame_id": int(frame_id),
                    "source_frame_id": samples.source_frame_id,
                    "sampling_json": self.sampling_json.as_posix() if self.sampling_json is not None else "",
                    "local_point_count": int(points.shape[0]),
                    "projection_hit_rate": 0.0,
                    "num_frame_points": 0,
                    "has_pixel_samples": bool(xy is not None),
                },
            )
        tree = cKDTree(np.asarray(scene_points, dtype=np.float32))
        dist, idx = tree.query(points, k=1, distance_upper_bound=self.nn_radius)
        hit = np.isfinite(dist) & (idx < len(scene_points))
        frame_points = sorted(set(int(v) for v in idx[hit].tolist()))
        mask_info: dict[int, set[int]] = {}
        positive_mask_samples = 0
        if frame_points and xy is not None and xy.shape[0] == hit.shape[0]:
            mask_ids = self._mask_ids_for_xy(np.asarray(mask_image), xy, samples.image_shape)
            for point_id, mask_id in zip(idx[hit], mask_ids[hit]):
                if int(mask_id) <= 0:
                    continue
                positive_mask_samples += 1
                mask_info.setdefault(int(mask_id), set()).add(int(point_id))
        elif frame_points:
            positive_ids = np.unique(np.asarray(mask_image)[np.asarray(mask_image) > 0]).astype(np.int64)
            if positive_ids.size == 1:
                mask_info[int(positive_ids[0])] = set(frame_points)
        return FrameProjection(
            mask_info=mask_info,
            frame_point_ids=frame_points,
            depth_max=0.0,
            diagnostics={
                "provider": self.name,
                "source": samples.source,
                "bss_frame_id": int(frame_id),
                "source_frame_id": samples.source_frame_id,
                "sampling_json": self.sampling_json.as_posix() if self.sampling_json is not None else "",
                "local_point_count": int(points.shape[0]),
                "projection_hit_rate": float(np.mean(hit)) if hit.size else 0.0,
                "num_frame_points": int(len(frame_points)),
                "num_masks_projected": int(len(mask_info)),
                "has_pixel_samples": bool(xy is not None),
                "positive_mask_samples": int(positive_mask_samples),
                "mask_support_point_assignments": int(sum(len(v) for v in mask_info.values())),
                "uses_d4rt": False,
                "uses_da3": False,
            },
        )
