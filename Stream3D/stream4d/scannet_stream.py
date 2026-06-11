from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ScanNetStream:
    def __init__(
        self,
        seq_name: str,
        backbone: str = "Cropformer",
        root: str | Path = "data/scannet/processed",
    ) -> None:
        self.seq_name = seq_name
        self.backbone = backbone
        self.root = Path(root) / seq_name
        self.rgb_dir = self.root / "color"
        self.depth_dir = self.root / "depth"
        self.pose_dir = self.root / "pose"
        self.intrinsic_dir = self.root / "intrinsic"
        self.mask_dir = self.root / f"output_{backbone}" / "mask"
        self.object_dir = self.root / f"output_{backbone}" / "object"
        self.mesh_path = self.root / f"{seq_name}_vh_clean_2.ply"
        self.depth_scale = 1000.0

    def validate(self, require_masks: bool = True) -> list[str]:
        errors: list[str] = []
        for label, path in [
            ("scene root", self.root),
            ("rgb dir", self.rgb_dir),
            ("depth dir", self.depth_dir),
            ("pose dir", self.pose_dir),
            ("intrinsic dir", self.intrinsic_dir),
            ("mesh", self.mesh_path),
        ]:
            if not path.exists():
                errors.append(f"missing {label}: {path}")
        if require_masks and not self.mask_dir.exists():
            errors.append(f"missing 2D mask dir for backbone={self.backbone}: {self.mask_dir}")
        return errors

    def frame_ids(self, stride: int = 10, max_frames: int | None = None) -> list[int]:
        image_paths = sorted(self.rgb_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not image_paths:
            raise FileNotFoundError(f"No RGB frames found in {self.rgb_dir}")
        selected = [int(p.stem) for p in image_paths if int(p.stem) % int(stride) == 0]
        if max_frames is not None:
            selected = selected[: int(max_frames)]
        return selected

    def load_rgb(self, frame_id: int) -> np.ndarray:
        path = self.rgb_dir / f"{int(frame_id)}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read RGB frame: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_mask(self, frame_id: int) -> np.ndarray:
        path = self.mask_dir / f"{int(frame_id)}.png"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read mask frame: {path}")
        if image.ndim == 3:
            image = image[..., 0]
        return image

    def load_depth(self, frame_id: int) -> np.ndarray:
        path = self.depth_dir / f"{int(frame_id)}.png"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Failed to read depth frame: {path}")
        return depth.astype(np.float32) / self.depth_scale

    def load_pose(self, frame_id: int) -> np.ndarray:
        path = self.pose_dir / f"{int(frame_id)}.txt"
        return np.loadtxt(path).astype(np.float32)

    def load_intrinsics(self) -> np.ndarray:
        path = self.intrinsic_dir / "intrinsic_depth.txt"
        return np.loadtxt(path).astype(np.float32)

    def load_window(self, frame_ids: list[int], require_masks: bool = True) -> dict[str, np.ndarray | list[int]]:
        rgbs = [self.load_rgb(fid) for fid in frame_ids]
        masks: list[np.ndarray] = []
        for fid, rgb in zip(frame_ids, rgbs):
            try:
                masks.append(self.load_mask(fid))
            except FileNotFoundError:
                if require_masks:
                    raise
                masks.append(np.zeros(rgb.shape[:2], dtype=np.int32))
        depths = [self.load_depth(fid) for fid in frame_ids]
        poses = [self.load_pose(fid) for fid in frame_ids]
        return {
            "frame_ids": list(frame_ids),
            "rgb": np.stack(rgbs, axis=0),
            "mask": np.stack(masks, axis=0),
            "depth": np.stack(depths, axis=0),
            "pose": np.stack(poses, axis=0),
            "intrinsics": self.load_intrinsics(),
        }
