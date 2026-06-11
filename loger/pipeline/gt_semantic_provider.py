"""GT semantic label discovery and loading utilities.

The v25 GT-semantic experiments are oracle diagnostics.  This module only
accepts explicit GT-style layouts and never falls back to predicted Stage-C
semantic masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IGNORE_LABEL = 255


KITTI_STEP_ID_TO_NAME: Dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    255: "void",
}


SEMANTIC_KITTI_ID_TO_NAME: Dict[int, str] = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on_rails",
    18: "truck",
    20: "other_vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other_ground",
    50: "building",
    51: "fence",
    52: "other_structure",
    60: "lane_marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic_sign",
    99: "other_object",
    252: "moving_car",
    253: "moving_bicyclist",
    254: "moving_person",
    255: "moving_motorcyclist",
    256: "moving_on_rails",
    257: "moving_bus",
    258: "moving_truck",
    259: "moving_other_vehicle",
}


CITYSCAPES_COLOR_TO_ID: Dict[Tuple[int, int, int], int] = {
    (128, 64, 128): 0,   # road
    (244, 35, 232): 1,   # sidewalk
    (70, 70, 70): 2,     # building
    (102, 102, 156): 3,  # wall
    (190, 153, 153): 4,  # fence
    (153, 153, 153): 5,  # pole
    (250, 170, 30): 6,   # traffic light
    (220, 220, 0): 7,    # traffic sign
    (107, 142, 35): 8,   # vegetation
    (152, 251, 152): 9,  # terrain
    (70, 130, 180): 10,  # sky
    (220, 20, 60): 11,   # person
    (255, 0, 0): 12,     # rider
    (0, 0, 142): 13,     # car
    (0, 0, 70): 14,      # truck
    (0, 60, 100): 15,    # bus
    (0, 80, 100): 16,    # train
    (0, 0, 230): 17,     # motorcycle
    (119, 11, 32): 18,   # bicycle
}


@dataclass(frozen=True)
class GTSemanticLayout:
    name: str
    kind: str
    label_dir: Path
    image_dir: Optional[Path] = None
    calib_path: Optional[Path] = None
    velodyne_dir: Optional[Path] = None
    point_label_dir: Optional[Path] = None
    frame_digits: int = 6
    suffix: str = ".png"
    semantic_id_encoding: str = "single_channel"
    note: str = ""

    @property
    def dense_image_map(self) -> bool:
        return self.kind in {
            "dense_id_png",
            "dense_rgb_png",
            "kitti_step_panoptic_png",
        }

    @property
    def point_projection(self) -> bool:
        return self.kind == "semantic_kitti_point_projection"

    def frame_name(self, frame: int) -> str:
        return f"{frame:0{self.frame_digits}d}{self.suffix}"

    def label_path(self, frame: int) -> Path:
        return self.label_dir / self.frame_name(frame)

    def image_path(self, frame: int) -> Optional[Path]:
        if self.image_dir is None:
            return None
        return self.image_dir / self.frame_name(frame)

    def has_frame(self, frame: int) -> bool:
        if self.point_projection:
            if self.velodyne_dir is None or self.point_label_dir is None:
                return False
            return (
                (self.velodyne_dir / f"{frame:06d}.bin").exists()
                and (self.point_label_dir / f"{frame:06d}.label").exists()
                and self.calib_path is not None
                and self.calib_path.exists()
            )
        return self.label_path(frame).exists()


@dataclass
class GTSemanticFrame:
    frame: int
    semantic: np.ndarray
    valid_mask: np.ndarray
    source_path: str
    layout_name: str
    coverage: float
    label_counts: Dict[int, int]


def _dedupe_layouts(layouts: Iterable[GTSemanticLayout]) -> List[GTSemanticLayout]:
    seen = set()
    out: List[GTSemanticLayout] = []
    for layout in layouts:
        key = (
            layout.name,
            layout.kind,
            str(layout.label_dir),
            str(layout.image_dir or ""),
            str(layout.velodyne_dir or ""),
            str(layout.point_label_dir or ""),
            layout.frame_digits,
            layout.suffix,
            layout.semantic_id_encoding,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(layout)
    return out


def _candidate_roots(
    sequence_root: Path,
    explicit_gt_root: Optional[Path],
) -> List[Path]:
    roots = [
        sequence_root,
        sequence_root.parent,
        sequence_root.parent.parent,
        sequence_root.parent.parent.parent,
    ]
    if explicit_gt_root is not None:
        roots.extend(
            [
                explicit_gt_root,
                explicit_gt_root / "data_semantics",
                explicit_gt_root / "KITTI",
                explicit_gt_root / "data" / "KITTI",
                explicit_gt_root / "BiSeNetv2" / "data" / "KITTI",
            ]
        )
    return [root.resolve() for root in roots if str(root)]


def discover_gt_semantic_layouts(
    *,
    sequence_root: Path,
    sequence: str,
    explicit_gt_root: Optional[Path] = None,
) -> List[GTSemanticLayout]:
    """Return plausible GT semantic layouts without walking the filesystem."""

    sequence_root = sequence_root.resolve()
    sequence = str(sequence)
    sequence_4 = f"{int(sequence):04d}" if sequence.isdigit() else sequence
    layouts: List[GTSemanticLayout] = []

    # Sequence-local dense labels sometimes used by converted datasets.
    for label_name in ("semantic", "semantics", "semantic_labels", "image_2_semantic"):
        layouts.append(
            GTSemanticLayout(
                name=f"sequence_local:{label_name}",
                kind="dense_id_png",
                label_dir=sequence_root / label_name,
                image_dir=sequence_root / "image_2",
                frame_digits=6,
                suffix=".png",
                semantic_id_encoding="single_channel",
            )
        )

    # SemanticKITTI-style point labels projected to image_2.
    layouts.append(
        GTSemanticLayout(
            name="semantic_kitti:sequence_point_projection",
            kind="semantic_kitti_point_projection",
            label_dir=sequence_root / "labels",
            image_dir=sequence_root / "image_2",
            calib_path=sequence_root / "calib.txt",
            velodyne_dir=sequence_root / "velodyne",
            point_label_dir=sequence_root / "labels",
            frame_digits=6,
            suffix=".label",
            semantic_id_encoding="semantic_kitti_uint32_lower16",
            note="Requires sequence velodyne/*.bin, labels/*.label, and calib Tr/Tr_velo_to_cam.",
        )
    )

    for root in _candidate_roots(sequence_root, explicit_gt_root):
        # KITTI semantic segmentation benchmark / common PyTorch repos:
        # training/{image_2,semantic,semantic_rgb}.  Filenames are often
        # 000000_10.png, so we check both the odometry-style and stereo2015 style.
        for base in (root, root / "data_semantics", root / "KITTI", root / "data" / "KITTI"):
            train = base / "training"
            layouts.extend(
                [
                    GTSemanticLayout(
                        name=f"kitti_semantic_benchmark:{train}:semantic",
                        kind="dense_id_png",
                        label_dir=train / "semantic",
                        image_dir=train / "image_2",
                        frame_digits=6,
                        suffix=".png",
                        semantic_id_encoding="single_channel",
                        note="KITTI semantic benchmark / PointPainting-style layout.",
                    ),
                    GTSemanticLayout(
                        name=f"kitti_semantic_benchmark:{train}:semantic_10",
                        kind="dense_id_png",
                        label_dir=train / "semantic",
                        image_dir=train / "image_2",
                        frame_digits=6,
                        suffix="_10.png",
                        semantic_id_encoding="single_channel",
                        note="KITTI Stereo2015 filenames often use *_10.png.",
                    ),
                    GTSemanticLayout(
                        name=f"kitti_semantic_benchmark:{train}:semantic_rgb",
                        kind="dense_rgb_png",
                        label_dir=train / "semantic_rgb",
                        image_dir=train / "image_2",
                        frame_digits=6,
                        suffix=".png",
                        semantic_id_encoding="cityscapes_rgb",
                        note="RGB color-coded semantic labels.",
                    ),
                    GTSemanticLayout(
                        name=f"kitti_semantic_benchmark:{train}:semantic_rgb_10",
                        kind="dense_rgb_png",
                        label_dir=train / "semantic_rgb",
                        image_dir=train / "image_2",
                        frame_digits=6,
                        suffix="_10.png",
                        semantic_id_encoding="cityscapes_rgb",
                        note="RGB color-coded semantic labels with Stereo2015 *_10 filenames.",
                    ),
                ]
            )

        # KITTI-STEP / DeepLab2-style layout:
        # panoptic_maps/{train,val}/{sequence_id}/{frame_id}.png with R=semantic.
        for split in ("train", "val", "training", "validation"):
            layouts.append(
                GTSemanticLayout(
                    name=f"kitti_step:{root}:panoptic_maps:{split}:{sequence_4}",
                    kind="kitti_step_panoptic_png",
                    label_dir=root / "panoptic_maps" / split / sequence_4,
                    image_dir=root / "images" / split / sequence_4,
                    frame_digits=6,
                    suffix=".png",
                    semantic_id_encoding="rgb_r_semantic_gb_instance",
                    note="KITTI-STEP panoptic PNG; R channel is semantic id.",
                )
            )

        # KITTI-360 dense 2D semantics. This is not odometry sequence 01, but
        # supporting it keeps the v25 audit aligned with common KITTI-family
        # 2D semantic training data.
        for split in ("train", "val"):
            for image_name in ("image_00", "image_01"):
                drive = f"2013_05_28_drive_{sequence_4}_sync"
                layouts.append(
                    GTSemanticLayout(
                        name=f"kitti360:{root}:data_2d_semantics:{split}:{drive}:{image_name}",
                        kind="dense_id_png",
                        label_dir=root / "data_2d_semantics" / split / drive / image_name / "semantic",
                        image_dir=root / "data_2d_raw" / drive / image_name / "data_rect",
                        frame_digits=10,
                        suffix=".png",
                        semantic_id_encoding="single_channel",
                        note="KITTI-360 2D semantic labels.",
                    )
                )

        # Explicit SemanticKITTI root can be passed as .../dataset or ... .
        seq_dir = root / "sequences" / sequence
        layouts.append(
            GTSemanticLayout(
                name=f"semantic_kitti:{seq_dir}:point_projection",
                kind="semantic_kitti_point_projection",
                label_dir=seq_dir / "labels",
                image_dir=seq_dir / "image_2",
                calib_path=seq_dir / "calib.txt",
                velodyne_dir=seq_dir / "velodyne",
                point_label_dir=seq_dir / "labels",
                frame_digits=6,
                suffix=".label",
                semantic_id_encoding="semantic_kitti_uint32_lower16",
                note="Requires SemanticKITTI labels and KITTI odometry velodyne scans.",
            )
        )

    return _dedupe_layouts(layouts)


def read_kitti_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        vals = np.array([float(x) for x in values.strip().split()], dtype=np.float64)
        if vals.size == 12:
            data[key.strip()] = vals.reshape(3, 4)
        elif vals.size == 9:
            data[key.strip()] = vals.reshape(3, 3)
    return data


def _homogeneous_4x4(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = mat
        return out
    raise ValueError(f"Unsupported transform shape: {mat.shape}")


def _projection_matrices(calib_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    calib = read_kitti_calib(calib_path)
    if "P2" not in calib:
        raise ValueError(f"Missing P2 in calibration file: {calib_path}")
    tr = None
    for key in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam"):
        if key in calib:
            tr = calib[key]
            break
    if tr is None:
        raise ValueError(
            f"Missing Tr/Tr_velo_to_cam in calibration file: {calib_path}"
        )
    r0 = calib.get("R0_rect", np.eye(3, dtype=np.float64))
    r0_4 = np.eye(4, dtype=np.float64)
    r0_4[:3, :3] = r0
    return calib["P2"], r0_4 @ _homogeneous_4x4(tr)


class GTSemanticProvider:
    def __init__(self, layout: GTSemanticLayout, *, image_size: Optional[Tuple[int, int]] = None):
        self.layout = layout
        self.image_size = image_size

    def load_frame(self, frame: int) -> GTSemanticFrame:
        if self.layout.kind == "semantic_kitti_point_projection":
            return self._load_projected_semantic_kitti(frame)
        return self._load_dense_png(frame)

    def _load_dense_png(self, frame: int) -> GTSemanticFrame:
        from PIL import Image  # type: ignore

        path = self.layout.label_path(frame)
        with Image.open(path) as img:
            arr = np.array(img)
        if self.layout.kind == "kitti_step_panoptic_png":
            if arr.ndim != 3 or arr.shape[2] < 1:
                raise ValueError(f"KITTI-STEP panoptic map must be RGB: {path}")
            semantic = arr[:, :, 0].astype(np.int32)
        elif self.layout.kind == "dense_rgb_png":
            semantic = np.full(arr.shape[:2], IGNORE_LABEL, dtype=np.int32)
            rgb = arr[:, :, :3]
            for color, label in CITYSCAPES_COLOR_TO_ID.items():
                mask = np.all(rgb == np.array(color, dtype=rgb.dtype), axis=-1)
                semantic[mask] = label
        else:
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            semantic = arr.astype(np.int32)
        valid = semantic != IGNORE_LABEL
        return self._frame_result(frame, semantic, valid, str(path))

    def _load_projected_semantic_kitti(self, frame: int) -> GTSemanticFrame:
        if self.layout.velodyne_dir is None or self.layout.point_label_dir is None:
            raise ValueError("Point projection layout is missing velodyne/label directories")
        if self.layout.calib_path is None:
            raise ValueError("Point projection layout is missing calibration path")
        image_size = self.image_size or self._infer_image_size(frame)
        if image_size is None:
            raise ValueError("Cannot infer image size for point-label projection")
        width, height = image_size
        p2, velo_to_rect = _projection_matrices(self.layout.calib_path)
        points_path = self.layout.velodyne_dir / f"{frame:06d}.bin"
        labels_path = self.layout.point_label_dir / f"{frame:06d}.label"
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(labels_path, dtype=np.uint32)
        if points.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Point/label count mismatch for frame {frame}: "
                f"{points.shape[0]} points vs {labels.shape[0]} labels"
            )
        semantic_ids = (labels & np.uint32(0xFFFF)).astype(np.int32)
        pts_h = np.ones((points.shape[0], 4), dtype=np.float64)
        pts_h[:, :3] = points[:, :3].astype(np.float64)
        rect = (velo_to_rect @ pts_h.T).T
        in_front = rect[:, 2] > 1e-6
        proj = (p2 @ rect.T).T
        uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        valid = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        semantic = np.full((height, width), IGNORE_LABEL, dtype=np.int32)
        zbuf = np.full((height, width), np.inf, dtype=np.float64)
        for px, py, depth, label in zip(u[valid], v[valid], rect[valid, 2], semantic_ids[valid]):
            if depth < zbuf[py, px]:
                zbuf[py, px] = depth
                semantic[py, px] = int(label)
        valid_mask = semantic != IGNORE_LABEL
        return self._frame_result(frame, semantic, valid_mask, str(labels_path))

    def _infer_image_size(self, frame: int) -> Optional[Tuple[int, int]]:
        if self.layout.image_dir is None:
            return None
        from PIL import Image  # type: ignore

        candidates = [
            self.layout.image_dir / f"{frame:06d}.png",
            self.layout.image_dir / f"{frame:010d}.png",
            self.layout.image_dir / f"{frame:06d}_10.png",
        ]
        for path in candidates:
            if path.exists():
                with Image.open(path) as img:
                    return int(img.size[0]), int(img.size[1])
        return None

    def _frame_result(
        self,
        frame: int,
        semantic: np.ndarray,
        valid_mask: np.ndarray,
        source_path: str,
    ) -> GTSemanticFrame:
        unique, counts = np.unique(semantic[valid_mask], return_counts=True)
        label_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        coverage = float(valid_mask.mean()) if valid_mask.size else 0.0
        return GTSemanticFrame(
            frame=frame,
            semantic=semantic,
            valid_mask=valid_mask,
            source_path=source_path,
            layout_name=self.layout.name,
            coverage=coverage,
            label_counts=label_counts,
        )
