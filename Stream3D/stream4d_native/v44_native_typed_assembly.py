from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STAGE1_GATE = {
    "4D_ARI": 0.485,
    "4D_purity": 0.875,
    "4D_completeness": 0.555,
    "temporal_span_mean": 1.70,
    "scene0081_ARI": 0.270,
    "mean_predictions_per_scene": 150.0,
    "duplicate_rate": 0.05,
    "conflict_rate": 0.10,
    "unknown_tube_ratio": 0.35,
    "birth_from_d4rt_tube_count": 0,
    "mixed_birth_count": 0,
}


DEFAULT_SCENES = [
    "scene0030_00",
    "scene0081_01",
    "scene0591_00",
    "scene0011_00",
    "scene0050_00",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_safe(row[key]), sort_keys=True) if isinstance(row.get(key), (dict, list, tuple)) else row.get(key, "") for key in keys})


def read_split(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read mask image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def _rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _xy_from_uv(uv: np.ndarray, shape: tuple[int, int]) -> tuple[int, int] | None:
    uv = np.asarray(uv, dtype=np.float32).reshape(2)
    if not np.isfinite(uv).all():
        return None
    if float(uv[0]) < 0.0 or float(uv[0]) > 1.0 or float(uv[1]) < 0.0 or float(uv[1]) > 1.0:
        return None
    height, width = int(shape[0]), int(shape[1])
    x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, max(width - 1, 0)))
    y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, max(height - 1, 0)))
    return x, y


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _mean(values: list[Any]) -> float | None:
    out: list[float] = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out.append(v)
    return float(np.mean(out)) if out else None


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) > 0.0 else 0.0


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * int(n)

    def find(self, item: int) -> int:
        item = int(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]
        return True


@dataclass
class V44Config:
    scannet_root: Path = Path("data/scannet/processed")
    cache_root: Path = Path("outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    backbone: str = "Cropformer"
    min_mask_area: int = 400
    min_visibility: float = 0.50
    min_confidence: float = 0.50
    max_tubes_per_window: int = 1920
    image_width: int = 640
    image_height: int = 480
    feature_backend: str = "rgb_stats"
    strategy: str = "core_first"
    seed: int = 4401
    core_area_ratio: float = 0.0040
    part_area_ratio: float = 0.0015
    mixed_area_ratio: float = 0.045
    mixed_variance_threshold: float = 0.19
    mixed_boundary_threshold: float = 0.18
    unknown_min_support: int = 2
    link_max_rank_gap: int = 2
    link_min_shared_tubes: int = 2
    link_min_score: float = 0.26
    link_min_score_recall: float = 0.18
    absorb_min_score: float = 0.44
    max_predictions_per_scene: int = 150
    unknown_label_base: int = 1_000_000


@dataclass(frozen=True)
class MaskKey:
    frame_id: int
    mask_id: int

    def text(self) -> str:
        return f"{int(self.frame_id)}:{int(self.mask_id)}"


@dataclass
class MaskMeasurement:
    index: int
    scene: str
    frame_id: int
    mask_id: int
    area: int
    image_area: int
    bbox_xyxy: tuple[int, int, int, int]
    center_xy: tuple[float, float]
    area_ratio: float
    mean_rgb: np.ndarray
    std_rgb: np.ndarray
    feature: np.ndarray
    rgb_variance: float
    core_nonempty: bool
    boundary_nonempty: bool
    boundary_contrast: float
    boundary_gradient: float
    prototype_count: int
    d4rt_support_count: int = 0
    d4rt_observation_count: int = 0
    mixedness: float = 0.0
    core_score: float = 0.0
    part_score: float = 0.0
    duplicate_score: float = 0.0
    role: str = "unknown"
    role_reasons: list[str] = field(default_factory=list)

    @property
    def key(self) -> MaskKey:
        return MaskKey(frame_id=int(self.frame_id), mask_id=int(self.mask_id))


@dataclass
class Objectlet:
    objectlet_id: int
    scene: str
    frame_id: int
    primary: MaskKey
    role: str
    feature: np.ndarray
    center_xy: tuple[float, float]
    area: int
    support: Counter[int] = field(default_factory=Counter)
    absorbed: list[MaskKey] = field(default_factory=list)
    mixed_source: bool = False
    pseudo_core: bool = False

    def all_keys(self) -> list[MaskKey]:
        return [self.primary, *self.absorbed]


@dataclass
class SceneRun:
    scene: str
    strategy: str
    status: str
    measurements: list[MaskMeasurement]
    objectlets: list[Objectlet]
    labels_pred: dict[int, int]
    gt_labels: dict[int, int]
    metrics: dict[str, Any]
    row: dict[str, Any]
    diagnostics: dict[str, Any]


def stage1_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "ari_pass": _value_ge(metrics.get("4D_ARI"), STAGE1_GATE["4D_ARI"]),
        "purity_pass": _value_ge(metrics.get("4D_purity"), STAGE1_GATE["4D_purity"]),
        "completeness_pass": _value_ge(metrics.get("4D_completeness"), STAGE1_GATE["4D_completeness"]),
        "temporal_span_pass": _value_ge(metrics.get("temporal_span_mean"), STAGE1_GATE["temporal_span_mean"]),
        "scene0081_pass": _value_ge(metrics.get("scene0081_ARI"), STAGE1_GATE["scene0081_ARI"]),
        "prediction_count_pass": _value_le(metrics.get("mean_predictions_per_scene"), STAGE1_GATE["mean_predictions_per_scene"]),
        "duplicate_rate_pass": _value_le(metrics.get("duplicate_rate"), STAGE1_GATE["duplicate_rate"]),
        "conflict_rate_pass": _value_le(metrics.get("conflict_rate"), STAGE1_GATE["conflict_rate"]),
        "unknown_tube_ratio_pass": _value_le(metrics.get("unknown_tube_ratio"), STAGE1_GATE["unknown_tube_ratio"]),
        "no_d4rt_birth_pass": _int_metric(metrics.get("birth_from_d4rt_tube_count")) == 0,
        "no_mixed_birth_pass": _int_metric(metrics.get("mixed_birth_count")) == 0,
    }
    checks["stage1_significant_gate_pass"] = bool(all(checks.values()))
    return checks


def _value_ge(value: Any, threshold: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v >= float(threshold))


def _value_le(value: Any, threshold: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v <= float(threshold))


def _int_metric(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ari(labels_true: list[int], labels_pred: list[int]) -> float | None:
    if len(labels_true) != len(labels_pred) or not labels_true:
        return None
    n = len(labels_true)
    if n < 2:
        return 1.0
    contingency: dict[tuple[int, int], int] = Counter(zip(labels_true, labels_pred))
    true_counts: Counter[int] = Counter(labels_true)
    pred_counts: Counter[int] = Counter(labels_pred)

    def comb2(x: int) -> int:
        return int(x * (x - 1) // 2)

    sum_comb = sum(comb2(v) for v in contingency.values())
    sum_true = sum(comb2(v) for v in true_counts.values())
    sum_pred = sum(comb2(v) for v in pred_counts.values())
    total = comb2(n)
    if total == 0:
        return 1.0
    expected = float(sum_true * sum_pred / total)
    denom = float(0.5 * (sum_true + sum_pred) - expected)
    if abs(denom) <= 1e-12:
        return 0.0
    return float((sum_comb - expected) / denom)


def cluster_metrics(labels_pred: dict[int, int], gt_labels: dict[int, int]) -> dict[str, Any]:
    labeled = [int(tube_id) for tube_id in sorted(labels_pred) if int(gt_labels.get(int(tube_id), 0)) > 0]
    true = [int(gt_labels[tube_id]) for tube_id in labeled]
    pred = [int(labels_pred[tube_id]) for tube_id in labeled]
    comp_to_labels: dict[int, Counter[int]] = defaultdict(Counter)
    gt_to_comps: dict[int, set[int]] = defaultdict(set)
    for tube_id in labeled:
        comp = int(labels_pred[tube_id])
        gt = int(gt_labels[tube_id])
        comp_to_labels[comp][gt] += 1
        gt_to_comps[gt].add(comp)
    purity_num = int(sum(max(counts.values()) for counts in comp_to_labels.values() if counts))
    completeness_num = 0
    for gt, comps in gt_to_comps.items():
        completeness_num += max(comp_to_labels[comp].get(gt, 0) for comp in comps)
    return {
        "ari": _ari(true, pred),
        "purity": float(purity_num / max(len(labeled), 1)),
        "completeness": float(completeness_num / max(len(labeled), 1)),
        "overmerge": int(sum(1 for counts in comp_to_labels.values() if len(counts) > 1)),
        "oversplit": int(sum(1 for comps in gt_to_comps.values() if len(comps) > 1)),
        "labeled_tube_count": int(len(labeled)),
    }


def load_tube_records(scene: str, config: V44Config) -> tuple[list[Any], dict[str, Any]]:
    from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
    from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache

    chunks, diag = load_scene_chunks_from_cache(
        Path(config.cache_root) / scene,
        max_tubes_per_window=int(config.max_tubes_per_window),
        image_width=int(config.image_width),
        image_height=int(config.image_height),
    )
    builder = D4RTNativeSceneBuilder(
        object(),
        {"model": {"input": {"clip_frames": 32}}},
        temporal_chunk_size=32,
        temporal_chunk_stride=16,
    )
    records = chunks_to_records(builder.stitch_to_canonical(chunks))
    frame_ids = sorted({int(v) for chunk in chunks for v in chunk.get("chunk", {}).get("frame_ids", [])})
    diag = {
        **diag,
        "cache_root": str(Path(config.cache_root) / scene),
        "cache_window_count": int(len(chunks)),
        "cache_frame_count": int(len(frame_ids)),
        "cache_frame_ids_head": frame_ids[:8],
        "cache_frame_ids_tail": frame_ids[-8:],
        "tube_count": int(len(records)),
    }
    return records, diag


def _frame_ids_from_cache(scene: str, config: V44Config) -> list[int]:
    scene_dir = Path(config.cache_root) / scene
    frame_ids: set[int] = set()
    for manifest_path in sorted(scene_dir.glob("carriers_window*_manifest.json")):
        payload = _load_json(manifest_path)
        values = payload.get("frame_ids") or payload.get("raw_frame_ids")
        if values:
            frame_ids.update(int(v) for v in values)
    if frame_ids:
        return sorted(frame_ids)
    for npz_path in sorted(scene_dir.glob("carriers_window*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            frame_count = int(np.asarray(data["uv_pred"]).shape[0])
            frame_ids.update(range(frame_count))
    return sorted(frame_ids)


def _visible(tube: Any, local_idx: int, config: V44Config) -> bool:
    uv = np.asarray(tube.uv[local_idx], dtype=np.float32)
    return bool(
        np.isfinite(uv).all()
        and 0.0 <= float(uv[0]) <= 1.0
        and 0.0 <= float(uv[1]) <= 1.0
        and float(tube.visibility[local_idx]) >= float(config.min_visibility)
        and float(tube.confidence[local_idx]) >= float(config.min_confidence)
    )


def collect_tube_mask_observations(
    *,
    scene: str,
    tubes: list[Any],
    labels_by_frame: dict[int, np.ndarray],
    config: V44Config,
) -> tuple[dict[MaskKey, Counter[int]], dict[int, Counter[MaskKey]], dict[str, Any]]:
    support_by_mask: dict[MaskKey, Counter[int]] = defaultdict(Counter)
    support_by_tube: dict[int, Counter[MaskKey]] = defaultdict(Counter)
    visible_obs = 0
    positive_obs = 0
    frame_hits: Counter[int] = Counter()
    for tube in tubes:
        frames = np.asarray(tube.target_frames_global, dtype=np.int64)
        for local_idx, frame_id in enumerate(frames.tolist()):
            label = labels_by_frame.get(int(frame_id))
            if label is None:
                continue
            if not _visible(tube, local_idx, config):
                continue
            visible_obs += 1
            xy = _xy_from_uv(np.asarray(tube.uv[local_idx], dtype=np.float32), label.shape)
            if xy is None:
                continue
            x, y = xy
            mask_id = int(label[y, x])
            frame_hits[int(frame_id)] += 1
            if mask_id <= 0:
                continue
            key = MaskKey(frame_id=int(frame_id), mask_id=int(mask_id))
            support_by_mask[key][int(tube.tube_id)] += 1
            support_by_tube[int(tube.tube_id)][key] += 1
            positive_obs += 1
    diag = {
        "scene": scene,
        "visible_tube_observation_count": int(visible_obs),
        "positive_mask_observation_count": int(positive_obs),
        "positive_observation_rate": float(positive_obs / max(visible_obs, 1)),
        "support_mask_count": int(len(support_by_mask)),
        "support_tube_count": int(len(support_by_tube)),
        "frame_observation_count": int(len(frame_hits)),
    }
    return support_by_mask, support_by_tube, diag


def _gradient_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb.astype(np.uint8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag / max(float(np.percentile(mag, 95)), 1e-6)


def _mask_feature(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    pixels = rgb[np.asarray(mask, dtype=bool)].astype(np.float32) / 255.0
    if pixels.size == 0:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32), 0.0
    mean = pixels.mean(axis=0).astype(np.float32)
    std = pixels.std(axis=0).astype(np.float32)
    variance = float(np.mean(std))
    feature = np.concatenate([mean, std], axis=0).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm > 1e-8:
        feature = feature / norm
    return feature, std, variance


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def build_mask_measurements(
    *,
    scene: str,
    frame_ids: list[int],
    labels_by_frame: dict[int, np.ndarray],
    support_by_mask: dict[MaskKey, Counter[int]],
    config: V44Config,
) -> tuple[list[MaskMeasurement], dict[str, Any]]:
    scene_root = Path(config.scannet_root) / scene
    measurements: list[MaskMeasurement] = []
    core_nonempty = 0
    boundary_nonempty = 0
    descriptor_frames = 0
    for frame_id in frame_ids:
        label = labels_by_frame.get(int(frame_id))
        if label is None:
            continue
        rgb_path = scene_root / "color" / f"{int(frame_id)}.jpg"
        if not rgb_path.exists():
            continue
        rgb = _rgb_image(rgb_path)
        grad = _gradient_map(rgb)
        image_area = int(label.shape[0] * label.shape[1])
        descriptor_frames += 1
        for mask_id in sorted(int(v) for v in np.unique(label) if int(v) > 0):
            mask = label == int(mask_id)
            area = int(np.count_nonzero(mask))
            if area < int(config.min_mask_area):
                continue
            kernel_size = 5 if area >= 1200 else 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            if int(np.count_nonzero(eroded)) == 0:
                eroded = mask
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            boundary = (dilated ^ eroded) & dilated
            feature, std, rgb_variance = _mask_feature(rgb, eroded)
            mean_rgb = (rgb[mask].astype(np.float32) / 255.0).mean(axis=0).astype(np.float32)
            outside_ring = boundary & (~mask)
            inside_ring = boundary & mask
            if np.any(inside_ring) and np.any(outside_ring):
                inside_mean = (rgb[inside_ring].astype(np.float32) / 255.0).mean(axis=0)
                outside_mean = (rgb[outside_ring].astype(np.float32) / 255.0).mean(axis=0)
                boundary_contrast = float(np.linalg.norm(inside_mean - outside_mean))
            else:
                boundary_contrast = 0.0
            boundary_gradient = float(np.mean(grad[boundary])) if np.any(boundary) else 0.0
            proto = 1
            if area > 7000 and rgb_variance > 0.12:
                proto = 2
            if area > 20000 and rgb_variance > 0.16:
                proto = 3
            if area > 50000 and rgb_variance > 0.20:
                proto = 4
            key = MaskKey(int(frame_id), int(mask_id))
            support = support_by_mask.get(key, Counter())
            meas = MaskMeasurement(
                index=len(measurements),
                scene=scene,
                frame_id=int(frame_id),
                mask_id=int(mask_id),
                area=area,
                image_area=image_area,
                bbox_xyxy=_bbox(mask),
                center_xy=_center(mask),
                area_ratio=float(area / max(image_area, 1)),
                mean_rgb=mean_rgb,
                std_rgb=std,
                feature=feature,
                rgb_variance=float(rgb_variance),
                core_nonempty=bool(np.any(eroded)),
                boundary_nonempty=bool(np.any(boundary)),
                boundary_contrast=float(boundary_contrast),
                boundary_gradient=float(boundary_gradient),
                prototype_count=int(proto),
                d4rt_support_count=int(len(support)),
                d4rt_observation_count=int(sum(support.values())),
            )
            _infer_role(meas, config)
            measurements.append(meas)
            core_nonempty += int(meas.core_nonempty)
            boundary_nonempty += int(meas.boundary_nonempty)
    diag = {
        "scene": scene,
        "feature_backend": str(config.feature_backend),
        "descriptor_frame_count": int(descriptor_frames),
        "mask_count": int(len(measurements)),
        "descriptor_success_rate": 1.0 if measurements else 0.0,
        "core_nonempty_rate": float(core_nonempty / max(len(measurements), 1)),
        "boundary_nonempty_rate": float(boundary_nonempty / max(len(measurements), 1)),
        "prototype_count_mean": _mean([m.prototype_count for m in measurements]),
        "feature_variance_mean": _mean([m.rgb_variance for m in measurements]),
        "boundary_contrast_mean": _mean([m.boundary_contrast for m in measurements]),
        "d4rt_supported_mask_rate": float(sum(1 for m in measurements if m.d4rt_support_count > 0) / max(len(measurements), 1)),
    }
    return measurements, diag


def _infer_role(meas: MaskMeasurement, config: V44Config) -> None:
    support_norm = min(1.0, math.log1p(float(meas.d4rt_support_count)) / math.log(80.0))
    area_core = min(1.0, float(meas.area_ratio) / max(float(config.core_area_ratio), 1e-6))
    area_part = 1.0 - min(1.0, float(meas.area_ratio) / max(float(config.part_area_ratio) * 4.0, 1e-6))
    semantic_cohesion = max(0.0, 1.0 - float(meas.rgb_variance) / 0.30)
    boundary_signal = min(1.0, max(float(meas.boundary_contrast), float(meas.boundary_gradient)))
    multimodal = min(1.0, (float(meas.prototype_count) - 1.0) / 3.0)
    large = min(1.0, float(meas.area_ratio) / max(float(config.mixed_area_ratio), 1e-6))
    meas.mixedness = float(
        0.35 * multimodal
        + 0.25 * min(1.0, float(meas.rgb_variance) / max(float(config.mixed_variance_threshold), 1e-6))
        + 0.20 * large
        + 0.20 * max(0.0, 1.0 - boundary_signal)
    )
    meas.core_score = float(0.38 * area_core + 0.27 * semantic_cohesion + 0.25 * support_norm + 0.10 * boundary_signal - 0.36 * meas.mixedness)
    meas.part_score = float(0.42 * area_part + 0.30 * support_norm + 0.18 * boundary_signal + 0.10 * semantic_cohesion - 0.20 * meas.mixedness)
    meas.duplicate_score = float(0.0)

    reasons: list[str] = []
    if meas.area_ratio >= float(config.mixed_area_ratio) and meas.rgb_variance >= float(config.mixed_variance_threshold):
        meas.role = "mixed"
        reasons.append("large_high_variance")
    elif meas.mixedness >= 0.70 and meas.prototype_count >= 3:
        meas.role = "mixed"
        reasons.append("multi_proto_mixedness")
    elif meas.d4rt_support_count < int(config.unknown_min_support) and meas.area_ratio < float(config.core_area_ratio):
        meas.role = "unknown"
        reasons.append("low_support_small")
    elif meas.core_score >= 0.43 and meas.mixedness < 0.66:
        meas.role = "core"
        reasons.append("core_score")
    elif meas.part_score >= 0.42 and meas.mixedness < 0.72:
        meas.role = "part"
        reasons.append("part_score")
    elif meas.core_score >= 0.35 and meas.d4rt_support_count >= int(config.unknown_min_support):
        meas.role = "core"
        reasons.append("relaxed_core_supported")
    else:
        meas.role = "unknown"
        reasons.append("ambiguous")
    meas.role_reasons = reasons


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float(math.sqrt(dx * dx + dy * dy))


def _absorb_score(part: MaskMeasurement, core: MaskMeasurement) -> float:
    sim = (_cosine(part.feature, core.feature) + 1.0) * 0.5
    gap = _bbox_gap(part.bbox_xyxy, core.bbox_xyxy)
    diag = math.sqrt(float(core.image_area))
    layout = max(0.0, 1.0 - gap / max(diag * 0.20, 1.0))
    support_overlap = 0.0
    return float(0.42 * sim + 0.38 * layout + 0.20 * support_overlap - 0.22 * part.mixedness)


def _make_objectlets(
    *,
    scene: str,
    measurements: list[MaskMeasurement],
    support_by_mask: dict[MaskKey, Counter[int]],
    config: V44Config,
) -> tuple[list[Objectlet], dict[str, Any]]:
    by_frame: dict[int, list[MaskMeasurement]] = defaultdict(list)
    for meas in measurements:
        by_frame[int(meas.frame_id)].append(meas)

    objectlets: list[Objectlet] = []
    objectlet_by_key: dict[MaskKey, int] = {}
    births_by_role: Counter[str] = Counter()
    absorb_rows: list[dict[str, Any]] = []
    allow_unknown_birth = str(config.strategy) in {"balanced_recall", "repair_completeness"}
    allow_part_pseudo = str(config.strategy) in {"pseudo_core", "balanced_recall", "repair_completeness"}
    strict_purity = str(config.strategy) in {"repair_purity", "core_first_strict"}

    for meas in measurements:
        birth = False
        pseudo = False
        role = meas.role
        if role == "core":
            birth = not (strict_purity and meas.core_score < 0.50)
        elif role == "part" and allow_part_pseudo and meas.d4rt_support_count >= max(3, int(config.unknown_min_support)):
            birth = True
            pseudo = True
        elif role == "unknown" and allow_unknown_birth and meas.d4rt_support_count >= 4 and meas.mixedness < 0.55:
            birth = True
            pseudo = True
        if not birth:
            continue
        oid = len(objectlets)
        obj = Objectlet(
            objectlet_id=oid,
            scene=scene,
            frame_id=int(meas.frame_id),
            primary=meas.key,
            role=role,
            feature=meas.feature,
            center_xy=meas.center_xy,
            area=int(meas.area),
            support=Counter(support_by_mask.get(meas.key, Counter())),
            mixed_source=(role == "mixed"),
            pseudo_core=bool(pseudo),
        )
        objectlets.append(obj)
        objectlet_by_key[meas.key] = oid
        births_by_role[role] += 1

    for frame_id, frame_measurements in by_frame.items():
        frame_cores = [m for m in frame_measurements if m.key in objectlet_by_key]
        if not frame_cores:
            continue
        for part in frame_measurements:
            if part.role not in {"part", "duplicate", "unknown"}:
                continue
            if part.key in objectlet_by_key:
                continue
            scores = [(_absorb_score(part, core), core) for core in frame_cores if core.role != "mixed"]
            if not scores:
                continue
            scores.sort(key=lambda item: item[0], reverse=True)
            best_score, best_core = scores[0]
            second = scores[1][0] if len(scores) > 1 else 0.0
            margin = float(best_score - second)
            threshold = float(config.absorb_min_score) + (0.06 if strict_purity else 0.0)
            if best_score >= threshold and (margin >= 0.02 or len(scores) == 1):
                oid = objectlet_by_key[best_core.key]
                objectlets[oid].absorbed.append(part.key)
                objectlets[oid].support.update(support_by_mask.get(part.key, Counter()))
                absorb_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "part": part.key.text(),
                        "core": best_core.key.text(),
                        "score": float(best_score),
                        "margin": float(margin),
                        "part_role": part.role,
                    }
                )

    diag = {
        "objectlet_count": int(len(objectlets)),
        "births_by_role": dict(births_by_role),
        "absorb_accept_count": int(len(absorb_rows)),
        "absorb_rows_sample": absorb_rows[:50],
        "mixed_birth_count": int(births_by_role.get("mixed", 0)),
        "part_only_birth_count": int(births_by_role.get("part", 0)),
        "pseudo_core_count": int(sum(1 for obj in objectlets if obj.pseudo_core)),
    }
    return objectlets, diag


def _maybe_shuffle_support(
    support_by_mask: dict[MaskKey, Counter[int]],
    *,
    seed: int,
) -> dict[MaskKey, Counter[int]]:
    rng = np.random.default_rng(int(seed))
    all_tubes = sorted({int(tube) for counter in support_by_mask.values() for tube in counter})
    if not all_tubes:
        return {key: Counter(counter) for key, counter in support_by_mask.items()}
    out: dict[MaskKey, Counter[int]] = {}
    for key, counter in support_by_mask.items():
        shuffled = Counter()
        values = list(counter.items())
        sampled = rng.choice(all_tubes, size=len(values), replace=len(values) > len(all_tubes))
        for (_, count), new_tube in zip(values, sampled.tolist()):
            shuffled[int(new_tube)] += int(count)
        out[key] = shuffled
    return out


def link_objectlets(
    objectlets: list[Objectlet],
    *,
    config: V44Config,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    n = len(objectlets)
    uf = UnionFind(n)
    component_frames: dict[int, set[int]] = {idx: {int(obj.frame_id)} for idx, obj in enumerate(objectlets)}
    frame_rank = {frame: rank for rank, frame in enumerate(sorted({int(obj.frame_id) for obj in objectlets}))}
    candidate_rows: list[dict[str, Any]] = []
    no_temporal = str(config.strategy) == "no_temporal"
    mask_only = str(config.strategy) == "mask_only"
    min_score = float(config.link_min_score_recall if str(config.strategy) in {"balanced_recall", "repair_completeness"} else config.link_min_score)
    if not no_temporal:
        for i, left in enumerate(objectlets):
            for j in range(i + 1, n):
                right = objectlets[j]
                rank_gap = abs(frame_rank.get(int(left.frame_id), 0) - frame_rank.get(int(right.frame_id), 0))
                if rank_gap <= 0 or rank_gap > int(config.link_max_rank_gap):
                    continue
                shared = sum((left.support & right.support).values())
                union = sum((left.support | right.support).values())
                tube_jaccard = float(shared / max(union, 1))
                appearance = (_cosine(left.feature, right.feature) + 1.0) * 0.5
                dx = float(left.center_xy[0] - right.center_xy[0])
                dy = float(left.center_xy[1] - right.center_xy[1])
                spatial = max(0.0, 1.0 - math.sqrt(dx * dx + dy * dy) / 850.0)
                if mask_only:
                    score = float(0.70 * appearance + 0.30 * spatial)
                    shared_gate = True
                else:
                    score = float(0.62 * tube_jaccard + 0.25 * appearance + 0.13 * spatial)
                    shared_gate = int(shared) >= int(config.link_min_shared_tubes)
                if score < min_score or not shared_gate:
                    continue
                candidate_rows.append(
                    {
                        "left": int(i),
                        "right": int(j),
                        "frame_left": int(left.frame_id),
                        "frame_right": int(right.frame_id),
                        "rank_gap": int(rank_gap),
                        "shared": int(shared),
                        "tube_jaccard": float(tube_jaccard),
                        "appearance": float(appearance),
                        "spatial": float(spatial),
                        "score": float(score),
                    }
                )
        for row in sorted(candidate_rows, key=lambda item: item["score"], reverse=True):
            left = int(row["left"])
            right = int(row["right"])
            root_left = uf.find(left)
            root_right = uf.find(right)
            if root_left == root_right:
                continue
            if component_frames[root_left] & component_frames[root_right]:
                continue
            if uf.union(root_left, root_right):
                new_root = uf.find(root_left)
                old_root = root_right if new_root == root_left else root_left
                component_frames[new_root] = component_frames.get(root_left, set()) | component_frames.get(root_right, set())
                component_frames.pop(old_root, None)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups[uf.find(idx)].append(idx)
    diag = {
        "link_candidate_count": int(len(candidate_rows)),
        "link_accept_count": int(n - len(groups)),
        "link_rows_sample": sorted(candidate_rows, key=lambda item: item["score"], reverse=True)[:100],
        "no_temporal": bool(no_temporal),
        "mask_only": bool(mask_only),
    }
    return {int(k): [int(v) for v in vals] for k, vals in groups.items()}, diag


def _assign_tubes_to_components(
    objectlets: list[Objectlet],
    components: dict[int, list[int]],
    gt_labels: dict[int, int],
    *,
    unknown_label_base: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    tube_votes: dict[int, Counter[int]] = defaultdict(Counter)
    component_supports: dict[int, set[int]] = {}
    object_id_by_root: dict[int, int] = {}
    for out_id, root in enumerate(sorted(components)):
        object_id_by_root[int(root)] = int(out_id)
        support: Counter[int] = Counter()
        for obj_idx in components[root]:
            support.update(objectlets[int(obj_idx)].support)
        component_supports[int(root)] = set(int(t) for t in support)
        for tube_id, count in support.items():
            tube_votes[int(tube_id)][int(out_id)] += int(count)

    labels_pred: dict[int, int] = {}
    unknown_count = 0
    assigned = 0
    conflict = 0
    next_unknown = int(unknown_label_base)
    for tube_id, gt in sorted(gt_labels.items()):
        if int(gt) <= 0:
            continue
        votes = tube_votes.get(int(tube_id), Counter())
        if not votes:
            labels_pred[int(tube_id)] = next_unknown
            next_unknown += 1
            unknown_count += 1
            continue
        top = votes.most_common()
        if len(top) > 1 and int(top[0][1]) == int(top[1][1]):
            conflict += 1
        labels_pred[int(tube_id)] = int(top[0][0])
        assigned += 1

    duplicate_pairs = 0
    roots = sorted(component_supports)
    for pos, left in enumerate(roots):
        a = component_supports[left]
        if not a:
            continue
        for right in roots[pos + 1 :]:
            b = component_supports[right]
            if not b:
                continue
            jaccard = len(a & b) / max(len(a | b), 1)
            duplicate_pairs += int(jaccard >= 0.80)
    diag = {
        "assigned_labeled_tube_count": int(assigned),
        "unknown_labeled_tube_count": int(unknown_count),
        "labeled_tube_count": int(sum(1 for v in gt_labels.values() if int(v) > 0)),
        "unknown_tube_ratio": float(unknown_count / max(sum(1 for v in gt_labels.values() if int(v) > 0), 1)),
        "assignment_conflict_count": int(conflict),
        "assignment_conflict_rate": float(conflict / max(assigned + unknown_count, 1)),
        "duplicate_pair_count": int(duplicate_pairs),
        "duplicate_rate": float(duplicate_pairs / max(len(roots), 1)),
    }
    return labels_pred, diag


def select_components(
    objectlets: list[Objectlet],
    components: dict[int, list[int]],
    *,
    max_components: int,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    if int(max_components) <= 0 or len(components) <= int(max_components):
        return components, {
            "selection_applied": False,
            "pre_selection_component_count": int(len(components)),
            "post_selection_component_count": int(len(components)),
            "dropped_component_count": 0,
        }
    scored: list[tuple[float, int]] = []
    for root, members in components.items():
        support = Counter()
        frames: set[int] = set()
        area = 0
        core_count = 0
        pseudo_count = 0
        for idx in members:
            obj = objectlets[int(idx)]
            support.update(obj.support)
            frames.add(int(obj.frame_id))
            area += int(obj.area)
            core_count += int(obj.role == "core")
            pseudo_count += int(obj.pseudo_core)
        score = (
            1.00 * float(len(support))
            + 0.12 * float(sum(support.values()))
            + 8.0 * float(len(frames))
            + 0.015 * math.sqrt(float(max(area, 0)))
            + 3.0 * float(core_count)
            + 1.0 * float(pseudo_count)
        )
        scored.append((float(score), int(root)))
    keep = {
        root
        for _, root in sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[: int(max_components)]
    }
    selected = {int(root): list(members) for root, members in components.items() if int(root) in keep}
    return selected, {
        "selection_applied": True,
        "pre_selection_component_count": int(len(components)),
        "post_selection_component_count": int(len(selected)),
        "dropped_component_count": int(len(components) - len(selected)),
        "selection_policy": "top_support_span_area_count_penalty",
        "max_components": int(max_components),
    }


def _role_counts(measurements: list[MaskMeasurement]) -> dict[str, int]:
    counts = Counter(m.role for m in measurements)
    return {role: int(counts.get(role, 0)) for role in ["core", "part", "mixed", "duplicate", "unknown"]}


def _measurement_rows(measurements: list[MaskMeasurement]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in measurements:
        rows.append(
            {
                "scene": m.scene,
                "frame_id": int(m.frame_id),
                "mask_id": int(m.mask_id),
                "area": int(m.area),
                "area_ratio": float(m.area_ratio),
                "d4rt_support_count": int(m.d4rt_support_count),
                "d4rt_observation_count": int(m.d4rt_observation_count),
                "rgb_variance": float(m.rgb_variance),
                "boundary_contrast": float(m.boundary_contrast),
                "boundary_gradient": float(m.boundary_gradient),
                "prototype_count": int(m.prototype_count),
                "mixedness": float(m.mixedness),
                "core_score": float(m.core_score),
                "part_score": float(m.part_score),
                "role": m.role,
                "role_reasons": ";".join(m.role_reasons),
            }
        )
    return rows


def _objectlet_rows(objectlets: list[Objectlet], components: dict[int, list[int]]) -> list[dict[str, Any]]:
    comp_by_idx: dict[int, int] = {}
    for out_id, root in enumerate(sorted(components)):
        for idx in components[root]:
            comp_by_idx[int(idx)] = int(out_id)
    rows: list[dict[str, Any]] = []
    for obj in objectlets:
        rows.append(
            {
                "objectlet_id": int(obj.objectlet_id),
                "object_id": int(comp_by_idx.get(int(obj.objectlet_id), -1)),
                "frame_id": int(obj.frame_id),
                "primary": obj.primary.text(),
                "role": obj.role,
                "support_tube_count": int(len(obj.support)),
                "support_observation_count": int(sum(obj.support.values())),
                "absorbed_count": int(len(obj.absorbed)),
                "absorbed": [key.text() for key in obj.absorbed],
                "pseudo_core": bool(obj.pseudo_core),
                "mixed_source": bool(obj.mixed_source),
            }
        )
    return rows


def _temporal_span_mean(objectlets: list[Objectlet], components: dict[int, list[int]]) -> float | None:
    spans: list[int] = []
    for root in sorted(components):
        frames = {int(objectlets[idx].frame_id) for idx in components[root]}
        if frames:
            spans.append(len(frames))
    return float(np.mean(spans)) if spans else None


def _load_gt_labels(scene: str, tubes: list[Any], config: V44Config) -> dict[int, int]:
    from stream4d.scannet_stream import ScanNetStream
    from tools.run_v26_object_quality_diagnostics import assign_gt_labels

    return assign_gt_labels(
        tubes,
        stream=ScanNetStream(seq_name=scene, backbone=config.backbone, root=config.scannet_root),
        min_visibility=float(config.min_visibility),
        min_confidence=float(config.min_confidence),
    )


def run_scene(scene: str, config: V44Config, *, output_root: Path | None = None) -> SceneRun:
    frame_ids = _frame_ids_from_cache(scene, config)
    scene_root = Path(config.scannet_root) / scene
    mask_dir = scene_root / f"output_{config.backbone}" / "mask"
    labels_by_frame: dict[int, np.ndarray] = {}
    missing_mask_frames: list[int] = []
    for frame_id in frame_ids:
        path = mask_dir / f"{int(frame_id)}.png"
        if path.exists():
            labels_by_frame[int(frame_id)] = _mask_image(path)
        else:
            missing_mask_frames.append(int(frame_id))

    tubes, tube_diag = load_tube_records(scene, config)
    support_by_mask, support_by_tube, support_diag = collect_tube_mask_observations(
        scene=scene,
        tubes=tubes,
        labels_by_frame=labels_by_frame,
        config=config,
    )
    effective_support = support_by_mask
    if str(config.strategy) == "shuffled_d4rt":
        effective_support = _maybe_shuffle_support(support_by_mask, seed=int(config.seed))
    if str(config.strategy) == "mask_only":
        effective_support = {key: Counter() for key in support_by_mask}

    measurements, descriptor_diag = build_mask_measurements(
        scene=scene,
        frame_ids=frame_ids,
        labels_by_frame=labels_by_frame,
        support_by_mask=effective_support,
        config=config,
    )
    objectlets, objectlet_diag = _make_objectlets(
        scene=scene,
        measurements=measurements,
        support_by_mask=effective_support,
        config=config,
    )
    raw_components, link_diag = link_objectlets(objectlets, config=config)
    components, selection_diag = select_components(
        objectlets,
        raw_components,
        max_components=int(config.max_predictions_per_scene),
    )
    gt_labels = _load_gt_labels(scene, tubes, config)
    labels_pred, assignment_diag = _assign_tubes_to_components(
        objectlets,
        components,
        gt_labels,
        unknown_label_base=int(config.unknown_label_base),
    )
    metrics = cluster_metrics(labels_pred, gt_labels)
    temporal_span = _temporal_span_mean(objectlets, components)
    role_counts = _role_counts(measurements)
    row = {
        "scene": scene,
        "strategy": str(config.strategy),
        "status": "ok",
        "feature_backend": str(config.feature_backend),
        "prediction_uses_gt": False,
        "prediction_uses_pose": False,
        "prediction_uses_rgbd": False,
        "prediction_uses_scannet_mesh": False,
        "gt_used_only_for_scoring": True,
        "mask_frame_count": int(len(labels_by_frame)),
        "missing_mask_frame_count": int(len(missing_mask_frames)),
        "tube_count": int(len(tubes)),
        "mask_count": int(len(measurements)),
        "core_candidate_count": int(role_counts.get("core", 0)),
        "part_candidate_count": int(role_counts.get("part", 0)),
        "mixed_candidate_count": int(role_counts.get("mixed", 0)),
        "duplicate_candidate_count": int(role_counts.get("duplicate", 0)),
        "unknown_candidate_count": int(role_counts.get("unknown", 0)),
        "unknown_rate": float(role_counts.get("unknown", 0) / max(len(measurements), 1)),
        "selected_objectlet_count": int(len(objectlets)),
        "selected_object_field_count": int(len(components)),
        "predictions_per_scene": int(len(components)),
        "temporal_span_mean": temporal_span,
        "4D_ARI": metrics.get("ari"),
        "4D_purity": metrics.get("purity"),
        "4D_completeness": metrics.get("completeness"),
        "overmerge": metrics.get("overmerge"),
        "oversplit": metrics.get("oversplit"),
        "metric_labeled_tube_count": metrics.get("labeled_tube_count"),
        "unknown_tube_ratio": assignment_diag["unknown_tube_ratio"],
        "conflict_rate": assignment_diag["assignment_conflict_rate"],
        "duplicate_rate": assignment_diag["duplicate_rate"],
        "birth_from_d4rt_tube_count": 0,
        "mixed_birth_count": objectlet_diag["mixed_birth_count"],
        "part_only_birth_count": objectlet_diag["part_only_birth_count"],
        "pseudo_core_count": objectlet_diag["pseudo_core_count"],
    }
    diagnostics = {
        "tube_cache": tube_diag,
        "support": support_diag,
        "descriptor": descriptor_diag,
        "role_counts": role_counts,
        "objectlets": objectlet_diag,
        "links": link_diag,
        "selection": selection_diag,
        "assignment": assignment_diag,
        "missing_mask_frames_head": missing_mask_frames[:20],
        "stage1_scene_gate": {
            "scene_ari_ge_0_270_if_scene0081": (row["4D_ARI"] is not None and float(row["4D_ARI"]) >= 0.270)
            if scene == "scene0081_01"
            else None,
            "prediction_count_pass": int(len(components)) <= int(config.max_predictions_per_scene),
            "no_d4rt_birth": True,
            "no_mixed_birth": int(objectlet_diag["mixed_birth_count"]) == 0,
        },
    }
    if output_root is not None:
        scene_dir = Path(output_root) / "scene_details" / scene
        write_json(scene_dir / "scene_summary.json", {"row": row, "diagnostics": diagnostics})
        write_csv(scene_dir / "mask_measurements.csv", _measurement_rows(measurements))
        write_csv(scene_dir / "objectlets.csv", _objectlet_rows(objectlets, components))
    return SceneRun(
        scene=scene,
        strategy=str(config.strategy),
        status="ok",
        measurements=measurements,
        objectlets=objectlets,
        labels_pred=labels_pred,
        gt_labels=gt_labels,
        metrics=metrics,
        row=row,
        diagnostics=diagnostics,
    )


def _offset_labels(scene_index: int, pred: dict[int, int], gt: dict[int, int]) -> tuple[dict[int, int], dict[int, int]]:
    key_base = int(scene_index) * 10_000_000
    pred_base = int(scene_index) * 10_000_000
    gt_base = int(scene_index) * 10_000_000
    pred_out = {key_base + int(tid): pred_base + int(label) for tid, label in pred.items()}
    gt_out = {
        key_base + int(tid): gt_base + int(label)
        for tid, label in gt.items()
        if int(label) > 0 and int(tid) in pred
    }
    return pred_out, gt_out


def summarize_runs(runs: list[SceneRun], *, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    aggregate_pred: dict[int, int] = {}
    aggregate_gt: dict[int, int] = {}
    for scene_index, run in enumerate(runs):
        pred, gt = _offset_labels(scene_index, run.labels_pred, run.gt_labels)
        aggregate_pred.update(pred)
        aggregate_gt.update(gt)
    agg = cluster_metrics(aggregate_pred, aggregate_gt) if aggregate_pred else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
        "labeled_tube_count": 0,
    }
    scene_rows = [run.row for run in runs]
    scene0081 = next((row for row in scene_rows if row.get("scene") == "scene0081_01"), {})
    aggregate_metrics = {
        "4D_ARI": agg.get("ari"),
        "4D_purity": agg.get("purity"),
        "4D_completeness": agg.get("completeness"),
        "temporal_span_mean": _mean([row.get("temporal_span_mean") for row in scene_rows]),
        "scene0081_ARI": scene0081.get("4D_ARI"),
        "mean_predictions_per_scene": _mean([row.get("predictions_per_scene") for row in scene_rows]),
        "duplicate_rate": _mean([row.get("duplicate_rate") for row in scene_rows]),
        "conflict_rate": _mean([row.get("conflict_rate") for row in scene_rows]),
        "unknown_tube_ratio": _mean([row.get("unknown_tube_ratio") for row in scene_rows]),
        "birth_from_d4rt_tube_count": int(sum(int(row.get("birth_from_d4rt_tube_count", 0) or 0) for row in scene_rows)),
        "mixed_birth_count": int(sum(int(row.get("mixed_birth_count", 0) or 0) for row in scene_rows)),
        "labeled_tube_count": agg.get("labeled_tube_count"),
        "overmerge": agg.get("overmerge"),
        "oversplit": agg.get("oversplit"),
    }
    gate = stage1_gate(aggregate_metrics)
    comparison: dict[str, Any] = {}
    if baseline:
        v37 = baseline.get("v37_best_metrics") or {}
        v41 = baseline.get("v41_1_best_metrics") or {}
        for prefix, ref in [("v37", v37), ("v41_1", v41)]:
            ref_ari = _extract_metric(ref, ["4D_ARI", "ari"])
            ref_comp = _extract_metric(ref, ["4D_completeness", "completeness"])
            ref_purity = _extract_metric(ref, ["4D_purity", "purity"])
            comparison[f"delta_ARI_vs_{prefix}"] = _delta(aggregate_metrics.get("4D_ARI"), ref_ari)
            comparison[f"delta_completeness_vs_{prefix}"] = _delta(aggregate_metrics.get("4D_completeness"), ref_comp)
            comparison[f"delta_purity_vs_{prefix}"] = _delta(aggregate_metrics.get("4D_purity"), ref_purity)
    return {
        "status": "ok",
        "scene_count": int(len(runs)),
        "scenes": [run.scene for run in runs],
        "strategy": runs[0].strategy if runs else "",
        "prediction_uses_gt": False,
        "prediction_uses_pose": False,
        "prediction_uses_rgbd": False,
        "prediction_uses_scannet_mesh": False,
        "gt_used_only_for_scoring": True,
        "metric_scope_note": "tube-level v44 typed-mask identity metrics; not ScanNet AP",
        "aggregate_metrics": aggregate_metrics,
        "gate": gate,
        "comparison_to_baselines": comparison,
        "baseline": baseline or {},
        "scene_rows": scene_rows,
    }


def _extract_metric(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _delta(value: Any, ref: Any) -> float | None:
    try:
        v = float(value)
        r = float(ref)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(v) and np.isfinite(r)):
        return None
    return float(v - r)


def load_baseline_bundle(v37_path: Path | None = None, v41_path: Path | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if v37_path is not None:
        out["v37_path"] = str(v37_path)
        if v37_path.exists():
            payload = _load_json(v37_path)
            out["v37_status"] = "ok"
            out["v37_final_status"] = payload.get("final_status")
            out["v37_best_metrics"] = payload.get("best_metrics", payload.get("aggregate_metrics", {}))
        else:
            out["v37_status"] = "missing"
    if v41_path is not None:
        out["v41_1_path"] = str(v41_path)
        if v41_path.exists():
            payload = _load_json(v41_path)
            out["v41_1_status"] = "ok"
            out["v41_1_final_status"] = payload.get("status") or payload.get("final_status")
            out["v41_1_best_metrics"] = payload.get("aggregate_metrics", payload.get("best_metrics", {}))
        else:
            out["v41_1_status"] = "missing"
    return out


def run_scenes(
    scenes: list[str],
    config: V44Config,
    *,
    output_root: Path,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    runs = [run_scene(scene, config, output_root=output_root) for scene in scenes]
    summary = summarize_runs(runs, baseline=baseline)
    write_json(output_root / "v44_native_typed_summary.json", summary)
    write_csv(output_root / "v44_native_typed_scene_rows.csv", [run.row for run in runs])
    write_json(
        output_root / "config_manifest.json",
        {
            "phase": "v44_native_typed_mask_assembly",
            "plan": "docs/stream4d_v44_typed_mask_assembly_plan.md",
            "strategy": str(config.strategy),
            "feature_backend": str(config.feature_backend),
            "prediction_uses_gt": False,
            "prediction_uses_pose": False,
            "prediction_uses_rgbd": False,
            "prediction_uses_scannet_mesh": False,
            "gt_used_only_for_scoring": True,
            "uses_old_candidate_first_route": False,
            "uses_v37_residual_repair": False,
            "d4rt_tubes_can_birth_object": False,
            "cache_root": str(config.cache_root),
            "scannet_root": str(config.scannet_root),
            "scenes": scenes,
        },
    )
    return summary
