from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_radio_checkpoint  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d  # noqa: E402


DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v92_phase1_source_container_registry/source_container_rows.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/audit/v92_phase4_semantic_region_affinity"

REGION_NODE_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "source_mask_id",
    "region_id",
    "region_index",
    "feature_y",
    "feature_x",
    "pixel_count",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "centroid_x",
    "centroid_y",
    "radio_feature_hash",
    "radio_feature_norm",
    "dino_feature_hash",
    "dino_feature_norm",
    "mean_rgb_r",
    "mean_rgb_g",
    "mean_rgb_b",
    "source_mean_cosine",
    "boundary_token",
    "center_distance_norm",
    "diagnostic_gt_id",
    "diagnostic_gt_fraction",
    "diagnostic_only_uses_gt",
    "uses_gt_for_prediction",
    "uses_future",
]

REGION_EDGE_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "source_mask_id",
    "region_id_a",
    "region_id_b",
    "edge_kind",
    "radio_cosine",
    "radio_contrast",
    "same_diagnostic_gt",
    "both_foreground_gt",
    "diagnostic_only_uses_gt",
    "uses_gt_for_prediction",
    "uses_future",
]

SEPARABILITY_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "source_mask_id",
    "region_node_count",
    "region_edge_count",
    "region_feature_variance_within_source",
    "region_graph_modularity_proxy",
    "source_internal_feature_boundary_strength",
    "adjacent_region_feature_contrast_mean",
    "foreground_seed_to_background_seed_margin_proxy",
    "mask_level_feature_vs_region_feature_gap",
    "source_internal_same_gt_different_gt_AUC",
    "foreground_background_region_AUC",
    "same_instance_region_recall_at_topk",
    "different_instance_region_false_positive_rate",
    "part_whole_region_affinity_diagnostic",
    "diagnostic_pair_count",
    "diagnostic_positive_pair_count",
    "diagnostic_negative_pair_count",
    "diagnostic_only_uses_gt",
    "uses_gt_for_prediction",
    "uses_future",
]

AUC_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scope",
    "scene_id",
    "metric_name",
    "auc",
    "sample_count",
    "positive_count",
    "negative_count",
    "score_mean_positive",
    "score_mean_negative",
    "diagnostic_only_uses_gt",
    "uses_gt_for_prediction",
    "uses_future",
]

FAILURE_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "failure_type",
    "repair_direction",
    "detail",
    "uses_gt_for_prediction",
    "uses_future",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_workspace_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_vec(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(path)


def _parse_scenes(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(np.asarray(vals, dtype=np.float32))) if vals else None


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pos = sum(1 for label in labels if int(label) == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _score, label in ordered[i:j] if int(label) == 1)
        i = j
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _read_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def _resize_label_nearest(label: np.ndarray, height: int, width: int) -> np.ndarray:
    if label.shape[:2] == (height, width):
        return label.astype(np.int64, copy=False)
    return cv2.resize(label.astype(np.int32), (int(width), int(height)), interpolation=cv2.INTER_NEAREST).astype(np.int64)


def _stable_sample_pairs(n: int, max_pairs: int, seed_text: str) -> list[tuple[int, int]]:
    if n < 2 or max_pairs <= 0:
        return []
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < max_pairs:
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        pairs.add((i, j))
    return sorted(pairs)


def _load_source_index(source_rows_path: Path, scenes: set[str]) -> dict[tuple[str, int], dict[int, dict[str, Any]]]:
    by_frame: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in _read_csv(source_rows_path):
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        if str(row.get("split", "dev")) != "dev":
            continue
        scene = str(row.get("scene_id", ""))
        if scene not in scenes:
            continue
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("source_mask_id", row.get("mask_id", -1)), -1)
        if frame_id < 0 or mask_id <= 0:
            continue
        if not _bool(row.get("mask_path_exists", "True")):
            continue
        key = (scene, int(frame_id))
        existing = by_frame[key].get(mask_id)
        if existing is None or str(row.get("variant_id", "")) == "B0_local_only":
            by_frame[key][mask_id] = row
    return by_frame


def _window_maps() -> dict[tuple[str, int], tuple[str, int]]:
    out: dict[tuple[str, int], tuple[str, int]] = {}
    for row in _read_csv(ROOT / "outputs/audit/v91_phase0_mv_ap_contract/window_support_rows.csv"):
        scene = str(row.get("scene_id", ""))
        window_id = str(row.get("window_id", ""))
        window_index = _int(row.get("window_index"), -1)
        start = _int(row.get("frame_id_start"), -1)
        end = _int(row.get("frame_id_end"), -1)
        if not scene or window_index < 0 or start < 0 or end < 0:
            continue
        for frame_id in range(start, end + 1, 5):
            out[(scene, int(frame_id))] = (window_id or f"w{window_index:04d}", int(window_index))
    return out


def _token_bbox(feature_y: int, feature_x: int, feature_h: int, feature_w: int, image_h: int, image_w: int) -> tuple[int, int, int, int]:
    x0 = int(math.floor(feature_x * image_w / max(1, feature_w)))
    x1 = int(math.ceil((feature_x + 1) * image_w / max(1, feature_w)))
    y0 = int(math.floor(feature_y * image_h / max(1, feature_h)))
    y1 = int(math.ceil((feature_y + 1) * image_h / max(1, feature_h)))
    x0 = max(0, min(image_w - 1, x0))
    x1 = max(x0 + 1, min(image_w, x1))
    y0 = max(0, min(image_h - 1, y0))
    y1 = max(y0 + 1, min(image_h, y1))
    return x0, y0, x1, y1


def _majority_gt(gt_patch: np.ndarray, mask_patch: np.ndarray) -> tuple[int, float]:
    vals = gt_patch[np.asarray(mask_patch, dtype=bool)]
    if vals.size == 0:
        vals = gt_patch.reshape(-1)
    if vals.size == 0:
        return 0, 0.0
    counts = Counter(int(v) for v in vals.tolist())
    gt_id, count = counts.most_common(1)[0]
    return int(gt_id), float(count / max(1, int(vals.size)))


def _region_rows_for_source(
    *,
    scene: str,
    frame_id: int,
    window_id: str,
    split: str,
    mask_id: int,
    mask_row: dict[str, Any],
    label: np.ndarray,
    gt: np.ndarray,
    rgb: np.ndarray,
    features: np.ndarray,
    run_id: str,
    min_region_pixels: int,
    max_pair_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    image_h, image_w = label.shape[:2]
    feature_h, feature_w, _feature_dim = features.shape
    small_label = _resize_label_nearest(label, feature_h, feature_w)
    small_mask = small_label == int(mask_id)
    if not np.any(small_mask):
        return [], [], None, []
    source_mask = label == int(mask_id)
    if not np.any(source_mask):
        return [], [], None, []
    ys, xs = np.nonzero(source_mask)
    source_cx = float(xs.mean())
    source_cy = float(ys.mean())
    source_scale = float(max(1.0, math.sqrt(float(np.count_nonzero(source_mask)))))
    token_coords = sorted((int(y), int(x)) for y, x in zip(*np.nonzero(small_mask)))
    node_vectors: list[np.ndarray] = []
    node_rows: list[dict[str, Any]] = []
    coord_to_region: dict[tuple[int, int], str] = {}
    for region_index, (fy, fx) in enumerate(token_coords):
        x0, y0, x1, y1 = _token_bbox(fy, fx, feature_h, feature_w, image_h, image_w)
        mask_patch = source_mask[y0:y1, x0:x1]
        pixel_count = int(np.count_nonzero(mask_patch))
        if pixel_count < int(min_region_pixels):
            continue
        region_pixels_y, region_pixels_x = np.nonzero(mask_patch)
        if region_pixels_x.size:
            centroid_x = float(x0 + region_pixels_x.mean())
            centroid_y = float(y0 + region_pixels_y.mean())
            rgb_patch = rgb[y0:y1, x0:x1][mask_patch]
            mean_rgb = rgb_patch.mean(axis=0) if rgb_patch.size else np.zeros((3,), dtype=np.float32)
        else:
            centroid_x = float((x0 + x1 - 1) / 2.0)
            centroid_y = float((y0 + y1 - 1) / 2.0)
            mean_rgb = np.zeros((3,), dtype=np.float32)
        vector = np.asarray(features[fy, fx], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            vector = vector / norm
        gt_id, gt_fraction = _majority_gt(gt[y0:y1, x0:x1], mask_patch)
        boundary = False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            yy = fy + dy
            xx = fx + dx
            if yy < 0 or yy >= feature_h or xx < 0 or xx >= feature_w or not bool(small_mask[yy, xx]):
                boundary = True
                break
        region_id = f"{scene}:{frame_id}:{mask_id}:r{region_index:04d}"
        coord_to_region[(fy, fx)] = region_id
        node_vectors.append(vector)
        node_rows.append(
            {
                "schema_version": "stream4d_v92_phase4_region_node_v1",
                "phase_id": "v92_phase4_semantic_region_affinity",
                "run_id": run_id,
                "scene_id": scene,
                "split": split,
                "window_id": window_id,
                "frame_id": int(frame_id),
                "source_mask_id": int(mask_id),
                "region_id": region_id,
                "region_index": int(region_index),
                "feature_y": int(fy),
                "feature_x": int(fx),
                "pixel_count": int(pixel_count),
                "bbox_x0": int(x0),
                "bbox_y0": int(y0),
                "bbox_x1": int(x1 - 1),
                "bbox_y1": int(y1 - 1),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "radio_feature_hash": _sha256_vec(vector),
                "radio_feature_norm": float(np.linalg.norm(vector)),
                "dino_feature_hash": "",
                "dino_feature_norm": "",
                "mean_rgb_r": float(mean_rgb[0]),
                "mean_rgb_g": float(mean_rgb[1]),
                "mean_rgb_b": float(mean_rgb[2]),
                "source_mean_cosine": "",
                "boundary_token": bool(boundary),
                "center_distance_norm": float(math.hypot(centroid_x - source_cx, centroid_y - source_cy) / source_scale),
                "diagnostic_gt_id": int(gt_id),
                "diagnostic_gt_fraction": float(gt_fraction),
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    if not node_rows:
        return [], [], None, []
    source_vec = np.mean(np.stack(node_vectors, axis=0), axis=0).astype(np.float32)
    source_norm = float(np.linalg.norm(source_vec))
    if source_norm > 1e-8:
        source_vec = source_vec / source_norm
    source_cosines = [_cosine(vec, source_vec) for vec in node_vectors]
    for row, score in zip(node_rows, source_cosines):
        row["source_mean_cosine"] = float(score)

    node_by_id = {row["region_id"]: row for row in node_rows}
    vector_by_id = {row["region_id"]: vec for row, vec in zip(node_rows, node_vectors)}
    edge_rows: list[dict[str, Any]] = []
    for fy, fx in token_coords:
        rid_a = coord_to_region.get((fy, fx))
        if not rid_a:
            continue
        for dy, dx in [(1, 0), (0, 1)]:
            rid_b = coord_to_region.get((fy + dy, fx + dx))
            if not rid_b:
                continue
            row_a = node_by_id[rid_a]
            row_b = node_by_id[rid_b]
            cos = _cosine(vector_by_id[rid_a], vector_by_id[rid_b])
            gt_a = _int(row_a.get("diagnostic_gt_id"), 0)
            gt_b = _int(row_b.get("diagnostic_gt_id"), 0)
            edge_rows.append(
                {
                    "schema_version": "stream4d_v92_phase4_region_edge_v1",
                    "phase_id": "v92_phase4_semantic_region_affinity",
                    "run_id": run_id,
                    "scene_id": scene,
                    "split": split,
                    "window_id": window_id,
                    "frame_id": int(frame_id),
                    "source_mask_id": int(mask_id),
                    "region_id_a": rid_a,
                    "region_id_b": rid_b,
                    "edge_kind": "feature_grid_4nbr_within_source_mask",
                    "radio_cosine": float(cos),
                    "radio_contrast": float(1.0 - cos),
                    "same_diagnostic_gt": bool(gt_a > 0 and gt_a == gt_b),
                    "both_foreground_gt": bool(gt_a > 0 and gt_b > 0),
                    "diagnostic_only_uses_gt": True,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    vectors = node_vectors
    gt_ids = [_int(row.get("diagnostic_gt_id"), 0) for row in node_rows]
    pair_scores: list[float] = []
    pair_labels: list[int] = []
    diff_scores: list[float] = []
    pairs = _stable_sample_pairs(len(node_rows), max_pair_samples, f"{scene}:{frame_id}:{mask_id}")
    for i, j in pairs:
        gt_i = gt_ids[i]
        gt_j = gt_ids[j]
        if gt_i <= 0 or gt_j <= 0:
            continue
        score = _cosine(vectors[i], vectors[j])
        same = int(gt_i == gt_j)
        pair_scores.append(float(score))
        pair_labels.append(same)
        if not same:
            diff_scores.append(float(score))
    same_auc = _auc(pair_scores, pair_labels)
    fg_labels = [1 if gt_id > 0 else 0 for gt_id in gt_ids]
    fg_auc = _auc(source_cosines, fg_labels)
    edge_contrasts = [float(row["radio_contrast"]) for row in edge_rows]
    boundary_scores = [float(row["source_mean_cosine"]) for row in node_rows if _bool(row.get("boundary_token"))]
    center_scores = [float(row["source_mean_cosine"]) for row in node_rows if not _bool(row.get("boundary_token"))]
    top_k = max(1, min(100, int(math.ceil(len(pair_scores) * 0.10)))) if pair_scores else 0
    recall_at_topk = 0.0
    if top_k:
        ranked = sorted(zip(pair_scores, pair_labels), key=lambda item: item[0], reverse=True)[:top_k]
        recall_at_topk = float(sum(int(label) for _score, label in ranked) / max(1, sum(pair_labels)))
    false_pos_rate = float(sum(1 for score in diff_scores if score >= 0.80) / max(1, len(diff_scores))) if diff_scores else 0.0
    unique_gt = sorted({gt for gt in gt_ids if gt > 0})
    feature_arr = np.stack(vectors, axis=0)
    variance = float(np.mean(np.var(feature_arr, axis=0))) if feature_arr.shape[0] > 1 else 0.0
    mean_cos = _mean(source_cosines)
    sep = {
        "schema_version": "stream4d_v92_phase4_separability_v1",
        "phase_id": "v92_phase4_semantic_region_affinity",
        "run_id": run_id,
        "scene_id": scene,
        "split": split,
        "window_id": window_id,
        "frame_id": int(frame_id),
        "source_mask_id": int(mask_id),
        "region_node_count": int(len(node_rows)),
        "region_edge_count": int(len(edge_rows)),
        "region_feature_variance_within_source": variance,
        "region_graph_modularity_proxy": float(_mean(edge_contrasts) / max(1e-8, variance)),
        "source_internal_feature_boundary_strength": _mean(edge_contrasts),
        "adjacent_region_feature_contrast_mean": _mean(edge_contrasts),
        "foreground_seed_to_background_seed_margin_proxy": float(_mean(center_scores) - _mean(boundary_scores)),
        "mask_level_feature_vs_region_feature_gap": float(1.0 - mean_cos),
        "source_internal_same_gt_different_gt_AUC": "" if same_auc is None else float(same_auc),
        "foreground_background_region_AUC": "" if fg_auc is None else float(fg_auc),
        "same_instance_region_recall_at_topk": recall_at_topk,
        "different_instance_region_false_positive_rate": false_pos_rate,
        "part_whole_region_affinity_diagnostic": "multi_gt_source" if len(unique_gt) > 1 else "single_or_no_gt_source",
        "diagnostic_pair_count": int(len(pair_scores)),
        "diagnostic_positive_pair_count": int(sum(pair_labels)),
        "diagnostic_negative_pair_count": int(len(pair_labels) - sum(pair_labels)),
        "diagnostic_only_uses_gt": True,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    auc_rows = _auc_rows_for_source(scene, run_id, pair_scores, pair_labels, source_cosines, fg_labels)
    return node_rows, edge_rows, sep, auc_rows


def _auc_rows_for_source(
    scene: str,
    run_id: str,
    pair_scores: list[float],
    pair_labels: list[int],
    fg_scores: list[float],
    fg_labels: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, scores, labels in [
        ("source_internal_same_gt_different_gt_AUC", pair_scores, pair_labels),
        ("foreground_background_region_AUC", fg_scores, fg_labels),
    ]:
        pos_scores = [float(s) for s, label in zip(scores, labels) if int(label) == 1]
        neg_scores = [float(s) for s, label in zip(scores, labels) if int(label) == 0]
        rows.append(
            {
                "schema_version": "stream4d_v92_phase4_semantic_auc_v1",
                "phase_id": "v92_phase4_semantic_region_affinity",
                "run_id": run_id,
                "scope": "source_mask",
                "scene_id": scene,
                "metric_name": metric_name,
                "auc": "" if not pos_scores or not neg_scores else float(_auc(scores, labels)),
                "sample_count": int(len(scores)),
                "positive_count": int(len(pos_scores)),
                "negative_count": int(len(neg_scores)),
                "score_mean_positive": _mean(pos_scores),
                "score_mean_negative": _mean(neg_scores),
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _aggregate_auc_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_metric_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("scope")) != "source_mask":
            continue
        by_metric_scene[(str(row.get("metric_name", "")), str(row.get("scene_id", "")))].append(row)
        by_metric_scene[(str(row.get("metric_name", "")), "ALL_DEV")].append(row)
    for (metric_name, scene), items in sorted(by_metric_scene.items()):
        auc_values = [_num(row.get("auc"), float("nan")) for row in items if str(row.get("auc", "")) != ""]
        pos = sum(_int(row.get("positive_count"), 0) for row in items)
        neg = sum(_int(row.get("negative_count"), 0) for row in items)
        out.append(
            {
                "schema_version": "stream4d_v92_phase4_semantic_auc_v1",
                "phase_id": "v92_phase4_semantic_region_affinity",
                "run_id": run_id,
                "scope": "scene" if scene != "ALL_DEV" else "all_dev",
                "scene_id": scene,
                "metric_name": metric_name,
                "auc": "" if not auc_values else float(np.average(np.asarray(auc_values, dtype=np.float32))),
                "sample_count": int(sum(_int(row.get("sample_count"), 0) for row in items)),
                "positive_count": int(pos),
                "negative_count": int(neg),
                "score_mean_positive": _mean([_num(row.get("score_mean_positive")) for row in items]),
                "score_mean_negative": _mean([_num(row.get("score_mean_negative")) for row in items]),
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _decision(summary: dict[str, Any]) -> str:
    if not summary.get("radio_available", False):
        return "SEMANTIC_FEATURE_MISSING"
    if int(summary.get("region_node_rows", 0)) <= 0:
        return "SEMANTIC_DIAGNOSTIC_INCONCLUSIVE"
    auc = summary.get("source_internal_same_gt_different_gt_AUC_mean")
    fg_auc = summary.get("foreground_background_region_AUC_mean")
    strong_count = int(summary.get("source_internal_auc_ge_0p60_count", 0))
    if auc is not None and float(auc) >= 0.60 and strong_count >= 10:
        return "SEMANTIC_REGION_SIGNAL_STRONG"
    if fg_auc is not None and float(fg_auc) >= 0.65 and strong_count >= 5:
        return "SEMANTIC_REGION_SIGNAL_STRONG"
    return "SEMANTIC_REGION_SIGNAL_WEAK"


def _run_extract(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve_workspace_path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    scenes = set(_parse_scenes(args.scenes))
    source_rows_path = _resolve_workspace_path(args.source_container_rows)
    source_index = _load_source_index(source_rows_path, scenes)
    frame_items = sorted(source_index.items())
    if int(args.max_frames) > 0:
        frame_items = frame_items[: int(args.max_frames)]
    availability = json.loads((ROOT / "outputs/audit/v46_loger_env_radio_radseg_availability_recheck_20260619/radio_vipe_availability.json").read_text(encoding="utf-8"))
    checkpoint = str(args.checkpoint).strip() or str(availability.get("radio_checkpoint") or locate_default_radio_checkpoint() or "")
    adapter = FrozenFeatureAdapter(
        backend="radio_radseg",
        device=str(args.device),
        checkpoint=checkpoint,
        radio_lang_model=str(args.radio_lang_model),
        radio_lang_align=bool(args.radio_lang_align),
        radio_slide_crop=int(args.radio_slide_crop),
        radio_slide_stride=int(args.radio_slide_stride),
    )
    window_map = _window_maps()
    region_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    sep_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    streams: dict[str, ScanNetStream] = {}
    for index, ((scene, frame_id), masks_by_id) in enumerate(frame_items, start=1):
        frame_started = time.time()
        if index % int(args.progress_every) == 0:
            print(f"[v92-phase4] frame={index}/{len(frame_items)} scene={scene} frame_id={frame_id} masks={len(masks_by_id)}", file=sys.stderr, flush=True)
        try:
            streams.setdefault(scene, ScanNetStream(scene, root=ROOT / "data/scannet/processed"))
            rgb = streams[scene].load_rgb(int(frame_id))
            first_mask_path = _resolve_workspace_path(next(iter(masks_by_id.values())).get("mask_path", ""))
            label = _read_label(first_mask_path)
            gt = _load_gt_2d(scene, int(frame_id), label.shape)
            feature_map = adapter.extract_dense_features(rgb)
        except Exception as exc:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v92_phase4_failure_v1",
                    "phase_id": "v92_phase4_semantic_region_affinity",
                    "run_id": args.run_id,
                    "scene_id": scene,
                    "failure_type": f"frame_feature_extract_failed:{type(exc).__name__}",
                    "repair_direction": "check RADIO checkpoint/import/RGB/mask paths; rerun dense extraction",
                    "detail": str(exc),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue
        features = np.asarray(feature_map.features, dtype=np.float32)
        written_nodes_before = len(region_rows)
        written_edges_before = len(edge_rows)
        window_id, _window_index = window_map.get((scene, int(frame_id)), ("", -1))
        for mask_id, mask_row in sorted(masks_by_id.items()):
            nodes, edges, sep, source_auc_rows = _region_rows_for_source(
                scene=scene,
                frame_id=int(frame_id),
                window_id=window_id,
                split=str(mask_row.get("split", "dev")),
                mask_id=int(mask_id),
                mask_row=mask_row,
                label=label,
                gt=gt,
                rgb=np.asarray(rgb),
                features=features,
                run_id=str(args.run_id),
                min_region_pixels=int(args.min_region_pixels),
                max_pair_samples=int(args.max_pair_samples_per_source),
            )
            region_rows.extend(nodes)
            edge_rows.extend(edges)
            if sep:
                sep_rows.append(sep)
            auc_rows.extend(source_auc_rows)
        frame_rows.append(
            {
                "scene_id": scene,
                "frame_id": int(frame_id),
                "source_mask_count": int(len(masks_by_id)),
                "feature_height": int(features.shape[0]),
                "feature_width": int(features.shape[1]),
                "feature_dim": int(features.shape[2]),
                "region_node_rows": int(len(region_rows) - written_nodes_before),
                "region_edge_rows": int(len(edge_rows) - written_edges_before),
                "runtime_sec": time.time() - frame_started,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    aggregate_auc_rows = _aggregate_auc_rows(auc_rows, str(args.run_id))
    auc_rows.extend(aggregate_auc_rows)
    summary = _build_summary(
        run_id=str(args.run_id),
        scenes=sorted(scenes),
        out=out,
        source_rows_path=source_rows_path,
        checkpoint=checkpoint,
        availability=availability,
        region_rows=region_rows,
        edge_rows=edge_rows,
        sep_rows=sep_rows,
        auc_rows=auc_rows,
        failure_rows=failure_rows,
        frame_rows=frame_rows,
        started=started,
        mode="extract",
        device=str(args.device),
    )
    _write_outputs(out, summary, region_rows, edge_rows, sep_rows, auc_rows, failure_rows, frame_rows)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def _load_rows_from_roots(roots: list[Path], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        rows.extend(_read_csv(root / filename))
    return rows


def _concat_csv_from_roots(roots: list[Path], filename: str, output_path: Path, fields: list[str]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fields)
        writer.writeheader()
        for root in roots:
            path = root / filename
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as in_handle:
                for row in csv.DictReader(in_handle):
                    writer.writerow({key: row.get(key, "") for key in fields})
                    row_count += 1
    return row_count


def _run_merge(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve_workspace_path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    roots = [_resolve_workspace_path(part) for part in str(args.merge_roots).split(",") if part.strip()]
    region_row_count = _concat_csv_from_roots(roots, "region_node_rows.csv", out / "region_node_rows.csv", REGION_NODE_FIELDS)
    edge_row_count = _concat_csv_from_roots(roots, "region_edge_rows.csv", out / "region_edge_rows.csv", REGION_EDGE_FIELDS)
    sep_rows = _load_rows_from_roots(roots, "semantic_separability_rows.csv")
    auc_source_rows = [row for row in _load_rows_from_roots(roots, "semantic_diagnostic_auc_rows.csv") if str(row.get("scope")) == "source_mask"]
    auc_rows = auc_source_rows + _aggregate_auc_rows(auc_source_rows, str(args.run_id))
    failure_rows = _load_rows_from_roots(roots, "semantic_failure_rows.csv")
    frame_rows = _load_rows_from_roots(roots, "frame_feature_rows.csv")
    child_summaries = []
    for root in roots:
        path = root / "summary.json"
        if path.exists():
            child_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    availability = {"radio_available": all(bool(s.get("radio_available", False)) for s in child_summaries)}
    checkpoint = next((str(s.get("radio_checkpoint", "")) for s in child_summaries if s.get("radio_checkpoint")), "")
    scenes = sorted({scene for summary in child_summaries for scene in summary.get("scenes", [])})
    summary = _build_summary(
        run_id=str(args.run_id),
        scenes=scenes,
        out=out,
        source_rows_path=_resolve_workspace_path(args.source_container_rows),
        checkpoint=checkpoint,
        availability=availability,
        region_rows=[],
        edge_rows=[],
        sep_rows=sep_rows,
        auc_rows=auc_rows,
        failure_rows=failure_rows,
        frame_rows=frame_rows,
        started=started,
        mode="merge",
        device="merged",
        region_row_count=region_row_count,
        edge_row_count=edge_row_count,
    )
    summary["merge_roots"] = [_rel(root) for root in roots]
    _write_csv(out / "semantic_separability_rows.csv", sep_rows, SEPARABILITY_FIELDS)
    _write_csv(out / "semantic_diagnostic_auc_rows.csv", auc_rows, AUC_FIELDS)
    _write_csv(out / "semantic_failure_rows.csv", failure_rows, FAILURE_FIELDS)
    _write_csv(out / "frame_feature_rows.csv", frame_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "region_node_rows.csv",
        out / "region_edge_rows.csv",
        out / "semantic_separability_rows.csv",
        out / "semantic_diagnostic_auc_rows.csv",
        out / "semantic_failure_rows.csv",
        out / "frame_feature_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def _build_summary(
    *,
    run_id: str,
    scenes: list[str],
    out: Path,
    source_rows_path: Path,
    checkpoint: str,
    availability: dict[str, Any],
    region_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    sep_rows: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    started: float,
    mode: str,
    device: str,
    region_row_count: int | None = None,
    edge_row_count: int | None = None,
) -> dict[str, Any]:
    same_auc_values = [
        _num(row.get("source_internal_same_gt_different_gt_AUC"), float("nan"))
        for row in sep_rows
        if str(row.get("source_internal_same_gt_different_gt_AUC", "")) != ""
    ]
    fg_auc_values = [
        _num(row.get("foreground_background_region_AUC"), float("nan"))
        for row in sep_rows
        if str(row.get("foreground_background_region_AUC", "")) != ""
    ]
    edge_contrast_values = [_num(row.get("adjacent_region_feature_contrast_mean"), 0.0) for row in sep_rows]
    variance_values = [_num(row.get("region_feature_variance_within_source"), 0.0) for row in sep_rows]
    source_internal_auc_ge_0p60_count = int(sum(1 for value in same_auc_values if float(value) >= 0.60))
    summary = {
        "phase_id": "v92_phase4_semantic_region_affinity",
        "schema": "stream4d_v92_phase4_semantic_region_affinity_summary_v1",
        "run_id": run_id,
        "mode": mode,
        "scenes": scenes,
        "source_container_rows": _rel(source_rows_path),
        "source_container_rows_sha256": _sha256(source_rows_path) if source_rows_path.exists() else "",
        "semantic_backend": "radio_radseg",
        "feature_layer": "radio_radseg_spatial_features",
        "radio_checkpoint": checkpoint,
        "radio_available": bool(availability.get("radio_available", True)),
        "device": device,
        "frame_rows": len(frame_rows),
        "region_node_rows": len(region_rows) if region_row_count is None else int(region_row_count),
        "region_edge_rows": len(edge_rows) if edge_row_count is None else int(edge_row_count),
        "semantic_separability_rows": len(sep_rows),
        "semantic_diagnostic_auc_rows": len(auc_rows),
        "semantic_failure_rows": len(failure_rows),
        "region_feature_variance_within_source_mean": _mean(variance_values),
        "adjacent_region_feature_contrast_mean": _mean(edge_contrast_values),
        "source_internal_same_gt_different_gt_AUC_mean": _median(same_auc_values),
        "foreground_background_region_AUC_mean": _median(fg_auc_values),
        "source_internal_auc_ge_0p60_count": source_internal_auc_ge_0p60_count,
        "source_internal_auc_available_count": len(same_auc_values),
        "foreground_background_auc_available_count": len(fg_auc_values),
        "diagnostic_only_uses_gt": True,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
        "output_artifacts": {
            "region_node_rows": _rel(out / "region_node_rows.csv"),
            "region_edge_rows": _rel(out / "region_edge_rows.csv"),
            "semantic_separability_rows": _rel(out / "semantic_separability_rows.csv"),
            "semantic_diagnostic_auc_rows": _rel(out / "semantic_diagnostic_auc_rows.csv"),
            "semantic_failure_rows": _rel(out / "semantic_failure_rows.csv"),
        },
    }
    summary["routing_label"] = _decision(summary)
    summary["decision"] = summary["routing_label"]
    return summary


def _write_outputs(
    out: Path,
    summary: dict[str, Any],
    region_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    sep_rows: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> None:
    _write_csv(out / "region_node_rows.csv", region_rows, REGION_NODE_FIELDS)
    _write_csv(out / "region_edge_rows.csv", edge_rows, REGION_EDGE_FIELDS)
    _write_csv(out / "semantic_separability_rows.csv", sep_rows, SEPARABILITY_FIELDS)
    _write_csv(out / "semantic_diagnostic_auc_rows.csv", auc_rows, AUC_FIELDS)
    _write_csv(out / "semantic_failure_rows.csv", failure_rows, FAILURE_FIELDS)
    _write_csv(out / "frame_feature_rows.csv", frame_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "region_node_rows.csv",
        out / "region_edge_rows.csv",
        out / "semantic_separability_rows.csv",
        out / "semantic_diagnostic_auc_rows.csv",
        out / "semantic_failure_rows.csv",
        out / "frame_feature_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v92 source-internal RADIO region affinity diagnostics.")
    parser.add_argument("--source-container-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--radio-lang-model", default="siglip2")
    parser.add_argument("--radio-lang-align", action="store_true")
    parser.add_argument("--radio-slide-crop", type=int, default=0)
    parser.add_argument("--radio-slide-stride", type=int, default=224)
    parser.add_argument("--min-region-pixels", type=int, default=1)
    parser.add_argument("--max-pair-samples-per-source", type=int, default=2048)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--run-id", default="v92_phase4_semantic_region_affinity")
    parser.add_argument("--merge-roots", default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if str(cli_args.merge_roots).strip():
        _run_merge(cli_args)
    else:
        _run_extract(cli_args)
