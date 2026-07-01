from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            out: dict[str, Any] = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    out[key] = json.dumps(json_safe(value), sort_keys=True)
                elif value is None:
                    out[key] = ""
                else:
                    out[key] = json_safe(value)
            writer.writerow(out)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def load_label(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int32, copy=False)


def load_mask_label(scene: str, frame_id: int) -> np.ndarray | None:
    return load_label(ROOT / "data/scannet/processed" / scene / "output_Cropformer/mask" / f"{int(frame_id)}.png")


def project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return ROOT.parent / path_obj
    return ROOT / path_obj


def resolve_mask_dir(mask_root: str | Path | None, scene: str) -> Path:
    if mask_root is None or str(mask_root).strip() == "":
        return ROOT / "data/scannet/processed" / scene / "output_Cropformer/mask"
    base = project_path(mask_root)
    candidates = [
        base / scene / "output_Cropformer" / "mask",
        base / scene / "mask",
        base / "output_Cropformer" / "mask",
        base / "mask",
        base,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_mask_label_from_root(scene: str, frame_id: int, mask_root: str | Path | None = None) -> np.ndarray | None:
    return load_label(resolve_mask_dir(mask_root, scene) / f"{int(frame_id)}.png")


def load_gt_label(scene: str, frame_id: int) -> np.ndarray | None:
    return load_label(ROOT / "data/scannet/processed" / scene / "instance/instance" / f"{int(frame_id)}.png")


def load_rgb(scene: str, frame_id: int) -> np.ndarray | None:
    path = ROOT / "data/scannet/processed" / scene / "color" / f"{int(frame_id)}.jpg"
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def dominant_gt(mask: np.ndarray, gt: np.ndarray | None) -> tuple[int | None, float | None]:
    if gt is None or not np.any(mask):
        return None, None
    if gt.shape != mask.shape:
        gt = cv2.resize(gt.astype(np.int32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    values, counts = np.unique(gt[np.asarray(mask, dtype=bool)], return_counts=True)
    pairs = [(int(value), int(count)) for value, count in zip(values, counts) if int(value) > 0]
    if not pairs:
        return None, None
    label, count = max(pairs, key=lambda item: item[1])
    return int(label), float(count / max(int(mask.sum()), 1))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou(a: Iterable[Any], b: Iterable[Any]) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0 + 1.0), max(0.0, iy1 - iy0 + 1.0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0 + 1.0) * max(0.0, ay1 - ay0 + 1.0)
    area_b = max(0.0, bx1 - bx0 + 1.0) * max(0.0, by1 - by0 + 1.0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


def color_feature(scene: str, frame_id: int, mask: np.ndarray) -> tuple[list[float], bool]:
    rgb = load_rgb(scene, frame_id)
    if rgb is None or not np.any(mask):
        return [], False
    if rgb.shape[:2] != mask.shape:
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    pixels = rgb[np.asarray(mask, dtype=bool)].astype(np.float32) / 255.0
    if pixels.size == 0:
        return [], False
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    hist_parts: list[np.ndarray] = []
    for channel in range(3):
        hist, _bins = np.histogram(pixels[:, channel], bins=4, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        denom = float(hist.sum())
        hist_parts.append(hist / denom if denom > 0.0 else hist)
    feature = np.concatenate([mean, std, *hist_parts]).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm > 0.0:
        feature = feature / norm
    return [float(x) for x in feature.tolist()], True


def cosine(left: list[float] | str | None, right: list[float] | str | None) -> float:
    if isinstance(left, str):
        left = json.loads(left) if left.strip() else []
    if isinstance(right, str):
        right = json.loads(right) if right.strip() else []
    if not left or not right:
        return 0.0
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 0.0 else float(np.dot(a, b) / denom)


def safe_mean(values: Iterable[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None and str(value) != "" and math.isfinite(float(value))]
    return float(np.mean(nums)) if nums else None


def safe_quantile(values: Iterable[Any], q: float) -> float | None:
    nums = [float(value) for value in values if value is not None and str(value) != "" and math.isfinite(float(value))]
    return float(np.quantile(nums, float(q))) if nums else None


def rank_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    pos_count = sum(1 for label, _score in pairs if label)
    neg_count = len(pairs) - pos_count
    if pos_count == 0 or neg_count == 0:
        return None
    pairs.sort(key=lambda item: item[1])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum_pos += avg_rank * sum(1 for label, _score in pairs[idx:end] if label)
        idx = end
    return float((rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))


def adjusted_rand_score(true_labels: list[str], pred_labels: list[str]) -> float:
    n = len(true_labels)
    if n <= 1:
        return 1.0
    table: Counter[tuple[str, str]] = Counter(zip(true_labels, pred_labels))
    true_counts: Counter[str] = Counter(true_labels)
    pred_counts: Counter[str] = Counter(pred_labels)
    comb = lambda count: float(count * (count - 1) / 2)
    sum_comb = sum(comb(value) for value in table.values())
    sum_true = sum(comb(value) for value in true_counts.values())
    sum_pred = sum(comb(value) for value in pred_counts.values())
    total = comb(n)
    if total <= 0.0:
        return 1.0
    expected = sum_true * sum_pred / total
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if abs(denom) <= 1e-12:
        return 0.0
    return float((sum_comb - expected) / denom)


def cluster_purity(true_labels: list[str], pred_labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    clusters: dict[str, Counter[str]] = defaultdict(Counter)
    for true, pred in zip(true_labels, pred_labels):
        clusters[pred][true] += 1
    return float(sum(max(counts.values()) for counts in clusters.values()) / len(true_labels))


def cluster_completeness(true_labels: list[str], pred_labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    gt_groups: dict[str, Counter[str]] = defaultdict(Counter)
    for true, pred in zip(true_labels, pred_labels):
        gt_groups[true][pred] += 1
    return float(sum(max(counts.values()) for counts in gt_groups.values()) / len(true_labels))


class UnionFind:
    def __init__(self, node_ids: Iterable[int]) -> None:
        self.parent = {int(node_id): int(node_id) for node_id in node_ids}
        self.members = {int(node_id): {int(node_id)} for node_id in node_ids}

    def find(self, node_id: int) -> int:
        node_id = int(node_id)
        parent = self.parent[node_id]
        if parent != node_id:
            self.parent[node_id] = self.find(parent)
        return self.parent[node_id]

    def union(self, left: int, right: int) -> bool:
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l == root_r:
            return False
        if len(self.members[root_l]) < len(self.members[root_r]):
            root_l, root_r = root_r, root_l
        for node_id in self.members[root_r]:
            self.parent[node_id] = root_l
        self.members[root_l].update(self.members[root_r])
        del self.members[root_r]
        return True
