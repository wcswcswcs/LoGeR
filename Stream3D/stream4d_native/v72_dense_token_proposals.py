from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_dinov2_checkpoint  # noqa: E402
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _score_free  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


@dataclass
class Proposal:
    proposal_id: str
    variant: str
    scene_id: str
    chunk_id: str
    frame_id: int
    source_mask_id: int
    source_mask_observation_id: str
    source_type: str
    source_broad_large_risk: bool
    source_underseg_proxy: bool
    source_semantic_entropy: float
    token_mask: np.ndarray
    eval_mask: np.ndarray
    proposal_area_ratio: float
    proposal_bbox: dict[str, int]
    semantic_entropy: float
    semantic_intra_variance: float
    semantic_boundary_divergence: float
    semantic_prototype_margin: float
    proposal_compactness_score: float
    proposal_background_proxy_score: float
    debug: dict[str, Any]
    majority_gt: int = 0
    majority_iou: float = 0.0
    source_majority_iou: float = 0.0


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(valid)) if valid else None


def _resize_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _resize_label(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != shape_hw:
        image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image, dtype=np.int64)


def _resize_binary(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    out = _resize_nearest(np.asarray(mask, dtype=bool), shape_hw)
    if not np.any(out) and np.any(mask):
        ys, xs = np.nonzero(mask)
        h, w = shape_hw
        cy = int(np.clip(round(float(ys.mean()) * float(h) / max(mask.shape[0], 1)), 0, int(h) - 1))
        cx = int(np.clip(round(float(xs.mean()) * float(w) / max(mask.shape[1], 1)), 0, int(w) - 1))
        out[cy, cx] = True
    return out


def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, eps)


def _entropy_from_intra(intra: float, scale: float) -> float:
    return float(1.0 - math.exp(-max(0.0, float(intra)) / max(1e-12, float(scale))))


def _token_coords(token_mask: np.ndarray) -> str:
    coords = np.argwhere(np.asarray(token_mask, dtype=bool))
    return ";".join(f"{int(y)}:{int(x)}" for y, x in coords)


def _bbox(mask: np.ndarray) -> dict[str, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    return {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1}


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[1:, :] != mask[:-1, :]
    out[:-1, :] |= mask[:-1, :] != mask[1:, :]
    out[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    out[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return out & mask


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pipeline_roots(path: Path, scenes: list[str]) -> dict[str, Path]:
    summary = _load_json(path)
    raw = summary.get("pipeline_roots") or {}
    return {scene: _rooted(raw[scene]) for scene in scenes if raw.get(scene)}


def _load_candidate_rows(path: Path, scenes: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            if scenes and scene not in scenes:
                continue
            row["frame_id"] = _int(row.get("frame_id"), -1)
            row["mask_id"] = _int(row.get("mask_id"), -1)
            row["area_ratio"] = _float(row.get("area_ratio"), 0.0)
            row["semantic_entropy"] = _float(row.get("semantic_entropy"), 1.0)
            row["semantic_intra_variance"] = _float(row.get("semantic_intra_mask_variance"), 0.0)
            row["semantic_prototype_margin"] = _float(row.get("semantic_prototype_margin"), 0.0)
            row["broad_large_risk"] = _bool(row.get("broad_background_risk")) or _bool(row.get("large_mask_risk")) or row["area_ratio"] >= 0.30
            row["underseg_proxy"] = _float(row.get("underseg_proxy_score"), 0.0) >= 0.75
            row["small_mask_risk"] = _bool(row.get("small_mask_risk"))
            out[str(row.get("chunk_id") or "")].append(row)
    return out


def _select_broad_sources(rows: list[dict[str, Any]], max_per_frame: int) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not (_bool(row.get("broad_large_risk")) or _bool(row.get("underseg_proxy"))):
            continue
        area = _float(row.get("area_ratio"), 0.0)
        if area < 0.01 or area > 0.85:
            continue
        by_frame[int(row.get("frame_id") or -1)].append(row)
    out: dict[int, list[dict[str, Any]]] = {}
    for frame_id, subset in by_frame.items():
        out[frame_id] = sorted(
            subset,
            key=lambda row: (
                _float(row.get("semantic_entropy"), 0.0),
                _float(row.get("area_ratio"), 0.0),
                _float(row.get("same_frame_overlap_count"), 0.0),
            ),
            reverse=True,
        )[: int(max_per_frame)]
    return out


def _clean_prototypes(rows: list[dict[str, Any]], max_per_frame: int) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        area = _float(row.get("area_ratio"), 0.0)
        if not (0.004 <= area <= 0.18):
            continue
        if _bool(row.get("broad_large_risk")) or _bool(row.get("underseg_proxy")) or _bool(row.get("small_mask_risk")):
            continue
        by_frame[int(row.get("frame_id") or -1)].append(row)
    out: dict[int, list[dict[str, Any]]] = {}
    for frame_id, subset in by_frame.items():
        out[frame_id] = sorted(
            subset,
            key=lambda row: (
                -_float(row.get("semantic_entropy"), 1.0),
                _float(row.get("semantic_prototype_margin"), 0.0),
                _float(row.get("area_ratio"), 0.0),
            ),
        )[: int(max_per_frame)]
    return out


def _deterministic_kmeans(features: np.ndarray, k: int, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    values = _l2_normalize(np.asarray(features, dtype=np.float32))
    n = int(values.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, values.shape[-1]), dtype=np.float32)
    k = max(1, min(int(k), n))
    mean = _l2_normalize(values.mean(axis=0, keepdims=True))[0]
    centers = [values[int(np.argmin(values @ mean))]]
    while len(centers) < k:
        sims = values @ np.stack(centers, axis=0).T
        distance = 1.0 - np.max(sims, axis=1)
        centers.append(values[int(np.argmax(distance))])
    center_arr = _l2_normalize(np.stack(centers, axis=0).astype(np.float32))
    labels = np.zeros((n,), dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        labels = np.argmax(values @ center_arr.T, axis=1).astype(np.int64)
        next_centers = center_arr.copy()
        for idx in range(k):
            member = values[labels == idx]
            if member.size:
                next_centers[idx] = member.mean(axis=0)
        center_arr = _l2_normalize(next_centers.astype(np.float32))
    return labels, center_arr


def _connected_subcomponents(mask: np.ndarray, min_tokens: int) -> list[np.ndarray]:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    count, labels = cv2.connectedComponents(mask_u8, connectivity=4)
    out: list[np.ndarray] = []
    for label in range(1, int(count)):
        comp = labels == label
        if int(comp.sum()) >= int(min_tokens):
            out.append(comp)
    return out


def _affinity_components(features: np.ndarray, source_mask: np.ndarray, threshold: float, min_tokens: int) -> list[np.ndarray]:
    h, w = source_mask.shape
    labels = np.full((h, w), -1, dtype=np.int64)
    current = 0
    vectors = _l2_normalize(np.asarray(features, dtype=np.float32))
    for y in range(h):
        for x in range(w):
            if not source_mask[y, x] or labels[y, x] >= 0:
                continue
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                base = vectors[cy, cx]
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not source_mask[ny, nx] or labels[ny, nx] >= 0:
                        continue
                    if float(np.dot(base, vectors[ny, nx])) < float(threshold):
                        continue
                    labels[ny, nx] = current
                    stack.append((ny, nx))
            current += 1
    out: list[np.ndarray] = []
    for label in range(current):
        comp = labels == label
        if int(comp.sum()) >= int(min_tokens):
            out.append(comp)
    return out


def _merge_adjacent_components(features: np.ndarray, components: list[np.ndarray], threshold: float, min_tokens: int) -> list[np.ndarray]:
    if len(components) < 2:
        return []
    shape = components[0].shape
    labels = np.full(shape, -1, dtype=np.int64)
    for idx, comp in enumerate(components):
        labels[np.asarray(comp, dtype=bool)] = int(idx)
    pairs: set[tuple[int, int]] = set()
    for a, b in ((labels[:-1, :], labels[1:, :]), (labels[:, :-1], labels[:, 1:])):
        valid = (a >= 0) & (b >= 0) & (a != b)
        if not np.any(valid):
            continue
        for left, right in zip(a[valid].tolist(), b[valid].tolist()):
            i, j = sorted((int(left), int(right)))
            pairs.add((i, j))
    if not pairs:
        return []
    vectors = _l2_normalize(np.asarray(features, dtype=np.float32))
    means: list[np.ndarray] = []
    for comp in components:
        vals = vectors[np.asarray(comp, dtype=bool)]
        if vals.size:
            means.append(_l2_normalize(vals.mean(axis=0, keepdims=True))[0])
        else:
            means.append(np.zeros((vectors.shape[-1],), dtype=np.float32))
    parent = list(range(len(components)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        if float(np.dot(means[i], means[j])) >= float(threshold):
            union(i, j)
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(components)):
        grouped[find(idx)].append(idx)
    merged: list[np.ndarray] = []
    seen: set[bytes] = set()
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        mask = np.zeros(shape, dtype=bool)
        for idx in indexes:
            mask |= np.asarray(components[idx], dtype=bool)
        if int(mask.sum()) < int(min_tokens):
            continue
        key = np.packbits(mask.reshape(-1).astype(np.uint8)).tobytes()
        if key in seen:
            continue
        seen.add(key)
        merged.append(mask)
    return merged


def _token_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)))
    union = int(np.count_nonzero(np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool)))
    return float(inter / union) if union else 0.0


def _proposal_from_token_mask(
    *,
    variant: str,
    source_type: str,
    scene_id: str,
    chunk_id: str,
    frame_id: int,
    source_row: dict[str, Any],
    token_mask: np.ndarray,
    features: np.ndarray,
    eval_shape_hw: tuple[int, int],
    proposal_index: int,
    entropy_scale: float,
    debug: dict[str, Any],
) -> Proposal | None:
    if not np.any(token_mask):
        return None
    eval_mask = _resize_binary(token_mask, eval_shape_hw)
    area = int(np.count_nonzero(eval_mask))
    if area <= 0:
        return None
    values = np.asarray(features, dtype=np.float32)[np.asarray(token_mask, dtype=bool)]
    if values.ndim != 2 or values.shape[0] == 0:
        return None
    pooled = _l2_normalize(values.mean(axis=0, keepdims=True))[0]
    sims = values @ pooled
    intra = float(np.mean(np.var(values, axis=0))) if values.shape[0] > 1 else 0.0
    entropy = _entropy_from_intra(intra, entropy_scale)
    boundary = _boundary(token_mask)
    if np.any(boundary) and np.any(~token_mask):
        inner = np.asarray(features, dtype=np.float32)[boundary].mean(axis=0)
        outer = np.asarray(features, dtype=np.float32)[~token_mask].mean(axis=0)
        denom = float(np.linalg.norm(inner) * np.linalg.norm(outer))
        boundary_div = float(1.0 - float(np.dot(inner, outer)) / max(denom, 1e-12))
    else:
        boundary_div = 0.0
    area_ratio = float(area / max(1, int(eval_shape_hw[0]) * int(eval_shape_hw[1])))
    source_entropy = _float(source_row.get("semantic_entropy"), 1.0)
    margin = float(np.quantile(sims, 0.75) - np.quantile(sims, 0.25)) if sims.size else 0.0
    compactness = max(0.0, 1.0 - entropy) + max(0.0, source_entropy - entropy) + 0.5 * boundary_div + 0.25 * margin
    background = 0.5 * entropy + (1.0 if area_ratio >= 0.30 else 0.0) + 0.25 * max(0.0, area_ratio / max(_float(source_row.get("area_ratio"), 1e-6), 1e-6))
    proposal_id = f"{variant}:{scene_id}:{frame_id}:{int(source_row.get('mask_id') or -1)}:{proposal_index:04d}"
    return Proposal(
        proposal_id=proposal_id,
        variant=variant,
        scene_id=scene_id,
        chunk_id=chunk_id,
        frame_id=int(frame_id),
        source_mask_id=int(source_row.get("mask_id") or -1),
        source_mask_observation_id=str(source_row.get("mask_observation_id") or ""),
        source_type=source_type,
        source_broad_large_risk=_bool(source_row.get("broad_large_risk")),
        source_underseg_proxy=_bool(source_row.get("underseg_proxy")),
        source_semantic_entropy=source_entropy,
        token_mask=np.asarray(token_mask, dtype=bool),
        eval_mask=np.asarray(eval_mask, dtype=bool),
        proposal_area_ratio=area_ratio,
        proposal_bbox=_bbox(eval_mask),
        semantic_entropy=entropy,
        semantic_intra_variance=intra,
        semantic_boundary_divergence=boundary_div,
        semantic_prototype_margin=margin,
        proposal_compactness_score=compactness,
        proposal_background_proxy_score=background,
        debug=debug,
    )


def _cap_and_nms(
    proposals: list[Proposal],
    max_count: int,
    nms_iou: float,
    *,
    small_threshold: float = 0.0,
    mid_threshold: float = 0.0,
    max_small: int | None = None,
    max_mid: int | None = None,
    max_large: int | None = None,
) -> list[Proposal]:
    kept: list[Proposal] = []
    bin_counts = {"small": 0, "mid": 0, "large": 0}
    for prop in sorted(proposals, key=lambda item: item.proposal_compactness_score, reverse=True):
        if len(kept) >= int(max_count):
            break
        area = float(prop.proposal_area_ratio)
        if small_threshold > 0.0 and area < small_threshold:
            bin_name = "small"
            bin_cap = max_small
        elif mid_threshold > 0.0 and area < mid_threshold:
            bin_name = "mid"
            bin_cap = max_mid
        else:
            bin_name = "large"
            bin_cap = max_large
        if bin_cap is not None and bin_counts[bin_name] >= int(bin_cap):
            continue
        if any(_token_iou(prop.token_mask, prev.token_mask) >= float(nms_iou) for prev in kept):
            continue
        kept.append(prop)
        bin_counts[bin_name] += 1
    return kept


def _accept_dense_proposal(proposal: Proposal | None, args: argparse.Namespace) -> bool:
    if proposal is None:
        return False
    area = float(proposal.proposal_area_ratio)
    if area < float(args.min_component_area_ratio):
        return False
    if area > float(args.max_dense_component_area_ratio):
        return False
    return True


def _mask_diagnostic(binary: np.ndarray, gt: np.ndarray, gt_area: dict[int, int]) -> tuple[int, float]:
    if not np.any(binary):
        return 0, 0.0
    labels, counts = np.unique(gt[np.asarray(binary, dtype=bool)], return_counts=True)
    best_gid = 0
    best_inter = 0
    for gid, inter in zip(labels, counts):
        gid_i = int(gid)
        if gid_i <= 0:
            continue
        if int(inter) > best_inter:
            best_gid = gid_i
            best_inter = int(inter)
    if best_gid <= 0 or best_inter <= 0:
        return 0, 0.0
    union = int(np.count_nonzero(binary)) + int(gt_area.get(best_gid, 0)) - best_inter
    return best_gid, float(best_inter / max(1, union))


def _evaluate_proposals(
    *,
    proposals: list[Proposal],
    frame_gt: dict[int, np.ndarray],
    variant: str,
) -> dict[str, Any]:
    by_frame: dict[int, list[Proposal]] = defaultdict(list)
    for prop in proposals:
        by_frame[int(prop.frame_id)].append(prop)
    acc = SparseSceneIoU()
    for frame_id, gt in frame_gt.items():
        pred = np.zeros(gt.shape, dtype=np.int64)
        for prop in sorted(by_frame.get(frame_id, []), key=lambda item: item.proposal_compactness_score, reverse=True):
            if prop.majority_gt <= 0:
                continue
            pred[(prop.eval_mask > 0) & (pred == 0)] = int(prop.majority_gt)
        acc.add(pred, gt)
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    return {
        "variant": variant,
        "proposal_count": len(proposals),
        "proposal_count_per_frame": len(proposals) / max(1, len(by_frame)),
        "proposal_area_ratio_mean": _mean([prop.proposal_area_ratio for prop in proposals]),
        "proposal_semantic_entropy_mean": _mean([prop.semantic_entropy for prop in proposals]),
        "proposal_source_entropy_mean": _mean([prop.source_semantic_entropy for prop in proposals]),
        "proposal_source_broad_rate": _mean([1.0 if prop.source_broad_large_risk else 0.0 for prop in proposals]),
        "proposal_from_broad_count": sum(1 for prop in proposals if prop.source_broad_large_risk),
        "proposal_majority_IoU_mean_diagnostic": _mean([prop.majority_iou for prop in proposals]),
        "proposal_IoU50_rate_diagnostic": _mean([1.0 if prop.majority_iou >= 0.50 else 0.0 for prop in proposals]),
        "proposal_background_proxy_rate": _mean([1.0 if prop.proposal_background_proxy_score >= 0.75 else 0.0 for prop in proposals]),
        "proposal_oracle_SF50_by_majority_GT_diagnostic": _score_free(summary),
        "proposal_oracle_AP50_by_majority_GT_diagnostic": summary.get("ap50"),
        "proposal_oracle_GT_best_IoU_mean_diagnostic": summary.get("gt_best_iou_mean"),
        "proposal_oracle_pred_best_IoU_median_diagnostic": summary.get("pred_best_iou_median"),
        "evaluated_gt_count": summary.get("evaluated_gt_count"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": True,
    }


def _baseline_existing_proposals(
    *,
    rows: list[dict[str, Any]],
    scene_id: str,
    chunk_id: str,
    frame_mask_eval: dict[int, np.ndarray],
    frame_gt: dict[int, np.ndarray],
    entropy_scale: float,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    for idx, row in enumerate(rows):
        frame_id = int(row.get("frame_id") or -1)
        label = frame_mask_eval.get(frame_id)
        gt = frame_gt.get(frame_id)
        if label is None or gt is None:
            continue
        eval_mask = label == int(row.get("mask_id") or -1)
        if not np.any(eval_mask):
            continue
        entropy = _float(row.get("semantic_entropy"), 1.0)
        proposal = Proposal(
            proposal_id=f"SP0_existing_masks_baseline:{str(row.get('mask_observation_id') or '')}",
            variant="SP0_existing_masks_baseline",
            scene_id=scene_id,
            chunk_id=chunk_id,
            frame_id=frame_id,
            source_mask_id=int(row.get("mask_id") or -1),
            source_mask_observation_id=str(row.get("mask_observation_id") or ""),
            source_type="existing_mask",
            source_broad_large_risk=_bool(row.get("broad_large_risk")),
            source_underseg_proxy=_bool(row.get("underseg_proxy")),
            source_semantic_entropy=entropy,
            token_mask=np.zeros((1, 1), dtype=bool),
            eval_mask=eval_mask,
            proposal_area_ratio=float(np.count_nonzero(eval_mask) / max(1, eval_mask.size)),
            proposal_bbox=_bbox(eval_mask),
            semantic_entropy=entropy,
            semantic_intra_variance=_float(row.get("semantic_intra_variance"), 0.0),
            semantic_boundary_divergence=0.0,
            semantic_prototype_margin=_float(row.get("semantic_prototype_margin"), 0.0),
            proposal_compactness_score=max(0.0, 1.0 - entropy) + _float(row.get("semantic_prototype_margin"), 0.0),
            proposal_background_proxy_score=0.5 * entropy + (1.0 if _bool(row.get("broad_large_risk")) else 0.0),
            debug={"baseline_index": idx, "entropy_scale": entropy_scale},
        )
        proposals.append(proposal)
    return proposals


def _generate_dense_for_frame(
    *,
    scene_id: str,
    chunk_id: str,
    frame_id: int,
    source_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    rgb: np.ndarray,
    mask_rgb: np.ndarray,
    eval_shape_hw: tuple[int, int],
    adapter: FrozenFeatureAdapter,
    args: argparse.Namespace,
) -> dict[str, list[Proposal]]:
    fmap = adapter.extract_dense_features(rgb)
    features = _l2_normalize(np.asarray(fmap.features, dtype=np.float32))
    token_shape = (int(features.shape[0]), int(features.shape[1]))
    source_token_masks: dict[str, np.ndarray] = {}
    proposals_by_variant: dict[str, list[Proposal]] = defaultdict(list)

    def source_mask(row: dict[str, Any]) -> np.ndarray:
        key = str(row.get("mask_observation_id") or "")
        if key not in source_token_masks:
            binary_rgb = mask_rgb == int(row.get("mask_id") or -1)
            source_token_masks[key] = _resize_binary(binary_rgb, token_shape)
        return source_token_masks[key]

    proto_vectors: list[tuple[dict[str, Any], np.ndarray]] = []
    for proto_row in clean_rows:
        token_mask = source_mask(proto_row)
        if int(token_mask.sum()) < int(args.min_tokens):
            continue
        values = features[token_mask]
        proto_vectors.append((proto_row, _l2_normalize(values.mean(axis=0, keepdims=True))[0]))

    proposal_index = 0
    for source_row in source_rows:
        source_tokens = source_mask(source_row)
        if int(source_tokens.sum()) < max(int(args.min_tokens) * 2, 3):
            continue
        coords = np.argwhere(source_tokens)
        token_values = features[source_tokens]
        per_source: dict[str, list[Proposal]] = defaultdict(list)
        for k in [int(part) for part in str(args.kmeans_k).split(",") if part.strip()]:
            labels, _centers = _deterministic_kmeans(token_values, k=k, iterations=int(args.kmeans_iterations))
            label_grid = np.full(token_shape, -1, dtype=np.int64)
            label_grid[source_tokens] = labels
            for label in sorted(set(int(v) for v in labels.tolist())):
                for comp in _connected_subcomponents(label_grid == label, int(args.min_tokens)):
                    proposal_index += 1
                    proposal = _proposal_from_token_mask(
                        variant="SP1_DINO_token_kmeans_within_broad",
                        source_type="dino_token_kmeans_component",
                        scene_id=scene_id,
                        chunk_id=chunk_id,
                        frame_id=frame_id,
                        source_row=source_row,
                        token_mask=comp,
                        features=features,
                        eval_shape_hw=eval_shape_hw,
                        proposal_index=proposal_index,
                        entropy_scale=float(args.entropy_variance_scale),
                        debug={"k": k, "token_count": int(comp.sum())},
                    )
                    if _accept_dense_proposal(proposal, args):
                        per_source[proposal.variant].append(proposal)
                        if (
                            proposal.semantic_entropy <= _float(source_row.get("semantic_entropy"), 1.0) - float(args.sp6_entropy_drop)
                            and proposal.proposal_area_ratio <= max(float(args.max_component_area_ratio), 0.65 * _float(source_row.get("area_ratio"), 0.0))
                        ):
                            proposal_index += 1
                            sp6 = _proposal_from_token_mask(
                                variant="SP6_background_suppressed_semantic_proposals",
                                source_type="dino_token_kmeans_background_suppressed",
                                scene_id=scene_id,
                                chunk_id=chunk_id,
                                frame_id=frame_id,
                                source_row=source_row,
                                token_mask=comp,
                                features=features,
                                eval_shape_hw=eval_shape_hw,
                                proposal_index=proposal_index,
                                entropy_scale=float(args.entropy_variance_scale),
                                debug={"k": k, "token_count": int(comp.sum()), "source_variant": "SP1"},
                            )
                            if _accept_dense_proposal(sp6, args):
                                per_source[sp6.variant].append(sp6)
        for threshold in [float(part) for part in str(args.affinity_thresholds).split(",") if part.strip()]:
            affinity_components = _affinity_components(features, source_tokens, threshold, int(args.min_tokens))
            for comp in affinity_components:
                proposal_index += 1
                proposal = _proposal_from_token_mask(
                    variant="SP2_DINO_affinity_connected_components",
                    source_type="dino_affinity_connected_component",
                    scene_id=scene_id,
                    chunk_id=chunk_id,
                    frame_id=frame_id,
                    source_row=source_row,
                    token_mask=comp,
                    features=features,
                    eval_shape_hw=eval_shape_hw,
                    proposal_index=proposal_index,
                    entropy_scale=float(args.entropy_variance_scale),
                    debug={"affinity_threshold": threshold, "token_count": int(comp.sum())},
                )
                if _accept_dense_proposal(proposal, args):
                    per_source[proposal.variant].append(proposal)
            for merge_threshold in [float(part) for part in str(args.component_merge_similarity_thresholds).split(",") if part.strip()]:
                merged_components = _merge_adjacent_components(
                    features,
                    affinity_components,
                    threshold=merge_threshold,
                    min_tokens=int(args.min_tokens),
                )
                for comp in merged_components:
                    proposal_index += 1
                    proposal = _proposal_from_token_mask(
                        variant="SP8_DINO_affinity_neighbor_merge_repair",
                        source_type="dino_affinity_neighbor_merged_component",
                        scene_id=scene_id,
                        chunk_id=chunk_id,
                        frame_id=frame_id,
                        source_row=source_row,
                        token_mask=comp,
                        features=features,
                        eval_shape_hw=eval_shape_hw,
                        proposal_index=proposal_index,
                        entropy_scale=float(args.entropy_variance_scale),
                        debug={
                            "affinity_threshold": threshold,
                            "component_merge_similarity_threshold": merge_threshold,
                            "token_count": int(comp.sum()),
                            "merged_from_component_count": int(len(affinity_components)),
                        },
                    )
                    if _accept_dense_proposal(proposal, args):
                        per_source[proposal.variant].append(proposal)
        for proto_row, proto_vec in proto_vectors[: int(args.max_prototypes_per_frame)]:
            sims = features @ proto_vec
            grow = source_tokens & (sims >= float(args.prototype_similarity_threshold))
            for comp in _connected_subcomponents(grow, int(args.min_tokens)):
                proposal_index += 1
                proposal = _proposal_from_token_mask(
                    variant="SP3_DINO_prototype_region_grow",
                    source_type="dino_prototype_region_grow",
                    scene_id=scene_id,
                    chunk_id=chunk_id,
                    frame_id=frame_id,
                    source_row=source_row,
                    token_mask=comp,
                    features=features,
                    eval_shape_hw=eval_shape_hw,
                    proposal_index=proposal_index,
                    entropy_scale=float(args.entropy_variance_scale),
                    debug={
                        "prototype_mask_observation_id": str(proto_row.get("mask_observation_id") or ""),
                        "prototype_similarity_threshold": float(args.prototype_similarity_threshold),
                        "token_count": int(comp.sum()),
                    },
                )
                if _accept_dense_proposal(proposal, args):
                    per_source[proposal.variant].append(proposal)
            hint_tokens = source_mask(proto_row)
            seed = source_tokens & hint_tokens
            seed_mode = "overlap"
            if int(seed.sum()) < int(args.min_tokens):
                kernel = np.ones((3, 3), dtype=np.uint8)
                adjacent = cv2.dilate(hint_tokens.astype(np.uint8), kernel, iterations=1).astype(bool)
                seed = source_tokens & adjacent
                seed_mode = "adjacent_band"
            if int(seed.sum()) >= int(args.min_tokens):
                seed_vec = proto_vec
                for hint_threshold in [float(part) for part in str(args.clean_hint_similarity_thresholds).split(",") if part.strip()]:
                    sims = features @ seed_vec
                    grow = source_tokens & (sims >= float(hint_threshold))
                    grow |= seed
                    for comp in _connected_subcomponents(grow, int(args.min_tokens)):
                        if int(np.count_nonzero(comp & seed)) < int(args.min_tokens):
                            continue
                        proposal_index += 1
                        proposal = _proposal_from_token_mask(
                            variant="SP4_clean_hint_constrained_region_grow",
                            source_type="clean_hint_constrained_region_grow",
                            scene_id=scene_id,
                            chunk_id=chunk_id,
                            frame_id=frame_id,
                            source_row=source_row,
                            token_mask=comp,
                            features=features,
                            eval_shape_hw=eval_shape_hw,
                            proposal_index=proposal_index,
                            entropy_scale=float(args.entropy_variance_scale),
                            debug={
                                "hint_mask_observation_id": str(proto_row.get("mask_observation_id") or ""),
                                "clean_hint_similarity_threshold": hint_threshold,
                                "seed_mode": seed_mode,
                                "seed_token_count": int(seed.sum()),
                                "token_count": int(comp.sum()),
                            },
                        )
                        if _accept_dense_proposal(proposal, args):
                            per_source[proposal.variant].append(proposal)

        for variant, subset in per_source.items():
            proposals_by_variant[variant].extend(
                _cap_and_nms(
                    subset,
                    max_count=int(args.max_proposals_per_source_variant),
                    nms_iou=float(args.token_nms_iou),
                )
            )

    capped: dict[str, list[Proposal]] = {}
    for variant, subset in proposals_by_variant.items():
        capped[variant] = _cap_and_nms(
            subset,
            max_count=int(args.max_proposals_per_frame_per_variant),
            nms_iou=float(args.token_nms_iou),
            small_threshold=float(args.area_bin_small_threshold),
            mid_threshold=float(args.area_bin_mid_threshold),
            max_small=int(args.area_bin_max_small_per_frame) if int(args.area_bin_max_small_per_frame) >= 0 else None,
            max_mid=int(args.area_bin_max_mid_per_frame) if int(args.area_bin_max_mid_per_frame) >= 0 else None,
            max_large=int(args.area_bin_max_large_per_frame) if int(args.area_bin_max_large_per_frame) >= 0 else None,
        )
    return capped


def _annotate_diagnostics(proposals: list[Proposal], frame_gt: dict[int, np.ndarray], frame_gt_area: dict[int, dict[int, int]], source_stats: dict[str, float]) -> None:
    for prop in proposals:
        gt = frame_gt.get(int(prop.frame_id))
        gt_area = frame_gt_area.get(int(prop.frame_id), {})
        if gt is None:
            continue
        prop.majority_gt, prop.majority_iou = _mask_diagnostic(prop.eval_mask, gt, gt_area)
        prop.source_majority_iou = float(source_stats.get(prop.source_mask_observation_id, 0.0))


def _proposal_rows(proposals: list[Proposal]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prop in proposals:
        rows.append(
            {
                "proposal_id": prop.proposal_id,
                "scene_id": prop.scene_id,
                "chunk_id": prop.chunk_id,
                "frame_id": prop.frame_id,
                "source_mask_ids": prop.source_mask_observation_id,
                "source_mask_id": prop.source_mask_id,
                "source_type": prop.source_type,
                "proposal_region_ref": "token_coords_37x37_in_csv" if prop.variant != "SP0_existing_masks_baseline" else "existing_mask_id",
                "proposal_token_grid_shape": "x".join(str(v) for v in prop.token_mask.shape),
                "proposal_token_coords": "" if prop.variant == "SP0_existing_masks_baseline" else _token_coords(prop.token_mask),
                "proposal_area_ratio": prop.proposal_area_ratio,
                "proposal_bbox": json.dumps(prop.proposal_bbox, sort_keys=True),
                "semantic_backend": "dinov2_timm",
                "semantic_entropy": prop.semantic_entropy,
                "semantic_intra_variance": prop.semantic_intra_variance,
                "semantic_boundary_divergence": prop.semantic_boundary_divergence,
                "semantic_prototype_id": "",
                "semantic_prototype_margin": prop.semantic_prototype_margin,
                "source_broad_large_risk": prop.source_broad_large_risk,
                "source_underseg_proxy": prop.source_underseg_proxy,
                "source_semantic_entropy": prop.source_semantic_entropy,
                "proposal_compactness_score": prop.proposal_compactness_score,
                "proposal_background_proxy_score": prop.proposal_background_proxy_score,
                "proposal_same_frame_overlap_count": "",
                "majority_gt_id_diagnostic": prop.majority_gt,
                "majority_iou_diagnostic": prop.majority_iou,
                "source_majority_iou_diagnostic": prop.source_majority_iou,
                "broad_to_subproposal_iou_gain_diagnostic": prop.majority_iou - prop.source_majority_iou,
                "broad_to_subproposal_entropy_drop": prop.source_semantic_entropy - prop.semantic_entropy,
                "uses_gt_for_prediction": False,
                "uses_gt_for_evaluation": True,
                "diagnostic_only": prop.variant != "SP0_existing_masks_baseline",
                "forbidden_for_method_table": prop.variant != "SP0_existing_masks_baseline",
                "variant": prop.variant,
                "debug_json": json.dumps(prop.debug, sort_keys=True),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    candidate_rows_path = _rooted(args.candidate_rows)
    witness_summary = _rooted(args.witness_summary)
    candidates_by_chunk = _load_candidate_rows(candidate_rows_path, set(scenes))
    pipeline_roots = _load_pipeline_roots(witness_summary, scenes)
    checkpoint = args.checkpoint or locate_default_dinov2_checkpoint()
    missing_rows: list[dict[str, Any]] = []
    if checkpoint is None:
        missing_rows.append({"name": "dinov2_checkpoint", "path": ""})
    if not candidate_rows_path.exists():
        missing_rows.append({"name": "candidate_rows", "path": _rel(candidate_rows_path)})
    if not witness_summary.exists():
        missing_rows.append({"name": "witness_summary", "path": _rel(witness_summary)})
    if missing_rows:
        _write_csv(output_root / "missing_input_rows.csv", missing_rows)
        summary = {"phase": "v72_phase2_dense_token_proposals", "decision": "FAIL_MISSING_INPUTS", "missing_input_count": len(missing_rows), "gate": {"pass": False}}
        _write_json(output_root / "dense_token_proposal_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    adapter = FrozenFeatureAdapter(
        backend=str(args.backend),
        device=str(args.device),
        checkpoint=str(checkpoint or ""),
        short_side=int(args.short_side),
    )
    proposal_rows_out: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    processed = 0
    dense_frame_count = 0
    for scene in scenes:
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "pipeline_root"})
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
                break
            chunk_id = str(chunk.get("chunk_id"))
            rows = candidates_by_chunk.get(chunk_id, [])
            if not rows:
                continue
            source_rows_by_frame = _select_broad_sources(rows, int(args.max_broad_sources_per_frame))
            if not any(source_rows_by_frame.values()):
                continue
            processed += 1
            print(f"[v72-dense-token] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if int(args.max_frames_per_chunk) > 0:
                frame_ids = frame_ids[: int(args.max_frames_per_chunk)]
            if not frame_ids:
                continue
            eval_shape_hw = tuple(int(value) for value in stream.load_depth(frame_ids[0]).shape)
            frame_gt: dict[int, np.ndarray] = {}
            frame_gt_area: dict[int, dict[int, int]] = {}
            frame_mask_eval: dict[int, np.ndarray] = {}
            frame_mask_rgb: dict[int, np.ndarray] = {}
            clean_by_frame = _clean_prototypes(rows, int(args.max_prototypes_per_frame))
            dense_by_variant: dict[str, list[Proposal]] = defaultdict(list)
            source_stats: dict[str, float] = {}

            for frame_id in frame_ids:
                gt = _load_gt_2d(scene, int(frame_id), eval_shape_hw)
                frame_gt[int(frame_id)] = gt
                labels, counts = np.unique(gt, return_counts=True)
                frame_gt_area[int(frame_id)] = {int(label): int(count) for label, count in zip(labels, counts) if int(label) > 0}
                mask_eval = _resize_label(mask_dir / f"{int(frame_id)}.png", eval_shape_hw)
                if mask_eval is not None:
                    frame_mask_eval[int(frame_id)] = mask_eval
                for row in rows:
                    if int(row.get("frame_id") or -1) != int(frame_id) or mask_eval is None:
                        continue
                    binary = mask_eval == int(row.get("mask_id") or -1)
                    _gid, iou = _mask_diagnostic(binary, gt, frame_gt_area[int(frame_id)])
                    source_stats[str(row.get("mask_observation_id") or "")] = iou

                sources = source_rows_by_frame.get(int(frame_id), [])
                if not sources:
                    continue
                try:
                    rgb = stream.load_rgb(int(frame_id))
                except FileNotFoundError:
                    missing_rows.append({"scene_id": scene, "frame_id": int(frame_id), "missing": "rgb"})
                    continue
                mask_rgb = _resize_label(mask_dir / f"{int(frame_id)}.png", rgb.shape[:2])
                if mask_rgb is None:
                    missing_rows.append({"scene_id": scene, "frame_id": int(frame_id), "missing": "mask_png"})
                    continue
                frame_mask_rgb[int(frame_id)] = mask_rgb
                try:
                    generated = _generate_dense_for_frame(
                        scene_id=scene,
                        chunk_id=chunk_id,
                        frame_id=int(frame_id),
                        source_rows=sources,
                        clean_rows=clean_by_frame.get(int(frame_id), []),
                        rgb=rgb,
                        mask_rgb=mask_rgb,
                        eval_shape_hw=eval_shape_hw,
                        adapter=adapter,
                        args=args,
                    )
                except Exception as exc:
                    missing_rows.append({"scene_id": scene, "frame_id": int(frame_id), "missing": f"dense_proposal_failed:{type(exc).__name__}:{exc}"})
                    continue
                dense_frame_count += 1
                for variant, subset in generated.items():
                    dense_by_variant[variant].extend(subset)

            baseline = _baseline_existing_proposals(
                rows=rows,
                scene_id=scene,
                chunk_id=chunk_id,
                frame_mask_eval=frame_mask_eval,
                frame_gt=frame_gt,
                entropy_scale=float(args.entropy_variance_scale),
            )
            proposal_source_by_variant = {"SP0_existing_masks_baseline": baseline, **dense_by_variant}
            for subset in proposal_source_by_variant.values():
                _annotate_diagnostics(subset, frame_gt, frame_gt_area, source_stats)
                proposal_rows_out.extend(_proposal_rows(subset))
            hybrid_by_variant = {
                f"SP7_existing_plus_{variant}": baseline + subset
                for variant, subset in dense_by_variant.items()
                if subset
            }
            all_by_variant = {**proposal_source_by_variant, **hybrid_by_variant}
            for variant, subset in sorted(all_by_variant.items()):
                row = _evaluate_proposals(proposals=subset, frame_gt=frame_gt, variant=variant)
                row.update(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk_id,
                        "dense_frame_count": sum(1 for prop in subset if prop.variant != "SP0_existing_masks_baseline"),
                        "broad_to_subproposal_GT_best_gain": _mean([prop.majority_iou - prop.source_majority_iou for prop in subset if prop.variant != "SP0_existing_masks_baseline"]),
                        "broad_to_subproposal_entropy_drop": _mean([prop.source_semantic_entropy - prop.semantic_entropy for prop in subset if prop.variant != "SP0_existing_masks_baseline"]),
                    }
                )
                metric_rows.append(row)
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break

    variant_summary_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in metric_rows}):
        subset = [row for row in metric_rows if row["variant"] == variant]
        variant_summary_rows.append(
            {
                "variant": variant,
                "chunk_count": len(subset),
                "proposal_count_per_chunk_mean": _mean([_float(row.get("proposal_count")) for row in subset]),
                "proposal_count_per_frame_mean": _mean([_float(row.get("proposal_count_per_frame")) for row in subset]),
                "proposal_oracle_SF50_mean": _mean([_float(row.get("proposal_oracle_SF50_by_majority_GT_diagnostic")) for row in subset]),
                "proposal_oracle_AP50_mean": _mean([_float(row.get("proposal_oracle_AP50_by_majority_GT_diagnostic")) for row in subset]),
                "proposal_oracle_GT_best_IoU_mean": _mean([_float(row.get("proposal_oracle_GT_best_IoU_mean_diagnostic")) for row in subset]),
                "proposal_majority_IoU_mean": _mean([_float(row.get("proposal_majority_IoU_mean_diagnostic")) for row in subset]),
                "proposal_IoU50_rate_mean": _mean([_float(row.get("proposal_IoU50_rate_diagnostic")) for row in subset]),
                "proposal_background_proxy_rate_mean": _mean([_float(row.get("proposal_background_proxy_rate")) for row in subset]),
                "proposal_source_broad_rate_mean": _mean([_float(row.get("proposal_source_broad_rate")) for row in subset]),
                "broad_to_subproposal_GT_best_gain_mean": _mean([_float(row.get("broad_to_subproposal_GT_best_gain")) for row in subset]),
                "broad_to_subproposal_entropy_drop_mean": _mean([_float(row.get("broad_to_subproposal_entropy_drop")) for row in subset]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_evaluation": True,
                "diagnostic_only": variant != "SP0_existing_masks_baseline",
                "forbidden_for_method_table": variant != "SP0_existing_masks_baseline",
            }
        )

    baseline_summary = next((row for row in variant_summary_rows if row["variant"] == "SP0_existing_masks_baseline"), {})
    dense_candidates = [row for row in variant_summary_rows if row["variant"] != "SP0_existing_masks_baseline"]
    best_dense = max(dense_candidates, key=lambda row: _float(row.get("proposal_oracle_SF50_mean"), -1.0), default={})
    sf50_gain = _float(best_dense.get("proposal_oracle_SF50_mean"), 0.0) - _float(baseline_summary.get("proposal_oracle_SF50_mean"), 0.0)
    majority_gain = _float(best_dense.get("proposal_majority_IoU_mean"), 0.0) - _float(baseline_summary.get("proposal_majority_IoU_mean"), 0.0)
    entropy_drop = _float(best_dense.get("broad_to_subproposal_entropy_drop_mean"), 0.0)
    background_drop = _float(baseline_summary.get("proposal_background_proxy_rate_mean"), 0.0) - _float(best_dense.get("proposal_background_proxy_rate_mean"), 1.0)
    phase2_pass = (
        bool(best_dense)
        and sf50_gain >= 0.10
        and majority_gain >= 0.08
        and entropy_drop >= 0.10
        and background_drop >= 0.10
        and _float(best_dense.get("proposal_count_per_frame_mean"), 9999.0) <= 60.0
        and _float(best_dense.get("proposal_count_per_chunk_mean"), 9999.0) <= 1200.0
    )
    summary = {
        "phase": "v72_phase2_dense_token_proposals",
        "decision": "PASS_V72_PHASE2_DENSE_TOKEN_PROPOSALS" if phase2_pass else "NO_GO_PHASE2_DENSE_TOKEN_PROPOSALS",
        "processed_chunk_count": processed,
        "dense_frame_count": dense_frame_count,
        "proposal_row_count": len(proposal_rows_out),
        "best_dense_variant": best_dense.get("variant"),
        "baseline_SP0_oracle_SF50": baseline_summary.get("proposal_oracle_SF50_mean"),
        "best_dense_oracle_SF50": best_dense.get("proposal_oracle_SF50_mean"),
        "best_dense_minus_SP0_oracle_SF50": sf50_gain,
        "baseline_SP0_majority_IoU": baseline_summary.get("proposal_majority_IoU_mean"),
        "best_dense_majority_IoU": best_dense.get("proposal_majority_IoU_mean"),
        "best_dense_minus_SP0_majority_IoU": majority_gain,
        "best_dense_broad_to_subproposal_entropy_drop": entropy_drop,
        "best_dense_background_proxy_drop_vs_SP0": background_drop,
        "gate": {
            "proposal_oracle_SF50_gain_ge_0p10": sf50_gain >= 0.10,
            "proposal_majority_IoU_gain_ge_0p08": majority_gain >= 0.08,
            "broad_to_subproposal_entropy_drop_ge_0p10": entropy_drop >= 0.10,
            "background_proxy_rate_drop_ge_0p10": background_drop >= 0.10,
            "proposal_count_per_frame_mean_le_60": _float(best_dense.get("proposal_count_per_frame_mean"), 9999.0) <= 60.0,
            "proposal_count_per_chunk_mean_le_1200": _float(best_dense.get("proposal_count_per_chunk_mean"), 9999.0) <= 1200.0,
            "uses_gt_for_prediction_false": True,
            "pass": phase2_pass,
        },
        "method_boundary": {
            "uses_gt_for_prediction": False,
            "gt_used_for_diagnostic_evaluation": True,
            "training_free": True,
            "backend": str(args.backend),
            "checkpoint": str(checkpoint or ""),
            "RADIO_unavailable": True,
        },
        "notes": [
            "This continuation recomputes DINOv2 dense patch tokens from RGB and generates new token-level proposal regions inside broad/underseg source masks.",
            "GT labels are used only after proposal generation for diagnostic majority-GT oracle grouping and IoU metrics.",
            "SP1/SP2/SP3/SP4/SP6 follow the Phase2 repair ladder: k-means, affinity components, prototype region grow, clean-mask constrained region grow, and background-suppressed token proposals.",
            "SP8 follows the plan repair direction for over-fragmented dense proposals by merging adjacent DINO affinity components using non-GT component-level feature similarity.",
            "SP7 hybrid rows evaluate existing masks plus dense token proposals; the proposal rows remain non-GT and are not duplicated in proposal_rows.csv.",
        ],
    }
    _write_csv(output_root / "proposal_rows.csv", proposal_rows_out)
    _write_csv(output_root / "proposal_metric_rows.csv", metric_rows)
    _write_csv(output_root / "proposal_variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_json(output_root / "dense_token_proposal_summary.json", summary)
    sha_rows = []
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "output"})
    for path in [candidate_rows_path, witness_summary]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "input"})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 Phase2 dense DINO token proposal repair.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase2_dense_token_proposals")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--max-frames-per-chunk", type=int, default=0)
    parser.add_argument("--backend", default="dinov2_timm", choices=["dinov2_timm", "rgb_stats"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--short-side", type=int, default=518)
    parser.add_argument("--max-broad-sources-per-frame", type=int, default=6)
    parser.add_argument("--max-prototypes-per-frame", type=int, default=8)
    parser.add_argument("--max-proposals-per-source-variant", type=int, default=6)
    parser.add_argument("--max-proposals-per-frame-per-variant", type=int, default=60)
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--kmeans-k", default="2,3,4")
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--affinity-thresholds", default="0.55,0.65")
    parser.add_argument("--component-merge-similarity-thresholds", default="0.62,0.70")
    parser.add_argument("--prototype-similarity-threshold", type=float, default=0.55)
    parser.add_argument("--clean-hint-similarity-thresholds", default="0.35,0.45")
    parser.add_argument("--token-nms-iou", type=float, default=0.85)
    parser.add_argument("--entropy-variance-scale", type=float, default=0.001)
    parser.add_argument("--sp6-entropy-drop", type=float, default=0.04)
    parser.add_argument("--max-component-area-ratio", type=float, default=0.22)
    parser.add_argument("--min-component-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-dense-component-area-ratio", type=float, default=0.75)
    parser.add_argument("--area-bin-small-threshold", type=float, default=0.0)
    parser.add_argument("--area-bin-mid-threshold", type=float, default=0.0)
    parser.add_argument("--area-bin-max-small-per-frame", type=int, default=-1)
    parser.add_argument("--area-bin-max-mid-per-frame", type=int, default=-1)
    parser.add_argument("--area-bin-max-large-per-frame", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
