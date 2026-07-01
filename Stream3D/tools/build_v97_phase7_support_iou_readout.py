#!/usr/bin/env python3
"""Build v97 Phase7 support-IoU CropFormer readout diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


PHASE_ID = "v97_phase7_support_iou_readout"
RUN_ID = "v97_phase7_support_iou_readout"
DEFAULT_PHASE0 = ROOT / "outputs/audit/v97_phase0_fact_lock"
DEFAULT_PHASE5 = ROOT / "outputs/audit/v97_phase5_object_birth_region_proxy_500k"
DEFAULT_PHASE6 = ROOT / "outputs/audit/v97_phase6_render_splat_C0_region_proxy_500k_gpu7_sparsefix"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase7_support_iou_readout"


@dataclass
class FrameInfo:
    label: np.ndarray
    mask_area: np.ndarray
    image_area: int
    mask_path: Path
    label_t: torch.Tensor | None = None
    mask_area_t: torch.Tensor | None = None


@dataclass
class SupportRecord:
    index: int
    row: dict[str, Any]
    ys: np.ndarray
    xs: np.ndarray
    values: np.ndarray
    object_score: float
    base_best: dict[str, Any]
    sigma_best: dict[str, Any]


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in materialized:
            for key in row:
                if str(key).startswith("_"):
                    continue
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _mask_path_lookup(source_rows: Path) -> dict[tuple[str, str, int], Path]:
    out: dict[tuple[str, str, int], Path] = {}
    for row in _read_csv(source_rows):
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        raw = row.get("mask_path", "")
        if raw and key not in out:
            out[key] = _project(raw)
    return out


def _load_object_scores(phase5_root: Path, variant_id: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_csv(phase5_root / "object_candidate_rows.csv"):
        if row.get("variant_id") != variant_id:
            continue
        out[row.get("object_id", "")] = {
            "object_score": float(_num(row.get("score"))),
            "micro_primitive_count": int(_num(row.get("micro_primitive_count"))),
            "frame_support_count": int(_num(row.get("frame_support_count"))),
            "masklet_support_count": int(_num(row.get("masklet_support_count"))),
            "source_mask_count": int(_num(row.get("source_mask_count"))),
        }
    return out


def _frame_info(frame_key: tuple[str, str, int], mask_lookup: dict[tuple[str, str, int], Path], cache: dict[tuple[str, str, int], FrameInfo], device: str) -> FrameInfo:
    if frame_key in cache:
        return cache[frame_key]
    path = mask_lookup.get(frame_key)
    if path is None or not path.exists():
        raise FileNotFoundError(f"missing CropFormer mask raster for frame={frame_key}: {path}")
    label = _load_label(path)
    positive = label[label > 0]
    max_label = int(np.max(positive)) if positive.size else 0
    mask_area = np.bincount(positive.astype(np.int64), minlength=max_label + 1).astype(np.int64, copy=False)
    label_t: torch.Tensor | None = None
    mask_area_t: torch.Tensor | None = None
    if device.startswith("cuda"):
        label_t = torch.as_tensor(label, dtype=torch.int64, device=device)
        mask_area_t = torch.as_tensor(mask_area, dtype=torch.int64, device=device)
    info = FrameInfo(label=label, mask_area=mask_area, image_area=int(label.size), mask_path=path, label_t=label_t, mask_area_t=mask_area_t)
    cache[frame_key] = info
    return info


def _load_sparse_support(phase6_root: Path) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    support_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_csv(phase6_root / "object_frame_support_rows.csv")):
        payload = dict(row)
        payload["object_frame_index"] = idx
        support_rows.append(payload)
    sparse: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in _read_csv(phase6_root / "heatmap_manifest_rows.csv"):
        raw = row.get("heatmap_shard_path", "")
        if not raw:
            continue
        path = _project(raw)
        with np.load(path, allow_pickle=False) as payload:
            indices = np.asarray(payload["object_frame_index"], dtype=np.int64)
            ys = np.asarray(payload["y"], dtype=np.int32)
            xs = np.asarray(payload["x"], dtype=np.int32)
            values = np.asarray(payload["value"], dtype=np.float32)
            for object_frame_index in np.unique(indices):
                mask = indices == int(object_frame_index)
                sparse[int(object_frame_index)] = (ys[mask], xs[mask], values[mask])
    return support_rows, sparse


def _dilate_sparse(ys: np.ndarray, xs: np.ndarray, radius: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0 or ys.size == 0:
        return ys.astype(np.int32, copy=False), xs.astype(np.int32, copy=False)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    y_parts = []
    x_parts = []
    for dy, dx in offsets:
        y_parts.append(ys.astype(np.int64, copy=False) + int(dy))
        x_parts.append(xs.astype(np.int64, copy=False) + int(dx))
    y_all = np.concatenate(y_parts)
    x_all = np.concatenate(x_parts)
    valid = (y_all >= 0) & (y_all < int(height)) & (x_all >= 0) & (x_all < int(width))
    linear = (y_all[valid] * int(width) + x_all[valid]).astype(np.int64, copy=False)
    if linear.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    linear = np.unique(linear)
    return (linear // int(width)).astype(np.int32, copy=False), (linear % int(width)).astype(np.int32, copy=False)


def _empty_candidate(support_area: int, support_area_ratio: float, source: str) -> dict[str, Any]:
    return {
        "candidate_rank": 0,
        "mask_id": 0,
        "support_iou": 0.0,
        "support_recall": 0.0,
        "mask_precision": 0.0,
        "intersection_pixels": 0,
        "support_area_pixels": int(support_area),
        "mask_area_pixels": 0,
        "selected_mask_area_ratio": 0.0,
        "support_area_ratio": float(support_area_ratio),
        "broad_area_ratio": 0.0,
        "broad_risk": 0.0,
        "conflict_risk": 1.0 if support_area > 0 else 0.0,
        "semantic_alignment": 1.0,
        "semantic_alignment_source": "neutral_no_dense_semantic_tensor",
        "readout_score": 0.0,
        "candidate_source": source,
    }


def _candidate_rows_for_support(
    *,
    info: FrameInfo,
    ys: np.ndarray,
    xs: np.ndarray,
    source: str,
    top_k: int,
    device: str,
) -> list[dict[str, Any]]:
    support_area = int(ys.size)
    support_area_ratio = float(support_area / max(1, info.image_area))
    if support_area <= 0:
        return [_empty_candidate(0, 0.0, source)]
    if device.startswith("cuda") and info.label_t is not None and info.mask_area_t is not None:
        y_t = torch.as_tensor(ys.astype(np.int64, copy=False), device=device)
        x_t = torch.as_tensor(xs.astype(np.int64, copy=False), device=device)
        labels = info.label_t[y_t, x_t]
        labels = labels[labels > 0]
        if labels.numel() == 0:
            return [_empty_candidate(support_area, support_area_ratio, source)]
        intersections = torch.bincount(labels, minlength=int(info.mask_area_t.numel()))
        ids = torch.nonzero(intersections > 0, as_tuple=False).flatten()
        ids = ids[ids > 0]
        if ids.numel() == 0:
            return [_empty_candidate(support_area, support_area_ratio, source)]
        inter = intersections[ids].to(torch.float32)
        mask_area = info.mask_area_t[ids].to(torch.float32)
        union = float(support_area) + mask_area - inter
        support_iou = inter / torch.clamp(union, min=1.0)
        support_recall = inter / max(1.0, float(support_area))
        precision = inter / torch.clamp(mask_area, min=1.0)
        area_ratio = mask_area / max(1.0, float(info.image_area))
        broad_area_ratio = mask_area / max(1.0, float(support_area))
        broad_risk = torch.clamp(area_ratio - float(support_area_ratio), min=0.0)
        conflict_risk = torch.clamp(1.0 - support_recall, min=0.0, max=1.0)
        score = 0.45 * support_iou + 0.25 * support_recall + 0.20 * precision + 0.10 - 0.10 * broad_risk - 0.15 * conflict_risk
        order = torch.argsort(score, descending=True)[: int(top_k)]
        ids_np = ids[order].detach().cpu().numpy()
        inter_np = inter[order].detach().cpu().numpy()
        mask_area_np = mask_area[order].detach().cpu().numpy()
        iou_np = support_iou[order].detach().cpu().numpy()
        recall_np = support_recall[order].detach().cpu().numpy()
        precision_np = precision[order].detach().cpu().numpy()
        area_ratio_np = area_ratio[order].detach().cpu().numpy()
        broad_ratio_np = broad_area_ratio[order].detach().cpu().numpy()
        broad_risk_np = broad_risk[order].detach().cpu().numpy()
        conflict_np = conflict_risk[order].detach().cpu().numpy()
        score_np = score[order].detach().cpu().numpy()
    else:
        labels = info.label[ys, xs]
        labels = labels[labels > 0]
        if labels.size == 0:
            return [_empty_candidate(support_area, support_area_ratio, source)]
        intersections = np.bincount(labels.astype(np.int64), minlength=info.mask_area.shape[0]).astype(np.float64, copy=False)
        ids_np = np.flatnonzero(intersections > 0)
        ids_np = ids_np[ids_np > 0]
        if ids_np.size == 0:
            return [_empty_candidate(support_area, support_area_ratio, source)]
        inter_np = intersections[ids_np]
        mask_area_np = info.mask_area[ids_np].astype(np.float64, copy=False)
        union = support_area + mask_area_np - inter_np
        iou_np = inter_np / np.maximum(1.0, union)
        recall_np = inter_np / max(1.0, float(support_area))
        precision_np = inter_np / np.maximum(1.0, mask_area_np)
        area_ratio_np = mask_area_np / max(1.0, float(info.image_area))
        broad_ratio_np = mask_area_np / max(1.0, float(support_area))
        broad_risk_np = np.maximum(0.0, area_ratio_np - support_area_ratio)
        conflict_np = np.clip(1.0 - recall_np, 0.0, 1.0)
        score_np = 0.45 * iou_np + 0.25 * recall_np + 0.20 * precision_np + 0.10 - 0.10 * broad_risk_np - 0.15 * conflict_np
        order = np.argsort(score_np)[::-1][: int(top_k)]
        ids_np = ids_np[order]
        inter_np = inter_np[order]
        mask_area_np = mask_area_np[order]
        iou_np = iou_np[order]
        recall_np = recall_np[order]
        precision_np = precision_np[order]
        area_ratio_np = area_ratio_np[order]
        broad_ratio_np = broad_ratio_np[order]
        broad_risk_np = broad_risk_np[order]
        conflict_np = conflict_np[order]
        score_np = score_np[order]
    rows: list[dict[str, Any]] = []
    for rank, mask_id in enumerate(ids_np, start=1):
        rows.append(
            {
                "candidate_rank": int(rank),
                "mask_id": int(mask_id),
                "support_iou": float(iou_np[rank - 1]),
                "support_recall": float(recall_np[rank - 1]),
                "mask_precision": float(precision_np[rank - 1]),
                "intersection_pixels": int(inter_np[rank - 1]),
                "support_area_pixels": int(support_area),
                "mask_area_pixels": int(mask_area_np[rank - 1]),
                "selected_mask_area_ratio": float(area_ratio_np[rank - 1]),
                "support_area_ratio": float(support_area_ratio),
                "broad_area_ratio": float(broad_ratio_np[rank - 1]),
                "broad_risk": float(broad_risk_np[rank - 1]),
                "conflict_risk": float(conflict_np[rank - 1]),
                "semantic_alignment": 1.0,
                "semantic_alignment_source": "neutral_no_dense_semantic_tensor",
                "readout_score": float(score_np[rank - 1]),
                "candidate_source": source,
            }
        )
    return rows if rows else [_empty_candidate(support_area, support_area_ratio, source)]


def _rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < order.size:
        j = i + 1
        while j < order.size and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    xr = _rankdata(xs)
    yr = _rankdata(ys)
    if float(np.std(xr)) <= 0.0 or float(np.std(yr)) <= 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _support_coords(record: SupportRecord, *, mode: str, radius: int, frame_info: FrameInfo) -> tuple[np.ndarray, np.ndarray]:
    if mode == "sigma":
        return _dilate_sparse(record.ys, record.xs, radius, frame_info.label.shape[0], frame_info.label.shape[1])
    return record.ys, record.xs


def _select(record: SupportRecord, variant_id: str, args: argparse.Namespace, frame_info: FrameInfo) -> dict[str, Any]:
    candidate = record.sigma_best if variant_id == "R6_uncertainty_sigma_snap" else record.base_best
    source_mode = "sigma" if variant_id == "R6_uncertainty_sigma_snap" else "base"
    support_ys, support_xs = _support_coords(record, mode=source_mode, radius=int(args.sigma_dilate_radius), frame_info=frame_info)
    support_area = int(support_ys.size)
    support_area_ratio = float(support_area / max(1, frame_info.image_area))
    mask_id = int(candidate.get("mask_id", 0))
    support_iou = float(candidate.get("support_iou", 0.0))
    recall = float(candidate.get("support_recall", 0.0))
    precision = float(candidate.get("mask_precision", 0.0))
    broad_ratio = float(candidate.get("broad_area_ratio", 0.0))
    broad_risk = float(candidate.get("broad_risk", 0.0))
    conflict = float(candidate.get("conflict_risk", 1.0))
    snap_good = (
        mask_id > 0
        and support_iou >= float(args.snap_iou_threshold)
        and recall >= float(args.snap_recall_threshold)
        and precision >= float(args.snap_precision_threshold)
    )
    broad = broad_ratio > float(args.broad_area_ratio_threshold) or broad_risk > float(args.broad_risk_threshold)
    conflict_high = conflict > float(args.conflict_risk_threshold)
    can_carve = mask_id > 0 and recall >= float(args.carve_recall_threshold)
    emitted = True
    mode = "no_emit"
    mask_source_type = "none"
    generated_mask_id = ""
    selected_mask_id = 0
    reason = ""
    if variant_id == "R0_snap_only":
        if snap_good and not broad:
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        else:
            emitted, reason = False, "snap_threshold_or_broad_risk_fail"
    elif variant_id == "R1_carve_if_broad":
        if can_carve and broad:
            mode, mask_source_type, selected_mask_id = "carved_mask", "cropformer_carved_mask", mask_id
        elif snap_good:
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        else:
            emitted, reason = False, "no_snap_or_carve_candidate"
    elif variant_id == "R2_support_only_fallback":
        if snap_good and not broad:
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        elif support_area >= int(args.min_pred_pixels):
            mode, mask_source_type, generated_mask_id = "support_only", "generated_support_mask", f"{record.row['object_id']}|{record.row['frame_id']}|support"
        else:
            emitted, reason = False, "support_too_small"
    elif variant_id == "R3_snap_or_carve_gtfree_rule":
        if snap_good and not broad:
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        elif can_carve:
            mode, mask_source_type, selected_mask_id = "carved_mask", "cropformer_carved_mask", mask_id
        elif support_area >= int(args.min_pred_pixels):
            mode, mask_source_type, generated_mask_id = "support_only", "generated_support_mask", f"{record.row['object_id']}|{record.row['frame_id']}|support"
        else:
            emitted, reason = False, "no_gtfree_readout_candidate"
    elif variant_id == "R4_semantic_alignment_snap":
        if snap_good and not broad and float(candidate.get("semantic_alignment", 0.0)) >= float(args.semantic_alignment_threshold):
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        else:
            emitted, reason = False, "semantic_snap_threshold_fail_or_no_dense_semantic"
    elif variant_id == "R5_conflict_aware_snap":
        if snap_good and not broad and not conflict_high:
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        elif can_carve and not conflict_high:
            mode, mask_source_type, selected_mask_id = "carved_mask", "cropformer_carved_mask", mask_id
        else:
            emitted, reason = False, "conflict_risk_or_threshold_fail"
    elif variant_id == "R6_uncertainty_sigma_snap":
        if snap_good and not broad:
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        elif can_carve:
            mode, mask_source_type, selected_mask_id = "carved_mask", "cropformer_carved_mask", mask_id
        elif support_area >= int(args.min_pred_pixels):
            mode, mask_source_type, generated_mask_id = "support_only", "generated_support_mask", f"{record.row['object_id']}|{record.row['frame_id']}|sigma_support"
        else:
            emitted, reason = False, "sigma_support_too_small"
    elif variant_id == "R7_hybrid_best_gtfree":
        if snap_good and not broad and not conflict_high:
            mode, mask_source_type, selected_mask_id = "snap_to_mask", "cropformer_mask", mask_id
        elif can_carve and not (conflict_high and support_iou < float(args.snap_iou_threshold)):
            mode, mask_source_type, selected_mask_id = "carved_mask", "cropformer_carved_mask", mask_id
        elif support_area >= int(args.min_pred_pixels):
            mode, mask_source_type, generated_mask_id = "support_only", "generated_support_mask", f"{record.row['object_id']}|{record.row['frame_id']}|hybrid_support"
        else:
            emitted, reason = False, "hybrid_no_candidate"
    else:
        raise ValueError(f"unknown variant: {variant_id}")
    if emitted and mode in {"support_only", "carved_mask"}:
        pred_pixels = support_area if mode == "support_only" else int(candidate.get("intersection_pixels", 0))
        if pred_pixels < int(args.min_pred_pixels):
            emitted = False
            reason = "pred_pixels_lt_min_pred_pixels"
    return {
        "schema_version": "stream4d_v97_phase7_mask_selection_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "scene_id": record.row.get("scene_id", ""),
        "window_id": record.row.get("window_id", ""),
        "object_id": record.row.get("object_id", ""),
        "frame_id": int(_num(record.row.get("frame_id"))),
        "object_frame_index": int(record.index),
        "readout_mode": mode,
        "mask_source_type": mask_source_type,
        "mask_id": int(selected_mask_id),
        "generated_mask_id": generated_mask_id,
        "emitted": bool(emitted),
        "no_emit_reason": reason,
        "object_score": float(record.object_score),
        "support_iou": support_iou,
        "support_recall": recall,
        "mask_precision": precision,
        "semantic_alignment": float(candidate.get("semantic_alignment", 1.0)),
        "semantic_alignment_source": candidate.get("semantic_alignment_source", "neutral_no_dense_semantic_tensor"),
        "broad_risk": broad_risk,
        "broad_area_ratio": broad_ratio,
        "conflict_risk": conflict,
        "support_area_ratio": support_area_ratio,
        "selected_mask_area_ratio": float(candidate.get("selected_mask_area_ratio", 0.0)),
        "readout_score": float(candidate.get("readout_score", 0.0)),
        "candidate_source": candidate.get("candidate_source", source_mode),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "_record": record,
        "_source_mode": source_mode,
    }


def _apply_wta(selections: list[dict[str, Any]], enabled: bool) -> int:
    if not enabled:
        return 0
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        if not row.get("emitted"):
            continue
        if row.get("mask_source_type") not in {"cropformer_mask", "cropformer_carved_mask"}:
            continue
        groups[(row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["mask_id"]))].append(row)
    dropped = 0
    for rows in groups.values():
        if len(rows) <= 1:
            continue
        winner = max(rows, key=lambda row: (float(row.get("object_score", 0.0)), float(row.get("readout_score", 0.0)), str(row.get("object_id", ""))))
        for row in rows:
            if row is winner:
                continue
            row["emitted"] = False
            row["readout_mode"] = "no_emit"
            row["no_emit_reason"] = "wta_duplicate_mask_drop"
            dropped += 1
    return dropped


def _selection_coords(selection: dict[str, Any], frame_info: FrameInfo, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    record: SupportRecord = selection["_record"]
    source_mode = str(selection.get("_source_mode", "base"))
    ys, xs = _support_coords(record, mode=source_mode, radius=int(args.sigma_dilate_radius), frame_info=frame_info)
    mode = selection.get("readout_mode")
    if mode == "support_only":
        return ys, xs
    if mode == "carved_mask":
        if int(args.carve_dilate_radius) > 0 and source_mode != "sigma":
            ys, xs = _dilate_sparse(ys, xs, int(args.carve_dilate_radius), frame_info.label.shape[0], frame_info.label.shape[1])
        labels = frame_info.label[ys, xs]
        keep = labels == int(selection.get("mask_id", 0))
        return ys[keep], xs[keep]
    return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)


def _evaluate_variant(
    *,
    variant_id: str,
    selections: list[dict[str, Any]],
    eval_frame_keys: set[tuple[str, str, int]],
    frame_cache: dict[tuple[str, str, int], FrameInfo],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int, dict[str, float]]:
    object_index: dict[str, int] = {}
    object_scores: dict[str, float] = {}
    index_to_object: dict[int, str] = {}
    scene_offsets = {scene: (idx + 1) * 1_000_000 for idx, scene in enumerate(sorted({key[0] for key in eval_frame_keys}))}
    by_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        if row.get("emitted"):
            by_frame[(row["scene_id"], row["window_id"], int(row["frame_id"]))].append(row)
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    missing_mask_raster_count = 0
    mask_bool_cache: dict[tuple[tuple[str, str, int], int], np.ndarray] = {}
    for frame_key in sorted(eval_frame_keys):
        scene, window, frame_id = frame_key
        info = frame_cache[frame_key]
        pred = np.zeros(info.label.shape, dtype=np.int64)
        rows = sorted(by_frame.get(frame_key, []), key=lambda row: (-float(row.get("object_score", 0.0)), -float(row.get("readout_score", 0.0)), str(row.get("object_id", ""))))
        for row in rows:
            oid = str(row["object_id"])
            if oid not in object_index:
                idx = len(object_index) + 1
                object_index[oid] = idx
                index_to_object[idx] = oid
            pred_id = object_index[oid]
            object_scores[oid] = max(float(object_scores.get(oid, 0.0)), float(row.get("object_score", 0.0)))
            mode = row.get("readout_mode")
            if mode == "snap_to_mask":
                cache_key = (frame_key, int(row.get("mask_id", 0)))
                mask = mask_bool_cache.get(cache_key)
                if mask is None:
                    mask = info.label == int(row.get("mask_id", 0))
                    mask_bool_cache[cache_key] = mask
                pixel_collision_count += int(np.count_nonzero((pred > 0) & mask))
                pred[(pred == 0) & mask] = pred_id
            else:
                ys, xs = _selection_coords(row, info, args)
                if ys.size == 0:
                    continue
                occupied = pred[ys, xs] > 0
                pixel_collision_count += int(np.count_nonzero(occupied))
                keep = ~occupied
                pred[ys[keep], xs[keep]] = pred_id
        gt = _load_gt_2d(scene, frame_id, info.label.shape)
        gt = np.where(gt > 0, gt + int(scene_offsets.get(scene, 0)), 0).astype(np.int64, copy=False)
        acc.add(pred, gt)
        frame_rows.append(
            {
                "schema_version": "stream4d_v97_phase7_preview_frame_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": scene,
                "window_id": window,
                "frame_id": int(frame_id),
                "status": "evaluated",
                "emitted_object_count": len(rows),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_eval": True,
                "uses_future": False,
            }
        )
    input_scores = np.ones((len(object_index),), dtype=np.float32)
    for oid, idx in object_index.items():
        input_scores[idx - 1] = float(object_scores.get(oid, 1.0))
    summary, iou, pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=int(args.min_pred_pixels),
        min_gt_pixels=int(args.min_gt_pixels),
        score_mode="input",
        input_scores=input_scores,
    )
    pred_best_by_object: dict[str, float] = {}
    if iou.shape[0] > 0:
        pred_best = np.max(iou, axis=1) if iou.shape[1] else np.zeros((iou.shape[0],), dtype=np.float32)
        for pid, best in zip(pred_ids, pred_best):
            oid = index_to_object.get(int(pid), "")
            if oid:
                pred_best_by_object[oid] = float(best)
    return summary, frame_rows, int(pixel_collision_count), int(missing_mask_raster_count), pred_best_by_object


def _evaluate_variant_scene(
    *,
    variant_id: str,
    selections: list[dict[str, Any]],
    eval_scene_frame_keys: set[tuple[str, int]],
    frame_cache: dict[tuple[str, str, int], FrameInfo],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int, dict[str, float], int]:
    object_index: dict[str, int] = {}
    object_scores: dict[str, float] = {}
    index_to_object: dict[int, str] = {}
    scene_offsets = {scene: (idx + 1) * 1_000_000 for idx, scene in enumerate(sorted({key[0] for key in eval_scene_frame_keys}))}
    frame_key_by_scene_frame: dict[tuple[str, int], tuple[str, str, int]] = {}
    mask_path_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    for frame_key, info in frame_cache.items():
        scene, _window, frame_id = frame_key
        scene_frame_key = (scene, int(frame_id))
        if scene_frame_key in eval_scene_frame_keys:
            frame_key_by_scene_frame.setdefault(scene_frame_key, frame_key)
            mask_path_sets[scene_frame_key].add(_rel(info.mask_path))
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        if row.get("emitted"):
            by_frame[(row["scene_id"], int(row["frame_id"]))].append(row)
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    missing_mask_raster_count = 0
    mask_path_conflict_count = sum(1 for paths in mask_path_sets.values() if len(paths) > 1)
    mask_bool_cache: dict[tuple[tuple[str, int], int], np.ndarray] = {}
    for scene_frame_key in sorted(eval_scene_frame_keys):
        scene, frame_id = scene_frame_key
        cache_key = frame_key_by_scene_frame.get(scene_frame_key)
        if cache_key is None:
            missing_mask_raster_count += 1
            continue
        info = frame_cache[cache_key]
        pred = np.zeros(info.label.shape, dtype=np.int64)
        rows = sorted(
            by_frame.get(scene_frame_key, []),
            key=lambda row: (-float(row.get("object_score", 0.0)), -float(row.get("readout_score", 0.0)), str(row.get("object_id", ""))),
        )
        for row in rows:
            oid = str(row["object_id"])
            if oid not in object_index:
                idx = len(object_index) + 1
                object_index[oid] = idx
                index_to_object[idx] = oid
            pred_id = object_index[oid]
            object_scores[oid] = max(float(object_scores.get(oid, 0.0)), float(row.get("object_score", 0.0)))
            mode = row.get("readout_mode")
            if mode == "snap_to_mask":
                bool_key = (scene_frame_key, int(row.get("mask_id", 0)))
                mask = mask_bool_cache.get(bool_key)
                if mask is None:
                    mask = info.label == int(row.get("mask_id", 0))
                    mask_bool_cache[bool_key] = mask
                pixel_collision_count += int(np.count_nonzero((pred > 0) & mask))
                pred[(pred == 0) & mask] = pred_id
            else:
                ys, xs = _selection_coords(row, info, args)
                if ys.size == 0:
                    continue
                occupied = pred[ys, xs] > 0
                pixel_collision_count += int(np.count_nonzero(occupied))
                keep = ~occupied
                pred[ys[keep], xs[keep]] = pred_id
        gt = _load_gt_2d(scene, frame_id, info.label.shape)
        gt = np.where(gt > 0, gt + int(scene_offsets.get(scene, 0)), 0).astype(np.int64, copy=False)
        acc.add(pred, gt)
        frame_rows.append(
            {
                "schema_version": "stream4d_v97_phase7_preview_frame_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "metric_name": "MV_AP_scene",
                "scene_id": scene,
                "window_id": "",
                "frame_id": int(frame_id),
                "status": "evaluated_scene_frame_dedup",
                "emitted_object_count": len(rows),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_eval": True,
                "uses_future": False,
            }
        )
    input_scores = np.ones((len(object_index),), dtype=np.float32)
    for oid, idx in object_index.items():
        input_scores[idx - 1] = float(object_scores.get(oid, 1.0))
    summary, iou, pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=int(args.min_pred_pixels),
        min_gt_pixels=int(args.min_gt_pixels),
        score_mode="input",
        input_scores=input_scores,
    )
    pred_best_by_object: dict[str, float] = {}
    if iou.shape[0] > 0:
        pred_best = np.max(iou, axis=1) if iou.shape[1] else np.zeros((iou.shape[0],), dtype=np.float32)
        for pid, best in zip(pred_ids, pred_best):
            oid = index_to_object.get(int(pid), "")
            if oid:
                pred_best_by_object[oid] = float(best)
    return summary, frame_rows, int(pixel_collision_count), int(missing_mask_raster_count), pred_best_by_object, int(mask_path_conflict_count)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Phase7 requested CUDA but torch.cuda.is_available() is false")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    phase0 = _read_json(_project(args.phase0_root) / "summary.json")
    b0_ap = float(phase0.get("B0_MV_AP_window", 0.0))
    b0_ap50 = float(phase0.get("B0_MV_AP50_window", 0.0))
    control_ap = float(phase0.get("best_control_MV_AP_window", 0.0))
    control_ap50 = float(phase0.get("best_control_MV_AP50_window", 0.0))
    required_ap = float(phase0.get("required_MV_AP_window", 0.0))
    required_ap50 = float(phase0.get("required_MV_AP50_window", 0.0))
    b0_scene_ap_raw = phase0.get("B0_MV_AP_scene", "")
    b0_scene_ap50_raw = phase0.get("B0_MV_AP50_scene", "")
    control_scene_ap_raw = phase0.get("best_control_MV_AP_scene", "")
    control_scene_ap50_raw = phase0.get("best_control_MV_AP50_scene", "")
    required_scene_ap_raw = phase0.get("required_MV_AP_scene", "")
    required_scene_ap50_raw = phase0.get("required_MV_AP50_scene", "")
    scene_comparator_available = all(
        value not in {"", None}
        for value in [b0_scene_ap_raw, b0_scene_ap50_raw, control_scene_ap_raw, control_scene_ap50_raw, required_scene_ap_raw, required_scene_ap50_raw]
    )
    b0_scene_ap = _num(b0_scene_ap_raw)
    b0_scene_ap50 = _num(b0_scene_ap50_raw)
    control_scene_ap = _num(control_scene_ap_raw)
    control_scene_ap50 = _num(control_scene_ap50_raw)
    required_scene_ap = _num(required_scene_ap_raw)
    required_scene_ap50 = _num(required_scene_ap50_raw)
    phase5_root = _project(args.phase5_root)
    phase6_root = _project(args.phase6_root)
    mask_lookup = _mask_path_lookup(_project(args.source_rows))
    object_meta = _load_object_scores(phase5_root, args.variant_id)
    support_rows, sparse = _load_sparse_support(phase6_root)
    if int(args.max_object_frames) > 0:
        support_rows = support_rows[: int(args.max_object_frames)]
    frame_cache: dict[tuple[str, str, int], FrameInfo] = {}
    support_iou_rows: list[dict[str, Any]] = []
    records: list[SupportRecord] = []
    missing_mask_raster_count = 0
    for row in support_rows:
        idx = int(row["object_frame_index"])
        scene = row.get("scene_id", "")
        window = row.get("window_id", "")
        frame_id = int(_num(row.get("frame_id")))
        frame_key = (scene, window, frame_id)
        try:
            info = _frame_info(frame_key, mask_lookup, frame_cache, device)
        except FileNotFoundError:
            missing_mask_raster_count += 1
            continue
        ys, xs, vals = sparse.get(idx, (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)))
        base_rows = _candidate_rows_for_support(info=info, ys=ys, xs=xs, source="base_sparse_support", top_k=int(args.top_k_candidates), device=device)
        sigma_y, sigma_x = _dilate_sparse(ys, xs, int(args.sigma_dilate_radius), info.label.shape[0], info.label.shape[1])
        sigma_rows = _candidate_rows_for_support(info=info, ys=sigma_y, xs=sigma_x, source="sigma_dilated_support", top_k=int(args.top_k_candidates), device=device)
        meta = object_meta.get(row.get("object_id", ""), {})
        object_score = float(meta.get("object_score", _num(row.get("support_peak"))))
        for source_rows in (base_rows, sigma_rows):
            for cand in source_rows:
                support_iou_rows.append(
                    {
                        "schema_version": "stream4d_v97_phase7_support_mask_iou_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "variant_id": args.variant_id,
                        "scene_id": scene,
                        "window_id": window,
                        "object_id": row.get("object_id", ""),
                        "frame_id": frame_id,
                        "object_frame_index": idx,
                        **cand,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        records.append(
            SupportRecord(
                index=idx,
                row=row,
                ys=ys,
                xs=xs,
                values=vals,
                object_score=object_score,
                base_best=base_rows[0],
                sigma_best=sigma_rows[0],
            )
        )
    variant_cfg = {
        "R0_snap_only": {"wta_duplicate_masks": False},
        "R1_carve_if_broad": {"wta_duplicate_masks": False},
        "R2_support_only_fallback": {"wta_duplicate_masks": False},
        "R3_snap_or_carve_gtfree_rule": {"wta_duplicate_masks": True},
        "R4_semantic_alignment_snap": {"wta_duplicate_masks": False},
        "R5_conflict_aware_snap": {"wta_duplicate_masks": True},
        "R6_uncertainty_sigma_snap": {"wta_duplicate_masks": True},
        "R7_hybrid_best_gtfree": {"wta_duplicate_masks": True},
    }
    eval_frame_keys = {(rec.row.get("scene_id", ""), rec.row.get("window_id", ""), int(_num(rec.row.get("frame_id")))) for rec in records}
    eval_scene_frame_keys = {(scene, frame_id) for scene, _window, frame_id in eval_frame_keys}
    all_selection_rows: list[dict[str, Any]] = []
    mv_object_frame_rows: list[dict[str, Any]] = []
    mv_object_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    preview_frame_rows: list[dict[str, Any]] = []
    for readout_variant, cfg in variant_cfg.items():
        selections = []
        for rec in records:
            frame_key = (rec.row.get("scene_id", ""), rec.row.get("window_id", ""), int(_num(rec.row.get("frame_id"))))
            selections.append(_select(rec, readout_variant, args, frame_cache[frame_key]))
        wta_drop_count = _apply_wta(selections, bool(cfg.get("wta_duplicate_masks", False)))
        all_selection_rows.extend(selections)
        emitted = [row for row in selections if row.get("emitted")]
        mode_counts = Counter(str(row.get("readout_mode", "no_emit")) for row in selections)
        duplicate_keys = [
            (row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["mask_id"]))
            for row in emitted
            if row.get("mask_source_type") in {"cropformer_mask", "cropformer_carved_mask"}
        ]
        same_frame_collision_count = sum(max(0, count - 1) for count in Counter(duplicate_keys).values())
        eval_summary, frame_rows, pixel_collision_count, eval_missing_count, pred_best_by_object = _evaluate_variant(
            variant_id=readout_variant,
            selections=selections,
            eval_frame_keys=eval_frame_keys,
            frame_cache=frame_cache,
            args=args,
        )
        for frame_row in frame_rows:
            frame_row["metric_name"] = "MV_AP_window"
        preview_frame_rows.extend(frame_rows)
        plain_summary, plain_frame_rows, plain_pixel_collision_count, plain_missing_count, plain_pred_best_by_object, plain_mask_conflict_count = _evaluate_variant_scene(
            variant_id=readout_variant,
            selections=selections,
            eval_scene_frame_keys=eval_scene_frame_keys,
            frame_cache=frame_cache,
            args=args,
        )
        preview_frame_rows.extend(plain_frame_rows)
        for row in emitted:
            mv_object_frame_rows.append(
                {
                    "schema_version": "stream4d_v97_phase7_mv_object_frame_mask_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": readout_variant,
                    "scene_id": row["scene_id"],
                    "window_id": row["window_id"],
                    "object_id": row["object_id"],
                    "frame_id": int(row["frame_id"]),
                    "mask_source_type": row["mask_source_type"],
                    "mask_id": int(row["mask_id"]),
                    "generated_mask_id": row["generated_mask_id"],
                    "readout_mode": row["readout_mode"],
                    "object_score": row["object_score"],
                    "support_iou": row["support_iou"],
                    "support_recall": row["support_recall"],
                    "mask_precision": row["mask_precision"],
                    "semantic_alignment": row["semantic_alignment"],
                    "broad_risk": row["broad_risk"],
                    "conflict_risk": row["conflict_risk"],
                    "support_area_ratio": row["support_area_ratio"],
                    "selected_mask_area_ratio": row["selected_mask_area_ratio"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        by_object = defaultdict(list)
        for row in emitted:
            by_object[str(row["object_id"])].append(row)
        for object_id in sorted({rec.row.get("object_id", "") for rec in records}):
            rows = by_object.get(object_id, [])
            meta = object_meta.get(object_id, {})
            mv_object_rows.append(
                {
                    "schema_version": "stream4d_v97_phase7_mv_object_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": readout_variant,
                    "object_id": object_id,
                    "object_score": float(meta.get("object_score", 0.0)),
                    "source_frame_count": sum(1 for rec in records if rec.row.get("object_id", "") == object_id),
                    "emitted_frame_count": len(rows),
                    "micro_primitive_count": meta.get("micro_primitive_count", ""),
                    "masklet_support_count": meta.get("masklet_support_count", ""),
                    "pred_best_iou_diagnostic": pred_best_by_object.get(object_id, ""),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        support_ious = [float(row["support_iou"]) for row in emitted]
        recalls = [float(row["support_recall"]) for row in emitted]
        precisions = [float(row["mask_precision"]) for row in emitted]
        broad_risks = [float(row["broad_risk"]) for row in emitted]
        conflicts = [float(row["conflict_risk"]) for row in emitted]
        score_list = []
        best_iou_list = []
        for object_id, best in pred_best_by_object.items():
            if object_id in object_meta:
                score_list.append(float(object_meta[object_id].get("object_score", 0.0)))
                best_iou_list.append(float(best))
        quality = {
            "schema_version": "stream4d_v97_phase7_readout_quality_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": readout_variant,
            "readout_mode_counts": dict(mode_counts),
            "support_to_selected_mask_IoU_mean": float(np.mean(support_ious)) if support_ious else 0.0,
            "support_recall_mean": float(np.mean(recalls)) if recalls else 0.0,
            "mask_precision_mean": float(np.mean(precisions)) if precisions else 0.0,
            "selected_mask_broad_risk_mean": float(np.mean(broad_risks)) if broad_risks else 0.0,
            "selected_mask_conflict_risk_mean": float(np.mean(conflicts)) if conflicts else 0.0,
            "wta_duplicate_drop_count": int(wta_drop_count),
            "pixel_collision_count": int(pixel_collision_count),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        quality_rows.append(quality)
        ap = eval_summary.get("ap")
        ap50 = eval_summary.get("ap50")
        ap25 = eval_summary.get("ap25")
        sf50 = (eval_summary.get("score_free_match_at_050") or {}).get("recall")
        sf25 = (eval_summary.get("score_free_match_at_025") or {}).get("recall")
        scene_ap = plain_summary.get("ap")
        scene_ap50 = plain_summary.get("ap50")
        scene_ap25 = plain_summary.get("ap25")
        scene_sf50 = (plain_summary.get("score_free_match_at_050") or {}).get("recall")
        scene_sf25 = (plain_summary.get("score_free_match_at_025") or {}).get("recall")
        progress_gate = (
            ap is not None
            and ap50 is not None
            and float(ap) >= b0_ap + 0.010
            and float(ap50) >= b0_ap50 + 0.020
            and float(ap) >= control_ap + 0.005
            and float(ap50) >= control_ap50 + 0.010
            and int(same_frame_collision_count) == 0
            and int(missing_mask_raster_count + eval_missing_count) == 0
        )
        strong_gate = (
            progress_gate
            and float(ap) >= required_ap
            and float(ap50) >= required_ap50
        )
        metric = {
            "schema_version": "stream4d_v97_phase7_variant_metric_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": readout_variant,
            "metric_scope": "segment_diagnostic",
            "MV_AP_window": ap,
            "MV_AP50_window": ap50,
            "MV_AP25_window": ap25,
            "MV_AP_scene": scene_ap,
            "MV_AP50_scene": scene_ap50,
            "MV_AP25_scene": scene_ap25,
            "ScoreFreeMatch50_window": sf50,
            "ScoreFreeMatch25_window": sf25,
            "ScoreFreeMatch50_scene": scene_sf50,
            "ScoreFreeMatch25_scene": scene_sf25,
            "object_count": len({rec.row.get("object_id", "") for rec in records}),
            "frame_mask_count": len(emitted),
            "same_frame_collision_count": int(same_frame_collision_count),
            "pixel_collision_count": int(pixel_collision_count),
            "scene_pixel_collision_count": int(plain_pixel_collision_count),
            "missing_mask_raster_count": int(missing_mask_raster_count + eval_missing_count),
            "scene_missing_mask_raster_count": int(plain_missing_count),
            "scene_mask_path_conflict_count": int(plain_mask_conflict_count),
            "scene_comparator_available": bool(scene_comparator_available),
            "support_to_selected_mask_IoU_mean": quality["support_to_selected_mask_IoU_mean"],
            "support_recall_mean": quality["support_recall_mean"],
            "mask_precision_mean": quality["mask_precision_mean"],
            "readout_mode_counts": json.dumps(dict(mode_counts), sort_keys=True),
            "selected_mask_broad_risk_mean": quality["selected_mask_broad_risk_mean"],
            "selected_mask_conflict_risk_mean": quality["selected_mask_conflict_risk_mean"],
            "object_score_vs_best_iou_spearman_diagnostic": _spearman(score_list, best_iou_list),
            "B0_MV_AP_window": b0_ap,
            "B0_MV_AP50_window": b0_ap50,
            "best_locked_control_MV_AP_window": control_ap,
            "best_locked_control_MV_AP50_window": control_ap50,
            "required_MV_AP_window": required_ap,
            "required_MV_AP50_window": required_ap50,
            "B0_MV_AP_scene": b0_scene_ap_raw,
            "B0_MV_AP50_scene": b0_scene_ap50_raw,
            "best_locked_control_MV_AP_scene": control_scene_ap_raw,
            "best_locked_control_MV_AP50_scene": control_scene_ap50_raw,
            "required_MV_AP_scene": required_scene_ap_raw,
            "required_MV_AP50_scene": required_scene_ap50_raw,
            "dev_progress_gate_pass": bool(progress_gate),
            "strong_dev_gate_pass": bool(strong_gate),
            "runtime_sec": "",
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
            "uses_future": False,
        }
        metric_rows.append(metric)
        gates = [
            ("MV_AP_window_ge_B0_plus_0p010", ap is not None and float(ap) >= b0_ap + 0.010, ap, b0_ap + 0.010),
            ("MV_AP50_window_ge_B0_plus_0p020", ap50 is not None and float(ap50) >= b0_ap50 + 0.020, ap50, b0_ap50 + 0.020),
            ("MV_AP_window_ge_best_control_plus_0p005", ap is not None and float(ap) >= control_ap + 0.005, ap, control_ap + 0.005),
            ("MV_AP50_window_ge_best_control_plus_0p010", ap50 is not None and float(ap50) >= control_ap50 + 0.010, ap50, control_ap50 + 0.010),
            ("MV_AP_scene_recorded", scene_ap is not None, scene_ap, "diagnostic local2history/scene metric; not Phase0 lock"),
            ("MV_AP50_scene_recorded", scene_ap50 is not None, scene_ap50, "diagnostic local2history/scene metric; not Phase0 lock"),
            ("same_frame_collision_count_eq_0", int(same_frame_collision_count) == 0, same_frame_collision_count, 0),
            ("scene_pixel_collision_count_eq_0_diagnostic", int(plain_pixel_collision_count) == 0, plain_pixel_collision_count, 0),
            ("scene_mask_path_conflict_count_eq_0_diagnostic", int(plain_mask_conflict_count) == 0, plain_mask_conflict_count, 0),
            ("missing_mask_raster_count_eq_0", int(missing_mask_raster_count + eval_missing_count) == 0, missing_mask_raster_count + eval_missing_count, 0),
            ("scene_missing_mask_raster_count_eq_0_diagnostic", int(plain_missing_count) == 0, plain_missing_count, 0),
            ("uses_gt_for_prediction_false", True, False, False),
            ("uses_future_false", True, False, False),
            ("strong_MV_AP_window_ge_required", ap is not None and float(ap) >= required_ap, ap, required_ap),
            ("strong_MV_AP50_window_ge_required", ap50 is not None and float(ap50) >= required_ap50, ap50, required_ap50),
            ("dev_progress_gate", bool(progress_gate), progress_gate, "all Phase7 dev progress criteria"),
            ("strong_dev_gate", bool(strong_gate), strong_gate, "dev progress and required MV_AP_window gates"),
        ]
        for gate, passed, observed, required in gates:
            gate_rows.append(
                {
                    "schema_version": "stream4d_v97_phase7_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": readout_variant,
                    "gate": gate,
                    "pass": bool(passed),
                    "observed": observed,
                    "required": required,
                    "metric_scope": "segment_diagnostic",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    runtime_sec = float(time.time() - started)
    for row in metric_rows:
        row["runtime_sec"] = runtime_sec
    best_variant = max(
        metric_rows,
        key=lambda row: (
            bool(row.get("strong_dev_gate_pass")),
            bool(row.get("dev_progress_gate_pass")),
            _num(row.get("MV_AP50_window")),
            _num(row.get("MV_AP_window")),
            _num(row.get("MV_AP50_scene")),
            _num(row.get("MV_AP_scene")),
            _num(row.get("ScoreFreeMatch50_window")),
            -int(_num(row.get("same_frame_collision_count"))),
        ),
        default={},
    )
    any_progress = any(_bool(row.get("dev_progress_gate_pass")) for row in metric_rows)
    any_strong = any(_bool(row.get("strong_dev_gate_pass")) for row in metric_rows)
    decision = "PASS_V97_PHASE7_STRONG_DIAGNOSTIC" if any_strong else ("PASS_V97_PHASE7_DEV_PROGRESS_DIAGNOSTIC" if any_progress else "NO_GO_V97_PHASE7_SUPPORT_IOU_READOUT")
    gpu_peak = float(torch.cuda.max_memory_allocated() / (1024.0**2)) if device.startswith("cuda") else 0.0
    _write_csv(output_root / "mv_object_rows.csv", mv_object_rows)
    _write_csv(output_root / "mv_object_frame_mask_rows.csv", mv_object_frame_rows)
    _write_csv(output_root / "mask_selection_rows.csv", all_selection_rows)
    _write_csv(output_root / "support_mask_iou_rows.csv", support_iou_rows)
    _write_csv(output_root / "readout_quality_rows.csv", quality_rows)
    _write_csv(output_root / "variant_metric_rows.csv", metric_rows)
    _write_csv(output_root / "variant_gate_rows.csv", gate_rows)
    _write_csv(output_root / "preview_frame_rows.csv", preview_frame_rows)
    summary = {
        "schema": "stream4d_v97_phase7_support_iou_readout_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "output_root": _rel(output_root),
        "phase5_root": _rel(phase5_root),
        "phase6_root": _rel(phase6_root),
        "source_rows": _rel(_project(args.source_rows)),
        "variant_id": args.variant_id,
        "metric_scope": "segment_diagnostic",
        "full_dev_gate_pass": False,
        "full_dev_gate_blocker": "Phase2/Phase6 frontier is segment_diagnostic, not full-dev decode scope",
        "readout_variant_count": len(variant_cfg),
        "object_frame_count": len(records),
        "eval_frame_count": len(eval_frame_keys),
        "eval_scene_frame_count": len(eval_scene_frame_keys),
        "missing_mask_raster_count": int(missing_mask_raster_count),
        "scene_metric_name": "MV_AP_scene",
        "scene_comparator_available": bool(scene_comparator_available),
        "scene_comparator_note": "" if scene_comparator_available else "Phase0 locks MV_AP_window; local2history/scene metric is recorded as MV_AP_scene diagnostic.",
        "semantic_dense_tensor_loaded": False,
        "semantic_alignment_policy": "neutral_no_dense_semantic_tensor; R4 is diagnostic only",
        "best_variant": best_variant,
        "metric_rows": metric_rows,
        "gate_rows": gate_rows,
        "can_enter_phase8_full": False,
        "can_enter_phase9_failure_decomposition": not any_strong,
        "runtime_sec": runtime_sec,
        "GPU_memory_peak_MB": gpu_peak,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": decision,
                "best_variant": best_variant.get("variant_id", ""),
                "best_MV_AP_window": best_variant.get("MV_AP_window", ""),
                "best_MV_AP50_window": best_variant.get("MV_AP50_window", ""),
                "best_MV_AP_scene": best_variant.get("MV_AP_scene", ""),
                "best_MV_AP50_scene": best_variant.get("MV_AP50_scene", ""),
                "metric_scope": "segment_diagnostic",
                "output_root": _rel(output_root),
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-root", default=str(DEFAULT_PHASE0))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5))
    parser.add_argument("--phase6-root", default=str(DEFAULT_PHASE6))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--variant-id", default="C0_cover_seed_plus_affinity_expand")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k-candidates", type=int, default=5)
    parser.add_argument("--max-object-frames", type=int, default=0)
    parser.add_argument("--min-pred-pixels", type=int, default=64)
    parser.add_argument("--min-gt-pixels", type=int, default=64)
    parser.add_argument("--snap-iou-threshold", type=float, default=0.08)
    parser.add_argument("--snap-recall-threshold", type=float, default=0.20)
    parser.add_argument("--snap-precision-threshold", type=float, default=0.02)
    parser.add_argument("--carve-recall-threshold", type=float, default=0.20)
    parser.add_argument("--broad-area-ratio-threshold", type=float, default=6.0)
    parser.add_argument("--broad-risk-threshold", type=float, default=0.030)
    parser.add_argument("--conflict-risk-threshold", type=float, default=0.75)
    parser.add_argument("--semantic-alignment-threshold", type=float, default=0.50)
    parser.add_argument("--sigma-dilate-radius", type=int, default=1)
    parser.add_argument("--carve-dilate-radius", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
