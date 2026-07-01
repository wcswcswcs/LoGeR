#!/usr/bin/env python3
"""Render v96 Phase5 object clusters by Triton splatting and mask snapping."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


PHASE_ID = "v96_phase6_render_snap"
RUN_ID = "v96_phase6_render_snap"
DEFAULT_PHASE5 = ROOT / "outputs/audit/v96_phase5_object_birth_w0020_segmented_r4_D3_repair5_overlap090"
DEFAULT_INCIDENCE = ROOT / "outputs/audit/v96_phase3_triton_incidence_w0020_segmented_r4_D3_repair1"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_repair5_overlap090"
B0_MV_AP_WINDOW = 0.023169647579624655
B0_MV_AP50_WINDOW = 0.07720796704691124


@triton.jit
def _splat_points_kernel(uv, heatmap, n_points: tl.constexpr, height: tl.constexpr, width: tl.constexpr, radius: tl.constexpr, sigma: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    side: tl.constexpr = radius * 2 + 1
    total: tl.constexpr = side * side
    point_idx = offsets // total
    off = offsets - point_idx * total
    valid = point_idx < n_points
    dx_i = off % side - radius
    dy_i = off // side - radius
    u = tl.load(uv + point_idx * 2, mask=valid, other=0.0)
    v = tl.load(uv + point_idx * 2 + 1, mask=valid, other=0.0)
    x = tl.cast(tl.floor(u + 0.5), tl.int32) + dx_i
    y = tl.cast(tl.floor(v + 0.5), tl.int32) + dy_i
    inside = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    dist2 = tl.cast(dx_i * dx_i + dy_i * dy_i, tl.float32)
    weight = tl.exp(-dist2 / (2.0 * sigma * sigma))
    tl.atomic_add(heatmap + y * width + x, weight, sem="relaxed", mask=inside)


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
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
        return {str(k): _jsonable(v) for k, v in value.items()}
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
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
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


def _load_selected_masklets(phase5_root: Path, family: str) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(phase5_root / "selected_masklet_rows.csv"):
        if row.get("family") == family:
            rows.append(row)
    if not rows:
        raise ValueError(f"no selected masklet rows for family={family} in {phase5_root}")
    return rows


def _load_incidence(root: Path, decode_variant: str) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(root / "incidence_event_rows.csv"):
        if row.get("decode_variant") == decode_variant:
            rows.append(row)
    return rows


def _cpu_splat(uv: np.ndarray, height: int, width: int, radius: int, sigma: float) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    for u, v in uv:
        cx = int(round(float(u)))
        cy = int(round(float(v)))
        for dy in range(-radius, radius + 1):
            y = cy + dy
            if y < 0 or y >= height:
                continue
            for dx in range(-radius, radius + 1):
                x = cx + dx
                if x < 0 or x >= width:
                    continue
                heatmap[y, x] += math.exp(-float(dx * dx + dy * dy) / (2.0 * float(sigma) * float(sigma)))
    return heatmap


def _triton_splat(uv_np: np.ndarray, height: int, width: int, radius: int, sigma: float, device: str) -> np.ndarray:
    if uv_np.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    uv_t = torch.as_tensor(uv_np.astype(np.float32, copy=False), device=device)
    heatmap = torch.zeros((height * width,), dtype=torch.float32, device=device)
    pixels_per_point = int((radius * 2 + 1) ** 2)
    block = 256
    grid = (triton.cdiv(int(uv_np.shape[0]) * pixels_per_point, block),)
    _splat_points_kernel[grid](uv_t, heatmap, int(uv_np.shape[0]), int(height), int(width), int(radius), float(sigma), block)
    return heatmap.reshape(height, width).detach().cpu().numpy()


def _support_iou(support: np.ndarray, selected_mask: np.ndarray) -> tuple[float, int, int, int]:
    inter = int(np.count_nonzero(support & selected_mask))
    support_area = int(np.count_nonzero(support))
    mask_area = int(np.count_nonzero(selected_mask))
    union = support_area + mask_area - inter
    return float(inter / union) if union > 0 else 0.0, inter, support_area, mask_area


def _best_support_mask(label: np.ndarray, support: np.ndarray) -> tuple[int, float, int, int, int]:
    support_area = int(np.count_nonzero(support))
    if support_area <= 0:
        return 0, 0.0, 0, 0, 0
    positive = label > 0
    if not np.any(positive):
        return 0, 0.0, 0, support_area, 0
    max_label = int(np.max(label))
    mask_areas = np.bincount(label[positive].astype(np.int64), minlength=max_label + 1)
    intersections = np.bincount(label[support & positive].astype(np.int64), minlength=max_label + 1)
    best_mask_id = 0
    best_iou = 0.0
    best_inter = 0
    best_area = 0
    for mask_id in np.flatnonzero(intersections):
        if mask_id <= 0:
            continue
        inter = int(intersections[mask_id])
        mask_area = int(mask_areas[mask_id])
        union = support_area + mask_area - inter
        iou = float(inter / union) if union > 0 else 0.0
        if (iou, inter, -mask_area, -mask_id) > (best_iou, best_inter, -best_area, -best_mask_id):
            best_mask_id = int(mask_id)
            best_iou = iou
            best_inter = inter
            best_area = mask_area
    return best_mask_id, best_iou, best_inter, support_area, best_area


def _candidate_tube_footprint(candidates: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, int, int], int], int]:
    area_by_key: dict[tuple[str, str, int, int], int] = {}
    for candidate in candidates:
        payload = candidate["payload"]
        key = (
            str(payload["scene_id"]),
            str(payload["window_id"]),
            int(payload["frame_id"]),
            int(candidate["emit_mask_id"]),
        )
        area = int(candidate.get("pred_pixel_count", 0))
        if area > area_by_key.get(key, 0):
            area_by_key[key] = area
    return area_by_key, int(sum(area_by_key.values()))


def _tube_jaccard(
    lhs: tuple[dict[tuple[str, str, int, int], int], int],
    rhs: tuple[dict[tuple[str, str, int, int], int], int],
) -> float:
    lhs_area, lhs_total = lhs
    rhs_area, rhs_total = rhs
    if lhs_total <= 0 or rhs_total <= 0:
        return 0.0
    if len(lhs_area) > len(rhs_area):
        lhs_area, rhs_area = rhs_area, lhs_area
    inter = 0
    for key, area in lhs_area.items():
        if key in rhs_area:
            inter += min(area, rhs_area[key])
    union = lhs_total + rhs_total - inter
    return float(inter / union) if union > 0 else 0.0


def _apply_object_tube_nms(
    candidates: list[dict[str, Any]],
    overlap_threshold: float,
) -> tuple[list[dict[str, Any]], int, int]:
    if not candidates:
        return [], 0, 0
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_object[str(candidate["payload"]["object_id"])].append(candidate)

    object_rows = []
    for object_id, rows in by_object.items():
        scores = [float(row["payload"].get("object_score", 0.0)) for row in rows]
        best_support = [float(row["payload"].get("best_support_iou", 0.0)) for row in rows]
        object_rows.append(
            {
                "object_id": object_id,
                "rows": rows,
                "score": float(max(scores)) if scores else 0.0,
                "frame_count": len(rows),
                "best_support_mean": float(np.mean(best_support)) if best_support else 0.0,
                "footprint": _candidate_tube_footprint(rows),
            }
        )

    kept: list[dict[str, Any]] = []
    suppressed_object_count = 0
    suppressed_frame_count = 0
    for row in sorted(
        object_rows,
        key=lambda item: (
            -float(item["score"]),
            -int(item["frame_count"]),
            -float(item["best_support_mean"]),
            str(item["object_id"]),
        ),
    ):
        max_overlap = 0.0
        for kept_row in kept:
            max_overlap = max(max_overlap, _tube_jaccard(row["footprint"], kept_row["footprint"]))
            if max_overlap >= overlap_threshold:
                break
        if max_overlap >= overlap_threshold:
            suppressed_object_count += 1
            suppressed_frame_count += len(row["rows"])
            continue
        kept.append(row)

    kept_ids = {str(row["object_id"]) for row in kept}
    filtered = [candidate for candidate in candidates if str(candidate["payload"]["object_id"]) in kept_ids]
    return filtered, int(suppressed_object_count), int(suppressed_frame_count)


def _score_object_payloads(
    payloads: list[dict[str, Any]],
    selected_by_object: dict[str, list[dict[str, str]]],
    object_qids: dict[str, set[str]],
    policy: str,
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, float]] = {}
    max_visible = 1.0
    max_qids = 1.0
    for oid, rows in selected_by_object.items():
        object_payloads = [row for row in payloads if row.get("object_id") == oid]
        masklet_scores = [_num(row.get("object_score")) for row in object_payloads]
        support_ious = [_num(row.get("support_iou")) for row in object_payloads]
        best_support_ious = [_num(row.get("best_support_iou")) for row in object_payloads]
        visible_counts = [_num(row.get("visible_micro_count")) for row in object_payloads]
        visible_mean = float(np.mean(visible_counts)) if visible_counts else 0.0
        qid_count = float(len(object_qids.get(oid, set())))
        max_visible = max(max_visible, visible_mean)
        max_qids = max(max_qids, qid_count)
        stats[oid] = {
            "selected_frame_count": float(len(rows)),
            "qid_count": qid_count,
            "masklet_score_mean": float(np.mean(masklet_scores)) if masklet_scores else 0.0,
            "masklet_score_max": float(np.max(masklet_scores)) if masklet_scores else 0.0,
            "support_iou_mean": float(np.mean(support_ious)) if support_ious else 0.0,
            "best_support_iou_mean": float(np.mean(best_support_ious)) if best_support_ious else 0.0,
            "best_support_iou_max": float(np.max(best_support_ious)) if best_support_ious else 0.0,
            "visible_micro_count_mean": visible_mean,
        }

    score_rows: list[dict[str, Any]] = []
    for oid, st in stats.items():
        frame_norm = min(1.0, st["selected_frame_count"] / 7.0)
        qid_norm = math.log1p(st["qid_count"]) / max(1e-6, math.log1p(max_qids))
        visible_norm = math.sqrt(st["visible_micro_count_mean"] / max(1.0, max_visible))
        support_mean = st["support_iou_mean"]
        best_support_mean = st["best_support_iou_mean"]
        best_support_max = st["best_support_iou_max"]
        masklet_mean = st["masklet_score_mean"]
        masklet_max = st["masklet_score_max"]
        if policy == "masklet_score":
            score = masklet_max
        elif policy == "frame_count":
            score = frame_norm
        elif policy == "frame_count_x_masklet":
            score = frame_norm * masklet_mean
        elif policy == "frame_count_x_support_iou":
            score = frame_norm * support_mean
        elif policy == "frame_visible_support":
            score = frame_norm * visible_norm * support_mean
        elif policy == "qid_frame_support":
            score = qid_norm * frame_norm * support_mean
        elif policy == "best_support_iou":
            score = best_support_max
        elif policy == "frame_count_x_best_support_iou":
            score = frame_norm * best_support_mean
        elif policy == "frame_count_x_masklet_x_best_support_iou":
            score = frame_norm * masklet_mean * best_support_mean
        else:
            raise ValueError(f"unknown score policy: {policy}")
        st["assigned_object_score"] = float(score)
        score_rows.append(
            {
                "schema_version": "stream4d_v96_object_score_policy_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "object_id": oid,
                "score_policy": policy,
                **st,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    for payload in payloads:
        oid = str(payload.get("object_id", ""))
        if policy == "masklet_score":
            payload["object_score"] = float(_num(payload.get("object_score")))
        else:
            payload["object_score"] = float(stats.get(oid, {}).get("assigned_object_score", 0.0))
        payload["score_policy"] = policy
    return score_rows


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _build_object_support(
    selected_rows: list[dict[str, str]],
    incidence_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str, int, int], set[str]], dict[str, set[str]], dict[tuple[str, str, int, str], list[tuple[float, float]]]]:
    node_to_qids: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    event_points: dict[tuple[str, str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for row in incidence_rows:
        scene = row.get("scene_id", "")
        window = row.get("window_id", "")
        frame_id = int(_num(row.get("target_frame_id")))
        mask_id = int(_num(row.get("center_mask_id")))
        qid = row.get("query_id", "")
        if mask_id > 0:
            node_to_qids[(scene, window, frame_id, mask_id)].add(qid)
        u = _num(row.get("u_tgt"), float("nan"))
        v = _num(row.get("v_tgt"), float("nan"))
        if math.isfinite(u) and math.isfinite(v):
            event_points[(scene, window, frame_id, qid)].append((float(u), float(v)))
    object_qids: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        key = (
            row.get("scene_id", ""),
            row.get("window_id", ""),
            int(_num(row.get("frame_id"))),
            int(_num(row.get("selected_mask_id"))),
        )
        object_qids[row.get("object_id", "")].update(node_to_qids.get(key, set()))
    return node_to_qids, object_qids, event_points


def _load_explicit_object_qids(phase5_root: Path, family: str) -> tuple[dict[str, set[str]], int]:
    path = phase5_root / "object_micro_query_rows.csv"
    if not path.exists():
        return {}, 0
    object_qids: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    for row in _read_csv(path):
        if row.get("family", family) != family:
            continue
        object_id = row.get("object_id", "")
        query_id = row.get("query_id", "")
        if not object_id or not query_id:
            continue
        object_qids[object_id].add(query_id)
        row_count += 1
    return object_qids, row_count


def _evaluate_variant(
    frame_predictions: dict[tuple[str, str, int], list[dict[str, Any]]],
    mask_lookup: dict[tuple[str, str, int], Path],
    *,
    variant: str,
    eval_frame_keys: set[tuple[str, str, int]],
    min_pred_pixels: int,
    min_gt_pixels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    object_index: dict[str, int] = {}
    object_scores: dict[str, float] = {}
    scene_gt_offsets = {scene: (idx + 1) * 1_000_000 for idx, scene in enumerate(sorted({key[0] for key in eval_frame_keys}))}
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    for key in sorted(set(eval_frame_keys) | set(frame_predictions)):
        rows = frame_predictions.get(key, [])
        scene, window, frame_id = key
        mask_path = mask_lookup.get(key)
        if mask_path is None or not mask_path.exists():
            frame_rows.append(
                {
                    "variant": variant,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "status": "missing_mask",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_eval": True,
                }
            )
            continue
        label = _load_label(mask_path)
        pred = np.zeros(label.shape, dtype=np.int64)
        for row in sorted(rows, key=lambda item: (-float(item.get("object_score", 0.0)), str(item.get("object_id", "")))):
            oid = str(row["object_id"])
            if oid not in object_index:
                object_index[oid] = len(object_index) + 1
                object_scores[oid] = 0.0
            object_scores[oid] = max(float(object_scores.get(oid, 0.0)), float(row.get("object_score", 0.0)))
            mask = np.asarray(row["pred_mask"], dtype=bool)
            pixel_collision_count += int(np.count_nonzero((pred > 0) & mask))
            pred[(pred == 0) & mask] = object_index[oid]
        gt = _load_gt_2d(scene, frame_id, label.shape)
        gt = np.where(gt > 0, gt + int(scene_gt_offsets.get(scene, 0)), 0).astype(np.int64, copy=False)
        acc.add(pred, gt)
        frame_rows.append(
            {
                "variant": variant,
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame_id,
                "status": "evaluated",
                "emitted_object_count": len(rows),
                "eval_frame_scope": "all_family_object_frame_keys_even_if_no_prediction",
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_eval": True,
                "gt_scene_offset": int(scene_gt_offsets.get(scene, 0)),
            }
        )
    input_scores = np.ones((len(object_index),), dtype=np.float32)
    for oid, idx in object_index.items():
        input_scores[int(idx) - 1] = float(object_scores.get(oid, 1.0))
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=min_pred_pixels,
        min_gt_pixels=min_gt_pixels,
        score_mode="input",
        input_scores=input_scores,
    )
    summary["input_score_count"] = int(input_scores.shape[0])
    summary["input_score_min"] = float(np.min(input_scores)) if input_scores.size else 0.0
    summary["input_score_max"] = float(np.max(input_scores)) if input_scores.size else 0.0
    return summary, frame_rows, pixel_collision_count


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase6 requires CUDA for Triton splatting; no CUDA device is available")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    phase5_root = _project(args.phase5_root)
    selected_rows = _load_selected_masklets(phase5_root, args.family)
    incidence_rows = _load_incidence(_project(args.incidence_root), args.decode_variant)
    mask_lookup = _mask_path_lookup(_project(args.source_rows))
    _node_to_qids, object_qids, event_points = _build_object_support(selected_rows, incidence_rows)
    explicit_object_qids, explicit_object_micro_query_count = _load_explicit_object_qids(phase5_root, args.family)
    object_qid_source = "selected_masklet_incidence_union"
    if explicit_object_qids:
        selected_object_ids = {row.get("object_id", "") for row in selected_rows}
        object_qids = {oid: qids for oid, qids in explicit_object_qids.items() if oid in selected_object_ids}
        object_qid_source = "explicit_object_micro_query_rows"

    label_cache: dict[tuple[str, str, int], np.ndarray] = {}
    rendered_support_rows: list[dict[str, Any]] = []
    support_iou_rows: list[dict[str, Any]] = []
    selected_frame_mask_rows: list[dict[str, Any]] = []
    mv_object_frame_mask_rows: list[dict[str, Any]] = []
    object_frame_payloads: list[dict[str, Any]] = []
    parity_row: dict[str, Any] | None = None

    selected_by_object = defaultdict(list)
    for row in selected_rows:
        selected_by_object[row.get("object_id", "")].append(row)

    for row in selected_rows:
        scene = row.get("scene_id", "")
        window = row.get("window_id", "")
        frame_id = int(_num(row.get("frame_id")))
        mask_id = int(_num(row.get("selected_mask_id")))
        object_id = row.get("object_id", "")
        frame_key = (scene, window, frame_id)
        mask_path = mask_lookup.get(frame_key)
        if mask_path is None or not mask_path.exists():
            rendered_support_rows.append(
                {
                    "schema_version": "stream4d_v96_rendered_support_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "object_id": object_id,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": frame_id,
                    "status": "missing_mask",
                    "missing_mask_raster": True,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue
        if frame_key not in label_cache:
            label_cache[frame_key] = _load_label(mask_path)
        label = label_cache[frame_key]
        uv: list[tuple[float, float]] = []
        for qid in object_qids.get(object_id, set()):
            uv.extend(event_points.get((scene, window, frame_id, qid), []))
        uv_np = np.asarray(uv, dtype=np.float32).reshape(-1, 2) if uv else np.zeros((0, 2), dtype=np.float32)
        heatmap = _triton_splat(uv_np, int(label.shape[0]), int(label.shape[1]), int(args.splat_radius), float(args.splat_sigma), device)
        if parity_row is None and uv_np.shape[0] > 0:
            sample_uv = uv_np[: int(args.parity_max_points)]
            cpu = _cpu_splat(sample_uv, int(label.shape[0]), int(label.shape[1]), int(args.splat_radius), float(args.splat_sigma))
            gpu = _triton_splat(sample_uv, int(label.shape[0]), int(label.shape[1]), int(args.splat_radius), float(args.splat_sigma), device)
            abs_err = np.abs(cpu - gpu)
            parity_row = {
                "schema_version": "stream4d_v96_triton_splat_parity_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame_id,
                "object_id": object_id,
                "sample_point_count": int(sample_uv.shape[0]),
                "cpu_vs_triton_abs_error_max": float(np.max(abs_err)) if abs_err.size else 0.0,
                "cpu_vs_triton_abs_error_mean": float(np.mean(abs_err)) if abs_err.size else 0.0,
                "cpu_vs_triton_positive_pixel_mismatch_rate": float(np.mean((cpu > 0) != (gpu > 0))) if abs_err.size else 0.0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        support = heatmap > float(args.support_threshold)
        if not np.any(support) and uv_np.shape[0] > 0:
            support = heatmap > 0.0
        support_dilated = _dilate(support, int(args.carve_dilate_radius))
        selected_mask = label == mask_id
        support_iou, support_inter, support_area, mask_area = _support_iou(support_dilated, selected_mask)
        best_mask_id, best_support_iou, best_support_inter, best_support_area, best_support_mask_area = _best_support_mask(label, support_dilated)
        best_support_mask = label == best_mask_id if best_mask_id > 0 else np.zeros(label.shape, dtype=bool)
        object_score = float(_num(row.get("masklet_score"), 0.0))
        rendered_support_rows.append(
            {
                "schema_version": "stream4d_v96_rendered_support_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "object_id": object_id,
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame_id,
                "visible_micro_count": int(uv_np.shape[0]),
                "support_area_pixels": int(support_area),
                "support_area_ratio": float(support_area / max(1, label.size)),
                "selected_mask_id": mask_id,
                "selected_mask_area_pixels": int(mask_area),
                "support_to_selected_mask_iou": support_iou,
                "best_support_mask_id": int(best_mask_id),
                "best_support_to_mask_iou": float(best_support_iou),
                "best_support_intersection_pixels": int(best_support_inter),
                "best_support_mask_area_pixels": int(best_support_mask_area),
                "status": "rendered",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        support_iou_rows.append(
            {
                "schema_version": "stream4d_v96_support_iou_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "object_id": object_id,
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame_id,
                "selected_mask_id": mask_id,
                "support_iou": support_iou,
                "support_intersection_pixels": support_inter,
                "support_area_pixels": support_area,
                "selected_mask_area_pixels": mask_area,
                "best_support_mask_id": int(best_mask_id),
                "best_support_iou": float(best_support_iou),
                "best_support_intersection_pixels": int(best_support_inter),
                "best_support_area_pixels": int(best_support_area),
                "best_support_mask_area_pixels": int(best_support_mask_area),
                "visible_micro_count": int(uv_np.shape[0]),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        object_frame_payloads.append(
            {
                "object_id": object_id,
                "scene_id": scene,
                "window_id": window,
                "frame_id": frame_id,
                "selected_mask_id": mask_id,
                "object_score": object_score,
                "selected_mask": selected_mask,
                "support": support,
                "support_dilated": support_dilated,
                "support_iou": support_iou,
                "best_support_mask_id": int(best_mask_id),
                "best_support_iou": float(best_support_iou),
                "best_support_mask": best_support_mask,
                "visible_micro_count": int(uv_np.shape[0]),
            }
        )

    object_score_policy_rows = _score_object_payloads(
        object_frame_payloads,
        selected_by_object,
        object_qids,
        str(args.score_policy),
    )
    object_score_by_id = {
        str(row["object_id"]): float(row.get("assigned_object_score", 0.0))
        for row in object_score_policy_rows
    }
    object_rank_by_id = {
        oid: rank
        for rank, oid in enumerate(
            sorted(
                object_score_by_id,
                key=lambda item: (
                    -float(object_score_by_id.get(item, 0.0)),
                    -len(selected_by_object.get(item, [])),
                    -len(object_qids.get(item, set())),
                    item,
                ),
            ),
            start=1,
        )
    }

    eval_frame_keys = {
        (str(payload["scene_id"]), str(payload["window_id"]), int(payload["frame_id"]))
        for payload in object_frame_payloads
    }
    variants = {
        "R0_support_only": {"mode": "support", "snap_threshold": 1.1},
        "R1_snap_to_mask": {"mode": "snap", "snap_threshold": 0.0},
        "R2_carved_mask": {"mode": "carve", "snap_threshold": 1.1},
        "R3_adaptive_snap_carve_fallback": {"mode": "adaptive", "snap_threshold": float(args.snap_iou_threshold)},
        "R4_uncertainty_radius": {"mode": "adaptive", "snap_threshold": max(0.0, float(args.snap_iou_threshold) - 0.02)},
        "R5_support_semantic_conflict_proxy": {"mode": "adaptive", "snap_threshold": float(args.snap_iou_threshold) + 0.03},
        "R6_snap_min3_frames": {"mode": "snap", "snap_threshold": 0.0, "min_object_frames": 3},
        "R7_snap_min5_frames": {"mode": "snap", "snap_threshold": 0.0, "min_object_frames": 5},
        "R8_snap_min3_qid500": {"mode": "snap", "snap_threshold": 0.0, "min_object_frames": 3, "min_object_qids": 500},
        "R9_snap_min6_frames": {"mode": "snap", "snap_threshold": 0.0, "min_object_frames": 6},
        "R10_snap_min7_frames": {"mode": "snap", "snap_threshold": 0.0, "min_object_frames": 7},
        "R11_snap_top16_score": {"mode": "snap", "snap_threshold": 0.0, "max_object_rank": 16},
        "R12_snap_top32_score": {"mode": "snap", "snap_threshold": 0.0, "max_object_rank": 32},
        "R13_snap_top64_score": {"mode": "snap", "snap_threshold": 0.0, "max_object_rank": 64},
        "R14_snap_top96_score": {"mode": "snap", "snap_threshold": 0.0, "max_object_rank": 96},
        "R15_snap_top128_score": {"mode": "snap", "snap_threshold": 0.0, "max_object_rank": 128},
        "R16_snap_support_iou_ge_0p10": {"mode": "snap", "snap_threshold": 0.0, "min_support_iou": 0.10},
        "R17_snap_support_iou_ge_0p20": {"mode": "snap", "snap_threshold": 0.0, "min_support_iou": 0.20},
        "R18_snap_support_iou_ge_0p30": {"mode": "snap", "snap_threshold": 0.0, "min_support_iou": 0.30},
        "R19_snap_support_iou_ge_0p40": {"mode": "snap", "snap_threshold": 0.0, "min_support_iou": 0.40},
        "R20_snap_support_iou_ge_0p50": {"mode": "snap", "snap_threshold": 0.0, "min_support_iou": 0.50},
        "R21_snap_best_support_iou_mask": {"mode": "best_snap", "min_best_support_iou": 0.0},
        "R22_snap_best_support_iou_ge_0p05": {"mode": "best_snap", "min_best_support_iou": 0.05},
        "R23_snap_best_support_iou_ge_0p10": {"mode": "best_snap", "min_best_support_iou": 0.10},
        "R24_snap_best_support_iou_ge_0p20": {"mode": "best_snap", "min_best_support_iou": 0.20},
        "R25_snap_best_support_iou_mask_wta": {"mode": "best_snap", "min_best_support_iou": 0.0, "wta_duplicate_masks": True},
        "R26_snap_best_support_iou_ge_0p20_wta": {"mode": "best_snap", "min_best_support_iou": 0.20, "wta_duplicate_masks": True},
        "R27_snap_best_support_iou_wta_objnms_0p15": {"mode": "best_snap", "min_best_support_iou": 0.0, "wta_duplicate_masks": True, "object_nms_overlap_threshold": 0.15},
        "R28_snap_best_support_iou_wta_objnms_0p30": {"mode": "best_snap", "min_best_support_iou": 0.0, "wta_duplicate_masks": True, "object_nms_overlap_threshold": 0.30},
        "R29_snap_best_support_iou_wta_objnms_0p50": {"mode": "best_snap", "min_best_support_iou": 0.0, "wta_duplicate_masks": True, "object_nms_overlap_threshold": 0.50},
        "R30_snap_best_support_iou_ge_0p05_wta": {"mode": "best_snap", "min_best_support_iou": 0.05, "wta_duplicate_masks": True},
        "R31_snap_best_support_iou_ge_0p10_wta": {"mode": "best_snap", "min_best_support_iou": 0.10, "wta_duplicate_masks": True},
        "R32_snap_best_support_iou_ge_0p05_wta_objnms_0p15": {"mode": "best_snap", "min_best_support_iou": 0.05, "wta_duplicate_masks": True, "object_nms_overlap_threshold": 0.15},
    }
    metric_rows: list[dict[str, Any]] = []
    preview_frame_rows: list[dict[str, Any]] = []
    for variant, cfg in variants.items():
        frame_predictions: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        no_emit_count = 0
        snap_count = 0
        carve_count = 0
        support_only_count = 0
        wta_duplicate_drop_count = 0
        object_nms_suppressed_object_count = 0
        object_nms_suppressed_frame_count = 0
        variant_emit_candidates: list[dict[str, Any]] = []

        def append_emission(candidate: dict[str, Any]) -> None:
            payload = candidate["payload"]
            pred_mask = candidate["pred_mask"]
            emit_mask_id = int(candidate["emit_mask_id"])
            policy = str(candidate["policy"])
            object_rank = int(candidate["object_rank"])
            frame_key = (payload["scene_id"], payload["window_id"], int(payload["frame_id"]))
            frame_predictions[frame_key].append(
                {
                    **payload,
                    "pred_mask": pred_mask,
                    "variant": variant,
                    "source_selected_mask_id": payload["selected_mask_id"],
                    "selected_mask_id": emit_mask_id,
                }
            )
            selected_frame_mask_rows.append(
                {
                    "schema_version": "stream4d_v96_selected_frame_mask_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "readout_variant": variant,
                    "object_id": payload["object_id"],
                    "scene_id": payload["scene_id"],
                    "window_id": payload["window_id"],
                    "frame_id": payload["frame_id"],
                    "source_selected_mask_id": payload["selected_mask_id"],
                    "selected_mask_id": emit_mask_id,
                    "readout_policy": policy,
                    "score_policy": args.score_policy,
                    "object_score": payload["object_score"],
                    "object_rank": object_rank,
                    "support_iou": payload["support_iou"],
                    "best_support_iou": payload["best_support_iou"],
                    "visible_micro_count": payload["visible_micro_count"],
                    "emitted": True,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            mv_object_frame_mask_rows.append(
                {
                    "schema_version": "stream4d_v96_mv_object_frame_mask_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "readout_variant": variant,
                    "object_id": payload["object_id"],
                    "scene_id": payload["scene_id"],
                    "window_id": payload["window_id"],
                    "frame_id": payload["frame_id"],
                    "source_selected_mask_id": payload["selected_mask_id"],
                    "selected_mask_id": emit_mask_id,
                    "pred_pixel_count": int(candidate["pred_pixel_count"]),
                    "readout_policy": policy,
                    "score_policy": args.score_policy,
                    "object_score": payload["object_score"],
                    "object_rank": object_rank,
                    "support_iou": payload["support_iou"],
                    "best_support_iou": payload["best_support_iou"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

        for payload in object_frame_payloads:
            object_rank = int(object_rank_by_id.get(str(payload["object_id"]), 10**9))
            if object_rank > int(cfg.get("max_object_rank", 10**9)):
                no_emit_count += 1
                continue
            if len(selected_by_object.get(payload["object_id"], [])) < int(cfg.get("min_object_frames", 1)):
                no_emit_count += 1
                continue
            if len(object_qids.get(payload["object_id"], set())) < int(cfg.get("min_object_qids", 0)):
                no_emit_count += 1
                continue
            if float(payload.get("support_iou", 0.0)) < float(cfg.get("min_support_iou", 0.0)):
                no_emit_count += 1
                continue
            mode = cfg["mode"]
            support_mask = payload["support_dilated"] if variant == "R4_uncertainty_radius" else payload["support"]
            selected_mask = payload["selected_mask"]
            emit_mask_id = int(payload["selected_mask_id"])
            pred_mask: np.ndarray
            policy = mode
            if mode == "support":
                pred_mask = support_mask
                support_only_count += 1
            elif mode == "snap":
                pred_mask = selected_mask
                snap_count += 1
            elif mode == "best_snap":
                if int(payload.get("best_support_mask_id", 0)) <= 0 or float(payload.get("best_support_iou", 0.0)) < float(cfg.get("min_best_support_iou", 0.0)):
                    no_emit_count += 1
                    continue
                pred_mask = payload["best_support_mask"]
                emit_mask_id = int(payload.get("best_support_mask_id", 0))
                policy = "best_support_iou_snap"
                snap_count += 1
            elif mode == "carve":
                pred_mask = selected_mask & payload["support_dilated"]
                carve_count += 1
            else:
                if payload["support_iou"] >= float(cfg["snap_threshold"]):
                    pred_mask = selected_mask
                    policy = "snap"
                    snap_count += 1
                else:
                    carved = selected_mask & payload["support_dilated"]
                    if np.count_nonzero(carved) >= int(args.min_pred_pixels):
                        pred_mask = carved
                        policy = "carve"
                        carve_count += 1
                    else:
                        pred_mask = support_mask
                        policy = "support_fallback"
                        support_only_count += 1
            pred_pixel_count = int(np.count_nonzero(pred_mask))
            if pred_pixel_count < int(args.min_pred_pixels):
                no_emit_count += 1
                continue
            candidate = {
                "payload": payload,
                "pred_mask": pred_mask,
                "emit_mask_id": emit_mask_id,
                "policy": policy,
                "object_rank": object_rank,
                "pred_pixel_count": pred_pixel_count,
            }
            if cfg.get("wta_duplicate_masks"):
                variant_emit_candidates.append(candidate)
            else:
                append_emission(candidate)
        if cfg.get("wta_duplicate_masks"):
            if "object_nms_overlap_threshold" in cfg:
                variant_emit_candidates, object_nms_suppressed_object_count, object_nms_suppressed_frame_count = _apply_object_tube_nms(
                    variant_emit_candidates,
                    float(cfg["object_nms_overlap_threshold"]),
                )
                no_emit_count += object_nms_suppressed_frame_count
            best_by_mask: dict[tuple[str, str, int, int], dict[str, Any]] = {}
            for candidate in variant_emit_candidates:
                payload = candidate["payload"]
                key = (
                    str(payload["scene_id"]),
                    str(payload["window_id"]),
                    int(payload["frame_id"]),
                    int(candidate["emit_mask_id"]),
                )
                current = best_by_mask.get(key)
                if current is None or (
                    float(payload.get("object_score", 0.0)),
                    -int(candidate["object_rank"]),
                    str(payload.get("object_id", "")),
                ) > (
                    float(current["payload"].get("object_score", 0.0)),
                    -int(current["object_rank"]),
                    str(current["payload"].get("object_id", "")),
                ):
                    best_by_mask[key] = candidate
            wta_duplicate_drop_count = len(variant_emit_candidates) - len(best_by_mask)
            no_emit_count += wta_duplicate_drop_count
            for candidate in best_by_mask.values():
                append_emission(candidate)
        eval_summary, frame_rows, pixel_collision_count = _evaluate_variant(
            frame_predictions,
            mask_lookup,
            variant=variant,
            eval_frame_keys=eval_frame_keys,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
        )
        preview_frame_rows.extend(frame_rows)
        emitted_keys = [
            (row["scene_id"], row["window_id"], int(row["frame_id"]), int(row["selected_mask_id"]))
            for rows in frame_predictions.values()
            for row in rows
        ]
        duplicate_mask_count = sum(max(0, count - 1) for count in Counter(emitted_keys).values())
        missing_mask_count = sum(1 for row in rendered_support_rows if _bool(row.get("missing_mask_raster", False)))
        ap = eval_summary.get("ap")
        ap50 = eval_summary.get("ap50")
        sf50 = (eval_summary.get("score_free_match_at_050") or {}).get("recall")
        emitted_scores = [float(row.get("object_score", 0.0)) for rows in frame_predictions.values() for row in rows]
        phase6_pass = (
            duplicate_mask_count == 0
            and missing_mask_count == 0
            and ap is not None
            and ap50 is not None
            and float(ap) >= B0_MV_AP_WINDOW + 0.010
            and float(ap50) >= B0_MV_AP50_WINDOW + 0.020
        )
        metric_rows.append(
            {
                "schema_version": "stream4d_v96_render_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "readout_variant": variant,
                "score_policy": args.score_policy,
                "max_object_rank": cfg.get("max_object_rank", ""),
                "min_support_iou": cfg.get("min_support_iou", ""),
                "min_best_support_iou": cfg.get("min_best_support_iou", ""),
                "wta_duplicate_masks": cfg.get("wta_duplicate_masks", ""),
                "wta_duplicate_drop_count": int(wta_duplicate_drop_count),
                "object_nms_overlap_threshold": cfg.get("object_nms_overlap_threshold", ""),
                "object_nms_suppressed_object_count": int(object_nms_suppressed_object_count),
                "object_nms_suppressed_frame_count": int(object_nms_suppressed_frame_count),
                "object_count": len({payload["object_id"] for payload in object_frame_payloads}),
                "emitted_object_count": len({row["object_id"] for rows in frame_predictions.values() for row in rows}),
                "emitted_object_frame_count": sum(len(rows) for rows in frame_predictions.values()),
                "visible_micro_count_per_object_frame_mean": float(np.mean([row["visible_micro_count"] for row in rendered_support_rows if row.get("status") == "rendered"])) if rendered_support_rows else 0.0,
                "support_to_selected_mask_IoU_mean": float(np.mean([_num(row.get("support_iou")) for row in support_iou_rows])) if support_iou_rows else 0.0,
                "snap_rate": float(snap_count / max(1, len(object_frame_payloads))),
                "carve_rate": float(carve_count / max(1, len(object_frame_payloads))),
                "support_only_rate": float(support_only_count / max(1, len(object_frame_payloads))),
                "no_emit_rate": float(no_emit_count / max(1, len(object_frame_payloads))),
                "same_frame_collision_count": int(duplicate_mask_count),
                "pixel_collision_count": int(pixel_collision_count),
                "missing_mask_raster_count": int(missing_mask_count),
                "MV_AP_window": ap,
                "MV_AP50_window": ap50,
                "MV_AP25_window": eval_summary.get("ap25"),
                "ScoreFreeMatch50_window": sf50,
                "ScoreFreeMatch25_window": (eval_summary.get("score_free_match_at_025") or {}).get("recall"),
                "B0_MV_AP_window": B0_MV_AP_WINDOW,
                "B0_MV_AP50_window": B0_MV_AP50_WINDOW,
                "object_score_unique_count": len({round(score, 12) for score in emitted_scores}),
                "object_score_min": float(np.min(emitted_scores)) if emitted_scores else 0.0,
                "object_score_max": float(np.max(emitted_scores)) if emitted_scores else 0.0,
                "phase6_gate_pass": bool(phase6_pass),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )

    mv_object_rows = []
    for object_id, rows in selected_by_object.items():
        mv_object_rows.append(
            {
                "schema_version": "stream4d_v96_mv_object_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "object_id": object_id,
                "family": args.family,
                "selected_frame_count": len(rows),
                "object_qid_count": len(object_qids.get(object_id, set())),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    gate_rows = []
    for row in metric_rows:
        variant = row["readout_variant"]
        gate_rows.extend(
            [
                {
                    "schema_version": "stream4d_v96_phase6_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "readout_variant": variant,
                    "gate": "same_frame_collision_count_eq_0",
                    "pass": int(row["same_frame_collision_count"]) == 0,
                    "observed": row["same_frame_collision_count"],
                    "required": 0,
                },
                {
                    "schema_version": "stream4d_v96_phase6_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "readout_variant": variant,
                    "gate": "missing_mask_raster_count_eq_0",
                    "pass": int(row["missing_mask_raster_count"]) == 0,
                    "observed": row["missing_mask_raster_count"],
                    "required": 0,
                },
                {
                    "schema_version": "stream4d_v96_phase6_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "readout_variant": variant,
                    "gate": "MV_AP_window_ge_B0_plus_0p010",
                    "pass": row["MV_AP_window"] is not None and float(row["MV_AP_window"]) >= B0_MV_AP_WINDOW + 0.010,
                    "observed": row["MV_AP_window"],
                    "required": B0_MV_AP_WINDOW + 0.010,
                },
                {
                    "schema_version": "stream4d_v96_phase6_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "readout_variant": variant,
                    "gate": "MV_AP50_window_ge_B0_plus_0p020",
                    "pass": row["MV_AP50_window"] is not None and float(row["MV_AP50_window"]) >= B0_MV_AP50_WINDOW + 0.020,
                    "observed": row["MV_AP50_window"],
                    "required": B0_MV_AP50_WINDOW + 0.020,
                },
                {
                    "schema_version": "stream4d_v96_phase6_gate_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "readout_variant": variant,
                    "gate": "variant_gate_to_controls",
                    "pass": bool(row["phase6_gate_pass"]),
                    "observed": row["phase6_gate_pass"],
                    "required": "all Phase6 pass criteria",
                },
            ]
        )
    best_variant = max(
        metric_rows,
        key=lambda row: (
            _num(row.get("MV_AP50_window")),
            _num(row.get("MV_AP_window")),
            _num(row.get("ScoreFreeMatch50_window")),
            -int(_num(row.get("same_frame_collision_count"))),
            -int(_num(row.get("missing_mask_raster_count"))),
            -int(_num(row.get("pixel_collision_count"))),
        ),
        default={},
    )
    phase6_pass = any(_bool(row.get("phase6_gate_pass")) for row in metric_rows)
    triton_peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    _write_csv(output_root / "rendered_support_rows.csv", rendered_support_rows)
    _write_csv(output_root / "support_iou_rows.csv", support_iou_rows)
    _write_csv(output_root / "selected_frame_mask_rows.csv", selected_frame_mask_rows)
    _write_csv(output_root / "mv_object_rows.csv", mv_object_rows)
    _write_csv(output_root / "mv_object_frame_mask_rows.csv", mv_object_frame_mask_rows)
    _write_csv(output_root / "object_score_policy_rows.csv", object_score_policy_rows)
    _write_csv(output_root / "render_variant_metric_rows.csv", metric_rows)
    _write_csv(output_root / "phase6_gate_rows.csv", gate_rows)
    _write_csv(output_root / "preview_frame_rows.csv", preview_frame_rows)
    _write_csv(output_root / "triton_splat_parity_rows.csv", [parity_row] if parity_row else [])
    summary = {
        "schema": "stream4d_v96_phase6_render_snap_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V96_PHASE6_RENDER_SNAP" if phase6_pass else "NO_GO_V96_PHASE6_RENDER_SNAP",
        "output_root": _rel(output_root),
        "phase5_root": _rel(phase5_root),
        "incidence_root": _rel(_project(args.incidence_root)),
        "family": args.family,
        "score_policy": args.score_policy,
        "selected_masklet_count": len(selected_rows),
        "object_count": len(selected_by_object),
        "object_qid_source": object_qid_source,
        "explicit_object_micro_query_count": int(explicit_object_micro_query_count),
        "rendered_object_frame_count": len(object_frame_payloads),
        "eval_frame_count": len(eval_frame_keys),
        "eval_frame_scope": "all family object-frame keys; variants with no prediction on a frame still evaluate GT",
        "best_variant": best_variant,
        "metric_rows": metric_rows,
        "gate_rows": gate_rows,
        "triton_splat_parity": parity_row,
        "GPU_memory_peak_MB": triton_peak_mb,
        "runtime_total_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "best_variant": best_variant.get("readout_variant", ""), "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render v96 Phase5 clusters with Triton splatting and mask snapping.")
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5))
    parser.add_argument("--incidence-root", default=str(DEFAULT_INCIDENCE))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--family", default="C_hybrid_cover_cluster")
    parser.add_argument("--decode-variant", default="D3_adaptive1024")
    parser.add_argument("--splat-radius", type=int, default=3)
    parser.add_argument("--splat-sigma", type=float, default=1.5)
    parser.add_argument("--support-threshold", type=float, default=0.05)
    parser.add_argument("--snap-iou-threshold", type=float, default=0.08)
    parser.add_argument("--carve-dilate-radius", type=int, default=2)
    parser.add_argument(
        "--score-policy",
        default="masklet_score",
        choices=[
            "masklet_score",
            "frame_count",
            "frame_count_x_masklet",
            "frame_count_x_support_iou",
            "frame_visible_support",
            "qid_frame_support",
            "best_support_iou",
            "frame_count_x_best_support_iou",
            "frame_count_x_masklet_x_best_support_iou",
        ],
    )
    parser.add_argument("--min-pred-pixels", type=int, default=64)
    parser.add_argument("--min-gt-pixels", type=int, default=64)
    parser.add_argument("--parity-max-points", type=int, default=1024)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
