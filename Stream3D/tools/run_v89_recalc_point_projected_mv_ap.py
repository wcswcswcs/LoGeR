from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v65_visualization_export import _load_scene_mesh  # noqa: E402
from tools.run_v65_scene_multiview_ap import (  # noqa: E402
    SparseSceneIoU,
    _load_gt_2d,
    _load_or_build_vertex_map,
    _read_label_png,
    _sha256,
    _summarize_iou,
    _top_iou_rows,
)


DEFAULT_OURS_VARIANTS = [
    "B0_local_only",
    "B1_M10_state_priority",
    "B2_DV5_confirmed_object_gain",
    "B3_history_with_local_fallback",
    "B4_state_priority_with_local_fallback",
    "B5_carrier_gated_frame_mask_readout",
    "B6_area_penalized_history_readout",
]
DEFAULT_CONTROL_VARIANTS = [
    "C0_semantic_only_control",
    "C1_shuffled_history_control",
    "C2_stale_history_control",
    "C3_size_matched_hash_control",
    "C4_single_largest_by_scene_control",
    "C5_local_only_area_rank_control",
]


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == ROOT.name:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: str | Path) -> str:
    p = _project(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    p = _project(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    p = _project(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = _project(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _frame_scope() -> dict[tuple[str, str], list[int]]:
    summary = _read_json(ROOT / "outputs/audit/v88_phase3_mv_ap_eval/mv_eval_summary.json")
    out: dict[tuple[str, str], list[int]] = {}
    for key, values in summary.get("frame_scope", {}).items():
        split, scene = str(key).split(":", 1)
        out[(split, scene)] = [int(v) for v in values]
    return out


def _mask_dir(scene: str) -> Path:
    candidates = [
        ROOT / "outputs/cache/v66_cropformer_chunk_masks" / scene / "stride_5/cropformer_conf_0p500/mask2former_hornet_3x/final_processed" / scene / "output_Cropformer/mask",
        ROOT / "outputs/cache/v65_cropformer_chunk_masks" / scene / "stride_5/cropformer_conf_0p500/mask2former_hornet_3x/final_processed" / scene / "output_Cropformer/mask",
        ROOT / "data/scannet/processed" / scene / "output_Cropformer/mask",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _raw_local_dir(local_export_root: Path, scene: str) -> Path:
    candidates = [
        local_export_root / f"raw_{scene}_dev",
        local_export_root / f"raw_{scene.replace('_00', '')}_dev",
    ]
    for path in candidates:
        if (path / "stream3d_local_object_rows.csv").exists():
            return path
    raise FileNotFoundError(f"missing raw local export for {scene} under {local_export_root}")


def _local_object_rows(local_export_root: Path, scene: str, source_step: str, window_index: int | None = None) -> tuple[Path, list[dict[str, Any]]]:
    raw_dir = _raw_local_dir(local_export_root, scene)
    object_rows = [row for row in _read_csv(raw_dir / "stream3d_local_object_rows.csv") if row.get("baseline_name") == source_step]
    if window_index is not None:
        object_rows = [row for row in object_rows if _int(row.get("window_index"), -1) == int(window_index)]
    if not object_rows:
        raise RuntimeError(f"no object rows for {scene} {source_step} window={window_index} in {raw_dir}")
    if not all(str(row.get("point_npz_path", "")).strip() for row in object_rows):
        raise RuntimeError(f"object rows lack point_npz_path; rerun Stream3D local export after point-id exporter patch: {raw_dir}")
    return raw_dir, object_rows


def _load_point_objects(
    local_export_root: Path,
    scene: str,
    source_step: str,
    vertex_count: int,
    window_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[int, str]]:
    raw_dir, object_rows = _local_object_rows(local_export_root, scene, source_step, window_index=window_index)
    object_ids = sorted({str(row["stream3d_local_object_id"]) for row in object_rows})
    object_to_idx = {obj: idx + 1 for idx, obj in enumerate(object_ids)}
    idx_to_obj = {idx: obj for obj, idx in object_to_idx.items()}
    scores = np.ones((len(object_ids),), dtype=np.float32)
    point_owner = np.zeros((vertex_count,), dtype=np.int32)
    point_owner_score = np.full((vertex_count,), -np.inf, dtype=np.float32)
    invalid_point_ids = 0
    assigned_points = 0
    overwritten_points = 0
    npz_cache: dict[Path, Any] = {}

    for row in object_rows:
        obj = str(row["stream3d_local_object_id"])
        idx = object_to_idx[obj]
        score = _num(row.get("object_score"), _num(row.get("point_count"), 1.0))
        scores[idx - 1] = float(score)
        npz_path = _project(str(row.get("point_npz_path", "")))
        if not npz_path.exists():
            raise FileNotFoundError(f"missing point npz: {npz_path}")
        if npz_path not in npz_cache:
            npz_cache[npz_path] = np.load(npz_path, allow_pickle=False)
        payload = npz_cache[npz_path]
        start = _int(row.get("point_slice_start"), -1)
        end = _int(row.get("point_slice_end"), -1)
        if start < 0 or end < start:
            raise ValueError(f"bad point slice for {obj}: {start}:{end}")
        point_ids = np.asarray(payload["point_ids"][start:end], dtype=np.int64)
        valid = (point_ids >= 0) & (point_ids < vertex_count)
        invalid_point_ids += int(point_ids.shape[0] - np.count_nonzero(valid))
        point_ids = point_ids[valid]
        if point_ids.size == 0:
            continue
        old_positive = point_owner[point_ids] > 0
        update = np.asarray(score >= point_owner_score[point_ids])
        overwritten_points += int(np.count_nonzero(old_positive & update))
        if np.any(update):
            chosen = point_ids[update]
            point_owner[chosen] = idx
            point_owner_score[chosen] = float(score)
            assigned_points += int(chosen.shape[0])

    for payload in npz_cache.values():
        try:
            payload.close()
        except Exception:
            pass
    diag = {
        "raw_dir": _rel(raw_dir),
        "source_step": source_step,
        "window_index": "" if window_index is None else int(window_index),
        "object_count": len(object_ids),
        "point_npz_file_count": len(npz_cache),
        "vertex_count": int(vertex_count),
        "assigned_unique_vertex_count": int(np.count_nonzero(point_owner > 0)),
        "assigned_point_write_count": int(assigned_points),
        "overwritten_point_write_count": int(overwritten_points),
        "invalid_point_id_count": int(invalid_point_ids),
        "materialization": "local_3d_object_point_ids_projected_to_2d_via_v65_depth_to_mesh_vertex_map",
    }
    return point_owner, scores, diag, idx_to_obj


def _local_window_frame_scope(frame_ids: list[int], window_indices: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    previous_end = -1
    last_valid = len(frame_ids) - 1
    for window_index in sorted({int(value) for value in window_indices}):
        end = min(max(int(window_index), 0), last_valid)
        start = min(max(previous_end + 1, 0), len(frame_ids))
        out[int(window_index)] = list(frame_ids[start : end + 1]) if end >= start else []
        previous_end = max(previous_end, end)
    return out


def _window_scoped_gt(gt: np.ndarray, window_index: int, gt_id_map: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.zeros(gt.shape, dtype=np.int64)
    for gt_id in np.unique(gt):
        gt_id = int(gt_id)
        if gt_id <= 0:
            continue
        key = (int(window_index), gt_id)
        mapped = gt_id_map.get(key)
        if mapped is None:
            mapped = len(gt_id_map) + 1
            gt_id_map[key] = mapped
        out[gt == gt_id] = int(mapped)
    return out


def _sum_diag(diags: list[dict[str, Any]], key: str) -> int:
    return int(sum(_int(diag.get(key), 0) for diag in diags))


def _evaluate_point_projected_global(
    *,
    scene: str,
    split: str,
    frame_ids: list[int],
    source_step: str,
    local_export_root: Path,
    score_mode: str,
    output_root: Path,
    vertex_cache_root: Path,
    vertex_nn_radius: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
    scene_points, _colors, mesh_path = _load_scene_mesh(scene)
    tree = cKDTree(scene_points)
    point_owner, scores, point_diag, idx_to_obj = _load_point_objects(local_export_root, scene, source_step, int(scene_points.shape[0]))
    acc = SparseSceneIoU()
    case_rows: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        gt = _load_gt_2d(scene, int(frame_id), shape_hw)
        vertex_idx, vertex_diag = _load_or_build_vertex_map(
            stream=stream,
            tree=tree,
            frame_id=int(frame_id),
            scene=scene,
            shape_hw=shape_hw,
            radius=float(vertex_nn_radius),
            cache_root=vertex_cache_root,
            use_cache=True,
        )
        pred = np.zeros(shape_hw, dtype=np.int64)
        valid = (vertex_idx >= 0) & (vertex_idx < point_owner.shape[0])
        owner = np.zeros((int(np.count_nonzero(valid)),), dtype=np.int64)
        if owner.size:
            owner = point_owner[vertex_idx[valid]].astype(np.int64, copy=False)
            pred_valid = owner > 0
            if np.any(pred_valid):
                tmp = pred[valid]
                tmp[pred_valid] = owner[pred_valid]
                pred[valid] = tmp
        acc.add(pred, gt)
        case_rows.append(
            {
                "split": split,
                "scene_id": scene,
                "variant": source_step,
                "method_family": "stream3d_local_point_projected",
                "frame_id": int(frame_id),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "nn_hit_pixels": vertex_diag.get("nn_hit_pixels", ""),
                "nn_hit_ratio": vertex_diag.get("nn_hit_ratio", ""),
                "score_mode": score_mode,
            }
        )
    input_scores = scores if score_mode == "input" else None
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode=score_mode,
        input_scores=input_scores,
    )
    top_rows = [
        {
            "split": split,
            "scene_id": scene,
            "variant": source_step,
            "method_family": "stream3d_local_point_projected",
            "score_mode": score_mode,
            "pred_id": row["pred_id"],
            "mv_object_id": idx_to_obj.get(int(row["pred_id"]), ""),
            "gt_id": row["gt_id"],
            "iou": row["iou"],
        }
        for row in _top_iou_rows(iou, pred_ids, gt_ids, top_k=100)
    ]
    metric = _metric_row(
        scene=scene,
        split=split,
        variant=source_step,
        method_family="stream3d_local_point_projected",
        score_mode=score_mode,
        frame_ids=frame_ids,
        summary=summary,
        extra={
            **point_diag,
            "mesh_path": _rel(mesh_path),
            "vertex_nn_radius": float(vertex_nn_radius),
            "vertex_cache_root": _rel(vertex_cache_root),
        },
    )
    return metric, case_rows, top_rows


def _evaluate_point_projected_local_window(
    *,
    scene: str,
    split: str,
    frame_ids: list[int],
    source_step: str,
    local_export_root: Path,
    score_mode: str,
    output_root: Path,
    vertex_cache_root: Path,
    vertex_nn_radius: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
    scene_points, _colors, mesh_path = _load_scene_mesh(scene)
    tree = cKDTree(scene_points)
    _raw_dir, object_rows = _local_object_rows(local_export_root, scene, source_step)
    window_indices = sorted({_int(row.get("window_index"), -1) for row in object_rows if _int(row.get("window_index"), -1) >= 0})
    window_scope = _local_window_frame_scope(frame_ids, window_indices)

    acc = SparseSceneIoU()
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    global_scores: list[float] = []
    global_idx_to_obj: dict[int, str] = {}
    gt_id_map: dict[tuple[int, int], int] = {}
    point_diags: list[dict[str, Any]] = []
    pred_offset = 0

    for window_index in window_indices:
        support_frames = window_scope.get(int(window_index), [])
        point_owner, scores, point_diag, idx_to_obj = _load_point_objects(
            local_export_root,
            scene,
            source_step,
            int(scene_points.shape[0]),
            window_index=int(window_index),
        )
        point_diags.append(point_diag)
        for local_idx, obj in idx_to_obj.items():
            global_idx_to_obj[pred_offset + int(local_idx)] = obj
        global_scores.extend(float(v) for v in scores.tolist())
        window_acc = SparseSceneIoU()
        for frame_id in support_frames:
            gt = _load_gt_2d(scene, int(frame_id), shape_hw)
            gt_window = _window_scoped_gt(gt, int(window_index), gt_id_map)
            vertex_idx, vertex_diag = _load_or_build_vertex_map(
                stream=stream,
                tree=tree,
                frame_id=int(frame_id),
                scene=scene,
                shape_hw=shape_hw,
                radius=float(vertex_nn_radius),
                cache_root=vertex_cache_root,
                use_cache=True,
            )
            pred_local = np.zeros(shape_hw, dtype=np.int64)
            valid = (vertex_idx >= 0) & (vertex_idx < point_owner.shape[0])
            if np.any(valid):
                owner = point_owner[vertex_idx[valid]].astype(np.int64, copy=False)
                pred_valid = owner > 0
                if np.any(pred_valid):
                    tmp = pred_local[valid]
                    tmp[pred_valid] = owner[pred_valid]
                    pred_local[valid] = tmp
            pred_global = np.zeros(shape_hw, dtype=np.int64)
            positive = pred_local > 0
            pred_global[positive] = pred_local[positive] + int(pred_offset)
            acc.add(pred_global, gt_window)
            window_acc.add(pred_local, gt)
            case_rows.append(
                {
                    "split": split,
                    "scene_id": scene,
                    "variant": source_step,
                    "method_family": "stream3d_local_point_projected",
                    "support_policy": "local_window_gt_projection",
                    "window_index": int(window_index),
                    "window_frame_count": int(len(support_frames)),
                    "frame_id": int(frame_id),
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "pred_positive_pixels": int(np.count_nonzero(pred_local > 0)),
                    "nn_hit_pixels": vertex_diag.get("nn_hit_pixels", ""),
                    "nn_hit_ratio": vertex_diag.get("nn_hit_ratio", ""),
                    "score_mode": score_mode,
                }
            )
        window_input_scores = scores if score_mode == "input" else None
        window_summary, _window_iou, _window_pred_ids, _window_gt_ids = _summarize_iou(
            accumulator=window_acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode=score_mode,
            input_scores=window_input_scores,
        )
        window_metric_rows.append(
            _metric_row(
                scene=scene,
                split=split,
                variant=source_step,
                method_family="stream3d_local_point_projected_window",
                score_mode=score_mode,
                frame_ids=support_frames if support_frames else [frame_ids[0]],
                summary=window_summary,
                extra={
                    **point_diag,
                    "support_policy": "local_window_gt_projection",
                    "window_index": int(window_index),
                    "window_frame_count": int(len(support_frames)),
                    "mesh_path": _rel(mesh_path),
                    "vertex_nn_radius": float(vertex_nn_radius),
                    "vertex_cache_root": _rel(vertex_cache_root),
                },
            )
        )
        pred_offset += int(scores.shape[0])

    input_scores = np.asarray(global_scores, dtype=np.float32) if score_mode == "input" else None
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode=score_mode,
        input_scores=input_scores,
    )
    top_rows = [
        {
            "split": split,
            "scene_id": scene,
            "variant": source_step,
            "method_family": "stream3d_local_point_projected",
            "support_policy": "local_window_gt_projection",
            "score_mode": score_mode,
            "pred_id": row["pred_id"],
            "mv_object_id": global_idx_to_obj.get(int(row["pred_id"]), ""),
            "gt_id": row["gt_id"],
            "iou": row["iou"],
        }
        for row in _top_iou_rows(iou, pred_ids, gt_ids, top_k=100)
    ]
    frame_counts = [len(values) for values in window_scope.values()]
    metric = _metric_row(
        scene=scene,
        split=split,
        variant=source_step,
        method_family="stream3d_local_point_projected",
        score_mode=score_mode,
        frame_ids=frame_ids,
        summary=summary,
        extra={
            "raw_dir": point_diags[0].get("raw_dir", "") if point_diags else "",
            "source_step": source_step,
            "object_count": _sum_diag(point_diags, "object_count"),
            "point_npz_file_count": _sum_diag(point_diags, "point_npz_file_count"),
            "vertex_count": int(scene_points.shape[0]),
            "assigned_unique_vertex_count": _sum_diag(point_diags, "assigned_unique_vertex_count"),
            "assigned_point_write_count": _sum_diag(point_diags, "assigned_point_write_count"),
            "overwritten_point_write_count": _sum_diag(point_diags, "overwritten_point_write_count"),
            "invalid_point_id_count": _sum_diag(point_diags, "invalid_point_id_count"),
            "materialization": "local_3d_object_point_ids_projected_to_2d_via_v65_depth_to_mesh_vertex_map",
            "support_policy": "local_window_gt_projection",
            "support_policy_detail": "GT instance ids are scoped by Stream3D local window; predictions are projected only on frames in the same local window.",
            "support_window_count": int(len(window_scope)),
            "support_window_frame_count_min": int(min(frame_counts)) if frame_counts else 0,
            "support_window_frame_count_max": int(max(frame_counts)) if frame_counts else 0,
            "support_window_frame_count_mean": float(np.mean(frame_counts)) if frame_counts else 0.0,
            "mesh_path": _rel(mesh_path),
            "vertex_nn_radius": float(vertex_nn_radius),
            "vertex_cache_root": _rel(vertex_cache_root),
        },
    )
    return metric, case_rows, top_rows, window_metric_rows


def _method_family(variant: str) -> str:
    if variant in DEFAULT_OURS_VARIANTS:
        return "ours_real"
    if variant in DEFAULT_CONTROL_VARIANTS:
        return "control"
    return "other"


def _object_scores(rows: list[dict[str, Any]], object_to_idx: dict[str, int]) -> np.ndarray:
    by_obj: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        obj = str(row.get("mv_object_id", ""))
        if obj in object_to_idx:
            by_obj[obj].append(_num(row.get("object_score"), 1.0))
    scores = np.ones((len(object_to_idx),), dtype=np.float32)
    for obj, idx in object_to_idx.items():
        vals = by_obj.get(obj, [1.0])
        scores[idx - 1] = float(sum(vals) / max(1, len(vals)))
    return scores


def _object_scores_scoped(rows: list[dict[str, Any]], object_to_idx: dict[str, int], frame_to_window: dict[int, int]) -> np.ndarray:
    by_obj: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        frame_id = _int(row.get("frame_id"), -1)
        if frame_id not in frame_to_window:
            continue
        obj = str(row.get("mv_object_id", ""))
        if not obj:
            continue
        scoped = f"w{int(frame_to_window[frame_id]):04d}|{obj}"
        if scoped in object_to_idx:
            by_obj[scoped].append(_num(row.get("object_score"), 1.0))
    scores = np.ones((len(object_to_idx),), dtype=np.float32)
    for obj, idx in object_to_idx.items():
        vals = by_obj.get(obj, [1.0])
        scores[idx - 1] = float(sum(vals) / max(1, len(vals)))
    return scores


def _evaluate_frame_mask_variant(
    *,
    scene: str,
    split: str,
    variant: str,
    frame_ids: list[int],
    rows: list[dict[str, Any]],
    score_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
    object_ids = sorted({str(row.get("mv_object_id", "")) for row in rows if str(row.get("mv_object_id", ""))})
    object_to_idx = {obj: idx + 1 for idx, obj in enumerate(object_ids)}
    idx_to_obj = {idx: obj for obj, idx in object_to_idx.items()}
    mask_to_obj: dict[tuple[int, int], int] = {}
    duplicate_conflicts = 0
    for row in rows:
        obj = str(row.get("mv_object_id", ""))
        if obj not in object_to_idx:
            continue
        key = (_int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        if key[0] < 0 or key[1] <= 0:
            continue
        idx = object_to_idx[obj]
        old = mask_to_obj.get(key)
        if old is not None and old != idx:
            duplicate_conflicts += 1
            continue
        mask_to_obj[key] = idx
    scores = _object_scores(rows, object_to_idx) if score_mode == "input" else None
    mask_dir = _mask_dir(scene)
    acc = SparseSceneIoU()
    case_rows: list[dict[str, Any]] = []
    missing_masks = 0
    for frame_id in frame_ids:
        gt = _load_gt_2d(scene, int(frame_id), shape_hw)
        pred = np.zeros(shape_hw, dtype=np.int64)
        mask_path = mask_dir / f"{int(frame_id)}.png"
        if mask_path.exists():
            mask = _read_label_png(mask_path, shape_hw)
            for mask_id in np.unique(mask):
                mask_id = int(mask_id)
                if mask_id <= 0:
                    continue
                label = mask_to_obj.get((int(frame_id), mask_id), 0)
                if label > 0:
                    pred[mask == mask_id] = label
        else:
            missing_masks += 1
        acc.add(pred, gt)
        case_rows.append(
            {
                "split": split,
                "scene_id": scene,
                "variant": variant,
                "method_family": _method_family(variant),
                "frame_id": int(frame_id),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "score_mode": score_mode,
                "mask_path": _rel(mask_path),
                "mask_exists": bool(mask_path.exists()),
            }
        )
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode=score_mode,
        input_scores=scores,
    )
    top_rows = [
        {
            "split": split,
            "scene_id": scene,
            "variant": variant,
            "method_family": _method_family(variant),
            "score_mode": score_mode,
            "pred_id": row["pred_id"],
            "mv_object_id": idx_to_obj.get(int(row["pred_id"]), ""),
            "gt_id": row["gt_id"],
            "iou": row["iou"],
        }
        for row in _top_iou_rows(iou, pred_ids, gt_ids, top_k=100)
    ]
    metric = _metric_row(
        scene=scene,
        split=split,
        variant=variant,
        method_family=_method_family(variant),
        score_mode=score_mode,
        frame_ids=frame_ids,
        summary=summary,
        extra={
            "object_count_from_input_rows": len(object_ids),
            "frame_mask_row_count": len(rows),
            "unique_frame_mask_count": len(mask_to_obj),
            "duplicate_frame_mask_conflict_count": duplicate_conflicts,
            "mask_dir": _rel(mask_dir),
            "missing_mask_raster_count": missing_masks,
            "materialization": "v87_v88_frame_mask_object_tube_contract",
        },
    )
    return metric, case_rows, top_rows


def _evaluate_frame_mask_variant_local_window(
    *,
    scene: str,
    split: str,
    variant: str,
    frame_ids: list[int],
    rows: list[dict[str, Any]],
    score_mode: str,
    local_export_root: Path,
    window_source_step: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
    shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
    _raw_dir, local_object_rows = _local_object_rows(local_export_root, scene, window_source_step)
    window_indices = sorted({_int(row.get("window_index"), -1) for row in local_object_rows if _int(row.get("window_index"), -1) >= 0})
    window_scope = _local_window_frame_scope(frame_ids, window_indices)
    frame_to_window = {
        int(frame_id): int(window_index)
        for window_index, support_frames in window_scope.items()
        for frame_id in support_frames
    }

    object_ids = sorted(
        {
            f"w{int(frame_to_window[_int(row.get('frame_id'), -1)]):04d}|{str(row.get('mv_object_id', ''))}"
            for row in rows
            if _int(row.get("frame_id"), -1) in frame_to_window and str(row.get("mv_object_id", ""))
        }
    )
    object_to_idx = {obj: idx + 1 for idx, obj in enumerate(object_ids)}
    idx_to_obj = {idx: obj for obj, idx in object_to_idx.items()}
    mask_to_obj: dict[tuple[int, int, int], int] = {}
    duplicate_conflicts = 0
    for row in rows:
        frame_id = _int(row.get("frame_id"), -1)
        window_index = frame_to_window.get(frame_id)
        if window_index is None:
            continue
        obj = str(row.get("mv_object_id", ""))
        if not obj:
            continue
        scoped = f"w{int(window_index):04d}|{obj}"
        if scoped not in object_to_idx:
            continue
        key = (int(window_index), int(frame_id), _int(row.get("mask_id"), -1))
        if key[2] <= 0:
            continue
        idx = object_to_idx[scoped]
        old = mask_to_obj.get(key)
        if old is not None and old != idx:
            duplicate_conflicts += 1
            continue
        mask_to_obj[key] = idx

    scores = _object_scores_scoped(rows, object_to_idx, frame_to_window) if score_mode == "input" else None
    mask_dir = _mask_dir(scene)
    acc = SparseSceneIoU()
    gt_id_map: dict[tuple[int, int], int] = {}
    case_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    missing_masks = 0
    for window_index, support_frames in sorted(window_scope.items()):
        window_acc = SparseSceneIoU()
        window_rows = [row for row in rows if frame_to_window.get(_int(row.get("frame_id"), -1)) == int(window_index)]
        window_object_ids = sorted(
            {
                f"w{int(window_index):04d}|{str(row.get('mv_object_id', ''))}"
                for row in window_rows
                if str(row.get("mv_object_id", ""))
            }
        )
        window_object_to_idx = {obj: object_to_idx[obj] for obj in window_object_ids if obj in object_to_idx}
        for frame_id in support_frames:
            gt = _load_gt_2d(scene, int(frame_id), shape_hw)
            gt_window = _window_scoped_gt(gt, int(window_index), gt_id_map)
            pred = np.zeros(shape_hw, dtype=np.int64)
            mask_path = mask_dir / f"{int(frame_id)}.png"
            if mask_path.exists():
                mask = _read_label_png(mask_path, shape_hw)
                for mask_id in np.unique(mask):
                    mask_id = int(mask_id)
                    if mask_id <= 0:
                        continue
                    label = mask_to_obj.get((int(window_index), int(frame_id), mask_id), 0)
                    if label > 0:
                        pred[mask == mask_id] = label
            else:
                missing_masks += 1
            acc.add(pred, gt_window)
            window_acc.add(pred, gt)
            case_rows.append(
                {
                    "split": split,
                    "scene_id": scene,
                    "variant": variant,
                    "method_family": _method_family(variant),
                    "support_policy": "local_window_gt_projection",
                    "window_index": int(window_index),
                    "window_frame_count": int(len(support_frames)),
                    "frame_id": int(frame_id),
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                    "score_mode": score_mode,
                    "mask_path": _rel(mask_path),
                    "mask_exists": bool(mask_path.exists()),
                }
            )
        window_scores = None
        if score_mode == "input":
            window_scores = np.ones((len(window_object_to_idx),), dtype=np.float32)
            for scoped, global_idx in window_object_to_idx.items():
                if scores is not None and 0 <= global_idx - 1 < scores.shape[0]:
                    window_scores[list(window_object_to_idx).index(scoped)] = float(scores[global_idx - 1])
        window_summary, _window_iou, _window_pred_ids, _window_gt_ids = _summarize_iou(
            accumulator=window_acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode=score_mode,
            input_scores=window_scores,
        )
        window_metric_rows.append(
            _metric_row(
                scene=scene,
                split=split,
                variant=variant,
                method_family=f"{_method_family(variant)}_window",
                score_mode=score_mode,
                frame_ids=support_frames if support_frames else [frame_ids[0]],
                summary=window_summary,
                extra={
                    "support_policy": "local_window_gt_projection",
                    "window_index": int(window_index),
                    "window_frame_count": int(len(support_frames)),
                    "materialization": "v87_v88_frame_mask_object_tube_contract_split_by_local_window",
                    "object_count_from_input_rows": len(window_object_to_idx),
                    "frame_mask_row_count": len(window_rows),
                    "mask_dir": _rel(mask_dir),
                },
            )
        )

    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode=score_mode,
        input_scores=scores,
    )
    top_rows = [
        {
            "split": split,
            "scene_id": scene,
            "variant": variant,
            "method_family": _method_family(variant),
            "support_policy": "local_window_gt_projection",
            "score_mode": score_mode,
            "pred_id": row["pred_id"],
            "mv_object_id": idx_to_obj.get(int(row["pred_id"]), ""),
            "gt_id": row["gt_id"],
            "iou": row["iou"],
        }
        for row in _top_iou_rows(iou, pred_ids, gt_ids, top_k=100)
    ]
    frame_counts = [len(values) for values in window_scope.values()]
    metric = _metric_row(
        scene=scene,
        split=split,
        variant=variant,
        method_family=_method_family(variant),
        score_mode=score_mode,
        frame_ids=frame_ids,
        summary=summary,
        extra={
            "object_count_from_input_rows": len(object_ids),
            "frame_mask_row_count": len(rows),
            "unique_frame_mask_count": len(mask_to_obj),
            "duplicate_frame_mask_conflict_count": duplicate_conflicts,
            "mask_dir": _rel(mask_dir),
            "missing_mask_raster_count": missing_masks,
            "materialization": "v87_v88_frame_mask_object_tube_contract_split_by_local_window",
            "support_policy": "local_window_gt_projection",
            "support_policy_detail": "method frame-mask tube objects are split by Stream3D local window; GT instance ids are scoped by window.",
            "support_window_count": int(len(window_scope)),
            "support_window_frame_count_min": int(min(frame_counts)) if frame_counts else 0,
            "support_window_frame_count_max": int(max(frame_counts)) if frame_counts else 0,
            "support_window_frame_count_mean": float(np.mean(frame_counts)) if frame_counts else 0.0,
        },
    )
    return metric, case_rows, top_rows, window_metric_rows


def _metric_row(
    *,
    scene: str,
    split: str,
    variant: str,
    method_family: str,
    score_mode: str,
    frame_ids: list[int],
    summary: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": scene,
        "split": split,
        "variant": variant,
        "method_family": method_family,
        "score_mode": score_mode,
        "frame_count": len(frame_ids),
        "frame_first": int(frame_ids[0]),
        "frame_last": int(frame_ids[-1]),
        "MV_AP": summary.get("ap"),
        "MV_AP50": summary.get("ap50"),
        "MV_AP25": summary.get("ap25"),
        "SF50_tp": summary.get("score_free_match_at_050", {}).get("tp"),
        "SF50_precision": summary.get("score_free_match_at_050", {}).get("precision"),
        "SF50_recall": summary.get("score_free_match_at_050", {}).get("recall"),
        "pred_object_count": summary.get("evaluated_pred_count"),
        "gt_object_count": summary.get("evaluated_gt_count"),
        "gt_best_iou_mean": summary.get("gt_best_iou_mean"),
        "gt_best_iou_median": summary.get("gt_best_iou_median"),
        "gt_best_iou_max": summary.get("gt_best_iou_max"),
        "gt_recall_best_iou_ge_050": summary.get("gt_recall_best_iou_ge_050"),
        "pred_best_iou_mean": summary.get("pred_best_iou_mean"),
        "score_unique_count": summary.get("score_unique_count"),
        "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        **extra,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_family"]), str(row["variant"]), str(row["score_mode"]))].append(row)
    out = []
    for (family, variant, score_mode), group in sorted(grouped.items()):
        out.append(
            {
                "method_family": family,
                "variant": variant,
                "score_mode": score_mode,
                "scene_count": len(group),
                "mean_MV_AP": float(np.mean([_num(r.get("MV_AP")) for r in group])),
                "mean_MV_AP50": float(np.mean([_num(r.get("MV_AP50")) for r in group])),
                "mean_MV_AP25": float(np.mean([_num(r.get("MV_AP25")) for r in group])),
                "mean_SF50_recall": float(np.mean([_num(r.get("SF50_recall")) for r in group])),
                "mean_pred_object_count": float(np.mean([_num(r.get("pred_object_count")) for r in group])),
                "mean_gt_object_count": float(np.mean([_num(r.get("gt_object_count")) for r in group])),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    output_root = _project(args.output_root)
    local_export_root = _project(args.local_export_root)
    vertex_cache_root = _project(args.vertex_cache_root)
    scenes = _parse_csv_list(args.scenes)
    stream3d_steps = _parse_csv_list(args.stream3d_steps)
    ours_variants = _parse_csv_list(args.ours_variants)
    control_variants = _parse_csv_list(args.control_variants)
    method_variants = ours_variants + control_variants
    score_modes = _parse_csv_list(args.score_modes)
    scope = _frame_scope()

    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    method_input_rows = _read_csv(args.method_frame_rows)

    for scene in scenes:
        frame_ids = scope.get(("dev", scene))
        if not frame_ids:
            raise RuntimeError(f"no dev frame scope for {scene}")
        for step in stream3d_steps:
            for score_mode in score_modes:
                if args.stream3d_support_policy == "local_window_gt_projection":
                    metric, cases, tops, windows = _evaluate_point_projected_local_window(
                        scene=scene,
                        split="dev",
                        frame_ids=frame_ids,
                        source_step=step,
                        local_export_root=local_export_root,
                        score_mode=score_mode,
                        output_root=output_root,
                        vertex_cache_root=vertex_cache_root,
                        vertex_nn_radius=float(args.vertex_nn_radius),
                    )
                    window_metric_rows.extend(windows)
                elif args.stream3d_support_policy == "global_frame_scope":
                    metric, cases, tops = _evaluate_point_projected_global(
                        scene=scene,
                        split="dev",
                        frame_ids=frame_ids,
                        source_step=step,
                        local_export_root=local_export_root,
                        score_mode=score_mode,
                        output_root=output_root,
                        vertex_cache_root=vertex_cache_root,
                        vertex_nn_radius=float(args.vertex_nn_radius),
                    )
                else:
                    raise ValueError(f"unsupported stream3d_support_policy={args.stream3d_support_policy}")
                metric_rows.append(metric)
                case_rows.extend(cases)
                top_rows.extend(tops)
                print(json.dumps(metric, sort_keys=True), flush=True)
        for variant in method_variants:
            rows = [
                row
                for row in method_input_rows
                if row.get("split") == "dev" and row.get("scene_id") == scene and row.get("source_variant", row.get("variant")) == variant
            ]
            if not rows:
                continue
            for score_mode in score_modes:
                if args.method_support_policy == "local_window_gt_projection":
                    metric, cases, tops, windows = _evaluate_frame_mask_variant_local_window(
                        scene=scene,
                        split="dev",
                        variant=variant,
                        frame_ids=frame_ids,
                        rows=rows,
                        score_mode=score_mode,
                        local_export_root=local_export_root,
                        window_source_step=args.method_window_source_step,
                    )
                    window_metric_rows.extend(windows)
                elif args.method_support_policy == "global_frame_scope":
                    metric, cases, tops = _evaluate_frame_mask_variant(
                        scene=scene,
                        split="dev",
                        variant=variant,
                        frame_ids=frame_ids,
                        rows=rows,
                        score_mode=score_mode,
                    )
                else:
                    raise ValueError(f"unsupported method_support_policy={args.method_support_policy}")
                metric_rows.append(metric)
                case_rows.extend(cases)
                top_rows.extend(tops)
                print(json.dumps(metric, sort_keys=True), flush=True)

    aggregate_rows = _aggregate(metric_rows)
    _write_csv(output_root / "mv_metric_rows.csv", metric_rows)
    _write_csv(output_root / "mv_eval_case_rows.csv", case_rows)
    _write_csv(output_root / "mv_top_iou_rows.csv", top_rows)
    _write_csv(output_root / "mv_aggregate_rows.csv", aggregate_rows)
    _write_csv(output_root / "mv_window_metric_rows.csv", window_metric_rows)
    summary = {
        "phase": "v89_recalc_point_projected_mv_ap",
        "runtime_sec": time.time() - t0,
        "scenes": scenes,
        "score_modes": score_modes,
        "stream3d_steps": stream3d_steps,
        "stream3d_support_policy": args.stream3d_support_policy,
        "method_support_policy": args.method_support_policy,
        "method_window_source_step": args.method_window_source_step,
        "ours_variants": ours_variants,
        "control_variants": control_variants,
        "local_export_root": _rel(local_export_root),
        "method_frame_rows": _rel(args.method_frame_rows),
        "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "bug_fixed": "Stream3D local-stage objects are evaluated from exported 3D point ids projected to 2D, not by reusing matched CropFormer mask ids as predictions.",
        "outputs": {
            "metric_rows": _rel(output_root / "mv_metric_rows.csv"),
            "aggregate_rows": _rel(output_root / "mv_aggregate_rows.csv"),
            "case_rows": _rel(output_root / "mv_eval_case_rows.csv"),
            "top_iou_rows": _rel(output_root / "mv_top_iou_rows.csv"),
            "window_metric_rows": _rel(output_root / "mv_window_metric_rows.csv"),
        },
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(
        output_root / "SHA256SUMS.json",
        {
            _rel(path): _sha256(_project(path))
            for path in [
                output_root / "mv_metric_rows.csv",
                output_root / "mv_aggregate_rows.csv",
                output_root / "mv_eval_case_rows.csv",
                output_root / "mv_top_iou_rows.csv",
                output_root / "mv_window_metric_rows.csv",
                output_root / "summary.json",
            ]
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute v89 MV_AP with Stream3D local 3D objects projected to 2D masks.")
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--score-modes", default="input")
    parser.add_argument("--stream3d-steps", default="S3D_L1_local_merged_masks")
    parser.add_argument("--ours-variants", default=",".join(DEFAULT_OURS_VARIANTS))
    parser.add_argument("--control-variants", default=",".join(DEFAULT_CONTROL_VARIANTS))
    parser.add_argument("--local-export-root", default="outputs/audit/v89_recalc_point_projected_mv_ap")
    parser.add_argument("--method-frame-rows", default="outputs/audit/v89_phase2_mv_tube_normalization/mv_object_frame_mask_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v89_recalc_point_projected_mv_ap/eval")
    parser.add_argument("--vertex-cache-root", default="outputs/cache/v66_scene_multiview_vertex_maps")
    parser.add_argument("--vertex-nn-radius", type=float, default=0.08)
    parser.add_argument(
        "--stream3d-support-policy",
        default="local_window_gt_projection",
        choices=["local_window_gt_projection", "global_frame_scope"],
    )
    parser.add_argument(
        "--method-support-policy",
        default="global_frame_scope",
        choices=["local_window_gt_projection", "global_frame_scope"],
    )
    parser.add_argument("--method-window-source-step", default="S3D_L1_local_merged_masks")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
