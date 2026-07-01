from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v101_phase1_f2_fragmentation_casebook"
PHASE0_DIR = AUDIT_ROOT / "v101_phase0_fact_lock"
PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PLAN_DOC = ROOT / "docs" / "stream4d_v101_geometry_provider_capability_f2_fragment_repair_plan.md"

if str(STREAM3D) not in sys.path:
    sys.path.insert(0, str(STREAM3D))

from tools.run_v65_scene_multiview_ap import _load_gt_2d  # noqa: E402


OBJECT_LIKE_MIN_AREA_RATIO = 0.005
BROAD_MASK_AREA_RATIO = 0.20


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if str(p).startswith("Stream3D/"):
        return ROOT / p
    return STREAM3D / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label png: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    arr = np.asarray(image, dtype=np.int64)
    if shape_hw is not None and tuple(arr.shape[:2]) != tuple(shape_hw):
        h, w = shape_hw
        arr = cv2.resize(arr, (int(w), int(h)), interpolation=cv2.INTER_NEAREST).astype(np.int64, copy=False)
    return arr


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _p50(values: list[float]) -> float:
    return float(np.percentile(values, 50)) if values else 0.0


def _p90(values: list[float]) -> float:
    return float(np.percentile(values, 90)) if values else 0.0


def _bucket(mask_area_ratio: float) -> str:
    if mask_area_ratio >= BROAD_MASK_AREA_RATIO:
        return "broad"
    if mask_area_ratio < OBJECT_LIKE_MIN_AREA_RATIO:
        return "small"
    return "object_like"


def _window_scoped_gt(gt: np.ndarray, chunk_key: str, gt_id_map: dict[tuple[str, int], int], reverse: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(gt, dtype=np.int64)
    for raw in np.unique(gt):
        raw_i = int(raw)
        if raw_i <= 0:
            continue
        key = (chunk_key, raw_i)
        if key not in gt_id_map:
            gt_id_map[key] = len(gt_id_map) + 1
            reverse[gt_id_map[key]] = raw_i
        out[gt == raw_i] = gt_id_map[key]
    return out


def _percentile_or_none(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _object_text(values: list[Any], limit: int = 32) -> str:
    vals = [str(v) for v in values]
    if len(vals) > limit:
        return "|".join(vals[:limit]) + f"|...(+{len(vals)-limit})"
    return "|".join(vals)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet"),
        pd.read_parquet(PHASE2C_DIR / "mv_object_rows.parquet"),
        pd.read_csv(PHASE2C_DIR / "frame_eval_rows.csv"),
    )


def _frame_groups(frame_eval: pd.DataFrame) -> dict[tuple[str, str, str, int], Path]:
    out: dict[tuple[str, str, str, int], Path] = {}
    for row in frame_eval.to_dict("records"):
        key = (str(row["dataset_split"]), str(row["scene_id"]), str(row["chunk_id"]), int(row["frame_id"]))
        out[key] = _project(row["mask_path"])
    return out


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    if not bool(phase0.get("phase0_pass")):
        raise RuntimeError("Phase0 did not pass; refusing to run Phase1 fragmentation diagnostic.")

    mask_rows_df, object_rows_df, frame_eval_df = _load_inputs()
    frame_mask_path = _frame_groups(frame_eval_df)
    object_meta = {str(r["mv_object_id"]): r for r in object_rows_df.to_dict("records")}

    gt_fragment_rows: list[dict[str, Any]] = []
    pred_object_rows: list[dict[str, Any]] = []
    fragment_pair_rows: list[dict[str, Any]] = []
    pred_gt_overlap_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    frame_audit_rows: list[dict[str, Any]] = []

    gt_fragment_count_values: list[float] = []
    union_minus_best_values: list[float] = []
    best_iou_values: list[float] = []
    union_iou_values: list[float] = []
    threshold_stats: dict[float, dict[str, Any]] = {
        0.0: {"fragment_counts": [], "union_gains": [], "ge2": 0, "gain10": 0},
        0.01: {"fragment_counts": [], "union_gains": [], "ge2": 0, "gain10": 0},
        0.05: {"fragment_counts": [], "union_gains": [], "ge2": 0, "gain10": 0},
        0.10: {"fragment_counts": [], "union_gains": [], "ge2": 0, "gain10": 0},
    }
    fragment_ge2_count = 0
    fragment_ge3_count = 0
    union_gain_gt_0p10_count = 0
    union_gain_gt_0p20_count = 0

    total_duplicate_mask_object_keys = 0
    total_missing_mask_frames = 0
    total_frame_count = 0
    total_gt_count = 0
    total_pred_count = 0

    group_cols = ["dataset_split", "scene_id", "chunk_id"]
    for (split, scene, chunk), chunk_rows_df in mask_rows_df.groupby(group_cols, sort=True):
        split = str(split)
        scene = str(scene)
        chunk = str(chunk)
        chunk_key = f"{scene}|{chunk}"
        rows = chunk_rows_df.sort_values(["frame_id", "selected_mask_id", "score", "mv_object_id"], ascending=[True, True, False, True]).to_dict("records")
        object_ids = sorted({str(r["mv_object_id"]) for r in rows})
        object_to_idx = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
        idx_to_object = {idx: oid for oid, idx in object_to_idx.items()}

        pred_area: defaultdict[int, int] = defaultdict(int)
        gt_area: defaultdict[int, int] = defaultdict(int)
        inter: defaultdict[tuple[int, int], int] = defaultdict(int)
        pair_frame_hits: defaultdict[tuple[int, int], int] = defaultdict(int)
        gt_visible_frames: defaultdict[int, int] = defaultdict(int)
        pred_visible_frames: defaultdict[int, int] = defaultdict(int)
        pred_mask_areas: defaultdict[int, list[int]] = defaultdict(list)
        pred_mask_area_ratios: defaultdict[int, list[float]] = defaultdict(list)
        pred_mask_buckets: defaultdict[int, list[str]] = defaultdict(list)
        pred_semantic_proxy: defaultdict[int, list[float]] = defaultdict(list)
        duplicate_key_count = 0
        missing_mask_count = 0
        pixel_collision_count = 0
        pred_positive_pixels = 0
        raw_gt_reverse: dict[int, int] = {}
        gt_id_map: dict[tuple[str, int], int] = {}

        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_frame[int(row["frame_id"])].append(row)

        for frame_id in sorted(by_frame):
            total_frame_count += 1
            frame_rows = by_frame[frame_id]
            mask_path = frame_mask_path.get((split, scene, chunk, int(frame_id)))
            if mask_path is None or not mask_path.exists():
                missing_mask_count += 1
                total_missing_mask_frames += 1
                shape_hw = (968, 1296)
                label = np.zeros(shape_hw, dtype=np.int64)
            else:
                label = _read_label(mask_path)
                shape_hw = tuple(int(v) for v in label.shape[:2])
            gt_raw = _load_gt_2d(scene, int(frame_id), shape_hw)
            gt = _window_scoped_gt(gt_raw, chunk_key, gt_id_map, raw_gt_reverse)
            pred = np.zeros(shape_hw, dtype=np.int64)
            label_ids, label_counts = np.unique(label[label > 0], return_counts=True)
            label_area = {int(k): int(v) for k, v in zip(label_ids, label_counts)}

            rows_by_mask: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in frame_rows:
                rows_by_mask[int(row["selected_mask_id"])].append(row)
            selected_rows: list[dict[str, Any]] = []
            for mask_id, vals in rows_by_mask.items():
                vals_sorted = sorted(vals, key=lambda r: (_num(r.get("score")), str(r.get("mv_object_id"))), reverse=True)
                chosen = vals_sorted[0]
                if len({str(v.get("mv_object_id", "")) for v in vals_sorted}) > 1:
                    duplicate_key_count += len(vals_sorted) - 1
                    total_duplicate_mask_object_keys += len(vals_sorted) - 1
                selected_rows.append(chosen)

            for row in sorted(selected_rows, key=lambda r: (-_num(r.get("score")), str(r.get("mv_object_id")))):
                oid = str(row["mv_object_id"])
                pred_idx = object_to_idx[oid]
                mask_id = int(row["selected_mask_id"])
                mask = label == mask_id
                area = int(label_area.get(mask_id, int(np.count_nonzero(mask))))
                ratio = float(area / max(1, label.size))
                pred_mask_areas[pred_idx].append(area)
                pred_mask_area_ratios[pred_idx].append(ratio)
                pred_mask_buckets[pred_idx].append(_bucket(ratio))
                pred_semantic_proxy[pred_idx].append(_num(row.get("v100_semantic_norm"), default=float("nan")))
                pixel_collision_count += int(np.count_nonzero((pred > 0) & mask))
                pred[(pred == 0) & mask] = pred_idx

            pred_pos = pred > 0
            gt_pos = gt > 0
            pred_positive_pixels += int(np.count_nonzero(pred_pos))

            if np.any(pred_pos):
                ids, counts = np.unique(pred[pred_pos], return_counts=True)
                for value, count in zip(ids, counts):
                    pred_area[int(value)] += int(count)
                    pred_visible_frames[int(value)] += 1
            if np.any(gt_pos):
                ids, counts = np.unique(gt[gt_pos], return_counts=True)
                for value, count in zip(ids, counts):
                    gt_area[int(value)] += int(count)
                    gt_visible_frames[int(value)] += 1
            both = pred_pos & gt_pos
            if np.any(both):
                pred_vals = pred[both]
                gt_vals = gt[both]
                base = int(np.max(gt_vals)) + 1
                encoded = pred_vals * base + gt_vals
                ids, counts = np.unique(encoded, return_counts=True)
                for value, count in zip(ids, counts):
                    pid = int(value // base)
                    gid = int(value % base)
                    inter[(pid, gid)] += int(count)
                    pair_frame_hits[(pid, gid)] += 1

            frame_audit_rows.append(
                {
                    "schema_version": "stream4d_v101_phase1_frame_audit_row_v1",
                    "phase_id": "v101_phase1_f2_fragmentation_casebook",
                    "dataset_split": split,
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "frame_id": int(frame_id),
                    "mask_path": _rel(mask_path) if mask_path is not None else "",
                    "mask_exists": bool(mask_path is not None and mask_path.exists()),
                    "selected_mask_rows": len(frame_rows),
                    "unique_selected_masks": len(rows_by_mask),
                    "duplicate_mask_object_keys": duplicate_key_count,
                    "pred_positive_pixels": int(np.count_nonzero(pred_pos)),
                    "gt_positive_pixels": int(np.count_nonzero(gt_pos)),
                    "pixel_collision_count_cumulative": pixel_collision_count,
                }
            )

        pred_ids = sorted(pred_area)
        gt_ids = sorted(gt_area)
        total_gt_count += len(gt_ids)
        total_pred_count += len(pred_ids)

        iou_by_pair: dict[tuple[int, int], float] = {}
        for (pid, gid), count in inter.items():
            union = int(pred_area[pid]) + int(gt_area[gid]) - int(count)
            iou_by_pair[(pid, gid)] = float(count / union) if union > 0 else 0.0

        pred_to_gt_iou: dict[int, list[tuple[int, float]]] = defaultdict(list)
        gt_to_pred_iou: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (pid, gid), iou in iou_by_pair.items():
            if iou <= 0:
                continue
            pred_to_gt_iou[pid].append((gid, iou))
            gt_to_pred_iou[gid].append((pid, iou))
            pred_gt_overlap_rows.append(
                {
                    "schema_version": "stream4d_v101_phase1_pred_gt_overlap_row_v1",
                    "phase_id": "v101_phase1_f2_fragmentation_casebook",
                    "dataset_split": split,
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "window_id": chunk,
                    "gt_window_id": gid,
                    "raw_gt_object_id": raw_gt_reverse.get(gid, ""),
                    "mv_object_id": idx_to_object[pid],
                    "pred_pixels": int(pred_area[pid]),
                    "GT_pixels": int(gt_area[gid]),
                    "intersection_pixels": int(inter.get((pid, gid), 0)),
                    "IoU": float(iou),
                    "gt_overlap_frame_count": int(pair_frame_hits.get((pid, gid), 0)),
                    "uses_gt_for_label": True,
                    "uses_gt_for_prediction": False,
                }
            )

        for gid in gt_ids:
            pred_iou_pairs = sorted(gt_to_pred_iou.get(gid, []), key=lambda x: x[1], reverse=True)
            fragment_ids = [pid for pid, _ in pred_iou_pairs]
            inter_sum = int(sum(inter.get((pid, gid), 0) for pid in fragment_ids))
            pred_union_area = int(sum(pred_area[pid] for pid in fragment_ids))
            union = pred_union_area + int(gt_area[gid]) - inter_sum
            union_iou = float(inter_sum / union) if union > 0 else 0.0
            best_iou = float(pred_iou_pairs[0][1]) if pred_iou_pairs else 0.0
            union_gain = float(union_iou - best_iou)
            fragment_count = len(fragment_ids)
            gt_fragment_count_values.append(float(fragment_count))
            union_minus_best_values.append(union_gain)
            best_iou_values.append(best_iou)
            union_iou_values.append(union_iou)
            fragment_ge2_count += int(fragment_count >= 2)
            fragment_ge3_count += int(fragment_count >= 3)
            union_gain_gt_0p10_count += int(union_gain > 0.10)
            union_gain_gt_0p20_count += int(union_gain > 0.20)
            for tau, stats in threshold_stats.items():
                if tau <= 0.0:
                    filtered = fragment_ids
                else:
                    filtered = [pid for pid, iou in pred_iou_pairs if iou >= tau]
                filtered_inter = int(sum(inter.get((pid, gid), 0) for pid in filtered))
                filtered_union_area = int(sum(pred_area[pid] for pid in filtered))
                filtered_union = filtered_union_area + int(gt_area[gid]) - filtered_inter
                filtered_union_iou = float(filtered_inter / filtered_union) if filtered_union > 0 else 0.0
                filtered_best = max([iou for pid, iou in pred_iou_pairs if pid in set(filtered)], default=0.0)
                filtered_gain = float(filtered_union_iou - filtered_best)
                stats["fragment_counts"].append(float(len(filtered)))
                stats["union_gains"].append(filtered_gain)
                stats["ge2"] += int(len(filtered) >= 2)
                stats["gain10"] += int(filtered_gain > 0.10)
            scores = [_num(object_meta.get(idx_to_object[pid], {}).get("object_score"), default=0.0) for pid in fragment_ids]
            frame_hits = [int(pair_frame_hits.get((pid, gid), 0)) for pid in fragment_ids]
            row = {
                "schema_version": "stream4d_v101_phase1_gt_fragment_row_v1",
                "phase_id": "v101_phase1_f2_fragmentation_casebook",
                "dataset_split": split,
                "scene_id": scene,
                "chunk_id": chunk,
                "window_id": chunk,
                "gt_window_id": gid,
                "raw_gt_object_id": raw_gt_reverse.get(gid, ""),
                "fragment_count": fragment_count,
                "pred_objects_per_GT": fragment_count,
                "best_pred_IoU": best_iou,
                "union_pred_IoU": union_iou,
                "union_minus_best_IoU": union_gain,
                "GT_pixels": int(gt_area[gid]),
                "visible_frame_count": int(gt_visible_frames.get(gid, 0)),
                "best_pred_object_id": idx_to_object.get(fragment_ids[0], "") if fragment_ids else "",
                "fragment_object_ids": _object_text([idx_to_object[pid] for pid in fragment_ids], limit=64),
                "fragment_score_distribution": _object_text([f"{v:.6f}" for v in scores], limit=64),
                "fragment_frame_overlap_distribution": _object_text(frame_hits, limit=64),
            }
            gt_fragment_rows.append(row)

            if fragment_count >= 2:
                for rank, (pa, pb) in enumerate(combinations(fragment_ids, 2)):
                    if rank >= 128:
                        break
                    shared_frames_a = int(pair_frame_hits.get((pa, gid), 0))
                    shared_frames_b = int(pair_frame_hits.get((pb, gid), 0))
                    fragment_pair_rows.append(
                        {
                            "schema_version": "stream4d_v101_phase1_fragment_pair_row_v1",
                            "phase_id": "v101_phase1_f2_fragmentation_casebook",
                            "dataset_split": split,
                            "scene_id": scene,
                            "chunk_id": chunk,
                            "window_id": chunk,
                            "gt_window_id": gid,
                            "raw_gt_object_id": raw_gt_reverse.get(gid, ""),
                            "obj_i": idx_to_object[pa],
                            "obj_j": idx_to_object[pb],
                            "obj_i_iou_to_gt": iou_by_pair.get((pa, gid), 0.0),
                            "obj_j_iou_to_gt": iou_by_pair.get((pb, gid), 0.0),
                            "obj_i_gt_intersection_pixels": inter.get((pa, gid), 0),
                            "obj_j_gt_intersection_pixels": inter.get((pb, gid), 0),
                            "obj_i_gt_overlap_frame_count": shared_frames_a,
                            "obj_j_gt_overlap_frame_count": shared_frames_b,
                            "frame_support_relation": "diagnostic_same_gt_fragment_pair",
                            "uses_gt_for_label": True,
                            "uses_gt_for_prediction": False,
                        }
                    )

        for pid in pred_ids:
            gt_iou_pairs = sorted(pred_to_gt_iou.get(pid, []), key=lambda x: x[1], reverse=True)
            best = gt_iou_pairs[0][1] if gt_iou_pairs else 0.0
            second = gt_iou_pairs[1][1] if len(gt_iou_pairs) > 1 else 0.0
            buckets = pred_mask_buckets.get(pid, [])
            broad_share = float(sum(1 for b in buckets if b == "broad") / max(1, len(buckets)))
            sem_vals = [v for v in pred_semantic_proxy.get(pid, []) if not math.isnan(float(v))]
            oid = idx_to_object[pid]
            meta = object_meta.get(oid, {})
            pred_object_rows.append(
                {
                    "schema_version": "stream4d_v101_phase1_pred_object_fragment_row_v1",
                    "phase_id": "v101_phase1_f2_fragmentation_casebook",
                    "dataset_split": split,
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "window_id": chunk,
                    "mv_object_id": oid,
                    "matched_GT_count": len(gt_iou_pairs),
                    "best_GT_IoU": float(best),
                    "second_best_GT_IoU": float(second),
                    "object_frame_count": meta.get("object_frame_count", int(pred_visible_frames.get(pid, 0))),
                    "mask_count": len(pred_mask_areas.get(pid, [])),
                    "mean_mask_area": float(np.mean(pred_mask_areas.get(pid, [0]))),
                    "mean_mask_area_ratio": float(np.mean(pred_mask_area_ratios.get(pid, [0.0]))),
                    "semantic_residual_coherence": float(np.mean(sem_vals)) if sem_vals else "",
                    "semantic_residual_coherence_source": "mean_v100_semantic_norm_proxy_not_pairwise_cosine",
                    "broad_mask_share": broad_share,
                    "same_frame_collision_with_other_objects": int(duplicate_key_count),
                    "pred_pixels": int(pred_area[pid]),
                    "visible_frame_count": int(pred_visible_frames.get(pid, 0)),
                    "object_score": meta.get("object_score", ""),
                }
            )

        top_rows = sorted(
            [r for r in gt_fragment_rows if r["dataset_split"] == split and r["scene_id"] == scene and r["chunk_id"] == chunk],
            key=lambda r: (int(r["fragment_count"]), float(r["union_minus_best_IoU"]), float(r["union_pred_IoU"])),
            reverse=True,
        )[:20]
        for rank, row in enumerate(top_rows, start=1):
            case = dict(row)
            case["schema_version"] = "stream4d_v101_phase1_casebook_top_fragmented_row_v1"
            case["rank_within_window"] = rank
            casebook_rows.append(case)

    gt_count = max(1, len(gt_fragment_rows))
    gt_fragment_count_mean = float(np.mean(gt_fragment_count_values)) if gt_fragment_count_values else 0.0
    union_minus_best_mean = float(np.mean(union_minus_best_values)) if union_minus_best_values else 0.0
    ge2_rate = float(fragment_ge2_count / gt_count)
    ge3_rate = float(fragment_ge3_count / gt_count)
    gain10_rate = float(union_gain_gt_0p10_count / gt_count)
    gain20_rate = float(union_gain_gt_0p20_count / gt_count)
    threshold_summary_rows: list[dict[str, Any]] = []
    for tau, stats in threshold_stats.items():
        counts = [float(v) for v in stats["fragment_counts"]]
        gains = [float(v) for v in stats["union_gains"]]
        threshold_summary_rows.append(
            {
                "schema_version": "stream4d_v101_phase1_fragment_threshold_summary_row_v1",
                "phase_id": "v101_phase1_f2_fragmentation_casebook",
                "pred_gt_iou_threshold": tau,
                "GT_count": len(gt_fragment_rows),
                "fragment_count_mean": float(np.mean(counts)) if counts else 0.0,
                "fragment_count_p50": _p50(counts),
                "fragment_count_p90": _p90(counts),
                "fragment_count_ge2_rate": float(stats["ge2"] / gt_count),
                "union_minus_best_IoU_mean": float(np.mean(gains)) if gains else 0.0,
                "union_minus_best_IoU_p90": _p90(gains),
                "GT_with_union_minus_best_gt_0p10_rate": float(stats["gain10"] / gt_count),
            }
        )

    fragmentation_confirmed = bool(
        ge2_rate >= 0.25 or union_minus_best_mean >= 0.08 or gain10_rate >= 0.20
    )
    merge_potential_confirmed = bool(union_minus_best_mean >= 0.08 or gain10_rate >= 0.20 or gain20_rate >= 0.10)

    gate_rows = [
        {
            "gate_id": "GT_fragment_count_ge2_rate_ge_0p25",
            "pass": ge2_rate >= 0.25,
            "expected": ">=0.25",
            "observed": ge2_rate,
            "severity": "diagnostic_route",
        },
        {
            "gate_id": "union_minus_best_IoU_mean_ge_0p08",
            "pass": union_minus_best_mean >= 0.08,
            "expected": ">=0.08",
            "observed": union_minus_best_mean,
            "severity": "diagnostic_route",
        },
        {
            "gate_id": "GT_with_union_minus_best_gt_0p10_rate_ge_0p20",
            "pass": gain10_rate >= 0.20,
            "expected": ">=0.20",
            "observed": gain10_rate,
            "severity": "diagnostic_route",
        },
        {
            "gate_id": "missing_mask_frames_zero",
            "pass": total_missing_mask_frames == 0,
            "expected": 0,
            "observed": total_missing_mask_frames,
            "severity": "required_quality",
        },
        {
            "gate_id": "uses_gt_only_for_diagnostic",
            "pass": True,
            "expected": "GT used only for fragmentation labels/IoU diagnostics",
            "observed": "script does not emit method prediction rows and does not select provider/threshold",
            "severity": "required",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v101_phase1_failure_row_v1",
            "phase_id": "v101_phase1_f2_fragmentation_casebook",
            "gate_id": row["gate_id"],
            "expected": row["expected"],
            "observed": row["observed"],
            "severity": row["severity"],
        }
        for row in gate_rows
        if row["severity"].startswith("required") and not bool(row["pass"])
    ]

    decision = (
        "PASS_FRAGMENTATION_CONFIRMED_ENTER_PHASE2"
        if fragmentation_confirmed and not failure_rows
        else "PASS_FRAGMENTATION_LOW_ROUTE_LOCAL2HISTORY"
        if not fragmentation_confirmed and not failure_rows
        else "BLOCK_PHASE2_REPAIR_PHASE1_INPUTS"
    )

    gt_csv = OUT_DIR / "gt_fragment_rows.csv"
    pred_csv = OUT_DIR / "pred_object_fragment_rows.csv"
    pair_csv = OUT_DIR / "fragment_pair_rows.csv"
    overlap_csv = OUT_DIR / "pred_gt_overlap_rows.csv"
    threshold_csv = OUT_DIR / "fragment_threshold_summary_rows.csv"
    casebook_csv = OUT_DIR / "casebook_top_fragmented_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    frame_csv = OUT_DIR / "frame_audit_rows.csv"

    _write_csv(gt_csv, gt_fragment_rows)
    _write_csv(pred_csv, pred_object_rows)
    _write_csv(pair_csv, fragment_pair_rows)
    _write_csv(overlap_csv, pred_gt_overlap_rows)
    _write_csv(threshold_csv, threshold_summary_rows)
    _write_csv(casebook_csv, casebook_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_csv(frame_csv, frame_audit_rows)

    summary = {
        "schema_version": "stream4d_v101_phase1_f2_fragmentation_casebook_summary_v1",
        "phase_id": "v101_phase1_f2_fragmentation_casebook",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase1_pass": not failure_rows,
        "fragmentation_confirmed": fragmentation_confirmed,
        "merge_potential_confirmed": merge_potential_confirmed,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "gt_fragment_row_count": len(gt_fragment_rows),
        "pred_object_fragment_row_count": len(pred_object_rows),
        "fragment_pair_row_count": len(fragment_pair_rows),
        "pred_gt_overlap_row_count": len(pred_gt_overlap_rows),
        "casebook_row_count": len(casebook_rows),
        "frame_count": total_frame_count,
        "gt_window_object_count": total_gt_count,
        "pred_window_object_count": total_pred_count,
        "duplicate_mask_object_key_count": total_duplicate_mask_object_keys,
        "missing_mask_frame_count": total_missing_mask_frames,
        "GT_fragment_count_mean": gt_fragment_count_mean,
        "GT_fragment_count_p50": _p50(gt_fragment_count_values),
        "GT_fragment_count_p90": _p90(gt_fragment_count_values),
        "GT_fragment_count_ge2_rate": ge2_rate,
        "GT_fragment_count_ge3_rate": ge3_rate,
        "union_minus_best_IoU_mean": union_minus_best_mean,
        "union_minus_best_IoU_p90": _p90(union_minus_best_values),
        "GT_with_union_minus_best_gt_0p10_rate": gain10_rate,
        "GT_with_union_minus_best_gt_0p20_rate": gain20_rate,
        "best_pred_IoU_mean": float(np.mean(best_iou_values)) if best_iou_values else 0.0,
        "union_pred_IoU_mean": float(np.mean(union_iou_values)) if union_iou_values else 0.0,
        "fragment_threshold_summary": threshold_summary_rows,
        "mask_area_bucket_policy": {
            "small": f"<{OBJECT_LIKE_MIN_AREA_RATIO}",
            "object_like": f"[{OBJECT_LIKE_MIN_AREA_RATIO},{BROAD_MASK_AREA_RATIO})",
            "broad": f">={BROAD_MASK_AREA_RATIO}",
            "unit": "mask_area_ratio_of_frame_pixels",
        },
        "semantic_residual_coherence_note": "pred_object rows use mean v100_semantic_norm as a proxy because pairwise RADIO residual features are not stored in Phase2c rows.",
        "config": {
            "phase0_summary": _rel(PHASE0_DIR / "summary.json"),
            "phase2c_mv_object_frame_mask_rows": _rel(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet"),
            "phase2c_mv_object_rows": _rel(PHASE2C_DIR / "mv_object_rows.parquet"),
            "phase2c_frame_eval_rows": _rel(PHASE2C_DIR / "frame_eval_rows.csv"),
            "plan_doc": _rel(PLAN_DOC),
            "OBJECT_LIKE_MIN_AREA_RATIO": OBJECT_LIKE_MIN_AREA_RATIO,
            "BROAD_MASK_AREA_RATIO": BROAD_MASK_AREA_RATIO,
            "config_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "script": _sha256(Path(__file__)),
                        "phase2c_masks": _sha256(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet"),
                        "phase2c_objects": _sha256(PHASE2C_DIR / "mv_object_rows.parquet"),
                        "phase2c_frame_eval": _sha256(PHASE2C_DIR / "frame_eval_rows.csv"),
                        "object_like_min": OBJECT_LIKE_MIN_AREA_RATIO,
                        "broad": BROAD_MASK_AREA_RATIO,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "gt_fragment_rows": _rel(gt_csv),
            "pred_object_fragment_rows": _rel(pred_csv),
            "fragment_pair_rows": _rel(pair_csv),
            "pred_gt_overlap_rows": _rel(overlap_csv),
            "fragment_threshold_summary_rows": _rel(threshold_csv),
            "casebook_top_fragmented_rows": _rel(casebook_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "frame_audit_rows": _rel(frame_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
