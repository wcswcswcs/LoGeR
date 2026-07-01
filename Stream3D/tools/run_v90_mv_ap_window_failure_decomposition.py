from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "outputs/audit/v90_phase1_variant_resurrection"
PHASE0 = ROOT / "outputs/audit/v90_phase0_mv_ap_contract"
OUT = ROOT / "outputs/audit/v90_phase2_failure_decomposition"

KEY_VARIANTS = [
    "B0_local_only",
    "R10_v82_local_B0_object_slot_config",
    "C0_semantic_only_control",
    "S3D_L1_local_merged_masks",
]


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def _rank_average(values: list[float], *, reverse: bool = True) -> list[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=reverse)
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for src_idx, _ in indexed[idx:end]:
            ranks[src_idx] = float(avg_rank)
        idx = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    rx = np.asarray(_rank_average([p[0] for p in pairs], reverse=False), dtype=np.float64)
    ry = np.asarray(_rank_average([p[1] for p in pairs], reverse=False), dtype=np.float64)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _max_cardinality_bipartite(edges: dict[int, set[int]]) -> int:
    match_gt_to_pred: dict[int, int] = {}

    def visit(pred_id: int, seen: set[int]) -> bool:
        for gt_id in sorted(edges.get(pred_id, set())):
            if gt_id in seen:
                continue
            seen.add(gt_id)
            current = match_gt_to_pred.get(gt_id)
            if current is None or visit(current, seen):
                match_gt_to_pred[gt_id] = pred_id
                return True
        return False

    matched = 0
    for pred_id in sorted(edges):
        if visit(pred_id, set()):
            matched += 1
    return matched


def _match_f1(tp: int, pred_count: int, gt_count: int) -> dict[str, Any]:
    fp = int(pred_count) - int(tp)
    fn = int(gt_count) - int(tp)
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "pred_count": int(pred_count),
        "gt_count": int(gt_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _window_id_from_mv_object_id(mv_object_id: str) -> str:
    match = re.search(r"\bw\d{4}\b", str(mv_object_id))
    return match.group(0) if match else ""


def _window_index_from_window_id(window_id: str) -> int:
    if not window_id.startswith("w"):
        return -1
    return _int(window_id[1:], -1)


def _gt_window_ranges(window_rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        scene = row.get("scene_id", "")
        by_scene[scene].append(
            {
                "window_id": row.get("window_id", ""),
                "window_index": _int(row.get("window_index"), -1),
                "gt_count": _int(row.get("window_scoped_gt_count"), 0),
            }
        )
    ranges: dict[str, list[dict[str, Any]]] = {}
    for scene, rows in by_scene.items():
        start = 1
        scene_ranges = []
        for row in sorted(rows, key=lambda r: int(r["window_index"])):
            count = int(row["gt_count"])
            end = start + count - 1
            scene_ranges.append({**row, "gt_id_start": start, "gt_id_end": end})
            start = end + 1
        ranges[scene] = scene_ranges
    return ranges


def _gt_window(scene_ranges: dict[str, list[dict[str, Any]]], scene: str, gt_id: int) -> tuple[str, int]:
    for row in scene_ranges.get(scene, []):
        if int(row["gt_id_start"]) <= int(gt_id) <= int(row["gt_id_end"]):
            return str(row["window_id"]), int(row["window_index"])
    return "", -1


def _load_object_scores(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row.get("variant_id", ""), row.get("scene_id", ""), row.get("mv_object_id", ""))
        if not all(key):
            continue
        if row.get("object_score", "") == "":
            continue
        buckets[key].append(_num(row.get("object_score")))
    for key, vals in buckets.items():
        out[key] = float(np.mean(vals))
    return out


def _load_metric_maps(metric_rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, list[dict[str, str]]]]:
    by_variant_scene: dict[tuple[str, str], dict[str, str]] = {}
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        variant = row.get("variant_id") or row.get("variant")
        scene = row.get("scene_id", "")
        if not variant or not scene:
            continue
        by_variant_scene[(variant, scene)] = row
        by_variant[variant].append(row)
    return by_variant_scene, by_variant


def _load_aggregate_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        variant = row.get("variant_id", "")
        if variant:
            out[variant] = row
    return out


def _collision_counts(frame_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, int]]:
    key_to_objects: dict[tuple[str, str, str, int, int], set[str]] = defaultdict(set)
    for row in frame_rows:
        variant = row.get("variant_id", "")
        scene = row.get("scene_id", "")
        window = row.get("window_id", "") or _window_id_from_mv_object_id(row.get("mv_object_id", ""))
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_id"), -1)
        obj = row.get("mv_object_id", "")
        if not variant or not scene or frame_id < 0 or mask_id <= 0 or not obj:
            continue
        key_to_objects[(variant, scene, window, frame_id, mask_id)].add(obj)
    out: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"same_frame_duplicate_count": 0, "same_mask_ownership_conflict_count": 0})
    for (variant, scene, window, _frame_id, _mask_id), objects in key_to_objects.items():
        if len(objects) > 1:
            out[(variant, scene, window)]["same_frame_duplicate_count"] += len(objects) - 1
            out[(variant, scene, window)]["same_mask_ownership_conflict_count"] += 1
    return out


def _stream_iou_stats(
    matrix_path: Path,
    object_scores: dict[tuple[str, str, str], float],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, int]],
    dict[tuple[str, str], dict[str, Any]],
]:
    gt_stats: dict[tuple[str, str, int], dict[str, Any]] = {}
    pred_stats: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    pair_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"pair_count": 0, "zero_iou_pair_count": 0})
    edge_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "pred_ids": set(),
            "gt_ids": set(),
            "edges25": defaultdict(set),
            "edges50": defaultdict(set),
        }
    )
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variant = row.get("variant_id") or row.get("variant", "")
            scene = row.get("scene_id", "")
            pred_id = _int(row.get("pred_id"), -1)
            mv_object_id = row.get("mv_object_id", "")
            gt_id = _int(row.get("gt_id"), -1)
            iou = _num(row.get("mv_iou", row.get("iou")), 0.0)
            if not variant or not scene or pred_id < 0 or gt_id < 0:
                continue
            pair_key = (variant, scene)
            pair_stats[pair_key]["pair_count"] += 1
            if iou <= 0.0:
                pair_stats[pair_key]["zero_iou_pair_count"] += 1
            edge_stats[pair_key]["pred_ids"].add(int(pred_id))
            edge_stats[pair_key]["gt_ids"].add(int(gt_id))
            if iou >= 0.25:
                edge_stats[pair_key]["edges25"][int(pred_id)].add(int(gt_id))
            if iou >= 0.50:
                edge_stats[pair_key]["edges50"][int(pred_id)].add(int(gt_id))
            gkey = (variant, scene, gt_id)
            gstat = gt_stats.setdefault(
                gkey,
                {
                    "best_iou": -1.0,
                    "best_pred_id": "",
                    "best_pred_object_id": "",
                    "best_pred_score": "",
                    "GT_to_pred_count_25": 0,
                    "GT_to_pred_count_50": 0,
                },
            )
            if iou >= 0.25:
                gstat["GT_to_pred_count_25"] += 1
            if iou >= 0.50:
                gstat["GT_to_pred_count_50"] += 1
            if iou > float(gstat["best_iou"]):
                score = object_scores.get((variant, scene, mv_object_id))
                gstat.update(
                    {
                        "best_iou": float(iou),
                        "best_pred_id": int(pred_id),
                        "best_pred_object_id": mv_object_id,
                        "best_pred_score": "" if score is None else float(score),
                    }
                )
            pkey = (variant, scene, pred_id, mv_object_id)
            pstat = pred_stats.setdefault(
                pkey,
                {
                    "best_iou": -1.0,
                    "best_gt_id": "",
                    "pred_to_GT_count_25": 0,
                    "pred_to_GT_count_50": 0,
                    "object_score": object_scores.get((variant, scene, mv_object_id), ""),
                },
            )
            if iou >= 0.25:
                pstat["pred_to_GT_count_25"] += 1
            if iou >= 0.50:
                pstat["pred_to_GT_count_50"] += 1
            if iou > float(pstat["best_iou"]):
                pstat.update({"best_iou": float(iou), "best_gt_id": int(gt_id)})
    match_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key, stat in edge_stats.items():
        pred_count = len(stat["pred_ids"])
        gt_count = len(stat["gt_ids"])
        tp25 = _max_cardinality_bipartite(stat["edges25"])
        tp50 = _max_cardinality_bipartite(stat["edges50"])
        match_stats[key] = {
            "score_free_Match25_window": _match_f1(tp25, pred_count, gt_count),
            "score_free_Match50_window": _match_f1(tp50, pred_count, gt_count),
        }
    return gt_stats, pred_stats, pair_stats, match_stats


def _build_gt_rows(
    gt_stats: dict[tuple[str, str, int], dict[str, Any]],
    gt_ranges: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for (variant, scene, gt_id), stat in sorted(gt_stats.items()):
        window_id, window_index = _gt_window(gt_ranges, scene, gt_id)
        best_iou = max(0.0, float(stat["best_iou"]))
        best_pred_object_id = "" if best_iou <= 0.0 else stat["best_pred_object_id"]
        best_pred_score = "" if best_iou <= 0.0 else stat["best_pred_score"]
        rows.append(
            {
                "variant_id": variant,
                "scene_id": scene,
                "window_id": window_id,
                "window_index": window_index,
                "gt_object_id": int(gt_id),
                "gt_area_sum": "",
                "best_pred_object_id": best_pred_object_id,
                "best_iou": best_iou,
                "best_pred_score": best_pred_score,
                "matched_at_25": bool(best_iou >= 0.25),
                "matched_at_50": bool(best_iou >= 0.50),
                "missing_frame_support_rate": "",
                "undercoverage_rate": "",
                "overcoverage_rate": "",
                "GT_to_pred_count_25": int(stat["GT_to_pred_count_25"]),
                "GT_to_pred_count_50": int(stat["GT_to_pred_count_50"]),
                "area_terms_note": "area_terms_unavailable_from_phase1_iou_matrix; not used for blocker labels",
            }
        )
    return rows


def _build_pred_rows(
    pred_stats: dict[tuple[str, str, int, str], dict[str, Any]],
    gt_ranges: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for (variant, scene, pred_id, mv_object_id), stat in sorted(pred_stats.items()):
        pred_window_id = _window_id_from_mv_object_id(mv_object_id)
        best_gt = _int(stat["best_gt_id"], -1)
        gt_window_id, gt_window_index = _gt_window(gt_ranges, scene, best_gt)
        best_iou = max(0.0, float(stat["best_iou"]))
        if best_iou <= 0.0:
            best_gt = -1
            gt_window_id, gt_window_index = "", -1
        rows.append(
            {
                "variant_id": variant,
                "scene_id": scene,
                "window_id": pred_window_id,
                "window_index": _window_index_from_window_id(pred_window_id),
                "pred_id": int(pred_id),
                "pred_object_id": mv_object_id,
                "object_score": stat.get("object_score", ""),
                "best_gt_object_id": best_gt if best_gt >= 0 else "",
                "best_gt_window_id": gt_window_id,
                "best_gt_window_index": gt_window_index,
                "best_iou": best_iou,
                "matched_at_25": bool(best_iou >= 0.25),
                "matched_at_50": bool(best_iou >= 0.50),
                "pred_to_GT_count_25": int(stat["pred_to_GT_count_25"]),
                "pred_to_GT_count_50": int(stat["pred_to_GT_count_50"]),
            }
        )
    return rows


def _build_grouping_rows(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    collision_map: dict[tuple[str, str, str], dict[str, int]],
    metric_map: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in gt_rows:
        key = (row["variant_id"], row["scene_id"], row["window_id"])
        collisions = collision_map.get(key, {})
        count25 = int(row["GT_to_pred_count_25"])
        if float(row["best_iou"]) < 0.25:
            err = "gt_uncovered_at_25"
        elif count25 > 1:
            err = "gt_fragmented_at_25"
        else:
            err = "gt_ok_at_25"
        rows.append(
            {
                "variant_id": row["variant_id"],
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "window_index": row["window_index"],
                "object_id_or_gt_id": row["gt_object_id"],
                "error_type": err,
                "GT_to_pred_count": count25,
                "pred_to_GT_count": "",
                "fragmentation_score": max(0, count25 - 1),
                "overmerge_score": "",
                "same_frame_duplicate_count": int(collisions.get("same_frame_duplicate_count", 0)),
                "same_mask_ownership_conflict_count": int(collisions.get("same_mask_ownership_conflict_count", 0)),
                "pre_wta_duplicate_frame_mask_conflict_count": _int(metric_map.get((row["variant_id"], row["scene_id"]), {}).get("duplicate_frame_mask_conflict_count"), 0),
                "threshold": 0.25,
            }
        )
    for row in pred_rows:
        key = (row["variant_id"], row["scene_id"], row["window_id"])
        collisions = collision_map.get(key, {})
        count25 = int(row["pred_to_GT_count_25"])
        if float(row["best_iou"]) < 0.25:
            err = "pred_unmatched_at_25"
        elif count25 > 1:
            err = "pred_overmerged_at_25"
        else:
            err = "pred_ok_at_25"
        rows.append(
            {
                "variant_id": row["variant_id"],
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "window_index": row["window_index"],
                "object_id_or_gt_id": row["pred_object_id"],
                "error_type": err,
                "GT_to_pred_count": "",
                "pred_to_GT_count": count25,
                "fragmentation_score": "",
                "overmerge_score": max(0, count25 - 1),
                "same_frame_duplicate_count": int(collisions.get("same_frame_duplicate_count", 0)),
                "same_mask_ownership_conflict_count": int(collisions.get("same_mask_ownership_conflict_count", 0)),
                "pre_wta_duplicate_frame_mask_conflict_count": _int(metric_map.get((row["variant_id"], row["scene_id"]), {}).get("duplicate_frame_mask_conflict_count"), 0),
                "threshold": 0.25,
            }
        )
    return rows


def _build_ranking_rows(pred_rows: list[dict[str, Any]], metric_map: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float | None]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        grouped[(row["variant_id"], row["scene_id"])].append(row)
    out: list[dict[str, Any]] = []
    spearman: dict[tuple[str, str], float | None] = {}
    for key, rows in grouped.items():
        valid_scores = [_num(r["object_score"], float("nan")) for r in rows]
        ious = [_num(r["best_iou"], 0.0) for r in rows]
        score_ranks = _rank_average([(-float("inf") if math.isnan(v) else v) for v in valid_scores], reverse=True)
        iou_ranks = _rank_average(ious, reverse=True)
        finite_scores = [v for v in valid_scores if not math.isnan(v)]
        finite_ious = [ious[idx] for idx, v in enumerate(valid_scores) if not math.isnan(v)]
        spearman[key] = _spearman(finite_scores, finite_ious)
        metric = metric_map.get(key, {})
        score_protocol = (
            f"score_mode={metric.get('score_mode', '')}; "
            f"score_unique_count={metric.get('score_unique_count', '')}; "
            "rank=1 is highest"
        )
        for idx, row in enumerate(rows):
            score = valid_scores[idx]
            out.append(
                {
                    "variant_id": row["variant_id"],
                    "scene_id": row["scene_id"],
                    "window_id": row["window_id"],
                    "window_index": row["window_index"],
                    "pred_object_id": row["pred_object_id"],
                    "object_score": "" if math.isnan(score) else score,
                    "best_gt_iou": row["best_iou"],
                    "score_rank": score_ranks[idx],
                    "best_iou_rank": iou_ranks[idx],
                    "score_rank_minus_iou_rank": float(score_ranks[idx] - iou_ranks[idx]),
                    "score_protocol": score_protocol,
                }
            )
    return out, spearman


def _variant_summary(
    variant: str,
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    grouping_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, str]],
    aggregate_map: dict[str, dict[str, str]],
    spearman: dict[tuple[str, str], float | None],
    match_stats: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    gt = [row for row in gt_rows if row["variant_id"] == variant]
    pred = [row for row in pred_rows if row["variant_id"] == variant]
    group = [row for row in grouping_rows if row["variant_id"] == variant]
    metric = [row for row in metric_rows if (row.get("variant_id") or row.get("variant")) == variant]
    scenes = sorted({row["scene_id"] for row in gt})
    sp_vals = [spearman.get((variant, scene)) for scene in scenes]
    sp_vals_f = [float(v) for v in sp_vals if v is not None and math.isfinite(float(v))]
    match25_vals = [
        _num(match_stats.get((variant, scene), {}).get("score_free_Match25_window", {}).get("f1"))
        for scene in scenes
    ]
    match50_vals = [
        _num(match_stats.get((variant, scene), {}).get("score_free_Match50_window", {}).get("f1"))
        for scene in scenes
    ]
    agg = aggregate_map.get(variant, {})
    match50_mean = _mean(match50_vals)
    return {
        "variant_id": variant,
        "MV_AP_window": _num(agg.get("mean_MV_AP_window"), None),
        "MV_AP50_window": _num(agg.get("mean_MV_AP50_window"), None),
        "score_free_Match50_window": match50_mean,
        "score_free_Match25_window": _mean(match25_vals),
        "MV_AP50_to_Match50_gap": _num(match50_mean) - _num(agg.get("mean_MV_AP50_window")),
        "mean_GT_best_IoU_window": _mean([_num(row["best_iou"]) for row in gt]),
        "median_GT_best_IoU_window": _median([_num(row["best_iou"]) for row in gt]),
        "gt_recall_best_iou_ge_25": _mean([1.0 if _num(row["best_iou"]) >= 0.25 else 0.0 for row in gt]),
        "gt_recall_best_iou_ge_50": _mean([1.0 if _num(row["best_iou"]) >= 0.50 else 0.0 for row in gt]),
        "pred_recall_best_iou_ge_25": _mean([1.0 if _num(row["best_iou"]) >= 0.25 else 0.0 for row in pred]),
        "pred_recall_best_iou_ge_50": _mean([1.0 if _num(row["best_iou"]) >= 0.50 else 0.0 for row in pred]),
        "fragmentation_rate": _mean([1.0 if row["error_type"] == "gt_fragmented_at_25" else 0.0 for row in group if str(row["error_type"]).startswith("gt_")]),
        "overmerge_rate": _mean([1.0 if row["error_type"] == "pred_overmerged_at_25" else 0.0 for row in group if str(row["error_type"]).startswith("pred_")]),
        "undercoverage_rate": _mean([1.0 if _num(row["best_iou"]) < 0.25 else 0.0 for row in gt]),
        "overcoverage_rate": _mean([1.0 if row["error_type"] == "pred_overmerged_at_25" else 0.0 for row in group if str(row["error_type"]).startswith("pred_")]),
        "score_vs_best_IoU_spearman": float(np.mean(sp_vals_f)) if sp_vals_f else None,
        "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in metric)),
        "missing_mask_raster_count": int(sum(_int(row.get("missing_mask_raster_count")) for row in metric)),
        "pre_wta_duplicate_frame_mask_conflict_count": int(sum(_int(row.get("duplicate_frame_mask_conflict_count")) for row in metric)),
        "pred_object_count_mean": _mean([_num(row.get("pred_object_count")) for row in metric]),
        "gt_object_count_mean": _mean([_num(row.get("gt_object_count")) for row in metric]),
    }


def _build_control_bias_rows(
    phase1_summary: dict[str, Any],
    metric_map: dict[tuple[str, str], dict[str, str]],
    aggregate_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    best_real = phase1_summary.get("best_real_variant", "")
    best_control = phase1_summary.get("best_control_variant", "")
    rows: list[dict[str, Any]] = []
    real_agg = aggregate_map.get(best_real, {})
    control_agg = aggregate_map.get(best_control, {})
    rows.append(
        {
            "scope": "aggregate_dev",
            "scene_id": "",
            "best_real_variant": best_real,
            "best_real_MV_AP_window": _num(real_agg.get("mean_MV_AP_window")),
            "best_control_variant": best_control,
            "best_control_MV_AP_window": _num(control_agg.get("mean_MV_AP_window")),
            "best_control_minus_best_real_MV_AP_window": _num(control_agg.get("mean_MV_AP_window")) - _num(real_agg.get("mean_MV_AP_window")),
            "control_bias_blocker": bool(_num(control_agg.get("mean_MV_AP_window")) >= _num(real_agg.get("mean_MV_AP_window"))),
        }
    )
    scenes = sorted({scene for variant, scene in metric_map if variant in {best_real, best_control}})
    for scene in scenes:
        real = metric_map.get((best_real, scene), {})
        control = metric_map.get((best_control, scene), {})
        rows.append(
            {
                "scope": "scene_dev",
                "scene_id": scene,
                "best_real_variant": best_real,
                "best_real_MV_AP_window": _num(real.get("MV_AP_window", real.get("MV_AP"))),
                "best_control_variant": best_control,
                "best_control_MV_AP_window": _num(control.get("MV_AP_window", control.get("MV_AP"))),
                "best_control_minus_best_real_MV_AP_window": _num(control.get("MV_AP_window", control.get("MV_AP"))) - _num(real.get("MV_AP_window", real.get("MV_AP"))),
                "control_bias_blocker": bool(_num(control.get("MV_AP_window", control.get("MV_AP"))) >= _num(real.get("MV_AP_window", real.get("MV_AP")))),
            }
        )
    return rows


def _build_casebook_rows(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    variants: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in variants:
        hard_gts = [row for row in gt_rows if row["variant_id"] == variant and _num(row["best_iou"]) < 0.50]
        hard_gts.sort(key=lambda row: (_num(row["best_iou"]), row["scene_id"], row["gt_object_id"]))
        for row in hard_gts[:50]:
            out.append(
                {
                    "variant_id": variant,
                    "scene_id": row["scene_id"],
                    "window_id": row["window_id"],
                    "case_type": "gt_best_iou_below_50",
                    "object_id": row["gt_object_id"],
                    "best_iou": row["best_iou"],
                    "evidence": f"best_pred={row['best_pred_object_id']}; matched25={row['matched_at_25']}; matched50={row['matched_at_50']}",
                    "next_debug_hint": "inspect local mask extent/object grouping before local2history",
                }
            )
        high_score_low_iou = [
            row
            for row in ranking_rows
            if row["variant_id"] == variant and _num(row["score_rank"], 1e9) <= 30 and _num(row["best_gt_iou"]) < 0.25
        ]
        high_score_low_iou.sort(key=lambda row: (_num(row["score_rank"]), -_num(row["score_rank_minus_iou_rank"])))
        for row in high_score_low_iou[:30]:
            out.append(
                {
                    "variant_id": variant,
                    "scene_id": row["scene_id"],
                    "window_id": row["window_id"],
                    "case_type": "high_score_low_iou",
                    "object_id": row["pred_object_id"],
                    "best_iou": row["best_gt_iou"],
                    "evidence": f"score_rank={row['score_rank']}; best_iou_rank={row['best_iou_rank']}; score={row['object_score']}",
                    "next_debug_hint": "score/readout calibration is harming AP precision envelope",
                }
            )
        low_score_good_iou = [
            row
            for row in ranking_rows
            if row["variant_id"] == variant
            and _num(row["best_gt_iou"]) >= 0.50
            and _num(row["score_rank_minus_iou_rank"], 0.0) > 50.0
        ]
        low_score_good_iou.sort(key=lambda row: (-_num(row["score_rank_minus_iou_rank"]), -_num(row["best_gt_iou"])))
        for row in low_score_good_iou[:30]:
            out.append(
                {
                    "variant_id": variant,
                    "scene_id": row["scene_id"],
                    "window_id": row["window_id"],
                    "case_type": "low_score_good_iou",
                    "object_id": row["pred_object_id"],
                    "best_iou": row["best_gt_iou"],
                    "evidence": f"score_rank={row['score_rank']}; best_iou_rank={row['best_iou_rank']}; score={row['object_score']}",
                    "next_debug_hint": "good local objects are ranked too late for AP",
                }
            )
        unmatched_preds = [row for row in pred_rows if row["variant_id"] == variant and _num(row["best_iou"]) < 0.25]
        unmatched_preds.sort(key=lambda row: (-_num(row["object_score"], 0.0), row["scene_id"], row["pred_object_id"]))
        for row in unmatched_preds[:30]:
            out.append(
                {
                    "variant_id": variant,
                    "scene_id": row["scene_id"],
                    "window_id": row["window_id"],
                    "case_type": "pred_unmatched_at_25",
                    "object_id": row["pred_object_id"],
                    "best_iou": row["best_iou"],
                    "evidence": f"object_score={row['object_score']}; pred_to_GT_count_25={row['pred_to_GT_count_25']}",
                    "next_debug_hint": "remove/support-gate unmatched predictions or repair extent",
                }
            )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    output_root = Path(args.output_root)
    matrix_path = Path(args.phase1_root) / "mv_iou_matrix_rows.csv"
    phase1_root = Path(args.phase1_root)
    phase0_root = Path(args.phase0_root)
    phase1_summary = json.loads((phase1_root / "summary.json").read_text(encoding="utf-8"))
    phase0_summary = json.loads((phase0_root / "summary.json").read_text(encoding="utf-8"))
    window_rows = _read_csv(phase0_root / "window_support_rows.csv")
    object_rows = _read_csv(phase1_root / "mv_object_rows.csv")
    frame_rows = _read_csv(phase1_root / "mv_object_frame_mask_rows.csv")
    metric_rows = _read_csv(phase1_root / "mv_metric_rows.csv")
    aggregate_rows = _read_csv(phase1_root / "mv_metric_aggregate_rows.csv")

    gt_ranges = _gt_window_ranges(window_rows)
    object_scores = _load_object_scores(object_rows)
    metric_map, _metric_by_variant = _load_metric_maps(metric_rows)
    aggregate_map = _load_aggregate_rows(aggregate_rows)
    collision_map = _collision_counts(frame_rows)

    gt_stats, pred_stats, pair_stats, match_stats = _stream_iou_stats(matrix_path, object_scores)
    gt_rows = _build_gt_rows(gt_stats, gt_ranges)
    pred_rows = _build_pred_rows(pred_stats, gt_ranges)
    grouping_rows = _build_grouping_rows(gt_rows, pred_rows, collision_map, metric_map)
    ranking_rows, spearman_by_scene = _build_ranking_rows(pred_rows, metric_map)
    control_bias_rows = _build_control_bias_rows(phase1_summary, metric_map, aggregate_map)

    variants = sorted({row["variant_id"] for row in gt_rows})
    variant_summaries = {
        variant: _variant_summary(variant, gt_rows, pred_rows, grouping_rows, metric_rows, aggregate_map, spearman_by_scene, match_stats)
        for variant in variants
    }
    casebook_variants = [v for v in KEY_VARIANTS if v in variant_summaries]
    best_real = str(phase1_summary.get("best_real_variant", ""))
    if best_real and best_real not in casebook_variants:
        casebook_variants.append(best_real)
    casebook_rows = _build_casebook_rows(gt_rows, pred_rows, ranking_rows, casebook_variants)

    best_control = str(phase1_summary.get("best_control_variant", ""))
    best_real_summary = variant_summaries.get(best_real, {})
    best_control_summary = variant_summaries.get(best_control, {})
    b0_summary = variant_summaries.get("B0_local_only", {})
    s3d_summary = variant_summaries.get("S3D_L1_local_merged_masks", {})

    labels: list[str] = []
    label_evidence: dict[str, Any] = {}
    same_frame_collision_count = _int(best_real_summary.get("same_frame_collision_count"), 0)
    missing_mask_raster_count = _int(best_real_summary.get("missing_mask_raster_count"), 0)
    if same_frame_collision_count > 0 or missing_mask_raster_count > 0 or not _bool(phase0_summary.get("formal_metric_source_eq_v65")):
        labels.append("SUPPORT_OR_EVALUATOR_BUG")
        label_evidence["SUPPORT_OR_EVALUATOR_BUG"] = {
            "same_frame_collision_count": same_frame_collision_count,
            "missing_mask_raster_count": missing_mask_raster_count,
            "formal_metric_source_eq_v65": phase0_summary.get("formal_metric_source_eq_v65"),
        }

    control_delta = _num(best_control_summary.get("MV_AP_window")) - _num(best_real_summary.get("MV_AP_window"))
    if control_delta >= 0.0:
        labels.append("CONTROL_BIAS_BLOCKER")
        label_evidence["CONTROL_BIAS_BLOCKER"] = {
            "best_control_variant": best_control,
            "best_real_variant": best_real,
            "best_control_minus_best_real_MV_AP_window": control_delta,
        }

    mean_gt_iou = _num(best_real_summary.get("mean_GT_best_IoU_window"))
    match50 = _num(best_real_summary.get("score_free_Match50_window"))
    if match50 < 0.35 and mean_gt_iou < 0.35:
        labels.append("EXTENT_BLOCKER")
        label_evidence["EXTENT_BLOCKER"] = {
            "criterion": "score_free_Match50_window < 0.35 and mean_GT_best_IoU_window < 0.35",
            "score_free_Match50_window": match50,
            "mean_GT_best_IoU_window": mean_gt_iou,
        }

    fragmentation_rate = _num(best_real_summary.get("fragmentation_rate"))
    overmerge_rate = _num(best_real_summary.get("overmerge_rate"))
    if fragmentation_rate >= 0.25 or overmerge_rate >= 0.25:
        labels.append("GROUPING_BLOCKER")
        label_evidence["GROUPING_BLOCKER"] = {
            "criterion": "fragmentation_rate >= 0.25 or overmerge_rate >= 0.25 at IoU 0.25",
            "fragmentation_rate": fragmentation_rate,
            "overmerge_rate": overmerge_rate,
        }

    rank_gap = _num(best_real_summary.get("MV_AP50_to_Match50_gap"))
    if rank_gap >= 0.10:
        labels.append("RANKING_BLOCKER")
        label_evidence["RANKING_BLOCKER"] = {
            "criterion": "score_free_Match50_window - MV_AP50_window >= 0.10",
            "MV_AP50_to_Match50_gap": rank_gap,
            "score_vs_best_IoU_spearman": best_real_summary.get("score_vs_best_IoU_spearman"),
        }

    if "SUPPORT_OR_EVALUATOR_BUG" in labels:
        recommended_next_phase = "repair_evaluator_or_materializer_before_algorithm_changes"
    elif "EXTENT_BLOCKER" in labels:
        recommended_next_phase = "Phase3_carrier_supported_mask_carving"
    elif "GROUPING_BLOCKER" in labels:
        recommended_next_phase = "Phase5_signed_constrained_clustering"
    elif "RANKING_BLOCKER" in labels:
        recommended_next_phase = "Phase7_score_calibration"
    elif "CONTROL_BIAS_BLOCKER" in labels:
        recommended_next_phase = "control_resistant_readout_with_new_controls"
    else:
        recommended_next_phase = "Phase3_per_plan_minimum_repair_ladder"

    output_root.mkdir(parents=True, exist_ok=True)
    gt_fields = [
        "variant_id",
        "scene_id",
        "window_id",
        "window_index",
        "gt_object_id",
        "gt_area_sum",
        "best_pred_object_id",
        "best_iou",
        "best_pred_score",
        "matched_at_25",
        "matched_at_50",
        "missing_frame_support_rate",
        "undercoverage_rate",
        "overcoverage_rate",
        "GT_to_pred_count_25",
        "GT_to_pred_count_50",
        "area_terms_note",
    ]
    _write_csv(output_root / "gt_top_iou_rows.csv", gt_rows, gt_fields)
    _write_csv(output_root / "pred_top_iou_rows.csv", pred_rows)
    _write_csv(output_root / "grouping_error_rows.csv", grouping_rows)
    _write_csv(output_root / "ranking_error_rows.csv", ranking_rows)
    _write_csv(output_root / "control_bias_rows.csv", control_bias_rows)
    _write_csv(output_root / "failure_casebook_rows.csv", casebook_rows)

    summary = {
        "phase": "v90_phase2_failure_decomposition",
        "schema": "stream4d_v90_phase2_failure_decomposition_v1",
        "phase2_pass": True,
        "runtime_sec": time.time() - t0,
        "metric_source": "Phase1 MV_AP rows are from tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou; Phase2 decomposes Phase1 full pred x GT IoU matrix.",
        "support_policy": phase1_summary.get("support_policy"),
        "inputs": {
            "phase1_summary": _rel(phase1_root / "summary.json"),
            "mv_iou_matrix_rows": _rel(matrix_path),
            "mv_object_rows": _rel(phase1_root / "mv_object_rows.csv"),
            "mv_object_frame_mask_rows": _rel(phase1_root / "mv_object_frame_mask_rows.csv"),
            "window_support_rows": _rel(phase0_root / "window_support_rows.csv"),
        },
        "row_counts": {
            "gt_top_iou_rows": len(gt_rows),
            "pred_top_iou_rows": len(pred_rows),
            "grouping_error_rows": len(grouping_rows),
            "ranking_error_rows": len(ranking_rows),
            "control_bias_rows": len(control_bias_rows),
            "failure_casebook_rows": len(casebook_rows),
        },
        "pair_stats": pair_stats,
        "score_free_match_stats_by_scene": match_stats,
        "best_real_variant": best_real,
        "best_control_variant": best_control,
        "variant_summaries": {
            key: variant_summaries[key]
            for key in sorted(set(KEY_VARIANTS + [best_real, best_control]))
            if key in variant_summaries
        },
        "required_metrics": {
            "best_real": best_real_summary,
            "best_control_minus_best_real_MV_AP_window": control_delta,
            "B0_to_Stream3D_gap": _num(s3d_summary.get("MV_AP_window")) - _num(b0_summary.get("MV_AP_window")),
        },
        "blocker_labels": labels,
        "blocker_label_evidence": label_evidence,
        "recommended_next_phase": recommended_next_phase,
        "area_terms_status": "gt_area_sum/missing_frame_support_rate/undercoverage_rate/overcoverage_rate are not derivable from Phase1 IoU-only matrix; output fields are present but blank and not used for blocker labels. Re-run evaluator with pred_area/gt_area/intersection dump if an extent subdiagnosis needs area decomposition.",
        "pre_wta_duplicate_conflict_note": "duplicate_frame_mask_conflict_count is retained as adapter/pre-WTA diagnostic; SUPPORT_OR_EVALUATOR_BUG uses same_frame_collision_count after materialized WTA plus missing raster/v65 sanity.",
        "outputs": {
            "gt_top_iou_rows": _rel(output_root / "gt_top_iou_rows.csv"),
            "pred_top_iou_rows": _rel(output_root / "pred_top_iou_rows.csv"),
            "grouping_error_rows": _rel(output_root / "grouping_error_rows.csv"),
            "ranking_error_rows": _rel(output_root / "ranking_error_rows.csv"),
            "control_bias_rows": _rel(output_root / "control_bias_rows.csv"),
            "failure_casebook_rows": _rel(output_root / "failure_casebook_rows.csv"),
        },
    }
    _write_json(output_root / "summary.json", summary)
    sha_paths = [
        output_root / "gt_top_iou_rows.csv",
        output_root / "pred_top_iou_rows.csv",
        output_root / "grouping_error_rows.csv",
        output_root / "ranking_error_rows.csv",
        output_root / "control_bias_rows.csv",
        output_root / "failure_casebook_rows.csv",
        output_root / "summary.json",
    ]
    _write_json(output_root / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in sha_paths})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose v90 local MV_AP_window failures using Phase1 full IoU matrix.")
    parser.add_argument("--phase1-root", type=Path, default=PHASE1)
    parser.add_argument("--phase0-root", type=Path, default=PHASE0)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
