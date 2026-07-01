#!/usr/bin/env python3
"""Independent class-agnostic AP audit for Stream4D v65 artifacts.

This script intentionally does not call Stream3D's evaluation entrypoint.  It
loads frozen prediction masks and GT ids, applies an explicit support policy,
then computes ScanNet-style AP over that support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCANNET_LABEL_IDS = [
    2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23,
    24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82,
    84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128,
    130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154, 155, 156,
    157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191,
    193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261,
    264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392,
    395, 399, 408, 417, 488, 540, 562, 570, 572, 581, 609, 748, 776, 1156,
    1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173,
    1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185,
    1186, 1187, 1188, 1189, 1190, 1191,
]


OVERLAPS = [round(float(x), 2) for x in np.append(np.arange(0.5, 0.95, 0.05), 0.25)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonable(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def percentile_or_none(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return finite_or_none(np.percentile(np.asarray(values, dtype=np.float64), q))


def load_scene_list(split_file: Path, pred_path: Path) -> list[str]:
    if split_file.exists():
        scenes = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        scenes = [scene[:-4] if scene.endswith(".npz") else scene for scene in scenes]
        available = {path.stem for path in pred_path.glob("*.npz")}
        return [scene for scene in scenes if scene in available]
    return sorted(path.stem for path in pred_path.glob("*.npz"))


def load_support_indices(scope: str, support_root: Path | None, scene: str, vertex_count: int) -> np.ndarray | None:
    if scope == "FULLMESH":
        return None
    if support_root is None:
        raise ValueError(f"--support-root is required for support scope {scope}")
    support_file = support_root / f"{scene}_pre_points.npy"
    if not support_file.exists():
        raise FileNotFoundError(f"missing support file: {support_file}")
    support = np.load(support_file)
    if support.ndim != 1:
        raise ValueError(f"support file must be 1-D: {support_file}")
    support = support.astype(np.int64, copy=False)
    if len(support) and (int(np.min(support)) < 0 or int(np.max(support)) >= vertex_count):
        raise ValueError(f"support indices out of range for {scene}: {support_file}")
    return support


def prepare_gt_arrays(
    gt_raw_ids: np.ndarray,
    *,
    class_agnostic: bool,
    base_label_id: int,
    valid_raw_label_ids: set[int],
    gt_instance_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    raw = gt_raw_ids.astype(np.int64, copy=False)
    raw_label_ids = raw // 1000
    raw_valid_instance = (raw >= 1000) & np.isin(
        raw_label_ids, np.asarray(sorted(valid_raw_label_ids), dtype=np.int64)
    )

    if class_agnostic and gt_instance_policy == "stream3d_no_class_legacy":
        # Reproduce Stream3D/evaluation/evaluate.py --no_class exactly.  This
        # intentionally converts semantic-only ids (<1000) into pseudo instances
        # and is kept only as a legacy support-scope cross-check.
        identity_ids = (raw % 1000) + base_label_id * 1000
        label_ids = np.full(raw.shape, int(base_label_id), dtype=np.int64)
        valid_instance_mask = np.ones(raw.shape, dtype=bool)
        void_mask = np.zeros(raw.shape, dtype=bool)
        transform = "legacy_stream3d: gt_eval_id = gt_id % 1000 + base_label_id * 1000; no void"
    elif class_agnostic and gt_instance_policy == "raw_instance_only":
        # Class-agnostic instance AP should ignore class labels, not merge raw
        # instance identities and not promote semantic-only ids into instances.
        identity_ids = raw.copy()
        label_ids = np.where(raw_valid_instance, int(base_label_id), 0).astype(np.int64)
        valid_instance_mask = raw_valid_instance
        void_mask = ~raw_valid_instance
        transform = (
            "raw_instance_only: valid GT iff raw gt_id>=1000 and raw_label in ScanNet valid ids; "
            "identity=raw gt_id; label=base_label_id; non-instance/invalid points are void"
        )
    elif not class_agnostic:
        identity_ids = raw.copy()
        label_ids = raw_label_ids.astype(np.int64, copy=False)
        valid_instance_mask = raw_valid_instance
        void_mask = ~np.isin(raw_label_ids, np.asarray(sorted(valid_raw_label_ids), dtype=np.int64))
        transform = "class_aware: valid GT iff raw gt_id>=1000 and raw_label in ScanNet valid ids"
    else:
        raise ValueError(f"unsupported gt_instance_policy={gt_instance_policy!r}")

    return identity_ids, label_ids, valid_instance_mask, void_mask, transform


def build_gt_infos(
    gt_identity_ids: np.ndarray,
    gt_label_ids: np.ndarray,
    valid_instance_mask: np.ndarray,
    valid_label_ids: set[int],
) -> dict[int, dict[str, Any]]:
    infos: dict[int, dict[str, Any]] = {}
    unique_ids, counts = np.unique(gt_identity_ids[valid_instance_mask], return_counts=True)
    for raw_id, count in zip(unique_ids.tolist(), counts.tolist()):
        inst_id = int(raw_id)
        if inst_id == 0:
            continue
        label_values = np.unique(gt_label_ids[(gt_identity_ids == inst_id) & valid_instance_mask])
        if len(label_values) != 1:
            raise ValueError(f"GT identity {inst_id} maps to multiple labels: {label_values.tolist()}")
        label_id = int(label_values[0])
        if label_id not in valid_label_ids:
            continue
        infos[inst_id] = {
            "instance_id": inst_id,
            "label_id": label_id,
            "vert_count": int(count),
            "med_dist": -1.0,
            "dist_conf": 0.0,
            "matched_pred": [],
        }
    return infos


def compute_scene_matches(
    *,
    scene: str,
    pred_file: Path,
    gt_file: Path,
    support_idx: np.ndarray | None,
    class_agnostic: bool,
    base_label_id: int,
    valid_label_ids: set[int],
    valid_raw_label_ids: set[int],
    gt_instance_policy: str,
    min_region_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_full = np.loadtxt(gt_file, dtype=np.int64)
    full_vertex_count = int(gt_full.shape[0])
    if support_idx is None:
        gt_support_raw = gt_full
        support_size = full_vertex_count
    else:
        gt_support_raw = gt_full[support_idx]
        support_size = int(support_idx.shape[0])

    gt_identity_ids, gt_label_ids, valid_instance_mask, bool_void, gt_transform = prepare_gt_arrays(
        gt_support_raw,
        class_agnostic=class_agnostic,
        base_label_id=base_label_id,
        valid_raw_label_ids=valid_raw_label_ids,
        gt_instance_policy=gt_instance_policy,
    )
    gt_infos = build_gt_infos(gt_identity_ids, gt_label_ids, valid_instance_mask, valid_label_ids)
    gt_list = list(gt_infos.values())

    with np.load(pred_file) as pred_npz:
        pred_masks = np.asarray(pred_npz["pred_masks"])
        pred_scores = np.asarray(pred_npz["pred_score"], dtype=np.float64)
        pred_classes = np.asarray(pred_npz["pred_classes"]) if "pred_classes" in pred_npz else np.full(len(pred_scores), base_label_id)

    if pred_masks.ndim != 2:
        raise ValueError(f"pred_masks must be [vertices, instances] in {pred_file}")
    if pred_masks.shape[0] != full_vertex_count:
        raise ValueError(
            f"prediction vertex count mismatch for {scene}: pred={pred_masks.shape[0]} gt={full_vertex_count}. "
            "This independent audit expects full-scene masks before support slicing."
        )
    if pred_masks.shape[1] != len(pred_scores):
        raise ValueError(f"pred_score length mismatch for {scene}")

    masks_support = pred_masks if support_idx is None else pred_masks[support_idx]
    masks_support = np.not_equal(masks_support, 0)
    pred_union = np.any(masks_support, axis=1) if masks_support.shape[1] else np.zeros(support_size, dtype=bool)

    pred_instances: list[dict[str, Any]] = []
    pred_best_ious: list[float] = []
    prediction_count_total = int(masks_support.shape[1])
    prediction_count_kept = 0
    pred_mask_points_total = 0

    for pred_id in range(masks_support.shape[1]):
        mask = masks_support[:, pred_id]
        vert_count = int(np.count_nonzero(mask))
        if vert_count < min_region_size:
            continue

        label_id = base_label_id if class_agnostic else int(pred_classes[pred_id])
        if label_id not in valid_label_ids:
            continue

        prediction_count_kept += 1
        pred_mask_points_total += vert_count
        pred_instance = {
            "filename": f"{pred_file.name}_{pred_id}",
            "scene": scene,
            "pred_id": int(pred_id),
            "label_id": int(label_id),
            "vert_count": vert_count,
            "confidence": float(pred_scores[pred_id]),
            "void_intersection": int(np.count_nonzero(bool_void & mask)),
            "matched_gt": [],
        }

        valid_matched = mask & valid_instance_mask
        matched_ids, intersections = np.unique(gt_identity_ids[valid_matched], return_counts=True)
        best_iou = 0.0
        for matched_id_raw, intersection_raw in zip(matched_ids.tolist(), intersections.tolist()):
            matched_id = int(matched_id_raw)
            intersection = int(intersection_raw)
            gt_info = gt_infos.get(matched_id)
            if gt_info is None:
                continue

            gt_copy = {key: value for key, value in gt_info.items() if key != "matched_pred"}
            pred_copy = {key: value for key, value in pred_instance.items() if key != "matched_gt"}
            gt_copy["intersection"] = intersection
            pred_copy["intersection"] = intersection
            pred_instance["matched_gt"].append(gt_copy)
            gt_infos[matched_id]["matched_pred"].append(pred_copy)

            if gt_info["vert_count"] >= min_region_size:
                denom = gt_info["vert_count"] + vert_count - intersection
                if denom > 0:
                    best_iou = max(best_iou, float(intersection) / float(denom))

        pred_instances.append(pred_instance)
        pred_best_ious.append(best_iou)

    gt_best_ious: list[float] = []
    duplicate_pred_counts_iou25: list[int] = []
    duplicate_pred_counts_iou50: list[int] = []
    valid_gt_count = 0
    for gt in gt_infos.values():
        if gt["instance_id"] < 1000 or gt["vert_count"] < min_region_size:
            continue
        valid_gt_count += 1
        best_iou = 0.0
        count25 = 0
        count50 = 0
        for pred in gt["matched_pred"]:
            denom = gt["vert_count"] + pred["vert_count"] - pred["intersection"]
            iou = float(pred["intersection"]) / float(denom) if denom > 0 else 0.0
            best_iou = max(best_iou, iou)
            if iou > 0.25:
                count25 += 1
            if iou > 0.50:
                count50 += 1
        gt_best_ious.append(best_iou)
        duplicate_pred_counts_iou25.append(max(0, count25 - 1))
        duplicate_pred_counts_iou50.append(max(0, count50 - 1))

    scene_match = {
        "scene": scene,
        "gt": gt_list,
        "pred": pred_instances,
    }
    scene_metrics = {
        "scene": scene,
        "full_vertex_count": full_vertex_count,
        "support_point_count": support_size,
        "support_ratio": float(support_size) / float(full_vertex_count) if full_vertex_count else None,
        "gt_instance_policy": gt_instance_policy,
        "gt_transform": gt_transform,
        "void_point_count": int(np.count_nonzero(bool_void)),
        "valid_instance_point_count": int(np.count_nonzero(valid_instance_mask)),
        "gt_instance_count_all": int(len(gt_infos)),
        "gt_instance_count_min_region": int(valid_gt_count),
        "prediction_count_total": prediction_count_total,
        "prediction_count_kept_min_region": int(prediction_count_kept),
        "pred_mask_points_total": int(pred_mask_points_total),
        "prediction_union_point_count": int(np.count_nonzero(pred_union)),
        "prediction_union_ratio": float(np.count_nonzero(pred_union)) / float(support_size) if support_size else None,
        "pred_best_iou_mean": finite_or_none(np.mean(pred_best_ious)) if pred_best_ious else None,
        "pred_best_iou_median": percentile_or_none(pred_best_ious, 50),
        "pred_best_iou_p90": percentile_or_none(pred_best_ious, 90),
        "gt_best_iou_mean": finite_or_none(np.mean(gt_best_ious)) if gt_best_ious else None,
        "gt_best_iou_median": percentile_or_none(gt_best_ious, 50),
        "gt_best_iou_p90": percentile_or_none(gt_best_ious, 90),
        "gt_best_iou_ge_025_ratio": float(np.mean(np.asarray(gt_best_ious) > 0.25)) if gt_best_ious else None,
        "gt_best_iou_ge_050_ratio": float(np.mean(np.asarray(gt_best_ious) > 0.50)) if gt_best_ious else None,
        "duplicate_predictions_per_gt_iou25_mean": finite_or_none(np.mean(duplicate_pred_counts_iou25)) if duplicate_pred_counts_iou25 else None,
        "duplicate_predictions_per_gt_iou50_mean": finite_or_none(np.mean(duplicate_pred_counts_iou50)) if duplicate_pred_counts_iou50 else None,
    }
    return scene_match, scene_metrics


def ap_from_examples(y_true: list[float], y_score: list[float], hard_false_negatives: int) -> float:
    if not y_score:
        return 0.0
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(y_score_arr)
    y_score_sorted = y_score_arr[order]
    y_true_sorted = y_true_arr[order]
    cumsum = np.cumsum(y_true_sorted)

    _, unique_indices = np.unique(y_score_sorted, return_index=True)
    num_prec_recall = len(unique_indices) + 1
    num_examples = len(y_score_sorted)
    num_true_examples = cumsum[-1]
    precision = np.zeros(num_prec_recall, dtype=np.float64)
    recall = np.zeros(num_prec_recall, dtype=np.float64)
    cumsum_with_sentinel = np.append(cumsum, 0.0)

    for idx_res, idx_scores in enumerate(unique_indices):
        true_before_threshold = cumsum_with_sentinel[idx_scores - 1]
        tp = num_true_examples - true_before_threshold
        fp = num_examples - idx_scores - tp
        fn = true_before_threshold + hard_false_negatives
        precision[idx_res] = float(tp) / float(tp + fp) if (tp + fp) else 0.0
        recall[idx_res] = float(tp) / float(tp + fn) if (tp + fn) else 0.0

    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_conv = np.copy(recall)
    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
    recall_for_conv = np.append(recall_for_conv, 0.0)
    step_widths = np.convolve(recall_for_conv, [-0.5, 0.0, 0.5], "valid")
    return float(np.dot(precision, step_widths))


def evaluate_threshold(scene_matches: list[dict[str, Any]], overlap_th: float, min_region_size: int) -> dict[str, Any]:
    pred_visited = {}
    for scene in scene_matches:
        for pred in scene["pred"]:
            pred_visited[pred["filename"]] = False

    y_true: list[float] = []
    y_score: list[float] = []
    hard_false_negatives = 0
    has_gt = False
    has_pred = False

    for scene in scene_matches:
        pred_instances = scene["pred"]
        gt_instances = [
            gt for gt in scene["gt"]
            if gt["instance_id"] >= 1000 and gt["vert_count"] >= min_region_size
        ]
        if gt_instances:
            has_gt = True
        if pred_instances:
            has_pred = True

        cur_true = np.ones(len(gt_instances), dtype=np.float64)
        cur_score = np.ones(len(gt_instances), dtype=np.float64) * (-float("inf"))
        cur_match = np.zeros(len(gt_instances), dtype=bool)

        for gti, gt in enumerate(gt_instances):
            found_match = False
            for pred in gt["matched_pred"]:
                if pred_visited[pred["filename"]]:
                    continue
                denom = gt["vert_count"] + pred["vert_count"] - pred["intersection"]
                overlap = float(pred["intersection"]) / float(denom) if denom > 0 else 0.0
                if overlap > overlap_th:
                    confidence = float(pred["confidence"])
                    if cur_match[gti]:
                        max_score = max(float(cur_score[gti]), confidence)
                        min_score = min(float(cur_score[gti]), confidence)
                        cur_score[gti] = max_score
                        cur_true = np.append(cur_true, 0.0)
                        cur_score = np.append(cur_score, min_score)
                        cur_match = np.append(cur_match, True)
                    else:
                        found_match = True
                        cur_match[gti] = True
                        cur_score[gti] = confidence
                        pred_visited[pred["filename"]] = True
            if not found_match:
                hard_false_negatives += 1

        y_true.extend(cur_true[cur_match].tolist())
        y_score.extend(cur_score[cur_match].tolist())

        for pred in pred_instances:
            found_gt = False
            for gt in pred["matched_gt"]:
                denom = gt["vert_count"] + pred["vert_count"] - gt["intersection"]
                overlap = float(gt["intersection"]) / float(denom) if denom > 0 else 0.0
                if overlap > overlap_th:
                    found_gt = True
                    break
            if found_gt:
                continue

            num_ignore = int(pred["void_intersection"])
            for gt in pred["matched_gt"]:
                if gt["instance_id"] < 1000:
                    num_ignore += int(gt["intersection"])
                if gt["vert_count"] < min_region_size:
                    num_ignore += int(gt["intersection"])
            proportion_ignore = float(num_ignore) / float(pred["vert_count"]) if pred["vert_count"] else 0.0
            if proportion_ignore <= overlap_th:
                y_true.append(0.0)
                y_score.append(float(pred["confidence"]))

    if has_gt and has_pred:
        ap = ap_from_examples(y_true, y_score, hard_false_negatives) if y_score else 0.0
    elif has_gt:
        ap = 0.0
    else:
        ap = float("nan")

    return {
        "overlap": float(overlap_th),
        "ap": finite_or_none(ap),
        "num_examples": int(len(y_score)),
        "num_true_examples": int(np.sum(np.asarray(y_true, dtype=np.float64))) if y_true else 0,
        "hard_false_negatives": int(hard_false_negatives),
        "has_gt": bool(has_gt),
        "has_pred": bool(has_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-path", required=True, type=Path)
    parser.add_argument("--gt-path", required=True, type=Path)
    parser.add_argument("--split-file", default=Path("Stream3D/splits/scannet.txt"), type=Path)
    parser.add_argument("--support-scope", default="FULLMESH", choices=["FULLMESH", "PREDICTION_UNION_ISLAND", "USED_FRAME_VISIBLE_SUPPORT", "SAME_SUPPORT_STREAM3D_PARITY"])
    parser.add_argument("--support-root", default=None, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", default=None, type=Path)
    parser.add_argument("--min-region-size", default=100, type=int)
    parser.add_argument("--class-agnostic", action="store_true", default=True)
    parser.add_argument("--base-label-id", default=SCANNET_LABEL_IDS[0], type=int)
    parser.add_argument(
        "--gt-instance-policy",
        default="raw_instance_only",
        choices=["raw_instance_only", "stream3d_no_class_legacy"],
        help=(
            "raw_instance_only is the strict class-agnostic instance AP policy: "
            "only raw gt_id>=1000 valid instances are GT. stream3d_no_class_legacy "
            "reproduces Stream3D's --no_class transform and may promote semantic-only ids."
        ),
    )
    parser.add_argument("--allow-diagnostic-only", action="store_true")
    args = parser.parse_args()

    pred_path = args.pred_path
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    gt_path = args.gt_path
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    scenes = load_scene_list(args.split_file, pred_path)
    if not scenes:
        raise RuntimeError(f"no scenes found in {pred_path}")

    manifest_path = pred_path / "config_manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("is_diagnostic_only") and not args.allow_diagnostic_only:
            raise RuntimeError(
                f"{manifest_path} is diagnostic-only; pass --allow-diagnostic-only to audit it explicitly"
            )

    valid_label_ids = set(SCANNET_LABEL_IDS)
    if args.class_agnostic:
        valid_label_ids = {int(args.base_label_id)}

    scene_matches: list[dict[str, Any]] = []
    scene_metrics: list[dict[str, Any]] = []
    support_hash_payload: list[dict[str, Any]] = []

    for scene in scenes:
        pred_file = pred_path / f"{scene}.npz"
        gt_file = gt_path / f"{scene}.txt"
        if not pred_file.exists() or not gt_file.exists():
            raise FileNotFoundError(f"missing pred/gt for scene {scene}: {pred_file}, {gt_file}")

        vertex_count = int(np.loadtxt(gt_file, dtype=np.int64).shape[0])
        support_idx = load_support_indices(args.support_scope, args.support_root, scene, vertex_count)
        if support_idx is None:
            support_hash_payload.append({"scene": scene, "support": "fullmesh", "vertex_count": vertex_count})
        else:
            support_hash_payload.append({
                "scene": scene,
                "support_file": str(args.support_root / f"{scene}_pre_points.npy"),
                "support_sha256": sha256_file(args.support_root / f"{scene}_pre_points.npy"),
                "support_count": int(support_idx.shape[0]),
            })

        match, metrics = compute_scene_matches(
            scene=scene,
            pred_file=pred_file,
            gt_file=gt_file,
            support_idx=support_idx,
            class_agnostic=args.class_agnostic,
            base_label_id=int(args.base_label_id),
            valid_label_ids=valid_label_ids,
            valid_raw_label_ids=set(SCANNET_LABEL_IDS),
            gt_instance_policy=args.gt_instance_policy,
            min_region_size=int(args.min_region_size),
        )
        scene_matches.append(match)
        scene_metrics.append(metrics)

    by_overlap = [evaluate_threshold(scene_matches, overlap, int(args.min_region_size)) for overlap in OVERLAPS]
    ap_values = [row["ap"] for row in by_overlap if row["ap"] is not None and not np.isclose(row["overlap"], 0.25)]
    ap50_values = [row["ap"] for row in by_overlap if row["ap"] is not None and np.isclose(row["overlap"], 0.50)]
    ap25_values = [row["ap"] for row in by_overlap if row["ap"] is not None and np.isclose(row["overlap"], 0.25)]

    all_pred_best = [m["pred_best_iou_median"] for m in scene_metrics if m["pred_best_iou_median"] is not None]
    all_gt_best = [m["gt_best_iou_median"] for m in scene_metrics if m["gt_best_iou_median"] is not None]

    output = {
        "script": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "pred_path": str(pred_path),
        "pred_manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "pred_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "pred_manifest": manifest,
        "gt_path": str(gt_path),
        "split_file": str(args.split_file),
        "scene_count": len(scenes),
        "scenes": scenes,
        "support_scope": args.support_scope,
        "support_root": str(args.support_root) if args.support_root else None,
        "support_policy_hash": sha256_jsonable(support_hash_payload),
        "support_policy_payload": support_hash_payload,
        "class_agnostic": bool(args.class_agnostic),
        "gt_instance_policy": args.gt_instance_policy,
        "class_agnostic_gt_transform": (
            "raw_instance_only: identity=raw gt_id for raw gt_id>=1000 valid-class instances; "
            "semantic-only/invalid points are void. stream3d_no_class_legacy: "
            "gt_id % 1000 + base_label_id * 1000."
        ),
        "base_label_id": int(args.base_label_id),
        "min_region_size": int(args.min_region_size),
        "overlaps": OVERLAPS,
        "AP": finite_or_none(np.mean(ap_values)) if ap_values else None,
        "AP50": finite_or_none(np.mean(ap50_values)) if ap50_values else None,
        "AP25": finite_or_none(np.mean(ap25_values)) if ap25_values else None,
        "by_overlap": by_overlap,
        "aggregate": {
            "support_point_count_mean": finite_or_none(np.mean([m["support_point_count"] for m in scene_metrics])),
            "support_point_count_min": int(min(m["support_point_count"] for m in scene_metrics)),
            "support_point_count_max": int(max(m["support_point_count"] for m in scene_metrics)),
            "full_vertex_count_mean": finite_or_none(np.mean([m["full_vertex_count"] for m in scene_metrics])),
            "gt_instance_count_mean": finite_or_none(np.mean([m["gt_instance_count_min_region"] for m in scene_metrics])),
            "full_scene_gt_instance_count_mean": finite_or_none(np.mean([m["gt_instance_count_min_region"] for m in scene_metrics])),
            "prediction_count_total": int(sum(m["prediction_count_total"] for m in scene_metrics)),
            "prediction_count_kept_min_region_total": int(sum(m["prediction_count_kept_min_region"] for m in scene_metrics)),
            "mean_predictions_per_scene": finite_or_none(np.mean([m["prediction_count_kept_min_region"] for m in scene_metrics])),
            "prediction_union_ratio_mean": finite_or_none(np.mean([m["prediction_union_ratio"] for m in scene_metrics])),
            "pred_best_iou_median_scene_median": percentile_or_none(all_pred_best, 50),
            "gt_best_iou_median_scene_median": percentile_or_none(all_gt_best, 50),
        },
        "scene_metrics": scene_metrics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(scene_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(scene_metrics)

    print(json.dumps({
        "output_json": str(args.output_json),
        "output_json_sha256": sha256_file(args.output_json),
        "support_scope": args.support_scope,
        "scene_count": len(scenes),
        "AP": output["AP"],
        "AP50": output["AP50"],
        "AP25": output["AP25"],
        "prediction_union_ratio_mean": output["aggregate"]["prediction_union_ratio_mean"],
        "gt_best_iou_median_scene_median": output["aggregate"]["gt_best_iou_median_scene_median"],
        "pred_best_iou_median_scene_median": output["aggregate"]["pred_best_iou_median_scene_median"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
