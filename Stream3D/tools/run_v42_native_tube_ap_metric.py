from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v36_external_downstream_assignment import _load_gt, _load_tubes


ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        current = float(value)
    except (TypeError, ValueError):
        return float(default)
    return current if np.isfinite(current) else float(default)


def _parse_json_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, got {type(parsed).__name__}")
    return [int(item) for item in parsed]


def _tube_iou(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return float(len(left & right) / union)


def _average_precision(tp_flags: list[int], fp_flags: list[int], num_gt: int) -> float | None:
    if num_gt <= 0:
        return None
    if len(tp_flags) != len(fp_flags):
        raise ValueError("tp_flags and fp_flags must have the same length")
    if not tp_flags:
        return 0.0
    tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
    recall = tp / float(num_gt)
    precision = tp / np.maximum(tp + fp, 1.0)
    mrec = np.concatenate([np.asarray([0.0]), recall, np.asarray([1.0])])
    mpre = np.concatenate([np.asarray([0.0]), precision, np.asarray([0.0])])
    for idx in range(mpre.size - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def _score_predictions_at_threshold(
    predictions: list[dict[str, Any]],
    gt_sets_by_scene: dict[str, list[dict[str, Any]]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    num_gt = sum(len(items) for items in gt_sets_by_scene.values())
    matched: set[tuple[str, int]] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    match_rows: list[dict[str, Any]] = []
    ordered = sorted(
        predictions,
        key=lambda row: (-float(row["score"]), str(row["scene"]), int(row["object_id"])),
    )
    for pred in ordered:
        scene = str(pred["scene"])
        pred_set = set(pred["tube_ids"])
        best_iou = 0.0
        best_gt_id: int | None = None
        for gt in gt_sets_by_scene.get(scene, []):
            gt_key = (scene, int(gt["gt_id"]))
            if gt_key in matched:
                continue
            current_iou = _tube_iou(pred_set, set(gt["tube_ids"]))
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_id = int(gt["gt_id"])
        is_tp = best_gt_id is not None and best_iou >= float(iou_threshold)
        if is_tp:
            matched.add((scene, int(best_gt_id)))
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
        match_rows.append(
            {
                "scene": scene,
                "object_id": int(pred["object_id"]),
                "score": float(pred["score"]),
                "threshold": float(iou_threshold),
                "best_iou": float(best_iou),
                "best_gt_id": "" if best_gt_id is None else int(best_gt_id),
                "is_true_positive": bool(is_tp),
            }
        )
    ap = _average_precision(tp_flags, fp_flags, num_gt)
    return {
        "threshold": float(iou_threshold),
        "AP": ap,
        "true_positive_count": int(sum(tp_flags)),
        "false_positive_count": int(sum(fp_flags)),
        "matched_gt_count": int(len(matched)),
        "gt_count": int(num_gt),
        "prediction_count": int(len(predictions)),
        "match_rows": match_rows,
    }


def _best_iou_rows(
    predictions: list[dict[str, Any]],
    gt_sets_by_scene: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preds_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        preds_by_scene[str(pred["scene"])].append(pred)
    for scene, gt_sets in gt_sets_by_scene.items():
        for gt in gt_sets:
            best_iou = 0.0
            best_object_id: int | None = None
            for pred in preds_by_scene.get(scene, []):
                current_iou = _tube_iou(set(pred["tube_ids"]), set(gt["tube_ids"]))
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_object_id = int(pred["object_id"])
            rows.append(
                {
                    "scene": scene,
                    "gt_id": int(gt["gt_id"]),
                    "gt_tube_count": int(len(gt["tube_ids"])),
                    "best_object_id": "" if best_object_id is None else int(best_object_id),
                    "best_tube_iou": float(best_iou),
                    "best_tube_iou_ge_25": bool(best_iou >= 0.25),
                    "best_tube_iou_ge_50": bool(best_iou >= 0.50),
                }
            )
    return rows


def _prediction_attribution_rows(
    predictions: list[dict[str, Any]],
    gt_labels_by_scene: dict[str, dict[int, int]],
    gt_sets_by_scene: dict[str, list[dict[str, Any]]],
    match_rows_at_50: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    match_by_prediction = {
        (str(row["scene"]), int(row["object_id"])): row
        for row in match_rows_at_50
        if "object_id" in row and "threshold" in row
    }
    gt_size: dict[tuple[str, int], int] = {}
    for scene, gt_sets in gt_sets_by_scene.items():
        for gt in gt_sets:
            gt_size[(scene, int(gt["gt_id"]))] = int(len(gt["tube_ids"]))
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        scene = str(pred["scene"])
        object_id = int(pred["object_id"])
        tube_ids = set(pred["tube_ids"])
        labels = gt_labels_by_scene.get(scene, {})
        counts = Counter(int(labels.get(int(tube_id), 0)) for tube_id in tube_ids)
        labeled_count = int(sum(count for gt_id, count in counts.items() if int(gt_id) > 0))
        unlabeled_count = int(counts.get(0, 0))
        gt_counts = {int(gt_id): int(count) for gt_id, count in counts.items() if int(gt_id) > 0}
        top_gt_id = None
        top_gt_count = 0
        if gt_counts:
            top_gt_id, top_gt_count = max(gt_counts.items(), key=lambda item: (item[1], -item[0]))
        top_gt_size = gt_size.get((scene, int(top_gt_id)), 0) if top_gt_id is not None else 0
        top_gt_fraction = float(top_gt_count / max(len(tube_ids), 1))
        top_gt_recall = float(top_gt_count / top_gt_size) if top_gt_size > 0 else 0.0
        gt_mass = np.asarray(list(gt_counts.values()), dtype=np.float64)
        if float(gt_mass.sum()) > 0.0:
            probs = gt_mass / float(gt_mass.sum())
            gt_entropy = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
        else:
            gt_entropy = 0.0
        match = match_by_prediction.get((scene, object_id), {})
        best_iou = _safe_float(match.get("best_iou", 0.0), default=0.0)
        ap50_tp = bool(match.get("is_true_positive", False))
        if labeled_count == 0:
            category = "no_gt_labeled_support"
        elif ap50_tp:
            category = "ap50_true_positive"
        elif best_iou >= 0.50:
            category = "duplicate_or_late_high_iou"
        elif unlabeled_count / max(len(tube_ids), 1) >= 0.50:
            category = "unlabeled_overmix"
        elif len(gt_counts) >= 3 and top_gt_fraction < 0.70:
            category = "multi_gt_overmix"
        elif top_gt_recall < 0.25:
            category = "low_recall_fragment"
        elif top_gt_fraction < 0.50:
            category = "mixed_low_purity"
        else:
            category = "partial_boundary_mismatch"
        rows.append(
            {
                "scene": scene,
                "variant": str(pred["variant"]),
                "source": str(pred["source"]),
                "object_id": object_id,
                "score": float(pred["score"]),
                "tube_count": int(len(tube_ids)),
                "labeled_tube_count": labeled_count,
                "unlabeled_tube_count": unlabeled_count,
                "unlabeled_fraction": float(unlabeled_count / max(len(tube_ids), 1)),
                "gt_component_count": int(len(gt_counts)),
                "top_gt_id": "" if top_gt_id is None else int(top_gt_id),
                "top_gt_tube_count": int(top_gt_count),
                "top_gt_size": int(top_gt_size),
                "top_gt_fraction": top_gt_fraction,
                "top_gt_recall": top_gt_recall,
                "gt_entropy_bits": gt_entropy,
                "ap50_best_iou": best_iou,
                "ap50_is_true_positive": ap50_tp,
                "ap50_best_gt_id": match.get("best_gt_id", ""),
                "fp_category": category,
            }
        )
    return rows


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(np.asarray(clean, dtype=np.float64))) if clean else None


def _fraction(values: list[bool]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _load_predictions(rows: list[dict[str, str]], scenes: list[str], variant: str, sources: set[str]) -> list[dict[str, Any]]:
    scene_set = set(scenes)
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("scene", "")) not in scene_set:
            continue
        if str(row.get("variant", "")) != str(variant):
            continue
        if sources and str(row.get("source", "")) not in sources:
            continue
        tube_ids = set(_parse_json_int_list(row.get("attached_tube_ids", "[]")))
        if not tube_ids:
            continue
        out.append(
            {
                "scene": str(row.get("scene", "")),
                "variant": str(row.get("variant", "")),
                "source": str(row.get("source", "")),
                "object_id": int(row.get("object_id", len(out))),
                "confidence": _safe_float(row.get("confidence", ""), default=0.0),
                "score": _safe_float(row.get("confidence", ""), default=0.0),
                "tube_ids": tube_ids,
                "tube_count": int(len(tube_ids)),
            }
        )
    return out


def _score_prediction(confidence: float, tube_count: int, score_mode: str) -> float:
    score_mode = str(score_mode)
    if score_mode == "confidence":
        return float(confidence)
    if score_mode == "confidence_log_tube_count":
        return float(confidence * np.log1p(max(int(tube_count), 0)))
    if score_mode == "confidence_sqrt_tube_count":
        return float(confidence * np.sqrt(max(int(tube_count), 0)))
    if score_mode == "confidence_tube_count":
        return float(confidence * max(int(tube_count), 0))
    raise ValueError(f"unsupported score_mode: {score_mode}")


def _gt_sets_from_labels(gt_labels: dict[int, int], tube_universe: set[int]) -> list[dict[str, Any]]:
    by_gt: dict[int, set[int]] = defaultdict(set)
    for tube_id, gt_id in gt_labels.items():
        tube_id = int(tube_id)
        gt_id = int(gt_id)
        if gt_id <= 0 or tube_id not in tube_universe:
            continue
        by_gt[gt_id].add(tube_id)
    return [
        {"gt_id": int(gt_id), "tube_ids": set(tube_ids), "tube_count": int(len(tube_ids))}
        for gt_id, tube_ids in sorted(by_gt.items())
        if tube_ids
    ]


def build_native_tube_ap(
    *,
    memory_object_rows_path: Path,
    cache_root: Path,
    scenes: list[str],
    variant: str,
    sources: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    memory_rows = _read_csv(memory_object_rows_path)
    raw_predictions = _load_predictions(memory_rows, scenes, variant, sources)
    gt_sets_by_scene: dict[str, list[dict[str, Any]]] = {}
    gt_labels_by_scene: dict[str, dict[int, int]] = {}
    predictions: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_args = argparse.Namespace(
            cache_root=str(cache_root),
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
        )
        tubes = _load_tubes(scene, scene_args)
        tube_universe = {int(tube.tube_id) for tube in tubes}
        gt_labels = _load_gt(scene, tubes, scene_args)
        gt_labels_by_scene[scene] = {int(k): int(v) for k, v in gt_labels.items()}
        gt_sets = _gt_sets_from_labels(gt_labels, tube_universe)
        gt_sets_by_scene[scene] = gt_sets
        scene_predictions = []
        for pred in raw_predictions:
            if str(pred["scene"]) != scene:
                continue
            tube_ids = set(pred["tube_ids"]) & tube_universe
            if not tube_ids:
                continue
            tube_count = int(len(tube_ids))
            if tube_count < int(args.min_pred_tube_count):
                continue
            if int(args.max_pred_tube_count) > 0 and tube_count > int(args.max_pred_tube_count):
                continue
            confidence = float(pred.get("confidence", pred.get("score", 0.0)))
            scene_predictions.append(
                {
                    **pred,
                    "tube_ids": tube_ids,
                    "tube_count": tube_count,
                    "confidence": confidence,
                    "score": _score_prediction(confidence, tube_count, str(args.score_mode)),
                }
            )
        predictions.extend(scene_predictions)
        labeled_tubes = {int(tube_id) for tube_id, gt_id in gt_labels.items() if int(gt_id) > 0 and int(tube_id) in tube_universe}
        covered_labeled = set().union(*(set(pred["tube_ids"]) for pred in scene_predictions)) & labeled_tubes if scene_predictions else set()
        scene_eval = _score_predictions_at_threshold(
            scene_predictions,
            {scene: gt_sets},
            iou_threshold=0.50,
        )
        scene_rows.append(
            {
                "scene": scene,
                "variant": variant,
                "sources": sorted(sources),
                "prediction_count": int(len(scene_predictions)),
                "gt_object_count": int(len(gt_sets)),
                "cache_tube_count": int(len(tube_universe)),
                "gt_labeled_tube_count": int(len(labeled_tubes)),
                "predicted_tube_count": int(len(set().union(*(set(pred["tube_ids"]) for pred in scene_predictions))) if scene_predictions else 0),
                "labeled_tube_coverage": float(len(covered_labeled) / len(labeled_tubes)) if labeled_tubes else None,
                "native_tube_AP50": scene_eval["AP"],
            }
        )
    thresholds = [0.50 + 0.05 * idx for idx in range(10)]
    threshold_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    result_50_full: dict[str, Any] | None = None
    for threshold in thresholds:
        result = _score_predictions_at_threshold(predictions, gt_sets_by_scene, iou_threshold=threshold)
        if abs(float(threshold) - 0.50) < 1e-9:
            result_50_full = result
        row = {key: value for key, value in result.items() if key != "match_rows"}
        threshold_rows.append(row)
        match_rows.extend(result["match_rows"])
    result_25 = _score_predictions_at_threshold(predictions, gt_sets_by_scene, iou_threshold=0.25)
    result_50 = threshold_rows[0]
    gt_best_rows = _best_iou_rows(predictions, gt_sets_by_scene)
    attribution_rows = _prediction_attribution_rows(
        predictions,
        gt_labels_by_scene,
        gt_sets_by_scene,
        (result_50_full or _score_predictions_at_threshold(predictions, gt_sets_by_scene, iou_threshold=0.50))[
            "match_rows"
        ],
    )
    attribution_counts = Counter(str(row["fp_category"]) for row in attribution_rows)
    ap_values = [float(row["AP"]) for row in threshold_rows if row.get("AP") is not None]
    prediction_tubes = set().union(*(set(pred["tube_ids"]) for pred in predictions)) if predictions else set()
    all_labeled_tubes = {
        int(tube_id)
        for gt_sets in gt_sets_by_scene.values()
        for gt in gt_sets
        for tube_id in set(gt["tube_ids"])
    }
    summary = {
        "phase": "v42_native_tube_space_ap",
        "status": "OK_NATIVE_TUBE_AP_COMPUTED" if sum(len(v) for v in gt_sets_by_scene.values()) > 0 else "NO_GO_NATIVE_TUBE_AP_NO_GT",
        "memory_object_rows": str(memory_object_rows_path),
        "cache_root": str(cache_root),
        "scenes": scenes,
        "variant": variant,
        "sources": sorted(sources),
        "score_mode": str(args.score_mode),
        "min_pred_tube_count": int(args.min_pred_tube_count),
        "max_pred_tube_count": int(args.max_pred_tube_count),
        "prediction_count": int(len(predictions)),
        "gt_object_count": int(sum(len(v) for v in gt_sets_by_scene.values())),
        "native_tube_AP": _mean(ap_values),
        "native_tube_AP50": result_50.get("AP"),
        "native_tube_AP25": result_25.get("AP"),
        "native_tube_true_positive_count_at_50": result_50.get("true_positive_count"),
        "native_tube_false_positive_count_at_50": result_50.get("false_positive_count"),
        "per_gt_best_tube_iou_mean": _mean([float(row["best_tube_iou"]) for row in gt_best_rows]),
        "per_gt_best_tube_iou_ge_25": _fraction([bool(row["best_tube_iou_ge_25"]) for row in gt_best_rows]),
        "per_gt_best_tube_iou_ge_50": _fraction([bool(row["best_tube_iou_ge_50"]) for row in gt_best_rows]),
        "prediction_attribution_counts": dict(sorted(attribution_counts.items())),
        "gt_labeled_tube_count": int(len(all_labeled_tubes)),
        "predicted_tube_count": int(len(prediction_tubes)),
        "labeled_tube_coverage": float(len(prediction_tubes & all_labeled_tubes) / len(all_labeled_tubes)) if all_labeled_tubes else None,
        "metric_scope": "d4rt_native_tube_space",
        "is_scannet_ap_result": False,
        "is_method_result": True,
        "is_diagnostic_only": False,
        "forbidden_for_method_table": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "phase8_gate_pass": False,
        "phase8_gate_blocker": "native tube-space AP is method-compatible but is not ScanNet mesh AP",
        "threshold_rows": threshold_rows,
        "scene_rows": scene_rows,
    }
    prediction_rows = [
        {
            "scene": str(pred["scene"]),
            "variant": str(pred["variant"]),
            "source": str(pred["source"]),
            "object_id": int(pred["object_id"]),
            "confidence": float(pred.get("confidence", pred["score"])),
            "score": float(pred["score"]),
            "tube_count": int(pred["tube_count"]),
            "tube_ids": sorted(int(tube_id) for tube_id in pred["tube_ids"]),
        }
        for pred in predictions
    ]
    gt_rows = [
        {
            "scene": scene,
            "gt_id": int(gt["gt_id"]),
            "tube_count": int(gt["tube_count"]),
            "tube_ids": sorted(int(tube_id) for tube_id in gt["tube_ids"]),
        }
        for scene, gt_sets in gt_sets_by_scene.items()
        for gt in gt_sets
    ]
    detail_rows = match_rows + gt_best_rows
    return prediction_rows, gt_rows, detail_rows, attribution_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v42 ObjectFields in D4RT native tube-set AP space.")
    parser.add_argument(
        "--memory-object-rows",
        default="outputs/audit/v42_streaming_memory_unioncap320_allframe_r1/memory_object_field_rows.csv",
    )
    parser.add_argument(
        "--cache-root",
        default="outputs/stream4d_debug_v42_semantic_occupancy_real_dino_q5_mf32_b1024/Q5",
    )
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--sources", default="dinov2_maskcut")
    parser.add_argument("--max-tubes-per-window", type=int, default=1024)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--min-visibility", type=float, default=0.50)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument(
        "--score-mode",
        default="confidence",
        choices=[
            "confidence",
            "confidence_log_tube_count",
            "confidence_sqrt_tube_count",
            "confidence_tube_count",
        ],
    )
    parser.add_argument("--min-pred-tube-count", type=int, default=1)
    parser.add_argument("--max-pred-tube-count", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/audit/v42_native_tube_ap_metric_allframe_r1")
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = {item.strip() for item in str(args.sources).split(",") if item.strip()}
    output_root = ROOT / str(args.output_root)
    prediction_rows, gt_rows, detail_rows, attribution_rows, summary = build_native_tube_ap(
        memory_object_rows_path=ROOT / str(args.memory_object_rows),
        cache_root=ROOT / str(args.cache_root),
        scenes=scenes,
        variant=str(args.variant),
        sources=sources,
        args=args,
    )
    _write_csv(output_root / "native_tube_ap_predictions.csv", prediction_rows)
    _write_csv(output_root / "native_tube_ap_gt_rows.csv", gt_rows)
    _write_csv(output_root / "native_tube_ap_detail_rows.csv", detail_rows)
    _write_csv(output_root / "native_tube_ap_prediction_attribution_rows.csv", attribution_rows)
    _write_csv(output_root / "native_tube_ap_scene_rows.csv", summary["scene_rows"])
    _write_csv(output_root / "native_tube_ap_threshold_rows.csv", summary["threshold_rows"])
    _write_json(output_root / "native_tube_ap_summary.json", summary)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "status": summary["status"],
                "prediction_count": summary["prediction_count"],
                "gt_object_count": summary["gt_object_count"],
                "native_tube_AP": summary["native_tube_AP"],
                "native_tube_AP50": summary["native_tube_AP50"],
                "native_tube_AP25": summary["native_tube_AP25"],
                "phase8_gate_pass": summary["phase8_gate_pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
