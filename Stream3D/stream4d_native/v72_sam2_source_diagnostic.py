from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v72_dense_token_proposals import _load_gt_2d, _load_pipeline_roots, _resize_label  # noqa: E402
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _summarize_iou  # noqa: E402
from tools.run_v66_local_chunk_eval import _score_free  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


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
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def _bbox(mask: np.ndarray) -> dict[str, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    return {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1}


def _frame_id_from_path(path: Path) -> int | None:
    match = re.search(r"frame0*([0-9]+)", path.stem)
    return int(match.group(1)) if match else None


def _load_sam2_masks(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    key = "masks" if "masks" in data.files else (data.files[0] if data.files else "")
    if not key:
        return None
    masks = np.asarray(data[key])
    if masks.ndim != 3:
        return None
    if masks.dtype != bool:
        masks = masks != 0
    resized: list[np.ndarray] = []
    h, w = int(shape_hw[0]), int(shape_hw[1])
    for mask in masks:
        if mask.shape[:2] == (h, w):
            resized.append(np.asarray(mask, dtype=bool))
        else:
            resized.append(cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool))
    return np.stack(resized, axis=0) if resized else np.zeros((0, h, w), dtype=bool)


def _mask_diagnostic(mask: np.ndarray, gt: np.ndarray, gt_area: dict[int, int]) -> tuple[int, float]:
    if not np.any(mask):
        return 0, 0.0
    labels, counts = np.unique(gt[np.asarray(mask, dtype=bool)], return_counts=True)
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
    union = int(np.count_nonzero(mask)) + int(gt_area.get(best_gid, 0)) - best_inter
    return best_gid, float(best_inter / max(1, union))


def _proposal_rows_for_masks(
    *,
    source: str,
    scene: str,
    frame_id: int,
    masks: list[np.ndarray],
    gt: np.ndarray,
    source_path: str,
) -> list[dict[str, Any]]:
    labels, counts = np.unique(gt, return_counts=True)
    gt_area = {int(label): int(count) for label, count in zip(labels, counts) if int(label) > 0}
    rows: list[dict[str, Any]] = []
    image_area = max(1, int(gt.shape[0]) * int(gt.shape[1]))
    for idx, mask in enumerate(masks):
        if not np.any(mask):
            continue
        majority_gt, majority_iou = _mask_diagnostic(mask, gt, gt_area)
        area = int(np.count_nonzero(mask))
        rows.append(
            {
                "source": source,
                "scene_id": scene,
                "frame_id": int(frame_id),
                "proposal_id": f"{source}:{scene}:{int(frame_id)}:{idx:05d}",
                "source_path": source_path,
                "mask_index": int(idx),
                "proposal_area_ratio": float(area / image_area),
                "proposal_bbox": json.dumps(_bbox(mask), sort_keys=True),
                "majority_gt_id_diagnostic": int(majority_gt),
                "majority_iou_diagnostic": float(majority_iou),
                "uses_gt_for_prediction": False,
                "uses_gt_for_evaluation": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        )
    return rows


def _source_frame_stats(rows: list[dict[str, Any]], gt: np.ndarray) -> dict[str, Any]:
    gt_ids = [int(label) for label in np.unique(gt) if int(label) > 0]
    gt_best: dict[int, float] = {gid: 0.0 for gid in gt_ids}
    for row in rows:
        gid = int(row.get("majority_gt_id_diagnostic") or 0)
        if gid > 0:
            gt_best[gid] = max(gt_best.get(gid, 0.0), _float(row.get("majority_iou_diagnostic"), 0.0))
    return {
        "proposal_count": len(rows),
        "proposal_count_per_frame": len(rows),
        "proposal_area_ratio_mean": _mean([_float(row.get("proposal_area_ratio"), 0.0) for row in rows]),
        "proposal_majority_IoU_mean_diagnostic": _mean([_float(row.get("majority_iou_diagnostic"), 0.0) for row in rows]),
        "proposal_IoU50_rate_diagnostic": _mean([1.0 if _float(row.get("majority_iou_diagnostic"), 0.0) >= 0.50 else 0.0 for row in rows]),
        "gt_best_IoU_mean_diagnostic": _mean(list(gt_best.values())),
        "gt_IoU50_coverage_rate_diagnostic": _mean([1.0 if value >= 0.50 else 0.0 for value in gt_best.values()]),
        "diagnostic_GT_count": len(gt_ids),
    }


def _source_oracle_summary(
    *,
    source_rows: list[dict[str, Any]],
    frame_gt: dict[tuple[str, int], np.ndarray],
    frame_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    best_by_gt_frame: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in source_rows:
        gid = int(row.get("majority_gt_id_diagnostic") or 0)
        if gid <= 0:
            continue
        key = (str(row.get("scene_id") or ""), int(row.get("frame_id") or -1), gid)
        old = best_by_gt_frame.get(key)
        if old is None or _float(row.get("majority_iou_diagnostic"), 0.0) > _float(old.get("majority_iou_diagnostic"), 0.0):
            best_by_gt_frame[key] = row
    acc = SparseSceneIoU()
    for key, gt in frame_gt.items():
        pred = np.zeros(gt.shape, dtype=np.int64)
        scene, frame_id = key
        selected = [row for item_key, row in best_by_gt_frame.items() if item_key[0] == scene and item_key[1] == frame_id]
        selected = sorted(selected, key=lambda row: _float(row.get("majority_iou_diagnostic"), 0.0), reverse=True)
        for row in selected:
            mask = frame_masks.get(str(row.get("proposal_id") or ""))
            if mask is None:
                continue
            pred[(mask > 0) & (pred == 0)] = int(row.get("majority_gt_id_diagnostic") or 0)
        acc.add(pred, gt)
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    return {
        "source_oracle_SF50_diagnostic": _score_free(summary),
        "source_oracle_AP50_diagnostic": summary.get("ap50"),
        "source_oracle_AP25_diagnostic": summary.get("ap25"),
        "source_oracle_GT_best_IoU_mean_diagnostic": summary.get("gt_best_iou_mean"),
        "source_oracle_pred_best_IoU_median_diagnostic": summary.get("pred_best_iou_median"),
        "source_oracle_uses_gt_for_selection": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    sam2_root = _rooted(args.sam2_root)
    witness_summary = _rooted(args.witness_summary)
    pipeline_roots = _load_pipeline_roots(witness_summary, scenes)
    missing_rows: list[dict[str, Any]] = []
    if not sam2_root.exists():
        missing_rows.append({"name": "sam2_root", "path": _rel(sam2_root)})
    if not witness_summary.exists():
        missing_rows.append({"name": "witness_summary", "path": _rel(witness_summary)})

    proposal_rows: list[dict[str, Any]] = []
    frame_metric_rows: list[dict[str, Any]] = []
    frame_gt: dict[tuple[str, int], np.ndarray] = {}
    frame_masks_by_source: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    processed_frames = 0
    for scene in scenes:
        scene_sam2 = sam2_root / scene
        files = sorted(scene_sam2.glob("sam2_frame*_masks.npz"), key=lambda path: _frame_id_from_path(path) or -1)
        if not files:
            missing_rows.append({"name": "sam2_scene_files", "scene_id": scene, "path": _rel(scene_sam2)})
            continue
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            missing_rows.append({"name": "pipeline_root", "scene_id": scene, "path": ""})
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        for path in files:
            frame_id = _frame_id_from_path(path)
            if frame_id is None:
                continue
            if int(args.max_frames_per_scene) > 0 and sum(1 for row in frame_metric_rows if row.get("scene_id") == scene and row.get("source") == "sam2_filtered_npz") >= int(args.max_frames_per_scene):
                break
            depth_shape = tuple(int(value) for value in stream.load_depth(int(frame_id)).shape)
            gt = _load_gt_2d(scene, int(frame_id), depth_shape)
            frame_gt[(scene, int(frame_id))] = gt

            sam2_masks_np = _load_sam2_masks(path, depth_shape)
            if sam2_masks_np is None:
                missing_rows.append({"name": "sam2_npz_load", "scene_id": scene, "frame_id": int(frame_id), "path": _rel(path)})
                continue
            sam2_masks = [sam2_masks_np[idx] for idx in range(int(sam2_masks_np.shape[0]))]
            sam2_rows = _proposal_rows_for_masks(source="sam2_filtered_npz", scene=scene, frame_id=int(frame_id), masks=sam2_masks, gt=gt, source_path=_rel(path))
            proposal_rows.extend(sam2_rows)
            for row, mask in zip(sam2_rows, [mask for mask in sam2_masks if np.any(mask)]):
                frame_masks_by_source["sam2_filtered_npz"][str(row.get("proposal_id"))] = mask
            sam2_metric = _source_frame_stats(sam2_rows, gt)
            sam2_metric.update({"source": "sam2_filtered_npz", "scene_id": scene, "frame_id": int(frame_id), "source_path": _rel(path)})
            frame_metric_rows.append(sam2_metric)

            label = _resize_label(mask_dir / f"{int(frame_id)}.png", depth_shape)
            if label is None:
                missing_rows.append({"name": "cropformer_mask_png", "scene_id": scene, "frame_id": int(frame_id), "path": _rel(mask_dir / f"{int(frame_id)}.png")})
            else:
                crop_masks = [label == int(mask_id) for mask_id in sorted(int(v) for v in np.unique(label) if int(v) > 0)]
                crop_rows = _proposal_rows_for_masks(
                    source="cropformer_flat_png",
                    scene=scene,
                    frame_id=int(frame_id),
                    masks=crop_masks,
                    gt=gt,
                    source_path=_rel(mask_dir / f"{int(frame_id)}.png"),
                )
                proposal_rows.extend(crop_rows)
                for row, mask in zip(crop_rows, [mask for mask in crop_masks if np.any(mask)]):
                    frame_masks_by_source["cropformer_flat_png"][str(row.get("proposal_id"))] = mask
                crop_metric = _source_frame_stats(crop_rows, gt)
                crop_metric.update({"source": "cropformer_flat_png", "scene_id": scene, "frame_id": int(frame_id), "source_path": _rel(mask_dir / f"{int(frame_id)}.png")})
                frame_metric_rows.append(crop_metric)
            processed_frames += 1

    variant_rows: list[dict[str, Any]] = []
    for source in sorted({str(row.get("source") or "") for row in frame_metric_rows}):
        subset = [row for row in frame_metric_rows if row.get("source") == source]
        source_prop_rows = [row for row in proposal_rows if row.get("source") == source]
        oracle = _source_oracle_summary(source_rows=source_prop_rows, frame_gt=frame_gt, frame_masks=frame_masks_by_source[source]) if source_prop_rows else {}
        variant_rows.append(
            {
                "source": source,
                "frame_count": len(subset),
                "proposal_count": sum(int(_float(row.get("proposal_count"), 0.0)) for row in subset),
                "proposal_count_per_frame_mean": _mean([_float(row.get("proposal_count_per_frame"), 0.0) for row in subset]),
                "proposal_area_ratio_mean": _mean([_float(row.get("proposal_area_ratio_mean"), 0.0) for row in subset]),
                "proposal_majority_IoU_mean_diagnostic": _mean([_float(row.get("proposal_majority_IoU_mean_diagnostic"), 0.0) for row in subset]),
                "proposal_IoU50_rate_diagnostic": _mean([_float(row.get("proposal_IoU50_rate_diagnostic"), 0.0) for row in subset]),
                "gt_best_IoU_mean_diagnostic": _mean([_float(row.get("gt_best_IoU_mean_diagnostic"), 0.0) for row in subset]),
                "gt_IoU50_coverage_rate_diagnostic": _mean([_float(row.get("gt_IoU50_coverage_rate_diagnostic"), 0.0) for row in subset]),
                "diagnostic_GT_count_mean": _mean([_float(row.get("diagnostic_GT_count"), 0.0) for row in subset]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_evaluation": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
                **oracle,
            }
        )

    crop = next((row for row in variant_rows if row.get("source") == "cropformer_flat_png"), {})
    sam2 = next((row for row in variant_rows if row.get("source") == "sam2_filtered_npz"), {})
    sam2_minus_crop_gt_best = _float(sam2.get("gt_best_IoU_mean_diagnostic"), 0.0) - _float(crop.get("gt_best_IoU_mean_diagnostic"), 0.0)
    sam2_minus_crop_oracle = _float(sam2.get("source_oracle_SF50_diagnostic"), 0.0) - _float(crop.get("source_oracle_SF50_diagnostic"), 0.0)
    source_signal = bool(sam2) and (
        sam2_minus_crop_gt_best >= float(args.min_gt_best_gain)
        or sam2_minus_crop_oracle >= float(args.min_oracle_sf50_gain)
    )
    complete = processed_frames >= len(scenes) * int(args.expected_frames_per_scene)
    decision = "PARTIAL_GO_STRONGER_SOURCE_SIGNAL_INCOMPLETE_COVERAGE" if source_signal else "NO_GO_SAM2_SOURCE_SIGNAL_NOT_BETTER"
    if not complete:
        decision += "_DIAGNOSTIC_ONLY"
    summary = {
        "phase": "v72_sam2_source_adequacy_diagnostic",
        "decision": decision,
        "processed_frame_count": processed_frames,
        "expected_frame_count": len(scenes) * int(args.expected_frames_per_scene),
        "coverage_complete_for_diagnostic": complete,
        "sam2_root": _rel(sam2_root),
        "cropformer_source": "pipeline_root/output_Cropformer/mask",
        "sam2_filtered_npz": sam2,
        "cropformer_flat_png": crop,
        "sam2_minus_cropformer_gt_best_IoU_mean": sam2_minus_crop_gt_best,
        "sam2_minus_cropformer_source_oracle_SF50": sam2_minus_crop_oracle,
        "source_diagnostic_supports_stronger_mask_source": source_signal,
        "can_replace_v72_phase2_full_source": False,
        "cannot_replace_reason": "SAM2 source is available only as a 4-frame-per-scene v51 diagnostic stack, not the full v72 chunk proposal universe.",
        "method_boundary": {
            "uses_gt_for_prediction": False,
            "gt_used_for_diagnostic_evaluation": True,
            "source_oracle_uses_gt_for_selection": True,
            "source_oracle_forbidden_for_method_table": True,
            "training_free": True,
        },
        "gate": {
            "all_inputs_present": not missing_rows,
            "sam2_source_signal_gt_best_gain_ge_threshold": sam2_minus_crop_gt_best >= float(args.min_gt_best_gain),
            "sam2_source_signal_oracle_sf50_gain_ge_threshold": sam2_minus_crop_oracle >= float(args.min_oracle_sf50_gain),
            "coverage_complete_for_diagnostic": complete,
            "uses_gt_for_prediction_false": True,
            "pass": False,
        },
    }
    _write_csv(output_root / "source_proposal_rows.csv", proposal_rows)
    _write_csv(output_root / "source_frame_metric_rows.csv", frame_metric_rows)
    _write_csv(output_root / "source_variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_json(output_root / "sam2_source_diagnostic_summary.json", summary)
    sha_rows = []
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "output"})
    for path in [sam2_root, witness_summary]:
        if path.is_file():
            sha_rows.append({"path": _rel(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path), "kind": "input"})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 targeted SAM2 source adequacy diagnostic.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--sam2-root", default="outputs/remask/v51_r2/sam2_tiny_probe5_4f_p64_crop1_containment_filtered")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase2_sam2_source_diagnostic")
    parser.add_argument("--max-frames-per-scene", type=int, default=4)
    parser.add_argument("--expected-frames-per-scene", type=int, default=4)
    parser.add_argument("--min-gt-best-gain", type=float, default=0.08)
    parser.add_argument("--min-oracle-sf50-gain", type=float, default=0.10)
    return parser.parse_args()
