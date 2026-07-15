#!/usr/bin/env python3
"""Run a controlled SAM2 prompt benchmark from LingBot-projected v107 points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GSAM2_ROOT = Path(os.environ.get("GSAM2_ROOT", str(ROOT / "Grounded-SAM-2"))).resolve()
for item in (GSAM2_ROOT, ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def load_reference_records(reference_root: Path) -> dict[int, dict[str, Any]]:
    summary = read_json(reference_root / "summary.json")
    records = {}
    for row in summary.get("records", []):
        frame_id = int(row["frame_id"])
        item = dict(row)
        item["label_path"] = resolve_path(str(row["label_path"]), reference_root)
        records[frame_id] = item
    return records


def load_points(points_json: Path) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    payload = read_json(points_json)
    out: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("rows", []):
        key = (int(row["target_frame_index"]), int(row["target_obj_id"]), int(row["source_frame_index"]))
        out[key].append(row)
    return out


def load_rows(rows_csv: Path) -> list[dict[str, Any]]:
    with rows_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def map_lingbot_xy_to_original(x: float, y: float, *, lingbot_hw: tuple[int, int], orig_hw: tuple[int, int]) -> tuple[float, float]:
    lh, lw = int(lingbot_hw[0]), int(lingbot_hw[1])
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    ox = float(x) * float(max(ow - 1, 1)) / float(max(lw - 1, 1))
    oy = float(y) * float(max(oh - 1, 1)) / float(max(lh - 1, 1))
    return ox, oy


def mask_metrics(mask: np.ndarray, ref: np.ndarray, target_label: np.ndarray, neg_ids: set[int]) -> dict[str, Any]:
    pred = np.asarray(mask).astype(bool)
    refb = np.asarray(ref).astype(bool)
    inter = int(np.count_nonzero(pred & refb))
    pred_area = int(np.count_nonzero(pred))
    ref_area = int(np.count_nonzero(refb))
    union = int(np.count_nonzero(pred | refb))
    neg_mask = np.isin(target_label, list(neg_ids)) if neg_ids else np.zeros_like(target_label, dtype=bool)
    neg_overlap = int(np.count_nonzero(pred & neg_mask))
    return {
        "mask_area_px": pred_area,
        "reference_area_px": ref_area,
        "intersection_px": inter,
        "iou_to_reference": float(inter / max(union, 1)),
        "precision_to_reference": float(inter / max(pred_area, 1)),
        "recall_to_reference": float(inter / max(ref_area, 1)),
        "negative_sibling_overlap_px": neg_overlap,
        "negative_sibling_overlap_rate": float(neg_overlap / max(pred_area, 1)),
    }


def choose_cases(rows: list[dict[str, Any]], points_by_case: dict[tuple[int, int, int], list[dict[str, Any]]], *, max_cases: int, per_lag: int) -> list[dict[str, Any]]:
    usable = [r for r in rows if str(r.get("usable_positive_negative_prompt", "")).lower() == "true"]
    usable = [
        r
        for r in usable
        if (int(r["target_frame_index"]), int(r["target_obj_id"]), int(r["source_frame_index"])) in points_by_case
    ]
    usable.sort(
        key=lambda r: (
            int(r.get("source_lag", 0)),
            -int(r.get("positive_visible_count", 0)),
            int(r.get("negative_reference_hits_target_obj_count", 0)),
            -int(r.get("negative_visible_count", 0)),
        )
    )
    selected: list[dict[str, Any]] = []
    lag_counts: dict[int, int] = defaultdict(int)
    for row in usable:
        lag = int(row.get("source_lag", 0))
        if lag_counts[lag] >= int(per_lag):
            continue
        selected.append(row)
        lag_counts[lag] += 1
        if len(selected) >= int(max_cases):
            break
    return selected


def build_sam2_predictor(args: argparse.Namespace):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = Path(args.sam2_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    model = build_sam2(str(args.sam2_model_cfg), str(checkpoint), device=str(args.device))
    if str(args.model_dtype).lower() in {"bf16", "bfloat16"}:
        model.to(dtype=torch.bfloat16)
    elif str(args.model_dtype).lower() in {"fp16", "float16"}:
        model.to(dtype=torch.float16)
    model.eval()
    return SAM2ImagePredictor(model), checkpoint


def autocast_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.device).startswith("cuda"):
        return {"enabled": False}
    dtype_name = str(args.model_dtype).lower()
    dtype = torch.float32
    enabled = False
    if dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
        enabled = True
    elif dtype_name in {"fp16", "float16"}:
        dtype = torch.float16
        enabled = True
    return {"device_type": "cuda", "dtype": dtype, "enabled": enabled}


def select_mask_by_score(masks: Any, scores: Any, ref_mask: np.ndarray, target_label: np.ndarray, neg_ids: set[int]) -> tuple[np.ndarray, dict[str, Any]]:
    mask_arr = np.asarray(masks)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    score_arr = np.asarray(scores if scores is not None else np.zeros((mask_arr.shape[0],), dtype=np.float32)).reshape(-1)
    candidates = []
    for idx, raw in enumerate(mask_arr):
        mask = np.squeeze(raw) > 0
        metrics = mask_metrics(mask, ref_mask, target_label, neg_ids)
        score = float(score_arr[idx]) if idx < len(score_arr) else 0.0
        metrics.update({"candidate_index": int(idx), "sam2_score": score})
        candidates.append((score, idx, mask, metrics))
    if not candidates:
        empty = np.zeros_like(ref_mask, dtype=bool)
        return empty, {"candidate_index": -1, "sam2_score": 0.0, **mask_metrics(empty, ref_mask, target_label, neg_ids)}
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, _idx, mask, metrics = candidates[0]
    oracle = max((item[3]["iou_to_reference"] for item in candidates), default=0.0)
    metrics["oracle_best_iou_among_multimask"] = float(oracle)
    return mask.astype(bool), metrics


def draw_case(
    *,
    rgb: np.ndarray,
    ref_mask: np.ndarray,
    pred_masks: dict[str, np.ndarray],
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    title: str,
) -> np.ndarray:
    base = rgb.copy()
    contours, _ = cv2.findContours(ref_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, contours, -1, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    for (x, y), label in zip(point_coords, point_labels):
        color = (25, 240, 60) if int(label) == 1 else (240, 50, 45)
        cv2.circle(base, (int(round(x)), int(round(y))), 7, color, -1, lineType=cv2.LINE_AA)
    panels = [base]
    for name, mask in pred_masks.items():
        panel = rgb.copy()
        color = np.asarray([40, 220, 255] if name == "G1_pos" else [255, 70, 170], dtype=np.float32)
        panel[mask] = (0.45 * panel[mask].astype(np.float32) + 0.55 * color).clip(0, 255).astype(np.uint8)
        cv2.drawContours(panel, contours, -1, (255, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(panel, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    merged = np.concatenate(panels, axis=1)
    header = 30
    out = np.zeros((merged.shape[0] + header, merged.shape[1], 3), dtype=np.uint8)
    out[:] = 12
    out[header:] = merged
    cv2.putText(out, title[:180], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def contact_sheet(paths: list[Path], out_path: Path, cols: int = 1, pad: int = 6) -> Path | None:
    images = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    canvas[:] = 24
    for idx, image in enumerate(images):
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        y = (idx // cols) * (h + pad)
        x = (idx % cols) * (w + pad)
        canvas[y : y + h, x : x + w] = image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-probe-root", required=True)
    parser.add_argument("--reference-run-root", required=True)
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed/scene0050_00")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows-csv", default="prompt_capsule_visibility_rows.csv")
    parser.add_argument("--points-json", default="prompt_capsule_visible_point_records.json")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--max-cases-per-lag", type=int, default=4)
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bf16", choices=["float32", "bf16", "float16"])
    parser.add_argument("--multimask-output", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    prompt_root = Path(args.prompt_probe_root)
    if not prompt_root.is_absolute():
        prompt_root = ROOT / prompt_root
    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root
    scene_root = Path(args.scene_root)
    if not scene_root.is_absolute():
        scene_root = ROOT / scene_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    probe_summary = read_json(prompt_root / "prompt_capsule_visibility_probe_summary.json")
    lingbot_hw = tuple(int(v) for v in probe_summary["raw_lingbot_geometry"].get("image_shape", [392, 518]))
    rows = load_rows(prompt_root / args.rows_csv)
    points_by_case = load_points(prompt_root / args.points_json)
    cases = choose_cases(rows, points_by_case, max_cases=int(args.max_cases), per_lag=int(args.max_cases_per_lag))
    reference_records = load_reference_records(reference_root)
    predictor, sam2_checkpoint = build_sam2_predictor(args)

    result_rows: list[dict[str, Any]] = []
    visual_paths: list[Path] = []
    current_frame_id: int | None = None
    current_rgb: np.ndarray | None = None
    labels_cache: dict[int, np.ndarray] = {}
    inference_start = time.time()
    for case_idx, row in enumerate(cases):
        frame_id = int(row["target_frame_id"])
        obj_id = int(row["target_obj_id"])
        if current_frame_id != frame_id:
            rgb_bgr = cv2.imread(str(scene_root / "color" / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(scene_root / "color" / f"{frame_id}.jpg")
            current_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
                predictor.set_image(current_rgb)
            current_frame_id = frame_id
        assert current_rgb is not None
        if frame_id not in labels_cache:
            labels_cache[frame_id] = load_label(Path(reference_records[frame_id]["label_path"]))
        target_label = labels_cache[frame_id]
        ref_mask = target_label == obj_id
        orig_hw = current_rgb.shape[:2]
        key = (int(row["target_frame_index"]), obj_id, int(row["source_frame_index"]))
        point_records = points_by_case[key]
        positives = [p for p in point_records if p.get("role") == "positive"]
        negatives = [p for p in point_records if p.get("role") == "negative"]
        neg_ids = {int(p["source_obj_id"]) for p in negatives}
        coords_pos = [map_lingbot_xy_to_original(float(p["target_x"]), float(p["target_y"]), lingbot_hw=lingbot_hw, orig_hw=orig_hw) for p in positives]
        coords_neg = [map_lingbot_xy_to_original(float(p["target_x"]), float(p["target_y"]), lingbot_hw=lingbot_hw, orig_hw=orig_hw) for p in negatives]
        variants = {
            "G1_pos": (coords_pos, [1] * len(coords_pos)),
            "G2_pos_neg": (coords_pos + coords_neg, [1] * len(coords_pos) + [0] * len(coords_neg)),
        }
        pred_masks: dict[str, np.ndarray] = {}
        case_metrics: dict[str, Any] = {}
        point_coords_for_vis = np.asarray(coords_pos + coords_neg, dtype=np.float32)
        point_labels_for_vis = np.asarray([1] * len(coords_pos) + [0] * len(coords_neg), dtype=np.int32)
        for variant, (coords, labels) in variants.items():
            if not coords:
                mask = np.zeros_like(ref_mask, dtype=bool)
                metrics = {"candidate_index": -1, "sam2_score": 0.0, **mask_metrics(mask, ref_mask, target_label, neg_ids)}
            else:
                point_coords = np.asarray(coords, dtype=np.float32)
                point_labels = np.asarray(labels, dtype=np.int32)
                with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
                    masks, scores, _logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=bool(args.multimask_output),
                    )
                mask, metrics = select_mask_by_score(masks, scores, ref_mask, target_label, neg_ids)
            pred_masks[variant] = mask
            for key_m, value in metrics.items():
                case_metrics[f"{variant}_{key_m}"] = value
        result = {
            "case_index": int(case_idx),
            "target_frame_id": frame_id,
            "target_obj_id": obj_id,
            "source_frame_id": int(row["source_frame_id"]),
            "source_lag": int(row.get("source_lag", 0)),
            "positive_point_count": int(len(coords_pos)),
            "negative_point_count": int(len(coords_neg)),
            **case_metrics,
        }
        result_rows.append(result)
        if len(visual_paths) < 16:
            title = (
                f"case={case_idx} frame={frame_id} obj={obj_id} lag={result['source_lag']} "
                f"G1_iou={result['G1_pos_iou_to_reference']:.3f} G2_iou={result['G2_pos_neg_iou_to_reference']:.3f}"
            )
            vis = draw_case(
                rgb=current_rgb,
                ref_mask=ref_mask,
                pred_masks=pred_masks,
                point_coords=point_coords_for_vis,
                point_labels=point_labels_for_vis,
                title=title,
            )
            path = output_root / "visual_overlays" / f"case{case_idx:03d}_frame{frame_id:06d}_obj{obj_id:04d}_lag{result['source_lag']:02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            visual_paths.append(path)

    inference_runtime = float(time.time() - inference_start)
    rows_csv = output_root / "sam2_prompt_benchmark_rows.csv"
    write_csv(rows_csv, result_rows)
    sheet = contact_sheet(visual_paths, output_root / "sam2_prompt_benchmark_contact_sheet.jpg", cols=1)
    def mean_of(key: str) -> float:
        vals = [float(row[key]) for row in result_rows if key in row]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "schema_version": "stream4d_v107_phase7_lingbot_sam2_prompt_benchmark_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "sam2_inference_runtime_sec": inference_runtime,
        "prompt_probe_root": rel(prompt_root),
        "reference_run_root": rel(reference_root),
        "scene_root": rel(scene_root),
        "case_count": len(result_rows),
        "sam2_checkpoint": rel(sam2_checkpoint),
        "sam2_checkpoint_sha256": sha256_file(sam2_checkpoint),
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "device": str(args.device),
        "model_dtype": str(args.model_dtype),
        "geometry_source": "LingBot-Map projected visible prompt points from prompt probe",
        "uses_scannet_pose_or_depth_for_projection": False,
        "G1_pos_mean_iou": mean_of("G1_pos_iou_to_reference"),
        "G2_pos_neg_mean_iou": mean_of("G2_pos_neg_iou_to_reference"),
        "G1_pos_mean_negative_sibling_overlap_rate": mean_of("G1_pos_negative_sibling_overlap_rate"),
        "G2_pos_neg_mean_negative_sibling_overlap_rate": mean_of("G2_pos_neg_negative_sibling_overlap_rate"),
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "visual_overlay_count": len(visual_paths),
        "visual_overlays": [rel(path) for path in visual_paths],
        "contact_sheet": rel(sheet) if sheet is not None else "",
        "contact_sheet_sha256": sha256_file(sheet) if sheet is not None else "",
        "audit_note": "Mask choice uses SAM2 predicted score, not reference IoU; reference labels are used only for evaluation.",
    }
    write_json(output_root / "sam2_prompt_benchmark_summary.json", summary)
    print(json.dumps({"summary": rel(output_root / "sam2_prompt_benchmark_summary.json"), "case_count": len(result_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
