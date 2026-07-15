#!/usr/bin/env python3
"""2D-only online reactivation shadow for v108 Phase10.

This tool prompts SAM2 image predictor on the target frame from a historical
2D capsule. It does not use LingBot geometry and does not mutate SAM2 memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = ROOT / "Stream3D"
GSAM2_ROOT = Path(os.environ.get("GSAM2_ROOT", str(ROOT / "Grounded-SAM-2"))).resolve()
for item in (ROOT, STREAM3D_ROOT, GSAM2_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from Stream3D.stream4d_v108.appearance_capsule import (  # noqa: E402
    cosine_similarity,
    rgb_shape_descriptor,
)
from Stream3D.stream4d_v108.geometry_capsule import (  # noqa: E402
    bbox_from_mask,
    sample_interior_points,
)
from Stream3D.stream4d_v108.reactivation import prompt_set_from_2d_capsule  # noqa: E402


DEFAULT_REFERENCE_ROOT = (
    ROOT
    / "Stream3D/outputs/audit/v107_phase35_scene0030_crossscene_p34mechanism_30f_20260714_1905"
    / "v107_phase8_g3_rolling_scheduler_smoke"
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str, base: Path = ROOT) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


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
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_records(reference_root: Path) -> dict[int, dict[str, Any]]:
    summary_path = reference_root / "summary.json"
    if not summary_path.exists():
        nested = reference_root / "v106_stateful_sam2_rolling_scene_stream" / "summary.json"
        if nested.exists():
            summary_path = nested
    summary = read_json(summary_path)
    records: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        item = dict(row)
        item["label_path"] = resolve_path(str(row["label_path"]), summary_path.parent)
        records[int(row["frame_id"])] = item
    return records


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def load_rgb(scene_root: Path, scene_id: str, frame_id: int) -> np.ndarray:
    bgr = cv2.imread(str(scene_root / scene_id / "color" / f"{int(frame_id)}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(scene_root / scene_id / "color" / f"{int(frame_id)}.jpg")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_ids(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def expand_box(box: tuple[int, int, int, int], image_hw: tuple[int, int], scale: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    h, w = int(image_hw[0]), int(image_hw[1])
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(1.0, x1 - x0 + 1.0) * float(scale)
    bh = max(1.0, y1 - y0 + 1.0) * float(scale)
    return (
        float(max(0.0, cx - 0.5 * bw)),
        float(max(0.0, cy - 0.5 * bh)),
        float(min(w - 1.0, cx + 0.5 * bw)),
        float(min(h - 1.0, cy + 0.5 * bh)),
    )


def random_box_like(box: tuple[float, float, float, float], image_hw: tuple[int, int], seed: int) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(int(seed))
    h, w = int(image_hw[0]), int(image_hw[1])
    bw = max(1.0, float(box[2] - box[0] + 1.0))
    bh = max(1.0, float(box[3] - box[1] + 1.0))
    max_x = max(0.0, w - bw)
    max_y = max(0.0, h - bh)
    x0 = float(rng.uniform(0.0, max_x)) if max_x > 0 else 0.0
    y0 = float(rng.uniform(0.0, max_y)) if max_y > 0 else 0.0
    return (x0, y0, min(w - 1.0, x0 + bw - 1.0), min(h - 1.0, y0 + bh - 1.0))


def mask_stats(mask: np.ndarray, target_mask: np.ndarray, label: np.ndarray, object_id: int) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    target_b = np.asarray(target_mask).astype(bool)
    h, w = mask_b.shape[:2]
    area = int(np.count_nonzero(mask_b))
    bbox = bbox_from_mask(mask_b)
    if bbox is None:
        bbox_list: list[int] = []
        bbox_area_frac = 0.0
        edge_touch = 0
    else:
        x0, y0, x1, y1 = bbox
        bbox_list = [int(x0), int(y0), int(x1), int(y1)]
        bbox_area_frac = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / max(1, h * w))
        edge_touch = int(x0 == 0) + int(y0 == 0) + int(x1 == w - 1) + int(y1 == h - 1)
    inter = int(np.count_nonzero(mask_b & target_b))
    union = int(np.count_nonzero(mask_b | target_b))
    other = np.asarray(label) > 0
    other &= np.asarray(label) != int(object_id)
    other_overlap = int(np.count_nonzero(mask_b & other))
    return {
        "candidate_area_px": int(area),
        "candidate_bbox_xyxy": bbox_list,
        "candidate_bbox_area_frac": float(bbox_area_frac),
        "candidate_edge_touch_count": int(edge_touch),
        "iou_to_reference_diagnostic": float(inter / max(union, 1)),
        "overlap_with_reference_px": int(inter),
        "overlap_with_other_object_px": int(other_overlap),
    }


def build_sam2_predictor(args: argparse.Namespace):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = resolve_path(str(args.sam2_checkpoint))
    model = build_sam2(str(args.sam2_model_cfg), str(checkpoint), device=str(args.device))
    dtype_name = str(args.model_dtype).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        model.to(dtype=torch.bfloat16)
    elif dtype_name in {"fp16", "float16"}:
        model.to(dtype=torch.float16)
    model.eval()
    return SAM2ImagePredictor(model), checkpoint


def autocast_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.device).startswith("cuda"):
        return {"enabled": False}
    dtype_name = str(args.model_dtype).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}
    if dtype_name in {"fp16", "float16"}:
        return {"device_type": "cuda", "dtype": torch.float16, "enabled": True}
    return {"enabled": False}


def predict_variant(
    *,
    predictor: Any,
    args: argparse.Namespace,
    point_coords: np.ndarray | None,
    point_labels: np.ndarray | None,
    box: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    kwargs: dict[str, Any] = {"multimask_output": bool(args.multimask_output)}
    if point_coords is not None and point_labels is not None and point_coords.size:
        kwargs["point_coords"] = np.asarray(point_coords, dtype=np.float32)
        kwargs["point_labels"] = np.asarray(point_labels, dtype=np.int32)
    if box is not None:
        kwargs["box"] = np.asarray(box, dtype=np.float32)
    if "point_coords" not in kwargs and "box" not in kwargs:
        raise ValueError("SAM2 image predictor requires a point or box prompt")
    with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
        masks, scores, _logits = predictor.predict(**kwargs)
    mask_arr = np.asarray(masks)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    score_arr = np.asarray(scores if scores is not None else np.zeros((mask_arr.shape[0],), dtype=np.float32)).reshape(-1)
    best_idx = int(np.argmax(score_arr)) if score_arr.size else 0
    best_idx = min(best_idx, mask_arr.shape[0] - 1)
    return np.squeeze(mask_arr[best_idx]) > 0, {
        "candidate_index": int(best_idx),
        "sam2_score_diagnostic": float(score_arr[best_idx]) if score_arr.size else 0.0,
        "candidate_count": int(mask_arr.shape[0]),
        "score_selection_note": "highest SAM2 score selected for visualization only; not an acceptance gate",
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.copy()
    mask_b = np.asarray(mask).astype(bool)
    if np.any(mask_b):
        c = np.asarray(color, dtype=np.float32)
        out[mask_b] = ((1.0 - float(alpha)) * out[mask_b].astype(np.float32) + float(alpha) * c).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return out


def draw_points(image: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> None:
    for idx, xy in enumerate(np.asarray(coords).reshape(-1, 2)):
        label = int(labels[idx])
        color = (30, 245, 70) if label == 1 else (250, 60, 50)
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        cv2.circle(image, (x, y), 7, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (x, y), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_box(image: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header = 34
    out = np.zeros((image.shape[0] + header, image.shape[1], 3), dtype=np.uint8)
    out[:] = 12
    out[header:] = image
    cv2.putText(out, text[:170], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def padded_crop_bounds(
    *,
    image_hw: tuple[int, int],
    masks: list[np.ndarray],
    boxes: list[tuple[float, float, float, float] | tuple[int, int, int, int]],
    pad_px: int,
) -> tuple[int, int, int, int]:
    h, w = int(image_hw[0]), int(image_hw[1])
    xs: list[int] = []
    ys: list[int] = []
    for mask in masks:
        bbox = bbox_from_mask(np.asarray(mask).astype(bool))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        xs.extend([int(x0), int(x1)])
        ys.extend([int(y0), int(y1)])
    for box in boxes:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        xs.extend([x0, x1])
        ys.extend([y0, y1])
    if not xs or not ys:
        return (0, 0, w - 1, h - 1)
    x0 = max(0, min(xs) - int(pad_px))
    y0 = max(0, min(ys) - int(pad_px))
    x1 = min(w - 1, max(xs) + int(pad_px))
    y1 = min(h - 1, max(ys) + int(pad_px))
    return (int(x0), int(y0), int(x1), int(y1))


def crop_image_and_mask(image: np.ndarray, mask: np.ndarray, crop_xyxy: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = crop_xyxy
    return image[y0 : y1 + 1, x0 : x1 + 1].copy(), mask[y0 : y1 + 1, x0 : x1 + 1].copy()


def shift_points(points_xy: np.ndarray, crop_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.copy()
    shifted = np.asarray(points_xy, dtype=np.float32).copy()
    shifted[:, 0] -= float(crop_xyxy[0])
    shifted[:, 1] -= float(crop_xyxy[1])
    return shifted


def shift_box(box: tuple[float, float, float, float], crop_xyxy: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    return (
        float(box[0] - crop_xyxy[0]),
        float(box[1] - crop_xyxy[1]),
        float(box[2] - crop_xyxy[0]),
        float(box[3] - crop_xyxy[1]),
    )


def make_zoom_panel(
    *,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    candidate_masks: dict[str, np.ndarray],
    prompt_points: np.ndarray,
    prompt_labels: np.ndarray,
    prompt_box: tuple[float, float, float, float],
    out_path: Path,
    title: str,
) -> Path:
    source_crop = padded_crop_bounds(image_hw=source_rgb.shape[:2], masks=[source_mask], boxes=[], pad_px=42)
    target_crop = padded_crop_bounds(
        image_hw=target_rgb.shape[:2],
        masks=[target_mask, *candidate_masks.values()],
        boxes=[prompt_box],
        pad_px=42,
    )
    src_rgb_c, src_mask_c = crop_image_and_mask(source_rgb, source_mask, source_crop)
    tgt_rgb_c, tgt_ref_c = crop_image_and_mask(target_rgb, target_mask, target_crop)
    prompt_points_c = shift_points(prompt_points, target_crop)
    prompt_box_c = shift_box(prompt_box, target_crop)

    source = overlay_mask(src_rgb_c, src_mask_c, color=(40, 220, 255), alpha=0.42)
    source = add_header(source, "source crop")
    target_ref = overlay_mask(tgt_rgb_c, tgt_ref_c, color=(40, 220, 255), alpha=0.30)
    draw_box(target_ref, prompt_box_c, (255, 220, 40))
    if prompt_points_c.size:
        draw_points(target_ref, prompt_points_c, prompt_labels)
    target_ref = add_header(target_ref, "target crop reference + prompt")

    panels = [source, target_ref]
    for variant, mask in candidate_masks.items():
        _rgb_c, cand_c = crop_image_and_mask(target_rgb, mask, target_crop)
        panel = overlay_mask(tgt_rgb_c, tgt_ref_c, color=(40, 220, 255), alpha=0.18)
        panel = overlay_mask(panel, cand_c, color=(255, 70, 190), alpha=0.45)
        draw_box(panel, prompt_box_c, (255, 220, 40))
        if prompt_points_c.size:
            draw_points(panel, prompt_points_c, prompt_labels)
        panels.append(add_header(panel, f"{variant} crop; visual review required"))

    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] == height:
            padded.append(panel)
            continue
        out = np.zeros((height, panel.shape[1], 3), dtype=np.uint8)
        out[:] = 12
        out[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(out)
    merged = np.concatenate(padded, axis=1)
    final = add_header(merged, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    return out_path


def make_panel(
    *,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    candidate_masks: dict[str, np.ndarray],
    prompt_points: np.ndarray,
    prompt_labels: np.ndarray,
    prompt_box: tuple[float, float, float, float],
    out_path: Path,
    title: str,
) -> Path:
    source = overlay_mask(source_rgb, source_mask, color=(40, 220, 255), alpha=0.38)
    source = add_header(source, "source dormant 2D capsule")
    target_ref = overlay_mask(target_rgb, target_mask, color=(40, 220, 255), alpha=0.28)
    draw_box(target_ref, prompt_box, (255, 220, 40))
    if prompt_points.size:
        draw_points(target_ref, prompt_points, prompt_labels)
    target_ref = add_header(target_ref, "target reference overlay + 2D prompt")
    panels = [source, target_ref]
    for variant, mask in candidate_masks.items():
        panel = overlay_mask(target_rgb, target_mask, color=(40, 220, 255), alpha=0.18)
        panel = overlay_mask(panel, mask, color=(255, 70, 190), alpha=0.42)
        draw_box(panel, prompt_box, (255, 220, 40))
        if prompt_points.size:
            draw_points(panel, prompt_points, prompt_labels)
        panels.append(add_header(panel, f"{variant}; visual review required"))
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] == height:
            padded.append(panel)
            continue
        out = np.zeros((height, panel.shape[1], 3), dtype=np.uint8)
        out[:] = 12
        out[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(out)
    merged = np.concatenate(padded, axis=1)
    final = add_header(merged, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0030_00")
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-frame-id", type=int, required=True)
    parser.add_argument("--target-frame-id", type=int, required=True)
    parser.add_argument("--object-ids", required=True)
    parser.add_argument("--box-expand", type=float, default=1.15)
    parser.add_argument("--positive-points", type=int, default=4)
    parser.add_argument("--source-core-min-distance-px", type=float, default=12.0)
    parser.add_argument("--multimask-output", type=int, default=1)
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bf16", choices=["float32", "bf16", "float16"])
    parser.add_argument("--seed", type=int, default=10810)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    scene_root = resolve_path(str(args.scene_root))
    reference_root = resolve_path(str(args.reference_run_root))
    output_root = resolve_path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    records = load_reference_records(reference_root)
    for frame_id in [int(args.source_frame_id), int(args.target_frame_id)]:
        if frame_id not in records:
            raise RuntimeError({"missing_frame": frame_id, "reference_root": rel(reference_root)})
    source_label = load_label(Path(records[int(args.source_frame_id)]["label_path"]))
    target_label = load_label(Path(records[int(args.target_frame_id)]["label_path"]))
    source_rgb = load_rgb(scene_root, str(args.scene_id), int(args.source_frame_id))
    target_rgb = load_rgb(scene_root, str(args.scene_id), int(args.target_frame_id))
    predictor, checkpoint = build_sam2_predictor(args)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
        predictor.set_image(target_rgb)

    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case_idx, object_id in enumerate(parse_ids(str(args.object_ids))):
        source_mask = source_label == int(object_id)
        target_mask = target_label == int(object_id)
        bbox = bbox_from_mask(source_mask)
        if bbox is None:
            continue
        prompt_box = expand_box(bbox, target_rgb.shape[:2], float(args.box_expand))
        points_yx, sample_stats = sample_interior_points(
            source_mask,
            count=int(args.positive_points),
            min_distance_px=float(args.source_core_min_distance_px),
            seed=int(args.seed) + int(object_id) * 17 + int(case_idx),
        )
        prompt_points = np.asarray([[float(x), float(y)] for y, x, _dist in points_yx], dtype=np.float32)
        prompt_labels = np.ones((prompt_points.shape[0],), dtype=np.int32)
        random_box = random_box_like(prompt_box, target_rgb.shape[:2], int(args.seed) + int(object_id) * 313)
        prompt_set = prompt_set_from_2d_capsule(
            frame_id=int(args.target_frame_id),
            target_global_object_id=int(object_id),
            positive_uv=[(float(x), float(y)) for y, x, _dist in points_yx],
            box_xyxy=prompt_box,
            source="2d_last_mask_capsule_no_lingbot",
        )
        variants: dict[str, tuple[np.ndarray | None, np.ndarray | None, tuple[float, float, float, float] | None]] = {
            "P1_points_only": (prompt_points, prompt_labels, None),
            "B1_box_only": (None, None, prompt_box),
            "PB1_points_box": (prompt_points, prompt_labels, prompt_box),
            "RANDOM_box_control": (None, None, random_box),
        }
        candidate_masks: dict[str, np.ndarray] = {}
        source_vec, source_meta = rgb_shape_descriptor(source_rgb, source_mask)
        for variant, (coords, labels, box) in variants.items():
            candidate, select = predict_variant(
                predictor=predictor,
                args=args,
                point_coords=coords,
                point_labels=labels,
                box=box,
            )
            candidate_masks[variant] = candidate
            cand_vec, cand_meta = rgb_shape_descriptor(target_rgb, candidate)
            row = {
                "case_index": int(case_idx),
                "scene_id": str(args.scene_id),
                "object_id": int(object_id),
                "source_frame_id": int(args.source_frame_id),
                "target_frame_id": int(args.target_frame_id),
                "variant": str(variant),
                "prompt_source": prompt_set.source,
                "positive_point_count": int(prompt_set.positive_count),
                "negative_point_count": int(prompt_set.negative_count),
                "box_xyxy": list(box) if box is not None else [],
                "source_bbox_xyxy": list(bbox),
                "source_sample_stats": sample_stats,
                "source_descriptor_meta": source_meta,
                "candidate_descriptor_meta": cand_meta,
                "rgb_shape_similarity_diagnostic": cosine_similarity(source_vec, cand_vec),
                "uses_lingbot_geometry": False,
                "occlusion_checked": False,
                "occlusion_note": "not checked in Phase10 2D-only shadow",
                **select,
                **mask_stats(candidate, target_mask, target_label, int(object_id)),
                "metrics_are_diagnostic_only": True,
                "visual_review_required": True,
            }
            rows.append(row)
        empty = np.zeros_like(target_mask, dtype=bool)
        rows.append(
            {
                "case_index": int(case_idx),
                "scene_id": str(args.scene_id),
                "object_id": int(object_id),
                "source_frame_id": int(args.source_frame_id),
                "target_frame_id": int(args.target_frame_id),
                "variant": "NO_WATCHER_empty_control",
                "prompt_source": "no_watcher_control",
                "positive_point_count": 0,
                "negative_point_count": 0,
                "box_xyxy": [],
                "source_bbox_xyxy": list(bbox),
                "uses_lingbot_geometry": False,
                "occlusion_checked": False,
                "occlusion_note": "not checked in Phase10 2D-only shadow",
                "candidate_index": -1,
                "sam2_score_diagnostic": 0.0,
                "candidate_count": 0,
                "score_selection_note": "control has no SAM2 prompt",
                **mask_stats(empty, target_mask, target_label, int(object_id)),
                "metrics_are_diagnostic_only": True,
                "visual_review_required": True,
            }
        )
        visual_variants = {name: candidate_masks[name] for name in ["B1_box_only", "PB1_points_box", "RANDOM_box_control"]}
        panel_path = output_root / "visual_checks" / (
            f"case_{case_idx:02d}_{args.scene_id}_src{int(args.source_frame_id):06d}_"
            f"tgt{int(args.target_frame_id):06d}_obj{int(object_id):04d}.png"
        )
        make_panel(
            source_rgb=source_rgb,
            target_rgb=target_rgb,
            source_mask=source_mask,
            target_mask=target_mask,
            candidate_masks=visual_variants,
            prompt_points=prompt_points,
            prompt_labels=prompt_labels,
            prompt_box=prompt_box,
            out_path=panel_path,
            title=f"Phase10 2D reactivation shadow scene={args.scene_id} obj={object_id}; metrics diagnostic only",
        )
        zoom_path = output_root / "visual_checks" / (
            f"zoom_case_{case_idx:02d}_{args.scene_id}_src{int(args.source_frame_id):06d}_"
            f"tgt{int(args.target_frame_id):06d}_obj{int(object_id):04d}.png"
        )
        make_zoom_panel(
            source_rgb=source_rgb,
            target_rgb=target_rgb,
            source_mask=source_mask,
            target_mask=target_mask,
            candidate_masks=visual_variants,
            prompt_points=prompt_points,
            prompt_labels=prompt_labels,
            prompt_box=prompt_box,
            out_path=zoom_path,
            title=f"Phase10 zoom scene={args.scene_id} obj={object_id}; visual judgment is primary",
        )
        case_summaries.append(
            {
                "case_index": int(case_idx),
                "scene_id": str(args.scene_id),
                "object_id": int(object_id),
                "source_frame_id": int(args.source_frame_id),
                "target_frame_id": int(args.target_frame_id),
                "source_bbox_xyxy": list(bbox),
                "prompt_box_xyxy": list(prompt_box),
                "prompt_positive_points_xy": prompt_points.tolist(),
                "visual_path": rel(panel_path),
                "visual_sha256": sha256_file(panel_path),
                "visual_zoom_path": rel(zoom_path),
                "visual_zoom_sha256": sha256_file(zoom_path),
                "uses_lingbot_geometry": False,
                "visual_review_required": True,
            }
        )

    rows_csv = output_root / "phase10_2d_reactivation_rows.csv"
    write_csv(rows_csv, rows)
    case_path = output_root / "case_summaries.json"
    write_json(case_path, {"cases": case_summaries})
    peak_cuda_mb = 0.0
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_cuda_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    summary_path = output_root / "phase10_2d_reactivation_shadow_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase10_2d_reactivation_shadow_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scene_id": str(args.scene_id),
        "source_frame_id": int(args.source_frame_id),
        "target_frame_id": int(args.target_frame_id),
        "object_ids": parse_ids(str(args.object_ids)),
        "case_count": int(len(case_summaries)),
        "row_count": int(len(rows)),
        "sam2_checkpoint": rel(checkpoint),
        "sam2_checkpoint_sha256": sha256_file(checkpoint),
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "device": str(args.device),
        "model_dtype": str(args.model_dtype),
        "peak_cuda_allocated_mb": float(peak_cuda_mb),
        "uses_lingbot_geometry": False,
        "occlusion_checked": False,
        "candidate_universe": "SAM2 image predictor generated masks from historical 2D capsule prompts; not base-mask filtering",
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "case_summaries": rel(case_path),
        "case_summaries_sha256": sha256_file(case_path),
        "visual_paths": [case["visual_path"] for case in case_summaries],
        "visual_zoom_paths": [case["visual_zoom_path"] for case in case_summaries],
        "visual_sha256": {case["visual_path"]: case["visual_sha256"] for case in case_summaries},
        "visual_zoom_sha256": {case["visual_zoom_path"]: case["visual_zoom_sha256"] for case in case_summaries},
        "controls": ["RANDOM_box_control", "NO_WATCHER_empty_control"],
        "acceptance_rule": "Metrics are diagnostic only; quality must be judged by high-resolution visual review.",
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "case_count": len(case_summaries), "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
