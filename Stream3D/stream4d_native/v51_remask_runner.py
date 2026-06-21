from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d_native.v47_common import ROOT, utc_now, write_json


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _parse_frames(frame_ids: str) -> list[int]:
    out: list[int] = []
    for part in str(frame_ids).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) for x in part.split(":")]
            if len(bits) == 2:
                start, stop = bits
                step = 1
            elif len(bits) == 3:
                start, stop, step = bits
            else:
                raise ValueError(f"Invalid frame range: {part}")
            out.extend(list(range(start, stop, step)))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def _load_rgb(scene: str, frame_id: int) -> np.ndarray:
    path = ROOT / "data/scannet/processed" / scene / "color" / f"{int(frame_id)}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _metadata_array(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(payload, sort_keys=True), dtype=object)


def _build_sam2_generator(
    sam2_root: str | Path,
    checkpoint: str | Path,
    model_cfg: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: int,
    min_mask_region_area: int,
) -> Any:
    sam2_root = Path(sam2_root).resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(str(model_cfg), str(Path(checkpoint).resolve()), device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        box_nms_thresh=float(box_nms_thresh),
        crop_n_layers=int(crop_n_layers),
        crop_nms_thresh=float(crop_nms_thresh),
        crop_overlap_ratio=float(crop_overlap_ratio),
        crop_n_points_downscale_factor=int(crop_n_points_downscale_factor),
        min_mask_region_area=int(min_mask_region_area),
    )


def run_sam2_remask(
    scene: str,
    frame_ids: str,
    output_root: str | Path,
    sam2_root: str | Path,
    checkpoint: str | Path,
    model_cfg: str,
    device: str = "cuda",
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    crop_n_layers: int = 0,
    crop_nms_thresh: float = 0.7,
    crop_overlap_ratio: float = 0.3413333333333333,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 0,
    min_area: int = 400,
) -> dict[str, Any]:
    frames = _parse_frames(frame_ids)
    output_root = ROOT / output_root if not Path(output_root).is_absolute() else Path(output_root)
    scene_dir = output_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    generator = _build_sam2_generator(
        sam2_root=sam2_root,
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        device=device,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )
    frame_rows: list[dict[str, Any]] = []
    total_masks = 0
    for frame_id in frames:
        frame_start = time.time()
        image = _load_rgb(scene, frame_id)
        anns = generator.generate(image)
        kept: list[dict[str, Any]] = []
        for ann in anns:
            mask = np.asarray(ann.get("segmentation"), dtype=bool)
            area = int(mask.sum())
            if area < int(min_area):
                continue
            kept.append(ann)
        kept.sort(key=lambda item: int(np.asarray(item.get("segmentation"), dtype=bool).sum()), reverse=True)
        if kept:
            masks = np.stack([np.asarray(ann["segmentation"], dtype=bool) for ann in kept], axis=0)
            scores = np.asarray([float(ann.get("predicted_iou", ann.get("stability_score", 1.0))) for ann in kept], dtype=np.float32)
            stability = np.asarray([float(ann.get("stability_score", np.nan)) for ann in kept], dtype=np.float32)
            areas = np.asarray([int(np.asarray(ann["segmentation"], dtype=bool).sum()) for ann in kept], dtype=np.int64)
            boxes = np.asarray([ann.get("bbox", [0, 0, 0, 0]) for ann in kept], dtype=np.float32)
        else:
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
            scores = np.zeros((0,), dtype=np.float32)
            stability = np.zeros((0,), dtype=np.float32)
            areas = np.zeros((0,), dtype=np.int64)
            boxes = np.zeros((0, 4), dtype=np.float32)
        metadata = {
            "source": "sam2_automatic",
            "model_cfg": str(model_cfg),
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_found": Path(checkpoint).exists(),
            "points_per_side": int(points_per_side),
            "pred_iou_thresh": float(pred_iou_thresh),
            "stability_score_thresh": float(stability_score_thresh),
            "box_nms_thresh": float(box_nms_thresh),
            "crop_n_layers": int(crop_n_layers),
            "crop_nms_thresh": float(crop_nms_thresh),
            "crop_overlap_ratio": float(crop_overlap_ratio),
            "crop_n_points_downscale_factor": int(crop_n_points_downscale_factor),
            "min_mask_region_area": int(min_mask_region_area),
            "min_area": int(min_area),
            "uses_gt_for_prediction": False,
            "preserves_nxhxw_stack": True,
        }
        out_path = scene_dir / f"sam2_frame{int(frame_id):06d}_masks.npz"
        np.savez_compressed(
            out_path,
            masks=masks,
            scores=scores,
            stability_scores=stability,
            boxes=boxes,
            areas=areas,
            source_metadata=_metadata_array(metadata),
        )
        total_masks += int(masks.shape[0])
        frame_rows.append(
            {
                "scene": scene,
                "frame_id": int(frame_id),
                "output_path": _rel(out_path),
                "rgb_shape": list(image.shape),
                "generated_mask_count_raw": int(len(anns)),
                "kept_mask_count": int(masks.shape[0]),
                "runtime_sec": time.time() - frame_start,
                "preserves_nxhxw_stack": True,
                "uses_gt_for_prediction": False,
            }
        )
    summary = {
        "phase": "v51_r2_sam2_remask_prepare",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "scene": scene,
        "frame_ids": frames,
        "output_root": _rel(output_root),
        "frame_count": len(frames),
        "total_kept_masks": int(total_masks),
        "mean_masks_per_frame": total_masks / max(len(frames), 1),
        "runtime_sec": time.time() - start,
        "sam2_root": str(Path(sam2_root).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_found": Path(checkpoint).exists(),
        "model_cfg": str(model_cfg),
        "device": device,
        "points_per_side": int(points_per_side),
        "pred_iou_thresh": float(pred_iou_thresh),
        "stability_score_thresh": float(stability_score_thresh),
        "box_nms_thresh": float(box_nms_thresh),
        "crop_n_layers": int(crop_n_layers),
        "crop_nms_thresh": float(crop_nms_thresh),
        "crop_overlap_ratio": float(crop_overlap_ratio),
        "crop_n_points_downscale_factor": int(crop_n_points_downscale_factor),
        "min_mask_region_area": int(min_mask_region_area),
        "min_area": int(min_area),
        "uses_gt_for_prediction": False,
        "preserves_nxhxw_stack": True,
        "frame_rows": frame_rows,
    }
    write_json(output_root / "sam2_remask_prepare_summary.json", summary)
    return summary
