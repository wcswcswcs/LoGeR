#!/usr/bin/env python3
"""Probe GT-free frame0 residual prompts for v105 piano-miss diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_gapadaptive_sam2.generated.yaml"
DEFAULT_SCANNET_ROOT = STREAM3D_ROOT / "data" / "scannet" / "processed"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    load_config,
    make_args,
    sample_component_adaptive_points_yx,
    setup_models,
    uncovered_from_masks,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    overlay_label,
    read_rgb,
    sha256_file,
    stable_seed,
)
from tools.build_v105_phase6_speculative_gap_birth import (  # noqa: E402
    filter_birth_masks,
    run_sam2_point_segment_choice_candidate_support,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16)


def _component_stats(mask: np.ndarray, min_area: int) -> dict[str, Any]:
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, int(n_labels))]
    kept = [area for area in areas if area >= int(min_area)]
    return {
        "component_count": int(len(areas)),
        "kept_component_count": int(len(kept)),
        "max_component_area": int(max(areas) if areas else 0),
        "area_ge_min_total": int(sum(kept)),
    }


def _piano_metrics(label: np.ndarray, scene_root: Path, frame_id: int, semantic_id: int) -> dict[str, Any]:
    sem = _read_label(scene_root / "label-filt" / f"{frame_id}.png")
    target = sem == int(semantic_id)
    covered = target & (label > 0)
    values, counts = np.unique(label[target], return_counts=True)
    top = []
    target_area = int(target.sum())
    for value, count in sorted(zip(values.tolist(), counts.tolist()), key=lambda pair: -pair[1])[:8]:
        top.append(
            {
                "pred_id": int(value),
                "overlap_px": int(count),
                "coverage_vs_target": float(count / target_area) if target_area else None,
                "pred_id_total_px": int((label == int(value)).sum()),
            }
        )
    return {
        "semantic_id": int(semantic_id),
        "semantic_area_px": int(target_area),
        "covered_px": int(covered.sum()),
        "uncovered_px": int((target & (label == 0)).sum()),
        "coverage_ratio": float(covered.sum() / target_area) if target_area else None,
        "top_pred_ids": top,
    }


def _masks_from_label(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([int(v) for v in np.unique(label) if int(v) > 0], dtype=np.int64)
    masks = np.stack([(label == int(v)) for v in ids], axis=0).astype(bool) if ids.size else np.zeros((0, *label.shape), dtype=bool)
    return ids, masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--base-label", required=True)
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--semantic-id", type=int, default=90)
    parser.add_argument("--choice-policy", default="max_candidate_support_valid_mask_per_point")
    parser.add_argument("--max-points", type=int, default=96)
    parser.add_argument("--min-component-area", type=int, default=800)
    parser.add_argument("--base-points-per-component", type=int, default=1)
    parser.add_argument("--area-per-extra-point", type=int, default=40000)
    parser.add_argument("--max-points-per-component", type=int, default=12)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.8)
    parser.add_argument("--min-birth-mask-area", type=int, default=100)
    parser.add_argument("--min-candidate-touch-area", type=int, default=32)
    parser.add_argument("--min-candidate-touch-ratio", type=float, default=0.01)
    parser.add_argument("--max-existing-overlap-ratio", type=float, default=0.95)
    parser.add_argument("--max-core-overlap-ratio", type=float, default=1.0)
    parser.add_argument("--max-births", type=int, default=8)
    parser.add_argument("--seed", type=int, default=105)
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = load_config(config_path)
    base_args = make_args(config, argparse.Namespace(config=str(config_path), scene_id=args.scene_id, rgb_root=None, frame_start=None, frame_stride=None, frame_count=1, frame_ids=str(args.frame_id), output_root=args.output_root, seed=args.seed, birth_dump_dir=None))
    base_args.offload_video_to_cpu = False
    base_args.offload_state_to_cpu = False
    base_args.frame_ids = str(args.frame_id)
    base_args.frame_count = 1
    base_args.output_root = str(_resolve(args.output_root))

    out_root = _resolve(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scene_root = _resolve(args.scannet_root) / args.scene_id
    rgb = read_rgb(scene_root / "color" / f"{int(args.frame_id)}.jpg")
    h, w = rgb.shape[:2]
    base_label_path = _resolve(args.base_label)
    base_label = _read_label(base_label_path)
    if base_label.shape != (h, w):
        raise ValueError(f"base label shape {base_label.shape} != rgb {(h, w)}")

    ids, masks = _masks_from_label(base_label)
    current_union = np.any(masks, axis=0) if masks.size else np.zeros((h, w), dtype=bool)
    candidate = uncovered_from_masks(masks, h, w).astype(bool)
    candidate_stats = _component_stats(candidate, int(args.min_component_area))
    seed = stable_seed(args.seed, args.scene_id, args.frame_id, args.choice_policy, "frame0-residual-prompt-probe")
    points_yx, point_meta = sample_component_adaptive_points_yx(
        candidate,
        max_points=int(args.max_points),
        min_component_area=int(args.min_component_area),
        base_points_per_component=int(args.base_points_per_component),
        area_per_extra_point=int(args.area_per_extra_point),
        max_points_per_component=int(args.max_points_per_component),
        seed=int(seed),
    )

    setup_t0 = time.time()
    models = setup_models(base_args)
    segmentor = models["segmentor"]
    setup_sec = time.time() - setup_t0
    decode_t0 = time.time()
    if int(points_yx.shape[0]) > 0:
        raw_masks, birth_stats = run_sam2_point_segment_choice_candidate_support(
            segmentor,
            rgb,
            points_yx=points_yx,
            support_mask=candidate,
            points_per_batch=int(base_args.points_per_batch),
            choice_policy=str(args.choice_policy),
            iou_threshold=float(args.pred_iou_thresh),
            stability_threshold=float(args.stability_score_thresh),
            stability_score_offset=float(base_args.stability_score_offset),
            model_mask_thresh=float(base_args.model_mask_thresh),
            box_nms_thresh=float(base_args.box_nms_thresh),
            empty_ratio=float(base_args.empty_ratio),
            apply_box_nms=False,
            nms_score_type="stability",
            support_min_area=int(args.min_candidate_touch_area),
            support_min_ratio=float(args.min_candidate_touch_ratio),
        )
    else:
        raw_masks = np.zeros((0, h, w), dtype=bool)
        birth_stats = {
            "choice_policy": str(args.choice_policy),
            "raw_multimask_option_count": 0,
            "prompt_with_good_mask_count": 0,
            "candidate_supported_option_count": 0,
            "prompt_with_candidate_supported_mask_count": 0,
            "pre_nms_mask_count": 0,
            "post_disjoint_mask_count": 0,
        }
    decode_sec = time.time() - decode_t0

    filtered, filter_records = filter_birth_masks(
        raw_masks,
        candidate=candidate,
        current_union=current_union,
        core=np.zeros((h, w), dtype=bool),
        min_birth_mask_area=int(args.min_birth_mask_area),
        min_candidate_touch_area=int(args.min_candidate_touch_area),
        min_candidate_touch_ratio=float(args.min_candidate_touch_ratio),
        max_existing_overlap_ratio=float(args.max_existing_overlap_ratio),
        max_core_overlap_ratio=float(args.max_core_overlap_ratio),
    )
    if filtered.size and int(args.max_births) > 0:
        areas = np.count_nonzero(filtered.reshape(filtered.shape[0], -1), axis=1)
        keep = np.argsort(areas)[::-1][: int(args.max_births)]
        keep.sort()
        accepted = filtered[keep]
    else:
        accepted = filtered

    next_id = int(ids.max()) + 1 if ids.size else 1
    merged_label = base_label.astype(np.uint16, copy=True)
    accepted_write_rows = []
    for local_idx, mask in enumerate(accepted.astype(bool)):
        write_mask = mask & (merged_label == 0)
        obj_id = int(next_id + local_idx)
        merged_label[write_mask] = obj_id
        accepted_write_rows.append(
            {
                "local_index": int(local_idx),
                "obj_id": int(obj_id),
                "raw_mask_area": int(mask.sum()),
                "written_area": int(write_mask.sum()),
                "dropped_existing_overlap_area": int((mask & (merged_label > 0) & ~write_mask).sum()),
            }
        )

    label_path = out_root / "frame0_residual_probe_label.png"
    overlay_path = out_root / "frame0_residual_probe_overlay.jpg"
    cv2.imwrite(str(label_path), merged_label.astype(np.uint16))
    cv2.imwrite(str(overlay_path), overlay_label(rgb, merged_label, alpha=0.58))

    accepted_rows = []
    mask_dir = out_root / "accepted_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(accepted.astype(bool)):
        path = mask_dir / f"accepted_{idx:03d}.png"
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        accepted_rows.append(
            {
                "local_index": int(idx),
                "obj_id": int(accepted_write_rows[idx]["obj_id"]),
                "mask_path": str(path),
                "mask_area": int(mask.sum()),
                "written_area": int(accepted_write_rows[idx]["written_area"]),
                "candidate_touch_area": int((mask & candidate).sum()),
                "existing_overlap_area": int((mask & current_union).sum()),
                "sha256": sha256_file(path),
            }
        )

    summary = {
        "schema_version": "stream4d_v105_frame0_residual_prompt_probe_v1",
        "scene_id": str(args.scene_id),
        "frame_id": int(args.frame_id),
        "config_path": str(config_path),
        "base_label": str(base_label_path),
        "base_label_sha256": sha256_file(base_label_path),
        "candidate_policy": "uncovered_from_existing_frame0_label_gt_free",
        "candidate_area": int(candidate.sum()),
        "candidate_stats": candidate_stats,
        "point_count": int(points_yx.shape[0]),
        "point_sampling_meta": point_meta,
        "setup_sec": float(setup_sec),
        "decode_sec": float(decode_sec),
        "birth_stats": birth_stats,
        "raw_birth_mask_count": int(raw_masks.shape[0]),
        "filtered_birth_mask_count": int(filtered.shape[0]),
        "accepted_birth_mask_count": int(accepted.shape[0]),
        "accepted_masks": accepted_rows,
        "filter_records": filter_records[:128],
        "base_semantic_metrics": _piano_metrics(base_label, scene_root, int(args.frame_id), int(args.semantic_id)),
        "merged_semantic_metrics": _piano_metrics(merged_label, scene_root, int(args.frame_id), int(args.semantic_id)),
        "outputs": {
            "label": str(label_path),
            "label_sha256": sha256_file(label_path),
            "overlay": str(overlay_path),
            "overlay_sha256": sha256_file(overlay_path),
        },
    }
    summary_path = out_root / "frame0_residual_prompt_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["outputs"]["summary"] = str(summary_path)
    summary["outputs"]["summary_sha256"] = sha256_file(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
