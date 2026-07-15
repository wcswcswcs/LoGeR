#!/usr/bin/env python3
"""Build a real v106 Phase7 inherited-only proxy speculative birth bank.

This tool is intentionally narrower than the full Phase7 production pipeline:
it starts from the C2 inherited-only handoff replay labels, derives all-frame
proxy-uncovered components, links them into simple tubes, decodes SAM2 point
prompts at tube anchors, and writes a birth_records.json that can be replayed by
tools/build_v105_phase5_frozen_birth_replay.py.

It does not use reference labels or exact B6 birth records to construct
candidates, prompts, or masks. Reference paths are only for downstream replay
and metric evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_gapadaptive_sam2.generated.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    load_config,
    make_args,
    run_sam2_point_segment_choice,
    sample_component_adaptive_points_yx,
    setup_models,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    parse_frame_ids,
    read_rgb,
    sha256_file,
    stable_seed,
)
from tools.build_v105_phase6_speculative_gap_birth import (  # noqa: E402
    component_area_stats,
    filter_birth_masks,
)


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_baseline_args(config_path: Path, cli: argparse.Namespace) -> SimpleNamespace:
    config = load_config(config_path)
    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=cli.scene_id,
        rgb_root=cli.rgb_root,
        frame_start=cli.frame_start,
        frame_stride=cli.frame_stride,
        frame_count=cli.frame_count,
        frame_ids=cli.frame_ids,
        output_root=cli.output_root,
        seed=cli.seed,
        birth_dump_dir="",
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(cli.output_root)
    return args


def summary_label_map(summary_path: Path) -> dict[int, Path]:
    payload = load_json(summary_path)
    out: dict[int, Path] = {}
    for row in payload.get("records", []):
        frame_id = int(row["frame_id"])
        label_path = resolve_path(str(row["label_path"]))
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        out[frame_id] = label_path
    return out


def load_label(path: Path, h: int, w: int) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    if label.shape[:2] != (h, w):
        label = cv2.resize(label.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.uint16, copy=False)


def write_mask(mask: np.ndarray, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    if not ok:
        raise IOError(f"failed to write mask: {path}")
    return int(np.count_nonzero(mask))


def component_records(mask: np.ndarray, *, frame_idx: int, frame_id: int, min_area: int, max_components: int) -> list[dict[str, Any]]:
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rows: list[dict[str, Any]] = []
    for label_id in range(1, int(n_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        rows.append(
            {
                "frame_index": int(frame_idx),
                "frame_id": int(frame_id),
                "component_label": int(label_id),
                "component_area": int(area),
                "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                "centroid_xy": [float(cx), float(cy)],
                "_mask": (labels == int(label_id)).astype(bool),
            }
        )
    rows.sort(key=lambda row: int(row["component_area"]), reverse=True)
    if int(max_components) > 0:
        rows = rows[: int(max_components)]
    for rank, row in enumerate(rows):
        row["component_rank"] = int(rank)
        row["component_id"] = f"f{int(frame_idx):03d}_c{int(rank):03d}"
    return rows


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return float(inter / union) if union else 0.0


def centroid_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = a["centroid_xy"]
    bx, by = b["centroid_xy"]
    return float(((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2) ** 0.5)


def link_tubes(
    components_by_frame: list[list[dict[str, Any]]],
    *,
    min_link_iou: float,
    max_centroid_distance: float,
) -> list[dict[str, Any]]:
    tubes: list[dict[str, Any]] = []
    active: dict[int, dict[str, Any]] = {}
    for frame_components in components_by_frame:
        next_active: dict[int, dict[str, Any]] = {}
        used_tubes: set[int] = set()
        for comp in frame_components:
            best_tube_idx: int | None = None
            best_score = -1.0
            for tube_idx, prev in active.items():
                if tube_idx in used_tubes:
                    continue
                iou = mask_iou(comp["_mask"], prev["_mask"])
                dist = centroid_distance(comp, prev)
                if iou < float(min_link_iou) and dist > float(max_centroid_distance):
                    continue
                score = float(iou) - 0.001 * float(dist)
                if score > best_score:
                    best_score = score
                    best_tube_idx = int(tube_idx)
            if best_tube_idx is None:
                tube_idx = len(tubes)
                tubes.append({"tube_id": f"tube_{tube_idx:04d}", "components": []})
            else:
                tube_idx = int(best_tube_idx)
            comp_record = {k: v for k, v in comp.items() if not str(k).startswith("_")}
            tubes[tube_idx]["components"].append(comp_record)
            tubes[tube_idx].setdefault("_masks", []).append(comp["_mask"])
            next_active[tube_idx] = comp
            used_tubes.add(tube_idx)
        active = next_active

    out: list[dict[str, Any]] = []
    for tube in tubes:
        comps = tube["components"]
        masks = tube["_masks"]
        areas = [int(c["component_area"]) for c in comps]
        frame_indices = [int(c["frame_index"]) for c in comps]
        max_idx = int(np.argmax(np.asarray(areas, dtype=np.int64))) if areas else 0
        out.append(
            {
                "tube_id": tube["tube_id"],
                "span_frames": int(len(comps)),
                "first_frame_index": int(min(frame_indices)) if frame_indices else None,
                "last_frame_index": int(max(frame_indices)) if frame_indices else None,
                "max_component_area": int(max(areas)) if areas else 0,
                "mean_component_area": float(np.mean(areas)) if areas else 0.0,
                "anchor_component_index": int(max_idx),
                "anchor_frame_index": int(comps[max_idx]["frame_index"]) if comps else None,
                "anchor_frame_id": int(comps[max_idx]["frame_id"]) if comps else None,
                "anchor_component_id": comps[max_idx]["component_id"] if comps else None,
                "components": comps,
                "_masks": masks,
            }
        )
    return out


def select_anchor_indices(tube: dict[str, Any], anchors_per_tube: int) -> list[int]:
    comps = tube["components"]
    if not comps:
        return []
    order = sorted(range(len(comps)), key=lambda idx: (-int(comps[idx]["component_area"]), int(comps[idx]["frame_index"])))
    selected: list[int] = []
    for idx in order:
        if len(selected) >= int(anchors_per_tube):
            break
        selected.append(int(idx))
    selected.sort(key=lambda idx: int(comps[idx]["frame_index"]))
    return selected


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def run(cli: argparse.Namespace) -> None:
    import torch

    output_root = resolve_path(cli.output_root)
    mask_dir = output_root / "birth_masks"
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_path(cli.config)
    args = make_baseline_args(config_path, cli)
    args.scene_id = str(cli.scene_id or args.scene_id)
    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = resolve_path(args.rgb_root) / args.scene_id / "color"
    frame_paths = [rgb_root / f"{int(frame_id)}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:8])
    rgbs = [read_rgb(path) for path in frame_paths]
    h, w = rgbs[0].shape[:2]

    inherited_birth_records_path = resolve_path(cli.inherited_birth_records)
    inherited_payload = load_json(inherited_birth_records_path)
    inherited_rows = list(inherited_payload.get("rows", []))
    inherited_proxy_summary_path = resolve_path(cli.inherited_proxy_summary)
    proxy_labels = summary_label_map(inherited_proxy_summary_path)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    setup_t0 = time.time()
    models = setup_models(args)
    setup_sec = time.time() - setup_t0
    segmentor = models["segmentor"]

    component_t0 = time.time()
    components_by_frame: list[list[dict[str, Any]]] = []
    proxy_frame_records: list[dict[str, Any]] = []
    proxy_unions: list[np.ndarray] = []
    for frame_idx, frame_id in enumerate(frame_ids):
        label_path = proxy_labels.get(int(frame_id))
        if label_path is None:
            raise FileNotFoundError({"missing_proxy_label_for_frame": int(frame_id), "summary": str(inherited_proxy_summary_path)})
        label = load_label(label_path, h, w)
        proxy_union = label > 0
        proxy_unions.append(proxy_union.astype(bool))
        gap = ~proxy_union
        stats = component_area_stats(gap, int(cli.min_component_area))
        comps = component_records(
            gap,
            frame_idx=int(frame_idx),
            frame_id=int(frame_id),
            min_area=int(cli.min_component_area),
            max_components=int(cli.max_components_per_frame),
        )
        components_by_frame.append(comps)
        proxy_frame_records.append(
            {
                "frame_index": int(frame_idx),
                "frame_id": int(frame_id),
                "proxy_label_path": str(label_path),
                "proxy_foreground_area": int(np.count_nonzero(proxy_union)),
                "proxy_gap_area": int(np.count_nonzero(gap)),
                "candidate_component_count": int(stats["component_count"]),
                "candidate_kept_component_count": int(len(comps)),
                "candidate_max_component_area": int(stats["max_component_area"]),
                "candidate_area_ge_min_total": int(stats["area_ge_min_total"]),
            }
        )
    component_sec = time.time() - component_t0

    tube_t0 = time.time()
    tubes = link_tubes(
        components_by_frame,
        min_link_iou=float(cli.tube_min_link_iou),
        max_centroid_distance=float(cli.tube_max_centroid_distance),
    )
    for tube in tubes:
        tube["persistent"] = bool(int(tube["span_frames"]) >= int(cli.min_persistence_frames))
        tube["large_gap"] = bool(int(tube["max_component_area"]) >= int(cli.large_gap_area_threshold))
        if str(cli.variant) == "S1_spec_large_gaps_only":
            tube["selected"] = bool(tube["persistent"] and tube["large_gap"])
        elif str(cli.variant) == "S2_spec_all_persistent_tubes":
            tube["selected"] = bool(tube["persistent"])
        else:
            raise ValueError(f"unsupported variant={cli.variant}")
        tube["selection_reason"] = (
            "selected"
            if tube["selected"]
            else "not_large_gap" if str(cli.variant) == "S1_spec_large_gaps_only" and not tube["large_gap"]
            else "not_persistent"
        )
    selected_tubes = [tube for tube in tubes if bool(tube["selected"])]
    if int(cli.max_selected_tubes) > 0:
        selected_tubes = sorted(selected_tubes, key=lambda row: (-int(row["max_component_area"]), str(row["tube_id"])))[: int(cli.max_selected_tubes)]
        selected_ids = {str(row["tube_id"]) for row in selected_tubes}
        for tube in tubes:
            if bool(tube["selected"]) and str(tube["tube_id"]) not in selected_ids:
                tube["selected"] = False
                tube["selection_reason"] = "limited_by_max_selected_tubes"
    tube_sec = time.time() - tube_t0

    inherited_out_rows: list[dict[str, Any]] = []
    max_existing_obj_id = 0
    for row in inherited_rows:
        copied = dict(row)
        copied.setdefault("phase5_role", "inherited")
        copied.setdefault("phase7_role", "inherited_proxy_seed")
        inherited_out_rows.append(copied)
        max_existing_obj_id = max(max_existing_obj_id, int(copied.get("obj_id", 0)))
    next_obj_id = int(cli.new_obj_id_start) if int(cli.new_obj_id_start) > 0 else max_existing_obj_id + 1001

    prompt_records: list[dict[str, Any]] = []
    birth_rows: list[dict[str, Any]] = []
    total_decode_sec = 0.0
    accepted_by_anchor: dict[int, list[np.ndarray]] = {}
    for tube in selected_tubes:
        masks = tube["_masks"]
        for anchor_component_index in select_anchor_indices(tube, int(cli.anchors_per_tube)):
            comp = tube["components"][anchor_component_index]
            frame_idx = int(comp["frame_index"])
            frame_id = int(comp["frame_id"])
            candidate = masks[anchor_component_index].astype(bool)
            point_seed = stable_seed(int(args.seed), args.scene_id, frame_id, str(tube["tube_id"]), str(cli.variant), "v106-phase7-specgap")
            points_yx, point_meta = sample_component_adaptive_points_yx(
                candidate,
                max_points=int(cli.max_points_per_anchor),
                min_component_area=int(cli.min_component_area),
                base_points_per_component=int(cli.base_points_per_component),
                area_per_extra_point=int(cli.area_per_extra_point),
                max_points_per_component=int(cli.max_points_per_component),
                seed=int(point_seed),
            )
            decode_t0 = time.time()
            if int(points_yx.shape[0]) > 0:
                raw_masks, birth_stats = run_sam2_point_segment_choice(
                    segmentor,
                    rgbs[frame_idx],
                    points_yx=points_yx,
                    region_mask=candidate,
                    points_per_batch=int(args.points_per_batch),
                    choice_policy="smallest_valid_mask_per_point",
                    iou_threshold=float(cli.pred_iou_thresh),
                    stability_threshold=float(cli.stability_score_thresh),
                    stability_score_offset=float(args.stability_score_offset),
                    model_mask_thresh=float(args.model_mask_thresh),
                    box_nms_thresh=float(args.box_nms_thresh),
                    empty_ratio=float(args.empty_ratio),
                    apply_box_nms=bool(cli.apply_box_nms),
                    nms_score_type=str(cli.nms_score_type),
                )
            else:
                raw_masks = np.zeros((0, h, w), dtype=bool)
                birth_stats = {
                    "choice_policy": "smallest_valid_mask_per_point",
                    "raw_multimask_option_count": 0,
                    "prompt_with_good_mask_count": 0,
                    "pre_nms_mask_count": 0,
                    "post_disjoint_mask_count": 0,
                    "apply_box_nms": bool(cli.apply_box_nms),
                    "nms_score_type": str(cli.nms_score_type),
                }
            decode_sec = time.time() - decode_t0
            total_decode_sec += float(decode_sec)

            frame_existing_union = proxy_unions[frame_idx].copy()
            for prior in accepted_by_anchor.get(frame_idx, []):
                frame_existing_union |= prior.astype(bool)
            filtered_masks, filter_records = filter_birth_masks(
                raw_masks,
                candidate=candidate,
                current_union=frame_existing_union,
                core=proxy_unions[frame_idx],
                min_birth_mask_area=int(cli.min_birth_mask_area),
                min_candidate_touch_area=int(cli.min_candidate_touch_area),
                min_candidate_touch_ratio=float(cli.min_candidate_touch_ratio),
                max_existing_overlap_ratio=float(cli.max_existing_overlap_ratio),
                max_core_overlap_ratio=float(cli.max_core_overlap_ratio),
            )
            if filtered_masks.size and int(cli.max_births_per_anchor) > 0:
                areas = np.count_nonzero(filtered_masks.reshape(filtered_masks.shape[0], -1), axis=1)
                keep = np.argsort(areas)[::-1][: int(cli.max_births_per_anchor)]
                keep.sort()
                filtered_masks = filtered_masks[keep]
            accepted_masks: list[np.ndarray] = []
            for local_idx, raw_mask in enumerate(filtered_masks.astype(bool)):
                write_union = proxy_unions[frame_idx].copy()
                for prior in accepted_by_anchor.get(frame_idx, []):
                    write_union |= prior.astype(bool)
                final_mask = raw_mask & ~write_union
                if int(np.count_nonzero(final_mask)) < int(cli.min_birth_mask_area):
                    continue
                obj_id = int(next_obj_id)
                next_obj_id += 1
                mask_path = mask_dir / f"frame_{frame_id:06d}_obj_{obj_id:06d}_{cli.variant}.png"
                area = write_mask(final_mask, mask_path)
                accepted_by_anchor.setdefault(frame_idx, []).append(final_mask.astype(bool))
                accepted_masks.append(final_mask.astype(bool))
                birth_rows.append(
                    {
                        "scene_id": str(args.scene_id),
                        "chunk_frame_index": int(frame_idx),
                        "frame_id": int(frame_id),
                        "obj_id": int(obj_id),
                        "phase5_role": "birth_new",
                        "phase7_role": "speculative_birth",
                        "source": "v106_phase7_inherited_proxy_specgap_birth",
                        "variant": str(cli.variant),
                        "tube_id": str(tube["tube_id"]),
                        "anchor_component_index": int(anchor_component_index),
                        "local_birth_index": int(local_idx),
                        "mask_path": str(mask_path),
                        "mask_area": int(area),
                        "raw_mask_area": int(np.count_nonzero(raw_mask)),
                        "candidate_touch_area": int(np.count_nonzero(raw_mask & candidate)),
                        "proxy_overlap_area": int(np.count_nonzero(raw_mask & proxy_unions[frame_idx])),
                        "full_speculative_prompt_decode": True,
                    }
                )
            prompt_record = {
                "tube_id": str(tube["tube_id"]),
                "variant": str(cli.variant),
                "anchor_component_index": int(anchor_component_index),
                "anchor_frame_index": int(frame_idx),
                "anchor_frame_id": int(frame_id),
                "candidate_area": int(np.count_nonzero(candidate)),
                "point_count": int(points_yx.shape[0]),
                "point_sampling_meta": point_meta,
                "decode_runtime_sec": float(decode_sec),
                "raw_birth_mask_count": int(raw_masks.shape[0]),
                "filtered_birth_mask_count": int(filtered_masks.shape[0]),
                "accepted_birth_mask_count": int(len(accepted_masks)),
                "birth_stats": birth_stats,
                "birth_filter_records": filter_records[:64],
            }
            prompt_records.append(prompt_record)
            print(json.dumps(prompt_record, ensure_ascii=True), flush=True)

    output_rows = inherited_out_rows + birth_rows
    birth_payload = {
        "schema_version": "stream4d_v106_phase7_inherited_proxy_specgap_birth_records_v1",
        "scene_id": str(args.scene_id),
        "variant": str(cli.variant),
        "frame_ids": [int(v) for v in frame_ids],
        "rows": output_rows,
        "audit": {
            "inherited_birth_records": str(inherited_birth_records_path),
            "inherited_birth_records_sha256": sha256_file(inherited_birth_records_path),
            "inherited_proxy_summary": str(inherited_proxy_summary_path),
            "inherited_proxy_summary_sha256": sha256_file(inherited_proxy_summary_path),
            "candidate_source": "proxy_label_uncovered_only",
            "reference_used_for_candidate_or_prompt": False,
            "alltracker_used": False,
            "full_speculative_prompt_decode": True,
        },
    }
    birth_records_path = output_root / "birth_records.json"
    write_json(birth_records_path, birth_payload)
    write_json(output_root / "proxy_frame_records.json", proxy_frame_records)
    write_json(output_root / "specgap_component_records.json", [[clean_record(c) for c in comps] for comps in components_by_frame])
    write_json(output_root / "specgap_tube_records.json", [clean_record(t) for t in tubes])
    write_json(output_root / "specgap_prompt_records.json", prompt_records)

    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    selected_count = sum(1 for tube in tubes if bool(tube.get("selected")))
    summary = {
        "schema_version": "stream4d_v106_phase7_inherited_proxy_specgap_birth_summary_v1",
        "scene_id": str(args.scene_id),
        "variant": str(cli.variant),
        "config_path": str(config_path),
        "birth_records_path": str(birth_records_path),
        "birth_records_sha256": sha256_file(birth_records_path),
        "frame_start": int(args.frame_start),
        "frame_stride": int(args.frame_stride),
        "frame_count": int(len(frame_ids)),
        "frame_ids": [int(v) for v in frame_ids],
        "inherited_seed_row_count": int(len(inherited_out_rows)),
        "tube_count": int(len(tubes)),
        "selected_tube_count": int(selected_count),
        "prompt_record_count": int(len(prompt_records)),
        "new_birth_record_count": int(len(birth_rows)),
        "total_output_row_count": int(len(output_rows)),
        "setup_sec": float(setup_sec),
        "component_runtime_sec": float(component_sec),
        "tube_link_runtime_sec": float(tube_sec),
        "gap_prompt_decode_latency_sec": float(total_decode_sec),
        "total_runtime_sec": float(setup_sec + component_sec + tube_sec + total_decode_sec),
        "peak_cuda_memory_mb": float(peak_mb),
        "model_provider": str(models.get("model_provider", "")),
        "segmentor_checkpoint": str(models.get("segmentor_checkpoint", "")),
        "segmentor_checkpoint_sha256": sha256_file(Path(models["segmentor_checkpoint"])) if models.get("segmentor_checkpoint") else "",
        "segmentor_cfg": str(models.get("segmentor_cfg", "")),
        "parameters": {
            "min_component_area": int(cli.min_component_area),
            "max_components_per_frame": int(cli.max_components_per_frame),
            "tube_min_link_iou": float(cli.tube_min_link_iou),
            "tube_max_centroid_distance": float(cli.tube_max_centroid_distance),
            "min_persistence_frames": int(cli.min_persistence_frames),
            "large_gap_area_threshold": int(cli.large_gap_area_threshold),
            "anchors_per_tube": int(cli.anchors_per_tube),
            "max_selected_tubes": int(cli.max_selected_tubes),
            "max_points_per_anchor": int(cli.max_points_per_anchor),
            "max_births_per_anchor": int(cli.max_births_per_anchor),
            "pred_iou_thresh": float(cli.pred_iou_thresh),
            "stability_score_thresh": float(cli.stability_score_thresh),
            "min_birth_mask_area": int(cli.min_birth_mask_area),
            "min_candidate_touch_area": int(cli.min_candidate_touch_area),
            "min_candidate_touch_ratio": float(cli.min_candidate_touch_ratio),
            "max_existing_overlap_ratio": float(cli.max_existing_overlap_ratio),
            "max_core_overlap_ratio": float(cli.max_core_overlap_ratio),
        },
        "honesty_note": (
            "This is a real inherited-only proxy prompt decode birth-bank probe. "
            "It is not a promotion by itself; replay and metric gates must pass separately."
        ),
    }
    write_json(output_root / "specgap_birth_summary.json", summary)
    print(json.dumps({"summary": str(output_root / "specgap_birth_summary.json"), "birth_records": str(birth_records_path)}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--inherited-birth-records", required=True)
    parser.add_argument("--inherited-proxy-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rgb-root", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--frame-ids", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variant", choices=["S1_spec_large_gaps_only", "S2_spec_all_persistent_tubes"], default="S2_spec_all_persistent_tubes")
    parser.add_argument("--min-component-area", type=int, default=512)
    parser.add_argument("--max-components-per-frame", type=int, default=12)
    parser.add_argument("--tube-min-link-iou", type=float, default=0.05)
    parser.add_argument("--tube-max-centroid-distance", type=float, default=96.0)
    parser.add_argument("--min-persistence-frames", type=int, default=2)
    parser.add_argument("--large-gap-area-threshold", type=int, default=4096)
    parser.add_argument("--anchors-per-tube", type=int, default=1)
    parser.add_argument("--max-selected-tubes", type=int, default=0)
    parser.add_argument("--new-obj-id-start", type=int, default=0)
    parser.add_argument("--max-points-per-anchor", type=int, default=128)
    parser.add_argument("--max-births-per-anchor", type=int, default=40)
    parser.add_argument("--base-points-per-component", type=int, default=1)
    parser.add_argument("--area-per-extra-point", type=int, default=4000)
    parser.add_argument("--max-points-per-component", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.8)
    parser.add_argument("--apply-box-nms", action="store_true")
    parser.add_argument("--nms-score-type", choices=["pred_iou", "stability"], default="stability")
    parser.add_argument("--min-birth-mask-area", type=int, default=96)
    parser.add_argument("--min-candidate-touch-area", type=int, default=32)
    parser.add_argument("--min-candidate-touch-ratio", type=float, default=0.01)
    parser.add_argument("--max-existing-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-core-overlap-ratio", type=float, default=0.35)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
