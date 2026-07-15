#!/usr/bin/env python3
"""Probe v107 reactivation through a real SAM2 video inference_state mutation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
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

from tools.run_v107_phase7_lingbot_sam2_prompt_benchmark import (  # noqa: E402
    jsonable,
    load_label,
    load_points,
    load_reference_records,
    load_rows,
    map_lingbot_xy_to_original,
    mask_metrics,
    read_json,
    rel,
    resolve_path,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-probe-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--reference-run-root", required=True)
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed/scene0050_00")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--events", default="0,12,18")
    parser.add_argument("--visual-events", default="")
    parser.add_argument("--confirm-reprompt", action="store_true")
    parser.add_argument("--online-select-neg-conflict-threshold", type=float, default=0.25)
    parser.add_argument("--online-select-min-g2-positive-support", type=float, default=0.50)
    parser.add_argument("--rows-csv", default="")
    parser.add_argument("--points-json", default="")
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--min-init-mask-area", type=int, default=64)
    parser.add_argument("--fallback-sibling-count", type=int, default=2)
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bf16", choices=["float32", "bf16", "float16"])
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--visual-pad", type=int, default=90)
    parser.add_argument("--visual-scale", type=int, default=2)
    return parser.parse_args()


def dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    value = str(args.model_dtype).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def autocast_for(args: argparse.Namespace):
    dtype = dtype_from_args(args)
    if str(args.device).startswith("cuda") and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def build_video_predictor(args: argparse.Namespace):
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = Path(args.sam2_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    predictor = build_sam2_video_predictor(str(args.sam2_model_cfg), str(checkpoint), device=str(args.device))
    dtype = dtype_from_args(args)
    if dtype != torch.float32:
        predictor.to(dtype=dtype)
    predictor.eval()
    return predictor, checkpoint


def infer_masks(predictor: Any, state: dict[str, Any], frame_idx: int, args: argparse.Namespace) -> tuple[list[int], np.ndarray]:
    with torch.inference_mode(), autocast_for(args):
        out_frame_idx, out_obj_ids, out_mask_logits = predictor.infer_single_frame(state, int(frame_idx))
    if int(out_frame_idx) != int(frame_idx):
        raise RuntimeError(f"infer_single_frame returned frame {out_frame_idx}, expected {frame_idx}")
    masks = (out_mask_logits > 0.0).detach().cpu().numpy().squeeze(1).astype(bool)
    return [int(v) for v in out_obj_ids], masks


def masks_from_add_result(frame_idx: int, obj_ids: Any, mask_logits: Any) -> tuple[list[int], np.ndarray]:
    _ = int(frame_idx)
    masks = (mask_logits > 0.0).detach().cpu().numpy().squeeze(1).astype(bool)
    return [int(v) for v in obj_ids], masks


def extract_mask(ids: list[int], masks: np.ndarray, obj_id: int, shape: tuple[int, int]) -> tuple[bool, np.ndarray]:
    if int(obj_id) not in ids:
        return False, np.zeros(shape, dtype=bool)
    idx = ids.index(int(obj_id))
    return True, np.asarray(masks[idx]).astype(bool)


def reconsolidate_stream_state_outputs(predictor: Any, state: dict[str, Any]) -> None:
    output_dict = state.get("output_dict", {})
    for storage_key, is_cond in (("cond_frame_outputs", True), ("non_cond_frame_outputs", False)):
        for prior_frame_idx in sorted(output_dict.get(storage_key, {}).keys()):
            consolidated_out = predictor._consolidate_temp_output_across_obj(
                state,
                int(prior_frame_idx),
                is_cond=bool(is_cond),
                run_mem_encoder=True,
            )
            output_dict[storage_key][int(prior_frame_idx)] = consolidated_out
            predictor._add_output_per_object(state, int(prior_frame_idx), consolidated_out, storage_key)


def add_source_masks(
    predictor: Any,
    state: dict[str, Any],
    *,
    source_label: np.ndarray,
    init_obj_ids: list[int],
    args: argparse.Namespace,
) -> tuple[list[int], np.ndarray]:
    for obj_id in init_obj_ids:
        mask = source_label == int(obj_id)
        with torch.inference_mode(), autocast_for(args):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=int(obj_id),
                mask=torch.from_numpy(mask.astype(np.float32)),
            )
    return infer_masks(predictor, state, 0, args)


def add_points_after_tracking_started(
    predictor: Any,
    state: dict[str, Any],
    *,
    frame_idx: int,
    obj_id: int,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    old_tracking_started = bool(state.get("tracking_has_started", False))
    old_obj_ids = [int(v) for v in state.get("obj_ids", [])]
    state.get("frames_already_tracked", {}).pop(int(frame_idx), None)
    state["tracking_has_started"] = False
    used_workaround = True
    try:
        with torch.inference_mode(), autocast_for(args):
            out_frame_idx, out_obj_ids, out_masks = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=point_coords.astype(np.float32),
                labels=point_labels.astype(np.int32),
                clear_old_points=True,
                normalize_coords=True,
            )
    finally:
        state["tracking_has_started"] = old_tracking_started
    new_obj_ids = [int(v) for v in state.get("obj_ids", [])]
    if old_tracking_started and len(new_obj_ids) > len(old_obj_ids):
        with torch.inference_mode(), autocast_for(args):
            reconsolidate_stream_state_outputs(predictor, state)
    ids, masks = masks_from_add_result(out_frame_idx, out_obj_ids, out_masks)
    return ids, masks, {
        "used_tracking_has_started_workaround": used_workaround,
        "state_obj_ids_before_readd": old_obj_ids,
        "state_obj_ids_after_readd": new_obj_ids,
    }


def load_events_from_pilot(pilot_root: Path, event_indices: set[int]) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    path = pilot_root / "reactivation_confirmation_records.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            event_index = int(row["event_index"])
            if event_index not in event_indices:
                continue
            if event_index in records and row.get("prompt_variant") != "G2_pos_neg":
                continue
            records[event_index] = {
                "event_index": event_index,
                "source_frame_id": int(row["source_frame_id"]),
                "attempt_frame_id": int(row["attempt_frame_id"]),
                "confirm_frame_id": int(row["confirm_frame_id"]),
                "global_id": int(row["global_id"]),
                "source_lag": int(row["source_lag"]),
            }
    missing = sorted(event_indices - set(records.keys()))
    if missing:
        raise RuntimeError(f"missing event indices in {path}: {missing}")
    return [records[idx] for idx in sorted(records.keys())]


def infer_lingbot_hw(prompt_root: Path, probe_summary: dict[str, Any]) -> tuple[int, int]:
    raw = probe_summary.get("raw_lingbot_geometry", {})
    npz_path_text = raw.get("npz_path", "")
    if not npz_path_text:
        shape = probe_summary.get("lingbot_image_shape") or probe_summary.get("image_shape")
        if shape:
            return int(shape[0]), int(shape[1])
        raise RuntimeError("cannot infer LingBot image shape: no npz_path or shape in summary")
    npz_path = resolve_path(str(npz_path_text), prompt_root)
    with np.load(npz_path) as data:
        if "depth" in data:
            return int(data["depth"].shape[-2]), int(data["depth"].shape[-1])
        if "images" in data:
            return int(data["images"].shape[-2]), int(data["images"].shape[-1])
    raise RuntimeError(f"cannot infer LingBot image shape from {npz_path}")


def selected_probe_file(prompt_root: Path, summary: dict[str, Any], arg_value: str, default_name: str, selected_key: str) -> Path:
    if arg_value:
        return resolve_path(arg_value, prompt_root)
    selected = summary.get(selected_key, "")
    if selected:
        return resolve_path(str(selected), prompt_root)
    return prompt_root / default_name


def frame_ids_for_event(source_frame: int, confirm_frame: int, step: int) -> list[int]:
    step_i = max(1, int(step))
    if int(confirm_frame) < int(source_frame):
        raise ValueError("confirm_frame must be >= source_frame")
    ids = list(range(int(source_frame), int(confirm_frame) + 1, step_i))
    if ids[-1] != int(confirm_frame):
        ids.append(int(confirm_frame))
    return ids


def write_video_frames(scene_root: Path, event_root: Path, frame_ids: list[int]) -> dict[int, int]:
    video_dir = event_root / "video_frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, int] = {}
    for video_idx, frame_id in enumerate(frame_ids):
        src = scene_root / "color" / f"{frame_id}.jpg"
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(src)
        out = video_dir / f"{video_idx:05d}.jpg"
        cv2.imwrite(str(out), bgr)
        mapping[int(frame_id)] = int(video_idx)
    return mapping


def choose_init_obj_ids(source_label: np.ndarray, target_obj_id: int, neg_obj_ids: set[int], args: argparse.Namespace) -> tuple[list[int], list[int], bool]:
    min_area = int(args.min_init_mask_area)
    init_ids: list[int] = [int(target_obj_id)]
    missing_neg: list[int] = []
    for obj_id in sorted(int(v) for v in neg_obj_ids):
        area = int(np.count_nonzero(source_label == int(obj_id)))
        if area >= min_area:
            init_ids.append(int(obj_id))
        else:
            missing_neg.append(int(obj_id))
    used_fallback = False
    if len(init_ids) == 1:
        candidates = []
        for obj_id in np.unique(source_label):
            obj_id_i = int(obj_id)
            if obj_id_i in {0, int(target_obj_id)}:
                continue
            area = int(np.count_nonzero(source_label == obj_id_i))
            if area >= min_area:
                candidates.append((area, obj_id_i))
        candidates.sort(reverse=True)
        for _area, obj_id_i in candidates[: max(1, int(args.fallback_sibling_count))]:
            init_ids.append(int(obj_id_i))
        used_fallback = True
    return init_ids, missing_neg, used_fallback


def points_for_event(
    points_by_case: dict[tuple[int, int, int], list[dict[str, Any]]],
    *,
    source_frame_index: int | None,
    source_frame_id: int,
    target_frame_id: int,
    target_obj_id: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for (_target_idx, obj_id, _source_idx), rows in points_by_case.items():
        if int(obj_id) != int(target_obj_id):
            continue
        for row in rows:
            if int(row["source_frame_id"]) == int(source_frame_id) and int(row["target_frame_id"]) == int(target_frame_id):
                candidates.append(row)
        if candidates:
            break
    if not candidates and source_frame_index is not None:
        key = (int(target_frame_id), int(target_obj_id), int(source_frame_index))
        candidates.extend(points_by_case.get(key, []))
    return candidates


def group_rows_by_case(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    out: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["source_frame_id"]), int(row["target_frame_id"]), int(row["target_obj_id"]))
        out[key] = row
    return out


def point_arrays(
    point_records: list[dict[str, Any]],
    *,
    lingbot_hw: tuple[int, int],
    orig_hw: tuple[int, int],
    include_negative: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    neg_ids: list[int] = []
    for row in point_records:
        role = str(row.get("role", ""))
        if role == "negative" and not include_negative:
            continue
        if role not in {"positive", "negative"}:
            continue
        x, y = map_lingbot_xy_to_original(float(row["target_x"]), float(row["target_y"]), lingbot_hw=lingbot_hw, orig_hw=orig_hw)
        coords.append((x, y))
        labels.append(1 if role == "positive" else 0)
        if role == "negative":
            neg_ids.append(int(row["source_obj_id"]))
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32), neg_ids


def prompt_point_rates(mask: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(mask).astype(bool)
    h, w = pred.shape[:2]
    pos_total = 0
    pos_inside = 0
    neg_total = 0
    neg_inside = 0
    for (x, y), label in zip(coords, labels, strict=False):
        xi = min(max(int(round(float(x))), 0), max(w - 1, 0))
        yi = min(max(int(round(float(y))), 0), max(h - 1, 0))
        inside = bool(pred[yi, xi])
        if int(label) == 1:
            pos_total += 1
            pos_inside += int(inside)
        elif int(label) == 0:
            neg_total += 1
            neg_inside += int(inside)
    return {
        "positive_point_count_for_online_rate": int(pos_total),
        "positive_point_support_rate": float(pos_inside / max(pos_total, 1)) if pos_total else 0.0,
        "candidate_negative_point_count_for_online_rate": int(neg_total),
        "candidate_negative_point_conflict_rate": float(neg_inside / max(neg_total, 1)) if neg_total else 0.0,
    }


def draw_zoom_overlay(
    *,
    rgb: np.ndarray,
    ref_mask: np.ndarray,
    pred_mask: np.ndarray,
    points: np.ndarray | None,
    labels: np.ndarray | None,
    title: str,
    output_path: Path,
    pad: int,
    scale: int,
    color: tuple[int, int, int],
) -> Path:
    canvas = rgb.copy()
    pred = np.asarray(pred_mask).astype(bool)
    overlay_color = np.asarray(color, dtype=np.float32)
    canvas[pred] = (0.45 * canvas[pred].astype(np.float32) + 0.55 * overlay_color).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(ref_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 3, lineType=cv2.LINE_AA)
    if points is not None and labels is not None:
        for (x, y), label in zip(points, labels, strict=False):
            point_color = (25, 240, 60) if int(label) == 1 else (240, 50, 45)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 8, point_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    ys, xs = np.where(ref_mask)
    if len(xs) == 0 or len(ys) == 0:
        x0, y0, x1, y1 = 0, 0, canvas.shape[1], canvas.shape[0]
    else:
        x0 = max(0, int(xs.min()) - int(pad))
        y0 = max(0, int(ys.min()) - int(pad))
        x1 = min(canvas.shape[1], int(xs.max()) + int(pad) + 1)
        y1 = min(canvas.shape[0], int(ys.max()) + int(pad) + 1)
    crop = canvas[y0:y1, x0:x1]
    if int(scale) > 1:
        crop = cv2.resize(crop, (crop.shape[1] * int(scale), crop.shape[0] * int(scale)), interpolation=cv2.INTER_NEAREST)
    header_h = 44
    out = np.zeros((crop.shape[0] + header_h, crop.shape[1], 3), dtype=np.uint8)
    out[:] = 18
    out[header_h:] = crop
    cv2.putText(out, title[:180], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return output_path


def event_frame_label(reference_records: dict[int, dict[str, Any]], frame_id: int) -> np.ndarray:
    if int(frame_id) not in reference_records:
        raise KeyError(f"frame {frame_id} not found in reference records")
    return load_label(Path(reference_records[int(frame_id)]["label_path"]))


def rgb_frame(scene_root: Path, frame_id: int) -> np.ndarray:
    bgr = cv2.imread(str(scene_root / "color" / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(scene_root / "color" / f"{frame_id}.jpg")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_variant(
    *,
    predictor: Any,
    event: dict[str, Any],
    variant: str,
    point_records: list[dict[str, Any]],
    confirm_point_records: list[dict[str, Any]],
    init_obj_ids: list[int],
    source_label: np.ndarray,
    attempt_label: np.ndarray,
    confirm_label: np.ndarray,
    frame_to_video_idx: dict[int, int],
    event_root: Path,
    scene_root: Path,
    lingbot_hw: tuple[int, int],
    args: argparse.Namespace,
    write_visuals: bool,
) -> dict[str, Any]:
    event_index = int(event["event_index"])
    target_obj_id = int(event["global_id"])
    source_frame_id = int(event["source_frame_id"])
    attempt_frame_id = int(event["attempt_frame_id"])
    confirm_frame_id = int(event["confirm_frame_id"])
    source_idx = frame_to_video_idx[source_frame_id]
    attempt_idx = frame_to_video_idx[attempt_frame_id]
    confirm_idx = frame_to_video_idx[confirm_frame_id]
    if source_idx != 0:
        raise RuntimeError(f"expected source video index 0, got {source_idx}")

    with torch.inference_mode(), autocast_for(args):
        state = predictor.init_state(
            video_path=str(event_root / "video_frames"),
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            offload_state_to_cpu=bool(args.offload_state_to_cpu),
            async_loading_frames=False,
        )
    init_ids_before = [int(v) for v in state.get("obj_ids", [])]
    source_out_ids, source_out_masks = add_source_masks(
        predictor,
        state,
        source_label=source_label,
        init_obj_ids=init_obj_ids,
        args=args,
    )
    source_target_present, _source_target_mask = extract_mask(source_out_ids, source_out_masks, target_obj_id, source_label.shape)
    state_obj_ids_after_source_init = [int(v) for v in state.get("obj_ids", [])]

    remove_started = time.time()
    with torch.inference_mode(), autocast_for(args):
        predictor.remove_object(state, target_obj_id, strict=True, need_output=False)
    remove_runtime_sec = float(time.time() - remove_started)
    state_obj_ids_after_remove = [int(v) for v in state.get("obj_ids", [])]

    dormant_target_seen = False
    dormant_frames_inferred: list[int] = []
    for idx in range(1, int(attempt_idx)):
        ids, _masks = infer_masks(predictor, state, idx, args)
        dormant_frames_inferred.append(int(idx))
        if target_obj_id in ids:
            dormant_target_seen = True

    include_negative = variant == "G2_pos_neg"
    eval_neg_ids = sorted({int(row["source_obj_id"]) for row in point_records if row.get("role") == "negative"})
    confirm_eval_neg_ids = sorted(
        {int(row["source_obj_id"]) for row in confirm_point_records if row.get("role") == "negative"}
    )
    if not confirm_eval_neg_ids:
        confirm_eval_neg_ids = list(eval_neg_ids)
    coords, labels, prompt_neg_ids = point_arrays(
        point_records,
        lingbot_hw=lingbot_hw,
        orig_hw=attempt_label.shape[:2],
        include_negative=include_negative,
    )
    attempt_candidate_coords, attempt_candidate_labels, _attempt_candidate_neg_ids = point_arrays(
        point_records,
        lingbot_hw=lingbot_hw,
        orig_hw=attempt_label.shape[:2],
        include_negative=True,
    )
    if coords.size == 0:
        raise RuntimeError(f"no prompt points for event {event_index} variant {variant}")
    readd_started = time.time()
    attempt_ids, attempt_masks, readd_state = add_points_after_tracking_started(
        predictor,
        state,
        frame_idx=attempt_idx,
        obj_id=target_obj_id,
        point_coords=coords,
        point_labels=labels,
        args=args,
    )
    readd_runtime_sec = float(time.time() - readd_started)
    attempt_target_present, attempt_target_mask = extract_mask(attempt_ids, attempt_masks, target_obj_id, attempt_label.shape)

    confirm_started = time.time()
    final_ids: list[int] = []
    final_masks = np.zeros((0, *confirm_label.shape), dtype=bool)
    confirm_reprompt_applied = False
    confirm_coords = np.zeros((0, 2), dtype=np.float32)
    confirm_labels = np.zeros((0,), dtype=np.int32)
    confirm_candidate_coords = np.zeros((0, 2), dtype=np.float32)
    confirm_candidate_labels = np.zeros((0,), dtype=np.int32)
    confirm_prompt_neg_ids: list[int] = []
    for idx in range(int(attempt_idx) + 1, int(confirm_idx)):
        final_ids, final_masks = infer_masks(predictor, state, idx, args)
    if bool(args.confirm_reprompt) and confirm_point_records:
        confirm_candidate_coords, confirm_candidate_labels, _confirm_candidate_neg_ids = point_arrays(
            confirm_point_records,
            lingbot_hw=lingbot_hw,
            orig_hw=confirm_label.shape[:2],
            include_negative=True,
        )
        confirm_coords, confirm_labels, confirm_prompt_neg_ids = point_arrays(
            confirm_point_records,
            lingbot_hw=lingbot_hw,
            orig_hw=confirm_label.shape[:2],
            include_negative=include_negative,
        )
        if confirm_coords.size > 0:
            final_ids, final_masks, confirm_readd_state = add_points_after_tracking_started(
                predictor,
                state,
                frame_idx=confirm_idx,
                obj_id=target_obj_id,
                point_coords=confirm_coords,
                point_labels=confirm_labels,
                args=args,
            )
            confirm_reprompt_applied = True
        else:
            final_ids, final_masks = infer_masks(predictor, state, int(confirm_idx), args)
    else:
        final_ids, final_masks = infer_masks(predictor, state, int(confirm_idx), args)
    confirm_runtime_sec = float(time.time() - confirm_started)
    confirm_target_present, confirm_target_mask = extract_mask(final_ids, final_masks, target_obj_id, confirm_label.shape)

    neg_id_set = set(int(v) for v in eval_neg_ids)
    attempt_metrics = mask_metrics(attempt_target_mask, attempt_label == target_obj_id, attempt_label, neg_id_set)
    attempt_point_metrics = prompt_point_rates(attempt_target_mask, attempt_candidate_coords, attempt_candidate_labels)
    confirm_neg_id_set = set(int(v) for v in confirm_eval_neg_ids)
    confirm_metrics = mask_metrics(confirm_target_mask, confirm_label == target_obj_id, confirm_label, confirm_neg_id_set)
    if confirm_candidate_coords.size == 0:
        confirm_candidate_coords, confirm_candidate_labels, _confirm_candidate_neg_ids = point_arrays(
            confirm_point_records if confirm_point_records else point_records,
            lingbot_hw=lingbot_hw,
            orig_hw=confirm_label.shape[:2],
            include_negative=True,
        )
    confirm_point_metrics = prompt_point_rates(confirm_target_mask, confirm_candidate_coords, confirm_candidate_labels)

    visual_paths = []
    if write_visuals:
        attempt_rgb = rgb_frame(scene_root, attempt_frame_id)
        confirm_rgb = rgb_frame(scene_root, confirm_frame_id)
        variant_color = (255, 70, 170) if variant == "G2_pos_neg" else (40, 220, 255)
        visual_paths.append(
            draw_zoom_overlay(
                rgb=attempt_rgb,
                ref_mask=attempt_label == target_obj_id,
                pred_mask=attempt_target_mask,
                points=coords,
                labels=labels,
                title=f"event{event_index:03d} {variant} attempt f{attempt_frame_id} obj{target_obj_id}",
                output_path=event_root / "highres_live_state_visuals" / f"event{event_index:03d}_{variant}_attempt_zoom_x{args.visual_scale}.jpg",
                pad=int(args.visual_pad),
                scale=int(args.visual_scale),
                color=variant_color,
            )
        )
        visual_paths.append(
            draw_zoom_overlay(
                rgb=confirm_rgb,
                ref_mask=confirm_label == target_obj_id,
                pred_mask=confirm_target_mask,
                points=confirm_coords if confirm_reprompt_applied else None,
                labels=confirm_labels if confirm_reprompt_applied else None,
                title=f"event{event_index:03d} {variant} confirm f{confirm_frame_id} obj{target_obj_id}",
                output_path=event_root / "highres_live_state_visuals" / f"event{event_index:03d}_{variant}_confirm_zoom_x{args.visual_scale}.jpg",
                pad=int(args.visual_pad),
                scale=int(args.visual_scale),
                color=variant_color,
            )
        )

    return {
        "schema_version": "stream4d_v107_phase8_sam2_live_state_reactivation_probe_record_v1",
        "event_index": event_index,
        "prompt_variant": variant,
        "source_frame_id": source_frame_id,
        "attempt_frame_id": attempt_frame_id,
        "confirm_frame_id": confirm_frame_id,
        "source_lag": int(event["source_lag"]),
        "global_id": target_obj_id,
        "video_frame_ids": [int(k) for k, _v in sorted(frame_to_video_idx.items(), key=lambda item: item[1])],
        "source_video_idx": int(source_idx),
        "attempt_video_idx": int(attempt_idx),
        "confirm_video_idx": int(confirm_idx),
        "init_obj_ids_requested": [int(v) for v in init_obj_ids],
        "state_obj_ids_before_source_init": init_ids_before,
        "state_obj_ids_after_source_init": state_obj_ids_after_source_init,
        "source_target_present_after_init": bool(source_target_present),
        "sam2_remove_object_called": True,
        "remove_object_runtime_sec": remove_runtime_sec,
        "state_obj_ids_after_remove": state_obj_ids_after_remove,
        "target_id_absent_after_remove": target_obj_id not in state_obj_ids_after_remove,
        "dormant_frames_inferred": dormant_frames_inferred,
        "target_id_seen_during_dormant_frames": bool(dormant_target_seen),
        "positive_prompt_count": int(np.count_nonzero(labels == 1)),
        "negative_prompt_count": int(np.count_nonzero(labels == 0)),
        "negative_candidate_source_obj_ids": sorted(neg_id_set),
        "negative_prompt_source_obj_ids": sorted(int(v) for v in prompt_neg_ids),
        "readd_runtime_sec": readd_runtime_sec,
        **readd_state,
        "target_id_present_after_readd_state": target_obj_id in [int(v) for v in state.get("obj_ids", [])],
        "target_id_present_in_attempt_output": bool(attempt_target_present),
        "attempt_iou_to_reference": attempt_metrics["iou_to_reference"],
        "attempt_precision_to_reference": attempt_metrics["precision_to_reference"],
        "attempt_recall_to_reference": attempt_metrics["recall_to_reference"],
        "attempt_mask_area_px": attempt_metrics["mask_area_px"],
        "attempt_negative_sibling_overlap_rate": attempt_metrics["negative_sibling_overlap_rate"],
        "attempt_positive_point_support_rate": attempt_point_metrics["positive_point_support_rate"],
        "attempt_candidate_negative_point_conflict_rate": attempt_point_metrics["candidate_negative_point_conflict_rate"],
        "confirm_runtime_sec": confirm_runtime_sec,
        "confirm_reprompt_requested": bool(args.confirm_reprompt),
        "confirm_reprompt_applied": bool(confirm_reprompt_applied),
        "confirm_positive_prompt_count": int(np.count_nonzero(confirm_labels == 1)) if confirm_reprompt_applied else 0,
        "confirm_negative_prompt_count": int(np.count_nonzero(confirm_labels == 0)) if confirm_reprompt_applied else 0,
        "confirm_negative_candidate_source_obj_ids": sorted(confirm_neg_id_set),
        "confirm_negative_prompt_source_obj_ids": sorted(int(v) for v in confirm_prompt_neg_ids),
        "state_obj_ids_after_confirm": [int(v) for v in state.get("obj_ids", [])],
        "target_id_present_in_confirm_output": bool(confirm_target_present),
        "confirm_iou_to_reference": confirm_metrics["iou_to_reference"],
        "confirm_precision_to_reference": confirm_metrics["precision_to_reference"],
        "confirm_recall_to_reference": confirm_metrics["recall_to_reference"],
        "confirm_mask_area_px": confirm_metrics["mask_area_px"],
        "confirm_negative_sibling_overlap_rate": confirm_metrics["negative_sibling_overlap_rate"],
        "confirm_positive_point_support_rate": confirm_point_metrics["positive_point_support_rate"],
        "confirm_candidate_negative_point_conflict_rate": confirm_point_metrics["candidate_negative_point_conflict_rate"],
        "same_client_obj_id_retained_after_confirm": target_obj_id in [int(v) for v in state.get("obj_ids", [])] and bool(confirm_target_present),
        "live_sam2_video_state_mutated": True,
        "online_gate_uses_reference_iou": False,
        "reference_labels_used_for_offline_evaluation_only": True,
        "visual_written": bool(write_visuals),
        "visual_paths": [rel(path) for path in visual_paths],
        "visual_sha256": {path.name: sha256_file(path) for path in visual_paths},
    }


def main() -> int:
    started = time.time()
    args = parse_args()
    prompt_root = Path(args.prompt_probe_root)
    if not prompt_root.is_absolute():
        prompt_root = ROOT / prompt_root
    pilot_root = Path(args.pilot_root)
    if not pilot_root.is_absolute():
        pilot_root = ROOT / pilot_root
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
    rows_path = selected_probe_file(
        prompt_root,
        probe_summary,
        str(args.rows_csv),
        "prompt_capsule_visibility_rows.csv",
        "selected_rows_csv",
    )
    points_path = selected_probe_file(
        prompt_root,
        probe_summary,
        str(args.points_json),
        "prompt_capsule_visible_point_records.json",
        "selected_visible_point_records",
    )
    rows = load_rows(rows_path)
    rows_by_case = group_rows_by_case(rows)
    points_by_case = load_points(points_path)
    lingbot_hw = infer_lingbot_hw(prompt_root, probe_summary)
    reference_records = load_reference_records(reference_root)
    event_indices = {int(v) for v in str(args.events).split(",") if str(v).strip()}
    visual_event_indices = (
        {int(v) for v in str(args.visual_events).split(",") if str(v).strip()}
        if str(args.visual_events).strip()
        else set(event_indices)
    )
    events = load_events_from_pilot(pilot_root, event_indices)
    predictor, checkpoint = build_video_predictor(args)

    records: list[dict[str, Any]] = []
    event_setup_records: list[dict[str, Any]] = []
    for event in events:
        event_index = int(event["event_index"])
        event_root = output_root / f"event{event_index:03d}"
        frame_ids = frame_ids_for_event(int(event["source_frame_id"]), int(event["confirm_frame_id"]), int(args.frame_step))
        frame_to_video_idx = write_video_frames(scene_root, event_root, frame_ids)
        source_label = event_frame_label(reference_records, int(event["source_frame_id"]))
        attempt_label = event_frame_label(reference_records, int(event["attempt_frame_id"]))
        confirm_label = event_frame_label(reference_records, int(event["confirm_frame_id"]))
        target_obj_id = int(event["global_id"])
        point_records = [
            row
            for row in points_for_event(
                points_by_case,
                source_frame_index=None,
                source_frame_id=int(event["source_frame_id"]),
                target_frame_id=int(event["attempt_frame_id"]),
                target_obj_id=target_obj_id,
            )
            if str(row.get("pose_mode", probe_summary.get("selected_pose_mode", ""))) == str(probe_summary.get("selected_pose_mode", "direct_as_c2w"))
        ]
        if not point_records:
            raise RuntimeError(f"no visible point records for event {event_index}: {event}")
        confirm_point_records = [
            row
            for row in points_for_event(
                points_by_case,
                source_frame_index=None,
                source_frame_id=int(event["source_frame_id"]),
                target_frame_id=int(event["confirm_frame_id"]),
                target_obj_id=target_obj_id,
            )
            if str(row.get("pose_mode", probe_summary.get("selected_pose_mode", ""))) == str(probe_summary.get("selected_pose_mode", "direct_as_c2w"))
        ]
        neg_ids = {int(row["source_obj_id"]) for row in point_records if row.get("role") == "negative"}
        confirm_neg_ids = {int(row["source_obj_id"]) for row in confirm_point_records if row.get("role") == "negative"}
        init_obj_ids, missing_neg_ids, used_fallback = choose_init_obj_ids(source_label, target_obj_id, neg_ids, args)
        source_area = int(np.count_nonzero(source_label == target_obj_id))
        attempt_area = int(np.count_nonzero(attempt_label == target_obj_id))
        confirm_area = int(np.count_nonzero(confirm_label == target_obj_id))
        case_row = rows_by_case.get((int(event["source_frame_id"]), int(event["attempt_frame_id"]), target_obj_id), {})
        event_setup_records.append(
            {
                "event_index": event_index,
                **event,
                "frame_ids": frame_ids,
                "frame_to_video_idx": frame_to_video_idx,
                "target_source_area_px": source_area,
                "target_attempt_area_px": attempt_area,
                "target_confirm_area_px": confirm_area,
                "visible_point_count": len(point_records),
                "positive_point_count": int(sum(1 for row in point_records if row.get("role") == "positive")),
                "negative_point_count": int(sum(1 for row in point_records if row.get("role") == "negative")),
                "negative_source_obj_ids": sorted(neg_ids),
                "confirm_visible_point_count": len(confirm_point_records),
                "confirm_positive_point_count": int(sum(1 for row in confirm_point_records if row.get("role") == "positive")),
                "confirm_negative_point_count": int(sum(1 for row in confirm_point_records if row.get("role") == "negative")),
                "confirm_negative_source_obj_ids": sorted(confirm_neg_ids),
                "init_obj_ids": init_obj_ids,
                "missing_negative_source_obj_ids_at_source": missing_neg_ids,
                "used_fallback_init_sibling_ids": bool(used_fallback),
                "usable_positive_negative_prompt": case_row.get("usable_positive_negative_prompt", ""),
            }
        )
        for variant in ("G1_pos", "G2_pos_neg"):
            record = run_variant(
                predictor=predictor,
                event=event,
                variant=variant,
                point_records=point_records,
                confirm_point_records=confirm_point_records,
                init_obj_ids=init_obj_ids,
                source_label=source_label,
                attempt_label=attempt_label,
                confirm_label=confirm_label,
                frame_to_video_idx=frame_to_video_idx,
                event_root=event_root,
                scene_root=scene_root,
                lingbot_hw=lingbot_hw,
                args=args,
                write_visuals=event_index in visual_event_indices,
            )
            records.append(record)
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    records_jsonl = output_root / "live_state_reactivation_records.jsonl"
    with records_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    records_csv = output_root / "live_state_reactivation_records.csv"
    write_csv(records_csv, records)
    setup_json = output_root / "live_state_reactivation_event_setup.json"
    write_json(setup_json, {"schema_version": "stream4d_v107_live_state_reactivation_event_setup_v1", "events": event_setup_records})

    def mean_for(variant: str, key: str) -> float:
        vals = [float(row[key]) for row in records if row["prompt_variant"] == variant and key in row]
        return float(np.mean(vals)) if vals else 0.0

    def rate_for(variant: str, key: str) -> float:
        vals = [bool(row[key]) for row in records if row["prompt_variant"] == variant and key in row]
        return float(sum(vals) / max(len(vals), 1)) if vals else 0.0

    def threshold_rate(variant: str, key: str, threshold: float, op: str = "ge") -> float:
        vals = [float(row[key]) for row in records if row["prompt_variant"] == variant and key in row]
        if not vals:
            return 0.0
        if op == "gt":
            return float(sum(v > float(threshold) for v in vals) / len(vals))
        return float(sum(v >= float(threshold) for v in vals) / len(vals))

    def online_selected_records() -> list[dict[str, Any]]:
        by_event: dict[int, dict[str, dict[str, Any]]] = {}
        for row in records:
            by_event.setdefault(int(row["event_index"]), {})[str(row["prompt_variant"])] = row
        selected: list[dict[str, Any]] = []
        neg_threshold = float(args.online_select_neg_conflict_threshold)
        min_g2_pos = float(args.online_select_min_g2_positive_support)
        for event_index in sorted(by_event):
            g1 = by_event[event_index].get("G1_pos")
            g2 = by_event[event_index].get("G2_pos_neg")
            if g1 is None or g2 is None:
                continue
            g1_conflict = float(g1.get("confirm_candidate_negative_point_conflict_rate", 0.0))
            g2_conflict = float(g2.get("confirm_candidate_negative_point_conflict_rate", 0.0))
            g2_pos_support = float(g2.get("confirm_positive_point_support_rate", 0.0))
            choose_g2 = bool(
                g1_conflict > neg_threshold
                and g2_pos_support >= min_g2_pos
                and g2_conflict <= g1_conflict
            )
            chosen = dict(g2 if choose_g2 else g1)
            chosen["online_selected_variant"] = "G2_pos_neg" if choose_g2 else "G1_pos"
            chosen["online_select_g1_negative_conflict_rate"] = g1_conflict
            chosen["online_select_g2_negative_conflict_rate"] = g2_conflict
            chosen["online_select_g2_positive_support_rate"] = g2_pos_support
            selected.append(chosen)
        return selected

    selected_records = online_selected_records()

    def selected_mean(key: str) -> float:
        vals = [float(row[key]) for row in selected_records if key in row]
        return float(np.mean(vals)) if vals else 0.0

    def selected_threshold_rate(key: str, threshold: float, op: str = "ge") -> float:
        vals = [float(row[key]) for row in selected_records if key in row]
        if not vals:
            return 0.0
        if op == "gt":
            return float(sum(v > float(threshold) for v in vals) / len(vals))
        return float(sum(v >= float(threshold) for v in vals) / len(vals))

    g3_selected_g2_count = int(sum(row.get("online_selected_variant") == "G2_pos_neg" for row in selected_records))

    visual_paths = [Path(ROOT / path) if not Path(path).is_absolute() else Path(path) for row in records for path in row.get("visual_paths", [])]
    summary = {
        "schema_version": "stream4d_v107_phase8_sam2_live_state_reactivation_probe_summary_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "status": "LIVE_STATE_MUTATION_PROBE_NOT_FULL_PHASE8_SCHEDULER",
        "audit_note": "This probe mutates a real SAM2 video inference_state via remove_object and add_new_points_or_box with the same client global id, but it is not the full v107 lifecycle scheduler/transaction manager.",
        "prompt_probe_root": rel(prompt_root),
        "pilot_root": rel(pilot_root),
        "reference_run_root": rel(reference_root),
        "scene_root": rel(scene_root),
        "rows_csv": rel(rows_path),
        "rows_csv_sha256": sha256_file(rows_path),
        "points_json": rel(points_path),
        "points_json_sha256": sha256_file(points_path),
        "lingbot_hw": list(lingbot_hw),
        "projection_geometry_source": probe_summary.get("projection_geometry_source", "LingBot-Map"),
        "uses_scannet_pose_or_depth_for_projection": False,
        "selected_pose_mode": probe_summary.get("selected_pose_mode", ""),
        "event_count": len(events),
        "visual_event_indices": sorted(visual_event_indices),
        "record_count": len(records),
        "sam2_checkpoint": rel(checkpoint),
        "sam2_checkpoint_sha256": sha256_file(checkpoint),
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "device": str(args.device),
        "model_dtype": str(args.model_dtype),
        "live_sam2_video_state_mutated": True,
        "sam2_remove_object_called": True,
        "sam2_add_new_points_or_box_called": True,
        "same_client_obj_id_readd_attempted": True,
        "confirm_reprompt_requested": bool(args.confirm_reprompt),
        "online_gate_uses_reference_iou": False,
        "reference_labels_used_for_offline_evaluation_only": True,
        "G3_online_select_policy": {
            "description": "Choose G2 only when the G1 candidate mask covers candidate negative prompt points; selection uses point support/conflict only, not reference IoU.",
            "g1_confirm_negative_point_conflict_threshold": float(args.online_select_neg_conflict_threshold),
            "min_g2_confirm_positive_support": float(args.online_select_min_g2_positive_support),
            "requires_g2_negative_conflict_no_worse_than_g1": True,
        },
        "G3_online_select_event_count": len(selected_records),
        "G3_online_select_g2_selected_count": g3_selected_g2_count,
        "G3_online_select_g2_selected_rate": float(g3_selected_g2_count / max(len(selected_records), 1)),
        "G3_online_select_confirm_mean_iou": selected_mean("confirm_iou_to_reference"),
        "G3_online_select_confirm_iou_ge_0_5_rate": selected_threshold_rate("confirm_iou_to_reference", 0.5),
        "G3_online_select_confirm_iou_ge_0_7_rate": selected_threshold_rate("confirm_iou_to_reference", 0.7),
        "G3_online_select_confirm_negative_sibling_overlap_rate_mean": selected_mean("confirm_negative_sibling_overlap_rate"),
        "G3_online_select_confirm_negative_sibling_overlap_gt_0_1_rate": selected_threshold_rate(
            "confirm_negative_sibling_overlap_rate", 0.1, op="gt"
        ),
        "G3_online_select_records": [
            {
                "event_index": int(row["event_index"]),
                "source_lag": int(row["source_lag"]),
                "global_id": int(row["global_id"]),
                "selected_variant": str(row["online_selected_variant"]),
                "confirm_iou_to_reference": float(row["confirm_iou_to_reference"]),
                "confirm_negative_sibling_overlap_rate": float(row["confirm_negative_sibling_overlap_rate"]),
                "g1_negative_point_conflict_rate": float(row["online_select_g1_negative_conflict_rate"]),
                "g2_negative_point_conflict_rate": float(row["online_select_g2_negative_conflict_rate"]),
                "g2_positive_support_rate": float(row["online_select_g2_positive_support_rate"]),
            }
            for row in selected_records
        ],
        "G1_pos_same_client_obj_id_retained_after_confirm_rate": rate_for("G1_pos", "same_client_obj_id_retained_after_confirm"),
        "G2_pos_neg_same_client_obj_id_retained_after_confirm_rate": rate_for("G2_pos_neg", "same_client_obj_id_retained_after_confirm"),
        "G1_pos_attempt_mean_iou": mean_for("G1_pos", "attempt_iou_to_reference"),
        "G2_pos_neg_attempt_mean_iou": mean_for("G2_pos_neg", "attempt_iou_to_reference"),
        "G1_pos_confirm_mean_iou": mean_for("G1_pos", "confirm_iou_to_reference"),
        "G2_pos_neg_confirm_mean_iou": mean_for("G2_pos_neg", "confirm_iou_to_reference"),
        "G1_pos_confirm_iou_ge_0_5_rate": threshold_rate("G1_pos", "confirm_iou_to_reference", 0.5),
        "G2_pos_neg_confirm_iou_ge_0_5_rate": threshold_rate("G2_pos_neg", "confirm_iou_to_reference", 0.5),
        "G1_pos_confirm_iou_ge_0_7_rate": threshold_rate("G1_pos", "confirm_iou_to_reference", 0.7),
        "G2_pos_neg_confirm_iou_ge_0_7_rate": threshold_rate("G2_pos_neg", "confirm_iou_to_reference", 0.7),
        "G1_pos_confirm_negative_sibling_overlap_rate_mean": mean_for("G1_pos", "confirm_negative_sibling_overlap_rate"),
        "G2_pos_neg_confirm_negative_sibling_overlap_rate_mean": mean_for("G2_pos_neg", "confirm_negative_sibling_overlap_rate"),
        "G1_pos_confirm_negative_sibling_overlap_gt_0_1_rate": threshold_rate("G1_pos", "confirm_negative_sibling_overlap_rate", 0.1, op="gt"),
        "G2_pos_neg_confirm_negative_sibling_overlap_gt_0_1_rate": threshold_rate("G2_pos_neg", "confirm_negative_sibling_overlap_rate", 0.1, op="gt"),
        "records_jsonl": rel(records_jsonl),
        "records_jsonl_sha256": sha256_file(records_jsonl),
        "records_csv": rel(records_csv),
        "records_csv_sha256": sha256_file(records_csv),
        "event_setup_json": rel(setup_json),
        "event_setup_json_sha256": sha256_file(setup_json),
        "highres_visual_count": len(visual_paths),
        "highres_visuals": [rel(path) for path in visual_paths],
        "highres_visual_sha256": {rel(path): sha256_file(path) for path in visual_paths},
    }
    summary_path = output_root / "live_state_reactivation_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "record_count": len(records), "status": summary["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
