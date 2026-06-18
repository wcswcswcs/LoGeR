#!/usr/bin/env python3
"""Append static roadside thing tracks from YOLOE detections.

This tool is an auditable automatic post-process. It does not use manual merge
groups. It runs YOLOE on every processed frame for a small static-roadside
prompt set, filters low-confidence detections, greedily associates detections
across nearby frames, and appends surviving tracks to an existing sparse
masklet file.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import clone_sparse, load_sparse, track_stats  # noqa: E402
from run_video_masklet_front_end import (  # noqa: E402
    _mask_to_box_np,
    _pack_mask_np,
    collect_image_paths,
    prepare_processing_image_paths,
)
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import (  # noqa: E402
    DEFAULT_SEMANTIC_WEIGHTS,
    YOLOEDetector,
    canonicalize_label,
    label_to_group,
)


@dataclass
class Det:
    frame_idx: int
    label: str
    mask: np.ndarray
    box: np.ndarray
    confidence: float
    area_ratio: float


@dataclass
class Track:
    track_id: int
    label: str
    dets: List[Det] = field(default_factory=list)

    @property
    def last(self) -> Det:
        return self.dets[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append YOLOE static thing tracks to sparse masklets.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--yoloe_model", default="yoloe-26l-seg.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yoloe_batch_size", type=int, default=4)
    parser.add_argument("--yoloe_imgsz", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument(
        "--prompts",
        default="traffic sign,pole,traffic light,barrier,traffic cone",
        help="Comma-separated YOLOE prompt labels.",
    )
    parser.add_argument(
        "--append_labels",
        default="traffic sign,pole,traffic light",
        help="Detected labels allowed to become appended thing tracks.",
    )
    parser.add_argument(
        "--label_map",
        default="traffic light=traffic sign",
        help="Comma-separated source=target label map applied before association.",
    )
    parser.add_argument("--box_threshold", type=float, default=0.30)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--min_conf", type=float, default=0.28)
    parser.add_argument(
        "--min_conf_by_label",
        default="traffic sign=0.30,pole=0.30,traffic light=0.20",
        help="Comma-separated label=value overrides.",
    )
    parser.add_argument("--min_area_ratio", type=float, default=0.00002)
    parser.add_argument("--max_area_ratio", type=float, default=0.040)
    parser.add_argument("--nms_iou", type=float, default=0.65)
    parser.add_argument("--max_assoc_gap", type=int, default=3)
    parser.add_argument("--match_min_box_iou", type=float, default=0.02)
    parser.add_argument("--match_max_center_px", type=float, default=90.0)
    parser.add_argument("--match_max_center_norm", type=float, default=3.0)
    parser.add_argument("--center_scale_floor", type=float, default=18.0)
    parser.add_argument("--min_track_frames", type=int, default=3)
    parser.add_argument("--min_track_mean_conf", type=float, default=0.28)
    parser.add_argument("--max_tracks_per_label", type=int, default=120)
    parser.add_argument("--merge_static_label_tracks", type=int, default=1)
    parser.add_argument("--merge_common_min_frames", type=int, default=2)
    parser.add_argument("--merge_min_duplicate_frames", type=int, default=2)
    parser.add_argument("--merge_min_mask_iou", type=float, default=0.05)
    parser.add_argument("--merge_min_containment", type=float, default=0.55)
    parser.add_argument("--merge_min_box_iou", type=float, default=0.01)
    parser.add_argument("--merge_max_center_norm", type=float, default=0.35)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--render_style", choices=["debug", "clean"], default="debug")
    return parser.parse_args()


def _parse_csv_labels(raw: str) -> List[str]:
    labels: List[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        label = canonicalize_label(part.strip())
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return labels


def _parse_conf_overrides(raw: str) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for part in str(raw or "").split(","):
        if not part.strip() or "=" not in part:
            continue
        label_raw, value_raw = part.split("=", 1)
        label = canonicalize_label(label_raw.strip())
        try:
            overrides[label] = float(value_raw)
        except ValueError:
            continue
    return overrides


def _parse_label_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for part in str(raw or "").replace(";", ",").split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        src_raw, dst_raw = item.split("=", 1)
        src = canonicalize_label(src_raw.strip())
        dst = canonicalize_label(dst_raw.strip())
        if src and dst:
            mapping[src] = dst
    return mapping


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)[:4]
    bb = np.asarray(b, dtype=np.float32).reshape(-1)[:4]
    if aa.size < 4 or bb.size < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in aa]
    bx1, by1, bx2, by2 = [float(v) for v in bb]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _center(box: np.ndarray) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
    return np.asarray([(float(arr[0]) + float(arr[2])) * 0.5, (float(arr[1]) + float(arr[3])) * 0.5])


def _box_scale(box: np.ndarray, floor: float) -> float:
    arr = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
    return max(float(floor), 1.0, float(arr[2] - arr[0]), float(arr[3] - arr[1]))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.logical_and(aa, bb).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(aa, bb).sum())
    return float(inter) / float(max(union, 1))


def _load_processing_frames(
    input_video: str,
    processing_max_side: int,
    frames_limit: int,
    expected_h: int,
    expected_w: int,
    num_frames: int,
) -> Tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    if int(frames_limit) > 0:
        image_paths = image_paths[: int(frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < int(num_frames):
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[: int(num_frames)]
    if tuple(proc_shape) != (int(expected_h), int(expected_w)):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _dedupe_frame_dets(dets: List[Det], nms_iou: float) -> List[Det]:
    kept: List[Det] = []
    for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
        duplicate = False
        for prev in kept:
            if prev.label != det.label:
                continue
            if _mask_iou(prev.mask, det.mask) >= float(nms_iou) or _box_iou(prev.box, det.box) >= float(nms_iou):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def _detections_from_yoloe(
    detector: YOLOEDetector,
    image_paths: Sequence[str],
    prompts: Sequence[str],
    append_labels: set[str],
    conf_overrides: Dict[str, float],
    label_map: Dict[str, str],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[List[Det]], Dict[str, Any]]:
    by_frame: List[List[Det]] = [[] for _ in image_paths]
    raw_count = 0
    label_raw: Dict[str, int] = {}
    label_kept: Dict[str, int] = {}
    t0 = time.time()
    batch = max(1, int(args.yoloe_batch_size))
    for start in range(0, len(image_paths), batch):
        end = min(len(image_paths), start + batch)
        outputs = detector.detect_batch(
            image_paths[start:end],
            thing_prompts=list(prompts),
            stuff_prompts=[],
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
        )
        for offset, frame_outputs in enumerate(outputs):
            frame_idx = start + offset
            frame_dets: List[Det] = []
            for raw in frame_outputs:
                raw_count += 1
                raw_label = canonicalize_label(str(raw.get("label", "")))
                label_raw[raw_label] = label_raw.get(raw_label, 0) + 1
                label = label_map.get(raw_label, raw_label)
                if label not in append_labels:
                    continue
                conf = float(raw.get("confidence", 0.0))
                min_conf = float(conf_overrides.get(raw_label, conf_overrides.get(label, args.min_conf)))
                if conf < min_conf:
                    continue
                mask = np.asarray(raw.get("mask"), dtype=np.uint8).astype(bool)
                if mask.shape != (H, W):
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                area_ratio = float(mask.sum()) / float(max(H * W, 1))
                if area_ratio < float(args.min_area_ratio) or area_ratio > float(args.max_area_ratio):
                    continue
                # Use the mask-derived box for association and duplicate merging.
                # YOLOE raw boxes can be looser than the final segmentation mask,
                # while all downstream audits also reason from saved masks.
                box = _mask_to_box_np(mask)
                frame_dets.append(
                    Det(
                        frame_idx=int(frame_idx),
                        label=label,
                        mask=mask,
                        box=box.astype(np.float32),
                        confidence=conf,
                        area_ratio=area_ratio,
                    )
                )
                label_kept[label] = label_kept.get(label, 0) + 1
            by_frame[frame_idx] = _dedupe_frame_dets(frame_dets, float(args.nms_iou))
        print(f"[append-static-thing] detected frames {start}-{end - 1}/{len(image_paths) - 1}", flush=True)
    summary = {
        "raw_detection_count": int(raw_count),
        "kept_detection_count": int(sum(len(x) for x in by_frame)),
        "raw_label_counts": label_raw,
        "kept_label_counts": label_kept,
        "elapsed_seconds": float(time.time() - t0),
    }
    return by_frame, summary


def _match_score(track: Track, det: Det, args: argparse.Namespace) -> Tuple[float, Dict[str, float]]:
    last = track.last
    gap = int(det.frame_idx - last.frame_idx)
    if gap <= 0 or gap > int(args.max_assoc_gap):
        return -1.0, {}
    if track.label != det.label:
        return -1.0, {}
    box_iou = _box_iou(last.box, det.box)
    center_dist = float(np.linalg.norm(_center(last.box) - _center(det.box)))
    scale = max(_box_scale(last.box, float(args.center_scale_floor)), _box_scale(det.box, float(args.center_scale_floor)))
    center_norm = float(center_dist / max(scale, 1e-6))
    passes_box = box_iou >= float(args.match_min_box_iou)
    passes_center = center_dist <= float(args.match_max_center_px) and center_norm <= float(args.match_max_center_norm)
    if not passes_box and not passes_center:
        return -1.0, {
            "box_iou": box_iou,
            "center_dist": center_dist,
            "center_norm": center_norm,
            "gap": float(gap),
        }
    score = box_iou * 4.0 + max(0.0, 1.0 - center_norm / max(float(args.match_max_center_norm), 1e-6))
    score += max(0.0, 1.0 - (gap - 1) / max(float(args.max_assoc_gap), 1.0)) * 0.25
    return float(score), {
        "box_iou": box_iou,
        "center_dist": center_dist,
        "center_norm": center_norm,
        "gap": float(gap),
    }


def _associate_detections(by_frame: Sequence[Sequence[Det]], args: argparse.Namespace) -> Tuple[List[Track], List[Dict[str, Any]]]:
    tracks: List[Track] = []
    active: List[Track] = []
    events: List[Dict[str, Any]] = []
    next_id = 0
    for frame_dets in by_frame:
        used_track_ids: set[int] = set()
        for det in sorted(frame_dets, key=lambda d: d.confidence, reverse=True):
            best_track: Optional[Track] = None
            best_score = -1.0
            best_info: Dict[str, float] = {}
            for tr in active:
                if tr.track_id in used_track_ids:
                    continue
                score, info = _match_score(tr, det, args)
                if score > best_score:
                    best_score = score
                    best_track = tr
                    best_info = info
            if best_track is None or best_score < 0:
                tr = Track(track_id=next_id, label=det.label, dets=[det])
                tracks.append(tr)
                active.append(tr)
                events.append(
                    {
                        "event": "new_track",
                        "track_id": int(tr.track_id),
                        "frame_idx": int(det.frame_idx),
                        "label": det.label,
                        "confidence": float(det.confidence),
                    }
                )
                next_id += 1
            else:
                prev_frame = int(best_track.last.frame_idx)
                best_track.dets.append(det)
                used_track_ids.add(best_track.track_id)
                events.append(
                    {
                        "event": "associate",
                        "track_id": int(best_track.track_id),
                        "prev_frame": prev_frame,
                        "frame_idx": int(det.frame_idx),
                        "label": det.label,
                        "score": float(best_score),
                        **best_info,
                    }
                )
        current_frame = frame_dets[0].frame_idx if frame_dets else None
        if current_frame is not None:
            active = [
                tr for tr in active
                if int(current_frame - tr.last.frame_idx) <= int(args.max_assoc_gap)
            ]
    return tracks, events


def _track_to_sparse_dict(track: Track, H: int, W: int, output_idx: int) -> Dict[str, Any]:
    label = canonicalize_label(track.label)
    group = int(label_to_group(label))
    out: Dict[str, Any] = {
        "mask_by_frame": {},
        "box_by_frame": {},
        "q_by_frame": {},
        "area_by_frame": {},
        "L_sem": label,
        "G_sem": group,
        "W_sem": float(DEFAULT_SEMANTIC_WEIGHTS.get(group, 0.15)),
        "source_type": "thing_tracked",
        "birth_frame": int(track.dets[0].frame_idx),
        "frame_height": int(H),
        "frame_width": int(W),
        "_static_thing_track_id": int(track.track_id),
        "_static_thing_output_idx": int(output_idx),
        "mask_source": "yoloe_static_seg_per_frame",
        "proposal_source": "yoloe_static_prompt_per_frame",
        "tracking_source": "yoloe_static_frame_assoc",
        "label_source": "yoloe_static_prompt_label",
        "semantic_resolver": "none_static_append_label",
        "sam3_status": "not_run",
        "postprocess_history": [
            {
                "tool": "append_yoloe_static_thing_tracks.py",
                "operation": "append_static_thing_track",
            }
        ],
    }
    for det in track.dets:
        mask = det.mask.astype(bool)
        out["mask_by_frame"][int(det.frame_idx)] = _pack_mask_np(mask)
        out["box_by_frame"][int(det.frame_idx)] = torch.from_numpy(_mask_to_box_np(mask)).float()
        out["q_by_frame"][int(det.frame_idx)] = float(det.confidence)
        out["area_by_frame"][int(det.frame_idx)] = float(det.area_ratio)
    return out


def _filter_tracks(tracks: Sequence[Track], args: argparse.Namespace) -> Tuple[List[Track], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    kept_by_label: Dict[str, List[Track]] = {}
    for tr in tracks:
        frames = [int(det.frame_idx) for det in tr.dets]
        mean_conf = float(np.mean([det.confidence for det in tr.dets])) if tr.dets else 0.0
        mean_area = float(np.mean([det.area_ratio for det in tr.dets])) if tr.dets else 0.0
        keep = len(frames) >= int(args.min_track_frames) and mean_conf >= float(args.min_track_mean_conf)
        row = {
            "candidate_track_id": int(tr.track_id),
            "label": tr.label,
            "start_frame": int(min(frames)) if frames else -1,
            "end_frame": int(max(frames)) if frames else -1,
            "visible_frames": int(len(frames)),
            "mean_confidence": mean_conf,
            "mean_area_ratio": mean_area,
            "kept": bool(keep),
            "drop_reason": "" if keep else "short_or_low_conf",
        }
        if keep:
            kept_by_label.setdefault(tr.label, []).append(tr)
        rows.append(row)

    kept: List[Track] = []
    for label, label_tracks in sorted(kept_by_label.items()):
        label_tracks = sorted(
            label_tracks,
            key=lambda tr: (len(tr.dets), float(np.mean([d.confidence for d in tr.dets]))),
            reverse=True,
        )
        limit = max(0, int(args.max_tracks_per_label))
        kept.extend(label_tracks[:limit] if limit else label_tracks)
        dropped_ids = {tr.track_id for tr in label_tracks[limit:]} if limit else set()
        if dropped_ids:
            for row in rows:
                if int(row["candidate_track_id"]) in dropped_ids:
                    row["kept"] = False
                    row["drop_reason"] = "max_tracks_per_label"
    kept_ids = {tr.track_id for tr in kept}
    rows.sort(key=lambda r: (str(r["label"]), int(r["start_frame"]), int(r["candidate_track_id"])))
    return [tr for tr in tracks if tr.track_id in kept_ids], rows


def _common_frame_metrics(left: Track, right: Track, args: argparse.Namespace) -> Dict[str, Any]:
    left_by_frame = {det.frame_idx: det for det in left.dets}
    right_by_frame = {det.frame_idx: det for det in right.dets}
    common = sorted(set(left_by_frame).intersection(right_by_frame))
    duplicate_frames = 0
    mask_ious: List[float] = []
    containments: List[float] = []
    box_ious: List[float] = []
    center_norms: List[float] = []
    for frame_idx in common:
        left_det = left_by_frame[frame_idx]
        right_det = right_by_frame[frame_idx]
        inter = int(np.logical_and(left_det.mask, right_det.mask).sum())
        left_area = int(left_det.mask.sum())
        right_area = int(right_det.mask.sum())
        union = int(np.logical_or(left_det.mask, right_det.mask).sum())
        mask_iou = float(inter) / float(max(union, 1))
        left_in_right = float(inter) / float(max(left_area, 1))
        right_in_left = float(inter) / float(max(right_area, 1))
        containment = max(left_in_right, right_in_left)
        box_iou = _box_iou(left_det.box, right_det.box)
        center_dist = float(np.linalg.norm(_center(left_det.box) - _center(right_det.box)))
        scale = max(
            _box_scale(left_det.box, float(args.center_scale_floor)),
            _box_scale(right_det.box, float(args.center_scale_floor)),
        )
        center_norm = float(center_dist / max(scale, 1e-6))
        mask_ious.append(mask_iou)
        containments.append(containment)
        box_ious.append(box_iou)
        center_norms.append(center_norm)
        overlap_duplicate = mask_iou >= float(args.merge_min_mask_iou) or containment >= float(args.merge_min_containment)
        center_duplicate = box_iou >= float(args.merge_min_box_iou) and center_norm <= float(args.merge_max_center_norm)
        if overlap_duplicate or center_duplicate:
            duplicate_frames += 1
    return {
        "common_frames": int(len(common)),
        "duplicate_frames": int(duplicate_frames),
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "max_mask_iou": float(np.max(mask_ious)) if mask_ious else 0.0,
        "mean_containment": float(np.mean(containments)) if containments else 0.0,
        "max_containment": float(np.max(containments)) if containments else 0.0,
        "mean_box_iou": float(np.mean(box_ious)) if box_ious else 0.0,
        "mean_center_norm": float(np.mean(center_norms)) if center_norms else 0.0,
    }


def _find(parent: Dict[int, int], value: int) -> int:
    root = value
    while parent[root] != root:
        root = parent[root]
    while parent[value] != value:
        value, parent[value] = parent[value], root
    return root


def _union(parent: Dict[int, int], left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _merge_track_group(group: Sequence[Track], new_track_id: int) -> Track:
    if not group:
        raise ValueError("empty track group")
    label = group[0].label
    by_frame: Dict[int, List[Det]] = {}
    for track in group:
        for det in track.dets:
            by_frame.setdefault(int(det.frame_idx), []).append(det)
    merged_dets: List[Det] = []
    for frame_idx in sorted(by_frame):
        dets = by_frame[frame_idx]
        if len(dets) == 1:
            merged_dets.append(dets[0])
            continue
        mask = np.zeros_like(dets[0].mask, dtype=bool)
        conf = max(float(det.confidence) for det in dets)
        for det in dets:
            mask |= det.mask.astype(bool)
        area_ratio = float(mask.sum()) / float(max(mask.shape[0] * mask.shape[1], 1))
        merged_dets.append(
            Det(
                frame_idx=int(frame_idx),
                label=label,
                mask=mask,
                box=_mask_to_box_np(mask).astype(np.float32),
                confidence=conf,
                area_ratio=area_ratio,
            )
        )
    return Track(track_id=int(new_track_id), label=label, dets=merged_dets)


def _merge_static_label_tracks(tracks: Sequence[Track], args: argparse.Namespace) -> Tuple[List[Track], List[Dict[str, Any]]]:
    if not int(args.merge_static_label_tracks) or len(tracks) <= 1:
        return list(tracks), []
    parent = {int(track.track_id): int(track.track_id) for track in tracks}
    by_id = {int(track.track_id): track for track in tracks}
    merge_rows: List[Dict[str, Any]] = []
    for left_idx in range(len(tracks)):
        for right_idx in range(left_idx + 1, len(tracks)):
            left = tracks[left_idx]
            right = tracks[right_idx]
            if left.label != right.label:
                continue
            metrics = _common_frame_metrics(left, right, args)
            if int(metrics["common_frames"]) < int(args.merge_common_min_frames):
                continue
            if int(metrics["duplicate_frames"]) < int(args.merge_min_duplicate_frames):
                continue
            _union(parent, int(left.track_id), int(right.track_id))
            merge_rows.append(
                {
                    "left_track_id": int(left.track_id),
                    "right_track_id": int(right.track_id),
                    "label": left.label,
                    **metrics,
                }
            )

    groups: Dict[int, List[Track]] = {}
    for track_id, track in by_id.items():
        groups.setdefault(_find(parent, track_id), []).append(track)
    merged: List[Track] = []
    next_track_id = 0
    for _root, group in sorted(groups.items(), key=lambda item: min(tr.track_id for tr in item[1])):
        if len(group) == 1:
            single = group[0]
            merged.append(Track(track_id=next_track_id, label=single.label, dets=list(single.dets)))
        else:
            merged.append(_merge_track_group(group, next_track_id))
        next_track_id += 1
    return merged, merge_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "candidate_track_id",
        "label",
        "start_frame",
        "end_frame",
        "visible_frames",
        "mean_confidence",
        "mean_area_ratio",
        "kept",
        "drop_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_events_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    prompts = _parse_csv_labels(args.prompts)
    append_labels = set(_parse_csv_labels(args.append_labels))
    conf_overrides = _parse_conf_overrides(args.min_conf_by_label)
    label_map = _parse_label_map(args.label_map)
    if not prompts:
        raise RuntimeError("No prompts provided.")
    if not append_labels:
        raise RuntimeError("No append_labels provided.")

    image_paths, temp_dirs = _load_processing_frames(
        args.input_video,
        int(args.processing_max_side),
        int(args.frames_limit),
        H,
        W,
        int(sparse.num_frames),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOEDetector(
        model_path=str(args.yoloe_model),
        device=str(args.device),
        batch_size=int(args.yoloe_batch_size),
        imgsz=int(args.yoloe_imgsz),
    )
    try:
        by_frame, detection_summary = _detections_from_yoloe(
            detector,
            image_paths,
            prompts,
            append_labels,
            conf_overrides,
            label_map,
            args,
            H,
            W,
        )
    finally:
        detector.release_gpu()

    candidate_tracks, assoc_events = _associate_detections(by_frame, args)
    kept_tracks, track_rows = _filter_tracks(candidate_tracks, args)
    merged_tracks, merge_rows = _merge_static_label_tracks(kept_tracks, args)

    out = clone_sparse(sparse)
    start_track_count = len(out.tracks)
    for offset, track in enumerate(merged_tracks):
        out.tracks.append(_track_to_sparse_dict(track, H, W, start_track_count + offset))
    out.num_masklets = len(out.tracks)
    out.debug = dict(out.debug)
    out.debug["append_yoloe_static_thing_tracks"] = {
        "format": "append_yoloe_static_thing_tracks_v1",
        "input_pt": str(args.input_pt),
        "yoloe_model": str(args.yoloe_model),
        "prompts": prompts,
        "append_labels": sorted(append_labels),
        "input_tracks": int(start_track_count),
        "candidate_tracks": int(len(candidate_tracks)),
        "kept_tracks_before_static_merge": int(len(kept_tracks)),
        "static_merge_events": int(len(merge_rows)),
        "appended_tracks": int(len(merged_tracks)),
        "args": {
            "box_threshold": float(args.box_threshold),
            "min_conf": float(args.min_conf),
            "min_conf_by_label": conf_overrides,
            "min_track_frames": int(args.min_track_frames),
            "max_assoc_gap": int(args.max_assoc_gap),
            "match_min_box_iou": float(args.match_min_box_iou),
            "match_max_center_px": float(args.match_max_center_px),
            "match_max_center_norm": float(args.match_max_center_norm),
            "label_map": label_map,
            "merge_static_label_tracks": int(args.merge_static_label_tracks),
            "merge_common_min_frames": int(args.merge_common_min_frames),
            "merge_min_duplicate_frames": int(args.merge_min_duplicate_frames),
        },
    }

    output_pt = out_dir / "sparse_masklets.pt"
    save_sparse_output(output_pt, out)
    create_tracking_video_v2(
        image_paths,
        out,
        str(out_dir / "overlay_final.mp4"),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style=str(args.render_style),
    )

    metrics = {
        "input_tracks": int(start_track_count),
        "output_tracks": int(len(out.tracks)),
        "candidate_tracks": int(len(candidate_tracks)),
        "kept_tracks_before_static_merge": int(len(kept_tracks)),
        "static_merge_events": int(len(merge_rows)),
        "appended_tracks": int(len(merged_tracks)),
        "detection_summary": detection_summary,
        "track_stats": track_stats(out),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "static_thing_tracks.csv", track_rows)
    _write_csv(out_dir / "static_thing_merge_events.csv", merge_rows)
    _write_events_jsonl(out_dir / "association_events.jsonl", assoc_events)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    for temp_dir in temp_dirs:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
