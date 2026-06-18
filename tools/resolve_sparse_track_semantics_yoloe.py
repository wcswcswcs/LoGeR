#!/usr/bin/env python3
"""Resolve sparse track labels with track-level YOLOE evidence.

This is the first resolver-style step after detection/propagation. It does not
create or manually merge tracks. It samples each existing track over time, runs
YOLOE with a fixed prompt set, aggregates per-label evidence over matched
detections, and only relabels a track when the winning label has enough support
and margin. The output records resolver evidence in each track's provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import _unpack_mask_np, load_sparse  # noqa: E402
from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import (  # noqa: E402
    DEFAULT_SEMANTIC_WEIGHTS,
    YOLOEDetector,
    canonicalize_label,
    label_to_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve sparse track labels with track-level YOLOE voting.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--yoloe_model", default="yoloe-26l-seg.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yoloe_batch_size", type=int, default=4)
    parser.add_argument("--yoloe_imgsz", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    driving_prompts = "car,person,traffic sign,road sign,sign board,billboard,traffic light,pole,barrier,traffic cone"
    parser.add_argument("--prompts", default=driving_prompts)
    parser.add_argument("--candidate_labels", default=driving_prompts)
    parser.add_argument("--relabel_from", default=driving_prompts)
    parser.add_argument("--max_frames_per_track", type=int, default=16, help="Frames sampled per track; <=0 samples every frame in the track.")
    parser.add_argument("--box_threshold", type=float, default=0.20)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--min_det_conf", type=float, default=0.10)
    parser.add_argument("--min_box_iou", type=float, default=0.05)
    parser.add_argument("--min_mask_iou", type=float, default=0.02)
    parser.add_argument("--min_containment", type=float, default=0.20)
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--min_winner_score", type=float, default=0.20)
    parser.add_argument("--min_margin", type=float, default=0.08)
    parser.add_argument("--render_video", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    return parser.parse_args()


def _parse_labels(raw: str) -> List[str]:
    labels: List[str] = []
    for part in str(raw or "").split(","):
        item = canonicalize_label(part.strip())
        if item and item not in labels:
            labels.append(item)
    return labels


def _frames(track: Dict[str, Any]) -> List[int]:
    return sorted(int(frame) for frame in track.get("mask_by_frame", {}).keys())


def _sample_frames(frames: Sequence[int], limit: int) -> List[int]:
    if limit <= 0 or len(frames) <= limit:
        return [int(frame) for frame in frames]
    indices = np.linspace(0, len(frames) - 1, int(limit))
    return sorted({int(frames[int(round(idx))]) for idx in indices})


def _load_frames(input_video: str, processing_max_side: int) -> Tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    image_paths, resize_tmp, _orig_shape, _proc_shape = prepare_processing_image_paths(image_paths, int(processing_max_side))
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    return image_paths, temp_dirs


def _box_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return np.zeros(4, dtype=np.float32)
    return arr[:4].astype(np.float32)


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    denom = left_area + right_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _mask_metrics(left: np.ndarray, right: np.ndarray) -> Tuple[float, float, float]:
    left = left.astype(bool)
    right = right.astype(bool)
    left_area = int(left.sum())
    right_area = int(right.sum())
    if left_area <= 0 or right_area <= 0:
        return 0.0, 0.0, 0.0
    inter = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return (
        float(inter / max(union, 1)),
        float(inter / max(left_area, 1)),
        float(inter / max(right_area, 1)),
    )


def _det_mask(det: Dict[str, Any], H: int, W: int) -> Optional[np.ndarray]:
    mask = det.get("mask")
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.shape[:2] != (H, W):
        return None
    return arr.astype(bool)


def _track_mask(track: Dict[str, Any], frame_idx: int, H: int, W: int) -> Optional[np.ndarray]:
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        packed = track.get("mask_by_frame", {}).get(str(int(frame_idx)))
    if packed is None:
        return None
    return _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)


def _best_detection_evidence(
    track: Dict[str, Any],
    frame_idx: int,
    detections: Sequence[Dict[str, Any]],
    candidate_labels: set[str],
    H: int,
    W: int,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    tmask = _track_mask(track, int(frame_idx), H, W)
    if tmask is None:
        return None
    tbox = _box_array(track.get("box_by_frame", {}).get(int(frame_idx)))
    best: Optional[Dict[str, Any]] = None
    for det in detections:
        label = canonicalize_label(str(det.get("label", "")))
        if label not in candidate_labels:
            continue
        conf = float(det.get("confidence", 0.0))
        if conf < float(args.min_det_conf):
            continue
        dbox = _box_array(det.get("box"))
        box_iou = _box_iou(tbox, dbox)
        mask_iou = 0.0
        track_in_det = 0.0
        det_in_track = 0.0
        dmask = _det_mask(det, H, W)
        if dmask is not None:
            mask_iou, track_in_det, det_in_track = _mask_metrics(tmask, dmask)
        containment = max(track_in_det, det_in_track)
        if (
            box_iou < float(args.min_box_iou)
            and mask_iou < float(args.min_mask_iou)
            and containment < float(args.min_containment)
        ):
            continue
        overlap = max(box_iou, mask_iou, containment)
        score = float(conf * overlap)
        row = {
            "frame_idx": int(frame_idx),
            "label": label,
            "confidence": conf,
            "score": score,
            "box_iou": box_iou,
            "mask_iou": mask_iou,
            "track_in_det": track_in_det,
            "det_in_track": det_in_track,
            "overlap": overlap,
        }
        if best is None or row["score"] > best["score"]:
            best = row
    return best


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse(Path(args.input_pt))
    H = int(sparse.frame_height)
    W = int(sparse.frame_width)
    prompts = _parse_labels(args.prompts)
    candidate_labels = set(_parse_labels(args.candidate_labels))
    relabel_from = set(_parse_labels(args.relabel_from))

    sampled_by_track: Dict[int, List[int]] = {}
    frames_to_detect: set[int] = set()
    for idx, track in enumerate(sparse.tracks):
        frames = _frames(track)
        sampled = _sample_frames(frames, int(args.max_frames_per_track))
        sampled_by_track[idx] = sampled
        frames_to_detect.update(sampled)

    image_paths, temp_dirs = _load_frames(str(args.input_video), int(args.processing_max_side))
    selected_frames = sorted(frame for frame in frames_to_detect if 0 <= frame < len(image_paths))
    detector = YOLOEDetector(
        model_path=str(args.yoloe_model),
        device=str(args.device),
        batch_size=int(args.yoloe_batch_size),
        imgsz=int(args.yoloe_imgsz),
    )
    detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    try:
        selected_paths = [image_paths[frame] for frame in selected_frames]
        detections = detector.detect_batch(
            selected_paths,
            thing_prompts=prompts,
            stuff_prompts=[],
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
        )
        detections_by_frame = {int(frame): list(dets) for frame, dets in zip(selected_frames, detections)}
    finally:
        detector.release_gpu()
        for temp_dir in temp_dirs:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    rows: List[Dict[str, Any]] = []
    changed = 0
    for idx, track in enumerate(sparse.tracks):
        current = canonicalize_label(str(track.get("L_sem", "")))
        evidence_sum: Dict[str, float] = {}
        support_frames: Dict[str, int] = {}
        best_events: List[Dict[str, Any]] = []
        for frame_idx in sampled_by_track.get(idx, []):
            event = _best_detection_evidence(
                track,
                int(frame_idx),
                detections_by_frame.get(int(frame_idx), []),
                candidate_labels,
                H,
                W,
                args,
            )
            if event is None:
                continue
            label = str(event["label"])
            evidence_sum[label] = evidence_sum.get(label, 0.0) + float(event["score"])
            support_frames[label] = support_frames.get(label, 0) + 1
            best_events.append(event)

        ranked = sorted(evidence_sum.items(), key=lambda item: item[1], reverse=True)
        winner = ranked[0][0] if ranked else current
        winner_score = float(ranked[0][1]) if ranked else 0.0
        runner_up = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = winner_score - runner_up
        support = int(support_frames.get(winner, 0))
        decision = "keep"
        if (
            current in relabel_from
            and winner != current
            and support >= int(args.min_support_frames)
            and winner_score >= float(args.min_winner_score)
            and margin >= float(args.min_margin)
        ):
            group = int(label_to_group(winner))
            track["L_sem"] = winner
            track["G_sem"] = group
            track["W_sem"] = float(DEFAULT_SEMANTIC_WEIGHTS.get(group, 0.15))
            history = list(track.get("postprocess_history", []))
            history.append(
                {
                    "tool": "resolve_sparse_track_semantics_yoloe.py",
                    "operation": "track_level_relabel",
                    "old_label": current,
                    "new_label": winner,
                    "winner_score": winner_score,
                    "runner_up_score": runner_up,
                    "support_frames": support,
                }
            )
            track["postprocess_history"] = history
            track["label_source"] = "track_level_yoloe_vote"
            track["semantic_resolver"] = "yoloe_track_vote_v1"
            changed += 1
            decision = "relabel"
        else:
            track.setdefault("semantic_resolver", "yoloe_track_vote_v1_checked")

        rows.append(
            {
                "track_index": int(idx),
                "old_label": current,
                "new_label": str(track.get("L_sem", current)),
                "decision": decision,
                "winner": winner,
                "winner_score": winner_score,
                "runner_up_score": runner_up,
                "margin": margin,
                "support_frames": support,
                "sampled_frames": len(sampled_by_track.get(idx, [])),
                "evidence_json": json.dumps(evidence_sum, ensure_ascii=False, sort_keys=True),
                "events_json": json.dumps(best_events[:20], ensure_ascii=False, sort_keys=True),
            }
        )

    output_pt = out_dir / "sparse_masklets.pt"
    sparse.debug = dict(sparse.debug)
    sparse.debug["resolve_sparse_track_semantics_yoloe"] = {
        "format": "resolve_sparse_track_semantics_yoloe_v1",
        "input_pt": str(args.input_pt),
        "prompts": prompts,
        "candidate_labels": sorted(candidate_labels),
        "relabel_from": sorted(relabel_from),
        "changed_tracks": int(changed),
        "sampled_frame_count": int(len(selected_frames)),
        "args": {
            "max_frames_per_track": int(args.max_frames_per_track),
            "min_support_frames": int(args.min_support_frames),
            "min_winner_score": float(args.min_winner_score),
            "min_margin": float(args.min_margin),
            "min_box_iou": float(args.min_box_iou),
            "min_mask_iou": float(args.min_mask_iou),
            "min_containment": float(args.min_containment),
        },
    }
    save_sparse_output(output_pt, sparse)

    with (out_dir / "semantic_resolution_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["track_index"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "changed_tracks": int(changed),
        "track_count": int(len(sparse.tracks)),
        "sampled_frame_count": int(len(selected_frames)),
    }
    (out_dir / "semantic_resolution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if int(args.render_video):
        image_paths, temp_dirs = _load_frames(str(args.input_video), int(args.processing_max_side))
        try:
            create_tracking_video_v2(
                image_paths[: sparse.num_frames],
                sparse,
                str(out_dir / "overlay_final.mp4"),
                fps=int(args.fps),
                mask_alpha=float(args.mask_alpha),
                render_style="debug",
            )
        finally:
            for temp_dir in temp_dirs:
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
