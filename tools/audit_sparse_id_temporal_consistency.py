#!/usr/bin/env python3
"""Audit sparse masklet temporal ID consistency at frame-level resolution."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import _unpack_mask_np, load_sparse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sparse masklet adjacent-frame ID continuity.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels", default="car,person", help="Comma-separated labels to audit; all/off audits all labels.")
    parser.add_argument("--center_scale_floor", type=float, default=24.0)
    parser.add_argument("--adjacent_min_box_iou", type=float, default=0.10)
    parser.add_argument("--adjacent_min_mask_iou", type=float, default=0.0)
    parser.add_argument("--adjacent_min_mask_containment", type=float, default=0.0)
    parser.add_argument("--adjacent_max_center_dist", type=float, default=0.80)
    parser.add_argument("--same_frame_min_box_iou", type=float, default=0.05)
    parser.add_argument("--same_frame_min_mask_iou", type=float, default=0.0)
    parser.add_argument("--same_frame_min_mask_containment", type=float, default=0.0)
    parser.add_argument("--same_frame_max_center_dist", type=float, default=0.60)
    parser.add_argument("--gap_max", type=int, default=35)
    parser.add_argument("--gap_min_box_iou", type=float, default=0.20)
    parser.add_argument("--gap_min_mask_iou", type=float, default=0.0)
    parser.add_argument("--gap_min_mask_containment", type=float, default=0.0)
    parser.add_argument("--gap_max_center_dist", type=float, default=0.80)
    parser.add_argument("--pair_duplicate_mask_iou", type=float, default=0.50)
    parser.add_argument("--pair_duplicate_mask_containment", type=float, default=0.70)
    parser.add_argument("--pair_conflict_mask_iou", type=float, default=0.05)
    parser.add_argument("--pair_conflict_mask_containment", type=float, default=0.10)
    parser.add_argument("--pair_low_conflict_max_ratio", type=float, default=0.20)
    parser.add_argument("--max_rows", type=int, default=5000)
    return parser.parse_args()


def label_set(raw: str) -> Optional[set[str]]:
    text = str(raw).strip()
    if not text or text.lower() in {"all", "off", "*"}:
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def box_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4]


def box_area(box: np.ndarray) -> float:
    return float(max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])))


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = box_area(left) + box_area(right) - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5])


def normalized_center_dist(left: np.ndarray, right: np.ndarray, scale_floor: float) -> float:
    left_w = max(1.0, float(left[2] - left[0]))
    left_h = max(1.0, float(left[3] - left[1]))
    right_w = max(1.0, float(right[2] - right[0]))
    right_h = max(1.0, float(right[3] - right[1]))
    scale = max(float(scale_floor), left_w, left_h, right_w, right_h)
    return float(np.linalg.norm(center(left) - center(right)) / max(scale, 1e-6))


def area_ratio(left: np.ndarray, right: np.ndarray) -> float:
    a = box_area(left)
    b = box_area(right)
    return float(min(a, b) / max(a, b, 1e-6))


def mask_array(track: Dict[str, Any], frame_idx: int, H: int, W: int) -> Optional[np.ndarray]:
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        packed = track.get("mask_by_frame", {}).get(str(int(frame_idx)))
    if packed is None:
        return None
    try:
        return _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
    except Exception:
        return None


def mask_overlap_metrics(
    left_track: Dict[str, Any],
    right_track: Dict[str, Any],
    frame_left: int,
    frame_right: int,
    H: int,
    W: int,
) -> Tuple[float, float, float]:
    left = mask_array(left_track, frame_left, H, W)
    right = mask_array(right_track, frame_right, H, W)
    if left is None or right is None:
        return 0.0, 0.0, 0.0
    left_area = int(left.sum())
    right_area = int(right.sum())
    if left_area <= 0 or right_area <= 0:
        return 0.0, 0.0, 0.0
    inter = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    mask_iou = float(inter) / float(max(union, 1))
    left_in_right = float(inter) / float(max(left_area, 1))
    right_in_left = float(inter) / float(max(right_area, 1))
    return mask_iou, left_in_right, right_in_left


def passes_mask_gate(
    mask_iou: float,
    left_in_right: float,
    right_in_left: float,
    min_mask_iou: float,
    min_mask_containment: float,
) -> bool:
    if float(min_mask_iou) <= 0.0 and float(min_mask_containment) <= 0.0:
        return True
    return (
        float(mask_iou) >= float(min_mask_iou)
        or max(float(left_in_right), float(right_in_left)) >= float(min_mask_containment)
    )


def frames_for(track: Dict[str, Any]) -> List[int]:
    return sorted(int(k) for k in track.get("mask_by_frame", {}).keys())


def pair_common_frame_stats(
    left_track: Dict[str, Any],
    right_track: Dict[str, Any],
    H: int,
    W: int,
    duplicate_mask_iou: float,
    duplicate_mask_containment: float,
    conflict_mask_iou: float,
    conflict_mask_containment: float,
) -> Dict[str, Any]:
    common = sorted(set(frames_for(left_track)).intersection(frames_for(right_track)))
    duplicate_frames: List[int] = []
    conflict_frames: List[int] = []
    mask_ious: List[float] = []
    containments: List[float] = []
    for frame_idx in common:
        mask_iou, left_in_right, right_in_left = mask_overlap_metrics(
            left_track,
            right_track,
            int(frame_idx),
            int(frame_idx),
            H,
            W,
        )
        containment = max(float(left_in_right), float(right_in_left))
        mask_ious.append(float(mask_iou))
        containments.append(float(containment))
        if mask_iou >= float(duplicate_mask_iou) or containment >= float(duplicate_mask_containment):
            duplicate_frames.append(int(frame_idx))
        if mask_iou < float(conflict_mask_iou) and containment < float(conflict_mask_containment):
            conflict_frames.append(int(frame_idx))
    common_count = len(common)
    return {
        "pair_common_frames": int(common_count),
        "pair_duplicate_frames": int(len(duplicate_frames)),
        "pair_conflict_frames": int(len(conflict_frames)),
        "pair_duplicate_ratio": float(len(duplicate_frames) / max(1, common_count)),
        "pair_conflict_ratio": float(len(conflict_frames) / max(1, common_count)),
        "pair_max_mask_iou": float(max(mask_ious) if mask_ious else 0.0),
        "pair_mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "pair_max_containment": float(max(containments) if containments else 0.0),
        "pair_mean_containment": float(np.mean(containments)) if containments else 0.0,
        "pair_duplicate_frame_preview": json.dumps(duplicate_frames[:20]),
        "pair_conflict_frame_preview": json.dumps(conflict_frames[:20]),
    }


def row_for_pair(
    frame_left: int,
    frame_right: int,
    left_idx: int,
    right_idx: int,
    left_track: Dict[str, Any],
    right_track: Dict[str, Any],
    left_box: np.ndarray,
    right_box: np.ndarray,
    scale_floor: float,
    kind: str,
    H: int,
    W: int,
    pair_duplicate_mask_iou: float,
    pair_duplicate_mask_containment: float,
    pair_conflict_mask_iou: float,
    pair_conflict_mask_containment: float,
) -> Dict[str, Any]:
    iou = box_iou(left_box, right_box)
    cdist = normalized_center_dist(left_box, right_box, scale_floor)
    ar = area_ratio(left_box, right_box)
    mask_iou, left_in_right, right_in_left = mask_overlap_metrics(
        left_track,
        right_track,
        frame_left,
        frame_right,
        H,
        W,
    )
    row = {
        "kind": kind,
        "frame_left": int(frame_left),
        "frame_right": int(frame_right),
        "gap": int(frame_right - frame_left),
        "left_track_index": int(left_idx),
        "right_track_index": int(right_idx),
        "left_track_id": str(left_track.get("track_id", left_idx)),
        "right_track_id": str(right_track.get("track_id", right_idx)),
        "label": str(left_track.get("L_sem", "")),
        "box_iou": float(iou),
        "mask_iou": float(mask_iou),
        "left_mask_in_right": float(left_in_right),
        "right_mask_in_left": float(right_in_left),
        "center_dist_norm": float(cdist),
        "area_ratio": float(ar),
        "left_box": json.dumps([round(float(x), 3) for x in left_box.tolist()]),
        "right_box": json.dumps([round(float(x), 3) for x in right_box.tolist()]),
    }
    row.update(
        pair_common_frame_stats(
            left_track,
            right_track,
            H,
            W,
            pair_duplicate_mask_iou,
            pair_duplicate_mask_containment,
            pair_conflict_mask_iou,
            pair_conflict_mask_containment,
        )
    )
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "frame_left",
        "frame_right",
        "gap",
        "left_track_index",
        "right_track_index",
        "left_track_id",
        "right_track_id",
        "label",
        "box_iou",
        "mask_iou",
        "left_mask_in_right",
        "right_mask_in_left",
        "center_dist_norm",
        "area_ratio",
        "left_length",
        "right_length",
        "pair_common_frames",
        "pair_duplicate_frames",
        "pair_conflict_frames",
        "pair_duplicate_ratio",
        "pair_conflict_ratio",
        "pair_max_mask_iou",
        "pair_mean_mask_iou",
        "pair_max_containment",
        "pair_mean_containment",
        "pair_duplicate_frame_preview",
        "pair_conflict_frame_preview",
        "left_box",
        "right_box",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def active_boxes_by_frame(tracks: List[Dict[str, Any]], labels: Optional[set[str]]) -> Dict[int, List[Tuple[int, Dict[str, Any], np.ndarray]]]:
    by_frame: Dict[int, List[Tuple[int, Dict[str, Any], np.ndarray]]] = {}
    for track_idx, track in enumerate(tracks):
        label = str(track.get("L_sem", ""))
        if labels is not None and label not in labels:
            continue
        for frame_idx, box in track.get("box_by_frame", {}).items():
            arr = box_array(box)
            if arr is None:
                continue
            by_frame.setdefault(int(frame_idx), []).append((track_idx, track, arr))
    return by_frame


def capped_append(rows: List[Dict[str, Any]], row: Dict[str, Any], max_rows: int) -> None:
    if len(rows) < max_rows:
        rows.append(row)


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    tracks: List[Dict[str, Any]] = list(sparse.tracks)
    H = int(getattr(sparse, "frame_height", 0))
    W = int(getattr(sparse, "frame_width", 0))
    labels = label_set(args.labels)
    by_frame = active_boxes_by_frame(tracks, labels)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adjacent_rows: List[Dict[str, Any]] = []
    same_frame_rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []
    max_rows = int(args.max_rows)
    scale_floor = float(args.center_scale_floor)

    max_frame = int(getattr(sparse, "num_frames", max(by_frame.keys()) + 1 if by_frame else 0))
    for frame_idx in range(max(0, max_frame - 1)):
        left_active = by_frame.get(frame_idx, [])
        right_active = by_frame.get(frame_idx + 1, [])
        left_indices_next = {idx for idx, _track, _box in right_active}
        right_indices_prev = {idx for idx, _track, _box in left_active}
        ending = [(idx, track, box) for idx, track, box in left_active if idx not in left_indices_next]
        starting = [(idx, track, box) for idx, track, box in right_active if idx not in right_indices_prev]
        for li, left_track, left_box in ending:
            for ri, right_track, right_box in starting:
                if li == ri or str(left_track.get("L_sem", "")) != str(right_track.get("L_sem", "")):
                    continue
                iou = box_iou(left_box, right_box)
                cdist = normalized_center_dist(left_box, right_box, scale_floor)
                mask_iou, left_in_right, right_in_left = mask_overlap_metrics(
                    left_track,
                    right_track,
                    frame_idx,
                    frame_idx + 1,
                    H,
                    W,
                )
                spatial_gate = iou >= float(args.adjacent_min_box_iou) or cdist <= float(args.adjacent_max_center_dist)
                mask_gate = passes_mask_gate(
                    mask_iou,
                    left_in_right,
                    right_in_left,
                    float(args.adjacent_min_mask_iou),
                    float(args.adjacent_min_mask_containment),
                )
                if spatial_gate and mask_gate:
                    capped_append(
                        adjacent_rows,
                        row_for_pair(
                            frame_idx,
                            frame_idx + 1,
                            li,
                            ri,
                            left_track,
                            right_track,
                            left_box,
                            right_box,
                            scale_floor,
                            "adjacent_switch_candidate",
                            H,
                            W,
                            float(args.pair_duplicate_mask_iou),
                            float(args.pair_duplicate_mask_containment),
                            float(args.pair_conflict_mask_iou),
                            float(args.pair_conflict_mask_containment),
                        ),
                        max_rows,
                    )

    for frame_idx, active in by_frame.items():
        for ai in range(len(active)):
            li, left_track, left_box = active[ai]
            for bi in range(ai + 1, len(active)):
                ri, right_track, right_box = active[bi]
                if str(left_track.get("L_sem", "")) != str(right_track.get("L_sem", "")):
                    continue
                iou = box_iou(left_box, right_box)
                cdist = normalized_center_dist(left_box, right_box, scale_floor)
                mask_iou, left_in_right, right_in_left = mask_overlap_metrics(
                    left_track,
                    right_track,
                    frame_idx,
                    frame_idx,
                    H,
                    W,
                )
                spatial_gate = iou >= float(args.same_frame_min_box_iou) or cdist <= float(args.same_frame_max_center_dist)
                mask_gate = passes_mask_gate(
                    mask_iou,
                    left_in_right,
                    right_in_left,
                    float(args.same_frame_min_mask_iou),
                    float(args.same_frame_min_mask_containment),
                )
                if spatial_gate and mask_gate:
                    capped_append(
                        same_frame_rows,
                        row_for_pair(
                            frame_idx,
                            frame_idx,
                            li,
                            ri,
                            left_track,
                            right_track,
                            left_box,
                            right_box,
                            scale_floor,
                            "same_frame_duplicate_candidate",
                            H,
                            W,
                            float(args.pair_duplicate_mask_iou),
                            float(args.pair_duplicate_mask_containment),
                            float(args.pair_conflict_mask_iou),
                            float(args.pair_conflict_mask_containment),
                        ),
                        max_rows,
                    )

    summaries = []
    for idx, track in enumerate(tracks):
        label = str(track.get("L_sem", ""))
        if labels is not None and label not in labels:
            continue
        fs = frames_for(track)
        if not fs:
            continue
        summaries.append((idx, track, fs[0], fs[-1], len(fs)))
    for li, left_track, _left_start, left_end, left_len in summaries:
        left_box = box_array(left_track.get("box_by_frame", {}).get(int(left_end)))
        if left_box is None:
            continue
        for ri, right_track, right_start, _right_end, right_len in summaries:
            if li == ri or str(left_track.get("L_sem", "")) != str(right_track.get("L_sem", "")):
                continue
            gap = int(right_start - left_end)
            if gap <= 1 or gap > int(args.gap_max):
                continue
            right_box = box_array(right_track.get("box_by_frame", {}).get(int(right_start)))
            if right_box is None:
                continue
            iou = box_iou(left_box, right_box)
            cdist = normalized_center_dist(left_box, right_box, scale_floor)
            mask_iou, left_in_right, right_in_left = mask_overlap_metrics(
                left_track,
                right_track,
                left_end,
                right_start,
                H,
                W,
            )
            spatial_gate = iou >= float(args.gap_min_box_iou) or cdist <= float(args.gap_max_center_dist)
            mask_gate = passes_mask_gate(
                mask_iou,
                left_in_right,
                right_in_left,
                float(args.gap_min_mask_iou),
                float(args.gap_min_mask_containment),
            )
            if spatial_gate and mask_gate:
                row = row_for_pair(
                    left_end,
                    right_start,
                    li,
                    ri,
                    left_track,
                    right_track,
                    left_box,
                    right_box,
                    scale_floor,
                    "gap_handoff_candidate",
                    H,
                    W,
                    float(args.pair_duplicate_mask_iou),
                    float(args.pair_duplicate_mask_containment),
                    float(args.pair_conflict_mask_iou),
                    float(args.pair_conflict_mask_containment),
                )
                row["left_length"] = int(left_len)
                row["right_length"] = int(right_len)
                capped_append(gap_rows, row, max_rows)

    adjacent_rows.sort(key=lambda r: (int(r["frame_left"]), -float(r["box_iou"]), float(r["center_dist_norm"])))
    same_frame_rows.sort(key=lambda r: (int(r["frame_left"]), -float(r["box_iou"]), float(r["center_dist_norm"])))
    gap_rows.sort(key=lambda r: (int(r["frame_left"]), int(r["frame_right"]), -float(r["box_iou"]), float(r["center_dist_norm"])))

    write_csv(out_dir / "adjacent_switch_candidates.csv", adjacent_rows)
    write_csv(out_dir / "same_frame_duplicate_candidates.csv", same_frame_rows)
    write_csv(out_dir / "gap_handoff_candidates.csv", gap_rows)
    summary = {
        "input_pt": str(args.input_pt),
        "track_count": int(len(tracks)),
        "num_frames": int(max_frame),
        "labels": sorted(labels) if labels is not None else "all",
        "adjacent_switch_candidate_count": int(len(adjacent_rows)),
        "same_frame_duplicate_candidate_count": int(len(same_frame_rows)),
        "same_frame_low_conflict_candidate_count": int(
            sum(float(row.get("pair_conflict_ratio", 1.0)) <= float(args.pair_low_conflict_max_ratio) for row in same_frame_rows)
        ),
        "gap_handoff_candidate_count": int(len(gap_rows)),
        "thresholds": {
            "center_scale_floor": float(scale_floor),
            "adjacent_min_box_iou": float(args.adjacent_min_box_iou),
            "adjacent_min_mask_iou": float(args.adjacent_min_mask_iou),
            "adjacent_min_mask_containment": float(args.adjacent_min_mask_containment),
            "adjacent_max_center_dist": float(args.adjacent_max_center_dist),
            "same_frame_min_box_iou": float(args.same_frame_min_box_iou),
            "same_frame_min_mask_iou": float(args.same_frame_min_mask_iou),
            "same_frame_min_mask_containment": float(args.same_frame_min_mask_containment),
            "same_frame_max_center_dist": float(args.same_frame_max_center_dist),
            "gap_max": int(args.gap_max),
            "gap_min_box_iou": float(args.gap_min_box_iou),
            "gap_min_mask_iou": float(args.gap_min_mask_iou),
            "gap_min_mask_containment": float(args.gap_min_mask_containment),
            "gap_max_center_dist": float(args.gap_max_center_dist),
            "pair_duplicate_mask_iou": float(args.pair_duplicate_mask_iou),
            "pair_duplicate_mask_containment": float(args.pair_duplicate_mask_containment),
            "pair_conflict_mask_iou": float(args.pair_conflict_mask_iou),
            "pair_conflict_mask_containment": float(args.pair_conflict_mask_containment),
            "pair_low_conflict_max_ratio": float(args.pair_low_conflict_max_ratio),
        },
        "preview": {
            "adjacent": adjacent_rows[:20],
            "same_frame": same_frame_rows[:20],
            "gap": gap_rows[:20],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
