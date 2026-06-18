#!/usr/bin/env python3
"""Audit whether proposal tracklet evidence survives into final sparse tracks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import load_sparse  # noqa: E402


VEHICLE_ALIASES = {"car", "van", "bus", "truck", "vehicle", "trailer"}
SIGN_ALIASES = {"traffic sign", "road sign", "sign board", "signboard", "billboard", "sign"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True, help="Stage-C cache root for one experiment.")
    parser.add_argument("--sparse_pt", required=True, help="Final sparse masklets .pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels", default="car,traffic sign")
    parser.add_argument("--states", default="confirmed,tentative")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--min_conf", type=float, default=0.0)
    parser.add_argument("--min_box_iou", type=float, default=0.05)
    parser.add_argument("--max_center_dist", type=float, default=0.80)
    parser.add_argument("--center_scale_floor", type=float, default=24.0)
    parser.add_argument("--same_label_only", type=int, default=1)
    return parser.parse_args()


def parse_set(raw: str) -> Optional[set[str]]:
    text = str(raw).strip()
    if not text or text.lower() in {"all", "*", "off"}:
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def canonical_label(label: str) -> str:
    text = str(label).strip().lower()
    if text in VEHICLE_ALIASES:
        return "car"
    if text in SIGN_ALIASES:
        return "traffic sign"
    return text


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
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


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


def center_dist_norm(left: np.ndarray, right: np.ndarray, scale_floor: float) -> float:
    left_w = max(1.0, float(left[2] - left[0]))
    left_h = max(1.0, float(left[3] - left[1]))
    right_w = max(1.0, float(right[2] - right[0]))
    right_h = max(1.0, float(right[3] - right[1]))
    scale = max(float(scale_floor), left_w, left_h, right_w, right_h)
    return float(np.linalg.norm(center(left) - center(right)) / max(scale, 1e-6))


def get_mapping_value(mapping: Dict[Any, Any], frame_idx: int) -> Any:
    if int(frame_idx) in mapping:
        return mapping[int(frame_idx)]
    return mapping.get(str(int(frame_idx)))


def load_proposals(cache_root: Path) -> Dict[int, Dict[str, Any]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    for path in sorted((cache_root / "chunks").glob("*/thing_proposals.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for prop in payload.get("proposals", []):
            prop_id = int(prop["proposal_id"])
            if prop_id in by_id:
                raise RuntimeError(f"duplicate proposal_id {prop_id} in {path}")
            rec = dict(prop)
            rec["chunk_path"] = str(path)
            by_id[prop_id] = rec
    return by_id


def load_tracklets(cache_root: Path) -> List[Dict[str, Any]]:
    path = cache_root / "final" / "proposal_tracklets.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload.get("tracklets", []))


def sparse_frame_boxes(sparse_pt: Path, labels: Optional[set[str]]) -> Tuple[int, Dict[int, List[Dict[str, Any]]]]:
    sparse = load_sparse(sparse_pt)
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for track_index, track in enumerate(sparse.tracks):
        label = canonical_label(track.get("L_sem", ""))
        if labels is not None and label not in labels:
            continue
        boxes = track.get("box_by_frame", {})
        for raw_frame, raw_box in boxes.items():
            frame_idx = int(raw_frame)
            box = box_array(raw_box)
            if box is None:
                continue
            by_frame.setdefault(frame_idx, []).append(
                {
                    "track_index": int(track_index),
                    "label": label,
                    "box": box,
                    "source_type": str(track.get("source_type", "")),
                }
            )
    return int(sparse.num_frames), by_frame


def frame_range_ok(frame_idx: int, start: int, end: int) -> bool:
    if frame_idx < int(start):
        return False
    return int(end) < 0 or frame_idx <= int(end)


def match_final(
    prop_label: str,
    prop_box: np.ndarray,
    final_boxes: List[Dict[str, Any]],
    same_label_only: bool,
    min_box_iou: float,
    max_center_dist: float,
    center_scale_floor: float,
) -> Tuple[bool, Dict[str, Any]]:
    best: Dict[str, Any] = {
        "track_index": "",
        "label": "",
        "box_iou": 0.0,
        "center_dist": 999.0,
    }
    for item in final_boxes:
        if same_label_only and str(item["label"]) != str(prop_label):
            continue
        iou = box_iou(prop_box, item["box"])
        cdist = center_dist_norm(prop_box, item["box"], center_scale_floor)
        better = (iou, -cdist) > (float(best["box_iou"]), -float(best["center_dist"]))
        if better:
            best = {
                "track_index": int(item["track_index"]),
                "label": str(item["label"]),
                "box_iou": float(iou),
                "center_dist": float(cdist),
            }
    ok = float(best["box_iou"]) >= float(min_box_iou) or float(best["center_dist"]) <= float(max_center_dist)
    return ok, best


def fmt_box(box: np.ndarray) -> str:
    return json.dumps([round(float(x), 3) for x in box.tolist()])


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_set(args.labels)
    labels = {canonical_label(x) for x in labels} if labels is not None else None
    states = parse_set(args.states)

    proposals = load_proposals(cache_root)
    tracklets = load_tracklets(cache_root)
    num_frames, final_by_frame = sparse_frame_boxes(Path(args.sparse_pt), labels)

    missing_rows: List[Dict[str, Any]] = []
    frame_rows: Dict[Tuple[int, str], Dict[str, int]] = {}
    tracklet_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "cache_root": str(cache_root),
        "sparse_pt": str(args.sparse_pt),
        "labels": sorted(labels) if labels is not None else "all",
        "states": sorted(states) if states is not None else "all",
        "num_frames": int(num_frames),
        "proposal_frames": 0,
        "covered_frames": 0,
        "missing_frames": 0,
        "per_label": {},
    }

    for tr in tracklets:
        label = canonical_label(tr.get("label", ""))
        state = str(tr.get("state", ""))
        if labels is not None and label not in labels:
            continue
        if states is not None and state not in states:
            continue
        prop_ids = [int(x) for x in tr.get("proposal_ids", []) if int(x) in proposals]
        frames: List[int] = []
        missing_count = 0
        covered_count = 0
        for prop_id in prop_ids:
            prop = proposals[prop_id]
            frame_idx = int(prop["frame_idx"])
            if not frame_range_ok(frame_idx, int(args.start), int(args.end)):
                continue
            if float(prop.get("confidence", 0.0)) < float(args.min_conf):
                continue
            prop_label = canonical_label(prop.get("label", label))
            if labels is not None and prop_label not in labels:
                continue
            prop_box = box_array(prop.get("box"))
            if prop_box is None:
                continue
            frames.append(frame_idx)
            final_boxes = final_by_frame.get(frame_idx, [])
            final_count = sum(1 for item in final_boxes if item["label"] == prop_label)
            key = (frame_idx, prop_label)
            row = frame_rows.setdefault(
                key,
                {"frame": frame_idx, "label": prop_label, "proposal_count": 0, "covered_count": 0, "missing_count": 0, "final_count": final_count},
            )
            row["proposal_count"] += 1
            ok, best = match_final(
                prop_label,
                prop_box,
                final_boxes,
                bool(int(args.same_label_only)),
                float(args.min_box_iou),
                float(args.max_center_dist),
                float(args.center_scale_floor),
            )
            label_summary = summary["per_label"].setdefault(
                prop_label,
                {"proposal_frames": 0, "covered_frames": 0, "missing_frames": 0},
            )
            summary["proposal_frames"] += 1
            label_summary["proposal_frames"] += 1
            if ok:
                covered_count += 1
                row["covered_count"] += 1
                summary["covered_frames"] += 1
                label_summary["covered_frames"] += 1
            else:
                missing_count += 1
                row["missing_count"] += 1
                summary["missing_frames"] += 1
                label_summary["missing_frames"] += 1
                missing_rows.append(
                    {
                        "frame": frame_idx,
                        "label": prop_label,
                        "proposal_tracklet_id": int(tr.get("tracklet_id", -1)),
                        "state": state,
                        "proposal_id": prop_id,
                        "confidence": float(prop.get("confidence", 0.0)),
                        "proposal_box": fmt_box(prop_box),
                        "best_final_track": best["track_index"],
                        "best_final_label": best["label"],
                        "best_box_iou": float(best["box_iou"]),
                        "best_center_dist": float(best["center_dist"]),
                        "final_same_label_count": int(final_count),
                    }
                )
        if frames:
            tracklet_rows.append(
                {
                    "proposal_tracklet_id": int(tr.get("tracklet_id", -1)),
                    "label": label,
                    "state": state,
                    "span_start": int(min(frames)),
                    "span_end": int(max(frames)),
                    "proposal_frames": int(len(frames)),
                    "covered_frames": int(covered_count),
                    "missing_frames": int(missing_count),
                }
            )

    for value in summary["per_label"].values():
        value["missing_ratio"] = float(value["missing_frames"] / max(1, value["proposal_frames"]))
    summary["missing_ratio"] = float(summary["missing_frames"] / max(1, summary["proposal_frames"]))

    def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    write_csv(
        output_dir / "missing_proposal_frames.csv",
        missing_rows,
        [
            "frame",
            "label",
            "proposal_tracklet_id",
            "state",
            "proposal_id",
            "confidence",
            "proposal_box",
            "best_final_track",
            "best_final_label",
            "best_box_iou",
            "best_center_dist",
            "final_same_label_count",
        ],
    )
    write_csv(
        output_dir / "per_frame_counts.csv",
        [frame_rows[k] for k in sorted(frame_rows)],
        ["frame", "label", "proposal_count", "covered_count", "missing_count", "final_count"],
    )
    write_csv(
        output_dir / "proposal_tracklet_coverage.csv",
        sorted(tracklet_rows, key=lambda x: (str(x["label"]), int(x["span_start"]), int(x["proposal_tracklet_id"]))),
        ["proposal_tracklet_id", "label", "state", "span_start", "span_end", "proposal_frames", "covered_frames", "missing_frames"],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
