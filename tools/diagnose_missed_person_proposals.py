#!/usr/bin/env python3
"""Audit high-confidence person proposals that are missing from final masklets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import load_sparse  # noqa: E402
from run_video_masklet_front_end import _unpack_mask_np, collect_image_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--proposal_chunks_root", required=True)
    parser.add_argument("--proposal_tracklets_pt", default="")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--overlay_video", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--output_jpg", required=True)
    parser.add_argument("--label", default="person")
    parser.add_argument("--min_conf", type=float, default=0.60)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--max_area_ratio", type=float, default=0.80)
    parser.add_argument("--max_coverage", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=24)
    parser.add_argument("--dedupe_same_tracklet", type=int, default=1)
    return parser.parse_args()


def _load_proposals(chunks_root: Path, label: str) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    for path in sorted(chunks_root.glob("chunk_*/thing_proposals.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for rec in payload.get("proposals", []):
            if str(rec.get("label", "")).lower() != label.lower():
                continue
            item = dict(rec)
            item["_chunk_path"] = str(path)
            proposals.append(item)
    return proposals


def _proposal_to_tracklet(path: Path) -> Dict[int, int]:
    if not path:
        return {}
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mapping: Dict[int, int] = {}
    for rec in payload.get("tracklets", []):
        tid = int(rec.get("tracklet_id", -1))
        for pid in rec.get("proposal_ids", []):
            mapping[int(pid)] = tid
    return mapping


def _as_mask(value: Any, height: int, width: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    mask = np.asarray(value)
    if mask.shape != (height, width):
        return None
    return mask.astype(bool)


def _as_box(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4].copy()


def _box_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


def _person_tracks_by_frame(tracks: List[Dict[str, Any]], label: str) -> Dict[int, List[int]]:
    by_frame: Dict[int, List[int]] = defaultdict(list)
    for idx, track in enumerate(tracks):
        if str(track.get("L_sem", "")).lower() != label.lower():
            continue
        if "thing" not in str(track.get("source_type", "")).lower():
            continue
        for frame_idx in track.get("mask_by_frame", {}).keys():
            by_frame[int(frame_idx)].append(idx)
    return by_frame


def _frame_union_and_best(
    sparse: Any,
    by_frame: Dict[int, List[int]],
    frame_idx: int,
    proposal_mask: np.ndarray,
    proposal_box: Optional[np.ndarray],
) -> Tuple[float, float, int, int]:
    height, width = int(sparse.frame_height), int(sparse.frame_width)
    union = np.zeros((height, width), dtype=bool)
    best_iou = 0.0
    best_idx = -1
    for idx in by_frame.get(int(frame_idx), []):
        track = sparse.tracks[idx]
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            continue
        mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), height, width).astype(bool)
        union |= mask
        box_iou = _box_iou(proposal_box, _as_box(track.get("box_by_frame", {}).get(int(frame_idx))))
        if box_iou > best_iou:
            best_iou = box_iou
            best_idx = idx
    area = int(proposal_mask.sum())
    if area <= 0:
        return 0.0, best_iou, best_idx, 0
    coverage = float(np.logical_and(proposal_mask, union).sum() / float(area))
    return coverage, best_iou, best_idx, int(union.sum())


def find_misses(args: argparse.Namespace) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    sparse = load_sparse(Path(args.input_pt))
    height, width = int(sparse.frame_height), int(sparse.frame_width)
    proposals = _load_proposals(Path(args.proposal_chunks_root), str(args.label))
    proposal_to_tracklet = _proposal_to_tracklet(Path(args.proposal_tracklets_pt)) if args.proposal_tracklets_pt else {}
    by_frame = _person_tracks_by_frame(sparse.tracks, str(args.label))

    misses: List[Dict[str, Any]] = []
    total_considered = 0
    for rec in proposals:
        conf = float(rec.get("confidence", 0.0))
        area_ratio = float(rec.get("area_ratio", 0.0))
        if conf < float(args.min_conf):
            continue
        if area_ratio < float(args.min_area_ratio) or area_ratio > float(args.max_area_ratio):
            continue
        mask = _as_mask(rec.get("mask"), height, width)
        if mask is None:
            continue
        box = _as_box(rec.get("box"))
        total_considered += 1
        frame_idx = int(rec.get("frame_idx", -1))
        coverage, best_box_iou, best_track, output_union_area = _frame_union_and_best(sparse, by_frame, frame_idx, mask, box)
        if coverage > float(args.max_coverage):
            continue
        score = float(conf * area_ratio * (1.0 - coverage))
        misses.append(
            {
                "score": score,
                "proposal_id": int(rec.get("proposal_id", -1)),
                "proposal_tracklet_id": int(proposal_to_tracklet.get(int(rec.get("proposal_id", -1)), -1)),
                "frame_idx": frame_idx,
                "label": str(rec.get("label", "")),
                "raw_label": str(rec.get("raw_label", "")),
                "confidence": conf,
                "area_ratio": area_ratio,
                "coverage_by_output_person": coverage,
                "best_output_box_iou": best_box_iou,
                "best_output_track_index": int(best_track),
                "proposal_area": int(mask.sum()),
                "output_person_union_area": int(output_union_area),
                "box": [] if box is None else [float(x) for x in box.tolist()],
                "chunk_path": str(rec.get("_chunk_path", "")),
            }
        )

    misses.sort(key=lambda row: (row["score"], row["confidence"], row["area_ratio"]), reverse=True)
    if int(args.dedupe_same_tracklet):
        deduped: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for row in misses:
            tid = int(row.get("proposal_tracklet_id", -1))
            if tid >= 0 and tid in seen:
                continue
            if tid >= 0:
                seen.add(tid)
            deduped.append(row)
        misses_for_top = deduped
    else:
        misses_for_top = misses

    summary = {
        "input_pt": str(args.input_pt),
        "proposal_chunks_root": str(args.proposal_chunks_root),
        "proposal_tracklets_pt": str(args.proposal_tracklets_pt),
        "label": str(args.label),
        "min_conf": float(args.min_conf),
        "min_area_ratio": float(args.min_area_ratio),
        "max_area_ratio": float(args.max_area_ratio),
        "max_coverage": float(args.max_coverage),
        "total_proposals": int(len(proposals)),
        "total_considered": int(total_considered),
        "miss_count": int(len(misses)),
        "miss_count_deduped_tracklet": int(len(misses_for_top)),
        "misses_by_tracklet_preview": Counter(int(row.get("proposal_tracklet_id", -1)) for row in misses).most_common(20),
    }
    return sparse, misses_for_top, summary


def _draw_mask_and_box(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    box: Optional[np.ndarray],
    sparse_height: int,
    sparse_width: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    if mask is not None:
        resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        color_arr = np.asarray(color, dtype=np.uint8)
        frame[resized] = (frame[resized] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    if box is not None:
        sx = frame.shape[1] / float(sparse_width)
        sy = frame.shape[0] / float(sparse_height)
        p1 = (int(round(float(box[0]) * sx)), int(round(float(box[1]) * sy)))
        p2 = (int(round(float(box[2]) * sx)), int(round(float(box[3]) * sy)))
        cv2.rectangle(frame, p1, p2, color, 2)


class _FrameReader:
    def __init__(self, input_path: str) -> None:
        self.input_path = str(input_path)
        self.image_paths: List[str] = []
        self.temp_dir: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        path = Path(input_path)
        if path.is_dir():
            self.image_paths, self.temp_dir = collect_image_paths(str(path), 0, -1, 1)
        else:
            self.cap = cv2.VideoCapture(str(input_path))
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None

    def is_opened(self) -> bool:
        return bool(self.image_paths) or self.cap is not None

    def read(self, frame_idx: int) -> Tuple[bool, Optional[np.ndarray]]:
        if self.image_paths:
            if frame_idx < 0 or frame_idx >= len(self.image_paths):
                return False, None
            frame = cv2.imread(self.image_paths[frame_idx], cv2.IMREAD_COLOR)
            return frame is not None, frame
        if self.cap is None:
            return False, None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


def write_outputs(args: argparse.Namespace, sparse: Any, misses: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "misses": misses[: int(args.top_k)]}, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_txt.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
        for row in misses[: int(args.top_k)]:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    cap = cv2.VideoCapture(str(args.overlay_video)) if args.overlay_video else None
    if cap is not None and not cap.isOpened():
        cap.release()
        cap = None
    raw_reader = _FrameReader(str(args.input_video))
    if not raw_reader.is_opened():
        raise RuntimeError(f"cannot open input frames: {args.input_video}")

    proposals_by_id = {int(rec.get("proposal_id", -1)): rec for rec in _load_proposals(Path(args.proposal_chunks_root), str(args.label))}
    sparse_height, sparse_width = int(sparse.frame_height), int(sparse.frame_width)
    tiles: List[np.ndarray] = []
    for rank, row in enumerate(misses[: int(args.top_k)], 1):
        frame_idx = int(row["frame_idx"])
        proposal_id = int(row["proposal_id"])
        rec = proposals_by_id.get(proposal_id)
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok_overlay, overlay = cap.read()
        else:
            ok_overlay, overlay = False, None
        ok_raw, raw = raw_reader.read(frame_idx)
        if not ok_overlay:
            overlay = raw.copy() if ok_raw and raw is not None else np.zeros((max(1, sparse_height), max(1, sparse_width), 3), dtype=np.uint8)
        if not ok_raw:
            raw = np.zeros_like(overlay)
        raw = cv2.resize(raw, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = _as_mask(rec.get("mask"), sparse_height, sparse_width) if rec is not None else None
        box = _as_box(rec.get("box")) if rec is not None else None
        _draw_mask_and_box(raw, mask, box, sparse_height, sparse_width, (0, 0, 255), 0.45)
        _draw_mask_and_box(overlay, mask, box, sparse_height, sparse_width, (0, 0, 255), 0.35)
        panel = np.concatenate([raw, overlay], axis=1)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 72), (0, 0, 0), -1)
        text1 = f"#{rank} f={frame_idx} pid={proposal_id} tid={row['proposal_tracklet_id']} conf={row['confidence']:.3f}"
        text2 = f"coverage={row['coverage_by_output_person']:.3f} best_box_iou={row['best_output_box_iou']:.3f} area={row['area_ratio']:.4f}"
        for y, text in [(24, text1), (52, text2)]:
            cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panel = cv2.resize(panel, (720, 220), interpolation=cv2.INTER_AREA)
        tiles.append(panel)
    if cap is not None:
        cap.release()
    raw_reader.release()

    if not tiles:
        grid = np.zeros((220, 720, 3), dtype=np.uint8)
        cv2.putText(grid, "no missed proposals", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cols = 2
        rows = int(math.ceil(len(tiles) / cols))
        blank = np.zeros_like(tiles[0])
        while len(tiles) < rows * cols:
            tiles.append(blank.copy())
        grid = np.concatenate(
            [np.concatenate(tiles[row * cols : (row + 1) * cols], axis=1) for row in range(rows)],
            axis=0,
        )
    output_jpg = Path(args.output_jpg)
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_jpg), grid)


def main() -> None:
    args = parse_args()
    sparse, misses, summary = find_misses(args)
    write_outputs(args, sparse, misses, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"output_json={args.output_json}")
    print(f"output_txt={args.output_txt}")
    print(f"output_jpg={args.output_jpg}")
    for row in misses[: int(args.top_k)]:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
