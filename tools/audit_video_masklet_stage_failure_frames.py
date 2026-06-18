#!/usr/bin/env python3
"""Render stage-wise evidence for selected Video Masklet frames.

This is an audit tool. It does not run detection, tracking, SAM, or merging.
It reads cached filtered proposals, review-package tracklet metadata, cached
SAM masks, and final sparse output to localize where a visible object vanished.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_video_masklet_front_end import _unpack_mask_np, get_colour  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit stage-wise frame failures.")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--proposals_pt", required=True)
    parser.add_argument("--review_tar", required=True)
    parser.add_argument("--sam3_pt", required=True)
    parser.add_argument("--sparse_pt", required=True)
    parser.add_argument("--prompt_probe_csv", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_alpha", type=float, default=0.45)
    return parser.parse_args()


def _parse_frames(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def _load_image(frames_dir: Path, frame_idx: int) -> np.ndarray:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = frames_dir / f"{frame_idx:06d}{suffix}"
        if path.exists():
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f"Frame {frame_idx} not found in {frames_dir}")


def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _to_box(raw: Any) -> Optional[np.ndarray]:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4].copy()


def _draw_items(
    image: np.ndarray,
    items: Sequence[Dict[str, Any]],
    title: str,
    H: int,
    W: int,
    mask_alpha: float,
) -> np.ndarray:
    out = image.copy()
    if out.shape[:2] != (H, W):
        out = cv2.resize(out, (W, H))
    for idx, item in enumerate(items):
        colour = get_colour(int(item.get("colour_idx", idx)))
        mask = item.get("mask")
        if mask is not None:
            mask_bool = np.asarray(mask).astype(bool)
            if mask_bool.shape[:2] == (H, W) and mask_bool.any():
                colour_np = np.asarray(colour, dtype=np.float32)
                out[mask_bool] = (
                    out[mask_bool].astype(np.float32) * (1.0 - mask_alpha)
                    + colour_np * mask_alpha
                ).astype(np.uint8)
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, colour, 2)
        box = _to_box(item.get("box"))
        if box is None and mask is not None:
            box = _mask_to_box(np.asarray(mask).astype(bool))
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = str(item.get("text", ""))
        if text:
            cv2.rectangle(out, (x1, max(0, y1 - 16)), (min(W - 1, x1 + max(80, 7 * len(text))), y1), (0, 0, 0), -1)
            cv2.putText(out, text, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _read_review_tracklets(review_tar: Path) -> List[Dict[str, Any]]:
    with tarfile.open(review_tar, "r:gz") as tar:
        member = tar.getmember("review_package/debug/proposal_tracklets.jsonl")
        f = tar.extractfile(member)
        if f is None:
            raise RuntimeError("proposal_tracklets.jsonl missing in review package")
        rows = []
        for raw in f:
            line = raw.decode("utf-8").strip()
            if line:
                rows.append(json.loads(line))
        return rows


def _load_prompt_rows(path: str, frames: set[int]) -> List[Dict[str, Any]]:
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["frame_idx"]) in frames:
                rows.append(row)
    return rows


def _proposal_rows(proposals: Sequence[Dict[str, Any]], frames: set[int]) -> List[Dict[str, Any]]:
    rows = []
    for p in proposals:
        frame_idx = int(p["frame_idx"])
        if frame_idx not in frames:
            continue
        box = _to_box(p.get("box"))
        rows.append(
            {
                "stage": "filtered_proposal",
                "frame_idx": frame_idx,
                "proposal_id": int(p["proposal_id"]),
                "tracklet_id": "",
                "final_track_index": "",
                "label": str(p.get("label", "")),
                "raw_label": str(p.get("raw_label", "")),
                "confidence": float(p.get("confidence", 0.0)),
                "area_ratio": float(p.get("area_ratio", 0.0)),
                "box": json.dumps([float(v) for v in box]) if box is not None else "",
                "state": "",
                "sam3_status": "",
            }
        )
    return rows


def _final_frame_items(sparse: Dict[str, Any], frame_idx: int, H: int, W: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, track in enumerate(sparse.get("tracks", [])):
        frames = [int(v) for v in track.get("frames", [])]
        if frame_idx not in frames:
            continue
        pos = frames.index(frame_idx)
        packed = np.asarray(track.get("packed_masks", [])[pos], dtype=np.uint8)
        mask = _unpack_mask_np(packed, H, W).astype(bool)
        box = track.get("boxes", [None])[pos]
        label = str(track.get("L_sem", ""))
        items.append(
            {
                "colour_idx": idx,
                "mask": mask,
                "box": box,
                "text": f"F{idx} {label}",
                "label": label,
                "track_index": idx,
            }
        )
    return items


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _parse_frames(args.frames)
    frame_set = set(frames)

    proposals_payload = torch.load(args.proposals_pt, map_location="cpu", weights_only=False)
    proposals = list(proposals_payload.get("proposals", []))
    tracklets = _read_review_tracklets(Path(args.review_tar))
    sam3_payload = torch.load(args.sam3_pt, map_location="cpu", weights_only=False)
    sparse = torch.load(args.sparse_pt, map_location="cpu", weights_only=False)
    H = int(sparse["frame_height"])
    W = int(sparse["frame_width"])

    prop_to_tracklet: Dict[int, Dict[str, Any]] = {}
    for tr in tracklets:
        for pid in tr.get("proposal_ids", []):
            prop_to_tracklet[int(pid)] = tr
    sam3_by_tid = {int(row["tracklet_id"]): row for row in sam3_payload.get("tracklets", [])}

    rows = _proposal_rows(proposals, frame_set)
    for row in rows:
        tr = prop_to_tracklet.get(int(row["proposal_id"]))
        if tr is None:
            row["state"] = "no_tracklet"
            continue
        tid = int(tr["proposal_tracklet_id"])
        row["tracklet_id"] = tid
        row["state"] = str(tr.get("state", ""))
        row["final_track_index"] = tr.get("output_track_index", "")
        sam3 = sam3_by_tid.get(tid, {})
        if bool(sam3.get("sam3_attempted", False)):
            row["sam3_status"] = "success" if bool(sam3.get("sam3_success", False)) else f"failed:{sam3.get('sam3_failure_reason')}"
        else:
            row["sam3_status"] = "not_attempted"

    prompt_rows = _load_prompt_rows(args.prompt_probe_csv, frame_set)
    with (out_dir / "stage_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "stage",
            "frame_idx",
            "proposal_id",
            "tracklet_id",
            "final_track_index",
            "label",
            "raw_label",
            "confidence",
            "area_ratio",
            "box",
            "state",
            "sam3_status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "prompt_probe_rows.csv").open("w", encoding="utf-8", newline="") as f:
        if prompt_rows:
            writer = csv.DictWriter(f, fieldnames=list(prompt_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prompt_rows)

    for frame_idx in frames:
        image = _load_image(Path(args.frames_dir), frame_idx)
        frame_props = [p for p in proposals if int(p["frame_idx"]) == frame_idx]
        proposal_items = []
        for p in frame_props:
            proposal_items.append(
                {
                    "colour_idx": int(p["proposal_id"]),
                    "mask": np.asarray(p.get("mask"), dtype=np.uint8).astype(bool),
                    "box": p.get("box"),
                    "text": f"P{int(p['proposal_id'])} {p.get('label')} {float(p.get('confidence', 0.0)):.2f}",
                }
            )

        tracklet_items = []
        for p in frame_props:
            tr = prop_to_tracklet.get(int(p["proposal_id"]))
            if tr is None:
                continue
            tid = int(tr["proposal_tracklet_id"])
            tracklet_items.append(
                {
                    "colour_idx": tid,
                    "mask": np.asarray(p.get("mask"), dtype=np.uint8).astype(bool),
                    "box": p.get("box"),
                    "text": f"T{tid} {tr.get('label')} {tr.get('state')} ->F{tr.get('output_track_index')}",
                }
            )

        sam3_items = []
        for row in sam3_payload.get("tracklets", []):
            tid = int(row["tracklet_id"])
            packed = row.get("packed_masks", {}).get(frame_idx)
            if packed is None:
                packed = row.get("packed_masks", {}).get(str(frame_idx))
            if packed is None:
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
            tr = next((item for item in tracklets if int(item["proposal_tracklet_id"]) == tid), {})
            sam3_items.append(
                {
                    "colour_idx": tid,
                    "mask": mask,
                    "box": _mask_to_box(mask),
                    "text": f"S{tid} {tr.get('label', '')}",
                }
            )

        final_items = _final_frame_items(sparse, frame_idx, H, W)
        views = {
            "00_raw": _draw_items(image, [], f"raw f{frame_idx}", H, W, float(args.mask_alpha)),
            "01_filtered_proposals": _draw_items(image, proposal_items, f"filtered proposals f{frame_idx}", H, W, float(args.mask_alpha)),
            "02_tracklets": _draw_items(image, tracklet_items, f"merged tracklets f{frame_idx}", H, W, float(args.mask_alpha)),
            "03_sam3": _draw_items(image, sam3_items, f"SAM3 refined f{frame_idx}", H, W, float(args.mask_alpha)),
            "04_final": _draw_items(image, final_items, f"final sparse f{frame_idx}", H, W, float(args.mask_alpha)),
        }
        panels = list(views.values())
        sheet = np.zeros((len(panels) * H, W, 3), dtype=np.uint8)
        for idx, panel in enumerate(panels):
            sheet[idx * H : (idx + 1) * H] = panel
        for name, panel in views.items():
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_{name}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_stage_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    summary = {
        "frames": frames,
        "proposals_pt": str(args.proposals_pt),
        "review_tar": str(args.review_tar),
        "sam3_pt": str(args.sam3_pt),
        "sparse_pt": str(args.sparse_pt),
        "filtered_proposal_rows": int(len(rows)),
        "prompt_probe_rows": int(len(prompt_rows)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
