#!/usr/bin/env python3
"""Visualize full and chunked top-level semantic_segmentation label maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch


PALETTE: Dict[str, Tuple[int, int, int]] = {
    "void": (0, 0, 0),
    "road": (104, 104, 104),
    "sky": (96, 180, 238),
    "grass": (74, 162, 74),
    "tree": (32, 120, 76),
    "vegetation": (42, 138, 82),
    "fence": (56, 100, 176),
    "guardrail": (72, 112, 184),
    "pole": (220, 188, 74),
    "traffic sign": (245, 214, 58),
    "car": (220, 74, 74),
    "truck": (200, 62, 62),
    "person": (235, 120, 60),
    "building": (160, 126, 192),
    "wall": (156, 156, 156),
    "sidewalk": (176, 176, 116),
    "terrain": (130, 108, 72),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="04")
    parser.add_argument("--image_dir", default="data/kitti/dataset/sequences/04/image_2")
    parser.add_argument("--full_pt", default="results/kitti_preprocess/04/sparse_masklets_with_semantic.pt")
    parser.add_argument("--cache_dir", default="results/kitti_preprocess/04/stage_c_cache_semantic_chunks")
    parser.add_argument("--output_dir", default="results/kitti_preprocess/04/semantic_segmentation_visual_audit")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--contact_max_rows", type=int, default=10)
    return parser.parse_args()


def _stable_colour(label: str) -> Tuple[int, int, int]:
    label = str(label)
    if label in PALETTE:
        return PALETTE[label]
    value = 2166136261
    for byte in label.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    hue = value % 180
    hsv = np.array([[[hue, 150, 230]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _colour_table(label_names: Sequence[str]) -> np.ndarray:
    colours = np.zeros((max(len(label_names), 1), 3), dtype=np.uint8)
    for idx, label in enumerate(label_names):
        colours[idx] = np.asarray(_stable_colour(str(label)), dtype=np.uint8)
    return colours


def _colourize(label_map: torch.Tensor, colours: np.ndarray) -> np.ndarray:
    arr = label_map.detach().cpu().numpy().astype(np.int64, copy=False)
    safe = np.clip(arr, 0, len(colours) - 1)
    return colours[safe]


def _read_rgb(image_dir: Path, frame_idx: int, height: int, width: int) -> np.ndarray:
    path = image_dir / f"{int(frame_idx):06d}.png"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def _overlay(rgb: np.ndarray, semantic_rgb: np.ndarray, alpha: float) -> np.ndarray:
    mask = semantic_rgb.sum(axis=2) > 0
    out = rgb.copy()
    out[mask] = (
        out[mask].astype(np.float32) * (1.0 - float(alpha))
        + semantic_rgb[mask].astype(np.float32) * float(alpha)
    ).astype(np.uint8)
    return out


def _put_small_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 250), 17), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _diff_image(full_map: torch.Tensor, chunk_map: torch.Tensor) -> np.ndarray:
    diff = (full_map.detach().cpu().numpy() != chunk_map.detach().cpu().numpy()).astype(np.uint8)
    out = np.zeros((*diff.shape, 3), dtype=np.uint8)
    out[diff > 0] = np.array([255, 0, 255], dtype=np.uint8)
    return out


def _load_index(cache_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in (cache_dir / "cache_index.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _select_samples(rows: Sequence[Dict[str, Any]], max_rows: int) -> List[Tuple[Dict[str, Any], int]]:
    if not rows:
        return []
    selected_indices = sorted({0, 1 if len(rows) > 1 else 0, len(rows) // 2, max(0, len(rows) - 2), len(rows) - 1})
    samples: List[Tuple[Dict[str, Any], int]] = []
    for idx in selected_indices:
        row = rows[idx]
        start, end = int(row["start_frame"]), int(row["end_frame"])
        frames = [start, max(start, min(end - 1, (start + end) // 2)), end - 1]
        for frame in frames:
            item = (row, int(frame))
            if item not in samples:
                samples.append(item)
    return samples[: int(max_rows)]


def _load_chunk_map(cache_dir: Path, row: Dict[str, Any]) -> torch.Tensor:
    payload = torch.load(cache_dir / row["chunk"] / "masklet.pt", map_location="cpu", weights_only=False)
    return payload["semantic_segmentation"]["label_maps"]


def _write_contact_sheet(
    samples: Sequence[Tuple[Dict[str, Any], int]],
    *,
    image_dir: Path,
    full_maps: torch.Tensor,
    cache_dir: Path,
    colours: np.ndarray,
    height: int,
    width: int,
    alpha: float,
    out_path: Path,
) -> List[Dict[str, Any]]:
    rows_img: List[np.ndarray] = []
    rows_report: List[Dict[str, Any]] = []
    chunk_cache: Dict[str, torch.Tensor] = {}
    headers = ["rgb", "full semantic", "chunk semantic", "full overlay", "diff"]
    header_img = np.zeros((22, width * len(headers), 3), dtype=np.uint8)
    for i, title in enumerate(headers):
        cv2.putText(header_img, title, (i * width + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    rows_img.append(header_img)
    for row, frame_idx in samples:
        chunk_name = str(row["chunk"])
        start = int(row["start_frame"])
        if chunk_name not in chunk_cache:
            chunk_cache[chunk_name] = _load_chunk_map(cache_dir, row)
        chunk_maps = chunk_cache[chunk_name]
        local_idx = int(frame_idx) - start
        full_map = full_maps[int(frame_idx)]
        chunk_map = chunk_maps[local_idx]
        full_rgb = _colourize(full_map, colours)
        chunk_rgb = _colourize(chunk_map, colours)
        rgb = _read_rgb(image_dir, int(frame_idx), height, width)
        diff = _diff_image(full_map, chunk_map)
        diff_pixels = int((full_map != chunk_map).sum().item())
        label = f"seq f{int(frame_idx):06d} {chunk_name}"
        panels = [
            _put_small_label(rgb, label),
            _put_small_label(full_rgb, label),
            _put_small_label(chunk_rgb, label),
            _put_small_label(_overlay(rgb, full_rgb, alpha), label),
            _put_small_label(diff, f"diff_pixels={diff_pixels}"),
        ]
        rows_img.append(np.concatenate(panels, axis=1))
        rows_report.append(
            {
                "frame": int(frame_idx),
                "chunk": chunk_name,
                "chunk_start": int(row["start_frame"]),
                "chunk_end": int(row["end_frame"]),
                "local_frame": int(local_idx),
                "diff_pixels": int(diff_pixels),
            }
        )
    contact = np.concatenate(rows_img, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    return rows_report


def _write_video(
    out_path: Path,
    *,
    image_dir: Path,
    maps: torch.Tensor,
    colours: np.ndarray,
    height: int,
    width: int,
    alpha: float,
    fps: int,
) -> None:
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
    for frame_idx in range(int(maps.shape[0])):
        rgb = _read_rgb(image_dir, frame_idx, height, width)
        sem_rgb = _colourize(maps[frame_idx], colours)
        frame = _overlay(rgb, sem_rgb, alpha)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _assemble_chunk_video_maps(rows: Sequence[Dict[str, Any]], cache_dir: Path, num_frames: int, height: int, width: int) -> torch.Tensor:
    out = torch.zeros((num_frames, height, width), dtype=torch.uint8)
    filled = torch.zeros((num_frames,), dtype=torch.bool)
    for row in rows:
        start, end = int(row["start_frame"]), int(row["end_frame"])
        chunk_maps = _load_chunk_map(cache_dir, row)
        for local_idx, frame_idx in enumerate(range(start, end)):
            if not bool(filled[frame_idx]):
                out[frame_idx] = chunk_maps[local_idx]
                filled[frame_idx] = True
    if not bool(filled.all().item()):
        missing = torch.where(~filled)[0].tolist()
        raise RuntimeError(f"Missing chunk semantic frames: {missing[:20]}")
    return out


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    full_pt = Path(args.full_pt)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(full_pt, map_location="cpu", weights_only=False)
    sem = payload["semantic_segmentation"]
    full_maps = sem["label_maps"].detach().cpu()
    height = int(payload["frame_height"])
    width = int(payload["frame_width"])
    label_names = list(sem.get("label_names", []))
    colours = _colour_table(label_names)
    rows = _load_index(cache_dir)
    samples = _select_samples(rows, int(args.contact_max_rows))

    contact_path = output_dir / f"seq{args.sequence}_semantic_full_vs_chunk_contact.png"
    sample_report = _write_contact_sheet(
        samples,
        image_dir=image_dir,
        full_maps=full_maps,
        cache_dir=cache_dir,
        colours=colours,
        height=height,
        width=width,
        alpha=float(args.alpha),
        out_path=contact_path,
    )

    chunk_video_maps = _assemble_chunk_video_maps(rows, cache_dir, int(payload["num_frames"]), height, width)
    max_diff = int((chunk_video_maps != full_maps).sum().item())
    full_video = output_dir / f"seq{args.sequence}_semantic_full_overlay.mp4"
    chunk_video = output_dir / f"seq{args.sequence}_semantic_chunk_overlay.mp4"
    _write_video(full_video, image_dir=image_dir, maps=full_maps, colours=colours, height=height, width=width, alpha=float(args.alpha), fps=int(args.fps))
    _write_video(chunk_video, image_dir=image_dir, maps=chunk_video_maps, colours=colours, height=height, width=width, alpha=float(args.alpha), fps=int(args.fps))

    summary = {
        "sequence": str(args.sequence),
        "image_dir": str(image_dir),
        "full_pt": str(full_pt),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "semantic_shape": list(full_maps.shape),
        "semantic_dtype": str(full_maps.dtype),
        "num_labels": len(label_names),
        "label_names": label_names,
        "num_chunks": len(rows),
        "contact_sheet": str(contact_path),
        "full_overlay_video": str(full_video),
        "chunk_overlay_video": str(chunk_video),
        "sample_rows": sample_report,
        "sample_diff_pixels_sum": int(sum(row["diff_pixels"] for row in sample_report)),
        "assembled_chunk_vs_full_diff_pixels": int(max_diff),
    }
    summary_path = output_dir / f"seq{args.sequence}_semantic_visual_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
