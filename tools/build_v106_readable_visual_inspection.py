#!/usr/bin/env python3
"""Build readable visual inspection assets from a v106 label summary.

The normal scene review video is useful for a quick skim, but it is too dense
for quality decisions. This tool keeps the model output fixed and rebuilds a
small inspection pack with full-size frames, mask-only views, edge-only views,
small-mask highlights, and 1:1 object crops.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_4dpm_style_per_frame_segmentors import PALETTE  # noqa: E402


DEFAULT_RGB_ROOT = STREAM3D_ROOT / "data" / "scannet" / "processed"

_PALETTE_ARRAY = np.asarray(PALETTE, dtype=np.uint8)
_PALETTE_LUT_U16 = np.zeros((65536, 3), dtype=np.uint8)
_PALETTE_LUT_U16[1:] = _PALETTE_ARRAY[(np.arange(1, 65536, dtype=np.int64) - 1) % int(len(_PALETTE_ARRAY))]
_EDGE_KERNEL_2 = np.ones((2, 2), dtype=np.uint8)
_EDGE_KERNEL_3 = np.ones((3, 3), dtype=np.uint8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.replace(";", ",").split(",") if part.strip()]


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_label(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    if shape is not None and label.shape[:2] != shape:
        label = cv2.resize(label, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.uint16, copy=False)


def write_rgb(path: Path, rgb: np.ndarray, quality: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def label_color(label: np.ndarray) -> np.ndarray:
    return _PALETTE_LUT_U16[label.astype(np.uint16, copy=False)]


def label_edges(label: np.ndarray, fg: np.ndarray | None = None) -> np.ndarray:
    if fg is None:
        fg = label > 0
    edge = cv2.morphologyEx(label.astype(np.uint16, copy=False), cv2.MORPH_GRADIENT, _EDGE_KERNEL_3)
    edge = (edge > 0) & fg
    return cv2.dilate(edge.astype(np.uint8), _EDGE_KERNEL_2, iterations=1).astype(bool)


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.62) -> None:
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def object_stats(label: np.ndarray) -> list[dict[str, Any]]:
    ids, counts = np.unique(label[label > 0], return_counts=True)
    stats: list[dict[str, Any]] = []
    for obj_id, area in zip(ids.tolist(), counts.tolist(), strict=False):
        ys, xs = np.where(label == int(obj_id))
        if ys.size == 0:
            continue
        stats.append(
            {
                "object_id": int(obj_id),
                "area": int(area),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
            }
        )
    stats.sort(key=lambda row: int(row["area"]), reverse=True)
    return stats


def draw_object_ids(img: np.ndarray, stats: list[dict[str, Any]], *, min_area: int) -> np.ndarray:
    out = img.copy()
    for row in stats:
        if int(row["area"]) < int(min_area):
            continue
        cx, cy = row["centroid_xy"]
        draw_text(out, str(int(row["object_id"])), (int(cx), int(cy)), scale=0.58)
    return out


def overlay_strong(rgb: np.ndarray, label: np.ndarray, *, alpha: float, stats: list[dict[str, Any]], id_min_area: int) -> np.ndarray:
    fg = label > 0
    color = label_color(label)
    blended = cv2.addWeighted(rgb, 1.0 - float(alpha), color, float(alpha), 0.0)
    blended[~fg] = rgb[~fg]
    edge = label_edges(label, fg)
    blended[edge] = np.array([255, 255, 255], dtype=np.uint8)
    return draw_object_ids(blended, stats, min_area=id_min_area)


def mask_only(label: np.ndarray, stats: list[dict[str, Any]], *, id_min_area: int) -> np.ndarray:
    fg = label > 0
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    out[fg] = label_color(label)[fg]
    out[label_edges(label, fg)] = np.array([255, 255, 255], dtype=np.uint8)
    return draw_object_ids(out, stats, min_area=id_min_area)


def edge_only(rgb: np.ndarray, label: np.ndarray, stats: list[dict[str, Any]], *, id_min_area: int) -> np.ndarray:
    out = rgb.copy()
    edge = label_edges(label)
    out[edge] = np.array([255, 255, 0], dtype=np.uint8)
    return draw_object_ids(out, stats, min_area=id_min_area)


def coverage_view(rgb: np.ndarray, label: np.ndarray) -> np.ndarray:
    fg = label > 0
    out = (rgb.astype(np.float32) * 0.42).astype(np.uint8)
    cyan = np.zeros_like(out)
    cyan[:, :] = np.array([0, 215, 255], dtype=np.uint8)
    covered = cv2.addWeighted(rgb, 0.45, cyan, 0.55, 0.0)
    out[fg] = covered[fg]
    out[label_edges(label, fg)] = np.array([255, 255, 0], dtype=np.uint8)
    return out


def small_mask_view(
    rgb: np.ndarray,
    label: np.ndarray,
    stats: list[dict[str, Any]],
    *,
    small_area: int,
    id_min_area: int,
) -> np.ndarray:
    out = rgb.copy()
    small_ids = [int(row["object_id"]) for row in stats if int(row["area"]) < int(small_area)]
    if small_ids:
        small = np.isin(label, np.asarray(small_ids, dtype=np.uint16))
        magenta = np.zeros_like(out)
        magenta[:, :] = np.array([255, 0, 255], dtype=np.uint8)
        mixed = cv2.addWeighted(out, 0.35, magenta, 0.65, 0.0)
        out[small] = mixed[small]
        out[label_edges(label, label > 0)] = np.array([255, 255, 0], dtype=np.uint8)
        out = draw_object_ids(out, stats, min_area=min(int(id_min_area), int(small_area)))
    else:
        draw_text(out, f"no masks smaller than {int(small_area)} px", (18, 36), scale=0.74)
    return out


def resolve_rgb_path(summary: dict[str, Any], row: dict[str, Any], rgb_root: Path) -> Path:
    if row.get("rgb_path"):
        return resolve_path(str(row["rgb_path"]))
    scene_id = str(summary["scene_id"])
    return rgb_root / scene_id / "color" / f"{int(row['frame_id'])}.jpg"


def selected_rows(records: list[dict[str, Any]], frame_indices: list[int], frame_ids: list[int]) -> list[dict[str, Any]]:
    by_index = {int(row["chunk_frame_index"]): row for row in records}
    by_frame_id = {int(row["frame_id"]): row for row in records}
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for idx in frame_indices:
        row = by_index.get(int(idx))
        if row is not None and int(row["chunk_frame_index"]) not in seen:
            rows.append(row)
            seen.add(int(row["chunk_frame_index"]))
    for frame_id in frame_ids:
        row = by_frame_id.get(int(frame_id))
        if row is not None and int(row["chunk_frame_index"]) not in seen:
            rows.append(row)
            seen.add(int(row["chunk_frame_index"]))
    rows.sort(key=lambda row: int(row["chunk_frame_index"]))
    return rows


def crop_box(bbox: list[int], shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)


def write_top_object_crops(
    *,
    frame_dir: Path,
    frame_index: int,
    frame_id: int,
    rgb: np.ndarray,
    label: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    stats: list[dict[str, Any]],
    top_k: int,
    margin: int,
) -> list[dict[str, Any]]:
    crop_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(stats[: int(top_k)], start=1):
        x0, y0, x1, y1 = crop_box(row["bbox_xyxy"], label.shape[:2], int(margin))
        base = f"frame_{frame_index:03d}_id_{frame_id:06d}_rank_{rank:02d}_obj_{int(row['object_id']):04d}"
        raw_path = frame_dir / "object_crops" / f"{base}_raw.jpg"
        overlay_path = frame_dir / "object_crops" / f"{base}_overlay.jpg"
        mask_path = frame_dir / "object_crops" / f"{base}_mask_only.jpg"
        write_rgb(raw_path, rgb[y0:y1, x0:x1])
        write_rgb(overlay_path, overlay[y0:y1, x0:x1])
        write_rgb(mask_path, mask[y0:y1, x0:x1])
        crop_rows.append(
            {
                "frame_index": int(frame_index),
                "frame_id": int(frame_id),
                "rank": int(rank),
                "object_id": int(row["object_id"]),
                "area": int(row["area"]),
                "bbox_xyxy": row["bbox_xyxy"],
                "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "raw_path": repo_rel(raw_path),
                "overlay_path": repo_rel(overlay_path),
                "mask_only_path": repo_rel(mask_path),
            }
        )
    return crop_rows


def write_video(
    *,
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    rgb_root: Path,
    fps: float,
    mode: str,
    alpha: float,
    small_area: int,
    id_min_area: int,
) -> dict[str, Any]:
    writer: cv2.VideoWriter | None = None
    width = 0
    height = 0
    try:
        for row in records:
            rgb = read_rgb(resolve_rgb_path(summary, row, rgb_root))
            label = read_label(resolve_path(str(row["label_path"])), rgb.shape[:2])
            stats = object_stats(label)
            if mode == "overlay":
                frame = overlay_strong(rgb, label, alpha=alpha, stats=stats, id_min_area=id_min_area)
            elif mode == "mask_only":
                frame = mask_only(label, stats, id_min_area=id_min_area)
            elif mode == "edge_only":
                frame = edge_only(rgb, label, stats, id_min_area=id_min_area)
            elif mode == "coverage":
                frame = coverage_view(rgb, label)
            elif mode == "small_mask":
                frame = small_mask_view(rgb, label, stats, small_area=small_area, id_min_area=id_min_area)
            else:
                raise ValueError(f"unsupported video mode: {mode}")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = bgr.shape[:2]
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open VideoWriter for {path}")
            writer.write(bgr)
    finally:
        if writer is not None:
            writer.release()
    return {
        "mode": mode,
        "path": repo_rel(path),
        "sha256": sha256_file(path),
        "frame_count": len(records),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
    }


def write_videos_one_pass(
    *,
    video_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    rgb_root: Path,
    fps: float,
    alpha: float,
    small_area: int,
    id_min_area: int,
) -> list[dict[str, Any]]:
    modes = ("overlay", "mask_only", "edge_only", "coverage", "small_mask")
    writers: dict[str, cv2.VideoWriter] = {}
    paths = {mode: video_dir / f"{mode}_{len(records)}f.mp4" for mode in modes}
    sizes: dict[str, tuple[int, int]] = {}
    try:
        for row in records:
            rgb = read_rgb(resolve_rgb_path(summary, row, rgb_root))
            label = read_label(resolve_path(str(row["label_path"])), rgb.shape[:2])
            stats = object_stats(label)
            frames = {
                "overlay": overlay_strong(rgb, label, alpha=alpha, stats=stats, id_min_area=id_min_area),
                "mask_only": mask_only(label, stats, id_min_area=id_min_area),
                "edge_only": edge_only(rgb, label, stats, id_min_area=id_min_area),
                "coverage": coverage_view(rgb, label),
                "small_mask": small_mask_view(rgb, label, stats, small_area=small_area, id_min_area=id_min_area),
            }
            for mode, frame in frames.items():
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if mode not in writers:
                    height, width = bgr.shape[:2]
                    video_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(paths[mode]),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open VideoWriter for {paths[mode]}")
                    writers[mode] = writer
                    sizes[mode] = (int(width), int(height))
                writers[mode].write(bgr)
    finally:
        for writer in writers.values():
            writer.release()
    return [
        {
            "mode": mode,
            "path": repo_rel(paths[mode]),
            "sha256": sha256_file(paths[mode]),
            "frame_count": len(records),
            "fps": float(fps),
            "width": int(sizes.get(mode, (0, 0))[0]),
            "height": int(sizes.get(mode, (0, 0))[1]),
        }
        for mode in modes
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--frame-indices", default="0,33,45,89")
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--overlay-alpha", type=float, default=0.62)
    parser.add_argument("--small-area", type=int, default=2048)
    parser.add_argument("--id-min-area", type=int, default=6000)
    parser.add_argument("--object-crop-top-k", type=int, default=8)
    parser.add_argument("--crop-margin", type=int, default=48)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--skip-videos", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = resolve_path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = list(summary.get("records", []))
    if not records:
        raise ValueError(f"summary has no records: {summary_path}")
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    rgb_root = resolve_path(args.rgb_root)
    frame_indices = parse_ints(str(args.frame_indices))
    frame_ids = parse_ints(str(args.frame_ids))
    if not frame_indices and not frame_ids:
        frame_indices = [0, len(records) // 2, len(records) - 1]
    rows = selected_rows(records, frame_indices, frame_ids)
    if not rows:
        raise ValueError("no selected rows matched --frame-indices/--frame-ids")

    manifest_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for row in rows:
        frame_index = int(row["chunk_frame_index"])
        frame_id = int(row["frame_id"])
        rgb = read_rgb(resolve_rgb_path(summary, row, rgb_root))
        label = read_label(resolve_path(str(row["label_path"])), rgb.shape[:2])
        stats = object_stats(label)
        overlay_img = overlay_strong(rgb, label, alpha=float(args.overlay_alpha), stats=stats, id_min_area=int(args.id_min_area))
        mask_img = mask_only(label, stats, id_min_area=int(args.id_min_area))
        edge_img = edge_only(rgb, label, stats, id_min_area=int(args.id_min_area))
        coverage_img = coverage_view(rgb, label)
        small_img = small_mask_view(
            rgb,
            label,
            stats,
            small_area=int(args.small_area),
            id_min_area=int(args.id_min_area),
        )

        base = f"frame_{frame_index:03d}_id_{frame_id:06d}"
        outputs = {
            "raw": frame_dir / f"{base}_raw.jpg",
            "overlay_strong": frame_dir / f"{base}_overlay_strong.jpg",
            "mask_only": frame_dir / f"{base}_mask_only.jpg",
            "edge_only": frame_dir / f"{base}_edge_only.jpg",
            "coverage": frame_dir / f"{base}_coverage.jpg",
            "small_mask": frame_dir / f"{base}_small_mask_highlight.jpg",
        }
        write_rgb(outputs["raw"], rgb)
        write_rgb(outputs["overlay_strong"], overlay_img)
        write_rgb(outputs["mask_only"], mask_img)
        write_rgb(outputs["edge_only"], edge_img)
        write_rgb(outputs["coverage"], coverage_img)
        write_rgb(outputs["small_mask"], small_img)
        object_rows.extend(
            write_top_object_crops(
                frame_dir=frame_dir,
                frame_index=frame_index,
                frame_id=frame_id,
                rgb=rgb,
                label=label,
                overlay=overlay_img,
                mask=mask_img,
                stats=stats,
                top_k=int(args.object_crop_top_k),
                margin=int(args.crop_margin),
            )
        )
        manifest_rows.append(
            {
                "frame_index": frame_index,
                "frame_id": frame_id,
                "visible_object_count": len(stats),
                "foreground_pixels": int(np.count_nonzero(label > 0)),
                "foreground_ratio": float(np.count_nonzero(label > 0) / label.size),
                "small_object_count": int(sum(1 for item in stats if int(item["area"]) < int(args.small_area))),
                "small_area_threshold": int(args.small_area),
                "raw_path": repo_rel(outputs["raw"]),
                "overlay_strong_path": repo_rel(outputs["overlay_strong"]),
                "mask_only_path": repo_rel(outputs["mask_only"]),
                "edge_only_path": repo_rel(outputs["edge_only"]),
                "coverage_path": repo_rel(outputs["coverage"]),
                "small_mask_highlight_path": repo_rel(outputs["small_mask"]),
            }
        )

    stats_csv = out_dir / "selected_frame_mask_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "frame_id",
                "visible_object_count",
                "foreground_pixels",
                "foreground_ratio",
                "small_object_count",
                "small_area_threshold",
                "raw_path",
                "overlay_strong_path",
                "mask_only_path",
                "edge_only_path",
                "coverage_path",
                "small_mask_highlight_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    crops_json = out_dir / "top_object_crops.json"
    crops_json.write_text(json.dumps(object_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    videos: list[dict[str, Any]] = []
    if not args.skip_videos:
        videos = write_videos_one_pass(
            video_dir=out_dir / "videos",
            records=records,
            summary=summary,
            rgb_root=rgb_root,
            fps=float(args.fps),
            alpha=float(args.overlay_alpha),
            small_area=int(args.small_area),
            id_min_area=int(args.id_min_area),
        )

    manifest = {
        "schema_version": "stream4d_v106_readable_visual_inspection_v1",
        "summary_path": repo_rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "scene_id": str(summary.get("scene_id")),
        "record_count": int(len(records)),
        "selected_frame_count": int(len(rows)),
        "selected_frames_csv": repo_rel(stats_csv),
        "top_object_crops_json": repo_rel(crops_json),
        "out_dir": repo_rel(out_dir),
        "overlay_alpha": float(args.overlay_alpha),
        "small_area_threshold": int(args.small_area),
        "id_min_area": int(args.id_min_area),
        "selected_frames": manifest_rows,
        "videos": videos,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "selected_frame_count": len(rows),
                "video_count": len(videos),
                "selected_frames_csv": str(stats_csv),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
