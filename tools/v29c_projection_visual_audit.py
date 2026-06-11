#!/usr/bin/env python3
"""Visual audit for v29C projected SemanticKITTI sparse 2D labels.

This tool intentionally produces human-inspectable PNGs. Numeric projection
gates are useful, but they do not prove that the projected 2D semantic anchors
line up with the KITTI image in a visually plausible way.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SEMANTIC_KITTI_ID_TO_NAME: Dict[int, str] = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on_rails",
    18: "truck",
    20: "other_vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other_ground",
    50: "building",
    51: "fence",
    52: "other_structure",
    60: "lane_marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic_sign",
    99: "other_object",
    252: "moving_car",
    253: "moving_bicyclist",
    254: "moving_person",
    255: "moving_motorcyclist",
    256: "moving_on_rails",
    257: "moving_bus",
    258: "moving_truck",
    259: "moving_other_vehicle",
}


SEMANTIC_KITTI_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (30, 30, 30),
    1: (255, 255, 255),
    10: (0, 0, 142),
    11: (119, 11, 32),
    13: (0, 60, 100),
    15: (0, 0, 230),
    16: (0, 80, 100),
    18: (0, 0, 70),
    20: (0, 0, 90),
    30: (220, 20, 60),
    31: (255, 0, 0),
    32: (255, 0, 120),
    40: (128, 64, 128),
    44: (250, 170, 160),
    48: (244, 35, 232),
    49: (81, 0, 81),
    50: (70, 70, 70),
    51: (190, 153, 153),
    52: (102, 102, 156),
    60: (230, 150, 140),
    70: (107, 142, 35),
    71: (152, 251, 152),
    72: (152, 251, 152),
    80: (153, 153, 153),
    81: (220, 220, 0),
    99: (255, 170, 0),
    252: (0, 0, 142),
    253: (255, 0, 0),
    254: (220, 20, 60),
    255: (255, 0, 120),
    256: (0, 80, 100),
    257: (0, 60, 100),
    258: (0, 0, 70),
    259: (0, 0, 90),
}


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    if not str(text or "").strip():
        return list(default)
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, 16)
    return ImageFont.load_default()


def _label_color(label: int) -> Tuple[int, int, int]:
    if label in SEMANTIC_KITTI_COLOR:
        return SEMANTIC_KITTI_COLOR[label]
    rng = np.random.default_rng(int(label) + 17)
    return tuple(int(x) for x in rng.integers(32, 240, size=3))


def _paint_sparse_points(shape: Tuple[int, int], sem: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    height, width = shape
    color = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.nonzero(valid)
    for y, x in zip(yy.tolist(), xx.tolist()):
        rgb = _label_color(int(sem[y, x]))
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        color[y0:y1, x0:x1, :] = rgb
    return color


def _overlay(image: Image.Image, color_points: np.ndarray, alpha: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask = np.any(color_points != 0, axis=2)
    out = rgb.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_points[mask].astype(np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _depth_color(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    vals = depth[valid]
    if vals.size == 0:
        return out
    lo = float(np.quantile(vals, 0.02))
    hi = float(np.quantile(vals, 0.98))
    norm = np.clip((depth - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    out[..., 0] = np.clip(255 * (1.0 - norm), 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(255 * (1.0 - np.abs(norm - 0.5) * 2.0), 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(255 * norm, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def _legend(labels: np.ndarray, counts: np.ndarray, width: int, height: int) -> Image.Image:
    font = _font()
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((12, 10), "Projected SemanticKITTI labels", fill=(0, 0, 0), font=font)
    y = 40
    for label, count in zip(labels[:14], counts[:14]):
        rgb = _label_color(int(label))
        draw.rectangle((14, y + 2, 34, y + 22), fill=rgb, outline=(0, 0, 0))
        name = SEMANTIC_KITTI_ID_TO_NAME.get(int(label), f"id{int(label)}")
        draw.text((44, y), f"{int(label):>3} {name:<18} {int(count)} px", fill=(0, 0, 0), font=font)
        y += 28
    return img


def _panel_title(img: Image.Image, title: str) -> Image.Image:
    font = _font()
    out = Image.new("RGB", (img.width, img.height + 28), (20, 20, 20))
    out.paste(img.convert("RGB"), (0, 28))
    draw = ImageDraw.Draw(out)
    draw.text((8, 5), title, fill=(255, 255, 255), font=font)
    return out


def _make_montage(panels: Sequence[Tuple[str, Image.Image]], legend: Image.Image, out_path: Path) -> None:
    titled = [_panel_title(img, title) for title, img in panels]
    w = max(p.width for p in titled)
    h = max(p.height for p in titled)
    resized = []
    for panel in titled:
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(panel, (0, 0))
        resized.append(canvas)
    montage = Image.new("RGB", (w * 2, h * 2 + legend.height), (0, 0, 0))
    montage.paste(resized[0], (0, 0))
    montage.paste(resized[1], (w, 0))
    montage.paste(resized[2], (0, h))
    montage.paste(resized[3], (w, h))
    legend_canvas = Image.new("RGB", (w * 2, legend.height), (245, 245, 245))
    legend_canvas.paste(legend, (0, 0))
    montage.paste(legend_canvas, (0, h * 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)


def _label_summary(sem: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    labels, counts = np.unique(sem[valid].astype(np.int32), return_counts=True)
    order = np.argsort(-counts)
    labels = labels[order]
    counts = counts[order]
    row: Dict[str, object] = {}
    height, width = sem.shape
    for label in (40, 48, 50, 70, 72, 10):
        mask = valid & (sem.astype(np.int32) == label)
        name = SEMANTIC_KITTI_ID_TO_NAME.get(label, f"id{label}")
        row[f"{name}_count"] = int(mask.sum())
        if mask.any():
            yy, xx = np.nonzero(mask)
            row[f"{name}_mean_y_ratio"] = float(yy.mean() / max(1, height - 1))
            row[f"{name}_mean_x_ratio"] = float(xx.mean() / max(1, width - 1))
        else:
            row[f"{name}_mean_y_ratio"] = None
            row[f"{name}_mean_x_ratio"] = None
    row["top_labels"] = " ".join(
        f"{int(label)}:{SEMANTIC_KITTI_ID_TO_NAME.get(int(label), f'id{int(label)}')}:{int(count)}"
        for label, count in zip(labels[:10], counts[:10])
    )
    return labels, counts, row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", default="/mnt/data/users/chengshun.wang/data/semantickitti_odometry/dataset/sequences/01")
    parser.add_argument("--cache-dir", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/projection_cache/seq01")
    parser.add_argument("--out-dir", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/projection_visual_audit")
    parser.add_argument("--frames", default="174,220,290,350,464,550,650,757,850,925")
    parser.add_argument("--point-radius", type=int, default=1)
    args = parser.parse_args()

    sequence_root = Path(args.sequence_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    frames = _parse_int_list(args.frames, [])
    rows: List[Dict[str, object]] = []

    for frame in frames:
        image_path = sequence_root / "image_2" / f"{frame:06d}.png"
        sem_path = cache_dir / f"{frame:06d}_sem_sparse.npy"
        depth_path = cache_dir / f"{frame:06d}_depth_sparse.npy"
        valid_path = cache_dir / f"{frame:06d}_valid_mask.npy"
        if not image_path.exists() or not sem_path.exists() or not valid_path.exists() or not depth_path.exists():
            rows.append({"frame": frame, "status": "missing_inputs"})
            continue
        image = Image.open(image_path).convert("RGB")
        sem = np.load(sem_path)
        depth = np.load(depth_path)
        valid = np.load(valid_path).astype(bool)
        if sem.shape != valid.shape or sem.shape != depth.shape:
            rows.append({"frame": frame, "status": "shape_mismatch", "sem_shape": str(sem.shape), "valid_shape": str(valid.shape)})
            continue

        color_points = _paint_sparse_points(sem.shape, sem, valid, max(0, int(args.point_radius)))
        semantic_overlay = _overlay(image, color_points, 0.82)
        semantic_points = Image.fromarray(color_points)
        depth_overlay = _overlay(image, _depth_color(depth, valid), 0.75)
        valid_mask = Image.fromarray(np.where(valid, 255, 0).astype(np.uint8)).convert("RGB")
        labels, counts, summary = _label_summary(sem, valid)
        legend = _legend(labels, counts, image.width * 2, 120)

        prefix = out_dir / f"frame_{frame:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(str(prefix) + "_rgb.png")
        semantic_overlay.save(str(prefix) + "_semantic_overlay.png")
        semantic_points.save(str(prefix) + "_semantic_points.png")
        depth_overlay.save(str(prefix) + "_depth_overlay.png")
        valid_mask.save(str(prefix) + "_valid_mask.png")
        _make_montage(
            [
                ("RGB", image),
                ("Projected semantic overlay", semantic_overlay),
                ("Projected semantic points only", semantic_points),
                ("Projected depth overlay", depth_overlay),
            ],
            legend,
            Path(str(prefix) + "_montage.png"),
        )

        valid_count = int(valid.sum())
        rows.append(
            {
                "frame": frame,
                "status": "ok",
                "valid_projected_pixels": valid_count,
                "projected_coverage": float(valid_count / max(1, valid.size)),
                **summary,
                "montage": str(prefix) + "_montage.png",
                "semantic_overlay": str(prefix) + "_semantic_overlay.png",
            }
        )

    _write_csv(out_dir / "projection_visual_audit_summary.csv", rows)
    summary = {
        "frames_requested": len(frames),
        "frames_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "frames": frames,
        "out_dir": str(out_dir),
        "visual_audit_note": "Human visual inspection is still required; these images are the audit evidence.",
    }
    (out_dir / "projection_visual_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["frames_ok"] == len(frames) else 1


if __name__ == "__main__":
    raise SystemExit(main())
