#!/usr/bin/env python3
"""Audit early-frame piano coverage for Stream4D v105 label maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCANNET_ROOT = REPO_ROOT / "Stream3D" / "data" / "scannet" / "processed"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_u16(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64, copy=False)


def _read_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.array(Image.open(path).convert("RGB"))


def _parse_frame_ids(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--source must be NAME=LABEL_DIR, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty source name in {item!r}")
        sources[name] = _resolve(path.strip())
    return sources


def _label_path_for_frame(label_dir: Path, frame_id: int) -> Path:
    frame_named = label_dir / f"frame_{frame_id:06d}.png"
    if frame_named.exists():
        return frame_named
    return label_dir / f"{frame_id}.png"


def _top_pred_ids(label: np.ndarray, target: np.ndarray, limit: int = 12) -> list[dict[str, Any]]:
    values, counts = np.unique(label[target], return_counts=True)
    target_area = int(target.sum())
    rows: list[dict[str, Any]] = []
    for value, count in sorted(zip(values.tolist(), counts.tolist()), key=lambda pair: -pair[1])[:limit]:
        pred_total = int((label == int(value)).sum())
        rows.append(
            {
                "pred_id": int(value),
                "overlap_px": int(count),
                "pred_id_total_px": int(pred_total),
                "coverage_vs_target": float(count / target_area) if target_area else None,
                "purity_vs_pred_id": float(count / pred_total) if pred_total else None,
            }
        )
    return rows


def _coverage_overlay(rgb: np.ndarray, target: np.ndarray, label: np.ndarray | None, title: str, scale: float) -> Image.Image:
    view = rgb.astype(np.float32)
    if label is None:
        tint = np.zeros_like(view)
        tint[target] = np.array([255, 220, 0], dtype=np.float32)
        alpha = target.astype(np.float32)[..., None] * 0.55
        view = view * (1.0 - alpha) + tint * alpha
    else:
        covered = target & (label > 0)
        missing = target & (label == 0)
        tint = np.zeros_like(view)
        tint[covered] = np.array([0, 220, 80], dtype=np.float32)
        tint[missing] = np.array([255, 40, 40], dtype=np.float32)
        alpha = target.astype(np.float32)[..., None] * 0.65
        view = view * (1.0 - alpha) + tint * alpha
    img = Image.fromarray(np.clip(view, 0, 255).astype(np.uint8))
    if scale != 1.0:
        w, h = img.size
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    pad = 4
    bbox = draw.textbbox((pad, pad), title, font=font)
    draw.rectangle((0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), title, fill=(255, 255, 255), font=font)
    return img


def _target_bbox(target: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(target)
    h, w = target.shape[:2]
    if yy.size == 0:
        return 0, 0, w, h
    x0 = max(0, int(xx.min()) - int(margin))
    y0 = max(0, int(yy.min()) - int(margin))
    x1 = min(w, int(xx.max()) + int(margin) + 1)
    y1 = min(h, int(yy.max()) + int(margin) + 1)
    return x0, y0, x1, y1


def _crop_arr(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return arr[y0:y1, x0:x1]


def _write_frame_crop_sheets(
    *,
    output_dir: Path,
    scene_root: Path,
    frames: list[int],
    semantic_id: int,
    sources: dict[str, Path],
    scale: float,
    margin: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for frame_id in frames:
        rgb = _read_rgb(scene_root / "color" / f"{frame_id}.jpg")
        sem = _read_u16(scene_root / "label-filt" / f"{frame_id}.png")
        target = sem == int(semantic_id)
        bbox = _target_bbox(target, int(margin))
        row = [
            _coverage_overlay(
                _crop_arr(rgb, bbox),
                _crop_arr(target, bbox),
                None,
                f"f{frame_id:03d} GT semantic={semantic_id}",
                scale,
            )
        ]
        for name, label_dir in sources.items():
            label_path = _label_path_for_frame(label_dir, frame_id)
            label = _read_u16(label_path) if label_path.exists() else np.zeros(target.shape, dtype=np.int64)
            row.append(
                _coverage_overlay(
                    _crop_arr(rgb, bbox),
                    _crop_arr(target, bbox),
                    _crop_arr(label, bbox),
                    f"f{frame_id:03d} {name}",
                    scale,
                )
            )
        widths = [tile.size[0] for tile in row]
        height = max(tile.size[1] for tile in row)
        sheet = Image.new("RGB", (sum(widths), height), color=(20, 20, 20))
        x = 0
        for tile in row:
            sheet.paste(tile, (x, 0))
            x += tile.size[0]
        out = output_dir / f"frame_{frame_id:06d}_semantic{int(semantic_id)}_crop.jpg"
        sheet.save(out, quality=94)
        written.append(out)
    return written


def _write_sheet(
    *,
    output_path: Path,
    scene_root: Path,
    frames: list[int],
    semantic_id: int,
    sources: dict[str, Path],
    scale: float,
) -> None:
    tiles: list[list[Image.Image]] = []
    for frame_id in frames:
        rgb = _read_rgb(scene_root / "color" / f"{frame_id}.jpg")
        sem = _read_u16(scene_root / "label-filt" / f"{frame_id}.png")
        target = sem == int(semantic_id)
        row = [_coverage_overlay(rgb, target, None, f"f{frame_id:03d} GT semantic={semantic_id}", scale)]
        for name, label_dir in sources.items():
            label_path = _label_path_for_frame(label_dir, frame_id)
            label = _read_u16(label_path) if label_path.exists() else None
            if label is not None and label.shape != target.shape:
                raise ValueError(f"shape mismatch for {label_path}: {label.shape} vs {target.shape}")
            row.append(_coverage_overlay(rgb, target, label, f"f{frame_id:03d} {name}", scale))
        tiles.append(row)
    if not tiles:
        return
    widths = [max(row[col].size[0] for row in tiles) for col in range(len(tiles[0]))]
    heights = [max(tile.size[1] for tile in row) for row in tiles]
    sheet = Image.new("RGB", (sum(widths), sum(heights)), color=(20, 20, 20))
    y = 0
    for row, row_h in zip(tiles, heights):
        x = 0
        for tile, col_w in zip(row, widths):
            sheet.paste(tile, (x, y))
            x += col_w
        y += row_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frames", default="0,5,10,15,20,25,30,35")
    parser.add_argument("--semantic-id", type=int, default=90)
    parser.add_argument("--semantic-name", default="piano")
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET_ROOT))
    parser.add_argument("--source", action="append", default=[], help="NAME=LABEL_DIR")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sheet-scale", type=float, default=0.35)
    parser.add_argument("--crop-scale", type=float, default=0.85)
    parser.add_argument("--crop-margin", type=int, default=40)
    parser.add_argument("--no-overview-sheet", action="store_true")
    args = parser.parse_args()

    frames = _parse_frame_ids(args.frames)
    sources = _parse_sources(args.source)
    scene_root = _resolve(args.scannet_root) / args.scene_id
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for frame_id in frames:
        sem_path = scene_root / "label-filt" / f"{frame_id}.png"
        sem = _read_u16(sem_path)
        target = sem == int(args.semantic_id)
        target_area = int(target.sum())
        for name, label_dir in sources.items():
            label_path = _label_path_for_frame(label_dir, frame_id)
            if not label_path.exists():
                records.append(
                    {
                        "scene_id": args.scene_id,
                        "frame_id": int(frame_id),
                        "source": name,
                        "label_path": str(label_path),
                        "missing_label": True,
                    }
                )
                continue
            label = _read_u16(label_path)
            if label.shape != target.shape:
                raise ValueError(f"shape mismatch for {label_path}: {label.shape} vs {target.shape}")
            covered = target & (label > 0)
            uncovered = target & (label == 0)
            top = _top_pred_ids(label, target)
            nonzero_ids = [row for row in top if row["pred_id"] != 0]
            significant_ids = [row for row in nonzero_ids if float(row["coverage_vs_target"] or 0.0) >= 0.01]
            records.append(
                {
                    "scene_id": args.scene_id,
                    "frame_id": int(frame_id),
                    "source": name,
                    "semantic_id": int(args.semantic_id),
                    "semantic_name": str(args.semantic_name),
                    "semantic_area_px": int(target_area),
                    "covered_px": int(covered.sum()),
                    "uncovered_px": int(uncovered.sum()),
                    "coverage_ratio": float(covered.sum() / target_area) if target_area else None,
                    "pred_id_count_nonzero_in_top": int(len(nonzero_ids)),
                    "pred_id_count_ge_1pct_in_top": int(len(significant_ids)),
                    "top_pred_ids": top,
                    "label_path": str(label_path),
                    "label_sha256": _sha256(label_path),
                    "semantic_path": str(sem_path),
                    "semantic_sha256": _sha256(sem_path),
                }
            )

    json_path = output_root / f"{args.scene_id}_semantic{int(args.semantic_id)}_coverage_records.json"
    csv_path = output_root / f"{args.scene_id}_semantic{int(args.semantic_id)}_coverage_summary.csv"
    sheet_path = output_root / f"{args.scene_id}_semantic{int(args.semantic_id)}_coverage_sheet.jpg"
    crop_dir = output_root / "frame_crops"
    payload = {
        "schema_version": "stream4d_v105_piano_early_frame_coverage_v1",
        "scene_id": args.scene_id,
        "semantic_id": int(args.semantic_id),
        "semantic_name": str(args.semantic_name),
        "frames": frames,
        "sources": {name: str(path) for name, path in sources.items()},
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "frame_id",
                "source",
                "semantic_id",
                "semantic_name",
                "semantic_area_px",
                "covered_px",
                "uncovered_px",
                "coverage_ratio",
                "pred_id_count_ge_1pct_in_top",
                "top_pred_ids_compact",
            ],
        )
        writer.writeheader()
        for row in records:
            if row.get("missing_label"):
                continue
            compact = ";".join(
                f"{item['pred_id']}:{item['overlap_px']}:{float(item['coverage_vs_target'] or 0.0):.6f}"
                for item in row["top_pred_ids"][:6]
            )
            writer.writerow(
                {
                    "scene_id": row["scene_id"],
                    "frame_id": row["frame_id"],
                    "source": row["source"],
                    "semantic_id": row["semantic_id"],
                    "semantic_name": row["semantic_name"],
                    "semantic_area_px": row["semantic_area_px"],
                    "covered_px": row["covered_px"],
                    "uncovered_px": row["uncovered_px"],
                    "coverage_ratio": row["coverage_ratio"],
                    "pred_id_count_ge_1pct_in_top": row["pred_id_count_ge_1pct_in_top"],
                    "top_pred_ids_compact": compact,
                }
            )
    crop_paths = _write_frame_crop_sheets(
        output_dir=crop_dir,
        scene_root=scene_root,
        frames=frames,
        semantic_id=int(args.semantic_id),
        sources=sources,
        scale=float(args.crop_scale),
        margin=int(args.crop_margin),
    )
    if not bool(args.no_overview_sheet):
        _write_sheet(
            output_path=sheet_path,
            scene_root=scene_root,
            frames=frames,
            semantic_id=int(args.semantic_id),
            sources=sources,
            scale=float(args.sheet_scale),
        )
    payload["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "sheet": str(sheet_path) if sheet_path.exists() else None,
        "frame_crops": [str(path) for path in crop_paths],
        "json_sha256": _sha256(json_path),
        "csv_sha256": _sha256(csv_path),
        "sheet_sha256": _sha256(sheet_path) if sheet_path.exists() else None,
        "frame_crop_sha256": {str(path): _sha256(path) for path in crop_paths},
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["outputs"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
