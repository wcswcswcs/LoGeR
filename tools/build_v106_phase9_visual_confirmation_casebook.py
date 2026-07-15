#!/usr/bin/env python3
"""Build visual confirmation artifacts for Stream4D v106 phase9 chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_abs(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"failed to read label: {path}")
    if label.ndim == 3:
        label = label[..., 0]
    return label.astype(np.int64, copy=False)


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    width = max(1, int(round(w * (height / max(h, 1)))))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def title_bar(width: int, text: str, height: int = 42) -> np.ndarray:
    bar = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(bar, text[:180], (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (15, 15, 15), 2, cv2.LINE_AA)
    return bar


def titled(img: np.ndarray, text: str) -> np.ndarray:
    return np.vstack([title_bar(img.shape[1], text), img])


def stable_color(label_id: int) -> tuple[int, int, int]:
    x = int(label_id) * 1103515245 + 12345
    b = 40 + ((x >> 0) & 0x7F)
    g = 80 + ((x >> 8) & 0x9F)
    r = 80 + ((x >> 16) & 0x9F)
    return int(b % 256), int(g % 256), int(r % 256)


def colorize_label(label: np.ndarray) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for label_id in np.unique(label):
        if label_id == 0:
            continue
        out[label == label_id] = stable_color(int(label_id))
    return out


def hard_foreground_diff(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if pred.shape != ref.shape:
        raise ValueError(f"label shape mismatch: pred={pred.shape} ref={ref.shape}")
    pred_fg = pred > 0
    ref_fg = ref > 0
    out = np.full((*pred.shape, 3), 20, dtype=np.uint8)
    out[pred_fg & ref_fg] = (180, 180, 180)
    out[pred_fg & ~ref_fg] = (40, 40, 230)
    out[~pred_fg & ref_fg] = (230, 170, 40)
    return out


def load_reference_overlay_map(reference_summary_path: Path) -> dict[int, Path]:
    data = load_json(reference_summary_path)
    result: dict[int, Path] = {}
    for record in data.get("records", []):
        overlay = record.get("overlay_path")
        frame_id = record.get("frame_id")
        if frame_id is not None and overlay:
            result[int(frame_id)] = ensure_abs(overlay)
    return result


def load_coverage_records(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    rows = load_json(path)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        result[(row["boundary"], int(row["frame_id"]))] = row
    return result


def collect_scene_frames(
    summaries: list[tuple[str, Path]],
    *,
    overlap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_frames: list[dict[str, Any]] = []
    chunk_meta: list[dict[str, Any]] = []
    for chunk_index, (chunk_name, summary_path) in enumerate(summaries):
        summary = load_json(summary_path)
        records = list(summary.get("records", []))
        start_idx = 0 if chunk_index == 0 else overlap
        used_records = records[start_idx:]
        chunk_meta.append(
            {
                "chunk_name": chunk_name,
                "summary_path": repo_rel(summary_path),
                "input_record_count": len(records),
                "used_record_count": len(used_records),
                "skipped_overlap_record_count": start_idx,
                "frame_ids": [int(r["frame_id"]) for r in used_records],
                "video_path": repo_rel(ensure_abs(summary.get("video_path"))) if summary.get("video_path") else None,
                "total_runtime_sec": summary.get("total_runtime_sec"),
                "tracking_runtime_sec": summary.get("total_tracking_runtime_sec")
                or summary.get("tracking_runtime_sec"),
                "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
            }
        )
        for record in used_records:
            overlay_path = ensure_abs(record["overlay_path"])
            label_path = ensure_abs(record["label_path"])
            scene_frames.append(
                {
                    "scene_frame_index": len(scene_frames),
                    "chunk_name": chunk_name,
                    "chunk_frame_index": int(record["chunk_frame_index"]),
                    "frame_id": int(record["frame_id"]),
                    "overlay_path": repo_rel(overlay_path),
                    "overlay_sha256": sha256_file(overlay_path),
                    "label_path": repo_rel(label_path),
                    "label_sha256": sha256_file(label_path),
                }
            )
    return scene_frames, {"chunks": chunk_meta}


def write_scene_video(scene_frames: list[dict[str, Any]], output_path: Path, *, fps: float) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = read_image(REPO_ROOT / scene_frames[0]["overlay_path"])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")
    for frame in scene_frames:
        img = read_image(REPO_ROOT / frame["overlay_path"])
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    cap = cv2.VideoCapture(str(output_path))
    decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else None
    cap.release()
    return {
        "path": repo_rel(output_path),
        "sha256": sha256_file(output_path),
        "input_frame_count": len(scene_frames),
        "decoded_frame_count_property": decoded,
        "fps": fps,
        "width": width,
        "height": height,
    }


def copy_fullres(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": repo_rel(dst), "sha256": sha256_file(dst)}


def build_panel(
    *,
    frame_id: int,
    boundary: str,
    pred_overlay: Path,
    ref_overlay: Path | None,
    pred_label: Path,
    ref_label: Path | None,
    diff_image: Path | None,
    output_path: Path,
) -> None:
    pred_img = read_image(pred_overlay)
    height = pred_img.shape[0]
    cells: list[np.ndarray] = []

    if ref_overlay and ref_overlay.exists():
        cells.append(titled(resize_to_height(read_image(ref_overlay), height), f"reference overlay frame {frame_id}"))
    else:
        blank = np.full_like(pred_img, 230)
        cells.append(titled(blank, f"reference overlay missing frame {frame_id}"))

    cells.append(titled(pred_img, f"candidate overlay frame {frame_id}"))

    pred = read_label(pred_label)
    if ref_label and ref_label.exists():
        ref = read_label(ref_label)
        if pred.shape == ref.shape:
            hard = hard_foreground_diff(pred, ref)
        else:
            hard = np.full((*pred.shape, 3), 30, dtype=np.uint8)
            cv2.putText(hard, "shape mismatch", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        hard = colorize_label(pred)
    cells.append(titled(resize_to_height(hard, height), "hard diff: red=extra pred, blue=missing ref"))

    if diff_image and diff_image.exists():
        cells.append(titled(resize_to_height(read_image(diff_image), height), "coverage diagnostic diff"))
    else:
        cells.append(titled(resize_to_height(colorize_label(pred), height), "candidate label colors"))

    max_cell_height = max(c.shape[0] for c in cells)
    padded: list[np.ndarray] = []
    for cell in cells:
        if cell.shape[0] < max_cell_height:
            pad = np.full((max_cell_height - cell.shape[0], cell.shape[1], 3), 245, dtype=np.uint8)
            cell = np.vstack([cell, pad])
        padded.append(cell)

    spacer = np.full((max_cell_height, 12, 3), 245, dtype=np.uint8)
    top = np.hstack([padded[0], spacer, padded[1]])
    bottom = np.hstack([padded[2], spacer, padded[3]])
    width = max(top.shape[1], bottom.shape[1])
    if top.shape[1] < width:
        top = np.hstack([top, np.full((top.shape[0], width - top.shape[1], 3), 245, dtype=np.uint8)])
    if bottom.shape[1] < width:
        bottom = np.hstack([bottom, np.full((bottom.shape[0], width - bottom.shape[1], 3), 245, dtype=np.uint8)])
    header = title_bar(width, f"{boundary} frame {frame_id} visual confirmation panel", height=48)
    panel = np.vstack([header, top, np.full((12, width, 3), 245, dtype=np.uint8), bottom])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panel):
        raise RuntimeError(f"failed to write panel: {output_path}")


def parse_key_frame(text: str) -> tuple[str, int]:
    boundary, frame = text.split(":", 1)
    return boundary, int(frame)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ensure_abs(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        ("c0", ensure_abs(args.c0_summary)),
        ("c1", ensure_abs(args.c1_summary)),
        ("c2", ensure_abs(args.c2_summary)),
    ]

    scene_frames, meta = collect_scene_frames(summaries, overlap=args.overlap)
    video_info = write_scene_video(scene_frames, output_root / "videos" / args.scene_video_name, fps=args.fps)
    write_json(output_root / "scene_frame_manifest.json", scene_frames)

    coverage_records = load_coverage_records(ensure_abs(args.coverage_records) if args.coverage_records else None)
    reference_overlay_maps: dict[str, dict[int, Path]] = {}
    summary_by_boundary = {
        "boundary_c0_c1": ensure_abs(args.c1_summary),
        "boundary_c1_c2": ensure_abs(args.c2_summary),
    }
    for boundary, summary_path in summary_by_boundary.items():
        summary = load_json(summary_path)
        ref_summary = summary.get("reference_summary_path")
        reference_overlay_maps[boundary] = load_reference_overlay_map(ensure_abs(ref_summary)) if ref_summary else {}

    key_frames = [parse_key_frame(x) for x in args.key_frame]
    key_outputs: list[dict[str, Any]] = []
    for boundary, frame_id in key_frames:
        summary_path = summary_by_boundary.get(boundary)
        if summary_path is None:
            raise ValueError(f"unsupported boundary: {boundary}")
        summary = load_json(summary_path)
        records = {int(r["frame_id"]): r for r in summary.get("records", [])}
        record = records.get(frame_id)
        if record is None:
            raise ValueError(f"frame {frame_id} not found in {boundary}: {summary_path}")

        cov = coverage_records.get((boundary, frame_id), {})
        pred_overlay = ensure_abs(record["overlay_path"])
        pred_label = ensure_abs(record["label_path"])
        ref_overlay = reference_overlay_maps.get(boundary, {}).get(frame_id)
        ref_label = ensure_abs(cov["reference_label_path"]) if cov.get("reference_label_path") else ensure_abs(record["reference_label_path"])
        diff_image = ensure_abs(cov["diff_image_path"]) if cov.get("diff_image_path") else None

        panel_path = output_root / "panels" / f"{boundary}_frame_{frame_id:06d}_panel.jpg"
        build_panel(
            frame_id=frame_id,
            boundary=boundary,
            pred_overlay=pred_overlay,
            ref_overlay=ref_overlay,
            pred_label=pred_label,
            ref_label=ref_label,
            diff_image=diff_image,
            output_path=panel_path,
        )

        fullres_dir = output_root / "fullres" / boundary / f"frame_{frame_id:06d}"
        fullres = {
            "pred_overlay": copy_fullres(pred_overlay, fullres_dir / pred_overlay.name),
            "pred_label": copy_fullres(pred_label, fullres_dir / pred_label.name),
            "panel": {"path": repo_rel(panel_path), "sha256": sha256_file(panel_path)},
        }
        if ref_overlay and ref_overlay.exists():
            fullres["reference_overlay"] = copy_fullres(ref_overlay, fullres_dir / f"reference_{ref_overlay.name}")
        if ref_label and ref_label.exists():
            fullres["reference_label"] = copy_fullres(ref_label, fullres_dir / f"reference_{ref_label.name}")
        if diff_image and diff_image.exists():
            fullres["coverage_diff"] = copy_fullres(diff_image, fullres_dir / diff_image.name)

        key_outputs.append(
            {
                "boundary": boundary,
                "frame_id": frame_id,
                "coverage_record": cov,
                "artifacts": fullres,
            }
        )

    summary = {
        "schema_version": "stream4d_v106_phase9_visual_confirmation_casebook_v1",
        "output_root": repo_rel(output_root),
        "scene_segment_video": video_info,
        "scene_segment_unique_frame_count": len(scene_frames),
        "overlap_frames_skipped_per_later_chunk": args.overlap,
        "key_frame_count": len(key_outputs),
        "key_frames": key_outputs,
        "inputs": {
            "c0_summary": repo_rel(ensure_abs(args.c0_summary)),
            "c1_summary": repo_rel(ensure_abs(args.c1_summary)),
            "c2_summary": repo_rel(ensure_abs(args.c2_summary)),
            "coverage_records": repo_rel(ensure_abs(args.coverage_records)) if args.coverage_records else None,
        },
        **meta,
    }
    write_json(output_root / "visual_confirmation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--c0-summary", required=True)
    parser.add_argument("--c1-summary", required=True)
    parser.add_argument("--c2-summary", required=True)
    parser.add_argument("--coverage-records")
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--scene-video-name", default="scene0050_00_three_chunk_unique90_visual_confirmation.mp4")
    parser.add_argument(
        "--key-frame",
        action="append",
        default=[],
        help="Boundary/frame pair such as boundary_c0_c1:4325. Can be repeated.",
    )
    args = parser.parse_args()
    if not args.key_frame:
        args.key_frame = [
            "boundary_c0_c1:4305",
            "boundary_c0_c1:4315",
            "boundary_c0_c1:4320",
            "boundary_c0_c1:4325",
            "boundary_c1_c2:4450",
            "boundary_c1_c2:4555",
        ]
    summary = build(args)
    print(
        json.dumps(
            {
                "output_root": summary["output_root"],
                "scene_segment_video": summary["scene_segment_video"],
                "scene_segment_unique_frame_count": summary["scene_segment_unique_frame_count"],
                "key_frame_count": summary["key_frame_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
