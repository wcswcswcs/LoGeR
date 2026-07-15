#!/usr/bin/env python3
"""Build full-frame visual audit sheets for Stream4D v105 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENES = ("scene0011_00", "scene0050_00")
INPUT_FRAME_INDICES = [i * 5 for i in range(32)]
EXPECTED_FRAME_COUNT = len(INPUT_FRAME_INDICES)

OLD_MATRIX_ROOT = Path("Stream3D/outputs/audit/v105_baseline_matrix_visual_videos_sam31_fixed_r3")
B3_ROOT = Path("Stream3D/outputs/audit/v105_sam31_shape_continuity_twoscenes_gpu7_r13")
B4_FASTSAMX_ROOT = Path("Stream3D/outputs/audit/v105_sgq_fastsamx_b4_gpu3_r20c")
B6_FASTSAMX_ROOT = Path("Stream3D/outputs/audit/v105_sgq_fastsamx_b6_gpu4_r20c")
B8_FASTSAMX_ROOT = Path("Stream3D/outputs/audit/v105_sgq_fastsamx_b8_gpu5_r20c")

BASELINES: dict[str, dict[str, str]] = {
    "B0": {
        "name": "cropformer_only",
        "stem": "B0_cropformer_only",
        "root": str(OLD_MATRIX_ROOT),
    },
    "B1": {
        "name": "fastsamx_only",
        "stem": "B1_fastsam_only",
        "root": str(B4_FASTSAMX_ROOT),
    },
    "B2": {
        "name": "4dpm_sam2_gap_tracking",
        "stem": "B2_4dpm_sam2_gap_tracking",
        "root": str(OLD_MATRIX_ROOT),
    },
    "B3": {
        "name": "cropformer_sam31_tracking",
        "stem": "B3_cropformer_sam31_tracking",
        "root": str(B3_ROOT),
    },
    "B4": {
        "name": "fastsamx_sam31_tracking",
        "stem": "B4_fastsam_sam31_tracking",
        "root": str(B4_FASTSAMX_ROOT),
    },
    "B5": {
        "name": "cropformer_sam2_video_tracker",
        "stem": "B5_cropformer_sam2_video_tracker",
        "root": str(OLD_MATRIX_ROOT),
    },
    "B6": {
        "name": "fastsamx_sam2_video_tracker",
        "stem": "B6_fastsam_sam2_video_tracker",
        "root": str(B6_FASTSAMX_ROOT),
    },
    "B7": {
        "name": "cropformer_edgetam_video_tracker",
        "stem": "B7_cropformer_edgetam_video_tracker",
        "root": str(OLD_MATRIX_ROOT),
    },
    "B8": {
        "name": "fastsamx_edgetam_video_tracker",
        "stem": "B8_fastsam_edgetam_video_tracker",
        "root": str(B8_FASTSAMX_ROOT),
    },
}


def rel(path: Path) -> str:
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


def source_video_path(baseline_id: str, scene: str) -> Path:
    spec = BASELINES[baseline_id]
    return REPO_ROOT / spec["root"] / "baselines" / "videos" / f"{spec['stem']}_{scene}.mp4"


def read_all_frames(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"opened": False, "frame_count_property": None, "fps": None}
    meta = {
        "opened": True,
        "frame_count_property": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, meta


def make_sheet(
    frames: list[np.ndarray],
    output_path: Path,
    *,
    baseline_id: str,
    baseline_name: str,
    scene: str,
    source_rel: str,
    thumb_width: int = 300,
    columns: int = 8,
) -> None:
    if not frames:
        raise ValueError("cannot make a sheet with no frames")

    first_h, first_w = frames[0].shape[:2]
    thumb_height = max(1, int(round(first_h * (thumb_width / max(first_w, 1)))))
    label_height = 28
    header_height = 42
    pad = 8
    rows = (len(frames) + columns - 1) // columns
    sheet_h = header_height + rows * (thumb_height + label_height + pad) + pad
    sheet_w = columns * (thumb_width + pad) + pad
    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)

    title = f"{baseline_id} {baseline_name} {scene} | all decoded frames | source: {source_rel}"
    cv2.putText(sheet, title[:190], (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        x = pad + col * (thumb_width + pad)
        y = header_height + row * (thumb_height + label_height + pad)
        frame_idx = INPUT_FRAME_INDICES[idx] if idx < len(INPUT_FRAME_INDICES) else None
        label = f"video_idx={idx:02d} input_frame={frame_idx if frame_idx is not None else 'extra'}"
        cv2.rectangle(sheet, (x, y), (x + thumb_width, y + label_height), (230, 230, 230), -1)
        cv2.putText(sheet, label, (x + 6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
        resized = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        y2 = y + label_height
        sheet[y2 : y2 + thumb_height, x : x + thumb_width] = resized

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"failed to write {output_path}")


def link_or_copy_video(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        data = src.read_bytes()
        dst.write_bytes(data)
    else:
        target = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        dst.symlink_to(target)


def build(output_root: Path, *, copy_videos: bool) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    sheet_dir = output_root / "frame_sheets"
    video_dir = output_root / "videos"
    rows: list[dict[str, Any]] = []

    for baseline_id in sorted(BASELINES):
        baseline_name = BASELINES[baseline_id]["name"]
        for scene in SCENES:
            src = source_video_path(baseline_id, scene)
            dst_name = f"{baseline_id}_{baseline_name}_{scene}.mp4"
            dst = video_dir / dst_name
            sheet = sheet_dir / f"{baseline_id}_{baseline_name}_{scene}_all32.jpg"
            row: dict[str, Any] = {
                "baseline_id": baseline_id,
                "baseline_name": baseline_name,
                "scene": scene,
                "source_video": rel(src),
                "unified_video": rel(dst),
                "sheet": rel(sheet),
                "expected_input_frame_indices": INPUT_FRAME_INDICES,
                "expected_frame_count": EXPECTED_FRAME_COUNT,
            }
            if not src.exists():
                row.update({"status": "missing_video", "frame_count": 0, "frame_count_pass": False})
                rows.append(row)
                continue

            link_or_copy_video(src, dst, copy=copy_videos)
            frames, meta = read_all_frames(src)
            frame_count = len(frames)
            row.update(meta)
            row.update(
                {
                    "status": "decoded" if frame_count else "decode_empty",
                    "frame_count": frame_count,
                    "frame_count_pass": frame_count == EXPECTED_FRAME_COUNT,
                    "input_stride_only_claim": "video_frames_are_expected_to_map_to_stride5_input_frames_0_5_..._155",
                    "source_sha256": sha256_file(src),
                }
            )
            if frames:
                make_sheet(
                    frames,
                    sheet,
                    baseline_id=baseline_id,
                    baseline_name=baseline_name,
                    scene=scene,
                    source_rel=rel(src),
                )
                row["sheet_sha256"] = sha256_file(sheet)
            rows.append(row)

    missing = [r for r in rows if r["status"] == "missing_video"]
    bad_counts = [r for r in rows if not r.get("frame_count_pass")]
    summary = {
        "schema": "stream4d_v105_baseline_visual_audit_v1",
        "status": "pass" if not missing and not bad_counts else "partial",
        "output_root": rel(output_root),
        "copy_videos": copy_videos,
        "expected_video_count": len(BASELINES) * len(SCENES),
        "decoded_video_count": sum(1 for r in rows if r["status"] == "decoded"),
        "missing_video_count": len(missing),
        "bad_frame_count_video_count": len(bad_counts),
        "expected_frame_count_per_video": EXPECTED_FRAME_COUNT,
        "input_frame_indices": INPUT_FRAME_INDICES,
        "baseline_sources": BASELINES,
        "videos": rows,
    }
    (output_root / "visual_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_root / "video_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="Stream3D/outputs/audit/v105_fastsamx_final_baseline_visual_audit_r20c",
    )
    parser.add_argument("--copy-videos", action="store_true", help="copy videos instead of creating symlinks")
    args = parser.parse_args()
    summary = build(REPO_ROOT / args.output_root, copy_videos=args.copy_videos)
    print(json.dumps({k: summary[k] for k in ("status", "decoded_video_count", "missing_video_count", "bad_frame_count_video_count", "output_root")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
