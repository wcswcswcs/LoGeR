#!/usr/bin/env python3
"""Build full-resolution v107 foreground overlay diagnostics.

Legend:
- red: candidate/v107 foreground only
- green: reference/v106 foreground only
- yellow: foreground overlap

This is an audit-only helper. It reads existing summaries and label PNGs and
does not modify any tracking output.
"""

from __future__ import annotations

import argparse
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

from tools.build_v106_readable_visual_inspection import (  # noqa: E402
    DEFAULT_RGB_ROOT,
    parse_ints,
    read_label,
    read_rgb,
    repo_rel,
    resolve_path,
    resolve_rgb_path,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


def load_summary(path_text: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(path_text)
    return path, json.loads(path.read_text(encoding="utf-8"))


def records_by_frame(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["frame_id"]): dict(row) for row in summary.get("records", [])}


def overlay_foreground(rgb: np.ndarray, candidate_label: np.ndarray, reference_label: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cand_fg = candidate_label > 0
    ref_fg = reference_label > 0
    overlap = cand_fg & ref_fg
    candidate_only = cand_fg & ~ref_fg
    reference_only = ref_fg & ~cand_fg

    canvas = rgb.copy()
    muted = (canvas.astype(np.float32) * 0.55).astype(np.uint8)
    canvas[:, :] = muted
    canvas[overlap] = np.array([255, 220, 0], dtype=np.uint8)
    canvas[candidate_only] = np.array([255, 30, 30], dtype=np.uint8)
    canvas[reference_only] = np.array([0, 220, 30], dtype=np.uint8)

    stats = {
        "candidate_px": int(np.count_nonzero(cand_fg)),
        "reference_px": int(np.count_nonzero(ref_fg)),
        "overlap_px": int(np.count_nonzero(overlap)),
        "candidate_only_px": int(np.count_nonzero(candidate_only)),
        "ref_only_px": int(np.count_nonzero(reference_only)),
    }
    return canvas, stats


def disagreement_crop_bbox(candidate_label: np.ndarray, reference_label: np.ndarray, margin: int) -> list[int]:
    disagreement = (candidate_label > 0) ^ (reference_label > 0)
    ys, xs = np.where(disagreement)
    h, w = candidate_label.shape[:2]
    if len(xs) == 0:
        return [0, 0, int(w), int(h)]
    x0 = max(0, int(xs.min()) - int(margin))
    y0 = max(0, int(ys.min()) - int(margin))
    x1 = min(int(w), int(xs.max()) + int(margin) + 1)
    y1 = min(int(h), int(ys.max()) + int(margin) + 1)
    return [x0, y0, x1, y1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--tag", default="v107")
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--crop-margin", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_summary_path, candidate_summary = load_summary(args.candidate_summary)
    reference_summary_path, reference_summary = load_summary(args.reference_summary)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_root = resolve_path(args.rgb_root)

    candidate_records = records_by_frame(candidate_summary)
    reference_records = records_by_frame(reference_summary)
    frame_ids = parse_ints(str(args.frame_ids))
    if not frame_ids:
        raise ValueError("--frame-ids must contain at least one frame id")

    rows: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        if int(frame_id) not in candidate_records:
            raise KeyError(f"candidate summary missing frame_id={frame_id}")
        if int(frame_id) not in reference_records:
            raise KeyError(f"reference summary missing frame_id={frame_id}")
        candidate_row = candidate_records[int(frame_id)]
        reference_row = reference_records[int(frame_id)]

        rgb = read_rgb(resolve_rgb_path(reference_summary, reference_row, rgb_root))
        candidate_label = read_label(resolve_path(str(candidate_row["label_path"])), rgb.shape[:2])
        reference_label = read_label(resolve_path(str(reference_row["label_path"])), rgb.shape[:2])
        if candidate_label.shape != reference_label.shape:
            candidate_label = cv2.resize(
                candidate_label,
                (reference_label.shape[1], reference_label.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if candidate_label.shape[:2] != rgb.shape[:2]:
            candidate_label = cv2.resize(candidate_label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            reference_label = cv2.resize(reference_label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay, stats = overlay_foreground(rgb, candidate_label, reference_label)
        x0, y0, x1, y1 = disagreement_crop_bbox(candidate_label, reference_label, margin=int(args.crop_margin))
        base = f"frame_{int(frame_id):06d}_{args.tag}_candidate_ref_foreground_overlay"
        full_path = out_dir / f"{base}_full.png"
        crop_path = out_dir / f"{base}_disagreement_crop.png"
        raw_crop_path = out_dir / f"{base}_raw_crop.png"
        write_rgb_png(full_path, overlay)
        write_rgb_png(crop_path, overlay[y0:y1, x0:x1])
        write_rgb_png(raw_crop_path, rgb[y0:y1, x0:x1])

        row = {
            "frame_id": int(frame_id),
            "candidate_summary": repo_rel(candidate_summary_path),
            "candidate_summary_sha256": sha256_file(candidate_summary_path),
            "reference_summary": repo_rel(reference_summary_path),
            "reference_summary_sha256": sha256_file(reference_summary_path),
            "candidate_label_path": repo_rel(resolve_path(str(candidate_row["label_path"]))),
            "reference_label_path": repo_rel(resolve_path(str(reference_row["label_path"]))),
            "overlay_convention": "red=v107_only green=v106_only yellow=overlap over RGB",
            "crop_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "full_overlay_path": repo_rel(full_path),
            "full_overlay_sha256": sha256_file(full_path),
            "disagreement_crop_path": repo_rel(crop_path),
            "disagreement_crop_sha256": sha256_file(crop_path),
            "raw_crop_path": repo_rel(raw_crop_path),
            "raw_crop_sha256": sha256_file(raw_crop_path),
            **stats,
        }
        stats_path = out_dir / f"{base}_stats.json"
        row["stats_path"] = repo_rel(stats_path)
        stats_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        row["stats_sha256"] = sha256_file(stats_path)
        rows.append(row)

    manifest = {
        "schema_version": "stream4d_v107_foreground_overlay_visuals_v1",
        "candidate_summary": repo_rel(candidate_summary_path),
        "candidate_summary_sha256": sha256_file(candidate_summary_path),
        "reference_summary": repo_rel(reference_summary_path),
        "reference_summary_sha256": sha256_file(reference_summary_path),
        "out_dir": repo_rel(out_dir),
        "frame_ids": [int(v) for v in frame_ids],
        "overlay_convention": "red=v107_only green=v106_only yellow=overlap over RGB",
        "rows": rows,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "frame_count": len(rows),
                "out_dir": str(out_dir),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
