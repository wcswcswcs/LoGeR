#!/usr/bin/env python3
"""Build a high-resolution label-backed visual for a single v107 event."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_points(
    points_json: Path,
    *,
    source_frame_id: int,
    target_frame_id: int,
    target_obj_id: int,
) -> list[dict]:
    rows = json.loads(points_json.read_text())["rows"]
    return [
        row
        for row in rows
        if row.get("source_frame_id") == source_frame_id
        and row.get("target_frame_id") == target_frame_id
        and row.get("target_obj_id") == target_obj_id
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--rgb-dir", default="Stream3D/data/scannet/processed/scene0050_00/color", type=Path)
    parser.add_argument("--points-json", required=True, type=Path)
    parser.add_argument("--frame-id", required=True, type=int)
    parser.add_argument("--source-frame-id", required=True, type=int)
    parser.add_argument("--reference-global-id", required=True, type=int)
    parser.add_argument("--target-obj-id", required=True, type=int)
    parser.add_argument("--live-obj-id", required=True, type=int)
    parser.add_argument("--label-value", required=True, type=int)
    parser.add_argument("--lingbot-width", default=518.0, type=float)
    parser.add_argument("--lingbot-height", default=392.0, type=float)
    parser.add_argument("--pad", default=260, type=int)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    rgb_path = args.rgb_dir / f"{args.frame_id}.jpg"
    label_path = (
        args.root
        / "v107_phase8_g3_rolling_scheduler_smoke"
        / "labels"
        / f"frame_{args.frame_id:06d}.png"
    )
    output_path = args.output_path or (
        args.root
        / "highres_event_visuals"
        / (
            f"event023_confirm_f{args.frame_id}_ref{args.reference_global_id}"
            f"_live{args.live_obj_id}_label{args.label_value}_pad{args.pad}_labelbacked.jpg"
        )
    )

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if label is None:
        raise FileNotFoundError(label_path)
    height, width = rgb.shape[:2]

    point_rows = _load_points(
        args.points_json,
        source_frame_id=args.source_frame_id,
        target_frame_id=args.frame_id,
        target_obj_id=args.target_obj_id,
    )
    positive_rows = [row for row in point_rows if row.get("role") == "positive"]
    negative_rows = [row for row in point_rows if row.get("role") == "negative"]

    def scale_xy(row: dict) -> tuple[float, float]:
        return (
            float(row["target_x"]) * width / args.lingbot_width,
            float(row["target_y"]) * height / args.lingbot_height,
        )

    points = [scale_xy(row) for row in point_rows]
    mask = label == args.label_value
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"label value {args.label_value} is absent in {label_path}")

    xmin = min(float(xs.min()), *(x for x, _ in points))
    xmax = max(float(xs.max()), *(x for x, _ in points))
    ymin = min(float(ys.min()), *(y for _, y in points))
    ymax = max(float(ys.max()), *(y for _, y in points))
    x0 = max(0, int(np.floor(xmin)) - args.pad)
    y0 = max(0, int(np.floor(ymin)) - args.pad)
    x1 = min(width, int(np.ceil(xmax)) + args.pad + 1)
    y1 = min(height, int(np.ceil(ymax)) + args.pad + 1)

    visual = rgb.copy()
    overlay = visual.copy()
    overlay[mask] = (255, 255, 0)
    visual = cv2.addWeighted(overlay, 0.48, visual, 0.52, 0)

    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visual, contours, -1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    for row in positive_rows:
        x, y = scale_xy(row)
        point = (int(round(x)), int(round(y)))
        cv2.circle(visual, point, 11, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(visual, point, 8, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for row in negative_rows:
        x, y = scale_xy(row)
        point = (int(round(x)), int(round(y)))
        cv2.circle(visual, point, 11, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.line(visual, (point[0] - 8, point[1] - 8), (point[0] + 8, point[1] + 8), (0, 0, 255), 3, lineType=cv2.LINE_AA)
        cv2.line(visual, (point[0] - 8, point[1] + 8), (point[0] + 8, point[1] - 8), (0, 0, 255), 3, lineType=cv2.LINE_AA)

    text = (
        f"event23 ref{args.reference_global_id} live{args.live_obj_id} "
        f"label{args.label_value} f{args.frame_id} "
        f"pos={len(positive_rows)} neg={len(negative_rows)} "
        f"LingBotMap {int(args.lingbot_width)}x{int(args.lingbot_height)}->{width}x{height}"
    )
    cv2.rectangle(visual, (x0 + 8, y0 + 8), (min(x1 - 8, x0 + 980), y0 + 42), (0, 0, 0), -1)
    cv2.putText(visual, text, (x0 + 16, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    crop = visual[y0:y1, x0:x1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"failed to write {output_path}")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "sha256": _sha256(output_path),
                "size_bytes": output_path.stat().st_size,
                "crop_xyxy": [x0, y0, x1, y1],
                "crop_shape": list(crop.shape),
                "mask_px": int(mask.sum()),
                "positive_points": len(positive_rows),
                "negative_points": len(negative_rows),
                "negative_source_obj_ids": [row.get("source_obj_id") for row in negative_rows],
                "positive_target_label_hits": sum(1 for row in positive_rows if row.get("target_label_at_point") == args.target_obj_id),
                "negative_target_label_values": [row.get("target_label_at_point") for row in negative_rows],
                "pose_modes": sorted({row.get("pose_mode") for row in point_rows}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
