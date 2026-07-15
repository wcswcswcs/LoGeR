#!/usr/bin/env python3
"""Build high-resolution per-event visual audit crops for v107 Phase8 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


PANEL_NAMES = ("prompt", "G1_pos", "G2_pos_neg")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def parse_events(text: str) -> list[int]:
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def find_event_image(pilot_root: Path, event_index: int) -> Path:
    matches = sorted((pilot_root / "visual_overlays").glob(f"event{event_index:03d}_*.jpg"))
    if not matches:
        raise FileNotFoundError(f"no event{event_index:03d} image in {pilot_root / 'visual_overlays'}")
    return matches[0]


def yellow_bbox(panel_rgb: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    # Reference contours are drawn in yellow in the Phase8 pilot composites.
    r = panel_rgb[:, :, 0]
    g = panel_rgb[:, :, 1]
    b = panel_rgb[:, :, 2]
    mask = (r > 180) & (g > 180) & (b < 120)
    ys, xs = np.where(mask)
    h, w = panel_rgb.shape[:2]
    if len(xs) == 0:
        return 0, 0, w, h
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(w, int(xs.max()) + int(pad) + 1)
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(h, int(ys.max()) + int(pad) + 1)
    min_w = min(w, max(96, int(0.22 * w)))
    min_h = min(h, max(96, int(0.22 * h)))
    if x1 - x0 < min_w:
        c = (x0 + x1) // 2
        x0 = max(0, c - min_w // 2)
        x1 = min(w, x0 + min_w)
        x0 = max(0, x1 - min_w)
    if y1 - y0 < min_h:
        c = (y0 + y1) // 2
        y0 = max(0, c - min_h // 2)
        y1 = min(h, y0 + min_h)
        y0 = max(0, y1 - min_h)
    return x0, y0, x1, y1


def split_event_image(path: Path, header_px: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    body = rgb[int(header_px) :]
    h, w = body.shape[:2]
    panel_h = h // 2
    panel_w = w // 3
    attempt = [body[0:panel_h, i * panel_w : (i + 1) * panel_w] for i in range(3)]
    confirm = [body[panel_h : 2 * panel_h, i * panel_w : (i + 1) * panel_w] for i in range(3)]
    return attempt, confirm


def concat_with_labels(panels: list[np.ndarray], labels: list[str], scale: int) -> np.ndarray:
    labeled = []
    for panel, label in zip(panels, labels):
        header_h = 34
        item = np.zeros((panel.shape[0] + header_h, panel.shape[1], 3), dtype=np.uint8)
        item[:] = 18
        item[header_h:] = panel
        cv2.putText(item, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        labeled.append(item)
    merged = np.concatenate(labeled, axis=1)
    if int(scale) != 1:
        merged = cv2.resize(merged, (merged.shape[1] * int(scale), merged.shape[0] * int(scale)), interpolation=cv2.INTER_CUBIC)
    return merged


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 96])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--header-px", type=int, default=32)
    parser.add_argument("--pad", type=int, default=80)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    pilot_root = Path(args.pilot_root)
    if not pilot_root.is_absolute():
        pilot_root = ROOT / pilot_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    rows: list[dict[str, Any]] = []
    for event_index in parse_events(args.events):
        source = find_event_image(pilot_root, event_index)
        attempt, confirm = split_event_image(source, int(args.header_px))
        for row_name, panels in (("attempt", attempt), ("confirm", confirm)):
            x0, y0, x1, y1 = yellow_bbox(panels[0], int(args.pad))
            crops = [panel[y0:y1, x0:x1] for panel in panels]
            out = concat_with_labels(crops, [f"{row_name}:{name}" for name in PANEL_NAMES], int(args.scale))
            out_path = output_root / f"event{event_index:03d}_{row_name}_target_zoom_x{int(args.scale)}.jpg"
            write_rgb(out_path, out)
            rows.append(
                {
                    "event_index": int(event_index),
                    "row": row_name,
                    "source_event_image": rel(source),
                    "output": rel(out_path),
                    "crop_xyxy_in_panel": [int(x0), int(y0), int(x1), int(y1)],
                    "scale": int(args.scale),
                    "note": "crop around yellow reference contour; columns are prompt, G1_pos, G2_pos_neg",
                }
            )
    manifest = {
        "schema_version": "stream4d_v107_phase8_single_event_visual_audit_v1",
        "pilot_root": rel(pilot_root),
        "event_indices": parse_events(args.events),
        "rows": rows,
    }
    manifest_path = output_root / "single_event_visual_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "image_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
