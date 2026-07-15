#!/usr/bin/env python3
"""Build full-size pairwise visual reviews for two v106 scene streams.

This is intentionally different from the compact multi-panel review videos:
each exported video shows one question only, at the original frame resolution.
The main difference view uses:

- red: covered by the reference run but missing from the candidate run
- cyan: covered by the candidate run but absent from the reference run
- gray: covered by both runs

The tool does not decide pass/fail by itself. It makes the visual evidence
large enough to inspect honestly.
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

from tools.build_v106_readable_visual_inspection import (  # noqa: E402
    DEFAULT_RGB_ROOT,
    draw_text,
    object_stats,
    overlay_strong,
    parse_ints,
    read_label,
    read_rgb,
    repo_rel,
    resolve_path,
    resolve_rgb_path,
)


_EDGE_KERNEL_3 = np.ones((3, 3), dtype=np.uint8)
_EDGE_KERNEL_5 = np.ones((5, 5), dtype=np.uint8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def blend_region(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    color_arr = np.asarray(color, dtype=np.float32)
    src = base[mask].astype(np.float32)
    base[mask] = np.clip(src * (1.0 - float(alpha)) + color_arr * float(alpha), 0, 255).astype(np.uint8)


def mask_edges(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, _EDGE_KERNEL_3) > 0
    return cv2.dilate(edge.astype(np.uint8), _EDGE_KERNEL_3, iterations=int(iterations)).astype(bool)


def connected_components(mask: np.ndarray, *, min_area: int, max_count: int) -> list[dict[str, Any]]:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    rows: list[dict[str, Any]] = []
    for idx in range(1, int(count)):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]
        rows.append(
            {
                "area": area,
                "bbox_xyxy": [x, y, x + w, y + h],
                "centroid_xy": [float(cx), float(cy)],
            }
        )
    rows.sort(key=lambda row: int(row["area"]), reverse=True)
    return rows[: int(max_count)]


def draw_components(
    img: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    bgr_color = tuple(int(v) for v in color)
    for rank, row in enumerate(rows, start=1):
        x0, y0, x1, y1 = [int(v) for v in row["bbox_xyxy"]]
        cv2.rectangle(img, (x0, y0), (x1, y1), bgr_color, 4)
        draw_text(img, f"{label_prefix}{rank} {int(row['area'])}", (x0 + 6, min(y1 - 8, y0 + 26)), scale=0.58)


def difference_views(
    *,
    rgb: np.ndarray,
    reference_label: np.ndarray,
    candidate_label: np.ndarray,
    min_component_area: int,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ref_fg = reference_label > 0
    cand_fg = candidate_label > 0
    both = ref_fg & cand_fg
    ref_only = ref_fg & ~cand_fg
    cand_only = cand_fg & ~ref_fg

    diff = (rgb.astype(np.float32) * 0.62).astype(np.uint8)
    blend_region(diff, both, (150, 150, 150), 0.45)
    blend_region(diff, ref_only, (255, 30, 30), 0.82)
    blend_region(diff, cand_only, (0, 220, 255), 0.80)
    diff[mask_edges(ref_only, iterations=1)] = np.array([255, 255, 255], dtype=np.uint8)
    diff[mask_edges(cand_only, iterations=1)] = np.array([0, 255, 255], dtype=np.uint8)

    missing_rows = connected_components(ref_only, min_area=min_component_area, max_count=max_components)
    extra_rows = connected_components(cand_only, min_area=min_component_area, max_count=max_components)
    draw_components(diff, missing_rows, color=(255, 30, 30), label_prefix="miss ")
    draw_components(diff, extra_rows, color=(0, 220, 255), label_prefix="extra ")
    draw_text(diff, "red=reference-only missing | cyan=candidate-only extra | gray=both", (18, 34), scale=0.72)

    diff_mask = np.zeros_like(rgb)
    diff_mask[both] = np.array([112, 112, 112], dtype=np.uint8)
    diff_mask[ref_only] = np.array([255, 30, 30], dtype=np.uint8)
    diff_mask[cand_only] = np.array([0, 220, 255], dtype=np.uint8)
    diff_mask[mask_edges(ref_only, iterations=1)] = np.array([255, 255, 255], dtype=np.uint8)
    diff_mask[mask_edges(cand_only, iterations=1)] = np.array([0, 255, 255], dtype=np.uint8)
    draw_components(diff_mask, missing_rows, color=(255, 30, 30), label_prefix="miss ")
    draw_components(diff_mask, extra_rows, color=(0, 220, 255), label_prefix="extra ")

    missing = rgb.copy()
    missing[~ref_only] = (missing[~ref_only].astype(np.float32) * 0.42).astype(np.uint8)
    blend_region(missing, ref_only, (255, 30, 30), 0.82)
    missing[mask_edges(ref_only, iterations=2)] = np.array([255, 255, 255], dtype=np.uint8)
    draw_components(missing, missing_rows, color=(255, 30, 30), label_prefix="miss ")
    draw_text(missing, "reference-only area: candidate missed these pixels", (18, 34), scale=0.72)

    extra = rgb.copy()
    extra[~cand_only] = (extra[~cand_only].astype(np.float32) * 0.42).astype(np.uint8)
    blend_region(extra, cand_only, (0, 220, 255), 0.80)
    extra[mask_edges(cand_only, iterations=2)] = np.array([0, 255, 255], dtype=np.uint8)
    draw_components(extra, extra_rows, color=(0, 220, 255), label_prefix="extra ")
    draw_text(extra, "candidate-only area: candidate added these pixels", (18, 34), scale=0.72)

    stats = {
        "reference_pixels": int(np.count_nonzero(ref_fg)),
        "candidate_pixels": int(np.count_nonzero(cand_fg)),
        "both_pixels": int(np.count_nonzero(both)),
        "reference_only_pixels": int(np.count_nonzero(ref_only)),
        "candidate_only_pixels": int(np.count_nonzero(cand_only)),
        "reference_only_large_components": missing_rows,
        "candidate_only_large_components": extra_rows,
    }
    return diff, diff_mask, missing, extra, stats


def selected_frame_keys(records: list[dict[str, Any]], frame_indices: list[int], frame_ids: list[int]) -> set[int]:
    if not frame_indices and not frame_ids:
        frame_indices = [0, len(records) // 2, len(records) - 1]
    by_index = {int(row["chunk_frame_index"]): int(row["frame_id"]) for row in records}
    by_frame_id = {int(row["frame_id"]): int(row["frame_id"]) for row in records}
    keys: set[int] = set()
    for idx in frame_indices:
        if int(idx) in by_index:
            keys.add(by_index[int(idx)])
    for frame_id in frame_ids:
        if int(frame_id) in by_frame_id:
            keys.add(by_frame_id[int(frame_id)])
    return keys


def aligned_pairs(reference_summary: dict[str, Any], candidate_summary: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ref_records = list(reference_summary.get("records", []))
    cand_records = list(candidate_summary.get("records", []))
    cand_by_frame = {int(row["frame_id"]): row for row in cand_records}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ref_row in ref_records:
        cand_row = cand_by_frame.get(int(ref_row["frame_id"]))
        if cand_row is not None:
            pairs.append((ref_row, cand_row))
    pairs.sort(key=lambda pair: int(pair[0]["chunk_frame_index"]))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--frame-indices", default="0,33,45,71,89")
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--overlay-alpha", type=float, default=0.58)
    parser.add_argument("--id-min-area", type=int, default=10000)
    parser.add_argument("--min-component-area", type=int, default=12000)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--skip-videos", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_summary_path = resolve_path(args.reference_summary)
    candidate_summary_path = resolve_path(args.candidate_summary)
    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    pairs = aligned_pairs(reference_summary, candidate_summary)
    if not pairs:
        raise ValueError("reference and candidate summaries have no aligned frame_id records")

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    videos_dir = out_dir / "videos"
    rgb_root = resolve_path(args.rgb_root)
    selected_keys = selected_frame_keys(
        list(reference_summary.get("records", [])),
        parse_ints(str(args.frame_indices)),
        parse_ints(str(args.frame_ids)),
    )

    video_specs = {
        "reference_overlay": videos_dir / f"reference_overlay_{len(pairs)}f.mp4",
        "candidate_overlay": videos_dir / f"candidate_overlay_{len(pairs)}f.mp4",
        "difference_on_rgb": videos_dir / f"difference_on_rgb_{len(pairs)}f.mp4",
        "difference_mask": videos_dir / f"difference_mask_{len(pairs)}f.mp4",
        "reference_only_missing": videos_dir / f"reference_only_missing_{len(pairs)}f.mp4",
        "candidate_only_extra": videos_dir / f"candidate_only_extra_{len(pairs)}f.mp4",
    }
    writers: dict[str, cv2.VideoWriter] = {}
    sizes: dict[str, tuple[int, int]] = {}
    stats_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    try:
        for ref_row, cand_row in pairs:
            frame_index = int(ref_row["chunk_frame_index"])
            frame_id = int(ref_row["frame_id"])
            rgb = read_rgb(resolve_rgb_path(reference_summary, ref_row, rgb_root))
            ref_label = read_label(resolve_path(str(ref_row["label_path"])), rgb.shape[:2])
            cand_label = read_label(resolve_path(str(cand_row["label_path"])), rgb.shape[:2])
            ref_stats = object_stats(ref_label)
            cand_stats = object_stats(cand_label)
            ref_overlay = overlay_strong(
                rgb,
                ref_label,
                alpha=float(args.overlay_alpha),
                stats=ref_stats,
                id_min_area=int(args.id_min_area),
            )
            cand_overlay = overlay_strong(
                rgb,
                cand_label,
                alpha=float(args.overlay_alpha),
                stats=cand_stats,
                id_min_area=int(args.id_min_area),
            )
            diff, diff_mask, missing, extra, diff_stats = difference_views(
                rgb=rgb,
                reference_label=ref_label,
                candidate_label=cand_label,
                min_component_area=int(args.min_component_area),
                max_components=int(args.max_components),
            )
            row = {
                "frame_index": frame_index,
                "frame_id": frame_id,
                "reference_visible_count": len(ref_stats),
                "candidate_visible_count": len(cand_stats),
                "reference_pixels": int(diff_stats["reference_pixels"]),
                "candidate_pixels": int(diff_stats["candidate_pixels"]),
                "both_pixels": int(diff_stats["both_pixels"]),
                "reference_only_pixels": int(diff_stats["reference_only_pixels"]),
                "candidate_only_pixels": int(diff_stats["candidate_only_pixels"]),
                "reference_only_large_component_count": len(diff_stats["reference_only_large_components"]),
                "candidate_only_large_component_count": len(diff_stats["candidate_only_large_components"]),
            }
            stats_rows.append(row)
            for kind, rows in (
                ("reference_only", diff_stats["reference_only_large_components"]),
                ("candidate_only", diff_stats["candidate_only_large_components"]),
            ):
                for rank, component in enumerate(rows, start=1):
                    component_rows.append(
                        {
                            "frame_index": frame_index,
                            "frame_id": frame_id,
                            "kind": kind,
                            "rank": rank,
                            "area": int(component["area"]),
                            "bbox_xyxy": json.dumps(component["bbox_xyxy"]),
                        }
                    )

            video_frames = {
                "reference_overlay": ref_overlay,
                "candidate_overlay": cand_overlay,
                "difference_on_rgb": diff,
                "difference_mask": diff_mask,
                "reference_only_missing": missing,
                "candidate_only_extra": extra,
            }
            if not bool(args.skip_videos):
                for mode, frame in video_frames.items():
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if mode not in writers:
                        height, width = bgr.shape[:2]
                        videos_dir.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(video_specs[mode]),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.fps),
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"failed to open VideoWriter for {video_specs[mode]}")
                        writers[mode] = writer
                        sizes[mode] = (int(width), int(height))
                    writers[mode].write(bgr)

            if frame_id in selected_keys:
                base = frames_dir / f"frame_{frame_index:03d}_id_{frame_id:06d}"
                paths = {
                    "raw": base / "raw.png",
                    "reference_overlay": base / "reference_overlay.png",
                    "candidate_overlay": base / "candidate_overlay.png",
                    "difference_on_rgb": base / "difference_on_rgb.png",
                    "difference_mask": base / "difference_mask.png",
                    "reference_only_missing": base / "reference_only_missing.png",
                    "candidate_only_extra": base / "candidate_only_extra.png",
                }
                for mode, path in paths.items():
                    write_rgb_png(path, {"raw": rgb, **video_frames}[mode])
                selected_rows.append({**row, **{f"{mode}_path": repo_rel(path) for mode, path in paths.items()}})
    finally:
        for writer in writers.values():
            writer.release()

    stats_csv = out_dir / "frame_difference_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    components_csv = out_dir / "large_difference_components.csv"
    with components_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "frame_id", "kind", "rank", "area", "bbox_xyxy"])
        writer.writeheader()
        writer.writerows(component_rows)

    selected_json = out_dir / "selected_full_resolution_frames.json"
    selected_json.write_text(json.dumps(selected_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    videos: list[dict[str, Any]] = []
    if not bool(args.skip_videos):
        for mode, path in video_specs.items():
            videos.append(
                {
                    "mode": mode,
                    "path": repo_rel(path),
                    "sha256": sha256_file(path),
                    "frame_count": len(pairs),
                    "fps": float(args.fps),
                    "width": int(sizes.get(mode, (0, 0))[0]),
                    "height": int(sizes.get(mode, (0, 0))[1]),
                }
            )

    manifest = {
        "schema_version": "stream4d_v106_pairwise_visual_compare_v1",
        "reference_summary_path": repo_rel(reference_summary_path),
        "reference_summary_sha256": sha256_file(reference_summary_path),
        "candidate_summary_path": repo_rel(candidate_summary_path),
        "candidate_summary_sha256": sha256_file(candidate_summary_path),
        "out_dir": repo_rel(out_dir),
        "scene_id": str(reference_summary.get("scene_id")),
        "aligned_frame_count": len(pairs),
        "selected_frame_count": len(selected_rows),
        "selected_frames_json": repo_rel(selected_json),
        "frame_difference_stats_csv": repo_rel(stats_csv),
        "large_difference_components_csv": repo_rel(components_csv),
        "legend": {
            "red": "covered by reference run, absent from candidate run",
            "cyan": "covered by candidate run, absent from reference run",
            "gray": "covered by both runs",
        },
        "min_component_area": int(args.min_component_area),
        "videos": videos,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "aligned_frame_count": len(pairs),
                "selected_frame_count": len(selected_rows),
                "video_count": len(videos),
                "stats_csv": str(stats_csv),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
