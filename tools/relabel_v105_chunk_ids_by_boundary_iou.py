#!/usr/bin/env python3
"""Relabel chunk-local v105 masks by boundary IoU.

This is a diagnostic/repair candidate for chunk-windowed full-scene videos.  It
does not change mask geometry: each chunk-local label id is remapped to a
scene-global id, using IoU matches between the previous chunk's last frame and
the next chunk's first frame.  Local ids born later inside a chunk are preserved
as new global ids.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_v105_fullscene_multichunk_repair import overlay_label, put_text  # noqa: E402


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except Exception:
        return 10**15


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32)


def label_areas(label: np.ndarray) -> dict[int, int]:
    return {int(v): int(np.count_nonzero(label == int(v))) for v in np.unique(label) if int(v) > 0}


def remap_label(label: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros(label.shape, dtype=np.uint16)
    for local_id, global_id in mapping.items():
        out[label == int(local_id)] = int(global_id)
    return out


def boundary_mapping(
    *,
    prev_global_label: np.ndarray,
    first_local_label: np.ndarray,
    first_local_ids: list[int],
    all_local_ids: list[int],
    min_iou: float,
    min_intersection_pixels: int,
    match_mode: str,
    next_global_id: int,
) -> tuple[dict[int, int], list[dict[str, Any]], int]:
    prev_areas = label_areas(prev_global_label)
    curr_areas = label_areas(first_local_label)
    candidates: list[tuple[float, int, int, int]] = []
    for local_id in first_local_ids:
        mask = first_local_label == int(local_id)
        overlap_ids, overlap_counts = np.unique(prev_global_label[mask], return_counts=True)
        for global_id_raw, intersection_raw in zip(overlap_ids.tolist(), overlap_counts.tolist(), strict=False):
            global_id = int(global_id_raw)
            intersection = int(intersection_raw)
            if global_id <= 0:
                continue
            union = int(curr_areas[local_id]) + int(prev_areas.get(global_id, 0)) - intersection
            iou = float(intersection) / float(max(union, 1))
            if iou >= float(min_iou) and intersection >= int(min_intersection_pixels):
                candidates.append((iou, intersection, int(local_id), int(global_id)))
    candidates.sort(reverse=True)

    mapping: dict[int, int] = {}
    assignments: list[dict[str, Any]] = []
    used_local: set[int] = set()
    used_global: set[int] = set()
    for iou, intersection, local_id, global_id in candidates:
        if local_id in used_local:
            continue
        if str(match_mode) == "one_to_one" and global_id in used_global:
            continue
        mapping[local_id] = global_id
        used_local.add(local_id)
        used_global.add(global_id)
        assignments.append(
            {
                "local_id": int(local_id),
                "global_id": int(global_id),
                "iou": float(iou),
                "intersection_pixels": int(intersection),
            }
        )

    for local_id in all_local_ids:
        if int(local_id) not in mapping:
            mapping[int(local_id)] = int(next_global_id)
            next_global_id += 1
    return mapping, assignments, next_global_id


def compute_boundary_diagnostics(mask_dir: Path, *, chunk_size: int, iou_match_threshold: float) -> dict[str, Any]:
    frame_ids = sorted(numeric_stem(path) for path in mask_dir.glob("*.png") if numeric_stem(path) < 10**12)
    records: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "pairs": 0,
            "matched_prev": 0,
            "changed": 0,
            "same": 0,
            "sum_best_iou": 0.0,
            "sum_visible_prev": 0,
            "sum_visible_next": 0,
        }
    )

    labels_cache: dict[int, np.ndarray] = {}

    def load(frame_id: int) -> np.ndarray:
        if int(frame_id) not in labels_cache:
            labels_cache[int(frame_id)] = read_label(mask_dir / f"{int(frame_id)}.png")
        return labels_cache[int(frame_id)]

    for idx in range(len(frame_ids) - 1):
        prev_label = load(frame_ids[idx])
        next_label = load(frame_ids[idx + 1])
        prev_ids = [int(v) for v in np.unique(prev_label) if int(v) > 0]
        next_ids = [int(v) for v in np.unique(next_label) if int(v) > 0]
        prev_areas = label_areas(prev_label)
        next_areas = label_areas(next_label)
        matched = 0
        same = 0
        changed = 0
        sum_iou = 0.0
        top_changed: list[dict[str, Any]] = []
        for prev_id in prev_ids:
            prev_mask = prev_label == int(prev_id)
            overlap_ids, overlap_counts = np.unique(next_label[prev_mask], return_counts=True)
            best_next = 0
            best_intersection = 0
            best_iou = 0.0
            for next_id_raw, intersection_raw in zip(overlap_ids.tolist(), overlap_counts.tolist(), strict=False):
                next_id = int(next_id_raw)
                intersection = int(intersection_raw)
                if next_id <= 0:
                    continue
                union = int(prev_areas[prev_id]) + int(next_areas.get(next_id, 0)) - intersection
                iou = float(intersection) / float(max(union, 1))
                if iou > best_iou:
                    best_iou = iou
                    best_next = next_id
                    best_intersection = intersection
            if best_next > 0 and best_iou >= float(iou_match_threshold):
                matched += 1
                sum_iou += best_iou
                if best_next == prev_id:
                    same += 1
                else:
                    changed += 1
                    top_changed.append(
                        {
                            "prev_id": int(prev_id),
                            "next_id": int(best_next),
                            "best_iou": float(best_iou),
                            "intersection_pixels": int(best_intersection),
                            "prev_area": int(prev_areas[prev_id]),
                            "next_area": int(next_areas.get(best_next, 0)),
                        }
                    )
        is_boundary = (idx + 1) % int(chunk_size) == 0
        record = {
            "pair_index": int(idx),
            "prev_frame_id": int(frame_ids[idx]),
            "next_frame_id": int(frame_ids[idx + 1]),
            "boundary_after_prev": bool(is_boundary),
            "prev_chunk_index": int(idx // int(chunk_size)),
            "next_chunk_index": int((idx + 1) // int(chunk_size)),
            "visible_prev": int(len(prev_ids)),
            "visible_next": int(len(next_ids)),
            "matched_prev_iou_ge_threshold": int(matched),
            "same_id_matches": int(same),
            "changed_id_matches": int(changed),
            "changed_rate": float(changed) / float(max(matched, 1)),
            "avg_best_iou": float(sum_iou) / float(max(matched, 1)),
            "top_changed_matches": sorted(top_changed, key=lambda item: float(item["best_iou"]), reverse=True)[:8],
        }
        records.append(record)
        key = "boundary" if is_boundary else "non_boundary"
        bucket = summary[key]
        bucket["pairs"] += 1
        bucket["matched_prev"] += matched
        bucket["changed"] += changed
        bucket["same"] += same
        bucket["sum_best_iou"] += sum_iou
        bucket["sum_visible_prev"] += len(prev_ids)
        bucket["sum_visible_next"] += len(next_ids)

    summary_out: dict[str, Any] = {}
    for key, bucket in summary.items():
        summary_out[key] = {
            **bucket,
            "changed_rate": float(bucket["changed"]) / float(max(bucket["matched_prev"], 1)),
            "avg_best_iou_weighted": float(bucket["sum_best_iou"]) / float(max(bucket["matched_prev"], 1)),
            "avg_visible_prev_per_pair": float(bucket["sum_visible_prev"]) / float(max(bucket["pairs"], 1)),
            "avg_visible_next_per_pair": float(bucket["sum_visible_next"]) / float(max(bucket["pairs"], 1)),
        }
    return {
        "frame_count": int(len(frame_ids)),
        "chunk_size": int(chunk_size),
        "chunk_count": int((len(frame_ids) + int(chunk_size) - 1) // int(chunk_size)),
        "iou_match_threshold": float(iou_match_threshold),
        "summary": summary_out,
        "worst_boundary_pairs": sorted(
            [record for record in records if record["boundary_after_prev"]],
            key=lambda item: (float(item["changed_rate"]), int(item["changed_id_matches"])),
            reverse=True,
        )[:12],
        "records": records,
    }


def verify_foreground_geometry(source_dir: Path, relabeled_dir: Path) -> dict[str, Any]:
    source_paths = sorted(source_dir.glob("*.png"), key=numeric_stem)
    diff_frames = 0
    label_identical_frames = 0
    first_diff: list[str] = []
    for src_path in source_paths:
        dst_path = relabeled_dir / src_path.name
        src = read_label(src_path)
        dst = read_label(dst_path)
        if not np.array_equal(src > 0, dst > 0):
            diff_frames += 1
            if len(first_diff) < 10:
                first_diff.append(src_path.name)
        if sha256_file(src_path) == sha256_file(dst_path):
            label_identical_frames += 1
    return {
        "checked_frames": int(len(source_paths)),
        "foreground_diff_frames": int(diff_frames),
        "label_identical_frames": int(label_identical_frames),
        "first_foreground_diff_frames": first_diff,
    }


def write_video(mask_dir: Path, *, scene_id: str, rgb_root: Path, output_root: Path, variant_id: str, fps: float, workers: int) -> dict[str, Any]:
    frame_ids = sorted(numeric_stem(path) for path in mask_dir.glob("*.png") if numeric_stem(path) < 10**12)
    color_dir = rgb_root / scene_id / "color"
    assembled_root = output_root / "assembled_scene_videos"
    overlay_dir = assembled_root / "overlays" / scene_id
    video_path = assembled_root / "videos" / f"{variant_id}_{scene_id}_full_stride5.mp4"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    def build_frame(item: tuple[int, int]) -> tuple[int, np.ndarray]:
        idx, frame_id = item
        rgb = cv2.imread(str(color_dir / f"{int(frame_id)}.jpg"), cv2.IMREAD_COLOR)
        if rgb is None:
            rgb = cv2.imread(str(color_dir / f"{int(frame_id)}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(color_dir / f"{int(frame_id)}.jpg")
        label = read_label(mask_dir / f"{int(frame_id)}.png")
        overlay = overlay_label(rgb, label)
        overlay = put_text(
            overlay,
            [
                f"{scene_id} chunk-boundary relabel frame_index={idx:04d} frame_id={int(frame_id):06d}",
                "diagnostic candidate: boundary IoU id remap; mask geometry unchanged",
            ],
        )
        cv2.imwrite(str(overlay_dir / f"{idx:04d}_frame_{int(frame_id):06d}.jpg"), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return idx, overlay

    started = time.time()
    frames: list[np.ndarray | None] = [None] * len(frame_ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(build_frame, item) for item in enumerate(frame_ids)]
        for future in concurrent.futures.as_completed(futures):
            idx, frame = future.result()
            frames[int(idx)] = frame

    writer: cv2.VideoWriter | None = None
    written = 0
    for frame in frames:
        if frame is None:
            continue
        if writer is None:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (frame.shape[1], frame.shape[0]))
        if writer is not None and writer.isOpened():
            writer.write(frame)
            written += 1
    if writer is not None:
        writer.release()
    decoded = 0
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return {
        "video_path": rel(video_path),
        "video_exists": bool(video_path.exists() and video_path.stat().st_size > 0),
        "video_size_bytes": int(video_path.stat().st_size) if video_path.exists() else 0,
        "video_sha256": sha256_file(video_path) if video_path.exists() else "",
        "video_frame_count_written": int(written),
        "video_frame_count_decoded": int(decoded),
        "overlay_dir": rel(overlay_dir),
        "wall_sec": float(time.time() - started),
    }


def zip_outputs(output_root: Path, zip_name: str) -> dict[str, Any]:
    zip_path = output_root / zip_name
    if zip_path.exists():
        zip_path.unlink()
    members = [
        output_root / "chunk_boundary_relabel_summary.json",
        output_root / "boundary_diagnostics_raw.json",
        output_root / "boundary_diagnostics_relabeled.json",
    ]
    masks_root = output_root / "masks"
    videos_root = output_root / "assembled_scene_videos"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for path in members:
            if path.exists():
                zf.write(path, path.relative_to(output_root).as_posix())
        for root in (masks_root, videos_root):
            if root.exists():
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        zf.write(path, path.relative_to(output_root).as_posix())
    return {
        "zip_path": rel(zip_path),
        "zip_exists": bool(zip_path.exists()),
        "zip_size_bytes": int(zip_path.stat().st_size) if zip_path.exists() else 0,
        "zip_sha256": sha256_file(zip_path) if zip_path.exists() else "",
    }


def run(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_mask_dir)
    if not source_dir.is_absolute():
        source_dir = REPO_ROOT / source_dir
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    variant_id = str(args.variant_id)
    scene_id = str(args.scene_id)
    relabeled_dir = output_root / "masks" / variant_id / scene_id / "mask"
    relabeled_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = sorted(numeric_stem(path) for path in source_dir.glob("*.png") if numeric_stem(path) < 10**12)
    if not frame_ids:
        raise FileNotFoundError({"source_mask_dir": rel(source_dir), "reason": "no numeric .png masks found"})

    started = time.time()
    next_global_id = 1
    previous_last_global: np.ndarray | None = None
    chunk_records: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, len(frame_ids), int(args.chunk_size))):
        end = min(start + int(args.chunk_size), len(frame_ids))
        chunk_frame_ids = frame_ids[start:end]
        chunk_labels = [read_label(source_dir / f"{int(frame_id)}.png") for frame_id in chunk_frame_ids]
        first_label = chunk_labels[0]
        first_local_ids = [int(v) for v in np.unique(first_label) if int(v) > 0]
        all_local_ids = sorted({int(v) for label in chunk_labels for v in np.unique(label) if int(v) > 0})
        if chunk_index == 0 or previous_last_global is None:
            mapping = {int(local_id): int(local_id) for local_id in all_local_ids}
            next_global_id = max(all_local_ids + [0]) + 1
            assignments: list[dict[str, Any]] = []
            mode = "identity_first_chunk_all_ids"
        else:
            mapping, assignments, next_global_id = boundary_mapping(
                prev_global_label=previous_last_global,
                first_local_label=first_label,
                first_local_ids=first_local_ids,
                all_local_ids=all_local_ids,
                min_iou=float(args.min_iou),
                min_intersection_pixels=int(args.min_intersection_pixels),
                match_mode=str(args.match_mode),
                next_global_id=int(next_global_id),
            )
            mode = f"boundary_iou_{args.match_mode}_plus_all_local_birth_ids"

        last_global: np.ndarray | None = None
        for frame_id, label in zip(chunk_frame_ids, chunk_labels, strict=False):
            relabeled = remap_label(label, mapping)
            cv2.imwrite(str(relabeled_dir / f"{int(frame_id)}.png"), relabeled)
            last_global = relabeled
        previous_last_global = last_global
        chunk_records.append(
            {
                "chunk_index": int(chunk_index),
                "start_index": int(start),
                "end_index": int(end - 1),
                "start_frame_id": int(chunk_frame_ids[0]),
                "end_frame_id": int(chunk_frame_ids[-1]),
                "mode": mode,
                "first_local_id_count": int(len(first_local_ids)),
                "all_local_id_count_in_chunk": int(len(all_local_ids)),
                "mapped_existing_count": int(len(assignments)),
                "new_global_count": int(len(all_local_ids) - len(assignments)) if chunk_index else int(len(all_local_ids)),
                "mapping_first40": {str(k): int(v) for k, v in sorted(mapping.items())[:40]},
                "assignments_top20": assignments[:20],
            }
        )

    geometry = verify_foreground_geometry(source_dir, relabeled_dir)
    raw_diag = compute_boundary_diagnostics(source_dir, chunk_size=int(args.chunk_size), iou_match_threshold=float(args.diagnostic_iou_threshold))
    relabeled_diag = compute_boundary_diagnostics(relabeled_dir, chunk_size=int(args.chunk_size), iou_match_threshold=float(args.diagnostic_iou_threshold))
    raw_diag_path = output_root / "boundary_diagnostics_raw.json"
    relabeled_diag_path = output_root / "boundary_diagnostics_relabeled.json"
    raw_diag_path.write_text(json.dumps(raw_diag, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    relabeled_diag_path.write_text(json.dumps(relabeled_diag, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    video_summary: dict[str, Any] | None = None
    if not bool(args.skip_video):
        rgb_root = Path(args.rgb_root)
        if not rgb_root.is_absolute():
            rgb_root = REPO_ROOT / rgb_root
        video_summary = write_video(
            relabeled_dir,
            scene_id=scene_id,
            rgb_root=rgb_root,
            output_root=output_root,
            variant_id=variant_id,
            fps=float(args.fps),
            workers=int(args.workers),
        )

    summary_path = output_root / "chunk_boundary_relabel_summary.json"
    summary = {
        "schema_version": "stream4d_v105_chunk_boundary_id_relabel_v1",
        "source_mask_dir": rel(source_dir),
        "output_root": rel(output_root),
        "relabeled_mask_dir": rel(relabeled_dir),
        "scene_id": scene_id,
        "variant_id": variant_id,
        "frame_count": int(len(frame_ids)),
        "chunk_size": int(args.chunk_size),
        "chunk_count": int((len(frame_ids) + int(args.chunk_size) - 1) // int(args.chunk_size)),
        "min_iou": float(args.min_iou),
        "min_intersection_pixels": int(args.min_intersection_pixels),
        "match_mode": str(args.match_mode),
        "diagnostic_iou_threshold": float(args.diagnostic_iou_threshold),
        "geometry_verification": geometry,
        "raw_boundary_summary": raw_diag["summary"],
        "relabeled_boundary_summary": relabeled_diag["summary"],
        "raw_boundary_diagnostics": rel(raw_diag_path),
        "relabeled_boundary_diagnostics": rel(relabeled_diag_path),
        "video_summary": video_summary,
        "chunk_records": chunk_records,
        "wall_sec": float(time.time() - started),
        "note": "Diagnostic repair candidate: relabels ids only. Mask foreground geometry must remain unchanged.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    zip_summary: dict[str, Any] | None = None
    if str(args.zip_name).strip():
        zip_summary = zip_outputs(output_root, str(args.zip_name))
        summary["zip_summary"] = zip_summary
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": rel(summary_path), "video": (video_summary or {}).get("video_path"), "zip": (zip_summary or {}).get("zip_path")}, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mask-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--variant-id", default="P6_scene0050_chunk_boundary_iou_relabel")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--min-intersection-pixels", type=int, default=512)
    parser.add_argument(
        "--match-mode",
        choices=("one_to_one", "split_friendly"),
        default="one_to_one",
        help=(
            "one_to_one preserves the current conservative matching. "
            "split_friendly lets multiple first-frame local fragments inherit the same previous global id."
        ),
    )
    parser.add_argument("--diagnostic-iou-threshold", type=float, default=0.05)
    parser.add_argument("--rgb-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-video", action="store_true", default=False)
    parser.add_argument("--zip-name", default="")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
