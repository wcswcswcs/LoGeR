#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULLSCENE_ROOT = REPO_ROOT / "Stream3D/outputs/audit/v105_fullscene_multichunk_repair_20260711"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Stream3D/outputs/audit/v105_fullscene_local2history_stitch_20260711"
STITCH_VARIANT_ID = "P6_period4_force50k_max1_scene_l2h_stitched_candidate_v1"


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except Exception:
        return 10**15


def _color_for_id(label_id: int) -> tuple[int, int, int]:
    value = int(label_id) * 1103515245 + 12345
    return (
        80 + ((value >> 16) & 127),
        80 + ((value >> 8) & 127),
        80 + ((value >> 0) & 127),
    )


def _overlay_label(rgb_bgr: np.ndarray, label: np.ndarray | None, *, alpha: float) -> np.ndarray:
    out = rgb_bgr.copy()
    if label is None:
        return out
    if label.ndim == 3:
        label = label[:, :, 0]
    if label.shape[:2] != out.shape[:2]:
        label = cv2.resize(label.astype(np.uint16), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    color_layer = np.zeros_like(out)
    mask_any = label > 0
    for label_id in ids:
        color_layer[label == label_id] = _color_for_id(label_id)
    out[mask_any] = np.clip(
        out[mask_any].astype(np.float32) * (1.0 - alpha) + color_layer[mask_any].astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    edges = np.zeros(label.shape[:2], dtype=np.uint8)
    for label_id in ids:
        m = (label == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours, -1, 255, 1)
    out[edges > 0] = (255, 255, 255)
    return out


def _put_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    height = 22 * len(lines) + 6
    cv2.rectangle(out, (0, 0), (out.shape[1], height), (0, 0, 0), thickness=-1)
    for idx, line in enumerate(lines):
        cv2.putText(out, line[:180], (8, 18 + 22 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def _mask_stats(label: np.ndarray) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    for label_id in ids:
        ys, xs = np.nonzero(label == label_id)
        if len(xs) == 0:
            continue
        stats[label_id] = {
            "area": float(len(xs)),
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "xmin": float(xs.min()),
            "xmax": float(xs.max()),
            "ymin": float(ys.min()),
            "ymax": float(ys.max()),
        }
    return stats


def _pair_intersections(prev: np.ndarray, curr: np.ndarray) -> dict[tuple[int, int], int]:
    mask = (prev > 0) & (curr > 0)
    if not np.any(mask):
        return {}
    max_curr = int(curr.max()) + 1
    combined = prev[mask].astype(np.int64) * max_curr + curr[mask].astype(np.int64)
    values, counts = np.unique(combined, return_counts=True)
    pairs: dict[tuple[int, int], int] = {}
    for value, count in zip(values, counts):
        prev_id = int(value // max_curr)
        curr_id = int(value % max_curr)
        pairs[(prev_id, curr_id)] = int(count)
    return pairs


def _boundary_matches(
    *,
    prev_global: np.ndarray,
    curr_local: np.ndarray,
    min_iou: float,
    min_overlap_min: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    prev_stats = _mask_stats(prev_global)
    curr_stats = _mask_stats(curr_local)
    intersections = _pair_intersections(prev_global, curr_local)
    candidates: list[dict[str, Any]] = []
    for (prev_id, curr_id), inter in intersections.items():
        prev_area = prev_stats.get(prev_id, {}).get("area", 0.0)
        curr_area = curr_stats.get(curr_id, {}).get("area", 0.0)
        if prev_area <= 0 or curr_area <= 0:
            continue
        union = prev_area + curr_area - float(inter)
        iou = float(inter) / union if union > 0 else 0.0
        overlap_min = float(inter) / min(prev_area, curr_area)
        if iou < min_iou and overlap_min < min_overlap_min:
            continue
        score = max(iou, overlap_min * 0.5)
        candidates.append(
            {
                "prev_global_id": int(prev_id),
                "curr_local_id": int(curr_id),
                "intersection": int(inter),
                "prev_area": int(prev_area),
                "curr_area": int(curr_area),
                "iou": iou,
                "overlap_min": overlap_min,
                "score": score,
            }
        )
    candidates.sort(key=lambda row: (float(row["score"]), float(row["iou"]), int(row["intersection"])), reverse=True)
    used_prev: set[int] = set()
    used_curr: set[int] = set()
    mapping: dict[int, int] = {}
    accepted: list[dict[str, Any]] = []
    for row in candidates:
        prev_id = int(row["prev_global_id"])
        curr_id = int(row["curr_local_id"])
        if prev_id in used_prev or curr_id in used_curr:
            continue
        used_prev.add(prev_id)
        used_curr.add(curr_id)
        mapping[curr_id] = prev_id
        accepted.append(row)
    unmatched_curr = [int(v) for v in curr_stats if int(v) not in mapping]
    large_unmatched_curr = [
        {"curr_local_id": int(v), "area": int(curr_stats[v]["area"])}
        for v in unmatched_curr
        if curr_stats[v]["area"] >= 10000
    ]
    audit = {
        "prev_global_id_count": len(prev_stats),
        "curr_local_id_count": len(curr_stats),
        "candidate_count": len(candidates),
        "matched_count": len(accepted),
        "unmatched_curr_count": len(unmatched_curr),
        "large_unmatched_curr_count": len(large_unmatched_curr),
        "large_unmatched_curr_first20": large_unmatched_curr[:20],
        "mean_iou": float(np.mean([row["iou"] for row in accepted])) if accepted else 0.0,
        "mean_overlap_min": float(np.mean([row["overlap_min"] for row in accepted])) if accepted else 0.0,
        "accepted_matches_first40": accepted[:40],
    }
    return mapping, audit


def _remap_label(label: np.ndarray, mapping: dict[int, int], next_id: int) -> tuple[np.ndarray, int, int]:
    local_ids = [int(v) for v in np.unique(label) if int(v) > 0]
    new_births = 0
    for local_id in local_ids:
        if local_id not in mapping:
            mapping[local_id] = next_id
            next_id += 1
            new_births += 1
    max_local = int(label.max()) if label.size else 0
    lookup = np.zeros(max_local + 1, dtype=np.uint16)
    for local_id, global_id in mapping.items():
        if local_id <= max_local:
            lookup[local_id] = int(global_id)
    return lookup[label].astype(np.uint16, copy=False), next_id, new_births


def _make_sheet(image_paths: list[Path], out_path: Path, title: str, thumb_w: int, thumb_h: int) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols, rows, header_h = 4, 2, 44
    sheet = np.full((header_h + rows * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    cv2.rectangle(sheet, (0, 0), (cols * thumb_w, header_h), (20, 20, 20), -1)
    cv2.putText(sheet, title[:150], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    readable = 0
    for idx, path in enumerate(image_paths[:8]):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            thumb = np.full((thumb_h, thumb_w, 3), 230, dtype=np.uint8)
            cv2.putText(thumb, "missing", (18, thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            readable += 1
            thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (170, 34), (0, 0, 0), -1)
        cv2.putText(thumb, f"frame {idx:02d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        y = header_h + (idx // cols) * thumb_h
        x = (idx % cols) * thumb_w
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return {"path": _rel(out_path), "readable_images": readable, "expected_images": len(image_paths[:8])}


def _scene_chunks(plan: dict[str, Any], scene_id: str, frame_ids: list[int]) -> list[dict[str, Any]]:
    chunks = [
        {
            "chunk_index": 0,
            "source": "first64_prefix",
            "start_index": 0,
            "frame_count": min(int(plan.get("first_prefix_count", 64)), len(frame_ids)),
        }
    ]
    jobs = [
        job
        for job in plan.get("planned_jobs", [])
        if isinstance(job, dict) and str(job.get("scene_id")) == scene_id
    ]
    for idx, job in enumerate(sorted(jobs, key=lambda row: int(row.get("start_index", 0))), start=1):
        chunks.append(
            {
                "chunk_index": idx,
                "source": "multichunk_window",
                "start_index": int(job["start_index"]),
                "frame_count": int(job["frame_count"]),
                "frame_start": int(job["frame_start"]),
            }
        )
    return chunks


def _build_scene(
    *,
    scene_id: str,
    source_mask_dir: Path,
    scene_color_dir: Path,
    plan: dict[str, Any],
    output_root: Path,
    alpha: float,
    min_iou: float,
    min_overlap_min: float,
    fps: float,
) -> dict[str, Any]:
    frame_ids = sorted([_numeric_stem(path) for path in source_mask_dir.glob("*.png") if _numeric_stem(path) < 10**12])
    chunks = _scene_chunks(plan, scene_id, frame_ids)
    mask_dir = output_root / "local2history_stitched" / "masks" / STITCH_VARIANT_ID / scene_id / "mask"
    overlay_dir = output_root / "local2history_stitched" / "overlays" / scene_id
    sheet_root = output_root / "local2history_stitched" / "sheet_groups" / scene_id
    boundary_sheet_root = output_root / "local2history_stitched" / "boundary_sheets" / scene_id
    video_path = output_root / "local2history_stitched" / "videos" / f"{STITCH_VARIANT_ID}_{scene_id}_full_stride5.mp4"
    for directory in (mask_dir, overlay_dir, sheet_root, boundary_sheet_root, video_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    next_global_id = 1
    prev_last_global: np.ndarray | None = None
    prev_last_frame_id: int | None = None
    chunk_records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []
    written_masks = 0
    written_overlays = 0
    sheet_records: list[dict[str, Any]] = []
    boundary_sheet_records: list[dict[str, Any]] = []
    writer: cv2.VideoWriter | None = None
    video_written = 0

    for chunk in chunks:
        start = int(chunk["start_index"])
        count = int(chunk["frame_count"])
        chunk_frame_ids = frame_ids[start : start + count]
        if not chunk_frame_ids:
            continue
        first_label = _read_label(source_mask_dir / f"{chunk_frame_ids[0]}.png")
        mapping: dict[int, int] = {}
        boundary_audit: dict[str, Any] | None = None
        if prev_last_global is not None:
            mapping, boundary_audit = _boundary_matches(
                prev_global=prev_last_global,
                curr_local=first_label,
                min_iou=min_iou,
                min_overlap_min=min_overlap_min,
            )
            boundary_audit.update(
                {
                    "scene_id": scene_id,
                    "prev_chunk_index": int(chunk["chunk_index"]) - 1,
                    "curr_chunk_index": int(chunk["chunk_index"]),
                    "prev_frame_id": int(prev_last_frame_id),
                    "curr_frame_id": int(chunk_frame_ids[0]),
                    "min_iou": float(min_iou),
                    "min_overlap_min": float(min_overlap_min),
                }
            )
            boundary_records.append(boundary_audit)

        chunk_new_births = 0
        chunk_overlay_paths: list[Path] = []
        local_ids_seen: set[int] = set()
        global_ids_seen: set[int] = set()
        last_global: np.ndarray | None = None
        for frame_pos, frame_id in enumerate(chunk_frame_ids):
            local = _read_label(source_mask_dir / f"{frame_id}.png")
            local_ids_seen.update(int(v) for v in np.unique(local) if int(v) > 0)
            global_label, next_global_id, new_births = _remap_label(local, mapping, next_global_id)
            chunk_new_births += int(new_births)
            global_ids_seen.update(int(v) for v in np.unique(global_label) if int(v) > 0)
            cv2.imwrite(str(mask_dir / f"{frame_id}.png"), global_label)
            written_masks += 1

            rgb = cv2.imread(str(scene_color_dir / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
            if rgb is None:
                rgb = cv2.imread(str(scene_color_dir / f"{frame_id}.png"), cv2.IMREAD_COLOR)
            if rgb is not None:
                overlay = _overlay_label(rgb, global_label, alpha=alpha)
                overlay = _put_text(
                    overlay,
                    [
                        f"{scene_id} {STITCH_VARIANT_ID} frame_index={start + frame_pos:04d} frame_id={frame_id:06d}",
                        f"ID-only local2history stitch candidate; chunk={chunk['chunk_index']} local geometry unchanged",
                    ],
                )
                overlay_path = overlay_dir / f"{start + frame_pos:04d}_frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                chunk_overlay_paths.append(overlay_path)
                written_overlays += 1
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (overlay.shape[1], overlay.shape[0]))
                if writer is not None and writer.isOpened():
                    writer.write(overlay)
                    video_written += 1
            last_global = global_label

        if chunk_overlay_paths:
            for group_idx, offset in enumerate(range(0, len(chunk_overlay_paths), 8)):
                group_paths = chunk_overlay_paths[offset : offset + 8]
                out_path = sheet_root / f"chunk_{int(chunk['chunk_index']):03d}_frames_{offset:03d}_{offset + len(group_paths) - 1:03d}.jpg"
                sheet_records.append(
                    {
                        "scene_id": scene_id,
                        "chunk_index": int(chunk["chunk_index"]),
                        "sheet_group_index": int(group_idx),
                        "frame_ids": [int(chunk_frame_ids[offset + idx]) for idx in range(len(group_paths))],
                        **_make_sheet(group_paths, out_path, f"{scene_id} chunk {int(chunk['chunk_index']):03d} l2h stitched frames {offset:03d}-{offset + len(group_paths) - 1:03d}", 480, 360),
                    }
                )
        if boundary_audit is not None:
            prev_paths = sorted(overlay_dir.glob("*.jpg"))[-(len(chunk_overlay_paths) + 4) : -len(chunk_overlay_paths)] if len(chunk_overlay_paths) else []
            curr_paths = chunk_overlay_paths[:4]
            if prev_paths and curr_paths:
                out_path = boundary_sheet_root / f"boundary_{int(chunk['chunk_index']) - 1:03d}_{int(chunk['chunk_index']):03d}.jpg"
                boundary_sheet_records.append(
                    {
                        "scene_id": scene_id,
                        "prev_chunk_index": int(chunk["chunk_index"]) - 1,
                        "curr_chunk_index": int(chunk["chunk_index"]),
                        "prev_frame_id": int(prev_last_frame_id),
                        "curr_frame_id": int(chunk_frame_ids[0]),
                        **_make_sheet(list(prev_paths) + curr_paths, out_path, f"{scene_id} boundary {int(chunk['chunk_index']) - 1:03d}->{int(chunk['chunk_index']):03d} l2h stitched", 480, 360),
                    }
                )

        chunk_records.append(
            {
                "scene_id": scene_id,
                "chunk_index": int(chunk["chunk_index"]),
                "source": chunk.get("source"),
                "start_index": start,
                "frame_count": len(chunk_frame_ids),
                "first_frame_id": int(chunk_frame_ids[0]),
                "last_frame_id": int(chunk_frame_ids[-1]),
                "local_id_count": len(local_ids_seen),
                "global_id_count": len(global_ids_seen),
                "new_global_birth_count": int(chunk_new_births),
                "inherited_from_previous_count": int(boundary_audit["matched_count"]) if boundary_audit else 0,
            }
        )
        prev_last_global = last_global
        prev_last_frame_id = int(chunk_frame_ids[-1])

    if writer is not None:
        writer.release()
    decoded_frames = 0
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            decoded_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    boundary_count = len(boundary_records)
    weak_boundaries = [
        row
        for row in boundary_records
        if int(row.get("matched_count", 0)) == 0 or int(row.get("large_unmatched_curr_count", 0)) > 0
    ]
    row = {
        "schema_version": "stream4d_v105_fullscene_l2h_stitch_scene_row_v1",
        "scene_id": scene_id,
        "variant_id": STITCH_VARIANT_ID,
        "source_variant": str(plan.get("variant_id")),
        "source_mask_dir": _rel(source_mask_dir),
        "mask_dir": _rel(mask_dir),
        "overlay_dir": _rel(overlay_dir),
        "video_path": _rel(video_path),
        "frame_count_expected": len(frame_ids),
        "mask_count_written": written_masks,
        "overlay_count_written": written_overlays,
        "video_frame_count_written": video_written,
        "video_frame_count_decoded": decoded_frames,
        "complete_scene_masks": bool(written_masks == len(frame_ids)),
        "complete_scene_video": bool(decoded_frames == len(frame_ids)),
        "chunk_count": len(chunk_records),
        "boundary_count": boundary_count,
        "weak_boundary_count": len(weak_boundaries),
        "weak_boundaries_first20": weak_boundaries[:20],
        "total_global_id_count": int(next_global_id - 1),
        "chunk_records": chunk_records,
        "boundary_records": boundary_records,
        "sheet_group_count": len(sheet_records),
        "boundary_sheet_count": len(boundary_sheet_records),
        "sheet_records_json": _rel(output_root / "sheet_records.json"),
        "boundary_sheet_records_json": _rel(output_root / "boundary_sheet_records.json"),
        "id_only_stitch_candidate": True,
        "mask_geometry_modified": False,
        "continuous_scene_level_id_claim": False,
        "claim_boundary": "Candidate ID-only local2history artifact; requires boundary visual review and user confirmation before a final continuous identity claim.",
    }
    return row, sheet_records, boundary_sheet_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a full-scene ID-only local2history stitch candidate from v105 chunk-windowed masks.")
    parser.add_argument("--fullscene-root", default=_rel(DEFAULT_FULLSCENE_ROOT))
    parser.add_argument("--output-root", default=_rel(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--min-iou", type=float, default=0.02)
    parser.add_argument("--min-overlap-min", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    fullscene_root = Path(args.fullscene_root)
    if not fullscene_root.is_absolute():
        fullscene_root = REPO_ROOT / fullscene_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    plan = _read_json(fullscene_root / "fullscene_multichunk_plan.json")
    summary = _read_json(fullscene_root / "fullscene_multichunk_summary.json")
    source_variant = str(summary.get("variant_id") or plan.get("variant_id"))
    scenes = [str(scene) for scene in plan.get("scenes", [])]
    scene_rows: list[dict[str, Any]] = []
    all_sheet_records: list[dict[str, Any]] = []
    all_boundary_sheet_records: list[dict[str, Any]] = []
    for scene_id in scenes:
        source_mask_dir = fullscene_root / "assembled_scene_videos" / "sgq_local" / "masks" / source_variant / scene_id / "mask"
        scene_color_dir = REPO_ROOT / "Stream3D/data/scannet/processed" / scene_id / "color"
        row, sheet_records, boundary_sheet_records = _build_scene(
            scene_id=scene_id,
            source_mask_dir=source_mask_dir,
            scene_color_dir=scene_color_dir,
            plan=plan,
            output_root=output_root,
            alpha=float(args.alpha),
            min_iou=float(args.min_iou),
            min_overlap_min=float(args.min_overlap_min),
            fps=float(args.fps),
        )
        scene_rows.append(row)
        all_sheet_records.extend(sheet_records)
        all_boundary_sheet_records.extend(boundary_sheet_records)

    _write_json(output_root / "scene_rows.json", scene_rows)
    _write_json(output_root / "sheet_records.json", all_sheet_records)
    _write_json(output_root / "boundary_sheet_records.json", all_boundary_sheet_records)
    all_complete_masks = bool(scene_rows) and all(bool(row.get("complete_scene_masks")) for row in scene_rows)
    all_complete_videos = bool(scene_rows) and all(bool(row.get("complete_scene_video")) for row in scene_rows)
    total_weak = sum(int(row.get("weak_boundary_count", 0)) for row in scene_rows)
    output_summary = {
        "schema_version": "stream4d_v105_fullscene_l2h_stitch_summary_v1",
        "variant_id": STITCH_VARIANT_ID,
        "source_fullscene_root": _rel(fullscene_root),
        "source_variant": source_variant,
        "output_root": _rel(output_root),
        "scene_count": len(scene_rows),
        "scene_rows_json": _rel(output_root / "scene_rows.json"),
        "sheet_records_json": _rel(output_root / "sheet_records.json"),
        "boundary_sheet_records_json": _rel(output_root / "boundary_sheet_records.json"),
        "all_complete_scene_masks": all_complete_masks,
        "all_complete_scene_videos": all_complete_videos,
        "scene_rows": scene_rows,
        "total_boundary_count": sum(int(row.get("boundary_count", 0)) for row in scene_rows),
        "total_weak_boundary_count": total_weak,
        "total_sheet_group_count": len(all_sheet_records),
        "total_boundary_sheet_count": len(all_boundary_sheet_records),
        "min_iou": float(args.min_iou),
        "min_overlap_min": float(args.min_overlap_min),
        "id_only_stitch_candidate": True,
        "mask_geometry_modified": False,
        "complete_scene_prediction_candidate": all_complete_masks,
        "continuous_scene_level_id_claim": False,
        "claim_boundary": "This is a candidate ID-only local2history stitch over complete scene masks. It does not claim final continuous scene-level identity until boundary visual review and user confirmation pass.",
    }
    _write_json(output_root / "fullscene_l2h_stitch_summary.json", output_summary)
    hash_summary = {
        "summary_sha256": _sha256(output_root / "fullscene_l2h_stitch_summary.json"),
        "scene_rows_sha256": _sha256(output_root / "scene_rows.json"),
        "sheet_records_sha256": _sha256(output_root / "sheet_records.json"),
        "boundary_sheet_records_sha256": _sha256(output_root / "boundary_sheet_records.json"),
    }
    _write_json(output_root / "hashes.json", hash_summary)
    print(json.dumps({**output_summary, "hashes": hash_summary}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
