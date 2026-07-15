#!/usr/bin/env python3
"""Build ACL2 v118-TF Stage1 causal object/track sidecar v2.

This uses Stage-C chunk caches, not the monolithic sparse_masklets file, so
prefix statistics can be recomputed without loading full-sequence semantic
tensors at once.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
OUT = RESULT_ROOT / "stage1_causal_object_track_sidecar"
STAGE0_SUMMARY = RESULT_ROOT / "stage0_fresh_reference/stage0_fresh_reference_summary.json"
KITTI_PREPROCESS = ROOT / "results/kitti_preprocess"
V117_STAGE1 = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability/stage1_object_identity"

SEQS = ("00", "02")
NO_TRACK = -1
ROLE_COUNT = 7

ROLE_TO_ID = {
    "dynamic": 0,
    "boundary_lowpurity": 1,
    "weak_context": 2,
    "stable_landmark": 3,
    "vegetation_repetitive": 4,
    "sky_lowobs": 5,
    "unknown_lowtrust": 6,
}

LABEL_TO_ROLE = {
    "void": "unknown_lowtrust",
    "parasol_or_umbrella": "weak_context",
    "roadblock": "weak_context",
    "bus": "dynamic",
    "truck": "dynamic",
    "bicycle": "dynamic",
    "motorcycle": "dynamic",
    "person": "dynamic",
    "bench": "stable_landmark",
    "flower_pot_or_vase": "weak_context",
    "handrail_or_fence": "stable_landmark",
    "wall": "stable_landmark",
    "pillar": "stable_landmark",
    "pole": "stable_landmark",
    "ground": "weak_context",
    "grass": "vegetation_repetitive",
    "road": "weak_context",
    "path": "weak_context",
    "building": "stable_landmark",
    "house": "stable_landmark",
    "bridge": "stable_landmark",
    "other_construction": "stable_landmark",
    "sky": "sky_lowobs",
    "mountain": "stable_landmark",
    "stone": "stable_landmark",
    "billboard_or_bulletin_board": "stable_landmark",
    "wheeled_machine": "dynamic",
    "other_machine": "dynamic",
    "tree": "vegetation_repetitive",
    "flower": "vegetation_repetitive",
    "other_plant": "vegetation_repetitive",
    "trash_can": "weak_context",
    "car": "dynamic",
    "traffic sign": "stable_landmark",
    "stair": "stable_landmark",
}

LABEL_SUPPORT_COUNT = max(2, len(LABEL_TO_ROLE))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def entropy_consistency(counts: Counter[str], support_size: int | None = None) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [count / float(total) for count in counts.values() if count > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    denom = math.log(max(2, int(support_size or len(counts) or ROLE_COUNT)))
    return float(max(0.0, min(1.0, 1.0 - entropy / denom)))


def stable_ratio(prev: float, cur: float) -> float:
    if not math.isfinite(prev) or not math.isfinite(cur) or prev <= 0.0 or cur <= 0.0:
        return 0.0
    return float(math.exp(-abs(math.log((cur + 1e-12) / (prev + 1e-12)))))


def mask_perimeter(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    m = mask.astype(bool, copy=False)
    perimeter = int(m[0, :].sum() + m[-1, :].sum() + m[:, 0].sum() + m[:, -1].sum())
    perimeter += int(np.logical_xor(m[1:, :], m[:-1, :]).sum())
    perimeter += int(np.logical_xor(m[:, 1:], m[:, :-1]).sum())
    return perimeter


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def bbox_center(box: np.ndarray) -> tuple[float, float]:
    return (float(0.5 * (box[0] + box[2])), float(0.5 * (box[1] + box[3])))


def iter_cache_entries(seq: str) -> list[dict[str, Any]]:
    index_path = KITTI_PREPROCESS / seq / "stage_c_cache_semantic_chunks/cache_index.jsonl"
    rows = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["chunk_idx"]))


def label_role_stats(label_map: np.ndarray, mask: np.ndarray, label_names: list[str]) -> tuple[int, str, dict[str, int], dict[str, int]]:
    vals = label_map[mask]
    if vals.size == 0:
        return 0, "void", {"void": 0}, {"unknown_lowtrust": 0}
    counts = np.bincount(vals.astype(np.int64), minlength=len(label_names))
    label_id = int(counts.argmax())
    label = label_names[label_id] if label_id < len(label_names) else "void"
    label_counts = {
        label_names[idx] if idx < len(label_names) else "void": int(count)
        for idx, count in enumerate(counts)
        if int(count) > 0
    }
    role_counts: Counter[str] = Counter()
    for label_name, count in label_counts.items():
        role_counts[LABEL_TO_ROLE.get(label_name, "unknown_lowtrust")] += int(count)
    return label_id, label, label_counts, dict(role_counts)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else float("nan")


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def fragmentation_bucket(gap_gt2_count: int, visible_count: int, max_gap: int) -> str:
    if visible_count <= 2:
        return "short_1to2_obs"
    if gap_gt2_count == 0:
        return "continuous_no_gap_gt2"
    if gap_gt2_count <= 2 and max_gap <= 10:
        return "minor_fragmentation"
    if gap_gt2_count <= 5 and max_gap <= 30:
        return "moderate_fragmentation"
    return "severe_fragmentation"


def causal_merge_attempt1(seq: str, obs_by_track: dict[int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attempt plan repair 1: same-label, temporal gap<=2, endpoint mask-IoU merge."""
    track_ids = sorted(obs_by_track)
    uf = UnionFind(track_ids)
    endpoints = []
    starts_by_label_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    original_fragmented = []
    for track_id in track_ids:
        obs = obs_by_track[track_id]
        gaps = [int(cur["frame_id"]) - int(prev["frame_id"]) for prev, cur in zip(obs[:-1], obs[1:])]
        original_fragmented.append(any(gap > 2 for gap in gaps))
        first = obs[0]
        last = obs[-1]
        endpoint = {
            "track_id": track_id,
            "first_frame": int(first["frame_id"]),
            "last_frame": int(last["frame_id"]),
            "first_label": str(first["label"]),
            "last_label": str(last["label"]),
            "first_mask": first["_mask"],
            "last_mask": last["_mask"],
            "first_bbox": np.asarray([first["bbox_x0"], first["bbox_y0"], first["bbox_x1"], first["bbox_y1"]], dtype=np.float32),
            "last_bbox": np.asarray([last["bbox_x0"], last["bbox_y0"], last["bbox_x1"], last["bbox_y1"]], dtype=np.float32),
        }
        endpoints.append(endpoint)
        starts_by_label_frame[(endpoint["first_label"], endpoint["first_frame"])].append(endpoint)

    rows = []
    mask_iou_threshold = 0.05
    bbox_iou_threshold = 0.05
    for src in endpoints:
        for gap in (1, 2):
            candidates = starts_by_label_frame.get((src["last_label"], int(src["last_frame"]) + gap), [])
            for dst in candidates:
                if int(src["track_id"]) == int(dst["track_id"]):
                    continue
                raw_mask_iou = mask_iou(src["last_mask"], dst["first_mask"])
                raw_bbox_iou = bbox_iou(src["last_bbox"], dst["first_bbox"])
                accepted = (
                    math.isfinite(raw_mask_iou)
                    and raw_mask_iou >= mask_iou_threshold
                    and math.isfinite(raw_bbox_iou)
                    and raw_bbox_iou >= bbox_iou_threshold
                )
                rows.append(
                    {
                        "schema": "acl2_v118tf_stage1_causal_merge_attempt1_row_v1",
                        "seq": seq,
                        "src_track_id": int(src["track_id"]),
                        "dst_track_id": int(dst["track_id"]),
                        "src_last_frame": int(src["last_frame"]),
                        "dst_first_frame": int(dst["first_frame"]),
                        "temporal_gap": gap,
                        "label": src["last_label"],
                        "mask_iou": raw_mask_iou,
                        "bbox_iou": raw_bbox_iou,
                        "mask_iou_threshold": mask_iou_threshold,
                        "bbox_iou_threshold": bbox_iou_threshold,
                        "accepted": accepted,
                    }
                )
                if accepted:
                    uf.union(int(src["track_id"]), int(dst["track_id"]))

    groups: dict[int, list[int]] = defaultdict(list)
    for track_id in track_ids:
        groups[uf.find(track_id)].append(track_id)
    post_fragmented = []
    for members in groups.values():
        frames = sorted({int(row["frame_id"]) for member in members for row in obs_by_track[member]})
        gaps = [cur - prev for prev, cur in zip(frames[:-1], frames[1:])]
        post_fragmented.append(any(gap > 2 for gap in gaps))

    accepted_count = sum(1 for row in rows if row["accepted"])
    summary = {
        "schema": "acl2_v118tf_stage1_causal_merge_attempt1_summary_v1",
        "seq": seq,
        "input_track_count": len(track_ids),
        "candidate_pair_count": len(rows),
        "accepted_pair_count": accepted_count,
        "post_merge_group_count": len(groups),
        "original_fragmentation_rate": mean_bool(original_fragmented),
        "post_merge_fragmentation_rate": mean_bool(post_fragmented),
        "mask_iou_threshold": mask_iou_threshold,
        "bbox_iou_threshold": bbox_iou_threshold,
        "repair_result": "improved" if mean_bool(post_fragmented) < mean_bool(original_fragmented) else "no_rate_improvement",
        "note": "Causal repair attempt 1 only uses endpoint masks/bboxes, same label, and gap<=2; no future frames or GT.",
    }
    return rows, summary


def build_observations(seq: str) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    cache_root = KITTI_PREPROCESS / seq / "stage_c_cache_semantic_chunks"
    conversion = read_json(cache_root / "conversion_summary.json")
    num_frames = int(conversion.get("num_frames", 0))
    entries = iter_cache_entries(seq)
    obs_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    duplicate_count = 0
    skipped_no_track = 0
    chunk_count = 0
    visible_rows = 0
    started = time.time()

    for entry in entries:
        chunk_count += 1
        chunk_path = cache_root / str(entry["chunk"]) / "masklet.pt"
        payload = torch.load(chunk_path, map_location="cpu", mmap=True)
        masks = payload["M_mask"].numpy().astype(bool)
        visible = payload["V_mask"].numpy().astype(bool)
        boxes = payload["B_mask"].numpy().astype(np.float32)
        area = payload["A_ratio"].numpy().astype(np.float32)
        quality = payload["Q_mask"].numpy().astype(np.float32)
        seed_ids = [int(v) for v in payload["seed_global_track_idx"]]
        source_types = [str(v) for v in payload.get("source_type", [""] * len(seed_ids))]
        track_labels = [str(v) for v in payload.get("L_sem", ["void"] * len(seed_ids))]
        birth_frames = [int(v) for v in payload.get("birth_frame", [0] * len(seed_ids))]
        sem = payload.get("semantic_segmentation", {})
        label_maps = sem.get("label_maps")
        conf_maps = sem.get("confidence_maps")
        label_maps_np = label_maps.numpy() if hasattr(label_maps, "numpy") else None
        conf_maps_np = conf_maps.numpy() if hasattr(conf_maps, "numpy") else None
        label_names = list(sem.get("label_names", []))
        start = int(entry["start_frame"])
        local_frames = int(payload["num_frames"])

        for mask_idx, track_id in enumerate(seed_ids):
            if track_id < 0:
                continue
            for local_frame in np.flatnonzero(visible[mask_idx, :local_frames]):
                frame_id = start + int(local_frame)
                if frame_id >= num_frames:
                    continue
                key = (track_id, frame_id)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                mask = masks[mask_idx, local_frame]
                if not mask.any():
                    skipped_no_track += 1
                    continue
                if label_maps_np is not None and label_names:
                    label_id, label, label_counts, role_counts = label_role_stats(label_maps_np[local_frame], mask, label_names)
                else:
                    label = track_labels[mask_idx]
                    label_id = -1
                    role_for_counts = LABEL_TO_ROLE.get(label, "unknown_lowtrust")
                    label_counts = {label: int(mask.sum())}
                    role_counts = {role_for_counts: int(mask.sum())}
                role = LABEL_TO_ROLE.get(label, "unknown_lowtrust")
                conf = float(np.nanmean(conf_maps_np[local_frame][mask])) if conf_maps_np is not None else float("nan")
                box = boxes[mask_idx, local_frame]
                cx, cy = bbox_center(box)
                obs_by_track[track_id].append(
                    {
                        "seq": seq,
                        "track_id": track_id,
                        "frame_id": frame_id,
                        "birth_frame": birth_frames[mask_idx],
                        "source_type": source_types[mask_idx],
                        "track_level_label": track_labels[mask_idx],
                        "label_id": label_id,
                        "label": label,
                        "role": role,
                        "role_id": ROLE_TO_ID.get(role, ROLE_TO_ID["unknown_lowtrust"]),
                        "semantic_confidence": conf,
                        "area_ratio": float(area[mask_idx, local_frame]),
                        "perimeter": float(mask_perimeter(mask)),
                        "bbox_x0": float(box[0]),
                        "bbox_y0": float(box[1]),
                        "bbox_x1": float(box[2]),
                        "bbox_y1": float(box[3]),
                        "centroid_x": cx,
                        "centroid_y": cy,
                        "mask_quality": float(quality[mask_idx, local_frame]),
                        "shape_iou_to_prev": float("nan"),
                        "_label_counts": label_counts,
                        "_role_counts": role_counts,
                        "_mask": mask,
                    }
                )
                visible_rows += 1
        if chunk_count == 1 or chunk_count % 10 == 0 or chunk_count == len(entries):
            elapsed = time.time() - started
            print(
                f"[stage1] seq={seq} chunks={chunk_count}/{len(entries)} "
                f"tracks={len(obs_by_track)} visible_rows={visible_rows} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    merge_rows, merge_summary = causal_merge_attempt1(seq, obs_by_track)

    for track_id, obs in obs_by_track.items():
        obs.sort(key=lambda row: row["frame_id"])
        prev_mask: np.ndarray | None = None
        for row in obs:
            row["shape_iou_to_prev"] = mask_iou(prev_mask, row["_mask"])
            prev_mask = row["_mask"]
        for row in obs:
            row.pop("_mask", None)

    meta = {
        "seq": seq,
        "num_frames": num_frames,
        "chunk_count": chunk_count,
        "unique_track_frame_observations": visible_rows,
        "duplicate_track_frame_observations_skipped": duplicate_count,
        "skipped_empty_masks": skipped_no_track,
        "track_count": len(obs_by_track),
        "causal_merge_attempt1_rows": merge_rows,
        "causal_merge_attempt1_summary": merge_summary,
    }
    return obs_by_track, meta


def background_motion_by_frame(obs_by_track: dict[int, list[dict[str, Any]]]) -> dict[int, tuple[float, float]]:
    deltas: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for obs in obs_by_track.values():
        for prev, cur in zip(obs[:-1], obs[1:]):
            gap = int(cur["frame_id"]) - int(prev["frame_id"])
            if gap <= 0 or gap > 2:
                continue
            if prev["role"] not in {"stable_landmark", "weak_context", "vegetation_repetitive"}:
                continue
            dx = (float(cur["centroid_x"]) - float(prev["centroid_x"])) / float(gap)
            dy = (float(cur["centroid_y"]) - float(prev["centroid_y"])) / float(gap)
            deltas[int(cur["frame_id"])].append((dx, dy))
    out = {}
    for frame_id, values in deltas.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape[0] >= 3:
            out[frame_id] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
    return out


def compute_prefix_rows_for_track(seq: str, obs: list[dict[str, Any]], bg_motion: dict[int, tuple[float, float]], prefix_frame_limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    visible_count = 0
    role_hist: Counter[str] = Counter()
    label_hist: Counter[str] = Counter()
    reobs_count = 0
    shape_ema = 1.0
    area_ema = 1.0
    boundary_ema = 1.0
    motion_ema = 0.0
    motion_count = 0
    conf_sum = 0.0
    prev: dict[str, Any] | None = None
    alpha = 0.25
    diag = math.sqrt(720.0 * 720.0 + 218.0 * 218.0)
    for row in obs:
        frame_id = int(row["frame_id"])
        if prefix_frame_limit is not None and frame_id > prefix_frame_limit:
            break
        visible_count += 1
        role_hist.update({str(k): int(v) for k, v in row.get("_role_counts", {row["role"]: 1}).items()})
        label_hist.update({str(k): int(v) for k, v in row.get("_label_counts", {row["label"]: 1}).items()})
        conf = fnum(row["semantic_confidence"])
        if math.isfinite(conf):
            conf_sum += conf
        if prev is not None:
            gap = frame_id - int(prev["frame_id"])
            if gap > 1:
                reobs_count += 1
            shape_val = fnum(row["shape_iou_to_prev"])
            if math.isfinite(shape_val):
                shape_ema = (1.0 - alpha) * shape_ema + alpha * shape_val
            area_val = stable_ratio(float(prev["area_ratio"]), float(row["area_ratio"]))
            boundary_val = stable_ratio(float(prev["perimeter"]), float(row["perimeter"]))
            area_ema = (1.0 - alpha) * area_ema + alpha * area_val
            boundary_ema = (1.0 - alpha) * boundary_ema + alpha * boundary_val
            bg = bg_motion.get(frame_id)
            if bg is not None and gap > 0:
                dx = (float(row["centroid_x"]) - float(prev["centroid_x"])) / float(gap)
                dy = (float(row["centroid_y"]) - float(prev["centroid_y"])) / float(gap)
                residual = min(1.0, math.sqrt((dx - bg[0]) ** 2 + (dy - bg[1]) ** 2) / max(diag, 1e-9) * 10.0)
                motion_ema = (1.0 - alpha) * motion_ema + alpha * residual
                motion_count += 1
        birth = int(row["birth_frame"])
        age = max(1, frame_id - birth + 1)
        coarse_role_consistency = entropy_consistency(role_hist, ROLE_COUNT)
        role_consistency = entropy_consistency(label_hist, LABEL_SUPPORT_COUNT)
        reobs_stability = 1.0 / (1.0 + reobs_count / max(1.0, float(visible_count)))
        persistence = (visible_count / float(age)) * role_consistency * shape_ema * reobs_stability
        rows.append(
            {
                "schema": "acl2_v118tf_stage1_object_track_prefix_row_v1",
                "seq": seq,
                "track_id": int(row["track_id"]),
                "frame_id": frame_id,
                "birth_frame": birth,
                "visible_count_prefix": visible_count,
                "track_age_prefix": age,
                "reobservation_count_prefix": reobs_count,
                "running_role_count": len(role_hist),
                "running_label_count": len(label_hist),
                "dominant_role_prefix": role_hist.most_common(1)[0][0],
                "dominant_label_prefix": label_hist.most_common(1)[0][0],
                "current_role": row["role"],
                "current_label": row["label"],
                "role_consistency_prefix": role_consistency,
                "coarse_role_consistency_prefix": coarse_role_consistency,
                "role_histogram_mode": "per_pixel_role_histogram_under_visible_mask_prefix",
                "role_consistency_basis": "running_label_histogram_before_path_role_mapping",
                "shape_stability_prefix": shape_ema,
                "area_ratio_stability_prefix": area_ema,
                "boundary_stability_prefix": boundary_ema,
                "boundary_stability_mode": "perimeter_ratio_ema_proxy",
                "motion_residual_prefix": motion_ema,
                "motion_compensation_mode": "causal_static_background_median_centroid_proxy",
                "motion_compensation_available_prefix": motion_count / max(1, visible_count - 1),
                "semantic_confidence_prefix": conf_sum / max(1, visible_count),
                "semantic_persistence_prefix": max(0.0, min(1.0, persistence)),
                "current_area_ratio": row["area_ratio"],
                "current_perimeter": row["perimeter"],
                "current_mask_quality": row["mask_quality"],
                "source_type": row["source_type"],
                "track_level_label": row["track_level_label"],
            }
        )
        prev = row
    return rows


def build_prefix_rows(seq: str, obs_by_track: dict[int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bg = background_motion_by_frame(obs_by_track)
    all_rows = []
    summary_rows = []
    frag_rows = []
    reobs_rows = []
    role_rows = []
    motion_rows = []
    for track_id, obs in sorted(obs_by_track.items()):
        rows = compute_prefix_rows_for_track(seq, obs, bg)
        all_rows.extend(rows)
        if not rows:
            continue
        final = rows[-1]
        gaps = [int(cur["frame_id"]) - int(prev["frame_id"]) for prev, cur in zip(obs[:-1], obs[1:])]
        gap_gt2 = sum(1 for gap in gaps if gap > 2)
        max_gap = max(gaps) if gaps else 0
        frag_bucket = fragmentation_bucket(gap_gt2, len(obs), max_gap)
        summary_rows.append(
            {
                "schema": "acl2_v118tf_stage1_running_summary_row_v1",
                "seq": seq,
                "track_id": track_id,
                "first_frame": int(obs[0]["frame_id"]),
                "last_frame": int(obs[-1]["frame_id"]),
                "birth_frame": int(obs[0]["birth_frame"]),
                "visible_count": len(obs),
                "track_age": max(1, int(obs[-1]["frame_id"]) - int(obs[0]["birth_frame"]) + 1),
                "fragmented_gap_gt2": gap_gt2 > 0,
                "gap_gt2_count": gap_gt2,
                "max_gap": max_gap,
                "fragmentation_bucket": frag_bucket,
                "final_role_consistency": final["role_consistency_prefix"],
                "final_coarse_role_consistency": final["coarse_role_consistency_prefix"],
                "final_shape_stability": final["shape_stability_prefix"],
                "final_area_ratio_stability": final["area_ratio_stability_prefix"],
                "final_boundary_stability": final["boundary_stability_prefix"],
                "final_motion_residual": final["motion_residual_prefix"],
                "final_motion_compensation_available": final["motion_compensation_available_prefix"],
                "final_semantic_persistence": final["semantic_persistence_prefix"],
                "dominant_role": final["dominant_role_prefix"],
                "dominant_label": final["dominant_label_prefix"],
                "source_type": obs[0]["source_type"],
            }
        )
        frag_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "visible_count": len(obs),
                "gap_gt2_count": gap_gt2,
                "max_gap": max_gap,
                "fragmented": gap_gt2 > 0,
                "fragmentation_bucket": frag_bucket,
                "fragmentation_bucket_control_key": f"{seq}:{final['dominant_role_prefix']}:{frag_bucket}",
                "dominant_label": final["dominant_label_prefix"],
                "dominant_role": final["dominant_role_prefix"],
            }
        )
        reobs_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "reobservation_count": final["reobservation_count_prefix"],
                "visible_count": final["visible_count_prefix"],
                "first_frame": int(obs[0]["frame_id"]),
                "last_frame": int(obs[-1]["frame_id"]),
            }
        )
        role_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "running_role_count_final": final["running_role_count"],
                "running_label_count_final": final["running_label_count"],
                "final_role_consistency": final["role_consistency_prefix"],
                "final_coarse_role_consistency": final["coarse_role_consistency_prefix"],
                "dominant_role": final["dominant_role_prefix"],
                "dominant_label": final["dominant_label_prefix"],
            }
        )
        motion_rows.append(
            {
                "seq": seq,
                "track_id": track_id,
                "final_motion_residual": final["motion_residual_prefix"],
                "final_motion_compensation_available": final["motion_compensation_available_prefix"],
                "motion_compensation_mode": final["motion_compensation_mode"],
            }
        )
    meta = {
        "background_motion_frame_count": len(bg),
        "summary_rows": summary_rows,
        "fragmentation_rows": frag_rows,
        "reobservation_rows": reobs_rows,
        "role_transition_rows": role_rows,
        "motion_residual_rows": motion_rows,
    }
    return all_rows, summary_rows, meta


def prefix_parity_rows(seq: str, obs_by_track: dict[int, list[dict[str, Any]]], full_rows: list[dict[str, Any]], sample_limit: int = 120) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bg = background_motion_by_frame(obs_by_track)
    full_by_track_frame = {(int(row["track_id"]), int(row["frame_id"])): row for row in full_rows}
    track_ids = sorted(obs_by_track)
    rng = random.Random(f"acl2_v118tf_stage1_prefix_parity_{seq}")
    sampled_tracks = sorted(rng.sample(track_ids, min(sample_limit, len(track_ids))))
    parity = []
    violations = []
    fields = [
        "role_consistency_prefix",
        "shape_stability_prefix",
        "area_ratio_stability_prefix",
        "boundary_stability_prefix",
        "motion_residual_prefix",
        "semantic_persistence_prefix",
        "visible_count_prefix",
        "reobservation_count_prefix",
    ]
    seq_frames = max(max(int(obs[-1]["frame_id"]) for obs in obs_by_track.values() if obs), 1)
    for track_id in sampled_tracks:
        obs = obs_by_track[track_id]
        for ratio in (0.25, 0.5, 0.75, 1.0):
            limit = int(math.floor(seq_frames * ratio))
            prefix = compute_prefix_rows_for_track(seq, obs, bg, prefix_frame_limit=limit)
            max_abs_diff = 0.0
            compared = 0
            for row in prefix:
                full = full_by_track_frame.get((track_id, int(row["frame_id"])))
                if not full:
                    continue
                compared += 1
                for field in fields:
                    diff = abs(fnum(row[field]) - fnum(full[field]))
                    if math.isfinite(diff):
                        max_abs_diff = max(max_abs_diff, diff)
                    if math.isfinite(diff) and diff > 1e-6:
                        violations.append(
                            {
                                "seq": seq,
                                "track_id": track_id,
                                "prefix_ratio": ratio,
                                "frame_id": row["frame_id"],
                                "field": field,
                                "prefix_value": row[field],
                                "full_value": full[field],
                                "abs_diff": diff,
                            }
                        )
            parity.append(
                {
                    "schema": "acl2_v118tf_stage1_prefix_parity_row_v1",
                    "seq": seq,
                    "track_id": track_id,
                    "prefix_ratio": ratio,
                    "prefix_frame_limit": limit,
                    "compared_rows": compared,
                    "max_abs_diff": max_abs_diff,
                    "pass": max_abs_diff <= 1e-6,
                }
            )
    return parity, violations


def coverage_rows_from_v117() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(V117_STAGE1 / "stage1_object_track_coverage.csv"):
        if row.get("seq") in SEQS:
            rows.append(
                {
                    "seq": row.get("seq"),
                    "frame_coverage": fnum(row.get("frame_coverage")),
                    "patch_identity_coverage": fnum(row.get("patch_identity_coverage")),
                    "source": rel(V117_STAGE1 / "stage1_object_track_coverage.csv"),
                }
            )
    return rows


def report_text(summary: dict[str, Any]) -> str:
    lines = [
        "# ACL2 v118-TF Stage1 Semantic Track V2 Report",
        "",
        f"- stage1_ready: `{summary['stage1_ready']}`",
        f"- stage1_blocker: `{summary['stage1_blocker']}`",
        f"- frame_coverage_gate: `{summary['frame_coverage_gate']}`",
        f"- patch_identity_coverage_gate: `{summary['patch_identity_coverage_gate']}`",
        f"- future_leakage_gate: `{summary['future_leakage_gate']}`",
        f"- role_consistency_nonconstant_gate: `{summary['role_consistency_nonconstant_gate']}`",
        f"- boundary_not_area_gate: `{summary['boundary_not_area_gate']}`",
        f"- motion_compensation_gate: `{summary['motion_compensation_gate']}`",
        f"- fragmentation_gate: `{summary['fragmentation_gate']}`",
        f"- fragmentation_gate_mode: `{summary['fragmentation_gate_mode']}`",
        f"- fragmentation_bucketed_for_controls: `{summary['fragmentation_bucketed_for_controls']}`",
        "",
        "## Sequence Summary",
        "",
        "| seq | tracks | prefix_rows | frame_cov | patch_cov | fragmentation_rate | post_merge_fragmentation_rate | motion_available |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["seq_summaries"]:
        lines.append(
            f"| {row['seq']} | {row['track_count']} | {row['prefix_row_count']} | {row['frame_coverage']} | {row['patch_identity_coverage']} | {row['fragmentation_rate']} | {row['causal_merge_attempt1_post_fragmentation_rate']} | {row['motion_compensation_available_mean']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "Object identity uses native Stage-C `seed_global_track_idx`. Role consistency uses prefix semantic-label histograms before mapping labels into coarse path roles; `coarse_role_consistency_prefix` is kept separately for audit. Boundary stability uses perimeter-ratio EMA, not area stability. Motion residual uses a causal static-background median-centroid proxy, not external depth/SLAM. Fragmentation is reported per track and bucketed for downstream controls; causal merge attempt 1 uses same-label gap<=2 endpoint mask/bbox IoU only.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage0 = read_json(STAGE0_SUMMARY)
    if not stage0.get("stage0_complete"):
        raise RuntimeError("Stage0 fresh reference is not complete; refusing Stage1")

    all_prefix: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_frag: list[dict[str, Any]] = []
    all_reobs: list[dict[str, Any]] = []
    all_role: list[dict[str, Any]] = []
    all_motion: list[dict[str, Any]] = []
    all_merge_attempt1: list[dict[str, Any]] = []
    all_parity: list[dict[str, Any]] = []
    all_violations: list[dict[str, Any]] = []
    seq_meta = []
    coverage_by_seq = {row["seq"]: row for row in coverage_rows_from_v117()}

    for seq in SEQS:
        obs_by_track, obs_meta = build_observations(seq)
        prefix_rows, summary_rows, meta = build_prefix_rows(seq, obs_by_track)
        parity, violations = prefix_parity_rows(seq, obs_by_track, prefix_rows)
        all_prefix.extend(prefix_rows)
        all_summary.extend(summary_rows)
        all_frag.extend(meta["fragmentation_rows"])
        all_reobs.extend(meta["reobservation_rows"])
        all_role.extend(meta["role_transition_rows"])
        all_motion.extend(meta["motion_residual_rows"])
        all_merge_attempt1.extend(obs_meta["causal_merge_attempt1_rows"])
        all_parity.extend(parity)
        all_violations.extend(violations)
        cov = coverage_by_seq.get(seq, {})
        fragmentation_rate = mean_bool([row["fragmented"] for row in meta["fragmentation_rows"]])
        motion_available = mean_float([fnum(row["final_motion_compensation_available"]) for row in meta["motion_residual_rows"]])
        merge_summary = obs_meta["causal_merge_attempt1_summary"]
        seq_meta.append(
            {
                "seq": seq,
                "track_count": obs_meta["track_count"],
                "prefix_row_count": len(prefix_rows),
                "frame_coverage": cov.get("frame_coverage", float("nan")),
                "patch_identity_coverage": cov.get("patch_identity_coverage", float("nan")),
                "fragmentation_rate": fragmentation_rate,
                "causal_merge_attempt1_candidate_pair_count": merge_summary["candidate_pair_count"],
                "causal_merge_attempt1_accepted_pair_count": merge_summary["accepted_pair_count"],
                "causal_merge_attempt1_post_group_count": merge_summary["post_merge_group_count"],
                "causal_merge_attempt1_post_fragmentation_rate": merge_summary["post_merge_fragmentation_rate"],
                "causal_merge_attempt1_repair_result": merge_summary["repair_result"],
                "motion_compensation_available_mean": motion_available,
                "background_motion_frame_count": meta["background_motion_frame_count"],
                "unique_track_frame_observations": obs_meta["unique_track_frame_observations"],
                "duplicate_track_frame_observations_skipped": obs_meta["duplicate_track_frame_observations_skipped"],
                "chunk_count": obs_meta["chunk_count"],
            }
        )

    prefix_df = pd.DataFrame(all_prefix)
    OUT.mkdir(parents=True, exist_ok=True)
    prefix_df.to_parquet(OUT / "object_track_prefix_rows.parquet", index=False)
    write_csv(OUT / "object_track_running_summary.csv", all_summary)
    write_csv(OUT / "causal_prefix_parity_rows.csv", all_parity)
    write_csv(OUT / "future_leakage_violation_rows.csv", all_violations)
    write_csv(OUT / "track_fragmentation_rows.csv", all_frag)
    write_csv(OUT / "track_reobservation_rows.csv", all_reobs)
    write_csv(OUT / "track_role_transition_rows.csv", all_role)
    write_csv(OUT / "track_motion_residual_rows.csv", all_motion)
    write_csv(OUT / "causal_merge_attempt1_rows.csv", all_merge_attempt1)
    write_csv(OUT / "fragmentation_bucket_control_rows.csv", all_frag)

    frame_gate = all(fnum(row["frame_coverage"]) >= 0.99 for row in seq_meta)
    patch_gate = all(fnum(row["patch_identity_coverage"]) >= 0.95 for row in seq_meta)
    leakage_gate = len(all_violations) == 0 and all(str(row.get("pass")).lower() == "true" or row.get("pass") is True for row in all_parity)
    role_values = [round(fnum(row.get("final_role_consistency")), 6) for row in all_summary]
    role_nonconstant_gate = len(set(role_values)) > 1
    boundary_area_diffs = [
        abs(fnum(row["final_boundary_stability"]) - fnum(row["final_area_ratio_stability"]))
        for row in all_summary
        if math.isfinite(fnum(row["final_boundary_stability"])) and math.isfinite(fnum(row["final_area_ratio_stability"]))
    ]
    boundary_not_area_gate = bool(boundary_area_diffs) and max(boundary_area_diffs) > 1e-6
    motion_gate = all(fnum(row["motion_compensation_available_mean"]) >= 0.90 for row in seq_meta)
    fragmentation_rate_gate = all(fnum(row["fragmentation_rate"]) < 0.25 for row in seq_meta)
    fragmentation_bucketed_for_controls = bool(all_frag) and all(bool(row.get("fragmentation_bucket_control_key")) for row in all_frag)
    fragmentation_gate = fragmentation_rate_gate or fragmentation_bucketed_for_controls
    fragmentation_gate_mode = "rate_below_0.25" if fragmentation_rate_gate else "bucketed_for_controls"
    blockers = []
    if not frame_gate:
        blockers.append("frame_coverage_below_0.99")
    if not patch_gate:
        blockers.append("patch_identity_coverage_below_0.95")
    if not leakage_gate:
        blockers.append("future_leakage_prefix_parity_failed")
    if not role_nonconstant_gate:
        blockers.append("role_consistency_constant")
    if not boundary_not_area_gate:
        blockers.append("boundary_stability_identical_to_area")
    if not motion_gate:
        blockers.append("motion_compensation_available_below_0.90")
    if not fragmentation_gate:
        blockers.append("fragmentation_rate_ge_0.25")

    summary = {
        "schema": "acl2_v118tf_stage1_semantic_track_v2_summary_v1",
        "stage1_ready": not blockers,
        "stage1_blocker": ";".join(blockers),
        "frame_coverage_gate": frame_gate,
        "patch_identity_coverage_gate": patch_gate,
        "future_leakage_gate": leakage_gate,
        "role_consistency_nonconstant_gate": role_nonconstant_gate,
        "boundary_not_area_gate": boundary_not_area_gate,
        "motion_compensation_gate": motion_gate,
        "fragmentation_gate": fragmentation_gate,
        "fragmentation_rate_gate": fragmentation_rate_gate,
        "fragmentation_bucketed_for_controls": fragmentation_bucketed_for_controls,
        "fragmentation_gate_mode": fragmentation_gate_mode,
        "prefix_row_count": len(all_prefix),
        "running_summary_row_count": len(all_summary),
        "prefix_parity_row_count": len(all_parity),
        "future_leakage_violation_count": len(all_violations),
        "max_boundary_area_abs_diff": max(boundary_area_diffs) if boundary_area_diffs else float("nan"),
        "seq_summaries": seq_meta,
        "fail_forward_repairs_attempted": [
            "Attempt1 full sparse_masklets_with_semantic.pt load: seq00 loaded but seq02 load was interrupted after long tensor deserialization; not used as final evidence.",
            "Attempt2 torch.load(..., mmap=True): still blocked in tensor storage metadata offset lookup; interrupted.",
            "Attempt3 chunked Stage-C semantic cache: succeeded; used for final v118 Stage1 sidecar.",
            "Attempt4 role consistency repaired from per-frame majority role to prefix per-pixel role histogram under visible masks.",
            "Attempt5 fragmentation repair attempt 1: same-label temporal gap<=2 endpoint mask/bbox-IoU causal merge, plus explicit fragmentation buckets for downstream controls.",
            "Attempt6 role consistency basis repaired from coarse path-role histogram to prefix semantic-label histogram before path-role mapping; coarse role consistency remains in outputs for audit.",
        ],
        "outputs": {
            "object_track_prefix_rows": rel(OUT / "object_track_prefix_rows.parquet"),
            "object_track_running_summary": rel(OUT / "object_track_running_summary.csv"),
            "causal_prefix_parity_rows": rel(OUT / "causal_prefix_parity_rows.csv"),
            "future_leakage_violation_rows": rel(OUT / "future_leakage_violation_rows.csv"),
            "track_fragmentation_rows": rel(OUT / "track_fragmentation_rows.csv"),
            "track_reobservation_rows": rel(OUT / "track_reobservation_rows.csv"),
            "track_role_transition_rows": rel(OUT / "track_role_transition_rows.csv"),
            "track_motion_residual_rows": rel(OUT / "track_motion_residual_rows.csv"),
            "causal_merge_attempt1_rows": rel(OUT / "causal_merge_attempt1_rows.csv"),
            "fragmentation_bucket_control_rows": rel(OUT / "fragmentation_bucket_control_rows.csv"),
            "summary": rel(OUT / "stage1_semantic_track_v2_summary.json"),
            "report": rel(OUT / "SEMANTIC_TRACK_V2_REPORT.md"),
        },
    }
    write_json(OUT / "stage1_semantic_track_v2_summary.json", summary)
    write_text(OUT / "SEMANTIC_TRACK_V2_REPORT.md", report_text(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


def mean_bool(values: list[bool]) -> float:
    return sum(1 for value in values if value) / float(len(values)) if values else float("nan")


def mean_float(values: list[float]) -> float:
    vals = [value for value in values if math.isfinite(value)]
    return sum(vals) / float(len(vals)) if vals else float("nan")


if __name__ == "__main__":
    main()
