#!/usr/bin/env python3
"""Build v117 Stage1 object/track identity sidecar from Stage-C seed tracks.

This stage is a cue-readiness artifact. It does not run geometry and does not
claim semantic causality. Native identity source priority is
seed_global_track_idx from the Stage-C semantic chunk cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability"
OUT = RESULT_ROOT / "stage1_object_identity"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
KITTI_PREPROCESS = ROOT / "results/kitti_preprocess"
SEQS = ("00", "02")
PATCH_SIZE = 14
HS_SIZE = 518
NO_TRACK = -1

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
    "crosswalk": "weak_context",
    "building": "stable_landmark",
    "house": "stable_landmark",
    "bridge": "stable_landmark",
    "other_construction": "stable_landmark",
    "sky": "sky_lowobs",
    "mountain": "stable_landmark",
    "billboard_or_bulletin_board": "stable_landmark",
    "wheeled_machine": "dynamic",
    "other_machine": "dynamic",
    "tree": "vegetation_repetitive",
    "stone": "stable_landmark",
    "flower": "vegetation_repetitive",
    "other_plant": "vegetation_repetitive",
    "trash_can": "weak_context",
    "car": "dynamic",
    "traffic sign": "stable_landmark",
    "stair": "stable_landmark",
}


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_cache_entries(cache_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with (cache_dir / "cache_index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["chunk_idx"]))


def resize_crop_nearest(arr: np.ndarray, size: int = HS_SIZE, patch_size: int = PATCH_SIZE) -> np.ndarray:
    image = Image.fromarray(arr.astype(np.int32), mode="I")
    w1, h1 = image.size
    scale = float(size) / float(max(w1, h1))
    new_size = tuple(int(round(x * scale)) for x in image.size)
    image = image.resize(new_size, Image.Resampling.NEAREST)
    w, h = image.size
    cx, cy = w // 2, h // 2
    halfw = ((2 * cx) // patch_size) * (patch_size // 2)
    halfh = ((2 * cy) // patch_size) * (patch_size // 2)
    if w == h:
        halfh = int(3 * halfw / 4)
    return np.asarray(image.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh)), dtype=np.int32)


def patch_dominant_track(track_map: np.ndarray) -> np.ndarray:
    h, w = track_map.shape
    patch_h = h // PATCH_SIZE
    patch_w = w // PATCH_SIZE
    out = np.full((patch_h * patch_w,), NO_TRACK, dtype=np.int32)
    idx = 0
    for py in range(patch_h):
        y0 = py * PATCH_SIZE
        y1 = y0 + PATCH_SIZE
        for px in range(patch_w):
            x0 = px * PATCH_SIZE
            x1 = x0 + PATCH_SIZE
            vals = track_map[y0:y1, x0:x1].reshape(-1)
            vals = vals[vals >= 0]
            if vals.size:
                counts = np.bincount(vals.astype(np.int64))
                out[idx] = int(counts.argmax())
            idx += 1
    return out


def robust_stability(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size <= 1:
        return 1.0 if vals.size == 1 else 0.0
    mean = float(vals.mean())
    if abs(mean) < 1e-9:
        return 0.0
    cv = float(vals.std()) / abs(mean)
    return float(max(0.0, min(1.0, 1.0 / (1.0 + cv))))


def centroid_motion_risk(boxes: np.ndarray) -> float:
    if boxes.shape[0] <= 1:
        return 0.0
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    dx = np.diff(cx)
    dy = np.diff(cy)
    diag = math.sqrt(720.0 * 720.0 + 218.0 * 218.0)
    speeds = np.sqrt(dx * dx + dy * dy) / max(diag, 1e-6)
    return float(max(0.0, min(1.0, np.nanmean(speeds) * 10.0))) if speeds.size else 0.0


def load_track_stats(seq: str) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    path = KITTI_PREPROCESS / seq / "sparse_masklets_with_semantic.pt"
    payload = torch.load(path, map_location="cpu")
    num_frames = int(payload["num_frames"])
    track_stats: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for track_id, track in enumerate(payload["tracks"]):
        frames = [int(v) for v in track.get("frames", [])]
        if not frames:
            continue
        label = str(track.get("L_sem", "void"))
        role = LABEL_TO_ROLE.get(label, "unknown_lowtrust")
        area = track.get("area_ratio")
        area_np = area.detach().cpu().numpy().astype(np.float32) if hasattr(area, "detach") else np.asarray(area, dtype=np.float32)
        boxes = track.get("boxes")
        box_np = boxes.detach().cpu().numpy().astype(np.float32) if hasattr(boxes, "detach") else np.asarray(boxes, dtype=np.float32)
        birth = int(track.get("birth_frame", min(frames)))
        age = max(frames) - min(frames) + 1
        visible_count = len(set(frames))
        area_stability = robust_stability(area_np)
        boundary_stability = area_stability
        motion_risk = centroid_motion_risk(box_np) if box_np.ndim == 2 and box_np.shape[1] >= 4 else 0.0
        role_consistency = 1.0
        persistence = max(0.0, min(1.0, (visible_count / float(age + 1)) * role_consistency * area_stability * boundary_stability))
        row = {
            "schema": "acl2_v117tf_stage1_track_summary_row_v1",
            "seq": seq,
            "object_or_track_id": track_id,
            "track_source": "seed_global_track_idx",
            "semantic_role": role,
            "semantic_role_id": ROLE_TO_ID.get(role, ROLE_TO_ID["unknown_lowtrust"]),
            "semantic_label": label,
            "semantic_confidence": float(track.get("W_sem", 0.0)),
            "track_age": age,
            "birth_frame": birth,
            "first_visible_frame": min(frames),
            "last_visible_frame": max(frames),
            "visible_count": visible_count,
            "reobservation_count": max(0, visible_count - 1),
            "miss_count": max(0, age - visible_count),
            "role_consistency": role_consistency,
            "area_stability": area_stability,
            "boundary_stability": boundary_stability,
            "boundary_stability_mode": "area_stability_proxy_no_boundary_edge_metric",
            "centroid_motion_risk": motion_risk,
            "object_persistence": persistence,
            "source_type": str(track.get("source_type", "")),
            "tracking_source": str(track.get("tracking_source", "")),
            "mask_source": str(track.get("mask_source", "")),
            "num_frames_in_sequence": num_frames,
        }
        track_stats[track_id] = row
        rows.append(row)
    return track_stats, rows


def build_seq(seq: str) -> dict[str, Any]:
    cache_dir = KITTI_PREPROCESS / seq / "stage_c_cache_semantic_chunks"
    conversion = read_json(cache_dir / "conversion_summary.json")
    num_frames = int(conversion["num_frames"])
    track_stats, track_rows = load_track_stats(seq)

    patch_count = None
    track_ids_by_frame: np.ndarray | None = None
    processed = np.zeros((num_frames,), dtype=bool)
    conflict_patch_count = 0
    duplicate_patch_count = 0
    sample_rows: list[dict[str, Any]] = []

    for entry in iter_cache_entries(cache_dir):
        chunk_dir = cache_dir / str(entry["chunk"])
        payload = torch.load(chunk_dir / "masklet.pt", map_location="cpu")
        masks = payload["M_mask"].numpy().astype(bool)
        visible = payload["V_mask"].numpy().astype(bool)
        seed_ids = [int(v) for v in payload["seed_global_track_idx"]]
        start = int(entry["start_frame"])
        end = int(entry["end_frame"])
        for local_frame, global_frame in enumerate(range(start, end)):
            if global_frame >= num_frames:
                continue
            track_map = np.full((int(payload["frame_height"]), int(payload["frame_width"])), NO_TRACK, dtype=np.int32)
            for mask_idx, seed_id in enumerate(seed_ids):
                if not visible[mask_idx, local_frame]:
                    continue
                track_map[masks[mask_idx, local_frame]] = seed_id
            patch_tracks = patch_dominant_track(resize_crop_nearest(track_map))
            if patch_count is None:
                patch_count = int(patch_tracks.shape[0])
                track_ids_by_frame = np.full((num_frames, patch_count), NO_TRACK, dtype=np.int32)
            assert track_ids_by_frame is not None
            if processed[global_frame]:
                old = track_ids_by_frame[global_frame]
                valid = (old >= 0) | (patch_tracks >= 0)
                duplicate_patch_count += int(valid.sum())
                conflict_patch_count += int(((old != patch_tracks) & valid).sum())
                continue
            track_ids_by_frame[global_frame] = patch_tracks
            processed[global_frame] = True
            if len(sample_rows) < 200:
                for patch_id, track_id in enumerate(patch_tracks[: min(16, len(patch_tracks))]):
                    stat = track_stats.get(int(track_id), {}) if int(track_id) >= 0 else {}
                    sample_rows.append(
                        {
                            "seq": seq,
                            "frame_id": global_frame,
                            "patch_id": patch_id,
                            "object_or_track_id": int(track_id),
                            "track_source": "seed_global_track_idx" if int(track_id) >= 0 else "no_track",
                            "semantic_role": stat.get("semantic_role", "unknown_lowtrust" if int(track_id) < 0 else ""),
                            "semantic_label": stat.get("semantic_label", ""),
                            "semantic_confidence": stat.get("semantic_confidence", ""),
                            "patch_purity": "",
                            "boundary_risk": "" if not stat else 1.0 - float(stat.get("boundary_stability", 0.0)),
                            "track_age": stat.get("track_age", ""),
                            "visible_count": stat.get("visible_count", ""),
                            "reobservation_count": stat.get("reobservation_count", ""),
                            "miss_count": stat.get("miss_count", ""),
                            "role_consistency": stat.get("role_consistency", ""),
                            "area_stability": stat.get("area_stability", ""),
                            "boundary_stability": stat.get("boundary_stability", ""),
                            "centroid_motion_risk": stat.get("centroid_motion_risk", ""),
                            "object_persistence": stat.get("object_persistence", ""),
                        }
                    )

    if track_ids_by_frame is None or patch_count is None:
        raise RuntimeError(f"no patch tracks built for seq {seq}")

    persistence = np.zeros_like(track_ids_by_frame, dtype=np.float32)
    role_consistency = np.zeros_like(track_ids_by_frame, dtype=np.float32)
    motion_risk = np.ones_like(track_ids_by_frame, dtype=np.float32)
    boundary_stability = np.zeros_like(track_ids_by_frame, dtype=np.float32)
    role_ids = np.full_like(track_ids_by_frame, ROLE_TO_ID["unknown_lowtrust"], dtype=np.int16)
    for track_id, stat in track_stats.items():
        mask = track_ids_by_frame == track_id
        persistence[mask] = float(stat["object_persistence"])
        role_consistency[mask] = float(stat["role_consistency"])
        motion_risk[mask] = float(stat["centroid_motion_risk"])
        boundary_stability[mask] = float(stat["boundary_stability"])
        role_ids[mask] = int(stat["semantic_role_id"])

    seq_out = OUT / seq
    seq_out.mkdir(parents=True, exist_ok=True)
    np.save(seq_out / "semantic_track_id.npy", track_ids_by_frame)
    np.save(seq_out / "semantic_track_persistence.npy", persistence)
    np.save(seq_out / "semantic_track_role_consistency.npy", role_consistency)
    np.save(seq_out / "semantic_track_motion_risk.npy", motion_risk)
    np.save(seq_out / "semantic_track_boundary_stability.npy", boundary_stability)
    np.save(seq_out / "semantic_track_role_id.npy", role_ids)
    write_csv(seq_out / "stage1_track_summary_rows.csv", track_rows)
    write_csv(seq_out / "stage1_object_rows_sample.csv", sample_rows)

    frame_coverage = float(processed.sum()) / float(num_frames)
    assigned = track_ids_by_frame >= 0
    patch_identity_coverage = float(assigned.sum()) / float(track_ids_by_frame.size)
    duplicate_conflict_rate = float(conflict_patch_count) / float(duplicate_patch_count) if duplicate_patch_count else 0.0
    role_mapping_coverage = 1.0 if all(row["semantic_role"] in ROLE_TO_ID for row in track_rows) else 0.0
    frag_by_label = Counter(row["semantic_label"] for row in track_rows)
    summary = {
        "schema": "acl2_v117tf_stage1_seq_summary_v1",
        "seq": seq,
        "track_source": "seed_global_track_idx",
        "num_frames": num_frames,
        "processed_frames": int(processed.sum()),
        "frame_coverage": frame_coverage,
        "patch_count": patch_count,
        "patch_identity_coverage": patch_identity_coverage,
        "role_mapping_coverage": role_mapping_coverage,
        "future_frame_leakage": False,
        "future_frame_leakage_note": "Stage-C native seed_global_track_idx is read as existing identity; this script does not perform future-frame smoothing.",
        "duplicate_conflicting_track_assignment_rate": duplicate_conflict_rate,
        "duplicate_patch_count": duplicate_patch_count,
        "conflict_patch_count": conflict_patch_count,
        "native_identity_track_count": len(track_rows),
        "pseudo_identity_track_count": 0,
        "track_fragmentation_statistics": {
            "track_count": len(track_rows),
            "labels_with_multiple_tracks_count": sum(1 for v in frag_by_label.values() if v > 1),
            "max_tracks_per_label": max(frag_by_label.values(), default=0),
            "top_labels_by_track_count": dict(frag_by_label.most_common(12)),
        },
        "outputs": {
            "semantic_track_id": rel(seq_out / "semantic_track_id.npy"),
            "semantic_track_persistence": rel(seq_out / "semantic_track_persistence.npy"),
            "semantic_track_role_consistency": rel(seq_out / "semantic_track_role_consistency.npy"),
            "semantic_track_motion_risk": rel(seq_out / "semantic_track_motion_risk.npy"),
            "semantic_track_boundary_stability": rel(seq_out / "semantic_track_boundary_stability.npy"),
            "semantic_track_role_id": rel(seq_out / "semantic_track_role_id.npy"),
            "track_summary_rows": rel(seq_out / "stage1_track_summary_rows.csv"),
            "object_rows_sample": rel(seq_out / "stage1_object_rows_sample.csv"),
        },
    }
    write_json(seq_out / "stage1_object_track_summary.json", summary)
    return summary


def report_text(overall: dict[str, Any], seq_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# v117 Stage1 Semantic Track Readiness",
        "",
        f"- stage1_ready: `{overall['stage1_ready']}`",
        f"- stage0_runtime_gate_status: `{overall['stage0_runtime_gate_status']}`",
        f"- frame_coverage_gate: `{overall['frame_coverage_gate']}`",
        f"- patch_identity_coverage_gate: `{overall['patch_identity_coverage_gate']}`",
        f"- role_mapping_gate: `{overall['role_mapping_gate']}`",
        f"- conflict_gate: `{overall['conflict_gate']}`",
        "",
        "| seq | frame_coverage | patch_identity_coverage | tracks | conflict_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in seq_summaries:
        lines.append(
            f"| {row['seq']} | {row['frame_coverage']} | {row['patch_identity_coverage']} | "
            f"{row['native_identity_track_count']} | {row['duplicate_conflicting_track_assignment_rate']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "This sidecar uses native Stage-C `seed_global_track_idx`. No pseudo-track was needed for 00/02.",
        "Boundary stability is an area-stability proxy because no explicit contour stability metric is present in the source tracks.",
        "This is a readiness artifact only; runtime action remains governed by the current Stage0 gate status.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="reuse existing per-seq Stage1 artifacts and rewrite only the top-level summary/report",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stage0_summary = read_json(STAGE0 / "stage0_frozen_facts.json")
    stage0_runtime_status = (
        "passed" if bool(stage0_summary.get("stage0_complete")) else "blocked_raw_reference_artifacts_missing"
    )
    if args.summary_only:
        seq_summaries = [read_json(OUT / seq / "stage1_object_track_summary.json") for seq in SEQS]
        missing = [seq for seq, summary in zip(SEQS, seq_summaries) if not summary]
        if missing:
            raise RuntimeError(f"missing per-seq Stage1 summaries for summary-only mode: {missing}")
    else:
        seq_summaries = [build_seq(seq) for seq in SEQS]
    coverage_rows = []
    sample_rows_all = []
    for summary in seq_summaries:
        seq = summary["seq"]
        coverage_rows.append(
            {
                "seq": seq,
                "frame_coverage": summary["frame_coverage"],
                "patch_identity_coverage": summary["patch_identity_coverage"],
                "role_mapping_coverage": summary["role_mapping_coverage"],
                "future_frame_leakage": summary["future_frame_leakage"],
                "duplicate_conflicting_track_assignment_rate": summary["duplicate_conflicting_track_assignment_rate"],
                "native_identity_track_count": summary["native_identity_track_count"],
                "pseudo_identity_track_count": summary["pseudo_identity_track_count"],
            }
        )
        sample_rows_all.extend(read_csv_dicts(OUT / seq / "stage1_object_rows_sample.csv"))

    frame_gate = all(row["frame_coverage"] >= 0.99 for row in seq_summaries)
    patch_gate = all(row["patch_identity_coverage"] >= 0.90 for row in seq_summaries)
    role_gate = all(row["role_mapping_coverage"] == 1.0 for row in seq_summaries)
    leakage_gate = all(not row["future_frame_leakage"] for row in seq_summaries)
    conflict_gate = all(row["duplicate_conflicting_track_assignment_rate"] <= 0.01 for row in seq_summaries)
    overall = {
        "schema": "acl2_v117tf_stage1_object_track_summary_v1",
        "stage1_ready": frame_gate and patch_gate and role_gate and leakage_gate and conflict_gate,
        "stage0_runtime_gate_status": stage0_runtime_status,
        "frame_coverage_gate": frame_gate,
        "patch_identity_coverage_gate": patch_gate,
        "role_mapping_gate": role_gate,
        "future_frame_leakage_gate": leakage_gate,
        "conflict_gate": conflict_gate,
        "seq_summaries": seq_summaries,
        "outputs": {
            "stage1_object_track_summary": rel(OUT / "stage1_object_track_summary.json"),
            "stage1_object_track_coverage": rel(OUT / "stage1_object_track_coverage.csv"),
            "stage1_object_rows_sample": rel(OUT / "stage1_object_rows_sample.csv"),
            "report": rel(OUT / "SEMANTIC_TRACK_READINESS_REPORT.md"),
        },
    }
    write_json(OUT / "stage1_object_track_summary.json", overall)
    write_csv(OUT / "stage1_object_track_coverage.csv", coverage_rows)
    write_csv(OUT / "stage1_object_rows_sample.csv", sample_rows_all[:500])
    write_text(OUT / "SEMANTIC_TRACK_READINESS_REPORT.md", report_text(overall, seq_summaries))
    print(json.dumps(clean_json(overall), indent=2, sort_keys=True, ensure_ascii=False))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
