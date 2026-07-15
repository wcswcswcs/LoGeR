#!/usr/bin/env python3
"""Build v117 Stage2 memory-provenance readiness artifacts.

This is a readiness/audit builder. It aligns the native Stage-C semantic track
identity to the memory-unit surfaces that are currently auditable, and reports
blocked or coarse provenance where real hooks/raw action targets are missing.
It does not run geometry and does not claim runtime causality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability"
OUT = RESULT_ROOT / "stage2_memory_provenance"
KITTI_PREPROCESS = ROOT / "results/kitti_preprocess"
STAGE1 = RESULT_ROOT / "stage1_object_identity"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
V115_STAGE2 = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control/stage2_alignment_cues"
V110R_STAGE3_ACTION_CONFIG = (
    ROOT
    / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
    / "stage3_pilot_00_02/action_config_rows.csv"
)

SEQS = ("00", "02")
NO_TRACK = -1
LINGBOT_GRID_H = 20
LINGBOT_GRID_W = 36
LINGBOT_PATCH_COUNT = LINGBOT_GRID_H * LINGBOT_GRID_W
LINGBOT_SPECIAL_TOKEN_COUNT = 6


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_int_list(value: Any) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    out: list[int] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def iter_cache_entries(cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (cache_dir / "cache_index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["chunk_idx"]))


def resize_track_grid(track_map: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(track_map.astype(np.int32), mode="I")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32).reshape(-1)


def seq_track_persistence(seq: str) -> dict[int, float]:
    rows = read_csv(STAGE1 / seq / "stage1_track_summary_rows.csv")
    out: dict[int, float] = {}
    for row in rows:
        try:
            out[int(row["object_or_track_id"])] = fnum(row.get("object_persistence"), 0.0)
        except (KeyError, ValueError):
            continue
    return out


def build_lingbot_patch_arrays() -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    for seq in SEQS:
        cache_dir = KITTI_PREPROCESS / seq / "stage_c_cache_semantic_chunks"
        conversion = read_json(cache_dir / "conversion_summary.json")
        num_frames = int(conversion["num_frames"])
        seq_arr = np.full((num_frames, LINGBOT_PATCH_COUNT), NO_TRACK, dtype=np.int32)
        processed = np.zeros((num_frames,), dtype=bool)
        duplicate_patch_count = 0
        conflict_patch_count = 0

        for entry in iter_cache_entries(cache_dir):
            chunk_dir = cache_dir / str(entry["chunk"])
            payload = torch.load(chunk_dir / "masklet.pt", map_location="cpu")
            masks = payload["M_mask"].numpy().astype(bool)
            visible = payload["V_mask"].numpy().astype(bool)
            seed_ids = [int(v) for v in payload["seed_global_track_idx"]]
            start = int(entry["start_frame"])
            end = int(entry["end_frame"])
            frame_h = int(payload["frame_height"])
            frame_w = int(payload["frame_width"])
            for local_frame, global_frame in enumerate(range(start, end)):
                if global_frame >= num_frames:
                    continue
                track_map = np.full((frame_h, frame_w), NO_TRACK, dtype=np.int32)
                for mask_idx, seed_id in enumerate(seed_ids):
                    if visible[mask_idx, local_frame]:
                        track_map[masks[mask_idx, local_frame]] = seed_id
                patch_tracks = resize_track_grid(track_map, LINGBOT_GRID_W, LINGBOT_GRID_H)
                if processed[global_frame]:
                    old = seq_arr[global_frame]
                    valid = (old >= 0) | (patch_tracks >= 0)
                    duplicate_patch_count += int(valid.sum())
                    conflict_patch_count += int(((old != patch_tracks) & valid).sum())
                    continue
                seq_arr[global_frame] = patch_tracks
                processed[global_frame] = True

        arrays[seq] = seq_arr
        assigned = seq_arr >= 0
        summaries.append(
            {
                "seq": seq,
                "model": "LingBot",
                "surface": "patch_grid_20x36",
                "num_frames": num_frames,
                "processed_frames": int(processed.sum()),
                "frame_coverage": float(processed.sum()) / float(num_frames),
                "patch_count": LINGBOT_PATCH_COUNT,
                "patch_provenance_coverage": float(assigned.sum()) / float(seq_arr.size),
                "duplicate_conflicting_track_assignment_rate": (
                    float(conflict_patch_count) / float(duplicate_patch_count) if duplicate_patch_count else 0.0
                ),
                "provenance_transform": "nearest_resize_to_lingbot_20x36_from_stage_c_native_track_map",
                "track_source": "seed_global_track_idx",
            }
        )
    return arrays, summaries


def save_padded_array(path: Path, seq_arrays: dict[str, np.ndarray], fill: int = NO_TRACK) -> dict[str, Any]:
    max_frames = max(arr.shape[0] for arr in seq_arrays.values())
    patch_count = next(iter(seq_arrays.values())).shape[1]
    out = np.full((len(SEQS), max_frames, patch_count), fill, dtype=np.int32)
    lengths = {}
    for seq_idx, seq in enumerate(SEQS):
        arr = seq_arrays[seq]
        out[seq_idx, : arr.shape[0], :] = arr
        lengths[seq] = int(arr.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return {"shape": list(out.shape), "seq_order": list(SEQS), "seq_lengths": lengths}


def save_lingbot_anchor(path: Path, seq_arrays: dict[str, np.ndarray], anchor_count: int = 8) -> dict[str, Any]:
    out = np.full((len(SEQS), anchor_count, LINGBOT_PATCH_COUNT), NO_TRACK, dtype=np.int32)
    for seq_idx, seq in enumerate(SEQS):
        arr = seq_arrays[seq]
        out[seq_idx, : min(anchor_count, arr.shape[0]), :] = arr[:anchor_count]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return {"shape": list(out.shape), "seq_order": list(SEQS), "anchor_frames_per_seq": anchor_count}


def build_lingbot_trajectory_from_actions(path: Path, seq_arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    action_rows = [
        row for row in read_csv(V110R_STAGE3_ACTION_CONFIG)
        if row.get("candidate_id") == "B1" and row.get("seq") in SEQS
    ]
    provenance_rows: list[dict[str, Any]] = []
    numeric_rows: list[list[int]] = []
    policy_to_idx: dict[str, int] = {}
    mapped_by_seq = {seq: 0 for seq in SEQS}
    total_by_seq = {seq: 0 for seq in SEQS}
    action_file_missing = 0

    for row in action_rows:
        seq = str(row.get("seq", ""))
        if seq not in seq_arrays:
            continue
        policy_id = str(row.get("policy_id", ""))
        policy_idx = policy_to_idx.setdefault(policy_id, len(policy_to_idx))
        action_file = Path(str(row.get("action_file", "")))
        action_file_exists = action_file.exists()
        if not action_file_exists:
            action_file_missing += 1
        arr = seq_arrays[seq]
        seq_idx = SEQS.index(seq)
        for frame_id in parse_int_list(row.get("selected_global_frame_indices")):
            total_by_seq[seq] += 1
            in_range = 0 <= frame_id < arr.shape[0]
            valid_tracks: np.ndarray
            if in_range:
                valid_tracks = arr[frame_id]
                valid_tracks = valid_tracks[valid_tracks >= 0]
            else:
                valid_tracks = np.empty((0,), dtype=np.int32)
            mapped = bool(valid_tracks.size)
            if mapped:
                mapped_by_seq[seq] += 1
                counts = Counter(int(v) for v in valid_tracks.tolist())
                top_tracks = counts.most_common(5)
                dominant_track = int(top_tracks[0][0])
                dominant_track_fraction = float(top_tracks[0][1]) / float(valid_tracks.size)
                top_track_ids = ";".join(str(k) for k, _ in top_tracks)
                top_track_counts = ";".join(str(v) for _, v in top_tracks)
            else:
                dominant_track = NO_TRACK
                dominant_track_fraction = 0.0
                top_track_ids = ""
                top_track_counts = ""
            numeric_rows.append([seq_idx, frame_id, policy_idx, int(mapped), dominant_track, int(valid_tracks.size)])
            provenance_rows.append(
                {
                    "schema": "acl2_v117tf_lingbot_trajectory_action_provenance_row_v1",
                    "seq": seq,
                    "policy_id": policy_id,
                    "policy_family": row.get("policy_family", ""),
                    "unit_id": f"lingbot:{seq}:trajectory_action:{policy_id}:{frame_id}:frame_aggregate",
                    "retained_or_target_frame_id": frame_id,
                    "token_type": "frame_aggregate",
                    "provenance_mode": "source_frame_track_distribution",
                    "mapped_to_stage1_track_grid": mapped,
                    "valid_track_patch_count": int(valid_tracks.size),
                    "dominant_track_id": dominant_track,
                    "dominant_track_fraction": dominant_track_fraction,
                    "top_track_ids": top_track_ids,
                    "top_track_counts": top_track_counts,
                    "action_file": rel(action_file) if action_file_exists else str(row.get("action_file", "")),
                    "action_file_exists": action_file_exists,
                    "source": rel(V110R_STAGE3_ACTION_CONFIG),
                }
            )

    out = np.asarray(numeric_rows, dtype=np.int32) if numeric_rows else np.empty((0, 6), dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    write_csv(OUT / "lingbot_trajectory_provenance_rows.csv", provenance_rows)
    stats_by_seq: dict[str, dict[str, Any]] = {}
    for seq in SEQS:
        total = total_by_seq[seq]
        mapped = mapped_by_seq[seq]
        coverage = float(mapped) / float(total) if total else 0.0
        stats_by_seq[seq] = {
            "action_target_rows": total,
            "mapped_action_target_rows": mapped,
            "trajectory_action_targets_provenance_coverage": coverage,
            "trajectory_blocker": "" if coverage >= 0.95 and action_file_missing == 0 else "trajectory_action_target_frame_aggregate_coverage_below_gate_or_action_file_missing",
            "trajectory_provenance_mode": "source_frame_track_distribution",
            "trajectory_action_config": rel(V110R_STAGE3_ACTION_CONFIG),
            "trajectory_action_file_missing_count": action_file_missing,
        }
    return {
        "shape": list(out.shape),
        "columns": ["seq_idx", "frame_id", "policy_idx", "mapped", "dominant_track_id", "valid_track_patch_count"],
        "provenance_rows": len(provenance_rows),
        "seq_order": list(SEQS),
        "policy_ids": sorted(policy_to_idx, key=policy_to_idx.get),
        "source": rel(V110R_STAGE3_ACTION_CONFIG),
        "row_csv": rel(OUT / "lingbot_trajectory_provenance_rows.csv"),
        "action_file_missing_count": action_file_missing,
    }, stats_by_seq


def summarize_lingbot_units(
    seq_arrays: dict[str, np.ndarray],
    patch_summaries: list[dict[str, Any]],
    trajectory_stats_by_seq: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in patch_summaries:
        seq = str(row["seq"])
        arr = seq_arrays[seq]
        persistence_by_track = seq_track_persistence(seq)
        valid = arr >= 0
        persistence_vals = np.asarray([persistence_by_track.get(int(v), 0.0) for v in arr[valid][:200000]], dtype=np.float32)
        persistence_mean = float(np.mean(persistence_vals)) if persistence_vals.size else 0.0
        anchor = arr[:8]
        anchor_coverage = float(np.count_nonzero(anchor >= 0)) / float(anchor.size) if anchor.size else 0.0
        local_coverage = float(np.count_nonzero(valid)) / float(arr.size)
        traj_stats = trajectory_stats_by_seq.get(seq, {})
        traj_coverage = float(traj_stats.get("trajectory_action_targets_provenance_coverage", 0.0))
        traj_count = int(traj_stats.get("action_target_rows", 0))
        traj_mode = str(traj_stats.get("trajectory_provenance_mode", "blocked_missing_raw_action_targets"))
        manifest.extend(
            [
                {
                    "model": "LingBot",
                    "memory_family": "Anchor Context",
                    "seq": seq,
                    "unit_id_schema": "(seq, anchor_frame_id, patch_id)",
                    "provenance_mode": "patch_track_id",
                    "unit_count": int(anchor.size),
                    "coverage": anchor_coverage,
                    "append_only": True,
                    "reliability_mode": "append_only_source_read",
                    "source": rel(OUT / "lingbot_anchor_provenance.npy"),
                },
                {
                    "model": "LingBot",
                    "memory_family": "Local Window source",
                    "seq": seq,
                    "unit_id_schema": "(seq, source_frame_id, patch_id)",
                    "provenance_mode": "patch_track_id",
                    "unit_count": int(arr.size),
                    "coverage": local_coverage,
                    "append_only": True,
                    "reliability_mode": "short_lived_source_correspondence",
                    "source": rel(OUT / "lingbot_local_provenance.npy"),
                },
                {
                    "model": "LingBot",
                    "memory_family": "Trajectory Memory compact unit",
                    "seq": seq,
                    "unit_id_schema": "(seq, policy_id, action_target_frame_id, token_type=frame_aggregate)",
                    "provenance_mode": traj_mode,
                    "unit_count": traj_count,
                    "coverage": traj_coverage,
                    "append_only": True,
                    "reliability_mode": "source_frame_aggregate_no_token_attention",
                    "source": rel(OUT / "lingbot_trajectory_provenance_rows.csv"),
                },
            ]
        )
        summary_rows.append(
            {
                "model": "LingBot",
                "seq": seq,
                "anchor_patch_provenance_coverage": anchor_coverage,
                "local_patch_provenance_coverage": local_coverage,
                "trajectory_action_targets_provenance_coverage": traj_coverage,
                "trajectory_blocker": traj_stats.get("trajectory_blocker", "trajectory_action_target_provenance_not_built"),
                "trajectory_action_target_rows": traj_count,
                "trajectory_mapped_action_target_rows": int(traj_stats.get("mapped_action_target_rows", 0)),
                "trajectory_provenance_mode": traj_mode,
                "sampled_object_persistence_mean": persistence_mean,
                **row,
            }
        )
    return manifest, summary_rows


def build_hs_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    seq_arrays: dict[str, np.ndarray] = {}
    hs_summary_rows: list[dict[str, Any]] = []
    for seq in SEQS:
        arr = np.load(STAGE1 / seq / "semantic_track_id.npy", mmap_mode="r")
        arr_np = np.asarray(arr, dtype=np.int32)
        seq_arrays[seq] = arr_np
        hs_summary_rows.append(
            {
                "model": "HorizonStream",
                "seq": seq,
                "surface": "Local KV",
                "patch_count": int(arr_np.shape[1]),
                "num_frames": int(arr_np.shape[0]),
                "local_kv_provenance_coverage": float(np.count_nonzero(arr_np >= 0)) / float(arr_np.size),
                "provenance_mode": "patch_track_id_from_v117_stage1_hs_grid",
                "source": rel(STAGE1 / seq / "semantic_track_id.npy"),
            }
        )

    hs_local_info = save_padded_array(OUT / "hs_local_provenance.npy", seq_arrays)

    gla_rows = read_csv(V115_STAGE2 / "hs_gla_state_quality_rows.csv")
    gla_numeric: list[list[float]] = []
    gla_manifest_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(gla_rows):
        case = row.get("case", "")
        seq_match = re.search(r"kitti_(\d\d)", case)
        seq = seq_match.group(1) if seq_match else row.get("seq", "")
        vals = [
            fnum(row.get("chunk_idx"), -1.0),
            fnum(row.get("chunk_start"), -1.0),
            fnum(row.get("chunk_end"), -1.0),
            fnum(row.get("global_layer_idx"), -1.0),
            fnum(row.get("state_new_norm"), 0.0),
            fnum(row.get("state_delta_norm"), 0.0),
            fnum(row.get("conv_state_norm"), 0.0),
            fnum(row.get("chunk_dynamic_mass_mean"), 0.0),
            fnum(row.get("chunk_boundary_mass_mean"), 0.0),
            fnum(row.get("chunk_stable_mass_mean"), 0.0),
            fnum(row.get("chunk_weak_context_mass_mean"), 0.0),
        ]
        gla_numeric.append(vals)
        gla_manifest_rows.append(
            {
                "model": "HorizonStream",
                "memory_family": "GLA state unit",
                "seq": seq,
                "unit_id_schema": "(seq, layer_id, chunk, channel_band)",
                "provenance_mode": "layer_chunk_band",
                "provenance_resolution": "layer_chunk_band",
                "unit_row_index": row_idx,
                "coverage": 1.0 if seq in SEQS or seq == "00/02" else 0.0,
                "append_only": False,
                "reliability_mode": "state_probe_chunk_summary_not_token_channel",
                "source": row.get("source_path", rel(V115_STAGE2 / "hs_gla_state_quality_rows.csv")),
            }
        )
    gla_array = np.asarray(gla_numeric, dtype=np.float32)
    np.save(OUT / "hs_gla_provenance.npy", gla_array)
    hs_gla_info = {
        "shape": list(gla_array.shape),
        "columns": [
            "chunk_idx",
            "chunk_start",
            "chunk_end",
            "global_layer_idx",
            "state_new_norm",
            "state_delta_norm",
            "conv_state_norm",
            "chunk_dynamic_mass_mean",
            "chunk_boundary_mass_mean",
            "chunk_stable_mass_mean",
            "chunk_weak_context_mass_mean",
        ],
        "provenance_resolution": "layer_chunk_band",
        "source_rows": len(gla_rows),
    }

    manifest: list[dict[str, Any]] = []
    for row in hs_summary_rows:
        manifest.append(
            {
                "model": "HorizonStream",
                "memory_family": "Local KV",
                "seq": row["seq"],
                "unit_id_schema": "(seq, frame_id, patch_id, layer_id)",
                "provenance_mode": row["provenance_mode"],
                "unit_count": int(row["num_frames"]) * int(row["patch_count"]),
                "coverage": row["local_kv_provenance_coverage"],
                "append_only": True,
                "reliability_mode": "source_patch_track_and_v115_probe_rows",
                "source": rel(OUT / "hs_local_provenance.npy"),
            }
        )
    manifest.extend(gla_manifest_rows)
    return manifest, hs_summary_rows, hs_local_info, hs_gla_info


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    stage0_boundary = (
        "Stage0 is passed in the current evidence freeze."
        if summary["stage0_runtime_gate_status"] == "passed"
        else "Runtime action is blocked until the Stage0 raw-reference gate is repaired."
    )
    lines = [
        "# v117 Stage2 Memory Provenance Readiness",
        "",
        f"- stage2_ready: `{summary['stage2_ready']}`",
        f"- stage0_runtime_gate_status: `{summary['stage0_runtime_gate_status']}`",
        f"- lingbot_anchor_patch_gate: `{summary['lingbot_anchor_patch_gate']}`",
        f"- lingbot_trajectory_action_target_gate: `{summary['lingbot_trajectory_action_target_gate']}`",
        f"- hs_local_kv_gate: `{summary['hs_local_kv_gate']}`",
        f"- gla_resolution_reported_gate: `{summary['gla_resolution_reported_gate']}`",
        "",
        "| model | seq | anchor/local/hs coverage | trajectory coverage | note |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        if row["model"] == "LingBot":
            trajectory_note = (
                f"trajectory {row.get('trajectory_provenance_mode', 'provenance')} coverage"
                if not str(row.get("trajectory_blocker", "")).strip()
                else str(row.get("trajectory_blocker", "trajectory blocked"))
            )
            lines.append(
                f"| LingBot | {row['seq']} | {row['local_patch_provenance_coverage']} | "
                f"{row['trajectory_action_targets_provenance_coverage']} | {trajectory_note} |"
            )
        elif row["model"] == "HorizonStream":
            lines.append(
                f"| HorizonStream | {row['seq']} | {row['local_kv_provenance_coverage']} |  | local KV ready; GLA chunk-band only |"
            )
    lines += [
        "",
        "## Boundary",
        "",
        "LingBot Anchor/Local patch provenance is derived from native Stage-C `seed_global_track_idx` projected to the 20x36 LingBot patch grid.",
        "LingBot Trajectory Memory action-target provenance uses frame-aggregate track distributions from v110R B1 selected action frames when available; it is not token-type attention-weighted provenance.",
        "HorizonStream GLA provenance is explicitly downgraded to `layer_chunk_band`; this is not token/channel-level same-space memory reliability.",
        stage0_boundary,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="reuse existing Stage2 arrays/CSVs and rewrite only readiness JSON/report",
    )
    parser.add_argument(
        "--repair-trajectory-provenance",
        action="store_true",
        help="reuse existing LingBot local provenance and rebuild only B1 trajectory action-target provenance",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stage0_summary = read_json(STAGE0 / "stage0_frozen_facts.json")
    stage0_runtime_status = (
        "passed" if bool(stage0_summary.get("stage0_complete")) else "blocked_raw_reference_artifacts_missing"
    )

    if args.repair_trajectory_provenance:
        summary = read_json(OUT / "stage2_memory_reliability_readiness.json")
        if not summary:
            raise RuntimeError("missing existing Stage2 readiness JSON for summary-only mode")
        local_meta = summary.get("array_metadata", {}).get("lingbot_local_provenance", {})
        local_arr = np.load(OUT / "lingbot_local_provenance.npy", mmap_mode="r")
        seq_order = [str(v) for v in local_meta.get("seq_order", SEQS)]
        seq_lengths = {str(k): int(v) for k, v in local_meta.get("seq_lengths", {}).items()}
        seq_arrays = {}
        for seq in SEQS:
            if seq not in seq_order:
                raise RuntimeError(f"seq {seq} missing from existing lingbot local provenance")
            seq_idx = seq_order.index(seq)
            length = seq_lengths.get(seq, int(local_arr.shape[1]))
            seq_arrays[seq] = np.asarray(local_arr[seq_idx, :length, :], dtype=np.int32)
        traj_info, trajectory_stats_by_seq = build_lingbot_trajectory_from_actions(
            OUT / "lingbot_trajectory_provenance.npy", seq_arrays
        )

        manifest_rows = read_csv(OUT / "stage2_memory_unit_manifest.csv")
        for row in manifest_rows:
            if row.get("model") == "LingBot" and row.get("memory_family") == "Trajectory Memory compact unit":
                seq = str(row.get("seq", ""))
                stats = trajectory_stats_by_seq.get(seq, {})
                row["unit_id_schema"] = "(seq, policy_id, action_target_frame_id, token_type=frame_aggregate)"
                row["provenance_mode"] = str(stats.get("trajectory_provenance_mode", "source_frame_track_distribution"))
                row["unit_count"] = str(int(stats.get("action_target_rows", 0)))
                row["coverage"] = str(float(stats.get("trajectory_action_targets_provenance_coverage", 0.0)))
                row["append_only"] = "True"
                row["reliability_mode"] = "source_frame_aggregate_no_token_attention"
                row["source"] = rel(OUT / "lingbot_trajectory_provenance_rows.csv")

        summary_rows = read_csv(OUT / "stage2_memory_provenance_summary.csv")
        for row in summary_rows:
            if row.get("model") == "LingBot":
                seq = str(row.get("seq", ""))
                stats = trajectory_stats_by_seq.get(seq, {})
                row["trajectory_action_targets_provenance_coverage"] = str(
                    float(stats.get("trajectory_action_targets_provenance_coverage", 0.0))
                )
                row["trajectory_blocker"] = str(stats.get("trajectory_blocker", ""))
                row["trajectory_action_target_rows"] = str(int(stats.get("action_target_rows", 0)))
                row["trajectory_mapped_action_target_rows"] = str(int(stats.get("mapped_action_target_rows", 0)))
                row["trajectory_provenance_mode"] = str(stats.get("trajectory_provenance_mode", ""))

        lingbot_anchor_gate = all(
            fnum(row.get("anchor_patch_provenance_coverage"), 0.0) >= 0.95
            for row in summary_rows if row.get("model") == "LingBot"
        )
        lingbot_trajectory_gate = all(
            fnum(row.get("trajectory_action_targets_provenance_coverage"), 0.0) >= 0.95
            for row in summary_rows if row.get("model") == "LingBot"
        )
        hs_local_gate = bool(summary.get("hs_local_kv_gate"))
        gla_resolution_gate = bool(summary.get("gla_resolution_reported_gate"))
        no_mismatch_gate = bool(summary.get("no_source_frame_token_index_mismatch_gate"))
        summary["stage0_runtime_gate_status"] = stage0_runtime_status
        summary["lingbot_anchor_patch_gate"] = lingbot_anchor_gate
        summary["lingbot_trajectory_action_target_gate"] = lingbot_trajectory_gate
        summary["stage2_ready"] = (
            lingbot_anchor_gate
            and lingbot_trajectory_gate
            and hs_local_gate
            and gla_resolution_gate
            and no_mismatch_gate
            and stage0_runtime_status == "passed"
        )
        blockers: list[str] = []
        if stage0_runtime_status != "passed":
            blockers.append("Stage0 raw-reference gate is still blocked")
        if not lingbot_trajectory_gate:
            blockers.append("LingBot trajectory/B1 action-target frame-aggregate provenance coverage is below gate")
        summary["blockers"] = blockers
        summary["limitations"] = [
            "LingBot trajectory provenance is frame-aggregate, not token-type attention weighted",
            "HorizonStream GLA provenance is layer_chunk_band, not token/channel-level",
        ]
        summary.setdefault("array_metadata", {})["lingbot_trajectory_provenance"] = traj_info
        summary.setdefault("outputs", {})["lingbot_trajectory_provenance_rows"] = rel(
            OUT / "lingbot_trajectory_provenance_rows.csv"
        )
        write_csv(OUT / "stage2_memory_unit_manifest.csv", manifest_rows)
        write_csv(OUT / "stage2_memory_provenance_summary.csv", summary_rows)
        write_json(OUT / "stage2_memory_reliability_readiness.json", summary)
        write_text(OUT / "MEMORY_PROVENANCE_READINESS_REPORT.md", report_text(summary, summary_rows))
        print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
        return

    if args.summary_only:
        summary = read_json(OUT / "stage2_memory_reliability_readiness.json")
        if not summary:
            raise RuntimeError("missing existing Stage2 readiness JSON for summary-only mode")
        lingbot_anchor_gate = bool(summary.get("lingbot_anchor_patch_gate"))
        lingbot_trajectory_gate = bool(summary.get("lingbot_trajectory_action_target_gate"))
        hs_local_gate = bool(summary.get("hs_local_kv_gate"))
        gla_resolution_gate = bool(summary.get("gla_resolution_reported_gate"))
        no_mismatch_gate = bool(summary.get("no_source_frame_token_index_mismatch_gate"))
        summary["stage0_runtime_gate_status"] = stage0_runtime_status
        summary["stage2_ready"] = (
            lingbot_anchor_gate
            and lingbot_trajectory_gate
            and hs_local_gate
            and gla_resolution_gate
            and no_mismatch_gate
            and stage0_runtime_status == "passed"
        )
        blockers: list[str] = []
        if stage0_runtime_status != "passed":
            blockers.append("Stage0 raw-reference gate is still blocked")
        if not lingbot_trajectory_gate:
            blockers.append("LingBot trajectory/B1 raw action target artifacts are missing")
        summary["blockers"] = blockers
        summary["limitations"] = [
            "HorizonStream GLA provenance is layer_chunk_band, not token/channel-level",
        ]
        summary_rows = read_csv(OUT / "stage2_memory_provenance_summary.csv")
        write_json(OUT / "stage2_memory_reliability_readiness.json", summary)
        write_text(OUT / "MEMORY_PROVENANCE_READINESS_REPORT.md", report_text(summary, summary_rows))
        print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
        return

    lingbot_arrays, lingbot_patch_summaries = build_lingbot_patch_arrays()
    anchor_info = save_lingbot_anchor(OUT / "lingbot_anchor_provenance.npy", lingbot_arrays)
    local_info = save_padded_array(OUT / "lingbot_local_provenance.npy", lingbot_arrays)
    traj_info, trajectory_stats_by_seq = build_lingbot_trajectory_from_actions(
        OUT / "lingbot_trajectory_provenance.npy", lingbot_arrays
    )
    lingbot_manifest, lingbot_summary_rows = summarize_lingbot_units(
        lingbot_arrays, lingbot_patch_summaries, trajectory_stats_by_seq
    )

    hs_manifest, hs_summary_rows, hs_local_info, hs_gla_info = build_hs_outputs()
    manifest_rows = lingbot_manifest + hs_manifest
    summary_rows = lingbot_summary_rows + hs_summary_rows

    missing_token_mismatch = any(row["frame_coverage"] < 0.99 for row in lingbot_patch_summaries)
    lingbot_anchor_gate = all(row["anchor_patch_provenance_coverage"] >= 0.95 for row in lingbot_summary_rows)
    lingbot_trajectory_gate = all(row["trajectory_action_targets_provenance_coverage"] >= 0.95 for row in lingbot_summary_rows)
    hs_local_gate = all(row["local_kv_provenance_coverage"] >= 0.95 for row in hs_summary_rows)
    gla_resolution_gate = hs_gla_info["provenance_resolution"] in {"layer_chunk_band", "token_channel", "token"}
    no_mismatch_gate = not missing_token_mismatch
    stage2_ready = (
        lingbot_anchor_gate
        and lingbot_trajectory_gate
        and hs_local_gate
        and gla_resolution_gate
        and no_mismatch_gate
        and stage0_runtime_status == "passed"
    )
    blockers: list[str] = []
    if stage0_runtime_status != "passed":
        blockers.append("Stage0 raw-reference gate is still blocked")
    if not lingbot_trajectory_gate:
        blockers.append("LingBot trajectory/B1 raw action target artifacts are missing")
    limitations = [
        "LingBot trajectory provenance is frame-aggregate, not token-type attention weighted",
        "HorizonStream GLA provenance is layer_chunk_band, not token/channel-level",
    ]

    summary = {
        "schema": "acl2_v117tf_stage2_memory_provenance_readiness_v1",
        "stage2_ready": stage2_ready,
        "stage0_runtime_gate_status": stage0_runtime_status,
        "lingbot_anchor_patch_gate": lingbot_anchor_gate,
        "lingbot_trajectory_action_target_gate": lingbot_trajectory_gate,
        "hs_local_kv_gate": hs_local_gate,
        "gla_resolution_reported_gate": gla_resolution_gate,
        "no_source_frame_token_index_mismatch_gate": no_mismatch_gate,
        "gla_provenance_resolution": hs_gla_info["provenance_resolution"],
        "blockers": blockers,
        "limitations": limitations,
        "array_metadata": {
            "lingbot_anchor_provenance": anchor_info,
            "lingbot_local_provenance": local_info,
            "lingbot_trajectory_provenance": traj_info,
            "hs_local_provenance": hs_local_info,
            "hs_gla_provenance": hs_gla_info,
        },
        "outputs": {
            "stage2_memory_unit_manifest": rel(OUT / "stage2_memory_unit_manifest.csv"),
            "stage2_memory_provenance_summary": rel(OUT / "stage2_memory_provenance_summary.csv"),
            "stage2_memory_reliability_readiness": rel(OUT / "stage2_memory_reliability_readiness.json"),
            "lingbot_anchor_provenance": rel(OUT / "lingbot_anchor_provenance.npy"),
            "lingbot_local_provenance": rel(OUT / "lingbot_local_provenance.npy"),
            "lingbot_trajectory_provenance": rel(OUT / "lingbot_trajectory_provenance.npy"),
            "lingbot_trajectory_provenance_rows": rel(OUT / "lingbot_trajectory_provenance_rows.csv"),
            "hs_local_provenance": rel(OUT / "hs_local_provenance.npy"),
            "hs_gla_provenance": rel(OUT / "hs_gla_provenance.npy"),
            "report": rel(OUT / "MEMORY_PROVENANCE_READINESS_REPORT.md"),
        },
    }
    write_csv(OUT / "stage2_memory_unit_manifest.csv", manifest_rows)
    write_csv(OUT / "stage2_memory_provenance_summary.csv", summary_rows)
    write_json(OUT / "stage2_memory_reliability_readiness.json", summary)
    write_text(OUT / "MEMORY_PROVENANCE_READINESS_REPORT.md", report_text(summary, summary_rows))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
