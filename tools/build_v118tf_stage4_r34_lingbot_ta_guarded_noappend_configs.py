#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R34 LingBot TA guarded no-append configs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r34_lingbot_ta_guarded_noappend"
SUPPORT = RESULT_ROOT / "stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_frame_semantic_support_rows.csv"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
ACTION_BUDGET_FRACTION = 0.10
TEMPORAL_BUCKETS = 4
RANDOM_SEED = 11834

METHODS = {
    "lingbot_map_stream_flashinfer_v118_r34_ta5_guarded_lowdrop": {
        "policy": "TA5_GUARDED_SEMANTIC_SUPPORT_LOW_DROP",
        "role": "candidate",
        "description": "drop the highest guarded semantic-support risk default base keyframes",
    },
    "lingbot_map_stream_flashinfer_v118_r34_ta5_guarded_reverse_keepdrop": {
        "policy": "TA5_REVERSE_HIGH_SUPPORT_DROP",
        "role": "reverse_control",
        "description": "drop the lowest guarded semantic-support risk default base keyframes",
    },
    "lingbot_map_stream_flashinfer_v118_r34_ta5_guarded_temporal_random": {
        "policy": "TA5_MATCHED_TEMPORAL_RANDOM_DROP",
        "role": "matched_temporal_random_control",
        "description": "drop a deterministic random same-count same-temporal-bucket subset",
    },
}

SUPPORT_FIELDS = [
    "max_semantic_persistence_prefix",
    "mean_semantic_persistence_prefix",
    "mean_current_mask_quality",
    "sum_current_area_ratio",
    "visible_track_rows",
    "unique_track_count",
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_support() -> dict[str, dict[int, dict[str, Any]]]:
    if not SUPPORT.exists():
        raise FileNotFoundError(SUPPORT)
    out: dict[str, dict[int, dict[str, Any]]] = {}
    with SUPPORT.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = str(row["seq"])
            frame = int(row["frame_id"])
            out.setdefault(seq, {})[frame] = row
    return out


def auto_keyframe_interval(num_frames: int, threshold: int = 320) -> int:
    return 1 if num_frames <= threshold else (num_frames + threshold - 1) // threshold


def default_base_keyframes(num_frames: int, scale_frames: int = 8) -> list[int]:
    interval = auto_keyframe_interval(num_frames)
    return [
        frame
        for stream_pos, frame in enumerate(range(scale_frames, num_frames))
        if interval <= 1 or stream_pos % interval == 0
    ]


def normalized_values(rows_by_frame: dict[int, dict[str, Any]], eligible: list[int]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for field in SUPPORT_FIELDS:
        values = [fnum(rows_by_frame[frame].get(field)) for frame in eligible]
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            out[field] = {frame: 0.5 for frame in eligible}
        else:
            out[field] = {
                frame: (fnum(rows_by_frame[frame].get(field)) - lo) / (hi - lo)
                for frame in eligible
            }
    return out


def guarded_risk_scores(rows_by_frame: dict[int, dict[str, Any]], eligible: list[int]) -> dict[int, float]:
    norm = normalized_values(rows_by_frame, eligible)
    scores: dict[int, float] = {}
    for frame in eligible:
        support = (
            0.30 * norm["max_semantic_persistence_prefix"][frame]
            + 0.25 * norm["mean_semantic_persistence_prefix"][frame]
            + 0.20 * norm["mean_current_mask_quality"][frame]
            + 0.10 * norm["sum_current_area_ratio"][frame]
            + 0.10 * norm["visible_track_rows"][frame]
            + 0.05 * norm["unique_track_count"][frame]
        )
        scores[frame] = 1.0 - support
    return scores


def temporal_bucket_counts(eligible: list[int], selected: list[int], bucket_count: int) -> dict[int, int]:
    position = {frame: idx for idx, frame in enumerate(eligible)}
    counts = {idx: 0 for idx in range(bucket_count)}
    denom = max(1, len(eligible))
    for frame in selected:
        bucket = min(bucket_count - 1, int(position[frame] * bucket_count / denom))
        counts[bucket] += 1
    return counts


def temporal_matched_random(eligible: list[int], selected: list[int], seq: str) -> list[int]:
    selected_set = set(selected)
    counts = temporal_bucket_counts(eligible, selected, TEMPORAL_BUCKETS)
    rng = random.Random(RANDOM_SEED + int(seq))
    out: list[int] = []
    denom = max(1, len(eligible))
    for bucket, count in counts.items():
        bucket_frames = [
            frame
            for idx, frame in enumerate(eligible)
            if min(TEMPORAL_BUCKETS - 1, int(idx * TEMPORAL_BUCKETS / denom)) == bucket
        ]
        pool = [frame for frame in bucket_frames if frame not in selected_set]
        if len(pool) < count:
            pool = list(bucket_frames)
        rng.shuffle(pool)
        out.extend(pool[:count])
    if len(out) < len(selected):
        remaining = [frame for frame in eligible if frame not in set(out) and frame not in selected_set]
        if len(remaining) < len(selected) - len(out):
            remaining = [frame for frame in eligible if frame not in set(out)]
        rng.shuffle(remaining)
        out.extend(remaining[: len(selected) - len(out)])
    return sorted(out[: len(selected)])


def selected_indices(rows_by_frame: dict[int, dict[str, Any]], seq: str) -> dict[str, list[int]]:
    num_frames = max(rows_by_frame) + 1
    eligible = [frame for frame in default_base_keyframes(num_frames) if frame in rows_by_frame]
    if not eligible:
        raise RuntimeError(f"no eligible default base keyframes for seq {seq}")
    budget = max(1, int(len(eligible) * ACTION_BUDGET_FRACTION))
    risk = guarded_risk_scores(rows_by_frame, eligible)
    ranked_high_risk = sorted(eligible, key=lambda frame: (-risk[frame], frame))
    ranked_low_risk = sorted(eligible, key=lambda frame: (risk[frame], frame))
    candidate = sorted(ranked_high_risk[:budget])
    return {
        "lingbot_map_stream_flashinfer_v118_r34_ta5_guarded_lowdrop": candidate,
        "lingbot_map_stream_flashinfer_v118_r34_ta5_guarded_reverse_keepdrop": sorted(ranked_low_risk[:budget]),
        "lingbot_map_stream_flashinfer_v118_r34_ta5_guarded_temporal_random": temporal_matched_random(
            eligible, candidate, seq
        ),
    }


def score_stats(rows_by_frame: dict[int, dict[str, Any]], frames: list[int], eligible: list[int]) -> dict[str, Any]:
    risks = guarded_risk_scores(rows_by_frame, eligible)
    values = [risks[frame] for frame in frames]
    stats = {
        "guarded_risk_min": min(values) if values else "",
        "guarded_risk_max": max(values) if values else "",
        "guarded_risk_median": median(values) if values else "",
    }
    for field in SUPPORT_FIELDS:
        raw = [fnum(rows_by_frame[frame].get(field)) for frame in frames]
        stats[f"{field}_median"] = median(raw) if raw else ""
    return stats


def write_main_config(config_dir: Path) -> None:
    (config_dir / "kitti_lingbot_flashinfer_r34_ta_guarded_noappend_full_reuse_v105gt.yaml").write_text(
        "\n".join(
            [
                f"workspace: {WORKSPACE}",
                "",
                "evaluation:",
                "  traj:",
                "    enable: true",
                "    vis: true",
                "  auc:",
                "    enable: false",
                "  depth:",
                "    enable: false",
                "  points:",
                "    enable: false",
                "",
                "datasets:",
                f"  - {DATASET}",
                "",
                "methods:",
                *[f"  - {method}" for method in METHODS],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_dataset_config(config_dir: Path) -> None:
    ds_dir = config_dir / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / f"{DATASET}.yaml").write_text(
        "\n".join(
            [
                "dataset: kitti",
                f"raw_data_root: {ROOT / 'data/kitti/dataset'}",
                "_target_size: [504, 280]",
                '_sequences: ["00", "01", "02", "05"]',
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_method_config(config_dir: Path, method: str, policy: str, seq_maps: dict[str, list[int]]) -> None:
    method_dir = config_dir / "methods"
    method_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "model: lingbot_map",
        "env: loger",
        f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
        "_device: cuda",
        "_use_amp: true",
        "_use_sdpa: false",
        "_image_size: 518",
        "_patch_size: 14",
        "_enable_3d_rope: true",
        "_num_scale_frames: 8",
        "_max_frame_num: 1024",
        "_kv_cache_sliding_window: 64",
        "_kv_cache_scale_frames: 8",
        "_auto_keyframe_threshold: 320",
        "_area_budget: 255000",
        "_align: 14",
        "_mode: streaming",
        "_keyframe_interval: auto",
        "_stage4_action_mode: force_non_keyframe",
        f"_stage4_action_label: {policy}",
        "_stage4_force_non_keyframe_indices_by_seq:",
    ]
    for seq in sorted(seq_maps):
        lines.append(f'  "{seq}": {json.dumps(seq_maps[seq], separators=(",", ":"))}')
    lines.append("")
    (method_dir / f"{method}.yaml").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(rows: list[dict[str, Any]], seq_maps_by_method: dict[str, dict[str, list[int]]]) -> None:
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "stage4_r34_lingbot_ta_guarded_noappend_manifest.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "acl2_v118tf_stage4_r34_lingbot_ta_guarded_noappend_manifest_v1",
        "stage": rel(STAGE),
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": rel(SUPPORT),
        "action_budget_fraction": ACTION_BUDGET_FRACTION,
        "temporal_buckets": TEMPORAL_BUCKETS,
        "random_seed": RANDOM_SEED,
        "action_mode": "force_non_keyframe",
        "risk_score": "1 - weighted normalized semantic-support fields",
        "support_fields": SUPPORT_FIELDS,
        "methods": METHODS,
        "force_non_keyframe_indices_by_method": seq_maps_by_method,
        "manifest_csv": rel(csv_path),
    }
    (out / "stage4_r34_lingbot_ta_guarded_noappend_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    support = read_support()
    config_dir = STAGE / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_main_config(config_dir)
    write_dataset_config(config_dir)

    seq_maps_by_method = {method: {} for method in METHODS}
    manifest_rows: list[dict[str, Any]] = []
    for seq, rows_by_frame in sorted(support.items()):
        if seq not in {"00", "02"}:
            continue
        num_frames = max(rows_by_frame) + 1
        interval = auto_keyframe_interval(num_frames)
        eligible = [frame for frame in default_base_keyframes(num_frames) if frame in rows_by_frame]
        selections = selected_indices(rows_by_frame, seq)
        candidate_frames = selections["lingbot_map_stream_flashinfer_v118_r34_ta5_guarded_lowdrop"]
        candidate_bucket_counts = temporal_bucket_counts(eligible, candidate_frames, TEMPORAL_BUCKETS)
        for method, frames in selections.items():
            seq_maps_by_method[method][seq] = frames
            manifest_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r34_lingbot_ta_guarded_noappend_manifest_row_v1",
                    "seq": seq,
                    "method": method,
                    "role": METHODS[method]["role"],
                    "policy": METHODS[method]["policy"],
                    "num_frames": num_frames,
                    "scale_frames": 8,
                    "auto_keyframe_interval": interval,
                    "eligible_default_base_keyframes": len(eligible),
                    "action_budget_fraction": ACTION_BUDGET_FRACTION,
                    "forced_non_keyframe_count": len(frames),
                    "candidate_temporal_bucket_counts": json.dumps(candidate_bucket_counts, sort_keys=True),
                    "selected_frame_ids": ";".join(str(frame) for frame in frames),
                    **score_stats(rows_by_frame, frames, eligible),
                }
            )

    for method, meta in METHODS.items():
        write_method_config(config_dir, method, meta["policy"], seq_maps_by_method[method])
    write_manifest(manifest_rows, seq_maps_by_method)
    print(
        json.dumps(
            {
                "stage": rel(STAGE),
                "config": rel(
                    config_dir / "kitti_lingbot_flashinfer_r34_ta_guarded_noappend_full_reuse_v105gt.yaml"
                ),
                "manifest_rows": len(manifest_rows),
                "methods": list(METHODS),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
