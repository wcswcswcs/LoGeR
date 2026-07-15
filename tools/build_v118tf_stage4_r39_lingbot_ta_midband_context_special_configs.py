#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R39 LingBot TA midband context-special configs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any

import build_v118tf_stage4_r34_lingbot_ta_guarded_noappend_configs as r34
import build_v118tf_stage4_r35_lingbot_ta_declustered_noappend_configs as r35
import build_v118tf_stage4_r36_lingbot_ta_tailcalibrated_noappend_configs as r36


ROOT = r34.ROOT
RESULT_ROOT = r34.RESULT_ROOT
STAGE = RESULT_ROOT / "stage4_r39_lingbot_ta_midband_context_special"
WORKSPACE = r34.WORKSPACE
DATASET = r34.DATASET
BASELINE_METHOD = r34.BASELINE_METHOD
ACTION_BUDGET_FRACTION = 0.10
TEMPORAL_BUCKETS = 4
MIN_FRAME_GAP = 60
MIDBAND_QUANTILE = 0.50
RANDOM_SEED = 11839

METHODS = {
    "lingbot_map_stream_flashinfer_v118_r39_ta5_midband_context_special": {
        "policy": "TA5_MIDBAND_CONTEXT_SPECIAL",
        "role": "candidate",
        "description": "convert de-clustered midband semantic-support base keyframes to context-only special-token carriers",
    },
    "lingbot_map_stream_flashinfer_v118_r39_ta5_highrisk_context_reverse": {
        "policy": "TA5_MIDBAND_CONTEXT_HIGH_RISK_REVERSE",
        "role": "reverse_control",
        "description": "context-only special-token reverse control on high guarded-risk frames with the same de-cluster gap",
    },
    "lingbot_map_stream_flashinfer_v118_r39_ta5_midband_context_temporal_random": {
        "policy": "TA5_MIDBAND_CONTEXT_MATCHED_TEMPORAL_RANDOM",
        "role": "matched_temporal_random_control",
        "description": "context-only special-token same-count temporal-bucket random control with the same de-cluster gap",
    },
}


def temporal_matched_random_gap(eligible: list[int], selected: list[int], seq: str) -> list[int]:
    selected_set = set(selected)
    counts = r34.temporal_bucket_counts(eligible, selected, TEMPORAL_BUCKETS)
    denom = max(1, len(eligible))
    bucket_pools = {
        bucket: [
            frame
            for idx, frame in enumerate(eligible)
            if min(TEMPORAL_BUCKETS - 1, int(idx * TEMPORAL_BUCKETS / denom)) == bucket
            and frame not in selected_set
        ]
        for bucket in range(TEMPORAL_BUCKETS)
    }

    def gap_ok(frame: int, chosen: list[int]) -> bool:
        return all(abs(frame - old) >= MIN_FRAME_GAP for old in chosen)

    best: list[int] = []
    for attempt in range(256):
        rng = random.Random(RANDOM_SEED + int(seq) * 997 + attempt)
        out: list[int] = []
        feasible = True
        for bucket in range(TEMPORAL_BUCKETS):
            quota = counts.get(bucket, 0)
            pool = list(bucket_pools[bucket])
            rng.shuffle(pool)
            picked = 0
            for frame in pool:
                if gap_ok(frame, out):
                    out.append(frame)
                    picked += 1
                    if picked == quota:
                        break
            if picked != quota:
                feasible = False
                break
        if len(out) > len(best):
            best = list(out)
        if feasible and len(out) == len(selected):
            return sorted(out)
    raise RuntimeError(
        f"unable to build same-count temporal-random gap control for seq {seq}: "
        f"target={len(selected)} best={len(best)} min_gap={MIN_FRAME_GAP} counts={dict(counts)}"
    )


def min_gap(frames: list[int]) -> int | str:
    if len(frames) < 2:
        return ""
    return min(b - a for a, b in zip(sorted(frames), sorted(frames)[1:]))


def selected_indices(rows_by_frame: dict[int, dict[str, Any]], seq: str) -> tuple[dict[str, list[int]], dict[str, Any]]:
    num_frames = max(rows_by_frame) + 1
    eligible = [frame for frame in r34.default_base_keyframes(num_frames) if frame in rows_by_frame]
    if not eligible:
        raise RuntimeError(f"no eligible default base keyframes for seq {seq}")
    budget = max(1, int(len(eligible) * ACTION_BUDGET_FRACTION))
    risk = r34.guarded_risk_scores(rows_by_frame, eligible)
    values = list(risk.values())
    risk_target = r36.quantile(values, MIDBAND_QUANTILE)
    ranked_mid = sorted(eligible, key=lambda frame: (abs(risk[frame] - risk_target), frame))
    ranked_high = sorted(eligible, key=lambda frame: (-risk[frame], frame))
    candidate = r35.greedy_gap_select(ranked_mid, budget, MIN_FRAME_GAP)
    reverse = r35.greedy_gap_select(ranked_high, budget, MIN_FRAME_GAP)
    return (
        {
            "lingbot_map_stream_flashinfer_v118_r39_ta5_midband_context_special": candidate,
            "lingbot_map_stream_flashinfer_v118_r39_ta5_highrisk_context_reverse": reverse,
            "lingbot_map_stream_flashinfer_v118_r39_ta5_midband_context_temporal_random": temporal_matched_random_gap(
                eligible, candidate, seq
            ),
        },
        {"risk_target_quantile": MIDBAND_QUANTILE, "risk_target": risk_target, "selection_mode": "midband_context_special_q50_all_sequences"},
    )


def score_stats(rows_by_frame: dict[int, dict[str, Any]], frames: list[int], eligible: list[int]) -> dict[str, Any]:
    risk = r34.guarded_risk_scores(rows_by_frame, eligible)
    values = [risk[frame] for frame in frames]
    out = {
        "guarded_risk_min": min(values) if values else "",
        "guarded_risk_max": max(values) if values else "",
        "guarded_risk_median": median(values) if values else "",
    }
    for field in r34.SUPPORT_FIELDS:
        raw = [r34.fnum(rows_by_frame[frame].get(field)) for frame in frames]
        out[f"{field}_median"] = median(raw) if raw else ""
    return out


def write_main_config(config_dir: Path) -> None:
    (config_dir / "kitti_lingbot_flashinfer_r39_ta_midband_context_special_full_reuse_v105gt.yaml").write_text(
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
        "_stage4_action_mode: context_only_special",
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
    csv_path = out / "stage4_r39_lingbot_ta_midband_context_special_manifest.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "acl2_v118tf_stage4_r39_lingbot_ta_midband_context_special_manifest_v1",
        "stage": r34.rel(STAGE),
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": r34.rel(r34.SUPPORT),
        "action_budget_fraction": ACTION_BUDGET_FRACTION,
        "temporal_buckets": TEMPORAL_BUCKETS,
        "min_frame_gap": MIN_FRAME_GAP,
        "midband_quantile": MIDBAND_QUANTILE,
        "random_seed": RANDOM_SEED,
        "action_mode": "context_only_special",
        "methods": METHODS,
        "force_context_indices_by_method": seq_maps_by_method,
        "manifest_csv": r34.rel(csv_path),
    }
    (out / "stage4_r39_lingbot_ta_midband_context_special_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    support = r34.read_support()
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
        interval = r34.auto_keyframe_interval(num_frames)
        eligible = [frame for frame in r34.default_base_keyframes(num_frames) if frame in rows_by_frame]
        selections, seq_meta = selected_indices(rows_by_frame, seq)
        candidate_frames = selections["lingbot_map_stream_flashinfer_v118_r39_ta5_midband_context_special"]
        candidate_bucket_counts = r34.temporal_bucket_counts(eligible, candidate_frames, TEMPORAL_BUCKETS)
        for method, frames in selections.items():
            seq_maps_by_method[method][seq] = frames
            manifest_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r39_lingbot_ta_midband_context_special_manifest_row_v1",
                    "seq": seq,
                    "method": method,
                    "role": METHODS[method]["role"],
                    "policy": METHODS[method]["policy"],
                    "num_frames": num_frames,
                    "scale_frames": 8,
                    "auto_keyframe_interval": interval,
                    "eligible_default_base_keyframes": len(eligible),
                    "action_budget_fraction": ACTION_BUDGET_FRACTION,
                    "min_frame_gap": MIN_FRAME_GAP,
                    "context_only_special_count": len(frames),
                    "candidate_temporal_bucket_counts": json.dumps(candidate_bucket_counts, sort_keys=True),
                    "selected_temporal_bucket_counts": json.dumps(
                        r34.temporal_bucket_counts(eligible, frames, TEMPORAL_BUCKETS), sort_keys=True
                    ),
                    "selected_min_gap": min_gap(frames),
                    "selected_frame_ids": ";".join(str(frame) for frame in frames),
                    **seq_meta,
                    **score_stats(rows_by_frame, frames, eligible),
                }
            )

    for method, meta in METHODS.items():
        write_method_config(config_dir, method, meta["policy"], seq_maps_by_method[method])
    write_manifest(manifest_rows, seq_maps_by_method)
    print(
        json.dumps(
            {
                "stage": r34.rel(STAGE),
                "config": r34.rel(config_dir / "kitti_lingbot_flashinfer_r39_ta_midband_context_special_full_reuse_v105gt.yaml"),
                "manifest_rows": len(manifest_rows),
                "methods": list(METHODS),
                "midband_quantile": MIDBAND_QUANTILE,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
