#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R33 LingBot TA soft context-gate configs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r33_lingbot_ta_soft_context_gate"
SUPPORT = RESULT_ROOT / "stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_frame_semantic_support_rows.csv"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
SCORE_FIELD = "max_semantic_persistence_prefix"
ACTION_BUDGET_FRACTION = 0.25
RANDOM_SEED = 11833
CONTEXT_TOKEN_MASK = [1.0, 1.0, 1.0, 1.0, 1.0, 0.25]

METHODS = {
    "lingbot_map_stream_flashinfer_v118_r33_ta3_soft_semantic_lowgate": {
        "policy": "TA3_SOFT_SEMANTIC_LOW_CONTEXT_GATE",
        "role": "candidate",
        "description": "soft context-token gate the lowest semantic-persistence default base keyframes",
    },
    "lingbot_map_stream_flashinfer_v118_r33_ta3_soft_reverse_highgate": {
        "policy": "TA3_SOFT_REVERSE_HIGH_CONTEXT_GATE",
        "role": "reverse_control",
        "description": "soft context-token gate the highest semantic-persistence default base keyframes",
    },
    "lingbot_map_stream_flashinfer_v118_r33_ta3_soft_random_gate": {
        "policy": "TA3_SOFT_MATCHED_RANDOM_CONTEXT_GATE",
        "role": "matched_random_control",
        "description": "soft context-token gate a deterministic random same-budget subset",
    },
}


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


def selected_indices(rows_by_frame: dict[int, dict[str, Any]], seq: str) -> dict[str, list[int]]:
    num_frames = max(rows_by_frame) + 1
    eligible = [frame for frame in default_base_keyframes(num_frames) if frame in rows_by_frame]
    if not eligible:
        raise RuntimeError(f"no eligible default base keyframes for seq {seq}")
    budget = max(1, int(len(eligible) * ACTION_BUDGET_FRACTION))
    ranked_low = sorted(eligible, key=lambda frame: (fnum(rows_by_frame[frame].get(SCORE_FIELD)), frame))
    ranked_high = list(reversed(ranked_low))
    rng = random.Random(RANDOM_SEED + int(seq))
    random_order = list(eligible)
    rng.shuffle(random_order)
    return {
        "lingbot_map_stream_flashinfer_v118_r33_ta3_soft_semantic_lowgate": sorted(ranked_low[:budget]),
        "lingbot_map_stream_flashinfer_v118_r33_ta3_soft_reverse_highgate": sorted(ranked_high[:budget]),
        "lingbot_map_stream_flashinfer_v118_r33_ta3_soft_random_gate": sorted(random_order[:budget]),
    }


def score_stats(rows_by_frame: dict[int, dict[str, Any]], frames: list[int]) -> dict[str, Any]:
    scores = [fnum(rows_by_frame[frame].get(SCORE_FIELD)) for frame in frames]
    return {
        "score_min": min(scores) if scores else "",
        "score_max": max(scores) if scores else "",
        "score_median": median(scores) if scores else "",
    }


def write_main_config(config_dir: Path) -> None:
    (config_dir / "kitti_lingbot_flashinfer_r33_ta_soft_context_gate_full_reuse_v105gt.yaml").write_text(
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
        "_stage4_action_mode: trajectory_context_token_mask",
        f"_stage4_action_label: {policy}",
        f"_stage4_context_token_mask: {json.dumps(CONTEXT_TOKEN_MASK, separators=(',', ':'))}",
        "_stage4_force_non_keyframe_indices_by_seq:",
    ]
    for seq in sorted(seq_maps):
        lines.append(f'  "{seq}": {json.dumps(seq_maps[seq], separators=(",", ":"))}')
    lines.append("")
    (method_dir / f"{method}.yaml").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(rows: list[dict[str, Any]], seq_maps_by_method: dict[str, dict[str, list[int]]]) -> None:
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "stage4_r33_lingbot_ta_soft_context_gate_manifest.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "acl2_v118tf_stage4_r33_lingbot_ta_soft_context_gate_manifest_v1",
        "stage": rel(STAGE),
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": rel(SUPPORT),
        "score_field": SCORE_FIELD,
        "action_budget_fraction": ACTION_BUDGET_FRACTION,
        "random_seed": RANDOM_SEED,
        "action_mode": "trajectory_context_token_mask",
        "context_token_mask": CONTEXT_TOKEN_MASK,
        "methods": METHODS,
        "force_context_indices_by_method": seq_maps_by_method,
        "manifest_csv": rel(csv_path),
    }
    (out / "stage4_r33_lingbot_ta_soft_context_gate_manifest.json").write_text(
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
        eligible = default_base_keyframes(num_frames)
        selections = selected_indices(rows_by_frame, seq)
        for method, frames in selections.items():
            seq_maps_by_method[method][seq] = frames
            stats = score_stats(rows_by_frame, frames)
            manifest_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r33_lingbot_ta_soft_context_gate_manifest_row_v1",
                    "seq": seq,
                    "method": method,
                    "role": METHODS[method]["role"],
                    "policy": METHODS[method]["policy"],
                    "num_frames": num_frames,
                    "scale_frames": 8,
                    "auto_keyframe_interval": interval,
                    "eligible_default_base_keyframes": len(eligible),
                    "action_budget_fraction": ACTION_BUDGET_FRACTION,
                    "soft_context_gate_count": len(frames),
                    "selected_frame_ids": ";".join(str(frame) for frame in frames),
                    "context_token_mask": ",".join(f"{value:g}" for value in CONTEXT_TOKEN_MASK),
                    **stats,
                }
            )

    for method, meta in METHODS.items():
        write_method_config(config_dir, method, meta["policy"], seq_maps_by_method[method])
    write_manifest(manifest_rows, seq_maps_by_method)
    print(
        json.dumps(
            {
                "stage": rel(STAGE),
                "config": rel(config_dir / "kitti_lingbot_flashinfer_r33_ta_soft_context_gate_full_reuse_v105gt.yaml"),
                "manifest_rows": len(manifest_rows),
                "methods": list(METHODS),
                "action_mode": "trajectory_context_token_mask",
                "context_token_mask": CONTEXT_TOKEN_MASK,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
