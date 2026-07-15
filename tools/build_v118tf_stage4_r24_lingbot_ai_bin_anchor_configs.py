#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R24 bin-balanced LingBot anchor configs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r24_lingbot_ai_bin_anchor"
SUPPORT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_frame_semantic_support_rows.csv"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
SCORE_FIELD = "max_semantic_persistence_prefix"
BUFFER_SIZE = 32
BIN_SIZE = 4
RANDOM_SEED = 11824

METHODS = {
    "lingbot_map_stream_flashinfer_v118_r24_ai3_semantic_bin_anchor_first32": {
        "policy": "AI3_SEMANTIC_BIN_BALANCED_ANCHOR_FIRST32",
        "role": "candidate",
    },
    "lingbot_map_stream_flashinfer_v118_r24_ai3_reverse_bin_lowsemantic_anchor_first32": {
        "policy": "AI3_REVERSE_BIN_LOW_SEMANTIC_ANCHOR_FIRST32",
        "role": "reverse_control",
    },
    "lingbot_map_stream_flashinfer_v118_r24_ai3_random_bin_anchor_first32": {
        "policy": "AI3_MATCHED_RANDOM_BIN_ANCHOR_FIRST32",
        "role": "matched_random_control",
    },
}


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_support() -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    with SUPPORT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(str(row["seq"]), {})[int(row["frame_id"])] = row
    return out


def bins() -> list[list[int]]:
    return [list(range(start, start + BIN_SIZE)) for start in range(0, BUFFER_SIZE, BIN_SIZE)]


def selected_indices(rows_by_frame: dict[int, dict[str, Any]], seq: str) -> dict[str, list[int]]:
    candidate: list[int] = []
    reverse: list[int] = []
    random_sel: list[int] = []
    rng = random.Random(RANDOM_SEED + int(seq))
    for bin_frames in bins():
        eligible = [frame for frame in bin_frames if frame in rows_by_frame]
        if not eligible:
            raise RuntimeError(f"empty first32 bin for seq {seq}: {bin_frames}")
        ranked = sorted(eligible, key=lambda frame: (fnum(rows_by_frame[frame].get(SCORE_FIELD)), frame))
        reverse.append(ranked[0])
        candidate.append(ranked[-1])
        random_sel.append(rng.choice(eligible))
    return {
        "lingbot_map_stream_flashinfer_v118_r24_ai3_semantic_bin_anchor_first32": sorted(candidate),
        "lingbot_map_stream_flashinfer_v118_r24_ai3_reverse_bin_lowsemantic_anchor_first32": sorted(reverse),
        "lingbot_map_stream_flashinfer_v118_r24_ai3_random_bin_anchor_first32": sorted(random_sel),
    }


def score_stats(rows_by_frame: dict[int, dict[str, Any]], frames: list[int]) -> dict[str, Any]:
    scores = [fnum(rows_by_frame[frame].get(SCORE_FIELD)) for frame in frames]
    return {
        "score_min": min(scores) if scores else "",
        "score_max": max(scores) if scores else "",
        "score_median": median(scores) if scores else "",
    }


def write_main_config(config_dir: Path) -> None:
    lines = [
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
    (config_dir / "kitti_lingbot_flashinfer_r24_ai_bin_anchor_full_reuse_v105gt.yaml").write_text(
        "\n".join(lines),
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
        "_stage4_action_mode: anchor_scale_frame_indices",
        f"_stage4_action_label: {policy}",
        "_stage4_scale_frame_indices_by_seq:",
    ]
    for seq in sorted(seq_maps):
        lines.append(f'  "{seq}": {json.dumps(seq_maps[seq], separators=(",", ":"))}')
    lines.append("")
    (method_dir / f"{method}.yaml").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(rows: list[dict[str, Any]], seq_maps_by_method: dict[str, dict[str, list[int]]]) -> None:
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "stage4_r24_lingbot_ai_bin_anchor_manifest.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "acl2_v118tf_stage4_r24_lingbot_ai_bin_anchor_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "dataset": DATASET,
        "support_rows": str(SUPPORT.relative_to(ROOT)),
        "score_field": SCORE_FIELD,
        "buffer_size": BUFFER_SIZE,
        "bin_size": BIN_SIZE,
        "bin_count": len(bins()),
        "random_seed": RANDOM_SEED,
        "methods": METHODS,
        "scale_frame_indices_by_method": seq_maps_by_method,
        "manifest_csv": str(csv_path.relative_to(ROOT)),
    }
    (out / "stage4_r24_lingbot_ai_bin_anchor_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
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
        selections = selected_indices(rows_by_frame, seq)
        for method, frames in selections.items():
            seq_maps_by_method[method][seq] = frames
            manifest_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r24_lingbot_ai_bin_anchor_manifest_row_v1",
                    "seq": seq,
                    "method": method,
                    "role": METHODS[method]["role"],
                    "policy": METHODS[method]["policy"],
                    "buffer_size": BUFFER_SIZE,
                    "bin_size": BIN_SIZE,
                    "bin_count": len(bins()),
                    "selected_frame_ids": ";".join(str(frame) for frame in frames),
                    **score_stats(rows_by_frame, frames),
                }
            )
    for method, meta in METHODS.items():
        write_method_config(config_dir, method, meta["policy"], seq_maps_by_method[method])
    write_manifest(manifest_rows, seq_maps_by_method)
    print(
        json.dumps(
            {
                "stage": str(STAGE.relative_to(ROOT)),
                "config": str((config_dir / "kitti_lingbot_flashinfer_r24_ai_bin_anchor_full_reuse_v105gt.yaml").relative_to(ROOT)),
                "manifest_rows": len(manifest_rows),
                "methods": list(METHODS),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
