#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R25 internal-read LingBot anchor configs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r25_lingbot_ai_internal_read_anchor"
SUPPORT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_frame_semantic_support_rows.csv"
TRACE_DIR = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
BUFFER_SIZE = 32
DEFAULT_ANCHOR_COUNT = 8
SCALE_FRAME_COUNT = 8
RANDOM_SEED = 11825

METHODS = {
    "lingbot_map_stream_flashinfer_v118_r25_ai1_internal_qk_nondefault_anchor_first32": {
        "policy": "AI1_INTERNAL_QK_NONDEFAULT_ANCHOR_FIRST32",
        "role": "candidate",
        "description": "choose the highest mean internal-QK cosine frames from first32 non-default local pages",
    },
    "lingbot_map_stream_flashinfer_v118_r25_ai1_reverse_lowinternal_nondefault_anchor_first32": {
        "policy": "AI1_REVERSE_LOW_INTERNAL_QK_NONDEFAULT_ANCHOR_FIRST32",
        "role": "reverse_control",
        "description": "choose the lowest mean internal-QK cosine frames from the same first32 non-default local-page pool",
    },
    "lingbot_map_stream_flashinfer_v118_r25_ai1_random_nondefault_anchor_first32": {
        "policy": "AI1_MATCHED_RANDOM_NONDEFAULT_ANCHOR_FIRST32",
        "role": "matched_random_control",
        "description": "choose deterministic random same-count anchors from the same first32 non-default local-page pool",
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


def read_internal_scores(seq: str) -> dict[int, dict[str, Any]]:
    path = TRACE_DIR / f"seq{seq}_flashinfer_trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    stats: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "read":
                continue
            if row.get("memory_family") != "local":
                continue
            if row.get("token_type") != "image_patch":
                continue
            frame_raw = row.get("source_frame_id")
            if frame_raw is None:
                continue
            frame = int(frame_raw)
            if not (DEFAULT_ANCHOR_COUNT <= frame < BUFFER_SIZE):
                continue
            score = fnum(row.get("qk_relevance_cosine"))
            softmax = fnum(row.get("qk_relevance_softmax"))
            rank = fnum(row.get("qk_relevance_rank"))
            bucket = stats.setdefault(
                frame,
                {
                    "frame_id": frame,
                    "read_rows": 0,
                    "sum_qk_cosine": 0.0,
                    "sum_qk_softmax": 0.0,
                    "sum_qk_rank": 0.0,
                    "last_read_time": 0,
                },
            )
            bucket["read_rows"] += 1
            bucket["sum_qk_cosine"] += score
            bucket["sum_qk_softmax"] += softmax
            bucket["sum_qk_rank"] += rank
            bucket["last_read_time"] = max(bucket["last_read_time"], int(row.get("last_read_time") or 0))
    for bucket in stats.values():
        n = int(bucket["read_rows"])
        bucket["mean_qk_cosine"] = bucket["sum_qk_cosine"] / n if n else 0.0
        bucket["mean_qk_softmax"] = bucket["sum_qk_softmax"] / n if n else 0.0
        bucket["mean_qk_rank"] = bucket["sum_qk_rank"] / n if n else 0.0
    return stats


def selected_indices(internal: dict[int, dict[str, Any]], seq: str) -> dict[str, list[int]]:
    eligible = [frame for frame in range(DEFAULT_ANCHOR_COUNT, BUFFER_SIZE) if frame in internal]
    if len(eligible) < SCALE_FRAME_COUNT:
        raise RuntimeError(f"not enough first{BUFFER_SIZE} non-default internal rows for seq {seq}: {len(eligible)}")
    ranked_low = sorted(eligible, key=lambda frame: (fnum(internal[frame].get("mean_qk_cosine")), frame))
    ranked_high = list(reversed(ranked_low))
    rng = random.Random(RANDOM_SEED + int(seq))
    random_order = list(eligible)
    rng.shuffle(random_order)
    return {
        "lingbot_map_stream_flashinfer_v118_r25_ai1_internal_qk_nondefault_anchor_first32": sorted(ranked_high[:SCALE_FRAME_COUNT]),
        "lingbot_map_stream_flashinfer_v118_r25_ai1_reverse_lowinternal_nondefault_anchor_first32": sorted(ranked_low[:SCALE_FRAME_COUNT]),
        "lingbot_map_stream_flashinfer_v118_r25_ai1_random_nondefault_anchor_first32": sorted(random_order[:SCALE_FRAME_COUNT]),
    }


def score_stats(internal: dict[int, dict[str, Any]], support: dict[int, dict[str, Any]], frames: list[int]) -> dict[str, Any]:
    qk = [fnum(internal[frame].get("mean_qk_cosine")) for frame in frames]
    softmax = [fnum(internal[frame].get("mean_qk_softmax")) for frame in frames]
    ranks = [fnum(internal[frame].get("mean_qk_rank")) for frame in frames]
    semantic = [fnum(support.get(frame, {}).get("max_semantic_persistence_prefix")) for frame in frames]
    return {
        "mean_qk_cosine_min": min(qk) if qk else "",
        "mean_qk_cosine_max": max(qk) if qk else "",
        "mean_qk_cosine_median": median(qk) if qk else "",
        "mean_qk_softmax_median": median(softmax) if softmax else "",
        "mean_qk_rank_median": median(ranks) if ranks else "",
        "semantic_persistence_median_diagnostic": median(semantic) if semantic else "",
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
    (config_dir / "kitti_lingbot_flashinfer_r25_ai_internal_read_anchor_full_reuse_v105gt.yaml").write_text(
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
    csv_path = out / "stage4_r25_lingbot_ai_internal_read_anchor_manifest.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "acl2_v118tf_stage4_r25_lingbot_ai_internal_read_anchor_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(SUPPORT.relative_to(ROOT)),
        "trace_dir": str(TRACE_DIR.relative_to(ROOT)),
        "score_field": "mean_qk_cosine_from_default_local_image_patch_read_rows",
        "buffer_size": BUFFER_SIZE,
        "eligible_frame_range": f"{DEFAULT_ANCHOR_COUNT}-{BUFFER_SIZE - 1}",
        "scale_frame_count": SCALE_FRAME_COUNT,
        "random_seed": RANDOM_SEED,
        "methods": METHODS,
        "scale_frame_indices_by_method": seq_maps_by_method,
        "manifest_csv": str(csv_path.relative_to(ROOT)),
    }
    (out / "stage4_r25_lingbot_ai_internal_read_anchor_manifest.json").write_text(
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
    for seq in ("00", "02"):
        internal = read_internal_scores(seq)
        selections = selected_indices(internal, seq)
        support_rows = support.get(seq, {})
        for method, frames in selections.items():
            seq_maps_by_method[method][seq] = frames
            manifest_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r25_lingbot_ai_internal_read_anchor_manifest_row_v1",
                    "seq": seq,
                    "method": method,
                    "role": METHODS[method]["role"],
                    "policy": METHODS[method]["policy"],
                    "buffer_size": BUFFER_SIZE,
                    "eligible_frame_range": f"{DEFAULT_ANCHOR_COUNT}-{BUFFER_SIZE - 1}",
                    "scale_frame_count": SCALE_FRAME_COUNT,
                    "selected_frame_ids": ";".join(str(frame) for frame in frames),
                    **score_stats(internal, support_rows, frames),
                }
            )
    for method, meta in METHODS.items():
        write_method_config(config_dir, method, meta["policy"], seq_maps_by_method[method])
    write_manifest(manifest_rows, seq_maps_by_method)
    print(
        json.dumps(
            {
                "stage": str(STAGE.relative_to(ROOT)),
                "config": str(
                    (config_dir / "kitti_lingbot_flashinfer_r25_ai_internal_read_anchor_full_reuse_v105gt.yaml").relative_to(ROOT)
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
