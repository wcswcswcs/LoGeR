#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R27 holdout calibrated-polarity anchor configs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor"
SUPPORT = RESULT_ROOT / "stage4_r27_holdout_cue_prep/summary/stage4_r27_holdout_frame_semantic_support_rows.csv"
TRACE_DIR = RESULT_ROOT / "stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
SEQS = ("01", "05")
BUFFER_SIZE = 32
DEFAULT_ANCHOR_COUNT = 8
SCALE_FRAME_COUNT = 8
RANDOM_SEED = 11827

METHODS = {
    "lingbot_map_stream_flashinfer_v118_r27_ai4_holdout_calibrated_polarity_anchor_first32": {
        "policy": "AI4_HOLDOUT_CALIBRATED_POLARITY_ANCHOR_FIRST32",
        "role": "candidate",
        "description": "apply the frozen R26 high-vs-low internal polarity rule to fresh 01/05 holdout cues",
    },
    "lingbot_map_stream_flashinfer_v118_r27_ai4_holdout_opposite_polarity_anchor_first32": {
        "policy": "AI4_HOLDOUT_OPPOSITE_POLARITY_CONTROL_ANCHOR_FIRST32",
        "role": "opposite_polarity_control",
        "description": "use the opposite high/low internal-QK polarity on the same fresh holdout cues",
    },
    "lingbot_map_stream_flashinfer_v118_r27_ai4_holdout_random_polarity_pool_anchor_first32": {
        "policy": "AI4_HOLDOUT_MATCHED_RANDOM_POLARITY_POOL_ANCHOR_FIRST32",
        "role": "matched_random_control",
        "description": "choose deterministic random same-count anchors from the same first32 non-default pool",
    },
}


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_role_count(raw: str, role: str) -> int:
    if not raw:
        return 0
    prefix = f"{role}:"
    total = 0
    for part in raw.split(";"):
        if part.startswith(prefix):
            try:
                total += int(part.split(":", 1)[1])
            except ValueError:
                pass
    return total


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
            bucket = stats.setdefault(
                frame,
                {
                    "frame_id": frame,
                    "read_rows": 0,
                    "sum_qk_cosine": 0.0,
                    "sum_qk_softmax": 0.0,
                    "sum_qk_rank": 0.0,
                },
            )
            bucket["read_rows"] += 1
            bucket["sum_qk_cosine"] += fnum(row.get("qk_relevance_cosine"))
            bucket["sum_qk_softmax"] += fnum(row.get("qk_relevance_softmax"))
            bucket["sum_qk_rank"] += fnum(row.get("qk_relevance_rank"))
    for bucket in stats.values():
        n = int(bucket["read_rows"])
        bucket["mean_qk_cosine"] = bucket["sum_qk_cosine"] / n if n else 0.0
        bucket["mean_qk_softmax"] = bucket["sum_qk_softmax"] / n if n else 0.0
        bucket["mean_qk_rank"] = bucket["sum_qk_rank"] / n if n else 0.0
    return stats


def high_low_random(internal: dict[int, dict[str, Any]], seq: str) -> tuple[list[int], list[int], list[int]]:
    eligible = [frame for frame in range(DEFAULT_ANCHOR_COUNT, BUFFER_SIZE) if frame in internal]
    if len(eligible) < SCALE_FRAME_COUNT:
        raise RuntimeError(f"not enough first{BUFFER_SIZE} non-default internal rows for seq {seq}: {len(eligible)}")
    ranked_low = sorted(eligible, key=lambda frame: (fnum(internal[frame].get("mean_qk_cosine")), frame))
    ranked_high = list(reversed(ranked_low))
    rng = random.Random(RANDOM_SEED + int(seq))
    random_order = list(eligible)
    rng.shuffle(random_order)
    return (
        sorted(ranked_high[:SCALE_FRAME_COUNT]),
        sorted(ranked_low[:SCALE_FRAME_COUNT]),
        sorted(random_order[:SCALE_FRAME_COUNT]),
    )


def calibration_decision(support: dict[int, dict[str, Any]], high_frames: list[int]) -> dict[str, Any]:
    stable_counts = [parse_role_count(str(support.get(frame, {}).get("top_roles", "")), "stable_landmark") for frame in high_frames]
    sky_best_count = sum(1 for frame in high_frames if str(support.get(frame, {}).get("best_track_role", "")) == "sky_lowobs")
    stable_median = median(stable_counts) if stable_counts else 0
    sky_best_fraction = sky_best_count / len(high_frames) if high_frames else 0.0
    high_internal_reliable = not (stable_median <= 1 and sky_best_fraction >= 0.75)
    return {
        "stable_count_median_in_high_internal_top8": stable_median,
        "sky_lowobs_best_fraction_in_high_internal_top8": sky_best_fraction,
        "polarity": "high_internal" if high_internal_reliable else "low_internal",
        "reason": (
            "high_internal_top8_has_sufficient_stable_landmark_support"
            if high_internal_reliable
            else "high_internal_top8_dominated_by_sky_lowobs_with_low_stable_support"
        ),
    }


def stats_for_frames(internal: dict[int, dict[str, Any]], support: dict[int, dict[str, Any]], frames: list[int]) -> dict[str, Any]:
    qk = [fnum(internal[frame].get("mean_qk_cosine")) for frame in frames]
    semantic = [fnum(support.get(frame, {}).get("max_semantic_persistence_prefix")) for frame in frames]
    stable = [parse_role_count(str(support.get(frame, {}).get("top_roles", "")), "stable_landmark") for frame in frames]
    dynamic = [parse_role_count(str(support.get(frame, {}).get("top_roles", "")), "dynamic") for frame in frames]
    return {
        "mean_qk_cosine_min": min(qk) if qk else "",
        "mean_qk_cosine_max": max(qk) if qk else "",
        "mean_qk_cosine_median": median(qk) if qk else "",
        "semantic_persistence_median_diagnostic": median(semantic) if semantic else "",
        "stable_count_median_diagnostic": median(stable) if stable else "",
        "dynamic_count_median_diagnostic": median(dynamic) if dynamic else "",
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
    (config_dir / "kitti_lingbot_flashinfer_r27_ai_holdout_calibrated_polarity_anchor_full_reuse_v105gt.yaml").write_text(
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


def write_manifest(rows: list[dict[str, Any]], seq_maps_by_method: dict[str, dict[str, list[int]]], calibration: dict[str, Any]) -> None:
    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_manifest.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema": "acl2_v118tf_stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_manifest_v1",
        "stage": str(STAGE.relative_to(ROOT)),
        "dataset": DATASET,
        "holdout_sequences": list(SEQS),
        "baseline_method": BASELINE_METHOD,
        "support_rows": str(SUPPORT.relative_to(ROOT)),
        "trace_dir": str(TRACE_DIR.relative_to(ROOT)),
        "score_field": "mean_qk_cosine_from_default_local_image_patch_read_rows",
        "pre_registered_rule_source": "R26 AI4 calibrated-polarity rule frozen before first R27 01/05 holdout action run",
        "calibration_rule": "if high-internal top8 stable_count_median<=1 and sky_lowobs_best_fraction>=0.75 then use low_internal else high_internal",
        "buffer_size": BUFFER_SIZE,
        "eligible_frame_range": f"{DEFAULT_ANCHOR_COUNT}-{BUFFER_SIZE - 1}",
        "scale_frame_count": SCALE_FRAME_COUNT,
        "random_seed": RANDOM_SEED,
        "methods": METHODS,
        "calibration_by_seq": calibration,
        "scale_frame_indices_by_method": seq_maps_by_method,
        "manifest_csv": str(csv_path.relative_to(ROOT)),
    }
    (out / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_manifest.json").write_text(
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
    calibration: dict[str, Any] = {}
    for seq in SEQS:
        internal = read_internal_scores(seq)
        high_frames, low_frames, random_frames = high_low_random(internal, seq)
        support_rows = support.get(seq, {})
        decision = calibration_decision(support_rows, high_frames)
        candidate_frames = high_frames if decision["polarity"] == "high_internal" else low_frames
        opposite_polarity = "low_internal" if decision["polarity"] == "high_internal" else "high_internal"
        opposite_frames = low_frames if decision["polarity"] == "high_internal" else high_frames
        calibration[seq] = {
            **decision,
            "opposite_polarity": opposite_polarity,
            "high_internal_frames": high_frames,
            "low_internal_frames": low_frames,
            "candidate_frames": candidate_frames,
            "opposite_frames": opposite_frames,
            "random_frames": random_frames,
        }
        selections = {
            "lingbot_map_stream_flashinfer_v118_r27_ai4_holdout_calibrated_polarity_anchor_first32": (candidate_frames, decision["polarity"]),
            "lingbot_map_stream_flashinfer_v118_r27_ai4_holdout_opposite_polarity_anchor_first32": (opposite_frames, opposite_polarity),
            "lingbot_map_stream_flashinfer_v118_r27_ai4_holdout_random_polarity_pool_anchor_first32": (random_frames, "random"),
        }
        for method, (frames, polarity) in selections.items():
            seq_maps_by_method[method][seq] = frames
            manifest_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_manifest_row_v1",
                    "seq": seq,
                    "method": method,
                    "role": METHODS[method]["role"],
                    "policy": METHODS[method]["policy"],
                    "actual_polarity": polarity,
                    "candidate_rule_polarity": decision["polarity"],
                    "calibration_reason": decision["reason"],
                    "buffer_size": BUFFER_SIZE,
                    "eligible_frame_range": f"{DEFAULT_ANCHOR_COUNT}-{BUFFER_SIZE - 1}",
                    "scale_frame_count": SCALE_FRAME_COUNT,
                    "selected_frame_ids": ";".join(str(frame) for frame in frames),
                    **stats_for_frames(internal, support_rows, frames),
                }
            )
    for method, meta in METHODS.items():
        write_method_config(config_dir, method, meta["policy"], seq_maps_by_method[method])
    write_manifest(manifest_rows, seq_maps_by_method, calibration)
    print(
        json.dumps(
            {
                "stage": str(STAGE.relative_to(ROOT)),
                "config": str(
                    (config_dir / "kitti_lingbot_flashinfer_r27_ai_holdout_calibrated_polarity_anchor_full_reuse_v105gt.yaml").relative_to(ROOT)
                ),
                "manifest_rows": len(manifest_rows),
                "methods": list(METHODS),
                "calibration_by_seq": calibration,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
