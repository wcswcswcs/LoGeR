#!/usr/bin/env python3
"""Run v107 Phase3 probation variants under an explicit Phase2 gate override."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml"
)
DEFAULT_REFERENCE = (
    ROOT
    / "Stream3D/outputs/audit/"
    / "v106_stateful_sam2_rolling_scene0050_area20k_e1_preprune6_maxvis45_labelcompact_noempty_full90_gpu6_20260713_1505"
    / "v106_stateful_sam2_rolling_scene_stream"
)
RUNNER = ROOT / "tools/run_v106_stateful_sam2_rolling_scene_stream.py"
PYTHON = Path(sys.executable)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def imread_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b)
    denom = int(np.count_nonzero(union))
    if denom == 0:
        return 1.0
    return float(np.count_nonzero(np.logical_and(a, b)) / denom)


def label_metrics(
    *,
    reference_root: Path,
    candidate_root: Path,
    large_region_area: int,
) -> dict[str, Any]:
    ref_summary_path = reference_root / "summary.json"
    cand_summary_path = candidate_root / "summary.json"
    ref_summary = read_json(ref_summary_path)
    cand_summary = read_json(cand_summary_path)
    ref_records = {int(row["frame_id"]): row for row in ref_summary.get("records", [])}
    cand_records = {int(row["frame_id"]): row for row in cand_summary.get("records", [])}

    missing_frames: list[int] = []
    extra_frames = sorted(set(cand_records) - set(ref_records))
    compared = 0
    pixel_mismatch_count = 0
    bad_frame_count = 0
    ref_fg_px = 0
    cand_fg_px = 0
    fg_intersection_px = 0
    large_ref_px = 0
    large_ref_hit_px = 0
    object_frame_count = 0
    object_frame_tor05_hit_count = 0
    object_frame_best_ious: list[float] = []
    merge_candidate_count = 0
    merge_error_proxy_count = 0
    frame_rows: list[dict[str, Any]] = []

    for frame_id in sorted(ref_records):
        if frame_id not in cand_records:
            missing_frames.append(int(frame_id))
            continue
        ref_label = imread_label(resolve_path(str(ref_records[frame_id]["label_path"]), reference_root))
        cand_label = imread_label(resolve_path(str(cand_records[frame_id]["label_path"]), candidate_root))
        same_shape = ref_label.shape == cand_label.shape
        if not same_shape:
            bad_frame_count += 1
            frame_rows.append({"frame_id": int(frame_id), "same_shape": False})
            continue
        compared += 1
        mismatch = int(np.count_nonzero(ref_label != cand_label))
        pixel_mismatch_count += mismatch
        if mismatch:
            bad_frame_count += 1
        ref_fg = ref_label > 0
        cand_fg = cand_label > 0
        ref_fg_count = int(np.count_nonzero(ref_fg))
        cand_fg_count = int(np.count_nonzero(cand_fg))
        fg_inter = int(np.count_nonzero(ref_fg & cand_fg))
        ref_fg_px += ref_fg_count
        cand_fg_px += cand_fg_count
        fg_intersection_px += fg_inter

        cand_ids = [int(v) for v in np.unique(cand_label).tolist() if int(v) > 0]
        cand_masks = {obj_id: cand_label == int(obj_id) for obj_id in cand_ids}
        ref_ids = [int(v) for v in np.unique(ref_label).tolist() if int(v) > 0]
        tor_hits = 0
        for ref_id in ref_ids:
            ref_mask = ref_label == int(ref_id)
            area = int(np.count_nonzero(ref_mask))
            if area <= 0:
                continue
            object_frame_count += 1
            if area >= int(large_region_area):
                large_ref_px += area
                large_ref_hit_px += int(np.count_nonzero(ref_mask & cand_fg))
            best_iou = 0.0
            for cand_mask in cand_masks.values():
                best_iou = max(best_iou, mask_iou(ref_mask, cand_mask))
            object_frame_best_ious.append(best_iou)
            if best_iou >= 0.5:
                object_frame_tor05_hit_count += 1
                tor_hits += 1

        ref_ids_for_merge = ref_ids
        for cand_id, cand_mask in cand_masks.items():
            cand_area = int(np.count_nonzero(cand_mask))
            if cand_area <= 0:
                continue
            merge_candidate_count += 1
            overlapping_refs = 0
            for ref_id in ref_ids_for_merge:
                ref_mask = ref_label == int(ref_id)
                overlap = int(np.count_nonzero(cand_mask & ref_mask))
                if overlap / max(cand_area, 1) >= 0.10:
                    overlapping_refs += 1
            if overlapping_refs >= 2:
                merge_error_proxy_count += 1

        frame_rows.append(
            {
                "frame_id": int(frame_id),
                "same_shape": True,
                "pixel_mismatch_count": int(mismatch),
                "reference_foreground_px": ref_fg_count,
                "candidate_foreground_px": cand_fg_count,
                "foreground_intersection_px": fg_inter,
                "reference_object_count": int(len(ref_ids)),
                "candidate_object_count": int(len(cand_ids)),
                "tor05_hits": int(tor_hits),
            }
        )

    foreground_recall = float(fg_intersection_px / ref_fg_px) if ref_fg_px else None
    foreground_precision = float(fg_intersection_px / cand_fg_px) if cand_fg_px else None
    tor05 = float(object_frame_tor05_hit_count / object_frame_count) if object_frame_count else None
    large_recall = float(large_ref_hit_px / large_ref_px) if large_ref_px else None
    merge_error_proxy_rate = (
        float(merge_error_proxy_count / merge_candidate_count) if merge_candidate_count else None
    )
    return {
        "schema_version": "stream4d_v107_phase3_reference_metric_proxy_v1",
        "reference_summary": {"path": rel(ref_summary_path), "sha256": sha256_file(ref_summary_path)},
        "candidate_summary": {"path": rel(cand_summary_path), "sha256": sha256_file(cand_summary_path)},
        "frame_count": int(len(ref_records)),
        "compared_frame_count": int(compared),
        "missing_frames": missing_frames,
        "extra_frames": [int(v) for v in extra_frames],
        "bad_frame_count": int(bad_frame_count),
        "pixel_mismatch_count": int(pixel_mismatch_count),
        "foreground_recall_vs_reference": foreground_recall,
        "foreground_precision_vs_reference": foreground_precision,
        "large_region_area_threshold_px": int(large_region_area),
        "large_region_recall_proxy": large_recall,
        "tor_0_5_proxy": tor05,
        "object_frame_count": int(object_frame_count),
        "object_frame_tor05_hit_count": int(object_frame_tor05_hit_count),
        "mean_best_object_frame_iou": float(np.mean(object_frame_best_ious)) if object_frame_best_ious else None,
        "median_best_object_frame_iou": float(np.median(object_frame_best_ious)) if object_frame_best_ious else None,
        "merge_error_proxy_count": int(merge_error_proxy_count),
        "merge_candidate_count": int(merge_candidate_count),
        "merge_error_proxy_rate": merge_error_proxy_rate,
        "label_exact_parity_pass": bool(
            compared == len(ref_records)
            and not missing_frames
            and not extra_frames
            and pixel_mismatch_count == 0
        ),
        "frame_rows": frame_rows,
    }


def metric_summary(summary_path: Path, reference_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = read_json(summary_path)
    stats = summary.get("v106_sam2_rolling_state", {}) or {}
    optimization = summary.get("v106_runtime_optimization", {}) or {}
    return {
        "summary_path": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "frame_count": int(summary.get("frame_count", len(summary.get("records", [])))),
        "wrapper_wall_time_sec": summary.get("wrapper_wall_time_sec"),
        "wrapper_total_with_v106_visual_export_wall_time_sec": summary.get(
            "wrapper_total_with_v106_visual_export_wall_time_sec"
        ),
        "total_runtime_sec": summary.get("total_runtime_sec"),
        "total_tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
        "total_gap_segmentation_runtime_sec": summary.get("total_gap_segmentation_runtime_sec"),
        "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
        "mean_visible_id_count": summary.get("mean_visible_id_count"),
        "mean_foreground_ratio": summary.get("mean_foreground_ratio"),
        "stream_add_masks_call_count": stats.get("stream_add_masks_call_count"),
        "stream_add_masks_input_mask_count": stats.get("stream_add_masks_input_mask_count"),
        "stream_add_masks_admitted_mask_count": stats.get("stream_add_masks_admitted_mask_count"),
        "stream_add_masks_skipped_mask_count": stats.get("stream_add_masks_skipped_mask_count"),
        "stream_add_masks_runtime_sec": stats.get("stream_add_masks_runtime_sec"),
        "reconsolidate_call_count": stats.get("reconsolidate_call_count"),
        "reconsolidate_runtime_sec": stats.get("reconsolidate_runtime_sec"),
        "reconsolidate_frame_output_count_sum": stats.get("reconsolidate_frame_output_count_sum"),
        "post_start_birth_filter_call_count": stats.get("post_start_birth_filter_call_count"),
        "post_start_birth_filter_immediate_area_admit_count": stats.get(
            "post_start_birth_filter_immediate_area_admit_count"
        ),
        "post_start_birth_filter_persistence_match_count": stats.get(
            "post_start_birth_filter_persistence_match_count"
        ),
        "post_start_birth_filter_persistence_admit_count": stats.get(
            "post_start_birth_filter_persistence_admit_count"
        ),
        "post_start_birth_filter_pending_max_count": stats.get("post_start_birth_filter_pending_max_count"),
        "post_start_birth_filter_appearance_enabled_count": stats.get(
            "post_start_birth_filter_appearance_enabled_count"
        ),
        "post_start_birth_filter_appearance_match_count": stats.get("post_start_birth_filter_appearance_match_count"),
        "post_start_birth_filter_appearance_admit_count": stats.get("post_start_birth_filter_appearance_admit_count"),
        "birth_transaction_enabled_count": stats.get("birth_transaction_enabled_count"),
        "birth_transaction_queue_add_count": stats.get("birth_transaction_queue_add_count"),
        "birth_transaction_queued_mask_count": stats.get("birth_transaction_queued_mask_count"),
        "birth_transaction_commit_count": stats.get("birth_transaction_commit_count"),
        "birth_transaction_committed_mask_count": stats.get("birth_transaction_committed_mask_count"),
        "birth_transaction_commit_runtime_sec": stats.get("birth_transaction_commit_runtime_sec"),
        "birth_transaction_reconsolidate_call_count": stats.get("birth_transaction_reconsolidate_call_count"),
        "birth_transaction_delay_frame_sum": stats.get("birth_transaction_delay_frame_sum"),
        "birth_transaction_max_delay_frames_observed": stats.get("birth_transaction_max_delay_frames_observed"),
        "birth_transaction_max_queue_mask_count": stats.get("birth_transaction_max_queue_mask_count"),
        "birth_transaction_max_queue_frame_count": stats.get("birth_transaction_max_queue_frame_count"),
        "optimization": optimization,
        "reference_metrics": reference_metrics or {},
    }


def base_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "P1_one_frame_probation",
            "description": "Require one repeated pending match before post-start birth admission.",
            "args": {
                "--birth-admission-every": "9999",
                "--birth-admission-max-per-frame": "0",
                "--birth-admission-persistence-iou": "0.35",
                "--birth-admission-persistence-hits": "2",
                "--birth-admission-pending-ttl": "3",
                "--birth-admission-persistence-min-area": "8000",
                "--birth-admission-persistence-max-per-frame": "0",
                "--birth-admission-immediate-area": "0",
            },
        },
        {
            "variant_id": "P2_two_frame_probation",
            "description": "Require two repeated pending matches before post-start birth admission.",
            "args": {
                "--birth-admission-every": "9999",
                "--birth-admission-max-per-frame": "0",
                "--birth-admission-persistence-iou": "0.35",
                "--birth-admission-persistence-hits": "3",
                "--birth-admission-pending-ttl": "4",
                "--birth-admission-persistence-min-area": "8000",
                "--birth-admission-persistence-max-per-frame": "0",
                "--birth-admission-immediate-area": "0",
            },
        },
        {
            "variant_id": "P4_two_frame_probation_large_immediate",
            "description": "Two-frame probation plus large unique-area immediate admission.",
            "args": {
                "--birth-admission-every": "9999",
                "--birth-admission-max-per-frame": "0",
                "--birth-admission-persistence-iou": "0.35",
                "--birth-admission-persistence-hits": "3",
                "--birth-admission-pending-ttl": "4",
                "--birth-admission-persistence-min-area": "8000",
                "--birth-admission-persistence-max-per-frame": "0",
                "--birth-admission-immediate-area": "80000",
            },
        },
    ]


def large_immediate_repair_variants() -> list[dict[str, Any]]:
    out = []
    for threshold in (60000, 40000, 20000):
        out.append(
            {
                "variant_id": f"P4_repair_large_immediate_{threshold}",
                "description": (
                    "Two-frame probation with lower large-object immediate-admission threshold "
                    f"({threshold} px) to repair reference recall."
                ),
                "args": {
                    "--birth-admission-every": "9999",
                    "--birth-admission-max-per-frame": "0",
                    "--birth-admission-persistence-iou": "0.35",
                    "--birth-admission-persistence-hits": "3",
                    "--birth-admission-pending-ttl": "4",
                    "--birth-admission-persistence-min-area": "8000",
                    "--birth-admission-persistence-max-per-frame": "0",
                    "--birth-admission-immediate-area": str(threshold),
                },
            }
        )
    return out


def large_immediate_fine_variants() -> list[dict[str, Any]]:
    out = []
    for threshold in (35000, 30000, 25000):
        out.append(
            {
                "variant_id": f"P4_fine_large_immediate_{threshold}",
                "description": (
                    "Fine sweep between exact-reference 20k and low-fidelity 40k immediate thresholds "
                    f"({threshold} px)."
                ),
                "args": {
                    "--birth-admission-every": "9999",
                    "--birth-admission-max-per-frame": "0",
                    "--birth-admission-persistence-iou": "0.35",
                    "--birth-admission-persistence-hits": "3",
                    "--birth-admission-pending-ttl": "4",
                    "--birth-admission-persistence-min-area": "8000",
                    "--birth-admission-persistence-max-per-frame": "0",
                    "--birth-admission-immediate-area": str(threshold),
                },
            }
        )
    return out


def appearance_watcher_variants() -> list[dict[str, Any]]:
    common = {
        "--birth-admission-every": "9999",
        "--birth-admission-max-per-frame": "0",
        "--birth-admission-persistence-iou": "0.35",
        "--birth-admission-persistence-hits": "3",
        "--birth-admission-pending-ttl": "4",
        "--birth-admission-persistence-min-area": "8000",
        "--birth-admission-persistence-max-per-frame": "0",
        "--birth-admission-appearance-enabled": None,
        "--birth-admission-appearance-min-iou": "0.02",
        "--birth-admission-appearance-max-area-ratio": "4.0",
    }
    return [
        {
            "variant_id": "P3_appwatch_immediate35000_strict",
            "description": "Two-frame probation plus RGB/centroid watcher, immediate threshold 35k, strict appearance.",
            "args": {
                **common,
                "--birth-admission-immediate-area": "35000",
                "--birth-admission-appearance-max-color-distance": "0.12",
                "--birth-admission-appearance-max-centroid-distance": "96",
            },
        },
        {
            "variant_id": "P3_appwatch_immediate35000_loose",
            "description": "Two-frame probation plus RGB/centroid watcher, immediate threshold 35k, looser appearance.",
            "args": {
                **common,
                "--birth-admission-immediate-area": "35000",
                "--birth-admission-appearance-max-color-distance": "0.18",
                "--birth-admission-appearance-max-centroid-distance": "128",
            },
        },
        {
            "variant_id": "P3_appwatch_immediate30000_loose",
            "description": "Two-frame probation plus RGB/centroid watcher, immediate threshold 30k, looser appearance.",
            "args": {
                **common,
                "--birth-admission-immediate-area": "30000",
                "--birth-admission-appearance-max-color-distance": "0.18",
                "--birth-admission-appearance-max-centroid-distance": "128",
            },
        },
    ]


def noappearance_control_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "P4_noapp_currentcode_immediate35000",
            "description": "Current-code no-appearance control for immediate threshold 35k.",
            "args": {
                "--birth-admission-every": "9999",
                "--birth-admission-max-per-frame": "0",
                "--birth-admission-persistence-iou": "0.35",
                "--birth-admission-persistence-hits": "3",
                "--birth-admission-pending-ttl": "4",
                "--birth-admission-persistence-min-area": "8000",
                "--birth-admission-persistence-max-per-frame": "0",
                "--birth-admission-immediate-area": "35000",
            },
        },
    ]


def appearance_hits2_variants() -> list[dict[str, Any]]:
    common = {
        "--birth-admission-every": "9999",
        "--birth-admission-max-per-frame": "0",
        "--birth-admission-persistence-iou": "0.35",
        "--birth-admission-persistence-hits": "2",
        "--birth-admission-pending-ttl": "3",
        "--birth-admission-persistence-min-area": "8000",
        "--birth-admission-persistence-max-per-frame": "0",
        "--birth-admission-appearance-enabled": None,
        "--birth-admission-appearance-min-iou": "0.02",
        "--birth-admission-appearance-max-color-distance": "0.18",
        "--birth-admission-appearance-max-centroid-distance": "128",
        "--birth-admission-appearance-max-area-ratio": "4.0",
    }
    return [
        {
            "variant_id": "P3_appwatch_hits2_immediate35000_loose",
            "description": "One-frame confirmation repair for appearance watcher, immediate threshold 35k.",
            "args": {**common, "--birth-admission-immediate-area": "35000"},
        },
        {
            "variant_id": "P3_appwatch_hits2_immediate30000_loose",
            "description": "One-frame confirmation repair for appearance watcher, immediate threshold 30k.",
            "args": {**common, "--birth-admission-immediate-area": "30000"},
        },
    ]


def variants(mode: str) -> list[dict[str, Any]]:
    if mode == "base":
        return base_variants()
    if mode == "large-immediate-repair":
        return large_immediate_repair_variants()
    if mode == "large-immediate-fine":
        return large_immediate_fine_variants()
    if mode == "appearance-watcher":
        return appearance_watcher_variants()
    if mode == "appearance-hits2":
        return appearance_hits2_variants()
    if mode == "noappearance-control":
        return noappearance_control_variants()
    if mode == "base-plus-repair":
        return (
            base_variants()
            + large_immediate_repair_variants()
            + large_immediate_fine_variants()
            + appearance_watcher_variants()
            + appearance_hits2_variants()
            + noappearance_control_variants()
        )
    raise ValueError(f"unsupported variant mode: {mode}")


def command_for_variant(
    *,
    output_root: Path,
    config: Path,
    scene_id: str,
    frame_start: int,
    frame_stride: int,
    frame_count: int,
    gpu: str,
    seed: int,
    variant: dict[str, Any],
) -> list[str]:
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--config",
        str(config),
        "--scene-id",
        str(scene_id),
        "--frame-start",
        str(int(frame_start)),
        "--frame-stride",
        str(int(frame_stride)),
        "--frame-count",
        str(int(frame_count)),
        "--output-root",
        str(output_root / str(variant["variant_id"])),
        "--gpu",
        str(gpu),
        "--seed",
        str(int(seed)),
        "--label-only-visual-export",
        "--compact-visual-video",
    ]
    for key, value in dict(variant["args"]).items():
        cmd.append(str(key))
        if value is not None:
            cmd.append(str(value))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--frame-start", type=int, default=4160)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=90)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--large-region-area", type=int, default=20000)
    parser.add_argument(
        "--variant-mode",
        choices=[
            "base",
            "large-immediate-repair",
            "large-immediate-fine",
            "appearance-watcher",
            "appearance-hits2",
            "noappearance-control",
            "base-plus-repair",
        ],
        default="base",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    phase3 = output_root / "phase3_override"
    phase3.mkdir(parents=True, exist_ok=True)
    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root

    env = os.environ.copy()
    if str(args.gpu).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    reference_metric = metric_summary(reference_root / "summary.json")
    rows: list[dict[str, Any]] = [
        {
            "variant_id": "P0_current_v106_reference",
            "description": "Current v106 best, used as reference; not rerun in this override sweep.",
            "status": "reference_only",
            "command": [],
            "returncode": 0,
            "run_root": rel(reference_root),
            **reference_metric,
        }
    ]
    run_records: list[dict[str, Any]] = []

    for variant in variants(str(args.variant_mode)):
        cmd = command_for_variant(
            output_root=output_root,
            config=config,
            scene_id=str(args.scene_id),
            frame_start=int(args.frame_start),
            frame_stride=int(args.frame_stride),
            frame_count=int(args.frame_count),
            gpu=str(args.gpu),
            seed=int(args.seed),
            variant=variant,
        )
        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        elapsed = float(time.time() - started)
        run_root = output_root / str(variant["variant_id"]) / "v106_stateful_sam2_rolling_scene_stream"
        summary_path = run_root / "summary.json"
        stdout_path = phase3 / f"{variant['variant_id']}.stdout.txt"
        stderr_path = phase3 / f"{variant['variant_id']}.stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        run_record = {
            "variant_id": str(variant["variant_id"]),
            "description": str(variant["description"]),
            "command": cmd,
            "returncode": int(proc.returncode),
            "elapsed_sec": elapsed,
            "stdout_path": rel(stdout_path),
            "stderr_path": rel(stderr_path),
            "run_root": rel(run_root),
            "summary_path": rel(summary_path) if summary_path.exists() else "",
        }
        run_records.append(run_record)
        if proc.returncode != 0 or not summary_path.exists():
            rows.append(
                {
                    "variant_id": str(variant["variant_id"]),
                    "description": str(variant["description"]),
                    "status": "run_failed",
                    **run_record,
                }
            )
            continue
        ref_metrics = label_metrics(
            reference_root=reference_root,
            candidate_root=run_root,
            large_region_area=int(args.large_region_area),
        )
        metric_path = phase3 / f"{variant['variant_id']}.reference_metrics.json"
        write_json(metric_path, ref_metrics)
        rows.append(
            {
                "variant_id": str(variant["variant_id"]),
                "description": str(variant["description"]),
                "status": "completed",
                **run_record,
                **metric_summary(summary_path, ref_metrics),
                "reference_metric_path": rel(metric_path),
            }
        )

    ref_recon = float(reference_metric.get("reconsolidate_runtime_sec") or 0.0)
    ref_calls = float(reference_metric.get("reconsolidate_call_count") or 0.0)
    ref_wall = float(reference_metric.get("wrapper_wall_time_sec") or 0.0)
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") not in {"completed", "reference_only"}:
            scored_rows.append(row)
            continue
        recon = float(row.get("reconsolidate_runtime_sec") or 0.0)
        calls = float(row.get("reconsolidate_call_count") or 0.0)
        wall = float(row.get("wrapper_wall_time_sec") or 0.0)
        metrics = row.get("reference_metrics", {}) or {}
        row["reconsolidation_runtime_reduction_vs_p0"] = (
            float((ref_recon - recon) / ref_recon) if ref_recon > 0 else None
        )
        row["reconsolidation_count_reduction_vs_p0"] = (
            float((ref_calls - calls) / ref_calls) if ref_calls > 0 else None
        )
        row["wall_reduction_vs_p0"] = float((ref_wall - wall) / ref_wall) if ref_wall > 0 else None
        row["phase3_proxy_gate"] = {
            "reconsolidation_runtime_reduction_ge_20pct": bool(
                row["reconsolidation_runtime_reduction_vs_p0"] is not None
                and row["reconsolidation_runtime_reduction_vs_p0"] >= 0.20
            ),
            "large_region_recall_ge_0_98": bool(
                metrics.get("large_region_recall_proxy") is not None
                and float(metrics["large_region_recall_proxy"]) >= 0.98
            ),
            "tor_0_5_ge_0_95": bool(
                metrics.get("tor_0_5_proxy") is not None and float(metrics["tor_0_5_proxy"]) >= 0.95
            ),
            "merge_error_proxy_not_above_p0": None,
            "note": "Proxy gate only; this run intentionally ignores the failed Phase2 live mutation gate.",
        }
        scored_rows.append(row)

    df_rows = []
    for row in scored_rows:
        metrics = row.get("reference_metrics", {}) or {}
        df_rows.append(
            {
                "variant_id": row.get("variant_id"),
                "status": row.get("status"),
                "returncode": row.get("returncode"),
                "wrapper_wall_time_sec": row.get("wrapper_wall_time_sec"),
                "reconsolidate_call_count": row.get("reconsolidate_call_count"),
                "reconsolidate_runtime_sec": row.get("reconsolidate_runtime_sec"),
                "stream_add_masks_input_mask_count": row.get("stream_add_masks_input_mask_count"),
                "stream_add_masks_admitted_mask_count": row.get("stream_add_masks_admitted_mask_count"),
                "stream_add_masks_skipped_mask_count": row.get("stream_add_masks_skipped_mask_count"),
                "appearance_match_count": row.get("post_start_birth_filter_appearance_match_count"),
                "appearance_admit_count": row.get("post_start_birth_filter_appearance_admit_count"),
                "foreground_recall_vs_reference": metrics.get("foreground_recall_vs_reference"),
                "foreground_precision_vs_reference": metrics.get("foreground_precision_vs_reference"),
                "large_region_recall_proxy": metrics.get("large_region_recall_proxy"),
                "tor_0_5_proxy": metrics.get("tor_0_5_proxy"),
                "merge_error_proxy_rate": metrics.get("merge_error_proxy_rate"),
                "reconsolidation_runtime_reduction_vs_p0": row.get("reconsolidation_runtime_reduction_vs_p0"),
                "wall_reduction_vs_p0": row.get("wall_reduction_vs_p0"),
            }
        )
    csv_path = phase3 / "phase3_override_probation_variant_table.csv"
    pd.DataFrame(df_rows).to_csv(csv_path, index=False)
    table_json = phase3 / "phase3_override_probation_variant_table.json"
    write_json(table_json, {"rows": df_rows, "row_count": len(df_rows)})

    completed = [row for row in scored_rows if row.get("status") == "completed"]
    candidates = [
        row
        for row in completed
        if (row.get("phase3_proxy_gate") or {}).get("reconsolidation_runtime_reduction_ge_20pct")
        and (row.get("phase3_proxy_gate") or {}).get("large_region_recall_ge_0_98")
        and (row.get("phase3_proxy_gate") or {}).get("tor_0_5_ge_0_95")
    ]
    best = None
    if completed:
        best = max(
            completed,
            key=lambda row: (
                float(row.get("reconsolidation_runtime_reduction_vs_p0") or -999.0),
                float((row.get("reference_metrics") or {}).get("tor_0_5_proxy") or -1.0),
            ),
        )
    decision = (
        "PASS_PHASE3_OVERRIDE_PROXY_GATE"
        if candidates
        else "NO_GO_PHASE3_OVERRIDE_PROBATION_PROXY_GATE"
    )
    summary = {
        "schema_version": "stream4d_v107_phase3_override_probation_sweep_summary_v1",
        "created_unix_time": time.time(),
        "override_reason": "User explicitly requested continuing after Phase2 gate failure.",
        "phase2_gate_status": "NO_GO_PHASE2_SAM2_MEMORY_PARITY_FAILED",
        "phase3_gate_is_proxy_only": True,
        "variant_mode": str(args.variant_mode),
        "scene_id": str(args.scene_id),
        "frame_start": int(args.frame_start),
        "frame_stride": int(args.frame_stride),
        "frame_count": int(args.frame_count),
        "config": {"path": rel(config), "sha256": sha256_file(config)},
        "reference_run_root": {"path": rel(reference_root), "summary_sha256": sha256_file(reference_root / "summary.json")},
        "run_records": run_records,
        "rows": scored_rows,
        "variant_table_csv": rel(csv_path),
        "variant_table_json": rel(table_json),
        "completed_variant_count": int(len(completed)),
        "failed_variant_count": int(sum(1 for row in scored_rows if row.get("status") == "run_failed")),
        "candidate_count": int(len(candidates)),
        "candidate_variant_ids": [str(row["variant_id"]) for row in candidates],
        "best_variant_id": str(best["variant_id"]) if best else "",
        "decision": decision,
        "next_allowed_family_under_override": (
            "Phase4 transactional admission only as exploratory override; do not claim v107 mainline success."
            if candidates
            else "Repair Phase3 probation/admission fidelity before Phase4, or explicitly accept lower-fidelity override."
        ),
    }
    summary_path = phase3 / "phase3_override_probation_sweep_summary.json"
    write_json(summary_path, summary)
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase3_override_probation_sweep_run_v1",
            "summary": rel(summary_path),
            "decision": decision,
            "completed_variant_count": summary["completed_variant_count"],
            "candidate_count": summary["candidate_count"],
            "best_variant_id": summary["best_variant_id"],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary": str(summary_path),
                "decision": decision,
                "completed_variant_count": summary["completed_variant_count"],
                "candidate_count": summary["candidate_count"],
                "best_variant_id": summary["best_variant_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["completed_variant_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
