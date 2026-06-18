#!/usr/bin/env python3
"""Batch-generate KITTI sparse masklets with top-level semantic segmentation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KITTI 00-10 Video Masklet + semantic PT preprocessing.")
    parser.add_argument("--sequences", default="00,01,02,03,04,05,06,07,08,09,10")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--input_root", default="data/kitti/dataset/sequences")
    parser.add_argument("--output_root", default="results/kitti_preprocess")
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--videomt_window_size", type=int, default=32)
    parser.add_argument("--dvisplus_window_size", type=int, default=32)
    parser.add_argument("--force", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--skip_sha256", type=int, default=0)
    return parser.parse_args()


def _split_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _image_dir(input_root: Path, seq: str) -> Path:
    return input_root / seq / "image_2"


def _count_frames(image_dir: Path) -> int:
    return len([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pt_summary(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation") or {}
    label_maps = sem.get("label_maps", None)
    confidence_maps = sem.get("confidence_maps", None)
    shape = list(label_maps.shape) if label_maps is not None else []
    dtype = str(label_maps.dtype) if label_maps is not None else ""
    confidence_shape = list(confidence_maps.shape) if confidence_maps is not None else []
    confidence_dtype = str(confidence_maps.dtype) if confidence_maps is not None else ""
    return {
        "format": payload.get("format"),
        "num_frames": int(payload.get("num_frames", -1)),
        "frame_height": int(payload.get("frame_height", -1)),
        "frame_width": int(payload.get("frame_width", -1)),
        "num_masklets": int(payload.get("num_masklets", len(payload.get("tracks", [])))),
        "num_tracks": int(len(payload.get("tracks", []))),
        "has_semantic_segmentation": bool(sem),
        "semantic_format": sem.get("format", ""),
        "semantic_shape": shape,
        "semantic_dtype": dtype,
        "semantic_num_labels": int(len(sem.get("label_names", []))) if sem else 0,
        "semantic_label_names": list(sem.get("label_names", [])) if sem else [],
        "has_semantic_confidence": bool(confidence_maps is not None),
        "semantic_confidence_shape": confidence_shape,
        "semantic_confidence_dtype": confidence_dtype,
    }


def _valid_final(path: Path, expected_frames: int) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        summary = _load_pt_summary(path)
    except Exception:
        return None
    if summary["format"] != "sparse_masklets_v1":
        return None
    if summary["num_frames"] != int(expected_frames):
        return None
    if not summary["has_semantic_segmentation"]:
        return None
    if summary["semantic_format"] != "semantic_label_maps_v1":
        return None
    if summary["semantic_shape"] != [
        summary["num_frames"],
        summary["frame_height"],
        summary["frame_width"],
    ]:
        return None
    if not summary["has_semantic_confidence"]:
        return None
    if summary["semantic_confidence_shape"] != [
        summary["num_frames"],
        summary["frame_height"],
        summary["frame_width"],
    ]:
        return None
    return summary


def _run_cmd(
    cmd: Sequence[str],
    log_path: Path,
    env: Dict[str, str],
    cwd: Path,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.time() - start
    return {
        "cmd": list(cmd),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
    }


def _stage_valid(
    path: Path,
    expected_frames: int,
    *,
    require_semantic_segmentation: bool = False,
    require_semantic_confidence: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        summary = _load_pt_summary(path)
        if summary.get("format") != "sparse_masklets_v1":
            return False
        if int(summary.get("num_frames", -1)) != int(expected_frames):
            return False
        if require_semantic_segmentation and not summary["has_semantic_segmentation"]:
            return False
        if require_semantic_confidence and not summary["has_semantic_confidence"]:
            return False
        if require_semantic_segmentation and summary["semantic_shape"] != [
            summary["num_frames"],
            summary["frame_height"],
            summary["frame_width"],
        ]:
            return False
        if require_semantic_confidence and summary["semantic_confidence_shape"] != [
            summary["num_frames"],
            summary["frame_height"],
            summary["frame_width"],
        ]:
            return False
        return True
    except Exception:
        return False


def _run_sequence(seq: str, gpu_queue: "queue.Queue[str]", args: argparse.Namespace) -> Dict[str, Any]:
    gpu = gpu_queue.get()
    try:
        input_root = (REPO_ROOT / args.input_root).resolve() if not Path(args.input_root).is_absolute() else Path(args.input_root)
        output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
        image_dir = _image_dir(input_root, seq)
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing KITTI image dir: {image_dir}")
        frame_count = _count_frames(image_dir)
        seq_root = output_root / seq
        log_dir = seq_root / "logs"
        seq_root.mkdir(parents=True, exist_ok=True)

        videomt_dir = seq_root / "videomt_l_vspw_w32_thingstuff"
        sam31_dir = seq_root / "sam31_textmatch_caronly_signmerge"
        carmerge_dir = seq_root / "sam31_textmatch_caronly_signmerge_carmerge_adjflicker"
        fused_dir = seq_root / "videomt_sam31_carmerge_dropcar_fusion"
        final_pt = seq_root / "sparse_masklets_with_semantic.pt"
        semantic_metrics = seq_root / "sparse_masklets_with_semantic.metrics.json"
        manifest_path = seq_root / "manifest.json"

        final_summary = None
        if int(args.resume) and not int(args.force):
            final_summary = _valid_final(final_pt, frame_count)
            if final_summary is not None:
                result = {
                    "sequence": seq,
                    "status": "skipped_existing_final",
                    "gpu": str(gpu),
                    "frame_count": int(frame_count),
                    "final_pt": str(final_pt),
                    "final_summary": final_summary,
                    "stages": [],
                    "manifest": str(manifest_path),
                }
                manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return result

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        py = sys.executable
        stages: List[Dict[str, Any]] = []

        videomt_pt = videomt_dir / "sparse_masklets.pt"
        if int(args.force) or not _stage_valid(
            videomt_pt,
            frame_count,
            require_semantic_segmentation=True,
            require_semantic_confidence=True,
        ):
            cmd = [
                py,
                "tools/run_videomt_vspw_sparse.py",
                "--input_video",
                str(image_dir),
                "--output_dir",
                str(videomt_dir),
                "--frames_limit",
                "0",
                "--processing_max_side",
                str(args.processing_max_side),
                "--window_size",
                str(args.videomt_window_size),
                "--drop_thing_labels",
                "0",
                "--render_video",
                "0",
                "--render_contact_sheet",
                "0",
            ]
            stage = _run_cmd(cmd, log_dir / "01_videomt.log", env, REPO_ROOT)
            stages.append({"name": "videomt_l_vspw_w32_thingstuff", **stage})
            if stage["returncode"] != 0:
                raise RuntimeError(f"{seq} VidEoMT failed; see {stage['log_path']}")
        else:
            stages.append({"name": "videomt_l_vspw_w32_thingstuff", "status": "skipped_existing", "output_pt": str(videomt_pt)})

        sam31_pt = sam31_dir / "sparse_masklets.pt"
        if int(args.force) or not _stage_valid(sam31_pt, frame_count):
            cmd = [
                py,
                "run_video_masklet_front_end_v3.py",
                "--input",
                str(image_dir),
                "--dataset_name",
                f"kitti{seq}",
                "--processing_max_side",
                str(args.processing_max_side),
                "--chunk_size",
                "32",
                "--chunk_overlap",
                "3",
                "--thing_detector",
                "yoloe",
                "--yoloe_model",
                "yoloe-26l-seg.pt",
                "--thing_run_stride",
                "1",
                "--tracker_backend",
                "mask_bytetrack",
                "--mot_handoff_merge",
                "1",
                "--mot_overlap_birth_carry_merge",
                "1",
                "--fusion_adjacent_handoff_merge",
                "1",
                "--fusion_v3_short_sign_fragment_merge",
                "1",
                "--fusion_v3_short_sign_fragment_labels",
                "traffic sign",
                "--fusion_v3_mot_stuff_fp_suppression",
                "1",
                "--fusion_v3_detector_only_low_value_filter",
                "1",
                "--fusion_v3_short_tail_motion_fill",
                "0",
                "--reliability_profile",
                "balanced",
                "--vos_backend",
                "sam31_multiplex",
                "--sam31_object_prompt_mode",
                "text_match",
                "--sam31_text_match_labels",
                "car",
                "--sam31_text_match_min_anchor_iou",
                "0.30",
                "--sam31_text_match_output_prob_thresh",
                "0.20",
                "--sam31_refinement_objects_per_session",
                "1",
                "--vos_prompt_only_eligible",
                "1",
                "--vos_target_labels",
                "car,traffic_sign",
                "--vos_skip_far_tiny_objects",
                "1",
                "--vos_max_tracks",
                "40",
                "--vos_use_detector_support_masks",
                "1",
                "--vos_stuff_prompt_gate",
                "1",
                "--stuff_backend",
                "dvisplus_vspw",
                "--dvisplus_vspw_window_size",
                str(args.dvisplus_window_size),
                "--cache_dir",
                "results/stage_c_v3_cache",
                "--cache_mode",
                "readwrite",
                "--experiment_id",
                f"kitti{seq}_preprocess_sam31_textmatch_caronly_signmerge_v1_yoloe26l_dvisplus",
                "--output_pt",
                str(sam31_pt),
                "--output_video",
                str(sam31_dir / "overlay_final.mp4"),
                "--metrics_dir",
                str(sam31_dir / "metrics"),
                "--render_video",
                "0",
                "--export_all_frames",
                "0",
                "--export_chunk_frames",
                "0",
                "--export_high_risk_windows",
                "0",
                "--make_review_package",
                "0",
                "--render_style",
                "clean",
            ]
            stage = _run_cmd(cmd, log_dir / "02_sam31_v3.log", env, REPO_ROOT)
            stages.append({"name": "sam31_v3_textmatch_caronly_signmerge", **stage})
            if stage["returncode"] != 0:
                raise RuntimeError(f"{seq} SAM3.1 v3 failed; see {stage['log_path']}")
        else:
            stages.append({"name": "sam31_v3_textmatch_caronly_signmerge", "status": "skipped_existing", "output_pt": str(sam31_pt)})

        carmerge_pt = carmerge_dir / "sparse_masklets.pt"
        if int(args.force) or not _stage_valid(carmerge_pt, frame_count):
            cmd = [
                py,
                "tools/merge_sparse_thing_tracks.py",
                "--input_pt",
                str(sam31_pt),
                "--input_video",
                str(image_dir),
                "--output_dir",
                str(carmerge_dir),
                "--render_video",
                "0",
                "--render_contact_sheet",
                "0",
                "--fast_metrics",
                "1",
                "--adjacent_flicker_merge_tracks",
                "1",
                "--adjacent_flicker_labels",
                "car",
                "--adjacent_flicker_min_support",
                "2",
                "--adjacent_flicker_min_box_iou",
                "0.02",
                "--adjacent_flicker_max_center_dist",
                "1.20",
                "--adjacent_flicker_min_area_ratio",
                "0.15",
                "--adjacent_flicker_center_scale_floor",
                "24",
                "--adjacent_flicker_single_min_box_iou",
                "0.02",
                "--adjacent_flicker_single_max_center_dist",
                "0.90",
                "--adjacent_flicker_single_min_area_ratio",
                "0.35",
            ]
            stage = _run_cmd(cmd, log_dir / "03_carmerge_adjflicker.log", env, REPO_ROOT)
            stages.append({"name": "sam31_carmerge_adjflicker", **stage})
            if stage["returncode"] != 0:
                raise RuntimeError(f"{seq} carmerge failed; see {stage['log_path']}")
        else:
            stages.append({"name": "sam31_carmerge_adjflicker", "status": "skipped_existing", "output_pt": str(carmerge_pt)})

        fused_pt = fused_dir / "sparse_masklets.pt"
        if int(args.force) or not _stage_valid(fused_pt, frame_count):
            cmd = [
                py,
                "tools/fuse_videomt_sam31_sparse.py",
                "--videomt_pt",
                str(videomt_pt),
                "--sam31_pt",
                str(carmerge_pt),
                "--input_video",
                str(image_dir),
                "--output_dir",
                str(fused_dir),
                "--fusion_role",
                "diagnostic_thingstuff_plus_sam31",
                "--subtract_sam31_from_videomt",
                "1",
                "--sam31_dilate_px",
                "1",
                "--drop_videomt_labels",
                "car",
                "--min_videomt_frames_after_subtract",
                "1",
                "--frames_limit",
                "0",
                "--processing_max_side",
                str(args.processing_max_side),
                "--render_video",
                "0",
                "--render_contact_sheet",
                "0",
            ]
            stage = _run_cmd(cmd, log_dir / "04_fuse_videomt_sam31.log", env, REPO_ROOT)
            stages.append({"name": "fuse_videomt_sam31_dropcar", **stage})
            if stage["returncode"] != 0:
                raise RuntimeError(f"{seq} fusion failed; see {stage['log_path']}")
        else:
            stages.append({"name": "fuse_videomt_sam31_dropcar", "status": "skipped_existing", "output_pt": str(fused_pt)})

        if int(args.force) or _valid_final(final_pt, frame_count) is None:
            cmd = [
                py,
                "tools/add_semantic_segmentation_to_sparse.py",
                "--input_pt",
                str(fused_pt),
                "--output_pt",
                str(final_pt),
                "--confidence_pt",
                str(videomt_pt),
                "--metrics_json",
                str(semantic_metrics),
                "--source",
                "videomt_l_vspw_w32_plus_sam31_textmatch_carmerge_dropcar",
                "--base_source_types",
                "stuff_static,structure_tracked",
                "--thing_source_types",
                "thing_tracked",
                "--label_order",
                "first_seen",
            ]
            stage = _run_cmd(cmd, log_dir / "05_add_semantic_segmentation.log", env, REPO_ROOT)
            stages.append({"name": "add_semantic_segmentation", **stage})
            if stage["returncode"] != 0:
                raise RuntimeError(f"{seq} semantic export failed; see {stage['log_path']}")
        else:
            stages.append({"name": "add_semantic_segmentation", "status": "skipped_existing", "output_pt": str(final_pt)})

        final_summary = _valid_final(final_pt, frame_count)
        if final_summary is None:
            raise RuntimeError(f"{seq} final validation failed: {final_pt}")
        result = {
            "sequence": seq,
            "status": "completed",
            "gpu": str(gpu),
            "frame_count": int(frame_count),
            "final_pt": str(final_pt),
            "final_sha256": "" if int(args.skip_sha256) else _sha256(final_pt),
            "final_summary": final_summary,
            "stages": stages,
            "manifest": str(manifest_path),
        }
        manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        result = {
            "sequence": seq,
            "status": "failed",
            "gpu": str(gpu),
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
            seq_root = output_root / seq
            seq_root.mkdir(parents=True, exist_ok=True)
            (seq_root / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return result
    finally:
        gpu_queue.put(gpu)


def main() -> None:
    args = parse_args()
    sequences = _split_csv(args.sequences)
    gpus = _split_csv(args.gpus)
    if not sequences:
        raise RuntimeError("No sequences requested")
    if not gpus:
        raise RuntimeError("No GPUs requested")

    gpu_queue: "queue.Queue[str]" = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    max_workers = min(int(args.max_parallel), len(gpus), len(sequences))
    start = time.time()
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {executor.submit(_run_sequence, seq, gpu_queue, args): seq for seq in sequences}
        for future in concurrent.futures.as_completed(future_to_seq):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    summary = {
        "status": "completed" if all(r.get("status") in {"completed", "skipped_existing_final"} for r in results) else "partial_or_failed",
        "elapsed_seconds": float(time.time() - start),
        "sequences": sequences,
        "gpus": gpus,
        "max_parallel": int(max_workers),
        "results": sorted(results, key=lambda r: str(r.get("sequence", ""))),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"batch_manifest": str(summary_path), **summary}, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
