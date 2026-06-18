#!/usr/bin/env python3
"""Refresh KITTI semantic confidence maps without re-running SAM or stuff VOS."""

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VidEoMT-only confidence refresh, add confidence to existing "
            "fused semantic PTs, and rebuild Stage-C chunks."
        )
    )
    parser.add_argument("--sequences", default="00,01,02,03,04,05,06,07,08,09,10")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_parallel", type=int, default=8)
    parser.add_argument("--post_max_parallel", type=int, default=2)
    parser.add_argument("--input_root", default="data/kitti/dataset/sequences")
    parser.add_argument("--output_root", default="results/kitti_preprocess")
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--videomt_window_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--chunk_overlap", type=int, default=3)
    parser.add_argument("--force_videomt", type=int, default=1)
    parser.add_argument("--force_semantic", type=int, default=1)
    parser.add_argument("--force_chunks", type=int, default=1)
    parser.add_argument("--skip_sha256", type=int, default=0)
    return parser.parse_args()


def _split_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


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


def _split_into_chunks(total_frames: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total_frames:
        return [(0, total_frames)]
    step = max(int(chunk_size) - int(overlap), 1)
    chunks: List[Tuple[int, int]] = []
    for start in range(0, int(total_frames), step):
        end = min(start + int(chunk_size), int(total_frames))
        chunks.append((start, end))
        if end == int(total_frames):
            break
    return chunks


def _load_semantic_summary(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    semantic = payload.get("semantic_segmentation") or {}
    label_maps = semantic.get("label_maps") if isinstance(semantic, dict) else None
    confidence_maps = semantic.get("confidence_maps") if isinstance(semantic, dict) else None
    out = {
        "path": str(path),
        "format": payload.get("format"),
        "num_frames": int(payload.get("num_frames", -1)),
        "frame_height": int(payload.get("frame_height", -1)),
        "frame_width": int(payload.get("frame_width", -1)),
        "num_tracks": int(len(payload.get("tracks", []))),
        "has_semantic_segmentation": bool(semantic),
        "semantic_format": semantic.get("format", "") if isinstance(semantic, dict) else "",
        "semantic_shape": list(label_maps.shape) if label_maps is not None else [],
        "semantic_dtype": str(label_maps.dtype) if label_maps is not None else "",
        "has_semantic_confidence": confidence_maps is not None,
        "semantic_confidence_shape": list(confidence_maps.shape) if confidence_maps is not None else [],
        "semantic_confidence_dtype": str(confidence_maps.dtype) if confidence_maps is not None else "",
    }
    if confidence_maps is not None:
        confidence_float = confidence_maps.float()
        out.update(
            {
                "semantic_confidence_min": float(confidence_float.min().item()) if confidence_float.numel() else 0.0,
                "semantic_confidence_max": float(confidence_float.max().item()) if confidence_float.numel() else 0.0,
                "semantic_confidence_mean": float(confidence_float.mean().item()) if confidence_float.numel() else 0.0,
            }
        )
    return out


def _valid_confidence_pt(path: Path, expected_frames: int) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        summary = _load_semantic_summary(path)
    except Exception:
        return None
    expected = [
        int(summary["num_frames"]),
        int(summary["frame_height"]),
        int(summary["frame_width"]),
    ]
    if summary["format"] != "sparse_masklets_v1":
        return None
    if int(summary["num_frames"]) != int(expected_frames):
        return None
    if not summary["has_semantic_segmentation"]:
        return None
    if not summary["has_semantic_confidence"]:
        return None
    if summary["semantic_shape"] != expected:
        return None
    if summary["semantic_confidence_shape"] != expected:
        return None
    return summary


def _run_cmd(cmd: Sequence[str], log_path: Path, env: Dict[str, str]) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at: {started_at}\n")
        log.write("cwd: " + str(REPO_ROOT) + "\n")
        log.write("CUDA_VISIBLE_DEVICES: " + str(env.get("CUDA_VISIBLE_DEVICES", "")) + "\n")
        log.write("$ " + " ".join(str(part) for part in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "cmd": [str(x) for x in cmd],
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(time.time() - start),
    }


def _paths(args: argparse.Namespace, seq: str) -> Dict[str, Path]:
    output_root = _resolve(args.output_root)
    seq_root = output_root / seq
    return {
        "seq_root": seq_root,
        "log_dir": seq_root / "logs",
        "videomt_dir": seq_root / "videomt_l_vspw_w32_thingstuff",
        "videomt_pt": seq_root / "videomt_l_vspw_w32_thingstuff" / "sparse_masklets.pt",
        "fused_pt": seq_root / "videomt_sam31_carmerge_dropcar_fusion" / "sparse_masklets.pt",
        "final_pt": seq_root / "sparse_masklets_with_semantic.pt",
        "semantic_metrics": seq_root / "sparse_masklets_with_semantic.metrics.json",
        "cache_dir": seq_root / "stage_c_cache_semantic_chunks",
        "validation_json": seq_root / "confidence_only_validation.json",
        "manifest_json": seq_root / "confidence_only_manifest.json",
    }


def _run_videomt(seq: str, gpu_queue: "queue.Queue[str]", args: argparse.Namespace) -> Dict[str, Any]:
    gpu = gpu_queue.get()
    try:
        input_root = _resolve(args.input_root)
        image_dir = _image_dir(input_root, seq)
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing KITTI image dir: {image_dir}")
        frame_count = _count_frames(image_dir)
        p = _paths(args, seq)
        p["seq_root"].mkdir(parents=True, exist_ok=True)

        existing = None if int(args.force_videomt) else _valid_confidence_pt(p["videomt_pt"], frame_count)
        if existing is not None:
            return {
                "sequence": seq,
                "stage": "videomt_confidence",
                "status": "skipped_existing",
                "gpu": str(gpu),
                "frame_count": int(frame_count),
                "videomt_pt": str(p["videomt_pt"]),
                "summary": existing,
            }

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cmd = [
            sys.executable,
            "tools/run_videomt_vspw_sparse.py",
            "--input_video",
            str(image_dir),
            "--output_dir",
            str(p["videomt_dir"]),
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
        stage = _run_cmd(cmd, p["log_dir"] / "06_videomt_confidence_only.log", env)
        if stage["returncode"] != 0:
            raise RuntimeError(f"VidEoMT confidence failed; see {stage['log_path']}")
        summary = _valid_confidence_pt(p["videomt_pt"], frame_count)
        if summary is None:
            raise RuntimeError(f"VidEoMT output validation failed: {p['videomt_pt']}")
        return {
            "sequence": seq,
            "stage": "videomt_confidence",
            "status": "completed",
            "gpu": str(gpu),
            "frame_count": int(frame_count),
            "videomt_pt": str(p["videomt_pt"]),
            "summary": summary,
            **stage,
        }
    except Exception as exc:
        return {
            "sequence": seq,
            "stage": "videomt_confidence",
            "status": "failed",
            "gpu": str(gpu),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        gpu_queue.put(gpu)


def _validate_chunks(seq: str, args: argparse.Namespace) -> Dict[str, Any]:
    p = _paths(args, seq)
    final = torch.load(p["final_pt"], map_location="cpu", weights_only=False)
    semantic = final.get("semantic_segmentation") or {}
    full_labels = semantic.get("label_maps")
    full_confidence = semantic.get("confidence_maps")
    if full_labels is None or full_confidence is None:
        raise RuntimeError(f"{p['final_pt']} missing semantic label_maps or confidence_maps")
    total_frames = int(final["num_frames"])
    height = int(final["frame_height"])
    width = int(final["frame_width"])
    expected_chunks = _split_into_chunks(total_frames, int(args.chunk_size), int(args.chunk_overlap))

    summary_path = p["cache_dir"] / "conversion_summary.json"
    cache_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(cache_summary.get("chunk_size", -1)) != int(args.chunk_size):
        raise RuntimeError(f"chunk_size mismatch in {summary_path}: {cache_summary.get('chunk_size')}")
    if int(cache_summary.get("chunk_overlap", -1)) != int(args.chunk_overlap):
        raise RuntimeError(f"chunk_overlap mismatch in {summary_path}: {cache_summary.get('chunk_overlap')}")
    if int(cache_summary.get("num_chunks", -1)) != len(expected_chunks):
        raise RuntimeError(f"num_chunks mismatch: got {cache_summary.get('num_chunks')} expected {len(expected_chunks)}")
    if not bool(cache_summary.get("has_semantic_confidence")):
        raise RuntimeError(f"{summary_path} does not report semantic confidence")
    if cache_summary.get("semantic_global_confidence_shape") != list(full_confidence.shape):
        raise RuntimeError("semantic_global_confidence_shape mismatch")

    checked = 0
    max_label_diff = 0
    max_confidence_abs_diff = 0.0
    rows: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(expected_chunks):
        chunk_dir = p["cache_dir"] / f"chunk_{idx:03d}_{start:06d}_{end:06d}"
        chunk_pt = chunk_dir / "masklet.pt"
        if not chunk_pt.exists():
            raise RuntimeError(f"Missing chunk PT: {chunk_pt}")
        chunk = torch.load(chunk_pt, map_location="cpu", weights_only=False)
        csem = chunk.get("semantic_segmentation") or {}
        c_labels = csem.get("label_maps")
        c_confidence = csem.get("confidence_maps")
        if c_labels is None or c_confidence is None:
            raise RuntimeError(f"{chunk_pt} missing semantic label_maps or confidence_maps")
        if int(csem.get("global_start_frame", -1)) != int(start) or int(csem.get("global_end_frame", -1)) != int(end):
            raise RuntimeError(f"{chunk_pt} semantic global range mismatch")
        expected_shape = [int(end - start), int(height), int(width)]
        if list(c_labels.shape) != expected_shape:
            raise RuntimeError(f"{chunk_pt} label shape mismatch: {list(c_labels.shape)} expected {expected_shape}")
        if list(c_confidence.shape) != expected_shape:
            raise RuntimeError(f"{chunk_pt} confidence shape mismatch: {list(c_confidence.shape)} expected {expected_shape}")
        label_diff = int((c_labels != full_labels[start:end]).sum().item())
        confidence_abs_diff = float((c_confidence.float() - full_confidence[start:end].float()).abs().max().item())
        max_label_diff = max(max_label_diff, label_diff)
        max_confidence_abs_diff = max(max_confidence_abs_diff, confidence_abs_diff)
        rows.append(
            {
                "chunk_idx": int(idx),
                "start_frame": int(start),
                "end_frame": int(end),
                "label_diff_pixels": int(label_diff),
                "confidence_max_abs_diff": float(confidence_abs_diff),
            }
        )
        checked += 1

    return {
        "sequence": seq,
        "status": "passed",
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "num_chunks": int(checked),
        "semantic_shape": list(full_labels.shape),
        "confidence_shape": list(full_confidence.shape),
        "max_label_diff_pixels": int(max_label_diff),
        "max_confidence_abs_diff": float(max_confidence_abs_diff),
        "first_chunk": rows[0] if rows else None,
        "last_chunk": rows[-1] if rows else None,
    }


def _run_postprocess(seq: str, args: argparse.Namespace) -> Dict[str, Any]:
    start_all = time.time()
    try:
        input_root = _resolve(args.input_root)
        image_dir = _image_dir(input_root, seq)
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing KITTI image dir: {image_dir}")
        frame_count = _count_frames(image_dir)
        p = _paths(args, seq)
        if not p["fused_pt"].exists():
            raise FileNotFoundError(f"Missing existing fused PT: {p['fused_pt']}")
        if not p["videomt_pt"].exists():
            raise FileNotFoundError(f"Missing VidEoMT confidence PT: {p['videomt_pt']}")

        stages: List[Dict[str, Any]] = []
        final_existing = None if int(args.force_semantic) else _valid_confidence_pt(p["final_pt"], frame_count)
        if final_existing is None:
            cmd = [
                sys.executable,
                "tools/add_semantic_segmentation_to_sparse.py",
                "--input_pt",
                str(p["fused_pt"]),
                "--output_pt",
                str(p["final_pt"]),
                "--confidence_pt",
                str(p["videomt_pt"]),
                "--metrics_json",
                str(p["semantic_metrics"]),
                "--source",
                "videomt_l_vspw_w32_plus_sam31_textmatch_carmerge_dropcar",
                "--base_source_types",
                "stuff_static,structure_tracked",
                "--thing_source_types",
                "thing_tracked",
                "--label_order",
                "first_seen",
                "--overwrite",
                "1",
            ]
            stage = _run_cmd(cmd, p["log_dir"] / "07_add_semantic_confidence_only.log", os.environ.copy())
            stages.append({"name": "add_semantic_confidence", **stage})
            if stage["returncode"] != 0:
                raise RuntimeError(f"semantic confidence export failed; see {stage['log_path']}")
        else:
            stages.append({"name": "add_semantic_confidence", "status": "skipped_existing", "output_pt": str(p["final_pt"])})

        final_summary = _valid_confidence_pt(p["final_pt"], frame_count)
        if final_summary is None:
            raise RuntimeError(f"Final semantic confidence validation failed: {p['final_pt']}")

        cmd = [
            sys.executable,
            "tools/convert_sparse_masklet_to_stage_c_cache.py",
            "--input_pt",
            str(p["final_pt"]),
            "--cache_dir",
            str(p["cache_dir"]),
            "--chunk_size",
            str(args.chunk_size),
            "--chunk_overlap",
            str(args.chunk_overlap),
            "--overwrite",
            "1" if int(args.force_chunks) else "0",
        ]
        stage = _run_cmd(cmd, p["log_dir"] / "08_rechunk_stage_c_confidence_only.log", os.environ.copy())
        stages.append({"name": "convert_stage_c_chunks", **stage})
        if stage["returncode"] != 0:
            raise RuntimeError(f"Stage-C conversion failed; see {stage['log_path']}")

        validation = _validate_chunks(seq, args)
        validation["validated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        p["validation_json"].write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

        result = {
            "sequence": seq,
            "status": "completed",
            "frame_count": int(frame_count),
            "videomt_pt": str(p["videomt_pt"]),
            "fused_input_pt": str(p["fused_pt"]),
            "final_pt": str(p["final_pt"]),
            "cache_dir": str(p["cache_dir"]),
            "semantic_metrics": str(p["semantic_metrics"]),
            "validation_json": str(p["validation_json"]),
            "final_summary": final_summary,
            "chunk_validation": validation,
            "stages": stages,
            "elapsed_seconds": float(time.time() - start_all),
        }
        if not int(args.skip_sha256):
            result["final_sha256"] = _sha256(p["final_pt"])
        p["manifest_json"].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        result = {
            "sequence": seq,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": float(time.time() - start_all),
        }
        try:
            p = _paths(args, seq)
            p["manifest_json"].parent.mkdir(parents=True, exist_ok=True)
            p["manifest_json"].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return result


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

    start = time.time()
    max_workers = min(int(args.max_parallel), len(gpus), len(sequences))
    videomt_results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {executor.submit(_run_videomt, seq, gpu_queue, args): seq for seq in sequences}
        for future in concurrent.futures.as_completed(future_to_seq):
            result = future.result()
            videomt_results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    successful = [str(r["sequence"]) for r in videomt_results if r.get("status") in {"completed", "skipped_existing"}]
    post_results: List[Dict[str, Any]] = []
    post_workers = min(max(int(args.post_max_parallel), 1), len(successful)) if successful else 0
    if post_workers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=post_workers) as executor:
            future_to_seq = {executor.submit(_run_postprocess, seq, args): seq for seq in successful}
            for future in concurrent.futures.as_completed(future_to_seq):
                result = future.result()
                post_results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": (
            "completed"
            if len(successful) == len(sequences) and all(r.get("status") == "completed" for r in post_results)
            else "partial_or_failed"
        ),
        "elapsed_seconds": float(time.time() - start),
        "sequences": sequences,
        "gpus": gpus,
        "max_parallel": int(max_workers),
        "post_max_parallel": int(post_workers),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "force_videomt": bool(int(args.force_videomt)),
        "force_semantic": bool(int(args.force_semantic)),
        "force_chunks": bool(int(args.force_chunks)),
        "videomt_results": sorted(videomt_results, key=lambda r: str(r.get("sequence", ""))),
        "postprocess_results": sorted(post_results, key=lambda r: str(r.get("sequence", ""))),
    }
    summary_path = output_root / "confidence_only_batch_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"batch_manifest": str(summary_path), **summary}, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
