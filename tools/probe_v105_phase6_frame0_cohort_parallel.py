#!/usr/bin/env python3
"""Probe cross-GPU frame0 object-cohort propagation for Stream4D v105.

This tool isolates the current Phase6 bottleneck: propagating the 46 frame0
seed/residual masks through SAM2. It does not decode new births and does not
claim end-to-end pipeline success.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_4dpm_largest_tracking_baseline import (  # noqa: E402
    disjoin_keep_order,
    label_from_id_masks,
    make_numeric_frame_dir,
    make_sheet_grid,
    write_video,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    annotate_frame,
    mask_stats,
    overlay_label,
    read_rgb,
    sha256_file,
)
from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    propagate_new_masks_chunked,
    setup_models,
)
from tools.build_v105_phase5_frozen_birth_replay import (  # noqa: E402
    install_video_feature_bank_patch,
    serializable_feature_bank_summary,
)
from tools.build_v105_phase6_speculative_gap_birth import (  # noqa: E402
    load_frame0_seed_rows,
    make_baseline_args,
    resolve_path,
)


def _load_phase6_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["scene_id", "frame_ids", "config_path", "frame0_seed_meta", "birth_records"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError({"missing_summary_keys": missing, "summary": str(path)})
    return payload


def _build_args_from_summary(summary: dict[str, Any], output_root: Path) -> SimpleNamespace:
    frame_ids = [int(v) for v in summary["frame_ids"]]
    config_path = resolve_path(str(summary["config_path"]))
    stride = int(frame_ids[1] - frame_ids[0]) if len(frame_ids) > 1 else 1
    cli = SimpleNamespace(
        config=str(config_path),
        scene_id=str(summary["scene_id"]),
        rgb_root=None,
        frame_start=int(frame_ids[0]),
        frame_stride=int(stride),
        frame_count=int(len(frame_ids)),
        frame_ids=",".join(str(v) for v in frame_ids),
        output_root=str(output_root),
        seed=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        propagation_chunk_size=0,
    )
    args = make_baseline_args(config_path, cli)
    args.output_root = str(output_root)
    args.offload_video_to_cpu = False
    args.offload_state_to_cpu = False
    args.propagation_chunk_size = 0
    return args


def _frame_paths(args: SimpleNamespace, frame_ids: list[int]) -> list[Path]:
    rgb_root = resolve_path(args.rgb_root) / str(args.scene_id) / "color"
    paths = [rgb_root / f"{int(frame_id)}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:8])
    return paths


def _load_frame0_objects(summary: dict[str, Any], frame_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int, int]:
    frame_ids = [int(v) for v in summary["frame_ids"]]
    first_rgb = read_rgb(frame_paths[0])
    h, w = int(first_rgb.shape[0]), int(first_rgb.shape[1])
    seed_path = resolve_path(str(summary["frame0_seed_meta"]["birth_records_path"]))
    seed_ids, seed_masks, _ = load_frame0_seed_rows(seed_path, frame_ids, str(summary["scene_id"]), h, w)

    residual_rows = [
        row
        for row in summary.get("birth_records", [])
        if int(row.get("chunk_frame_index", -1)) == 0 and str(row.get("source")) == "frame0_residual_repair"
    ]
    residual_rows.sort(key=lambda row: int(row["obj_id"]))
    residual_ids: list[int] = []
    residual_masks: list[np.ndarray] = []
    for row in residual_rows:
        mask_path = resolve_path(str(row["mask_path"]))
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(mask_path)
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        residual_ids.append(int(row["obj_id"]))
        residual_masks.append((image > 0).astype(bool))
    if residual_masks:
        obj_ids = np.concatenate([seed_ids.astype(np.int64), np.asarray(residual_ids, dtype=np.int64)])
        masks = np.concatenate([seed_masks.astype(bool), np.stack(residual_masks, axis=0).astype(bool)], axis=0)
    else:
        obj_ids = seed_ids.astype(np.int64)
        masks = seed_masks.astype(bool)
    order = np.argsort(obj_ids)
    return obj_ids[order].astype(np.int64), masks[order].astype(bool), h, w


def _parse_index_group_spec(spec: str, object_count: int) -> list[int]:
    """Parse compact object-position groups like "0-22+38" or "0:23+38"."""
    indices: list[int] = []
    for raw_token in str(spec).replace(",", "+").split("+"):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", 1)
            start = int(start_text)
            end = int(end_text)
            values = list(range(start, end))
        elif "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            values = list(range(start, end + 1))
        else:
            values = [int(token)]
        for value in values:
            if value < 0 or value >= int(object_count):
                raise ValueError({"index_out_of_range": value, "object_count": int(object_count), "spec": spec})
            indices.append(int(value))
    if not indices:
        raise ValueError({"empty_index_group_spec": spec})
    if len(indices) != len(set(indices)):
        raise ValueError({"duplicate_indices": indices, "spec": spec})
    return indices


def _sanitize_group_name(text: str) -> str:
    safe = []
    for char in str(text):
        safe.append(char if char.isalnum() else "_")
    return "".join(safe).strip("_") or "group"


def _pack_outputs(
    propagated: dict[int, dict[int, np.ndarray]],
    *,
    frame_count: int,
    obj_ids: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    packed_len = (int(h) * int(w) + 7) // 8
    packed = np.zeros((int(frame_count), int(obj_ids.size), int(packed_len)), dtype=np.uint8)
    for frame_idx in range(int(frame_count)):
        frame_outputs = propagated.get(int(frame_idx), {})
        for obj_pos, obj_id in enumerate(obj_ids.tolist()):
            mask = frame_outputs.get(int(obj_id))
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            packed[int(frame_idx), int(obj_pos)] = np.packbits(mask.reshape(-1), bitorder="little")
    return packed


def _unpack_mask(packed: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.unpackbits(packed, bitorder="little", count=int(h) * int(w)).reshape((int(h), int(w))).astype(bool)


def _read_worker_npz(path: Path) -> dict[str, Any]:
    data = np.load(path)
    return {
        "packed_masks": data["packed_masks"],
        "obj_ids": data["obj_ids"].astype(np.int64),
        "frame_ids": data["frame_ids"].astype(np.int64),
        "h": int(data["h"][0]),
        "w": int(data["w"][0]),
    }


def _build_labels_from_packed(parts: list[dict[str, Any]], label_dir: Path, *, empty_ratio: float) -> list[Path]:
    label_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [int(v) for v in parts[0]["frame_ids"].tolist()]
    h, w = int(parts[0]["h"]), int(parts[0]["w"])
    id_to_series: dict[int, np.ndarray] = {}
    for part in parts:
        for pos, obj_id in enumerate(part["obj_ids"].tolist()):
            id_to_series[int(obj_id)] = part["packed_masks"][:, int(pos), :]
    obj_ids = np.asarray(sorted(id_to_series), dtype=np.int64)
    label_paths: list[Path] = []
    for frame_idx, frame_id in enumerate(frame_ids):
        masks = np.stack([_unpack_mask(id_to_series[int(obj_id)][frame_idx], h, w) for obj_id in obj_ids], axis=0)
        disjoint, keep = disjoin_keep_order(masks, h, w, empty_ratio=float(empty_ratio))
        label = label_from_id_masks(obj_ids[keep], disjoint[keep], h, w)
        label_path = label_dir / f"frame_{int(frame_id):06d}.png"
        cv2.imwrite(str(label_path), label)
        label_paths.append(label_path)
    return label_paths


def _compare_label_dirs(candidate_dir: Path, reference_dir: Path) -> dict[str, Any]:
    ref_by_name = {path.name: path for path in reference_dir.glob("*.png")}
    exact = 0
    common = 0
    pixel_diff_total = 0
    fg_ious: list[float] = []
    hard: dict[str, Any] = {}
    worst: tuple[float, str, int, int] | None = None
    hard_names = {"frame_004190.png", "frame_004225.png", "frame_004240.png", "frame_004310.png", "frame_004315.png"}
    for cand_path in sorted(candidate_dir.glob("*.png")):
        ref_path = ref_by_name.get(cand_path.name)
        if ref_path is None:
            continue
        cand = cv2.imread(str(cand_path), cv2.IMREAD_UNCHANGED)
        ref = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
        if cand is None or ref is None:
            continue
        common += 1
        diff = int(np.count_nonzero(cand != ref))
        pixel_diff_total += diff
        exact += int(diff == 0)
        cand_fg = cand > 0
        ref_fg = ref > 0
        union = int(np.count_nonzero(cand_fg | ref_fg))
        inter = int(np.count_nonzero(cand_fg & ref_fg))
        fg_iou = 1.0 if union == 0 else float(inter) / float(union)
        fg_delta = int(np.count_nonzero(cand_fg) - np.count_nonzero(ref_fg))
        fg_ious.append(fg_iou)
        if worst is None or fg_iou < worst[0]:
            worst = (float(fg_iou), cand_path.name, int(diff), int(fg_delta))
        if cand_path.name in hard_names:
            hard[cand_path.name] = {
                "foreground_iou": float(fg_iou),
                "pixel_diff": int(diff),
                "foreground_delta": int(fg_delta),
            }
    return {
        "common_frames": int(common),
        "exact_frames": int(exact),
        "pixel_diff_total": int(pixel_diff_total),
        "mean_foreground_iou": float(np.mean(fg_ious)) if fg_ious else None,
        "min_foreground_iou": float(np.min(fg_ious)) if fg_ious else None,
        "worst_frame": {
            "foreground_iou": float(worst[0]),
            "label": worst[1],
            "pixel_diff": int(worst[2]),
            "foreground_delta": int(worst[3]),
        }
        if worst is not None
        else None,
        "hard_frames": hard,
    }


def _write_overlays(
    label_paths: list[Path],
    frame_paths: list[Path],
    overlay_dir: Path,
    *,
    title: str,
    fps: float,
    sheet_cell_width: int,
) -> tuple[list[str], str]:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = overlay_dir.parent / f"{overlay_dir.name}_sheets"
    video_dir = overlay_dir.parent / f"{overlay_dir.name}_videos"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    overlay_paths: list[Path] = []
    for idx, (label_path, frame_path) in enumerate(zip(label_paths, frame_paths, strict=True)):
        rgb = read_rgb(frame_path)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        stats = mask_stats(label)
        overlay = overlay_label(rgb, label)
        annotated = annotate_frame(
            overlay,
            f"{title} frame {idx:02d} / id {frame_path.stem}",
            [f"ids={stats['visible_id_count']} fg={stats['foreground_ratio']:.3f}"],
        )
        out = overlay_dir / f"frame_{idx:02d}_id_{frame_path.stem}.jpg"
        annotated.save(out, quality=95)
        overlay_paths.append(out)
    sheet_paths: list[str] = []
    for start in range(0, len(overlay_paths), 8):
        part = overlay_paths[start : start + 8]
        end = start + len(part) - 1
        sheet = sheet_dir / f"{title}_frames_{start:02d}_{end:02d}_4x2.jpg"
        make_sheet_grid(part, sheet, int(sheet_cell_width), cols=4)
        sheet_paths.append(str(sheet))
    video_path = video_dir / f"{title}_chunk0.mp4"
    write_video(overlay_paths, video_path, fps=float(fps))
    return sheet_paths, str(video_path)


def _write_compare_frames(serial_dir: Path, parallel_dir: Path, frame_paths: list[Path], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_frame_ids = {4190, 4225, 4240, 4310, 4315}
    paths: list[str] = []
    for frame_path in frame_paths:
        frame_id = int(frame_path.stem)
        if frame_id not in selected_frame_ids:
            continue
        serial_label = cv2.imread(str(serial_dir / f"frame_{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        parallel_label = cv2.imread(str(parallel_dir / f"frame_{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        rgb = read_rgb(frame_path)
        serial_overlay_raw = overlay_label(rgb, serial_label)
        parallel_overlay_raw = overlay_label(rgb, parallel_label)
        serial_overlay = (
            np.asarray(serial_overlay_raw.convert("RGB"))
            if isinstance(serial_overlay_raw, Image.Image)
            else np.asarray(serial_overlay_raw).astype(np.uint8)
        )
        parallel_overlay = (
            np.asarray(parallel_overlay_raw.convert("RGB"))
            if isinstance(parallel_overlay_raw, Image.Image)
            else np.asarray(parallel_overlay_raw).astype(np.uint8)
        )
        serial_fg = serial_label > 0
        parallel_fg = parallel_label > 0
        diff = np.zeros_like(rgb)
        diff[..., 0] = ((parallel_fg & ~serial_fg) * 255).astype(np.uint8)
        diff[..., 1] = ((serial_fg & ~parallel_fg) * 255).astype(np.uint8)
        diff[..., 2] = (((parallel_label != serial_label) & (parallel_fg & serial_fg)) * 255).astype(np.uint8)
        panel = np.concatenate([rgb, serial_overlay, parallel_overlay, diff], axis=1)
        image = Image.fromarray(panel)
        out = out_dir / f"frame_{frame_path.stem}_raw_serial_parallel_diff.jpg"
        image.save(out, quality=95)
        paths.append(str(out))
    return paths


def run_worker(args: argparse.Namespace) -> None:
    import torch

    worker_root = Path(args.output_root)
    worker_root.mkdir(parents=True, exist_ok=True)
    summary = _load_phase6_summary(resolve_path(args.phase6_summary))
    frame_ids = [int(v) for v in summary["frame_ids"]]
    baseline_args = _build_args_from_summary(summary, worker_root)
    frame_paths = _frame_paths(baseline_args, frame_ids)
    all_obj_ids, all_masks, h, w = _load_frame0_objects(summary, frame_paths)
    if str(args.object_indices).strip():
        object_positions = _parse_index_group_spec(str(args.object_indices), int(all_obj_ids.size))
        obj_ids = all_obj_ids[np.asarray(object_positions, dtype=np.int64)]
        masks = all_masks[np.asarray(object_positions, dtype=np.int64)]
        object_selection = {"mode": "index_group", "spec": str(args.object_indices), "positions": object_positions}
    else:
        start, end = [int(v) for v in str(args.object_range).split(":", 1)]
        object_positions = list(range(start, end))
        obj_ids = all_obj_ids[start:end]
        masks = all_masks[start:end]
        object_selection = {"mode": "range", "spec": str(args.object_range), "positions": object_positions}
    if obj_ids.size == 0:
        raise ValueError({"empty_worker_object_range": args.object_range})

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    total_t0 = time.time()
    setup_t0 = time.time()
    models = setup_models(baseline_args)
    setup_sec = time.time() - setup_t0
    tracker_model = models["tracker_model"]
    feature_t0 = time.time()
    feature_info = install_video_feature_bank_patch(
        tracker_model,
        frame_ids=frame_ids,
        frame_paths=frame_paths,
        storage_device=str(args.video_feature_bank_storage_device),
        video_gpu_hot_window=int(args.video_gpu_hot_window),
    )
    feature_install_sec = time.time() - feature_t0
    video_dir = make_numeric_frame_dir(frame_paths, worker_root)
    template_t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        video_state_template = tracker_model.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
    video_state_template["cached_features"] = {}
    template_init_sec = time.time() - template_t0
    propagation_t0 = time.time()
    chunk_records: list[dict[str, Any]] = []
    propagated = propagate_new_masks_chunked(
        tracker_model,
        tracker=str(baseline_args.tracker_backend),
        video_dir=video_dir,
        seed_frame=0,
        obj_ids=obj_ids,
        masks=masks,
        total_frames=int(len(frame_ids)),
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        chunk_size=0,
        feature_bank_frame_offset=0,
        video_state_template=video_state_template,
        chunk_runtime_records=chunk_records,
    )
    propagation_sec = time.time() - propagation_t0
    pack_t0 = time.time()
    packed = _pack_outputs(propagated, frame_count=len(frame_ids), obj_ids=obj_ids, h=h, w=w)
    npz_path = worker_root / "packed_masks.npz"
    np.savez(
        npz_path,
        packed_masks=packed,
        obj_ids=obj_ids.astype(np.int64),
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        h=np.asarray([h], dtype=np.int64),
        w=np.asarray([w], dtype=np.int64),
    )
    pack_write_sec = time.time() - pack_t0
    total_sec = time.time() - total_t0
    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    worker_summary = {
        "schema_version": "stream4d_v105_frame0_cohort_worker_v1",
        "phase6_summary": str(resolve_path(args.phase6_summary)),
        "phase6_summary_sha256": sha256_file(resolve_path(args.phase6_summary)),
        "scene_id": str(summary["scene_id"]),
        "frame_ids": frame_ids,
        "object_range": str(args.object_range),
        "object_indices": str(args.object_indices),
        "object_selection": object_selection,
        "object_count": int(obj_ids.size),
        "obj_ids": [int(v) for v in obj_ids.tolist()],
        "setup_sec": float(setup_sec),
        "feature_bank_install_sec": float(feature_install_sec),
        "video_state_template_init_sec": float(template_init_sec),
        "propagation_sec": float(propagation_sec),
        "propagation_chunk_runtime_records": chunk_records,
        "pack_write_sec": float(pack_write_sec),
        "total_worker_sec": float(total_sec),
        "peak_cuda_memory_mb": float(peak_mb),
        "packed_npz": str(npz_path),
        "video_feature_bank": serializable_feature_bank_summary(feature_info),
    }
    summary_path = worker_root / "worker_summary.json"
    summary_path.write_text(json.dumps(worker_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"worker_summary": str(summary_path), "packed_npz": str(npz_path)}, ensure_ascii=True), flush=True)


def _launch_worker(
    *,
    gpu: str,
    object_range: str,
    object_indices: str,
    output_root: Path,
    phase6_summary: Path,
    log_path: Path,
    storage_device: str,
    hot_window: int,
) -> subprocess.Popen[Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--phase6-summary",
        str(phase6_summary),
        "--object-range",
        object_range,
        "--object-indices",
        object_indices,
        "--output-root",
        str(output_root),
        "--video-feature-bank-storage-device",
        str(storage_device),
        "--video-gpu-hot-window",
        str(hot_window),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    log_fh = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)


def run_coordinator(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase6_summary = resolve_path(args.phase6_summary)
    summary = _load_phase6_summary(phase6_summary)
    baseline_args = _build_args_from_summary(summary, output_root / "_coord")
    frame_ids = [int(v) for v in summary["frame_ids"]]
    frame_paths = _frame_paths(baseline_args, frame_ids)
    all_obj_ids, _, h, w = _load_frame0_objects(summary, frame_paths)
    splits = [part.strip() for part in str(args.parallel_splits).split(",") if part.strip()]
    gpus = [part.strip() for part in str(args.gpus).split(",") if part.strip()]
    index_groups = [part.strip() for part in str(args.parallel_index_groups).split(";") if part.strip()]
    if index_groups:
        if len(index_groups) != len(gpus):
            raise ValueError({"parallel_index_groups": index_groups, "gpus": gpus, "reason": "counts must match"})
        expected = set(range(int(all_obj_ids.size)))
        seen: list[int] = []
        for group in index_groups:
            seen.extend(_parse_index_group_spec(group, int(all_obj_ids.size)))
        if set(seen) != expected or len(seen) != len(expected):
            raise ValueError(
                {
                    "parallel_index_groups": index_groups,
                    "reason": "groups must cover every object position exactly once",
                    "missing": sorted(expected - set(seen)),
                    "duplicates": sorted({value for value in seen if seen.count(value) > 1}),
                }
            )
        worker_specs = [("0:0", group) for group in index_groups]
    else:
        if len(splits) != len(gpus):
            raise ValueError({"parallel_splits": splits, "gpus": gpus, "reason": "counts must match"})
        worker_specs = [(split, "") for split in splits]

    serial_summary: dict[str, Any] | None = None
    serial_npz: Path | None = None
    if bool(args.run_serial_reference):
        serial_root = output_root / "serial_full"
        serial_log = output_root / "logs" / "serial_full_gpu.log"
        serial_t0 = time.time()
        proc = _launch_worker(
            gpu=str(gpus[0]),
            object_range=f"0:{int(all_obj_ids.size)}",
            object_indices="",
            output_root=serial_root,
            phase6_summary=phase6_summary,
            log_path=serial_log,
            storage_device=str(args.video_feature_bank_storage_device),
            hot_window=int(args.video_gpu_hot_window),
        )
        rc = proc.wait()
        serial_wall_sec = time.time() - serial_t0
        if rc != 0:
            raise RuntimeError({"serial_worker_failed": int(rc), "log": str(serial_log)})
        serial_summary_path = serial_root / "worker_summary.json"
        serial_summary = json.loads(serial_summary_path.read_text(encoding="utf-8"))
        serial_summary["coordinator_wall_sec"] = float(serial_wall_sec)
        serial_summary["log"] = str(serial_log)
        serial_npz = serial_root / "packed_masks.npz"

    parallel_t0 = time.time()
    processes: list[tuple[str, str, str, Path, Path, subprocess.Popen[Any]]] = []
    for gpu, (object_range, object_indices) in zip(gpus, worker_specs, strict=True):
        object_name = _sanitize_group_name(object_indices if object_indices else object_range)
        worker_root = output_root / f"parallel_gpu{gpu}_objs_{object_name}"
        log = output_root / "logs" / f"parallel_gpu{gpu}_objs_{object_name}.log"
        proc = _launch_worker(
            gpu=gpu,
            object_range=object_range,
            object_indices=object_indices,
            output_root=worker_root,
            phase6_summary=phase6_summary,
            log_path=log,
            storage_device=str(args.video_feature_bank_storage_device),
            hot_window=int(args.video_gpu_hot_window),
        )
        processes.append((gpu, object_range, object_indices, worker_root, log, proc))
    parallel_rows: list[dict[str, Any]] = []
    for gpu, object_range, object_indices, worker_root, log, proc in processes:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                {
                    "parallel_worker_failed": int(rc),
                    "gpu": gpu,
                    "range": object_range,
                    "indices": object_indices,
                    "log": str(log),
                }
            )
        worker_summary_path = worker_root / "worker_summary.json"
        row = json.loads(worker_summary_path.read_text(encoding="utf-8"))
        row["gpu"] = str(gpu)
        row["log"] = str(log)
        parallel_rows.append(row)
    parallel_wall_sec = time.time() - parallel_t0

    serial_parts: list[dict[str, Any]] = []
    if serial_npz is not None:
        serial_parts = [_read_worker_npz(serial_npz)]
        _build_labels_from_packed(serial_parts, output_root / "labels_serial_full", empty_ratio=float(args.empty_ratio))
    parallel_parts = [_read_worker_npz(Path(row["packed_npz"])) for row in parallel_rows]
    _build_labels_from_packed(parallel_parts, output_root / "labels_parallel_merged", empty_ratio=float(args.empty_ratio))

    compare: dict[str, Any] | None = None
    reference_label_dir = Path(args.reference_label_dir) if str(args.reference_label_dir) else None
    if serial_parts:
        compare = _compare_label_dirs(output_root / "labels_parallel_merged", output_root / "labels_serial_full")
    elif reference_label_dir is not None:
        compare = _compare_label_dirs(output_root / "labels_parallel_merged", reference_label_dir)
    parallel_sheets, parallel_video = _write_overlays(
        sorted((output_root / "labels_parallel_merged").glob("*.png")),
        frame_paths,
        output_root / "overlays_parallel_merged",
        title="frame0_parallel_merged",
        fps=float(args.fps),
        sheet_cell_width=int(args.sheet_cell_width),
    )
    serial_sheets: list[str] = []
    serial_video: str | None = None
    compare_frames: list[str] = []
    if serial_parts:
        serial_sheets, serial_video = _write_overlays(
            sorted((output_root / "labels_serial_full").glob("*.png")),
            frame_paths,
            output_root / "overlays_serial_full",
            title="frame0_serial_full",
            fps=float(args.fps),
            sheet_cell_width=int(args.sheet_cell_width),
        )
        compare_frames = _write_compare_frames(
            output_root / "labels_serial_full",
            output_root / "labels_parallel_merged",
            frame_paths,
            output_root / "compare_frames",
        )
    elif reference_label_dir is not None:
        compare_frames = _write_compare_frames(
            reference_label_dir,
            output_root / "labels_parallel_merged",
            frame_paths,
            output_root / "compare_frames",
        )

    summary_out = {
        "schema_version": "stream4d_v105_frame0_cohort_parallel_probe_v1",
        "phase6_summary": str(phase6_summary),
        "phase6_summary_sha256": sha256_file(phase6_summary),
        "scene_id": str(summary["scene_id"]),
        "frame_ids": frame_ids,
        "frame_count": int(len(frame_ids)),
        "frame0_object_count": int(all_obj_ids.size),
        "frame0_obj_ids": [int(v) for v in all_obj_ids.tolist()],
        "h": int(h),
        "w": int(w),
        "gpus": gpus,
        "parallel_splits": splits,
        "parallel_index_groups": index_groups,
        "serial_reference": serial_summary,
        "reference_serial_worker_summary": str(args.reference_serial_worker_summary)
        if str(args.reference_serial_worker_summary)
        else None,
        "reference_label_dir": str(reference_label_dir) if reference_label_dir is not None else None,
        "parallel_workers": parallel_rows,
        "parallel_coordinator_wall_sec": float(parallel_wall_sec),
        "parallel_worker_total_critical_path_sec": float(max(float(row["total_worker_sec"]) for row in parallel_rows)),
        "parallel_worker_propagation_critical_path_sec": float(max(float(row["propagation_sec"]) for row in parallel_rows)),
        "parallel_worker_peak_cuda_memory_mb_max": float(max(float(row["peak_cuda_memory_mb"]) for row in parallel_rows)),
        "serial_vs_parallel_label_compare": compare,
        "parallel_sheets": parallel_sheets,
        "parallel_video": parallel_video,
        "serial_sheets": serial_sheets,
        "serial_video": serial_video,
        "compare_frame_paths": compare_frames,
        "interpretation_boundary": (
            "This is a frame0 propagation-only probe. It excludes later Phase6 birth decode, "
            "later birth propagation, reconciliation, and cold unified-runner stages."
        ),
    }
    speed_reference = serial_summary
    if speed_reference is None and str(args.reference_serial_worker_summary):
        speed_reference_path = Path(args.reference_serial_worker_summary)
        speed_reference = json.loads(speed_reference_path.read_text(encoding="utf-8"))
    if speed_reference:
        serial_prop = float(speed_reference["propagation_sec"])
        parallel_prop_cp = float(summary_out["parallel_worker_propagation_critical_path_sec"])
        serial_total = float(speed_reference["total_worker_sec"])
        parallel_wall = float(summary_out["parallel_coordinator_wall_sec"])
        summary_out["parallel_speedup_prop_critical_path_vs_serial"] = (
            float(serial_prop / parallel_prop_cp) if parallel_prop_cp > 0 else None
        )
        summary_out["parallel_speedup_cold_worker_wall_vs_serial"] = (
            float(serial_total / parallel_wall) if parallel_wall > 0 else None
        )
    summary_path = output_root / "frame0_cohort_parallel_probe_summary.json"
    summary_path.write_text(json.dumps(summary_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "parallel_video": parallel_video, "parallel_sheets": parallel_sheets}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["coordinator", "worker"], default="coordinator")
    parser.add_argument("--phase6-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--object-range", default="0:0")
    parser.add_argument("--object-indices", default="")
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--parallel-splits", default="0:23,23:46")
    parser.add_argument("--parallel-index-groups", default="")
    parser.add_argument("--run-serial-reference", action="store_true", default=False)
    parser.add_argument("--reference-label-dir", default="")
    parser.add_argument("--reference-serial-worker-summary", default="")
    parser.add_argument("--video-feature-bank-storage-device", default="cuda")
    parser.add_argument("--video-gpu-hot-window", type=int, default=1)
    parser.add_argument("--empty-ratio", type=float, default=0.001)
    parser.add_argument("--sheet-cell-width", type=int, default=520)
    parser.add_argument("--fps", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "worker":
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
