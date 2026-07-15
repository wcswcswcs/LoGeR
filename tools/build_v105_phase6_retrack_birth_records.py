#!/usr/bin/env python3
"""Retrack frozen v105 Phase6 birth records with one SAM2 video state.

This is a scheduler probe. It does not discover new births or decode prompts;
it tests whether already accepted birth masks can be added as conditioning
frames before a single SAM2 propagation pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_gapadaptive_sam2.generated.yaml"

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
    parse_frame_ids,
    read_rgb,
    sha256_file,
)
from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    add_masks_to_stream_state,
    load_config,
    make_args,
    setup_models,
)
from tools.build_v105_phase5_frozen_birth_replay import (  # noqa: E402
    empty_feature_bank_summary,
    install_video_feature_bank_patch,
    serializable_feature_bank_summary,
)
from tools.build_v105_phase6_speculative_gap_birth import load_gray_bool, resolve_path  # noqa: E402


def make_baseline_args(config_path: Path, cli: argparse.Namespace) -> SimpleNamespace:
    config = load_config(config_path)
    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=cli.scene_id,
        rgb_root=cli.rgb_root,
        frame_start=cli.frame_start,
        frame_stride=cli.frame_stride,
        frame_count=cli.frame_count,
        frame_ids=cli.frame_ids,
        output_root=cli.output_root,
        seed=cli.seed,
        birth_dump_dir="",
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(cli.output_root)
    args.offload_video_to_cpu = bool(cli.offload_video_to_cpu)
    args.offload_state_to_cpu = bool(cli.offload_state_to_cpu)
    return args


def load_phase6_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "birth_records" not in payload:
        raise ValueError(f"phase6 summary has no birth_records: {path}")
    return payload


def load_seed_rows_from_phase6_summary(phase6_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    seed_meta = phase6_summary.get("frame0_seed_meta", {})
    if not isinstance(seed_meta, dict):
        return [], None
    seed_path_text = seed_meta.get("birth_records_path")
    if not seed_path_text:
        return [], None
    seed_path = resolve_path(str(seed_path_text))
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("rows", [])
        if str(row.get("source")) == "frame0_seed"
    ]
    return rows, str(seed_path)


def load_birth_record_masks(
    birth_records: list[dict[str, Any]],
    *,
    scene_id: str,
    frame_ids: list[int],
    h: int,
    w: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    frame_id_to_idx = {int(frame_id): idx for idx, frame_id in enumerate(frame_ids)}
    grouped: dict[int, list[tuple[int, np.ndarray, dict[str, Any]]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    for row in birth_records:
        if str(row.get("scene_id")) != str(scene_id):
            skipped.append({"reason": "scene_mismatch", "row": row})
            continue
        frame_id = int(row.get("frame_id", -1))
        if frame_id not in frame_id_to_idx:
            skipped.append({"reason": "frame_not_requested", "row": row})
            continue
        obj_id = int(row["obj_id"])
        mask_path = resolve_path(str(row["mask_path"]))
        mask = load_gray_bool(mask_path, h, w)
        grouped[int(frame_id_to_idx[frame_id])].append((obj_id, mask, row))
    out: dict[int, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for chunk_idx, rows in grouped.items():
        rows.sort(key=lambda item: int(item[0]))
        ids = np.asarray([int(item[0]) for item in rows], dtype=np.int64)
        masks = np.stack([item[1].astype(bool) for item in rows], axis=0).astype(bool)
        out[int(chunk_idx)] = (ids, masks, [item[2] for item in rows])
    if skipped:
        out[-1] = (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, h, w), dtype=bool),
            skipped[:64],
        )
    return out


def read_reference_label_paths(summary_path: Path | None) -> dict[int, Path]:
    if summary_path is None:
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    labels: dict[int, Path] = {}
    for row in payload.get("records", []):
        frame_id = int(row["frame_id"])
        label_path = resolve_path(str(row["label_path"]))
        labels[frame_id] = label_path
    return labels


def load_label(path: Path, h: int, w: int) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    if label.shape[:2] != (h, w):
        label = cv2.resize(label.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.uint16, copy=False)


def compare_label(label: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    if label.shape != ref.shape:
        raise ValueError(f"label shape mismatch: {label.shape} vs {ref.shape}")
    pred_fg = label > 0
    ref_fg = ref > 0
    inter = int(np.count_nonzero(pred_fg & ref_fg))
    union = int(np.count_nonzero(pred_fg | ref_fg))
    return {
        "exact_equal": bool(np.array_equal(label, ref)),
        "pixel_diff_count": int(np.count_nonzero(label != ref)),
        "pixel_equal_ratio": float(np.mean(label == ref)),
        "foreground_iou": float(inter) / float(union) if union else 1.0,
    }


def run(cli: argparse.Namespace) -> None:
    import torch

    config_path = resolve_path(cli.config)
    args = make_baseline_args(config_path, cli)
    args.scene_id = str(cli.scene_id or args.scene_id)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = resolve_path(args.rgb_root) / args.scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])

    output_root = resolve_path(cli.output_root)
    label_dir = output_root / "labels"
    overlay_dir = output_root / "overlays"
    sheet_dir = output_root / "sheets"
    video_dir_out = output_root / "videos"
    for directory in (label_dir, overlay_dir, sheet_dir, video_dir_out):
        directory.mkdir(parents=True, exist_ok=True)

    rgbs = [read_rgb(path) for path in frame_paths]
    h, w = rgbs[0].shape[:2]
    video_dir = make_numeric_frame_dir(frame_paths, output_root)
    phase6_summary_path = resolve_path(cli.phase6_summary)
    phase6_summary = load_phase6_summary(phase6_summary_path)
    seed_rows, seed_birth_records_path = load_seed_rows_from_phase6_summary(phase6_summary)
    combined_birth_records = list(seed_rows) + list(phase6_summary.get("birth_records", []))
    birth_groups = load_birth_record_masks(
        combined_birth_records,
        scene_id=args.scene_id,
        frame_ids=frame_ids,
        h=h,
        w=w,
    )
    skipped_records = birth_groups.pop(-1, (np.zeros((0,), dtype=np.int64), np.zeros((0, h, w), dtype=bool), []))[2]
    reference_summary_path = resolve_path(cli.reference_summary) if str(cli.reference_summary).strip() else None
    reference_labels = read_reference_label_paths(reference_summary_path)

    t_setup = time.time()
    models = setup_models(args)
    setup_sec = time.time() - t_setup
    tracker_model = models["tracker_model"]

    feature_bank_info: dict[str, Any] = empty_feature_bank_summary()
    total_t0 = time.time()
    if bool(cli.use_video_feature_bank):
        feature_bank_info = install_video_feature_bank_patch(
            tracker_model,
            frame_ids=frame_ids,
            frame_paths=frame_paths,
            storage_device=str(cli.video_feature_bank_storage_device),
            video_gpu_hot_window=int(cli.video_gpu_hot_window),
        )

    all_obj_ids = sorted(
        {
            int(obj_id)
            for obj_ids, _masks, _rows in birth_groups.values()
            for obj_id in obj_ids.tolist()
        }
    )
    microbatch_size = int(cli.object_microbatch_size)
    if microbatch_size <= 0:
        object_batches = [all_obj_ids]
    else:
        object_batches = [
            all_obj_ids[start : start + microbatch_size]
            for start in range(0, len(all_obj_ids), microbatch_size)
        ]
    if not object_batches:
        object_batches = [[]]

    add_records: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    init_sec = 0.0
    add_total_sec = 0.0
    propagate_sec = 0.0
    frame_output_lists: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for batch_index, batch_obj_ids in enumerate(object_batches):
        batch_obj_id_set = {int(v) for v in batch_obj_ids}
        batch_birth_groups: dict[int, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
        for chunk_idx, (obj_ids, masks, rows) in birth_groups.items():
            keep = np.asarray([int(v) in batch_obj_id_set for v in obj_ids.tolist()], dtype=bool)
            if not np.any(keep):
                continue
            kept_ids = obj_ids[keep]
            kept_masks = masks[keep]
            kept_rows = [row for row, flag in zip(rows, keep.tolist(), strict=False) if bool(flag)]
            batch_birth_groups[int(chunk_idx)] = (kept_ids, kept_masks, kept_rows)

        init_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = tracker_model.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                async_loading_frames=False,
            )
        batch_init_sec = time.time() - init_t0
        init_sec += batch_init_sec

        batch_add_sec = 0.0
        for chunk_idx in sorted(batch_birth_groups):
            obj_ids, masks, rows = batch_birth_groups[int(chunk_idx)]
            add_t0 = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                add_masks_to_stream_state(
                    tracker_model,
                    state,
                    tracker=str(args.tracker_backend),
                    frame_idx=int(chunk_idx),
                    obj_ids=obj_ids,
                    masks=masks,
                )
            add_sec = time.time() - add_t0
            add_total_sec += add_sec
            batch_add_sec += add_sec
            add_records.append(
                {
                    "batch_index": int(batch_index),
                    "chunk_frame_index": int(chunk_idx),
                    "frame_id": int(frame_ids[int(chunk_idx)]),
                    "object_count": int(obj_ids.size),
                    "obj_id_min": int(obj_ids.min()) if obj_ids.size else None,
                    "obj_id_max": int(obj_ids.max()) if obj_ids.size else None,
                    "add_runtime_sec": float(add_sec),
                    "source_record_count": int(len(rows)),
                }
            )

        propagate_t0 = time.time()
        batch_output_count = 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in tracker_model.propagate_in_video(state):
                masks = (out_mask_logits > 0.0).detach().cpu().numpy().squeeze(1).astype(bool)
                ids = np.asarray([int(v) for v in out_obj_ids], dtype=np.int64)
                for obj_id, mask in zip(ids.tolist(), masks, strict=False):
                    frame_output_lists[int(out_frame_idx)].append((int(obj_id), mask.astype(bool)))
                    batch_output_count += 1
        batch_propagate_sec = time.time() - propagate_t0
        propagate_sec += batch_propagate_sec

        try:
            tracker_model.reset_state(state)
        except Exception:
            pass
        try:
            state.clear()
        except Exception:
            pass
        torch.cuda.empty_cache()

        batch_records.append(
            {
                "batch_index": int(batch_index),
                "object_count": int(len(batch_obj_ids)),
                "obj_id_min": int(min(batch_obj_ids)) if batch_obj_ids else None,
                "obj_id_max": int(max(batch_obj_ids)) if batch_obj_ids else None,
                "birth_group_count": int(len(batch_birth_groups)),
                "state_init_runtime_sec": float(batch_init_sec),
                "add_masks_runtime_sec": float(batch_add_sec),
                "propagate_runtime_sec": float(batch_propagate_sec),
                "frame_output_mask_count": int(batch_output_count),
            }
        )

    overlay_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    exact_equal_count = 0
    pixel_diff_total = 0
    for chunk_idx, frame_id in enumerate(frame_ids):
        output_items = sorted(frame_output_lists.get(int(chunk_idx), []), key=lambda item: int(item[0]))
        if output_items:
            ids_pre = np.asarray([int(item[0]) for item in output_items], dtype=np.int64)
            masks_pre = np.stack([item[1].astype(bool) for item in output_items], axis=0)
        else:
            ids_pre = np.zeros((0,), dtype=np.int64)
            masks_pre = np.zeros((0, h, w), dtype=bool)
        if masks_pre.size:
            masks_all, keep = disjoin_keep_order(masks_pre.astype(bool), h, w, empty_ratio=float(args.empty_ratio))
            masks_out = masks_all[keep]
            ids_out = ids_pre[keep]
        else:
            masks_out = np.zeros((0, h, w), dtype=bool)
            ids_out = np.zeros((0,), dtype=np.int64)
        label = label_from_id_masks(ids_out, masks_out, h, w)
        label_path = label_dir / f"frame_{int(frame_id):06d}.png"
        cv2.imwrite(str(label_path), label)
        stats = mask_stats(label)
        ref_compare = None
        ref_label_path = reference_labels.get(int(frame_id))
        if ref_label_path is not None and ref_label_path.exists():
            ref_label = load_label(ref_label_path, h, w)
            ref_compare = compare_label(label, ref_label)
            exact_equal_count += int(bool(ref_compare["exact_equal"]))
            pixel_diff_total += int(ref_compare["pixel_diff_count"])
        overlay = overlay_label(rgbs[int(chunk_idx)], label)
        annotated = annotate_frame(
            overlay,
            f"phase6 retrack frozen births frame {chunk_idx:02d} / id {int(frame_id)}",
            [
                f"ids={stats['visible_id_count']} fg={float(stats['foreground_ratio']):.4f}",
                f"ref_equal={ref_compare['exact_equal'] if ref_compare else 'n/a'}",
            ],
        )
        overlay_path = overlay_dir / f"frame_{int(chunk_idx):02d}_id_{int(frame_id):06d}.jpg"
        annotated.save(overlay_path, quality=95)
        overlay_paths.append(overlay_path)
        records.append(
            {
                "chunk_frame_index": int(chunk_idx),
                "frame_id": int(frame_id),
                "label_path": str(label_path),
                "overlay_path": str(overlay_path),
                "output_object_id_count": int(ids_out.size),
                "visible_id_count": int(stats["visible_id_count"]),
                "foreground_ratio": float(stats["foreground_ratio"]),
                "reference_label_path": str(ref_label_path) if ref_label_path is not None else None,
                "reference_compare": ref_compare,
            }
        )

    sheet_paths: list[str] = []
    for start in range(0, len(overlay_paths), 8):
        part = overlay_paths[start : start + 8]
        end = start + len(part) - 1
        sheet_path = sheet_dir / f"phase6_retrack_birth_records_{args.scene_id}_frames_{start:02d}_{end:02d}_4x2.jpg"
        make_sheet_grid(part, sheet_path, int(args.sheet_cell_width), cols=4)
        sheet_paths.append(str(sheet_path))
    video_path = video_dir_out / f"phase6_retrack_birth_records_{args.scene_id}_chunk0.mp4"
    write_video(overlay_paths, video_path, fps=float(args.fps))

    total_sec = time.time() - total_t0
    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    summary = {
        "schema_version": "stream4d_v105_phase6_retrack_birth_records_summary_v1",
        "scene_id": str(args.scene_id),
        "config_path": str(config_path),
        "phase6_summary": str(phase6_summary_path),
        "phase6_summary_sha256": sha256_file(phase6_summary_path),
        "seed_birth_records_path": seed_birth_records_path,
        "seed_birth_record_count": int(len(seed_rows)),
        "reference_summary": str(reference_summary_path) if reference_summary_path is not None else None,
        "frame_ids": [int(v) for v in frame_ids],
        "frame_count": int(len(frame_ids)),
        "setup_sec": float(setup_sec),
        "total_runtime_sec": float(total_sec),
        "stream_state_init_runtime_sec": float(init_sec),
        "add_masks_total_runtime_sec": float(add_total_sec),
        "single_state_propagate_runtime_sec": float(propagate_sec),
        "object_microbatch_size": int(microbatch_size),
        "object_state_count": int(len(object_batches)),
        "peak_cuda_memory_mb": float(peak_mb),
        "use_video_feature_bank": bool(cli.use_video_feature_bank),
        "video_feature_bank": serializable_feature_bank_summary(feature_bank_info),
        "birth_group_count": int(len(birth_groups)),
        "birth_object_count": int(sum(int(v[0].size) for v in birth_groups.values())),
        "combined_birth_record_count": int(len(combined_birth_records)),
        "skipped_birth_record_count": int(len(skipped_records)),
        "skipped_birth_records_first64": skipped_records[:64],
        "add_records": add_records,
        "batch_records": batch_records,
        "reference_exact_equal_frame_count": int(exact_equal_count),
        "reference_compared_frame_count": int(sum(1 for row in records if row["reference_compare"] is not None)),
        "reference_pixel_diff_total": int(pixel_diff_total),
        "visual_gate_status": "manual_review_required",
        "records": records,
        "sheet_paths": sheet_paths,
        "video_path": str(video_path),
    }
    summary_path = output_root / "phase6_retrack_birth_records_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "video": str(video_path), "sheets": sheet_paths}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--phase6-summary", required=True)
    parser.add_argument("--reference-summary", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rgb-root", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--frame-ids", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=False)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=False)
    parser.add_argument("--use-video-feature-bank", action="store_true")
    parser.add_argument("--video-feature-bank-storage-device", default="cuda")
    parser.add_argument("--video-gpu-hot-window", type=int, default=0)
    parser.add_argument(
        "--object-microbatch-size",
        type=int,
        default=0,
        help="0 means all objects in one state. Positive values split sorted object ids into multiple SAM2 states.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
