#!/usr/bin/env python3
"""Audit SAM2 object-memory API semantics for Stream4D v107 Phase0/Phase2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base  # noqa: E402
from tools.audit_v106_sam2_rolling_state import (  # noqa: E402
    _rolling_add_frame,
    install_rolling_state_support,
    reset_rolling_stats,
)


SAM2_VIDEO_PREDICTOR = REPO_ROOT / "Grounded-SAM-2/sam2/sam2_video_predictor.py"
V106_ROLLING_ADAPTER = REPO_ROOT / "tools/audit_v106_sam2_rolling_state.py"
V105_BASELINE_LOOP = REPO_ROOT / "tools/audit_v105_baseline_x_sam2_twostage_tracking.py"
CURRENT_BEST_CONFIG = (
    REPO_ROOT
    / "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml"
)


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def source_contract() -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v107_sam2_memory_source_contract_v1",
        "source_files": {
            "sam2_video_predictor": {
                "path": str(SAM2_VIDEO_PREDICTOR.relative_to(REPO_ROOT)),
                "sha256": sha256_file(SAM2_VIDEO_PREDICTOR),
            },
            "v106_rolling_adapter": {
                "path": str(V106_ROLLING_ADAPTER.relative_to(REPO_ROOT)),
                "sha256": sha256_file(V106_ROLLING_ADAPTER),
            },
            "v105_baseline_loop": {
                "path": str(V105_BASELINE_LOOP.relative_to(REPO_ROOT)),
                "sha256": sha256_file(V105_BASELINE_LOOP),
            },
        },
        "inspected_code_points": [
            {
                "file": "Grounded-SAM-2/sam2/sam2_video_predictor.py",
                "lines": "142-174",
                "finding": "SAM2 _obj_id_to_idx refuses new object IDs after tracking_has_started=True.",
            },
            {
                "file": "Grounded-SAM-2/sam2/sam2_video_predictor.py",
                "lines": "333-414",
                "finding": "add_new_mask records mask input, runs single-object inference, and consolidates temp output across all objects.",
            },
            {
                "file": "Grounded-SAM-2/sam2/sam2_video_predictor.py",
                "lines": "1181-1292",
                "finding": "remove_object removes mappings, reindexes remaining objects, slices packed state tensors, and resets state if removing the only object.",
            },
            {
                "file": "tools/audit_v105_baseline_x_sam2_twostage_tracking.py",
                "lines": "711-764",
                "finding": "v106 permits post-start births by temporarily setting tracking_has_started=False, then reconsolidating prior outputs when object count grows.",
            },
        ],
        "source_contract": {
            "public_api_direct_new_id_after_tracking_allowed": False,
            "remove_object_reindexes_runtime_object_indices": True,
            "remove_last_object_resets_state": True,
            "post_start_add_requires_v106_workaround_or_reset": True,
            "stable_global_id_mapping_required": True,
            "same_numeric_runtime_id_readd_status": "requires_probe",
            "batch_add_reconsolidation_status": "requires_phase2_probe",
        },
        "phase0_interpretation": (
            "The source audit is enough to forbid direct live scheduler changes before Phase2. "
            "A stable global-id layer must hide SAM2 runtime index shifts."
        ),
    }


def make_args(config_path: Path, scene_id: str, frame_start: int, frame_stride: int, frame_count: int, output_root: Path):
    config = base.load_config(config_path)
    cli = SimpleNamespace(
        config=str(config_path),
        scene_id=scene_id,
        rgb_root=None,
        frame_start=frame_start,
        frame_stride=frame_stride,
        frame_count=frame_count,
        frame_ids="",
        output_root=str(output_root),
        seed=107,
        birth_dump_dir="",
    )
    args = base.make_args(config, cli)
    args.model_dtype = "bfloat16"
    args.runtime_num_maskmem = int(config.get("sam2", {}).get("runtime_num_maskmem", 3))
    args.runtime_max_obj_ptrs_in_encoder = int(config.get("sam2", {}).get("runtime_max_obj_ptrs_in_encoder", 8))
    args.runtime_max_cond_frames_in_attn = int(config.get("sam2", {}).get("runtime_max_cond_frames_in_attn", 4))
    return args


def rectangle_masks(height: int, width: int) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    masks[101] = np.zeros((height, width), dtype=bool)
    masks[101][height // 5 : height // 2, width // 6 : width // 3] = True
    masks[102] = np.zeros((height, width), dtype=bool)
    masks[102][height // 4 : height // 2, width // 2 : (2 * width) // 3] = True
    masks[103] = np.zeros((height, width), dtype=bool)
    masks[103][height // 2 : (3 * height) // 4, width // 3 : width // 2] = True
    return masks


def mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float | None:
    if mask_a is None or mask_b is None:
        return None
    a = mask_a.astype(bool, copy=False)
    b = mask_b.astype(bool, copy=False)
    union = np.logical_or(a, b)
    denom = int(np.count_nonzero(union))
    if denom == 0:
        return 1.0
    return float(np.count_nonzero(np.logical_and(a, b)) / denom)


def mask_for_id(ids: np.ndarray, masks: np.ndarray, obj_id: int) -> np.ndarray | None:
    matches = np.where(ids.astype(np.int64) == int(obj_id))[0]
    if matches.size == 0:
        return None
    return masks[int(matches[0])].astype(bool, copy=False)


def add_initial_masks(predictor: Any, state: dict[str, Any], masks: dict[int, np.ndarray]) -> None:
    base.add_masks_to_stream_state(
        predictor,
        state,
        tracker="sam2",
        frame_idx=0,
        obj_ids=np.asarray([101, 102], dtype=np.int64),
        masks=np.stack([masks[101], masks[102]], axis=0),
    )


def infer_frame(predictor: Any, state: dict[str, Any], rgbs: list[np.ndarray], frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    _rolling_add_frame(predictor, state, frame_idx=int(frame_idx), rgb=rgbs[int(frame_idx)])
    return base.infer_stream_frame(predictor, state, frame_idx=int(frame_idx))


def run_gpu_smoke(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    import torch

    if str(args.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    frame_ids = base.parse_frame_ids("", int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = REPO_ROOT / "Stream3D/data/scannet/processed" / str(args.scene_id) / "color"
    rgbs = [base.read_rgb(rgb_root / f"{int(frame_id)}.jpg") for frame_id in frame_ids]
    height, width = rgbs[0].shape[:2]
    masks = rectangle_masks(height, width)

    model_args = make_args(config_path, str(args.scene_id), int(args.frame_start), int(args.frame_stride), int(args.frame_count), output_root)
    reset_rolling_stats()
    t0 = time.time()
    models = base.setup_models(model_args)
    predictor = models["tracker_model"]
    install_rolling_state_support(predictor)
    model_load_sec = float(time.time() - t0)

    def new_state() -> dict[str, Any]:
        state = predictor.init_state(
            video_path=None,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        _rolling_add_frame(predictor, state, frame_idx=0, rgb=rgbs[0])
        add_initial_masks(predictor, state, masks)
        infer_frame(predictor, state, rgbs, 0)
        infer_frame(predictor, state, rgbs, 1)
        return state

    direct_state = new_state()
    direct_error = None
    direct_add_allowed = True
    try:
        predictor.add_new_mask(
            inference_state=direct_state,
            frame_idx=1,
            obj_id=999,
            mask=torch.from_numpy(masks[103].astype(np.float32)),
        )
    except Exception as exc:  # noqa: BLE001 - probe records API behavior
        direct_add_allowed = False
        direct_error = f"{type(exc).__name__}: {exc}"

    control_state = new_state()
    control_ids_f2, control_masks_f2 = infer_frame(predictor, control_state, rgbs, 2)

    experiment_state = new_state()
    add_t0 = time.time()
    base.add_masks_to_stream_state(
        predictor,
        experiment_state,
        tracker="sam2",
        frame_idx=1,
        obj_ids=np.asarray([103], dtype=np.int64),
        masks=np.stack([masks[103]], axis=0),
    )
    post_start_add_sec = float(time.time() - add_t0)
    exp_ids_f2, exp_masks_f2 = infer_frame(predictor, experiment_state, rgbs, 2)
    existing_iou = {
        str(obj_id): mask_iou(
            mask_for_id(control_ids_f2, control_masks_f2, obj_id),
            mask_for_id(exp_ids_f2, exp_masks_f2, obj_id),
        )
        for obj_id in (101, 102)
    }

    remove_t0 = time.time()
    remove_error = None
    try:
        ids_after_remove, updated_frames = predictor.remove_object(
            experiment_state, 103, strict=True, need_output=False
        )
    except Exception as exc:  # noqa: BLE001 - probe records API behavior
        ids_after_remove = list(experiment_state.get("obj_ids", []))
        updated_frames = []
        remove_error = f"{type(exc).__name__}: {exc}"
    remove_sec = float(time.time() - remove_t0)

    readd_error = None
    readd_same_numeric_success = True
    readd_t0 = time.time()
    try:
        base.add_masks_to_stream_state(
            predictor,
            experiment_state,
            tracker="sam2",
            frame_idx=2,
            obj_ids=np.asarray([103], dtype=np.int64),
            masks=np.stack([masks[103]], axis=0),
        )
    except Exception as exc:  # noqa: BLE001 - probe records API behavior
        readd_same_numeric_success = False
        readd_error = f"{type(exc).__name__}: {exc}"
    readd_sec = float(time.time() - readd_t0)

    try:
        final_ids_f3, final_masks_f3 = infer_frame(predictor, experiment_state, rgbs, 3)
        final_infer_error = None
    except Exception as exc:  # noqa: BLE001 - probe records API behavior
        final_ids_f3 = np.asarray([], dtype=np.int64)
        final_masks_f3 = np.zeros((0, height, width), dtype=bool)
        final_infer_error = f"{type(exc).__name__}: {exc}"

    peak_cuda_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None
    torch.cuda.empty_cache()

    return {
        "schema_version": "stream4d_v107_sam2_memory_gpu_smoke_v1",
        "scene_id": str(args.scene_id),
        "frame_ids": [int(v) for v in frame_ids],
        "gpu": str(args.gpu),
        "model_load_sec": model_load_sec,
        "direct_new_id_after_tracking_allowed": bool(direct_add_allowed),
        "direct_new_id_after_tracking_error": direct_error,
        "v106_post_start_add_success": True,
        "v106_post_start_add_runtime_sec": post_start_add_sec,
        "control_frame2_obj_ids": [int(v) for v in control_ids_f2.tolist()],
        "experiment_frame2_obj_ids": [int(v) for v in exp_ids_f2.tolist()],
        "existing_object_frame2_iou_after_add": existing_iou,
        "remove_object_error": remove_error,
        "remove_runtime_sec": remove_sec,
        "ids_after_remove": [int(v) for v in ids_after_remove],
        "remove_updated_frame_count": int(len(updated_frames)),
        "same_numeric_id_readd_success_with_v106_workaround": bool(readd_same_numeric_success),
        "same_numeric_id_readd_error": readd_error,
        "same_numeric_id_readd_runtime_sec": readd_sec,
        "final_frame3_obj_ids": [int(v) for v in final_ids_f3.tolist()],
        "final_frame3_mask_count": int(final_masks_f3.shape[0]),
        "final_infer_error": final_infer_error,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "interpretation": (
            "This smoke probe is a mechanism check on synthetic rectangle prompts, not a Phase2 pass. "
            "Phase2 still needs A0-A6 frozen-frame parity with real masks."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(CURRENT_BEST_CONFIG.relative_to(REPO_ROOT)))
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--frame-start", type=int, default=4160)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--run-gpu-smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    phase0_dir = output_root / "phase0"
    phase2_dir = output_root / "phase2"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir.mkdir(parents=True, exist_ok=True)

    contract = source_contract()
    gpu_smoke = None
    if bool(args.run_gpu_smoke):
        gpu_smoke = run_gpu_smoke(args, output_root)
        contract["gpu_smoke_summary"] = {
            "path": str((phase2_dir / "sam2_memory_gpu_smoke.json").relative_to(REPO_ROOT)),
            "direct_new_id_after_tracking_allowed": gpu_smoke["direct_new_id_after_tracking_allowed"],
            "same_numeric_id_readd_success_with_v106_workaround": gpu_smoke[
                "same_numeric_id_readd_success_with_v106_workaround"
            ],
            "existing_object_frame2_iou_after_add": gpu_smoke["existing_object_frame2_iou_after_add"],
        }
        (phase2_dir / "sam2_memory_gpu_smoke.json").write_text(
            json.dumps(gpu_smoke, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    contract_path = phase0_dir / "sam2_memory_api_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "contract": str(contract_path),
                "gpu_smoke": str(phase2_dir / "sam2_memory_gpu_smoke.json") if gpu_smoke is not None else "",
                "run_gpu_smoke": bool(args.run_gpu_smoke),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
