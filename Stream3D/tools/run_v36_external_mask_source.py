from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from stream4d.scannet_stream import ScanNetStream
from tools.run_v26_object_quality_diagnostics import _json_safe


SAM3_PATHS = [
    "../ckpts/SAM3/sam3.1_multiplex.pt",
    "../ckpts/SAM3/sam3.pt",
    "ckpts/SAM3/sam3.1_multiplex.pt",
    "ckpts/SAM3/sam3.pt",
]

EFFICIENTSAM3_PATHS = [
    "../ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_m_mobileclip_s0_ctx16.pt",
    "../ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
    "../ckpts/EfficientSAM3/stage1_all_converted/efficient_sam3_efficientvit_s.pt",
    "ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_m_mobileclip_s0_ctx16.pt",
    "ckpts/EfficientSAM3/stage1_sam3p1/efficient_sam3p1_efficientvit_l_mobileclip_s0_ctx16.pt",
    "ckpts/EfficientSAM3/stage1_all_converted/efficient_sam3_efficientvit_s.pt",
]

DINO_PATHS = [
    "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
    "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
]


class TimeoutError(RuntimeError):
    pass


def _alarm(_signum: int, _frame: Any) -> None:
    raise TimeoutError("external mask source timeout")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _locate(paths: list[str], cwd: Path) -> tuple[bool, str | None, list[str]]:
    checked = []
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = cwd / path
        checked.append(str(path))
        if path.exists():
            return True, str(path), checked
    return False, None, checked


def _efficient_sam3_model_name(checkpoint: str) -> str:
    name = Path(checkpoint).name
    if "efficientvit_l" in name:
        return "b2"
    if "efficientvit_m" in name:
        return "b1"
    return "b0"


def _parse_frame_ids(value: str) -> list[int]:
    frame_ids: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        frame_ids.append(int(item))
    return frame_ids


def _select_frame_ids(ids: list[int], mode: str, requested: list[int] | None = None) -> list[int]:
    if requested:
        available = set(int(frame_id) for frame_id in ids)
        missing = [int(frame_id) for frame_id in requested if int(frame_id) not in available]
        if missing:
            raise FileNotFoundError(f"requested frames are not available: {missing[:16]}")
        return [int(frame_id) for frame_id in requested]
    if mode == "frame0":
        return [ids[0]]
    if mode == "sample8":
        if len(ids) <= 8:
            return ids
        keep = np.linspace(0, len(ids) - 1, num=8, dtype=np.int64)
        return [ids[int(idx)] for idx in keep.tolist()]
    if mode == "sample32":
        if len(ids) <= 32:
            return ids
        keep = np.linspace(0, len(ids) - 1, num=32, dtype=np.int64)
        return [ids[int(idx)] for idx in keep.tolist()]
    if mode == "sample64":
        if len(ids) <= 64:
            return ids
        keep = np.linspace(0, len(ids) - 1, num=64, dtype=np.int64)
        return [ids[int(idx)] for idx in keep.tolist()]
    if mode == "all_masks":
        return ids
    if mode == "probe5_full32":
        return ids[:32]
    raise ValueError(f"unknown mode: {mode}")


def _frame_ids(scene: str, mode: str, requested: list[int] | None = None) -> list[int]:
    stream = ScanNetStream(seq_name=scene)
    paths = sorted(stream.rgb_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    ids = [int(path.stem) for path in paths]
    if not ids:
        raise FileNotFoundError(f"no RGB frames for {scene}: {stream.rgb_dir}")
    return _select_frame_ids(ids, mode, requested)


def _save_mask_npz(
    output_dir: Path,
    source: str,
    frame_id: int,
    masks: list[np.ndarray],
    scores: list[float] | None = None,
    *,
    compressed: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = scores or [1.0] * len(masks)
    if masks:
        arr = np.stack([mask.astype(bool) for mask in masks], axis=0)
    else:
        arr = np.zeros((0, 1, 1), dtype=bool)
    save = np.savez_compressed if compressed else np.savez
    save(output_dir / f"{source}_frame{int(frame_id):06d}_masks.npz", masks=arr, scores=np.asarray(scores, dtype=np.float32))
    preview = np.zeros(arr.shape[1:], dtype=np.uint16) if arr.size else np.zeros((1, 1), dtype=np.uint16)
    for idx, mask in enumerate(masks, start=1):
        preview[np.asarray(mask, dtype=bool)] = idx
    cv2.imwrite(str(output_dir / f"{source}_frame{int(frame_id):06d}_label.png"), preview)


def _load_instance_map(stream: ScanNetStream, frame_id: int) -> np.ndarray | None:
    path = stream.root / "instance" / "instance" / f"{int(frame_id)}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64)


def _external_mask_metrics(stream: ScanNetStream, masks_by_frame: dict[int, list[np.ndarray]]) -> dict[str, Any]:
    total_regions = 0
    mixed_regions = 0
    best_iou: dict[tuple[int, int], float] = defaultdict(float)
    gt_keys: set[tuple[int, int]] = set()
    for frame_id, masks in sorted(masks_by_frame.items()):
        instance = _load_instance_map(stream, frame_id)
        if instance is None:
            continue
        for gt in sorted(int(v) for v in np.unique(instance) if int(v) > 0):
            gt_keys.add((int(frame_id), int(gt)))
        for mask in masks:
            if mask.shape != instance.shape:
                mask = cv2.resize(mask.astype(np.uint8), (instance.shape[1], instance.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask = np.asarray(mask, dtype=bool)
            if int(mask.sum()) <= 0:
                continue
            total_regions += 1
            values, counts = np.unique(instance[mask], return_counts=True)
            positive = [(int(v), int(c)) for v, c in zip(values.tolist(), counts.tolist()) if int(v) > 0 and int(c) >= 16]
            labeled = sum(count for _, count in positive)
            if len(positive) > 1:
                dominant = max(count for _, count in positive) / max(labeled, 1)
                if dominant < 0.95:
                    mixed_regions += 1
            for gt, overlap in positive:
                gt_mask = instance == int(gt)
                denom = int(mask.sum()) + int(gt_mask.sum()) - int(overlap)
                if denom > 0:
                    key = (int(frame_id), int(gt))
                    best_iou[key] = max(best_iou[key], float(overlap / denom))
    return {
        "diagnostic_scope": "2D instance PNGs for evaluated frames only",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "mixed_region_rate": float(mixed_regions / max(total_regions, 1)),
        "GT_object_coverage@0.10": float(sum(1 for key in gt_keys if best_iou.get(key, 0.0) >= 0.10) / max(len(gt_keys), 1)),
        "GT_object_coverage@0.25": float(sum(1 for key in gt_keys if best_iou.get(key, 0.0) >= 0.25) / max(len(gt_keys), 1)),
        "evaluated_frame_instance_count": int(len(gt_keys)),
    }


def _watershed_masks(stream: ScanNetStream, frame_id: int) -> list[np.ndarray]:
    rgb = stream.load_rgb(frame_id)
    mask = stream.load_mask(frame_id)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out: list[np.ndarray] = []
    for value in sorted(int(v) for v in np.unique(mask) if int(v) > 0):
        binary = (mask == value).astype(np.uint8)
        if int(binary.sum()) < 32:
            continue
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        if float(dist.max()) <= 0.0:
            out.append(binary.astype(bool))
            continue
        _, seeds = cv2.threshold(dist, 0.35 * float(dist.max()), 255, 0)
        seeds = seeds.astype(np.uint8)
        n, markers = cv2.connectedComponents(seeds)
        if n <= 2:
            out.append(binary.astype(bool))
            continue
        markers = markers + 1
        markers[binary == 0] = 0
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        labels = cv2.watershed(color, markers.astype(np.int32))
        for label in sorted(int(v) for v in np.unique(labels) if int(v) > 1):
            part = labels == label
            if int(part.sum()) >= 32:
                out.append(part)
    return out


def _run_watershed(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    stream = ScanNetStream(seq_name=args.scene)
    available = sorted((int(path.stem) for path in stream.mask_dir.glob("*.png")), key=int)
    if not available:
        raise FileNotFoundError(f"no Cropformer masks for {args.scene}: {stream.mask_dir}")
    frames = _select_frame_ids(available, args.mode, _parse_frame_ids(str(args.frame_ids)))
    total = 0
    masks_by_frame: dict[int, list[np.ndarray]] = {}
    for frame_id in frames:
        masks = _watershed_masks(stream, frame_id)
        _save_mask_npz(output_dir, "watershed", frame_id, masks, compressed=not bool(args.uncompressed_mask_npz))
        masks_by_frame[int(frame_id)] = masks
        total += len(masks)
    result = {"integration_pass": True, "mask_count": int(total), "region_count": int(total), "frames": frames}
    result.update(_external_mask_metrics(stream, masks_by_frame))
    return result


def _run_dinov2_maskcut(args: argparse.Namespace, output_dir: Path, cwd: Path) -> dict[str, Any]:
    found, checkpoint, checked = _locate(DINO_PATHS, cwd)
    if not found or checkpoint is None:
        return {"integration_pass": False, "failure_stage": "checkpoint", "checkpoint_found": False, "attempted_paths": checked}
    import torch
    import torch.nn.functional as F
    import timm

    device = "cuda:0" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    stream = ScanNetStream(seq_name=args.scene)
    frames = _frame_ids(args.scene, args.mode, _parse_frame_ids(str(args.frame_ids)))
    total = 0
    masks_by_frame: dict[int, list[np.ndarray]] = {}
    for frame_id in frames:
        rgb = stream.load_rgb(frame_id)
        resized = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        tensor = torch.from_numpy(((resized - np.asarray([0.485, 0.456, 0.406])) / np.asarray([0.229, 0.224, 0.225])).transpose(2, 0, 1)).float()[None].to(device)
        with torch.inference_mode():
            out = model.forward_features(tensor)
            tokens = out["x_norm_patchtokens"] if isinstance(out, dict) and "x_norm_patchtokens" in out else out[:, 1:, :]
            tokens = F.normalize(tokens.float(), dim=-1).squeeze(0).detach().cpu().numpy()
        scores = tokens[:, 0]
        grid = int(round(np.sqrt(tokens.shape[0])))
        score_map = cv2.resize(scores.reshape(grid, grid), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        threshold = float(np.quantile(score_map, 0.70))
        binary = score_map >= threshold
        n, labels = cv2.connectedComponents(binary.astype(np.uint8))
        masks = [(labels == label) for label in range(1, n) if int((labels == label).sum()) >= 64]
        _save_mask_npz(output_dir, "dinov2_maskcut", frame_id, masks, compressed=not bool(args.uncompressed_mask_npz))
        masks_by_frame[int(frame_id)] = masks
        total += len(masks)
    result = {
        "integration_pass": True,
        "checkpoint_found": True,
        "checkpoint": checkpoint,
        "mask_count": int(total),
        "region_count": int(total),
        "frames": frames,
    }
    result.update(_external_mask_metrics(stream, masks_by_frame))
    return result


def _run_sam3(args: argparse.Namespace, output_dir: Path, cwd: Path) -> dict[str, Any]:
    found, checkpoint, checked = _locate(SAM3_PATHS, cwd)
    if not found or checkpoint is None:
        return {"integration_pass": False, "failure_stage": "checkpoint", "checkpoint_found": False, "attempted_paths": checked}
    sys.path.insert(0, str((cwd / "../third_party/sam3").resolve()))
    import torch
    from sam3 import build_sam3_predictor

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(0)
    stream = ScanNetStream(seq_name=args.scene)
    frames = _frame_ids(args.scene, args.mode, _parse_frame_ids(str(args.frame_ids)))
    resource_path = str(stream.rgb_dir)
    model = build_sam3_predictor(checkpoint_path=checkpoint, compile=False, async_loading_frames=False)
    response = model.handle_request({"type": "start_session", "resource_path": resource_path})
    session_id = response["session_id"]
    output = model.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": str(args.prompt),
        }
    )
    if hasattr(model, "shutdown"):
        model.shutdown()
    mask_count = _count_nested_masks(output)
    _write_json(output_dir / "sam3_raw_response_summary.json", _summarize_response(output))
    return {
        "integration_pass": True,
        "checkpoint_found": True,
        "checkpoint": checkpoint,
        "prompt": str(args.prompt),
        "mask_count": int(mask_count),
        "region_count": int(mask_count),
        "frames": frames,
    }


def _run_efficientsam3(args: argparse.Namespace, output_dir: Path, cwd: Path) -> dict[str, Any]:
    found, checkpoint, checked = _locate(EFFICIENTSAM3_PATHS, cwd)
    if not found or checkpoint is None:
        return {"integration_pass": False, "failure_stage": "checkpoint", "checkpoint_found": False, "attempted_paths": checked}
    sys.path.insert(0, str((cwd / "../third_party/efficientsam3/sam3").resolve()))
    import torch
    from sam3 import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda:0" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    model = build_efficientsam3_image_model(
        bpe_path=str((cwd / "../third_party/efficientsam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz").resolve()),
        enable_inst_interactivity=True,
        checkpoint_path=checkpoint,
        load_from_HF=False,
        backbone_type="efficientvit",
        model_name=_efficient_sam3_model_name(checkpoint),
    )
    model.to(device).eval()
    processor = Sam3Processor(model)
    stream = ScanNetStream(seq_name=args.scene)
    frames = _frame_ids(args.scene, args.mode, _parse_frame_ids(str(args.frame_ids)))
    total = 0
    masks_by_frame: dict[int, list[np.ndarray]] = {}
    for frame_id in frames:
        image = Image.fromarray(stream.load_rgb(frame_id))
        state = processor.set_image(image)
        box = np.asarray([[0, 0, image.width - 1, image.height - 1]], dtype=np.float32)
        with torch.inference_mode():
            masks, scores, _ = model.predict_inst(state, point_coords=None, point_labels=None, box=box, multimask_output=True)
        masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        scores_np = scores.detach().cpu().numpy().reshape(-1).tolist() if hasattr(scores, "detach") else list(np.asarray(scores).reshape(-1))
        mask_list = [(mask > 0) for mask in masks_np.reshape((-1,) + masks_np.shape[-2:])]
        _save_mask_npz(
            output_dir,
            "efficientsam3",
            frame_id,
            mask_list,
            [float(v) for v in scores_np[: len(mask_list)]],
            compressed=not bool(args.uncompressed_mask_npz),
        )
        masks_by_frame[int(frame_id)] = mask_list
        total += len(mask_list)
    result = {
        "integration_pass": True,
        "checkpoint_found": True,
        "checkpoint": checkpoint,
        "mask_count": int(total),
        "region_count": int(total),
        "frames": frames,
    }
    result.update(_external_mask_metrics(stream, masks_by_frame))
    return result


def _count_nested_masks(value: Any) -> int:
    if isinstance(value, dict):
        count = 0
        for key, item in value.items():
            if str(key).lower() in {"masks", "pred_masks"}:
                try:
                    return int(len(item))
                except TypeError:
                    return 1
            count += _count_nested_masks(item)
        return count
    if isinstance(value, list):
        return sum(_count_nested_masks(item) for item in value)
    return 0


def _summarize_response(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return str(type(value))
    if isinstance(value, dict):
        return {str(k): _summarize_response(v, depth + 1) for k, v in value.items() if str(k) not in {"masks", "pred_masks"}}
    if isinstance(value, list):
        return [_summarize_response(item, depth + 1) for item in value[:5]]
    if hasattr(value, "shape"):
        return {"type": str(type(value)), "shape": list(value.shape)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value))


def run(args: argparse.Namespace) -> dict[str, Any]:
    cwd = Path.cwd()
    output_dir = Path(args.output_root) / str(args.source) / str(args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(args.timeout_sec))
    stage = "start"
    try:
        stage = "inference"
        if args.source == "watershed":
            result = _run_watershed(args, output_dir)
        elif args.source == "dinov2_maskcut":
            result = _run_dinov2_maskcut(args, output_dir, cwd)
        elif args.source == "sam3":
            stage = "model_build_or_inference"
            result = _run_sam3(args, output_dir, cwd)
        elif args.source == "efficientsam3":
            stage = "model_build_or_inference"
            result = _run_efficientsam3(args, output_dir, cwd)
        else:
            raise ValueError(f"unsupported source: {args.source}")
        result.setdefault("failure_stage", "")
    except Exception as exc:
        result = {
            "integration_pass": False,
            "failure_stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
        }
    finally:
        signal.alarm(0)
    result.update(
        {
            "source": str(args.source),
            "mode": str(args.mode),
            "scene": str(args.scene),
            "runtime_sec": float(time.time() - start),
            "output_dir": str(output_dir),
            "note": "external source smoke; downstream integration is recorded separately when available",
            "mask_npz_compressed": not bool(args.uncompressed_mask_npz),
            "requested_frame_ids": _parse_frame_ids(str(args.frame_ids)),
        }
    )
    result.setdefault("downstream_E_assignment_ARI", None)
    result.setdefault("mixed_region_rate", None)
    result.setdefault("GT_object_coverage@0.10", None)
    result.setdefault("GT_object_coverage@0.25", None)
    _write_json(output_dir / "summary.json", result)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sam3", "efficientsam3", "dinov2_maskcut", "watershed"], required=True)
    parser.add_argument("--mode", choices=["frame0", "sample8", "sample32", "sample64", "all_masks", "probe5_full32"], default="frame0")
    parser.add_argument("--scene", default="scene0081_01")
    parser.add_argument("--output-root", default="outputs/audit/v36_external_mask_source")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--uncompressed-mask-npz", action="store_true")
    parser.add_argument("--frame-ids", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
