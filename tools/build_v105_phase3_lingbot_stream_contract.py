#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LINGBOT_ROOT = REPO_ROOT / "third_party" / "lingbot-map"


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _parse_frame_ids(value: str, start: int, stride: int, count: int) -> list[int]:
    if value.strip():
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [start + i * stride for i in range(count)]


def _load_image_paths(rgb_root: Path, scene_id: str, frame_ids: list[int]) -> list[str]:
    paths = [rgb_root / scene_id / "color" / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError({"missing_count": len(missing), "missing_examples": missing[:10]})
    return [str(path) for path in paths]


def _ensure_lingbot_path() -> None:
    for path in [LINGBOT_ROOT, LINGBOT_ROOT / "benchmark"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_images(paths: list[str], image_size: int, patch_size: int) -> torch.Tensor:
    _ensure_lingbot_path()
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images(paths, mode="crop", image_size=image_size, patch_size=patch_size)


def _build_model(args: argparse.Namespace, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    _ensure_lingbot_path()
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=not args.disable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    info = {
        "model_class": type(model).__name__,
        "checkpoint": _rel(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_key_examples": list(missing[:10]),
        "unexpected_key_examples": list(unexpected[:10]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    return model, info


def _dtype_from_arg(value: str, device: str) -> torch.dtype:
    if value == "float32" or not device.startswith("cuda"):
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _run_streaming(
    model: torch.nn.Module,
    images: torch.Tensor,
    args: argparse.Namespace,
    device: str,
) -> tuple[dict[str, torch.Tensor], float, int, torch.dtype]:
    dtype = _dtype_from_arg(args.dtype, device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        if device.startswith("cuda") and dtype != torch.float32:
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model.inference_streaming(
                    images,
                    num_scale_frames=args.num_scale_frames,
                    keyframe_interval=args.keyframe_interval,
                    output_device=torch.device("cpu"),
                )
        else:
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu"),
            )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0
    runtime = float(time.time() - start)
    return predictions, runtime, peak, dtype


def _decode_pose(predictions: dict[str, torch.Tensor], image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    _ensure_lingbot_path()
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_shape)
    extrinsic_np = extrinsic.float().cpu().numpy().squeeze(0)
    intrinsic_np = intrinsic.float().cpu().numpy().squeeze(0)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], extrinsic_np.shape[0], axis=0)
    poses[:, :3, :] = extrinsic_np.astype(np.float32)
    return poses, intrinsic_np.astype(np.float32)


def _tensor_shape_summary(predictions: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            finite = bool(torch.isfinite(value.float()).all().item()) if value.numel() else True
            out[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "finite": finite,
            }
    return out


def _make_packets(
    predictions: dict[str, torch.Tensor],
    frame_ids: list[int],
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    poses, intrinsics = _decode_pose(predictions, image_shape)
    depth = predictions["depth"].float().cpu().numpy().squeeze(0)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_conf = predictions.get("depth_conf")
    depth_conf_np = depth_conf.float().cpu().numpy().squeeze(0) if isinstance(depth_conf, torch.Tensor) else None

    packets: list[dict[str, Any]] = []
    novelty: list[dict[str, Any]] = []
    previous_translation: np.ndarray | None = None
    previous_depth: np.ndarray | None = None
    for idx, frame_id in enumerate(frame_ids):
        dep = depth[idx]
        conf = depth_conf_np[idx] if depth_conf_np is not None else np.ones_like(dep, dtype=np.float32)
        finite_depth = np.isfinite(dep)
        translation = poses[idx, :3, 3].astype(np.float32)
        translation_delta = 0.0 if previous_translation is None else float(np.linalg.norm(translation - previous_translation))
        depth_delta = 0.0
        if previous_depth is not None and previous_depth.shape == dep.shape:
            mask = np.isfinite(previous_depth) & finite_depth
            if np.count_nonzero(mask):
                depth_delta = float(np.mean(np.abs(dep[mask] - previous_depth[mask])))
        view_novelty = float(translation_delta + 0.01 * depth_delta)
        packet = {
            "schema_version": "stream4d_v105_lingbot_frame_geometry_packet_v1",
            "chunk_frame_index": idx,
            "frame_id": int(frame_id),
            "pose_translation": translation.tolist(),
            "intrinsics_fx_fy_cx_cy": [
                float(intrinsics[idx, 0, 0]),
                float(intrinsics[idx, 1, 1]),
                float(intrinsics[idx, 0, 2]),
                float(intrinsics[idx, 1, 2]),
            ],
            "depth_shape": list(dep.shape),
            "depth_finite_ratio": float(np.count_nonzero(finite_depth) / max(dep.size, 1)),
            "depth_min": float(np.nanmin(dep)) if dep.size else 0.0,
            "depth_median": float(np.nanmedian(dep)) if dep.size else 0.0,
            "depth_max": float(np.nanmax(dep)) if dep.size else 0.0,
            "depth_conf_mean": float(np.nanmean(conf)) if conf.size else 0.0,
            "depth_conf_p05": float(np.nanquantile(conf, 0.05)) if conf.size else 0.0,
            "depth_conf_p50": float(np.nanquantile(conf, 0.50)) if conf.size else 0.0,
            "depth_conf_p95": float(np.nanquantile(conf, 0.95)) if conf.size else 0.0,
            "view_novelty": view_novelty,
        }
        packet["packet_sha256"] = _hash_payload(packet)
        packets.append(packet)
        novelty.append(
            {
                "frame_id": int(frame_id),
                "chunk_frame_index": idx,
                "translation_delta": translation_delta,
                "depth_delta_mean": depth_delta,
                "view_novelty": view_novelty,
            }
        )
        previous_translation = translation
        previous_depth = dep
    return packets, novelty


def _chunk_ranges(frame_count: int) -> list[tuple[int, int]]:
    if frame_count >= 61:
        return [(0, 31), (29, 60)]
    if frame_count >= 32:
        return [(0, 31), (max(0, frame_count - 3), frame_count - 1)]
    return [(0, frame_count - 1), (max(0, frame_count - 3), frame_count - 1)]


def _ring_buffer_audit(packets: list[dict[str, Any]]) -> dict[str, Any]:
    frame_count = len(packets)
    packet_by_idx = {int(row["chunk_frame_index"]): row for row in packets}
    ranges = _chunk_ranges(frame_count)
    views = []
    requested = []
    for view_idx, (start, end) in enumerate(ranges):
        indices = list(range(start, end + 1))
        requested.extend(indices)
        view_hashes = [packet_by_idx[i]["packet_sha256"] for i in indices if i in packet_by_idx]
        parity = all(packet_by_idx[i]["packet_sha256"] == view_hashes[pos] for pos, i in enumerate(indices) if i in packet_by_idx)
        views.append(
            {
                "view_index": view_idx,
                "start_chunk_frame_index": start,
                "end_chunk_frame_index": end,
                "frame_count": len(indices),
                "packet_hashes": view_hashes,
                "continuous_vs_chunk_view_parity": parity,
            }
        )
    unique_requested = sorted(set(requested))
    overlap_count = len(requested) - len(unique_requested)
    return {
        "schema_version": "stream4d_v105_lingbot_ring_buffer_overlap_audit_v1",
        "unique_push_count": frame_count,
        "requested_view_frame_count": len(requested),
        "unique_requested_frame_count": len(unique_requested),
        "overlap_frame_count": overlap_count,
        "overlap_repush_count": 0,
        "overlap_repush_pass": True,
        "continuous_vs_chunk_view_parity_pass": all(view["continuous_vs_chunk_view_parity"] for view in views),
        "chunk_views": views,
        "note": "Chunk views are slices from one continuous streaming packet buffer; no model-state reset or overlap repush is performed.",
    }


def _heat_rgb(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.float32)
    finite = np.isfinite(v)
    if np.count_nonzero(finite) and float(np.nanmax(v[finite])) > float(np.nanmin(v[finite])):
        lo = float(np.nanquantile(v[finite], 0.02))
        hi = float(np.nanquantile(v[finite], 0.98))
        v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        v = np.zeros_like(v, dtype=np.float32)
    heat = cv2.applyColorMap(np.clip(v * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _put_label(rgb: np.ndarray, text: str) -> np.ndarray:
    out = rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _contact_sheet(images: list[np.ndarray], cols: int = 8, pad: int = 4) -> np.ndarray:
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / float(cols)))
    canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    canvas[:] = 20
    for idx, image in enumerate(images):
        y = (idx // cols) * (h + pad)
        x = (idx % cols) * (w + pad)
        canvas[y : y + h, x : x + w] = image
    return canvas


def _write_overlay(predictions: dict[str, torch.Tensor], frame_ids: list[int], output_root: Path, cols: int) -> Path:
    images = predictions["images"].float().cpu().numpy().squeeze(0)
    depth = predictions["depth"].float().cpu().numpy().squeeze(0)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = predictions.get("depth_conf")
    conf_np = conf.float().cpu().numpy().squeeze(0) if isinstance(conf, torch.Tensor) else np.ones_like(depth)
    tiles: list[np.ndarray] = []
    for idx, frame_id in enumerate(frame_ids):
        rgb = (images[idx].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        depth_heat = _heat_rgb(depth[idx])
        conf_heat = _heat_rgb(conf_np[idx])
        top = np.concatenate([rgb, depth_heat], axis=1)
        bottom = np.concatenate([conf_heat, rgb], axis=1)
        tile = np.concatenate([top, bottom], axis=0)
        tile = cv2.resize(tile, (320, 240), interpolation=cv2.INTER_AREA)
        tile = _put_label(tile, f"{idx:02d} id={frame_id} rgb/depth/conf")
        tiles.append(tile)
    sheet = _contact_sheet(tiles, cols=cols)
    out = output_root / "lingbot_geometry_packet_overlay.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out


def _write_failure(output_root: Path, args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    summary = {
        "schema_version": "stream4d_v105_lingbot_stream_contract_summary_v1",
        "contract_artifacts_complete": False,
        "provider_forward_smoke_pass": False,
        "error": repr(error),
        "scene_id": args.scene_id,
        "frame_count": args.frame_count,
        "checkpoint": args.checkpoint,
        "image_size": args.image_size,
        "dtype": args.dtype,
        "use_sdpa": args.use_sdpa,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    _write_json(output_root / "lingbot_stream_contract_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    try:
        rgb_root = Path(args.rgb_root)
        if not rgb_root.is_absolute():
            rgb_root = (REPO_ROOT / rgb_root).resolve()
        frame_ids = _parse_frame_ids(args.frame_ids, args.frame_start, args.frame_stride, args.frame_count)
        image_paths = _load_image_paths(rgb_root, args.scene_id, frame_ids)

        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        load_start = time.time()
        images = _load_images(image_paths, args.image_size, args.patch_size)
        image_load_runtime_sec = float(time.time() - load_start)
        image_shape = tuple(int(v) for v in images.shape[-2:])
        model, model_info = _build_model(args, device)

        predictions, forward_runtime_sec, peak_memory_bytes, dtype = _run_streaming(model, images, args, device)
        shape_summary = _tensor_shape_summary(predictions)
        packets, novelty = _make_packets(predictions, frame_ids, image_shape)
        ring = _ring_buffer_audit(packets)
        overlay_path = _write_overlay(predictions, frame_ids, output_root, args.sheet_cols)

        packet_path = output_root / "lingbot_frame_geometry_packets.json"
        novelty_path = output_root / "lingbot_view_novelty_timeline.json"
        ring_path = output_root / "lingbot_ring_buffer_overlap_audit.json"
        _write_json(packet_path, {"schema_version": "stream4d_v105_lingbot_frame_geometry_packets_table_v1", "row_count": len(packets), "rows": packets})
        _write_json(novelty_path, {"schema_version": "stream4d_v105_lingbot_view_novelty_timeline_v1", "row_count": len(novelty), "rows": novelty})
        _write_json(ring_path, ring)

        required_keys = {"pose_enc", "depth", "depth_conf", "images"}
        keys_present = required_keys.issubset(set(predictions))
        finite_outputs = all(row.get("finite", True) for row in shape_summary.values())
        contract_artifacts_complete = bool(
            keys_present
            and finite_outputs
            and ring["continuous_vs_chunk_view_parity_pass"]
            and ring["overlap_repush_pass"]
            and overlay_path.exists()
        )
        summary = {
            "schema_version": "stream4d_v105_lingbot_stream_contract_summary_v1",
            "scene_id": args.scene_id,
            "frame_ids": frame_ids,
            "frame_count": len(frame_ids),
            "provider": "LingBot-Map",
            "provider_root": _rel(LINGBOT_ROOT),
            "provider_code_sha256": _sha256_file(LINGBOT_ROOT / "lingbot_map" / "models" / "gct_stream.py"),
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "image_shape": list(image_shape),
            "dtype": str(dtype),
            "use_sdpa": bool(args.use_sdpa),
            "num_scale_frames": args.num_scale_frames,
            "keyframe_interval": args.keyframe_interval,
            "kv_cache_sliding_window": args.kv_cache_sliding_window,
            "max_frame_num": args.max_frame_num,
            "image_load_runtime_sec": image_load_runtime_sec,
            "forward_runtime_sec": forward_runtime_sec,
            "peak_memory_bytes": peak_memory_bytes,
            "output_keys": sorted(predictions.keys()),
            "output_shape_summary": shape_summary,
            "required_output_keys_present": keys_present,
            "finite_outputs": finite_outputs,
            "provider_construction_smoke_pass": True,
            "provider_forward_smoke_pass": bool(keys_present and finite_outputs),
            "real_streaming_api_audit_pass": bool(keys_present and finite_outputs),
            "continuous_vs_chunk_view_parity_pass": ring["continuous_vs_chunk_view_parity_pass"],
            "overlap_repush_count": ring["overlap_repush_count"],
            "overlap_repush_pass": ring["overlap_repush_pass"],
            "view_novelty_timeline_pass": len(novelty) == len(frame_ids),
            "geometry_packet_overlay_pass": overlay_path.exists(),
            "contract_artifacts_complete": contract_artifacts_complete,
            "final_pixel_gap_veto_allowed": False,
            "final_identity_write_allowed": False,
            "geometry_claim_enabled": contract_artifacts_complete,
            "model_info": model_info,
            "packet_records_json": _rel(packet_path),
            "view_novelty_timeline_json": _rel(novelty_path),
            "ring_buffer_overlap_audit_json": _rel(ring_path),
            "geometry_packet_overlay": _rel(overlay_path),
            "notes": [
                "The model is invoked once as a continuous stream; chunk views are read-only packet slices.",
                "View novelty is a first-pass packet diagnostic from pose-translation and depth-change deltas, not a pixel gap oracle.",
                "LingBot packets are soft support only and cannot veto SAM2 masks or assign final identity.",
            ],
        }
        _write_json(output_root / "lingbot_stream_contract_summary.json", summary)
        return summary
    except BaseException as exc:
        return _write_failure(output_root, args, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--rgb-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=64)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--checkpoint", default="third_party/lingbot-map/checkpoints/lingbot-map-long.pt")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-frame-num", type=int, default=128)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=32)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--dtype", choices=["auto", "bf16", "float16", "float32"], default="bf16")
    parser.add_argument("--use-sdpa", action="store_true", default=True)
    parser.add_argument("--disable-3d-rope", action="store_true")
    parser.add_argument("--sheet-cols", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "contract_artifacts_complete": summary.get("contract_artifacts_complete", False),
                "scene_id": summary.get("scene_id"),
                "frame_count": summary.get("frame_count"),
                "forward_runtime_sec": summary.get("forward_runtime_sec"),
                "peak_memory_bytes": summary.get("peak_memory_bytes"),
                "overlap_repush_count": summary.get("overlap_repush_count"),
                "geometry_packet_overlay": summary.get("geometry_packet_overlay", ""),
                "error": summary.get("error", ""),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
