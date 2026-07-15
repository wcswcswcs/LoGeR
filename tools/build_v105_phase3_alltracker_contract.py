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

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLTRACKER_ROOT = REPO_ROOT / "third_party" / "alltracker"


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


def _parse_frame_ids(value: str, start: int, stride: int, count: int) -> list[int]:
    if value.strip():
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [start + i * stride for i in range(count)]


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_rgb(rgb: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def _read_label(path: Path, hw: tuple[int, int]) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    h, w = hw
    if label.shape[:2] != (h, w):
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.int32, copy=False)


def _load_alltracker(checkpoint: Path, window_len: int, device: str) -> torch.nn.Module:
    if str(ALLTRACKER_ROOT) not in sys.path:
        sys.path.insert(0, str(ALLTRACKER_ROOT))
    from nets.alltracker import Net

    model = Net(seqlen=window_len)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _make_windows(frame_count: int, window_len: int, relay_stride: int) -> list[dict[str, int]]:
    starts: list[int] = []
    start = 0
    while start + window_len < frame_count:
        starts.append(start)
        start += relay_stride
    starts.append(start)
    return [
        {
            "start": int(s),
            "end_inclusive": int(s + window_len - 1),
            "start_frame_index": int(s),
            "end_frame_index_inclusive": int(min(frame_count - 1, s + window_len - 1)),
        }
        for s in starts
    ]


def _run_alltracker(
    model: torch.nn.Module,
    frames_rgb: list[np.ndarray],
    inference_iters: int,
    window_len: int,
    relay_stride: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    stacked = np.stack(frames_rgb, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().float().unsqueeze(0).to(device)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        flows, visconf, _, _ = model.forward_sliding(
            tensor,
            iters=inference_iters,
            sw=None,
            is_training=False,
            window_len=window_len,
            stride=relay_stride,
        )
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0
    runtime = float(time.time() - start)
    return flows.detach().cpu(), visconf.detach().cpu(), runtime, peak


def _normalise_alltracker_outputs(
    flows: torch.Tensor,
    visconf: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return flow[T,2,H,W], visibility[T,H,W], confidence[T,H,W].

    AllTracker's two-frame branch returns a pair flow without an explicit time
    dimension. For provider smoke we add frame0 identity coverage and use the
    returned pair field for frame1. Multi-frame relay already returns T.
    """
    flow0 = flows[0].numpy()
    vis0 = visconf[0].numpy()
    inserted_identity_frames: list[int] = []
    if flow0.ndim == 3 and frame_count == 2:
        full_flow = np.zeros((2, 2, height, width), dtype=flow0.dtype)
        full_flow[1] = flow0
        full_visibility = np.ones((2, height, width), dtype=vis0.dtype)
        full_confidence = np.ones((2, height, width), dtype=vis0.dtype)
        full_visibility[1] = vis0[0]
        full_confidence[1] = vis0[1]
        inserted_identity_frames.append(0)
        return full_flow, full_visibility, full_confidence, inserted_identity_frames
    if flow0.ndim != 4:
        raise ValueError({"unexpected_flow_shape": list(flow0.shape), "frame_count": frame_count})
    return flow0, vis0[:, 0], vis0[:, 1], inserted_identity_frames


def _scatter_coverage(
    flow_xy: np.ndarray,
    confidence: np.ndarray,
    visibility: np.ndarray,
    source_fg: np.ndarray,
    conf_thr: float,
    visibility_thr: float,
    envelope_conf_thr: float,
    envelope_visibility_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = source_fg.shape
    yy, xx = np.nonzero(source_fg)
    if yy.size == 0:
        return np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)

    x_target = np.rint(xx.astype(np.float32) + flow_xy[0, yy, xx]).astype(np.int32)
    y_target = np.rint(yy.astype(np.float32) + flow_xy[1, yy, xx]).astype(np.int32)
    in_bounds = (x_target >= 0) & (x_target < w) & (y_target >= 0) & (y_target < h)

    conf_vals = confidence[yy, xx]
    vis_vals = visibility[yy, xx]
    core_ok = in_bounds & (conf_vals >= conf_thr) & (vis_vals >= visibility_thr)
    envelope_ok = in_bounds & (conf_vals >= envelope_conf_thr) & (vis_vals >= envelope_visibility_thr)

    core = np.zeros((h, w), dtype=bool)
    envelope = np.zeros((h, w), dtype=bool)
    core[y_target[core_ok], x_target[core_ok]] = True
    envelope[y_target[envelope_ok], x_target[envelope_ok]] = True
    return core, envelope


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _heat_overlay(rgb: np.ndarray, values: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    v = values.astype(np.float32)
    if np.nanmax(v) > np.nanmin(v):
        v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
    else:
        v = np.zeros_like(v, dtype=np.float32)
    heat = cv2.applyColorMap(np.clip(v * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip(rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _put_label(rgb: np.ndarray, text: str) -> np.ndarray:
    out = rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _contact_sheet(images: list[np.ndarray], cols: int = 8, pad: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("no images for sheet")
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / float(cols)))
    canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    canvas[:] = 24
    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y : y + h, x : x + w] = image
    return canvas


def _safe_ratio(num: int | float, den: int | float) -> float:
    if den == 0:
        return 1.0
    return float(num) / float(den)


def _write_png(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rgb_root = (REPO_ROOT / args.rgb_root).resolve() if not Path(args.rgb_root).is_absolute() else Path(args.rgb_root)
    label_root = Path(args.x1_label_dir)
    if not label_root.is_absolute():
        label_root = (REPO_ROOT / label_root).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()

    frame_ids = _parse_frame_ids(args.frame_ids, args.frame_start, args.frame_stride, args.frame_count)
    canonical_hw = (args.canonical_height, args.canonical_width)
    frame_paths = [rgb_root / args.scene_id / "color" / f"{frame_id}.jpg" for frame_id in frame_ids]
    label_paths = [label_root / f"frame_{frame_id:06d}.png" for frame_id in frame_ids]

    missing_frames = [str(p) for p in frame_paths if not p.exists()]
    if missing_frames:
        raise FileNotFoundError({"missing_frame_count": len(missing_frames), "missing_examples": missing_frames[:10]})
    missing_labels = [str(p) for p in label_paths if not p.exists()]
    if missing_labels and not bool(args.allow_missing_diagnostic_labels):
        raise FileNotFoundError({"missing_count": len(missing_labels), "missing_examples": missing_labels[:10]})
    if not label_paths[0].exists():
        raise FileNotFoundError(
            {
                "missing_frame0_source_label": str(label_paths[0]),
                "repair": "provide a label for the first frame; future labels are optional only with --allow-missing-diagnostic-labels",
            }
        )

    frames_rgb = [_resize_rgb(_read_rgb(path), canonical_hw) for path in frame_paths]
    labels = [_read_label(path, canonical_hw) if path.exists() else None for path in label_paths]
    source_fg = labels[0] > 0

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    construction_start = time.time()
    model = _load_alltracker(checkpoint, args.window_len, device)
    construction_runtime_sec = float(time.time() - construction_start)

    flows, visconf, forward_runtime_sec, peak_memory_bytes = _run_alltracker(
        model,
        frames_rgb,
        args.inference_iters,
        args.window_len,
        args.relay_stride,
        device,
    )

    flow_np, vis_np, conf_np, inserted_identity_frames = _normalise_alltracker_outputs(
        flows,
        visconf,
        len(frame_ids),
        args.canonical_height,
        args.canonical_width,
    )
    no_nan = bool(np.isfinite(flow_np).all() and np.isfinite(vis_np).all() and np.isfinite(conf_np).all())
    shape_ok = tuple(flow_np.shape) == (len(frame_ids), 2, args.canonical_height, args.canonical_width)
    forward_smoke_pass = bool(no_nan and shape_ok)

    envelope_conf_thr = args.envelope_conf_thr if args.envelope_conf_thr is not None else args.conf_thr * 0.5
    envelope_visibility_thr = (
        args.envelope_visibility_thr if args.envelope_visibility_thr is not None else args.visibility_thr * 0.5
    )

    coverage_dir = output_root / "coverage_masks"
    overlay_dir = output_root / "overlays"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    confidence_sheet_tiles: list[np.ndarray] = []
    visibility_sheet_tiles: list[np.ndarray] = []
    proxy_sheet_tiles: list[np.ndarray] = []
    coverage_sheet_tiles: list[np.ndarray] = []

    for idx, frame_id in enumerate(frame_ids):
        core, envelope = _scatter_coverage(
            flow_np[idx],
            conf_np[idx],
            vis_np[idx],
            source_fg,
            args.conf_thr,
            args.visibility_thr,
            envelope_conf_thr,
            envelope_visibility_thr,
        )
        core = _dilate(core, args.core_dilate)
        envelope = _dilate(envelope, args.envelope_dilate)
        diagnostic_label_available = labels[idx] is not None
        exact_fg = labels[idx] > 0 if diagnostic_label_available else np.zeros_like(source_fg, dtype=bool)
        uncertain_band = envelope & (~core)
        foreground_outside_envelope = exact_fg & (~envelope) if diagnostic_label_available else np.zeros_like(envelope)
        foreground_outside_core = exact_fg & (~core) if diagnostic_label_available else np.zeros_like(core)
        false_envelope = envelope & (~exact_fg) if diagnostic_label_available else np.zeros_like(envelope)

        cv2.imwrite(str(coverage_dir / f"frame_{frame_id:06d}_core.png"), (core.astype(np.uint8) * 255))
        cv2.imwrite(str(coverage_dir / f"frame_{frame_id:06d}_envelope.png"), (envelope.astype(np.uint8) * 255))
        cv2.imwrite(
            str(coverage_dir / f"frame_{frame_id:06d}_foreground_outside_envelope.png"),
            (foreground_outside_envelope.astype(np.uint8) * 255),
        )

        conf_tile = _put_label(_heat_overlay(frames_rgb[idx], conf_np[idx]), f"{idx:02d} id={frame_id} conf")
        vis_tile = _put_label(_heat_overlay(frames_rgb[idx], vis_np[idx]), f"{idx:02d} id={frame_id} vis")
        coverage_tile = frames_rgb[idx].copy()
        coverage_tile = _overlay_mask(coverage_tile, envelope, (255, 196, 0), 0.45)
        coverage_tile = _overlay_mask(coverage_tile, core, (0, 220, 60), 0.55)
        coverage_tile = _put_label(coverage_tile, f"{idx:02d} core/envelope")
        proxy_tile = frames_rgb[idx].copy()
        proxy_tile = _overlay_mask(proxy_tile, uncertain_band, (255, 196, 0), 0.35)
        if diagnostic_label_available:
            proxy_tile = _overlay_mask(proxy_tile, foreground_outside_core, (255, 80, 0), 0.55)
            proxy_tile = _overlay_mask(proxy_tile, foreground_outside_envelope, (255, 0, 0), 0.75)
            proxy_label = f"{idx:02d} red=fg outside envelope"
        else:
            proxy_label = f"{idx:02d} diagnostic label unavailable"
        proxy_tile = _put_label(proxy_tile, proxy_label)

        confidence_sheet_tiles.append(conf_tile)
        visibility_sheet_tiles.append(vis_tile)
        coverage_sheet_tiles.append(coverage_tile)
        proxy_sheet_tiles.append(proxy_tile)

        exact_area = int(np.count_nonzero(exact_fg))
        source_area = int(np.count_nonzero(source_fg))
        envelope_area = int(np.count_nonzero(envelope))
        core_area = int(np.count_nonzero(core))
        records.append(
            {
                "chunk_frame_index": idx,
                "frame_id": frame_id,
                "source_foreground_area": source_area,
                "diagnostic_label_available": bool(diagnostic_label_available),
                "exact_foreground_area": exact_area,
                "core_area": core_area,
                "envelope_area": envelope_area,
                "uncertain_band_area": int(np.count_nonzero(uncertain_band)),
                "foreground_outside_core_area": int(np.count_nonzero(foreground_outside_core)),
                "foreground_outside_envelope_area": int(np.count_nonzero(foreground_outside_envelope)),
                "false_envelope_area": int(np.count_nonzero(false_envelope)),
                "foreground_core_recall": (
                    _safe_ratio(int(np.count_nonzero(exact_fg & core)), exact_area)
                    if diagnostic_label_available
                    else None
                ),
                "foreground_envelope_recall": (
                    _safe_ratio(int(np.count_nonzero(exact_fg & envelope)), exact_area)
                    if diagnostic_label_available
                    else None
                ),
                "foreground_outside_envelope_ratio": _safe_ratio(
                    int(np.count_nonzero(foreground_outside_envelope)), exact_area
                )
                if diagnostic_label_available
                else None,
                "false_envelope_area_ratio": (
                    _safe_ratio(int(np.count_nonzero(false_envelope)), exact_area)
                    if diagnostic_label_available
                    else None
                ),
                "confidence_mean": float(np.mean(conf_np[idx])),
                "confidence_p05": float(np.quantile(conf_np[idx], 0.05)),
                "confidence_p50": float(np.quantile(conf_np[idx], 0.5)),
                "confidence_p95": float(np.quantile(conf_np[idx], 0.95)),
                "visibility_mean": float(np.mean(vis_np[idx])),
                "visibility_p05": float(np.quantile(vis_np[idx], 0.05)),
                "visibility_p50": float(np.quantile(vis_np[idx], 0.5)),
                "visibility_p95": float(np.quantile(vis_np[idx], 0.95)),
            }
        )

    confidence_sheet = output_root / f"{args.scene_id}_alltracker_confidence_sheet.jpg"
    visibility_sheet = output_root / f"{args.scene_id}_alltracker_visibility_sheet.jpg"
    coverage_sheet = output_root / f"{args.scene_id}_alltracker_core_envelope_sheet.jpg"
    proxy_sheet = output_root / f"{args.scene_id}_alltracker_proxy_gap_sheet.jpg"
    _write_png(confidence_sheet, _contact_sheet(confidence_sheet_tiles, cols=args.sheet_cols))
    _write_png(visibility_sheet, _contact_sheet(visibility_sheet_tiles, cols=args.sheet_cols))
    _write_png(coverage_sheet, _contact_sheet(coverage_sheet_tiles, cols=args.sheet_cols))
    _write_png(proxy_sheet, _contact_sheet(proxy_sheet_tiles, cols=args.sheet_cols))

    row_count = len(records)
    available_records = [row for row in records if bool(row.get("diagnostic_label_available"))]
    min_envelope_recall = min((row["foreground_envelope_recall"] for row in available_records), default=None)
    min_core_recall = min((row["foreground_core_recall"] for row in available_records), default=None)
    mean_outside_envelope = (
        float(np.mean([row["foreground_outside_envelope_ratio"] for row in available_records]))
        if available_records
        else None
    )
    max_outside_envelope = max((row["foreground_outside_envelope_ratio"] for row in available_records), default=None)

    records_path = output_root / "alltracker_frame_records.json"
    _write_json(
        records_path,
        {
            "schema_version": "stream4d_v105_phase3_alltracker_frame_records_v1",
            "row_count": row_count,
            "rows": records,
        },
    )

    summary = {
        "schema_version": "stream4d_v105_phase3_alltracker_contract_summary_v1",
        "scene_id": args.scene_id,
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "source_label_dir": _rel(label_root),
        "source_label_frame0_only_allowed": bool(args.allow_missing_diagnostic_labels),
        "diagnostic_label_available_frame_count": int(len(available_records)),
        "diagnostic_label_missing_frame_count": int(len(frame_ids) - len(available_records)),
        "canonical_height": args.canonical_height,
        "canonical_width": args.canonical_width,
        "provider": "AllTracker",
        "provider_root": _rel(ALLTRACKER_ROOT),
        "wrapper_reference": _rel(REPO_ROOT / "third_party/4D_PM/frontend/alltracker/wrapper.py"),
        "model_class": type(model).__name__,
        "checkpoint": _rel(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "provider_code_sha256": _sha256_file(ALLTRACKER_ROOT / "nets" / "alltracker.py"),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "window_len": args.window_len,
        "relay_stride": args.relay_stride,
        "relay_windows": _make_windows(len(frame_ids), args.window_len, args.relay_stride),
        "inference_iters": args.inference_iters,
        "conf_thr": args.conf_thr,
        "visibility_thr": args.visibility_thr,
        "envelope_conf_thr": envelope_conf_thr,
        "envelope_visibility_thr": envelope_visibility_thr,
        "core_dilate": args.core_dilate,
        "envelope_dilate": args.envelope_dilate,
        "construction_runtime_sec": construction_runtime_sec,
        "forward_runtime_sec": forward_runtime_sec,
        "peak_memory_bytes": peak_memory_bytes,
        "parameter_count": _count_params(model),
        "flow_shape": list(flow_np.shape),
        "raw_flow_shape": list(flows[0].shape),
        "raw_visconf_shape": list(visconf[0].shape),
        "normalised_vis_shape": list(vis_np.shape),
        "inserted_identity_frames_for_two_frame_smoke": inserted_identity_frames,
        "no_nan": no_nan,
        "shape_ok": shape_ok,
        "provider_construction_smoke_pass": True,
        "provider_forward_smoke_pass": forward_smoke_pass,
        "relay_coverage_artifact_pass": row_count == len(frame_ids),
        "confidence_visibility_overlay_pass": confidence_sheet.exists() and visibility_sheet.exists(),
        "proxy_gap_sheet_pass": proxy_sheet.exists(),
        "contract_artifacts_complete": bool(
            forward_smoke_pass
            and row_count == len(frame_ids)
            and confidence_sheet.exists()
            and visibility_sheet.exists()
            and coverage_sheet.exists()
            and proxy_sheet.exists()
        ),
        "diagnostic_only": True,
        "final_mask_write_allowed": False,
        "final_identity_write_allowed": False,
        "records_json": _rel(records_path),
        "confidence_sheet": _rel(confidence_sheet),
        "visibility_sheet": _rel(visibility_sheet),
        "coverage_sheet": _rel(coverage_sheet),
        "proxy_gap_sheet": _rel(proxy_sheet),
        "min_foreground_envelope_recall": min_envelope_recall,
        "min_foreground_core_recall": min_core_recall,
        "mean_foreground_outside_envelope_ratio": mean_outside_envelope,
        "max_foreground_outside_envelope_ratio": max_outside_envelope,
        "notes": [
            "AllTracker is used only as approximate future coverage provider.",
            "Foreground recall compares proxy coverage against X1 label foreground as a diagnostic; it is not a tuned GT gate.",
            "Proxy gap sheets are visual diagnostics and do not alter SAM2 masks or object identities.",
        ],
    }
    summary_path = output_root / "alltracker_contract_summary.json"
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--rgb-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--x1-label-dir", required=True)
    parser.add_argument(
        "--allow-missing-diagnostic-labels",
        action="store_true",
        default=False,
        help="Allow only the frame0 source label to exist; future label-based proxy recall diagnostics are then written as unavailable.",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--checkpoint", default="third_party/alltracker/alltracker.pth")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--canonical-height", type=int, default=240)
    parser.add_argument("--canonical-width", type=int, default=320)
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--relay-stride", type=int, default=8)
    parser.add_argument("--inference-iters", type=int, default=4)
    parser.add_argument("--conf-thr", type=float, default=0.1)
    parser.add_argument("--visibility-thr", type=float, default=0.5)
    parser.add_argument("--envelope-conf-thr", type=float, default=None)
    parser.add_argument("--envelope-visibility-thr", type=float, default=None)
    parser.add_argument("--core-dilate", type=int, default=1)
    parser.add_argument("--envelope-dilate", type=int, default=3)
    parser.add_argument("--sheet-cols", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "summary": summary["contract_artifacts_complete"],
                "scene_id": summary["scene_id"],
                "forward_runtime_sec": summary["forward_runtime_sec"],
                "peak_memory_bytes": summary["peak_memory_bytes"],
                "min_foreground_envelope_recall": summary["min_foreground_envelope_recall"],
                "proxy_gap_sheet": summary["proxy_gap_sheet"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
