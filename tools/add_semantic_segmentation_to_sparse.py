#!/usr/bin/env python3
"""Append dense semantic label maps to a sparse_masklets_v1 payload.

The added semantic segmentation lives at the top level and intentionally does
not change the existing track list. This keeps sparse instance masklets
backward-compatible while giving downstream models a direct [T,H,W] semantic
label tensor.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add top-level semantic_segmentation to sparse_masklets_v1 PT.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--confidence_pt", default="")
    parser.add_argument("--metrics_json", default="")
    parser.add_argument("--source", default="videomt_sam31_fusion")
    parser.add_argument("--void_label", default="void")
    parser.add_argument(
        "--base_source_types",
        default="stuff_static,structure_tracked",
        help="Comma-separated source_type values rendered before thing tracks.",
    )
    parser.add_argument(
        "--thing_source_types",
        default="thing_tracked",
        help="Comma-separated source_type values rendered after base tracks and allowed to override them.",
    )
    parser.add_argument(
        "--label_order",
        choices=["first_seen", "alphabetical"],
        default="first_seen",
        help="Controls semantic label id assignment after the void label.",
    )
    parser.add_argument("--overwrite", type=int, default=1)
    return parser.parse_args()


def _split_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _label_of(track: Dict[str, Any]) -> str:
    return str(track.get("L_sem", track.get("label", ""))).strip()


def _source_type_of(track: Dict[str, Any]) -> str:
    return str(track.get("source_type", "")).strip()


def _packed_to_numpy(packed: Any) -> np.ndarray:
    if isinstance(packed, torch.Tensor):
        return packed.detach().cpu().numpy().astype(np.uint8, copy=False)
    return np.asarray(packed, dtype=np.uint8)


def _unpack_mask(packed: Any, height: int, width: int) -> np.ndarray:
    arr = _packed_to_numpy(packed)
    return np.unpackbits(arr, bitorder="big", count=int(height) * int(width)).reshape(int(height), int(width)).astype(bool)


def _track_sort_key(
    indexed_track: Tuple[int, Dict[str, Any]],
    base_source_types: set[str],
    thing_source_types: set[str],
) -> Tuple[int, int]:
    index, track = indexed_track
    source_type = _source_type_of(track)
    if source_type in base_source_types:
        priority = 0
    elif source_type in thing_source_types:
        priority = 2
    else:
        priority = 1
    return priority, int(index)


def _collect_labels(
    tracks: Sequence[Dict[str, Any]],
    void_label: str,
    label_order: str,
) -> Tuple[List[str], Dict[str, int]]:
    labels: List[str] = [str(void_label)]
    seen = {str(void_label)}
    if str(label_order) == "alphabetical":
        ordered = sorted({_label_of(track) for track in tracks if _label_of(track)})
    else:
        ordered = []
        for track in tracks:
            label = _label_of(track)
            if label and label not in ordered:
                ordered.append(label)
    for label in ordered:
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels, {label: idx for idx, label in enumerate(labels)}


def _semantic_dtype(num_labels: int) -> Tuple[np.dtype, torch.dtype, str]:
    if int(num_labels) <= 255:
        return np.dtype(np.uint8), torch.uint8, "uint8"
    return np.dtype(np.int16), torch.int16, "int16"


def _render_semantic_maps(
    payload: Dict[str, Any],
    labels: Sequence[str],
    label_to_id: Dict[str, int],
    base_source_types: set[str],
    thing_source_types: set[str],
    confidence_maps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    height = int(payload["frame_height"])
    width = int(payload["frame_width"])
    num_frames = int(payload["num_frames"])
    np_dtype, torch_dtype, dtype_name = _semantic_dtype(len(labels))
    label_maps = np.zeros((num_frames, height, width), dtype=np_dtype)
    confidence_tensor: Optional[torch.Tensor] = None
    if confidence_maps is not None:
        confidence_tensor = confidence_maps.detach().cpu().float().clone()
        expected = (num_frames, height, width)
        if tuple(int(x) for x in confidence_tensor.shape) != expected:
            raise SystemExit(
                f"confidence_maps shape mismatch: got {tuple(confidence_tensor.shape)} expected {expected}"
            )

    tracks = list(payload.get("tracks", []))
    rendered_frames = 0
    skipped_frames = 0
    skipped_out_of_range = 0
    thing_override_pixels = 0
    thing_override_frames = 0
    pixels_written_by_label: Counter[str] = Counter()
    frames_written_by_label: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter(_source_type_of(track) for track in tracks)
    label_counts: Counter[str] = Counter(_label_of(track) for track in tracks)

    ordered_tracks = sorted(enumerate(tracks), key=lambda item: _track_sort_key(item, base_source_types, thing_source_types))
    for _track_index, track in ordered_tracks:
        label = _label_of(track)
        if not label:
            continue
        label_id = int(label_to_id[label])
        source_type = _source_type_of(track)
        frames = [int(x) for x in track.get("frames", [])]
        packed_masks = list(track.get("packed_masks", []))
        for idx, frame_idx in enumerate(frames):
            if idx >= len(packed_masks):
                skipped_frames += 1
                continue
            if frame_idx < 0 or frame_idx >= num_frames:
                skipped_out_of_range += 1
                continue
            mask = _unpack_mask(packed_masks[idx], height, width)
            if not mask.any():
                skipped_frames += 1
                continue
            label_maps[frame_idx][mask] = label_id
            if confidence_tensor is not None and source_type in thing_source_types:
                frame_conf = confidence_tensor[frame_idx]
                frame_max = float(frame_conf.max().item()) if frame_conf.numel() else 0.0
                frame_conf[mask] = frame_max
                confidence_tensor[frame_idx] = frame_conf
                thing_override_pixels += int(mask.sum())
                thing_override_frames += 1
            rendered_frames += 1
            pixels_written_by_label[label] += int(mask.sum())
            frames_written_by_label[label] += 1

    tensor = torch.as_tensor(label_maps, dtype=torch_dtype)
    debug = {
        "semantic_dtype": dtype_name,
        "rendered_track_frames": int(rendered_frames),
        "skipped_empty_or_missing_track_frames": int(skipped_frames),
        "skipped_out_of_range_track_frames": int(skipped_out_of_range),
        "input_track_label_counts": dict(sorted(label_counts.items())),
        "input_track_source_type_counts": dict(sorted(source_type_counts.items())),
        "frames_written_by_label": dict(sorted(frames_written_by_label.items())),
        "pixels_written_by_label": dict(sorted(pixels_written_by_label.items())),
        "nonvoid_pixels": int(np.count_nonzero(label_maps)),
        "total_pixels": int(label_maps.size),
        "thing_override_pixels": int(thing_override_pixels),
        "thing_override_frames": int(thing_override_frames),
    }
    if confidence_tensor is not None:
        debug.update(
            {
                "confidence_dtype": str(confidence_tensor.dtype),
                "confidence_shape": list(confidence_tensor.shape),
                "confidence_min": float(confidence_tensor.min().item()) if confidence_tensor.numel() else 0.0,
                "confidence_max": float(confidence_tensor.max().item()) if confidence_tensor.numel() else 0.0,
                "confidence_mean": float(confidence_tensor.mean().item()) if confidence_tensor.numel() else 0.0,
            }
        )
    return tensor, confidence_tensor, debug


def _load_confidence_maps(
    confidence_pt: Path,
    *,
    total_frames: int,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    payload = torch.load(confidence_pt, map_location="cpu", weights_only=False)
    semantic = payload.get("semantic_segmentation", {}) if isinstance(payload, dict) else {}
    confidence_maps = None
    if isinstance(semantic, dict):
        confidence_maps = semantic.get("confidence_maps")
    if confidence_maps is None and isinstance(payload, dict):
        confidence_maps = payload.get("confidence_maps")
    if confidence_maps is None:
        raise SystemExit(f"{confidence_pt} missing semantic_segmentation.confidence_maps")
    if isinstance(confidence_maps, torch.Tensor):
        maps = confidence_maps.detach().cpu()
    else:
        maps = torch.as_tensor(confidence_maps)
    expected = (int(total_frames), int(height), int(width))
    if int(maps.ndim) != 3 or tuple(int(x) for x in maps.shape) != expected:
        raise SystemExit(f"confidence_maps shape mismatch: got {tuple(maps.shape)} expected {expected}")
    maps = maps.float().contiguous()
    debug = {
        "confidence_pt": str(confidence_pt),
        "confidence_shape": list(maps.shape),
        "confidence_dtype": str(maps.dtype),
        "confidence_min": float(maps.min().item()) if maps.numel() else 0.0,
        "confidence_max": float(maps.max().item()) if maps.numel() else 0.0,
        "confidence_mean": float(maps.mean().item()) if maps.numel() else 0.0,
    }
    return maps, debug


def main() -> None:
    args = parse_args()
    input_pt = Path(args.input_pt)
    output_pt = Path(args.output_pt)
    if output_pt.exists() and not int(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing output: {output_pt}")

    payload = torch.load(input_pt, map_location="cpu", weights_only=False)
    if payload.get("format") != "sparse_masklets_v1":
        raise ValueError(f"Unsupported input format: {payload.get('format')}")

    base_source_types = set(_split_csv(args.base_source_types))
    thing_source_types = set(_split_csv(args.thing_source_types))
    labels, label_to_id = _collect_labels(payload.get("tracks", []), str(args.void_label), str(args.label_order))
    confidence_maps = None
    confidence_debug: Dict[str, Any] = {}
    if str(args.confidence_pt).strip():
        confidence_maps, confidence_debug = _load_confidence_maps(
            Path(args.confidence_pt),
            total_frames=int(payload["num_frames"]),
            height=int(payload["frame_height"]),
            width=int(payload["frame_width"]),
        )
    label_maps, confidence_maps, render_debug = _render_semantic_maps(
        payload,
        labels,
        label_to_id,
        base_source_types,
        thing_source_types,
        confidence_maps=confidence_maps,
    )

    semantic = {
        "format": "semantic_label_maps_v1",
        "source": str(args.source),
        "frame_height": int(payload["frame_height"]),
        "frame_width": int(payload["frame_width"]),
        "num_frames": int(payload["num_frames"]),
        "label_names": list(labels),
        "label_to_id": dict(label_to_id),
        "label_maps": label_maps,
        "label_dtype": str(render_debug["semantic_dtype"]),
        "confidence_maps": confidence_maps if confidence_maps is not None else None,
        "confidence_source": confidence_debug.get("confidence_pt"),
        "confidence_policy": {
            "base_confidence_source": "videomt_vss_pred_confidence_maps",
            "thing_override_policy": "set thing-source pixels to per-frame max VidEoMT confidence",
            "thing_source_types": sorted(thing_source_types),
        },
        "priority_policy": {
            "void_id": 0,
            "base_source_types": sorted(base_source_types),
            "middle_source_types": "all source_type values not listed as base or thing",
            "thing_source_types": sorted(thing_source_types),
            "thing_override": True,
        },
        "provenance": {
            "input_pt": str(input_pt),
            "confidence_pt": confidence_debug.get("confidence_pt"),
            "source": str(args.source),
            "notes": "Dense semantic label maps are reconstructed from sparse tracks; confidence maps come from VidEoMT VSS and thing-source pixels are filled with the per-frame VidEoMT max confidence.",
        },
        "debug": render_debug,
    }
    if confidence_maps is not None:
        semantic["debug"].update(confidence_debug)
    payload["semantic_segmentation"] = semantic
    payload.setdefault("debug", {})
    payload["debug"]["semantic_segmentation_export"] = {
        "format": "semantic_segmentation_export_v1",
        "input_pt": str(input_pt),
        "output_pt": str(output_pt),
        "num_labels": int(len(labels)),
        "label_names": list(labels),
        "label_dtype": str(render_debug["semantic_dtype"]),
        "confidence_pt": confidence_debug.get("confidence_pt"),
        "priority_policy": semantic["priority_policy"],
        "confidence_policy": semantic["confidence_policy"],
        "render_debug": render_debug,
    }

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)

    summary = {
        "input_pt": str(input_pt),
        "output_pt": str(output_pt),
        "format": payload.get("format"),
        "num_frames": int(payload["num_frames"]),
        "frame_height": int(payload["frame_height"]),
        "frame_width": int(payload["frame_width"]),
        "num_tracks": int(len(payload.get("tracks", []))),
        "semantic_format": semantic["format"],
        "semantic_shape": list(label_maps.shape),
        "semantic_dtype": str(render_debug["semantic_dtype"]),
        "num_labels": int(len(labels)),
        "label_names": list(labels),
        "nonvoid_pixels": int(render_debug["nonvoid_pixels"]),
        "has_confidence_maps": confidence_maps is not None,
        "confidence_shape": list(confidence_maps.shape) if confidence_maps is not None else [],
        "confidence_dtype": str(confidence_maps.dtype) if confidence_maps is not None else "",
        "confidence_min": float(confidence_maps.min().item()) if confidence_maps is not None else None,
        "confidence_max": float(confidence_maps.max().item()) if confidence_maps is not None else None,
        "confidence_mean": float(confidence_maps.mean().item()) if confidence_maps is not None else None,
        "confidence_pt": confidence_debug.get("confidence_pt"),
        "thing_override_pixels": int(render_debug.get("thing_override_pixels", 0)),
        "thing_override_frames": int(render_debug.get("thing_override_frames", 0)),
        "priority_policy": semantic["priority_policy"],
    }
    metrics_path = Path(args.metrics_json) if str(args.metrics_json).strip() else output_pt.with_suffix(".semantic_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
