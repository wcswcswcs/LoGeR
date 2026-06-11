from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream
from tools.export_d4rt_grid_surfel_field_v8 import _grid_sources


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def _stats(name: str, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    diff = np.abs(left - right)
    finite = np.isfinite(diff)
    values = diff[finite]
    if values.size == 0:
        return {
            f"{name}_num_finite": 0,
            f"{name}_max_abs": None,
            f"{name}_mean_abs": None,
            f"{name}_p50_abs": None,
            f"{name}_p90_abs": None,
            f"{name}_p99_abs": None,
        }
    return {
        f"{name}_num_finite": int(values.size),
        f"{name}_max_abs": float(np.max(values)),
        f"{name}_mean_abs": float(np.mean(values)),
        f"{name}_p50_abs": float(np.percentile(values, 50)),
        f"{name}_p90_abs": float(np.percentile(values, 90)),
        f"{name}_p99_abs": float(np.percentile(values, 99)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# v8 D4RT Adapter vs Official Helper Diagnostic",
        "",
        "This is a Lane 1 blocker diagnostic. It does not read GT labels and does not report AP.",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Diff Stats", ""])
    for key, value in payload.get("diff_stats", {}).items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--seq-name", default="scene0050_00")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--grid-margin-ratio", type=float, default=0.02)
    parser.add_argument("--mask-aware-min-points-per-mask", type=int, default=0)
    parser.add_argument("--min-mask-area", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    d4rt_root = Path(args.d4rt_root).resolve()
    sys.path.insert(0, str(d4rt_root))
    from infer_track_3d import _infer_tracks, _resize_video

    adapter = D4RTAdapter(
        d4rt_root=d4rt_root,
        model_config=args.d4rt_config,
        ckpt_path=args.d4rt_ckpt,
        device=args.device,
    )
    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone, root=args.scannet_root)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    frame_ids = stream.frame_ids(stride=int(args.frame_stride), max_frames=int(args.max_frames))
    data = stream.load_window(frame_ids)
    sources, source_diag = _grid_sources(
        masks=np.asarray(data["mask"]),
        frame_ids=frame_ids,
        grid_size=int(args.grid_size),
        grid_margin_ratio=float(args.grid_margin_ratio),
        mask_aware_min_points_per_mask=int(args.mask_aware_min_points_per_mask),
        min_mask_area=int(args.min_mask_area),
    )

    adapter_batch = adapter.infer_carriers(
        video_rgb_uint8=np.asarray(data["rgb"]),
        src_uv_norm=sources.src_uv,
        src_frame_local=sources.src_frame,
        carrier_id=sources.carrier_id,
        src_frame_global=sources.src_frame_global,
        src_xy=sources.src_xy,
        src_mask_id=sources.src_mask_id,
        query_chunk_size=int(args.query_chunk_size),
    )

    video_model_rgb = _resize_video(np.asarray(data["rgb"]), adapter.image_hw).astype(np.uint8)
    official = _infer_tracks(
        model=adapter.model,
        video_model_rgb=video_model_rgb,
        query_uv_norm=sources.src_uv,
        query_src_indices_global=sources.src_frame,
        query_chunk_size=int(args.query_chunk_size),
    )

    adapter_uv_qt2 = np.transpose(adapter_batch.uv_pred, (1, 0, 2))
    adapter_xyz_qt3 = np.transpose(adapter_batch.xyz_ref, (1, 0, 2))
    adapter_vis_qt = np.transpose(adapter_batch.visibility_prob, (1, 0))
    adapter_conf_qt = np.transpose(adapter_batch.confidence_prob, (1, 0))
    official_vis_prob = _sigmoid_np(np.asarray(official["tracks_visibility_logits"], dtype=np.float32))
    official_conf_prob = _sigmoid_np(np.asarray(official["tracks_confidence"], dtype=np.float32))

    diff_stats: dict[str, Any] = {}
    diff_stats.update(_stats("uv", adapter_uv_qt2, official["tracks_uv_norm"]))
    diff_stats.update(_stats("xyz_ref0", adapter_xyz_qt3, official["tracks_xyz_ref0"]))
    diff_stats.update(_stats("visibility_prob", adapter_vis_qt, official_vis_prob))
    diff_stats.update(_stats("confidence_prob", adapter_conf_qt, official_conf_prob))

    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "uses_gt": False,
        "is_method_result": False,
        "purpose": "adapter-vs-official-helper-equivalence",
        "python": sys.executable,
        "d4rt_root": str(d4rt_root),
        "seq_name": args.seq_name,
        "frame_ids": [int(v) for v in frame_ids],
        "grid_size": int(args.grid_size),
        "grid_margin_ratio": float(args.grid_margin_ratio),
        "num_queries": int(sources.src_uv.shape[0]),
        "num_frames": int(len(frame_ids)),
        "adapter_aspect_policy": "native_scan_width_over_height",
        "official_helper_aspect_policy": "resized_clip_width_over_height",
        "adapter_last_infer": adapter.last_infer_diagnostics,
        "source_diagnostics": source_diag,
        "diff_stats": diff_stats,
    }

    prefix = Path(args.output_prefix)
    _write_json(prefix.with_suffix(".json"), payload)
    _write_markdown(prefix.with_suffix(".md"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
