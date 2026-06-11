from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .appearance_memory import attach_proposal_features
from .carrier_store import CarrierBatch
from .diagnostics import carrier_diagnostics, local_props_payload, write_json
from .export_scannet import ScanNetExporter
from .local_4d_filter import Local4DFilter
from .mask_evidence import MaskEvidenceBuilder
from .object_memory import ObjectMemory4D
from .object_memory_v2 import ObjectMemory4DV2
from .scannet_stream import ScanNetStream


def _windows(frame_ids: list[int], window_size: int, window_stride: int) -> list[list[int]]:
    if not frame_ids:
        return []
    window_size = max(1, int(window_size))
    window_stride = max(1, int(window_stride))
    out: list[list[int]] = []
    start = 0
    while start < len(frame_ids):
        cur = frame_ids[start : start + window_size]
        if cur:
            out.append(cur)
        if start + window_size >= len(frame_ids):
            break
        start += window_stride
    return out


def _load_carrier_batch(path: Path) -> CarrierBatch:
    payload = np.load(path)
    return CarrierBatch(
        carrier_id=payload["carrier_id"],
        src_frame=payload["src_frame"],
        src_uv=payload["src_uv"],
        xyz_ref=payload["xyz_ref"],
        uv_pred=payload["uv_pred"],
        visibility_prob=payload["visibility_prob"],
        confidence_prob=payload["confidence_prob"],
        valid=payload["valid"],
        xyz_local=payload["xyz_local"] if "xyz_local" in payload.files else None,
        src_frame_global=payload["src_frame_global"] if "src_frame_global" in payload.files else None,
        src_xy=payload["src_xy"] if "src_xy" in payload.files else None,
        src_mask_id=payload["src_mask_id"] if "src_mask_id" in payload.files else None,
    )


def _load_rgb_mask_window(stream: ScanNetStream, frame_window: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rgbs = [stream.load_rgb(frame_id) for frame_id in frame_window]
    masks = [stream.load_mask(frame_id) for frame_id in frame_window]
    return np.stack(rgbs, axis=0), np.stack(masks, axis=0)


def _build_memory(args: argparse.Namespace):
    if args.memory_version == "old":
        return ObjectMemory4D(
            history_match_threshold=args.history_match_threshold,
            lost_tolerance_windows=args.lost_tolerance_windows,
        )
    if args.memory_version == "v2":
        return ObjectMemory4DV2(
            history_match_threshold=args.history_match_threshold,
            lost_tolerance_windows=args.lost_tolerance_windows,
            carrier_weight=args.memory_v2_carrier_weight,
            appearance_weight=args.memory_v2_appearance_weight,
            geometry_weight=args.memory_v2_geometry_weight,
            motion_weight=args.memory_v2_motion_weight,
            conflict_weight=args.memory_v2_conflict_weight,
            geometry_sigma=args.memory_v2_geometry_sigma,
            motion_sigma=args.memory_v2_motion_sigma,
            min_carrier_score=args.memory_v2_min_carrier_score,
        )
    raise ValueError(f"Unsupported memory version: {args.memory_version}")


def _process_sequence(args: argparse.Namespace) -> dict:
    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))

    all_frames = stream.frame_ids(stride=args.frame_stride, max_frames=args.max_frames)
    windows = _windows(all_frames, args.window_size, args.window_stride)
    input_debug_dir = Path(args.input_debug_root) / args.seq_name
    debug_dir = Path(args.debug_root) / args.seq_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    evidence_builder = MaskEvidenceBuilder(rho_min=args.rho_min)
    local_filter = Local4DFilter(local_ioc_threshold=args.local_ioc_threshold)
    memory = _build_memory(args)
    support_uv: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    timeline: list[dict] = []
    t0 = time.time()

    for window_idx, frame_window in enumerate(windows):
        window_t0 = time.time()
        carrier_path = input_debug_dir / f"carriers_window{window_idx:03d}.npz"
        if not carrier_path.exists():
            raise FileNotFoundError(f"Missing replay carrier file: {carrier_path}")
        batch = _load_carrier_batch(carrier_path)
        rgb, masks = _load_rgb_mask_window(stream, frame_window)
        observations, evidence_diag = evidence_builder.build(batch, masks=masks, frame_ids=frame_window)
        for obs in observations:
            for cid, uv, weight in zip(obs.carrier_ids.tolist(), obs.uv_norm, obs.weights.tolist()):
                support_uv[(int(obs.frame_id), int(cid))] = (uv.astype(np.float32), float(weight))
        proposals, local_diag = local_filter.run(observations)
        feature_diag: dict[str, float | str] = {}
        if args.memory_version == "v2":
            feature_diag = attach_proposal_features(
                proposals,
                rgb_window=rgb,
                masks_window=masks,
                frame_ids=frame_window,
                bins=args.memory_v2_appearance_bins,
                max_pixels_per_mask=args.memory_v2_appearance_max_pixels_per_mask,
                max_masks_per_proposal=args.memory_v2_appearance_max_masks_per_proposal,
            )
        memory_diag = memory.update(proposals, window_index=window_idx)
        merged_diag = {
            "window_index": int(window_idx),
            "frame_start": int(frame_window[0]),
            "frame_end": int(frame_window[-1]),
            "num_frames": int(len(frame_window)),
            "seconds": float(time.time() - window_t0),
            **carrier_diagnostics(batch),
            **evidence_diag,
            **local_diag,
            **feature_diag,
            **memory_diag,
        }
        timeline.append(merged_diag)
        print(
            f"[stream4d-replay] seq={args.seq_name} window={window_idx + 1}/{len(windows)} "
            f"props={int(merged_diag['num_local_proposals'])} "
            f"objects={int(merged_diag['num_objects'])} sec={merged_diag['seconds']:.2f}",
            flush=True,
        )
        write_json(debug_dir / f"local_props_window{window_idx:03d}.json", local_props_payload(observations, proposals, merged_diag))

    write_json(debug_dir / "object_memory.json", {"timeline": timeline, **memory.to_jsonable()})
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=args.export_nn_radius,
        export_support_mode=args.export_support_mode,
        export_mask_sample_stride=args.export_mask_sample_stride,
        export_mask_max_pixels=args.export_mask_max_pixels,
        export_max_masks_per_object=args.export_max_masks_per_object,
        export_mask_min_relative_coverage=args.export_mask_min_relative_coverage,
        export_core_nn_radius=args.export_core_nn_radius,
        export_fringe_nn_radius=args.export_fringe_nn_radius,
        export_fringe_radius=args.export_fringe_radius,
        export_fringe_max_ratio=args.export_fringe_max_ratio,
        export_point_dilate_radius=args.export_point_dilate_radius,
        export_min_points_per_object=args.export_min_points_per_object,
        export_score_mode=args.export_score_mode,
    )
    export_diag = exporter.export_rgbd_eval(memory, support_uv=support_uv)
    summary = {
        "seq_name": args.seq_name,
        "backbone": args.backbone,
        "input_debug_root": args.input_debug_root,
        "memory_version": args.memory_version,
        "num_windows": int(len(windows)),
        "num_frames": int(len(all_frames)),
        "total_seconds": float(time.time() - t0),
        "timeline": timeline,
        "export": export_diag,
    }
    write_json(debug_dir / "summary.json", summary)
    print(
        f"[stream4d-replay] seq={args.seq_name} done objects={int(export_diag['num_exported_objects'])} "
        f"points={int(export_diag['num_exported_points'])} "
        f"hit_rate={export_diag['export_nn_hit_rate']:.4f} total_sec={summary['total_seconds']:.2f}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--rho-min", type=float, default=0.35)
    parser.add_argument("--local-ioc-threshold", type=float, default=0.25)
    parser.add_argument("--history-match-threshold", type=float, default=0.30)
    parser.add_argument("--lost-tolerance-windows", type=int, default=3)
    parser.add_argument("--memory-version", default="v2", choices=["old", "v2"])
    parser.add_argument("--memory-v2-carrier-weight", type=float, default=0.55)
    parser.add_argument("--memory-v2-appearance-weight", type=float, default=0.25)
    parser.add_argument("--memory-v2-geometry-weight", type=float, default=0.20)
    parser.add_argument("--memory-v2-motion-weight", type=float, default=0.0)
    parser.add_argument("--memory-v2-conflict-weight", type=float, default=0.30)
    parser.add_argument("--memory-v2-geometry-sigma", type=float, default=0.35)
    parser.add_argument("--memory-v2-motion-sigma", type=float, default=0.35)
    parser.add_argument("--memory-v2-min-carrier-score", type=float, default=0.0)
    parser.add_argument("--memory-v2-appearance-bins", type=int, default=8)
    parser.add_argument("--memory-v2-appearance-max-pixels-per-mask", type=int, default=2048)
    parser.add_argument("--memory-v2-appearance-max-masks-per-proposal", type=int, default=8)
    parser.add_argument("--export-nn-radius", type=float, default=0.08)
    parser.add_argument(
        "--export-support-mode",
        default="carrier_uv",
        choices=["carrier_uv", "mask_backproject", "hybrid", "core_fringe", "component_densify"],
    )
    parser.add_argument("--export-mask-sample-stride", type=int, default=1)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-max-masks-per-object", type=int, default=0)
    parser.add_argument("--export-mask-min-relative-coverage", type=float, default=0.0)
    parser.add_argument("--export-core-nn-radius", type=float, default=None)
    parser.add_argument("--export-fringe-nn-radius", type=float, default=None)
    parser.add_argument("--export-fringe-radius", type=float, default=0.05)
    parser.add_argument("--export-fringe-max-ratio", type=float, default=0.35)
    parser.add_argument("--export-point-dilate-radius", type=float, default=0.0)
    parser.add_argument("--export-min-points-per-object", type=int, default=0)
    parser.add_argument("--export-score-mode", default="one", choices=["one", "area", "reliability", "observations", "dense_quality"])
    parser.add_argument("--input-debug-root", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--debug-root", default="outputs/stream4d_replay")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = _process_sequence(args)
    write_json(Path(args.debug_root) / f"{args.output_config}_run_summary.json", {"summaries": [summary], "errors": []})


if __name__ == "__main__":
    main()
