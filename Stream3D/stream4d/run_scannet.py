from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .appearance_memory import attach_proposal_features
from .carrier_sampler import CarrierSampler
from .d4rt_adapter import D4RTAdapter
from .diagnostics import carrier_diagnostics, local_props_payload, save_overlay, write_json
from .export_scannet import ScanNetExporter
from .local_4d_filter import Local4DFilter
from .mask_evidence import MaskEvidenceBuilder
from .object_memory import ObjectMemory4D
from .object_memory_v2 import ObjectMemory4DV2
from .scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


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


def _seq_names(args: argparse.Namespace) -> list[str]:
    if args.seq_name:
        return [args.seq_name]
    if args.seq_list:
        with Path(args.seq_list).open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    raise ValueError("Provide --seq-name or --seq-list")


def _process_sequence(args: argparse.Namespace, adapter: D4RTAdapter) -> dict:
    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    pred_path = Path("data/prediction") / f"{args.output_config}_class_agnostic" / f"{args.seq_name}.npz"
    tmp_path = Path("data/TMP") / args.output_config / f"{args.seq_name}_pre_points.npy"
    if args.skip_existing and pred_path.exists() and tmp_path.exists():
        print(f"[stream4d] seq={args.seq_name} skip existing prediction={pred_path}", flush=True)
        return {
            "seq_name": args.seq_name,
            "backbone": args.backbone,
            "d4rt_config": str(args.d4rt_config),
            "d4rt_ckpt": str(args.d4rt_ckpt),
            "skipped_existing": True,
        }
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    all_frames = stream.frame_ids(stride=args.frame_stride, max_frames=args.max_frames)
    windows = _windows(all_frames, args.window_size, args.window_stride)
    debug_dir = Path(args.debug_root) / args.seq_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    sampler = CarrierSampler(
        max_points_per_mask=args.max_points_per_mask,
        min_points_per_mask=args.min_points_per_mask,
        strategy=args.sampling_strategy,
        seed=args.seed,
    )
    evidence_builder = MaskEvidenceBuilder(rho_min=args.rho_min)
    local_filter = Local4DFilter(local_ioc_threshold=args.local_ioc_threshold)
    if args.memory_version == "old":
        memory = ObjectMemory4D(
            history_match_threshold=args.history_match_threshold,
            lost_tolerance_windows=args.lost_tolerance_windows,
        )
    elif args.memory_version == "v2":
        memory = ObjectMemory4DV2(
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
    else:
        raise ValueError(f"Unsupported memory version: {args.memory_version}")
    support_uv: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    timeline: list[dict] = []

    t0 = time.time()
    for window_idx, frame_window in enumerate(windows):
        window_t0 = time.time()
        print(
            f"[stream4d] seq={args.seq_name} window={window_idx + 1}/{len(windows)} "
            f"frames={frame_window[0]}..{frame_window[-1]}",
            flush=True,
        )
        data = stream.load_window(frame_window)
        masks = data["mask"]
        sources = sampler.sample(masks=masks, frame_ids=frame_window)
        sources.save_npz(debug_dir / f"carrier_sources_window{window_idx:03d}.npz")

        batch = adapter.infer_carriers(
            video_rgb_uint8=data["rgb"],
            src_uv_norm=sources.src_uv,
            src_frame_local=sources.src_frame,
            carrier_id=sources.carrier_id,
            src_frame_global=sources.src_frame_global,
            src_xy=sources.src_xy,
            src_mask_id=sources.src_mask_id,
            query_chunk_size=args.query_chunk_size,
        )
        batch.save_npz(debug_dir / f"carriers_window{window_idx:03d}.npz")
        cache_diag = carrier_diagnostics(batch)
        cache_manifest = {
            "seq_name": args.seq_name,
            "window_id": int(window_idx),
            "frame_indices": [int(v) for v in frame_window],
            "raw_frame_ids": [int(v) for v in frame_window],
            "num_frames": int(len(frame_window)),
            "num_carriers": int(batch.carrier_id.shape[0]),
            "num_target_frames": int(batch.valid.shape[0]),
            "checkpoint_path": str(args.d4rt_ckpt),
            "checkpoint_sha256": "not_computed_in_run_scannet",
            "checkpoint_size_bytes": int(getattr(adapter, "checkpoint_size_bytes", 0)),
            "clip_frames": int(getattr(adapter, "clip_frames", 0)),
            "query_chunk_size": int(args.query_chunk_size),
            "uv_in01_rate": float(cache_diag.get("uv_in01_rate", 0.0)),
            "visibility_rate_mean": float(cache_diag.get("visibility_prob_mean", 0.0)),
            "confidence_mean": float(cache_diag.get("confidence_prob_mean", 0.0)),
            **getattr(adapter, "last_infer_diagnostics", {}),
        }
        write_json(debug_dir / f"carriers_window{window_idx:03d}_manifest.json", cache_manifest)

        observations, evidence_diag = evidence_builder.build(batch, masks=masks, frame_ids=frame_window)
        for obs in observations:
            for cid, uv, weight in zip(obs.carrier_ids.tolist(), obs.uv_norm, obs.weights.tolist()):
                support_uv[(int(obs.frame_id), int(cid))] = (uv.astype(np.float32), float(weight))
        proposals, local_diag = local_filter.run(observations)
        feature_diag: dict[str, float | str] = {}
        if args.memory_version == "v2":
            feature_diag = attach_proposal_features(
                proposals,
                rgb_window=data["rgb"],
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
            "num_sampled_carriers": int(sources.carrier_id.shape[0]),
            "seconds": float(time.time() - window_t0),
            **carrier_diagnostics(batch),
            **evidence_diag,
            **local_diag,
            **feature_diag,
            **memory_diag,
        }
        timeline.append(merged_diag)
        print(
            f"[stream4d] seq={args.seq_name} window={window_idx + 1} "
            f"carriers={merged_diag['num_sampled_carriers']} "
            f"props={int(merged_diag['num_local_proposals'])} "
            f"objects={int(merged_diag['num_objects'])} "
            f"sec={merged_diag['seconds']:.2f}",
            flush=True,
        )
        write_json(debug_dir / f"local_props_window{window_idx:03d}.json", local_props_payload(observations, proposals, merged_diag))

        if args.save_overlays:
            for local_idx, frame_id in enumerate(frame_window[: min(4, len(frame_window))]):
                save_overlay(
                    debug_dir / "overlays" / f"window{window_idx:03d}_frame{int(frame_id)}.png",
                    data["rgb"][local_idx],
                    batch.uv_pred[local_idx],
                )

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
    if args.export_mode == "rgbd_eval":
        export_diag = exporter.export_rgbd_eval(memory, support_uv=support_uv)
    elif args.export_mode == "d4rt_nn":
        export_diag = exporter.export_d4rt_nn(memory)
    else:
        raise ValueError(f"Unsupported export mode: {args.export_mode}")

    summary = {
        "seq_name": args.seq_name,
        "backbone": args.backbone,
        "d4rt_config": str(args.d4rt_config),
        "d4rt_ckpt": str(args.d4rt_ckpt),
        "memory_version": args.memory_version,
        "num_windows": int(len(windows)),
        "num_frames": int(len(all_frames)),
        "total_seconds": float(time.time() - t0),
        "timeline": timeline,
        "export": export_diag,
    }
    write_json(debug_dir / "summary.json", summary)
    rgbd_bridge = args.export_mode == "rgbd_eval"
    manifest = build_prediction_manifest(
        output_config=args.output_config,
        is_method_result=not rgbd_bridge,
        is_diagnostic_only=rgbd_bridge,
        uses_gt=False,
        gt_usage="none",
        source_configs=[],
        pre_points_policy="recompute",
        support_policy=args.export_support_mode,
        notes=(
            "Generated by stream4d.run_scannet. export_mode=rgbd_eval uses ScanNet RGB-D/pose/mesh "
            "as a bridge to materialize predictions and is diagnostic-only under the v21.3 native guard."
            if rgbd_bridge
            else "Generated by stream4d.run_scannet from D4RT-native prediction outputs; no GT is used."
        ),
        extra={
            "seq_scope": args.seq_name,
            "d4rt_config": str(args.d4rt_config),
            "d4rt_ckpt": str(args.d4rt_ckpt),
            "memory_version": args.memory_version,
            "num_windows": int(len(windows)),
            "num_frames": int(len(all_frames)),
            "uses_rgbd_for_prediction": bool(rgbd_bridge),
            "uses_pose_for_prediction": bool(rgbd_bridge),
            "uses_scannet_mesh_for_prediction": bool(rgbd_bridge),
            "uses_gt_for_prediction": False,
            "uses_gt_sim3_for_prediction": False,
            "uses_rgbd_for_evaluation": bool(rgbd_bridge),
            "forbidden_for_method_table": bool(rgbd_bridge),
            "geometry_source": "rgbd_eval_bridge" if rgbd_bridge else "d4rt_native",
            "alignment_source": "none",
            "chunking_policy": f"manual_window_size_{args.window_size}_stride_{args.window_stride}",
            "opend4rt_reference_policy": "legacy_stream4d_adapter",
            "is_method_result": not rgbd_bridge,
            "is_diagnostic_only": rgbd_bridge,
        },
    )
    write_prediction_manifest(args.output_config, manifest)
    print(
        f"[stream4d] seq={args.seq_name} done objects={int(export_diag['num_exported_objects'])} "
        f"points={int(export_diag['num_exported_points'])} "
        f"hit_rate={export_diag['export_nn_hit_rate']:.4f} total_sec={summary['total_seconds']:.2f}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--seq-name", default="")
    parser.add_argument("--seq-list", default="")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--max-points-per-mask", type=int, default=32)
    parser.add_argument("--min-points-per-mask", type=int, default=4)
    parser.add_argument("--sampling-strategy", default="uniform_mask_pixels")
    parser.add_argument("--rho-min", type=float, default=0.35)
    parser.add_argument("--local-ioc-threshold", type=float, default=0.25)
    parser.add_argument("--history-match-threshold", type=float, default=0.30)
    parser.add_argument("--lost-tolerance-windows", type=int, default=3)
    parser.add_argument("--memory-version", default="old", choices=["old", "v2"])
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
    parser.add_argument("--export-mode", default="rgbd_eval", choices=["rgbd_eval", "d4rt_nn"])
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
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
    parser.add_argument("--export-score-mode", default="one", choices=["one", "area"])
    parser.add_argument("--output-config", default="stream4d_scannet")
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--debug-root", default="outputs/stream4d_debug")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seqs = _seq_names(args)
    adapter = D4RTAdapter(
        d4rt_root=args.d4rt_root,
        model_config=args.d4rt_config,
        ckpt_path=args.d4rt_ckpt,
        device=args.device,
    )
    summaries = []
    errors = []
    for seq_name in seqs:
        args.seq_name = seq_name
        print(f"[stream4d] start seq={seq_name}", flush=True)
        try:
            summaries.append(_process_sequence(args, adapter))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(f"[stream4d][ERROR] seq={seq_name} {message}", flush=True)
            errors.append({"seq_name": seq_name, "error": message})
    write_json(Path(args.debug_root) / f"{args.output_config}_run_summary.json", {"summaries": summaries, "errors": errors})


if __name__ == "__main__":
    main()
