from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .carrier_store import CarrierBatch
from .diagnostics import carrier_diagnostics, write_json
from .evidence_graph import EvidenceGraphBuilder, EvidenceGraphMemory
from .export_scannet import ScanNetExporter
from .mask_evidence import MaskEvidenceBuilder, MaskObservation
from .replay_memory import _load_carrier_batch, _windows
from .scannet_stream import ScanNetStream


def _load_masks_window(stream: ScanNetStream, frame_window: list[int]) -> np.ndarray:
    return np.stack([stream.load_mask(frame_id) for frame_id in frame_window], axis=0)


def _collect_observations(
    args: argparse.Namespace,
    stream: ScanNetStream,
    windows: list[list[int]],
) -> tuple[list[MaskObservation], dict[tuple[int, int], tuple[np.ndarray, float]], list[dict[str, float]]]:
    input_debug_dir = Path(args.input_debug_root) / args.seq_name
    evidence_builder = MaskEvidenceBuilder(rho_min=args.rho_min)
    observations_all: list[MaskObservation] = []
    support_uv: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    timeline: list[dict[str, float]] = []
    for window_idx, frame_window in enumerate(windows):
        t0 = time.time()
        carrier_path = input_debug_dir / f"carriers_window{window_idx:03d}.npz"
        if not carrier_path.exists():
            raise FileNotFoundError(f"Missing replay carrier file: {carrier_path}")
        batch: CarrierBatch = _load_carrier_batch(carrier_path)
        masks = _load_masks_window(stream, frame_window)
        observations, evidence_diag = evidence_builder.build(batch, masks=masks, frame_ids=frame_window)
        for obs in observations:
            for cid, uv, weight in zip(obs.carrier_ids.tolist(), obs.uv_norm, obs.weights.tolist()):
                key = (int(obs.frame_id), int(cid))
                prev = support_uv.get(key)
                if prev is None or float(weight) > float(prev[1]):
                    support_uv[key] = (uv.astype(np.float32), float(weight))
        observations_all.extend(observations)
        row = {
            "window_index": float(window_idx),
            "frame_start": float(frame_window[0]),
            "frame_end": float(frame_window[-1]),
            "num_frames": float(len(frame_window)),
            "seconds": float(time.time() - t0),
            **carrier_diagnostics(batch),
            **evidence_diag,
        }
        timeline.append(row)
        print(
            f"[evidence-graph] seq={args.seq_name} window={window_idx + 1}/{len(windows)} "
            f"observations={len(observations)} sec={row['seconds']:.2f}",
            flush=True,
        )
    return observations_all, support_uv, timeline


def _process_sequence(args: argparse.Namespace) -> dict:
    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    all_frames = stream.frame_ids(stride=args.frame_stride, max_frames=args.max_frames)
    windows = _windows(all_frames, args.window_size, args.window_stride)
    debug_dir = Path(args.debug_root) / args.seq_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    observations, support_uv, timeline = _collect_observations(args, stream, windows)
    builder = EvidenceGraphBuilder(
        min_shared_carriers=args.graph_min_shared_carriers,
        min_carrier_ioc=args.graph_min_carrier_ioc,
        min_component_observations=args.graph_min_component_observations,
        min_component_carriers=args.graph_min_component_carriers,
        min_node_carriers=args.graph_min_node_carriers,
        min_node_coverage=args.graph_min_node_coverage,
        edge_coverage_power=args.graph_edge_coverage_power,
    )
    result = builder.build(observations)
    memory = EvidenceGraphMemory(result.objects, result.diagnostics)
    write_json(debug_dir / "object_memory.json", memory.to_jsonable())

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
        export_enable_wta=args.export_enable_wta,
        export_wta_score_mode=args.export_wta_score_mode,
        export_wta_min_conflict_owners=args.export_wta_min_conflict_owners,
        densify_boundary_erosion=args.densify_boundary_erosion,
        densify_small_mask_area=args.densify_small_mask_area,
        densify_seed_distance_px=args.densify_seed_distance_px,
        densify_min_seed_pixels=args.densify_min_seed_pixels,
        densify_seed_keep_mode=args.densify_seed_keep_mode,
        densify_seed_min_support_views=args.densify_seed_min_support_views,
        densify_mask_selection_mode=args.densify_mask_selection_mode,
    )
    export_diag = exporter.export_rgbd_eval(memory, support_uv=support_uv)
    summary = {
        "seq_name": args.seq_name,
        "backbone": args.backbone,
        "memory_version": "evidence_graph",
        "input_debug_root": args.input_debug_root,
        "num_windows": int(len(windows)),
        "num_frames": int(len(all_frames)),
        "total_seconds": float(time.time() - t0),
        "timeline": timeline,
        "graph": result.diagnostics,
        "export": export_diag,
    }
    write_json(debug_dir / "summary.json", summary)
    write_json(Path(args.debug_root) / f"{args.output_config}_run_summary.json", {"summaries": [summary], "errors": []})
    print(
        f"[evidence-graph] seq={args.seq_name} done objects={int(export_diag['num_exported_objects'])} "
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
    parser.add_argument("--graph-min-shared-carriers", type=int, default=2)
    parser.add_argument("--graph-min-carrier-ioc", type=float, default=0.50)
    parser.add_argument("--graph-min-component-observations", type=int, default=1)
    parser.add_argument("--graph-min-component-carriers", type=int, default=1)
    parser.add_argument("--graph-min-node-carriers", type=int, default=1)
    parser.add_argument("--graph-min-node-coverage", type=float, default=0.0)
    parser.add_argument("--graph-edge-coverage-power", type=float, default=0.0)
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
    parser.add_argument(
        "--export-score-mode",
        default="one",
        choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"],
    )
    parser.add_argument("--export-enable-wta", action="store_true")
    parser.add_argument(
        "--export-wta-score-mode",
        default="evidence_density",
        choices=["evidence_quality", "evidence_density", "observations", "carriers", "compactness"],
    )
    parser.add_argument("--export-wta-min-conflict-owners", type=int, default=2)
    parser.add_argument("--densify-boundary-erosion", type=int, default=1)
    parser.add_argument("--densify-small-mask-area", type=int, default=400)
    parser.add_argument("--densify-seed-distance-px", type=float, default=32.0)
    parser.add_argument("--densify-min-seed-pixels", type=int, default=1)
    parser.add_argument(
        "--densify-seed-keep-mode",
        default="none",
        choices=["none", "supported", "boundary", "component", "all"],
    )
    parser.add_argument("--densify-seed-min-support-views", type=int, default=1)
    parser.add_argument(
        "--densify-mask-selection-mode",
        default="coverage",
        choices=[
            "coverage",
            "seed_density",
            "component_seed_density",
            "kept_seed_density",
            "coverage_component_density",
            "coverage_kept_density",
            "kept_ratio",
        ],
    )
    parser.add_argument("--input-debug-root", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--debug-root", default="outputs/stream4d_evidence_graph")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _process_sequence(args)


if __name__ == "__main__":
    main()
