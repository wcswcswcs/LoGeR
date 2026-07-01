#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "Stream3D" / "outputs" / "audit" / "v98_1_da3_variant_geometry_quality_scene0050"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _add_toggle(server: viser.ViserServer, label: str, handle: Any, visible: bool) -> None:
    toggle = server.gui.add_checkbox(label, visible)

    @toggle.on_update
    def _(_: Any) -> None:
        handle.visible = bool(toggle.value)


def _metric_snippet(row: dict[str, Any]) -> dict[str, Any]:
    surface = row["surface_refined_sim3"]["geometry_metrics"]
    pose = row["pose_orientation_sim3"]["geometry_metrics"]
    return {
        "variant_key": row["variant_key"],
        "display_name": row["display_name"],
        "surface_refined_chamfer_l2_mean_m": surface["chamfer_l2_mean_m"],
        "surface_refined_accuracy_p90_m": surface["accuracy_da3_to_gt_m"]["p90"],
        "surface_refined_completeness_p90_m": surface["completeness_gt_to_da3_m"]["p90"],
        "surface_refined_fscore_0p10m": surface["fscore"]["0.10m"]["fscore"],
        "pose_orientation_chamfer_l2_mean_m": pose["chamfer_l2_mean_m"],
        "pose_orientation_accuracy_p90_m": pose["accuracy_da3_to_gt_m"]["p90"],
        "pose_orientation_completeness_p90_m": pose["completeness_gt_to_da3_m"]["p90"],
        "pose_orientation_fscore_0p10m": pose["fscore"]["0.10m"]["fscore"],
    }


def _d4rt_metric_snippet(summary: dict[str, Any]) -> dict[str, Any] | None:
    row = summary.get("d4rt_geometry")
    if not isinstance(row, dict):
        return None
    metrics = row.get("geometry_metrics_against_input_visible_gt")
    if not isinstance(metrics, dict):
        return None
    return {
        "display_name": row.get("display_name", "D4RT self-stitched"),
        "transform": row.get("csv_row", {}).get("transform", "overlap_self_stitch_then_depth_pose_sim3"),
        "chamfer_l2_mean_m": metrics.get("chamfer_l2_mean_m"),
        "accuracy_p90_m": metrics.get("accuracy_da3_to_gt_m", {}).get("p90"),
        "completeness_p90_m": metrics.get("completeness_gt_to_da3_m", {}).get("p90"),
        "fscore_0p10m": metrics.get("fscore", {}).get("0.10m", {}).get("fscore"),
    }


def _d4rt_frame_slice(
    points: np.ndarray,
    colors: np.ndarray,
    frame_ids: np.ndarray,
    unique_frames: np.ndarray,
    frame_slider_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    slider_index = int(np.clip(frame_slider_index, 0, max(int(unique_frames.shape[0]) - 1, 0)))
    frame_id = int(unique_frames[slider_index])
    keep = np.asarray(frame_ids, dtype=np.int64) == int(frame_id)
    return (
        np.asarray(points[keep], dtype=np.float32),
        np.asarray(colors[keep], dtype=np.uint8),
        frame_id,
    )


def serve(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    summary_path = Path(args.summary_json) if args.summary_json else output_root / "geometry_quality_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    npz_path = Path(args.viewer_npz) if args.viewer_npz else Path(summary["outputs"]["viewer_npz"])
    payload = np.load(npz_path)

    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=float(args.grid_width),
        height=float(args.grid_width),
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )

    gt = server.scene.add_point_cloud(
        "/GT ScanNet mesh RGB",
        points=np.asarray(payload["gt_points"], dtype=np.float32),
        colors=np.asarray(payload["gt_colors"], dtype=np.uint8),
        point_size=float(args.gt_point_size),
        point_shape="circle",
        visible=True,
    )
    _add_toggle(server, "GT ScanNet mesh RGB", gt, True)

    layers: list[str] = ["GT ScanNet mesh RGB"]
    for row in summary["variants"]:
        key = row["variant_key"]
        display_name = row["display_name"]
        colors = np.asarray(payload[f"{key}_colors"], dtype=np.uint8)
        surface_visible = key == args.default_variant
        surface = server.scene.add_point_cloud(
            f"/{display_name} RGB surface-refined Sim3",
            points=np.asarray(payload[f"{key}_surface_points"], dtype=np.float32),
            colors=colors,
            point_size=float(args.da3_point_size),
            point_shape="circle",
            visible=surface_visible,
        )
        pose = server.scene.add_point_cloud(
            f"/{display_name} RGB pose-orientation Sim3",
            points=np.asarray(payload[f"{key}_pose_points"], dtype=np.float32),
            colors=colors,
            point_size=float(args.da3_point_size),
            point_shape="circle",
            visible=False,
        )
        _add_toggle(server, f"{display_name} surface-refined", surface, surface_visible)
        _add_toggle(server, f"{display_name} pose-orientation", pose, False)
        layers.extend(
            [
                f"{display_name} RGB surface-refined Sim3",
                f"{display_name} RGB pose-orientation Sim3",
            ]
        )

    d4rt_info = summary.get("d4rt_geometry")
    d4rt_visible = bool(isinstance(d4rt_info, dict) and "d4rt_points" in payload.files)
    d4rt_status: dict[str, Any] | None = None
    if d4rt_visible:
        d4rt_display_name = str(d4rt_info.get("display_name", "D4RT self-stitched"))
        d4rt_layer_name = f"{d4rt_display_name} RGB depth-pose Sim3"
        d4rt_points = np.asarray(payload["d4rt_points"], dtype=np.float32)
        d4rt_colors = np.asarray(payload["d4rt_colors"], dtype=np.uint8)
        d4rt_frame_ids = np.asarray(payload["d4rt_frame_ids"], dtype=np.int64) if "d4rt_frame_ids" in payload.files else None
        frame_mode = str(args.d4rt_frame_mode)
        can_use_slider = d4rt_frame_ids is not None and d4rt_frame_ids.shape[0] == d4rt_points.shape[0] and np.unique(d4rt_frame_ids).shape[0] > 1
        if frame_mode in {"slider", "slider_with_aggregate"} and can_use_slider:
            unique_frames = np.unique(d4rt_frame_ids)
            initial_index = int(np.argmin(np.abs(unique_frames.astype(np.int64) - int(args.d4rt_initial_frame))))
            frame_points, frame_colors, initial_frame_id = _d4rt_frame_slice(
                d4rt_points,
                d4rt_colors,
                d4rt_frame_ids,
                unique_frames,
                initial_index,
            )
            d4rt = server.scene.add_point_cloud(
                f"/{d4rt_layer_name} frame",
                points=frame_points,
                colors=frame_colors,
                point_size=float(args.d4rt_point_size),
                point_shape="circle",
                visible=True,
            )
            _add_toggle(server, f"{d4rt_display_name} frame", d4rt, True)
            frame_slider = server.gui.add_slider(
                "D4RT frame index",
                min=0,
                max=max(int(unique_frames.shape[0]) - 1, 0),
                step=1,
                initial_value=int(initial_index),
            )

            @frame_slider.on_update
            def _(_: Any) -> None:
                selected_points, selected_colors, _frame_id = _d4rt_frame_slice(
                    d4rt_points,
                    d4rt_colors,
                    d4rt_frame_ids,
                    unique_frames,
                    int(frame_slider.value),
                )
                d4rt.points = selected_points
                d4rt.colors = selected_colors

            layers.append(f"{d4rt_layer_name} frame-slider")
            d4rt_status = {
                "frame_mode": frame_mode,
                "aggregate_point_count": int(d4rt_points.shape[0]),
                "unique_frame_count": int(unique_frames.shape[0]),
                "initial_slider_index": int(initial_index),
                "initial_frame_id": int(initial_frame_id),
                "initial_frame_point_count": int(frame_points.shape[0]),
                "frame_ids_min": int(unique_frames.min()),
                "frame_ids_max": int(unique_frames.max()),
            }
            if frame_mode == "slider_with_aggregate":
                d4rt_aggregate = server.scene.add_point_cloud(
                    f"/{d4rt_layer_name} aggregate",
                    points=d4rt_points,
                    colors=d4rt_colors,
                    point_size=float(args.d4rt_point_size),
                    point_shape="circle",
                    visible=False,
                )
                _add_toggle(server, f"{d4rt_display_name} aggregate", d4rt_aggregate, False)
                layers.append(f"{d4rt_layer_name} aggregate")
        else:
            d4rt = server.scene.add_point_cloud(
                f"/{d4rt_layer_name}",
                points=d4rt_points,
                colors=d4rt_colors,
                point_size=float(args.d4rt_point_size),
                point_shape="circle",
                visible=True,
            )
            _add_toggle(server, d4rt_display_name, d4rt, True)
            layers.append(d4rt_layer_name)
            d4rt_status = {
                "frame_mode": "aggregate",
                "aggregate_point_count": int(d4rt_points.shape[0]),
                "requested_frame_mode": frame_mode,
                "can_use_slider": bool(can_use_slider),
            }

    d4rt_metric = _d4rt_metric_snippet(summary)
    status = {
        "viewer": "v98_1_da3_variant_geometry_viewer",
        "pid": int(os.getpid()),
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{args.port}",
        "scene_id": summary["scene_id"],
        "summary_json": str(summary_path),
        "viewer_npz": str(npz_path),
        "diagnostic_only": True,
        "default_visible_variant": args.default_variant,
        "layers": layers,
        "metric_snippets": [_metric_snippet(row) for row in summary["variants"]],
        "d4rt_metric_snippet": d4rt_metric,
        "d4rt_viewer": d4rt_status,
    }
    if args.status_json:
        status_path = Path(args.status_json)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True, default=_json_default), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if float(args.smoke_seconds) > 0.0:
        deadline = time.time() + float(args.smoke_seconds)
        while time.time() < deadline:
            time.sleep(0.25)
        server.stop()
        return status
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve v98.1 DA3 variant geometry comparison in Viser.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--viewer-npz", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--default-variant", default="large")
    parser.add_argument("--gt-point-size", type=float, default=0.008)
    parser.add_argument("--da3-point-size", type=float, default=0.010)
    parser.add_argument("--d4rt-point-size", type=float, default=0.010)
    parser.add_argument("--d4rt-frame-mode", choices=["aggregate", "slider", "slider_with_aggregate"], default="aggregate")
    parser.add_argument("--d4rt-initial-frame", type=int, default=0)
    parser.add_argument("--grid-width", type=float, default=8.0)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
