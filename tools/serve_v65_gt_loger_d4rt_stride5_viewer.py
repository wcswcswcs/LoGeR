#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _status_path(path: str) -> Path | None:
    return Path(path) if path else None


def serve(args: argparse.Namespace) -> dict[str, Any]:
    loger = _load_npz(Path(args.loger_viewer_npz))
    d4rt = _load_npz(Path(args.d4rt_visual_npz))
    gt_points = loger["gt_points"].astype(np.float32)
    gt_colors = loger["gt_colors"].astype(np.uint8)
    loger_points = loger["loger_points"].astype(np.float32)
    loger_colors = loger["loger_colors"].astype(np.uint8)
    d4rt_points = d4rt["points"].astype(np.float32)
    if "colors" not in d4rt:
        raise KeyError(f"{args.d4rt_visual_npz} has no colors array; regenerate D4RT visual_points.npz with RGB export")
    d4rt_colors = d4rt["colors"].astype(np.uint8)

    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/v65_gt_loger_d4rt/grid",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    gt_handle = server.scene.add_point_cloud(
        "/v65_gt_loger_d4rt/GT geometry",
        points=gt_points,
        colors=gt_colors,
        point_size=args.gt_point_size,
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    loger_handle = server.scene.add_point_cloud(
        "/v65_gt_loger_d4rt/LoGeR stride5",
        points=loger_points,
        colors=loger_colors,
        point_size=args.loger_point_size,
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    d4rt_handle = server.scene.add_point_cloud(
        "/v65_gt_loger_d4rt/D4RT stride5 all-conf RGB",
        points=d4rt_points,
        colors=d4rt_colors,
        point_size=args.d4rt_point_size,
        point_shape="circle",
        visible=True,
        precision="float32",
    )

    gt_toggle = server.gui.add_checkbox("GT geometry", True)
    loger_toggle = server.gui.add_checkbox("LoGeR stride5", True)
    d4rt_toggle = server.gui.add_checkbox("D4RT stride5 all-conf RGB", True)

    @gt_toggle.on_update
    def _(_: Any) -> None:
        gt_handle.visible = bool(gt_toggle.value)

    @loger_toggle.on_update
    def _(_: Any) -> None:
        loger_handle.visible = bool(loger_toggle.value)

    @d4rt_toggle.on_update
    def _(_: Any) -> None:
        d4rt_handle.visible = bool(d4rt_toggle.value)

    status = {
        "viewer": "v65_gt_loger_d4rt_stride5_allconf",
        "url": f"http://localhost:{args.port}",
        "host": args.host,
        "port": int(args.port),
        "layers": ["GT geometry", "LoGeR stride5", "D4RT stride5 all-conf RGB"],
        "toggles": ["GT geometry", "LoGeR stride5", "D4RT stride5 all-conf RGB"],
        "gt_point_count": int(gt_points.shape[0]),
        "loger_point_count": int(loger_points.shape[0]),
        "d4rt_point_count": int(d4rt_points.shape[0]),
        "d4rt_color_source": "ScanNet RGB sampled at D4RT predicted uv/frame_id",
        "loger_viewer_npz": args.loger_viewer_npz,
        "d4rt_visual_npz": args.d4rt_visual_npz,
    }
    status_path = _status_path(args.status_json)
    if status_path is not None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GT / LoGeR stride5 / D4RT stride5 all-confidence RGB viewer.")
    parser.add_argument("--loger-viewer-npz", required=True)
    parser.add_argument("--d4rt-visual-npz", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--gt-point-size", type=float, default=0.006)
    parser.add_argument("--loger-point-size", type=float, default=0.009)
    parser.add_argument("--d4rt-point-size", type=float, default=0.014)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
