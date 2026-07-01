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


def serve(args: argparse.Namespace) -> dict[str, Any]:
    viewer_npz = Path(args.viewer_npz)
    payload = _load_npz(viewer_npz)
    gt_points = payload["gt_points"].astype(np.float32)
    gt_colors = payload["gt_colors"].astype(np.uint8)
    loger_points = payload["loger_points"].astype(np.float32)
    loger_colors = payload["loger_colors"].astype(np.uint8)
    if gt_points.ndim != 2 or gt_points.shape[1] != 3:
        raise ValueError(f"gt_points must be Nx3, got {gt_points.shape}")
    if loger_points.ndim != 2 or loger_points.shape[1] != 3:
        raise ValueError(f"loger_points must be Nx3, got {loger_points.shape}")

    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    gt_handle = server.scene.add_point_cloud(
        "/GT geometry",
        points=gt_points,
        colors=gt_colors,
        point_size=args.gt_point_size,
        point_shape="circle",
        visible=True,
    )
    loger_handle = server.scene.add_point_cloud(
        "/LoGeR geometry",
        points=loger_points,
        colors=loger_colors,
        point_size=args.loger_point_size,
        point_shape="circle",
        visible=True,
    )

    gt_toggle = server.gui.add_checkbox("GT geometry", True, hint="Show or hide ScanNet GT mesh point cloud.")
    loger_toggle = server.gui.add_checkbox("LoGeR geometry", True, hint="Show or hide Sim3-aligned LoGeR point cloud.")

    @gt_toggle.on_update
    def _(_: Any) -> None:
        gt_handle.visible = bool(gt_toggle.value)

    @loger_toggle.on_update
    def _(_: Any) -> None:
        loger_handle.visible = bool(loger_toggle.value)

    status = {
        "viewer": "loger_scannet_gt_loger",
        "viewer_npz": str(viewer_npz),
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{args.port}",
        "gt_point_count": int(gt_points.shape[0]),
        "loger_point_count": int(loger_points.shape[0]),
        "layers": ["GT geometry", "LoGeR geometry"],
        "toggles": ["GT geometry", "LoGeR geometry"],
    }
    if args.status_json:
        status_path = Path(args.status_json)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if args.smoke_seconds > 0:
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
    parser = argparse.ArgumentParser(description="Serve a two-layer GT/LoGeR ScanNet Viser viewer.")
    parser.add_argument("--viewer-npz", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--gt-point-size", type=float, default=0.015)
    parser.add_argument("--loger-point-size", type=float, default=0.012)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
