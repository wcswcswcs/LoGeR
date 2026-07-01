#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "Stream3D"
    / "outputs"
    / "audit"
    / "v99_phase10aa_da3_d4rt_sim3_alignment_scene0011"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _load_layers(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def serve(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    summary_path = Path(args.summary_json)
    layers_path = Path(args.layers_npz)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    layers = _load_layers(layers_path)

    import viser  # type: ignore

    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/v99_da3_d4rt_alignment/grid",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    da3_dense = server.scene.add_point_cloud(
        "/v99_da3_d4rt_alignment/DA3 dense",
        points=layers["da3_dense_points"].astype(np.float32),
        colors=layers["da3_dense_colors"].astype(np.uint8),
        point_size=float(args.da3_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    da3_corr = server.scene.add_point_cloud(
        "/v99_da3_d4rt_alignment/DA3 correspondence",
        points=layers["da3_correspondence_points"].astype(np.float32),
        colors=layers["da3_correspondence_colors"].astype(np.uint8),
        point_size=float(args.correspondence_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    d4rt_raw = server.scene.add_point_cloud(
        "/v99_da3_d4rt_alignment/D4RT raw self-stitched",
        points=layers["d4rt_raw_points"].astype(np.float32),
        colors=layers["d4rt_raw_colors"].astype(np.uint8),
        point_size=float(args.d4rt_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    d4rt_aligned = server.scene.add_point_cloud(
        "/v99_da3_d4rt_alignment/D4RT aligned to DA3 Sim3",
        points=layers["d4rt_aligned_points"].astype(np.float32),
        colors=layers["d4rt_aligned_colors"].astype(np.uint8),
        point_size=float(args.d4rt_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )

    handles = {
        "DA3 dense": da3_dense,
        "DA3 correspondence": da3_corr,
        "D4RT raw self-stitched": d4rt_raw,
        "D4RT aligned to DA3 Sim3": d4rt_aligned,
    }
    toggles = {
        name: server.gui.add_checkbox(name, bool(handle.visible))
        for name, handle in handles.items()
    }

    for name, toggle in toggles.items():
        @toggle.on_update
        def _(_: Any, layer_name: str = name) -> None:
            handles[layer_name].visible = bool(toggles[layer_name].value)

    sim3 = summary.get("sim3_d4rt_to_da3", {})
    metrics = summary.get("metrics", {})
    aligned_res = metrics.get("aligned_pair_residual_m", {})
    raw_res = metrics.get("raw_pair_residual_m", {})
    server.gui.add_text("scene", str(summary.get("scene_id", "")), disabled=True)
    server.gui.add_number("Sim3 scale", float(sim3.get("scale", 0.0) or 0.0), disabled=True)
    server.gui.add_number("raw p90", float(raw_res.get("p90", 0.0) or 0.0), disabled=True)
    server.gui.add_number("aligned p90", float(aligned_res.get("p90", 0.0) or 0.0), disabled=True)
    server.gui.add_number("aligned points", int(layers["d4rt_aligned_points"].shape[0]), disabled=True)
    server.gui.add_number("DA3 dense points", int(layers["da3_dense_points"].shape[0]), disabled=True)

    status = {
        "phase": "v99_phase10aa_da3_d4rt_sim3_alignment_viewer",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{int(args.port)}",
        "summary_json": str(summary_path),
        "layers_npz": str(layers_path),
        "layers": {
            "DA3 dense": int(layers["da3_dense_points"].shape[0]),
            "DA3 correspondence": int(layers["da3_correspondence_points"].shape[0]),
            "D4RT raw self-stitched": int(layers["d4rt_raw_points"].shape[0]),
            "D4RT aligned to DA3 Sim3": int(layers["d4rt_aligned_points"].shape[0]),
        },
        "contract": summary.get("contract", {}),
        "sim3_d4rt_to_da3": {
            "scale": sim3.get("scale"),
            "rotation_det": sim3.get("rotation_det"),
            "translation_norm": sim3.get("translation_norm"),
        },
        "metrics": {
            "raw_pair_residual_m": raw_res,
            "aligned_pair_residual_m": aligned_res,
        },
    }
    status_path = output_root / "viewer_status.json"
    _write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True, default=_json_default), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary-json", default=str(DEFAULT_OUTPUT_ROOT / "summary.json"))
    parser.add_argument("--layers-npz", default=str(DEFAULT_OUTPUT_ROOT / "da3_d4rt_sim3_alignment_layers.npz"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--da3-point-size", type=float, default=0.006)
    parser.add_argument("--d4rt-point-size", type=float, default=0.009)
    parser.add_argument("--correspondence-point-size", type=float, default=0.007)
    return parser.parse_args()


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
