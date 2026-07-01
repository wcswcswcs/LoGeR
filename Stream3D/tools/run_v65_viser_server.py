#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_visualization_export import D4RT_DEBUG_ROOT, VIS_ROOT, run_v65_viser_server_status, serve_v65_viser


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v65 viser import/server smoke or serve exported scene data.")
    parser.add_argument("--output-root", default=VIS_ROOT)
    parser.add_argument("--serve", action="store_true", help="Start a live Viser server and keep it running.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind in --serve mode.")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind in --serve mode.")
    parser.add_argument("--scene", default="scene0050_00", help="Scene to show in --serve mode; use 'all' only for multi-scene audit.")
    parser.add_argument(
        "--pred-config",
        default="v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
        help="Prediction config rendered as the method/pred overlay in --serve mode.",
    )
    parser.add_argument("--d4rt-debug-root", default=D4RT_DEBUG_ROOT, help="D4RT debug cache used for aligned carrier layer.")
    parser.add_argument(
        "--d4rt-mode",
        default="self_stitched_scale_normalized_eval_sim3",
        help="D4RTCarrierProjectionProvider mode for the main aligned D4RT layer.",
    )
    parser.add_argument(
        "--max-d4rt-points",
        type=int,
        default=0,
        help="Aligned D4RT point cap; 0 means no sampling/cap.",
    )
    args = parser.parse_args()
    if args.serve:
        serve_v65_viser(
            args.output_root,
            host=args.host,
            port=args.port,
            scene_id=args.scene,
            pred_config=args.pred_config,
            d4rt_debug_root=args.d4rt_debug_root,
            d4rt_mode=args.d4rt_mode,
            max_d4rt_points=args.max_d4rt_points,
        )
        return
    status = run_v65_viser_server_status(args.output_root)
    print(
        {
            "summary": f"{args.output_root}/viser_server_status.json",
            "viser_import_ok": status["viser_import_ok"],
            "viser_version": status["viser_version"],
            "python_executable": status["python_executable"],
            "gate": status["gate"],
            "scene_point_clouds_added": status["scene_point_clouds_added"],
        }
    )


if __name__ == "__main__":
    main()
