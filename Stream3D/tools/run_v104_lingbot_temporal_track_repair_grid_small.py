#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import run_v104_lingbot_temporal_track_local_mv_ap as base  # noqa: E402


DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v104_lingbot_map_only_phase10_temporal_track_repair_grid_small"


def _tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p").rstrip("0").rstrip("p")


def _make_variant(
    variant_id: str,
    *,
    sigma_3d: float,
    sigma_2d: float,
    w3d: float,
    area_frac: float,
    threshold: float,
    max_gap: int,
) -> dict[str, object]:
    remain = 1.0 - float(w3d)
    warea = remain * area_frac
    w2d = remain - warea
    return {
        "variant_id": variant_id,
        "sigma_3d": sigma_3d,
        "sigma_2d": sigma_2d,
        "sigma_log_area": 0.85,
        "w3d": w3d,
        "w2d": w2d,
        "warea": warea,
        "threshold": threshold,
        "max_gap": max_gap,
        "min_frames": 2,
        "non_broad_only": False,
    }


def _variants() -> list[dict[str, object]]:
    variants: list[dict[str, object]] = [
        _make_variant(
            "S0_2dshape_control_replay_s2d008_thr050_gap1",
            sigma_3d=1.0,
            sigma_2d=0.08,
            w3d=0.0,
            area_frac=0.30,
            threshold=0.50,
            max_gap=1,
        )
    ]
    for sigma_3d in [0.60, 0.90]:
        for w3d in [0.02, 0.05, 0.10, 0.15, 0.20]:
            for threshold in [0.45, 0.50, 0.55]:
                variants.append(
                    _make_variant(
                        (
                            f"S1_w3d{_tag(w3d)}_s3d{_tag(sigma_3d)}_"
                            f"s2d0p08_area0p30_thr{_tag(threshold)}_gap1"
                        ),
                        sigma_3d=sigma_3d,
                        sigma_2d=0.08,
                        w3d=w3d,
                        area_frac=0.30,
                        threshold=threshold,
                        max_gap=1,
                    )
                )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small v104 LingBot temporal repair grid near the 2D/shape control.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-root", default=str(base.DEFAULT_FEATURE_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(base.DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(base.DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(base.DEFAULT_BASELINE_ROWS))
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base.VARIANTS = _variants()
    forwarded = argparse.Namespace(
        output_root=args.output_root,
        feature_root=args.feature_root,
        scene0011_phase2_root=args.scene0011_phase2_root,
        scene0050_phase2_root=args.scene0050_phase2_root,
        baseline_rows=args.baseline_rows,
        variants="",
        min_pred_pixels=args.min_pred_pixels,
        min_gt_pixels=args.min_gt_pixels,
        cupy_device_id=args.cupy_device_id,
        disable_cupy_iou=args.disable_cupy_iou,
        force=args.force,
    )
    summary = base.build(forwarded)
    return 0 if summary.get("phase10_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
