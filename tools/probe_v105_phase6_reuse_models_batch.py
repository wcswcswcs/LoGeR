#!/usr/bin/env python3
"""Run multiple Phase6 chunks in one process while reusing SAM2 models.

This is a speed probe, not a promoted pipeline runner. It monkeypatches the
Phase6 module's setup_models symbol so repeated Phase6 runs on one GPU can
reuse the already constructed image predictor and video predictor. Outputs are
still written by the original Phase6 runner for each chunk.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_4dpm_style_per_frame_segmentors import sha256_file  # noqa: E402
from tools.build_v105_phase6_speculative_gap_birth import build_parser as build_phase6_parser  # noqa: E402
from tools.build_v105_phase6_speculative_gap_birth import run as run_phase6  # noqa: E402
import tools.build_v105_phase6_speculative_gap_birth as phase6_module  # noqa: E402


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _format_template(template: str, frame_start: int, frame_count: int) -> str:
    return str(template).format(frame_start=int(frame_start), frame_count=int(frame_count))


def _build_phase6_cli(args: argparse.Namespace, frame_start: int, frame_count: int, output_root: Path) -> argparse.Namespace:
    argv = [
        "--config",
        str(args.config),
        "--scene-id",
        str(args.scene_id),
        "--frame-start",
        str(int(frame_start)),
        "--frame-count",
        str(int(frame_count)),
        "--frame0-birth-records",
        _format_template(str(args.frame0_birth_records_template), frame_start, frame_count),
        "--alltracker-dir",
        _format_template(str(args.alltracker_dir_template), frame_start, frame_count),
        "--x0-summary",
        str(args.x0_summary),
        "--x1-summary",
        _format_template(str(args.x1_summary_template), frame_start, frame_count),
        "--output-root",
        str(output_root),
    ]
    argv.extend(args.phase6_args)
    return build_phase6_parser().parse_args(argv)


def run(args: argparse.Namespace) -> None:
    import torch

    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_starts = [int(part) for part in str(args.frame_starts).split(",") if part.strip()]
    if not frame_starts:
        raise ValueError("--frame-starts must contain at least one start frame")

    original_setup_models = phase6_module.setup_models
    cached_models: dict[str, Any] | None = None
    tracker_base_get_image_feature: Any | None = None
    tracker_base_forward_image: Any | None = None
    setup_call_count = 0
    setup_reuse_count = 0
    setup_build_sec = 0.0

    def cached_setup_models(baseline_args: Any) -> dict[str, Any]:
        nonlocal cached_models
        nonlocal setup_call_count, setup_reuse_count, setup_build_sec
        nonlocal tracker_base_get_image_feature, tracker_base_forward_image
        setup_call_count += 1
        if cached_models is None:
            t0 = time.time()
            cached_models = original_setup_models(baseline_args)
            setup_build_sec += time.time() - t0
            tracker_model = cached_models["tracker_model"]
            tracker_base_get_image_feature = tracker_model._get_image_feature
            tracker_base_forward_image = tracker_model.forward_image
            return cached_models
        setup_reuse_count += 1
        tracker_model = cached_models["tracker_model"]
        if tracker_base_get_image_feature is not None:
            tracker_model._get_image_feature = tracker_base_get_image_feature  # type: ignore[method-assign]
        if tracker_base_forward_image is not None:
            tracker_model.forward_image = tracker_base_forward_image  # type: ignore[method-assign]
        return cached_models

    rows: list[dict[str, Any]] = []
    total_t0 = time.time()
    phase6_module.setup_models = cached_setup_models
    try:
        for frame_start in frame_starts:
            chunk_output = output_root / f"phase6_{str(args.scene_id).split('_')[0]}_start{int(frame_start):04d}_f{int(args.frame_count)}"
            if chunk_output.exists() and not bool(args.force):
                raise FileExistsError({"output_exists": str(chunk_output), "repair": "use --force or a fresh --output-root"})
            cli = _build_phase6_cli(args, int(frame_start), int(args.frame_count), chunk_output)
            t0 = time.time()
            run_phase6(cli)
            row_sec = time.time() - t0
            summary_path = chunk_output / "phase6_speculative_gap_birth_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "frame_start": int(frame_start),
                    "frame_count": int(args.frame_count),
                    "output_root": str(chunk_output),
                    "summary": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "outer_runtime_sec": float(row_sec),
                    "phase6_total_runtime_sec": float(summary.get("total_runtime_sec", 0.0)),
                    "phase6_setup_sec_reported": float(summary.get("setup_sec", 0.0)),
                    "phase6_tracking_sec": float(summary.get("total_tracking_runtime_sec", 0.0)),
                    "phase6_birth_decode_sec": float(summary.get("total_birth_decode_runtime_sec", 0.0)),
                    "peak_cuda_memory_mb": float(summary.get("peak_cuda_memory_mb", 0.0)),
                }
            )
            torch.cuda.empty_cache()
    finally:
        phase6_module.setup_models = original_setup_models

    total_sec = time.time() - total_t0
    summary_out = {
        "schema_version": "stream4d_v105_phase6_reuse_models_batch_probe_v1",
        "scene_id": str(args.scene_id),
        "frame_starts": frame_starts,
        "frame_count": int(args.frame_count),
        "config": str(args.config),
        "output_root": str(output_root),
        "summary_name": str(args.summary_name),
        "phase6_args": list(args.phase6_args),
        "setup_call_count": int(setup_call_count),
        "setup_reuse_count": int(setup_reuse_count),
        "setup_build_sec": float(setup_build_sec),
        "total_outer_runtime_sec": float(total_sec),
        "rows": rows,
        "gpu_memory_peak_mb_after_batch": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
        "boundary": "Speed probe only; each chunk output is still produced by the original Phase6 runner.",
    }
    summary_path = output_root / str(args.summary_name)
    summary_path.write_text(json.dumps(summary_out, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "total_outer_runtime_sec": total_sec}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--frame-starts", required=True)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--config", default="configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml")
    parser.add_argument("--x0-summary", required=True)
    parser.add_argument("--x1-summary-template", required=True)
    parser.add_argument("--frame0-birth-records-template", required=True)
    parser.add_argument("--alltracker-dir-template", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--summary-name",
        default="phase6_reuse_models_batch_summary.json",
        help="Batch summary filename under --output-root. Useful when multiple GPU batches share one output root.",
    )
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument(
        "--phase6-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded to build_v105_phase6_speculative_gap_birth.py. Place this last.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
