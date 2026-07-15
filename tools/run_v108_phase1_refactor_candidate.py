#!/usr/bin/env python3
"""Run a v108 Phase1 refactor candidate and optional label parity check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v108.rolling_baseline import (  # noqa: E402
    RollingBaselineCase,
    run_phase1_refactor_candidate,
)


DEFAULT_CONFIG = "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-label-dir", default="")
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--no-label-only-visual-export", action="store_true", default=False)
    parser.add_argument("--no-compact-visual-video", action="store_true", default=False)
    args = parser.parse_args()

    reference_label_dir = None
    if args.reference_label_dir.strip():
        reference_label_dir = REPO_ROOT / args.reference_label_dir
    case = RollingBaselineCase(
        case_name=args.case_name,
        scene_id=args.scene_id,
        frame_start=int(args.frame_start),
        frame_stride=int(args.frame_stride),
        frame_count=int(args.frame_count),
        gpu=str(args.gpu),
        config=args.config,
        output_root=REPO_ROOT / args.output_root,
        reference_label_dir=reference_label_dir,
        model_dtype=args.model_dtype,
        label_only_visual_export=not bool(args.no_label_only_visual_export),
        compact_visual_video=not bool(args.no_compact_visual_video),
    )
    result = run_phase1_refactor_candidate(case, REPO_ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") != "OK":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
