#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v106.pipeline import V106Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stream4D v106 sequential SAM2 objectlet pipeline stages.")
    parser.add_argument(
        "--config",
        default="configs/v106/streaming_sam2_objectlet_default.yaml",
        help="v106 YAML/JSON config path.",
    )
    parser.add_argument(
        "--stages",
        default="phase0,phase1",
        help="Comma-separated stages to run. Supported: phase0,phase1,phase2,phase3,phase4,phase5,phase6,phase7,phase8,phase9,phase10.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for v106 artifacts.",
    )
    parser.add_argument("--force", action="store_true", help="Replace output-root if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = V106Pipeline(
        repo_root=REPO_ROOT,
        config_path=(REPO_ROOT / args.config),
        output_root=(REPO_ROOT / args.output_root),
        force=args.force,
    )
    result = pipeline.run(args.stages.split(","))
    print(f"wrote {result['output_root']}")
    print(f"all_requested_stages_pass={result['all_requested_stages_pass']}")


if __name__ == "__main__":
    main()
