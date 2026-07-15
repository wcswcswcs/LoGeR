#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HorizonStream pipeline with deterministic torch/CUDA settings for v115 parity checks."
    )
    parser.add_argument("--horizonstream-root", type=Path, default=Path("third_party/HorizonStream"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    if not args.pipeline_args:
        parser.error("pipeline arguments are required after --")
    return args


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)
    horizon_root = args.horizonstream_root.resolve()
    os.chdir(horizon_root)
    sys.path.insert(0, str(horizon_root))
    sys.argv = ["run_pipeline.py", *args.pipeline_args]
    import run_pipeline

    run_pipeline.main()


if __name__ == "__main__":
    main()
