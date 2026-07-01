from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v91_mask_feature_store import merge_mask_feature_stores  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = []
    for text in args.inputs:
        path = Path(text)
        inputs.append(path if path.is_absolute() else ROOT / path)
    output = Path(args.output_root)
    output = output if output.is_absolute() else ROOT / output
    manifest = merge_mask_feature_stores(
        inputs,
        output,
        metadata={
            "source": "merge_v91_radio_mask_feature_stores.py",
            "input_stores": [adaptive._rel(path) for path in inputs],
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    )
    print(json.dumps(adaptive._jsonable(manifest), indent=2, sort_keys=True), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge v91 RADIO mask feature NPZ stores.")
    parser.add_argument("--output-root", default="outputs/audit/v91_radio_mask_features_npz")
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
