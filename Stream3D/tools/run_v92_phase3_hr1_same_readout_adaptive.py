#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v91_phase4_adaptive_uncertainty_materialization as v91_adaptive


DEFAULT_SUPPORT_ROWS = ROOT / "outputs/audit/v92_phase3_d4rt_highres/highres_native_carrier_support_rows.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/audit/v92_phase3_hr1_same_readout_adaptive_materialization"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_workspace_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO_ROOT / path
    return ROOT / path


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    support_rows = _resolve_workspace_path(args.support_rows)
    output_root = _resolve_workspace_path(args.output_root)
    if not support_rows.exists():
        raise FileNotFoundError(support_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    v91_adaptive.OUT = output_root
    v91_adaptive.SUPPORT_ROWS = support_rows
    summary = v91_adaptive.run(argparse.Namespace())
    wrapper = {
        "schema": "stream4d_v92_phase3_hr1_same_readout_adaptive_wrapper_v1",
        "phase_id": "v92_phase3_d4rt_highres",
        "run_id": "v92_phase3_hr1_same_readout_adaptive_materialization",
        "decision": summary.get("decision", ""),
        "support_rows": _rel(support_rows),
        "support_rows_sha256": _sha256(support_rows),
        "output_root": _rel(output_root),
        "wrapped_script": "tools/run_v91_phase4_adaptive_uncertainty_materialization.py",
        "wrapped_protocol": "same adaptive uncertainty materialization family, only SUPPORT_ROWS and OUT are rebound",
        "uses_gt_for_prediction": bool(summary.get("uses_gt_for_prediction", False)),
        "uses_future": bool(summary.get("uses_future", False)),
        "uses_rgbd_pose_mesh": False,
        "best_variant_id": summary.get("best_variant_id", ""),
        "best_delta_vs_phase8_best_MV_AP_window": summary.get("best_delta_vs_phase8_best_MV_AP_window", ""),
        "best_delta_vs_phase8_best_MV_AP50_window": summary.get("best_delta_vs_phase8_best_MV_AP50_window", ""),
        "runtime_sec": time.time() - started,
    }
    _write_json(output_root / "v92_wrapper_summary.json", wrapper)
    print(json.dumps(_jsonable(wrapper), indent=2, sort_keys=True), flush=True)
    return wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v91 adaptive same-readout materialization with v92 HR1 support rows.")
    parser.add_argument("--support-rows", type=Path, default=DEFAULT_SUPPORT_ROWS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
