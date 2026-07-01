#!/usr/bin/env python3
"""Audit the canonical MV_AP implementation contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V65 = ROOT / "tools/run_v65_scene_multiview_ap.py"
V89 = ROOT / "tools/run_v89_recalc_point_projected_mv_ap.py"
SCENE_ADAPTER = ROOT / "tools/build_v98_1_canonical_scene_metrics.py"

EXPECTED_THRESHOLDS = [round(float(x), 2) for x in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    _require(V65.exists(), f"missing canonical AP file: {V65}", errors)
    _require(V89.exists(), f"missing canonical window adapter: {V89}", errors)
    if V65.exists():
        module = _load_module(V65)
        thresholds = [round(float(x), 2) for x in getattr(module, "AP_THRESHOLDS", [])]
        _require(thresholds == EXPECTED_THRESHOLDS, f"AP_THRESHOLDS mismatch: {thresholds}", errors)
        _require(hasattr(module, "SparseSceneIoU"), "SparseSceneIoU missing from v65", errors)
        _require(hasattr(module, "_summarize_iou"), "_summarize_iou missing from v65", errors)
        _require(hasattr(module, "_ap_from_scores"), "_ap_from_scores missing from v65", errors)
    if V89.exists():
        text = V89.read_text(encoding="utf-8")
        _require("_evaluate_frame_mask_variant_local_window" in text, "local-window adapter function missing from v89", errors)
        _require("local_window_gt_projection" in text, "local-window support_policy missing from v89", errors)
        _require("_window_scoped_gt" in text, "window-scoped GT helper missing from v89", errors)
        _require("w{int(window_index):04d}|" in text, "window-scoped prediction id prefix missing from v89", errors)
    if SCENE_ADAPTER.exists():
        text = SCENE_ADAPTER.read_text(encoding="utf-8")
        _require("scene_level_raw_gt_no_window_split" in text, "scene adapter scope marker missing", errors)
        _require("_summarize_iou" in text and "SparseSceneIoU" in text, "scene adapter must call v65 AP core", errors)
        _require("window_scoped_gt" not in text, "scene adapter must not window-scope GT", errors)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: canonical MV_AP contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
