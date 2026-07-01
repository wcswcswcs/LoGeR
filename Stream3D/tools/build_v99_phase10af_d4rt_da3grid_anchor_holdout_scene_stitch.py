#!/usr/bin/env python3
"""Run Phase10Y D4RT anchor scene stitching with DA3-grid D4RT provider artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase10y_d4rt_anchor_holdout_scene_stitch as phase10y  # noqa: E402


def main() -> int:
    phase10y.OUT_DIR = AUDIT_ROOT / "v99_phase10af_d4rt_da3grid_anchor_holdout_scene_stitch"
    phase10y.D4RT_ROOTS = [
        AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0011",
        AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0050",
    ]
    return phase10y.main()


if __name__ == "__main__":
    raise SystemExit(main())
