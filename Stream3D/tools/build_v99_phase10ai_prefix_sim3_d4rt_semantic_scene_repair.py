#!/usr/bin/env python3
"""Run Phase10Z semantic+D4RT repair using Phase10AH prefix-Sim3-aligned D4RT candidates."""

from __future__ import annotations

import sys
from pathlib import Path


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase10z_d4rt_verifier_semantic_scene_repair as phase10z  # noqa: E402


def main() -> int:
    phase10z.OUT_DIR = AUDIT_ROOT / "v99_phase10ai_prefix_sim3_d4rt_semantic_scene_repair"
    phase10z.PHASE10Y_DIR = AUDIT_ROOT / "v99_phase10ah_prefix_sim3_aligned_anchor_scene_stitch"
    return phase10z.main()


if __name__ == "__main__":
    raise SystemExit(main())
