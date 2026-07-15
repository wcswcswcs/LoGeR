#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_v103_supp_r5_phaseR5_3_gt_coverage_inconsistency as base  # noqa: E402


PHASE_ID = "v103_supp_r6_phaseR6_6_gt_coverage_inconsistency"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r6_phase6_gt_coverage_inconsistency"
DEFAULT_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_feature"
DEFAULT_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_local_ap"
DEFAULT_ANCHOR_ONLY_ROOT = AUDIT_ROOT / "_missing_v103_supp_r6_anchor_only_skip"


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    return base._jsonable(value)


def _patch_base() -> None:
    base.PHASE_ID = PHASE_ID
    base.DEFAULT_OUT = DEFAULT_OUT
    base.DEFAULT_FEATURE_ROOT = DEFAULT_FEATURE_ROOT
    base.DEFAULT_LOCAL_AP_ROOT = DEFAULT_LOCAL_AP_ROOT
    base.DEFAULT_ANCHOR_ONLY_ROOT = DEFAULT_ANCHOR_ONLY_ROOT


def build(args: argparse.Namespace) -> dict[str, Any]:
    _patch_base()
    summary = base.build(args)
    out = _project(args.output_root)
    summary = dict(summary)
    summary.pop("phase_r5_3_diag_complete", None)
    summary.update(
        {
            "schema_version": "stream4d_v103_supp_r6_phaseR6_6_gt_coverage_inconsistency_summary_v1",
            "phase_id": PHASE_ID,
            "decision": "DIAGNOSTIC_ONLY_R6_6_AFTER_R6_2_LOCAL_AP_NO_GO",
            "phase_r6_6_diag_complete": True,
            "r6_feature_root": base._rel(_project(args.feature_root)),
            "r6_local_ap_root": base._rel(_project(args.local_ap_root)),
            "truthfulness_note": (
                "R6-6 is GT-only coverage and same-object multi-view fragmentation diagnostic. "
                "It does not pick thresholds, does not alter method predictions, and does not by itself prove method success."
            ),
        }
    )
    base._write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R6-6 GT coverage and 3D inconsistency diagnostic.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--local-ap-root", default=str(DEFAULT_LOCAL_AP_ROOT))
    parser.add_argument("--anchor-only-root", default=str(DEFAULT_ANCHOR_ONLY_ROOT))
    parser.add_argument("--current-phase6d-root", default=str(base.DEFAULT_CURRENT_PHASE6D_ROOT))
    parser.add_argument("--phaseS1-root", default=str(base.DEFAULT_PHASES1_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(base.DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(base.DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
