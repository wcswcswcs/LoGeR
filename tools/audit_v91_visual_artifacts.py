#!/usr/bin/env python3
"""Audit v91 visual rediscovery artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT


DEFAULT_DIR = ROOT / "phase11_visual_rediscovery"
REQUIRED_CATEGORIES = [
    "semantic_topology_tracklet_panels",
    "regime_classifier_panels",
    "regime_conditioned_mode_panels",
    "policy_state_panels",
    "delayed_commit_panels",
    "carrier_or_blocked_panels",
    "counterfactual_or_blocked_panels",
    "runtime_or_blocked_panels",
    "ttt_or_blocked_panels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    args = parse_args()
    matrix_path = args.visual_dir / "visual_requirement_matrix.csv"
    rows = pd.read_csv(matrix_path) if matrix_path.exists() else pd.DataFrame()
    checks = []
    categories_ok = 0
    for category in REQUIRED_CATEGORIES:
        sub = rows[rows["category"].astype(str) == category] if len(rows) else pd.DataFrame()
        ok = False
        size = 0
        rgb = False
        if len(sub):
            size = int(pd.to_numeric(sub["size_bytes"], errors="coerce").fillna(0).max())
            rgb = bool(sub["rgb_prev_curr_available"].map(_bool).all())
            ok = bool(size > 1000 and rgb)
        categories_ok += int(ok)
        checks.append({"check": f"{category}_present_nonempty_rgb", "pass": ok, "size_bytes": size, "rgb_prev_curr_available": rgb})
    review_coverage = float(categories_ok / max(len(REQUIRED_CATEGORIES), 1))
    insight_exists = (args.visual_dir / "visual_insight.md").exists() and (args.visual_dir / "visual_insight.md").stat().st_size > 0
    no_fake = True
    if len(rows) and "is_fake_route_runtime_ttt_panel" in rows:
        no_fake = not bool(rows["is_fake_route_runtime_ttt_panel"].map(_bool).any())
    gate = bool(review_coverage >= 0.80 and all(row["pass"] for row in checks) and insight_exists and no_fake)
    summary = {
        "phase": "Phase11_visual_artifact_audit",
        "visual_integrity_gate_pass": gate,
        "required_categories_present": bool(all(row["pass"] for row in checks)),
        "review_coverage": review_coverage,
        "visual_insight_exists": insight_exists,
        "no_fake_route_runtime_ttt_panels": no_fake,
        "panel_rows": int(len(rows)),
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "visual_integrity_gate_failed"
    write_csv(args.visual_dir / "visual_requirement_matrix_audit.csv", checks)
    write_json(args.visual_dir / "visual_integrity_audit.json", summary)
    print(f"visual_integrity_gate_pass={summary['visual_integrity_gate_pass']}")
    print(f"review_coverage={summary['review_coverage']}")
    print(f"visual_insight_exists={summary['visual_insight_exists']}")
    print(f"no_fake_route_runtime_ttt_panels={summary['no_fake_route_runtime_ttt_panels']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
