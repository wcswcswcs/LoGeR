#!/usr/bin/env python3
"""Build a closure matrix for ACL2 v74-TF predeclared action families.

This is an audit helper: it does not run new experiments or tune thresholds.
It only reads landed Phase5/fallback artifacts and writes a compact JSON/CSV
matrix showing whether any predeclared family remains eligible for promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PREDECLARED_FAMILIES = [
    "component_leave_one_out_SWA",
    "component_boost_SWA",
    "component_veto_SWA",
    "route_swap_geometry_context_transient",
    "refresh_hold_flip",
    "stable_anchor_floor_short",
    "harmful_no_persistent_TTT_if_available",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object JSON at {path}")
    return data


def _family_for(intervention_type: str) -> str | None:
    if intervention_type in PREDECLARED_FAMILIES:
        return intervention_type
    if intervention_type.startswith("route_swap_geometry_context_transient"):
        return "route_swap_geometry_context_transient"
    if intervention_type.startswith("stable_anchor_floor"):
        return "stable_anchor_floor_short"
    return None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any, default: float = -999.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            _as_int(row.get("support_chunk_count")),
            _as_float(row.get("median_delta_J_scale")),
        ),
    )


def build_report(root: Path) -> dict[str, Any]:
    phase5_path = (
        root
        / "report_final"
        / "phase5_online_memory_intervention_after_refresh_hold_flip_09_validation"
        / "online_intervention_summary.json"
    )
    geometry01_path = (
        root
        / "phase5_refresh_hold_flip_radio_qscale_holdalpha005_top4"
        / "phaseE_multichunk_summary_full11_geometry_fallback.json"
    )
    geometry09_path = (
        root
        / "phase5_refresh_hold_flip_09_radio_qscale_holdalpha005_top8"
        / "phaseE_multichunk_summary_top8_gt09_geometry_fallback.json"
    )

    phase5 = _load_json(phase5_path)
    geometry01 = _load_json(geometry01_path)
    geometry09 = _load_json(geometry09_path)

    families: dict[str, list[dict[str, Any]]] = {
        family: [] for family in PREDECLARED_FAMILIES
    }
    for row in phase5.get("intervention_summary", []):
        if not isinstance(row, dict):
            continue
        family = _family_for(str(row.get("intervention_type", "")))
        if family is not None:
            families[family].append(row)

    summary: list[dict[str, Any]] = []
    for family in PREDECLARED_FAMILIES:
        family_rows = families[family]
        seq01_rows = [
            row
            for row in family_rows
            if str(row.get("seq", "01")) in ("01", "None", "")
        ]
        seq09_rows = [row for row in family_rows if str(row.get("seq")) == "09"]
        best01 = _best_row(seq01_rows)
        best09 = _best_row(seq09_rows)

        status = "tested_no_01_support"
        reason = "no KITTI01 causal support beating controls"
        if family == "refresh_hold_flip":
            status = "01_support_09_non_reversal_failed"
            reason = (
                "KITTI01 support exists but KITTI09 median delta is negative "
                "and non_reversal_gate_pass is false"
            )
        elif family == "stable_anchor_floor_short":
            reason = "stable-anchor variants tested; no variant produced support chunks"
        elif family == "route_swap_geometry_context_transient":
            reason = "pair/query variants tested; no variant produced support chunks"

        summary.append(
            {
                "family": family,
                "variant_count": len(family_rows),
                "tested": bool(family_rows),
                "best01_intervention_type": best01.get("intervention_type", ""),
                "best01_hook_active_chunks": best01.get("hook_active_chunks", ""),
                "best01_positive_chunks": best01.get("positive_chunks", ""),
                "best01_support_chunks": best01.get("support_chunks", ""),
                "best01_support_chunk_count": _as_int(
                    best01.get("support_chunk_count")
                ),
                "best01_median_delta_J_scale": best01.get("median_delta_J_scale"),
                "best01_causal_support": bool(best01.get("causal_support")),
                "best09_intervention_type": best09.get("intervention_type", ""),
                "best09_positive_chunks": best09.get("positive_chunks", ""),
                "best09_support_chunks": best09.get("support_chunks", ""),
                "best09_support_chunk_count": _as_int(
                    best09.get("support_chunk_count")
                ),
                "best09_median_delta_J_scale": best09.get("median_delta_J_scale"),
                "best09_non_reversal_gate_pass": bool(
                    best09.get("non_reversal_gate_pass")
                ),
                "status": status,
                "promotion_allowed": False,
                "closure_reason": reason,
            }
        )

    return {
        "plan": "docs/ACL2_v74TF_TrainingFree_SemanticMemoryControl_FullPlan.md",
        "phase5_source": str(phase5_path),
        "phase5_gate_pass": phase5.get("phase5_gate_pass"),
        "phase5_01_gate_pass": phase5.get("phase5_01_gate_pass"),
        "phase5_09_gate_pass": phase5.get("phase5_09_gate_pass"),
        "blocked_reason": phase5.get("blocked_reason"),
        "predeclared_families": summary,
        "geometry_fallback": {
            "01_summary": str(geometry01_path),
            "09_summary": str(geometry09_path),
            "phaseE_01_gate_pass": geometry01.get("phaseE_gate_pass"),
            "phaseE_09_gate_pass": geometry09.get("phaseE_gate_pass"),
            "phaseE_01_head_tail_pass_count": geometry01.get(
                "head_tail_pass_count"
            ),
            "phaseE_01_overlap_pass_count": geometry01.get("overlap_pass_count"),
            "phaseE_09_head_tail_pass_count": geometry09.get(
                "head_tail_pass_count"
            ),
            "phaseE_09_overlap_pass_count": geometry09.get("overlap_pass_count"),
            "promotion_allowed": False,
            "closure_reason": (
                "geometry-only fallback fails mechanism gate and KITTI09 support, "
                "so it is not a Phase7 candidate"
            ),
        },
        "phase7_allowed": False,
        "phase8_allowed": False,
        "final_closure": (
            "No legal predeclared Phase5 action family or geometry-only fallback "
            "remains eligible for Phase6/Phase7 promotion under the FullPlan stop rules."
        ),
    }


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "predeclared_action_family_closure.json"
    csv_path = out_dir / "predeclared_action_family_closure.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = report["predeclared_families"]
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/kitti01_hmc_v2/"
            "acl2_v74tf_training_free_semantic_memory_control"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to ROOT/report_final/phase5_predeclared_action_family_closure",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (
        args.root / "report_final" / "phase5_predeclared_action_family_closure"
    )
    report = build_report(args.root)
    json_path, csv_path = write_report(report, out_dir)
    print(
        json.dumps(
            {
                "out_json": str(json_path),
                "out_csv": str(csv_path),
                "family_count": len(report["predeclared_families"]),
                "tested_all": all(
                    row["tested"] for row in report["predeclared_families"]
                ),
                "phase5_gate_pass": report["phase5_gate_pass"],
                "phase7_allowed": report["phase7_allowed"],
                "phase8_allowed": report["phase8_allowed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
