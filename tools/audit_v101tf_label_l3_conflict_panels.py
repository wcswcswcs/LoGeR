#!/usr/bin/env python3
"""Build text panels for v101 label/L3 conflict audit.

The plan asks for conflict panels when label and L3 disagree.  These panels are
Markdown/CSV evidence bundles, not runtime artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
TRACK_T = ROOT / "trackT_drift_target_relabel"
TRACK_Q2 = ROOT / "trackQ2_scale_update_admission"
TRACK_V = ROOT / "trackV_anchor_scale_observability"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("case_id", ""): row for row in rows if row.get("case_id")}


def panel_class(row: dict[str, Any]) -> tuple[str, str]:
    label = row.get("case_label", "")
    taxonomy = row.get("target_taxonomy", "")
    split = row.get("split_target_kind", "")
    conflict = row.get("label_l3_conflict", "")
    if taxonomy == "GOOD_HIGH_L3_CONTAMINATED" or "good_high_l3" in str(conflict):
        return "good_high_l3_contaminated", "Exclude from SAFE_GOOD and use only as diagnostic high-harm conflict."
    if taxonomy == "LOCAL_BAD_NOT_HANDOFF" or "bad_low_l3" in str(conflict):
        return "bad_low_l3_local_not_handoff", "Do not force into HANDOFF target; keep as local/read failure diagnostic."
    if split == "L3_HIGH_HARM_TARGET" and label == "good":
        return "binary_good_l3_harm_conflict", "Keep binary-label and L3-harm diagnostics separate."
    if taxonomy == "MULTIMODE_LOWOBS_ABSTAIN":
        return "lowobs_multimode_abstain", "Exclude from action gate until observability/control evidence improves."
    return "mixed_or_ambiguous", "Keep diagnostic-only; do not use for action gate."


def main() -> None:
    target_rows = by_case(read_rows(TRACK_T / "target_universe_v101.csv"))
    conflict_rows = read_rows(TRACK_T / "label_l3_conflict_rows.csv")
    split_rows = by_case(read_rows(TRACK_T / "target_split_l3_metric_rows.csv"))
    q2_rows = by_case(read_rows(TRACK_Q2 / "admission_rows.csv"))
    geom_case_rows = by_case(read_rows(TRACK_V / "per_anchor_geometry_case_summary.csv"))
    good_ext_rows = by_case(read_rows(TRACK_T / "v95_good_control_extension_rows.csv"))

    panel_rows: list[dict[str, Any]] = []
    conflict_case_ids = sorted({row.get("case_id", "") for row in conflict_rows if row.get("case_id")})
    for case_id in conflict_case_ids:
        target = target_rows.get(case_id, {})
        split = split_rows.get(case_id, {})
        q2 = q2_rows.get(case_id, {})
        geom = geom_case_rows.get(case_id, {})
        good_ext = good_ext_rows.get(case_id, {})
        merged: dict[str, Any] = {
            "case_id": case_id,
            "seq": target.get("seq", ""),
            "case_label": target.get("case_label", ""),
            "failure_type": target.get("failure_type", ""),
            "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
            "label_l3_conflict": target.get("label_l3_conflict", ""),
            "target_taxonomy": target.get("target_taxonomy", ""),
            "target_reason": target.get("target_reason", ""),
            "split_target_kind": split.get("split_target_kind", ""),
            "binary_good_high_L3_conflict": split.get("binary_good_high_L3_conflict", ""),
            "action_eligible_handoff_target": split.get("action_eligible_handoff_target", ""),
            "q2_admission_decision": q2.get("admission_decision", ""),
            "q2_proxy_selected_delay_or_no_scale": split.get("q_proxy_selected_delay_or_no_scale", ""),
            "O_scale_repaired_mean": geom.get("O_scale_repaired_mean", ""),
            "O_scale_repaired_p25": geom.get("O_scale_repaired_p25", ""),
            "geometry_available_anchor_frac": geom.get("geometry_available_anchor_frac", ""),
            "v95_good_control_extension_exclusion": good_ext.get("exclusion_reason", ""),
        }
        klass, recommendation = panel_class(merged)
        merged["panel_class"] = klass
        merged["recommendation"] = recommendation
        merged["runtime_action_allowed"] = False
        panel_rows.append(merged)

    counts = Counter(row["panel_class"] for row in panel_rows)
    summary = {
        "schema": "acl2_v101_label_l3_conflict_panels_v1",
        "panel_count": len(panel_rows),
        "case_count": len(conflict_case_ids),
        "panel_class_counts": dict(counts),
        "runtime_action_allowed": False,
        "claim": "Conflict panels are diagnostic only and separate binary-label from L3-harm evidence.",
    }
    write_rows(TRACK_T / "label_l3_conflict_panel_rows.csv", panel_rows)
    write_rows(
        TRACK_T / "label_l3_conflict_panel_manifest.csv",
        [
            {
                "path": str(TRACK_T / "label_l3_conflict_panels.md"),
                "description": "Markdown label/L3 conflict panel report",
                "exists": True,
            },
            {
                "path": str(TRACK_T / "label_l3_conflict_panel_rows.csv"),
                "description": "Structured label/L3 conflict panel rows",
                "exists": True,
            },
        ],
    )
    write_json(TRACK_T / "label_l3_conflict_panel_summary.json", summary)
    lines = [
        "# Track T Label/L3 Conflict Panels",
        "",
        f"- Panel count: {summary['panel_count']}",
        f"- Panel class counts: `{json.dumps(dict(counts), sort_keys=True)}`",
        "- Runtime action allowed: false",
        "",
    ]
    for row in panel_rows:
        lines.extend(
            [
                f"## {row['case_id']}",
                "",
                f"- label: `{row['case_label']}`",
                f"- failure_type: `{row['failure_type']}`",
                f"- L3: `{row['L3_handoff_transfer_penalty_proxy']}`",
                f"- label_l3_conflict: `{row['label_l3_conflict']}`",
                f"- target_taxonomy: `{row['target_taxonomy']}`",
                f"- split_target_kind: `{row['split_target_kind']}`",
                f"- Q2 decision: `{row['q2_admission_decision']}`",
                f"- repaired O_scale mean: `{row['O_scale_repaired_mean']}`",
                f"- panel_class: `{row['panel_class']}`",
                f"- recommendation: {row['recommendation']}",
                "",
            ]
        )
    write_text(TRACK_T / "label_l3_conflict_panels.md", "\n".join(lines))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
