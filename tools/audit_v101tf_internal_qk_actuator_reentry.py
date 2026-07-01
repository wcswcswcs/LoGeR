#!/usr/bin/env python3
"""Audit V83/V85 internal-QK cue actuator re-entry for ACL2 v101.

This is a read-only Outcome-D follow-up. It summarizes already measured v95
action-surface variants for the strong TrackG internal-QK cue, and it does not
authorize runtime action.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


V95_ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
V101_ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
OUT = V101_ROOT / "final_decision"

TRACKG_SUMMARY = V95_ROOT / "trackG_swa_internal_cue_eval_v1" / "summary.json"

FAMILIES = [
    {
        "family_id": "baseline_merge_mode",
        "label": "TrackE baseline internal cue action surface",
        "path": V95_ROOT / "trackE_internal_cue_action_surface_v1",
        "component_summary": None,
    },
    {
        "family_id": "alpha_delta",
        "label": "TrackE alpha/delta internal cue action surface",
        "path": V95_ROOT / "trackE_internal_cue_alphadelta_action_surface_v1",
        "component_summary": V95_ROOT / "trackE_internal_cue_alphadelta_component_strict_gate" / "summary.json",
    },
    {
        "family_id": "high_alpha",
        "label": "TrackE high-alpha internal cue action surface",
        "path": V95_ROOT / "trackE_internal_cue_highalpha_action_surface_v1",
        "component_summary": V95_ROOT / "trackE_internal_cue_highalpha_component_strict_gate" / "summary.json",
    },
    {
        "family_id": "old_actuator_alpha02",
        "label": "TrackH old-actuator alpha02 internal cue action surface",
        "path": V95_ROOT / "trackH_internal_cue_oldactuator_alpha02_action_surface_v1",
        "component_summary": V95_ROOT / "trackH_internal_cue_oldactuator_alpha02_component_strict_gate" / "summary.json",
    },
    {
        "family_id": "swa_source_gate_stable_v",
        "label": "TrackE SWA source gate stable-V action surface",
        "path": V95_ROOT / "trackE_swa_source_gate_stable_v_action_surface_v1",
        "component_summary": None,
    },
    {
        "family_id": "swa_source_replace_stable_v",
        "label": "TrackE SWA source replace stable-V action surface",
        "path": V95_ROOT / "trackE_swa_source_replace_stable_v_action_surface_v1",
        "component_summary": None,
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fnum(value: Any) -> float | None:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def compact(value: Any, max_len: int = 700) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def median_or_blank(values: list[float]) -> float | str:
    return median(values) if values else ""


def max_or_blank(values: list[float]) -> float | str:
    return max(values) if values else ""


def best_metric_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}

    def key(row: dict[str, str]) -> tuple[int, float, float, float, float]:
        return (
            1 if truthy(row.get("candidate_action_surface_gate_pass")) else 0,
            fnum(row.get("bad_handoff_median_improvement")) or -1e99,
            fnum(row.get("bad_handoff_max_improvement")) or -1e99,
            fnum(row.get("selected_bad_sequence_coverage")) or -1e99,
            fnum(row.get("selected_bad_count")) or -1e99,
        )

    return max(rows, key=key)


def family_summary(family: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    family_id = str(family["family_id"])
    path = Path(family["path"])
    summary = read_json(path / "summary.json")
    metrics = read_csv(path / "internal_cue_action_surface_metrics.csv")
    oracle = read_csv(path / "internal_cue_oracle_metrics.csv")
    per_pair = read_csv(path / "per_pair_best_handoff_rows.csv")
    component = read_json(family["component_summary"]) if family.get("component_summary") else {}

    best = best_metric_row(metrics)
    pass_rows = [row for row in metrics if truthy(row.get("candidate_action_surface_gate_pass"))]
    med_values = [v for row in metrics if (v := fnum(row.get("bad_handoff_median_improvement"))) is not None]
    max_values = [v for row in metrics if (v := fnum(row.get("bad_handoff_max_improvement"))) is not None]
    oracle_meds = [v for row in oracle if (v := fnum(row.get("oracle_bad_handoff_median_improvement"))) is not None]
    oracle_maxes = [v for row in oracle if (v := fnum(row.get("oracle_bad_handoff_max_improvement"))) is not None]
    bad_pair_rows = [row for row in per_pair if str(row.get("case_label_offline_only", "")).lower() == "bad"]
    bad_pair_improvements = [
        v for row in bad_pair_rows if (v := fnum(row.get("max_handoff_improvement"))) is not None
    ]
    component_strict_pass = truthy(component.get("strict_gate_pass")) if component else ""

    out = {
        "family_id": family_id,
        "family_label": family["label"],
        "family_dir": str(path),
        "summary_gate_pass": summary.get("gate_pass", ""),
        "summary_runtime_action_allowed": summary.get("runtime_action_allowed", ""),
        "summary_blocker": summary.get("blocker", ""),
        "metric_row_count": len(metrics),
        "candidate_action_surface_pass_count": len(pass_rows),
        "family_action_surface_gate_pass": len(pass_rows) > 0,
        "best_cue_id": best.get("cue_id", ""),
        "best_variant": best.get("variant", ""),
        "best_candidate_action_surface_gate_pass": truthy(best.get("candidate_action_surface_gate_pass")),
        "best_selected_bad_count": best.get("selected_bad_count", ""),
        "best_selected_good_count": best.get("selected_good_count", ""),
        "best_selected_bad_sequence_coverage": best.get("selected_bad_sequence_coverage", ""),
        "best_bad_handoff_median_improvement": fnum(best.get("bad_handoff_median_improvement")),
        "best_bad_handoff_max_improvement": fnum(best.get("bad_handoff_max_improvement")),
        "best_bad_handoff_min_improvement": fnum(best.get("bad_handoff_min_improvement")),
        "best_actual_minus_best_same_count_control": fnum(best.get("actual_minus_best_same_count_control")),
        "best_bad_handoff_median_ge_threshold": truthy(best.get("bad_handoff_median_ge_threshold")),
        "best_beats_same_count_controls_ge_0p05": truthy(best.get("beats_same_count_controls_ge_0p05")),
        "max_metric_bad_handoff_median_improvement": max_or_blank(med_values),
        "max_metric_bad_handoff_max_improvement": max_or_blank(max_values),
        "oracle_row_count": len(oracle),
        "oracle_best_bad_handoff_median_improvement": max_or_blank(oracle_meds),
        "oracle_best_bad_handoff_max_improvement": max_or_blank(oracle_maxes),
        "per_pair_best_row_count": len(per_pair),
        "per_pair_bad_row_count": len(bad_pair_rows),
        "per_pair_bad_max_handoff_improvement_max": max_or_blank(bad_pair_improvements),
        "per_pair_bad_max_handoff_improvement_median": median_or_blank(bad_pair_improvements),
        "per_pair_bad_ge_0p05_count": sum(1 for value in bad_pair_improvements if value >= 0.05),
        "per_pair_bad_negative_best_count": sum(1 for value in bad_pair_improvements if value < 0.0),
        "component_summary_path": str(family["component_summary"] or ""),
        "component_strict_gate_pass": component_strict_pass,
        "component_strict_gate_pass_count": component.get("strict_gate_pass_count", ""),
        "component_best_variant": component.get("best_variant", ""),
        "component_best_bad_handoff_median_improvement": component.get("best_bad_handoff_median_improvement", ""),
        "component_blocker": component.get("blocker", ""),
    }

    candidate_rows: list[dict[str, Any]] = []
    for row in metrics:
        candidate_rows.append(
            {
                "family_id": family_id,
                "family_label": family["label"],
                **row,
            }
        )

    per_pair_rows: list[dict[str, Any]] = []
    for row in per_pair:
        per_pair_rows.append(
            {
                "family_id": family_id,
                "family_label": family["label"],
                **row,
            }
        )

    return out, candidate_rows, per_pair_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trackg = read_json(TRACKG_SUMMARY)
    trackg_best = trackg.get("best_method_safe_internal", {})

    family_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    per_pair_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        row, candidates, per_pair = family_summary(family)
        family_rows.append(row)
        candidate_rows.extend(candidates)
        per_pair_rows.extend(per_pair)

    def row_float(row: dict[str, Any], key: str) -> float:
        value = fnum(row.get(key))
        return value if value is not None else -1e99

    family_rows.sort(
        key=lambda row: (
            not bool(row["family_action_surface_gate_pass"]),
            -row_float(row, "best_bad_handoff_median_improvement"),
            -row_float(row, "best_bad_handoff_max_improvement"),
        )
    )
    candidate_rows.sort(
        key=lambda row: (
            not truthy(row.get("candidate_action_surface_gate_pass")),
            -(fnum(row.get("bad_handoff_median_improvement")) or -1e99),
            -(fnum(row.get("bad_handoff_max_improvement")) or -1e99),
        )
    )

    family_fields = [
        "family_id",
        "family_label",
        "family_dir",
        "summary_gate_pass",
        "summary_runtime_action_allowed",
        "summary_blocker",
        "metric_row_count",
        "candidate_action_surface_pass_count",
        "family_action_surface_gate_pass",
        "best_cue_id",
        "best_variant",
        "best_candidate_action_surface_gate_pass",
        "best_selected_bad_count",
        "best_selected_good_count",
        "best_selected_bad_sequence_coverage",
        "best_bad_handoff_median_improvement",
        "best_bad_handoff_max_improvement",
        "best_bad_handoff_min_improvement",
        "best_actual_minus_best_same_count_control",
        "best_bad_handoff_median_ge_threshold",
        "best_beats_same_count_controls_ge_0p05",
        "max_metric_bad_handoff_median_improvement",
        "max_metric_bad_handoff_max_improvement",
        "oracle_row_count",
        "oracle_best_bad_handoff_median_improvement",
        "oracle_best_bad_handoff_max_improvement",
        "per_pair_best_row_count",
        "per_pair_bad_row_count",
        "per_pair_bad_max_handoff_improvement_max",
        "per_pair_bad_max_handoff_improvement_median",
        "per_pair_bad_ge_0p05_count",
        "per_pair_bad_negative_best_count",
        "component_summary_path",
        "component_strict_gate_pass",
        "component_strict_gate_pass_count",
        "component_best_variant",
        "component_best_bad_handoff_median_improvement",
        "component_blocker",
    ]
    candidate_fields = [
        "family_id",
        "family_label",
        "cue_id",
        "variant",
        "candidate_action_surface_gate_pass",
        "cue_bad_recall",
        "cue_good_FPR",
        "cue_balanced_accuracy",
        "selected_pair_count",
        "selected_bad_count",
        "selected_good_count",
        "selected_sequence_coverage",
        "selected_bad_sequence_coverage",
        "bad_handoff_median_improvement",
        "bad_handoff_max_improvement",
        "bad_handoff_min_improvement",
        "bad_runtime_proxy_median_I_J",
        "bad_boundary_median_improvement",
        "bad_scale_median_improvement",
        "good_handoff_median_worsen",
        "good_handoff_max_worsen",
        "actual_minus_best_same_count_control",
        "bad_handoff_median_ge_threshold",
        "good_handoff_protection_gate",
        "sequence_coverage_ge_3",
        "beats_same_count_controls_ge_0p05",
        "selected_pairs",
        "selected_bad_pairs",
        "selected_good_pairs",
    ]
    per_pair_fields = [
        "family_id",
        "family_label",
        "pair_id",
        "seq",
        "case_label_offline_only",
        "best_variant_by_handoff",
        "max_handoff_improvement",
        "I_J_runtime_proxy_at_best_handoff",
        "boundary_improvement_at_best_handoff",
        "scale_improvement_at_best_handoff",
    ]

    write_csv(OUT / "internal_qk_actuator_reentry_family_summary.csv", family_rows, family_fields)
    write_csv(OUT / "internal_qk_actuator_reentry_candidate_metrics.csv", candidate_rows, candidate_fields)
    write_csv(OUT / "internal_qk_actuator_reentry_per_pair_best_rows.csv", per_pair_rows, per_pair_fields)

    best_family = family_rows[0] if family_rows else {}
    per_pair_bad_values = [
        v for row in per_pair_rows
        if str(row.get("case_label_offline_only", "")).lower() == "bad"
        if (v := fnum(row.get("max_handoff_improvement"))) is not None
    ]
    summary = {
        "schema": "acl2_v101_internal_qk_actuator_reentry_audit_v1",
        "trackG_summary": str(TRACKG_SUMMARY),
        "trackG_gate_pass": trackg.get("gate_pass"),
        "trackG_runtime_action_allowed": trackg.get("runtime_action_allowed"),
        "trackG_best_cue_id": trackg_best.get("cue_id", ""),
        "trackG_best_balanced_accuracy": trackg_best.get("balanced_accuracy", ""),
        "trackG_best_bad_recall": trackg_best.get("bad_recall", ""),
        "trackG_best_good_FPR": trackg_best.get("good_FPR", ""),
        "trackG_best_selected_sequence_coverage": trackg_best.get("selected_sequence_coverage", ""),
        "trackG_best_selected_pair_ids": trackg_best.get("selected_pair_ids", ""),
        "family_count": len(family_rows),
        "metric_candidate_row_count": len(candidate_rows),
        "action_surface_passing_family_count": sum(
            1 for row in family_rows if row["family_action_surface_gate_pass"]
        ),
        "candidate_action_surface_pass_row_count": sum(
            1 for row in candidate_rows if truthy(row.get("candidate_action_surface_gate_pass"))
        ),
        "component_strict_family_count": sum(1 for row in family_rows if row["component_summary_path"]),
        "component_strict_passing_family_count": sum(
            1 for row in family_rows if row["component_strict_gate_pass"] is True
        ),
        "best_family_id": best_family.get("family_id", ""),
        "best_variant": best_family.get("best_variant", ""),
        "best_cue_id": best_family.get("best_cue_id", ""),
        "best_family_action_surface_gate_pass": best_family.get("family_action_surface_gate_pass", False),
        "best_bad_handoff_median_improvement": best_family.get("best_bad_handoff_median_improvement", ""),
        "best_bad_handoff_max_improvement": best_family.get("best_bad_handoff_max_improvement", ""),
        "best_actual_minus_best_same_count_control": best_family.get(
            "best_actual_minus_best_same_count_control", ""
        ),
        "best_selected_bad_count": best_family.get("best_selected_bad_count", ""),
        "best_selected_good_count": best_family.get("best_selected_good_count", ""),
        "best_selected_bad_sequence_coverage": best_family.get("best_selected_bad_sequence_coverage", ""),
        "global_per_pair_bad_max_handoff_improvement": max_or_blank(per_pair_bad_values),
        "global_per_pair_bad_max_handoff_improvement_median": median_or_blank(per_pair_bad_values),
        "global_per_pair_bad_ge_0p05_count": sum(1 for value in per_pair_bad_values if value >= 0.05),
        "runtime_action_allowed": False,
        "full_validation_allowed": False,
        "v101_goal_achieved": False,
        "claim": (
            "TrackG internal-QK cue remains a diagnostic selector candidate, but all audited v95 measured "
            "actuator/action-surface families fail the handoff-improvement/control gate."
        ),
    }
    with (OUT / "internal_qk_actuator_reentry_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    report = [
        "# Internal-QK Actuator Re-entry Audit",
        "",
        "This audit summarizes measured v95 action-surface families for the V83/V85 internal-QK cue.",
        "It is diagnostic only and does not authorize runtime action.",
        "",
        "## TrackG Cue",
        "",
        f"- TrackG gate_pass: `{summary['trackG_gate_pass']}`",
        f"- best cue: `{summary['trackG_best_cue_id']}`",
        f"- balanced accuracy: `{summary['trackG_best_balanced_accuracy']}`",
        f"- bad recall: `{summary['trackG_best_bad_recall']}`",
        f"- good FPR: `{summary['trackG_best_good_FPR']}`",
        f"- selected sequence coverage: `{summary['trackG_best_selected_sequence_coverage']}`",
        f"- selected pairs: `{summary['trackG_best_selected_pair_ids']}`",
        "",
        "## Action-Surface Families",
        "",
        f"- families audited: `{summary['family_count']}`",
        f"- metric candidate rows: `{summary['metric_candidate_row_count']}`",
        f"- passing action-surface families: `{summary['action_surface_passing_family_count']}`",
        f"- passing candidate rows: `{summary['candidate_action_surface_pass_row_count']}`",
        f"- component-strict passing families: `{summary['component_strict_passing_family_count']}`",
        f"- best family: `{summary['best_family_id']}`",
        f"- best variant: `{summary['best_variant']}`",
        f"- best bad handoff median improvement: `{summary['best_bad_handoff_median_improvement']}`",
        f"- best bad handoff max improvement: `{summary['best_bad_handoff_max_improvement']}`",
        f"- best actual minus same-count control: `{summary['best_actual_minus_best_same_count_control']}`",
        f"- global per-pair bad max handoff improvement: `{summary['global_per_pair_bad_max_handoff_improvement']}`",
        f"- per-pair bad rows >= 0.05 handoff improvement: `{summary['global_per_pair_bad_ge_0p05_count']}`",
        "",
        "## Interpretation",
        "",
    ]
    if summary["action_surface_passing_family_count"]:
        report.append(
            "At least one measured action-surface family passed this diagnostic re-entry gate, but runtime remains disallowed until a predeclared measured-control rerun is performed."
        )
    else:
        report.append(
            "The internal-QK selector is strong offline, but the audited measured actuator families are too weak: none passes the action-surface gate, and no bad per-pair best handoff improvement reaches 0.05."
        )
    report.append("")
    report.append("Runtime action remains disallowed.")
    (OUT / "internal_qk_actuator_reentry_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
