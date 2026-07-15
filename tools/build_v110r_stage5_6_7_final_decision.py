#!/usr/bin/env python3
"""Finalize ACL2 v110R Stage5/6/7 and final decision artifacts.

This script is intentionally read-only with respect to model/runtime outputs. It
consolidates the registered v110R evidence after Stage4, writes the Stage5
composition skip decision, the Stage6 semantic-causality decision, the Stage7
LoGeR comparison blocker, and the final decision report.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
STAGE3 = RESULT_ROOT / "stage3_pilot_00_02"
STAGE4 = RESULT_ROOT / "stage4_full_00_01_02_05_validation"
STAGE5 = RESULT_ROOT / "stage5_composition_search"
STAGE6 = RESULT_ROOT / "stage6_semantic_causality_decision"
CROSS = RESULT_ROOT / "cross_model_comparison"
FINAL = RESULT_ROOT / "final_decision"
V105_LOGER = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline"
V109 = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"

SEQUENCES = ("00", "01", "02", "05")
FINAL_TAXONOMY = "FULL_ATE_BOOST_INTERNAL_SCHEDULE_BASELINE_ONLY"
SEMANTIC_MARGIN = 0.02


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["schema", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(val) for val in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def max_rel_harm(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return max([max(0.0, -v) for v in vals], default=float("nan"))


def by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key, "")): row for row in rows}


def best_stage4_policy(policy_rows: list[dict[str, str]]) -> dict[str, str]:
    pass_rows = [row for row in policy_rows if boolish(row.get("strong_improvement_pass"))]
    source = pass_rows or policy_rows
    if not source:
        return {}
    return max(source, key=lambda row: fnum(row.get("median_full_rel", "nan")))


def policy_full_rows(policy_id: str, full_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        [row for row in full_rows if row.get("policy_id") == policy_id],
        key=lambda row: row.get("seq", ""),
    )


def f19_by_seq() -> dict[str, dict[str, str]]:
    return {
        row.get("seq", ""): row
        for row in read_csv(STAGE0 / "f19_champion_metrics.csv")
        if row.get("seq")
    }


def baseline_by_seq() -> dict[str, dict[str, str]]:
    return {
        row.get("seq", ""): row
        for row in read_csv(STAGE0 / "frozen_baseline_table.csv")
        if row.get("seq")
    }


def build_stage5(policy_rows: list[dict[str, str]], stage4_summary: dict[str, Any]) -> dict[str, Any]:
    geometry_pass_rows = [row for row in policy_rows if boolish(row.get("stage4_geometry_pass"))]
    strong_pass_rows = [row for row in policy_rows if boolish(row.get("strong_improvement_pass"))]
    pass_candidates = sorted({row.get("candidate_id", "") for row in geometry_pass_rows if row.get("candidate_id")})
    pass_surfaces = sorted({row.get("surface_id", "") for row in geometry_pass_rows if row.get("surface_id")})
    failed_candidates = sorted(
        {
            row.get("candidate_id", "")
            for row in policy_rows
            if row.get("candidate_id") and not boolish(row.get("stage4_geometry_pass"))
        }
    )
    composition_allowed = len(pass_candidates) >= 2 and len(pass_surfaces) >= 2
    reason = (
        "composition_allowed_by_multiple_complementary_geometry_pass_surfaces"
        if composition_allowed
        else "composition_not_run_only_B1_surface_passed_stage4_geometry_E1_F2_failed"
    )

    combo_config_rows: list[dict[str, Any]] = []
    combo_metric_rows: list[dict[str, Any]] = []
    if not composition_allowed:
        combo_config_rows.append(
            {
                "schema": "acl2_v110r_stage5_combo_config_row_v1",
                "combo_id": "SKIP_NO_COMPLEMENTARY_SURFACE_PAIR",
                "composition_search_run": False,
                "reason": reason,
                "geometry_pass_candidate_ids": ";".join(pass_candidates),
                "geometry_pass_surface_ids": ";".join(pass_surfaces),
                "stage4_failed_candidate_ids": ";".join(failed_candidates),
            }
        )

    summary = {
        "schema": "acl2_v110r_stage5_composition_summary_v1",
        "composition_search_run": composition_allowed,
        "composition_allowed": composition_allowed,
        "reason": reason,
        "stage4_taxonomy": stage4_summary.get("taxonomy", ""),
        "geometry_pass_candidate_ids": pass_candidates,
        "strong_improvement_policy_ids": [row.get("policy_id", "") for row in strong_pass_rows],
        "failed_candidate_ids": failed_candidates,
        "outputs": {
            "combo_config_rows": rel(STAGE5 / "combo_config_rows.csv"),
            "combo_metric_rows": rel(STAGE5 / "combo_metric_rows.csv"),
            "combo_interaction_report": rel(STAGE5 / "combo_interaction_report.md"),
            "stage5_summary": rel(STAGE5 / "stage5_summary.json"),
        },
    }

    report = [
        "# ACL2 v110R Stage5 Composition Search Decision",
        "",
        f"composition_search_run: {composition_allowed}",
        f"reason: {reason}",
        f"geometry_pass_candidate_ids: {pass_candidates}",
        f"strong_improvement_policy_ids: {summary['strong_improvement_policy_ids']}",
        f"failed_candidate_ids: {failed_candidates}",
        "",
        "## Interpretation",
        "",
        "The registered v110R order only allows limited composition when individual candidates are complementary. Stage4 produced one passing surface family, B1, while E1 and F2 failed the four-sequence geometry gate. A composition sweep is therefore not evidence-backed in this run.",
    ]
    write_csv(STAGE5 / "combo_config_rows.csv", combo_config_rows)
    write_csv(
        STAGE5 / "combo_metric_rows.csv",
        combo_metric_rows,
        fieldnames=[
            "schema",
            "combo_id",
            "metric_available",
            "blocker",
        ],
    )
    write_json(STAGE5 / "stage5_summary.json", summary)
    write_text(STAGE5 / "combo_interaction_report.md", "\n".join(report))
    return summary


def stage4_policy_lookup(policy_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row.get("candidate_id", ""), row.get("policy_family", "")): row
        for row in policy_rows
    }


def build_stage6(
    policy_rows: list[dict[str, str]],
    semantic_rows: list[dict[str, str]],
    stage4_summary: dict[str, Any],
) -> dict[str, Any]:
    lookup = stage4_policy_lookup(policy_rows)
    decision_rows: list[dict[str, Any]] = []
    for sem in semantic_rows:
        candidate_id = sem.get("candidate_id", "")
        plus = lookup.get((candidate_id, "semantic_plus_internal"), {})
        only = lookup.get((candidate_id, "semantic_only"), {})
        plus_geometry = boolish(plus.get("stage4_geometry_pass"))
        strong_pass = boolish(plus.get("strong_improvement_pass")) or boolish(only.get("strong_improvement_pass"))
        plus_minus_only = fnum(sem.get("semantic_plus_minus_semantic_only_median", "nan"))
        semantic_separated = math.isfinite(plus_minus_only) and plus_minus_only >= SEMANTIC_MARGIN
        stronger_controls_available = False
        aware_pass = bool(plus_geometry and strong_pass and semantic_separated and stronger_controls_available)
        safety_pass = False
        if plus_geometry and strong_pass and not semantic_separated:
            category = "STRONG_INTERNAL_OR_SCHEDULE_MEMORY_CONTROL_BASELINE"
            blocker = "semantic_plus_internal_equals_semantic_only; role_rotation_same_bucket_random_same_count_stage4_controls_missing"
        elif not plus_geometry:
            category = "NO_GO_SINGLE_SEQUENCE_OR_FULL_GEOMETRY_FAIL"
            blocker = "stage4_full_geometry_gate_failed"
        else:
            category = "NO_GO_SEMANTIC_CONTROLS_MATCH_ALL_EFFECTS"
            blocker = "semantic_causality_controls_not_satisfied"
        decision_rows.append(
            {
                "schema": "acl2_v110r_stage6_semantic_causality_row_v1",
                "candidate_id": candidate_id,
                "surface_id": sem.get("surface_id", ""),
                "semantic_plus_policy_id": sem.get("semantic_plus_policy_id", ""),
                "semantic_only_policy_id": sem.get("semantic_only_policy_id", ""),
                "stage4_geometry_pass": plus_geometry,
                "stage4_strong_improvement_pass": strong_pass,
                "semantic_plus_median_full_rel": sem.get("semantic_plus_median_full_rel", ""),
                "semantic_only_median_full_rel": sem.get("semantic_only_median_full_rel", ""),
                "semantic_plus_minus_semantic_only_median": plus_minus_only,
                "semantic_plus_beats_semantic_only_by_0p02": semantic_separated,
                "semantic_plus_beats_internal_by_0p02": "",
                "semantic_plus_beats_semantic_shuffle_best_by_0p02": "",
                "semantic_plus_beats_same_bucket_random_p95_by_0p02": "",
                "role_rotation_cannot_match_on_3_of_4": "",
                "same_count_random_cannot_match_on_3_of_4": "",
                "low_risk_reverse_worse_than_candidate": "",
                "stronger_stage6_controls_available": stronger_controls_available,
                "semantic_aware_method_candidate_pass": aware_pass,
                "semantic_safety_filter_pass": safety_pass,
                "stage6_category": category,
                "blocker": blocker,
            }
        )

    f19_control = read_json(V109 / "stage2_f19_keyframe_controls/f19_keyframe_control_summary.json")
    if f19_control:
        decision_rows.append(
            {
                "schema": "acl2_v110r_stage6_semantic_causality_row_v1",
                "candidate_id": "F19_frozen_reference",
                "surface_id": "F",
                "semantic_plus_policy_id": "F19_dynamic_or_special_admitted_high_risk_else_weak_context",
                "semantic_only_policy_id": "",
                "stage4_geometry_pass": True,
                "stage4_strong_improvement_pass": False,
                "semantic_plus_median_full_rel": f19_control.get("f19_median_full_rel_improvement", ""),
                "semantic_only_median_full_rel": "",
                "semantic_plus_minus_semantic_only_median": "",
                "semantic_plus_beats_semantic_only_by_0p02": False,
                "semantic_plus_beats_internal_by_0p02": "",
                "semantic_plus_beats_semantic_shuffle_best_by_0p02": "",
                "semantic_plus_beats_same_bucket_random_p95_by_0p02": "",
                "role_rotation_cannot_match_on_3_of_4": "",
                "same_count_random_cannot_match_on_3_of_4": False,
                "low_risk_reverse_worse_than_candidate": "",
                "stronger_stage6_controls_available": True,
                "semantic_aware_method_candidate_pass": False,
                "semantic_safety_filter_pass": False,
                "stage6_category": "STRONG_INTERNAL_OR_SCHEDULE_MEMORY_CONTROL_BASELINE",
                "blocker": f19_control.get("blocker", "same_count_keyframe_control_matches_f19_on_multiple_sequences"),
            }
        )

    semantic_aware_pass_any = any(boolish(row.get("semantic_aware_method_candidate_pass")) for row in decision_rows)
    safety_filter_pass_any = any(boolish(row.get("semantic_safety_filter_pass")) for row in decision_rows)
    strong_baseline_rows = [
        row
        for row in decision_rows
        if row.get("stage6_category") == "STRONG_INTERNAL_OR_SCHEDULE_MEMORY_CONTROL_BASELINE"
    ]
    b1_row = next((row for row in decision_rows if row.get("candidate_id") == "B1"), {})
    summary = {
        "schema": "acl2_v110r_stage6_semantic_causality_summary_v1",
        "semantic_aware_method_candidate_pass_any": semantic_aware_pass_any,
        "semantic_safety_filter_pass_any": safety_filter_pass_any,
        "strong_internal_or_schedule_baseline_any": bool(strong_baseline_rows),
        "best_geometry_candidate_id": "B1" if b1_row else stage4_summary.get("best_policy_by_median_full_rel", ""),
        "final_stage6_category": (
            "STRONG_INTERNAL_OR_SCHEDULE_MEMORY_CONTROL_BASELINE"
            if strong_baseline_rows
            else "NO_GO_SEMANTIC_CONTROLS_MATCH_ALL_EFFECTS"
        ),
        "taxonomy": (
            "STAGE6_INTERNAL_SCHEDULE_BASELINE_ONLY"
            if strong_baseline_rows
            else "STAGE6_NO_GO_SEMANTIC_CAUSALITY_FAIL"
        ),
        "blocker": (
            "semantic_plus_internal_and_semantic_only_match; stronger semantic controls absent_or_matched"
            if strong_baseline_rows
            else "semantic_causality_controls_failed"
        ),
        "decision_row_count": len(decision_rows),
        "outputs": {
            "semantic_causality_rows": rel(STAGE6 / "semantic_causality_rows.csv"),
            "semantic_causality_report": rel(STAGE6 / "semantic_causality_report.md"),
            "semantic_safety_filter_report": rel(STAGE6 / "semantic_safety_filter_report.md"),
            "stage6_summary": rel(STAGE6 / "stage6_summary.json"),
            "semantic_content_not_causal": rel(RESULT_ROOT / "SEMANTIC_CONTENT_NOT_CAUSAL_YET.md"),
        },
    }

    causality_report = [
        "# ACL2 v110R Stage6 Semantic Causality Decision",
        "",
        f"semantic_aware_method_candidate_pass_any: {semantic_aware_pass_any}",
        f"semantic_safety_filter_pass_any: {safety_filter_pass_any}",
        f"strong_internal_or_schedule_baseline_any: {bool(strong_baseline_rows)}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        "",
        "## Candidate Decisions",
        "",
    ]
    for row in decision_rows:
        causality_report.append(
            "- {candidate_id}: category={category} geometry_pass={geo} strong_pass={strong} "
            "plus_minus_semantic_only={delta} semantic_aware_pass={aware} blocker={blocker}".format(
                candidate_id=row.get("candidate_id", ""),
                category=row.get("stage6_category", ""),
                geo=row.get("stage4_geometry_pass", ""),
                strong=row.get("stage4_strong_improvement_pass", ""),
                delta=row.get("semantic_plus_minus_semantic_only_median", ""),
                aware=row.get("semantic_aware_method_candidate_pass", ""),
                blocker=row.get("blocker", ""),
            )
        )
    causality_report.extend(
        [
            "",
            "## Interpretation",
            "",
            "B1 is a real four-sequence full-ATE improvement candidate, but its semantic_plus_internal and semantic_only variants are numerically identical in Stage4. That means the current evidence supports a strong internal/cache-schedule baseline, not a semantic-aware method claim.",
        ]
    )

    safety_report = [
        "# ACL2 v110R Semantic Safety Filter Report",
        "",
        "semantic_safety_filter_pass_any: False",
        "",
        "The safety-filter gate requires semantic behavior to reduce harm against an internal control. Stage4 did not run internal-only four-sequence controls for B1/E1/F2, and the semantic_plus_internal rows match semantic_only exactly. Therefore no semantic safety-filter claim is allowed.",
    ]

    content_not_causal = [
        "# ACL2 v110R Semantic Content Not Causal Yet",
        "",
        "## Decision",
        "",
        "`No-Go` for claiming a semantic-aware LingBot memory-control method in v110R.",
        "",
        "Stage4 found a strong B1 full-ATE improvement, but semantic content is not isolated: `B1_semantic_plus_internal` and `B1_semantic_only` have identical four-sequence metrics, and the registered stronger controls are missing or already matched in prior F19 evidence.",
        "",
        "## Key Evidence",
        "",
        f"- Stage4 taxonomy: `{stage4_summary.get('taxonomy', '')}`",
        "- B1 median full relative improvement: `0.17413068803456322`",
        "- B1 mean full relative improvement: `0.18754824888948118`",
        "- B1 improved sequences: `4/4`",
        "- B1 max full harm: `0.0`",
        "- B1 local-window median harm: `0.017321397910578965`",
        "- B1 semantic_plus_internal minus semantic_only median: `0.0`",
        f"- F19 keyframe-control blocker: `{f19_control.get('blocker', '')}`",
        "",
        "## Interpretation",
        "",
        "The result should be retained as a strong internal/cache-schedule baseline. It must not be reported as a semantic-aware method until semantic content beats matched schedule/count/bucket/role controls under the registered margins.",
        "",
        "## Recommended Next Step",
        "",
        "Move the next method-development attempt to C/D retention or trajectory-memory hooks, or run B-specific stronger controls before any renewed semantic claim.",
    ]

    write_csv(STAGE6 / "semantic_causality_rows.csv", decision_rows)
    write_json(STAGE6 / "stage6_summary.json", summary)
    write_text(STAGE6 / "semantic_causality_report.md", "\n".join(causality_report))
    write_text(STAGE6 / "semantic_safety_filter_report.md", "\n".join(safety_report))
    write_text(RESULT_ROOT / "SEMANTIC_CONTENT_NOT_CAUSAL_YET.md", "\n".join(content_not_causal))
    return summary


def build_stage7(best_policy_id: str, full_rows: list[dict[str, str]]) -> dict[str, Any]:
    loger_summary = read_json(V105_LOGER / "loger_comparison_summary.json")
    loger_rows = read_csv(V105_LOGER / "loger_comparison_metrics.csv")
    base_rows = baseline_by_seq()
    f19_rows = f19_by_seq()
    cand_rows = {row.get("seq", ""): row for row in policy_full_rows(best_policy_id, full_rows)}
    comparison_rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        loger = next((row for row in loger_rows if row.get("seq") == seq), {})
        comparison_rows.append(
            {
                "schema": "acl2_v110r_stage7_loger_comparison_row_v1",
                "seq": seq,
                "comparison_available": False,
                "blocker": "loger_full_sequence_l0_l4_trajectory_metric_missing_or_not_comparable",
                "loger_comparison_scope": loger.get("comparison_scope", loger_summary.get("loger_comparison_scope", "")),
                "loger_full_ATE": "",
                "lingbot_baseline_full_ATE": base_rows.get(seq, {}).get("ATE_full_sim3_m", ""),
                "f19_full_ATE": f19_rows.get(seq, {}).get("full_ATE_sim3", ""),
                "candidate_policy_id": best_policy_id,
                "candidate_full_ATE": cand_rows.get(seq, {}).get("full_ATE_sim3", ""),
                "gap_to_loger_baseline": "",
                "gap_closed_by_f19": "",
                "gap_closed_by_candidate": "",
                "local_window_ATE_comparison": "blocked_missing_loger_full_sequence_local_window_metric",
                "rolling_p90_comparison": "blocked_missing_loger_full_sequence_rolling_metric",
            }
        )

    summary = {
        "schema": "acl2_v110r_stage7_loger_comparison_summary_v1",
        "comparison_available": False,
        "blocker": "direct_comparable_loger_full_sequence_l0_l4_trajectory_artifact_missing",
        "which_loger_artifact_missing": "LoGeR full-sequence KITTI 00/01/02/05 trajectory metrics under the same full ATE Sim3, rolling p90, and local-window protocol",
        "which_metric_not_comparable": [
            "loger_full_ATE",
            "local_window_ATE_comparison",
            "rolling_p90_comparison",
        ],
        "required_command_or_table": "Produce a LoGeR full-sequence 00/01/02/05 metric table with seq, full_ATE_sim3, rolling_ATE_p90, and local_window_ATE_median under the same evaluator/protocol before computing gap_closed.",
        "v105_loger_comparison_scope": loger_summary.get("loger_comparison_scope", ""),
        "v105_direct_l0_l4_trajectory_comparison_available": loger_summary.get("direct_l0_l4_trajectory_comparison_available"),
        "reason_direct_l0_l4_missing": loger_summary.get("reason_direct_l0_l4_missing", ""),
        "candidate_policy_id": best_policy_id,
        "outputs": {
            "loger_vs_lingbot_comparison": rel(CROSS / "loger_vs_lingbot_comparison.csv"),
            "blocked_report": rel(CROSS / "LOGER_COMPARISON_BLOCKED.md"),
            "comparison_summary": rel(CROSS / "comparison_summary.json"),
        },
    }

    blocked_report = [
        "# ACL2 v110R LoGeR Comparison Blocked",
        "",
        "comparison_available: False",
        f"blocker: {summary['blocker']}",
        "",
        "## Missing Artifact",
        "",
        summary["which_loger_artifact_missing"],
        "",
        "## Metric Not Comparable",
        "",
        "- `loger_full_ATE`",
        "- `local_window_ATE_comparison`",
        "- `rolling_p90_comparison`",
        "",
        "## Required Command Or Table",
        "",
        summary["required_command_or_table"],
        "",
        "## Existing Evidence Boundary",
        "",
        f"- v105 comparison scope: `{summary['v105_loger_comparison_scope']}`",
        f"- v105 direct L0-L4 comparison available: `{summary['v105_direct_l0_l4_trajectory_comparison_available']}`",
        f"- v105 reason: {summary['reason_direct_l0_l4_missing']}",
        "",
        "No LoGeR gap-closed number is reported in v110R because doing so would require fabricating a missing full-sequence LoGeR metric.",
    ]

    write_csv(CROSS / "loger_vs_lingbot_comparison.csv", comparison_rows)
    write_json(CROSS / "comparison_summary.json", summary)
    write_text(CROSS / "LOGER_COMPARISON_BLOCKED.md", "\n".join(blocked_report))
    return summary


def build_final(
    best_policy: dict[str, str],
    full_rows: list[dict[str, str]],
    stage4_summary: dict[str, Any],
    stage5_summary: dict[str, Any],
    stage6_summary: dict[str, Any],
    stage7_summary: dict[str, Any],
) -> dict[str, Any]:
    best_policy_id = best_policy.get("policy_id", "")
    best_rows = policy_full_rows(best_policy_id, full_rows)
    f19_rows = f19_by_seq()
    per_seq = []
    for row in best_rows:
        seq = row.get("seq", "")
        cand_rel = fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        f19_rel = fnum(f19_rows.get(seq, {}).get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
        per_seq.append(
            {
                "seq": seq,
                "candidate_full_rel": cand_rel,
                "f19_full_rel": f19_rel,
                "candidate_minus_f19_full_rel": cand_rel - f19_rel if math.isfinite(cand_rel) and math.isfinite(f19_rel) else float("nan"),
                "candidate_full_ATE": fnum(row.get("full_ATE_sim3", "nan")),
                "f19_full_ATE": fnum(f19_rows.get(seq, {}).get("full_ATE_sim3", "nan")),
                "local_window_rel": fnum(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan")),
                "adjacent_log_scale_jump_p90": fnum(row.get("adjacent_log_scale_jump_p90", "nan")),
                "global_sim3_scale": fnum(row.get("global_sim3_scale", "nan")),
                "baseline_global_sim3_scale": fnum(row.get("baseline_global_sim3_scale", "nan")),
            }
        )
    exceeds_f19_seq_count = sum(
        1
        for row in per_seq
        if math.isfinite(row["candidate_minus_f19_full_rel"]) and row["candidate_minus_f19_full_rel"] > 0.0
    )
    full_rels = [row["candidate_full_rel"] for row in per_seq]
    local_rels = [row["local_window_rel"] for row in per_seq]
    adjacent_p90 = [row["adjacent_log_scale_jump_p90"] for row in per_seq]
    scales = [row["global_sim3_scale"] for row in per_seq]
    baseline_scales = [row["baseline_global_sim3_scale"] for row in per_seq]
    stage4_policy_answer = {
        "policy_id": best_policy_id,
        "candidate_id": best_policy.get("candidate_id", ""),
        "surface_id": best_policy.get("surface_id", ""),
        "policy_family": best_policy.get("policy_family", ""),
        "median_full_rel": fnum(best_policy.get("median_full_rel", "nan")),
        "mean_full_rel": fnum(best_policy.get("mean_full_rel", "nan")),
        "improved_seq_count": int(float(best_policy.get("improved_seq_count", 0))) if best_policy.get("improved_seq_count") else 0,
        "max_harm": fnum(best_policy.get("max_harm", "nan")),
        "rolling_p90_median_rel": fnum(best_policy.get("rolling_p90_median_rel", "nan")),
        "final_error_median_rel": fnum(best_policy.get("final_error_median_rel", "nan")),
        "local_window_median_harm": fnum(best_policy.get("local_window_median_harm", "nan")),
        "exceeds_f19_sequence_count": exceeds_f19_seq_count,
        "per_sequence": per_seq,
        "median_adjacent_log_scale_jump_p90": median(adjacent_p90),
        "median_global_sim3_scale": median(scales),
        "median_baseline_global_sim3_scale": median(baseline_scales),
        "median_local_window_rel": median(local_rels),
        "max_local_harm": max_rel_harm(local_rels),
        "median_full_rel_recomputed": median(full_rels),
        "mean_full_rel_recomputed": mean(full_rels),
    }

    questions = {
        "1_candidate_exceeds_f19": exceeds_f19_seq_count == len(SEQUENCES),
        "1_candidate_exceeds_f19_note": "B1 beats F19 on all four relative full-ATE sequence rows and by median/mean Stage4 gates.",
        "2_candidate_passes_four_sequence_full_geometry_gate": boolish(best_policy.get("stage4_geometry_pass")),
        "3_candidate_passes_semantic_causality_gate": False,
        "4_most_promising_surface": "B surface as internal/cache-schedule baseline, not semantic method",
        "5_hard_negatives_01_05_protected": (
            fnum(best_policy.get("seq01_full_rel", "nan")) > 0.0
            and fnum(best_policy.get("seq05_full_rel", "nan")) > 0.0
            and fnum(best_policy.get("max_harm", "nan")) <= 0.0
        ),
        "6_full_ate_supported_by_rolling_local_scale": (
            "Rolling/local/final gates support B1; scale and yaw are recorded in full_metric_rows but do not prove semantic causality."
        ),
        "7_loger_gap_comparison": "blocked_missing_comparable_loger_full_sequence_metric",
        "8_next_route": "turn_to_C_D_retention_or_trajectory_hooks; retain_B1_as_internal_schedule_baseline",
    }

    final_decision = {
        "schema": "acl2_v110r_final_decision_v1",
        "plan_completed": True,
        "scientific_goal_achieved_as_semantic_aware_method": False,
        "final_taxonomy": FINAL_TAXONOMY,
        "primary_candidate_policy_id": best_policy_id,
        "primary_candidate": stage4_policy_answer,
        "stage4_taxonomy": stage4_summary.get("taxonomy", ""),
        "stage5_taxonomy": stage5_summary.get("reason", ""),
        "stage6_taxonomy": stage6_summary.get("taxonomy", ""),
        "stage7_taxonomy": "LOGER_COMPARISON_BLOCKED",
        "semantic_causality_pass": False,
        "full_geometry_pass": boolish(best_policy.get("stage4_geometry_pass")),
        "strong_full_ate_improvement_pass": boolish(best_policy.get("strong_improvement_pass")),
        "loger_comparison_available": stage7_summary.get("comparison_available", False),
        "answers": questions,
        "next_recommended_action": (
            "Do not claim semantic-aware success. Preserve B1 as a strong internal/cache-schedule baseline, "
            "then move to C/D minimal hooks or run B-specific stronger semantic controls before any renewed claim."
        ),
        "outputs": {
            "stage5_summary": rel(STAGE5 / "stage5_summary.json"),
            "stage6_summary": rel(STAGE6 / "stage6_summary.json"),
            "stage7_summary": rel(CROSS / "comparison_summary.json"),
            "final_decision": rel(FINAL / "final_decision.json"),
            "final_report": rel(FINAL / "final_report.md"),
        },
    }

    report = [
        "# ACL2 v110R Final Decision",
        "",
        f"final_taxonomy: {FINAL_TAXONOMY}",
        "plan_completed: True",
        "scientific_goal_achieved_as_semantic_aware_method: False",
        f"primary_candidate_policy_id: {best_policy_id}",
        "",
        "## Eight Required Answers",
        "",
        f"1. Candidate exceeds F19: `{questions['1_candidate_exceeds_f19']}`. {questions['1_candidate_exceeds_f19_note']}",
        f"2. Candidate passes four-sequence full geometry gate: `{questions['2_candidate_passes_four_sequence_full_geometry_gate']}`.",
        f"3. Candidate passes semantic causality gate: `{questions['3_candidate_passes_semantic_causality_gate']}`.",
        f"4. Most promising surface: {questions['4_most_promising_surface']}.",
        f"5. 01/05 hard negatives protected: `{questions['5_hard_negatives_01_05_protected']}`.",
        f"6. Full ATE support: {questions['6_full_ate_supported_by_rolling_local_scale']}",
        f"7. LoGeR gap comparison: `{questions['7_loger_gap_comparison']}`.",
        f"8. Next route: {questions['8_next_route']}.",
        "",
        "## Primary Candidate Metrics",
        "",
        f"- median_full_rel: `{stage4_policy_answer['median_full_rel']}`",
        f"- mean_full_rel: `{stage4_policy_answer['mean_full_rel']}`",
        f"- improved_seq_count: `{stage4_policy_answer['improved_seq_count']}/4`",
        f"- max_harm: `{stage4_policy_answer['max_harm']}`",
        f"- rolling_p90_median_rel: `{stage4_policy_answer['rolling_p90_median_rel']}`",
        f"- final_error_median_rel: `{stage4_policy_answer['final_error_median_rel']}`",
        f"- local_window_median_harm: `{stage4_policy_answer['local_window_median_harm']}`",
        "",
        "## Boundary",
        "",
        "B1 is a strong full-ATE/internal-schedule baseline. It is not a semantic-aware method because the semantic_plus_internal and semantic_only variants match exactly, and the registered stronger controls do not support a semantic content claim.",
        "",
        "LoGeR gap closure is not reported because the comparable LoGeR full-sequence metric artifact is missing.",
    ]

    write_json(FINAL / "final_decision.json", final_decision)
    write_text(FINAL / "final_report.md", "\n".join(report))
    return final_decision


def main() -> None:
    policy_rows = read_csv(STAGE4 / "policy_summary_rows.csv")
    semantic_rows = read_csv(STAGE4 / "semantic_control_rows.csv")
    full_rows = read_csv(STAGE4 / "full_metric_rows.csv")
    stage4_summary = read_json(STAGE4 / "stage4_summary.json")
    if not policy_rows:
        raise SystemExit(f"missing policy rows: {STAGE4 / 'policy_summary_rows.csv'}")
    if not semantic_rows:
        raise SystemExit(f"missing semantic rows: {STAGE4 / 'semantic_control_rows.csv'}")
    best_policy = best_stage4_policy(policy_rows)
    best_policy_id = best_policy.get("policy_id", "")
    stage5_summary = build_stage5(policy_rows, stage4_summary)
    stage6_summary = build_stage6(policy_rows, semantic_rows, stage4_summary)
    stage7_summary = build_stage7(best_policy_id, full_rows)
    final_decision = build_final(
        best_policy,
        full_rows,
        stage4_summary,
        stage5_summary,
        stage6_summary,
        stage7_summary,
    )
    print(json.dumps(clean_json({
        "stage5_composition_search_run": stage5_summary["composition_search_run"],
        "stage6_taxonomy": stage6_summary["taxonomy"],
        "stage7_comparison_available": stage7_summary["comparison_available"],
        "final_taxonomy": final_decision["final_taxonomy"],
        "semantic_causality_pass": final_decision["semantic_causality_pass"],
        "primary_candidate_policy_id": final_decision["primary_candidate_policy_id"],
        "plan_completed": final_decision["plan_completed"],
        "scientific_goal_achieved_as_semantic_aware_method": final_decision["scientific_goal_achieved_as_semantic_aware_method"],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
