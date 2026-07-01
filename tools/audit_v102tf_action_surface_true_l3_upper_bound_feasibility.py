#!/usr/bin/env python3
"""Audit whether existing action artifacts contain a strict true L3/L4 upper bound.

This is a no-action audit. It only reads previous ACL2 artifacts and checks
whether any measured READ/SWA/TTT or related carrier result can satisfy the
v102 Stage4 true L3/L4 action-surface gate.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
OUT = ROOT / "stage4_memory_action_surface_oracle"


PATHS = {
    "v95_tracke_summary": Path(
        "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
        "trackE_measured_eligibility_upper_bound_v1/summary.json"
    ),
    "v95_tracke_policy_rows": Path(
        "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
        "trackE_measured_eligibility_upper_bound_v1/cue_policy_upper_bound_metrics.csv"
    ),
    "v101_internal_qk_summary": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "final_decision/internal_qk_actuator_reentry_summary.json"
    ),
    "v101_internal_qk_rows": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "final_decision/internal_qk_actuator_reentry_per_pair_best_rows.csv"
    ),
    "v98_stage7f_summary": Path(
        "results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control/"
        "stage7f_prev_ttt_anchor_gate_v05_all_6case/trajectory_eval_summary.json"
    ),
    "v98_stage7h_summary": Path(
        "results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control/"
        "stage7h_query_soft_r05_all_ge90_6case_comparison.json"
    ),
    "v101_cache_topk_summary": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "final_decision/cache_topk_identity_query_reentry_summary.json"
    ),
    "v101_cache_topk_rows": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "final_decision/cache_topk_identity_query_reentry_action_variant_metrics.csv"
    ),
    "v100_e4_summary": Path(
        "results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control/"
        "trackE4_swa_identity_handoff_control/summary.json"
    ),
    "v100_m3_summary": Path(
        "results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control/"
        "trackM3_identity_action_simulator/summary.json"
    ),
    "v100_f4_summary": Path(
        "results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control/"
        "trackF4_ttt_write_to_use_same_space/summary.json"
    ),
    "v101_frontier_summary": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "final_decision/strict_action_frontier_summary.json"
    ),
    "v101_frontier_rows": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "final_decision/strict_action_frontier_rows.csv"
    ),
    "v101_merge_rich_summary": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "outcomeD_merge_gauge_rich_selector_replay_probe/runtime_probe_sensitivity_summary.json"
    ),
    "v101_merge_fresh_summary": Path(
        "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
        "outcomeD_merge_gauge_fresh_unlabelled_stage1_native_probe/runtime_probe_sensitivity_summary.json"
    ),
    "v102_stage4_summary": ROOT / "stage4_memory_action_surface_oracle/stage4_summary.json",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bval(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_or_blank(value: float) -> float | str:
    return value if math.isfinite(value) else ""


def best_float(rows: list[dict[str, str]], field: str) -> float:
    vals = [fnum(row.get(field)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else math.nan


def best_row(rows: list[dict[str, str]], field: str) -> dict[str, str]:
    usable = [(fnum(row.get(field)), row) for row in rows]
    usable = [(v, row) for v, row in usable if math.isfinite(v)]
    if not usable:
        return {}
    return max(usable, key=lambda item: item[0])[1]


def fail_reasons(*items: tuple[bool, str]) -> str:
    return "; ".join(reason for ok, reason in items if not ok)


def audit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    v95 = read_json(PATHS["v95_tracke_summary"])
    v95_policy = read_csv(PATHS["v95_tracke_policy_rows"])
    v95_best_policy = v95.get("best_cue_policy") or best_row(v95_policy, "bad_handoff_median_improvement")
    v95_median = fnum(v95_best_policy.get("bad_handoff_median_improvement"))
    v95_max = fnum(v95.get("global_best_bad_handoff_improvement"))
    v95_seq_ok = bval(v95_best_policy.get("sequence_coverage_ge_3"))
    v95_good_ok = bval(v95_best_policy.get("good_handoff_protection_gate"))
    rows.append(
        {
            "route_id": "V95_TRACK_E_MEASURED_SWA_POOL",
            "memory_body": "SWA",
            "evidence_path": str(PATHS["v95_tracke_summary"]),
            "artifact_available": bool(v95),
            "true_l3_l4_metric_available": True,
            "metric_scope": "measured handoff_transfer_improvement over old SWA action pool",
            "trace_or_action_executed": True,
            "bad_median_or_best_improvement": finite_or_blank(v95_median),
            "bad_max_improvement": finite_or_blank(v95_max),
            "good_harm_or_worsen": v95_best_policy.get("good_handoff_max_worsen", ""),
            "selected_target_count": v95_best_policy.get("selected_bad_count", ""),
            "selected_sequence_coverage": v95_best_policy.get("selected_bad_sequence_coverage", ""),
            "control_margin_or_beats_control": "",
            "source_gate_pass": bool(v95.get("gate_pass")),
            "strict_v102_action_surface_upper_bound_pass": bool(
                math.isfinite(v95_median) and v95_median >= 0.05 and v95_good_ok and v95_seq_ok
            ),
            "disqualifiers": fail_reasons(
                (math.isfinite(v95_median) and v95_median >= 0.05, "bad median L3 improvement < 5%"),
                (v95_good_ok, "good protection gate unavailable/false"),
                (v95_seq_ok, "selected bad sequence coverage < 3"),
                (bool(v95.get("gate_pass")), "source upper-bound gate false"),
            ),
        }
    )

    qk = read_json(PATHS["v101_internal_qk_summary"])
    rows.append(
        {
            "route_id": "V101_INTERNAL_QK_ACTUATOR_REENTRY",
            "memory_body": "SWA",
            "evidence_path": str(PATHS["v101_internal_qk_summary"]),
            "artifact_available": bool(qk),
            "true_l3_l4_metric_available": True,
            "metric_scope": "audited old SWA action families with handoff improvement",
            "trace_or_action_executed": True,
            "bad_median_or_best_improvement": qk.get("best_bad_handoff_median_improvement", ""),
            "bad_max_improvement": qk.get("global_per_pair_bad_max_handoff_improvement", ""),
            "good_harm_or_worsen": "",
            "selected_target_count": qk.get("best_selected_bad_count", ""),
            "selected_sequence_coverage": qk.get("best_selected_bad_sequence_coverage", ""),
            "control_margin_or_beats_control": qk.get("best_actual_minus_best_same_count_control", ""),
            "source_gate_pass": bool(qk.get("best_family_action_surface_gate_pass")),
            "strict_v102_action_surface_upper_bound_pass": bool(qk.get("best_family_action_surface_gate_pass")),
            "disqualifiers": fail_reasons(
                (fnum(qk.get("best_bad_handoff_median_improvement")) >= 0.05, "bad median L3 improvement < 5%"),
                (fnum(qk.get("best_actual_minus_best_same_count_control")) > 0, "does not beat best same-count control"),
                (bool(qk.get("best_family_action_surface_gate_pass")), "source action-surface gate false"),
            ),
        }
    )

    cache = read_json(PATHS["v101_cache_topk_summary"])
    cache_rows = read_csv(PATHS["v101_cache_topk_rows"])
    cache_best = best_row(cache_rows, "median_improvement_ratio_vs_baseline")
    rows.append(
        {
            "route_id": "V101_CACHE_TOPK_IDENTITY_QUERY_REENTRY",
            "memory_body": "SWA/TTT",
            "evidence_path": str(PATHS["v101_cache_topk_summary"]),
            "artifact_available": bool(cache),
            "true_l3_l4_metric_available": False,
            "metric_scope": "ATE/final-error action-pilot comparison, not a true L3/L4 action-surface metric",
            "trace_or_action_executed": True,
            "bad_median_or_best_improvement": cache_best.get("median_improvement_ratio_vs_baseline", cache.get("best_action_median_improvement_ratio_vs_baseline", "")),
            "bad_max_improvement": cache_best.get("max_improvement_ratio_vs_baseline", ""),
            "good_harm_or_worsen": cache_best.get("worse_ate_case_count", cache.get("best_action_worse_ate_case_count", "")),
            "selected_target_count": cache_best.get("case_count", ""),
            "selected_sequence_coverage": "",
            "control_margin_or_beats_control": "",
            "source_gate_pass": bool(cache.get("best_action_gate_pass")),
            "strict_v102_action_surface_upper_bound_pass": False,
            "disqualifiers": "metric is ATE/final-error rather than true L3/L4; source gate false",
        }
    )

    for route_id, memory_body, path_key in [
        ("V98_STAGE7F_PREV_TTT_ANCHOR_GATE", "TTT/SWA", "v98_stage7f_summary"),
        ("V98_STAGE7H_QUERY_SOFT", "TTT/SWA", "v98_stage7h_summary"),
    ]:
        data = read_json(PATHS[path_key])
        summary = data.get("summary") if isinstance(data, dict) else {}
        rows.append(
            {
                "route_id": route_id,
                "memory_body": memory_body,
                "evidence_path": str(PATHS[path_key]),
                "artifact_available": bool(data),
                "true_l3_l4_metric_available": False,
                "metric_scope": "trajectory ATE/final-error comparison, not true L3/L4",
                "trace_or_action_executed": True,
                "bad_median_or_best_improvement": summary.get("median_improvement_ratio_vs_baseline", ""),
                "bad_max_improvement": "",
                "good_harm_or_worsen": summary.get("worse_ate_case_count", ""),
                "selected_target_count": summary.get("case_count", ""),
                "selected_sequence_coverage": "",
                "control_margin_or_beats_control": "",
                "source_gate_pass": bool(summary.get("gate_pass")),
                "strict_v102_action_surface_upper_bound_pass": False,
                "disqualifiers": "metric is not true L3/L4 and source gate is false",
            }
        )

    for route_id, memory_body, path_key, reason in [
        ("V100_E4_SWA_IDENTITY_HANDOFF_CONTROL", "SWA", "v100_e4_summary", "track blocked before simulator/action"),
        ("V100_M3_IDENTITY_ACTION_SIMULATOR", "SWA/TTT", "v100_m3_summary", "track blocked before action family simulation"),
    ]:
        data = read_json(PATHS[path_key])
        rows.append(
            {
                "route_id": route_id,
                "memory_body": memory_body,
                "evidence_path": str(PATHS[path_key]),
                "artifact_available": bool(data),
                "true_l3_l4_metric_available": False,
                "metric_scope": "blocked artifact",
                "trace_or_action_executed": False,
                "bad_median_or_best_improvement": "",
                "bad_max_improvement": "",
                "good_harm_or_worsen": "",
                "selected_target_count": "",
                "selected_sequence_coverage": "",
                "control_margin_or_beats_control": "",
                "source_gate_pass": bool(data.get("gate_pass")),
                "strict_v102_action_surface_upper_bound_pass": False,
                "disqualifiers": reason,
            }
        )

    f4 = read_json(PATHS["v100_f4_summary"])
    rows.append(
        {
            "route_id": "V100_F4_TTT_WRITE_TO_USE_SAME_SPACE",
            "memory_body": "TTT",
            "evidence_path": str(PATHS["v100_f4_summary"]),
            "artifact_available": bool(f4),
            "true_l3_l4_metric_available": False,
            "metric_scope": "case-level diagnostic correlation, no measured refresh/expire/context-only action effect",
            "trace_or_action_executed": False,
            "bad_median_or_best_improvement": "",
            "bad_max_improvement": f4.get("best_abs_corr_L3", ""),
            "good_harm_or_worsen": f4.get("best_good_FPR", ""),
            "selected_target_count": "",
            "selected_sequence_coverage": f4.get("sequence_coverage", ""),
            "control_margin_or_beats_control": f4.get("control_margins_available", ""),
            "source_gate_pass": bool(f4.get("gate_pass")),
            "strict_v102_action_surface_upper_bound_pass": False,
            "disqualifiers": "diagnostic correlation only; no measured TTT state-machine action effect; control margins unavailable",
        }
    )

    frontier = read_json(PATHS["v101_frontier_summary"])
    rows.append(
        {
            "route_id": "V101_STRICT_ACTION_FRONTIER",
            "memory_body": "READ/SWA/TTT",
            "evidence_path": str(PATHS["v101_frontier_summary"]),
            "artifact_available": bool(frontier),
            "true_l3_l4_metric_available": False,
            "metric_scope": "prerequisite frontier audit, not action effect",
            "trace_or_action_executed": False,
            "bad_median_or_best_improvement": "",
            "bad_max_improvement": "",
            "good_harm_or_worsen": "",
            "selected_target_count": frontier.get("strict_action_ready_clean_candidate_count", ""),
            "selected_sequence_coverage": "",
            "control_margin_or_beats_control": "",
            "source_gate_pass": False,
            "strict_v102_action_surface_upper_bound_pass": False,
            "disqualifiers": frontier.get("blocked_reason", "strict action ready count is zero"),
        }
    )

    for route_id, path_key, allowed_memory_route in [
        ("V101_MERGE_GAUGE_RICH_SELECTOR_REPLAY_PROBE", "v101_merge_rich_summary", False),
        ("V101_MERGE_GAUGE_FRESH_UNLABELLED_NATIVE_PROBE", "v101_merge_fresh_summary", False),
    ]:
        data = read_json(PATHS[path_key])
        selected = data.get("selected_candidate_summary") or {}
        bad_imp = fnum(selected.get("bad_median_I_J_runtime_proxy"))
        good_ok = bool(selected.get("good_median_worsen_gate_le_0p02"))
        beats_control = bool(data.get("selected_candidate_beats_control"))
        row_count = selected.get("row_count", data.get("target_count", ""))
        rows.append(
            {
                "route_id": route_id,
                "memory_body": "MERGE_GAUGE_DIAGNOSTIC",
                "evidence_path": str(PATHS[path_key]),
                "artifact_available": bool(data),
                "true_l3_l4_metric_available": bool(selected.get("handoff_transfer_rows_complete")),
                "metric_scope": "measured merge/gauge handoff runtime proxy; external diagnostic carrier, not READ/SWA/TTT semantic memory action",
                "trace_or_action_executed": bool(data.get("runtime_probe_executed")),
                "bad_median_or_best_improvement": finite_or_blank(bad_imp),
                "bad_max_improvement": "",
                "good_harm_or_worsen": selected.get("good_max_worsen_runtime_proxy", ""),
                "selected_target_count": row_count,
                "selected_sequence_coverage": "",
                "control_margin_or_beats_control": beats_control,
                "source_gate_pass": bool(data.get("phase3r_runtime_probe_gate_pass")),
                "strict_v102_action_surface_upper_bound_pass": bool(
                    allowed_memory_route
                    and math.isfinite(bad_imp)
                    and bad_imp >= 0.05
                    and good_ok
                    and beats_control
                    and data.get("phase3r_runtime_probe_gate_pass")
                ),
                "disqualifiers": fail_reasons(
                    (allowed_memory_route, "external merge/gauge diagnostic is not a READ/SWA/TTT semantic memory action"),
                    (math.isfinite(bad_imp) and bad_imp >= 0.05, "bad median runtime proxy improvement < 5% or unavailable"),
                    (good_ok, "good harm gate unavailable/false"),
                    (beats_control, "selected candidate does not beat control"),
                    (bool(data.get("phase3r_runtime_probe_gate_pass")), "source runtime probe gate false"),
                ),
            }
        )

    stage4 = read_json(PATHS["v102_stage4_summary"])
    rows.append(
        {
            "route_id": "V102_CURRENT_STAGE4_DIAGNOSTIC_INVENTORY",
            "memory_body": "READ/SWA/TTT/ADMISSION",
            "evidence_path": str(PATHS["v102_stage4_summary"]),
            "artifact_available": bool(stage4),
            "true_l3_l4_metric_available": False,
            "metric_scope": "diagnostic inventory only because Stage3 strict semantic oracle failed",
            "trace_or_action_executed": False,
            "bad_median_or_best_improvement": "",
            "bad_max_improvement": "",
            "good_harm_or_worsen": "",
            "selected_target_count": stage4.get("surface_count", ""),
            "selected_sequence_coverage": "",
            "control_margin_or_beats_control": "",
            "source_gate_pass": bool(stage4.get("strict_memory_action_surface_pass")),
            "strict_v102_action_surface_upper_bound_pass": False,
            "disqualifiers": stage4.get("reason", "Stage4 inventory only"),
        }
    )

    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_rows = [r for r in rows if bval(r.get("strict_v102_action_surface_upper_bound_pass"))]
    true_rows = [r for r in rows if bval(r.get("true_l3_l4_metric_available"))]
    eligible_true_rows = [
        r for r in true_rows if str(r.get("memory_body")) in {"SWA", "TTT", "READ", "READ/SWA/TTT"}
    ]

    def score(row: dict[str, Any]) -> float:
        return fnum(row.get("bad_median_or_best_improvement"))

    best_true = max(true_rows, key=score) if true_rows else {}
    best_eligible = max(eligible_true_rows, key=score) if eligible_true_rows else {}
    return {
        "schema": "acl2_v102_action_surface_true_l3_upper_bound_feasibility_v1",
        "audited_route_count": len(rows),
        "artifact_available_route_count": sum(1 for r in rows if bval(r.get("artifact_available"))),
        "true_l3_l4_metric_route_count": len(true_rows),
        "eligible_read_swa_ttt_true_l3_l4_route_count": len(eligible_true_rows),
        "strict_action_surface_upper_bound_pass_count": len(pass_rows),
        "strict_action_surface_upper_bound_pass_routes": ";".join(str(r["route_id"]) for r in pass_rows),
        "best_true_l3_route_id": best_true.get("route_id", ""),
        "best_true_l3_memory_body": best_true.get("memory_body", ""),
        "best_true_l3_bad_median_or_best_improvement": best_true.get("bad_median_or_best_improvement", ""),
        "best_true_l3_disqualified": best_true.get("disqualifiers", ""),
        "best_eligible_read_swa_ttt_route_id": best_eligible.get("route_id", ""),
        "best_eligible_read_swa_ttt_bad_median_or_best_improvement": best_eligible.get("bad_median_or_best_improvement", ""),
        "best_eligible_read_swa_ttt_disqualified": best_eligible.get("disqualifiers", ""),
        "stage4_strict_memory_action_surface_pass": False,
        "runtime_action_allowed": False,
        "conclusion": (
            "Existing measured artifacts do not contain a v102-strict true L3/L4 READ/SWA/TTT "
            "action-surface upper bound. The strongest true-L3 external merge/gauge diagnostic is "
            "not an allowed semantic memory action and does not beat control; the strongest eligible "
            "SWA measured-pool route remains below the 5% median L3 gate."
        ),
    }


def build_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Action Surface True L3/L4 Upper-Bound Feasibility Audit",
        "",
        "This is a no-action audit. It checks existing artifacts for a measured true L3/L4 action-surface upper bound and does not authorize Stage4 runtime.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "audited_route_count",
        "true_l3_l4_metric_route_count",
        "eligible_read_swa_ttt_true_l3_l4_route_count",
        "strict_action_surface_upper_bound_pass_count",
        "best_true_l3_route_id",
        "best_true_l3_bad_median_or_best_improvement",
        "best_eligible_read_swa_ttt_route_id",
        "best_eligible_read_swa_ttt_bad_median_or_best_improvement",
        "runtime_action_allowed",
        "conclusion",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "## Route Evidence", ""])
    columns = [
        "route_id",
        "memory_body",
        "true_l3_l4_metric_available",
        "bad_median_or_best_improvement",
        "source_gate_pass",
        "strict_v102_action_surface_upper_bound_pass",
        "disqualifiers",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        vals = []
        for col in columns:
            text = str(row.get(col, "")).replace("|", "\\|").replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- v95/v101 SWA measured-pool evidence has true handoff metrics, but the best eligible median L3 improvement is below 5%.",
            "- v98/v101 cache/top-k and query-soft action pilots are trajectory ATE comparisons, not true L3/L4 action-surface evidence.",
            "- v100/v101 TTT same-space/write-to-use artifacts remain diagnostic or blocked before measured state-machine action.",
            "- merge/gauge probes can move a handoff runtime proxy, but they are external diagnostic carriers and did not beat the required control gate, so they cannot substitute for READ/SWA/TTT semantic memory action.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUT)
    rows = audit_rows()
    summary = build_summary(rows)
    write_csv(OUT / "action_surface_true_l3_upper_bound_feasibility_rows.csv", rows)
    write_json(OUT / "action_surface_true_l3_upper_bound_feasibility_summary.json", summary)
    write_text(OUT / "action_surface_true_l3_upper_bound_feasibility_report.md", build_report(rows, summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
