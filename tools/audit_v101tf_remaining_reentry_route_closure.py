#!/usr/bin/env python3
"""Close the remaining Outcome-D re-entry routes for ACL2 v101.

Routes 4 and 5 are not fresh runtime-action candidates: READ is retained as an
instrumentation/provider route after v96 Stage7 full No-Go, and semantic-anchor
state is frozen until a new action-ready target universe exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


V96_ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
V101_ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
OUT = V101_ROOT / "final_decision"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compact(value: Any, max_len: int = 900) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v96 = read_json(V96_ROOT / "final_decision" / "summary.json")
    v101 = read_json(V101_ROOT / "final_decision" / "summary.json")
    v101_new = read_json(OUT / "new_v100_schema_universe_feasibility_summary.json")

    v96_stage7 = v96.get("stage7_full_validation", {})
    v96_trackd = v96.get("trackD_read_gauge_preserving_action_pilots", {})

    rows = [
        {
            "route_id": "read_provider_only_reentry",
            "closure_status": "provider_only_not_runtime_action",
            "positive_evidence": compact(
                {
                    "mechanism_success": v96.get("mechanism_success"),
                    "diagnostic_success": v96.get("diagnostic_success"),
                    "trackD_gate_pass": v96_trackd.get("gate_pass"),
                    "trackD_completed_pilot_count": v96_trackd.get("completed_pilot_count"),
                    "trackD_best_bad_improvement_vs_baseline": v96_trackd.get(
                        "best_bad_improvement_vs_baseline"
                    ),
                    "trackD_best_candidate_margin_vs_random_same_mass": v96_trackd.get(
                        "best_candidate_margin_vs_random_same_mass"
                    ),
                }
            ),
            "blocking_evidence": compact(
                {
                    "runtime_action_allowed": v96.get("runtime_action_allowed"),
                    "method_success": v96.get("method_success"),
                    "full_method_success": v96.get("full_method_success"),
                    "stage7_gate_pass": v96_stage7.get("gate_pass"),
                    "stage7_candidate_count": v96_stage7.get("candidate_count"),
                    "stage7_best_candidate": v96_stage7.get("best_candidate_by_delta_ate"),
                    "stage7_best_delta_ate": v96_stage7.get("best_delta_aligned_ate_rmse_m"),
                    "stage7_best_delta_final_error": v96_stage7.get("best_delta_final_error_m"),
                    "stage7_strict_reason": v96_stage7.get("best_strict_full_gate_reason"),
                }
            ),
            "action_allowed": False,
            "recommended_next_step": (
                "Use READ only as a diagnostic/provider/instrumentation source. Do not revive READ L07 or "
                "weak-context skip as a runtime action without a new full-gate design."
            ),
        },
        {
            "route_id": "semantic_anchor_state_freeze_until_new_universe",
            "closure_status": "frozen_until_new_action_ready_universe",
            "positive_evidence": compact(
                {
                    "trackU_gate_pass": v101.get("trackU_gate_pass"),
                    "trackW_gate_pass": v101.get("trackW_gate_pass"),
                    "core_v100_schema_ready_clean_candidate_count": v101_new.get(
                        "core_v100_schema_ready_clean_candidate_count"
                    ),
                    "clean_candidate_count": v101_new.get("clean_candidate_count"),
                }
            ),
            "blocking_evidence": compact(
                {
                    "trackT_gate_pass": v101.get("trackT_gate_pass"),
                    "trackV_gate_pass": v101.get("trackV_gate_pass"),
                    "trackQ2_true_stage_pass": v101.get("trackQ2_true_stage_pass"),
                    "clean_handoff_candidate_count": v101_new.get("clean_handoff_candidate_count"),
                    "safe_good_candidate_count": v101_new.get("safe_good_candidate_count"),
                    "strict_action_ready_clean_candidate_count": v101_new.get(
                        "strict_action_ready_clean_candidate_count"
                    ),
                    "new_universe_available_from_existing_artifacts": v101_new.get(
                        "new_universe_available_from_existing_artifacts"
                    ),
                    "q2_proxy_only": v101_new.get("q2_proxy_only"),
                }
            ),
            "action_allowed": False,
            "recommended_next_step": (
                "Freeze semantic-anchor action work until a new v100-schema clean handoff target universe exists "
                "with same-space trace, per-anchor geometry, identity/query-head controls, write-to-use chain, and Q2 true-stage pass."
            ),
        },
    ]

    write_csv(
        OUT / "remaining_reentry_route_closure_rows.csv",
        rows,
        [
            "route_id",
            "closure_status",
            "positive_evidence",
            "blocking_evidence",
            "action_allowed",
            "recommended_next_step",
        ],
    )

    summary = {
        "schema": "acl2_v101_remaining_reentry_route_closure_v1",
        "closed_route_count": len(rows),
        "action_allowed_route_count": sum(1 for row in rows if row["action_allowed"]),
        "read_provider_route_action_allowed": False,
        "read_provider_mechanism_success": v96.get("mechanism_success"),
        "read_provider_stage7_candidate_count": v96_stage7.get("candidate_count"),
        "read_provider_stage7_gate_pass": v96_stage7.get("gate_pass"),
        "read_provider_stage7_best_delta_ate": v96_stage7.get("best_delta_aligned_ate_rmse_m"),
        "read_provider_stage7_best_delta_final_error": v96_stage7.get("best_delta_final_error_m"),
        "read_provider_stage7_strict_reason": v96_stage7.get("best_strict_full_gate_reason"),
        "semantic_anchor_freeze_action_allowed": False,
        "semantic_anchor_trackU_gate_pass": v101.get("trackU_gate_pass"),
        "semantic_anchor_trackW_gate_pass": v101.get("trackW_gate_pass"),
        "semantic_anchor_trackT_gate_pass": v101.get("trackT_gate_pass"),
        "semantic_anchor_trackV_gate_pass": v101.get("trackV_gate_pass"),
        "semantic_anchor_trackQ2_true_stage_pass": v101.get("trackQ2_true_stage_pass"),
        "semantic_anchor_strict_action_ready_clean_candidate_count": v101_new.get(
            "strict_action_ready_clean_candidate_count"
        ),
        "semantic_anchor_new_universe_available_from_existing_artifacts": v101_new.get(
            "new_universe_available_from_existing_artifacts"
        ),
        "runtime_action_allowed": False,
        "full_validation_allowed": False,
        "v101_goal_achieved": False,
    }
    with (OUT / "remaining_reentry_route_closure_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    report = [
        "# Remaining Re-entry Route Closure",
        "",
        "This audit closes Outcome-D routes 4 and 5 under existing artifacts.",
        "",
        "## Summary",
        "",
        f"- closed routes: `{summary['closed_route_count']}`",
        f"- action-allowed routes: `{summary['action_allowed_route_count']}`",
        f"- READ mechanism success: `{summary['read_provider_mechanism_success']}`",
        f"- READ Stage7 candidate count: `{summary['read_provider_stage7_candidate_count']}`",
        f"- READ Stage7 gate pass: `{summary['read_provider_stage7_gate_pass']}`",
        f"- READ Stage7 best delta ATE: `{summary['read_provider_stage7_best_delta_ate']}`",
        f"- READ Stage7 best delta final error: `{summary['read_provider_stage7_best_delta_final_error']}`",
        f"- semantic-anchor strict action-ready clean candidates: `{summary['semantic_anchor_strict_action_ready_clean_candidate_count']}`",
        f"- new universe available from existing artifacts: `{summary['semantic_anchor_new_universe_available_from_existing_artifacts']}`",
        "",
        "## Interpretation",
        "",
        "READ remains useful as provider/instrumentation only; semantic-anchor state remains frozen until a new action-ready target universe exists. Neither route authorizes runtime action.",
    ]
    (OUT / "remaining_reentry_route_closure_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
