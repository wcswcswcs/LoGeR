#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
DIAG = RESULT_ROOT / "diagnostics"
REPORTS = RESULT_ROOT / "reports"
STAGE1 = RESULT_ROOT / "stage1_hook_audit"
STAGE2 = RESULT_ROOT / "stage2_alignment_cues"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(label: str, prefix: str, kind: str) -> dict[str, Any]:
    summary = read_json(DIAG / f"{prefix}_summary.json")
    agg = summary.get("aggregate", {})
    return {
        "kind": kind,
        "label": label,
        "prefix": prefix,
        "candidate_name": agg.get("candidate_name", ""),
        "median_full_ATE_rel_improvement": agg.get("median_full_ATE_rel_improvement"),
        "median_rolling_p90_rel_improvement": agg.get("median_rolling_p90_rel_improvement"),
        "median_segment_scale_rel_improvement": agg.get("median_segment_scale_rel_improvement"),
        "max_full_ATE_harm_rel": agg.get("max_full_ATE_harm_rel"),
        "segment_scale_not_worse_all": agg.get("segment_scale_not_worse_all"),
        "pilot_geometry_gate_pass": agg.get("pilot_geometry_gate", {}).get("pass", False),
        "source_path": rel(DIAG / f"{prefix}_summary.json"),
    }


def manifest_status(case: str) -> dict[str, Any]:
    path = DIAG / case / "run_manifest.json"
    data = read_json(path)
    return {
        "case": case,
        "returncode": data.get("returncode", ""),
        "manifest": rel(path),
        "output_root": data.get("output_root", ""),
        "log_path": data.get("log_path", ""),
    }


def fmt_pct(value: Any) -> str:
    try:
        if value is None or value == "":
            return "NA"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "NA"


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def main() -> None:
    rows = [
        metric_row(
            "HS_HG1_pose_head_gate_semantic_internal_mild",
            "stage_hs_hg1_pose_head_gate_semantic_internal_mild_rerun1",
            "candidate_full_pilot",
        ),
        metric_row(
            "HS_HG2_sparse_pose_head_gate_risk_suppress_semantic_internal_medium",
            "stage_hs_hg2_sparse_pose_head_gate_risk_suppress_semantic_internal_medium_rerun1",
            "candidate_full_pilot_repair",
        ),
        metric_row(
            "HS_GQ1_layer23_state_delta_gain_semantic_internal_mild_chunkblock1_notrace",
            "stage_hs_gq1_layer23_state_delta_gain_chunkblock1_rerun1_notrace",
            "candidate_full_pilot_reduced_memory_config_specific",
        ),
    ]
    gq_decision = read_json(DIAG / "stage_hs_gq_decision_summary.json")
    gq_status = gq_decision.get("gq_full_status_rows", [])
    lingbot_full = read_json(RESULT_ROOT / "stage5_lingbot_a2_l2_query_full_pilot_00_02/query_full_metric_summary.json")
    lingbot_control = read_json(RESULT_ROOT / "stage5_lingbot_l2_matched_controls_00_02/l2_control_metric_summary.json")
    lingbot_special_repair = read_json(RESULT_ROOT / "stage5_lingbot_l2_special_weight_repair_00_02/l2_control_metric_summary.json")
    lingbot_policy_rows = read_csv(RESULT_ROOT / "stage5_lingbot_a2_l2_query_full_pilot_00_02/policy_summary_rows.csv")
    lingbot_special_policy_rows = read_csv(RESULT_ROOT / "stage5_lingbot_l2_special_weight_repair_00_02/policy_summary_rows.csv")
    lingbot_l2_candidate = find_row(lingbot_policy_rows, "policy_id", "LB_L2_special_query_local_risk_suppress_smoke")
    lingbot_special_candidate = find_row(lingbot_special_policy_rows, "policy_id", "LB_L2_special_query_local_special_weight_repair")
    stage1 = read_json(STAGE1 / "stage1_hook_audit_summary.json")
    stage2 = read_json(STAGE2 / "stage2_alignment_cues_summary.json")
    stage0 = read_json(RESULT_ROOT / "stage0_evidence_freeze/stage0_summary.json")
    stage1_complete = (
        stage1.get("stage1_status") == "pass"
        and bool(stage1.get("strict_pose_noop_parity_pass"))
        and bool(stage1.get("hook_level_noop_parity_pass"))
    )
    stage2_complete = stage2.get("stage2_status") == "pass"
    hg1_pass = bool(rows[0].get("pilot_geometry_gate_pass"))
    hg2_pass = bool(rows[1].get("pilot_geometry_gate_pass"))
    gq1_config_pass = bool(rows[2].get("pilot_geometry_gate_pass"))
    any_geometry_pass = any(bool(row.get("pilot_geometry_gate_pass")) for row in rows)
    lingbot_plan_gate_pass_count = int(lingbot_full.get("pilot_geometry_gate_pass_count") or 0)
    lingbot_positive_safety_count = int(lingbot_full.get("positive_safety_gate_pass_count") or 0)
    lingbot_l2_controls_complete = bool(
        lingbot_control.get("metric_complete")
        and lingbot_control.get("all_action_fidelity")
    )
    final_taxonomy = "NO_GO_TRACE_PASS_NO_GEOMETRY_EFFECT"
    decision_reason = (
        "HS-HG non-value-scaling actions produced valid action traces and full 00/02 pilot metrics, "
        "but median geometry gains stayed below the 5% gate; the sparse repair fixed head-fraction fidelity "
        "but introduced segment-scale not-worse failure. HS-GQ token/state actions reached smoke/action fidelity; "
        "GQ3/GQ4/GQ1 default full promotion remained OOM on 22GB GPUs after in-place, no-trace/no-audit, "
        "and selected-layer repairs. The only completed GQ1 full 00/02 metrics use chunk_block_num=1 and layer23, "
        "which is a reduced-memory config-specific pilot and still misses the 5% geometry gate. "
        "LingBot A2/L2 query/context-specific hooks were repaired and ran smoke/full 00/02 plus L2 matched controls; "
        "after correcting the summarizer threshold, no LingBot policy reaches the plan-level 5% pilot geometry gate. "
        "The first L2 matched-control subset did not match the weak positive L2 signal, but that signal remains below method-level geometry success. "
        "A follow-up special-query cue-weight repair using v112 w_special_query_mild was also tested and made the L2 candidate negative, with a shuffled control exceeding it."
    )
    summary = {
        "schema": "acl2_v115tf_final_decision_summary_v4",
        "final_taxonomy": final_taxonomy,
        "decision_reason": decision_reason,
        "stage0_complete": stage0.get("stage0_complete"),
        "stage1_status": stage1.get("stage1_status", ""),
        "stage2_status": stage2.get("stage2_status", ""),
        "hs_hg_full_pilot_rows": rows[:2],
        "hs_non_value_scaling_pilot_rows": rows,
        "hs_hg_any_geometry_pass": hg1_pass or hg2_pass,
        "hs_gq_config_specific_geometry_pass": gq1_config_pass,
        "hs_any_geometry_pass": any_geometry_pass,
        "hs_gq_smoke_gate_summary": read_json(STAGE1 / "hs_gq3_smoke_gate_summary.json"),
        "hs_gq_decision_summary": gq_decision,
        "hs_gq_full_status": gq_status,
        "lingbot_runtime_status": {
            "query_full_taxonomy": lingbot_full.get("taxonomy", ""),
            "query_full_metric_complete": bool(lingbot_full.get("metric_complete")),
            "query_full_all_action_fidelity": bool(lingbot_full.get("all_action_fidelity")),
            "positive_safety_gate_pass_count": lingbot_positive_safety_count,
            "plan_geometry_gate_pass_count": lingbot_plan_gate_pass_count,
            "l2_candidate_median_full_rel": lingbot_l2_candidate.get("median_full_rel", ""),
            "l2_candidate_rolling_p90_median_rel": lingbot_l2_candidate.get("rolling_p90_median_rel", ""),
            "l2_candidate_positive_safety_gate_pass": lingbot_l2_candidate.get("positive_safety_gate_pass", ""),
            "l2_candidate_plan_geometry_gate_pass": lingbot_l2_candidate.get("pilot_geometry_gate_pass", ""),
            "l2_control_taxonomy": lingbot_control.get("taxonomy", ""),
            "l2_controls_complete": lingbot_l2_controls_complete,
            "l2_control_matches_or_exceeds_candidate": bool(lingbot_control.get("control_matches_or_exceeds_candidate")),
            "l2_best_control_median_full_rel": lingbot_control.get("best_control_median_full_rel"),
            "l2_best_control_rolling_p90_median_rel": lingbot_control.get("best_control_rolling_p90_median_rel"),
            "l2_special_weight_repair_taxonomy": lingbot_special_repair.get("taxonomy", ""),
            "l2_special_weight_repair_complete": bool(
                lingbot_special_repair.get("metric_complete")
                and lingbot_special_repair.get("all_action_fidelity")
            ),
            "l2_special_weight_repair_candidate_median_full_rel": lingbot_special_candidate.get("median_full_rel", ""),
            "l2_special_weight_repair_candidate_rolling_p90_median_rel": lingbot_special_candidate.get("rolling_p90_median_rel", ""),
            "l2_special_weight_repair_control_matches_or_exceeds": bool(lingbot_special_repair.get("control_matches_or_exceeds_candidate")),
        },
        "lingbot_query_full_summary": lingbot_full,
        "lingbot_l2_control_summary": lingbot_control,
        "lingbot_l2_special_weight_repair_summary": lingbot_special_repair,
        "controls_required": False,
        "controls_required_reason": "No HS or LingBot candidate passed the plan-level 5% geometry pilot gate. The L2 matched-control and special-weight repair controls were run as extra fail-forward evidence, not as semantic-success promotion.",
        "minimum_completion": {
            "stage0_complete": bool(stage0.get("stage0_complete")),
            "stage1_complete": stage1_complete,
            "stage1_strict_pose_noop_parity_pass": bool(stage1.get("strict_pose_noop_parity_pass")),
            "stage1_hook_level_noop_parity_pass": bool(stage1.get("hook_level_noop_parity_pass")),
            "stage2_complete": stage2_complete,
            "horizonstream_non_value_scaling_action_entered_00_02_pilot": True,
            "lingbot_non_b1_h1_hook_audit_done": True,
            "lingbot_non_b1_h1_runtime_pilot_attempted": bool(lingbot_full.get("metric_complete")),
            "lingbot_non_b1_h1_plan_geometry_pass": lingbot_plan_gate_pass_count > 0,
            "lingbot_l2_matched_controls_complete": lingbot_l2_controls_complete,
            "lingbot_l2_special_weight_repair_complete": bool(lingbot_special_repair.get("metric_complete") and lingbot_special_repair.get("all_action_fidelity")),
        },
    }
    write_json(DIAG / "stage_hs_final_decision_summary.json", summary)
    write_csv(DIAG / "stage_hs_final_decision_rows.csv", rows)
    lines = [
        "# ACL2 v115TF Final Decision Summary",
        "",
        f"Final taxonomy: `{final_taxonomy}`",
        "",
        decision_reason,
        "",
        "| candidate | median full ATE | rolling p90 | max harm | segment scale | pilot pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt_pct(row["median_full_ATE_rel_improvement"]),
                    fmt_pct(row["median_rolling_p90_rel_improvement"]),
                    fmt_pct(row["max_full_ATE_harm_rel"]),
                    fmt_pct(row["median_segment_scale_rel_improvement"]),
                    str(row["pilot_geometry_gate_pass"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "LingBot runtime addendum:",
            "",
            f"- Query/context full pilot taxonomy: `{lingbot_full.get('taxonomy', '')}`; plan 5% gate pass count `{lingbot_plan_gate_pass_count}`, positive-safety count `{lingbot_positive_safety_count}`.",
            f"- L2 special candidate median full ATE: {fmt_pct(lingbot_l2_candidate.get('median_full_rel'))}; rolling p90: {fmt_pct(lingbot_l2_candidate.get('rolling_p90_median_rel'))}; plan gate `{lingbot_l2_candidate.get('pilot_geometry_gate_pass', '')}`.",
            f"- L2 matched-control taxonomy: `{lingbot_control.get('taxonomy', '')}`; best control median full ATE {fmt_pct(lingbot_control.get('best_control_median_full_rel'))}, best rolling {fmt_pct(lingbot_control.get('best_control_rolling_p90_median_rel'))}.",
            f"- L2 special-weight repair taxonomy: `{lingbot_special_repair.get('taxonomy', '')}`; candidate median full ATE {fmt_pct(lingbot_special_candidate.get('median_full_rel'))}, rolling {fmt_pct(lingbot_special_candidate.get('rolling_p90_median_rel'))}.",
            "",
            "Stage completion:",
            "",
            f"- Stage0 complete: {bool(stage0.get('stage0_complete'))}.",
            f"- Stage1 complete: {stage1_complete}; strict deterministic smoke parity max diff is recorded in `stage1_hook_audit/hs_noop_trace_parity_summary.json`.",
            f"- Stage2 complete: {stage2_complete}.",
            "",
            "Key blockers:",
            "",
        "- HS-LA attention-logit path is blocked by fused SDPA hiding attention probabilities/logits.",
        "- HS-HG1/HS-HG2 full pilots did not pass the 5% median geometry gate.",
        "- HS-GQ3/GQ4/GQ1 default full promotion OOMed on 00/02 after in-place, no-trace/no-audit, and selected-layer repairs.",
        "- HS-GQ1 layer23 chunk_block_num=1 completed 00/02 full metrics, but it is a reduced-memory config-specific pilot and did not pass the geometry gate.",
        "- LingBot A2/L2 query/context-specific runtime hooks now pass smoke/full action fidelity, but no policy reaches the plan-level 5% geometry gate.",
        "- LingBot L2 matched controls do not explain the weak positive L2 signal, but the signal remains below method-level geometry success and still lacks the full semantic-causality suite / 00/01/02/05 promotion.",
        "- LingBot L2 special-query cue-weight repair using `w_special_query_mild` worsened the candidate and was exceeded by a shuffled control.",
        "",
    ]
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "V115_FINAL_DECISION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"final_taxonomy": final_taxonomy, "hs_hg_any_geometry_pass": hg1_pass or hg2_pass, "lingbot_plan_gate_pass_count": lingbot_plan_gate_pass_count}, sort_keys=True))


if __name__ == "__main__":
    main()
