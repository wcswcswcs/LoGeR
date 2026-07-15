#!/usr/bin/env python3
"""Build ACL2 v107TF Stage0 evidence freeze from v105 and v106R artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106R = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control"
V107 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
STAGE0 = V107 / "stage0_evidence_freeze"

ARTIFACTS = [
    ("v105_full_kitti_metrics", V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"),
    ("v105_full_metric_summary", V105 / "stage1_lingbot_baseline/full_sequence_metrics/stage1_full_metric_summary.json"),
    ("v105_local_window_rows", V105 / "stage1_lingbot_baseline/full_sequence_metrics/local_window_rows.csv"),
    ("v105_trace_summary", V105 / "stage2_gca_trace/trace_summary.json"),
    ("v105_trace_parity_rows", V105 / "stage2_gca_trace/no_action_parity_rows.csv"),
    ("v105_stage3_oracle_summary", V105 / "stage3_lingbot_oracle/stage3_summary.json"),
    ("v105_stage3_oracle_sweep_rows", V105 / "stage3_lingbot_oracle/oracle_policy_sweep_metrics.csv"),
    ("v105_stage4_action_summary", V105 / "stage4_lingbot_action_pilot_or_blocked/stage4_summary.json"),
    ("v105_stage4_action_rows", V105 / "stage4_lingbot_action_pilot_or_blocked/action_metric_rows.csv"),
    ("v105_stage4_action_aggregate", V105 / "stage4_lingbot_action_pilot_or_blocked/action_aggregate_metrics.csv"),
    ("v105_headlocal_policy_summary", V105 / "stage4_lingbot_headlocal_trace/headlocal_policy_summary.json"),
    ("v105_headlocal_relaxed_selected_rows", V105 / "stage4_lingbot_headlocal_trace/headlocal_relaxed_selected_rows.csv"),
    ("v105_platform_decision", V105 / "stage5_cross_platform/platform_decision.md"),
    ("v106r_stage0_known_facts", V106R / "stage0_v105_evidence_freeze/v105_known_facts.json"),
    ("v106r_stage0_forbidden_repeat", V106R / "stage0_v105_evidence_freeze/forbidden_repeat_list.md"),
    ("v106r_initial_stage1_summary", V106R / "stage1_memory_operation_map/stage1_summary.json"),
    ("v106r_targeted_summary", V106R / "stage1_memory_operation_map/targeted_trace/targeted_trace_summary.json"),
    ("v106r_target_manifest", V106R / "stage1_memory_operation_map/targeted_trace/target_manifest.csv"),
    ("v106r_stage1_no_memory_lever", V106R / "stage1_memory_operation_map/stage1_no_memory_lever_found.md"),
    ("v106r_non_readout_observability", V106R / "stage1_memory_operation_map/non_readout_operation_observability_report.md"),
    ("v106r_final_decision_json", V106R / "final_decision/final_decision.json"),
    ("v106r_final_decision_md", V106R / "final_decision/final_decision.md"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def artifact_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in ARTIFACTS:
        rows.append(
            {
                "schema": "acl2_v107tf_stage0_artifact_manifest_row_v1",
                "artifact_name": name,
                "path": rel(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "status": "available" if path.exists() else "missing",
            }
        )
    return rows


def full_kitti_baseline() -> dict[str, Any]:
    path = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
    if not path.is_file():
        return {}
    out: dict[str, Any] = {}
    for row in read_csv(path):
        seq = row["seq"]
        out[seq] = {
            "frames": int(float(row["frames"])),
            "ATE_full_sim3_m": fnum(row, "ATE_full_sim3_m"),
            "final_error_m": fnum(row, "final_error_m"),
            "rolling_ATE_p90": fnum(row, "rolling_ATE_p90"),
            "local_window_ATE_median": fnum(row, "local_window_ATE_median"),
        }
    return out


def allowed_negative_controls() -> list[dict[str, Any]]:
    return [
        {
            "schema": "acl2_v107tf_allowed_negative_control_row_v1",
            "control_name": "v105_headlocal_relaxed_context_only_demote",
            "allowed_use": "negative_control_only",
            "runtime_promotion_allowed": False,
            "reason": "v105 showed bad L3 movement but severe good harm.",
            "source": rel(V105 / "stage4_lingbot_action_pilot_or_blocked/action_aggregate_metrics.csv"),
        },
        {
            "schema": "acl2_v107tf_allowed_negative_control_row_v1",
            "control_name": "v106r_readout_attention_mass_levers",
            "allowed_use": "negative_control_only",
            "runtime_promotion_allowed": False,
            "reason": "v106R readout-only levers did not pass discovery and are not cache-operation levers.",
            "source": rel(V106R / "stage1_memory_operation_map/targeted_trace/targeted_memory_lever_rank.csv"),
        },
        {
            "schema": "acl2_v107tf_allowed_negative_control_row_v1",
            "control_name": "same_count_random_or_role_rotation",
            "allowed_use": "control_margin_only",
            "runtime_promotion_allowed": False,
            "reason": "Control baselines for Stage3/Stage5; never method claims.",
            "source": "plan_section_8_10",
        },
    ]


def known_facts(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = {row["artifact_name"]: bool(row["exists"]) for row in manifest_rows}
    v105_trace = load_json(V105 / "stage2_gca_trace/trace_summary.json")
    v105_stage3 = load_json(V105 / "stage3_lingbot_oracle/stage3_summary.json")
    v105_stage4 = load_json(V105 / "stage4_lingbot_action_pilot_or_blocked/stage4_summary.json")
    v105_headlocal = load_json(V105 / "stage4_lingbot_headlocal_trace/headlocal_policy_summary.json")
    v106r_stage1 = load_json(V106R / "stage1_memory_operation_map/stage1_summary.json")
    v106r_targeted = load_json(V106R / "stage1_memory_operation_map/targeted_trace/targeted_trace_summary.json")
    v106r_final = load_json(V106R / "final_decision/final_decision.json")
    relaxed = (v105_headlocal.get("relaxed_candidate") or {}) if isinstance(v105_headlocal, dict) else {}
    relaxed_metrics = (v105_stage4.get("semantic_headlocal_relaxed_context_only_metrics") or {}) if isinstance(v105_stage4, dict) else {}
    baseline = full_kitti_baseline()
    facts = {
        "schema": "acl2_v107tf_stage0_v105_v106r_known_facts_v1",
        "v105_full_kitti_baseline_available": bool(baseline),
        "v105_full_kitti_baseline_by_seq": baseline,
        "v105_full_kitti_sequences": sorted(baseline),
        "v105_trace_parity_pass": bool(v105_trace.get("stage2_trace_parity_pass", False)),
        "v105_trace_error_rows": v105_trace.get("trace_error_rows", ""),
        "v105_stage3_oracle_pass": v105_stage3.get("stage3_lingbot_oracle_pass", ""),
        "v105_stage4_action_pass": v105_stage4.get("stage4_action_pass", ""),
        "v105_stage4_action_rows_available": available.get("v105_stage4_action_rows", False),
        "v105_headlocal_relaxed_selected_count": relaxed.get("selected_count", ""),
        "v105_headlocal_relaxed_bad_seq_coverage": relaxed.get("bad_seq_coverage", ""),
        "v105_headlocal_relaxed_bad_l3_improvement": relaxed_metrics.get("bad_l3_median_improvement", ""),
        "v105_headlocal_relaxed_good_median_harm": relaxed_metrics.get("good_median_harm", ""),
        "v105_headlocal_relaxed_good_max_harm": relaxed_metrics.get("good_max_harm", ""),
        "v106r_taxonomy": v106r_final.get("taxonomy", ""),
        "v106r_stage1_initial_discovery_pass": v106r_final.get("stage1_initial_discovery_pass", ""),
        "v106r_stage1_targeted_trace_parity_pass": v106r_final.get("stage1_targeted_trace_parity_pass", ""),
        "v106r_stage1_targeted_discovery_pass": v106r_final.get("stage1_targeted_discovery_pass", ""),
        "v106r_new_full_kitti_ate_available": v106r_final.get("new_full_kitti_ate_available", ""),
        "v106r_initial_operation_types_present": v106r_stage1.get("operation_types_present", []),
        "v106r_initial_missing_operation_types": v106r_stage1.get("missing_operation_types", []),
        "v106r_targeted_operation_types_present": v106r_targeted.get("operation_types_present", []),
        "v106r_targeted_missing_operation_types": v106r_targeted.get("missing_operation_types", []),
        "v106r_targeted_max_abs_corr_L3": v106r_targeted.get("max_abs_corr_L3", ""),
        "v106r_targeted_max_abs_corr_rolling": v106r_targeted.get("max_abs_corr_rolling", ""),
        "v106r_targeted_trace_error_rows": v106r_targeted.get("trace_error_rows", ""),
        "v106r_target_count": len(read_csv(V106R / "stage1_memory_operation_map/targeted_trace/target_manifest.csv"))
        if (V106R / "stage1_memory_operation_map/targeted_trace/target_manifest.csv").is_file()
        else 0,
        "required_artifacts_missing": [row["artifact_name"] for row in manifest_rows if not row["exists"]],
    }
    facts["stage0_pass"] = bool(
        facts["v105_full_kitti_baseline_available"]
        and facts["v105_trace_parity_pass"]
        and facts["v105_stage4_action_rows_available"]
        and facts["v106r_taxonomy"]
        and facts["v106r_targeted_operation_types_present"] == ["readout"]
        and set(facts["v106r_targeted_missing_operation_types"]) >= {"update", "retention", "initialization", "budget_eviction"}
    )
    return facts


def write_forbidden_repeat(path: Path) -> None:
    lines = [
        "# v107TF Forbidden Repeat List",
        "",
        "These paths are allowed only as explicit negative controls or audit references.",
        "",
        "- Directly promoting v105 `semantic_headlocal_relaxed_context_only_demote`.",
        "- Continuing attention-mass-only / head-selected-count-only / semantic-label-only action rules.",
        "- Re-running v106R readout lever threshold sweeps as if they were cache-operation discovery.",
        "- Claiming method effect from debug32/debug96 or trace movement alone.",
        "- Using MoGe-2 or LingBot-Depth as pose/depth post-processing.",
        "",
        "Carry-forward rationale:",
        "",
        "- v105 headlocal relaxed action had causal movement but severe good harm.",
        "- v106R targeted trace parity was healthy, but observed operation type was readout-only.",
        "- v107TF must observe cache operations before action selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing_report(path: Path, missing: list[str]) -> None:
    if not missing:
        return
    lines = [
        "# Stage0 Missing Artifacts Report",
        "",
        "Stage0 searched v105/v106R artifacts and found the following missing items:",
        "",
    ]
    lines.extend(f"- {item}" for item in missing)
    lines.extend(
        [
            "",
            "Per plan, missing artifacts are documented rather than silently skipped.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, facts: dict[str, Any]) -> None:
    lines = [
        "# Stage0 Evidence Freeze Report",
        "",
        f"- stage0_pass: `{facts['stage0_pass']}`",
        f"- v105_full_kitti_baseline_available: `{facts['v105_full_kitti_baseline_available']}`",
        f"- v105_trace_parity_pass: `{facts['v105_trace_parity_pass']}`",
        f"- v105_stage4_action_pass: `{facts['v105_stage4_action_pass']}`",
        f"- v106r_taxonomy: `{facts['v106r_taxonomy']}`",
        f"- v106r_targeted_operation_types_present: `{','.join(facts['v106r_targeted_operation_types_present'])}`",
        f"- v106r_targeted_missing_operation_types: `{','.join(facts['v106r_targeted_missing_operation_types'])}`",
        "",
        "Interpretation:",
        "",
        "Stage0 freezes v105 as a runnable LingBot platform and v106R as a readout-only No-Go. "
        "This enables v107TF cache-operation instrumentation but does not allow runtime action.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> dict[str, Any]:
    STAGE0.mkdir(parents=True, exist_ok=True)
    manifest = artifact_manifest()
    facts = known_facts(manifest)
    missing = facts["required_artifacts_missing"]
    write_csv(STAGE0 / "available_artifact_manifest.csv", manifest)
    (STAGE0 / "v105_v106r_known_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_forbidden_repeat(STAGE0 / "forbidden_repeat_list.md")
    write_csv(STAGE0 / "allowed_negative_controls.csv", allowed_negative_controls())
    write_missing_report(STAGE0 / "stage0_missing_artifacts_report.md", missing)
    write_report(STAGE0 / "stage0_freeze_report.md", facts)

    for src, dst_name in [
        (V105 / "stage5_cross_platform/platform_decision.md", "v105_platform_decision_copy.md"),
        (V106R / "final_decision/final_decision.md", "v106r_final_decision_copy.md"),
        (V106R / "stage1_memory_operation_map/non_readout_operation_observability_report.md", "v106r_non_readout_observability_copy.md"),
    ]:
        if src.is_file():
            shutil.copy2(src, STAGE0 / dst_name)

    summary = {
        "schema": "acl2_v107tf_stage0_summary_v1",
        "stage0_pass": facts["stage0_pass"],
        "artifact_count": len(manifest),
        "available_artifact_count": sum(1 for row in manifest if row["exists"]),
        "missing_artifacts": missing,
        "outputs": {
            "known_facts": rel(STAGE0 / "v105_v106r_known_facts.json"),
            "forbidden_repeat_list": rel(STAGE0 / "forbidden_repeat_list.md"),
            "allowed_negative_controls": rel(STAGE0 / "allowed_negative_controls.csv"),
            "available_artifact_manifest": rel(STAGE0 / "available_artifact_manifest.csv"),
            "report": rel(STAGE0 / "stage0_freeze_report.md"),
        },
    }
    (STAGE0 / "stage0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
