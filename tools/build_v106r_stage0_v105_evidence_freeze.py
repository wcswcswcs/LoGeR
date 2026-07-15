#!/usr/bin/env python3
"""Build ACL2 v106R Stage0 v105 evidence freeze artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106R = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control"
STAGE0 = V106R / "stage0_v105_evidence_freeze"


ARTIFACTS = [
    ("lingbot_full_kitti_baseline_metrics", V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"),
    ("lingbot_full_kitti_baseline_summary", V105 / "stage1_lingbot_baseline/full_sequence_metrics/stage1_full_metric_summary.json"),
    ("gca_trace_no_action_parity_summary", V105 / "stage2_gca_trace/trace_summary.json"),
    ("gca_trace_no_action_parity_rows", V105 / "stage2_gca_trace/no_action_parity_rows.csv"),
    ("stage3_oracle_sweep_rows", V105 / "stage3_lingbot_oracle/oracle_policy_sweep_metrics.csv"),
    ("stage3_oracle_summary", V105 / "stage3_lingbot_oracle/stage3_summary.json"),
    ("stage3_oracle_decision", V105 / "stage3_lingbot_oracle/stage3_decision.md"),
    ("stage4_action_metric_rows", V105 / "stage4_lingbot_action_pilot_or_blocked/action_metric_rows.csv"),
    ("stage4_action_aggregate_metrics", V105 / "stage4_lingbot_action_pilot_or_blocked/action_aggregate_metrics.csv"),
    ("stage4_action_summary", V105 / "stage4_lingbot_action_pilot_or_blocked/stage4_summary.json"),
    ("stage4_headlocal_policy_summary", V105 / "stage4_lingbot_headlocal_trace/headlocal_policy_summary.json"),
    ("stage4_headlocal_relaxed_selected_rows", V105 / "stage4_lingbot_headlocal_trace/headlocal_relaxed_selected_rows.csv"),
    ("stage4_headlocal_selected_rows", V105 / "stage4_lingbot_headlocal_trace/headlocal_selected_rows.csv"),
    ("stage5_cross_platform_summary", V105 / "stage5_cross_platform/stage5_summary.json"),
    ("stage5_platform_decision", V105 / "stage5_cross_platform/platform_decision.md"),
    ("final_decision_json", V105 / "final_decision/final_decision.json"),
]

FORBIDDEN_REPEATS = [
    "frame-level semantic_geometry_write_filter small fixes",
    "semantic_only_reject_write_filter small fixes",
    "headlocal relaxed context-only demotion threshold tuning",
    "whole-frame demotion",
    "whole-context demotion",
    "attention-mass-only action rules",
    "head-selected-count-only action rules",
    "semantic-label-only masks",
    "MoGe post-processing of LingBot depth/pose",
    "windowed overlap Sim(3) post-processing as a memory-control method",
    "LoGeR provider expansion or A5 query-soft sweeps",
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
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.exists() else path.relative_to(ROOT).as_posix()


def artifact_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in ARTIFACTS:
        rows.append(
            {
                "schema": "acl2_v106r_stage0_available_artifact_row_v1",
                "artifact_name": name,
                "path": rel(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "status": "available" if path.exists() else "missing",
            }
        )
    return rows


def selected_manifest_rows() -> list[dict[str, Any]]:
    path = V105 / "stage4_lingbot_headlocal_trace/headlocal_relaxed_selected_rows.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        selected_heads = [item.strip() for item in row.get("selected_heads", "").split(",") if item.strip()]
        for head in selected_heads or [""]:
            rows.append(
                {
                    "schema": "acl2_v106r_stage0_v105_headlocal_selected_manifest_row_v1",
                    "seq": row.get("seq", ""),
                    "sample_position": row.get("sample_position", ""),
                    "original_frame": row.get("original_frame", ""),
                    "head_id": head.removeprefix("h"),
                    "bad_label": row.get("bad_label", ""),
                    "good_label": row.get("good_label", ""),
                    "sim3_residual_m": row.get("sim3_residual_m", ""),
                    "policy": row.get("policy", ""),
                    "top_labels": row.get("top_labels", ""),
                    "source_artifact": rel(path),
                }
            )
    return rows


def action_surface_summary_rows() -> list[dict[str, Any]]:
    aggregate = V105 / "stage4_lingbot_action_pilot_or_blocked/action_aggregate_metrics.csv"
    if not aggregate.exists():
        return []
    rows: list[dict[str, Any]] = []
    forbidden_labels = {
        "semantic_geometry_write_filter",
        "semantic_only_reject_write_filter",
        "semantic_headlocal_relaxed_context_only_demote",
    }
    for row in read_csv(aggregate):
        action = row.get("action_label", "")
        rows.append(
            {
                "schema": "acl2_v106r_stage0_v105_action_surface_summary_row_v1",
                "action_label": action,
                "bad_l3_median_improvement": row.get("bad_l3_median_improvement", ""),
                "good_median_harm": row.get("good_median_harm", ""),
                "good_max_harm": row.get("good_max_harm", ""),
                "rolling_worse_fraction_gt_0p05": row.get("rolling_worse_fraction_gt_0p05", ""),
                "bad_pair_count": row.get("bad_pair_count", ""),
                "good_pair_count": row.get("good_pair_count", ""),
                "trace_fidelity_pass": row.get("trace_fidelity_pass", ""),
                "forbidden_repeat_surface": action in forbidden_labels,
                "promotion_allowed_in_v106r": False if action in forbidden_labels else "",
                "source_artifact": rel(aggregate),
            }
        )
    return rows


def known_facts(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = {row["artifact_name"]: bool(row["exists"]) for row in manifest_rows}
    baseline_rows = read_csv(V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv") if available.get("lingbot_full_kitti_baseline_metrics") else []
    trace_summary = load_json(V105 / "stage2_gca_trace/trace_summary.json") if available.get("gca_trace_no_action_parity_summary") else {}
    stage3_summary = load_json(V105 / "stage3_lingbot_oracle/stage3_summary.json") if available.get("stage3_oracle_summary") else {}
    stage4_summary = load_json(V105 / "stage4_lingbot_action_pilot_or_blocked/stage4_summary.json") if available.get("stage4_action_summary") else {}
    headlocal_summary = load_json(V105 / "stage4_lingbot_headlocal_trace/headlocal_policy_summary.json") if available.get("stage4_headlocal_policy_summary") else {}
    stage5_summary = load_json(V105 / "stage5_cross_platform/stage5_summary.json") if available.get("stage5_cross_platform_summary") else {}

    relaxed = headlocal_summary.get("relaxed_candidate", {}) or {}
    stage4_relaxed = stage4_summary.get("semantic_headlocal_relaxed_context_only_metrics", {}) or {}
    baseline_by_seq = {
        row["seq"]: {
            "frames": int(float(row["frames"])),
            "ATE_full_sim3_m": float(row["ATE_full_sim3_m"]),
            "final_error_m": float(row["final_error_m"]),
            "rolling_ATE_p90": float(row["rolling_ATE_p90"]),
            "local_window_ATE_median": float(row["local_window_ATE_median"]),
        }
        for row in baseline_rows
    }
    return {
        "schema": "acl2_v106r_stage0_v105_known_facts_v1",
        "lingbot_full_kitti_baseline_available": bool(baseline_rows),
        "lingbot_full_kitti_sequences": sorted(baseline_by_seq),
        "lingbot_full_kitti_baseline_by_seq": baseline_by_seq,
        "stage2_trace_parity_pass": bool(trace_summary.get("stage2_trace_parity_pass", False)),
        "stage2_trace_sequences": [row.get("seq") for row in trace_summary.get("parity_rows", [])],
        "stage2_trace_error_rows": trace_summary.get("trace_error_rows", ""),
        "stage3_oracle_sweep_available": available.get("stage3_oracle_sweep_rows", False),
        "stage3_lingbot_oracle_pass": stage3_summary.get("stage3_lingbot_oracle_pass", ""),
        "stage4_action_metric_rows_available": available.get("stage4_action_metric_rows", False),
        "stage4_action_pass": stage4_summary.get("stage4_action_pass", ""),
        "stage4_run_result_rows": stage4_summary.get("run_result_rows", ""),
        "stage4_run_failures": stage4_summary.get("run_failures", ""),
        "headlocal_relaxed_action_available": bool(relaxed),
        "headlocal_relaxed_selected_count": relaxed.get("selected_count", ""),
        "headlocal_relaxed_bad_l3_improvement": stage4_relaxed.get("bad_l3_median_improvement", ""),
        "headlocal_relaxed_good_median_harm": stage4_relaxed.get("good_median_harm", ""),
        "headlocal_relaxed_good_max_harm": stage4_relaxed.get("good_max_harm", ""),
        "headlocal_relaxed_bad_seq_coverage": relaxed.get("bad_seq_coverage", ""),
        "lingbot_go": stage5_summary.get("lingbot_go", False),
        "loger_frozen": True,
        "final_decision_json_available": available.get("final_decision_json", False),
        "final_decision_substitute": rel(V105 / "stage5_cross_platform/platform_decision.md") if available.get("stage5_platform_decision", False) else "",
        "no_repeat_guard_active": True,
        "v106r_stage0_pass": bool(
            baseline_rows
            and trace_summary.get("stage2_trace_parity_pass", False)
            and available.get("stage4_action_metric_rows", False)
        ),
    }


def write_forbidden_repeat_list(path: Path) -> None:
    lines = [
        "# v106R Forbidden Repeat List",
        "",
        "These routes are allowed only as explicit negative controls. They must not be promoted as v106R runtime candidates.",
        "",
    ]
    for item in FORBIDDEN_REPEATS:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Carry-forward evidence:",
            "",
            "- v105 frame-level and coarse semantic/geometric write filters did not pass action gates.",
            "- v105 headlocal relaxed context-only demotion improved bad L3 but caused severe good harm.",
            "- v106R must discover memory-operation levers before choosing action surfaces.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, facts: dict[str, Any], missing: list[str]) -> None:
    status = "PASS" if facts["v106r_stage0_pass"] else "INCOMPLETE"
    lines = [
        "# Stage0 v105 Evidence Freeze Report",
        "",
        f"- stage0_status: `{status}`",
        f"- lingbot_full_kitti_baseline_available: `{facts['lingbot_full_kitti_baseline_available']}`",
        f"- stage2_trace_parity_pass: `{facts['stage2_trace_parity_pass']}`",
        f"- stage4_action_metric_rows_available: `{facts['stage4_action_metric_rows_available']}`",
        f"- headlocal_relaxed_action_available: `{facts['headlocal_relaxed_action_available']}`",
        f"- final_decision_json_available: `{facts['final_decision_json_available']}`",
        "",
        "Missing or substituted artifacts:",
        "",
    ]
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Key carry-forward facts:",
            "",
            f"- full KITTI sequences: `{','.join(facts['lingbot_full_kitti_sequences'])}`",
            f"- headlocal relaxed bad L3 improvement: `{facts['headlocal_relaxed_bad_l3_improvement']}`",
            f"- headlocal relaxed good median harm: `{facts['headlocal_relaxed_good_median_harm']}`",
            f"- headlocal relaxed good max harm: `{facts['headlocal_relaxed_good_max_harm']}`",
            f"- lingbot_go from v105 decision: `{facts['lingbot_go']}`",
            "",
            "Conclusion:",
            "",
            "Stage0 freezes v105 as an analyzable LingBot platform with known unsafe action surfaces. "
            "It does not claim v106R success; it enables Stage1 memory operation discovery.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> dict[str, Any]:
    STAGE0.mkdir(parents=True, exist_ok=True)
    manifest_rows = artifact_manifest()
    facts = known_facts(manifest_rows)
    missing = [row["artifact_name"] for row in manifest_rows if not row["exists"]]

    write_csv(STAGE0 / "available_artifact_manifest.csv", manifest_rows)
    write_csv(STAGE0 / "v105_action_surface_summary.csv", action_surface_summary_rows())
    write_csv(STAGE0 / "v105_headlocal_selected_manifest.csv", selected_manifest_rows())
    (STAGE0 / "v105_known_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_forbidden_repeat_list(STAGE0 / "forbidden_repeat_list.md")
    write_report(STAGE0 / "stage0_freeze_report.md", facts, missing)

    # Keep a narrow copy of the v105 platform decision for reviewer convenience.
    decision = V105 / "stage5_cross_platform/platform_decision.md"
    if decision.exists():
        shutil.copy2(decision, STAGE0 / "v105_platform_decision_copy.md")

    summary = {
        "schema": "acl2_v106r_stage0_summary_v1",
        "stage0_pass": facts["v106r_stage0_pass"],
        "missing_artifacts": missing,
        "artifact_count": len(manifest_rows),
        "available_artifact_count": sum(1 for row in manifest_rows if row["exists"]),
        "outputs": {
            "known_facts": (STAGE0 / "v105_known_facts.json").relative_to(ROOT).as_posix(),
            "forbidden_repeat_list": (STAGE0 / "forbidden_repeat_list.md").relative_to(ROOT).as_posix(),
            "available_artifact_manifest": (STAGE0 / "available_artifact_manifest.csv").relative_to(ROOT).as_posix(),
            "v105_action_surface_summary": (STAGE0 / "v105_action_surface_summary.csv").relative_to(ROOT).as_posix(),
            "v105_headlocal_selected_manifest": (STAGE0 / "v105_headlocal_selected_manifest.csv").relative_to(ROOT).as_posix(),
            "report": (STAGE0 / "stage0_freeze_report.md").relative_to(ROOT).as_posix(),
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
