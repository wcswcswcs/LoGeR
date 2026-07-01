#!/usr/bin/env python3
"""Build ACL2 v105-TF cross-platform decision artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE1 = RESULT_ROOT / "stage1_lingbot_baseline"
STAGE2 = RESULT_ROOT / "stage2_gca_trace"
STAGE3 = RESULT_ROOT / "stage3_lingbot_oracle"
STAGE4 = RESULT_ROOT / "stage4_lingbot_action_pilot_or_blocked"
STAGE5 = RESULT_ROOT / "stage5_cross_platform"
LOGER_STAGE5 = RESULT_ROOT / "stage5_loger_witness_disambiguation"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def bool_text(value: Any) -> str:
    return str(bool(value)).lower()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_indices(raw: str) -> set[int]:
    return {int(x) for x in str(raw or "").split(";") if x != ""}


def build_admission_false_positive_note(stage4_summary: dict[str, Any]) -> None:
    sem = stage4_summary.get("semantic_geometry_metrics", {})
    action_label = str(sem.get("action_label", ""))
    forced_by_seq: dict[str, set[int]] = {}
    if action_label:
        for row in read_csv(STAGE4 / "action_config_rows.csv"):
            if row.get("action_label") != action_label:
                continue
            seq = f"{int(float(row.get('seq', 0) or 0)):02d}"
            forced_by_seq.setdefault(seq, set()).update(parse_indices(row.get("force_non_keyframe_indices", "")))

    good_rows: list[dict[str, str]] = []
    for row in read_csv(STAGE3 / "frame_semantic_geometry_rows.csv"):
        seq = f"{int(float(row.get('seq', 0) or 0)):02d}"
        sample_position = int(float(row.get("sample_position", 0) or 0))
        if sample_position in forced_by_seq.get(seq, set()) and parse_bool(row.get("good_label")):
            out = dict(row)
            out["seq"] = seq
            good_rows.append(out)

    lines = [
        "# Stage4 Admission False Positive Panels",
        "",
        f"The current best semantic action `{action_label}` admitted safe-good frames into its forced action set.",
        "The runtime action changed the intended KV write path, but good harm was not controlled.",
        "",
        f"- action_label: `{action_label}`",
        f"- semantic_geometry_good_median_harm: `{sem.get('good_median_harm')}`",
        f"- semantic_geometry_good_max_harm: `{sem.get('good_max_harm')}`",
        f"- selected_good_rows: `{len(good_rows)}`",
        "",
        "| seq | sample_position | original_frame | sim3_residual_m | top_labels |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in good_rows:
        lines.append(
            "| {seq} | {sample_position} | {original_frame} | {sim3_residual_m} | {top_labels} |".format(
                seq=row.get("seq"),
                sample_position=row.get("sample_position"),
                original_frame=row.get("original_frame"),
                sim3_residual_m=row.get("sim3_residual_m"),
                top_labels=str(row.get("top_labels", "")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- The action captured bad frames, but its runtime effect still harmed safe-good frames.",
            "- This action family should not be reported as a pass unless good harm is controlled.",
            "- A future action would need finer token/path routing than frame-level KV write suppression.",
            "",
        ]
    )
    (STAGE4 / "admission_false_positive_panels.md").write_text("\n".join(lines), encoding="utf-8")


def build() -> dict[str, Any]:
    STAGE5.mkdir(parents=True, exist_ok=True)
    LOGER_STAGE5.mkdir(parents=True, exist_ok=True)

    stage1 = load_json(STAGE1 / "loger_comparison_summary.json")
    stage1_full_path = STAGE1 / "full_sequence_metrics/stage1_full_metric_summary.json"
    stage1_full = load_json(stage1_full_path) if stage1_full_path.is_file() else {}
    stage2 = load_json(STAGE2 / "trace_summary.json")
    stage3 = load_json(STAGE3 / "stage3_combined_summary.json")
    stage4 = load_json(STAGE4 / "stage4_summary.json")
    build_admission_false_positive_note(stage4)

    full_stage1_complete = bool(stage1_full.get("full_stage1_stream_default_complete"))
    full_rows = stage1_full.get("metric_rows", [])
    full_ate_by_seq = {
        row.get("seq"): row.get("ATE_full_sim3_m")
        for row in full_rows
        if isinstance(row, dict)
    }

    lingbot_go = (
        full_stage1_complete
        and bool(stage2.get("stage2_trace_parity_pass"))
        and bool(stage3.get("stage3_lingbot_oracle_pass"))
        and bool(stage4.get("stage4_action_pass"))
    )
    lingbot_nogo = not lingbot_go

    strict_positive_ids = stage1.get("v104_exact_witness_strict_positive_case_ids", [])
    safe_good_ids = stage1.get("v104_exact_witness_safe_good_case_ids", [])
    loger_witness_pass = (
        len(strict_positive_ids) >= 2
        and len(safe_good_ids) == 0
        and bool(stage1.get("v104_runtime_action_allowed"))
    )

    comparison_rows = [
        {
            "schema": "acl2_v105tf_stage5_model_comparison_row_v1",
            "platform": "LingBot-Map",
            "baseline_can_run": True,
            "kitti_drift_available": True,
            "l0_l4_metrics_align": "full_stage1_00_01_02_05_stream_default_complete",
            "full_stage1_stream_default_complete": full_stage1_complete,
            "full_ate_seq00": full_ate_by_seq.get("00", ""),
            "full_ate_seq01": full_ate_by_seq.get("01", ""),
            "full_ate_seq02": full_ate_by_seq.get("02", ""),
            "full_ate_seq05": full_ate_by_seq.get("05", ""),
            "trace_only_parity_pass": bool(stage2.get("stage2_trace_parity_pass")),
            "semantic_geometry_oracle_pass": bool(stage3.get("stage3_lingbot_oracle_pass")),
            "context_witness_action_surface_pass": bool(stage4.get("stage4_action_pass")),
            "good_harm_controlled": False,
            "engineering_cost": "moderate_high_for_true_token_path_routing",
            "scientific_clarity": "good_trace_oracle_but_action_counterevidence",
            "decision": "No-Go for main-platform switch; keep as architecture reference",
        },
        {
            "schema": "acl2_v105tf_stage5_model_comparison_row_v1",
            "platform": "LoGeR",
            "baseline_can_run": "already_available_from_prior_acl2_runs",
            "kitti_drift_available": "available_in_prior_failure_atlas",
            "l0_l4_metrics_align": "not_aligned_to_lingbot_debug96_in_v105",
            "trace_only_parity_pass": bool(stage1.get("v104_no_action_parity_pass")),
            "semantic_geometry_oracle_pass": bool(stage1.get("v104_stage2_semantic_diagnostic_pass")),
            "context_witness_action_surface_pass": loger_witness_pass,
            "good_harm_controlled": False,
            "engineering_cost": "lower_for_existing_pipeline_but_provider_join_blocked",
            "scientific_clarity": "witness_exists_but_hard_negative_collision_unresolved",
            "decision": "Continue only with witness role disambiguation, not provider expansion",
        },
    ]
    write_csv(STAGE5 / "model_comparison_table.csv", comparison_rows)

    (LOGER_STAGE5 / "hard_negative_witness_collision.md").write_text(
        "# LoGeR Hard Negative Witness Collision\n\n"
        f"- strict_positive_exact_witness_cases: `{strict_positive_ids}`\n"
        f"- safe_good_exact_witness_cases: `{safe_good_ids}`\n"
        "- Interpretation: existing exact witness still collides with hard safe-good cases, so it cannot be "
        "promoted to `SCALE_GAUGE_EVIDENCE` without a context/local-only explanation.\n",
        encoding="utf-8",
    )
    (LOGER_STAGE5 / "loger_witness_disambiguation_failure.md").write_text(
        "# LoGeR Witness Disambiguation Failure\n\n"
        "LoGeR witness disambiguation does not pass in v105.\n\n"
        f"- selected_strict_or_exploratory_positives: `{len(strict_positive_ids)}`\n"
        f"- hard_negative_scale_gauge_evidence_hits: `{len(safe_good_ids)}`\n"
        f"- runtime_action_allowed_from_v104: `{bool_text(stage1.get('v104_runtime_action_allowed'))}`\n"
        "- failure_reason: safe-good hard negatives and positives remain indistinguishable under the carried "
        "exact-witness evidence; available v105 LoGeR artifact is case-level, not full L0-L4 trajectory-aligned.\n",
        encoding="utf-8",
    )

    decision_lines = [
        "# Platform Decision",
        "",
        "v105 goal is not achieved.",
        "",
        "LingBot-Map result:",
        f"- full_stage1_stream_default_complete: `{bool_text(full_stage1_complete)}`",
        f"- full_ATE_seq00: `{full_ate_by_seq.get('00', '')}`",
        f"- full_ATE_seq01: `{full_ate_by_seq.get('01', '')}`",
        f"- full_ATE_seq02: `{full_ate_by_seq.get('02', '')}`",
        f"- full_ATE_seq05: `{full_ate_by_seq.get('05', '')}`",
        f"- trace_only_parity_pass: `{bool_text(stage2.get('stage2_trace_parity_pass'))}`",
        f"- stage3_oracle_pass: `{bool_text(stage3.get('stage3_lingbot_oracle_pass'))}`",
        f"- stage4_action_pass: `{bool_text(stage4.get('stage4_action_pass'))}`",
        f"- semantic_geometry_bad_l3_median_improvement: `{stage4.get('semantic_geometry_metrics', {}).get('bad_l3_median_improvement')}`",
        f"- semantic_geometry_good_median_harm: `{stage4.get('semantic_geometry_metrics', {}).get('good_median_harm')}`",
        "",
        "LoGeR result:",
        f"- exact_witness_strict_positive_cases: `{strict_positive_ids}`",
        f"- exact_witness_safe_good_cases: `{safe_good_ids}`",
        f"- witness_disambiguation_pass: `{bool_text(loger_witness_pass)}`",
        "",
        "Decision:",
        "- Do not switch main method development to LingBot in this round.",
        "- Keep LingBot as architecture reference for anchor/local/trajectory/context role taxonomy.",
        "- Continue LoGeR only through witness role disambiguation and hard-negative separation; do not do provider expansion.",
        "",
    ]
    (STAGE5 / "platform_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    summary = {
        "schema": "acl2_v105tf_stage5_cross_platform_decision_summary_v1",
        "v105_goal_achieved": False,
        "lingbot_full_stage1_stream_default_complete": full_stage1_complete,
        "lingbot_full_ate_by_seq": full_ate_by_seq,
        "lingbot_go": lingbot_go,
        "lingbot_nogo": lingbot_nogo,
        "loger_witness_disambiguation_pass": loger_witness_pass,
        "stage4_action_pass": bool(stage4.get("stage4_action_pass")),
        "artifacts": {
            "model_comparison_table": str(STAGE5 / "model_comparison_table.csv"),
            "platform_decision": str(STAGE5 / "platform_decision.md"),
            "admission_false_positive_panels": str(STAGE4 / "admission_false_positive_panels.md"),
            "loger_witness_failure": str(LOGER_STAGE5 / "loger_witness_disambiguation_failure.md"),
            "hard_negative_collision": str(LOGER_STAGE5 / "hard_negative_witness_collision.md"),
        },
    }
    (STAGE5 / "stage5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
