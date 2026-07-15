#!/usr/bin/env python3
"""Finalize ACL2 v106R when Stage1 finds no promotable memory lever."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V106R = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control"
STAGE0 = V106R / "stage0_v105_evidence_freeze"
STAGE1 = V106R / "stage1_memory_operation_map"
TARGETED = STAGE1 / "targeted_trace"
STAGE2 = V106R / "stage2_semantic_increment_map"
STAGE3 = V106R / "stage3_role_disambiguation"
STAGE4 = V106R / "stage4_action_surface_screen"
STAGE5 = V106R / "stage5_runtime_pilot_or_blocked"
STAGE6 = V106R / "stage6_full_validation_or_blocked"
FINAL = V106R / "final_decision"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def bullet_lines(rows: list[str]) -> list[str]:
    return [f"- {row}" for row in rows]


def baseline_lines(facts: dict[str, Any]) -> list[str]:
    lines = [
        "| seq | frames | full_ATE_sim3_m | final_error_m | rolling_ATE_p90 | local_window_ATE_median |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seq, row in facts["lingbot_full_kitti_baseline_by_seq"].items():
        lines.append(
            f"| {seq} | {row['frames']} | {row['ATE_full_sim3_m']} | "
            f"{row['final_error_m']} | {row['rolling_ATE_p90']} | {row['local_window_ATE_median']} |"
        )
    return lines


def target_lines(targets: list[dict[str, str]]) -> list[str]:
    lines = [
        "| target_id | kind | seq | frames | handoff_transfer_penalty | adjacent_log_scale_jump | local_sim3_ate_rmse_m |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in targets:
        frames = f"{row['target_frame_start']}-{row['target_frame_end']}"
        lines.append(
            f"| {row['target_id']} | {row['target_kind']} | {row['seq']} | {frames} | "
            f"{row['handoff_transfer_penalty']} | {row['adjacent_log_scale_jump']} | "
            f"{row['local_sim3_ate_rmse_m']} |"
        )
    return lines


def top_lever_lines(rows: list[dict[str, str]], n: int = 8) -> list[str]:
    lines = [
        "| lever_id | abs_corr_L3 | bad_recall | good_FPR | same_count_random_margin | target_coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:n]:
        lines.append(
            f"| {row['lever_id']} | {row['abs_corr_L3']} | {row['bad_recall']} | "
            f"{row['good_FPR']} | {row['same_count_random_margin']} | {row['target_coverage']} |"
        )
    return lines


def main() -> None:
    facts = read_json(STAGE0 / "v105_known_facts.json")
    stage1 = read_json(STAGE1 / "stage1_summary.json")
    targeted = read_json(TARGETED / "targeted_trace_summary.json")
    targets = read_csv(TARGETED / "target_manifest.csv")
    rank_rows = read_csv(TARGETED / "targeted_memory_lever_rank.csv")

    stage1_pass = bool(stage1.get("stage1_discovery_pass"))
    targeted_pass = bool(targeted.get("targeted_stage1_discovery_pass"))
    parity_pass = bool(targeted.get("targeted_trace_parity_pass"))
    taxonomy = "NO_GO_NO_MEMORY_LEVER_FOUND"
    action_allowed = bool(stage1_pass and targeted_pass and parity_pass)

    no_go_lines = [
        "# Stage1 No Memory Lever Found",
        "",
        "Decision: `NO_GO_NO_MEMORY_LEVER_FOUND`.",
        "",
        "This is an action-blocking Stage1 result, not a runtime method result. "
        "Per the v106R plan, action stages must stop when Stage1 has no lever with a stable L3 or rolling association.",
        "",
        "Initial v105-trace map:",
        "",
        *bullet_lines(
            [
                f"memory_operation_rows: `{stage1['memory_operation_rows']}`",
                f"lever_count: `{stage1['lever_count']}`",
                f"levers_sequence_coverage_ge2: `{stage1['levers_sequence_coverage_ge2']}`",
                f"max_abs_corr_L3: `{stage1['max_abs_corr_L3']}`",
                f"max_abs_corr_rolling: `{stage1['max_abs_corr_rolling']}`",
                f"max_same_count_random_margin: `{stage1['max_same_count_random_margin']}`",
                f"stage1_discovery_pass: `{stage1['stage1_discovery_pass']}`",
                f"trace_sequences: `{','.join(stage1['trace_sequences'])}`",
            ]
        ),
        "",
        "Corrected targeted trace map:",
        "",
        *bullet_lines(
            [
                f"targeted_trace_parity_pass: `{targeted['targeted_trace_parity_pass']}`",
                f"failed_parity_rows: `{targeted['failed_parity_rows']}`",
                f"trace_error_rows: `{targeted['trace_error_rows']}`",
                f"context_role_resolved_ratio_min: `{targeted['context_role_resolved_ratio_min']}`",
                f"gca_context_topk_rows: `{targeted['gca_context_topk_rows']}`",
                f"kv_cache_provenance_rows: `{targeted['kv_cache_provenance_rows']}`",
                f"targeted_memory_operation_rows: `{targeted['targeted_memory_operation_rows']}`",
                f"targeted_lever_count: `{targeted['targeted_lever_count']}`",
                f"levers_sequence_coverage_ge2: `{targeted['levers_sequence_coverage_ge2']}`",
                f"max_abs_corr_L3: `{targeted['max_abs_corr_L3']}`",
                f"max_abs_corr_rolling: `{targeted['max_abs_corr_rolling']}`",
                f"max_same_count_random_margin: `{targeted['max_same_count_random_margin']}`",
                f"targeted_stage1_discovery_pass: `{targeted['targeted_stage1_discovery_pass']}`",
                f"operation_types_present: `{','.join(targeted.get('operation_types_present', []))}`",
                f"missing_operation_types: `{','.join(targeted.get('missing_operation_types', []))}`",
            ]
        ),
        "",
        "Target windows used after safe-L3 selection repair:",
        "",
        *target_lines(targets),
        "",
        "Top targeted levers:",
        "",
        *top_lever_lines(rank_rows),
        "",
        "Interpretation:",
        "",
        "- The trace/parity path is healthy, so this is not a trace blocker.",
        "- The strongest targeted lever has `abs_corr_L3` far below the Stage1 gate `0.45`.",
        "- Several high-recall rows also have high good FPR, so they are not safe action surfaces.",
        "- No Stage2 semantic-increment or Stage4 action run is allowed from these evidence rows.",
    ]
    write_text(STAGE1 / "stage1_no_memory_lever_found.md", "\n".join(no_go_lines) + "\n")

    observability = [
        "# Non-Readout Operation Observability Report",
        "",
        "Conclusion: current targeted trace supports GCA readout ranking, but does not expose a content-dependent semantic update/retention selector.",
        "",
        *bullet_lines(
            [
                f"operation_types_present: `{','.join(targeted.get('operation_types_present', []))}`",
                f"missing_operation_types: `{','.join(targeted.get('missing_operation_types', []))}`",
                "Current SDPA trace rows are `kv_cache_provenance` and `gca_context_topk`.",
                "`kv_cache_provenance` records cache dimensions and policy metadata, not per-evidence semantic update decisions.",
                "LingBot fixed keyframe mode with keyframe_interval=1 appends each streaming frame; cache eviction follows fixed scale-frame plus sliding-window policy.",
                "Therefore update/retention/initialization/budget-eviction families were not promoted as semantic-aware levers in v106R.",
            ]
        ),
        "",
        "Audited source paths:",
        "",
        *bullet_lines(
            [
                "third_party/lingbot-map/lingbot_map/models/gct_stream_window_v2.py",
                "third_party/lingbot-map/lingbot_map/layers/attention.py",
                "tools/build_v106r_stage1_targeted_trace_summary.py",
            ]
        ),
    ]
    write_text(STAGE1 / "non_readout_operation_observability_report.md", "\n".join(observability) + "\n")

    blocked_common = [
        "Blocked by Stage1 `NO_GO_NO_MEMORY_LEVER_FOUND`.",
        "",
        "Reason:",
        "",
        *bullet_lines(
            [
                "Stage1 initial map and corrected targeted trace both failed the discovery gate.",
                "The v106R plan explicitly forbids continuing action stages when Stage1 has no memory lever.",
                "No runtime action or full KITTI validation was run after this gate.",
            ]
        ),
    ]
    write_text(STAGE2 / "semantic_increment_not_run_due_stage1_no_memory_lever.md", "# Stage2 Blocked\n\n" + "\n".join(blocked_common) + "\n")
    write_text(STAGE3 / "role_assignment_not_run_due_stage1_no_memory_lever.md", "# Stage3 Blocked\n\n" + "\n".join(blocked_common) + "\n")
    write_text(STAGE4 / "action_surface_screen_blocked_by_stage1_no_memory_lever.md", "# Stage4 Blocked\n\n" + "\n".join(blocked_common) + "\n")
    write_text(STAGE5 / "runtime_failure_report.md", "# Stage5 Runtime Pilot Blocked\n\n" + "\n".join(blocked_common) + "\n")
    write_text(STAGE6 / "full_validation_blocked.md", "# Stage6 Full Validation Blocked\n\n" + "\n".join(blocked_common) + "\n")

    decision = {
        "schema": "acl2_v106r_final_decision_v1",
        "taxonomy": taxonomy,
        "objective_status": "completed_as_stage1_no_go",
        "stage0_pass": bool(facts["v106r_stage0_pass"]),
        "stage1_initial_discovery_pass": stage1_pass,
        "stage1_targeted_trace_parity_pass": parity_pass,
        "stage1_targeted_discovery_pass": targeted_pass,
        "action_stages_allowed": action_allowed,
        "semantic_increment_run": False,
        "role_disambiguation_run": False,
        "runtime_action_run": False,
        "full_validation_run": False,
        "new_full_kitti_ate_available": False,
        "baseline_full_kitti_reference_by_seq": facts["lingbot_full_kitti_baseline_by_seq"],
        "stage1_initial_summary": rel(STAGE1 / "stage1_summary.json"),
        "stage1_targeted_summary": rel(TARGETED / "targeted_trace_summary.json"),
        "stage1_no_memory_lever_found": rel(STAGE1 / "stage1_no_memory_lever_found.md"),
        "non_readout_operation_observability_report": rel(STAGE1 / "non_readout_operation_observability_report.md"),
    }
    write_json(FINAL / "final_decision.json", decision)

    decision_md = [
        "# ACL2 v106R Final Decision",
        "",
        f"Taxonomy: `{taxonomy}`",
        "",
        "This run did not produce a runtime LingBot method. It completed as a Stage1 No-Go because no observed GCA memory readout lever passed the discovery gate after targeted trace repair.",
        "",
        "Full KITTI ATE status:",
        "",
        "- No new v106R action/full-validation ATE exists, because Stage2-6 were blocked by the Stage1 gate.",
        "- The table below is the frozen v105 LingBot full-sequence baseline reference, not a new v106R method result.",
        "",
        *baseline_lines(facts),
        "",
        "Key Stage1 evidence:",
        "",
        *bullet_lines(
            [
                f"Initial map pass: `{stage1_pass}` with max_abs_corr_L3 `{stage1['max_abs_corr_L3']}` and max_abs_corr_rolling `{stage1['max_abs_corr_rolling']}`.",
                f"Targeted trace parity pass: `{parity_pass}` with failed_parity_rows `{targeted['failed_parity_rows']}` and trace_error_rows `{targeted['trace_error_rows']}`.",
                f"Targeted discovery pass: `{targeted_pass}` with max_abs_corr_L3 `{targeted['max_abs_corr_L3']}`.",
                f"Targeted operation rows: `{targeted['targeted_memory_operation_rows']}`.",
                f"Operation types present: `{','.join(targeted.get('operation_types_present', []))}`.",
                f"Operation types not content-observed: `{','.join(targeted.get('missing_operation_types', []))}`.",
            ]
        ),
        "",
        "Hypothesis status:",
        "",
        *bullet_lines(
            [
                "H1 not supported for observed GCA readout levers: correlations are far below the Stage1 threshold.",
                "H2 semantic increment was not run because Stage1 had no promotable top lever.",
                "H3 local/reference action was not run; v105 headlocal action remains forbidden repeat evidence.",
                "H4 update/retention remains unpromoted in this run because current trace does not expose content-dependent semantic update/retention decisions.",
                "H5 special-token readout is not supported as the main carrier in this targeted evidence; its top abs_corr_L3 is low.",
            ]
        ),
        "",
        "Allowed next move:",
        "",
        "A future run would need a richer cache-operation trace or a different observable operation surface before any semantic action can be justified. Re-running v105 headlocal demotion is explicitly disallowed by the v106R no-repeat guard.",
    ]
    write_text(FINAL / "final_decision.md", "\n".join(decision_md) + "\n")

    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
