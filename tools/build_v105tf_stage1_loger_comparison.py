#!/usr/bin/env python3
"""Build the ACL2 v105-TF Stage 1 LoGeR comparison audit rows.

This intentionally does not turn v104 LoGeR case-level evidence into fake
sequence trajectory metrics.  The output records what is actually comparable
and writes a blocker note for missing full-sequence L0-L4 LoGeR trajectories.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE1 = V105 / "stage1_lingbot_baseline"
LINGBOT_ROWS = STAGE1 / "debug96_metrics/stage1_debug96_all_metric_rows.csv"
V104 = ROOT / "results/acl2_v104tf_strict_provider_evidence_eligibility_state_machine_memory_control"
PROVIDER_CASES = V104 / "stage1_provider/provider_case_summary.csv"
STAGE2_CASES = V104 / "stage2_oracle/evidence_state_case_summary.csv"
STAGE3_BOUNDARY = V104 / "stage3_state_machine/boundary_state_rows.csv"
STAGE1_PROVIDER_SUMMARY = V104 / "stage1_provider/stage1_provider_summary.json"
STAGE2_SUMMARY = V104 / "stage2_oracle/stage2_summary.json"
STAGE3_SUMMARY = V104 / "stage3_state_machine/state_machine_summary.json"
FINAL_DECISION = V104 / "stage7_final_decision/final_decision.json"
NO_ACTION_PARITY = V104 / "stage1_provider/selected11_no_action_parity_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


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


def group_counts(rows: list[dict[str, str]], key: str) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        seq = row.get("seq", "")
        if seq:
            counts[seq][row.get(key, "")] += 1
    return counts


def compact_counts(counter: Counter[str]) -> str:
    return ";".join(f"{k}:{v}" for k, v in sorted(counter.items()) if k)


def lingbot_stream_sequences() -> list[str]:
    rows = read_csv(LINGBOT_ROWS) if LINGBOT_ROWS.is_file() else []
    return sorted(
        {
            row["seq"]
            for row in rows
            if row.get("model") == "LingBot"
            and row.get("mode") == "streaming"
            and row.get("setting") == "default"
        }
    )


def build() -> dict[str, Any]:
    provider = read_csv(PROVIDER_CASES)
    stage2 = read_csv(STAGE2_CASES)
    stage3 = read_csv(STAGE3_BOUNDARY)
    provider_summary = read_json(STAGE1_PROVIDER_SUMMARY)
    stage2_summary = read_json(STAGE2_SUMMARY)
    stage3_summary = read_json(STAGE3_SUMMARY)
    final = read_json(FINAL_DECISION)
    no_action = read_json(NO_ACTION_PARITY)

    state_counts = group_counts(stage2, "evidence_state")
    boundary_counts = group_counts(stage3, "boundary_state")
    rows: list[dict[str, Any]] = []
    for seq in sorted({row["seq"] for row in provider if row.get("seq")}):
        seq_rows = [row for row in provider if row.get("seq") == seq]
        selected = [row for row in seq_rows if truthy(row.get("selected_action_case", ""))]
        blocker_counts = Counter(row.get("first_blocker", "") for row in seq_rows)
        rows.append(
            {
                "schema": "acl2_v105tf_stage1_loger_case_level_comparison_row_v1",
                "seq": seq,
                "model": "LoGeR",
                "comparison_scope": "v104_case_level_boundary_witness_not_full_trajectory",
                "comparable_to_lingbot_debug96_l0_l4": False,
                "source_result_root": V104.relative_to(ROOT).as_posix(),
                "case_count": len(seq_rows),
                "selected_action_case_count": len(selected),
                "provider_join_pass_count": sum(truthy(r.get("provider_join_pass", "")) for r in seq_rows),
                "strict_identity_candidate_count": sum(truthy(r.get("strict_identity_candidate", "")) for r in seq_rows),
                "nonproxy_current_support_count": sum(truthy(r.get("nonproxy_current_support", "")) for r in seq_rows),
                "semantic_class_fallback_count": sum(truthy(r.get("semantic_class_fallback", "")) for r in seq_rows),
                "proxy_only_count": sum(truthy(r.get("proxy_only", "")) for r in seq_rows),
                "stage2_evidence_state_counts": compact_counts(state_counts.get(seq, Counter())),
                "stage3_boundary_state_counts": compact_counts(boundary_counts.get(seq, Counter())),
                "dominant_first_blockers": compact_counts(blocker_counts),
                "runtime_action_allowed": False,
                "full_sequence_l0_l4_available": False,
                "full_ATE": "",
                "rolling_ATE_p90": "",
                "adjacent_log_scale_jump_p90": "",
                "handoff_transfer_penalty_median": "",
            }
        )

    summary = {
        "schema": "acl2_v105tf_stage1_loger_comparison_summary_v1",
        "lingbot_stream_default_debug_sequences": lingbot_stream_sequences(),
        "loger_sequences_with_case_level_evidence": sorted({row["seq"] for row in rows}),
        "loger_comparison_scope": "case_level_v104_boundary_witness_and_state_machine",
        "direct_l0_l4_trajectory_comparison_available": False,
        "reason_direct_l0_l4_missing": (
            "v104 artifacts contain selected boundary/case-level witness and provider-control evidence, "
            "not full-sequence LoGeR trajectories aligned to the LingBot debug96 frame universe."
        ),
        "v104_final_taxonomy": final.get("final_taxonomy"),
        "v104_goal_achieved": final.get("goal_achieved"),
        "v104_runtime_action_allowed": final.get("runtime_action_allowed"),
        "v104_stage1_provider_pass": final.get("stage1_provider_pass"),
        "v104_stage2_semantic_diagnostic_pass": final.get("stage2_semantic_diagnostic_pass"),
        "v104_stage3_state_machine_pass": final.get("stage3_state_machine_pass"),
        "v104_no_action_parity_pass": no_action.get("no_trace_pose_sha_parity_pass"),
        "v104_provider_join_failure_rate": provider_summary.get("provider_join_failure_rate"),
        "v104_exact_witness_safe_good_case_ids": provider_summary.get("exact_witness_safe_good_case_ids", []),
        "v104_exact_witness_strict_positive_case_ids": provider_summary.get("exact_witness_strict_positive_case_ids", []),
        "v104_tracked_local_diagnostic_case_ids": stage3_summary.get("tracked_local_diagnostic_case_ids", []),
        "v104_semantic_good_FPR_reduction": stage2_summary.get("semantic_good_FPR_reduction"),
        "v104_semantic_balanced_accuracy_gain": stage2_summary.get("semantic_balanced_accuracy_gain"),
    }

    write_csv(STAGE1 / "loger_comparison_metrics.csv", rows)
    (STAGE1 / "loger_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    blocker_lines = [
        "# LoGeR Comparison Scope Blocker",
        "",
        "This file records why Stage1 has a LoGeR comparison but not a direct L0-L4 trajectory baseline.",
        "",
        f"- LingBot stream-default debug sequences: `{summary['lingbot_stream_default_debug_sequences']}`",
        f"- LoGeR v104 case-level sequences: `{summary['loger_sequences_with_case_level_evidence']}`",
        "- Direct L0-L4 comparison available: `false`",
        f"- Reason: {summary['reason_direct_l0_l4_missing']}",
        f"- v104 final taxonomy: `{summary['v104_final_taxonomy']}`",
        f"- v104 no-action parity pass: `{summary['v104_no_action_parity_pass']}`",
        f"- v104 provider join failure rate: `{summary['v104_provider_join_failure_rate']}`",
        f"- v104 exact witness strict positives: `{summary['v104_exact_witness_strict_positive_case_ids']}`",
        f"- v104 exact witness safe-good cases: `{summary['v104_exact_witness_safe_good_case_ids']}`",
        "",
        "Interpretation: this is enough to compare failure taxonomy and witness-role pressure, but not enough to claim LoGeR/LingBot full-trajectory metric parity.",
    ]
    (STAGE1 / "loger_comparison_scope_blocker.md").write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
