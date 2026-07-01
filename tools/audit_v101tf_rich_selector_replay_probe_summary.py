#!/usr/bin/env python3
"""Summarize the v101 rich-selector replay probe.

The replay probe is a measured reproducibility check over the already selected
rich-selector rows.  It is not a fresh holdout and therefore cannot authorize
runtime action even if local metrics look positive.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
PROBE = ROOT / "outcomeD_merge_gauge_rich_selector_replay_probe"
DRY = ROOT / "outcomeD_merge_gauge_rich_selector_replay_probe_dryrun"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def append_unique_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    if line not in lines:
        lines.append(line)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def upsert_section(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    section = f"\n\n## {heading}\n\n{body.strip()}\n"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = f"\n## {heading}\n"
    if marker in text:
        prefix, rest = text.split(marker, 1)
        next_heading = rest.find("\n## ")
        if next_heading >= 0:
            text = prefix.rstrip() + section + rest[next_heading:]
        else:
            text = prefix.rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.lstrip() + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    dry_manifest = read_json(DRY / "runtime_probe_manifest.json")
    manifest = read_json(PROBE / "runtime_probe_manifest.json")
    sensitivity = read_json(PROBE / "runtime_probe_sensitivity_summary.json")
    variant_rows = read_rows(PROBE / "runtime_probe_variant_summary.csv")
    selected = sensitivity.get("selected_candidate_summary") or {}

    jobs = manifest.get("jobs") or []
    skipped_existing_count = sum(1 for row in jobs if row.get("skipped_existing"))
    rerun_count = sum(1 for row in jobs if not row.get("skipped_existing"))
    summary = {
        "schema": "acl2_v101_rich_selector_replay_probe_summary_v1",
        "dry_run_target_count": dry_manifest.get("target_count", ""),
        "dry_run_job_count": dry_manifest.get("job_count", ""),
        "replay_probe_target_count": manifest.get("target_count", ""),
        "replay_probe_job_count": manifest.get("job_count", ""),
        "replay_probe_completed_count": manifest.get("completed_count", ""),
        "replay_probe_failed_count": manifest.get("failed_count", ""),
        "replay_probe_all_completed": manifest.get("all_completed", False),
        "replay_probe_skipped_existing_count": skipped_existing_count,
        "replay_probe_rerun_count_after_skip_existing": rerun_count,
        "runtime_probe_executed": sensitivity.get("runtime_probe_executed", False),
        "runtime_probe_metric_row_count": sensitivity.get("metric_row_count", ""),
        "runtime_probe_effect_row_count": sensitivity.get("effect_row_count", ""),
        "phase3r_runtime_probe_gate_pass": sensitivity.get("phase3r_runtime_probe_gate_pass", False),
        "selected_candidate_variant": sensitivity.get("selected_candidate_variant", ""),
        "selected_candidate_bad_median_I_J_runtime_proxy": selected.get("bad_median_I_J_runtime_proxy", ""),
        "selected_candidate_good_max_worsen_runtime_proxy": selected.get("good_max_worsen_runtime_proxy", ""),
        "selected_candidate_sequence_coverage": selected.get("sequence_coverage", ""),
        "best_control_bad_median_I_J_runtime_proxy": sensitivity.get(
            "best_control_bad_median_I_J_runtime_proxy", ""
        ),
        "selected_candidate_beats_control": sensitivity.get("selected_candidate_beats_control", False),
        "original_phase3_gate_pass": sensitivity.get("original_phase3_gate_pass", ""),
        "original_balanced_probe_gate_pass": sensitivity.get("original_balanced_probe_gate_pass", ""),
        "blocker": sensitivity.get("blocker", ""),
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "claim": (
            "Measured replay over rich-selected existing rows only; not a fresh holdout, not action authorization."
        ),
    }

    write_json(FINAL / "rich_selector_replay_probe_summary.json", summary)
    write_rows(FINAL / "rich_selector_replay_probe_variant_summary.csv", variant_rows)
    write_text(
        PROBE / "failure_report.md",
        "Rich selector replay probe completed after retry, but it did not pass the measured runtime-probe gate because the selected candidate did not beat the measured control. This directory is a replay diagnostic, not runtime authorization.",
    )
    write_text(
        PROBE / "what_would_have_to_be_true_to_pass.md",
        "The selected replay would need to beat measured controls on the selected pair set, preserve good cases, pass the runtime-probe gate, and then be replicated on a fresh predeclared holdout before M4/runtime promotion.",
    )
    write_text(
        PROBE / "control_gap_report.md",
        "Control gap: selected merge_alpha_0p2 bad median I/J is below the best measured control bad median I/J on the same selected pair set.",
    )
    write_text(
        PROBE / "next_attempt_recommendation.md",
        "Do not promote this selected replay. A next attempt needs a fresh predeclared measured-control rerun or new holdout evidence, not another retrospective selected-set replay.",
    )
    write_rows(
        PROBE / "false_positive_false_negative_rows.csv",
        [
            {
                "cue_name": "rich_selector_replay_probe",
                "row_kind": "control_beating_failure",
                "selected_candidate_variant": summary["selected_candidate_variant"],
                "selected_candidate_bad_median_I_J_runtime_proxy": summary[
                    "selected_candidate_bad_median_I_J_runtime_proxy"
                ],
                "best_control_bad_median_I_J_runtime_proxy": summary[
                    "best_control_bad_median_I_J_runtime_proxy"
                ],
                "claim_level": "measured_replay_no_action",
            }
        ],
    )
    write_text(
        DRY / "failure_report.md",
        "Dry-run manifest only. It verifies command construction for the rich-selector replay probe but contains no measured outcome and cannot authorize action.",
    )
    write_text(
        DRY / "what_would_have_to_be_true_to_pass.md",
        "A dry-run cannot pass an experimental gate. The actual replay must complete and then pass measured control-beating gates on a fresh predeclared evaluation.",
    )
    write_text(
        DRY / "control_gap_report.md",
        "Dry-run has no measured controls; it only records the planned command/job structure.",
    )
    write_text(
        DRY / "next_attempt_recommendation.md",
        "Use the actual replay probe output and summary for measured evidence. Do not treat the dry-run as data.",
    )
    write_rows(
        DRY / "false_positive_false_negative_rows.csv",
        [
            {
                "cue_name": "rich_selector_replay_probe_dryrun",
                "row_kind": "dry_run_no_metrics",
                "dry_run_target_count": summary["dry_run_target_count"],
                "dry_run_job_count": summary["dry_run_job_count"],
                "claim_level": "command_feasibility_only",
            }
        ],
    )

    report = [
        "# Rich Selector Replay Probe Summary",
        "",
        "This probe reran the rich-selector selected pair set with measured runtime variants. It is useful as a reproducibility and control-beating check, but it is not a fresh holdout because the pair set came from the retrospective rich-selector screen.",
        "",
        "## Execution",
        "",
        f"- dry-run target count: `{summary['dry_run_target_count']}`",
        f"- dry-run job count: `{summary['dry_run_job_count']}`",
        f"- replay target count: `{summary['replay_probe_target_count']}`",
        f"- replay job count: `{summary['replay_probe_job_count']}`",
        f"- replay completed count: `{summary['replay_probe_completed_count']}`",
        f"- replay failed count after retry: `{summary['replay_probe_failed_count']}`",
        f"- all completed after retry: `{summary['replay_probe_all_completed']}`",
        f"- skipped-existing count in retry manifest: `{summary['replay_probe_skipped_existing_count']}`",
        f"- rerun count in retry manifest: `{summary['replay_probe_rerun_count_after_skip_existing']}`",
        "",
        "## Measured Result",
        "",
        f"- metric rows: `{summary['runtime_probe_metric_row_count']}`",
        f"- effect rows: `{summary['runtime_probe_effect_row_count']}`",
        f"- selected candidate variant: `{summary['selected_candidate_variant']}`",
        f"- selected bad median I/J: `{summary['selected_candidate_bad_median_I_J_runtime_proxy']}`",
        f"- best control bad median I/J: `{summary['best_control_bad_median_I_J_runtime_proxy']}`",
        f"- selected candidate beats control: `{summary['selected_candidate_beats_control']}`",
        f"- phase3r runtime probe gate pass: `{summary['phase3r_runtime_probe_gate_pass']}`",
        f"- blocker: `{summary['blocker']}`",
        "",
        "## Conclusion",
        "",
        "The selected `merge_alpha_0p2` replay improves bad rows locally and protects good rows, but it does not beat the measured control on the same selected set. Therefore this replay does not promote the rich selector to M4/runtime/full validation.",
    ]
    (FINAL / "rich_selector_replay_probe_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    recommendation = (
        "A measured replay over the rich-selector selected pair set completed after low-concurrency retry, but it "
        "did not pass the runtime probe gate: selected `merge_alpha_0p2` bad median I/J="
        f"{summary['selected_candidate_bad_median_I_J_runtime_proxy']} versus best control="
        f"{summary['best_control_bad_median_I_J_runtime_proxy']}, selected_candidate_beats_control="
        f"{summary['selected_candidate_beats_control']}. This is not a fresh holdout and does not authorize M4/runtime."
    )
    upsert_section(FINAL / "next_attempt_recommendation.md", "Rich Selector Replay Probe Follow-up", recommendation)
    append_unique_line(
        FINAL / "remaining_blockers.md",
        "- Rich-selector replay probe completed, but selected candidate did not beat measured control; M4/runtime/full validation remain unauthorized.",
    )
    append_unique_line(
        FINAL / "failure_report.md",
        "- Rich-selector replay probe follow-up: measured selected replay completed but phase3r_runtime_probe_gate_pass=false.",
    )
    append_unique_line(
        FINAL / "control_gap_report.md",
        "- Rich-selector replay control gap: selected `merge_alpha_0p2` did not beat measured control on the selected pair set.",
    )

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
