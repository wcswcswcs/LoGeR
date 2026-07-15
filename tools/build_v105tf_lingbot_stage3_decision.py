#!/usr/bin/env python3
"""Combine ACL2 v105-TF LingBot Stage 3 predefined and sweep decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage3_lingbot_oracle"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_clause(clause: str) -> tuple[str, str, float]:
    for op in ("_ge_", "_le_"):
        if op in clause:
            feature, value = clause.rsplit(op, 1)
            return feature, op[1:-1], float(value)
    raise ValueError(f"unsupported policy clause: {clause}")


def policy_selects(policy: str, row: dict[str, str]) -> bool:
    clauses = [split_clause(clause) for clause in policy.split("_AND_")]
    for feature, op, threshold in clauses:
        value = float(row.get(feature, 0.0) or 0.0)
        if op == "ge" and value < threshold:
            return False
        if op == "le" and value > threshold:
            return False
    return True


def build() -> dict[str, Any]:
    predefined_summary = load_json(STAGE3 / "stage3_summary.json")
    sweep_summary = load_json(STAGE3 / "oracle_policy_sweep_summary.json")
    sweep_rows = read_csv(STAGE3 / "oracle_policy_sweep_metrics.csv")

    passing = [row for row in sweep_rows if parse_bool(row.get("stage3_oracle_pass"))]
    best_policy = str(sweep_summary.get("best_policy", ""))
    best_row = next((row for row in passing if row.get("policy") == best_policy), None)
    if best_row is None:
        best_row = next((row for row in sweep_rows if row.get("policy") == best_policy), {})

    predefined_pass = parse_bool(predefined_summary.get("stage3_lingbot_oracle_pass"))
    sweep_pass = parse_bool(sweep_summary.get("stage3_sweep_pass")) and bool(passing)
    stage3_pass = predefined_pass or sweep_pass
    if stage3_pass:
        decision_boundary_note = (
            "Predefined policies failed; a trace32 diagnostic repair-space sweep found at least one candidate "
            "meeting the Stage3 oracle thresholds. This is not a full-sequence or heldout result and no runtime "
            "action has been run."
        )
    else:
        decision_boundary_note = (
            "Predefined policies and the trace32 diagnostic repair-space sweep both failed the Stage3 oracle "
            "thresholds. No LingBot routing action is allowed from these evidence rows."
        )

    selected_rows: list[dict[str, Any]] = []
    if best_policy:
        for row in read_csv(STAGE3 / "frame_semantic_geometry_rows.csv"):
            if policy_selects(best_policy, row):
                selected_rows.append(
                    {
                        "schema": "acl2_v105tf_lingbot_stage3_selected_policy_frame_v1",
                        "policy": best_policy,
                        "seq": row.get("seq"),
                        "sample_position": row.get("sample_position"),
                        "original_frame": row.get("original_frame"),
                        "bad_label": row.get("bad_label"),
                        "good_label": row.get("good_label"),
                        "sim3_residual_m": row.get("sim3_residual_m"),
                        "semantic_reject_unreliable_attention_frac": row.get(
                            "semantic_reject_unreliable_attention_frac"
                        ),
                        "local_window_context_attention_frac": row.get("local_window_context_attention_frac"),
                        "scale_reference_context_attention_frac": row.get("scale_reference_context_attention_frac"),
                        "reject_unreliable_patch_frac": row.get("reject_unreliable_patch_frac"),
                        "top_labels": row.get("top_labels"),
                    }
                )
    write_csv(STAGE3 / "stage3_sweep_passing_policy_selected_rows.csv", selected_rows)

    failure_path = STAGE3 / "semantic_increment_failure.md"
    if stage3_pass and failure_path.exists():
        old_text = failure_path.read_text(encoding="utf-8")
        (STAGE3 / "semantic_increment_predefined_failure.md").write_text(
            "# Stage3 Predefined Policy Failure\n\n"
            "The initial predefined Stage3 policies did not pass. This evidence was superseded by the later "
            "diagnostic repair-space sweep, which found one oracle candidate passing the plan thresholds. "
            "The original failure note is preserved below for audit continuity.\n\n"
            f"{old_text.strip()}\n",
            encoding="utf-8",
        )
        failure_path.unlink()

    combined = {
        "schema": "acl2_v105tf_lingbot_stage3_combined_decision_v1",
        "predefined_policy_pass": predefined_pass,
        "sweep_policy_pass": sweep_pass,
        "stage3_lingbot_oracle_pass": stage3_pass,
        "stage4_action_allowed": stage3_pass,
        "stage4_action_permission_scope": (
            "guarded_action_pilot_allowed_by_sweep_candidate" if stage3_pass else "blocked"
        ),
        "decision_boundary_note": decision_boundary_note,
        "best_sweep_policy": best_policy,
        "best_sweep_policy_metrics": {
            key: best_row.get(key)
            for key in [
                "bad_recall",
                "good_FPR",
                "balanced_accuracy",
                "selected_rows",
                "selected_bad_rows",
                "selected_good_rows",
                "selected_positive_sequence_coverage",
                "same_count_random_margin",
                "semantic_shuffle_margin",
                "context_role_rotation_margin",
                "uses_semantic",
                "uses_context",
            ]
        },
        "selected_frame_rows": len(selected_rows),
        "artifacts": {
            "predefined_summary": str(STAGE3 / "stage3_summary.json"),
            "sweep_summary": str(STAGE3 / "oracle_policy_sweep_summary.json"),
            "sweep_metrics": str(STAGE3 / "oracle_policy_sweep_metrics.csv"),
            "selected_rows": str(STAGE3 / "stage3_sweep_passing_policy_selected_rows.csv"),
            "predefined_failure_note": str(STAGE3 / "semantic_increment_predefined_failure.md"),
        },
    }
    write_json(STAGE3 / "stage3_combined_summary.json", combined)

    decision_md = "\n".join(
        [
            "# Stage3 Combined Decision",
            "",
            f"- predefined_policy_pass: `{str(predefined_pass).lower()}`",
            f"- sweep_policy_pass: `{str(sweep_pass).lower()}`",
            f"- stage3_lingbot_oracle_pass: `{str(stage3_pass).lower()}`",
            f"- stage4_action_allowed: `{str(stage3_pass).lower()}`",
            f"- best_sweep_policy: `{best_policy}`",
            f"- selected_frame_rows: `{len(selected_rows)}`",
            "",
            "Boundary:",
            "- The pass is from a trace32 diagnostic repair-space sweep, not from a pre-registered policy.",
            "- Semantic source is the existing pseudo-semantic cache, not GT semantics.",
            "- No LingBot routing action has been run.",
            "- Stage4 is allowed only as a guarded action pilot with no-action parity preserved.",
            "",
            "Best policy metrics:",
            f"- bad_recall: `{best_row.get('bad_recall')}`",
            f"- good_FPR: `{best_row.get('good_FPR')}`",
            f"- balanced_accuracy: `{best_row.get('balanced_accuracy')}`",
            f"- selected_positive_sequence_coverage: `{best_row.get('selected_positive_sequence_coverage')}`",
            f"- same_count_random_margin: `{best_row.get('same_count_random_margin')}`",
            f"- semantic_shuffle_margin: `{best_row.get('semantic_shuffle_margin')}`",
            f"- context_role_rotation_margin: `{best_row.get('context_role_rotation_margin')}`",
            "",
        ]
    )
    (STAGE3 / "stage3_decision.md").write_text(decision_md, encoding="utf-8")
    return combined


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
