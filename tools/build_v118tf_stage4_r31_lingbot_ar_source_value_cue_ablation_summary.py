#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R31 LingBot AR source-value cue ablations."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import median
from typing import Any


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r31")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r31_lingbot_ar_source_value_cue_ablation")

import build_v118tf_stage4_r28_lingbot_ar_anchor_read_summary as base_summary


ROOT = base_summary.ROOT
RESULT_ROOT = base_summary.RESULT_ROOT
STAGE = base_summary.STAGE
STAGE_TAG = base_summary.STAGE_TAG
DATASET = base_summary.DATASET
BASELINE_METHOD = base_summary.BASELINE_METHOD
MANIFEST = base_summary.MANIFEST
REGISTRY = base_summary.REGISTRY
SEQS = base_summary.SEQS
R30_ROWS = (
    RESULT_ROOT
    / "stage4_r30_lingbot_ar_source_value_scaling/summary/stage4_r30_lingbot_ar_anchor_read_rows.csv"
)


def rel(path: Path) -> str:
    return base_summary.rel(path)


def fnum(value: Any) -> float | None:
    return base_summary.fnum(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    return base_summary.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    return base_summary.write_csv(path, rows)


def variant_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        role = str(row.get("role", ""))
        if not role.startswith("candidate_"):
            continue
        variant = role.removeprefix("candidate_")
        seq = str(row.get("seq"))
        ate = fnum(row.get("ate"))
        baseline_ate = fnum(row.get("baseline_ate"))
        rel_improvement = fnum(row.get("ate_rel_improvement_vs_default"))
        bucket = by_variant.setdefault(
            variant,
            {
                "role": role,
                "seqs": {},
                "ate_rel_improvements": [],
                "ates": [],
            },
        )
        bucket["seqs"][seq] = {
            "method": row.get("method", ""),
            "policy": row.get("policy", ""),
            "ate": ate,
            "baseline_ate": baseline_ate,
            "ate_rel_improvement_vs_default": rel_improvement,
            "action_fidelity_pass": row.get("action_fidelity_pass"),
            "action_log_rows": row.get("action_log_rows"),
            "weight_min_observed": row.get("weight_min_observed"),
            "weight_max_observed": row.get("weight_max_observed"),
        }
        if rel_improvement is not None:
            bucket["ate_rel_improvements"].append(rel_improvement)
        if ate is not None:
            bucket["ates"].append(ate)
    for variant, bucket in by_variant.items():
        rels = list(bucket.pop("ate_rel_improvements"))
        max_harm = abs(min(rels)) if rels and min(rels) < 0 else 0.0
        bucket["median_ate_rel_improvement_vs_default"] = median(rels) if rels else ""
        bucket["mean_ate_rel_improvement_vs_default"] = sum(rels) / len(rels) if rels else ""
        bucket["max_harm"] = max_harm
        bucket["both_sequences_improve"] = len(rels) == len(SEQS) and all(value > 0 for value in rels)
        bucket["pilot_gate"] = (
            len(rels) == len(SEQS)
            and median(rels) >= 0.03
            and max_harm <= 0.01
        )
    return by_variant


def r30_control_rows() -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_csv(R30_ROWS):
        role = row.get("role", "")
        if role not in {"reverse_control", "shuffled_control"}:
            continue
        seq = str(row.get("seq"))
        out.setdefault(role, {})[seq] = row
    return out


def compare_with_controls(stats: dict[str, Any], controls: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for variant, bucket in stats.items():
        variant_cmp: dict[str, Any] = {}
        for control_role, control_by_seq in controls.items():
            per_seq: dict[str, Any] = {}
            deltas = []
            for seq, seq_row in bucket["seqs"].items():
                ate = fnum(seq_row.get("ate"))
                ctrl = fnum(control_by_seq.get(seq, {}).get("ate"))
                if ate is None or ctrl is None:
                    continue
                delta = ate - ctrl
                deltas.append(delta)
                per_seq[seq] = {
                    "variant_ate": ate,
                    control_role: ctrl,
                    "variant_minus_control": delta,
                }
            variant_cmp[control_role] = {
                "all_variant_better_than_control": bool(deltas) and all(delta < 0 for delta in deltas),
                "median_variant_minus_control": median(deltas) if deltas else "",
                "per_seq": per_seq,
            }
        comparisons[variant] = variant_cmp
    return comparisons


def best_variants(stats: dict[str, Any], control_comparisons: dict[str, Any]) -> dict[str, Any]:
    ranked = []
    for variant, bucket in stats.items():
        median_rel = fnum(bucket.get("median_ate_rel_improvement_vs_default"))
        max_harm = fnum(bucket.get("max_harm"))
        if median_rel is None:
            continue
        better_all_controls = all(
            ctrl_cmp.get("all_variant_better_than_control") is True
            for ctrl_cmp in control_comparisons.get(variant, {}).values()
        )
        ranked.append(
            {
                "variant": variant,
                "median_ate_rel_improvement_vs_default": median_rel,
                "max_harm": max_harm,
                "both_sequences_improve": bucket.get("both_sequences_improve"),
                "pilot_gate": bucket.get("pilot_gate"),
                "better_all_r30_controls": bool(control_comparisons.get(variant)) and better_all_controls,
            }
        )
    ranked.sort(key=lambda row: (row["median_ate_rel_improvement_vs_default"], -float(row["max_harm"] or 0.0)), reverse=True)
    safe_ranked = [
        row
        for row in ranked
        if row.get("both_sequences_improve") is True and float(row.get("max_harm") or 0.0) <= 0.01
    ]
    validated = [
        row
        for row in ranked
        if row.get("pilot_gate") is True and row.get("better_all_r30_controls") is True
    ]
    return {
        "ranked_by_median_rel": ranked,
        "best_by_median_rel": ranked[0] if ranked else {},
        "best_safe": safe_ranked[0] if safe_ranked else {},
        "validated_candidate_variants": validated,
    }


def add_registry_row(row: dict[str, Any]) -> None:
    return base_summary.add_registry_row(row)


def main() -> None:
    manifest = base_summary.read_json(MANIFEST)
    if not manifest:
        raise FileNotFoundError(MANIFEST)
    rows = base_summary.summarize_rows(manifest)
    complete = all(row["eval_exists"] and row["complete_marker_exists"] and row["action_trace_exists"] for row in rows)
    action_fidelity = all(row.get("action_fidelity_pass") is True for row in rows)
    stats = variant_stats(rows)
    controls = r30_control_rows()
    control_comparisons = compare_with_controls(stats, controls)
    best = best_variants(stats, control_comparisons)
    validated = best["validated_candidate_variants"]

    if not complete:
        decision = "AR_SOURCE_VALUE_CUE_ABLATION_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "AR_SOURCE_VALUE_CUE_ABLATION_ACTION_FIDELITY_NO_GO"
    elif validated:
        decision = "AR_SOURCE_VALUE_CUE_ABLATION_CANDIDATE_FOUND_REQUIRES_FRESH_VALIDATION"
    else:
        decision = "AR_SOURCE_VALUE_CUE_ABLATION_NO_STABLE_CONTROL_RESISTANT_CUE"

    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"stage4_{STAGE_TAG}_lingbot_ar_source_value_cue_ablation_rows.csv"
    summary_path = out / f"stage4_{STAGE_TAG}_lingbot_ar_source_value_cue_ablation_summary.json"
    report_path = out / f"STAGE4_{STAGE_TAG.upper()}_LINGBOT_AR_SOURCE_VALUE_CUE_ABLATION_REPORT.md"
    write_csv(rows_path, rows)
    summary = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_source_value_cue_ablation_summary_v1",
        f"stage4_{STAGE_TAG}_decision": decision,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "action_mode": manifest.get("action_mode", ""),
        "intervention_form": manifest.get("intervention_form", ""),
        "complete": complete,
        "action_fidelity": action_fidelity,
        "row_count": len(rows),
        "r30_control_rows": rel(R30_ROWS) if R30_ROWS.exists() else str(R30_ROWS),
        "r30_controls_available": bool(controls),
        "variant_stats": stats,
        "control_comparisons": control_comparisons,
        "best_variants": best,
        "boundary": (
            "R31 is a 00/02 cue ablation over the already verified source-value hook. A candidate here still "
            "requires fresh validation and the remaining mechanism-dissection checks before any global claim."
        ),
        "outputs": {
            "rows": rel(rows_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = [
        f"# Stage4-{STAGE_TAG.upper()} LingBot AR Source-Value Cue Ablation Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"action_fidelity: `{action_fidelity}`",
        "",
        "```json",
        json.dumps(
            {
                "best_variants": best,
                "variant_stats": stats,
                "control_comparisons": control_comparisons,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": f"Stage4-{STAGE_TAG.upper()}",
            "surface_or_branch": "LB-AR",
            "status": decision,
            "artifact": rel(summary_path),
            "notes": "LB-AR source-value cue ablation over fixed source frames 0-7; dev-only 00/02, global success remains false",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
