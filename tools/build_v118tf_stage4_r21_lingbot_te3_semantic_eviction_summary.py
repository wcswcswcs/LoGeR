#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R21 LingBot TE3 semantic eviction repair."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r21_lingbot_te3_semantic_eviction"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"

METHOD_POLICIES = {
    "lingbot_map_stream_flashinfer_v118_r21_te3_semantic_persistence": ("TE3_SEMANTIC_PERSISTENCE_ONLY", "candidate"),
    "lingbot_map_stream_flashinfer_v118_r21_te3_reverse_semantic": ("TE3_REVERSE_SEMANTIC_PERSISTENCE", "reverse_control"),
    "lingbot_map_stream_flashinfer_v118_r21_te3_random_semantic": ("TE3_RANDOM_SEMANTIC_PERSISTENCE", "matched_random_control"),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / "eval/traj.json"


def trace_stats(path: Path) -> dict:
    rows = 0
    counts: Counter[str] = Counter()
    action_rows = 0
    changed_action_rows = 0
    missing_support_rows = 0
    selected_scores = []
    default_scores = []
    selected_minus_default_scores = []
    selected_track_counts = []
    default_track_counts = []
    selected_roles: Counter[str] = Counter()
    selected_labels: Counter[str] = Counter()
    if not path.exists():
        return {
            "trace_exists": False,
            "trace_rows": 0,
            "row_type_counts": {},
            "te_action_rows": 0,
            "changed_action_rows": 0,
            "semantic_support_missing_rows": 0,
            "selected_semantic_score_median": "",
            "default_fifo_semantic_score_median": "",
            "selected_minus_default_semantic_score_median": "",
            "selected_semantic_track_count_median": "",
            "default_fifo_semantic_track_count_median": "",
            "selected_role_counts": {},
            "selected_label_counts": {},
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            row_type = row.get("row_type", "")
            counts[row_type] += 1
            if row_type == "eviction" and row.get("action_branch") == "LB-TE":
                action_rows += 1
                if row.get("action_changed_eviction_order") is True:
                    changed_action_rows += 1
                if row.get("policy_selected_semantic_support_missing") or row.get("default_fifo_semantic_support_missing"):
                    missing_support_rows += 1
                selected = fnum(row.get("policy_selected_semantic_score"))
                default = fnum(row.get("default_fifo_semantic_score"))
                if selected is not None:
                    selected_scores.append(selected)
                if default is not None:
                    default_scores.append(default)
                if selected is not None and default is not None:
                    selected_minus_default_scores.append(selected - default)
                selected_count = fnum(row.get("policy_selected_semantic_track_count"))
                default_count = fnum(row.get("default_fifo_semantic_track_count"))
                if selected_count is not None:
                    selected_track_counts.append(selected_count)
                if default_count is not None:
                    default_track_counts.append(default_count)
                role = str(row.get("policy_selected_semantic_role", ""))
                label = str(row.get("policy_selected_semantic_label", ""))
                if role:
                    selected_roles[role] += 1
                if label:
                    selected_labels[label] += 1
    return {
        "trace_exists": True,
        "trace_rows": rows,
        "row_type_counts": dict(sorted(counts.items())),
        "te_action_rows": action_rows,
        "changed_action_rows": changed_action_rows,
        "semantic_support_missing_rows": missing_support_rows,
        "selected_semantic_score_median": median(selected_scores) if selected_scores else "",
        "default_fifo_semantic_score_median": median(default_scores) if default_scores else "",
        "selected_minus_default_semantic_score_median": median(selected_minus_default_scores) if selected_minus_default_scores else "",
        "selected_semantic_track_count_median": median(selected_track_counts) if selected_track_counts else "",
        "default_fifo_semantic_track_count_median": median(default_track_counts) if default_track_counts else "",
        "selected_role_counts": dict(selected_roles.most_common(10)),
        "selected_label_counts": dict(selected_labels.most_common(10)),
    }


def summarize_rows() -> list[dict]:
    rows = []
    trace_dir = STAGE / "runtime_full_thread8/traces"
    for method, (policy, role) in METHOD_POLICIES.items():
        for seq in ("00", "02"):
            eval_json = metric_path(seq, method)
            trace = trace_dir / f"{method}_seq{seq}.jsonl"
            metrics = read_json(eval_json)
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            row = {
                "schema": "acl2_v118tf_stage4_r21_lingbot_te3_semantic_eviction_row_v1",
                "dataset": DATASET,
                "seq": seq,
                "method": method,
                "branch": "LB-TE",
                "policy": policy,
                "role": role,
                "eval_exists": eval_json.exists(),
                "complete_marker_exists": (WORKSPACE / DATASET / seq / method / ".complete.json").exists(),
                "ate": metrics.get("ate", ""),
                "rpe_rot": metrics.get("rpe_rot", ""),
                "rpe_trans": metrics.get("rpe_trans", ""),
                "baseline_method": BASELINE_METHOD,
                "baseline_ate": baseline.get("ate", ""),
                "baseline_rpe_rot": baseline.get("rpe_rot", ""),
                "baseline_rpe_trans": baseline.get("rpe_trans", ""),
                "ate_rel_improvement_vs_default": (
                    (baseline_ate - ate) / baseline_ate
                    if ate is not None and baseline_ate not in (None, 0.0)
                    else ""
                ),
                "eval_json": str(eval_json.relative_to(ROOT)) if eval_json.exists() else str(eval_json),
                "trace": str(trace.relative_to(ROOT)) if trace.exists() else str(trace),
                **trace_stats(trace),
            }
            rows.append(row)
    return rows


def compare(rows: list[dict]) -> dict:
    candidate = {row["seq"]: row for row in rows if row["role"] == "candidate"}
    controls = {
        role: {row["seq"]: row for row in rows if row["role"] == role}
        for role in ("reverse_control", "matched_random_control")
    }
    comparisons = {}
    for control_role, control_rows in controls.items():
        seqs = sorted(candidate.keys() & control_rows.keys())
        metric_summary = {}
        for metric in ("ate", "rpe_rot", "rpe_trans"):
            deltas = []
            per_seq = {}
            for seq in seqs:
                c = fnum(candidate[seq].get(metric))
                ctrl = fnum(control_rows[seq].get(metric))
                if c is None or ctrl is None:
                    continue
                delta = c - ctrl
                deltas.append(delta)
                per_seq[seq] = {
                    "candidate": c,
                    control_role: ctrl,
                    "candidate_minus_control": delta,
                }
            metric_summary[metric] = {
                "all_candidate_better_than_control": bool(deltas) and all(delta < 0 for delta in deltas),
                "median_candidate_minus_control": median(deltas) if deltas else "",
                "per_seq": per_seq,
            }
        comparisons[control_role] = {"seqs": seqs, "metrics": metric_summary}
    rels = [fnum(row.get("ate_rel_improvement_vs_default")) for row in candidate.values()]
    rels = [value for value in rels if value is not None]
    max_harm = abs(min(rels)) if rels and min(rels) < 0 else 0.0
    comparisons["baseline"] = {
        "ate_rel_improvement_vs_default": {
            "median": median(rels) if rels else "",
            "max_harm": max_harm,
            "both_sequences_improve": bool(rels) and all(value > 0 for value in rels),
            "pilot_gate": bool(rels) and median(rels) >= 0.03 and max_harm <= 0.01,
        }
    }
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = summarize_rows()
    complete = all(row["eval_exists"] and row["trace_exists"] for row in rows)
    action_fidelity = all(int(row["te_action_rows"]) > 0 and int(row["changed_action_rows"]) > 0 for row in rows)
    semantic_support_present = all(int(row["semantic_support_missing_rows"]) == 0 for row in rows)
    comparisons = compare(rows)
    candidate_better_all_controls = all(
        comparisons[role]["metrics"]["ate"]["all_candidate_better_than_control"]
        for role in ("reverse_control", "matched_random_control")
    )
    baseline_gate = comparisons["baseline"]["ate_rel_improvement_vs_default"]["pilot_gate"]
    if not complete:
        decision = "TE3_SEMANTIC_EVICTION_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "TE3_SEMANTIC_EVICTION_ACTION_FIDELITY_NO_GO"
    elif not semantic_support_present:
        decision = "TE3_SEMANTIC_EVICTION_SEMANTIC_SUPPORT_INCOMPLETE"
    elif candidate_better_all_controls and baseline_gate:
        decision = "TE3_SEMANTIC_EVICTION_PROMOTION_CANDIDATE_FOUND"
    else:
        decision = "TE3_SEMANTIC_EVICTION_CONTROL_OR_BASELINE_NO_GO"

    out = STAGE / "summary"
    summary = {
        "schema": "acl2_v118tf_stage4_r21_lingbot_te3_semantic_eviction_summary_v1",
        "stage4_r21_decision": decision,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "complete": complete,
        "action_fidelity": action_fidelity,
        "semantic_support_present": semantic_support_present,
        "row_count": len(rows),
        "candidate_better_all_controls": candidate_better_all_controls,
        "baseline_gate": baseline_gate,
        "comparisons": comparisons,
        "boundary": (
            "R21 tests TE3 semantic persistence on actual local patch page eviction. "
            "A success claim requires candidate to beat reverse and matched random controls and improve over default FlashInfer baseline."
        ),
        "outputs": {
            "rows": str((out / "stage4_r21_lingbot_te3_semantic_eviction_rows.csv").relative_to(ROOT)),
            "summary": str((out / "stage4_r21_lingbot_te3_semantic_eviction_summary.json").relative_to(ROOT)),
            "report": str((out / "STAGE4_R21_LINGBOT_TE3_SEMANTIC_EVICTION_REPORT.md").relative_to(ROOT)),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r21_lingbot_te3_semantic_eviction_rows.csv", rows)
    (out / "stage4_r21_lingbot_te3_semantic_eviction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# Stage4-R21 LingBot TE3 Semantic Eviction Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"action_fidelity: `{action_fidelity}`",
        f"semantic_support_present: `{semantic_support_present}`",
        "",
        "```json",
        json.dumps(comparisons, indent=2, sort_keys=True),
        "```",
    ]
    (out / "STAGE4_R21_LINGBOT_TE3_SEMANTIC_EVICTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
