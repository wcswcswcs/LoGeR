#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R18 LingBot registered variants."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r18_lingbot_stage4_variant_expansion"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"

METHOD_POLICIES = {
    "lingbot_map_stream_flashinfer_v118_r18_tr2_segment_diversity": ("LB-TR", "TR2_QK_PLUS_SEGMENT_DIVERSITY", "candidate"),
    "lingbot_map_stream_flashinfer_v118_r18_tr1_reverse_qk": ("LB-TR", "TR1_REVERSE_QK_TOPK", "reverse_control"),
    "lingbot_map_stream_flashinfer_v118_r18_te2_redundancy": ("LB-TE", "TE2_REDUNDANCY_PRUNING", "candidate"),
    "lingbot_map_stream_flashinfer_v118_r18_te1_reverse": ("LB-TE", "TE1_REVERSE_READ_UTILITY", "reverse_control"),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def trace_stats(path: Path) -> dict:
    rows = 0
    counts = Counter()
    action_rows = 0
    changed_action_rows = 0
    if not path.exists():
        return {
            "trace_exists": False,
            "trace_rows": 0,
            "row_type_counts": {},
            "action_rows": 0,
            "changed_action_rows": 0,
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            row_type = row.get("row_type", "")
            counts[row_type] += 1
            if row.get("action_branch") == "LB-TR" and row_type == "action":
                action_rows += 1
                if row.get("action_changed_visible_table") is True:
                    changed_action_rows += 1
            if row.get("action_branch") == "LB-TE" and row_type == "eviction":
                action_rows += 1
                if row.get("action_changed_eviction_order") is True:
                    changed_action_rows += 1
    return {
        "trace_exists": True,
        "trace_rows": rows,
        "row_type_counts": dict(sorted(counts.items())),
        "action_rows": action_rows,
        "changed_action_rows": changed_action_rows,
    }


def metric_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / "eval/traj.json"


def summarize_rows() -> list[dict]:
    rows = []
    trace_dir = STAGE / "runtime_full_thread8/traces"
    for method, (branch, policy, role) in METHOD_POLICIES.items():
        for seq in ("00", "02"):
            eval_json = metric_path(seq, method)
            trace = trace_dir / f"{method}_seq{seq}.jsonl"
            metrics = read_json(eval_json)
            baseline_metrics = read_json(metric_path(seq, BASELINE_METHOD))
            row = {
                "schema": "acl2_v118tf_stage4_r18_lingbot_variant_row_v1",
                "dataset": DATASET,
                "seq": seq,
                "method": method,
                "branch": branch,
                "policy": policy,
                "role": role,
                "eval_exists": eval_json.exists(),
                "complete_marker_exists": (WORKSPACE / DATASET / seq / method / ".complete.json").exists(),
                "ate": metrics.get("ate", ""),
                "rpe_rot": metrics.get("rpe_rot", ""),
                "rpe_trans": metrics.get("rpe_trans", ""),
                "baseline_method": BASELINE_METHOD,
                "baseline_ate": baseline_metrics.get("ate", ""),
                "baseline_rpe_rot": baseline_metrics.get("rpe_rot", ""),
                "baseline_rpe_trans": baseline_metrics.get("rpe_trans", ""),
                "eval_json": str(eval_json.relative_to(ROOT)) if eval_json.exists() else str(eval_json),
                "trace": str(trace.relative_to(ROOT)) if trace.exists() else str(trace),
                **trace_stats(trace),
            }
            ate = fnum(row["ate"])
            baseline_ate = fnum(row["baseline_ate"])
            row["ate_rel_improvement_vs_default"] = (
                (baseline_ate - ate) / baseline_ate
                if ate is not None and baseline_ate not in (None, 0.0)
                else ""
            )
            rows.append(row)
    return rows


def compare_by_branch(rows: list[dict]) -> dict:
    out = {}
    for branch in sorted({row["branch"] for row in rows}):
        candidates = {row["seq"]: row for row in rows if row["branch"] == branch and row["role"] == "candidate"}
        controls = {row["seq"]: row for row in rows if row["branch"] == branch and row["role"] == "reverse_control"}
        seqs = sorted(candidates.keys() & controls.keys())
        metrics = {}
        for metric in ("ate", "rpe_rot", "rpe_trans"):
            per_seq = {}
            deltas = []
            for seq in seqs:
                c = fnum(candidates[seq].get(metric))
                ctrl = fnum(controls[seq].get(metric))
                if c is None or ctrl is None:
                    continue
                delta = c - ctrl
                deltas.append(delta)
                per_seq[seq] = {
                    "candidate": c,
                    "reverse_control": ctrl,
                    "candidate_minus_control": delta,
                }
            metrics[metric] = {
                "all_candidate_better_than_control": bool(deltas) and all(delta < 0 for delta in deltas),
                "median_candidate_minus_control": median(deltas) if deltas else "",
                "per_seq": per_seq,
            }
        rels = [fnum(row.get("ate_rel_improvement_vs_default")) for row in candidates.values()]
        rels = [x for x in rels if x is not None]
        out[branch] = {
            "seqs": seqs,
            "metrics": metrics,
            "ate_rel_improvement_vs_default": {
                "median": median(rels) if rels else "",
                "max_harm": abs(min(rels)) if rels and min(rels) < 0 else 0.0,
                "both_sequences_improve": bool(rels) and all(x > 0 for x in rels),
                "pilot_gate": bool(rels) and median(rels) >= 0.03 and (abs(min(rels)) if min(rels) < 0 else 0.0) <= 0.01,
            },
        }
    return out


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
    changed = all(int(row["changed_action_rows"]) > 0 for row in rows if row["role"] in {"candidate", "reverse_control"})
    comparisons = compare_by_branch(rows)
    promoted = [
        branch
        for branch, summary in comparisons.items()
        if summary["metrics"]["ate"]["all_candidate_better_than_control"]
        and summary["ate_rel_improvement_vs_default"]["pilot_gate"]
    ]
    decision = "FULL_VARIANT_RUNTIME_INCOMPLETE"
    if complete and changed and promoted:
        decision = "FULL_VARIANT_PROMOTION_CANDIDATE_FOUND"
    elif complete and changed:
        decision = "FULL_VARIANT_CONTROL_OR_BASELINE_NO_GO"
    elif complete:
        decision = "FULL_VARIANT_ACTION_FIDELITY_NO_GO"

    out = STAGE / "summary"
    summary = {
        "schema": "acl2_v118tf_stage4_r18_lingbot_variant_summary_v1",
        "stage4_r18_decision": decision,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "complete": complete,
        "all_candidate_control_rows_changed_action": changed,
        "row_count": len(rows),
        "promoted_branches": promoted,
        "branch_comparisons": comparisons,
        "boundary": (
            "R18 expands registered TR/TE variants. A branch can only promote if candidate "
            "beats reverse control and improves over the default FlashInfer baseline."
        ),
        "outputs": {
            "rows": str((out / "stage4_r18_lingbot_variant_rows.csv").relative_to(ROOT)),
            "summary": str((out / "stage4_r18_lingbot_variant_summary.json").relative_to(ROOT)),
            "report": str((out / "STAGE4_R18_LINGBOT_VARIANT_REPORT.md").relative_to(ROOT)),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r18_lingbot_variant_rows.csv", rows)
    (out / "stage4_r18_lingbot_variant_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = [
        "# Stage4-R18 LingBot Variant Expansion Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"all_candidate_control_rows_changed_action: `{changed}`",
        f"promoted_branches: `{promoted}`",
        "",
        "Lower ATE/RPE is better. Relative improvement is measured against `lingbot_map_stream_flashinfer_v118_r15_full`.",
        "",
        "```json",
        json.dumps(comparisons, indent=2, sort_keys=True),
        "```",
    ]
    (out / "STAGE4_R18_LINGBOT_VARIANT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
