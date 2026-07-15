#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R17 LingBot retrieval/retention actions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r17_lingbot_retrieval_retention_action"

METHOD_POLICIES = {
    "lingbot_map_stream_flashinfer_v118_r17_tr1_qk_topk": ("LB-TR", "TR1_QK_TOPK", "candidate"),
    "lingbot_map_stream_flashinfer_v118_r17_tr1_random_topk": ("LB-TR", "TR1_RANDOM_TOPK", "matched_control"),
    "lingbot_map_stream_flashinfer_v118_r17_te1_read_utility": ("LB-TE", "TE1_READ_UTILITY_ONLY", "candidate"),
    "lingbot_map_stream_flashinfer_v118_r17_te1_random": ("LB-TE", "TE1_RANDOM_READ_UTILITY", "matched_control"),
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
    tr_rows = 0
    te_rows = 0
    qk_rows = 0
    read_rows = 0
    eviction_rows = 0
    selected_counts = []
    default_counts = []
    if not path.exists():
        return {
            "trace_exists": False,
            "trace_rows": 0,
            "row_type_counts": {},
            "action_rows": 0,
            "changed_action_rows": 0,
            "tr_action_rows": 0,
            "te_action_rows": 0,
            "read_rows": 0,
            "qk_read_rows": 0,
            "eviction_rows": 0,
            "selected_special_page_count_median": "",
            "default_special_page_count_median": "",
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            row_type = row.get("row_type", "")
            counts[row_type] += 1
            if row_type == "read":
                read_rows += 1
                if "qk_relevance_cosine" in row:
                    qk_rows += 1
            if row_type == "eviction":
                eviction_rows += 1
            if row.get("action_branch") == "LB-TR" and row_type == "action":
                tr_rows += 1
                action_rows += 1
                selected_counts.append(fnum(row.get("selected_special_page_count")))
                default_counts.append(fnum(row.get("default_special_page_count")))
                if row.get("action_changed_visible_table") is True:
                    changed_action_rows += 1
            if row.get("action_branch") == "LB-TE" and row_type == "eviction":
                te_rows += 1
                action_rows += 1
                if row.get("action_changed_eviction_order") is True:
                    changed_action_rows += 1
    selected_counts = [x for x in selected_counts if x is not None]
    default_counts = [x for x in default_counts if x is not None]
    return {
        "trace_exists": True,
        "trace_rows": rows,
        "row_type_counts": dict(sorted(counts.items())),
        "action_rows": action_rows,
        "changed_action_rows": changed_action_rows,
        "tr_action_rows": tr_rows,
        "te_action_rows": te_rows,
        "read_rows": read_rows,
        "qk_read_rows": qk_rows,
        "eviction_rows": eviction_rows,
        "selected_special_page_count_median": median(selected_counts) if selected_counts else "",
        "default_special_page_count_median": median(default_counts) if default_counts else "",
    }


def summarize_runtime(runtime: str, dataset: str, workspace: Path, trace_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    for method, (branch, policy, role) in METHOD_POLICIES.items():
        for seq in ("00", "02"):
            eval_json = workspace / dataset / seq / method / "eval/traj.json"
            trace = trace_dir / f"{method}_seq{seq}.jsonl"
            metrics = read_json(eval_json)
            stats = trace_stats(trace)
            row = {
                "schema": "acl2_v118tf_stage4_r17_lingbot_action_row_v1",
                "runtime": runtime,
                "dataset": dataset,
                "seq": seq,
                "method": method,
                "branch": branch,
                "policy": policy,
                "role": role,
                "eval_exists": eval_json.exists(),
                "workspace_complete": (workspace / dataset / seq / method / "metadata.json").exists(),
                "ate": metrics.get("ate", ""),
                "rpe_rot": metrics.get("rpe_rot", ""),
                "rpe_trans": metrics.get("rpe_trans", ""),
                "eval_json": str(eval_json.relative_to(ROOT)) if eval_json.exists() else str(eval_json),
                "trace": str(trace.relative_to(ROOT)) if trace.exists() else str(trace),
                "complete_marker_exists": (
                    (workspace / dataset / seq / method / ".complete.json").exists()
                    or (workspace / dataset / seq / method / "metadata.json").exists()
                ),
                **stats,
            }
            rows.append(row)
    complete = all(row["eval_exists"] and row["trace_exists"] for row in rows)
    action_fidelity_rows = [row for row in rows if row["action_rows"]]
    changed_rows = [row for row in rows if int(row["changed_action_rows"]) > 0]
    summary = {
        "runtime": runtime,
        "dataset": dataset,
        "row_count": len(rows),
        "complete": bool(complete),
        "action_fidelity_row_count": len(action_fidelity_rows),
        "changed_action_row_count": len(changed_rows),
        "all_rows_have_eval_and_trace": bool(complete),
    }
    return rows, summary


def summarize_control_comparisons(rows: list[dict], runtime: str) -> dict:
    comparisons = {}
    runtime_rows = [row for row in rows if row["runtime"] == runtime and row["eval_exists"]]
    for branch in sorted({row["branch"] for row in runtime_rows}):
        candidate_rows = [row for row in runtime_rows if row["branch"] == branch and row["role"] == "candidate"]
        control_rows = [row for row in runtime_rows if row["branch"] == branch and row["role"] == "matched_control"]
        candidate_by_seq = {row["seq"]: row for row in candidate_rows}
        control_by_seq = {row["seq"]: row for row in control_rows}
        common = sorted(candidate_by_seq.keys() & control_by_seq.keys())
        metric_summary = {}
        for metric in ("ate", "rpe_rot", "rpe_trans"):
            deltas = []
            per_seq = {}
            for seq in common:
                candidate_value = fnum(candidate_by_seq[seq].get(metric))
                control_value = fnum(control_by_seq[seq].get(metric))
                if candidate_value is None or control_value is None:
                    continue
                delta = candidate_value - control_value
                deltas.append(delta)
                per_seq[seq] = {
                    "candidate": candidate_value,
                    "matched_control": control_value,
                    "candidate_minus_control": delta,
                }
            metric_summary[metric] = {
                "mean_candidate_minus_control": mean(deltas) if deltas else "",
                "all_candidate_better_than_control": bool(deltas) and all(delta < 0 for delta in deltas),
                "per_seq": per_seq,
            }
        comparisons[branch] = {
            "common_seq_count": len(common),
            "seqs": common,
            "metrics": metric_summary,
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
    out = STAGE / "summary"
    trace32_rows, trace32_summary = summarize_runtime(
        "trace32",
        "kitti_v118_r17_00_02_trace32",
        STAGE / "workspace_trace32",
        STAGE / "runtime_trace32/traces",
    )
    rows = trace32_rows
    summaries = {"trace32": trace32_summary}
    full_workspace = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
    full_runtime = "full"
    full_trace_dir = STAGE / "runtime_full/traces"
    thread8_trace_dir = STAGE / "runtime_full_thread8/traces"
    if thread8_trace_dir.exists():
        full_runtime = "full_thread8"
        full_trace_dir = thread8_trace_dir
    if full_trace_dir.exists():
        full_rows, full_summary = summarize_runtime(
            full_runtime,
            "kitti_v105_00_01_02_05",
            full_workspace,
            full_trace_dir,
        )
        rows += full_rows
        summaries[full_runtime] = full_summary
    trace32_complete = trace32_summary["complete"]
    trace32_action_pressure = trace32_summary["changed_action_row_count"] > 0
    full_summary = summaries.get(full_runtime, {}) if full_trace_dir.exists() else {}
    control_comparisons = summarize_control_comparisons(rows, full_runtime) if full_summary else {}
    full_ate_control_pass = bool(control_comparisons) and all(
        branch_summary["metrics"]["ate"]["all_candidate_better_than_control"]
        for branch_summary in control_comparisons.values()
    )
    decision = "TRACE32_RUNTIME_PASS_FULL_ACTION_FIDELITY_PENDING"
    if not trace32_complete:
        decision = "TRACE32_RUNTIME_INCOMPLETE"
    elif full_summary.get("complete") and full_summary.get("changed_action_row_count", 0) > 0:
        decision = (
            "FULL_ACTION_FIDELITY_AVAILABLE_CONTROL_PASS_GEOMETRY_GATE_PENDING"
            if full_ate_control_pass
            else "FULL_ACTION_FIDELITY_AVAILABLE_CONTROL_NO_GO"
        )
    summary = {
        "schema": "acl2_v118tf_stage4_r17_lingbot_action_summary_v1",
        "stage4_r17_decision": decision,
        "global_goal_achieved": False,
        "trace32_complete": trace32_complete,
        "trace32_action_pressure_observed": bool(trace32_action_pressure),
        "runtime_summaries": summaries,
        "full_runtime_selected": full_runtime if full_summary else "",
        "control_comparisons": control_comparisons,
        "boundary": (
            "Trace32 is a runtime smoke. Full 00/02 action-fidelity and matched-control gates "
            "are required before any LB-TR/LB-TE success claim; lower ATE/RPE is better."
        ),
        "outputs": {
            "rows": str((out / "stage4_r17_lingbot_action_rows.csv").relative_to(ROOT)),
            "summary": str((out / "stage4_r17_lingbot_action_summary.json").relative_to(ROOT)),
            "report": str((out / "STAGE4_R17_LINGBOT_ACTION_REPORT.md").relative_to(ROOT)),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r17_lingbot_action_rows.csv", rows)
    (out / "stage4_r17_lingbot_action_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Stage4-R17 LingBot Retrieval/Retention Action Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"trace32_complete: `{trace32_complete}`",
        f"trace32_action_pressure_observed: `{trace32_action_pressure}`",
        f"full_runtime_selected: `{summary['full_runtime_selected']}`",
        "",
        "Trace32 is only a runtime smoke when it does not create enough special pages or eviction pressure.",
        "",
        "Matched-control comparisons use candidate minus control; negative is better.",
        "```json",
        json.dumps(control_comparisons, indent=2, sort_keys=True),
        "```",
    ]
    (out / "STAGE4_R17_LINGBOT_ACTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
