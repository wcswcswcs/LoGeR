#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R19 LingBot TR topK calibration."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r19_lingbot_tr_topk_calibration"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"

METHODS = {
    "lingbot_map_stream_flashinfer_v118_r19_tr1_qk_topk2": {"topk": 2, "role": "candidate", "policy": "TR1_QK_TOPK"},
    "lingbot_map_stream_flashinfer_v118_r19_tr1_random_topk2": {"topk": 2, "role": "matched_control", "policy": "TR1_RANDOM_TOPK"},
    "lingbot_map_stream_flashinfer_v118_r19_tr1_qk_topk4": {"topk": 4, "role": "candidate", "policy": "TR1_QK_TOPK"},
    "lingbot_map_stream_flashinfer_v118_r19_tr1_random_topk4": {"topk": 4, "role": "matched_control", "policy": "TR1_RANDOM_TOPK"},
    "lingbot_map_stream_flashinfer_v118_r19_tr1_qk_topk8": {"topk": 8, "role": "candidate", "policy": "TR1_QK_TOPK"},
    "lingbot_map_stream_flashinfer_v118_r19_tr1_random_topk8": {"topk": 8, "role": "matched_control", "policy": "TR1_RANDOM_TOPK"},
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


def metric_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / "eval/traj.json"


def trace_stats(path: Path) -> dict:
    rows = 0
    counts = Counter()
    action_rows = 0
    changed_action_rows = 0
    selected_counts = []
    if not path.exists():
        return {
            "trace_exists": False,
            "trace_rows": 0,
            "row_type_counts": {},
            "action_rows": 0,
            "changed_action_rows": 0,
            "selected_special_page_count_median": "",
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            counts[row.get("row_type", "")] += 1
            if row.get("action_branch") == "LB-TR" and row.get("row_type") == "action":
                action_rows += 1
                if row.get("action_changed_visible_table") is True:
                    changed_action_rows += 1
                selected = fnum(row.get("selected_special_page_count"))
                if selected is not None:
                    selected_counts.append(selected)
    return {
        "trace_exists": True,
        "trace_rows": rows,
        "row_type_counts": dict(sorted(counts.items())),
        "action_rows": action_rows,
        "changed_action_rows": changed_action_rows,
        "selected_special_page_count_median": median(selected_counts) if selected_counts else "",
    }


def summarize_rows() -> list[dict]:
    rows = []
    trace_dir = STAGE / "runtime_full_thread8/traces"
    for method, meta in METHODS.items():
        for seq in ("00", "02"):
            eval_json = metric_path(seq, method)
            trace = trace_dir / f"{method}_seq{seq}.jsonl"
            metrics = read_json(eval_json)
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            row = {
                "schema": "acl2_v118tf_stage4_r19_lingbot_tr_topk_row_v1",
                "dataset": DATASET,
                "seq": seq,
                "method": method,
                "branch": "LB-TR",
                "policy": meta["policy"],
                "topk": meta["topk"],
                "role": meta["role"],
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
    out = {}
    for topk in sorted({int(row["topk"]) for row in rows}):
        qk = {row["seq"]: row for row in rows if int(row["topk"]) == topk and row["role"] == "candidate"}
        rnd = {row["seq"]: row for row in rows if int(row["topk"]) == topk and row["role"] == "matched_control"}
        seqs = sorted(qk.keys() & rnd.keys())
        metrics = {}
        for metric in ("ate", "rpe_rot", "rpe_trans"):
            deltas = []
            per_seq = {}
            for seq in seqs:
                c = fnum(qk[seq].get(metric))
                ctrl = fnum(rnd[seq].get(metric))
                if c is None or ctrl is None:
                    continue
                delta = c - ctrl
                deltas.append(delta)
                per_seq[seq] = {
                    "candidate": c,
                    "matched_control": ctrl,
                    "candidate_minus_control": delta,
                }
            metrics[metric] = {
                "all_candidate_better_than_control": bool(deltas) and all(delta < 0 for delta in deltas),
                "median_candidate_minus_control": median(deltas) if deltas else "",
                "per_seq": per_seq,
            }
        rels = [fnum(row.get("ate_rel_improvement_vs_default")) for row in qk.values()]
        rels = [x for x in rels if x is not None]
        max_harm = abs(min(rels)) if rels and min(rels) < 0 else 0.0
        out[str(topk)] = {
            "seqs": seqs,
            "metrics": metrics,
            "ate_rel_improvement_vs_default": {
                "median": median(rels) if rels else "",
                "max_harm": max_harm,
                "both_sequences_improve": bool(rels) and all(x > 0 for x in rels),
                "pilot_gate": bool(rels) and median(rels) >= 0.03 and max_harm <= 0.01,
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
    action_changed = all(int(row["changed_action_rows"]) > 0 for row in rows)
    any_action_changed = any(int(row["changed_action_rows"]) > 0 for row in rows)
    comparisons = compare(rows)
    promoted_topks = [
        topk
        for topk, result in comparisons.items()
        if result["metrics"]["ate"]["all_candidate_better_than_control"]
        and result["ate_rel_improvement_vs_default"]["pilot_gate"]
    ]
    noop_topks = [
        topk
        for topk in sorted(comparisons)
        if not any(int(row["changed_action_rows"]) > 0 for row in rows if str(row["topk"]) == topk)
    ]
    if not complete:
        decision = "TR_TOPK_CALIBRATION_RUNTIME_INCOMPLETE"
    elif not any_action_changed:
        decision = "TR_TOPK_CALIBRATION_ACTION_FIDELITY_NO_GO"
    elif promoted_topks:
        decision = "TR_TOPK_CALIBRATION_PROMOTION_CANDIDATE_FOUND"
    else:
        decision = "TR_TOPK_CALIBRATION_BASELINE_OR_CONTROL_NO_GO_WITH_NOOP_TOPKS" if noop_topks else "TR_TOPK_CALIBRATION_BASELINE_OR_CONTROL_NO_GO"

    out = STAGE / "summary"
    summary = {
        "schema": "acl2_v118tf_stage4_r19_lingbot_tr_topk_summary_v1",
        "stage4_r19_decision": decision,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "complete": complete,
        "all_rows_changed_action": action_changed,
        "any_rows_changed_action": any_action_changed,
        "noop_topks": noop_topks,
        "row_count": len(rows),
        "promoted_topks": promoted_topks,
        "topk_comparisons": comparisons,
        "boundary": "R19 only calibrates LB-TR topK. Promotion requires QK topK to beat random topK and default FlashInfer baseline.",
        "outputs": {
            "rows": str((out / "stage4_r19_lingbot_tr_topk_rows.csv").relative_to(ROOT)),
            "summary": str((out / "stage4_r19_lingbot_tr_topk_summary.json").relative_to(ROOT)),
            "report": str((out / "STAGE4_R19_LINGBOT_TR_TOPK_REPORT.md").relative_to(ROOT)),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r19_lingbot_tr_topk_rows.csv", rows)
    (out / "stage4_r19_lingbot_tr_topk_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = [
        "# Stage4-R19 LingBot TR topK Calibration Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"promoted_topks: `{promoted_topks}`",
        "",
        "```json",
        json.dumps(comparisons, indent=2, sort_keys=True),
        "```",
    ]
    (out / "STAGE4_R19_LINGBOT_TR_TOPK_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
