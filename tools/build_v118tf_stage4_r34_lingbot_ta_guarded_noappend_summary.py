#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R34 LingBot TA guarded no-append repair."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r34_lingbot_ta_guarded_noappend"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
MANIFEST = STAGE / "summary/stage4_r34_lingbot_ta_guarded_noappend_manifest.json"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / "eval/traj.json"


def complete_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / ".complete.json"


def selected_set(manifest: dict[str, Any], method: str, seq: str) -> set[int]:
    return {
        int(value)
        for value in manifest.get("force_non_keyframe_indices_by_method", {})
        .get(method, {})
        .get(seq, [])
    }


def action_trace_stats(path: Path, selected: set[int]) -> dict[str, Any]:
    rows = 0
    counts: Counter[str] = Counter()
    selected_seen: set[int] = set()
    selected_forced: set[int] = set()
    selected_base_keyframes: set[int] = set()
    selected_final_non_keyframe: set[int] = set()
    selected_skip_append: set[int] = set()
    if not path.exists():
        return {
            "action_trace_exists": False,
            "action_trace_rows": 0,
            "action_trace_schema_counts": {},
            "selected_action_seen_count": 0,
            "selected_forced_non_keyframe_count": 0,
            "selected_base_keyframe_count": 0,
            "selected_final_non_keyframe_count": 0,
            "selected_skip_append_count": 0,
            "selected_action_fidelity": 0.0 if selected else "",
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            counts[str(row.get("schema", ""))] += 1
            frame = row.get("sample_position")
            if frame is None:
                continue
            frame = int(frame)
            if frame not in selected:
                continue
            selected_seen.add(frame)
            if row.get("forced_non_keyframe") is True:
                selected_forced.add(frame)
            if row.get("base_is_keyframe") is True:
                selected_base_keyframes.add(frame)
            if row.get("final_is_keyframe") is False:
                selected_final_non_keyframe.add(frame)
            if row.get("skip_append") is True:
                selected_skip_append.add(frame)
    good = selected_forced & selected_base_keyframes & selected_final_non_keyframe & selected_skip_append
    return {
        "action_trace_exists": True,
        "action_trace_rows": rows,
        "action_trace_schema_counts": dict(sorted(counts.items())),
        "selected_action_seen_count": len(selected_seen),
        "selected_forced_non_keyframe_count": len(selected_forced),
        "selected_base_keyframe_count": len(selected_base_keyframes),
        "selected_final_non_keyframe_count": len(selected_final_non_keyframe),
        "selected_skip_append_count": len(selected_skip_append),
        "selected_action_fidelity": len(good) / len(selected) if selected else "",
    }


def fi_trace_stats(path: Path) -> dict[str, Any]:
    rows = 0
    counts: Counter[str] = Counter()
    if not path.exists():
        return {"fi_trace_exists": False, "fi_trace_rows": 0, "fi_row_type_counts": {}}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            counts[str(row.get("row_type", ""))] += 1
    return {
        "fi_trace_exists": True,
        "fi_trace_rows": rows,
        "fi_row_type_counts": dict(sorted(counts.items())),
        "fi_append_rows": counts.get("append", 0),
        "fi_rollback_rows": counts.get("rollback", 0),
    }


def summarize_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    action_dir = STAGE / "runtime_full_thread8/action_traces"
    trace_dir = STAGE / "runtime_full_thread8/traces"
    methods = manifest.get("methods", {})
    manifest_rows = {
        (row.get("method", ""), row.get("seq", "")): row
        for row in read_csv(STAGE / "summary/stage4_r34_lingbot_ta_guarded_noappend_manifest.csv")
    }
    for method, meta in methods.items():
        for seq in ("00", "02"):
            selected = selected_set(manifest, method, seq)
            metrics = read_json(metric_path(seq, method))
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            action_path = action_dir / f"{method}_seq{seq}.jsonl"
            fi_path = trace_dir / f"{method}_seq{seq}.jsonl"
            manifest_row = manifest_rows.get((method, seq), {})
            row = {
                "schema": "acl2_v118tf_stage4_r34_lingbot_ta_guarded_noappend_row_v1",
                "dataset": DATASET,
                "seq": seq,
                "method": method,
                "branch": "LB-TA",
                "policy": meta.get("policy", ""),
                "role": meta.get("role", ""),
                "forced_non_keyframe_count": len(selected),
                "eval_exists": metric_path(seq, method).exists(),
                "complete_marker_exists": complete_path(seq, method).exists(),
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
                "guarded_risk_median": manifest_row.get("guarded_risk_median", ""),
                "guarded_risk_min": manifest_row.get("guarded_risk_min", ""),
                "guarded_risk_max": manifest_row.get("guarded_risk_max", ""),
                "eval_json": rel(metric_path(seq, method)) if metric_path(seq, method).exists() else str(metric_path(seq, method)),
                "action_trace": rel(action_path) if action_path.exists() else str(action_path),
                "fi_trace": rel(fi_path) if fi_path.exists() else str(fi_path),
                **action_trace_stats(action_path, selected),
                **fi_trace_stats(fi_path),
            }
            rows.append(row)
    return rows


def compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = {row["seq"]: row for row in rows if row["role"] == "candidate"}
    controls = {
        role: {row["seq"]: row for row in rows if row["role"] == role}
        for role in ("reverse_control", "matched_temporal_random_control")
    }
    comparisons: dict[str, Any] = {}
    for role, control_rows in controls.items():
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
                    role: ctrl,
                    "candidate_minus_control": delta,
                }
            metric_summary[metric] = {
                "all_candidate_better_than_control": bool(deltas) and all(delta < 0 for delta in deltas),
                "median_candidate_minus_control": median(deltas) if deltas else "",
                "per_seq": per_seq,
            }
        comparisons[role] = {"seqs": seqs, "metrics": metric_summary}
    rels = [fnum(row.get("ate_rel_improvement_vs_default")) for row in candidate.values()]
    rels = [value for value in rels if value is not None]
    max_harm = abs(min(rels)) if rels and min(rels) < 0 else 0.0
    comparisons["baseline"] = {
        "ate_rel_improvement_vs_default": {
            "median": median(rels) if rels else "",
            "max_harm": max_harm,
            "both_sequences_improve": bool(rels) and all(value > 0 for value in rels),
            "pilot_gate": bool(rels) and median(rels) >= 0.03 and max_harm <= 0.01,
            "per_seq": {
                seq: candidate[seq].get("ate_rel_improvement_vs_default", "")
                for seq in sorted(candidate)
            },
        }
    }
    return comparisons


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv(REGISTRY)
    fields: list[str] = []
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)
    for key in row:
        if key not in fields:
            fields.append(key)
    kept = [
        old
        for old in rows
        if not (
            old.get("stage") == row.get("stage")
            and old.get("surface_or_branch") == row.get("surface_or_branch")
            and old.get("artifact") == row.get("artifact")
        )
    ]
    kept.append({key: row.get(key, "") for key in fields})
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v118-TF Stage4-R34 LB-TA Guarded No-Append Report",
        "",
        f"- decision: `{summary['decision']}`",
        f"- complete: `{summary['complete']}`",
        f"- action_fidelity: `{summary['action_fidelity']}`",
        f"- baseline_gate: `{summary['baseline_gate']}`",
        f"- candidate_better_all_controls: `{summary['candidate_better_all_controls']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "## Rows",
        "",
        "| seq | role | method | ATE | rel vs default | action fidelity |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['seq']}` | `{row['role']}` | `{row['method']}` | "
            f"{row['ate']} | {row['ate_rel_improvement_vs_default']} | {row['selected_action_fidelity']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "R34 uses a fixed-budget hard no-append intervention on default base keyframes. The candidate uses only causal semantic-support fields from the R20 bridge rows; it does not claim missing internal-candidate or direct trajectory-semantic provenance.",
    ]
    write_text(STAGE / "summary/STAGE4_R34_LINGBOT_TA_GUARDED_NOAPPEND_REPORT.md", "\n".join(lines))


def main() -> None:
    manifest = read_json(MANIFEST)
    rows = summarize_rows(manifest)
    rows_csv = STAGE / "summary/stage4_r34_lingbot_ta_guarded_noappend_rows.csv"
    write_csv(rows_csv, rows)
    comparisons = compare(rows)
    complete = bool(rows) and all(row["eval_exists"] and row["complete_marker_exists"] for row in rows)
    action_fidelity = bool(rows) and all(fnum(row.get("selected_action_fidelity")) == 1.0 for row in rows)
    candidate_better_all_controls = all(
        comparisons.get(role, {}).get("metrics", {}).get("ate", {}).get("all_candidate_better_than_control") is True
        for role in ("reverse_control", "matched_temporal_random_control")
    )
    baseline_gate = bool(
        comparisons.get("baseline", {})
        .get("ate_rel_improvement_vs_default", {})
        .get("pilot_gate", False)
    )
    global_goal = complete and action_fidelity and candidate_better_all_controls and baseline_gate
    if global_goal:
        decision = "TA_GUARDED_NOAPPEND_PASS"
    elif candidate_better_all_controls and not baseline_gate:
        decision = "TA_GUARDED_NOAPPEND_CONTROL_PASS_BASELINE_GATE_FAIL"
    else:
        decision = "TA_GUARDED_NOAPPEND_CONTROL_OR_BASELINE_NO_GO"
    summary = {
        "schema": "acl2_v118tf_stage4_r34_lingbot_ta_guarded_noappend_summary_v1",
        "stage": rel(STAGE),
        "branch": "LB-TA",
        "decision": decision,
        "complete": complete,
        "action_fidelity": action_fidelity,
        "candidate_better_all_controls": candidate_better_all_controls,
        "baseline_gate": baseline_gate,
        "global_goal_achieved": global_goal,
        "row_count": len(rows),
        "comparisons": comparisons,
        "outputs": {
            "manifest": rel(MANIFEST),
            "rows_csv": rel(rows_csv),
            "report": rel(STAGE / "summary/STAGE4_R34_LINGBOT_TA_GUARDED_NOAPPEND_REPORT.md"),
            "runtime": rel(STAGE / "runtime_full_thread8"),
        },
        "boundary": (
            "Fixed-budget guarded no-append tests whether R22's seq02 harm can be reduced by "
            "using combined causal semantic-support evidence rather than max persistence alone. "
            "No missing internal-candidate or GT-derived runtime cue is claimed."
        ),
    }
    write_json(STAGE / "summary/stage4_r34_lingbot_ta_guarded_noappend_summary.json", summary)
    build_report(summary, rows)
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R34",
            "surface_or_branch": "LB-TA",
            "status": decision,
            "artifact": rel(STAGE / "summary/stage4_r34_lingbot_ta_guarded_noappend_summary.json"),
            "notes": "Fixed-budget guarded semantic-support hard no-append with reverse and temporal-random controls",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
