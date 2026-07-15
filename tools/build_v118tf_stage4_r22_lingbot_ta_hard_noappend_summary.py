#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R22 LingBot TA hard no-append repair."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r22_lingbot_ta_hard_noappend"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
MANIFEST = STAGE / "summary/stage4_r22_lingbot_ta_hard_noappend_manifest.json"


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


def complete_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / ".complete.json"


def forced_set(manifest: dict, method: str, seq: str) -> set[int]:
    return {
        int(value)
        for value in manifest.get("force_non_keyframe_indices_by_method", {})
        .get(method, {})
        .get(seq, [])
    }


def action_trace_stats(path: Path, forced: set[int]) -> dict:
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
            "selected_action_fidelity": 0.0 if forced else "",
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
            if frame not in forced:
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
        "selected_action_fidelity": len(good) / len(forced) if forced else "",
    }


def fi_trace_stats(path: Path, forced: set[int]) -> dict:
    rows = 0
    counts: Counter[str] = Counter()
    rollback_frames: set[int] = set()
    append_frames: set[int] = set()
    if not path.exists():
        return {
            "fi_trace_exists": False,
            "fi_trace_rows": 0,
            "fi_row_type_counts": {},
            "selected_rollback_frame_count": 0,
            "selected_rollback_coverage": 0.0 if forced else "",
            "selected_append_frame_count": 0,
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            row_type = str(row.get("row_type", ""))
            counts[row_type] += 1
            frame = row.get("source_frame_id")
            if frame is None:
                continue
            frame = int(frame)
            if row_type == "rollback":
                rollback_frames.add(frame)
            elif row_type == "append":
                append_frames.add(frame)
    selected_rollback = rollback_frames & forced
    selected_append = append_frames & forced
    source_frames = rollback_frames | append_frames
    source_frame_max = max(source_frames) if source_frames else ""
    source_frame_id_matches_input_frame = (
        isinstance(source_frame_max, int)
        and bool(forced)
        and source_frame_max >= max(forced)
    )
    return {
        "fi_trace_exists": True,
        "fi_trace_rows": rows,
        "fi_row_type_counts": dict(sorted(counts.items())),
        "fi_rollback_rows": counts.get("rollback", 0),
        "fi_source_frame_id_max": source_frame_max,
        "fi_source_frame_id_matches_input_frame": source_frame_id_matches_input_frame,
        "selected_rollback_frame_count": len(selected_rollback),
        "selected_rollback_coverage": len(selected_rollback) / len(forced) if forced else "",
        "selected_append_frame_count": len(selected_append),
    }


def summarize_rows(manifest: dict) -> list[dict]:
    rows = []
    action_dir = STAGE / "runtime_full_thread8/action_traces"
    trace_dir = STAGE / "runtime_full_thread8/traces"
    methods = manifest.get("methods", {})
    for method, meta in methods.items():
        for seq in ("00", "02"):
            forced = forced_set(manifest, method, seq)
            eval_json = metric_path(seq, method)
            metrics = read_json(eval_json)
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            action_path = action_dir / f"{method}_seq{seq}.jsonl"
            fi_path = trace_dir / f"{method}_seq{seq}.jsonl"
            row = {
                "schema": "acl2_v118tf_stage4_r22_lingbot_ta_hard_noappend_row_v1",
                "dataset": DATASET,
                "seq": seq,
                "method": method,
                "branch": "LB-TA",
                "policy": meta.get("policy", ""),
                "role": meta.get("role", ""),
                "forced_non_keyframe_count": len(forced),
                "eval_exists": eval_json.exists(),
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
                "eval_json": str(eval_json.relative_to(ROOT)) if eval_json.exists() else str(eval_json),
                "action_trace": str(action_path.relative_to(ROOT)) if action_path.exists() else str(action_path),
                "fi_trace": str(fi_path.relative_to(ROOT)) if fi_path.exists() else str(fi_path),
                **action_trace_stats(action_path, forced),
                **fi_trace_stats(fi_path, forced),
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
    manifest = read_json(MANIFEST)
    if not manifest:
        raise FileNotFoundError(MANIFEST)
    rows = summarize_rows(manifest)
    complete = all(row["eval_exists"] and row["action_trace_exists"] and row["fi_trace_exists"] for row in rows)
    action_fidelity = all(
        fnum(row.get("selected_action_fidelity")) == 1.0
        and int(row.get("fi_rollback_rows") or 0) > 0
        for row in rows
    )
    comparisons = compare(rows)
    candidate_better_all_controls = all(
        comparisons[role]["metrics"]["ate"]["all_candidate_better_than_control"]
        for role in ("reverse_control", "matched_random_control")
    )
    baseline_gate = comparisons["baseline"]["ate_rel_improvement_vs_default"]["pilot_gate"]
    if not complete:
        decision = "TA_HARD_NOAPPEND_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "TA_HARD_NOAPPEND_ACTION_FIDELITY_NO_GO"
    elif candidate_better_all_controls and baseline_gate:
        decision = "TA_HARD_NOAPPEND_PROMOTION_CANDIDATE_FOUND"
    else:
        decision = "TA_HARD_NOAPPEND_CONTROL_OR_BASELINE_NO_GO"

    out = STAGE / "summary"
    summary = {
        "schema": "acl2_v118tf_stage4_r22_lingbot_ta_hard_noappend_summary_v1",
        "stage4_r22_decision": decision,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "complete": complete,
        "action_fidelity": action_fidelity,
        "row_count": len(rows),
        "candidate_better_all_controls": candidate_better_all_controls,
        "baseline_gate": baseline_gate,
        "comparisons": comparisons,
        "boundary": (
            "R22 tests LB-TA hard trajectory-admission no-append on default base keyframes. "
            "It does not claim exact trajectory semantic token provenance. Per-frame action success is gated on GCT action rows; "
            "FlashInfer rollback rows are used only as runtime path evidence because their source_frame_id field is not an input-frame id."
        ),
        "outputs": {
            "rows": str((out / "stage4_r22_lingbot_ta_hard_noappend_rows.csv").relative_to(ROOT)),
            "summary": str((out / "stage4_r22_lingbot_ta_hard_noappend_summary.json").relative_to(ROOT)),
            "report": str((out / "STAGE4_R22_LINGBOT_TA_HARD_NOAPPEND_REPORT.md").relative_to(ROOT)),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r22_lingbot_ta_hard_noappend_rows.csv", rows)
    (out / "stage4_r22_lingbot_ta_hard_noappend_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# Stage4-R22 LingBot TA Hard No-Append Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"action_fidelity: `{action_fidelity}`",
        "",
        "```json",
        json.dumps(comparisons, indent=2, sort_keys=True),
        "```",
    ]
    (out / "STAGE4_R22_LINGBOT_TA_HARD_NOAPPEND_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
