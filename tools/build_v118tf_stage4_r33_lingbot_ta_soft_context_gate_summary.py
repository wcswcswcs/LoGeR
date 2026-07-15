#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R33 LingBot TA soft context-gate repair."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r33_lingbot_ta_soft_context_gate"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
MANIFEST = STAGE / "summary/stage4_r33_lingbot_ta_soft_context_gate_manifest.json"
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
        for value in manifest.get("force_context_indices_by_method", {})
        .get(method, {})
        .get(seq, [])
    }


def parse_mask(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            return tuple()
    return tuple(values)


def action_trace_stats(path: Path, selected: set[int], expected_mask: tuple[float, ...]) -> dict[str, Any]:
    rows = 0
    counts: Counter[str] = Counter()
    selected_seen: set[int] = set()
    selected_context_only: set[int] = set()
    selected_base_keyframes: set[int] = set()
    selected_final_non_keyframe: set[int] = set()
    selected_not_skip_append: set[int] = set()
    selected_token_mask_ok: set[int] = set()
    if not path.exists():
        return {
            "action_trace_exists": False,
            "action_trace_rows": 0,
            "action_trace_schema_counts": {},
            "selected_action_seen_count": 0,
            "selected_context_only_count": 0,
            "selected_base_keyframe_count": 0,
            "selected_final_non_keyframe_count": 0,
            "selected_not_skip_append_count": 0,
            "selected_token_mask_ok_count": 0,
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
            if row.get("forced_context_only") is True and row.get("context_only_append") is True:
                selected_context_only.add(frame)
            if row.get("base_is_keyframe") is True:
                selected_base_keyframes.add(frame)
            if row.get("final_is_keyframe") is False:
                selected_final_non_keyframe.add(frame)
            if row.get("skip_append") is False:
                selected_not_skip_append.add(frame)
            if parse_mask(str(row.get("token_type_mask", ""))) == expected_mask:
                selected_token_mask_ok.add(frame)
    good = (
        selected_context_only
        & selected_base_keyframes
        & selected_final_non_keyframe
        & selected_not_skip_append
        & selected_token_mask_ok
    )
    return {
        "action_trace_exists": True,
        "action_trace_rows": rows,
        "action_trace_schema_counts": dict(sorted(counts.items())),
        "selected_action_seen_count": len(selected_seen),
        "selected_context_only_count": len(selected_context_only),
        "selected_base_keyframe_count": len(selected_base_keyframes),
        "selected_final_non_keyframe_count": len(selected_final_non_keyframe),
        "selected_not_skip_append_count": len(selected_not_skip_append),
        "selected_token_mask_ok_count": len(selected_token_mask_ok),
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
    expected_mask = tuple(float(value) for value in manifest.get("context_token_mask", []))
    for method, meta in methods.items():
        for seq in ("00", "02"):
            selected = selected_set(manifest, method, seq)
            eval_json = metric_path(seq, method)
            metrics = read_json(eval_json)
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            action_path = action_dir / f"{method}_seq{seq}.jsonl"
            fi_path = trace_dir / f"{method}_seq{seq}.jsonl"
            row = {
                "schema": "acl2_v118tf_stage4_r33_lingbot_ta_soft_context_gate_row_v1",
                "dataset": DATASET,
                "seq": seq,
                "method": method,
                "branch": "LB-TA",
                "policy": meta.get("policy", ""),
                "role": meta.get("role", ""),
                "soft_context_gate_count": len(selected),
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
                "eval_json": rel(eval_json) if eval_json.exists() else str(eval_json),
                "action_trace": rel(action_path) if action_path.exists() else str(action_path),
                "fi_trace": rel(fi_path) if fi_path.exists() else str(fi_path),
                **action_trace_stats(action_path, selected, expected_mask),
                **fi_trace_stats(fi_path),
            }
            rows.append(row)
    return rows


def compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = {row["seq"]: row for row in rows if row["role"] == "candidate"}
    controls = {
        role: {row["seq"]: row for row in rows if row["role"] == role}
        for role in ("reverse_control", "matched_random_control")
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
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    manifest = read_json(MANIFEST)
    if not manifest:
        raise FileNotFoundError(MANIFEST)
    rows = summarize_rows(manifest)
    complete = all(row["eval_exists"] and row["action_trace_exists"] and row["fi_trace_exists"] for row in rows)
    action_fidelity = all(
        fnum(row.get("selected_action_fidelity")) == 1.0 and int(row.get("fi_trace_rows") or 0) > 0
        for row in rows
    )
    comparisons = compare(rows)
    candidate_better_all_controls = all(
        comparisons[role]["metrics"]["ate"]["all_candidate_better_than_control"]
        for role in ("reverse_control", "matched_random_control")
    )
    baseline_gate = comparisons["baseline"]["ate_rel_improvement_vs_default"]["pilot_gate"]
    if not complete:
        decision = "TA_SOFT_CONTEXT_GATE_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "TA_SOFT_CONTEXT_GATE_ACTION_FIDELITY_NO_GO"
    elif candidate_better_all_controls and baseline_gate:
        decision = "TA_SOFT_CONTEXT_GATE_PROMOTION_CANDIDATE_FOUND"
    else:
        decision = "TA_SOFT_CONTEXT_GATE_CONTROL_OR_BASELINE_NO_GO"

    out = STAGE / "summary"
    summary = {
        "schema": "acl2_v118tf_stage4_r33_lingbot_ta_soft_context_gate_summary_v1",
        "stage4_r33_decision": decision,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "complete": complete,
        "action_fidelity": action_fidelity,
        "row_count": len(rows),
        "candidate_better_all_controls": candidate_better_all_controls,
        "baseline_gate": baseline_gate,
        "comparisons": comparisons,
        "manifest": rel(MANIFEST),
        "boundary": (
            "R33 tests LB-TA soft context-token admission on the same fixed-budget default base-keyframe universe as R22. "
            "Selected frames are context-only appends with the pre-registered token mask, not hard skipped. "
            "A success claim requires action fidelity, candidate better than reverse/random controls, and default-baseline geometry gate."
        ),
        "outputs": {
            "rows": rel(out / "stage4_r33_lingbot_ta_soft_context_gate_rows.csv"),
            "summary": rel(out / "stage4_r33_lingbot_ta_soft_context_gate_summary.json"),
            "report": rel(out / "STAGE4_R33_LINGBOT_TA_SOFT_CONTEXT_GATE_REPORT.md"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r33_lingbot_ta_soft_context_gate_rows.csv", rows)
    write_json(out / "stage4_r33_lingbot_ta_soft_context_gate_summary.json", summary)
    report = [
        "# Stage4-R33 LingBot TA Soft Context-Gate Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"action_fidelity: `{action_fidelity}`",
        f"candidate_better_all_controls: `{candidate_better_all_controls}`",
        f"baseline_gate: `{baseline_gate}`",
        "",
        "```json",
        json.dumps(comparisons, indent=2, sort_keys=True),
        "```",
    ]
    (out / "STAGE4_R33_LINGBOT_TA_SOFT_CONTEXT_GATE_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R33",
            "surface_or_branch": "LB-TA",
            "status": decision,
            "artifact": rel(out / "stage4_r33_lingbot_ta_soft_context_gate_summary.json"),
            "notes": (
                "LB-TA soft context-token gate candidate/reverse/random full 00/02; "
                "global goal remains false unless controls and baseline gate pass"
            ),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
