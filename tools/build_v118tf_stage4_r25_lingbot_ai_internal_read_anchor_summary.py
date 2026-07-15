#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R25 internal-read LingBot anchor repair."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r25_lingbot_ai_internal_read_anchor"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
MANIFEST = STAGE / "summary/stage4_r25_lingbot_ai_internal_read_anchor_manifest.json"


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


def selected_set(manifest: dict, method: str, seq: str) -> set[int]:
    return {
        int(value)
        for value in manifest.get("scale_frame_indices_by_method", {})
        .get(method, {})
        .get(seq, [])
    }


def parse_scale_frame_list(raw: Any) -> set[int]:
    if raw is None:
        return set()
    text = str(raw).strip()
    if not text:
        return set()
    return {int(part) for part in text.split(",") if part.strip()}


def action_trace_stats(path: Path, selected: set[int]) -> dict:
    rows = 0
    scale_rows = 0
    scale_seen: set[int] = set()
    matching_list_rows = 0
    selected_sorted = ",".join(str(value) for value in sorted(selected))
    if not path.exists():
        return {
            "action_trace_exists": False,
            "action_trace_rows": 0,
            "anchor_scale_action_rows": 0,
            "selected_scale_seen_count": 0,
            "selected_scale_coverage": 0.0 if selected else "",
            "scale_frame_list_match_rows": 0,
            "scale_frame_list_match": False,
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if row.get("anchor_scale_frame") is not True:
                continue
            scale_rows += 1
            frame = row.get("sample_position")
            if frame is not None:
                scale_seen.add(int(frame))
            if ",".join(str(value) for value in sorted(parse_scale_frame_list(row.get("scale_frame_indices")))) == selected_sorted:
                matching_list_rows += 1
    return {
        "action_trace_exists": True,
        "action_trace_rows": rows,
        "anchor_scale_action_rows": scale_rows,
        "selected_scale_seen_count": len(scale_seen & selected),
        "selected_scale_coverage": len(scale_seen & selected) / len(selected) if selected else "",
        "scale_frame_list_match_rows": matching_list_rows,
        "scale_frame_list_match": bool(selected) and matching_list_rows >= len(selected),
    }


def summarize_rows(manifest: dict) -> list[dict]:
    rows = []
    action_dir = STAGE / "runtime_full_thread8/action_traces"
    for method, meta in manifest.get("methods", {}).items():
        for seq in ("00", "02"):
            selected = selected_set(manifest, method, seq)
            eval_json = metric_path(seq, method)
            metrics = read_json(eval_json)
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            action_path = action_dir / f"{method}_seq{seq}.jsonl"
            rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r25_lingbot_ai_internal_read_anchor_row_v1",
                    "dataset": DATASET,
                    "seq": seq,
                    "method": method,
                    "branch": "LB-AI",
                    "policy": meta.get("policy", ""),
                    "role": meta.get("role", ""),
                    "selected_scale_frame_count": len(selected),
                    "selected_scale_frames": ";".join(str(value) for value in sorted(selected)),
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
                    **action_trace_stats(action_path, selected),
                }
            )
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
                per_seq[seq] = {"candidate": c, role: ctrl, "candidate_minus_control": delta}
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
    complete = all(row["eval_exists"] and row["action_trace_exists"] for row in rows)
    action_fidelity = all(
        fnum(row.get("selected_scale_coverage")) == 1.0
        and int(row.get("anchor_scale_action_rows") or 0) == int(row.get("selected_scale_frame_count") or 0)
        and row.get("scale_frame_list_match") is True
        for row in rows
    )
    comparisons = compare(rows)
    candidate_better_all_controls = all(
        comparisons[role]["metrics"]["ate"]["all_candidate_better_than_control"]
        for role in ("reverse_control", "matched_random_control")
    )
    baseline_gate = comparisons["baseline"]["ate_rel_improvement_vs_default"]["pilot_gate"]
    if not complete:
        decision = "AI_INTERNAL_READ_ANCHOR_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "AI_INTERNAL_READ_ANCHOR_ACTION_FIDELITY_NO_GO"
    elif candidate_better_all_controls and baseline_gate:
        decision = "AI_INTERNAL_READ_ANCHOR_PROMOTION_CANDIDATE_FOUND"
    else:
        decision = "AI_INTERNAL_READ_ANCHOR_CONTROL_OR_BASELINE_NO_GO"

    out = STAGE / "summary"
    summary = {
        "schema": "acl2_v118tf_stage4_r25_lingbot_ai_internal_read_anchor_summary_v1",
        "stage4_r25_decision": decision,
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
            "R25 tests LB-AI internal-QK anchor initialization using default FlashInfer read rows. "
            "Candidate and controls select 8 anchors from the same first32 non-default local-frame pool, excluding default 0-7 anchors to avoid anchor-family/read-count leakage."
        ),
        "outputs": {
            "rows": str((out / "stage4_r25_lingbot_ai_internal_read_anchor_rows.csv").relative_to(ROOT)),
            "summary": str((out / "stage4_r25_lingbot_ai_internal_read_anchor_summary.json").relative_to(ROOT)),
            "report": str((out / "STAGE4_R25_LINGBOT_AI_INTERNAL_READ_ANCHOR_REPORT.md").relative_to(ROOT)),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r25_lingbot_ai_internal_read_anchor_rows.csv", rows)
    (out / "stage4_r25_lingbot_ai_internal_read_anchor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# Stage4-R25 LingBot AI Internal Read Anchor Summary",
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
    (out / "STAGE4_R25_LINGBOT_AI_INTERNAL_READ_ANCHOR_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
