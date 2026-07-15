#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R27 holdout calibrated-polarity anchor validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor"
R26_STAGE = RESULT_ROOT / "stage4_r26_lingbot_ai_calibrated_polarity_anchor"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
MANIFEST = STAGE / "summary/stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_manifest.json"
R26_ROWS = R26_STAGE / "summary/stage4_r26_lingbot_ai_calibrated_polarity_anchor_rows.csv"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
SEQS = ("01", "05")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


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


def summarize_r27_rows(manifest: dict) -> list[dict]:
    rows = []
    action_dir = STAGE / "runtime_full_thread8/action_traces"
    for method, meta in manifest.get("methods", {}).items():
        for seq in SEQS:
            selected = selected_set(manifest, method, seq)
            eval_json = metric_path(seq, method)
            metrics = read_json(eval_json)
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            action_path = action_dir / f"{method}_seq{seq}.jsonl"
            rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_row_v1",
                    "source_stage": "R27_holdout",
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
                    "eval_json": rel(eval_json) if eval_json.exists() else str(eval_json),
                    "action_trace": rel(action_path) if action_path.exists() else str(action_path),
                    **action_trace_stats(action_path, selected),
                }
            )
    return rows


def normalize_r26_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("seq") not in {"00", "02"}:
            continue
        if row.get("role") not in {"candidate", "opposite_polarity_control", "matched_random_control"}:
            continue
        copied: dict[str, Any] = dict(row)
        copied["source_stage"] = "R26_dev"
        out.append(copied)
    return out


def compare_by_role(rows: list[dict[str, Any]], seqs: set[str]) -> dict:
    candidate = {row["seq"]: row for row in rows if row.get("role") == "candidate" and row.get("seq") in seqs}
    controls = {
        role: {row["seq"]: row for row in rows if row.get("role") == role and row.get("seq") in seqs}
        for role in ("opposite_polarity_control", "matched_random_control")
    }
    comparisons = {}
    for role, control_rows in controls.items():
        common = sorted(candidate.keys() & control_rows.keys())
        metric_summary = {}
        for metric in ("ate", "rpe_rot", "rpe_trans"):
            deltas = []
            per_seq = {}
            for seq in common:
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

        margin_rows = []
        for seq in common:
            c_rel = fnum(candidate[seq].get("ate_rel_improvement_vs_default"))
            ctrl_rel = fnum(control_rows[seq].get("ate_rel_improvement_vs_default"))
            if c_rel is None or ctrl_rel is None:
                continue
            margin_rows.append(c_rel - ctrl_rel)
        comparisons[role] = {
            "seqs": common,
            "metrics": metric_summary,
            "rel_improvement_margin_vs_control": {
                "per_seq": {
                    seq: fnum(candidate[seq].get("ate_rel_improvement_vs_default"))
                    - fnum(control_rows[seq].get("ate_rel_improvement_vs_default"))
                    for seq in common
                    if fnum(candidate[seq].get("ate_rel_improvement_vs_default")) is not None
                    and fnum(control_rows[seq].get("ate_rel_improvement_vs_default")) is not None
                },
                "all_margins_gt_0p01": bool(margin_rows) and all(value > 0.01 for value in margin_rows),
                "median_margin": median(margin_rows) if margin_rows else "",
            },
        }
    return comparisons


def four_sequence_ate_gate(combined_rows: list[dict[str, Any]]) -> dict:
    candidate = {row["seq"]: row for row in combined_rows if row.get("role") == "candidate"}
    rel_by_seq = {
        seq: fnum(row.get("ate_rel_improvement_vs_default"))
        for seq, row in candidate.items()
        if fnum(row.get("ate_rel_improvement_vs_default")) is not None
    }
    rels = list(rel_by_seq.values())
    harms = [-value for value in rels if value < 0]
    return {
        "required_seqs": ["00", "01", "02", "05"],
        "available_seqs": sorted(rel_by_seq),
        "rel_improvement_by_seq": rel_by_seq,
        "median_full_ate_relative_improvement": median(rels) if rels else "",
        "mean_full_ate_relative_improvement": mean(rels) if rels else "",
        "improved_sequence_count": sum(1 for value in rels if value > 0),
        "max_full_ate_harm": max(harms) if harms else 0.0,
        "median_gate_ge_0p05": bool(rels) and len(rels) == 4 and median(rels) >= 0.05,
        "mean_gate_gt_0": bool(rels) and len(rels) == 4 and mean(rels) > 0,
        "improved_sequences_gate_ge_3_of_4": len(rels) == 4 and sum(1 for value in rels if value > 0) >= 3,
        "max_harm_gate_le_0p01": len(rels) == 4 and (max(harms) if harms else 0.0) <= 0.01,
    }


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
    r27_rows = summarize_r27_rows(manifest)
    r26_rows = normalize_r26_rows(read_csv(R26_ROWS))
    combined_rows = r26_rows + r27_rows
    complete = all(row["eval_exists"] and row["action_trace_exists"] for row in r27_rows)
    action_fidelity = all(
        fnum(row.get("selected_scale_coverage")) == 1.0
        and int(row.get("anchor_scale_action_rows") or 0) == int(row.get("selected_scale_frame_count") or 0)
        and row.get("scale_frame_list_match") is True
        for row in r27_rows
    )
    holdout_comparisons = compare_by_role(r27_rows, {"01", "05"})
    four_seq_comparisons = compare_by_role(combined_rows, {"00", "01", "02", "05"})
    four_seq_ate = four_sequence_ate_gate(combined_rows)
    holdout_candidate_better_all_controls = all(
        holdout_comparisons[role]["metrics"]["ate"]["all_candidate_better_than_control"]
        for role in ("opposite_polarity_control", "matched_random_control")
    )
    four_seq_controls_not_matched = all(
        four_seq_comparisons[role]["rel_improvement_margin_vs_control"]["all_margins_gt_0p01"]
        for role in ("opposite_polarity_control", "matched_random_control")
    )
    four_seq_ate_gate = all(
        four_seq_ate[key] is True
        for key in (
            "median_gate_ge_0p05",
            "mean_gate_gt_0",
            "improved_sequences_gate_ge_3_of_4",
            "max_harm_gate_le_0p01",
        )
    )
    non_ate_risk_gates = {
        "rolling_p90_median_improvement_gt_0": "not_available_from_current_lingbot_traj_evaluator",
        "final_error_median_not_worse_gt_2pct": "not_available_from_current_lingbot_traj_evaluator",
        "lingbot_local_window_median_harm_le_2pct": "not_available_from_current_lingbot_traj_evaluator",
    }
    non_ate_risk_gate_complete = False
    ate_control_holdout_pass = bool(
        complete
        and action_fidelity
        and holdout_candidate_better_all_controls
        and four_seq_controls_not_matched
        and four_seq_ate_gate
    )
    if not complete:
        decision = "AI4_HOLDOUT_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "AI4_HOLDOUT_ACTION_FIDELITY_NO_GO"
    elif not ate_control_holdout_pass:
        decision = "AI4_HOLDOUT_CONTROL_OR_FOURSEQ_ATE_NO_GO"
    else:
        decision = "AI4_HOLDOUT_ATE_CONTROL_PASS_NON_ATE_RISK_GATES_PENDING"

    out = STAGE / "summary"
    summary = {
        "schema": "acl2_v118tf_stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_summary_v1",
        "stage4_r27_decision": decision,
        "global_goal_achieved": bool(ate_control_holdout_pass and non_ate_risk_gate_complete),
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "complete": complete,
        "action_fidelity": action_fidelity,
        "row_count": len(r27_rows),
        "holdout_candidate_better_all_controls": holdout_candidate_better_all_controls,
        "four_sequence_controls_not_matched_within_0p01": four_seq_controls_not_matched,
        "four_sequence_ate_gate": four_seq_ate_gate,
        "ate_control_holdout_pass": ate_control_holdout_pass,
        "non_ate_risk_gate_complete": non_ate_risk_gate_complete,
        "non_ate_risk_gates": non_ate_risk_gates,
        "calibration_by_seq": manifest.get("calibration_by_seq", {}),
        "holdout_comparisons": holdout_comparisons,
        "four_sequence_comparisons": four_seq_comparisons,
        "four_sequence_ate": four_seq_ate,
        "boundary": (
            "R27 applies the frozen R26 AI4 polarity rule to fresh 01/05 holdout cues. The summary can validate "
            "ATE/control gates from existing LingBot evaluator outputs, but it does not fabricate rolling/final/local-window "
            "risk gates that are not emitted by the current evaluator."
        ),
        "outputs": {
            "rows": rel(out / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_rows.csv"),
            "combined_rows": rel(out / "stage4_r27_combined_r26_r27_ate_control_rows.csv"),
            "summary": rel(out / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_summary.json"),
            "report": rel(out / "STAGE4_R27_LINGBOT_AI_HOLDOUT_CALIBRATED_POLARITY_ANCHOR_REPORT.md"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_rows.csv", r27_rows)
    write_csv(out / "stage4_r27_combined_r26_r27_ate_control_rows.csv", combined_rows)
    (out / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# Stage4-R27 LingBot AI Holdout Calibrated Polarity Anchor Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"action_fidelity: `{action_fidelity}`",
        f"ate_control_holdout_pass: `{ate_control_holdout_pass}`",
        "",
        "```json",
        json.dumps(
            {
                "calibration_by_seq": summary["calibration_by_seq"],
                "four_sequence_ate": four_seq_ate,
                "holdout_comparisons": holdout_comparisons,
                "four_sequence_controls_not_matched_within_0p01": four_seq_controls_not_matched,
                "non_ate_risk_gates": non_ate_risk_gates,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
    ]
    (out / "STAGE4_R27_LINGBOT_AI_HOLDOUT_CALIBRATED_POLARITY_ANCHOR_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R27",
            "surface_or_branch": "LB-AI",
            "status": decision,
            "artifact": rel(out / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_summary.json"),
            "notes": "Frozen R26 AI4 calibrated-polarity rule applied to fresh 01/05 holdout; global success remains false unless all plan risk gates are available and pass",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
