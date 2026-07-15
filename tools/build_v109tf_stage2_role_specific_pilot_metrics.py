#!/usr/bin/env python3
"""Build v109TF Stage2 role-specific pilot metrics for KITTI 00/02."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402


base = stage2m.base

RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
CORE = RESULT_ROOT / "stage2_f_core_ablation"
OUT = RESULT_ROOT / "stage2_role_specific_pilot"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"
SEQUENCES = ("00", "02")

PILOT_MEDIAN_REL_MIN = 0.05
PILOT_MAX_HARM = 0.02
PILOT_F1_TOL = 0.005
SUMMARY_ROW_SCHEMA = "acl2_v109tf_stage2_role_specific_pilot_summary_row_v1"
SUMMARY_SCHEMA = "acl2_v109tf_stage2_role_specific_pilot_summary_v1"
SUMMARY_JSON = "role_specific_pilot_summary.json"
REPORT_MD = "role_specific_pilot_report.md"
REPORT_TITLE = "# ACL2 v109TF Stage2 Role-Specific Pilot Report"
SCOPE_NOTE = "00/02 role-specific pilot only; not a full KITTI success claim"
GATE_SCOPE = "KITTI 00/02 pilot only"
INTERPRETATION_TEXT = (
    "This is a 00/02 role-specific pilot. Passing it can only nominate a full-KITTI follow-up "
    "candidate; it is not a v109 semantic-aware success claim by itself."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_float(value: Any) -> float:
    return stage2m.safe_float(value)


def safe_rc(row: dict[str, str] | None) -> int:
    return stage2m.safe_rc(row)


def row_rel(row: dict[str, Any] | None) -> float:
    return stage2m.row_rel(row)


def role_phase_status_for(
    cfg: dict[str, str],
    latest: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v109tf_stage2_role_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v109tf_stage2_role_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def role_action_fidelity_row(
    cfg: dict[str, str],
    latest: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    action_rows = base.load_jsonl(action_file)
    expected = base.parse_indices(cfg.get("selected_global_frame_indices", ""))
    expected_field = cfg.get("expected_action_field", "")
    mode = cfg.get("stage2_action_mode") or cfg.get("stage4_action_mode", "")
    observed = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get(expected_field, False))
    }
    base_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get("base_is_keyframe", False))
    }
    final_keyframes = {
        int(float(row.get("sample_position", -1)))
        for row in action_rows
        if base.boolish(row.get("final_is_keyframe", False))
    }
    effective: set[int] = set()
    trace_error_rows = 0
    for row in action_rows:
        try:
            sample = int(float(row.get("sample_position", -1)))
        except ValueError:
            trace_error_rows += 1
            continue
        if mode == "anchor_special_only":
            if (
                base.boolish(row.get("forced_anchor_only", False))
                and base.boolish(row.get("forced_context_only", False))
                and base.boolish(row.get("context_only_append", False))
                and str(row.get("context_only_special_mode", "")) == "scale_only"
            ):
                effective.add(sample)
        elif base.boolish(row.get(expected_field, False)):
            effective.add(sample)

    missing = expected - observed
    unexpected = observed - expected
    ineffective = expected - effective
    action_fidelity_pass = (
        action_file.exists()
        and observed == expected
        and effective == expected
        and trace_error_rows == 0
    )
    run_name = f"kitti_lingbot_v109tf_stage2_role_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    return {
        "schema": "acl2_v109tf_stage2_role_action_fidelity_row_v1",
        "surface_id": cfg["surface_id"],
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage2_action_mode": mode,
        "stage4_action_mode": cfg.get("stage4_action_mode", mode),
        "expected_action_field": expected_field,
        "expected_action_frame_count": len(expected),
        "observed_action_frame_count": len(observed),
        "action_effective_frame_count": len(effective),
        "action_noop_frame_count": len(ineffective),
        "expected_keyframe_count": len(expected),
        "observed_keyframe_count": len(observed & base_keyframes),
        "special_token_operation_count": len(effective) if mode == "anchor_special_only" else "",
        "trace_error_rows": trace_error_rows,
        "action_file_exists": action_file.exists(),
        "action_fidelity_pass": action_fidelity_pass,
        "observed_action_indices": ";".join(str(x) for x in sorted(observed)),
        "effective_action_indices": ";".join(str(x) for x in sorted(effective)),
        "missing_expected_indices": ";".join(str(x) for x in sorted(missing)),
        "unexpected_observed_indices": ";".join(str(x) for x in sorted(unexpected)),
        "ineffective_expected_indices": ";".join(str(x) for x in sorted(ineffective)),
        "base_keyframe_count_observed_log": len(base_keyframes),
        "final_keyframe_count_observed_log": len(final_keyframes),
        "action_log_rows": len(action_rows),
        "action_file": rel(action_file),
        "run_worker_returncode": run_row.get("returncode", ""),
        "run_worker_duration_sec": run_row.get("duration_sec", ""),
    }


def install_role_overrides() -> None:
    stage2m.OUT = OUT
    stage2m.CONFIG_ROWS = CONFIG_ROWS
    stage2m.RUN_RESULTS = RUN_RESULTS
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = role_phase_status_for
    stage2m.action_fidelity_row = role_action_fidelity_row


def build_policy_selected_rows(fidelity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fidelity_by_key = {
        (row["policy_id"], row["seq"]): row
        for row in fidelity_rows
    }
    semantic_by_key = {
        (row["policy_id"], row["seq"], row["source_frame"]): row
        for row in read_csv(OUT / "role_source_frame_rows.csv")
    }
    rows: list[dict[str, Any]] = []
    for snap in read_csv(OUT / "role_keyframe_snap_rows.csv"):
        policy_id = snap["policy_id"]
        seq = snap["seq"]
        source = snap["source_frame"]
        snapped = snap["snapped_base_keyframe"]
        sem = semantic_by_key.get((policy_id, seq, source), {})
        fidelity = fidelity_by_key.get((policy_id, seq), {})
        effective = base.parse_indices(str(fidelity.get("effective_action_indices", "")))
        observed = base.parse_indices(str(fidelity.get("observed_action_indices", "")))
        snapped_int = int(float(snapped))
        rows.append(
            {
                "schema": "acl2_v109tf_stage2_role_policy_selected_frame_row_v1",
                "surface_id": snap["surface_id"],
                "policy_id": policy_id,
                "policy_family": snap["policy_family"],
                "seq": seq,
                "source_frame": source,
                "snapped_base_keyframe": snapped,
                "snap_distance": snap.get("distance", ""),
                "snap_accepted": snap.get("accepted", ""),
                "snapped_observed_action": snapped_int in observed,
                "snapped_effective_action": snapped_int in effective,
                "role_score": sem.get("role_score", ""),
                "stable_structure_mass": sem.get("stable_structure_mass", ""),
                "dynamic_mass": sem.get("dynamic_mass", ""),
                "boundary_mass": sem.get("boundary_mass", ""),
                "weak_context_mass": sem.get("weak_context_mass", ""),
                "road_ground_mass": sem.get("road_ground_mass", ""),
                "sky_lowobs_mass": sem.get("sky_lowobs_mass", ""),
                "semantic_trust_mean": sem.get("semantic_trust_mean", ""),
                "semantic_purity_mean": sem.get("semantic_purity_mean", ""),
                "semantic_continuity_score": sem.get("semantic_continuity_score", ""),
                "semantic_boundary_risk": sem.get("semantic_boundary_risk", ""),
                "special_token_count": sem.get("special_token_count", ""),
                "cache_append_count": sem.get("cache_append_count", ""),
                "trajectory_write_count": sem.get("trajectory_write_count", ""),
            }
        )
    return rows


def f1_core_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in read_csv(CORE / "full_metric_rows.csv"):
        if row.get("policy_id") == stage2m.SEMANTIC_PLUS and row.get("seq") in SEQUENCES:
            rows[row["seq"]] = row
    return rows


def role_summary_rows(
    full_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_policy[row["policy_id"]].append(row)
    for row in rolling_rows:
        rolling_by_policy[row["policy_id"]].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[row["policy_id"]].append(row)

    f1_by_seq = f1_core_rows()
    rows: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        policy_full = sorted(by_policy[policy_id], key=lambda row: row["seq"])
        rels = [row_rel(row) for row in policy_full]
        finals = [safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan")) for row in policy_full]
        locals_ = [safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan")) for row in policy_full]
        roll_rels = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        vs_f1: list[float] = []
        for row in policy_full:
            seq = row["seq"]
            role_ate = safe_float(row.get("full_ATE_sim3", "nan"))
            f1_ate = safe_float(f1_by_seq.get(seq, {}).get("full_ATE_sim3", "nan"))
            vs_f1.append(base.rel_improvement(f1_ate, role_ate))
        metric_available_count = sum(1 for row in policy_full if bool(row.get("metric_available")))
        action_fidelity_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool(row.get("action_fidelity_pass")))
        improved_count = sum(1 for value in rels if math.isfinite(value) and value > 0.0)
        pre_gate = bool(
            len(policy_full) == len(SEQUENCES)
            and metric_available_count == len(SEQUENCES)
            and action_fidelity_count == len(SEQUENCES)
            and base.median(rels) >= PILOT_MEDIAN_REL_MIN
            and improved_count == len(SEQUENCES)
            and base.max_rel_harm(rels) <= PILOT_MAX_HARM
            and base.max_rel_harm(locals_) <= PILOT_MAX_HARM
        )
        rows.append(
            {
                "schema": SUMMARY_ROW_SCHEMA,
                "row_type": "policy_summary",
                "surface_id": "F",
                "policy_id": policy_id,
                "policy_family": policy_full[0].get("policy_family", "") if policy_full else "",
                "sequence_count": len(policy_full),
                "metric_available_count": metric_available_count,
                "action_fidelity_pass_count": action_fidelity_count,
                "median_full_rel_improvement": base.median(rels),
                "mean_full_rel_improvement": base.mean(rels),
                "num_seq_improved": improved_count,
                "num_seq_worse": sum(1 for value in rels if math.isfinite(value) and value < 0.0),
                "max_harm": base.max_rel_harm(rels),
                "median_rolling_p90_rel_improvement": base.median(roll_rels),
                "median_final_error_rel_improvement": base.median(finals),
                "median_local_window_rel_improvement": base.median(locals_),
                "local_window_max_harm": base.max_rel_harm(locals_),
                "median_full_ATE_relative_improvement_vs_stage2_F1_same_seq": base.median(vs_f1),
                "mean_full_ATE_relative_improvement_vs_stage2_F1_same_seq": base.mean(vs_f1),
                "role_beats_stage2_F1_sequence_count": sum(1 for value in vs_f1 if math.isfinite(value) and value > 0.0),
                "role_within_0p005_of_stage2_F1_sequence_count": sum(
                    1 for value in vs_f1 if math.isfinite(value) and value >= -PILOT_F1_TOL
                ),
                "role_pilot_pre_gate_pass": pre_gate,
                "pilot_scope_note": SCOPE_NOTE,
            }
        )
    best = max(rows, key=lambda row: safe_float(row["median_full_rel_improvement"])) if rows else {}
    return rows, best


def taxonomy(metric_complete: bool, all_action_fidelity: bool, summary_rows: list[dict[str, Any]]) -> tuple[str, bool, str]:
    if not metric_complete:
        return "ROLE_PILOT_METRICS_NOT_COMPLETE", False, "role_pilot_metrics_not_complete"
    if not all_action_fidelity:
        return "ROLE_PILOT_ACTION_FIDELITY_FAIL", False, "role_pilot_action_fidelity_fail"
    pre_gate_rows = [row for row in summary_rows if bool(row.get("role_pilot_pre_gate_pass"))]
    if not pre_gate_rows:
        return "ROLE_PILOT_NO_ROLE_SURPASSES_GATE", False, "no_role_specific_policy_passed_00_02_pre_gate"
    best_vs_f1 = max(
        safe_float(row.get("median_full_ATE_relative_improvement_vs_stage2_F1_same_seq", "nan"))
        for row in pre_gate_rows
    )
    if math.isfinite(best_vs_f1) and best_vs_f1 >= -PILOT_F1_TOL:
        return "ROLE_PILOT_CANDIDATE_FOR_FULL_KITTI", True, ""
    return "ROLE_PILOT_BEATS_BASELINE_BUT_NOT_STAGE2_F1", False, "role_candidate_does_not_match_stage2_f1_on_00_02"


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        REPORT_TITLE,
        "",
        f"metric_complete: {summary['metric_complete']}",
        f"role_pilot_pre_gate_any_pass: {summary['role_pilot_pre_gate_any_pass']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        f"observed_run_workers: {summary['observed_run_worker_count']}/{summary['expected_run_worker_count']}",
        f"observed_evaluate_rows: {summary['observed_evaluate_count']}/{summary['expected_run_worker_count']}",
        f"observed_report_rows: {summary['observed_report_count']}/{summary['expected_run_worker_count']}",
        "",
        "## Policy Summary",
        "",
    ]
    for row in rows:
        lines.append(
            "- {policy_id}: median_full_rel={median_full_rel_improvement} "
            "improved={num_seq_improved}/{sequence_count} max_harm={max_harm} "
            "local_max_harm={local_window_max_harm} vs_F1_median={median_full_ATE_relative_improvement_vs_stage2_F1_same_seq} "
            "pre_gate={role_pilot_pre_gate_pass}".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            INTERPRETATION_TEXT,
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    install_role_overrides()
    config_rows = read_csv(CONFIG_ROWS)
    run_result_rows = read_csv(RUN_RESULTS)
    latest = stage2m.latest_run_results(run_result_rows)
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    selected_rows = build_policy_selected_rows(fidelity_rows)
    summary_rows, best_row = role_summary_rows(full_rows, rolling_rows, fidelity_rows)

    expected_run_workers = len(config_rows)
    run_worker_rows = stage2m.latest_phase_rows(latest, "run_worker")
    evaluate_rows = stage2m.latest_phase_rows(latest, "evaluate")
    report_rows = stage2m.latest_phase_rows(latest, "report")
    all_run_worker_success = (
        len(run_worker_rows) >= expected_run_workers
        and all(safe_rc(row) == 0 for row in run_worker_rows)
    )
    all_evaluate_success = (
        len(evaluate_rows) >= expected_run_workers
        and all(safe_rc(row) == 0 for row in evaluate_rows)
    )
    all_action_fidelity = (
        len(fidelity_rows) == expected_run_workers
        and all(bool(row["action_fidelity_pass"]) for row in fidelity_rows)
    )
    metric_complete = (
        len(full_rows) == expected_run_workers
        and all(bool(row["metric_available"]) for row in full_rows)
        and all(bool(row["all_metric_phase_success"]) for row in full_rows)
        and all_run_worker_success
        and all_evaluate_success
    )
    pre_gate_any = any(bool(row.get("role_pilot_pre_gate_pass")) for row in summary_rows)
    tax, pilot_pass, blocker = taxonomy(metric_complete, all_action_fidelity, summary_rows)

    write_csv(OUT / "no_action_control_rows.csv", stage2m.no_action_rows())
    write_csv(OUT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT / "full_metric_rows.csv", full_rows)
    write_csv(OUT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(OUT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(OUT / "policy_selected_frame_rows.csv", selected_rows)
    write_csv(OUT / "role_specific_summary_rows.csv", summary_rows)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "role_pilot_pass": pilot_pass,
        "role_pilot_pre_gate_any_pass": pre_gate_any,
        "metric_complete": metric_complete,
        "taxonomy": tax,
        "blocker": blocker,
        "expected_run_worker_count": expected_run_workers,
        "observed_run_worker_count": len(run_worker_rows),
        "observed_evaluate_count": len(evaluate_rows),
        "observed_report_count": len(report_rows),
        "all_run_worker_success": all_run_worker_success,
        "all_evaluate_success": all_evaluate_success,
        "all_action_fidelity": all_action_fidelity,
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_selected_frame_row_count": len(selected_rows),
        "best_policy_by_median_full_rel": best_row,
        "gate_definition": {
            "scope": GATE_SCOPE,
            "median_full_ATE_relative_improvement_min": PILOT_MEDIAN_REL_MIN,
            "improved_sequence_count_required": len(SEQUENCES),
            "max_sequence_full_ATE_harm": PILOT_MAX_HARM,
            "local_window_ATE_median_max_harm": PILOT_MAX_HARM,
            "stage2_F1_same_seq_tolerance": PILOT_F1_TOL,
        },
        "outputs": {
            "no_action_control_rows": rel(OUT / "no_action_control_rows.csv"),
            "action_fidelity_rows": rel(OUT / "action_fidelity_rows.csv"),
            "full_metric_rows": rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(OUT / "local_handoff_metric_rows.csv"),
            "policy_selected_frame_rows": rel(OUT / "policy_selected_frame_rows.csv"),
            "role_specific_summary_rows": rel(OUT / "role_specific_summary_rows.csv"),
            "role_specific_summary_json": rel(OUT / SUMMARY_JSON),
            "role_specific_report_md": rel(OUT / REPORT_MD),
        },
    }
    write_json(OUT / SUMMARY_JSON, summary)
    write_text(OUT / REPORT_MD, build_report(summary, summary_rows))
    return summary


def main() -> None:
    print(json.dumps(base.clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
