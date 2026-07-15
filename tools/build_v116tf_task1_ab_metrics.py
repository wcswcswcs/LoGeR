#!/usr/bin/env python3
"""Build ACL2 v116-TF Task1 A1+B1 00/02 pilot metrics."""

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

RESULT_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"
OUT = RESULT_ROOT / "task1_ab"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"
SEQUENCES = ("00", "02")

B1_MEDIAN_FULL_REL = 0.17413068803456322
B1_TOL = 0.02
PILOT_PROMOTION_GATE = 0.03
MAX_HARM_GATE = 0.01


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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
    path.write_text(json.dumps(base.clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    return stage2m.safe_float(value)


def safe_rc(row: dict[str, str] | None) -> int:
    return stage2m.safe_rc(row)


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_indices(raw: str) -> set[int]:
    return {int(float(x)) for x in str(raw).replace(",", ";").split(";") if x.strip()}


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v116tf_task1_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v116tf_task1_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def action_fidelity_row(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    action_file = Path(cfg["action_file"])
    action_rows = base.load_jsonl(action_file)
    expected_scale = parse_indices(cfg.get("scale_frame_indices", ""))
    expected_b1 = parse_indices(cfg.get("b1_force_non_keyframe_indices", ""))
    observed_scale = set()
    observed_b1 = set()
    effective_b1 = set()
    scope_leakage = False
    for row in action_rows:
        try:
            sample = int(float(row.get("sample_position", -1)))
        except (TypeError, ValueError):
            continue
        if boolish(row.get("anchor_scale_frame", False)):
            observed_scale.add(sample)
        if boolish(row.get("forced_non_keyframe", False)):
            observed_b1.add(sample)
            if boolish(row.get("skip_append", False)) and not boolish(row.get("final_is_keyframe", True)):
                effective_b1.add(sample)
        if expected_scale and boolish(row.get("forced_context_only", False)):
            scope_leakage = True
    scale_pass = observed_scale == expected_scale
    b1_pass = observed_b1 == expected_b1 and effective_b1 == expected_b1
    action_fidelity_pass = action_file.exists() and scale_pass and b1_pass and not scope_leakage
    run_name = f"kitti_lingbot_v116tf_task1_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    return {
        "schema": "acl2_v116tf_task1_action_fidelity_row_v1",
        "task": "Task1_AB",
        "candidate_id": cfg.get("candidate_id", ""),
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "action_name": cfg["action_name"],
        "stage4_action_mode": cfg.get("stage4_action_mode", ""),
        "target_frame_count": len(expected_scale),
        "effective_frame_count": len(observed_scale),
        "b1_target_frame_count": len(expected_b1),
        "b1_effective_frame_count": len(effective_b1),
        "expected_scale_indices": ";".join(str(x) for x in sorted(expected_scale)),
        "observed_scale_indices": ";".join(str(x) for x in sorted(observed_scale)),
        "expected_b1_indices": ";".join(str(x) for x in sorted(expected_b1)),
        "observed_b1_indices": ";".join(str(x) for x in sorted(observed_b1)),
        "effective_b1_indices": ";".join(str(x) for x in sorted(effective_b1)),
        "scale_action_fidelity_pass": scale_pass,
        "b1_action_fidelity_pass": b1_pass,
        "scope_leakage": scope_leakage,
        "action_fidelity_pass": action_fidelity_pass,
        "action_log_rows": len(action_rows),
        "action_file": rel(action_file),
        "run_worker_returncode": run_row.get("returncode", ""),
        "run_worker_duration_sec": run_row.get("duration_sec", ""),
    }


def install_metric_overrides() -> None:
    stage2m.OUT = OUT
    stage2m.CONFIG_ROWS = CONFIG_ROWS
    stage2m.RUN_RESULTS = RUN_RESULTS
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = phase_status_for
    stage2m.action_fidelity_row = action_fidelity_row


def augment_rows(rows: list[dict[str, Any]], config_rows: list[dict[str, str]]) -> None:
    cfg_by_key = {(row["policy_id"], row["seq"]): row for row in config_rows}
    for row in rows:
        row["schema"] = str(row.get("schema", "")).replace("acl2_v109tf_stage2", "acl2_v116tf_task1")
        cfg = cfg_by_key.get((str(row.get("policy_id", "")), str(row.get("seq", ""))), {})
        for key in (
            "candidate_id",
            "M",
            "a1_mode",
            "num_anchor",
            "scale_frame_indices",
            "b1_force_non_keyframe_indices",
            "b1_expected_count",
        ):
            row[key] = cfg.get(key, "")


def policy_summary_rows(
    full_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    fidelity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        full_by_policy[str(row["policy_id"])].append(row)
    for row in rolling_rows:
        rolling_by_policy[str(row["policy_id"])].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[str(row["policy_id"])].append(row)

    out: list[dict[str, Any]] = []
    for policy_id, rows in sorted(full_by_policy.items()):
        rels = [safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan")) for row in rows]
        rolling = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        fids = fidelity_by_policy.get(policy_id, [])
        median_full = base.median(rels)
        max_harm = base.max_rel_harm(rels)
        action_pass_all = len(fids) == len(SEQUENCES) and all(boolish(row.get("action_fidelity_pass")) for row in fids)
        promotion_gate = median_full >= PILOT_PROMOTION_GATE and max_harm <= MAX_HARM_GATE and action_pass_all
        b1_keep_gate = median_full >= (B1_MEDIAN_FULL_REL - B1_TOL)
        out.append(
            {
                "schema": "acl2_v116tf_task1_policy_summary_row_v1",
                "task": "Task1_AB",
                "candidate_id": rows[0].get("candidate_id", ""),
                "policy_id": policy_id,
                "policy_family": rows[0].get("policy_family", ""),
                "M": rows[0].get("M", ""),
                "a1_mode": rows[0].get("a1_mode", ""),
                "sequence_count": len(rows),
                "seq_full_rel": json.dumps({row.get("seq", ""): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan")) for row in rows}, sort_keys=True),
                "median_full_rel": median_full,
                "mean_full_rel": base.mean(rels),
                "min_seq_full_rel": min([v for v in rels if math.isfinite(v)], default=float("nan")),
                "improved_seq_count": sum(1 for v in rels if math.isfinite(v) and v > 0),
                "max_harm": max_harm,
                "rolling_p90_median_rel": base.median(rolling),
                "action_fidelity_pass_all": action_pass_all,
                "pilot_promotion_gate_pass": promotion_gate,
                "b1_keep_gate_pass": b1_keep_gate,
                "median_minus_b1_reference": median_full - B1_MEDIAN_FULL_REL if math.isfinite(median_full) else float("nan"),
            }
        )
    return out


def control_comparison_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy = {row["policy_id"]: row for row in policy_rows}
    b1 = by_policy.get("AB0_B1_semantic_only_reference", {})
    default = by_policy.get("AB_CTRL_A1_default_first8_plus_B1", {})
    out: list[dict[str, Any]] = []
    for row in policy_rows:
        if row["policy_id"] in {"AB0_B1_semantic_only_reference", "AB_CTRL_A1_default_first8_plus_B1"}:
            continue
        median = safe_float(row.get("median_full_rel", "nan"))
        b1_med = safe_float(b1.get("median_full_rel", "nan"))
        default_med = safe_float(default.get("median_full_rel", "nan"))
        out.append(
            {
                "schema": "acl2_v116tf_task1_control_comparison_row_v1",
                "candidate_id": row.get("candidate_id", ""),
                "policy_id": row.get("policy_id", ""),
                "candidate_median": median,
                "b1_reference_median": b1_med,
                "a1_default_plus_b1_median": default_med,
                "candidate_minus_b1": median - b1_med if math.isfinite(median) and math.isfinite(b1_med) else float("nan"),
                "candidate_minus_a1_default_plus_b1": (
                    median - default_med if math.isfinite(median) and math.isfinite(default_med) else float("nan")
                ),
                "matched_controls_status": "not_run_yet_controls_required_if_candidate_ge_3pct",
            }
        )
    return out


def build_report(summary: dict[str, Any], policies: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v116-TF Task1 A1+B1 00/02 Pilot Report",
        "",
        f"task_status: `{summary['task_status']}`",
        f"failure_type: `{summary['failure_type']}`",
        f"primary_blocker: `{summary['primary_blocker']}`",
        f"metric_complete: `{summary['metric_complete']}`",
        f"all_action_fidelity: `{summary['all_action_fidelity']}`",
        "",
        "## Candidate Geometry",
        "",
        "| policy_id | median_full_rel | max_harm | rolling_p90_median_rel | action_fidelity | promotion_gate | b1_keep_gate |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in sorted(policies, key=lambda item: safe_float(item.get("median_full_rel", "nan")), reverse=True):
        lines.append(
            f"| {row.get('policy_id','')} | {row.get('median_full_rel','')} | {row.get('max_harm','')} | "
            f"{row.get('rolling_p90_median_rel','')} | {row.get('action_fidelity_pass_all','')} | "
            f"{row.get('pilot_promotion_gate_pass','')} | {row.get('b1_keep_gate_pass','')} |"
        )
    lines += [
        "",
        "## B1 Reference Comparisons",
        "",
        "| policy_id | candidate_minus_b1 | candidate_minus_a1_default_plus_b1 | matched_controls_status |",
        "|---|---:|---:|---|",
    ]
    for row in controls:
        lines.append(
            f"| {row.get('policy_id','')} | {row.get('candidate_minus_b1','')} | "
            f"{row.get('candidate_minus_a1_default_plus_b1','')} | {row.get('matched_controls_status','')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This report is a 00/02 pilot only. Semantic causality controls are not complete unless TASK1_CONTROL_COMPARISON contains matched control metrics beyond B1/default composition.",
    ]
    return "\n".join(lines)


def main() -> int:
    install_metric_overrides()
    config_rows = read_csv(CONFIG_ROWS)
    run_rows = read_csv(RUN_RESULTS)
    latest = stage2m.latest_run_results(run_rows)
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        augment_rows(rows, config_rows)
    policies = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    controls = control_comparison_rows(policies)

    expected_run_worker = len(config_rows)
    observed_prepare = sum(1 for row in latest.values() if row.get("phase") == "prepare" and safe_rc(row) == 0)
    observed_run_worker = sum(1 for row in latest.values() if row.get("phase") == "run_worker" and safe_rc(row) == 0)
    observed_evaluate = sum(1 for row in latest.values() if row.get("phase") == "evaluate" and safe_rc(row) == 0)
    observed_report = sum(1 for row in latest.values() if row.get("phase") == "report" and safe_rc(row) == 0)
    metric_complete = (
        len(full_rows) == expected_run_worker
        and observed_prepare >= len(SEQUENCES)
        and observed_run_worker >= expected_run_worker
        and observed_evaluate >= expected_run_worker
        and all(boolish(row.get("metric_available")) for row in full_rows)
    )
    all_action_fidelity = len(fidelity_rows) == expected_run_worker and all(
        boolish(row.get("action_fidelity_pass")) for row in fidelity_rows
    )
    promising = [row for row in policies if boolish(row.get("pilot_promotion_gate_pass")) and row["policy_id"] != "AB0_B1_semantic_only_reference"]
    best = max(policies, key=lambda row: safe_float(row.get("median_full_rel", "nan")), default={})

    if not metric_complete:
        task_status = "INCOMPLETE"
        failure_type = "RUN_OR_METRIC_INCOMPLETE"
        primary_blocker = "not_all_task1_manifest_rows_have_successful_prepare_run_worker_evaluate_metrics"
    elif not all_action_fidelity:
        task_status = "NO_GO_ACTION_FIDELITY"
        failure_type = "ACTION_FIDELITY_FAIL"
        primary_blocker = "A1 scale frame or B1 no-append action rows did not match expected indices"
    elif promising:
        task_status = "GEOMETRY_PROMISING_CONTROLS_REQUIRED"
        failure_type = "GEOMETRY_PROMISING"
        primary_blocker = "matched semantic controls not yet run"
    else:
        best_rel = safe_float(best.get("median_full_rel", "nan"))
        if math.isfinite(best_rel) and best_rel < 0.01:
            failure_type = "GEOMETRY_WEAK"
        elif math.isfinite(best_rel) and best_rel < PILOT_PROMOTION_GATE:
            failure_type = "GEOMETRY_DIAGNOSTIC_ONLY"
        else:
            failure_type = "GEOMETRY_BELOW_PROMOTION_OR_B1_KEEP_GATE"
        task_status = "PILOT_NO_PROMOTION"
        primary_blocker = "no_AB_candidate_met_promotion_gate"

    summary = {
        "schema": "acl2_v116tf_task1_decision_summary_v1",
        "task": "Task1_AB",
        "task_status": task_status,
        "failure_type": failure_type,
        "primary_blocker": primary_blocker,
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action_fidelity,
        "expected_prepare_count": len(SEQUENCES),
        "observed_prepare_count": observed_prepare,
        "expected_run_worker_count": expected_run_worker,
        "observed_run_worker_count": observed_run_worker,
        "observed_evaluate_count": observed_evaluate,
        "observed_report_count": observed_report,
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "best_policy_id": best.get("policy_id", ""),
        "best_median_full_rel": best.get("median_full_rel", ""),
        "promising_policy_ids": [row["policy_id"] for row in promising],
        "next_allowed_attempt": "run matched AB controls if promising else inspect action scope and stop/one mechanism repair per plan",
        "next_disallowed_attempt": "claim semantic causality without A1 random/B1 shuffle/same-count controls",
        "whether_branch_stops": task_status == "PILOT_NO_PROMOTION" and failure_type == "GEOMETRY_WEAK",
        "outputs": {
            "run_manifest": rel(OUT / "TASK1_RUN_MANIFEST.csv"),
            "geometry_metrics": rel(OUT / "TASK1_GEOMETRY_METRICS.csv"),
            "action_fidelity": rel(OUT / "TASK1_ACTION_FIDELITY.csv"),
            "control_comparison": rel(OUT / "TASK1_CONTROL_COMPARISON.csv"),
            "decision_summary": rel(OUT / "TASK1_DECISION_SUMMARY.json"),
            "report": rel(OUT / "TASK1_REPORT.md"),
        },
    }

    write_csv(OUT / "TASK1_GEOMETRY_METRICS.csv", full_rows)
    write_csv(OUT / "TASK1_ROLLING_METRICS.csv", rolling_rows)
    write_csv(OUT / "TASK1_LOCAL_HANDOFF_METRICS.csv", local_rows)
    write_csv(OUT / "TASK1_ACTION_FIDELITY.csv", fidelity_rows)
    write_csv(OUT / "TASK1_POLICY_SUMMARY.csv", policies)
    write_csv(OUT / "TASK1_CONTROL_COMPARISON.csv", controls)
    write_json(OUT / "TASK1_DECISION_SUMMARY.json", summary)
    write_text(OUT / "TASK1_REPORT.md", build_report(summary, policies, controls))
    print(json.dumps(base.clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
