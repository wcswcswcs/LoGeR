#!/usr/bin/env python3
"""Summarize ACL2 v116-TF Task1 matched controls."""

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
OUT = RESULT_ROOT / "task1_ab_controls"
MAIN = RESULT_ROOT / "task1_ab"
CONFIG_ROWS = OUT / "action_config_rows.csv"
RUN_RESULTS = OUT / "run_results.csv"
WORKSPACE = OUT / "workspace"
SEQUENCES = ("00", "02")
CAUSAL_MARGIN = 0.02


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
            run_name = f"kitti_lingbot_v116tf_task1_control_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v116tf_task1_control_{cfg['policy_id']}_{seq}_{phase}"
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
    observed_scale: set[int] = set()
    observed_b1: set[int] = set()
    effective_b1: set[int] = set()
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
    run_name = f"kitti_lingbot_v116tf_task1_control_{cfg['policy_id']}_{cfg['seq']}_run_worker"
    run_row = latest.get((run_name, "run_worker"), {})
    return {
        "schema": "acl2_v116tf_task1_control_action_fidelity_row_v1",
        "task": "Task1_AB_controls",
        "policy_id": cfg["policy_id"],
        "policy_family": cfg["policy_family"],
        "control_component": cfg.get("control_component", ""),
        "control_mode": cfg.get("control_mode", ""),
        "seq": cfg["seq"],
        "dataset": cfg["dataset"],
        "method": cfg["method"],
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
        row["schema"] = str(row.get("schema", "")).replace("acl2_v109tf_stage2", "acl2_v116tf_task1_control")
        cfg = cfg_by_key.get((str(row.get("policy_id", "")), str(row.get("seq", ""))), {})
        for key in (
            "control_component",
            "control_mode",
            "M",
            "num_anchor",
            "scale_frame_indices",
            "b1_force_non_keyframe_indices",
            "b1_expected_count",
            "b1_reference_count",
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
        out.append(
            {
                "schema": "acl2_v116tf_task1_control_policy_summary_row_v1",
                "task": "Task1_AB_controls",
                "policy_id": policy_id,
                "policy_family": rows[0].get("policy_family", ""),
                "control_component": rows[0].get("control_component", ""),
                "control_mode": rows[0].get("control_mode", ""),
                "M": rows[0].get("M", ""),
                "sequence_count": len(rows),
                "seq_full_rel": json.dumps({row.get("seq", ""): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan")) for row in rows}, sort_keys=True),
                "median_full_rel": base.median(rels),
                "mean_full_rel": base.mean(rels),
                "min_seq_full_rel": min([v for v in rels if math.isfinite(v)], default=float("nan")),
                "improved_seq_count": sum(1 for v in rels if math.isfinite(v) and v > 0),
                "max_harm": base.max_rel_harm(rels),
                "rolling_p90_median_rel": base.median(rolling),
                "action_fidelity_pass_all": len(fids) == len(SEQUENCES) and all(boolish(row.get("action_fidelity_pass")) for row in fids),
            }
        )
    return out


def best_row(rows: list[dict[str, Any]], *, component: str | None = None, m: str | None = None) -> dict[str, Any]:
    candidates = rows
    if component is not None:
        candidates = [row for row in candidates if row.get("control_component") == component]
    if m is not None:
        candidates = [row for row in candidates if str(row.get("M", "")) == str(m)]
    return max(candidates, key=lambda row: safe_float(row.get("median_full_rel", "nan")), default={})


def comparison_rows(main_rows: list[dict[str, str]], control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_main = {row["policy_id"]: row for row in main_rows}
    b1 = by_main.get("AB0_B1_semantic_only_reference", {})
    out: list[dict[str, Any]] = []
    b1_med = safe_float(b1.get("median_full_rel", "nan"))
    for row in control_rows:
        med = safe_float(row.get("median_full_rel", "nan"))
        if row.get("control_component") == "B1":
            out.append(
                {
                    "schema": "acl2_v116tf_task1_control_comparison_row_v1",
                    "comparison_family": "B1_semantic_vs_B1_control",
                    "control_policy_id": row.get("policy_id", ""),
                    "control_median_full_rel": med,
                    "reference_policy_id": "AB0_B1_semantic_only_reference",
                    "reference_median_full_rel": b1_med,
                    "reference_minus_control": b1_med - med if math.isfinite(b1_med) and math.isfinite(med) else float("nan"),
                    "causal_margin_required": CAUSAL_MARGIN,
                }
            )
    for m in ("16", "24", "32"):
        semantic = [
            row for row in main_rows
            if str(row.get("M", "")) == m and row.get("policy_id", "").startswith("AB") and "CTRL" not in row.get("policy_id", "")
        ]
        best_sem = max(semantic, key=lambda row: safe_float(row.get("median_full_rel", "nan")), default={})
        best_ctl = best_row(control_rows, component="A1", m=m)
        sem_med = safe_float(best_sem.get("median_full_rel", "nan"))
        ctl_med = safe_float(best_ctl.get("median_full_rel", "nan"))
        out.append(
            {
                "schema": "acl2_v116tf_task1_control_comparison_row_v1",
                "comparison_family": "A1_semantic_best_vs_A1_control_best",
                "M": m,
                "semantic_policy_id": best_sem.get("policy_id", ""),
                "semantic_median_full_rel": sem_med,
                "control_policy_id": best_ctl.get("policy_id", ""),
                "control_median_full_rel": ctl_med,
                "semantic_minus_control": sem_med - ctl_med if math.isfinite(sem_med) and math.isfinite(ctl_med) else float("nan"),
                "semantic_minus_b1_reference": sem_med - b1_med if math.isfinite(sem_med) and math.isfinite(b1_med) else float("nan"),
                "causal_margin_required": CAUSAL_MARGIN,
            }
        )
    return out


def decision_summary(
    main_rows: list[dict[str, str]],
    control_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    metric_complete: bool,
    all_action_fidelity: bool,
    counts: dict[str, int],
) -> dict[str, Any]:
    by_main = {row["policy_id"]: row for row in main_rows}
    b1_med = safe_float(by_main.get("AB0_B1_semantic_only_reference", {}).get("median_full_rel", "nan"))
    b1_controls = [row for row in control_rows if row.get("control_component") == "B1"]
    best_b1_control = max(b1_controls, key=lambda row: safe_float(row.get("median_full_rel", "nan")), default={})
    best_b1_control_med = safe_float(best_b1_control.get("median_full_rel", "nan"))
    b1_margin = b1_med - best_b1_control_med if math.isfinite(b1_med) and math.isfinite(best_b1_control_med) else float("nan")
    a1_comparisons = [row for row in comparisons if row.get("comparison_family") == "A1_semantic_best_vs_A1_control_best"]
    a1_margin_min = min(
        [safe_float(row.get("semantic_minus_control", "nan")) for row in a1_comparisons if math.isfinite(safe_float(row.get("semantic_minus_control", "nan")))],
        default=float("nan"),
    )
    a1_incremental_max = max(
        [safe_float(row.get("semantic_minus_b1_reference", "nan")) for row in a1_comparisons if math.isfinite(safe_float(row.get("semantic_minus_b1_reference", "nan")))],
        default=float("nan"),
    )
    b1_control_margin_pass = math.isfinite(b1_margin) and b1_margin >= CAUSAL_MARGIN
    a1_control_margin_pass = math.isfinite(a1_margin_min) and a1_margin_min >= CAUSAL_MARGIN
    a1_incremental_pass = math.isfinite(a1_incremental_max) and a1_incremental_max >= CAUSAL_MARGIN

    if not metric_complete:
        task_status = "CONTROLS_INCOMPLETE"
        blocker = "not_all_control_manifest_rows_have_successful_prepare_run_worker_evaluate_metrics"
    elif not all_action_fidelity:
        task_status = "NO_GO_CONTROL_ACTION_FIDELITY"
        blocker = "control action rows did not match expected scale/B1 forced indices"
    elif not b1_control_margin_pass:
        task_status = "NO_GO_B1_CONTROL_MATCHES"
        blocker = "B1 matched control matches or exceeds semantic-only within causal margin"
    elif not a1_incremental_pass:
        task_status = "NO_GO_A1_NO_INCREMENTAL_BENEFIT"
        blocker = "A1 semantic selectors do not improve over B1 reference by causal margin"
    elif not a1_control_margin_pass:
        task_status = "NO_GO_A1_CONTROL_MATCHES"
        blocker = "A1 random/shuffle/role controls match semantic A1 selectors within causal margin"
    else:
        task_status = "TASK1_AB_CONTROLS_PASS"
        blocker = ""

    return {
        "schema": "acl2_v116tf_task1_control_decision_summary_v1",
        "task": "Task1_AB_controls",
        "task_status": task_status,
        "blocker": blocker,
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action_fidelity,
        "expected_prepare_count": len(SEQUENCES),
        **counts,
        "b1_reference_median_full_rel": b1_med,
        "best_b1_control_policy_id": best_b1_control.get("policy_id", ""),
        "best_b1_control_median_full_rel": best_b1_control_med,
        "b1_semantic_minus_best_control": b1_margin,
        "b1_control_margin_pass": b1_control_margin_pass,
        "a1_semantic_minus_best_control_min": a1_margin_min,
        "a1_control_margin_pass": a1_control_margin_pass,
        "a1_semantic_minus_b1_reference_max": a1_incremental_max,
        "a1_incremental_pass": a1_incremental_pass,
        "causal_margin_required": CAUSAL_MARGIN,
        "outputs": {
            "control_policy_summary": rel(OUT / "TASK1_CONTROL_POLICY_SUMMARY.csv"),
            "control_comparison": rel(OUT / "TASK1_CONTROL_COMPARISON.csv"),
            "decision_summary": rel(OUT / "TASK1_CONTROL_DECISION_SUMMARY.json"),
            "report": rel(OUT / "TASK1_CONTROL_REPORT.md"),
        },
    }


def build_report(summary: dict[str, Any], controls: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v116-TF Task1 Matched Control Report",
        "",
        f"task_status: `{summary['task_status']}`",
        f"blocker: `{summary['blocker']}`",
        f"metric_complete: `{summary['metric_complete']}`",
        f"all_action_fidelity: `{summary['all_action_fidelity']}`",
        "",
        "## Control Ranking",
        "",
        "| policy_id | component | mode | M | median_full_rel | action_fidelity |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in sorted(controls, key=lambda item: safe_float(item.get("median_full_rel", "nan")), reverse=True):
        lines.append(
            f"| {row.get('policy_id','')} | {row.get('control_component','')} | {row.get('control_mode','')} | "
            f"{row.get('M','')} | {row.get('median_full_rel','')} | {row.get('action_fidelity_pass_all','')} |"
        )
    lines += [
        "",
        "## Comparisons",
        "",
        "| comparison_family | reference/semantic | control | delta | margin_required |",
        "|---|---|---|---:|---:|",
    ]
    for row in comparisons:
        if row.get("comparison_family") == "B1_semantic_vs_B1_control":
            lines.append(
                f"| {row.get('comparison_family','')} | {row.get('reference_policy_id','')} | "
                f"{row.get('control_policy_id','')} | {row.get('reference_minus_control','')} | {row.get('causal_margin_required','')} |"
            )
        else:
            lines.append(
                f"| {row.get('comparison_family','')} M={row.get('M','')} | {row.get('semantic_policy_id','')} | "
                f"{row.get('control_policy_id','')} | {row.get('semantic_minus_control','')} | {row.get('causal_margin_required','')} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "This report completes the Task1 matched-control gate only when all control metrics and action fidelity pass. "
        "A semantic claim also requires the B1 semantic reference and A1 semantic selectors to beat their matched controls by the registered margin.",
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
    controls = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    main_rows = read_csv(MAIN / "TASK1_POLICY_SUMMARY.csv")
    comparisons = comparison_rows(main_rows, controls)

    expected_run_worker = len(config_rows)
    counts = {
        "observed_prepare_count": sum(1 for row in latest.values() if row.get("phase") == "prepare" and safe_rc(row) == 0),
        "expected_run_worker_count": expected_run_worker,
        "observed_run_worker_count": sum(1 for row in latest.values() if row.get("phase") == "run_worker" and safe_rc(row) == 0),
        "observed_evaluate_count": sum(1 for row in latest.values() if row.get("phase") == "evaluate" and safe_rc(row) == 0),
        "observed_report_count": sum(1 for row in latest.values() if row.get("phase") == "report" and safe_rc(row) == 0),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
    }
    metric_complete = (
        len(full_rows) == expected_run_worker
        and counts["observed_prepare_count"] >= len(SEQUENCES)
        and counts["observed_run_worker_count"] >= expected_run_worker
        and counts["observed_evaluate_count"] >= expected_run_worker
        and all(boolish(row.get("metric_available")) for row in full_rows)
    )
    all_action_fidelity = len(fidelity_rows) == expected_run_worker and all(
        boolish(row.get("action_fidelity_pass")) for row in fidelity_rows
    )
    summary = decision_summary(main_rows, controls, comparisons, metric_complete, all_action_fidelity, counts)

    write_csv(OUT / "TASK1_CONTROL_GEOMETRY_METRICS.csv", full_rows)
    write_csv(OUT / "TASK1_CONTROL_ROLLING_METRICS.csv", rolling_rows)
    write_csv(OUT / "TASK1_CONTROL_LOCAL_HANDOFF_METRICS.csv", local_rows)
    write_csv(OUT / "TASK1_CONTROL_ACTION_FIDELITY.csv", fidelity_rows)
    write_csv(OUT / "TASK1_CONTROL_POLICY_SUMMARY.csv", controls)
    write_csv(OUT / "TASK1_CONTROL_COMPARISON.csv", comparisons)
    write_json(OUT / "TASK1_CONTROL_DECISION_SUMMARY.json", summary)
    write_text(OUT / "TASK1_CONTROL_REPORT.md", build_report(summary, controls, comparisons))
    print(json.dumps(base.clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
