#!/usr/bin/env python3
"""Summarize ACL2 v112TF H1/T4 semantic lifetime pilot/full metrics."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402
import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
MODE = os.environ.get("ACL2_V112TF_H1_T4_MODE", "pilot").strip().lower()
if MODE not in {"pilot", "full"}:
    raise ValueError("ACL2_V112TF_H1_T4_MODE must be 'pilot' or 'full'")
RUN_LABEL = "full" if MODE == "full" else "pilot"
PILOT = RESULT_ROOT / (
    "stage6_h1_t4_full_validation_00_01_02_05"
    if MODE == "full"
    else "stage5_h1_t4_semantic_lifetime_pilot_00_02"
)
CONFIG_ROWS = PILOT / "action_config_rows.csv"
RUN_RESULTS = PILOT / "run_results.csv"
WORKSPACE = PILOT / "workspace"
REPORT_NAME = "H1_T4_SEMANTIC_LIFETIME_FULL_REPORT.md" if MODE == "full" else "H1_T4_SEMANTIC_LIFETIME_PILOT_REPORT.md"
SEQUENCES = ("00", "01", "02", "05") if MODE == "full" else ("00", "02")
B1_SAME_SEQ = {
    "00": 0.3983909144016795,
    "01": 0.0035407050871187795,
    "02": 0.21012151584586794,
    "05": 0.13813986022325847,
}


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
    path.write_text(json.dumps(stage3m.base.clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def phase_status_for(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> tuple[dict[str, Any], bool, bool]:
    status: dict[str, Any] = {}
    all_metric_phase_success = True
    all_phase_success = True
    seq = cfg["seq"]
    for phase in ("prepare", "run_worker", "evaluate", "report"):
        if phase == "prepare":
            run_name = f"kitti_lingbot_v112tf_h1_t4_prepare_{seq}"
        else:
            run_name = f"kitti_lingbot_v112tf_h1_t4_{cfg['policy_id']}_{seq}_{phase}"
        row = latest.get((run_name, phase))
        rc = stage2m.safe_rc(row)
        status[f"{phase}_returncode"] = rc
        status[f"{phase}_duration_sec"] = row.get("duration_sec", "") if row else ""
        if phase in {"prepare", "run_worker", "evaluate"}:
            all_metric_phase_success = all_metric_phase_success and rc == 0
        all_phase_success = all_phase_success and rc == 0
    return status, all_metric_phase_success, all_phase_success


def install_overrides() -> None:
    stage2m.OUT = PILOT
    stage2m.CONFIG_ROWS = CONFIG_ROWS
    stage2m.RUN_RESULTS = RUN_RESULTS
    stage2m.WORKSPACE = WORKSPACE
    stage2m.SEQUENCES = SEQUENCES
    stage2m.phase_status_for = phase_status_for
    original_action_fidelity = stage2m.action_fidelity_row

    def action_fidelity(cfg: dict[str, str], latest: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
        row = original_action_fidelity(cfg, latest)
        run_name = f"kitti_lingbot_v112tf_h1_t4_{cfg['policy_id']}_{cfg['seq']}_run_worker"
        run_row = latest.get((run_name, "run_worker"), {})
        row["schema"] = "acl2_v112tf_h1_t4_action_fidelity_row_v1"
        row["run_worker_returncode"] = run_row.get("returncode", "")
        row["run_worker_duration_sec"] = run_row.get("duration_sec", "")
        row["mask_mode"] = cfg.get("mask_mode", "")
        row["camera_token_weight_mean"] = cfg.get("camera_token_weight_mean", "")
        row["register_token_weight_mean"] = cfg.get("register_token_weight_mean", "")
        row["anchor_token_weight_mean"] = cfg.get("anchor_token_weight_mean", "")
        return row

    stage2m.action_fidelity_row = action_fidelity


def policy_summary_rows(full_rows: list[dict[str, Any]], rolling_rows: list[dict[str, Any]], fidelity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rolling_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fidelity_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_policy[str(row["policy_id"])].append(row)
    for row in rolling_rows:
        rolling_by_policy[str(row["policy_id"])].append(row)
    for row in fidelity_rows:
        fidelity_by_policy[str(row["policy_id"])].append(row)
    out: list[dict[str, Any]] = []
    for policy_id in sorted(by_policy):
        rows = by_policy[policy_id]
        rels_by_seq = {
            str(row.get("seq", "")): safe_float(row.get("full_ATE_sim3_relative_improvement_vs_baseline", "nan"))
            for row in rows
        }
        rels = [rels_by_seq.get(seq, float("nan")) for seq in SEQUENCES]
        rolling = [
            safe_float(row.get("rolling_ATE_p90_relative_improvement_vs_baseline", "nan"))
            for row in rolling_by_policy.get(policy_id, [])
        ]
        local = [
            safe_float(row.get("local_window_ATE_rel_improvement_vs_baseline_median", "nan"))
            for row in rows
        ]
        final_errors = [
            safe_float(row.get("final_error_relative_improvement_vs_baseline", "nan"))
            for row in rows
        ]
        median_full = stage3m.base.median(rels)
        mean_full = stage3m.base.mean(rels)
        max_harm = stage3m.base.max_rel_harm(rels)
        min_rel = min([value for value in rels if math.isfinite(value)], default=float("nan"))
        rolling_median = stage3m.base.median(rolling)
        local_harm = stage3m.base.max_rel_harm(local)
        final_error_median = stage3m.base.median(final_errors)
        action_pass_count = sum(1 for row in fidelity_by_policy.get(policy_id, []) if bool_value(row.get("action_fidelity_pass")))
        all_action = action_pass_count == len(SEQUENCES)
        improved_count = sum(1 for value in rels if math.isfinite(value) and value > 0.0)
        if MODE == "full":
            gate = bool(
                len(rows) == len(SEQUENCES)
                and all_action
                and math.isfinite(median_full)
                and median_full >= 0.05
                and improved_count >= 3
                and max_harm <= 0.01
                and math.isfinite(rolling_median)
                and rolling_median > 0.0
                and local_harm <= 0.02
                and math.isfinite(final_error_median)
                and final_error_median >= -0.02
            )
        else:
            gate = bool(
                len(rows) == len(SEQUENCES)
                and all_action
                and math.isfinite(median_full)
                and median_full >= 0.05
                and math.isfinite(min_rel)
                and min_rel >= -0.02
                and max_harm <= 0.02
            )
        b1_gap = stage3m.base.median([rels_by_seq.get(seq, float("nan")) - B1_SAME_SEQ[seq] for seq in SEQUENCES])
        sample = rows[0]
        row_out: dict[str, Any] = {
                "schema": "acl2_v112tf_h1_t4_policy_summary_row_v1",
                "candidate_id": "H1_T4",
                "surface_id": sample.get("surface_id", ""),
                "policy_id": policy_id,
                "policy_family": sample.get("policy_family", ""),
                "sequence_count": len(rows),
                "metric_complete": len(rows) == len(SEQUENCES) and all(bool_value(row.get("metric_available")) for row in rows),
                "action_fidelity_pass_count": action_pass_count,
                "all_action_fidelity": all_action,
                "median_full_rel": median_full,
                "mean_full_rel": mean_full,
                "min_seq_full_rel": min_rel,
                "improved_seq_count": improved_count,
                "max_harm": max_harm,
                "rolling_p90_median_rel": rolling_median,
                "local_window_median_harm": local_harm,
                "final_error_median_rel": final_error_median,
                "gate_pass": gate,
                "pilot_gate_pass": gate if MODE == "pilot" else "",
                "full_gate_pass": gate if MODE == "full" else "",
                "median_full_rel_minus_b1_same_seq": b1_gap,
                "mask_mode": sample.get("mask_mode", ""),
                "camera_token_weight_mean": sample.get("camera_token_weight_mean", ""),
                "register_token_weight_mean": sample.get("register_token_weight_mean", ""),
                "anchor_token_weight_mean": sample.get("anchor_token_weight_mean", ""),
                "claim_boundary": (
                    "00/01/02/05 full validation; semantic-aware claim still requires matched controls."
                    if MODE == "full"
                    else "00/02 pilot only; requires full 00/01/02/05 validation and controls before promotion."
                ),
        }
        for seq in SEQUENCES:
            row_out[f"seq{seq}_full_rel"] = rels_by_seq.get(seq, "")
        out.append(row_out)
    return out


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: safe_float(row.get("median_full_rel", "nan")), reverse=True)
    lines = [
        f"# ACL2 v112TF H1/T4 Semantic Lifetime {RUN_LABEL.title()} Report",
        "",
        f"metric_complete: `{summary['metric_complete']}`",
        f"all_action_fidelity: `{summary['all_action_fidelity']}`",
        f"taxonomy: `{summary['taxonomy']}`",
        f"blocker: `{summary['blocker']}`",
        "",
        "## Ranking",
        "",
    ]
    for row in ranked:
        lines.append(
            "- {policy}: median={median} mean={mean} min={minrel} harm={harm} rolling={rolling} local_harm={local_harm} gate={gate} vs_B1_same_seq={b1gap}".format(
                policy=row.get("policy_id", ""),
                median=row.get("median_full_rel", ""),
                mean=row.get("mean_full_rel", ""),
                minrel=row.get("min_seq_full_rel", ""),
                harm=row.get("max_harm", ""),
                rolling=row.get("rolling_p90_median_rel", ""),
                local_harm=row.get("local_window_median_harm", ""),
                gate=row.get("gate_pass", ""),
                b1gap=row.get("median_full_rel_minus_b1_same_seq", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This is a 00/01/02/05 full validation for a non-B1/A1 v112 surface. "
                "It cannot be promoted to semantic-aware success without matched controls."
                if MODE == "full"
                else "This is a 00/02 pilot for a non-B1/A1 v112 surface. It cannot be promoted without four-sequence validation and semantic controls."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    install_overrides()
    config_rows = read_csv(CONFIG_ROWS)
    latest = stage2m.latest_run_results(read_csv(RUN_RESULTS))
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        for row in rows:
            row["schema"] = str(row.get("schema", "")).replace("acl2_v109tf_stage2", "acl2_v112tf_h1_t4")
            cfg = next((cfg for cfg in config_rows if cfg["policy_id"] == row.get("policy_id") and cfg["seq"] == row.get("seq")), {})
            row["mask_mode"] = cfg.get("mask_mode", "")
            row["camera_token_weight_mean"] = cfg.get("camera_token_weight_mean", "")
            row["register_token_weight_mean"] = cfg.get("register_token_weight_mean", "")
            row["anchor_token_weight_mean"] = cfg.get("anchor_token_weight_mean", "")

    policy_rows = policy_summary_rows(full_rows, rolling_rows, fidelity_rows)
    observed_counts: dict[str, int] = {}
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            phase = str(row.get("phase", ""))
            observed_counts[phase] = observed_counts.get(phase, 0) + 1

    metric_complete = len(full_rows) == len(config_rows) and all(bool_value(row.get("metric_available")) for row in full_rows)
    all_action = len(fidelity_rows) == len(config_rows) and all(bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows)
    pass_rows = [row for row in policy_rows if bool_value(row.get("gate_pass")) and row.get("policy_id") != "H1_no_action_mask_all1_default_off"]
    best = max(policy_rows, key=lambda row: safe_float(row.get("median_full_rel", "nan"))) if policy_rows else {}
    if pass_rows and MODE == "full":
        taxonomy = "H1_T4_FULL_GEOMETRY_PASS_SEMANTIC_CONTROLS_PENDING"
        blocker = "semantic_controls_pending"
    elif pass_rows:
        taxonomy = "H1_T4_00_02_PILOT_GEOMETRY_PASS_SEMANTIC_CONTROLS_PENDING"
        blocker = "full_00_01_02_05_validation_and_controls_pending"
    elif MODE == "full":
        taxonomy = "H1_T4_FULL_NO_GEOMETRY_PASS"
        blocker = "full_geometry_gate_not_passed"
    else:
        taxonomy = "H1_T4_00_02_PILOT_NO_GEOMETRY_PASS"
        blocker = "no_non_default_policy_passed_00_02_pilot_gate"

    summary = {
        "schema": f"acl2_v112tf_h1_t4_lifetime_{RUN_LABEL}_metric_summary_v1",
        "mode": MODE,
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "taxonomy": taxonomy,
        "blocker": blocker,
        "observed_prepare_count": observed_counts.get("prepare", 0),
        "observed_run_worker_count": observed_counts.get("run_worker", 0),
        "observed_evaluate_count": observed_counts.get("evaluate", 0),
        "observed_report_count": observed_counts.get("report", 0),
        "expected_run_worker_count": len(config_rows),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "policy_summary_row_count": len(policy_rows),
        "gate_policy_ids": [row["policy_id"] for row in pass_rows],
        "pilot_gate_policy_ids": [row["policy_id"] for row in pass_rows] if MODE == "pilot" else [],
        "full_gate_policy_ids": [row["policy_id"] for row in pass_rows] if MODE == "full" else [],
        "best_policy_by_median_full_rel": best.get("policy_id", ""),
        "best_policy_median_full_rel": best.get("median_full_rel", ""),
        "best_policy_mean_full_rel": best.get("mean_full_rel", ""),
        "semantic_causality_claim_allowed": False,
        "semantic_causality_claim_blocker": (
            "Full validation has no random/same-bucket/schedule/semantic-shuffle controls."
            if MODE == "full"
            else "00/02 pilot has no random/same-bucket/schedule/semantic-shuffle controls."
        ),
        "outputs": {
            "full_metric_rows": rel(PILOT / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(PILOT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(PILOT / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(PILOT / "action_fidelity_rows.csv"),
            "policy_summary_rows": rel(PILOT / "policy_summary_rows.csv"),
            "report": rel(PILOT / REPORT_NAME),
            "summary": rel(PILOT / "h1_t4_metric_summary.json"),
        },
    }
    write_csv(PILOT / "full_metric_rows.csv", full_rows)
    write_csv(PILOT / "rolling_metric_rows.csv", rolling_rows)
    write_csv(PILOT / "local_handoff_metric_rows.csv", local_rows)
    write_csv(PILOT / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(PILOT / "policy_summary_rows.csv", policy_rows)
    write_json(PILOT / "h1_t4_metric_summary.json", summary)
    write_text(PILOT / REPORT_NAME, report_text(summary, policy_rows))
    print(json.dumps(stage3m.base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
