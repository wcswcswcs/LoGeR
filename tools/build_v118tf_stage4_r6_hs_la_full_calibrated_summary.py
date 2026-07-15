#!/usr/bin/env python3
"""Summarize v118 Stage4-R6 HS-LA full-calibrated selected-query pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v113hs_action_metric_summary import finite_values, rel_improvement
from build_v113hs_baseline_metric_summary import summarize_sequence
from build_v118tf_stage4_r5_hs_la_internal_qk_summary import (
    REGISTRY,
    ROOT,
    V113_ROOT,
    clean_json,
    numeric,
    read_csv_rows,
    read_json,
    rel,
    stat,
    write_csv,
    write_json,
    write_text,
)


RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r6_hs_la_full_calibrated_selected_query"
OUT = STAGE / "summary"
PREFIX = "stage4_r6_hs_la5_full_calibrated_selected_query_tiny_tight"
VARIANT = "HS_LA5_full_calibrated_selected_query_tiny_tight"
CONTROL = "full_calibrated_selected_query"
SEQS = ["00", "02"]


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv_rows(REGISTRY)
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


def manifest_status(seq: str, *, max_frames: int = 0) -> dict[str, Any]:
    suffix = f"max{max_frames}_kitti_{seq}" if max_frames else f"full_kitti_{seq}"
    case = f"{PREFIX}_{suffix}"
    manifest = STAGE / "diagnostics" / case / "run_manifest.json"
    data = read_json(manifest)
    return {
        "seq": seq,
        "case": case,
        "max_frames": int(max_frames),
        "returncode": data.get("returncode"),
        "action": data.get("action"),
        "control": data.get("control"),
        "gpu": data.get("gpu"),
        "manifest": rel(manifest),
        "output_root": data.get("output_root"),
        "log_path": data.get("log_path"),
        "command": data.get("command"),
    }


def action_audit_summary(seq: str, *, max_frames: int = 0) -> dict[str, Any]:
    suffix = f"max{max_frames}_kitti_{seq}" if max_frames else f"full_kitti_{seq}"
    path = STAGE / "diagnostics" / f"{PREFIX}_{suffix}" / "hs_la_action_rows.csv"
    if not path.exists():
        return {"seq": seq, "max_frames": int(max_frames), "action_audit_rows": 0, "audit_path": rel(path)}
    df = pd.read_csv(path)
    selected = df[df.get("scope", pd.Series(dtype=str)).astype(str) == "pose_query"].copy()
    row: dict[str, Any] = {
        "seq": seq,
        "max_frames": int(max_frames),
        "action_audit_rows": int(len(df)),
        "selected_pose_query_rows": int(len(selected)),
        "audit_path": rel(path),
        "controls": ";".join(sorted({str(v) for v in df.get("control", pd.Series(dtype=object)).dropna().unique()})),
        "calibration_modes": ";".join(
            sorted({str(v) for v in df.get("calibration_mode", pd.Series(dtype=object)).dropna().unique()})
        ),
    }
    for col in [
        "bias_abs_mean",
        "bias_std",
        "bias_min",
        "bias_max",
        "read_reliability_gate_mean",
        "semantic_direction_std_mean",
        "internal_qk_positive_fraction",
        "attention_read_reliability_mean",
        "qk_logit_top1_minus_top2_mean",
        "selected_output_delta_rel_norm",
    ]:
        row.update(stat(numeric(selected, col), col))
    return row


def summarize_pair(seq: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_root = V113_ROOT / "outputs" / f"baseline_kitti_{seq}"
    candidate_root = STAGE / "outputs" / f"{PREFIX}_full_kitti_{seq}"
    base = summarize_sequence(baseline_root, seq)
    cand = summarize_sequence(candidate_root, seq)
    base["variant"] = "v113_baseline_default_no_loop"
    cand["variant"] = VARIANT
    row: dict[str, Any] = {
        "seq": seq,
        "baseline_variant": base["variant"],
        "candidate_variant": cand["variant"],
        "baseline_output_root": rel(baseline_root),
        "candidate_output_root": rel(candidate_root),
    }
    for metric in [
        "full_ATE_sim3_rmse",
        "rolling_ate_p90",
        "final_error_sim3_aligned",
        "segment_scale_log_error_median_abs",
        "adjacent_log_scale_jump_p90_abs",
        "rpe_delta1_translation_mean",
        "rpe_delta1_rotation_deg_mean",
        "global_sim3_scale",
    ]:
        b = base.get(metric)
        c = cand.get(metric)
        row[f"baseline_{metric}"] = b
        row[f"candidate_{metric}"] = c
        row[f"{metric}_rel_improvement"] = rel_improvement(b, c)
        if b is not None and c is not None:
            row[f"{metric}_abs_delta_candidate_minus_baseline"] = float(c - b)
    return base, cand, row


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    agg = summary["aggregate"]
    lines = [
        "# ACL2 v118-TF Stage4-R6 HS-LA Full-Calibrated Selected-Query Pilot",
        "",
        f"- candidate: `{agg['candidate_name']}`",
        f"- control: `{agg['control']}`",
        f"- pilot_gate_plan13_2_pass: `{agg['pilot_gate_plan13_2']['pass']}`",
        f"- median_full_ATE_rel_improvement: `{agg['median_full_ATE_rel_improvement']}`",
        f"- median_rolling_p90_rel_improvement: `{agg['median_rolling_p90_rel_improvement']}`",
        f"- max_full_ATE_harm_rel: `{agg['max_full_ATE_harm_rel']}`",
        f"- segment_scale_not_worse_all: `{agg['segment_scale_not_worse_all']}`",
        "",
        "| seq | baseline ATE | candidate ATE | full ATE rel | rolling p90 rel | segment scale rel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | {row['baseline_full_ATE_sim3_rmse']} | {row['candidate_full_ATE_sim3_rmse']} | "
            f"{row['full_ATE_sim3_rmse_rel_improvement']} | {row['rolling_ate_p90_rel_improvement']} | "
            f"{row['segment_scale_log_error_median_abs_rel_improvement']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "This run tests the pre-registered HS-LA full calibrated selected-query logit intervention: semantic direction, internal QK alignment, and selected-query read-focus reliability are coupled on the local-attention carrier. It is not a success unless the pilot gate and later controls pass.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = [manifest_status("00", max_frames=128)] + [manifest_status(seq) for seq in SEQS]
    audit_rows = [action_audit_summary("00", max_frames=128)] + [action_audit_summary(seq) for seq in SEQS]
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for seq in SEQS:
        base, cand, comp = summarize_pair(seq)
        metric_rows.extend([base, cand])
        comparison_rows.append(comp)

    full = finite_values(comparison_rows, "full_ATE_sim3_rmse_rel_improvement")
    rolling = finite_values(comparison_rows, "rolling_ate_p90_rel_improvement")
    segment = finite_values(comparison_rows, "segment_scale_log_error_median_abs_rel_improvement")
    harms = [max(0.0, -value) for value in full]
    median_full = float(np.median(full)) if full else None
    median_rolling = float(np.median(rolling)) if rolling else None
    max_harm = float(max(harms)) if harms else None
    pilot_pass = bool(
        full
        and rolling
        and median_full is not None
        and median_rolling is not None
        and max_harm is not None
        and median_full >= 0.03
        and median_rolling > 0.0
        and max_harm <= 0.01
    )
    segment_not_worse_all = bool(segment and all(value >= 0 for value in segment))
    aggregate = {
        "schema": "acl2_v118tf_stage4_r6_hs_la_full_calibrated_metric_summary_v1",
        "candidate_name": VARIANT,
        "control": CONTROL,
        "seqs": SEQS,
        "baseline_name": "v113_baseline_default_no_loop",
        "median_full_ATE_rel_improvement": median_full,
        "median_rolling_p90_rel_improvement": median_rolling,
        "median_segment_scale_rel_improvement": float(np.median(segment)) if segment else None,
        "max_full_ATE_harm_rel": max_harm,
        "improved_seq_count_full_ATE": int(sum(value > 0 for value in full)),
        "full_ATE_not_worse_all": bool(full and all(value >= -0.01 for value in full)),
        "segment_scale_not_worse_all": segment_not_worse_all,
        "pilot_gate_plan13_2": {
            "thresholds": {
                "median_full_ATE_rel_improvement_ge": 0.03,
                "median_rolling_p90_rel_improvement_gt": 0.0,
                "max_full_ATE_harm_rel_le": 0.01,
            },
            "pass": pilot_pass,
        },
        "full_geometry_safety": {
            "segment_scale_not_worse_all": segment_not_worse_all,
            "pass": bool(pilot_pass and segment_not_worse_all),
        },
        "semantic_causality_gate": {
            "status": "pending_controls_after_pilot_gate_pass" if pilot_pass else "not_triggered_pilot_gate_failed",
            "pass": False,
            "reason": (
                "The HS-LA full-calibrated selected-query action passed the plan 13.2 pilot gate, but semantic-causality controls have not been run."
                if pilot_pass
                else "The HS-LA full-calibrated selected-query action did not reach the plan 13.2 pilot gate."
            ),
        },
    }
    summary = {
        "schema": "acl2_v118tf_stage4_r6_hs_la_full_calibrated_summary_v1",
        "aggregate": aggregate,
        "metric_rows": metric_rows,
        "comparison_rows": comparison_rows,
        "manifest_rows": manifest_rows,
        "action_audit_rows": audit_rows,
        "outputs": {
            "summary": rel(OUT / "stage4_r6_hs_la_full_calibrated_summary.json"),
            "report": rel(OUT / "STAGE4_R6_HS_LA_FULL_CALIBRATED_REPORT.md"),
            "comparison_rows": rel(OUT / "stage4_r6_hs_la_full_calibrated_comparison_rows.csv"),
            "metric_rows": rel(OUT / "stage4_r6_hs_la_full_calibrated_metric_rows.csv"),
            "action_audit_summary": rel(OUT / "stage4_r6_hs_la_full_calibrated_action_audit_summary.csv"),
            "manifest_summary": rel(OUT / "stage4_r6_hs_la_full_calibrated_manifest_summary.csv"),
        },
    }
    write_csv(OUT / "stage4_r6_hs_la_full_calibrated_metric_rows.csv", metric_rows)
    write_csv(OUT / "stage4_r6_hs_la_full_calibrated_comparison_rows.csv", comparison_rows)
    write_csv(OUT / "stage4_r6_hs_la_full_calibrated_action_audit_summary.csv", audit_rows)
    write_csv(OUT / "stage4_r6_hs_la_full_calibrated_manifest_summary.csv", manifest_rows)
    write_json(OUT / "stage4_r6_hs_la_full_calibrated_summary.json", summary)
    write_text(OUT / "STAGE4_R6_HS_LA_FULL_CALIBRATED_REPORT.md", report_text(summary, comparison_rows))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R6",
            "surface_or_branch": "HS-LA",
            "status": "PASS_PILOT_GATE_PENDING_CONTROLS" if pilot_pass else "NO_GO_PILOT_GATE_FAILED",
            "artifact": rel(OUT / "stage4_r6_hs_la_full_calibrated_summary.json"),
            "notes": "HS-LA full-calibrated selected-query full 00/02 pilot complete; controls not triggered unless pilot gate passes",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
