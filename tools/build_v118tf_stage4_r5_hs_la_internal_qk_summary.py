#!/usr/bin/env python3
"""Summarize v118 Stage4-R5 HS-LA internal-QK full pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v113hs_action_metric_summary import finite_values, rel_improvement
from build_v113hs_baseline_metric_summary import summarize_sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
V113_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
STAGE = RESULT_ROOT / "stage4_r5_hs_la_internal_qk_only"
OUT = STAGE / "summary"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
PREFIX = "stage4_r5_hs_la1_internal_qk_only_tiny_tight"
VARIANT = "HS_LA1_internal_qk_only_tiny_tight"
CONTROL = "internal_qk_only"
SEQS = ["00", "02"]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def stat(values: pd.Series, name: str) -> dict[str, Any]:
    if values.empty:
        return {f"{name}_mean": None, f"{name}_p10": None, f"{name}_p90": None, f"{name}_max": None}
    return {
        f"{name}_mean": float(values.mean()),
        f"{name}_p10": float(values.quantile(0.1)),
        f"{name}_p90": float(values.quantile(0.9)),
        f"{name}_max": float(values.max()),
    }


def manifest_status(seq: str) -> dict[str, Any]:
    case = f"{PREFIX}_full_kitti_{seq}"
    manifest = STAGE / "diagnostics" / case / "run_manifest.json"
    data = read_json(manifest)
    return {
        "seq": seq,
        "case": case,
        "returncode": data.get("returncode"),
        "action": data.get("action"),
        "control": data.get("control"),
        "gpu": data.get("gpu"),
        "manifest": rel(manifest),
        "output_root": data.get("output_root"),
        "log_path": data.get("log_path"),
        "command": data.get("command"),
    }


def action_audit_summary(seq: str) -> dict[str, Any]:
    path = STAGE / "diagnostics" / f"{PREFIX}_full_kitti_{seq}" / "hs_la_action_rows.csv"
    if not path.exists():
        return {"seq": seq, "action_audit_rows": 0, "audit_path": rel(path)}
    df = pd.read_csv(path)
    selected = df[df.get("scope", pd.Series(dtype=str)).astype(str) == "pose_query"].copy()
    row: dict[str, Any] = {
        "seq": seq,
        "action_audit_rows": int(len(df)),
        "selected_pose_query_rows": int(len(selected)),
        "audit_path": rel(path),
        "controls": ";".join(sorted({str(v) for v in df.get("control", pd.Series(dtype=object)).dropna().unique()})),
    }
    for col in [
        "bias_abs_mean",
        "bias_std",
        "bias_min",
        "bias_max",
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
        "# ACL2 v118-TF Stage4-R5 HS-LA Internal-QK Full Pilot",
        "",
        f"- candidate: `{agg['candidate_name']}`",
        f"- control: `{agg['control']}`",
        f"- pilot_geometry_gate_pass: `{agg['pilot_geometry_gate']['pass']}`",
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
        "This run tests a non-semantic selected-query internal-QK HS-LA action. If the geometry gate fails, semantic-causality controls are not promoted.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = [manifest_status(seq) for seq in SEQS]
    audit_rows = [action_audit_summary(seq) for seq in SEQS]
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
    gate_pass = bool(
        full
        and rolling
        and segment
        and (float(np.median(full)) >= 0.05 or float(np.median(rolling)) >= 0.05)
        and max(harms) <= 0.01
        and all(value >= 0 for value in segment)
    )
    aggregate = {
        "schema": "acl2_v118tf_stage4_r5_hs_la_internal_qk_metric_summary_v1",
        "candidate_name": VARIANT,
        "control": CONTROL,
        "seqs": SEQS,
        "baseline_name": "v113_baseline_default_no_loop",
        "median_full_ATE_rel_improvement": float(np.median(full)) if full else None,
        "median_rolling_p90_rel_improvement": float(np.median(rolling)) if rolling else None,
        "median_segment_scale_rel_improvement": float(np.median(segment)) if segment else None,
        "max_full_ATE_harm_rel": float(max(harms)) if harms else None,
        "improved_seq_count_full_ATE": int(sum(value > 0 for value in full)),
        "segment_scale_not_worse_all": bool(segment and all(value >= 0 for value in segment)),
        "pilot_geometry_gate": {
            "thresholds": {
                "median_full_ATE_rel_improvement_ge": 0.05,
                "or_median_rolling_p90_rel_improvement_ge": 0.05,
                "max_full_ATE_harm_rel_le": 0.01,
                "segment_scale_not_worse_all": True,
            },
            "pass": gate_pass,
        },
        "semantic_causality_gate": {
            "status": "pending_controls_after_geometry_gate_pass" if gate_pass else "not_triggered_geometry_gate_failed",
            "pass": False,
            "reason": (
                "The non-semantic HS-LA internal-QK action passed the geometry pilot gate, but semantic-causality controls have not been run."
                if gate_pass
                else "The non-semantic HS-LA internal-QK action failed the geometry pilot gate."
            ),
        },
        "failed_attempts": {
            "seq00_oom_attempt1_manifest": rel(STAGE / "failed_attempts" / f"{PREFIX}_full_kitti_00_oom_attempt1_run_manifest.json"),
            "seq00_oom_attempt1_log": rel(STAGE / "failed_attempts" / f"{PREFIX}_full_kitti_00_oom_attempt1.log"),
        },
    }
    summary = {
        "schema": "acl2_v118tf_stage4_r5_hs_la_internal_qk_summary_v1",
        "aggregate": aggregate,
        "metric_rows": metric_rows,
        "comparison_rows": comparison_rows,
        "manifest_rows": manifest_rows,
        "action_audit_rows": audit_rows,
        "outputs": {
            "summary": rel(OUT / "stage4_r5_hs_la_internal_qk_summary.json"),
            "report": rel(OUT / "STAGE4_R5_HS_LA_INTERNAL_QK_REPORT.md"),
            "comparison_rows": rel(OUT / "stage4_r5_hs_la_internal_qk_comparison_rows.csv"),
            "metric_rows": rel(OUT / "stage4_r5_hs_la_internal_qk_metric_rows.csv"),
            "action_audit_summary": rel(OUT / "stage4_r5_hs_la_internal_qk_action_audit_summary.csv"),
            "manifest_summary": rel(OUT / "stage4_r5_hs_la_internal_qk_manifest_summary.csv"),
        },
    }
    write_csv(OUT / "stage4_r5_hs_la_internal_qk_metric_rows.csv", metric_rows)
    write_csv(OUT / "stage4_r5_hs_la_internal_qk_comparison_rows.csv", comparison_rows)
    write_csv(OUT / "stage4_r5_hs_la_internal_qk_action_audit_summary.csv", audit_rows)
    write_csv(OUT / "stage4_r5_hs_la_internal_qk_manifest_summary.csv", manifest_rows)
    write_json(OUT / "stage4_r5_hs_la_internal_qk_summary.json", summary)
    write_text(OUT / "STAGE4_R5_HS_LA_INTERNAL_QK_REPORT.md", report_text(summary, comparison_rows))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R5",
            "surface_or_branch": "HS-LA",
            "status": "PASS_GEOMETRY_GATE_PENDING_CONTROLS" if gate_pass else "NO_GO_GEOMETRY_GATE_FAILED",
            "artifact": rel(OUT / "stage4_r5_hs_la_internal_qk_summary.json"),
            "notes": "HS-LA internal-QK-only full 00/02 pilot complete; semantic controls not triggered unless geometry gate passes",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
