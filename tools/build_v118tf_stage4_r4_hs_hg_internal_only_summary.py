#!/usr/bin/env python3
"""Summarize v118 Stage4-R4 HS-HG internal-only full pilots."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v113hs_action_metric_summary import finite_values, rel_improvement
from build_v113hs_baseline_metric_summary import summarize_sequence


ROOT = Path(__file__).resolve().parents[1]
V113_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE4 = RESULT_ROOT / "stage4_r4_hs_hg_internal_only"
OUT = STAGE4 / "summary"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"

SEQS = ["00", "02"]
PREFIX = "stage4_r4_hs_hg4_internal_only_mild"
VARIANT = "HS_HG4_internal_only_head_gate_mild"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a v118 Stage4-R4 HS-HG full pilot.")
    parser.add_argument("--stage4-root", type=Path, default=STAGE4)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--variant", default=VARIANT)
    parser.add_argument("--summary-subdir", default="summary")
    return parser.parse_args()


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
        value = float(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def manifest_status(seq: str) -> dict[str, Any]:
    case = f"{PREFIX}_full_kitti_{seq}"
    manifest = STAGE4 / "diagnostics" / case / "run_manifest.json"
    data = read_json(manifest)
    return {
        "seq": seq,
        "case": case,
        "returncode": data.get("returncode", ""),
        "action": data.get("action", ""),
        "control": data.get("control", ""),
        "gpu": data.get("gpu", ""),
        "manifest": rel(manifest),
        "output_root": data.get("output_root", ""),
        "log_path": data.get("log_path", ""),
        "command": data.get("command", ""),
    }


def action_audit_summary(seq: str) -> dict[str, Any]:
    path = STAGE4 / "diagnostics" / f"{PREFIX}_full_kitti_{seq}" / "hs_hg_action_gate_rows.csv"
    if not path.exists():
        return {"seq": seq, "action_audit_rows": 0, "audit_path": rel(path)}
    df = pd.read_csv(path)
    out: dict[str, Any] = {"seq": seq, "action_audit_rows": int(len(df)), "audit_path": rel(path)}
    for col in [
        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
        "gate_row_mean_std",
        "changed_head_fraction_abs_gt_1e_4",
        "internal_head_q_mean",
        "internal_head_q_std",
    ]:
        if col not in df:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        out[f"{col}_mean"] = float(values.mean())
        out[f"{col}_p10"] = float(np.percentile(values, 10))
        out[f"{col}_p90"] = float(np.percentile(values, 90))
    out["controls"] = ";".join(sorted({str(v) for v in df.get("control", pd.Series(dtype=object)).dropna().unique()}))
    out["rowmean_neutral_values"] = ";".join(
        sorted({str(v) for v in df.get("rowmean_neutral", pd.Series(dtype=object)).dropna().unique()})
    )
    return out


def summarize_pair(seq: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_root = V113_ROOT / "outputs" / f"baseline_kitti_{seq}"
    candidate_root = STAGE4 / "outputs" / f"{PREFIX}_full_kitti_{seq}"
    base = summarize_sequence(baseline_root, seq)
    cand = summarize_sequence(candidate_root, seq)
    base["variant"] = "v113_baseline_default_no_loop"
    cand["variant"] = VARIANT
    cmp_row: dict[str, Any] = {
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
        cmp_row[f"baseline_{metric}"] = b
        cmp_row[f"candidate_{metric}"] = c
        cmp_row[f"{metric}_rel_improvement"] = rel_improvement(b, c)
        if b is not None and c is not None:
            cmp_row[f"{metric}_abs_delta_candidate_minus_baseline"] = float(c - b)
    return base, cand, cmp_row


def report_text(summary: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> str:
    agg = summary["aggregate"]
    lines = [
        "# ACL2 v118-TF Stage4-R4 HS-HG Internal-Only Summary",
        "",
        f"- candidate: `{agg['candidate_name']}`",
        f"- pilot_geometry_gate_pass: `{agg['pilot_geometry_gate']['pass']}`",
        f"- median_full_ATE_rel_improvement: `{agg['median_full_ATE_rel_improvement']}`",
        f"- median_rolling_p90_rel_improvement: `{agg['median_rolling_p90_rel_improvement']}`",
        f"- max_full_ATE_harm_rel: `{agg['max_full_ATE_harm_rel']}`",
        f"- segment_scale_not_worse_all: `{agg['segment_scale_not_worse_all']}`",
        "",
        "| seq | baseline ATE | candidate ATE | full ATE rel | rolling p90 rel | segment scale rel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['seq']} | {row['baseline_full_ATE_sim3_rmse']} | {row['candidate_full_ATE_sim3_rmse']} | "
            f"{row['full_ATE_sim3_rmse_rel_improvement']} | {row['rolling_ate_p90_rel_improvement']} | "
            f"{row['segment_scale_log_error_median_abs_rel_improvement']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "This R4 run is a non-semantic HS-HG head-output repair. It does not establish semantic-aware geometry improvement. If the geometry pilot gate fails, semantic-causality controls are not promoted from this candidate.",
    ]
    return "\n".join(lines)


def main() -> None:
    global STAGE4, OUT, PREFIX, VARIANT
    args = parse_args()
    STAGE4 = args.stage4_root
    OUT = STAGE4 / args.summary_subdir
    PREFIX = args.prefix
    VARIANT = args.variant
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    manifest_rows = [manifest_status(seq) for seq in SEQS]
    audit_rows = [action_audit_summary(seq) for seq in SEQS]
    for seq in SEQS:
        base, cand, cmp_row = summarize_pair(seq)
        metric_rows.extend([base, cand])
        comparison_rows.append(cmp_row)

    full_ate_improvements = finite_values(comparison_rows, "full_ATE_sim3_rmse_rel_improvement")
    rolling_improvements = finite_values(comparison_rows, "rolling_ate_p90_rel_improvement")
    segment_improvements = finite_values(comparison_rows, "segment_scale_log_error_median_abs_rel_improvement")
    full_ate_harms = [max(0.0, -value) for value in full_ate_improvements]
    median_threshold = 0.05
    max_harm_threshold = 0.01
    gate_pass = bool(
        full_ate_improvements
        and rolling_improvements
        and segment_improvements
        and (
            float(np.median(full_ate_improvements)) >= median_threshold
            or float(np.median(rolling_improvements)) >= median_threshold
        )
        and max(full_ate_harms) <= max_harm_threshold
        and all(value >= 0 for value in segment_improvements)
    )
    controls = sorted({str(row.get("control", "")) for row in manifest_rows if row.get("control")})
    if not controls:
        controls = sorted(
            {
                str(row.get("controls", ""))
                for row in audit_rows
                if row.get("controls")
            }
        )
    control_label = ";".join(controls) if controls else "unknown"
    semantic_gate = (
        {
            "status": "pending_controls_after_geometry_gate_pass",
            "pass": False,
            "reason": "The non-semantic HS-HG4 repair passed the geometry pilot gate, but semantic-causality controls have not been run by this summary step.",
        }
        if gate_pass
        else {
            "status": "not_triggered_geometry_gate_failed",
            "pass": False,
            "reason": "The non-semantic HS-HG4 repair failed the geometry pilot gate.",
        }
    )
    aggregate = {
        "schema": "acl2_v118tf_stage4_r4_hs_hg_internal_only_metric_summary_v1",
        "seqs": SEQS,
        "baseline_name": "v113_baseline_default_no_loop",
        "candidate_name": VARIANT,
        "action": VARIANT,
        "control": control_label,
        "median_full_ATE_rel_improvement": float(np.median(full_ate_improvements)) if full_ate_improvements else None,
        "median_rolling_p90_rel_improvement": float(np.median(rolling_improvements)) if rolling_improvements else None,
        "median_segment_scale_rel_improvement": float(np.median(segment_improvements)) if segment_improvements else None,
        "max_full_ATE_harm_rel": float(max(full_ate_harms)) if full_ate_harms else None,
        "improved_seq_count_full_ATE": int(sum(value > 0 for value in full_ate_improvements)),
        "segment_scale_not_worse_all": bool(segment_improvements and all(value >= 0 for value in segment_improvements)),
        "pilot_geometry_gate": {
            "thresholds": {
                "median_full_ATE_rel_improvement_ge": median_threshold,
                "or_median_rolling_p90_rel_improvement_ge": median_threshold,
                "max_full_ATE_harm_rel_le": max_harm_threshold,
                "segment_scale_not_worse_all": True,
            },
            "pass": gate_pass,
        },
        "semantic_causality_gate": semantic_gate,
    }
    summary = {
        "schema": "acl2_v118tf_stage4_r4_hs_hg_internal_only_summary_v1",
        "aggregate": aggregate,
        "metric_rows": metric_rows,
        "comparison_rows": comparison_rows,
        "manifest_rows": manifest_rows,
        "action_audit_rows": audit_rows,
        "outputs": {
            "metric_rows": rel(OUT / "stage4_r4_hs_hg_internal_only_metric_rows.csv"),
            "comparison_rows": rel(OUT / "stage4_r4_hs_hg_internal_only_comparison_rows.csv"),
            "action_audit_summary": rel(OUT / "stage4_r4_hs_hg_internal_only_action_audit_summary.csv"),
            "manifest_summary": rel(OUT / "stage4_r4_hs_hg_internal_only_manifest_summary.csv"),
            "summary": rel(OUT / "stage4_r4_hs_hg_internal_only_summary.json"),
            "report": rel(OUT / "STAGE4_R4_HS_HG_INTERNAL_ONLY_REPORT.md"),
        },
    }

    write_csv(OUT / "stage4_r4_hs_hg_internal_only_metric_rows.csv", metric_rows)
    write_csv(OUT / "stage4_r4_hs_hg_internal_only_comparison_rows.csv", comparison_rows)
    write_csv(OUT / "stage4_r4_hs_hg_internal_only_action_audit_summary.csv", audit_rows)
    write_csv(OUT / "stage4_r4_hs_hg_internal_only_manifest_summary.csv", manifest_rows)
    write_json(OUT / "stage4_r4_hs_hg_internal_only_summary.json", summary)
    write_text(OUT / "STAGE4_R4_HS_HG_INTERNAL_ONLY_REPORT.md", report_text(summary, comparison_rows))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R4",
            "surface_or_branch": "HS-HG",
            "status": "NO_GO_GEOMETRY_GATE_FAILED" if not gate_pass else "PASS_GEOMETRY_GATE_PENDING_CONTROLS",
            "artifact": rel(OUT / "stage4_r4_hs_hg_internal_only_summary.json"),
            "notes": f"HS_HG4 full 00/02 pilot complete under control={control_label}; semantic controls not triggered unless geometry gate passes",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
