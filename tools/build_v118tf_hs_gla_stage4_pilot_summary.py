#!/usr/bin/env python3
"""Summarize v118 Stage4-R11 HS-GLA write/retention pilot matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v113hs_action_metric_summary import rel_improvement
from build_v113hs_baseline_metric_summary import summarize_sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
V113_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
STAGE = RESULT_ROOT / "stage4_r11_hs_gla_write_retention_pilot"
OUT = STAGE / "summary"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def stats(df: pd.DataFrame, col: str, prefix: str) -> dict[str, Any]:
    values = numeric(df, col)
    if values.empty:
        return {
            f"{prefix}_{col}_mean": None,
            f"{prefix}_{col}_p10": None,
            f"{prefix}_{col}_p90": None,
            f"{prefix}_{col}_max": None,
        }
    return {
        f"{prefix}_{col}_mean": float(values.mean()),
        f"{prefix}_{col}_p10": float(values.quantile(0.1)),
        f"{prefix}_{col}_p90": float(values.quantile(0.9)),
        f"{prefix}_{col}_max": float(values.max()),
    }


def median(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def case_name(prefix: str, seq: str, max_frames: int) -> str:
    suffix = f"max{max_frames}_kitti_{seq}" if max_frames > 0 else f"full_kitti_{seq}"
    return f"{prefix}_{suffix}"


def audit_summary(stage_root: Path, prefix: str, seq: str, max_frames: int) -> dict[str, Any]:
    case = stage_root / "diagnostics" / case_name(prefix, seq, max_frames)
    gq = case / "hs_gq_action_gate_rows.csv"
    state = case / "hs_gq_state_action_rows.csv"
    row: dict[str, Any] = {
        "gq_action_rows": 0,
        "gq_state_action_rows": 0,
        "gq_action_path": rel(gq),
        "gq_state_action_path": rel(state),
    }
    if gq.exists():
        df = pd.read_csv(gq)
        row["gq_action_rows"] = int(len(df))
        row["gq_scopes"] = ";".join(sorted({str(v) for v in df.get("scope", pd.Series(dtype=object)).dropna().unique()}))
        row["gq_calibration_modes"] = ";".join(sorted({str(v) for v in df.get("calibration_mode", pd.Series(dtype=object)).dropna().unique()}))
        row["gq_intervention_forms"] = ";".join(sorted({str(v) for v in df.get("intervention_form", pd.Series(dtype=object)).dropna().unique()}))
        for col in ["gate_std", "changed_frame_fraction_abs_gt_1e_4", "internal_target_q_std"]:
            row.update(stats(df, col, "gq"))
    if state.exists():
        df = pd.read_csv(state)
        row["gq_state_action_rows"] = int(len(df))
        row["gq_state_scopes"] = ";".join(sorted({str(v) for v in df.get("scope", pd.Series(dtype=object)).dropna().unique()}))
        row["gq_state_calibration_modes"] = ";".join(sorted({str(v) for v in df.get("calibration_mode", pd.Series(dtype=object)).dropna().unique()}))
        row["gq_state_intervention_forms"] = ";".join(sorted({str(v) for v in df.get("intervention_form", pd.Series(dtype=object)).dropna().unique()}))
        row["gq_state_changed_rows"] = int(df.get("changed_state", pd.Series(dtype=object)).astype(str).str.lower().isin({"true", "1"}).sum())
        row["gq_state_first_chunk_rows"] = int(df.get("first_chunk_no_prior", pd.Series(dtype=object)).astype(str).str.lower().isin({"true", "1"}).sum())
        for col in ["state_delta_gate", "state_delta_rel_norm_raw", "state_delta_rel_norm_after", "delta_pressure"]:
            row.update(stats(df, col, "gq_state"))
    return row


def summarize_run(job: dict[str, str]) -> dict[str, Any]:
    seq = job["seq"]
    prefix = job["prefix"]
    max_frames = int(job.get("max_frames") or 0)
    manifest_path = STAGE / "diagnostics" / case_name(prefix, seq, max_frames) / "run_manifest.json"
    manifest = read_json(manifest_path)
    baseline_root = V113_ROOT / "outputs" / f"baseline_kitti_{seq}"
    candidate_root = STAGE / "outputs" / case_name(prefix, seq, max_frames)
    base = summarize_sequence(baseline_root, seq)
    cand = summarize_sequence(candidate_root, seq)
    row: dict[str, Any] = {
        "branch": job["branch"],
        "run_type": job["run_type"],
        "variant": job["variant"],
        "seq": seq,
        "action": job["action"],
        "control": job.get("control", ""),
        "intervention_form": job.get("intervention_form", ""),
        "returncode": manifest.get("returncode"),
        "gpu": manifest.get("gpu"),
        "manifest": rel(manifest_path),
        "candidate_output_root": rel(candidate_root),
        "command": manifest.get("command", job.get("command", "")),
    }
    for metric in [
        "full_ATE_sim3_rmse",
        "rolling_ate_p90",
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
    row.update(audit_summary(STAGE, prefix, seq, max_frames))
    return row


def pilot_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [row.get("full_ATE_sim3_rmse_rel_improvement") for row in rows]
    rolling = [row.get("rolling_ate_p90_rel_improvement") for row in rows]
    med_full = median(full)
    med_rolling = median(rolling)
    harms = [max(0.0, -float(v)) for v in full if v is not None and np.isfinite(float(v))]
    max_harm = float(max(harms)) if harms else None
    passed = bool(
        med_full is not None
        and med_rolling is not None
        and max_harm is not None
        and med_full >= 0.03
        and med_rolling > 0.0
        and max_harm <= 0.01
    )
    return {
        "median_full_ATE_rel_improvement": med_full,
        "median_rolling_p90_rel_improvement": med_rolling,
        "max_full_ATE_harm_rel": max_harm,
        "pass": passed,
        "thresholds": {
            "median_full_ATE_rel_improvement_ge": 0.03,
            "median_rolling_p90_rel_improvement_gt": 0.0,
            "max_full_ATE_harm_rel_le": 0.01,
        },
    }


def report_text(summary: dict[str, Any]) -> str:
    lines = [
        "# ACL2 v118-TF Stage4-R11 HS-GLA Write/Retention Pilot",
        "",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- all_runtime_returncodes_zero: `{summary['aggregate']['all_runtime_returncodes_zero']}`",
        f"- candidate_pilot_gate_pass_count: `{summary['aggregate']['candidate_pilot_gate_pass_count']}`",
        f"- direct_gamma_hook_status: `{summary['direct_gamma_hook']['status']}`",
        "",
        "| branch | variant | type | median full ATE rel | median rolling p90 rel | max harm | gate | control |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in summary["variant_summary"]:
        gate = row["pilot_gate"]
        lines.append(
            f"| {row['branch']} | {row['variant']} | {row['run_type']} | "
            f"{gate['median_full_ATE_rel_improvement']} | {gate['median_rolling_p90_rel_improvement']} | "
            f"{gate['max_full_ATE_harm_rel']} | {gate['pass']} | {row.get('control', '')} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "HGR rows are retention/state reliability fallback probes through state-delta gain. They are not direct KDA gamma or decay modulation because the current HorizonStream wrapper does not expose such a runtime hook.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = read_csv_rows(STAGE / "matrix_job_manifest.csv")
    rows = [summarize_run(job) for job in jobs]
    variant_summary: list[dict[str, Any]] = []
    for key in sorted({(row["branch"], row["variant"], row["run_type"], row.get("control", "")) for row in rows}):
        branch, variant, run_type, control = key
        subset = [row for row in rows if (row["branch"], row["variant"], row["run_type"], row.get("control", "")) == key]
        variant_summary.append(
            {
                "branch": branch,
                "variant": variant,
                "run_type": run_type,
                "control": control,
                "intervention_form": subset[0].get("intervention_form", "") if subset else "",
                "pilot_gate": pilot_gate(subset),
                "returncodes": sorted({row.get("returncode") for row in subset}),
                "audit_rows": {
                    "gq_action_rows": int(sum(int(row.get("gq_action_rows") or 0) for row in subset)),
                    "gq_state_action_rows": int(sum(int(row.get("gq_state_action_rows") or 0) for row in subset)),
                    "gq_state_changed_rows": int(sum(int(row.get("gq_state_changed_rows") or 0) for row in subset)),
                },
            }
        )
    candidate_pass_count = int(sum(1 for row in variant_summary if row["run_type"] == "candidate" and row["pilot_gate"]["pass"]))
    aggregate = {
        "schema": "acl2_v118tf_hs_gla_stage4_pilot_aggregate_v1",
        "all_runtime_returncodes_zero": bool(rows and all(str(row.get("returncode")) == "0" for row in rows)),
        "candidate_run_type_count": int(sum(1 for row in variant_summary if row["run_type"] == "candidate")),
        "matched_control_run_type_count": int(sum(1 for row in variant_summary if row["run_type"] == "matched_control")),
        "candidate_pilot_gate_pass_count": candidate_pass_count,
        "decision": (
            "HS_GLA_STAGE4_PILOT_GATE_PASS_PENDING_EXPANDED_CONTROLS"
            if candidate_pass_count > 0
            else "HS_GLA_STAGE4_NO_PILOT_GATE_PASS"
        ),
    }
    direct_gamma_hook = {
        "status": "NOT_EXPOSED_CURRENT_HORIZONSTREAM_WRAPPER_FALLBACK_TO_STATE_DELTA_PROXY",
        "evidence": [
            "third_party/HorizonStream/horizonstream/runtime/layers/attention.py constructs KDA without a retention/decay runtime parameter.",
            "GLAAttention.forward passes hidden_states, attention_mask, past_key_values, use_cache, and output_attentions only.",
        ],
        "hgr_interpretation": "HGR candidates are retention proxies through bounded state-delta gain, not direct gamma modulation.",
    }
    summary = {
        "schema": "acl2_v118tf_hs_gla_stage4_pilot_summary_v1",
        "global_goal_achieved": False,
        "aggregate": aggregate,
        "direct_gamma_hook": direct_gamma_hook,
        "variant_summary": variant_summary,
        "rows": rows,
        "outputs": {
            "summary": rel(OUT / "hs_gla_stage4_pilot_summary.json"),
            "report": rel(OUT / "HS_GLA_STAGE4_PILOT_REPORT.md"),
            "rows": rel(OUT / "hs_gla_stage4_pilot_rows.csv"),
            "variant_summary": rel(OUT / "hs_gla_stage4_pilot_variant_summary.csv"),
        },
    }
    write_csv(OUT / "hs_gla_stage4_pilot_rows.csv", rows)
    flat_variants = []
    for row in variant_summary:
        flat = {k: v for k, v in row.items() if k not in {"pilot_gate", "audit_rows"}}
        flat.update({f"pilot_{k}": v for k, v in row["pilot_gate"].items() if k != "thresholds"})
        flat.update(row["audit_rows"])
        flat_variants.append(flat)
    write_csv(OUT / "hs_gla_stage4_pilot_variant_summary.csv", flat_variants)
    write_json(OUT / "hs_gla_stage4_pilot_summary.json", summary)
    write_text(OUT / "HS_GLA_STAGE4_PILOT_REPORT.md", report_text(summary))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-HSGLA-PilotMatrix",
            "surface_or_branch": "HS-GLA",
            "status": aggregate["decision"],
            "artifact": rel(OUT / "hs_gla_stage4_pilot_summary.json"),
            "notes": "HS-GW/HGR 00/02 full pilot matrix with matched random-sign controls; HGR direct gamma not exposed, so recorded as state-delta retention proxy",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
