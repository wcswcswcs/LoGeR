#!/usr/bin/env python3
"""Summarize v118 HS-GLA OOM repair runs."""

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
DEFAULT_OOM_STAGE = RESULT_ROOT / "stage4_r11_hs_gla_write_retention_pilot"
STAGE = RESULT_ROOT / "stage4_r12_hs_gla_oom_repair"
OUT = STAGE / "summary"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def metric_or_missing(root: Path, seq: str) -> dict[str, Any]:
    try:
        return summarize_sequence(root, seq)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return {"seq": seq, "output_root": rel(root), "metric_missing": True, "metric_error": type(exc).__name__}


def case_name(prefix: str, seq: str, max_frames: int) -> str:
    suffix = f"max{max_frames}_kitti_{seq}" if max_frames > 0 else f"full_kitti_{seq}"
    return f"{prefix}_{suffix}"


def audit_summary(job: dict[str, str]) -> dict[str, Any]:
    case = STAGE / "diagnostics" / case_name(job["prefix"], job["seq"], int(job.get("max_frames") or 0))
    row: dict[str, Any] = {"gq_action_rows": 0, "gq_state_action_rows": 0}
    for name, path in [
        ("gq", case / "hs_gq_action_gate_rows.csv"),
        ("gq_state", case / "hs_gq_state_action_rows.csv"),
    ]:
        row[f"{name}_path"] = rel(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row[f"{name}_rows"] = int(len(df))
        row[f"{name}_calibration_modes"] = ";".join(sorted({str(v) for v in df.get("calibration_mode", pd.Series(dtype=object)).dropna().unique()}))
        row[f"{name}_intervention_forms"] = ";".join(sorted({str(v) for v in df.get("intervention_form", pd.Series(dtype=object)).dropna().unique()}))
        for col in ["gate_std", "state_delta_gate", "changed_frame_fraction_abs_gt_1e_4"]:
            vals = numeric(df, col)
            if not vals.empty:
                row[f"{name}_{col}_mean"] = float(vals.mean())
                row[f"{name}_{col}_max"] = float(vals.max())
    return row


def summarize_job(job: dict[str, str]) -> dict[str, Any]:
    seq = job["seq"]
    max_frames = int(job.get("max_frames") or 0)
    prefix = job["prefix"]
    manifest = STAGE / "diagnostics" / case_name(prefix, seq, max_frames) / "run_manifest.json"
    data = read_json(manifest)
    row: dict[str, Any] = {
        "phase": job["phase"],
        "variant": job["variant"],
        "seq": seq,
        "action": job["action"],
        "control": job.get("control", ""),
        "intervention_form": job.get("intervention_form", ""),
        "returncode": data.get("returncode"),
        "chunk_block_num": data.get("chunk_block_num", job.get("chunk_block_num", "")),
        "gq_layer_filter": data.get("gq_layer_filter", job.get("gq_layer_filter", "")),
        "trace_enable": data.get("trace_enable", job.get("trace_enable", "")),
        "action_audit_enable": data.get("action_audit_enable", job.get("action_audit_enable", "")),
        "manifest": rel(manifest),
        "candidate_output_root": rel(STAGE / "outputs" / case_name(prefix, seq, max_frames)),
        "command": data.get("command", job.get("command", "")),
    }
    row.update(audit_summary(job))
    if job["phase"] == "full_chunkblock1_layer23_notrace" and str(data.get("returncode")) == "0":
        baseline = metric_or_missing(V113_ROOT / "outputs" / f"baseline_kitti_{seq}", seq)
        candidate = metric_or_missing(STAGE / "outputs" / case_name(prefix, seq, max_frames), seq)
        for metric in [
            "full_ATE_sim3_rmse",
            "rolling_ate_p90",
            "segment_scale_log_error_median_abs",
            "adjacent_log_scale_jump_p90_abs",
            "rpe_delta1_translation_mean",
            "rpe_delta1_rotation_deg_mean",
        ]:
            b = baseline.get(metric)
            c = candidate.get(metric)
            row[f"baseline_{metric}"] = b
            row[f"candidate_{metric}"] = c
            row[f"{metric}_rel_improvement"] = rel_improvement(b, c)
    return row


def median(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [row.get("full_ATE_sim3_rmse_rel_improvement") for row in rows]
    rolling = [row.get("rolling_ate_p90_rel_improvement") for row in rows]
    med_full = median(full)
    med_roll = median(rolling)
    harms = [max(0.0, -float(v)) for v in full if v is not None and np.isfinite(float(v))]
    max_harm = float(max(harms)) if harms else None
    passed = bool(med_full is not None and med_roll is not None and max_harm is not None and med_full >= 0.03 and med_roll > 0.0 and max_harm <= 0.01)
    return {"median_full_ATE_rel_improvement": med_full, "median_rolling_p90_rel_improvement": med_roll, "max_full_ATE_harm_rel": max_harm, "pass": passed}


def default_oom_summary() -> dict[str, Any]:
    rows = read_csv_rows(DEFAULT_OOM_STAGE / "matrix_run_results.csv")
    logs = list((DEFAULT_OOM_STAGE / "scheduler_logs").glob("*.stdout.log"))
    oom_count = 0
    for path in logs:
        try:
            if "torch.OutOfMemoryError" in path.read_text(encoding="utf-8", errors="ignore"):
                oom_count += 1
        except OSError:
            pass
    return {
        "stage": rel(DEFAULT_OOM_STAGE),
        "run_result_rows": len(rows),
        "returncode_1_rows": int(sum(1 for row in rows if str(row.get("returncode")) == "1")),
        "oom_log_count": oom_count,
        "classification": "DEFAULT_FULL_GLA_OOM_ON_22GB",
    }


def report_text(summary: dict[str, Any]) -> str:
    lines = [
        "# ACL2 v118-TF HS-GLA OOM Repair Summary",
        "",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- default_full_status: `{summary['default_full_oom']['classification']}`",
        f"- repair_decision: `{summary['aggregate']['decision']}`",
        "",
        "| variant | type | median full ATE rel | median rolling p90 rel | max harm | gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary["variant_summary"]:
        g = row["pilot_gate"]
        lines.append(f"| {row['variant']} | {row['run_type']} | {g['median_full_ATE_rel_improvement']} | {g['median_rolling_p90_rel_improvement']} | {g['max_full_ATE_harm_rel']} | {g['pass']} |")
    lines += [
        "",
        "## Boundary",
        "",
        "Default HS-GLA full promotion is OOM-blocked on the available 22GB GPUs. The chunk_block_num=1, layer23 runs are reduced-memory, config-specific repairs and must not be reported as default full success.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = read_csv_rows(STAGE / "repair_job_manifest.csv")
    rows = [summarize_job(job) for job in jobs]
    full_rows = [row for row in rows if row["phase"] == "full_chunkblock1_layer23_notrace"]
    variant_summary = []
    for variant in sorted({row["variant"] for row in full_rows}):
        subset = [row for row in full_rows if row["variant"] == variant]
        run_type = "matched_control" if "RANDOM" in variant else "candidate"
        variant_summary.append({"variant": variant, "run_type": run_type, "pilot_gate": gate(subset), "returncodes": sorted({row.get("returncode") for row in subset})})
    pass_count = int(sum(1 for row in variant_summary if row["run_type"] == "candidate" and row["pilot_gate"]["pass"]))
    all_repair_ok = bool(rows and all(str(row.get("returncode")) == "0" for row in rows))
    aggregate = {
        "schema": "acl2_v118tf_hs_gla_oom_repair_aggregate_v1",
        "all_repair_returncodes_zero": all_repair_ok,
        "candidate_pilot_gate_pass_count": pass_count,
        "decision": "HS_GLA_REDUCED_REPAIR_PASS_PENDING_CONTROLS" if pass_count > 0 else ("HS_GLA_REDUCED_REPAIR_NO_PILOT_GATE_PASS" if all_repair_ok else "HS_GLA_REPAIR_RUNTIME_INCOMPLETE"),
    }
    summary = {
        "schema": "acl2_v118tf_hs_gla_oom_repair_summary_v1",
        "global_goal_achieved": False,
        "default_full_oom": default_oom_summary(),
        "aggregate": aggregate,
        "variant_summary": variant_summary,
        "rows": rows,
        "outputs": {
            "summary": rel(OUT / "hs_gla_oom_repair_summary.json"),
            "report": rel(OUT / "HS_GLA_OOM_REPAIR_REPORT.md"),
            "rows": rel(OUT / "hs_gla_oom_repair_rows.csv"),
        },
    }
    write_csv(OUT / "hs_gla_oom_repair_rows.csv", rows)
    write_json(OUT / "hs_gla_oom_repair_summary.json", summary)
    write_text(OUT / "HS_GLA_OOM_REPAIR_REPORT.md", report_text(summary))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-HSGLA-OOMRepair",
            "surface_or_branch": "HS-GLA",
            "status": aggregate["decision"],
            "artifact": rel(OUT / "hs_gla_oom_repair_summary.json"),
            "notes": "Default HS-GLA full matrix OOMed on 22GB; repair uses max128 action smoke plus chunk_block_num=1 layer23 reduced full pilots",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
