#!/usr/bin/env python3
"""Summarize the v118 Stage4-R5 HS-LA internal-QK action smoke run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r5_hs_la_internal_qk_only"
OUT = STAGE / "summary"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
CASE = "stage4_r5_hs_la1_internal_qk_only_tiny_tight_max128_kitti_00"


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


def stat(df: pd.DataFrame, col: str) -> dict[str, Any]:
    values = numeric(df, col)
    if values.empty:
        return {f"{col}_min": None, f"{col}_p10": None, f"{col}_p90": None, f"{col}_max": None}
    return {
        f"{col}_min": float(values.min()),
        f"{col}_p10": float(values.quantile(0.1)),
        f"{col}_p90": float(values.quantile(0.9)),
        f"{col}_max": float(values.max()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = STAGE / "diagnostics" / CASE / "run_manifest.json"
    audit_path = STAGE / "diagnostics" / CASE / "hs_la_action_rows.csv"
    eval_path = STAGE / "outputs" / CASE / "eval_summary.json"
    manifest = read_json(manifest_path)
    eval_summary = read_json(eval_path)
    df = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    selected = df[df.get("scope", pd.Series(dtype=str)).astype(str) == "pose_query"].copy() if not df.empty else pd.DataFrame()
    bias_abs_max = float(numeric(selected, "bias_max").abs().max()) if not selected.empty else None
    bias_min_abs_max = float(numeric(selected, "bias_min").abs().max()) if not selected.empty else None
    bias_bound = max(v for v in [bias_abs_max, bias_min_abs_max] if v is not None) if not selected.empty else None
    delta = numeric(selected, "selected_output_delta_rel_norm")
    lambda_values = numeric(selected, "lambda_risk")
    mu_values = numeric(selected, "mu_stable")
    smoke_pass = bool(
        manifest.get("returncode") == 0
        and len(selected) > 0
        and bias_bound is not None
        and bias_bound <= 0.050001
        and not delta.empty
        and float(delta.max()) > 0.0
        and float(delta.max()) < 0.05
        and not lambda_values.empty
        and float(lambda_values.abs().max()) == 0.0
        and not mu_values.empty
        and float(mu_values.abs().max()) == 0.0
    )
    summary = {
        "schema": "acl2_v118tf_stage4_r5_hs_la_internal_qk_smoke_summary_v1",
        "status": "SMOKE_PASS_ACTION_FIDELITY_FULL_PILOT_PENDING" if smoke_pass else "SMOKE_NO_GO_ACTION_FIDELITY_FAILED",
        "smoke_pass": smoke_pass,
        "global_goal_achieved": False,
        "case": CASE,
        "manifest_returncode": manifest.get("returncode"),
        "action": manifest.get("action"),
        "control": manifest.get("control"),
        "selected_pose_query_rows": int(len(selected)),
        "total_audit_rows": int(len(df)),
        "bias_bound_abs_max": bias_bound,
        "lambda_risk_abs_max": float(lambda_values.abs().max()) if not lambda_values.empty else None,
        "mu_stable_abs_max": float(mu_values.abs().max()) if not mu_values.empty else None,
        **stat(selected, "bias_std"),
        **stat(selected, "bias_abs_mean"),
        **stat(selected, "attention_entropy_before"),
        **stat(selected, "attention_entropy_after"),
        **stat(selected, "selected_output_delta_rel_norm"),
        "artifacts": {
            "manifest": rel(manifest_path),
            "audit": rel(audit_path),
            "eval_summary": rel(eval_path),
            "summary": rel(OUT / "stage4_r5_hs_la_internal_qk_smoke_summary.json"),
            "report": rel(OUT / "STAGE4_R5_HS_LA_INTERNAL_QK_SMOKE_REPORT.md"),
        },
        "boundary": "Smoke verifies the internal-QK-only HS-LA action path and bounded action fidelity only. Full 00/02 geometry pilot and Stage4 controls are still required.",
    }
    write_json(OUT / "stage4_r5_hs_la_internal_qk_smoke_summary.json", summary)
    report = "\n".join(
        [
            "# ACL2 v118-TF Stage4-R5 HS-LA Internal-QK Smoke",
            "",
            f"- status: `{summary['status']}`",
            f"- smoke_pass: `{summary['smoke_pass']}`",
            f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
            f"- selected_pose_query_rows: `{summary['selected_pose_query_rows']}`",
            f"- bias_bound_abs_max: `{summary['bias_bound_abs_max']}`",
            f"- selected_output_delta_rel_norm_p90: `{summary.get('selected_output_delta_rel_norm_p90')}`",
            f"- selected_output_delta_rel_norm_max: `{summary.get('selected_output_delta_rel_norm_max')}`",
            "",
            "## Boundary",
            "",
            summary["boundary"],
        ]
    )
    write_text(OUT / "STAGE4_R5_HS_LA_INTERNAL_QK_SMOKE_REPORT.md", report)
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R5-Smoke",
            "surface_or_branch": "HS-LA",
            "status": summary["status"],
            "artifact": rel(OUT / "stage4_r5_hs_la_internal_qk_smoke_summary.json"),
            "notes": "HS-LA internal-QK-only tiny/tight max128 smoke; full pilot and controls pending",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
