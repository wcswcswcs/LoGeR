#!/usr/bin/env python3
"""Build the v118 HS-LA Stage4 candidate/control matrix."""

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
OUT = RESULT_ROOT / "stage4_hs_la_control_matrix" / "summary"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
SEQS = ["00", "02"]


RUNS = [
    ("HLA1", "candidate", RESULT_ROOT / "stage4_r5_hs_la_internal_qk_only", "stage4_r5_hs_la1_internal_qk_only_tiny_tight", "internal_qk_only"),
    ("HLA1", "random", RESULT_ROOT / "stage4_r5_hs_la_internal_qk_only_random_control", "stage4_r5_hs_la1_internal_qk_only_tiny_tight_random_logit", "same_magnitude_random_logit"),
    ("HLA1", "reverse", RESULT_ROOT / "stage4_r5_hs_la_internal_qk_only_reverse_control", "stage4_r5_hs_la1_internal_qk_only_tiny_tight_reverse_internal", "internal_qk_reverse"),
    ("HLA2", "candidate", RESULT_ROOT / "stage4_r7_hs_la_semantic_role_only", "stage4_r7_hs_la2_semantic_role_only_tiny_tight", "semantic_role_only"),
    ("HLA2", "random", RESULT_ROOT / "stage4_r7_hs_la_semantic_role_only_random_control", "stage4_r7_hs_la2_semantic_role_only_tiny_tight_random_logit", "same_magnitude_random_logit"),
    ("HLA2", "reverse", RESULT_ROOT / "stage4_r7_hs_la_semantic_role_only_reverse_control", "stage4_r7_hs_la2_semantic_role_only_tiny_tight_reverse_semantic", "reverse_semantic_role"),
    ("HLA3", "candidate", RESULT_ROOT / "stage4_r9_hs_la_dynamic_mismatch_suppress", "stage4_r9_hs_la3_dynamic_mismatch_suppress_tiny_tight", "dynamic_mismatch_suppress"),
    ("HLA3", "random", RESULT_ROOT / "stage4_r9_hs_la_dynamic_mismatch_suppress_random_control", "stage4_r9_hs_la3_dynamic_mismatch_suppress_tiny_tight_random_logit", "same_magnitude_random_logit"),
    ("HLA3", "reverse", RESULT_ROOT / "stage4_r9_hs_la_dynamic_mismatch_suppress_reverse_control", "stage4_r9_hs_la3_dynamic_mismatch_suppress_tiny_tight_reverse_semantic", "reverse_semantic_role"),
    ("HLA4", "candidate", RESULT_ROOT / "stage4_r8_hs_la_persistent_aligned_boost", "stage4_r8_hs_la4_persistent_aligned_boost_tiny_tight", "persistent_aligned_boost"),
    ("HLA4", "random", RESULT_ROOT / "stage4_r8_hs_la_persistent_aligned_boost_random_control", "stage4_r8_hs_la4_persistent_aligned_boost_tiny_tight_random_logit", "same_magnitude_random_logit"),
    ("HLA4", "reverse", RESULT_ROOT / "stage4_r8_hs_la_persistent_aligned_boost_reverse_control", "stage4_r8_hs_la4_persistent_aligned_boost_tiny_tight_reverse_semantic", "reverse_semantic_role"),
    ("HLA5", "candidate", RESULT_ROOT / "stage4_r6_hs_la_full_calibrated_selected_query", "stage4_r6_hs_la5_full_calibrated_selected_query_tiny_tight", "full_calibrated_selected_query"),
    ("HLA5", "random", RESULT_ROOT / "stage4_r6_hs_la_full_calibrated_selected_query_random_control", "stage4_r6_hs_la5_full_calibrated_selected_query_tiny_tight_random_logit", "same_magnitude_random_logit"),
    ("HLA5", "reverse", RESULT_ROOT / "stage4_r6_hs_la_full_calibrated_selected_query_reverse_control", "stage4_r6_hs_la5_full_calibrated_selected_query_tiny_tight_reverse_semantic", "reverse_semantic_role"),
]


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def quantiles(df: pd.DataFrame, col: str, prefix: str) -> dict[str, Any]:
    vals = numeric(df, col)
    if vals.empty:
        return {f"{prefix}_{col}_mean": None, f"{prefix}_{col}_p90": None, f"{prefix}_{col}_max": None}
    return {
        f"{prefix}_{col}_mean": float(vals.mean()),
        f"{prefix}_{col}_p90": float(vals.quantile(0.9)),
        f"{prefix}_{col}_max": float(vals.max()),
    }


def audit_summary(stage_root: Path, prefix: str, seq: str) -> dict[str, Any]:
    path = stage_root / "diagnostics" / f"{prefix}_full_kitti_{seq}" / "hs_la_action_rows.csv"
    if not path.exists():
        return {"audit_rows": 0, "selected_pose_query_rows": 0, "audit_path": rel(path)}
    df = pd.read_csv(path)
    selected = df[df.get("scope", pd.Series(dtype=str)).astype(str) == "pose_query"].copy()
    row: dict[str, Any] = {
        "audit_rows": int(len(df)),
        "selected_pose_query_rows": int(len(selected)),
        "audit_path": rel(path),
        "calibration_modes": ";".join(sorted({str(v) for v in df.get("calibration_mode", pd.Series(dtype=object)).dropna().unique()})),
        "controls_seen": ";".join(sorted({str(v) for v in df.get("control", pd.Series(dtype=object)).dropna().unique()})),
    }
    for col in ["bias_abs_mean", "selected_output_delta_rel_norm"]:
        row.update(quantiles(selected, col, "audit"))
    return row


def summarize_run(branch: str, run_type: str, stage_root: Path, prefix: str, control: str, seq: str) -> dict[str, Any]:
    manifest_path = stage_root / "diagnostics" / f"{prefix}_full_kitti_{seq}" / "run_manifest.json"
    manifest = read_json(manifest_path)
    baseline_root = V113_ROOT / "outputs" / f"baseline_kitti_{seq}"
    candidate_root = stage_root / "outputs" / f"{prefix}_full_kitti_{seq}"
    base = summarize_sequence(baseline_root, seq)
    cand = summarize_sequence(candidate_root, seq)
    row: dict[str, Any] = {
        "branch": branch,
        "run_type": run_type,
        "seq": seq,
        "control": control,
        "prefix": prefix,
        "returncode": manifest.get("returncode"),
        "manifest": rel(manifest_path),
        "candidate_output_root": rel(candidate_root),
    }
    for metric in [
        "full_ATE_sim3_rmse",
        "rolling_ate_p90",
        "segment_scale_log_error_median_abs",
        "adjacent_log_scale_jump_p90_abs",
        "rpe_delta1_translation_mean",
        "rpe_delta1_rotation_deg_mean",
    ]:
        b = base.get(metric)
        c = cand.get(metric)
        row[f"baseline_{metric}"] = b
        row[f"candidate_{metric}"] = c
        row[f"{metric}_rel_improvement"] = rel_improvement(b, c)
    row.update(audit_summary(stage_root, prefix, seq))
    return row


def median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def hla6_smoke() -> dict[str, Any]:
    stage = RESULT_ROOT / "stage4_r10_hs_la_patch_query_control"
    prefix = "stage4_r10_hs_la6_patch_query_control_tiny_tight"
    case = stage / "diagnostics" / f"{prefix}_max128_kitti_00"
    audit = case / "hs_la_action_rows.csv"
    manifest = read_json(case / "run_manifest.json")
    scopes: dict[str, int] = {}
    selected_rows = 0
    delta_rows = 0
    if audit.exists():
        df = pd.read_csv(audit)
        scopes = {str(k): int(v) for k, v in df.get("scope", pd.Series(dtype=str)).value_counts(dropna=False).items()}
        selected_rows = int(pd.to_numeric(df.get("selected_query_count", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum())
        delta_rows = int(pd.to_numeric(df.get("selected_output_delta_rel_norm", pd.Series(dtype=float)), errors="coerce").dropna().shape[0])
    return {
        "branch": "HLA6",
        "run_type": "smoke",
        "returncode": manifest.get("returncode"),
        "manifest": rel(case / "run_manifest.json"),
        "audit_path": rel(audit),
        "scope_counts": scopes,
        "selected_query_positive_rows": selected_rows,
        "delta_rows": delta_rows,
        "effective_action": bool(selected_rows > 0 and delta_rows > 0),
        "status": "STRUCTURAL_NO_EFFECT_PATCH_QUERY_NOT_REACHED" if selected_rows == 0 else "SMOKE_EFFECTIVE",
    }


def report_text(summary: dict[str, Any]) -> str:
    agg = summary["aggregate"]
    lines = [
        "# ACL2 v118-TF HS-LA Stage4 Control Matrix",
        "",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- candidate_pilot_gate_pass_count: `{agg['candidate_pilot_gate_pass_count']}`",
        f"- hla6_effective_action: `{summary['hla6_smoke']['effective_action']}`",
        "",
        "| branch | candidate median full ATE rel | candidate median rolling p90 rel | max harm | random median full ATE rel | reverse median full ATE rel | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for branch in ["HLA1", "HLA2", "HLA3", "HLA4", "HLA5"]:
        row = agg["branch_summary"][branch]
        lines.append(
            f"| {branch} | {row.get('candidate_median_full_ATE_rel')} | {row.get('candidate_median_rolling_p90_rel')} | "
            f"{row.get('candidate_max_harm')} | {row.get('random_median_full_ATE_rel')} | "
            f"{row.get('reverse_median_full_ATE_rel')} | {row.get('decision')} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "No HS-LA branch in this matrix reaches the plan 13.2 pilot gate. Controls are recorded for attribution, but no semantic-aware geometry success is claimed.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for branch, run_type, stage_root, prefix, control in RUNS:
        for seq in SEQS:
            rows.append(summarize_run(branch, run_type, stage_root, prefix, control, seq))

    branch_summary: dict[str, dict[str, Any]] = {}
    for branch in ["HLA1", "HLA2", "HLA3", "HLA4", "HLA5"]:
        subset = [row for row in rows if row["branch"] == branch]
        by_type = {kind: [row for row in subset if row["run_type"] == kind] for kind in ["candidate", "random", "reverse"]}
        cand_full = [row.get("full_ATE_sim3_rmse_rel_improvement") for row in by_type["candidate"]]
        cand_roll = [row.get("rolling_ate_p90_rel_improvement") for row in by_type["candidate"]]
        cand_harm = [max(0.0, -float(v)) for v in cand_full if v is not None and np.isfinite(v)]
        cand_pass = bool(
            median(cand_full) is not None
            and median(cand_roll) is not None
            and median(cand_full) >= 0.03
            and median(cand_roll) > 0.0
            and (max(cand_harm) if cand_harm else 0.0) <= 0.01
        )
        branch_summary[branch] = {
            "candidate_median_full_ATE_rel": median(cand_full),
            "candidate_median_rolling_p90_rel": median(cand_roll),
            "candidate_max_harm": float(max(cand_harm)) if cand_harm else None,
            "candidate_pilot_gate_pass": cand_pass,
            "random_median_full_ATE_rel": median([row.get("full_ATE_sim3_rmse_rel_improvement") for row in by_type["random"]]),
            "reverse_median_full_ATE_rel": median([row.get("full_ATE_sim3_rmse_rel_improvement") for row in by_type["reverse"]]),
            "decision": "PASS_PILOT_GATE_PENDING_CAUSAL_CONTROLS" if cand_pass else "NO_GO_PILOT_GATE_FAILED",
        }

    hla6 = hla6_smoke()
    aggregate = {
        "schema": "acl2_v118tf_hs_la_stage4_control_matrix_aggregate_v1",
        "branch_summary": branch_summary,
        "candidate_pilot_gate_pass_count": int(sum(1 for row in branch_summary.values() if row["candidate_pilot_gate_pass"])),
        "control_run_count": int(sum(1 for row in rows if row["run_type"] in {"random", "reverse"})),
        "all_runtime_returncodes_zero": bool(all(str(row.get("returncode")) == "0" for row in rows)),
        "decision": "HS_LA_STAGE4_NO_PILOT_GATE_PASS_CONTROLS_RECORDED",
    }
    summary = {
        "schema": "acl2_v118tf_hs_la_stage4_control_matrix_summary_v1",
        "global_goal_achieved": False,
        "aggregate": aggregate,
        "hla6_smoke": hla6,
        "rows": rows,
        "outputs": {
            "summary": rel(OUT / "hs_la_stage4_control_matrix_summary.json"),
            "report": rel(OUT / "HS_LA_STAGE4_CONTROL_MATRIX_REPORT.md"),
            "rows": rel(OUT / "hs_la_stage4_control_matrix_rows.csv"),
        },
    }
    write_csv(OUT / "hs_la_stage4_control_matrix_rows.csv", rows)
    write_json(OUT / "hs_la_stage4_control_matrix_summary.json", summary)
    write_text(OUT / "HS_LA_STAGE4_CONTROL_MATRIX_REPORT.md", report_text(summary))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-HSLA-ControlMatrix",
            "surface_or_branch": "HS-LA",
            "status": aggregate["decision"],
            "artifact": rel(OUT / "hs_la_stage4_control_matrix_summary.json"),
            "notes": "HLA1-HLA5 candidate/random/reverse controls summarized; HLA6 smoke had no effective patch-query action; v118 global goal not achieved",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
