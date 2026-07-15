#!/usr/bin/env python3
"""Summarize v118 Stage3-R5 HS-LA selected-query internal QK audit."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage3_r5_hs_la_internal_qk_audit"
OUT = STAGE / "summary"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
PREFIX = "stage3_r5_hs_la0_internal_qk_audit"
SEQS = ["00", "02"]
STAGE1_PREFIX = RESULT_ROOT / "stage1_causal_object_track_sidecar/object_track_prefix_rows.parquet"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(val) for val in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def signal_stats(df: pd.DataFrame, col: str) -> dict[str, Any]:
    values = numeric(df.get(col, pd.Series(dtype=float))).replace([np.inf, -np.inf], np.nan).dropna()
    total = int(len(df))
    if total == 0 or values.empty:
        return {
            f"{col}_coverage": 0.0,
            f"{col}_p10": None,
            f"{col}_p90": None,
            f"{col}_span": 0.0,
            f"{col}_finite_count": int(values.shape[0]),
        }
    p10 = float(np.percentile(values.to_numpy(dtype=float), 10))
    p90 = float(np.percentile(values.to_numpy(dtype=float), 90))
    return {
        f"{col}_coverage": float(values.shape[0]) / float(total),
        f"{col}_p10": p10,
        f"{col}_p90": p90,
        f"{col}_span": float(p90 - p10),
        f"{col}_finite_count": int(values.shape[0]),
    }


def temporal_nonconstant(df: pd.DataFrame, col: str) -> tuple[bool, int, int]:
    work = df.copy()
    work["_bucket"] = (
        work.get("seq", pd.Series([""] * len(work), index=work.index)).astype(str)
        + ":chunk:"
        + numeric(work.get("chunk_idx", pd.Series([-1] * len(work), index=work.index))).fillna(-1).astype(int).astype(str)
    )
    eligible = 0
    varied = 0
    for _bucket, group in work.groupby("_bucket"):
        values = numeric(group[col]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 2:
            continue
        eligible += 1
        if float(values.max() - values.min()) > 1e-9:
            varied += 1
    return bool(eligible > 0 and varied / eligible >= 0.50), eligible, varied


def semantic_persistence_by_chunk() -> dict[tuple[str, int], float]:
    if not STAGE1_PREFIX.exists():
        return {}
    prefix = pd.read_parquet(STAGE1_PREFIX, columns=["seq", "frame_id", "semantic_persistence_prefix"])
    prefix["frame_id"] = numeric(prefix["frame_id"]).astype(int)
    prefix["semantic_persistence_prefix"] = numeric(prefix["semantic_persistence_prefix"])
    out: dict[tuple[str, int], float] = {}
    for seq in SEQS:
        seq_rows = prefix[prefix["seq"].astype(str) == seq]
        if seq_rows.empty:
            continue
        chunks = pd.concat(
            [
                load_audit_rows(seq)[["seq", "chunk_idx", "chunk_start", "chunk_end"]].drop_duplicates(),
            ],
            ignore_index=True,
        )
        for row in chunks.itertuples(index=False):
            values = seq_rows[
                (seq_rows["frame_id"] >= int(row.chunk_start))
                & (seq_rows["frame_id"] < int(row.chunk_end))
            ]["semantic_persistence_prefix"].dropna()
            if not values.empty:
                out[(str(row.seq), int(row.chunk_idx))] = float(values.mean())
    return out


def load_audit_rows(seq: str) -> pd.DataFrame:
    path = STAGE / "diagnostics" / f"{PREFIX}_full_kitti_{seq}" / "hs_la_action_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["seq"] = df.get("seq", seq).astype(str).str.zfill(2)
    return df


def manifest_row(seq: str) -> dict[str, Any]:
    manifest = STAGE / "diagnostics" / f"{PREFIX}_full_kitti_{seq}" / "run_manifest.json"
    data = read_json(manifest)
    return {
        "seq": seq,
        "manifest": rel(manifest),
        "returncode": data.get("returncode"),
        "action": data.get("action"),
        "control": data.get("control"),
        "gpu": data.get("gpu"),
        "command": data.get("command"),
    }


def summarize_seq(seq: str, persistence: dict[tuple[str, int], float]) -> tuple[dict[str, Any], pd.DataFrame]:
    df = load_audit_rows(seq)
    if df.empty:
        return {"seq": seq, "row_count_total": 0, "stage3_r5_ready_contribution": False}, df
    selected = df[df["scope"].astype(str) == "pose_query"].copy()
    selected["chunk_semantic_persistence_mean"] = [
        persistence.get((str(row.seq).zfill(2), int(row.chunk_idx)), np.nan)
        for row in selected.itertuples(index=False)
    ]
    candidate_col = "qk_logit_top1_minus_top2_mean"
    reliability_col = "attention_read_reliability_mean"
    cand_temporal, cand_eligible, cand_varied = temporal_nonconstant(selected, candidate_col)
    rel_temporal, rel_eligible, rel_varied = temporal_nonconstant(selected, reliability_col)
    delta = numeric(selected.get("selected_output_delta_rel_norm", pd.Series(dtype=float))).replace([np.inf, -np.inf], np.nan).dropna()
    candidate_values = numeric(selected[candidate_col]).replace([np.inf, -np.inf], np.nan)
    reliability_values = numeric(selected[reliability_col]).replace([np.inf, -np.inf], np.nan)
    persistence_values = numeric(selected["chunk_semantic_persistence_mean"]).replace([np.inf, -np.inf], np.nan)
    corr_frame = pd.DataFrame(
        {
            "candidate": candidate_values,
            "reliability": reliability_values,
            "persistence": persistence_values,
        }
    ).dropna()
    if len(corr_frame) >= 3:
        candidate_semantic_spearman = float(corr_frame["candidate"].corr(corr_frame["persistence"], method="spearman"))
        reliability_semantic_spearman = float(corr_frame["reliability"].corr(corr_frame["persistence"], method="spearman"))
    else:
        candidate_semantic_spearman = None
        reliability_semantic_spearman = None
    row = {
        "seq": seq,
        "row_count_total": int(len(df)),
        "selected_pose_query_row_count": int(len(selected)),
        "selected_scope_fraction_total": float(len(selected)) / float(len(df)) if len(df) else 0.0,
        "selected_query_scope_coverage": 1.0 if len(selected) else 0.0,
        **signal_stats(selected, candidate_col),
        **signal_stats(selected, reliability_col),
        "candidate_temporal_nonconstant_gate": cand_temporal,
        "candidate_temporal_bucket_count": cand_eligible,
        "candidate_temporal_varied_bucket_count": cand_varied,
        "reliability_temporal_nonconstant_gate": rel_temporal,
        "reliability_temporal_bucket_count": rel_eligible,
        "reliability_temporal_varied_bucket_count": rel_varied,
        "selected_output_delta_rel_norm_max_abs": float(delta.abs().max()) if not delta.empty else None,
        "noop_selected_delta_gate": bool((not delta.empty) and float(delta.abs().max()) <= 1e-12),
        "candidate_semantic_persistence_spearman": candidate_semantic_spearman,
        "reliability_semantic_persistence_spearman": reliability_semantic_spearman,
        "semantic_persistence_correlation_gate": bool(
            candidate_semantic_spearman is not None
            and reliability_semantic_spearman is not None
            and abs(candidate_semantic_spearman) < 0.95
            and abs(reliability_semantic_spearman) < 0.95
        ),
    }
    row["candidate_gate"] = bool(
        row[f"{candidate_col}_coverage"] >= 0.90
        and row[f"{candidate_col}_span"] >= 0.10
        and row["candidate_temporal_nonconstant_gate"]
    )
    row["reliability_gate"] = bool(
        row[f"{reliability_col}_coverage"] >= 0.90
        and row[f"{reliability_col}_span"] >= 0.10
        and row["reliability_temporal_nonconstant_gate"]
    )
    row["stage3_r5_ready_contribution"] = bool(
        row["candidate_gate"]
        and row["reliability_gate"]
        and row["noop_selected_delta_gate"]
        and row["semantic_persistence_correlation_gate"]
    )
    return row, selected


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v118-TF Stage3-R5 HS-LA Selected-Query Internal QK Audit",
        "",
        f"- stage3_r5_decision: `{summary['stage3_r5_decision']}`",
        f"- hs_la_selected_query_ready_for_stage4: `{summary['hs_la_selected_query_ready_for_stage4']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| seq | selected rows | cand span | rel span | cand rho sem | rel rho sem | noop delta max | ready contribution |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | {row['selected_pose_query_row_count']} | "
            f"{row.get('qk_logit_top1_minus_top2_mean_span')} | "
            f"{row.get('attention_read_reliability_mean_span')} | "
            f"{row.get('candidate_semantic_persistence_spearman')} | "
            f"{row.get('reliability_semantic_persistence_spearman')} | "
            f"{row.get('selected_output_delta_rel_norm_max_abs')} | "
            f"{row.get('stage3_r5_ready_contribution')} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "R5 repairs the HS-LA selected-query internal-audit blocker only. It does not prove geometry improvement and does not claim semantic-aware causality. Non-selected large-query local attention calls remain outside this selected-query branch. Stage4 carrier attribution is still required before any HS-LA runtime policy can be promoted.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    persistence = semantic_persistence_by_chunk()
    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for seq in SEQS:
        row, selected = summarize_seq(seq, persistence)
        rows.append(row)
        if not selected.empty:
            selected_frames.append(selected)
    manifests = [manifest_row(seq) for seq in SEQS]
    all_manifests_ok = all(row.get("returncode") == 0 for row in manifests)
    ready = bool(all_manifests_ok and rows and all(row.get("stage3_r5_ready_contribution") for row in rows))
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    selected_out = OUT / "stage3_r5_hs_la_selected_query_rows.csv"
    if not selected_all.empty:
        selected_all.to_csv(selected_out, index=False)
    else:
        write_csv(selected_out, [])
    summary = {
        "schema": "acl2_v118tf_stage3_r5_hs_la_internal_qk_audit_summary_v1",
        "stage3_r5_decision": (
            "PARTIAL_BRANCH_READY_HS_LA_SELECTED_QUERY_QK_ONLY"
            if ready
            else "NO_GO_HS_LA_SELECTED_QUERY_AUDIT_INCOMPLETE_OR_GATE_FAILED"
        ),
        "hs_la_selected_query_ready_for_stage4": ready,
        "ready_for_stage4_branches": ["HS-LA"] if ready else [],
        "global_goal_achieved": False,
        "all_manifests_returncode_0": all_manifests_ok,
        "candidate_col": "qk_logit_top1_minus_top2_mean",
        "reliability_col": "attention_read_reliability_mean",
        "seq_rows": rows,
        "manifest_rows": manifests,
        "input_artifacts": {
            "stage1_object_track_prefix_rows": rel(STAGE1_PREFIX),
            "seq00_audit": rel(STAGE / "diagnostics" / f"{PREFIX}_full_kitti_00" / "hs_la_action_rows.csv"),
            "seq02_audit": rel(STAGE / "diagnostics" / f"{PREFIX}_full_kitti_02" / "hs_la_action_rows.csv"),
        },
        "outputs": {
            "summary": rel(OUT / "stage3_r5_hs_la_internal_qk_audit_summary.json"),
            "readiness_rows": rel(OUT / "stage3_r5_hs_la_readiness_rows.csv"),
            "selected_query_rows": rel(selected_out),
            "manifest_summary": rel(OUT / "stage3_r5_hs_la_manifest_summary.csv"),
            "report": rel(OUT / "STAGE3_R5_HS_LA_INTERNAL_QK_AUDIT_REPORT.md"),
        },
        "boundary": "HS-LA selected-query readiness only; Stage4 carrier attribution remains required before policy promotion.",
    }
    write_csv(OUT / "stage3_r5_hs_la_readiness_rows.csv", rows)
    write_csv(OUT / "stage3_r5_hs_la_manifest_summary.csv", manifests)
    write_json(OUT / "stage3_r5_hs_la_internal_qk_audit_summary.json", summary)
    write_text(OUT / "STAGE3_R5_HS_LA_INTERNAL_QK_AUDIT_REPORT.md", report_text(summary, rows))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage3-R5",
            "surface_or_branch": "HS-LA",
            "status": summary["stage3_r5_decision"],
            "artifact": rel(OUT / "stage3_r5_hs_la_internal_qk_audit_summary.json"),
            "notes": "HS-LA selected-query internal QK/read-focus audit; geometry and semantic-causality not claimed",
        }
    )
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
