#!/usr/bin/env python3
"""Audit a v118 Stage3-R4 internal-reliability repair attempt.

This script is intentionally additive. It does not rewrite the original v118
Stage3 or final decision artifacts. It tests whether non-semantic internal
signals already materialized in earlier HorizonStream/LingBot artifacts can
legitimately unblock any v118 branch.
"""

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
OUT = RESULT_ROOT / "stage3_r4_internal_reliability_repair"
STAGE2_SUMMARY = RESULT_ROOT / "stage2_memory_entry_provenance/stage2_memory_entry_provenance_summary.json"
STAGE3_ROWS = RESULT_ROOT / "stage3_internal_signal_readiness/stage3_surface_signal_gate_rows.csv"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
V115_CUES = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control/stage2_alignment_cues"


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def stats(series: pd.Series) -> dict[str, Any]:
    values = numeric(series)
    total = int(len(values))
    finite = values[np.isfinite(values)]
    if total == 0:
        return {"row_count": 0, "nonmissing_count": 0, "coverage": 0.0, "p10": None, "p90": None, "span": 0.0}
    if finite.empty:
        return {"row_count": total, "nonmissing_count": 0, "coverage": 0.0, "p10": None, "p90": None, "span": 0.0}
    p10, p90 = np.percentile(finite.to_numpy(dtype=float), [10, 90])
    return {
        "row_count": total,
        "nonmissing_count": int(finite.shape[0]),
        "coverage": float(finite.shape[0]) / float(total),
        "p10": float(p10),
        "p90": float(p90),
        "span": float(p90 - p10),
    }


def temporal_nonconstant(df: pd.DataFrame, value_col: str) -> bool:
    if df.empty or value_col not in df:
        return False
    work = df.copy()
    if "chunk_idx" in work:
        seq = work.get("seq", pd.Series([""] * len(work), index=work.index)).astype(str)
        chunk = numeric(work["chunk_idx"]).fillna(-1).astype(int).astype(str)
        work["_bucket"] = seq + ":chunk:" + chunk
    elif "frame_idx" in work:
        seq = work.get("seq", pd.Series([""] * len(work), index=work.index)).astype(str)
        frame = numeric(work["frame_idx"]).fillna(0).astype(int)
        work["_bucket"] = seq + ":seg:" + (frame // 500).astype(str)
    else:
        work["_bucket"] = "all"

    eligible = 0
    varied = 0
    for _bucket, group in work.groupby("_bucket"):
        values = numeric(group[value_col]).dropna()
        if len(values) < 2:
            continue
        eligible += 1
        if float(values.max() - values.min()) > 1e-9:
            varied += 1
    return eligible > 0 and varied / eligible >= 0.50


def signal_gate(df: pd.DataFrame, candidate_col: str, reliability_col: str) -> dict[str, Any]:
    cand = stats(df[candidate_col]) if candidate_col in df else stats(pd.Series(dtype=float))
    rels = stats(df[reliability_col]) if reliability_col in df else stats(pd.Series(dtype=float))
    temporal = temporal_nonconstant(df, candidate_col)
    return {
        "candidate_row_count": cand["row_count"],
        "candidate_nonmissing_coverage": cand["coverage"],
        "candidate_p10": cand["p10"],
        "candidate_p90": cand["p90"],
        "candidate_p10_p90_span": cand["span"],
        "candidate_gate": bool(cand["coverage"] >= 0.90 and cand["span"] >= 0.10),
        "reliability_row_count": rels["row_count"],
        "reliability_nonmissing_coverage": rels["coverage"],
        "reliability_p10": rels["p10"],
        "reliability_p90": rels["p90"],
        "reliability_p10_p90_span": rels["span"],
        "reliability_gate": bool(rels["coverage"] >= 0.90 and rels["span"] >= 0.10),
        "temporal_bucket_nonconstant_gate": temporal,
    }


def make_branch_row(
    *,
    branch: str,
    surface: str,
    source_path: Path | None,
    df: pd.DataFrame,
    candidate_col: str,
    reliability_col: str,
    candidate_mode: str,
    reliability_mode: str,
    semantic_free: bool,
    stage2_ready: bool,
    operation_specific: bool,
    stage4_partner_ready: bool = True,
    blocker_notes: list[str] | None = None,
) -> dict[str, Any]:
    gate = signal_gate(df, candidate_col, reliability_col)
    blockers = list(blocker_notes or [])
    if not stage2_ready:
        blockers.append("stage2_surface_not_ready")
    if not gate["candidate_gate"]:
        blockers.append("candidate_gate_failed")
    if not gate["reliability_gate"]:
        blockers.append("reliability_gate_failed")
    if not gate["temporal_bucket_nonconstant_gate"]:
        blockers.append("candidate_constant_or_unavailable_within_temporal_bucket")
    if not semantic_free:
        blockers.append("semantic_terms_used")
    if not operation_specific:
        blockers.append("not_operation_specific_reliability")
    if not stage4_partner_ready:
        blockers.append("stage4_required_partner_branch_not_ready")
    ready = (
        stage2_ready
        and gate["candidate_gate"]
        and gate["reliability_gate"]
        and gate["temporal_bucket_nonconstant_gate"]
        and semantic_free
        and operation_specific
        and stage4_partner_ready
    )
    return {
        "schema": "acl2_v118tf_stage3_r4_branch_readiness_row_v1",
        "branch": branch,
        "surface": surface,
        "source": rel(source_path) if source_path is not None else "",
        "candidate_col": candidate_col,
        "reliability_col": reliability_col,
        "candidate_mode": candidate_mode,
        "reliability_mode": reliability_mode,
        "semantic_free": semantic_free,
        "stage2_ready": stage2_ready,
        "operation_specific": operation_specific,
        "stage4_partner_ready": stage4_partner_ready,
        "stage3_r4_ready_for_stage4": ready,
        "blockers": ";".join(blockers),
        **gate,
    }


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
    kept.append({k: row.get(k, "") for k in fields})
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v118-TF Stage3-R4 Internal Reliability Repair",
        "",
        f"- stage3_r4_complete: `{summary['stage3_r4_complete']}`",
        f"- ready_for_stage4_count: `{summary['ready_for_stage4_count']}`",
        f"- ready_for_stage4_branches: `{', '.join(summary['ready_for_stage4_branches']) or 'none'}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| branch | ready | cand span | rel span | blockers |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['branch']} | {row['stage3_r4_ready_for_stage4']} | "
            f"{row['candidate_p10_p90_span']} | {row['reliability_p10_p90_span']} | {row['blockers']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "R4 found one branch-level Stage3 repair: HS-HG can be treated as a non-semantic, operation-specific head-output reliability branch because both candidate and reliability signals come from internal head traces (`internal_head_q_std` and `internal_head_q_mean`).",
        "",
        "This does not achieve v118. It only reopens HS-HG for Stage4 counterfactual attribution. Existing v115 HS-HG full-pilot geometry improves full ATE by about 0.3% median, below the 5% geometry gate, and semantic causality controls were not run.",
        "",
        "HS-LA remains blocked because no selected-query local-attention reliability row is present. HS-GW/HS-GR remain blocked because direct KDA gamma or fixed-reference state reliability is still unavailable. HS-MR has usable readout uncertainty, but the plan requires pairing it with HS-LA/HS-GW/HS-GR, not HS-HG alone.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage2 = read_json(STAGE2_SUMMARY)
    if not stage2.get("stage2_complete"):
        raise RuntimeError("Stage2 incomplete; refusing R4 audit")
    ready_surfaces = set(stage2.get("ready_surfaces", []))

    head_path = V115_CUES / "hs_head_reliability_rows.csv"
    gla_path = V115_CUES / "hs_gla_state_quality_rows.csv"
    mrt_path = V115_CUES / "hs_mrt_scale_safety_rows.csv"
    head = load_df(head_path)
    gla = load_df(gla_path)
    mrt = load_df(mrt_path)

    rows: list[dict[str, Any]] = []
    rows.append(
        make_branch_row(
            branch="HS-HG",
            surface="HS-Local",
            source_path=head_path,
            df=head,
            candidate_col="internal_head_q_std",
            reliability_col="internal_head_q_mean",
            candidate_mode="nonsemantic_internal_head_disagreement",
            reliability_mode="nonsemantic_internal_head_alignment_mean",
            semantic_free=True,
            stage2_ready="HS-Local" in ready_surfaces,
            operation_specific=True,
        )
    )
    rows.append(
        make_branch_row(
            branch="HS-LA",
            surface="HS-Local",
            source_path=None,
            df=pd.DataFrame(),
            candidate_col="selected_query_attention_entropy_or_alignment",
            reliability_col="local_kv_read_reliability",
            candidate_mode="missing_selected_query_local_attention_internal_rows",
            reliability_mode="missing_local_kv_reliability_rows",
            semantic_free=True,
            stage2_ready="HS-Local" in ready_surfaces,
            operation_specific=False,
            blocker_notes=["head_gate_rows_do_not_substitute_selected_query_local_attention"],
        )
    )
    rows.append(
        make_branch_row(
            branch="HS-GW",
            surface="HS-GLA",
            source_path=gla_path,
            df=gla,
            candidate_col="state_delta_norm",
            reliability_col="state_delta_rel_norm",
            candidate_mode="nonsemantic_state_delta_candidate",
            reliability_mode="state_delta_relative_norm_not_fixed_reference_or_direct_gamma",
            semantic_free=True,
            stage2_ready="HS-GLA" in ready_surfaces,
            operation_specific=False,
            blocker_notes=["direct_kda_gamma_unavailable", "fixed_reference_state_reliability_unavailable"],
        )
    )
    rows.append(
        make_branch_row(
            branch="HS-GR",
            surface="HS-GLA",
            source_path=gla_path,
            df=gla,
            candidate_col="state_delta_norm",
            reliability_col="state_delta_rel_norm",
            candidate_mode="nonsemantic_state_delta_candidate",
            reliability_mode="state_delta_relative_norm_not_fixed_reference_or_direct_gamma",
            semantic_free=True,
            stage2_ready="HS-GLA" in ready_surfaces,
            operation_specific=False,
            blocker_notes=["direct_kda_gamma_unavailable", "fixed_reference_state_reliability_unavailable"],
        )
    )
    rows.append(
        make_branch_row(
            branch="HS-MR",
            surface="HS-MRT",
            source_path=mrt_path,
            df=mrt.assign(abs_predicted_metric_scale_delta=numeric(mrt.get("predicted_metric_scale_delta", pd.Series(dtype=float))).abs()),
            candidate_col="abs_predicted_metric_scale_delta",
            reliability_col="metric_readout_feature_norm",
            candidate_mode="nonsemantic_mrt_scale_delta_uncertainty",
            reliability_mode="nonsemantic_mrt_readout_feature_norm",
            semantic_free=True,
            stage2_ready="HS-MRT" in ready_surfaces,
            operation_specific=True,
            stage4_partner_ready=False,
            blocker_notes=["plan_requires_hs_mr_pair_with_hs_la_or_hs_gw_or_hs_gr"],
        )
    )
    rows.extend(
        [
            make_branch_row(
                branch="LB-AI",
                surface="LB-Anchor",
                source_path=None,
                df=pd.DataFrame(),
                candidate_col="anchor_internal_candidate",
                reliability_col="anchor_memory_reliability",
                candidate_mode="missing_lingbot_anchor_internal_rows",
                reliability_mode="missing_lingbot_anchor_reliability_rows",
                semantic_free=True,
                stage2_ready="LB-Anchor" in ready_surfaces,
                operation_specific=False,
            ),
            make_branch_row(
                branch="LB-AR",
                surface="LB-Anchor",
                source_path=None,
                df=pd.DataFrame(),
                candidate_col="anchor_read_internal_candidate",
                reliability_col="anchor_read_memory_reliability",
                candidate_mode="missing_lingbot_anchor_read_internal_rows",
                reliability_mode="missing_lingbot_anchor_read_reliability_rows",
                semantic_free=True,
                stage2_ready="LB-Anchor" in ready_surfaces,
                operation_specific=False,
            ),
            make_branch_row(
                branch="LB-LR",
                surface="LB-Local",
                source_path=None,
                df=pd.DataFrame(),
                candidate_col="local_read_internal_candidate",
                reliability_col="local_read_memory_reliability",
                candidate_mode="missing_lingbot_local_read_internal_rows",
                reliability_mode="missing_lingbot_local_read_reliability_rows",
                semantic_free=True,
                stage2_ready="LB-Local" in ready_surfaces,
                operation_specific=False,
            ),
            make_branch_row(
                branch="LB-TA/TR/TE",
                surface="LB-Trajectory",
                source_path=RESULT_ROOT / "stage2_memory_entry_provenance/smoke_lingbot_flashinfer_trace.jsonl",
                df=pd.DataFrame(),
                candidate_col="trajectory_internal_candidate",
                reliability_col="trajectory_memory_reliability",
                candidate_mode="missing_default_flashinfer_trajectory_internal_candidate_rows",
                reliability_mode="missing_default_flashinfer_trajectory_memory_reliability_rows",
                semantic_free=True,
                stage2_ready="LB-Trajectory" in ready_surfaces,
                operation_specific=False,
            ),
            make_branch_row(
                branch="LB-CT",
                surface="LB-Local",
                source_path=None,
                df=pd.DataFrame(),
                candidate_col="context_token_route_internal_candidate",
                reliability_col="context_token_memory_reliability",
                candidate_mode="missing_compact_context_token_internal_rows",
                reliability_mode="missing_compact_context_token_reliability_rows",
                semantic_free=True,
                stage2_ready="LB-Local" in ready_surfaces,
                operation_specific=False,
            ),
        ]
    )

    ready = [row["branch"] for row in rows if row["stage3_r4_ready_for_stage4"]]
    summary = {
        "schema": "acl2_v118tf_stage3_r4_internal_reliability_repair_summary_v1",
        "stage3_r4_complete": True,
        "global_goal_achieved": False,
        "ready_for_stage4_count": len(ready),
        "ready_for_stage4_branches": ready,
        "blocked_branches": [row["branch"] for row in rows if not row["stage3_r4_ready_for_stage4"]],
        "stage3_r4_decision": "PARTIAL_BRANCH_READY_HS_HG_ONLY" if ready else "NO_GO_R4_NO_BRANCH_READY",
        "stage4_runtime_launched": False,
        "boundary": (
            "R4 is a readiness repair only. It does not claim v118 semantic-aware geometry improvement "
            "and it does not launch Stage4/5/6 runtime."
        ),
        "outputs": {
            "branch_readiness": rel(OUT / "stage3_r4_branch_readiness_rows.csv"),
            "summary": rel(OUT / "stage3_r4_internal_reliability_repair_summary.json"),
            "report": rel(OUT / "STAGE3_R4_INTERNAL_RELIABILITY_REPAIR_REPORT.md"),
        },
        "input_artifacts": {
            "stage2_summary": rel(STAGE2_SUMMARY),
            "stage3_rows": rel(STAGE3_ROWS),
            "hs_head_reliability_rows": rel(head_path),
            "hs_gla_state_quality_rows": rel(gla_path),
            "hs_mrt_scale_safety_rows": rel(mrt_path),
        },
    }

    write_csv(OUT / "stage3_r4_branch_readiness_rows.csv", rows)
    write_json(OUT / "stage3_r4_internal_reliability_repair_summary.json", summary)
    write_text(OUT / "STAGE3_R4_INTERNAL_RELIABILITY_REPAIR_REPORT.md", report_text(summary, rows))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage3-R4",
            "surface_or_branch": "HS-HG",
            "status": summary["stage3_r4_decision"],
            "artifact": rel(OUT / "stage3_r4_internal_reliability_repair_summary.json"),
            "notes": "HS-HG branch-level Stage3 repair ready for Stage4 attribution only; v118 global goal not achieved",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
