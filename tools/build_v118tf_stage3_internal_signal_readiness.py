#!/usr/bin/env python3
"""Build ACL2 v118-TF Stage3 per-surface internal-signal readiness."""

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
OUT = RESULT_ROOT / "stage3_internal_signal_readiness"
STAGE2_SUMMARY = RESULT_ROOT / "stage2_memory_entry_provenance/stage2_memory_entry_provenance_summary.json"
V117_STAGE3 = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability/stage3_internal_reliability"


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


def load_df(name: str) -> pd.DataFrame:
    path = V117_STAGE3 / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def stats(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    total = int(len(numeric))
    ok = numeric[np.isfinite(numeric)]
    if total == 0:
        return {"row_count": 0, "nonmissing_count": 0, "coverage": 0.0, "p10": None, "p90": None, "span": 0.0}
    if ok.empty:
        return {"row_count": total, "nonmissing_count": 0, "coverage": 0.0, "p10": None, "p90": None, "span": 0.0}
    p10 = float(np.percentile(ok.to_numpy(dtype=float), 10))
    p90 = float(np.percentile(ok.to_numpy(dtype=float), 90))
    return {
        "row_count": total,
        "nonmissing_count": int(ok.shape[0]),
        "coverage": float(ok.shape[0]) / float(total),
        "p10": p10,
        "p90": p90,
        "span": p90 - p10,
    }


def varied_within_temporal_bucket(df: pd.DataFrame, value_col: str) -> bool:
    if df.empty or value_col not in df:
        return False
    work = df.copy()
    if "chunk_idx" in work and work["chunk_idx"].notna().any():
        work["bucket"] = work["seq"].astype(str) + ":chunk:" + work["chunk_idx"].fillna(-1).astype(int).astype(str)
    else:
        unit = work.get("unit_id", pd.Series(range(len(work)), index=work.index)).astype(str)
        frame = unit.str.extract(r":(\d+)(?::frame_aggregate)?$", expand=False)
        frame_num = pd.to_numeric(frame, errors="coerce").fillna(0).astype(int)
        work["bucket"] = work["seq"].astype(str) + ":seg:" + (frame_num // 500).astype(str)
    varied = 0
    eligible = 0
    for _bucket, group in work.groupby("bucket"):
        vals = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        eligible += 1
        if float(vals.max() - vals.min()) > 1e-9:
            varied += 1
    return eligible > 0 and varied / eligible >= 0.50


def mode_text(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df:
        return ""
    vals = sorted({str(v) for v in df[col].dropna().unique()})
    return ";".join(vals)


def semantic_proxy_mode(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
            "semantic_object_persistence_proxy",
            "object_persistence_proxy",
            "semantic_stable_minus_risk",
            "chunk_semantic_state_norm_proxy",
        ]
    )


def surface_row(
    *,
    surface: str,
    candidate_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    candidate_col: str = "candidate_gain",
    reliability_col: str = "memory_reliability",
    required: bool = True,
    stage2_status: str = "ready",
) -> dict[str, Any]:
    cand_stats = stats(candidate_df[candidate_col]) if not candidate_df.empty and candidate_col in candidate_df else stats(pd.Series(dtype=float))
    rel_stats = (
        stats(reliability_df[reliability_col])
        if not reliability_df.empty and reliability_col in reliability_df
        else stats(pd.Series(dtype=float))
    )
    cand_mode = mode_text(candidate_df, "candidate_gain_mode")
    rel_mode = mode_text(reliability_df, "reliability_mode")
    candidate_proxy = semantic_proxy_mode(cand_mode)
    reliability_proxy = semantic_proxy_mode(rel_mode)
    candidate_gate = cand_stats["coverage"] >= 0.90 and cand_stats["span"] >= 0.10 and not candidate_proxy
    reliability_gate = rel_stats["coverage"] >= 0.90 and rel_stats["span"] >= 0.10 and not reliability_proxy
    temporal_gate = varied_within_temporal_bucket(candidate_df, candidate_col)
    ready = bool(required and stage2_status == "ready" and candidate_gate and reliability_gate and temporal_gate)
    blockers = []
    if stage2_status != "ready":
        blockers.append(f"stage2_surface_not_ready:{stage2_status}")
    if cand_stats["coverage"] < 0.90:
        blockers.append("candidate_nonmissing_coverage_below_0.90")
    if cand_stats["span"] < 0.10:
        blockers.append("candidate_p10_p90_span_below_0.10")
    if candidate_proxy:
        blockers.append("candidate_is_semantic_proxy_not_internal")
    if rel_stats["coverage"] < 0.90:
        blockers.append("reliability_nonmissing_coverage_below_0.90")
    if rel_stats["span"] < 0.10:
        blockers.append("reliability_p10_p90_span_below_0.10")
    if reliability_proxy:
        blockers.append("reliability_is_semantic_proxy_not_memory_internal")
    if not temporal_gate:
        blockers.append("candidate_constant_or_unavailable_within_temporal_bucket")
    return {
        "schema": "acl2_v118tf_stage3_surface_signal_gate_row_v1",
        "surface": surface,
        "stage2_status": stage2_status,
        "candidate_row_count": cand_stats["row_count"],
        "candidate_nonmissing_coverage": cand_stats["coverage"],
        "candidate_p10": cand_stats["p10"],
        "candidate_p90": cand_stats["p90"],
        "candidate_p10_p90_span": cand_stats["span"],
        "candidate_mode": cand_mode,
        "candidate_gate": candidate_gate,
        "reliability_row_count": rel_stats["row_count"],
        "reliability_nonmissing_coverage": rel_stats["coverage"],
        "reliability_p10": rel_stats["p10"],
        "reliability_p90": rel_stats["p90"],
        "reliability_p10_p90_span": rel_stats["span"],
        "reliability_mode": rel_mode,
        "reliability_gate": reliability_gate,
        "prefix_causal_availability": True,
        "semantic_proxy_blocker": candidate_proxy or reliability_proxy,
        "temporal_bucket_nonconstant_gate": temporal_gate,
        "stage3_surface_ready": ready,
        "blockers": ";".join(blockers),
    }


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lb_trajectory = next((row for row in rows if row.get("surface") == "LB-Trajectory"), {})
    if lb_trajectory.get("stage2_status") == "ready":
        lb_boundary = (
            "LingBot trajectory Stage2 default FlashInfer provenance is ready, but Stage3 still rejects its "
            "semantic persistence rows as proxy evidence rather than internal admission/reliability cues."
        )
    else:
        lb_boundary = "LingBot trajectory remains Stage2-blocked or debug-only and cannot enter Stage4."
    lines = [
        "# ACL2 v118-TF Stage3 Internal Signal Readiness Report",
        "",
        f"- stage3_complete: `{summary['stage3_complete']}`",
        f"- any_surface_ready_for_stage4: `{summary['any_surface_ready_for_stage4']}`",
        f"- ready_surface_count: `{summary['ready_surface_count']}`",
        "",
        "| surface | ready | cand span | rel span | blocker |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['surface']} | {row['stage3_surface_ready']} | {row['candidate_p10_p90_span']} | {row['reliability_p10_p90_span']} | {row['blockers']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        (
            "Stage3 did not promote semantic persistence proxies into internal cues. "
            + lb_boundary
            + " HorizonStream GLA keeps a state-delta approximation, but reliability span is below the v118 pre-Stage4 threshold."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage2 = read_json(STAGE2_SUMMARY)
    if not stage2.get("stage2_complete"):
        raise RuntimeError("Stage2 is incomplete; refusing Stage3")
    ready_surfaces = set(stage2.get("ready_surfaces", []))
    candidate = load_df("candidate_update_rows.csv")
    reliability = load_df("memory_reliability_rows.csv")

    def c(surface: str) -> pd.DataFrame:
        return candidate[candidate["surface"] == surface].copy() if not candidate.empty else pd.DataFrame()

    def r(family: str) -> pd.DataFrame:
        return reliability[reliability["memory_family"] == family].copy() if not reliability.empty else pd.DataFrame()

    rows = [
        surface_row(
            surface="LB-Anchor",
            candidate_df=pd.DataFrame(),
            reliability_df=pd.DataFrame(),
            stage2_status="ready" if "LB-Anchor" in ready_surfaces else "stage2_blocked",
        ),
        surface_row(
            surface="LB-Local",
            candidate_df=pd.DataFrame(),
            reliability_df=pd.DataFrame(),
            stage2_status="ready" if "LB-Local" in ready_surfaces else "stage2_blocked",
        ),
        surface_row(
            surface="LB-Trajectory",
            candidate_df=c("trajectory_admission_proxy"),
            reliability_df=r("append_only_source_read_proxy"),
            stage2_status="ready" if "LB-Trajectory" in ready_surfaces else "stage2_blocked",
        ),
        surface_row(
            surface="HS-Local",
            candidate_df=c("local_pose_head_probe"),
            reliability_df=r("local_kv_head_probe"),
            stage2_status="ready" if "HS-Local" in ready_surfaces else "stage2_blocked",
        ),
        surface_row(
            surface="HS-GLA",
            candidate_df=c("gla_state_probe_chunk_band"),
            reliability_df=r("gla_state_chunk_band"),
            stage2_status="ready" if "HS-GLA" in ready_surfaces else "stage2_blocked",
        ),
        surface_row(
            surface="HS-MRT",
            candidate_df=c("mrt_readout_diagnostic"),
            reliability_df=pd.DataFrame(),
            stage2_status="ready" if "HS-MRT" in ready_surfaces else "stage2_blocked",
        ),
    ]
    ready = [row["surface"] for row in rows if row["stage3_surface_ready"]]
    if "LB-Trajectory" in ready_surfaces:
        trajectory_attempt = (
            "Rejected LingBot trajectory semantic persistence proxy even after Stage2 default FlashInfer "
            "provenance passed; candidate/reliability mode is not internal."
        )
    else:
        trajectory_attempt = (
            "Rejected LingBot trajectory semantic persistence proxy because Stage2 default FlashInfer "
            "provenance is blocked and candidate mode is not internal."
        )
    summary = {
        "schema": "acl2_v118tf_stage3_internal_signal_readiness_summary_v1",
        "stage3_complete": True,
        "any_surface_ready_for_stage4": bool(ready),
        "ready_surface_count": len(ready),
        "ready_surfaces": ready,
        "blocked_surfaces": [row["surface"] for row in rows if not row["stage3_surface_ready"]],
        "stage3_decision": "NO_GO_INTERNAL_SIGNAL_READINESS_BLOCKED" if not ready else "PARTIAL_SURFACE_READY",
        "fail_forward_attempts_recorded": [
            "Reused v117 internal rows only when their mode was not semantic persistence substitution.",
            "Evaluated HS local internal-std candidate separately from semantic-mixed reliability.",
            "Evaluated HS GLA state-delta approximation separately from direct KDA write weights.",
            trajectory_attempt,
        ],
        "outputs": {
            "surface_signal_gate_rows": rel(OUT / "stage3_surface_signal_gate_rows.csv"),
            "summary": rel(OUT / "stage3_internal_signal_readiness_summary.json"),
            "report": rel(OUT / "STAGE3_INTERNAL_SIGNAL_READINESS_REPORT.md"),
        },
    }
    write_csv(OUT / "stage3_surface_signal_gate_rows.csv", rows)
    write_json(OUT / "stage3_internal_signal_readiness_summary.json", summary)
    write_text(OUT / "STAGE3_INTERNAL_SIGNAL_READINESS_REPORT.md", report_text(summary, rows))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
