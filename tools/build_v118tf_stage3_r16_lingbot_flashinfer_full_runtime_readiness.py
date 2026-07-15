#!/usr/bin/env python3
"""Summarize v118 Stage3-R16 LingBot FlashInfer full runtime readiness."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage3_r14_lingbot_flashinfer_internal_signal_probe"
OUT = STAGE / "runtime_full_summary"
TRACE_ROOT = STAGE / "runtime_full"
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
SEQS = ("00", "02")


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def stats(values: list[float], prefix: str, total: int | None = None) -> dict[str, Any]:
    denom = len(values) if total is None else total
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if denom == 0 or finite.size == 0:
        return {
            f"{prefix}_coverage": 0.0,
            f"{prefix}_p10": None,
            f"{prefix}_p90": None,
            f"{prefix}_span": 0.0,
            f"{prefix}_finite_count": int(finite.size),
        }
    p10, p90 = np.percentile(finite, [10, 90])
    return {
        f"{prefix}_coverage": float(finite.size) / float(denom),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p90": float(p90),
        f"{prefix}_span": float(p90 - p10),
        f"{prefix}_finite_count": int(finite.size),
    }


def temporal_nonconstant(rows: list[dict[str, Any]], key: str) -> tuple[bool, int, int]:
    buckets: dict[int, list[float]] = {}
    for row in rows:
        frame = int(fnum(row.get("last_read_time"), -1))
        if frame < 0:
            continue
        buckets.setdefault(frame // 500, []).append(fnum(row.get(key), float("nan")))
    eligible = 0
    varied = 0
    for values in buckets.values():
        finite = [value for value in values if math.isfinite(value)]
        if len(finite) < 2:
            continue
        eligible += 1
        if max(finite) - min(finite) > 1e-9:
            varied += 1
    return bool(eligible > 0 and varied / eligible >= 0.50), eligible, varied


def enrich_trajectory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    read_counts = [fnum(row.get("read_count")) for row in rows]
    max_read = max(read_counts) if read_counts else 1.0
    max_read = max(max_read, 1.0)
    out = []
    for row in rows:
        visible = max(int(fnum(row.get("visible_page_count"), 1.0)), 1)
        rank = max(int(fnum(row.get("qk_relevance_rank"), visible)), 1)
        if visible <= 1:
            rank_score = 1.0
        else:
            rank_score = 1.0 - float(rank - 1) / float(visible - 1)
        enriched = dict(row)
        enriched["trajectory_qk_rank_score"] = rank_score
        enriched["trajectory_read_count_norm"] = fnum(row.get("read_count")) / max_read
        out.append(enriched)
    return out


def summarize_seq(seq: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace = TRACE_ROOT / f"seq{seq}_flashinfer_trace.jsonl"
    rows = read_jsonl(trace)
    read_rows = [row for row in rows if row.get("operation_type") == "read_visible_page"]
    qk_rows = [row for row in read_rows if row.get("internal_signal_source") == "flashinfer_online_page_qk_summary"]
    trajectory = enrich_trajectory_rows([row for row in qk_rows if row.get("memory_family") == "trajectory_special"])
    complete = WORKSPACE / DATASET / seq / METHOD / ".complete.json"
    eval_json = WORKSPACE / DATASET / seq / METHOD / "eval/traj.json"
    candidate_temporal, candidate_bucket_count, candidate_varied_count = temporal_nonconstant(trajectory, "trajectory_qk_rank_score")
    reliability_temporal, reliability_bucket_count, reliability_varied_count = temporal_nonconstant(trajectory, "trajectory_read_count_norm")
    candidate_stats = stats([fnum(row.get("trajectory_qk_rank_score"), float("nan")) for row in trajectory], "candidate_rank_score", len(trajectory))
    reliability_stats = stats([fnum(row.get("trajectory_read_count_norm"), float("nan")) for row in trajectory], "read_count_reliability", len(trajectory))
    qk_coverage = float(len(qk_rows)) / float(len(read_rows)) if read_rows else 0.0
    candidate_gate = bool(candidate_stats["candidate_rank_score_coverage"] >= 0.90 and candidate_stats["candidate_rank_score_span"] >= 0.10 and candidate_temporal)
    reliability_gate = bool(reliability_stats["read_count_reliability_coverage"] >= 0.90 and reliability_stats["read_count_reliability_span"] >= 0.10 and reliability_temporal)
    blockers = []
    if not complete.exists():
        blockers.append("workspace_complete_missing")
    if not eval_json.exists():
        blockers.append("eval_json_missing")
    if qk_coverage < 1.0:
        blockers.append("qk_read_coverage_below_1.0")
    if not trajectory:
        blockers.append("trajectory_special_read_rows_missing")
    if not candidate_gate:
        blockers.append("candidate_rank_score_gate_failed")
    if not reliability_gate:
        blockers.append("read_count_reliability_gate_failed")
    return (
        {
            "seq": seq,
            "trace": rel(trace),
            "workspace_complete": complete.exists(),
            "eval_json": rel(eval_json),
            "total_trace_rows": len(rows),
            "read_row_count": len(read_rows),
            "qk_read_row_count": len(qk_rows),
            "qk_coverage_over_reads": qk_coverage,
            "trajectory_special_read_rows": len(trajectory),
            "candidate_temporal_nonconstant_gate": candidate_temporal,
            "candidate_temporal_bucket_count": candidate_bucket_count,
            "candidate_temporal_varied_bucket_count": candidate_varied_count,
            "reliability_temporal_nonconstant_gate": reliability_temporal,
            "reliability_temporal_bucket_count": reliability_bucket_count,
            "reliability_temporal_varied_bucket_count": reliability_varied_count,
            "candidate_gate": candidate_gate,
            "reliability_gate": reliability_gate,
            "retrieval_retention_stage3_seq_ready": bool(complete.exists() and eval_json.exists() and qk_coverage == 1.0 and trajectory and candidate_gate and reliability_gate),
            "blockers": ";".join(blockers),
            **candidate_stats,
            **reliability_stats,
            **read_json(eval_json),
        },
        [
            {
                "seq": seq,
                "source_frame_id": row.get("source_frame_id"),
                "last_read_time": row.get("last_read_time"),
                "memory_entry_id": row.get("memory_entry_id"),
                "qk_relevance_rank": row.get("qk_relevance_rank"),
                "visible_page_count": row.get("visible_page_count"),
                "trajectory_qk_rank_score": row.get("trajectory_qk_rank_score"),
                "trajectory_read_count_norm": row.get("trajectory_read_count_norm"),
                "qk_relevance_cosine": row.get("qk_relevance_cosine"),
                "qk_relevance_softmax": row.get("qk_relevance_softmax"),
                "read_count": row.get("read_count"),
            }
            for row in trajectory
        ],
    )


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v118-TF Stage3-R16 LingBot FlashInfer Full Runtime Readiness",
        "",
        f"- stage3_r16_decision: `{summary['stage3_r16_decision']}`",
        f"- lb_retrieval_retention_ready_for_stage4: `{summary['lb_retrieval_retention_ready_for_stage4']}`",
        f"- lb_admission_ready_for_stage4: `{summary['lb_admission_ready_for_stage4']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| seq | ready | read rows | trajectory rows | cand span | rel span | ATE | blockers |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | {row['retrieval_retention_stage3_seq_ready']} | {row['read_row_count']} | "
            f"{row['trajectory_special_read_rows']} | {row['candidate_rank_score_span']} | "
            f"{row['read_count_reliability_span']} | {row.get('ate')} | {row['blockers']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "R16 upgrades LingBot trajectory retrieval/retention from missing-default-FlashInfer-runtime to full 00/02 internal read-signal readiness. It does not cover trajectory admission because no pre-append admission candidate rows were generated. It also does not claim geometry improvement: the method is a default-backend trace run, not a promoted action policy.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seq_rows = []
    trajectory_rows = []
    for seq in SEQS:
        row, rows = summarize_seq(seq)
        seq_rows.append(row)
        trajectory_rows.extend(rows)
    retrieval_ready = bool(seq_rows and all(row["retrieval_retention_stage3_seq_ready"] for row in seq_rows))
    ready_branches = ["LB-TR", "LB-TE"] if retrieval_ready else []
    summary = {
        "schema": "acl2_v118tf_stage3_r16_lingbot_flashinfer_full_runtime_readiness_summary_v1",
        "stage3_r16_decision": (
            "FULL_DEFAULT_FLASHINFER_READ_RETRIEVAL_RETENTION_READY_STAGE4_PENDING"
            if retrieval_ready
            else "NO_GO_FULL_DEFAULT_FLASHINFER_READ_SIGNAL_GATE_FAILED"
        ),
        "lb_retrieval_retention_ready_for_stage4": retrieval_ready,
        "lb_admission_ready_for_stage4": False,
        "ready_for_stage4_branches": ready_branches,
        "blocked_branches": ["LB-TA"] + ([] if retrieval_ready else ["LB-TR", "LB-TE"]),
        "global_goal_achieved": False,
        "dataset": DATASET,
        "method": METHOD,
        "config": rel(STAGE / "configs/kitti_lingbot_flashinfer_r15_full_reuse_v105gt.yaml"),
        "workspace": rel(WORKSPACE),
        "seq_rows": seq_rows,
        "outputs": {
            "seq_rows": rel(OUT / "stage3_r16_full_runtime_readiness_rows.csv"),
            "trajectory_rows": rel(OUT / "stage3_r16_full_runtime_trajectory_rows.csv"),
            "summary": rel(OUT / "stage3_r16_lingbot_flashinfer_full_runtime_readiness_summary.json"),
            "report": rel(OUT / "STAGE3_R16_LINGBOT_FLASHINFER_FULL_RUNTIME_READINESS_REPORT.md"),
        },
        "boundary": (
            "Full 00/02 default-FlashInfer read-signal readiness for LB-TR/LB-TE only; "
            "LB-TA admission and any geometry-improvement/semantic-causality claim remain pending."
        ),
    }
    write_csv(OUT / "stage3_r16_full_runtime_readiness_rows.csv", seq_rows)
    write_csv(OUT / "stage3_r16_full_runtime_trajectory_rows.csv", trajectory_rows)
    write_json(OUT / "stage3_r16_lingbot_flashinfer_full_runtime_readiness_summary.json", summary)
    write_text(OUT / "STAGE3_R16_LINGBOT_FLASHINFER_FULL_RUNTIME_READINESS_REPORT.md", report_text(summary, seq_rows))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage3-R16",
            "surface_or_branch": "LB-TR/LB-TE",
            "status": summary["stage3_r16_decision"],
            "artifact": rel(OUT / "stage3_r16_lingbot_flashinfer_full_runtime_readiness_summary.json"),
            "notes": "Full 00/02 default-FlashInfer read-signal readiness for retrieval/retention only; LB-TA and Stage4 action policies pending",
        }
    )
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
