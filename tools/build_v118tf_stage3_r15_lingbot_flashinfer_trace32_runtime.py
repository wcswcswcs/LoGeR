#!/usr/bin/env python3
"""Summarize v118 Stage3-R15 LingBot FlashInfer trace32 runtime probe."""

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
OUT = STAGE / "runtime_trace32_summary"
TRACE_ROOT = STAGE / "runtime_trace32"
WORKSPACE = STAGE / "workspace_trace32"
DATASET = "kitti_v118_r14_00_02_trace32"
METHOD = "lingbot_map_stream_flashinfer_v118_r14_trace"
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


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = finite_values(rows, key)
    total = len(rows)
    if total == 0 or values.size == 0:
        return {f"{key}_coverage": 0.0, f"{key}_p10": None, f"{key}_p90": None, f"{key}_span": 0.0}
    p10, p90 = np.percentile(values, [10, 90])
    return {
        f"{key}_coverage": float(values.size) / float(total),
        f"{key}_p10": float(p10),
        f"{key}_p90": float(p90),
        f"{key}_span": float(p90 - p10),
    }


def summarize_seq(seq: str) -> dict[str, Any]:
    trace = TRACE_ROOT / f"seq{seq}_flashinfer_trace.jsonl"
    rows = read_jsonl(trace)
    read_rows = [row for row in rows if row.get("operation_type") == "read_visible_page"]
    qk_rows = [row for row in read_rows if row.get("internal_signal_source") == "flashinfer_online_page_qk_summary"]
    trajectory_rows = [row for row in qk_rows if row.get("memory_family") == "trajectory_special"]
    complete = WORKSPACE / DATASET / seq / METHOD / ".complete.json"
    eval_json = WORKSPACE / DATASET / seq / METHOD / "eval/traj.json"
    row = {
        "seq": seq,
        "trace": rel(trace),
        "workspace_complete": complete.exists(),
        "eval_json": rel(eval_json),
        "total_trace_rows": len(rows),
        "read_row_count": len(read_rows),
        "qk_read_row_count": len(qk_rows),
        "trajectory_special_read_rows": len(trajectory_rows),
        "qk_coverage_over_reads": float(len(qk_rows)) / float(len(read_rows)) if read_rows else 0.0,
        "runtime_trace32_pass": bool(complete.exists() and eval_json.exists() and read_rows and len(qk_rows) == len(read_rows) and trajectory_rows),
    }
    row.update(read_json(eval_json))
    for key in ("qk_relevance_cosine", "qk_relevance_softmax", "read_entropy_normalized", "read_count"):
        row.update(stats(qk_rows, key))
    for key, value in stats(trajectory_rows, "qk_relevance_softmax").items():
        row["trajectory_" + key] = value
    return row


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v118-TF Stage3-R15 LingBot FlashInfer Trace32 Runtime",
        "",
        f"- stage3_r15_decision: `{summary['stage3_r15_decision']}`",
        f"- trace32_runtime_pass: `{summary['trace32_runtime_pass']}`",
        f"- full_lingbot_runtime_ready_for_stage4: `{summary['full_lingbot_runtime_ready_for_stage4']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| seq | pass | read rows | qk rows | trajectory rows | ATE | qk span | softmax span |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | {row['runtime_trace32_pass']} | {row['read_row_count']} | "
            f"{row['qk_read_row_count']} | {row['trajectory_special_read_rows']} | {row.get('ate')} | "
            f"{row.get('qk_relevance_cosine_span')} | {row.get('qk_relevance_softmax_span')} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "This is a real KITTI trace32 default-FlashInfer runtime smoke for 00/02. It validates that the default backend can run with v118 internal-read trace fields on real frames. It is not a full-sequence 00/02 Stage3 gate and does not authorize LB-TA/LB-TR/LB-TE Stage4 promotion by itself.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [summarize_seq(seq) for seq in SEQS]
    trace32_pass = bool(rows and all(row["runtime_trace32_pass"] for row in rows))
    summary = {
        "schema": "acl2_v118tf_stage3_r15_lingbot_flashinfer_trace32_runtime_summary_v1",
        "stage3_r15_decision": (
            "TRACE32_DEFAULT_FLASHINFER_INTERNAL_SIGNAL_RUNTIME_PASS_FULL_SEQUENCE_PENDING"
            if trace32_pass
            else "NO_GO_TRACE32_DEFAULT_FLASHINFER_INTERNAL_SIGNAL_RUNTIME_FAILED"
        ),
        "trace32_runtime_pass": trace32_pass,
        "full_lingbot_runtime_ready_for_stage4": False,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "method": METHOD,
        "config": rel(STAGE / "configs/kitti_lingbot_flashinfer_r14_trace32.yaml"),
        "workspace": rel(WORKSPACE),
        "seq_rows": rows,
        "outputs": {
            "seq_rows": rel(OUT / "stage3_r15_trace32_runtime_rows.csv"),
            "summary": rel(OUT / "stage3_r15_lingbot_flashinfer_trace32_runtime_summary.json"),
            "report": rel(OUT / "STAGE3_R15_LINGBOT_FLASHINFER_TRACE32_RUNTIME_REPORT.md"),
        },
        "boundary": "Real trace32 pass only; full KITTI 00/02 default-FlashInfer internal cue runtime remains pending.",
    }
    write_csv(OUT / "stage3_r15_trace32_runtime_rows.csv", rows)
    write_json(OUT / "stage3_r15_lingbot_flashinfer_trace32_runtime_summary.json", summary)
    write_text(OUT / "STAGE3_R15_LINGBOT_FLASHINFER_TRACE32_RUNTIME_REPORT.md", report_text(summary, rows))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage3-R15",
            "surface_or_branch": "LB-Trajectory",
            "status": summary["stage3_r15_decision"],
            "artifact": rel(OUT / "stage3_r15_lingbot_flashinfer_trace32_runtime_summary.json"),
            "notes": "Real KITTI 00/02 trace32 default-FlashInfer internal signal runtime passed; full sequence still pending",
        }
    )
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
