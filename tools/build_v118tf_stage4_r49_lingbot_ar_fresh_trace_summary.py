#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R49 fresh no-action trace baselines."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"


def parse_seq_env(name: str, default: str) -> tuple[str, ...]:
    seqs = tuple(part.strip().zfill(2) for part in os.environ.get(name, default).replace(";", ",").split(",") if part.strip())
    return seqs or tuple(part.strip().zfill(2) for part in default.split(",") if part.strip())


STAGE_TAG = os.environ.get("ACL2_V118_FRESH_TRACE_TAG", "r49").strip().lower() or "r49"
STAGE = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_TRACE_STAGE_SLUG", "stage4_r49_lingbot_ar_fresh_trace_baseline")
SUMMARY_DIR = STAGE / "summary"
RUNTIME = STAGE / "runtime_full"
WORKSPACE = STAGE / "workspace"
METHOD = os.environ.get("ACL2_V118_FRESH_TRACE_METHOD", f"lingbot_map_stream_flashinfer_v118_{STAGE_TAG}_fresh_trace")
SEQS = parse_seq_env("ACL2_V118_FRESH_TRACE_SEQS", "04,03")
SEQ_LABEL = "/".join(SEQS)
DATASET_PREFIX = os.environ.get("ACL2_V118_FRESH_DATASET_PREFIX", "kitti_v118_r49_fresh_seq")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def trace_stats(seq: str) -> dict[str, Any]:
    trace = RUNTIME / f"seq{seq}_flashinfer_trace.jsonl"
    total = read = qk = anchor_patch_read = local_patch_read = trajectory_special_read = 0
    anchor_frames: set[int] = set()
    if trace.is_file():
        with trace.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                is_read = row.get("row_type") == "read"
                if is_read:
                    read += 1
                if row.get("internal_signal_source") == "flashinfer_online_page_qk_summary":
                    qk += 1
                if is_read and row.get("memory_family") == "anchor" and row.get("token_type") == "image_patch":
                    anchor_patch_read += 1
                    frame = row.get("source_frame_id")
                    if frame is not None:
                        anchor_frames.add(int(frame))
                if is_read and row.get("memory_family") == "local" and row.get("token_type") == "image_patch":
                    local_patch_read += 1
                if is_read and row.get("memory_family") == "trajectory_special":
                    trajectory_special_read += 1
    return {
        "trace": rel(trace),
        "trace_exists": trace.is_file(),
        "trace_rows": total,
        "read_rows": read,
        "qk_rows": qk,
        "qk_coverage_over_reads": qk / read if read else 0.0,
        "anchor_image_patch_read_rows": anchor_patch_read,
        "anchor_source_frame_coverage_0_7": len(set(range(8)) & anchor_frames) / 8.0,
        "local_image_patch_read_rows": local_patch_read,
        "trajectory_special_read_rows": trajectory_special_read,
    }


def summarize_seq(seq: str) -> dict[str, Any]:
    dataset = f"{DATASET_PREFIX}{seq}"
    method_root = WORKSPACE / dataset / seq / METHOD
    complete = method_root / ".complete.json"
    eval_json = method_root / "eval/traj.json"
    eval_data = read_json(eval_json)
    trace = trace_stats(seq)
    return {
        "schema": "acl2_v118tf_stage4_r49_fresh_trace_summary_row_v1",
        "seq": seq,
        "dataset": dataset,
        "method": METHOD,
        "complete": complete.is_file(),
        "eval_available": eval_json.is_file(),
        "ate": eval_data.get("ate"),
        "rpe_rot": eval_data.get("rpe_rot"),
        "rpe_trans": eval_data.get("rpe_trans"),
        **trace,
        "fresh_trace_ready": bool(
            complete.is_file()
            and eval_json.is_file()
            and trace["trace_exists"]
            and trace["trace_rows"] > 0
            and trace["qk_rows"] == trace["read_rows"]
            and trace["anchor_image_patch_read_rows"] > 0
            and trace["anchor_source_frame_coverage_0_7"] >= 1.0
        ),
    }


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_seq(seq) for seq in SEQS]
    ready = bool(rows and all(row["fresh_trace_ready"] for row in rows))
    decision_key = f"stage4_{STAGE_TAG}_decision"
    summary = {
        "schema": "acl2_v118tf_stage4_r49_lingbot_ar_fresh_trace_summary_v1",
        decision_key: (
            "FRESH_TRACE_BASELINE_READY_FOR_SUPPORT_AND_TOKEN_BUILD"
            if ready
            else "NO_GO_FRESH_TRACE_BASELINE_INCOMPLETE"
        ),
        "global_goal_achieved": False,
        "boundary": (
            f"{STAGE_TAG.upper()} is a no-action fresh baseline/internal-trace build for {SEQ_LABEL}. "
            "It does not evaluate the R47 candidate or controls."
        ),
        "sequence_count": len(rows),
        "fresh_trace_ready": ready,
        "rows": rows,
        "outputs": {
            "rows": rel(SUMMARY_DIR / "stage4_r49_fresh_trace_rows.csv"),
            "summary": rel(SUMMARY_DIR / "stage4_r49_fresh_trace_summary.json"),
            "report": rel(SUMMARY_DIR / "STAGE4_R49_FRESH_TRACE_REPORT.md"),
        },
        "next_step": (
            f"Build fresh semantic support and token tensors for {SEQ_LABEL}, then pre-register and run "
            "R47 candidate/opposite/random controls."
        ),
    }
    write_csv(SUMMARY_DIR / "stage4_r49_fresh_trace_rows.csv", rows)
    write_json(SUMMARY_DIR / "stage4_r49_fresh_trace_summary.json", summary)
    lines = [
        f"# ACL2 v118 Stage4-{STAGE_TAG.upper()} Fresh Trace Baseline",
        "",
        f"- decision: `{summary[decision_key]}`",
        f"- fresh_trace_ready: `{summary['fresh_trace_ready']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| seq | complete | eval | ATE | RPE rot | RPE trans | trace rows | read rows | anchor read rows | qk coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | {row['complete']} | {row['eval_available']} | {row['ate']} | "
            f"{row['rpe_rot']} | {row['rpe_trans']} | {row['trace_rows']} | {row['read_rows']} | "
            f"{row['anchor_image_patch_read_rows']} | {row['qk_coverage_over_reads']} |"
        )
    lines += ["", "## Boundary", "", summary["boundary"]]
    (SUMMARY_DIR / "STAGE4_R49_FRESH_TRACE_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
