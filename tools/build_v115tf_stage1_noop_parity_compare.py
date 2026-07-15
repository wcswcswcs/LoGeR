#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
STAGE1 = RESULT_ROOT / "stage1_hook_audit"
DEFAULT_NOTRACE = RESULT_ROOT / "outputs/stage1_parity_notrace_kitti00_max12_gpu4_nooffload_rerun1"
DEFAULT_TRACE = RESULT_ROOT / "outputs/stage1_parity_traceonly_kitti00_max12_gpu4_nooffload_rerun1"
DEFAULT_SUMMARY = STAGE1 / "hs_noop_trace_parity_summary_nooffload_gpu4_rerun1.json"
DEFAULT_ROWS = STAGE1 / "hs_noop_trace_parity_rows_nooffload_gpu4_rerun1.csv"
CANONICAL_SUMMARY = STAGE1 / "hs_noop_trace_parity_summary.json"
CANONICAL_ROWS = STAGE1 / "hs_noop_trace_parity_rows.csv"
PRIOR_SUMMARY = STAGE1 / "hs_noop_trace_parity_summary_prior_offload_failed_20260708.json"
PRIOR_ROWS = STAGE1 / "hs_noop_trace_parity_rows_prior_offload_failed_20260708.csv"
THRESHOLD = 1e-6
LEFT_ROOT = DEFAULT_NOTRACE
RIGHT_ROOT = DEFAULT_TRACE


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_numeric_json(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric_json(item, child))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child = f"{prefix}[{idx}]"
            out.update(flatten_numeric_json(item, child))
    elif isinstance(value, bool) or value is None:
        return out
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)
    return out


def numeric_diff(left: np.ndarray, right: np.ndarray) -> tuple[float, float, int]:
    if left.shape != right.shape:
        return float("inf"), float("inf"), -1
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return float(np.max(diff)) if diff.size else 0.0, float(np.mean(diff)) if diff.size else 0.0, int(diff.size)


def compare_numeric_file(rel_path: str, kind: str) -> dict[str, Any]:
    left_path = LEFT_ROOT / rel_path
    right_path = RIGHT_ROOT / rel_path
    row: dict[str, Any] = {
        "kind": kind,
        "relative_path": rel_path,
        "notrace_path": rel(left_path),
        "trace_path": rel(right_path),
        "exists_notrace": left_path.exists(),
        "exists_trace": right_path.exists(),
        "pass": False,
        "max_abs_diff": "",
        "mean_abs_diff": "",
        "num_values": "",
        "shape_notrace": "",
        "shape_trace": "",
    }
    if not left_path.exists() or not right_path.exists():
        return row
    if left_path.suffix == ".npy":
        left = np.load(left_path)
        right = np.load(right_path)
    else:
        left = np.loadtxt(left_path)
        right = np.loadtxt(right_path)
    row["shape_notrace"] = list(left.shape)
    row["shape_trace"] = list(right.shape)
    max_abs, mean_abs, num_values = numeric_diff(np.asarray(left), np.asarray(right))
    row["max_abs_diff"] = max_abs
    row["mean_abs_diff"] = mean_abs
    row["num_values"] = num_values
    row["pass"] = max_abs <= THRESHOLD
    return row


def compare_json_numeric(rel_path: str) -> list[dict[str, Any]]:
    left_path = LEFT_ROOT / rel_path
    right_path = RIGHT_ROOT / rel_path
    if not left_path.exists() or not right_path.exists():
        return [
            {
                "kind": "json_numeric",
                "relative_path": rel_path,
                "metric": "",
                "exists_notrace": left_path.exists(),
                "exists_trace": right_path.exists(),
                "pass": False,
                "max_abs_diff": "",
                "mean_abs_diff": "",
                "num_values": "",
            }
        ]
    left = flatten_numeric_json(read_json(left_path))
    right = flatten_numeric_json(read_json(right_path))
    keys = sorted(set(left) | set(right))
    rows: list[dict[str, Any]] = []
    for key in keys:
        exists_left = key in left
        exists_right = key in right
        max_abs = abs(left[key] - right[key]) if exists_left and exists_right else float("inf")
        rows.append(
            {
                "kind": "json_numeric",
                "relative_path": rel_path,
                "metric": key,
                "exists_notrace": exists_left,
                "exists_trace": exists_right,
                "pass": max_abs <= THRESHOLD,
                "max_abs_diff": max_abs,
                "mean_abs_diff": max_abs,
                "num_values": 1 if exists_left and exists_right else -1,
            }
        )
    return rows


def parity_rows() -> list[dict[str, Any]]:
    rows = [
        compare_numeric_file("00/02/poses/abs_pose.txt", "pose"),
        compare_numeric_file("00/02/poses/gt_abs_pose.txt", "gt_pose"),
        compare_numeric_file("00/02/poses/intri.txt", "intrinsics"),
    ]
    for rel_path in sorted((LEFT_ROOT / "00/02/depth/dpt").glob("*.npy")):
        rows.append(compare_numeric_file(rel_path.relative_to(LEFT_ROOT).as_posix(), "depth_dpt"))
    for rel_path in sorted((LEFT_ROOT / "00/02/depth/conf").glob("*.npy")):
        rows.append(compare_numeric_file(rel_path.relative_to(LEFT_ROOT).as_posix(), "depth_conf"))
    rows.extend(compare_json_numeric("00/02/eval/trajectory_metrics.json"))
    rows.extend(compare_json_numeric("eval_summary.json"))
    return rows


def maybe_preserve_prior() -> None:
    if CANONICAL_SUMMARY.exists() and not PRIOR_SUMMARY.exists():
        PRIOR_SUMMARY.write_text(CANONICAL_SUMMARY.read_text(encoding="utf-8"), encoding="utf-8")
    if CANONICAL_ROWS.exists() and not PRIOR_ROWS.exists():
        PRIOR_ROWS.write_text(CANONICAL_ROWS.read_text(encoding="utf-8"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v115 Stage1 HorizonStream smoke parity artifacts.")
    parser.add_argument("--left-root", type=Path, default=DEFAULT_NOTRACE)
    parser.add_argument("--right-root", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--comparison", default="matched_nooffload_gpu4_fresh_to_fresh")
    parser.add_argument("--trace-root", default=rel(RESULT_ROOT / "diagnostics/stage1_parity_traceonly_kitti00_max12_gpu4_nooffload_rerun1"))
    parser.add_argument("--promote-canonical", action="store_true")
    return parser.parse_args()


def main() -> None:
    global LEFT_ROOT, RIGHT_ROOT
    args = parse_args()
    LEFT_ROOT = args.left_root
    RIGHT_ROOT = args.right_root
    if args.promote_canonical:
        maybe_preserve_prior()
    rows = parity_rows()
    max_diffs = [
        float(row["max_abs_diff"])
        for row in rows
        if row.get("max_abs_diff") not in ("", None) and str(row.get("max_abs_diff")) != "inf"
    ]
    failed = [row for row in rows if not row.get("pass")]
    summary = {
        "schema": "acl2_v115tf_stage1_noop_trace_parity_v2",
        "comparison": args.comparison,
        "threshold": THRESHOLD,
        "pass": not failed,
        "max_abs_diff": max(max_diffs) if max_diffs else float("inf"),
        "failed_row_count": len(failed),
        "row_count": len(rows),
        "left_output_root": rel(LEFT_ROOT),
        "right_output_root": rel(RIGHT_ROOT),
        "notrace_output_root": rel(LEFT_ROOT),
        "trace_output_root": rel(RIGHT_ROOT),
        "trace_root": args.trace_root,
        "prior_failed_summary_preserved": rel(PRIOR_SUMMARY) if PRIOR_SUMMARY.exists() else "",
        "rows_path": rel(args.rows),
    }
    write_csv(args.rows, rows)
    write_json(args.summary, summary)
    if args.promote_canonical:
        write_csv(CANONICAL_ROWS, rows)
        write_json(CANONICAL_SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
