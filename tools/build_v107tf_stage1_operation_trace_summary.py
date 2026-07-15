#!/usr/bin/env python3
"""Summarize ACL2 v107TF Stage1 cache-operation traces and no-action parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "third_party/lingbot-map/benchmark"
sys.path.insert(0, str(BENCH))

from benchmark.io.image import load_exr  # noqa: E402
from benchmark.io.intrinsics import read_intrinsics  # noqa: E402


V107 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
DEFAULT_STAGE1 = V107 / "stage1_cache_operation_instrumentation"
STAGE1 = DEFAULT_STAGE1
WORKSPACE = STAGE1 / "workspace"
TARGET_MANIFEST = STAGE1 / "target_manifest.csv"
RUN_MANIFEST = STAGE1 / "run_manifest.csv"
CONFIG_SUMMARY = STAGE1 / "config_generation_summary.json"
REQUIRED_GATE_OPS = {
    "initialization",
    "trajectory_write",
    "retention",
    "eviction",
    "cache_append",
    "special_token_update",
}


def set_stage1_root(stage1_root: Path) -> None:
    global STAGE1, WORKSPACE, TARGET_MANIFEST, RUN_MANIFEST, CONFIG_SUMMARY
    STAGE1 = stage1_root.resolve()
    WORKSPACE = STAGE1 / "workspace"
    TARGET_MANIFEST = STAGE1 / "target_manifest.csv"
    RUN_MANIFEST = STAGE1 / "run_manifest.csv"
    CONFIG_SUMMARY = STAGE1 / "config_generation_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def bool_s(value: bool) -> str:
    return "true" if value else "false"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "row_type": "trace_error",
                        "error": f"json_decode_error:{path.name}:{line_number}:{exc}",
                    }
                )
    return rows


def load_traj(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            mats.append(mat)
    if not mats:
        return None
    return np.stack(mats, axis=0)


def max_abs_diff_arrays(lhs: np.ndarray | None, rhs: np.ndarray | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    if lhs.shape != rhs.shape:
        return math.inf
    diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    if diff.size == 0 or np.all(np.isnan(diff)):
        return 0.0
    return float(np.nanmax(diff))


def max_abs_diff_exr_dir(lhs_root: Path, rhs_root: Path, name: str) -> tuple[float | None, str]:
    lhs_files = sorted((lhs_root / name).glob("*.exr")) if (lhs_root / name).is_dir() else []
    rhs_files = sorted((rhs_root / name).glob("*.exr")) if (rhs_root / name).is_dir() else []
    if not lhs_files or not rhs_files:
        return None, f"{name}_exr_missing"
    if [path.name for path in lhs_files] != [path.name for path in rhs_files]:
        return math.inf, f"{name}_filename_mismatch"
    max_diff = 0.0
    for lhs_file, rhs_file in zip(lhs_files, rhs_files):
        lhs = load_exr(lhs_file)
        rhs = load_exr(rhs_file)
        if lhs.shape != rhs.shape:
            return math.inf, f"{name}_shape_mismatch:{lhs_file.name}"
        diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
        if diff.size and not np.all(np.isnan(diff)):
            max_diff = max(max_diff, float(np.nanmax(diff)))
    return max_diff, ""


def load_intr(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return read_intrinsics(path)


def parse_list(raw: str | None) -> set[str]:
    if raw is None or raw.strip() == "":
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def selected_trace_cases(seqs: set[str], target_kinds: set[str], target_ids: set[str]) -> list[dict[str, Any]]:
    targets = {row["target_id"]: row for row in read_csv(TARGET_MANIFEST)}
    out: list[dict[str, Any]] = []
    for row in read_csv(RUN_MANIFEST):
        if row["phase"] != "run_worker_trace":
            continue
        target = dict(targets[row["target_id"]])
        target.update(row)
        if seqs and target["seq"] not in seqs:
            continue
        if target_kinds and target["target_kind"] not in target_kinds:
            continue
        if target_ids and target["target_id"] not in target_ids:
            continue
        out.append(target)
    return out


def compare_case(target: dict[str, Any], trace_rows: list[dict[str, Any]], notrace_method: str, trace_method: str) -> dict[str, Any]:
    dataset = target["dataset"]
    seq = target["seq"]
    notrace_root = WORKSPACE / dataset / seq / notrace_method
    trace_root = WORKSPACE / dataset / seq / trace_method
    reasons: list[str] = []
    if not (notrace_root / ".complete.json").is_file():
        reasons.append("notrace_complete_missing")
    if not (trace_root / ".complete.json").is_file():
        reasons.append("trace_complete_missing")

    notrace_traj = notrace_root / "traj.txt"
    trace_traj = trace_root / "traj.txt"
    pose_sha_equal = sha256_file(notrace_traj) != "" and sha256_file(notrace_traj) == sha256_file(trace_traj)
    pose_diff = max_abs_diff_arrays(load_traj(notrace_traj), load_traj(trace_traj))
    intr_diff = max_abs_diff_arrays(load_intr(notrace_root / "intrinsics.txt"), load_intr(trace_root / "intrinsics.txt"))
    depth_diff, depth_reason = max_abs_diff_exr_dir(notrace_root, trace_root, "depth")
    conf_diff, conf_reason = max_abs_diff_exr_dir(notrace_root, trace_root, "confidence")

    if pose_diff is None:
        reasons.append("pose_missing")
    if intr_diff is None:
        reasons.append("intrinsics_missing")
    if depth_reason:
        reasons.append(depth_reason)
    if conf_reason:
        reasons.append(conf_reason)

    trace_error_count = sum(1 for row in trace_rows if row.get("row_type") == "trace_error")
    operation_rows = [row for row in trace_rows if row.get("row_type") == "cache_operation"]
    operation_types = sorted({str(row.get("operation_type", "")) for row in operation_rows if row.get("operation_type", "")})
    observable_false = sum(1 for row in operation_rows if str(row.get("operation_type_observable", "true")).lower() == "false")

    if not operation_rows:
        reasons.append("operation_trace_empty")
    if trace_error_count:
        reasons.append(f"trace_error_rows:{trace_error_count}")
    if observable_false:
        reasons.append(f"operation_type_observable_false_rows:{observable_false}")

    depth_ok = depth_diff is not None and depth_diff == 0.0
    intr_ok = intr_diff is not None and intr_diff == 0.0
    conf_ok = conf_diff is not None and conf_diff == 0.0
    complete_ok = (notrace_root / ".complete.json").is_file() and (trace_root / ".complete.json").is_file()
    parity_pass = (
        complete_ok
        and pose_sha_equal
        and depth_ok
        and intr_ok
        and conf_ok
        and trace_error_count == 0
        and len(operation_rows) > 0
        and observable_false == 0
    )

    if not pose_sha_equal:
        reasons.append("pose_sha_not_equal")
    if depth_diff is not None and not depth_ok:
        reasons.append("depth_diff_nonzero")
    if intr_diff is not None and not intr_ok:
        reasons.append("intrinsics_diff_nonzero")
    if conf_diff is not None and not conf_ok:
        reasons.append("confidence_diff_nonzero")

    return {
        "schema": "acl2_v107tf_stage1_operation_trace_parity_row_v1",
        "target_id": target["target_id"],
        "target_kind": target["target_kind"],
        "dataset": dataset,
        "seq": seq,
        "notrace_method": notrace_method,
        "trace_method": trace_method,
        "pose_sha_equal": bool_s(pose_sha_equal),
        "pose_max_abs_diff": "" if pose_diff is None else pose_diff,
        "depth_max_abs_diff": "" if depth_diff is None else depth_diff,
        "intrinsics_max_abs_diff": "" if intr_diff is None else intr_diff,
        "confidence_max_abs_diff": "" if conf_diff is None else conf_diff,
        "operation_row_count": len(operation_rows),
        "observed_operation_types": ",".join(operation_types),
        "observed_operation_type_count": len(operation_types),
        "trace_error_row_count": trace_error_count,
        "operation_type_observable_false_rows": observable_false,
        "parity_pass": bool_s(parity_pass),
        "failure_reason": ";".join(dict.fromkeys(reasons)),
        "notrace_root": rel(notrace_root),
        "trace_root": rel(trace_root),
        "trace_file": rel(Path(target["trace_file"])),
    }


def build_summary(seqs: set[str], target_kinds: set[str], target_ids: set[str]) -> dict[str, Any]:
    config = json.loads(CONFIG_SUMMARY.read_text(encoding="utf-8"))
    notrace_method = config["notrace_method"]
    trace_method = config["trace_method"]
    cases = selected_trace_cases(seqs, target_kinds, target_ids)
    parity_rows: list[dict[str, Any]] = []
    operation_rows_out: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []

    for target in cases:
        trace_rows = load_jsonl(Path(target["trace_file"]))
        all_trace_rows.extend(trace_rows)
        for row in trace_rows:
            if row.get("row_type") != "cache_operation":
                continue
            out = dict(row)
            out.setdefault("target_id", target["target_id"])
            out.setdefault("target_kind", target["target_kind"])
            out.setdefault("target_window_index", target["window_index"])
            out.setdefault("trace_start_idx", target["trace_start_idx"])
            out.setdefault("trace_end_idx_exclusive", target["trace_end_idx_exclusive"])
            operation_rows_out.append(out)
        parity_rows.append(compare_case(target, trace_rows, notrace_method, trace_method))

    write_csv(STAGE1 / "operation_trace_rows.csv", operation_rows_out)
    write_csv(STAGE1 / "operation_trace_parity_rows.csv", parity_rows)

    operation_types = sorted(
        {str(row.get("operation_type", "")) for row in operation_rows_out if row.get("operation_type", "")}
    )
    gate_ops_seen = sorted(set(operation_types) & REQUIRED_GATE_OPS)
    non_readout = sorted(op for op in operation_types if op != "readout")
    all_parity_pass = bool(parity_rows) and all(row["parity_pass"] == "true" for row in parity_rows)
    trace_error_rows = sum(1 for row in all_trace_rows if row.get("row_type") == "trace_error")
    stage1_pass = (
        all_parity_pass
        and len(operation_types) >= 3
        and len(non_readout) >= 1
        and len(gate_ops_seen) >= 2
        and trace_error_rows == 0
    )
    summary = {
        "schema": "acl2_v107tf_stage1_operation_trace_summary_v1",
        "case_count": len(cases),
        "parity_row_count": len(parity_rows),
        "trace_parity_pass": all_parity_pass,
        "stage1_pass": stage1_pass,
        "operation_row_count": len(operation_rows_out),
        "observed_operation_types": operation_types,
        "observed_operation_type_count": len(operation_types),
        "non_readout_operation_types": non_readout,
        "required_gate_operation_types_seen": gate_ops_seen,
        "trace_error_rows": trace_error_rows,
        "outputs": {
            "operation_trace_rows": rel(STAGE1 / "operation_trace_rows.csv"),
            "operation_trace_parity_rows": rel(STAGE1 / "operation_trace_parity_rows.csv"),
            "operation_trace_summary": rel(STAGE1 / "operation_trace_summary.json"),
            "no_action_parity_report": rel(STAGE1 / "no_action_parity_report.md"),
        },
    }
    write_text(STAGE1 / "operation_trace_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    report_lines = [
        "# ACL2 v107TF Stage1 No-Action Parity Report",
        "",
        f"- case_count: `{len(cases)}`",
        f"- trace_parity_pass: `{stage1_pass and all_parity_pass}`",
        f"- stage1_pass: `{stage1_pass}`",
        f"- operation_row_count: `{len(operation_rows_out)}`",
        f"- trace_error_rows: `{trace_error_rows}`",
        f"- observed_operation_types: `{', '.join(operation_types)}`",
        f"- required_gate_operation_types_seen: `{', '.join(gate_ops_seen)}`",
        "",
        "Parity rows are in `operation_trace_parity_rows.csv`; flattened operation rows are in `operation_trace_rows.csv`.",
    ]
    write_text(STAGE1 / "no_action_parity_report.md", "\n".join(report_lines))

    if not stage1_pass:
        write_text(
            STAGE1 / "CACHE_OPERATION_OBSERVABILITY_BLOCKED.md",
            "\n".join(
                [
                    "# CACHE_OPERATION_OBSERVABILITY_BLOCKED",
                    "",
                    f"- trace_parity_pass: `{all_parity_pass}`",
                    f"- operation_row_count: `{len(operation_rows_out)}`",
                    f"- observed_operation_types: `{', '.join(operation_types)}`",
                    f"- trace_error_rows: `{trace_error_rows}`",
                    "",
                    "Stage1 did not meet the operation observability gate. Do not enter Stage3/Stage4 from this summary.",
                ]
            ),
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--target-kinds", default="")
    parser.add_argument("--target-ids", default="")
    args = parser.parse_args()
    set_stage1_root(args.stage1_root)
    build_summary(
        seqs=parse_list(args.seqs),
        target_kinds=parse_list(args.target_kinds),
        target_ids=parse_list(args.target_ids),
    )


if __name__ == "__main__":
    main()
