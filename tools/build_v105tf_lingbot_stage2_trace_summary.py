#!/usr/bin/env python3
"""Summarize ACL2 v105-TF LingBot Stage 2 no-action trace parity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "third_party/lingbot-map/benchmark"
sys.path.insert(0, str(BENCH))

from benchmark.io.image import load_exr  # noqa: E402
from benchmark.io.intrinsics import read_intrinsics  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
STAGE0_SUMMARY = RESULT_ROOT / "stage0_repo_env_audit/stage0_summary.json"
STAGE2 = RESULT_ROOT / "stage2_gca_trace"
WORKSPACE = STAGE2 / "workspace"
TRACE_DIR = STAGE2 / "raw_trace"
SEQUENCES = ["00", "02"]
NOTRACE_METHOD = "lingbot_map_stream_default_stage2_notrace"
TRACE_METHOD = "lingbot_map_stream_default_stage2_trace"


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.write_text(text, encoding="utf-8")


def bool_s(value: bool) -> str:
    return "true" if value else "false"


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
    if diff.size == 0:
        return 0.0
    if np.all(np.isnan(diff)):
        return 0.0
    return float(np.nanmax(diff))


def load_intr(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return read_intrinsics(path)


def exr_files(root: Path, name: str) -> list[Path]:
    directory = root / name
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.exr"))


def max_abs_diff_exr_dir(lhs_root: Path, rhs_root: Path, name: str) -> tuple[float | None, str]:
    lhs_files = exr_files(lhs_root, name)
    rhs_files = exr_files(rhs_root, name)
    if not lhs_files or not rhs_files:
        return None, f"{name}_exr_missing"
    lhs_names = [p.name for p in lhs_files]
    rhs_names = [p.name for p in rhs_files]
    if lhs_names != rhs_names:
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
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


def trace_path(dataset: str, seq: str) -> Path:
    return TRACE_DIR / f"{dataset}_{seq}_{TRACE_METHOD}.jsonl"


def trace_rows_for(dataset: str, seq: str) -> list[dict[str, Any]]:
    rows = load_jsonl(trace_path(dataset, seq))
    for row in rows:
        row.setdefault("dataset", dataset)
        row.setdefault("seq", seq)
        row.setdefault("method", TRACE_METHOD)
    return rows


def compare_case(dataset: str, seq: str, trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    notrace_root = WORKSPACE / dataset / seq / NOTRACE_METHOD
    trace_root = WORKSPACE / dataset / seq / TRACE_METHOD
    reasons: list[str] = []

    if not (notrace_root / ".complete.json").is_file():
        reasons.append("notrace_complete_missing")
    if not (trace_root / ".complete.json").is_file():
        reasons.append("trace_complete_missing")

    notrace_traj = notrace_root / "traj.txt"
    trace_traj = trace_root / "traj.txt"
    pose_sha_equal = sha256_file(notrace_traj) != "" and sha256_file(notrace_traj) == sha256_file(trace_traj)
    pose_diff = max_abs_diff_arrays(load_traj(notrace_traj), load_traj(trace_traj))
    if pose_diff is None:
        reasons.append("pose_missing")

    intr_diff = max_abs_diff_arrays(load_intr(notrace_root / "intrinsics.txt"), load_intr(trace_root / "intrinsics.txt"))
    if intr_diff is None:
        reasons.append("intrinsics_missing")

    depth_diff, depth_reason = max_abs_diff_exr_dir(notrace_root, trace_root, "depth")
    if depth_reason:
        reasons.append(depth_reason)

    conf_diff, conf_reason = max_abs_diff_exr_dir(notrace_root, trace_root, "confidence")
    if conf_reason:
        reasons.append(conf_reason)

    trace_error_count = sum(1 for row in trace_rows if row.get("row_type") == "trace_error")
    gca_count = sum(1 for row in trace_rows if row.get("row_type") == "gca_context_topk")
    kv_count = sum(1 for row in trace_rows if row.get("row_type") == "kv_cache_provenance")
    trace_payload_exists = bool(gca_count and kv_count)
    if not trace_payload_exists:
        reasons.append("trace_payload_empty_or_incomplete")
    if trace_error_count:
        reasons.append(f"trace_error_rows:{trace_error_count}")

    pose_ok = pose_sha_equal or (pose_diff is not None and pose_diff <= 1e-6)
    depth_ok = depth_diff is not None and depth_diff <= 1e-6
    intr_ok = intr_diff is not None and intr_diff <= 1e-6
    conf_ok = conf_diff is not None and conf_diff <= 1e-6
    complete_ok = (notrace_root / ".complete.json").is_file() and (trace_root / ".complete.json").is_file()
    parity_pass = complete_ok and pose_ok and depth_ok and intr_ok and conf_ok and trace_payload_exists and trace_error_count == 0

    if pose_diff is not None and not pose_ok:
        reasons.append("pose_diff_gt_1e-6")
    if depth_diff is not None and not depth_ok:
        reasons.append("depth_diff_gt_1e-6")
    if intr_diff is not None and not intr_ok:
        reasons.append("intrinsics_diff_gt_1e-6")
    if conf_diff is not None and not conf_ok:
        reasons.append("confidence_diff_gt_1e-6")

    return {
        "schema": "acl2_v105tf_lingbot_stage2_no_action_parity_row_v1",
        "dataset": dataset,
        "seq": seq,
        "notrace_method": NOTRACE_METHOD,
        "trace_method": TRACE_METHOD,
        "pose_sha_equal": bool_s(pose_sha_equal),
        "pose_max_abs_diff": "" if pose_diff is None else pose_diff,
        "depth_max_abs_diff": "" if depth_diff is None else depth_diff,
        "intrinsics_max_abs_diff": "" if intr_diff is None else intr_diff,
        "confidence_max_abs_diff": "" if conf_diff is None else conf_diff,
        "trace_payload_exists": bool_s(trace_payload_exists),
        "trace_row_count": len(trace_rows),
        "gca_context_topk_row_count": gca_count,
        "kv_cache_provenance_row_count": kv_count,
        "trace_error_row_count": trace_error_count,
        "job_returncode": 0 if complete_ok else "",
        "parity_pass": bool_s(parity_pass),
        "failure_reason": ";".join(dict.fromkeys(reasons)),
        "notrace_root": notrace_root.relative_to(ROOT).as_posix(),
        "trace_root": trace_root.relative_to(ROOT).as_posix(),
        "trace_file": trace_path(dataset, seq).relative_to(ROOT).as_posix(),
    }


def project_rows(rows: Iterable[dict[str, Any]], row_type: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("row_type") != row_type:
            continue
        selected.append(row)
    return selected


def aggregate_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    weights: defaultdict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("seq", ""),
            row.get("method", ""),
            row.get("trace_backend", ""),
            row.get("cache_mode", ""),
            row.get("key_context_role", ""),
            row.get("key_token_role", ""),
            row.get("query_token_role", ""),
        )
        if key not in groups:
            groups[key] = {
                "schema": "acl2_v105tf_lingbot_stage2_context_role_token_row_v1",
                "dataset": key[0],
                "seq": key[1],
                "method": key[2],
                "trace_backend": key[3],
                "cache_mode": key[4],
                "key_context_role": key[5],
                "key_token_role": key[6],
                "query_token_role": key[7],
                "row_count": 0,
                "global_idx_values": set(),
            }
        groups[key]["row_count"] += 1
        groups[key]["global_idx_values"].add(str(row.get("global_idx", "")))
        try:
            weights[key].append(float(row.get("attention_weight", 0.0)))
        except (TypeError, ValueError):
            weights[key].append(0.0)

    out: list[dict[str, Any]] = []
    for key, row in sorted(groups.items()):
        vals = weights[key]
        row = dict(row)
        row["attention_weight_sum"] = float(np.sum(vals)) if vals else 0.0
        row["attention_weight_mean"] = float(np.mean(vals)) if vals else 0.0
        row["attention_weight_max"] = float(np.max(vals)) if vals else 0.0
        row["global_idx_values"] = ",".join(sorted(x for x in row["global_idx_values"] if x != ""))
        out.append(row)
    return out


def build_backend_comparison(parity_rows: list[dict[str, Any]], flashinfer_available: bool) -> str:
    all_pass = all(row["parity_pass"] == "true" for row in parity_rows)
    lines = [
        "# ACL2 v105-TF Stage2 trace backend comparison",
        "",
        f"- Trace backend exercised: SDPA_TRACE",
        f"- FlashInfer available in recommended env: {bool_s(flashinfer_available)}",
        "- FlashInfer no-trace/trace comparison: not run because Stage0 found FlashInfer unavailable in `loger`.",
        "- Trace action status: trace-only; no routing action was run in Stage2.",
        f"- Overall SDPA_TRACE no-action parity pass: {bool_s(all_pass)}",
        "",
        "| seq | pose_sha_equal | pose_max_abs_diff | depth_max_abs_diff | intrinsics_max_abs_diff | confidence_max_abs_diff | trace_row_count | parity_pass | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in parity_rows:
        lines.append(
            f"| {row['seq']} | {row['pose_sha_equal']} | {row['pose_max_abs_diff']} | "
            f"{row['depth_max_abs_diff']} | {row['intrinsics_max_abs_diff']} | "
            f"{row['confidence_max_abs_diff']} | {row['trace_row_count']} | "
            f"{row['parity_pass']} | {row['failure_reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build() -> dict[str, Any]:
    stage0 = json.loads(STAGE0_SUMMARY.read_text(encoding="utf-8"))
    flashinfer_available = bool(stage0["environment"]["conda"]["flashinfer_available_in_recommended_env"])
    all_trace_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []

    for seq in SEQUENCES:
        dataset = f"kitti_v105_seq{seq}_trace32"
        trace_rows = trace_rows_for(dataset, seq)
        all_trace_rows.extend(trace_rows)
        parity_rows.append(compare_case(dataset, seq, trace_rows))

    gca_rows = project_rows(all_trace_rows, "gca_context_topk")
    kv_rows = project_rows(all_trace_rows, "kv_cache_provenance")
    context_rows = aggregate_context(gca_rows)

    write_csv(STAGE2 / "no_action_parity_rows.csv", parity_rows)
    write_csv(STAGE2 / "gca_context_trace_rows.csv", gca_rows)
    write_csv(STAGE2 / "kv_cache_provenance_rows.csv", kv_rows)
    write_csv(STAGE2 / "context_role_token_rows.csv", context_rows)
    write_text(STAGE2 / "trace_backend_comparison.md", build_backend_comparison(parity_rows, flashinfer_available))

    failed = [row for row in parity_rows if row["parity_pass"] != "true"]
    failure_path = STAGE2 / "TRACE_PARITY_FAILURE.md"
    if failed:
        lines = ["# TRACE_PARITY_FAILURE", ""]
        for row in failed:
            lines.append(f"- seq {row['seq']}: {row['failure_reason'] or 'unknown_failure'}")
        lines.append("")
        write_text(failure_path, "\n".join(lines))
    elif failure_path.exists():
        failure_path.unlink()

    summary = {
        "schema": "acl2_v105tf_lingbot_stage2_trace_summary_v1",
        "stage2_trace_parity_pass": not failed,
        "parity_rows": parity_rows,
        "gca_context_trace_rows": len(gca_rows),
        "kv_cache_provenance_rows": len(kv_rows),
        "context_role_token_rows": len(context_rows),
        "trace_error_rows": sum(1 for row in all_trace_rows if row.get("row_type") == "trace_error"),
    }
    write_text(STAGE2 / "trace_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
