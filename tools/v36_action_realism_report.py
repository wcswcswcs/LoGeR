#!/usr/bin/env python3
"""Aggregate v36 H0 action-realism smoke rows.

The report is intentionally conservative: metrics not instrumented in landed
runtime logs are recorded as unavailable and prevent H0 from passing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _status(run_dir: Path) -> str:
    status = run_dir / "run_status.txt"
    if status.exists():
        text = status.read_text(encoding="utf-8", errors="replace")
        for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
            if " DONE " in f" {line} " or line.endswith(" DONE") or " DONE " in line:
                return "DONE"
            if " FAIL " in f" {line} " or line.endswith(" FAIL") or " FAIL " in line:
                return "FAIL"
        return text.strip()
    if (run_dir / "FAIL").exists():
        return "FAIL"
    if (run_dir / "DONE").exists():
        return "DONE"
    return "MISSING_STATUS"


def _float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mask_indices(mask: str, num_frames: int, h: int, w: int) -> Set[int]:
    mask = str(mask or "none")
    out: Set[int] = set()
    total = max(0, int(num_frames) * int(h) * int(w))
    if mask in {"none", ""}:
        return out
    if mask in {"all_patch_skip", "all_dynamic_role", "all_static_role"}:
        return set(range(total))
    for t in range(int(num_frames)):
        for y in range(int(h)):
            for x in range(int(w)):
                keep = False
                if mask == "center_box_skip":
                    keep = (round(0.25 * h) <= y < round(0.75 * h)) and (round(0.25 * w) <= x < round(0.75 * w))
                elif mask == "left_half_skip":
                    keep = x < max(1, int(w) // 2)
                elif mask == "random_20pct_skip":
                    keep = ((t * 73856093 + y * 19349663 + x * 83492791) % 100) < 20
                if keep:
                    out.add((t * int(h) + y) * int(w) + x)
    return out


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _post_zp_action_ratio(trace_dir: Any) -> Tuple[float, float, int]:
    if not trace_dir:
        return math.nan, math.nan, 0
    path = Path(str(trace_dir)) / "basis_projection_coefficients.csv"
    if not path.exists():
        return math.nan, math.nan, 0
    ratios: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            native = _float(row.get("native_delta_norm"), math.nan)
            action = _float(row.get("action_delta_norm"), math.nan)
            if math.isfinite(native) and native > 1e-12 and math.isfinite(action):
                ratios.append(float(action / native))
    if not ratios:
        return math.nan, math.nan, 0
    return float(max(ratios)), float(sum(ratios) / len(ratios)), int(len(ratios))


def collect(run_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    action_rows: List[Dict[str, Any]] = []
    hook_rows: List[Dict[str, Any]] = []
    masks: List[Dict[str, Any]] = []
    swa_rows: List[Dict[str, Any]] = []
    ttt_rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        hmc_rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
        hook_jsonl = _read_jsonl(run_dir / "hook_effect_summary.jsonl")
        context_rows = _read_jsonl(run_dir / "context_skip_summary.jsonl")
        latest = hmc_rows[-1] if hmc_rows else {}
        v36 = latest.get("v36_synthetic_intervention") or {}
        if not isinstance(v36, dict):
            v36 = {}
        trace = latest.get("control_trace") or {}
        v11_trace = latest.get("v11_projection_trace_summary") or {}
        v18 = v11_trace.get("v18_true_action_artifact_summary") if isinstance(v11_trace, dict) else {}
        trace_dir = v18.get("v18_artifact_dir") if isinstance(v18, dict) else None
        post_zp_max_ratio, post_zp_mean_ratio, post_zp_rows = _post_zp_action_ratio(trace_dir)
        action_row = {
            "run_name": run_dir.name,
            "status": _status(run_dir),
            "chunk_idx": latest.get("chunk_idx"),
            "start_frame": latest.get("start_frame"),
            "end_frame": latest.get("end_frame"),
            "mask": v36.get("mask", ""),
            "path": v36.get("path", ""),
            "action": v36.get("action", ""),
            "role": v36.get("role", ""),
            "token_count": v36.get("token_count", ""),
            "patch_token_count": v36.get("patch_token_count", ""),
            "selected_fraction": v36.get("selected_fraction", ""),
            "num_frames": v36.get("num_frames", ""),
            "patch_grid_h": v36.get("patch_grid_h", ""),
            "patch_grid_w": v36.get("patch_grid_w", ""),
            "mask_sha16": v36.get("mask_sha16", ""),
            "implemented_paths": ",".join(trace.get("implemented_paths", []) or []),
            "context_empty_source_events": 0,
            "max_context_source_skip_tokens": 0,
            "min_mean_keep_ratio": 1.0,
            "ttt_update_norm_changed_ratio": latest.get("ttt_update_norm_changed_ratio", ""),
            "post_zp_delta_norm_changed_ratio": (
                post_zp_max_ratio
                if math.isfinite(post_zp_max_ratio)
                else latest.get("post_zp_delta_norm_changed_ratio", "")
            ),
        }
        keep_vals: List[float] = []
        empty_total = 0
        skip_max = 0
        for row in context_rows:
            keep_vals.append(_float(row.get("mean_context_source_keep_ratio"), 1.0))
            empty_total += _int(row.get("num_context_empty_source_events"), 0)
            skip_max = max(skip_max, _int(row.get("max_context_source_skip_tokens"), 0))
            hook_rows.append({
                "run_name": run_dir.name,
                "path": row.get("path", ""),
                "num_context_source_skip_applied": row.get("num_context_source_skip_applied", ""),
                "mean_context_source_keep_ratio": row.get("mean_context_source_keep_ratio", ""),
                "max_context_source_skip_tokens": row.get("max_context_source_skip_tokens", ""),
                "num_context_empty_source_events": row.get("num_context_empty_source_events", ""),
            })
        if keep_vals:
            action_row["min_mean_keep_ratio"] = min(keep_vals)
        action_row["context_empty_source_events"] = empty_total
        action_row["max_context_source_skip_tokens"] = skip_max
        action_rows.append(action_row)
        if v36:
            masks.append(v36)
        for hook_obj in hook_jsonl:
            summary = hook_obj.get("hook_effect_summary") or {}
            if not isinstance(summary, dict):
                continue
            for path_name, value in summary.items():
                if not isinstance(value, dict):
                    continue
                hook_rows.append({
                    "run_name": run_dir.name,
                    "path": path_name,
                    "num_context_source_skip_applied": value.get("num_context_source_skip_applied", ""),
                    "mean_context_source_keep_ratio": value.get("mean_context_source_keep_ratio", ""),
                    "max_context_source_skip_tokens": value.get("max_context_source_skip_tokens", ""),
                    "num_context_empty_source_events": value.get("num_context_empty_source_events", ""),
                })
                if path_name == "swa_read":
                    for key in (
                        "num_calls", "num_source_gate_applied", "mean_swa_gate", "mean_abs_gate_delta",
                        "max_abs_gate_delta", "max_d_prev_tokens", "max_history_tokens",
                        "num_swa_overlap_source_gate_applied", "mean_swa_overlap_source_gate",
                        "mean_swa_overlap_source_gate_delta", "num_swa_overlap_source_replace_applied",
                    ):
                        if key in value:
                            swa_rows.append({"run_name": run_dir.name, "metric": key, "value": value.get(key)})
                if path_name == "ttt_apply":
                    for key in ("num_calls", "num_enabled_layers"):
                        if key in value:
                            ttt_rows.append({"run_name": run_dir.name, "metric": key, "value": value.get(key)})
        for key in (
            "memory_ttt_mean_rel_diff", "memory_ttt_max_rel_diff",
            "memory_ttt_w0_mean_rel_diff", "memory_ttt_w1_mean_rel_diff", "memory_ttt_w2_mean_rel_diff",
            "memory_ttt_w0_max_rel_diff", "memory_ttt_w1_max_rel_diff", "memory_ttt_w2_max_rel_diff",
        ):
            if key in latest:
                ttt_rows.append({"run_name": run_dir.name, "metric": key, "value": latest.get(key)})
        if isinstance(v11_trace, dict) and v11_trace:
            for key in (
                "token_group_written", "role_rows", "matrix_layers", "projection_action_mode",
                "logging_only_no_action_change", "projection_harmful_energy", "projection_helpful_energy",
                "ttt_update_conflict_energy",
            ):
                if key in v11_trace:
                    ttt_rows.append({"run_name": run_dir.name, "metric": f"v11_{key}", "value": v11_trace.get(key)})
        if isinstance(v18, dict) and v18:
            for key in ("v18_true_tensor_basis", "v18_artifact_layers", "v18_artifact_coeff_rows", "v18_artifact_dir"):
                if key in v18:
                    ttt_rows.append({"run_name": run_dir.name, "metric": key, "value": v18.get(key)})
        if post_zp_rows:
            ttt_rows.extend([
                {"run_name": run_dir.name, "metric": "post_zp_action_delta_over_native_max", "value": post_zp_max_ratio},
                {"run_name": run_dir.name, "metric": "post_zp_action_delta_over_native_mean", "value": post_zp_mean_ratio},
                {"run_name": run_dir.name, "metric": "post_zp_action_delta_rows", "value": post_zp_rows},
            ])
    jaccard_rows: List[Dict[str, Any]] = []
    indexed: List[Tuple[str, Set[int]]] = []
    for m in masks:
        name = f"{m.get('path','')}:{m.get('action','')}:{m.get('mask','')}"
        idx = _mask_indices(
            str(m.get("mask", "none")),
            _int(m.get("num_frames"), 0),
            _int(m.get("patch_grid_h"), 0),
            _int(m.get("patch_grid_w"), 0),
        )
        indexed.append((name, idx))
    for name_a, idx_a in indexed:
        for name_b, idx_b in indexed:
            jaccard_rows.append({"action_a": name_a, "action_b": name_b, "jaccard": _jaccard(idx_a, idx_b)})
    return action_rows, hook_rows, jaccard_rows, swa_rows, ttt_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    action_rows, hook_rows, jaccard_rows, swa_rows, ttt_rows = collect(run_root)
    _write_csv(
        out_dir / "action_tensor_summary.csv",
        action_rows,
        [
            "run_name", "status", "chunk_idx", "start_frame", "end_frame", "mask", "path", "action",
            "role", "token_count", "patch_token_count", "selected_fraction", "num_frames",
            "patch_grid_h", "patch_grid_w", "mask_sha16", "implemented_paths",
            "context_empty_source_events", "max_context_source_skip_tokens", "min_mean_keep_ratio",
            "ttt_update_norm_changed_ratio", "post_zp_delta_norm_changed_ratio",
        ],
    )
    _write_csv(
        out_dir / "source_attention_mass_removed.csv",
        hook_rows,
        [
            "run_name", "path", "num_context_source_skip_applied", "mean_context_source_keep_ratio",
            "max_context_source_skip_tokens", "num_context_empty_source_events",
        ],
    )
    _write_csv(out_dir / "action_jaccard_matrix.csv", jaccard_rows, ["action_a", "action_b", "jaccard"])
    # These are only landed runtime summaries.  They are not upgraded into
    # uninstrumented attention-mass or post-zeropower claims.
    _write_csv(out_dir / "swa_cache_effect_summary.csv", swa_rows, ["run_name", "metric", "value"])
    _write_csv(out_dir / "ttt_update_effect_summary.csv", ttt_rows, ["run_name", "metric", "value"])

    done_rows = [r for r in action_rows if str(r.get("status")) == "DONE"]
    context_empty = sum(_int(r.get("context_empty_source_events"), 0) for r in done_rows)
    synthetic_effect_rows = [
        r for r in done_rows
        if _int(r.get("max_context_source_skip_tokens"), 0) > 0 and _float(r.get("min_mean_keep_ratio"), 1.0) < 0.98
    ]
    attention_mass_instrumented = False
    ttt_instrumented = any(
        str(r.get("post_zp_delta_norm_changed_ratio", "")) not in {"", "None", "nan"}
        for r in done_rows
    )
    gate_pass = bool(
        done_rows
        and context_empty == 0
        and synthetic_effect_rows
        and attention_mass_instrumented
        and ttt_instrumented
    )
    missing: List[str] = []
    if not attention_mass_instrumented:
        missing.append("attention-mass-removed instrumentation")
    if not ttt_instrumented:
        missing.append("TTT post-zp/update-norm instrumentation")
    summary = {
        "h0_gate_pass": gate_pass,
        "rows_found": len(action_rows),
        "rows_done": len(done_rows),
        "context_empty_source_events_total": context_empty,
        "synthetic_source_effect_rows": len(synthetic_effect_rows),
        "attention_mass_removed_instrumented": attention_mass_instrumented,
        "ttt_post_zp_delta_instrumented": ttt_instrumented,
        "blocked_reason": None if gate_pass else (
            "H0 cannot pass until " + " and ".join(missing) + " is present for the landed smoke rows."
            if missing else "H0 conditions were not met by the landed smoke rows."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hook_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# v36 H0 Action Realism Audit",
        "",
        f"rows_found = {summary['rows_found']}",
        f"rows_done = {summary['rows_done']}",
        f"context_empty_source_events_total = {summary['context_empty_source_events_total']}",
        f"synthetic_source_effect_rows = {summary['synthetic_source_effect_rows']}",
        f"attention_mass_removed_instrumented = {summary['attention_mass_removed_instrumented']}",
        f"ttt_post_zp_delta_instrumented = {summary['ttt_post_zp_delta_instrumented']}",
        f"h0_gate_pass = {summary['h0_gate_pass']}",
        "",
        "Boundary: unavailable metrics are not imputed.",
    ]
    if summary["blocked_reason"]:
        report.extend(["", f"Blocked reason: {summary['blocked_reason']}"])
    (out_dir / "hook_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
