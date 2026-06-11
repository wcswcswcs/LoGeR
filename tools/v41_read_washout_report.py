#!/usr/bin/env python3
"""Proxy washout attribution for v41 READ h10-effective / h15-failed rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _nested(row: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _sum(rows: Iterable[Mapping[str, Any]], keys: List[str]) -> float:
    total = 0.0
    found = False
    for row in rows:
        for key in keys:
            value = _f(_nested(row, key) if "." in key else row.get(key), float("nan"))
            if math.isfinite(value):
                total += value
                found = True
                break
    return total if found else float("nan")


def _by_chunk(rows: Iterable[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        chunk = int(_f(row.get("chunk_idx", row.get("chunk", -1)), -1))
        if chunk >= 0:
            out[chunk] = row
    return out


def _metric_row(path: Path, parent: str, chunk: int, candidate: str) -> Dict[str, str]:
    for row in _read_csv(path):
        if row.get("parent") == parent and int(_f(row.get("chunk"), -1)) == int(chunk) and row.get("candidate") == candidate:
            return row
    return {}


def _ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-9:
        return float("nan")
    return abs(num) / (abs(den) + 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--h10-run-dir", type=Path, required=True)
    parser.add_argument("--h15-run-dir", type=Path, required=True)
    parser.add_argument("--h10-effects", type=Path, required=True)
    parser.add_argument("--h15-effects", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    h10_state = _read_jsonl(args.h10_run_dir / "hmc_state_hash.jsonl")
    h15_state = _read_jsonl(args.h15_run_dir / "hmc_state_hash.jsonl")
    h10_by = _by_chunk(h10_state)
    h15_by = _by_chunk(h15_state)
    common = sorted(set(h10_by) & set(h15_by))
    tail = sorted(set(h15_by) - set(h10_by))
    h10_common = [h10_by[c] for c in common]
    h15_common = [h15_by[c] for c in common]
    h15_tail = [h15_by[c] for c in tail]

    h10 = _metric_row(args.h10_effects, args.parent, args.chunk, args.candidate)
    h15 = _metric_row(args.h15_effects, args.parent, args.chunk, args.candidate)
    metrics = {
        "ATE_delta": (_f(h10.get("ATE_delta_vs_base")), _f(h15.get("ATE_delta_vs_base"))),
        "rolling100_best_delta": (_f(h10.get("rolling_100f_best_delta_vs_base")), _f(h15.get("rolling_100f_best_delta_vs_base"))),
        "stress_200_300_delta": (_f(h10.get("intersection_200_300_delta_vs_base")), _f(h15.get("intersection_200_300_delta_vs_base"))),
        "downstream_400_600_delta": (_f(h10.get("intersection_400_600_delta_vs_base")), _f(h15.get("intersection_400_600_delta_vs_base"))),
    }

    path_rows = []
    specs = [
        (
            "ttt_state",
            ["memory_ttt_mean_rel_diff", "memory_side_effect.ttt_state_diff.mean_rel_diff"],
        ),
        (
            "frame_attention_bias",
            ["control_trace.hook_effect_summary.frame_attention.mean_abs_bias"],
        ),
        (
            "chunk_attention_source_keep",
            ["control_trace.hook_effect_summary.chunk_attention.mean_context_source_keep_ratio"],
        ),
        (
            "swa_source_replace",
            ["control_trace.hook_effect_summary.swa_read.mean_swa_overlap_source_replace_alpha"],
        ),
    ]
    for name, keys in specs:
        h10_total = _sum(h10_state, keys)
        h15_common_total = _sum(h15_common, keys)
        h15_tail_total = _sum(h15_tail, keys)
        path_rows.append({
            "path": name,
            "h10_total": h10_total,
            "h15_common_total": h15_common_total,
            "h15_tail_total": h15_tail_total,
            "h15_tail_over_h10_ratio": h15_tail_total / (h10_total + 1e-9) if math.isfinite(h10_total) else float("nan"),
            "evidence": "proxy_from_hmc_state_hash_jsonl",
        })

    hash_rows = []
    for chunk in common:
        h10_row = h10_by[chunk]
        h15_row = h15_by[chunk]
        hash_rows.append({
            "chunk_idx": chunk,
            "h10_hash_H_next": h10_row.get("hash_H_next", ""),
            "h15_hash_H_next": h15_row.get("hash_H_next", ""),
            "hash_H_next_equal": h10_row.get("hash_H_next") == h15_row.get("hash_H_next"),
        })

    metric_rows = []
    for name, (h10_val, h15_val) in metrics.items():
        metric_rows.append({
            "metric": name,
            "h10_delta": h10_val,
            "h15_delta": h15_val,
            "abs_h15_over_h10": _ratio(h15_val, h10_val),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "read_washout_path_proxy.csv", path_rows)
    _write_csv(args.out_dir / "read_washout_metric_durability.csv", metric_rows)
    _write_csv(args.out_dir / "read_washout_h10_h15_hash.csv", hash_rows)
    summary = {
        "parent": args.parent,
        "chunk": args.chunk,
        "candidate": args.candidate,
        "evidence_level": "proxy_only_no_tensor_state_snapshots",
        "common_chunk_count": len(common),
        "tail_chunk_count": len(tail),
        "h15_tail_chunks": tail,
        "metric_durability": {row["metric"]: row["abs_h15_over_h10"] for row in metric_rows},
        "path_tail_over_h10": {row["path"]: row["h15_tail_over_h10_ratio"] for row in path_rows},
        "interpretation_boundary": "JSONL proxy only; no tensor-state overwrite norm is claimed.",
    }
    _write_json(args.out_dir / "read_washout_summary.json", summary)
    lines = [
        "# v41 READ Washout Attribution",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "| Metric | h10 delta | h15 delta | abs durability |",
        "|---|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(f"| `{row['metric']}` | {_f(row['h10_delta']):.10f} | {_f(row['h15_delta']):.10f} | {_f(row['abs_h15_over_h10']):.10f} |")
    lines.extend([
        "",
        "| Path proxy | h10 total | h15 tail total | tail/h10 |",
        "|---|---:|---:|---:|",
    ])
    for row in path_rows:
        lines.append(f"| `{row['path']}` | {_f(row['h10_total']):.10f} | {_f(row['h15_tail_total']):.10f} | {_f(row['h15_tail_over_h10_ratio']):.10f} |")
    lines.extend([
        "",
        "Boundary: this is proxy-only attribution from landed JSONL summaries. It does not claim tensor-state overwrite proof.",
        "",
    ])
    (args.out_dir / "read_washout_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
