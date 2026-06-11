#!/usr/bin/env python3
"""Track F lightweight washout attribution for v31 h10/h15 diagnostic rows.

This script intentionally uses only landed JSONL/CSV artifacts. If tensor state
snapshots are absent, it does not pretend to compute true tensor overwrite
norms; it writes proxy ratios from HMC hash/debug summaries and marks the
evidence level accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


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


def _read_metric_row(path: Path, candidate: str, chunk: int | None = None, horizon: int | None = None) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not (row.get("candidate") == candidate or row.get("candidate_id") == candidate):
                continue
            if chunk is not None and str(row.get("chunk", "")) != str(chunk):
                continue
            if horizon is not None and str(row.get("horizon", "")) != str(horizon):
                continue
            return dict(row)
    return {}


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _chunk_keyed(rows: Iterable[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        try:
            chunk = int(row.get("chunk_idx", row.get("chunk", -1)))
        except (TypeError, ValueError):
            continue
        if chunk >= 0:
            out[chunk] = row
    return out


def _nested_get(row: Mapping[str, Any], dotted: str, default: Any = float("nan")) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _sum_metric(rows: Iterable[Mapping[str, Any]], dotted: str) -> float:
    total = 0.0
    found = False
    for row in rows:
        value = _to_float(_nested_get(row, dotted))
        if math.isfinite(value):
            total += value
            found = True
    return total if found else float("nan")


def _mean_metric(rows: Iterable[Mapping[str, Any]], dotted: str) -> float:
    vals = [_to_float(_nested_get(row, dotted)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _hash_match_rows(h10: Dict[int, Mapping[str, Any]], h15: Dict[int, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for chunk in sorted(set(h10) & set(h15)):
        a = h10[chunk]
        b = h15[chunk]
        rows.append({
            "chunk_idx": chunk,
            "h10_hash_H_next": a.get("hash_H_next", ""),
            "h15_hash_H_next": b.get("hash_H_next", ""),
            "hash_H_next_equal": a.get("hash_H_next") == b.get("hash_H_next"),
            "h10_controlled_output_state_hash": a.get("controlled_output_state_hash", ""),
            "h15_controlled_output_state_hash": b.get("controlled_output_state_hash", ""),
            "controlled_output_hash_equal": a.get("controlled_output_state_hash") == b.get("controlled_output_state_hash"),
            "h10_ttt_mean_rel_diff": _nested_get(a, "memory_side_effect.ttt_state_diff.mean_rel_diff"),
            "h15_ttt_mean_rel_diff": _nested_get(b, "memory_side_effect.ttt_state_diff.mean_rel_diff"),
            "h10_frame_mean_abs_bias": _nested_get(a, "control_trace.hook_effect_summary.frame_attention.mean_abs_bias"),
            "h15_frame_mean_abs_bias": _nested_get(b, "control_trace.hook_effect_summary.frame_attention.mean_abs_bias"),
        })
    return rows


def _role_rows(h10_rows: List[Mapping[str, Any]], h15_rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    def collect(rows: List[Mapping[str, Any]], split: str) -> Dict[tuple[str, str, str], float]:
        acc: Dict[tuple[str, str, str], float] = defaultdict(float)
        for row in rows:
            by_path = row.get("fine_label_path_role_counts")
            if not isinstance(by_path, Mapping):
                continue
            for path, labels in by_path.items():
                if not isinstance(labels, Mapping):
                    continue
                for label, roles in labels.items():
                    if not isinstance(roles, Mapping):
                        continue
                    for role, count in roles.items():
                        acc[(str(path), str(label), str(role))] += _to_float(count)
        return acc

    h10 = collect(h10_rows, "h10")
    h15 = collect(h15_rows, "h15")
    out: List[Dict[str, Any]] = []
    for key in sorted(set(h10) | set(h15)):
        h10_count = h10.get(key, 0.0)
        h15_count = h15.get(key, 0.0)
        out.append({
            "path": key[0],
            "fine_label_id": key[1],
            "role_id": key[2],
            "h10_total_count": h10_count,
            "h15_total_count": h15_count,
            "h15_minus_h10": h15_count - h10_count,
        })
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    f = _to_float(value)
    if math.isfinite(f):
        return f"{f:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h10-run-dir", required=True, type=Path)
    parser.add_argument("--h15-run-dir", required=True, type=Path)
    parser.add_argument("--h10-report-csv", required=True, type=Path)
    parser.add_argument("--h15-report-csv", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--h10-horizon", type=int, default=10)
    parser.add_argument("--h15-horizon", type=int, default=15)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    h10_state = _read_jsonl(args.h10_run_dir / "hmc_state_hash.jsonl")
    h15_state = _read_jsonl(args.h15_run_dir / "hmc_state_hash.jsonl")
    h10_sem = _read_jsonl(args.h10_run_dir / "semantic_memory_path_summary.jsonl")
    h15_sem = _read_jsonl(args.h15_run_dir / "semantic_memory_path_summary.jsonl")

    h10_by_chunk = _chunk_keyed(h10_state)
    h15_by_chunk = _chunk_keyed(h15_state)
    common_chunks = sorted(set(h10_by_chunk) & set(h15_by_chunk))
    tail_chunks = sorted(set(h15_by_chunk) - set(h10_by_chunk))
    h10_common = [h10_by_chunk[c] for c in common_chunks]
    h15_common = [h15_by_chunk[c] for c in common_chunks]
    h15_tail = [h15_by_chunk[c] for c in tail_chunks]

    h10_metric = _read_metric_row(args.h10_report_csv, args.candidate, args.chunk, args.h10_horizon)
    h15_metric = _read_metric_row(args.h15_report_csv, args.candidate, args.chunk, args.h15_horizon)
    h10_ate = _to_float(h10_metric.get("ATE_delta_vs_H9"))
    h15_ate = _to_float(h15_metric.get("ATE_delta_vs_H9"))
    h10_seg = _to_float(h10_metric.get("intersection_200_300_delta_vs_H9"))
    h15_seg = _to_float(h15_metric.get("intersection_200_300_delta_vs_H9"))

    h10_ttt_total = _sum_metric(h10_state, "memory_side_effect.ttt_state_diff.mean_rel_diff")
    h15_common_ttt_total = _sum_metric(h15_common, "memory_side_effect.ttt_state_diff.mean_rel_diff")
    h15_tail_ttt_total = _sum_metric(h15_tail, "memory_side_effect.ttt_state_diff.mean_rel_diff")
    h10_frame_bias_total = _sum_metric(h10_state, "control_trace.hook_effect_summary.frame_attention.mean_abs_bias")
    h15_tail_frame_bias_total = _sum_metric(h15_tail, "control_trace.hook_effect_summary.frame_attention.mean_abs_bias")
    h10_swa_replace_total = _sum_metric(h10_state, "control_trace.hook_effect_summary.swa_read.mean_swa_overlap_source_replace_alpha")
    h15_tail_swa_replace_total = _sum_metric(h15_tail, "control_trace.hook_effect_summary.swa_read.mean_swa_overlap_source_replace_alpha")

    path_rows = [
        {
            "path": "ttt",
            "evidence": "proxy_from_hmc_state_hash_jsonl",
            "h10_total_side_effect": h10_ttt_total,
            "h15_common_side_effect": h15_common_ttt_total,
            "h15_tail_side_effect": h15_tail_ttt_total,
            "tail_over_h10_ratio": h15_tail_ttt_total / (h10_ttt_total + 1e-8) if math.isfinite(h10_ttt_total) else float("nan"),
            "common_chunks": len(common_chunks),
            "tail_chunks": len(tail_chunks),
        },
        {
            "path": "frame_attention_bias",
            "evidence": "proxy_from_hook_effect_summary",
            "h10_total_side_effect": h10_frame_bias_total,
            "h15_tail_side_effect": h15_tail_frame_bias_total,
            "tail_over_h10_ratio": h15_tail_frame_bias_total / (h10_frame_bias_total + 1e-8) if math.isfinite(h10_frame_bias_total) else float("nan"),
            "common_chunks": len(common_chunks),
            "tail_chunks": len(tail_chunks),
        },
        {
            "path": "swa_source_replace",
            "evidence": "proxy_from_hook_effect_summary",
            "h10_total_side_effect": h10_swa_replace_total,
            "h15_tail_side_effect": h15_tail_swa_replace_total,
            "tail_over_h10_ratio": h15_tail_swa_replace_total / (h10_swa_replace_total + 1e-8) if math.isfinite(h10_swa_replace_total) else float("nan"),
            "common_chunks": len(common_chunks),
            "tail_chunks": len(tail_chunks),
        },
    ]
    _write_csv(args.out_dir / "path_overwrite_ratio.csv", path_rows)
    (args.out_dir / "path_overwrite_ratio.json").write_text(
        json.dumps(path_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    hash_rows = _hash_match_rows(h10_by_chunk, h15_by_chunk)
    _write_csv(args.out_dir / "h10_h15_state_norms.csv", hash_rows)
    role_rows = _role_rows(h10_sem, h15_sem)
    _write_csv(args.out_dir / "label_role_mass_h10_h15.csv", role_rows)

    summary = {
        "candidate": args.candidate,
        "evidence_level": "proxy_only_no_tensor_state_snapshots",
        "h10_ATE_delta_vs_H9": h10_ate,
        "h15_ATE_delta_vs_H9": h15_ate,
        "h10_200_300_delta_vs_H9": h10_seg,
        "h15_200_300_delta_vs_H9": h15_seg,
        "durability_abs_h15_ate_over_h10_ate": abs(h15_ate) / (abs(h10_ate) + 1e-8) if math.isfinite(h10_ate) else float("nan"),
        "durability_abs_h15_200_300_over_h10_200_300": abs(h15_seg) / (abs(h10_seg) + 1e-8) if math.isfinite(h10_seg) else float("nan"),
        "common_chunk_count": len(common_chunks),
        "tail_chunk_count": len(tail_chunks),
        "common_hash_H_next_equal_count": sum(1 for row in hash_rows if row["hash_H_next_equal"]),
        "common_controlled_output_hash_equal_count": sum(1 for row in hash_rows if row["controlled_output_hash_equal"]),
        "h10_ttt_mean_rel_diff_sum": h10_ttt_total,
        "h15_tail_ttt_mean_rel_diff_sum": h15_tail_ttt_total,
        "h15_tail_ttt_over_h10_ratio": h15_tail_ttt_total / (h10_ttt_total + 1e-8) if math.isfinite(h10_ttt_total) else float("nan"),
        "h10_ttt_mean_rel_diff_mean": _mean_metric(h10_state, "memory_side_effect.ttt_state_diff.mean_rel_diff"),
        "h15_tail_ttt_mean_rel_diff_mean": _mean_metric(h15_tail, "memory_side_effect.ttt_state_diff.mean_rel_diff"),
        "h15_tail_frame_bias_over_h10_ratio": h15_tail_frame_bias_total / (h10_frame_bias_total + 1e-8) if math.isfinite(h10_frame_bias_total) else float("nan"),
        "h15_tail_swa_replace_over_h10_ratio": h15_tail_swa_replace_total / (h10_swa_replace_total + 1e-8) if math.isfinite(h10_swa_replace_total) else float("nan"),
        "tensor_state_snapshots_available": False,
    }
    (args.out_dir / "track_f_washout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md = [
        "# v31 Track F Washout Attribution",
        "",
        f"Candidate: `{args.candidate}`",
        "",
        "Evidence level: `proxy_only_no_tensor_state_snapshots`.",
        "No `.pt` HMC/merge state snapshots were found in the h10/h15 rollout dirs, so this report does not claim full tensor overwrite attribution.",
        "",
        "## Trajectory Durability",
        "",
        "| Metric | h10 | h15 | abs durability |",
        "|---|---:|---:|---:|",
        f"| ATE delta vs H9 | {_fmt(h10_ate)} | {_fmt(h15_ate)} | {_fmt(summary['durability_abs_h15_ate_over_h10_ate'])} |",
        f"| [200,300) delta vs H9 | {_fmt(h10_seg)} | {_fmt(h15_seg)} | {_fmt(summary['durability_abs_h15_200_300_over_h10_200_300'])} |",
        "",
        "## Path Overwrite Proxy",
        "",
        "| Path | h10 total | h15 tail total | tail/h10 | Evidence |",
        "|---|---:|---:|---:|---|",
    ]
    for row in path_rows:
        md.append(
            f"| `{row['path']}` | {_fmt(row.get('h10_total_side_effect'))} | "
            f"{_fmt(row.get('h15_tail_side_effect'))} | {_fmt(row.get('tail_over_h10_ratio'))} | "
            f"`{row['evidence']}` |"
        )
    md.extend([
        "",
        "## Interpretation",
        "",
        f"- Common chunks compared: `{len(common_chunks)}`.",
        f"- h15-only tail chunks: `{len(tail_chunks)}`.",
        f"- Common `hash_H_next` exact matches: `{summary['common_hash_H_next_equal_count']}/{len(common_chunks)}`.",
        f"- TTT tail/h10 side-effect proxy ratio: `{_fmt(summary['h15_tail_ttt_over_h10_ratio'])}`.",
        f"- Frame-bias tail/h10 proxy ratio: `{_fmt(summary['h15_tail_frame_bias_over_h10_ratio'])}`.",
        f"- SWA replace tail/h10 proxy ratio: `{_fmt(summary['h15_tail_swa_replace_over_h10_ratio'])}`.",
        "",
        "The h15 rollout adds extra chunks after the h10 endpoint. The path proxy rows above indicate which landed hook/state summaries continue to move in that tail.",
        "This is a lightweight attribution from JSONL summaries only, not a full tensor-state proof.",
        "",
    ])
    (args.out_dir / "memory_path_washout_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
