#!/usr/bin/env python3
"""Summarize v36B H0C action-smoke source-skip distinguishability."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _last_path_summary(run_dir: Path, path_name: str) -> Dict[str, object]:
    rows = [r for r in _read_jsonl(run_dir / "context_skip_summary.jsonl") if r.get("path") == path_name]
    return rows[-1] if rows else {}


def _last_role_summary(run_dir: Path) -> Dict[str, object]:
    rows = _read_jsonl(run_dir / "semantic_role_summary.jsonl")
    return rows[-1] if rows else {}


def _role_distribution(summary: Dict[str, object], field: str) -> Dict[str, float]:
    counts = summary.get("path_role_counts", {})
    if not isinstance(counts, dict):
        return {}
    raw = counts.get(field, {})
    if not isinstance(raw, dict):
        return {}
    total = sum(float(v or 0.0) for v in raw.values())
    if total <= 0:
        return {}
    return {str(k): float(v or 0.0) / total for k, v in raw.items()}


def _dist_l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def _status_done(run_dir: Path, run_name: str) -> bool:
    text = (run_dir / "run_status.txt").read_text(encoding="utf-8", errors="replace") if (run_dir / "run_status.txt").exists() else ""
    return f"DONE {run_name}" in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--run-prefix", default="V36B_H0C_SMOKE_R1_H9")
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--candidates", default="V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_01,FG_SEM_02,FG_SEM_03,FG_SEM_04,FG_SEM_05")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rollout_root = Path(args.rollout_root)
    candidates = [x.strip() for x in str(args.candidates).split(",") if x.strip()]
    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for candidate in candidates:
        run_name = f"{args.run_prefix}_{candidate}_chunk{args.chunk}_h{args.horizon}_globalgate_H9parent_SWKS3"
        run_dir = rollout_root / run_name
        if not _status_done(run_dir, run_name):
            missing.append({"candidate": candidate, "run_name": run_name, "reason": "missing_or_not_done"})
            continue
        frame = _last_path_summary(run_dir, "frame_attention")
        chunk = _last_path_summary(run_dir, "chunk_attention")
        roles = _last_role_summary(run_dir)
        rows.append({
            "candidate": candidate,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "semantic_role_policy": roles.get("semantic_role_policy"),
            "semantic_memory_paths": roles.get("semantic_memory_paths"),
            "frame_keep_ratio": frame.get("mean_context_source_keep_ratio"),
            "frame_max_skip_tokens": frame.get("max_context_source_skip_tokens"),
            "frame_empty_source_events": frame.get("num_context_empty_source_events"),
            "global_keep_ratio": chunk.get("mean_context_source_keep_ratio"),
            "global_max_skip_tokens": chunk.get("max_context_source_skip_tokens"),
            "global_empty_source_events": chunk.get("num_context_empty_source_events"),
            "frame_role_distribution": json.dumps(_role_distribution(roles, "R_frame_tok"), sort_keys=True),
            "global_role_distribution": json.dumps(_role_distribution(roles, "R_global_tok"), sort_keys=True),
        })

    pairs: List[Dict[str, object]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            frame_diff = abs(float(left.get("frame_keep_ratio") or 1.0) - float(right.get("frame_keep_ratio") or 1.0))
            global_diff = abs(float(left.get("global_keep_ratio") or 1.0) - float(right.get("global_keep_ratio") or 1.0))
            left_roles = json.loads(str(left.get("frame_role_distribution") or "{}"))
            right_roles = json.loads(str(right.get("frame_role_distribution") or "{}"))
            role_l1 = _dist_l1(left_roles, right_roles)
            gate = frame_diff >= 0.02 or global_diff >= 0.02 or role_l1 >= 0.02
            pairs.append({
                "left": left["candidate"],
                "right": right["candidate"],
                "frame_keep_ratio_diff": frame_diff,
                "global_keep_ratio_diff": global_diff,
                "frame_role_l1": role_l1,
                "action_distinguishable": bool(gate),
            })

    non_base_pairs = [p for p in pairs if p["left"] != "V31_BASE_H9_REFERENCE" and p["right"] != "V31_BASE_H9_REFERENCE"]
    any_non_base_distinct = any(bool(p["action_distinguishable"]) for p in non_base_pairs)
    any_source_skip_effect = any(
        float(r.get("frame_max_skip_tokens") or 0.0) > 0.0 or float(r.get("global_max_skip_tokens") or 0.0) > 0.0
        for r in rows
        if r["candidate"] != "V31_BASE_H9_REFERENCE"
    )
    context_empty_total = sum(
        int(r.get("frame_empty_source_events") or 0) + int(r.get("global_empty_source_events") or 0)
        for r in rows
    )
    summary = {
        "h0c_action_smoke_gate_pass": bool(rows and not missing and any_non_base_distinct and any_source_skip_effect and context_empty_total == 0),
        "rows_done": len(rows),
        "missing_rows": len(missing),
        "any_non_base_pair_distinguishable": bool(any_non_base_distinct),
        "any_source_skip_effect": bool(any_source_skip_effect),
        "context_empty_source_events_total": context_empty_total,
        "gate_rule": "At least one non-base FG pair differs by keep ratio or frame-role L1 >= 0.02; at least one source skip effect; no empty source events.",
        "boundary": f"H0C smoke only checks action/hook distinguishability on chunk{args.chunk} h{args.horizon} before H1; it is not trajectory evidence.",
    }
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "h0c_action_smoke_rows.csv", rows)
    _write_csv(out_dir / "h0c_action_smoke_pairs.csv", pairs)
    _write_csv(out_dir / "h0c_action_smoke_missing.csv", missing)
    _write_json(out_dir / "h0c_action_smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["h0c_action_smoke_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
