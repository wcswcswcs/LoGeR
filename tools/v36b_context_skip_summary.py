#!/usr/bin/env python3
"""Collect v36B context source-skip hook metrics from short rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


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


def _done(run_dir: Path, run_name: str) -> bool:
    status = run_dir / "run_status.txt"
    if not status.exists():
        return False
    return f"DONE {run_name}" in status.read_text(encoding="utf-8", errors="replace")


def _path_row(rows: List[Dict[str, object]], path_name: str) -> Dict[str, object]:
    matching = [r for r in rows if r.get("path") == path_name]
    if not matching:
        return {}
    return matching[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--candidates", default="V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_04")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rollout_root = Path(args.rollout_root)
    chunks = [int(x) for x in str(args.chunks).split(",") if x.strip()]
    candidates = [x.strip() for x in str(args.candidates).split(",") if x.strip()]
    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for chunk in chunks:
        for candidate in candidates:
            run_name = f"{args.run_prefix}_{candidate}_chunk{chunk}_h{args.horizon}_globalgate_H9parent_SWKS3"
            run_dir = rollout_root / run_name
            if not _done(run_dir, run_name):
                missing.append({"chunk": chunk, "candidate": candidate, "run_name": run_name, "reason": "missing_or_not_done"})
                continue
            skip_rows = _read_jsonl(run_dir / "context_skip_summary.jsonl")
            frame = _path_row(skip_rows, "frame_attention")
            global_row = _path_row(skip_rows, "chunk_attention")
            hook_rows = _read_jsonl(run_dir / "hook_effect_summary.jsonl")
            implemented_paths = hook_rows[-1].get("implemented_paths") if hook_rows else []
            rows.append({
                "chunk": chunk,
                "horizon": int(args.horizon),
                "candidate": candidate,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "implemented_paths": ",".join(implemented_paths) if isinstance(implemented_paths, list) else implemented_paths,
                "frame_max_context_source_skip_tokens": frame.get("max_context_source_skip_tokens"),
                "frame_mean_context_source_keep_ratio": frame.get("mean_context_source_keep_ratio"),
                "frame_context_empty_source_events": frame.get("num_context_empty_source_events"),
                "global_max_context_source_skip_tokens": global_row.get("max_context_source_skip_tokens"),
                "global_mean_context_source_keep_ratio": global_row.get("mean_context_source_keep_ratio"),
                "global_context_empty_source_events": global_row.get("num_context_empty_source_events"),
                "frame_attention_mass_available": frame.get("attention_mass_available", False),
                "frame_attention_mass_removed_before": frame.get("mean_attention_mass_removed_before"),
                "frame_attention_mass_removed_after": frame.get("mean_attention_mass_removed_after"),
                "frame_attention_mass_retained_before": frame.get("mean_attention_mass_retained_before"),
                "frame_attention_mass_retained_after": frame.get("mean_attention_mass_retained_after"),
                "global_attention_mass_available": global_row.get("attention_mass_available", False),
                "global_attention_mass_removed_before": global_row.get("mean_attention_mass_removed_before"),
                "global_attention_mass_removed_after": global_row.get("mean_attention_mass_removed_after"),
                "global_attention_mass_retained_before": global_row.get("mean_attention_mass_retained_before"),
                "global_attention_mass_retained_after": global_row.get("mean_attention_mass_retained_after"),
                "attention_mass_removed_available": bool(frame.get("attention_mass_available", False))
                or bool(global_row.get("attention_mass_available", False)),
                "attention_mass_status": (
                    "sampled-qk-softmax-mass"
                    if bool(frame.get("attention_mass_available", False)) or bool(global_row.get("attention_mass_available", False))
                    else "attention-mass-unverified"
                ),
            })

    context_empty_total = sum(
        int(r.get("frame_context_empty_source_events") or 0) + int(r.get("global_context_empty_source_events") or 0)
        for r in rows
    )
    source_effect_rows = [
        r for r in rows
        if float(r.get("frame_max_context_source_skip_tokens") or 0) > 0
        or float(r.get("global_max_context_source_skip_tokens") or 0) > 0
    ]
    attention_mass_rows = [r for r in rows if bool(r.get("attention_mass_removed_available", False))]
    summary = {
        "rows": len(rows),
        "missing_rows": len(missing),
        "context_empty_source_events_total": context_empty_total,
        "source_effect_rows": len(source_effect_rows),
        "attention_mass_removed_available": bool(attention_mass_rows),
        "attention_mass_rows": len(attention_mass_rows),
        "attention_mass_status": "sampled-qk-softmax-mass" if attention_mass_rows else "attention-mass-unverified",
        "all_rows_done": len(rows) == len(chunks) * len(candidates) and not missing,
    }
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "context_skip_rollout_summary.csv", rows)
    _write_csv(out_dir / "context_skip_missing_rows.csv", missing)
    _write_json(out_dir / "context_skip_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["all_rows_done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
