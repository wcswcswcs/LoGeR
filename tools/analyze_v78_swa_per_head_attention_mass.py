#!/usr/bin/env python3
"""Export per-head SWA overlap-bias attention-mass diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=run_dir containing hook_effect_summary.jsonl")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--control", default="")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-comparison-csv", type=Path, default=None)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    name, path = raw.split("=", 1)
    return name.strip(), Path(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_run(name: str, run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for rec_idx, rec in enumerate(_read_jsonl(run_dir / "hook_effect_summary.jsonl")):
        swa = ((rec.get("hook_effect_summary") or {}).get("swa_read") or {})
        selected_lift = swa.get("swa_overlap_attention_mass_selected_lift_by_head")
        source_lift = swa.get("swa_overlap_attention_mass_source_lift_by_head")
        if not isinstance(selected_lift, list) or not isinstance(source_lift, list):
            continue
        selected_before = swa.get("swa_overlap_attention_mass_selected_before_by_head") or []
        selected_after = swa.get("swa_overlap_attention_mass_selected_after_by_head") or []
        source_before = swa.get("swa_overlap_attention_mass_source_before_by_head") or []
        source_after = swa.get("swa_overlap_attention_mass_source_after_by_head") or []
        head_count = min(len(selected_lift), len(source_lift))
        record = {
            "run": name,
            "run_dir": str(run_dir),
            "record_idx": int(rec_idx),
            "trace_chunk_idx": rec.get("chunk_idx"),
            "trace_start_frame": rec.get("start_frame"),
            "trace_end_frame": rec.get("end_frame"),
            "attention_mass_available": bool(swa.get("attention_mass_available", False)),
            "selected_top_head_by_lift": swa.get("swa_overlap_attention_mass_selected_top_head_by_lift"),
            "selected_top_head_lift": swa.get("swa_overlap_attention_mass_selected_top_head_lift"),
            "source_top_head_by_lift": swa.get("swa_overlap_attention_mass_source_top_head_by_lift"),
            "source_top_head_lift": swa.get("swa_overlap_attention_mass_source_top_head_lift"),
            "head_count": int(head_count),
        }
        record_rows.append(record)
        for head in range(head_count):
            out_rows.append({
                **record,
                "head": int(head),
                "selected_before": _finite(selected_before[head]) if head < len(selected_before) else None,
                "selected_after": _finite(selected_after[head]) if head < len(selected_after) else None,
                "selected_lift": _finite(selected_lift[head]),
                "source_before": _finite(source_before[head]) if head < len(source_before) else None,
                "source_after": _finite(source_after[head]) if head < len(source_after) else None,
                "source_lift": _finite(source_lift[head]),
            })
    return out_rows, record_rows


def _comparison_rows(rows: list[dict[str, Any]], candidate: str, control: str) -> list[dict[str, Any]]:
    cand = [row for row in rows if row.get("run") == candidate]
    ctrl = [row for row in rows if row.get("run") == control]
    ctrl_by_key = {
        (int(row.get("record_idx", -1)), int(row.get("head", -1))): row
        for row in ctrl
    }
    out: list[dict[str, Any]] = []
    for row in cand:
        key = (int(row.get("record_idx", -1)), int(row.get("head", -1)))
        other = ctrl_by_key.get(key)
        if not other:
            continue
        sel = _finite(row.get("selected_lift"))
        sel_c = _finite(other.get("selected_lift"))
        src = _finite(row.get("source_lift"))
        src_c = _finite(other.get("source_lift"))
        out.append({
            "record_idx": key[0],
            "head": key[1],
            "candidate": candidate,
            "control": control,
            "candidate_selected_lift": sel,
            "control_selected_lift": sel_c,
            "candidate_minus_control_selected_lift": sel - sel_c if sel is not None and sel_c is not None else None,
            "candidate_source_lift": src,
            "control_source_lift": src_c,
            "candidate_minus_control_source_lift": src - src_c if src is not None and src_c is not None else None,
        })
    return sorted(
        out,
        key=lambda row: _finite(row.get("candidate_minus_control_selected_lift")) or -math.inf,
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for raw in args.run:
        name, run_dir = _parse_run(raw)
        rows, records = _extract_run(name, run_dir)
        all_rows.extend(rows)
        record_rows.extend(records)
    comparison = _comparison_rows(all_rows, args.candidate, args.control) if args.candidate and args.control else []
    comp_csv = args.out_comparison_csv or args.out_csv.with_name(args.out_csv.stem + "_comparison.csv")
    _write_csv(args.out_csv, all_rows)
    if comparison:
        _write_csv(comp_csv, comparison)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "schema": "v78_swa_per_head_attention_mass_v1",
                "runs": [raw for raw in args.run],
                "records": record_rows,
                "rows": len(all_rows),
                "comparison_rows": len(comparison),
                "comparison_csv": str(comp_csv) if comparison else "",
                "top_candidate_minus_control_selected_lift": comparison[:5],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote_csv={args.out_csv} rows={len(all_rows)}")
    if comparison:
        print(f"wrote_csv={comp_csv} rows={len(comparison)}")
    print(f"wrote_json={args.out_json}")


if __name__ == "__main__":
    main()
