#!/usr/bin/env python3
"""Summarize ACL2 v80 Phase1 good/bad case-bank outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CASE_BANK_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank"
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(row.get("case_type", "") for row in rows)
    good_seqs = sorted({row.get("seq", "") for row in rows if row.get("case_type") == "good"})
    bad_seqs = sorted({row.get("seq", "") for row in rows if row.get("case_type") == "bad"})
    missing = Counter(field for row in rows for field in str(row.get("missing_fields", "")).split(";") if field)
    semantic_rows = sum(str(row.get("semantic_available", "")).lower() == "true" for row in rows)
    return {
        "rows": len(rows),
        "good": by_type.get("good", 0),
        "bad": by_type.get("bad", 0),
        "good_seqs": good_seqs,
        "bad_seqs": bad_seqs,
        "semantic_available_rows": semantic_rows,
        "missing_field_counts": dict(missing),
        "gate_pass": by_type.get("good", 0) >= 12
        and by_type.get("bad", 0) >= 12
        and len(good_seqs) >= 3
        and len(bad_seqs) >= 3
        and semantic_rows == len(rows),
    }


def write_balance_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v80 Phase1 Good/Bad Case Balance",
        "",
        "| memory | rows | good | bad | good_seqs | bad_seqs | semantic_rows | gate_pass |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for memory in ("short", "mid", "long"):
        item = summary["memory_body_summary"][memory]
        lines.append(
            "| {memory} | {rows} | {good} | {bad} | {good_seqs} | {bad_seqs} | {semantic_rows} | {gate} |".format(
                memory=memory,
                rows=item["rows"],
                good=item["good"],
                bad=item["bad"],
                good_seqs=",".join(item["good_seqs"]),
                bad_seqs=",".join(item["bad_seqs"]),
                semantic_rows=item["semantic_available_rows"],
                gate=item["gate_pass"],
            )
        )
    lines += ["", f"Phase1 gate pass: `{summary['phase1_gate_pass']}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-bank-dir", type=Path, default=DEFAULT_CASE_BANK_DIR)
    args = parser.parse_args()

    files = {
        "short": args.case_bank_dir / "short_single_chunk_cases.csv",
        "mid": args.case_bank_dir / "mid_adjacent_pair_cases.csv",
        "long": args.case_bank_dir / "long_five_chunk_cases.csv",
    }
    memory_summary = {memory: summarize_rows(read_rows(path)) for memory, path in files.items()}
    summary = {
        "schema": "acl2_v80tf_phase1_case_bank_summary_v1",
        "case_bank_dir": str(args.case_bank_dir),
        "memory_body_summary": memory_summary,
        "phase1_gate_pass": all(item["gate_pass"] for item in memory_summary.values()),
    }
    write_json(args.case_bank_dir / "case_bank_summary_resummarized.json", summary)
    write_balance_md(args.case_bank_dir / "good_bad_case_balance.md", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
