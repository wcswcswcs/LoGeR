#!/usr/bin/env python3
"""Summarize and validate the ACL2 v81 long-window cluster bank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_BANK = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase1_long_window_cluster_bank/long_window_cluster_rows.csv"
)
DEFAULT_OUT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase1_long_window_cluster_bank/long_window_cluster_resummary.json"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def nonempty(value: Any) -> bool:
    return value not in (None, "")


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    bad = [row for row in rows if row.get("case_type") == "bad"]
    good = [row for row in rows if row.get("case_type") in {"good", "false_positive"}]
    seq02 = [
        row for row in rows
        if row.get("seq") == "02" and int(row["chunk_start"]) <= 66 and int(row["chunk_end"]) >= 62
    ]
    missing_selected = [
        row.get("window_id") for row in rows
        if not nonempty(row.get("selected_low_support_ratio")) or int(float(row.get("selected_chunk_evidence_count") or 0)) <= 0
    ]
    missing_semantic = [
        row.get("window_id") for row in rows
        if not nonempty(row.get("stable_mass")) or not nonempty(row.get("harm_mass")) or not nonempty(row.get("context_mass"))
    ]
    gate = (
        len(bad) >= 12
        and len(good) >= 12
        and len({row.get("seq") for row in rows}) >= 3
        and bool(seq02)
        and not missing_selected
        and not missing_semantic
    )
    return {
        "schema": "acl2_v81_phase1_long_window_cluster_resummary_v1",
        "gate_pass": gate,
        "row_count": len(rows),
        "bad_long_windows": len(bad),
        "good_or_false_positive_windows": len(good),
        "seqs_covered": sorted({str(row.get("seq")) for row in rows}),
        "case_type_counts": {
            case_type: sum(1 for row in rows if row.get("case_type") == case_type)
            for case_type in ("bad", "good", "false_positive")
        },
        "seq02_62_70_cluster_rows": [row.get("window_id") for row in seq02],
        "missing_selected_rows": missing_selected,
        "missing_semantic_rows": missing_semantic,
        "radio_available_rows": sum(str(row.get("has_radio")).lower() == "true" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = read_rows(args.bank)
    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
