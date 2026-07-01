#!/usr/bin/env python3
"""Audit v81 READ/SWA confirmation quality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROWS = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase4_read_swa_confirmation/read_swa_confirmation_rows.csv"
)
DEFAULT_OUT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase4_read_swa_confirmation/confirmation_quality_summary.json"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key)
        return None if value in (None, "") else float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = read_rows(args.rows)
    read_vals = [value for row in rows if (value := f(row, "read_confirmed_stable_mass")) is not None]
    swa_vals = [value for row in rows if (value := f(row, "swa_confirmed_stable_mass")) is not None]
    align_vals = [value for row in rows if (value := f(row, "read_swa_alignment")) is not None]
    random_vals = [value for row in rows if (value := f(row, "random_confirmation_alignment")) is not None]
    summary: dict[str, Any] = {
        "schema": "acl2_v81_phase4_read_swa_confirmation_quality_v1",
        "row_count": len(rows),
        "read_confirmed_rows": len(read_vals),
        "swa_confirmed_rows": len(swa_vals),
        "alignment_rows": len(align_vals),
        "read_confirmed_stable_mass_mean": float(np.mean(read_vals)) if read_vals else None,
        "swa_confirmed_stable_mass_mean": float(np.mean(swa_vals)) if swa_vals else None,
        "read_swa_alignment_mean": float(np.mean(align_vals)) if align_vals else None,
        "random_alignment_available": bool(random_vals),
        "random_confirmation_alignment_mean": float(np.mean(random_vals)) if random_vals else None,
        "read_confirmed_stable_mass_nonzero": any(value > 0 for value in read_vals),
        "swa_confirmed_stable_mass_nonzero": any(value > 0 for value in swa_vals),
        "alignment_ge_0_30": bool(align_vals and float(np.mean(align_vals)) >= 0.30),
        "actual_beats_random": bool(random_vals and align_vals and float(np.mean(align_vals)) > float(np.mean(random_vals))),
        "confirmation_scope": "diagnostic_existing_v80_hook_proxy",
    }
    summary["gate_pass"] = bool(
        summary["read_confirmed_stable_mass_nonzero"]
        and summary["swa_confirmed_stable_mass_nonzero"]
        and summary["alignment_ge_0_30"]
        and summary["actual_beats_random"]
        and summary["confirmation_scope"] == "v81_action_ready"
    )
    summary["decision"] = "usable_for_ttt_action" if summary["gate_pass"] else "diagnostic_only_not_action_ready"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
