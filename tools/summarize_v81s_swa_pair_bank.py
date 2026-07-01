#!/usr/bin/env python3
"""Summarize v81S SWA adjacent-pair case bank outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BANK_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS2_swa_good_bad_pair_bank"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    args = parser.parse_args()

    rows_path = args.bank_dir / "swa_good_bad_pair_bank.csv"
    summary_path = args.bank_dir / "swa_good_bad_pair_bank_summary.json"
    rows = _read_rows(rows_path)
    summary = _read_json(summary_path)
    case_counts = Counter(row.get("case_type", "") for row in rows)
    seqs = sorted({row.get("seq", "") for row in rows if row.get("seq", "")})
    quality_risk_rows = sum(str(row.get("artifact_quality_risk", "")).lower() == "true" for row in rows)
    out = {
        "schema": "acl2_v81s_swa_pair_bank_resummary_v1",
        "bank_dir": str(args.bank_dir),
        "rows": len(rows),
        "case_counts": dict(case_counts),
        "seq_coverage": seqs,
        "quality_risk_rows": int(quality_risk_rows),
        "gate": summary.get("gate", {}),
    }
    _write_json(args.bank_dir / "swa_pair_bank_resummary.json", out)
    lines = [
        "# v81S SWA Pair Bank Summary",
        "",
        f"rows: {len(rows)}",
        f"case_counts: {dict(case_counts)}",
        f"seq_coverage: {seqs}",
        f"quality_risk_rows: {quality_risk_rows}",
        f"phaseS2_gate_pass: {summary.get('gate', {}).get('phaseS2_gate_pass')}",
        "",
        "Note: empty SWA/READ action fields are retained as missing evidence rather than synthesized values.",
    ]
    (args.bank_dir / "swa_pair_bank_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
