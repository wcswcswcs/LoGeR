#!/usr/bin/env python3
"""Print a compact summary for ACL2 v84 Phase2 Memory Ruler ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SUMMARY = Path("results/acl2_v84tf_memory_ruler_audit/phase2_memory_ruler_ledger/phase2_ledger_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    keys = [
        "phase2_gate_pass",
        "rows",
        "bad_rows",
        "good_or_false_positive_rows",
        "sequence_coverage",
        "score_available_high_quality_ratio",
        "positive_support_gate_pass",
        "ruler_anchor_pair_count",
        "ruler_anchor_token_count",
    ]
    print(json.dumps({key: data.get(key) for key in keys}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

