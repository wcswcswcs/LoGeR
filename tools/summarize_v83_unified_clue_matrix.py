#!/usr/bin/env python3
"""Print ACL2 v83 Phase1 unified clue matrix summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SUMMARY = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase1_unified_clue_matrix/clue_matrix_summary.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
