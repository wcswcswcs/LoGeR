#!/usr/bin/env python3
"""Print ACL2 v83 Phase2 clue sufficiency summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SUMMARY = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase2_clue_sufficiency/clue_sufficiency_summary.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
