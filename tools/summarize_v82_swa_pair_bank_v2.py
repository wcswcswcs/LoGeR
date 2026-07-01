#!/usr/bin/env python3
"""Print the v82 SWA pair-bank v2 summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SUMMARY = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase2_swa_pair_bank_v2/swa_pair_bank_v2_summary.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
