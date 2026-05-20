#!/usr/bin/env python3
"""Tiny numeric helper for v16 shell launchers."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "double":
        raise SystemExit("Usage: v16_candidate_value.py double VALUE")
    print(f"{float(sys.argv[2]) * 2.0:.8g}")


if __name__ == "__main__":
    main()
