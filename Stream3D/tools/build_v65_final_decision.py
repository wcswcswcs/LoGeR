#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_final_eval import FINAL_ROOT, build_v65_final_decision, write_v65_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v65 final decision artifacts.")
    parser.add_argument("--output-root", default=FINAL_ROOT)
    args = parser.parse_args()
    payload = build_v65_final_decision()
    write_v65_final_decision(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/final_decision.json",
            "decision_labels": summary["decision_labels"],
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
