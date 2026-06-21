from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import build_fact_lock, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v44 fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v44_fact_lock")
    args = parser.parse_args()
    payload = build_fact_lock(ROOT)
    out = ROOT / args.output_root
    write_json(out / "fact_lock.json", payload)
    write_csv(out / "fact_lock_rows.csv", payload["rows"])
    print(json.dumps({"fact_lock": str(out / "fact_lock.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
