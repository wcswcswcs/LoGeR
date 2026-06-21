from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import ap_bridge_diagnostic, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Report v44 AP bridge diagnostic boundary.")
    parser.add_argument("--output-root", default="outputs/audit/v44_ap_bridge")
    args = parser.parse_args()
    payload = ap_bridge_diagnostic(ROOT)
    out = ROOT / args.output_root
    write_json(out / "ap_bridge_diagnostic.json", payload)
    write_csv(out / "ap_bridge_rows.csv", payload["rows"])
    print(json.dumps({"summary": str(out / "ap_bridge_diagnostic.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
