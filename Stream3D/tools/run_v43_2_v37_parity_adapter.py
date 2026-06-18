from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v37_object_field_adapter import build_v37_adapter_summary, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v37-4d-root", default="outputs/audit/v37_4d_if_allowed_i4_sparse")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_v37_parity_adapter")
    args = parser.parse_args()
    payload = build_v37_adapter_summary(ROOT, v37_4d_root=args.v37_4d_root)
    out = ROOT / args.output_root
    write_json(out / "adapter_parity_summary.json", payload)
    write_csv(out / "adapter_scene_rows.csv", payload.get("scene_rows", []))
    write_csv(out / "adapter_all_summary_rows.csv", payload.get("all_summary_rows", []))
    print(json.dumps({"summary": str(out / "adapter_parity_summary.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
