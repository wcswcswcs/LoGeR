from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.matching_error_profiler import build_error_profile
from stream4d_native.v37_object_field_adapter import read_json, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v37-4d-root", default="outputs/audit/v43_2_v37_parity_adapter/v37_4d_rerun_with_counts")
    parser.add_argument("--adapter-summary", default="outputs/audit/v43_2_v37_parity_adapter/adapter_parity_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_matching_error_profiler")
    args = parser.parse_args()
    adapter = read_json(ROOT / args.adapter_summary) or {}
    payload = build_error_profile(ROOT, v37_4d_root=args.v37_4d_root, adapter_summary=adapter)
    out = ROOT / args.output_root
    write_json(out / "matching_error_profiler_summary.json", payload)
    write_csv(out / "diagnostic_loss_rows.csv", payload.get("diagnostic_loss_rows", []))
    write_csv(out / "hard_scene_root_cause_rows.csv", payload.get("hard_scene_root_cause_rows", []))
    print(json.dumps({"summary": str(out / "matching_error_profiler_summary.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
