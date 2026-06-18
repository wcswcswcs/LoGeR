from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.local_semantic_residual_matching import run_semantic_residual
from stream4d_native.v37_object_field_adapter import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-summary", default="outputs/audit/v43_2_v37_parity_adapter/adapter_parity_summary.json")
    parser.add_argument("--profiler-summary", default="outputs/audit/v43_2_matching_error_profiler/matching_error_profiler_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_semantic_residual_matching")
    args = parser.parse_args()
    adapter = read_json(ROOT / args.adapter_summary) or {}
    profiler = read_json(ROOT / args.profiler_summary) or {}
    payload = run_semantic_residual(ROOT, adapter_summary=adapter, profiler_summary=profiler)
    out = ROOT / args.output_root
    write_json(out / "semantic_residual_summary.json", payload)
    print(json.dumps({"summary": str(out / "semantic_residual_summary.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
