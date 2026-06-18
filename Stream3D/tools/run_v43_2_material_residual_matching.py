from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.material_residual_matching import run_material_residual
from stream4d_native.v37_object_field_adapter import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-summary", default="outputs/audit/v43_2_semantic_residual_matching/semantic_residual_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_material_residual_matching")
    args = parser.parse_args()
    semantic = read_json(ROOT / args.semantic_summary) or {}
    payload = run_material_residual(ROOT, semantic_summary=semantic)
    out = ROOT / args.output_root
    write_json(out / "material_residual_summary.json", payload)
    print(json.dumps({"summary": str(out / "material_residual_summary.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
