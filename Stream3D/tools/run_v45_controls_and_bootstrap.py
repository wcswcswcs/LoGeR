from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.stage1_scale_aware_typed_assembly import read_json, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v45 controls/bootstrap status.")
    parser.add_argument("--stage1-summary", default="outputs/audit/v45_stage1_full/stage1_full_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v45_controls")
    args = parser.parse_args()
    stage1 = read_json(ROOT / args.stage1_summary) or {}
    diag = stage1.get("typed_energy_diagnostic") or {}
    controls = diag.get("controls") or {}
    payload = {
        "phase": "v45_controls_and_bootstrap",
        "created_at": utc_now(),
        "status": "blocked_stage1_not_method" if not stage1.get("stage1_run_as_method") else "ready",
        "controls": controls,
        "bootstrap_delta_ARI_lower95": None,
        "bootstrap_delta_completeness_lower95": None,
        "bootstrap_status": "blocked_until_full_probe5_stage1_method_run" if not stage1.get("stage1_run_as_method") else "not_run",
        "gate": {"pass": bool(stage1.get("stage1_run_as_method") and diag.get("gate", {}).get("pass"))},
    }
    out = ROOT / args.output_root
    write_json(out / "controls_and_bootstrap.json", payload)
    print(json.dumps({"summary": str(out / "controls_and_bootstrap.json"), "status": payload["status"], "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

