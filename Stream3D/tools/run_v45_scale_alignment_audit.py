from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.scale_aware_d4rt_scene import read_csv_rows, scale_alignment_guard_audit
from stream4d_native.v44_typed_mask_assembly import write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v45 scale alignment guard audit from v44 scale diagnostics.")
    parser.add_argument("--ratio-rows", default="outputs/audit/v44_chunk_scale_diagnostic_probe5/chunk_scale_canonical_relaxed030_ratio_rows.csv")
    parser.add_argument("--window-rows", default="outputs/audit/v44_chunk_scale_diagnostic_probe5/chunk_scale_canonical_relaxed030_window_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v45_scale_alignment")
    parser.add_argument("--no-block-outside-10pct", action="store_true")
    args = parser.parse_args()
    payload = scale_alignment_guard_audit(
        ratio_rows=read_csv_rows(ROOT / args.ratio_rows),
        window_rows=read_csv_rows(ROOT / args.window_rows),
        block_outside_10pct=not bool(args.no_block_outside_10pct),
    )
    out = ROOT / args.output_root
    write_json(out / "scale_alignment_audit.json", payload)
    write_csv(out / "scale_guard_rows.csv", payload["scale_guard_rows"])
    print(json.dumps({"summary": str(out / "scale_alignment_audit.json"), "gate": payload["gate"], "outside_10pct_scale_pair_count": payload["outside_10pct_scale_pair_count"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

