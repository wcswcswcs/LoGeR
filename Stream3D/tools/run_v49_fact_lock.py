from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_fact_lock, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 0 fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v49_fact_lock")
    args = parser.parse_args()
    payload = build_fact_lock()
    write_bundle(args.output_root, "fact_lock", payload, {"fact_lock_rows": payload["fact_rows"]})
    print({"summary": f"{args.output_root}/fact_lock.json", "gate": payload["gate"]})


if __name__ == "__main__":
    main()
