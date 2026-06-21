from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_shared_observation, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 7 shared observation audit.")
    parser.add_argument("--output-root", default="outputs/audit/v49_shared_observation")
    args = parser.parse_args()
    payload = build_shared_observation()
    write_bundle(args.output_root, "shared_observation_summary", payload)
    print({"summary": f"{args.output_root}/shared_observation_summary.json", "gate": payload["gate"]})


if __name__ == "__main__":
    main()
