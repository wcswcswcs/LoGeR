from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_final_decision, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v49 final decision.")
    parser.add_argument("--output-root", default="outputs/audit/v49_final_decision")
    args = parser.parse_args()
    payload = build_final_decision()
    write_bundle(args.output_root, "v49_final_decision", payload)
    print({"summary": f"{args.output_root}/v49_final_decision.json", "final_label": payload["final_label"]})


if __name__ == "__main__":
    main()
