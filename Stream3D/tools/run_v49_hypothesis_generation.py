from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_hypothesis_generation, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 3 object hypothesis generation.")
    parser.add_argument("--output-root", default="outputs/audit/v49_hypothesis_generation")
    parser.add_argument("--max-hypotheses", type=int, default=5000)
    args = parser.parse_args()
    payload = build_hypothesis_generation(max_hypotheses=args.max_hypotheses)
    write_bundle(args.output_root, "hypothesis_generation_summary", payload, {"hypothesis_rows": payload["hypothesis_rows"]})
    print({"summary": f"{args.output_root}/hypothesis_generation_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
