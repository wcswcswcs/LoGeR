from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_hypothesis_generation, write_v50_hypothesis_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 6 whole-object hypothesis generation audit.")
    parser.add_argument(
        "--output-root",
        default="outputs/audit/v50_hypothesis_generation",
        help="Output directory under the Stream3D root.",
    )
    parser.add_argument(
        "--max-hypothesis-rows",
        type=int,
        default=12000,
        help="Maximum deduplicated hypotheses to write for audit.",
    )
    args = parser.parse_args()

    payload = build_v50_hypothesis_generation(max_hypothesis_rows=args.max_hypothesis_rows)
    write_v50_hypothesis_generation(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/hypothesis_summary.json",
            "gate": payload["gate"],
            "hypothesis_count": payload["summary"]["hypothesis_count"],
            "GT_object_has_hypothesis@0.25": payload["summary"]["GT_object_has_hypothesis@0.25"],
            "GT_object_has_hypothesis@0.50": payload["summary"]["GT_object_has_hypothesis@0.50"],
            "hypothesis_purity@topk": payload["summary"]["hypothesis_purity@topk"],
        }
    )


if __name__ == "__main__":
    main()
