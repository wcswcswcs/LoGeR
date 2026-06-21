from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_relation_propagation, write_v50_relation_propagation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 4 relation propagation audit.")
    parser.add_argument(
        "--output-root",
        default="outputs/audit/v50_relation_propagation",
        help="Output directory under the Stream3D root.",
    )
    parser.add_argument(
        "--max-affinity-rows",
        type=int,
        default=20000,
        help="Maximum component affinity rows to write for audit.",
    )
    args = parser.parse_args()

    payload = build_v50_relation_propagation(max_affinity_rows=args.max_affinity_rows)
    write_v50_relation_propagation(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/propagation_summary.json",
            "gate": payload["gate"],
            "component_pair_count": payload["summary"]["component_pair_count"],
            "same_GT_pair_AUC": payload["summary"]["same_GT_pair_AUC"],
            "propagation_real_minus_shuffled_AUC": payload["summary"]["propagation_real_minus_shuffled_AUC"],
            "propagation_real_minus_no_temporal_AUC": payload["summary"]["propagation_real_minus_no_temporal_AUC"],
        }
    )


if __name__ == "__main__":
    main()
