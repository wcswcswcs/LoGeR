from __future__ import annotations

import argparse

from stream4d_native.v51_semantic_reliability import build_v51_semantic_reliability, write_v51_semantic_reliability


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v51-r2 semantic reliability guard for selected key masks.")
    parser.add_argument("--keymask-root", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_semantic_reliability")
    parser.add_argument("--mask-observation-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument("--vote-rows-path", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--contradiction-threshold", type=float, default=0.80)
    args = parser.parse_args()
    payload = build_v51_semantic_reliability(
        keymask_root=args.keymask_root,
        mask_observation_table=args.mask_observation_table,
        vote_rows_path=args.vote_rows_path,
        contradiction_threshold=args.contradiction_threshold,
    )
    write_v51_semantic_reliability(args.output_root, payload)
    print({"summary": f"{args.output_root}/semantic_reliability_summary.json", "gate": payload["gate"], "metrics": payload["summary"]})


if __name__ == "__main__":
    main()
