from __future__ import annotations

import argparse

from stream4d_native.v55_chunk_roles import build_v55_chunk_roles, write_v55_chunk_roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v55 Phase 1 chunk role inference.")
    parser.add_argument("--phase0-rows", default="outputs/audit/v55_phase0_fact_lock/v54_failure_decomposition_rows.csv")
    parser.add_argument("--anchor-threshold", type=float, default=0.62)
    parser.add_argument("--output-root", default="outputs/audit/v55_chunk_roles")
    parser.add_argument("--visualization-root", default="outputs/audit/v55_visualizations/chunk_roles")
    args = parser.parse_args()
    payload = build_v55_chunk_roles(phase0_rows_path=args.phase0_rows, anchor_threshold=args.anchor_threshold)
    write_v55_chunk_roles(args.output_root, payload, visualization_root=args.visualization_root)
    summary = payload["summary"]
    print(
        {
            "chunk_roles": f"{args.output_root}/chunk_role_summary.json",
            "gate": summary["gate"],
            "anchor_chunk_count": summary["anchor_chunk_count"],
            "update_chunk_count": summary["update_chunk_count"],
            "role_separation_score": summary["role_separation_score"],
            "role_repairs_applied": summary["role_repairs_applied"],
        }
    )


if __name__ == "__main__":
    main()
