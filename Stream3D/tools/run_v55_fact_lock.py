from __future__ import annotations

import argparse

from stream4d_native.v55_fact_lock import build_v55_fact_lock, write_v55_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v55 Phase 0 fact lock and v54 failure decomposition.")
    parser.add_argument("--output-root", default="outputs/audit/v55_phase0_fact_lock")
    parser.add_argument("--visualization-root", default="outputs/audit/v55_visualizations/phase0")
    args = parser.parse_args()
    payload = build_v55_fact_lock()
    write_v55_fact_lock(args.output_root, payload, visualization_root=args.visualization_root)
    summary = payload["summary"]
    print(
        {
            "fact_lock": f"{args.output_root}/fact_lock.json",
            "gate": summary["gate"],
            "v54_final_label": summary["v54_final_label"],
            "v54_chunks_per_scene_mean": summary["v54_chunks_per_scene_mean"],
            "v54_reprojection_success_rate": summary["v54_reprojection_success_rate"],
        }
    )


if __name__ == "__main__":
    main()
