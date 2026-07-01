from __future__ import annotations

import argparse

from stream4d_native.v59_fact_lock import build_v59_fact_lock, write_v59_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v59 Phase0 SOMA-Manifold fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v59_phase0_fact_lock")
    parser.add_argument("--visualization-root", default="outputs/audit/v59_visualizations/phase0")
    args = parser.parse_args()

    payload = build_v59_fact_lock()
    outputs = write_v59_fact_lock(args.output_root, payload, args.visualization_root)
    fact = payload["fact_lock"]
    print(
        {
            "fact_lock": outputs["fact_lock"],
            "metric_rows": outputs["metric_rows"],
            "failure_chain_rows": outputs["failure_chain_rows"],
            "visualization_status": outputs["visualization_status"],
            "gate": fact["gate"],
            "v58_phase1_dino_recall@3": fact["v58_phase1_dino_recall@3"],
            "v58_phase2_actionable_count": fact["v58_phase2_actionable_count"],
            "v58_phase2_deferred_count": fact["v58_phase2_deferred_count"],
            "v58_phase3_Q6_valid_rate": fact["v58_phase3_Q6_valid_rate"],
            "v58_phase3_expanded_Q6_valid_rate": fact["v58_phase3_expanded_Q6_valid_rate"],
            "expanded_candidate_quality_drop_observed": fact["expanded_candidate_quality_drop_observed"],
        }
    )


if __name__ == "__main__":
    main()
