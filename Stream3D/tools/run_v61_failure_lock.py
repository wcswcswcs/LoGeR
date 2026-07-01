from __future__ import annotations

import argparse

from stream4d_native.v61_failure_lock import (
    build_v61_failure_lock,
    write_v61_failure_lock,
    write_v61_phase0_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v61 Phase0 v60 failure/unit-mismatch lock.")
    parser.add_argument("--output-root", default="outputs/audit/v61_phase0_failure_lock")
    parser.add_argument("--visualization-root", default="outputs/audit/v61_visualizations/phase0")
    args = parser.parse_args()
    result = build_v61_failure_lock()
    outputs = write_v61_failure_lock(result, args.output_root)
    visuals = write_v61_phase0_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "failure_lock": outputs["failure_lock"],
            "unit_mismatch_rows": outputs["unit_mismatch_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "material_node_count": summary["v60_material_node_count"],
            "observation_node_count": summary["v60_observation_node_count"],
            "material_state_coverage_rate": summary["material_state_coverage_rate"],
            "observation_state_coverage_rate": summary["observation_state_coverage_rate"],
        }
    )


if __name__ == "__main__":
    main()
