from __future__ import annotations

import argparse

from stream4d_native.v60_fact_lock import (
    build_v60_fact_lock,
    write_v60_fact_lock,
    write_v60_phase0_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v60 Phase0 v59 artifact integrity and calibrated same-category fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v60_phase0_fact_lock")
    parser.add_argument("--visualization-root", default="outputs/audit/v60_visualizations/phase0")
    args = parser.parse_args()
    result = build_v60_fact_lock()
    outputs = write_v60_fact_lock(result, args.output_root)
    visuals = write_v60_phase0_visualizations(result, args.visualization_root)
    fact = result["fact_lock"]
    print(
        {
            "fact_lock": outputs["fact_lock"],
            "same_category_gate_rows": outputs["same_category_gate_rows"],
            "artifact_integrity_tree": outputs["artifact_integrity_tree"],
            "visualization_status": visuals["visualization_status"],
            "gate": fact["gate"],
            "v59_final_label": fact["v59_final_label"],
            "phase2_same_category_gate_calibrated_pass": fact["phase2_same_category_gate_calibrated_pass"],
            "method_false_count": fact["phase2_same_category_method_false_count"],
            "method_pair_count": fact["phase2_same_category_method_pair_count"],
            "method_wilson_upper95": fact["phase2_same_category_calibrated"]["method_wilson_upper95"],
        }
    )


if __name__ == "__main__":
    main()
