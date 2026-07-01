from __future__ import annotations

import argparse

from stream4d_native.v61_stress_eval import (
    V61StressEvalConfig,
    build_v61_stress_eval,
    write_v61_stress_eval,
    write_v61_stress_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v61 Phase5 stress diagnostic proxy.")
    parser.add_argument("--output-root", default="outputs/audit/v61_stress")
    parser.add_argument("--visualization-root", default="outputs/audit/v61_visualizations/stress")
    args = parser.parse_args()
    cfg = V61StressEvalConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v61_stress_eval(cfg)
    outputs = write_v61_stress_eval(result, args.output_root)
    visuals = write_v61_stress_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "stress_summary": outputs["stress_summary"],
            "stress_metric_rows": outputs["stress_metric_rows"],
            "stress_setting_rows": outputs["stress_setting_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "stress_real_minus_mask_only_ARI_pass_count": summary["stress_real_minus_mask_only_ARI_pass_count"],
            "stress_real_minus_v56_expanded_ARI_pass_count": summary["stress_real_minus_v56_expanded_ARI_pass_count"],
            "reactivation_precision_diagnostic": summary["reactivation_precision_diagnostic"],
        }
    )


if __name__ == "__main__":
    main()
