from __future__ import annotations

import argparse

from stream4d_native.v62_refinement_robustness import (
    V62RefinementRobustnessConfig,
    build_v62_refinement_robustness,
    write_v62_refinement_robustness,
    write_v62_refinement_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 4 refinement robustness.")
    parser.add_argument("--output-root", default="outputs/audit/v62_refinement_robustness")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/refinement_robustness")
    args = parser.parse_args()
    cfg = V62RefinementRobustnessConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v62_refinement_robustness(cfg)
    outputs = write_v62_refinement_robustness(result, args.output_root)
    visuals = write_v62_refinement_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"]})


if __name__ == "__main__":
    main()

