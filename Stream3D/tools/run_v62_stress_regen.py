from __future__ import annotations

import argparse

from stream4d_native.v62_stress_regen import V62StressRegenConfig, build_v62_stress_regen, write_v62_stress_regen, write_v62_stress_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 6 lightweight graph-regeneration stress.")
    parser.add_argument("--output-root", default="outputs/audit/v62_stress_regen")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/stress_regen")
    args = parser.parse_args()
    cfg = V62StressRegenConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v62_stress_regen(cfg)
    outputs = write_v62_stress_regen(result, args.output_root)
    visuals = write_v62_stress_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"]})


if __name__ == "__main__":
    main()

