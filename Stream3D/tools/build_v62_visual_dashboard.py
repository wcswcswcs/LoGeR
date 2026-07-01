from __future__ import annotations

import argparse

from stream4d_native.v62_visualization import V62VisualizationConfig, build_v62_visual_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v62 visual dashboard HTML.")
    parser.add_argument("--final-decision-path", default="outputs/audit/v62_final/final_decision.json")
    parser.add_argument("--output-path", default="outputs/audit/v62_visualizations/v62_dashboard.html")
    args = parser.parse_args()
    result = build_v62_visual_dashboard(V62VisualizationConfig(final_decision_path=args.final_decision_path, output_path=args.output_path))
    print(result)


if __name__ == "__main__":
    main()

