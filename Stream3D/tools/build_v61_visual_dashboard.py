from __future__ import annotations

import argparse

from stream4d_native.v61_visualization import V61DashboardConfig, build_v61_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v61 HTML dashboard.")
    parser.add_argument("--final-decision", default="outputs/audit/v61_final_decision/final_decision.json")
    parser.add_argument("--output", default="outputs/audit/v61_visualizations/v61_dashboard.html")
    args = parser.parse_args()
    outputs = build_v61_dashboard(V61DashboardConfig(final_decision_path=args.final_decision, output_path=args.output))
    print(outputs)


if __name__ == "__main__":
    main()
