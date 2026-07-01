from __future__ import annotations

import argparse

from stream4d_native.v63_full_eval import build_v63_visual_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v63 final HTML dashboard.")
    parser.add_argument("--final-decision-path", default="outputs/audit/v63_final/final_decision.json")
    parser.add_argument("--output-path", default="outputs/audit/v63_visualizations/v63_dashboard.html")
    args = parser.parse_args()
    print(build_v63_visual_dashboard(args.final_decision_path, args.output_path))


if __name__ == "__main__":
    main()
