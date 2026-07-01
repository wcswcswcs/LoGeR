from __future__ import annotations

import argparse

from stream4d_native.v64r2_visualization import (
    build_v64r2_dashboard_html,
    write_v64r2_dashboard,
    write_v64r2_summary_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v64-r2 HTML dashboard.")
    parser.add_argument("--final-decision-path", default="outputs/audit/v64r2_final/final_decision.json")
    parser.add_argument("--output-path", default="outputs/audit/v64r2_visualizations/v64r2_dashboard.html")
    args = parser.parse_args()
    html = build_v64r2_dashboard_html(final_decision_path=args.final_decision_path)
    write_v64r2_dashboard(args.output_path, html)
    pngs = write_v64r2_summary_visualizations()
    print({"dashboard": args.output_path, "pngs": pngs})


if __name__ == "__main__":
    main()
