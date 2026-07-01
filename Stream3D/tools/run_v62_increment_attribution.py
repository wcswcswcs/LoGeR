from __future__ import annotations

import argparse

from stream4d_native.v62_increment_attribution import (
    V62IncrementAttributionConfig,
    build_v62_increment_attribution,
    write_v62_increment_attribution,
    write_v62_increment_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 2 increment attribution.")
    parser.add_argument("--output-root", default="outputs/audit/v62_increment_attribution")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/increment")
    args = parser.parse_args()
    cfg = V62IncrementAttributionConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v62_increment_attribution(cfg)
    outputs = write_v62_increment_attribution(result, args.output_root)
    visuals = write_v62_increment_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"]})


if __name__ == "__main__":
    main()

