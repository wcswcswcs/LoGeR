from __future__ import annotations

import argparse

from stream4d_native.v62_decircularization import (
    V62DecircularizationConfig,
    build_v62_decircularization,
    write_v62_decircularization,
    write_v62_decircularization_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 1 de-circularization audit.")
    parser.add_argument("--output-root", default="outputs/audit/v62_decircularization")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/decircularization")
    args = parser.parse_args()
    cfg = V62DecircularizationConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v62_decircularization(cfg)
    outputs = write_v62_decircularization(result, args.output_root)
    visuals = write_v62_decircularization_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"]})


if __name__ == "__main__":
    main()

