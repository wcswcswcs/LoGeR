from __future__ import annotations

import argparse

from stream4d_native.v62_active_query_refresh import (
    V62ActiveQueryRefreshConfig,
    build_v62_active_query_refresh,
    write_v62_active_query_refresh,
    write_v62_query_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 5 active query candidate refresh.")
    parser.add_argument("--output-root", default="outputs/audit/v62_active_query_refresh")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/query_refresh")
    args = parser.parse_args()
    cfg = V62ActiveQueryRefreshConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v62_active_query_refresh(cfg)
    outputs = write_v62_active_query_refresh(result, args.output_root)
    visuals = write_v62_query_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"], "claim_status": result["summary"]["claim_status"]})


if __name__ == "__main__":
    main()

