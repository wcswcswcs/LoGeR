from __future__ import annotations

import argparse

from stream4d_native.v64r2_active_query_optional import (
    build_v64r2_active_query_optional,
    write_v64r2_active_query_optional,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Track D optional active-query decision.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_active_query_optional")
    args = parser.parse_args()
    payload = build_v64r2_active_query_optional()
    write_v64r2_active_query_optional(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/optional_query_summary.json",
            "active_query_status": payload["active_query_status"],
            "blocks_scannet_ap": payload["blocks_scannet_ap"],
            "blocks_dynamic": payload["blocks_dynamic"],
        }
    )


if __name__ == "__main__":
    main()
