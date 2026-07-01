#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stream4d_native.v65_geometry_contract import (
    GEOM_ROOT,
    build_v65_geometry_contract,
    write_v65_geometry_contract,
)


def main() -> None:
    payload = build_v65_geometry_contract()
    write_v65_geometry_contract(GEOM_ROOT, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{GEOM_ROOT}/geometry_contract_summary.json",
            "gate": summary["gate"],
            "geometry_status": summary["geometry_status"],
            "row_count": summary["geometry_metric_row_count"],
        }
    )


if __name__ == "__main__":
    main()
