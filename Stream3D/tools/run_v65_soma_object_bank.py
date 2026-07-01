from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_soma_object_bank import (
    V65SOMAObjectBankConfig,
    build_v65_soma_object_bank,
    write_v65_soma_object_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v65 SOMA method-safe object bank without GT geometry.")
    parser.add_argument("--object-rows", default="outputs/audit/v64r2_native_contract/object_field_rows.csv")
    parser.add_argument("--material-rows", default="outputs/audit/v64r2_native_contract/material_state_rows.csv")
    parser.add_argument(
        "--carrier-rows",
        default="outputs/audit/v53_native_carrier_materialization/objectlet_native_carrier_rows.csv",
    )
    parser.add_argument("--output-root", default="outputs/audit/v65_soma_object_bank")
    parser.add_argument(
        "--allow-unverified-component-join",
        action="store_true",
        help="Debug only: join v64 object/material rows to v53 carrier rows by same text component_id.",
    )
    args = parser.parse_args()
    cfg = V65SOMAObjectBankConfig(
        object_rows_path=args.object_rows,
        material_rows_path=args.material_rows,
        carrier_rows_path=args.carrier_rows,
        output_root=args.output_root,
        allow_unverified_component_join=args.allow_unverified_component_join,
    )
    result = build_v65_soma_object_bank(cfg)
    paths = write_v65_soma_object_bank(result, args.output_root)
    summary = result["summary"]
    print(
        {
            "summary": paths["summary"],
            "object_count": summary["object_count"],
            "material_assignment_count": summary["material_assignment_count"],
            "object_support_row_count": summary["object_support_row_count"],
            "native_carrier_support_row_count": summary["native_carrier_support_row_count"],
            "native_support_join_available": summary["native_support_join_available"],
            "gate": summary["gate"],
            "blockers": summary["blockers"],
        }
    )


if __name__ == "__main__":
    main()
