from __future__ import annotations

import argparse

from stream4d_native.v61_native_field import (
    V61NativeFieldConfig,
    build_v61_native_field,
    write_v61_native_field,
    write_v61_native_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v61 Phase6 native semantic-material field export.")
    parser.add_argument("--output-root", default="outputs/audit/v61_native_field")
    parser.add_argument("--visualization-root", default="outputs/audit/v61_visualizations/native_field")
    args = parser.parse_args()
    cfg = V61NativeFieldConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v61_native_field(cfg)
    outputs = write_v61_native_field(result, args.output_root)
    visuals = write_v61_native_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "native_field_summary": outputs["native_field_summary"],
            "native_history_rows": outputs["native_history_rows"],
            "native_carrier_state_rows": outputs["native_carrier_state_rows"],
            "shortcut_ledger_rows": outputs["shortcut_ledger_rows"],
            "ap_metric_rows": outputs["ap_metric_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "history_object_count": summary["history_object_count"],
            "confirmed_carrier_count": summary["confirmed_carrier_count"],
            "method_safe_native_support_available": summary["method_safe_native_support_available"],
        }
    )


if __name__ == "__main__":
    main()
