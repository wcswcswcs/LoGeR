from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_ap_diagnostic, write_v50_ap_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 8 AP smoke/diagnostic export.")
    parser.add_argument("--output-root", default="outputs/audit/v50_ap_diagnostic")
    parser.add_argument("--export-mask-sample-stride", type=int, default=4)
    parser.add_argument("--export-mask-max-pixels", type=int, default=30000)
    args = parser.parse_args()

    payload = build_v50_ap_diagnostic(
        export_mask_sample_stride=args.export_mask_sample_stride,
        export_mask_max_pixels=args.export_mask_max_pixels,
    )
    write_v50_ap_diagnostic(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/ap_export_summary.json",
            "gate": payload["gate"],
            "best_AP": payload["summary"]["best_AP"],
            "best_AP50": payload["summary"]["best_AP50"],
            "best_AP25": payload["summary"]["best_AP25"],
            "best_AP_variant": payload["summary"]["best_AP_variant"],
        }
    )


if __name__ == "__main__":
    main()
