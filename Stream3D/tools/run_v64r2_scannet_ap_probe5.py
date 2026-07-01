from __future__ import annotations

import argparse

from stream4d_native.v64r2_scannet_ap_eval import build_v64r2_scannet_ap_probe5, write_v64r2_scannet_ap_probe5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase B1 ScanNet AP smoke on probe5.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_scannet_ap_probe5")
    parser.add_argument("--output-config-prefix", default="v64r2_probe5_v53_bridge")
    parser.add_argument("--export-mask-sample-stride", type=int, default=4)
    parser.add_argument("--export-mask-max-pixels", type=int, default=30000)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    args = parser.parse_args()
    payload = build_v64r2_scannet_ap_probe5(
        output_root=args.output_root,
        output_config_prefix=args.output_config_prefix,
        export_mask_sample_stride=args.export_mask_sample_stride,
        export_mask_max_pixels=args.export_mask_max_pixels,
        export_nn_radius=args.export_nn_radius,
    )
    write_v64r2_scannet_ap_probe5(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/ap_smoke_summary.json",
            "scannet_ap_status": summary["scannet_ap_status"],
            "best_diagnostic_AP": summary["best_diagnostic_AP"],
            "best_diagnostic_AP50": summary["best_diagnostic_AP50"],
            "best_diagnostic_AP25": summary["best_diagnostic_AP25"],
            "method_safe_AP_available": summary["method_safe_AP_available"],
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
