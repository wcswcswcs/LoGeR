from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_key_mask_selection, write_v50_key_mask_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 3 key-mask selection audit.")
    parser.add_argument("--output-root", default="outputs/audit/v50_key_masks")
    args = parser.parse_args()
    payload = build_v50_key_mask_selection()
    write_v50_key_mask_selection(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/key_mask_summary.json",
            "gate": payload["gate"],
            "selected_variant": payload["selected_variant"],
            "key_mask_count": payload["summary"]["key_mask_count"],
            "component_coverage": payload["summary"]["component_coverage"],
        }
    )


if __name__ == "__main__":
    main()
