from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_mask_source_audit, write_v50_mask_source_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 1 mask source audit.")
    parser.add_argument("--output-root", default="outputs/audit/v50_mask_source_audit")
    parser.add_argument("--max-overlap-rows", type=int, default=500)
    args = parser.parse_args()
    payload = build_v50_mask_source_audit(max_overlap_rows=args.max_overlap_rows)
    write_v50_mask_source_audit(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/mask_source_summary.json",
            "gate": payload["gate"],
            "mask_count": payload["summary"]["mask_count"],
            "component_lattice_fallback_edge_count": payload["summary"]["component_lattice_fallback_edge_count"],
        }
    )


if __name__ == "__main__":
    main()
