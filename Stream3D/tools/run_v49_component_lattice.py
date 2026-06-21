from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_component_lattice, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 2 component lattice audit.")
    parser.add_argument("--output-root", default="outputs/audit/v49_component_lattice")
    args = parser.parse_args()
    payload = build_component_lattice()
    write_bundle(
        args.output_root,
        "component_lattice_summary",
        payload,
        {
            "component_lattice_scale_rows": payload["scale_rows"],
            "component_lattice_containment_rows": payload["containment_rows"],
            "top_fragmented_gt_objects": payload["top_fragmented_gt_objects"],
        },
    )
    print({"summary": f"{args.output_root}/component_lattice_summary.json", "gate": payload["gate"]})


if __name__ == "__main__":
    main()
