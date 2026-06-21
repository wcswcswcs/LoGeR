from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_component_completion_atlas, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 1 component completion atlas.")
    parser.add_argument("--output-root", default="outputs/audit/v49_component_completion_atlas")
    parser.add_argument("--pair-limit-per-scene", type=int, default=120)
    parser.add_argument("--set-limit", type=int, default=5000)
    args = parser.parse_args()
    payload = build_component_completion_atlas(pair_limit_per_scene=args.pair_limit_per_scene, set_limit=args.set_limit)
    write_bundle(
        args.output_root,
        "component_completion_atlas_summary",
        payload,
        {
            "component_pair_error_rows": payload["pair_rows"],
            "component_set_candidate_rows": payload["component_set_rows"],
        },
    )
    print({"summary": f"{args.output_root}/component_completion_atlas_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
