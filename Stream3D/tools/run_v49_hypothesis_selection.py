from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_hypothesis_selection, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 6 hypothesis selection solver.")
    parser.add_argument("--output-root", default="outputs/audit/v49_hypothesis_selection")
    parser.add_argument("--max-per-scene", type=int, default=150)
    args = parser.parse_args()
    payload = build_hypothesis_selection(max_per_scene=args.max_per_scene)
    write_bundle(
        args.output_root,
        "hypothesis_selection_summary",
        payload,
        {"hypothesis_selection_rows": payload["selection_rows"], "selected_hypothesis_rows": payload["selected_hypothesis_rows"]},
    )
    print({"summary": f"{args.output_root}/hypothesis_selection_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
