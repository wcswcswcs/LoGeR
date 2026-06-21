from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_semantic_backend_availability_audit, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 frozen semantic backend availability audit.")
    parser.add_argument("--output-root", default="outputs/audit/v49_semantic_backend_availability")
    args = parser.parse_args()
    payload = build_semantic_backend_availability_audit()
    write_bundle(
        args.output_root,
        "semantic_backend_availability_summary",
        payload,
        {
            "semantic_source_rows": payload["semantic_source_rows"],
            "prediction_input_rows": payload["prediction_input_rows"],
            "v48_backend_rows": payload["v48_backend_rows"],
        },
    )
    print({"summary": f"{args.output_root}/semantic_backend_availability_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
