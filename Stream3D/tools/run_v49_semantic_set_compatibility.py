from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_semantic_set_compatibility, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 4 semantic set compatibility audit.")
    parser.add_argument("--output-root", default="outputs/audit/v49_semantic_set")
    args = parser.parse_args()
    payload = build_semantic_set_compatibility()
    write_bundle(args.output_root, "semantic_set_compatibility_summary", payload, {"semantic_backend_rows": payload["backend_rows"]})
    print({"summary": f"{args.output_root}/semantic_set_compatibility_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
