from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import project_path, build_failure_autopsy, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 10 failure autopsy.")
    parser.add_argument("--output-root", default="outputs/audit/v49_failure_autopsy")
    args = parser.parse_args()
    payload = build_failure_autopsy()
    write_bundle(args.output_root, "failure_autopsy_summary", payload, {"failure_layers": payload["failure_layers"]})
    out = project_path(args.output_root)
    (out / "failure_summary.md").write_text(payload["failure_summary_md"] + "\n", encoding="utf-8")
    print({"summary": f"{args.output_root}/failure_autopsy_summary.json", "failure_label": payload["final_failure_label"]})


if __name__ == "__main__":
    main()
