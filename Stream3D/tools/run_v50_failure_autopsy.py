from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_failure_autopsy, write_v50_failure_autopsy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 10 failure autopsy.")
    parser.add_argument("--output-root", default="outputs/audit/v50_failure_autopsy")
    args = parser.parse_args()
    payload = build_v50_failure_autopsy()
    write_v50_failure_autopsy(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/failure_autopsy_summary.json",
            "failure_labels": payload["failure_labels"],
        }
    )


if __name__ == "__main__":
    main()
