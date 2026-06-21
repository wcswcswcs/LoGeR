from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_semantic_guard, write_v50_semantic_guard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 5 semantic guard audit.")
    parser.add_argument("--output-root", default="outputs/audit/v50_semantic_guard")
    args = parser.parse_args()

    payload = build_v50_semantic_guard()
    write_v50_semantic_guard(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/semantic_guard_summary.json",
            "gate": payload["gate"],
            "selected_policy": payload["summary"]["selected_policy"],
        }
    )


if __name__ == "__main__":
    main()
