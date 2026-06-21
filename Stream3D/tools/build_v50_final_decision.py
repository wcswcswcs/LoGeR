from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_final_decision, write_v50_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v50 final decision bundle.")
    parser.add_argument("--output-root", default="outputs/audit/v50_final_decision")
    args = parser.parse_args()
    payload = build_v50_final_decision()
    write_v50_final_decision(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/v50_final_decision.json",
            "final_label": payload["final_label"],
            "no_go_labels": payload["no_go_labels"],
        }
    )


if __name__ == "__main__":
    main()
