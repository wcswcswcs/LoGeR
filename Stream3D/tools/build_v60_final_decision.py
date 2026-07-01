from __future__ import annotations

import argparse

from stream4d_native.v60_final_decision import build_v60_final_decision, write_v60_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v60 final decision artifacts.")
    parser.add_argument("--output-root", default="outputs/audit/v60_final_decision")
    args = parser.parse_args()
    result = build_v60_final_decision()
    outputs = write_v60_final_decision(result, args.output_root)
    decision = result["decision"]
    print(
        {
            "final_decision": outputs["final_decision"],
            "final_eval_rows": outputs["final_eval_rows"],
            "final_label": decision["final_label"],
            "partial_label": decision["partial_label"],
            "goal_achieved": decision["goal_achieved"],
            "first_hard_blocker": decision["first_hard_blocker"],
        }
    )


if __name__ == "__main__":
    main()
