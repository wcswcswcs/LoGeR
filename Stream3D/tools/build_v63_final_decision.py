from __future__ import annotations

import argparse

from stream4d_native.v63_full_eval import V63FullEvalConfig, build_v63_final_decision, write_v63_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v63 final decision from phase summaries.")
    parser.add_argument("--output-root", default="outputs/audit/v63_final")
    args = parser.parse_args()
    cfg = V63FullEvalConfig(output_root=args.output_root)
    result = build_v63_final_decision(cfg)
    outputs = write_v63_final_decision(result, args.output_root)
    print(
        {
            "outputs": outputs,
            "decision_label": result["final_decision"]["decision_label"],
            "blocked_claims": result["final_decision"]["blocked_claims"],
            "final_gate": result["final_decision"]["final_gate"],
        }
    )


if __name__ == "__main__":
    main()
