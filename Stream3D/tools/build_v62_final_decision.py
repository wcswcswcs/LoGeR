from __future__ import annotations

import argparse

from stream4d_native.v62_full_eval import V62FullEvalConfig, build_v62_final_decision, write_v62_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v62 final decision table.")
    parser.add_argument("--output-root", default="outputs/audit/v62_final")
    args = parser.parse_args()
    cfg = V62FullEvalConfig(output_root=args.output_root)
    result = build_v62_final_decision(cfg)
    outputs = write_v62_final_decision(result, args.output_root)
    print({"outputs": outputs, "decision_label": result["final_decision"]["decision_label"], "blocked_claims": result["final_decision"]["blocked_claims"]})


if __name__ == "__main__":
    main()

