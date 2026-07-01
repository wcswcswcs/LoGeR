from __future__ import annotations

import argparse

from stream4d_native.v61_full_eval import V61FullEvalConfig, build_v61_final_decision, write_v61_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v61 final decision.")
    parser.add_argument("--output-root", default="outputs/audit/v61_final_decision")
    args = parser.parse_args()
    cfg = V61FullEvalConfig(output_root=args.output_root)
    result = build_v61_final_decision(cfg)
    outputs = write_v61_final_decision(result, args.output_root)
    summary = result["summary"]
    print(
        {
            "final_decision": outputs["final_decision"],
            "final_metric_rows": outputs["final_metric_rows"],
            "decision_label": summary["decision_label"],
            "core_global_embedding_go": summary["core_global_embedding_go"],
            "blocked_claims": summary["blocked_claims"],
            "go_gate": summary["go_gate"],
        }
    )


if __name__ == "__main__":
    main()
