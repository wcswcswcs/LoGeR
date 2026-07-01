from __future__ import annotations

import argparse

from stream4d_native.v56_full_eval import build_v56_full_eval, write_v56_full_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v56 evidence-typed full audit evaluation.")
    parser.add_argument("--output-root", default="outputs/audit")
    args = parser.parse_args()
    payload = build_v56_full_eval()
    paths = write_v56_full_eval(payload, output_root=args.output_root)
    final = payload["final_summary"]
    print(
        {
            "final_decision": paths["v56_final_decision/final_decision.json"],
            "final_label": final["final_label"],
            "partial_label": final["partial_label"],
            "core_ARI": final["core_4D_ARI"],
            "core_purity": final["core_purity"],
            "core_completeness": final["core_completeness"],
            "expanded_ARI": final["expanded_4D_ARI"],
            "expanded_purity": final["expanded_purity"],
            "expanded_completeness": final["expanded_completeness"],
            "native_field_available": final["native_field_available"],
        }
    )


if __name__ == "__main__":
    main()

