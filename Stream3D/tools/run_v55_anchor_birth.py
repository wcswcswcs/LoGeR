from __future__ import annotations

import argparse

from stream4d_native.v55_anchor_birth import build_v55_anchor_birth, write_v55_anchor_birth


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v55 Phase 3 anchor chunk object birth.")
    parser.add_argument("--output-root", default="outputs/audit/v55_anchor_birth")
    parser.add_argument("--visualization-root", default="outputs/audit/v55_visualizations/anchor_birth")
    args = parser.parse_args()
    payload = build_v55_anchor_birth()
    write_v55_anchor_birth(args.output_root, payload, visualization_root=args.visualization_root)
    summary = payload["summary"]
    print(
        {
            "anchor_birth": f"{args.output_root}/anchor_birth_summary.json",
            "gate": summary["gate"],
            "birth_candidate_count": summary["birth_candidate_count"],
            "accepted_birth_count": summary["accepted_birth_count"],
            "birth_purity_diagnostic": summary["birth_purity_diagnostic"],
            "birth_completeness_diagnostic": summary["birth_completeness_diagnostic"],
        }
    )


if __name__ == "__main__":
    main()
