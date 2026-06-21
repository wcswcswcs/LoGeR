from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_hypothesis_selection, write_v50_hypothesis_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 7 hypothesis selection audit.")
    parser.add_argument(
        "--output-root",
        default="outputs/audit/v50_hypothesis_selection",
        help="Output directory under the Stream3D root.",
    )
    args = parser.parse_args()

    payload = build_v50_hypothesis_selection()
    write_v50_hypothesis_selection(args.output_root, payload)
    best = payload["summary"]["best_real_row"]
    print(
        {
            "summary": f"{args.output_root}/selection_summary.json",
            "gate": payload["gate"],
            "best_solver_variant": best["solver_variant"],
            "ARI": best["ARI"],
            "purity": best["purity"],
            "completeness": best["completeness"],
            "conflict_rate": best["conflict_rate"],
            "ap_smoke_queue_count": payload["summary"]["ap_smoke_queue_count"],
            "ap_diagnostic_queue_count": payload["summary"]["ap_diagnostic_queue_count"],
        }
    )


if __name__ == "__main__":
    main()
