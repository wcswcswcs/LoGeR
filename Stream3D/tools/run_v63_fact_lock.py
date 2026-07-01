from __future__ import annotations

import argparse

from stream4d_native.v63_fact_lock import V63FactLockConfig, build_v63_fact_lock, write_v63_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock Stream4D v63 Phase 0 facts from real v62 artifacts.")
    parser.add_argument("--output-root", default="outputs/audit/v63_phase0_fact_lock")
    parser.add_argument("--visualization-root", default="outputs/audit/v63_visualizations/phase0")
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    cfg = V63FactLockConfig(
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        tolerance=args.tolerance,
    )
    result = build_v63_fact_lock(cfg)
    outputs = write_v63_fact_lock(result, cfg)
    gate = result["summary"]["gate"]
    print(
        {
            "outputs": outputs,
            "gate": gate,
            "method_status": result["summary"]["method_status"],
            "missing_artifacts": result["summary"]["missing_artifacts"],
        }
    )


if __name__ == "__main__":
    main()
