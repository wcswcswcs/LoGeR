from __future__ import annotations

import argparse

from stream4d_native.v58_fact_lock import build_v58_fact_lock, write_v58_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v58 Phase 0 v56 fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v58_phase0_fact_lock")
    args = parser.parse_args()
    payload = build_v58_fact_lock()
    outputs = write_v58_fact_lock(args.output_root, payload)
    fact_lock = payload["fact_lock"]
    print(
        {
            "fact_lock": outputs["fact_lock"],
            "v56_baseline_rows": outputs["v56_baseline_rows"],
            "dashboard": outputs["dashboard"],
            "dashboard_status": outputs["dashboard_status"],
            "gate": fact_lock["gate"],
            "v56_final_label": fact_lock["v56_final_label"],
            "v56_partial_label": fact_lock["v56_partial_label"],
            "v56_core_purity": fact_lock["v56_core_purity"],
            "v56_promoted_component_count": fact_lock["v56_promoted_component_count"],
            "v56_native_field_available": fact_lock["v56_native_field_available"],
        }
    )


if __name__ == "__main__":
    main()

