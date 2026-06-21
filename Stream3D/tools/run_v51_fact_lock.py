from __future__ import annotations

import argparse

from stream4d_native.v51_remask_source_discovery import build_v51_fact_lock, write_v51_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v51-r2 Phase0 fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_fact_lock")
    args = parser.parse_args()
    payload = build_v51_fact_lock()
    write_v51_fact_lock(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/fact_lock.json",
            "gate": payload["gate"],
            "stream3d_current_mask_source": payload["fact_map"].get("stream3d_current_mask_source"),
            "stream3d_current_mask_overlap_capable": payload["fact_map"].get("stream3d_current_mask_overlap_capable"),
        }
    )


if __name__ == "__main__":
    main()
