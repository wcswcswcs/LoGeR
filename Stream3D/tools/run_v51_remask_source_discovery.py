from __future__ import annotations

import argparse

from stream4d_native.v51_remask_source_discovery import build_v51_source_discovery, write_v51_source_discovery


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v51-r2 Phase1 mask source discovery.")
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_source_discovery")
    parser.add_argument("--skip-external-npz", action="store_true")
    parser.add_argument("--max-npz-roots", type=int, default=None)
    args = parser.parse_args()
    payload = build_v51_source_discovery(
        include_external_npz=not bool(args.skip_external_npz),
        max_npz_roots=args.max_npz_roots,
    )
    write_v51_source_discovery(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/source_discovery_summary.json",
            "gate": payload["gate"],
            "selected_source_id": payload["summary"].get("selected_source_id"),
            "stream3d_current_source": payload["summary"].get("stream3d_current_source"),
        }
    )


if __name__ == "__main__":
    main()
