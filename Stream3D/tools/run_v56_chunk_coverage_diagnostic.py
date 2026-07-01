from __future__ import annotations

import argparse

from stream4d_native.v56_chunk_coverage_diagnostic import (
    build_v56_chunk_coverage_diagnostic,
    write_v56_chunk_coverage_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v56 chunk role and evidence coverage diagnostic.")
    parser.add_argument("--output-root", default="outputs/audit/v56_chunk_coverage_diagnostic")
    args = parser.parse_args()
    payload = build_v56_chunk_coverage_diagnostic()
    write_v56_chunk_coverage_diagnostic(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/chunk_coverage_summary.json",
            "evidence_chunk_count": summary["evidence_chunk_count"],
            "evidence_chunks_with_best_objectlets": summary["evidence_chunks_with_best_objectlets"],
            "evidence_chunks_without_best_objectlets": summary["evidence_chunks_without_best_objectlets"],
            "evidence_chunks_with_c3_confirmed_update": summary["evidence_chunks_with_c3_confirmed_update"],
            "bridge_chunks_without_best_objectlets": summary["bridge_chunks_without_best_objectlets"],
            "total_best_objectlets_in_evidence_chunks": summary["total_best_objectlets_in_evidence_chunks"],
            "total_c3_confirmed_updates": summary["total_c3_confirmed_updates"],
            "total_c3_no_evidence_history_count": summary["total_c3_no_evidence_history_count"],
        }
    )


if __name__ == "__main__":
    main()
