#!/usr/bin/env python3
"""Compare two label directories for v108 Phase1 parity diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v108.artifacts import ArtifactWriter  # noqa: E402
from Stream3D.stream4d_v108.phase1_parity import compare_label_dirs, summarize_parity  # noqa: E402


def parse_frame_ids(text: str) -> list[int] | None:
    text = text.strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-label-dir", required=True)
    parser.add_argument("--reference-label-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    candidate_dir = REPO_ROOT / args.candidate_label_dir
    reference_dir = REPO_ROOT / args.reference_label_dir
    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = compare_label_dirs(candidate_dir, reference_dir, parse_frame_ids(args.frame_ids))
    summary = summarize_parity(rows)
    summary.update(
        {
            "candidate_label_dir": args.candidate_label_dir,
            "reference_label_dir": args.reference_label_dir,
            "selfcheck": bool(args.selfcheck),
            "v108_experiment_run": False,
            "gpu_used": False,
            "acceptance_gate": False,
            "diagnostic_only": True,
        }
    )

    writer = ArtifactWriter(output_root)
    writer.write_csv("label_parity_rows.csv", rows, "stream4d_v108_phase1_label_parity_rows_v1")
    summary["artifact_manifest"] = writer.manifest()
    writer.write_json("label_parity_summary.json", summary, "stream4d_v108_phase1_label_parity_summary_v1")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.selfcheck and not summary["all_label_equal"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
