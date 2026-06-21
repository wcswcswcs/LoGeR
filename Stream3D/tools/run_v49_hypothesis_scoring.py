from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_hypothesis_scoring, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 5 hypothesis scoring.")
    parser.add_argument("--output-root", default="outputs/audit/v49_hypothesis_scoring")
    args = parser.parse_args()
    payload = build_hypothesis_scoring()
    write_bundle(
        args.output_root,
        "hypothesis_scoring_summary",
        payload,
        {"scored_hypothesis_rows": payload["hypothesis_rows"], "score_auc_rows": payload["score_auc_rows"]},
    )
    print({"summary": f"{args.output_root}/hypothesis_scoring_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
