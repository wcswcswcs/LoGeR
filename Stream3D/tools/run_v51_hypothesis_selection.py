from __future__ import annotations

import argparse

from stream4d_native.v51_hypothesis_selection import build_v51_hypothesis_selection, write_v51_hypothesis_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and select v51-r2 hyperedge whole-object hypotheses.")
    parser.add_argument("--hyperedge-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_hypothesis_selection")
    args = parser.parse_args()
    payload = build_v51_hypothesis_selection(args.hyperedge_root, args.semantic_root)
    write_v51_hypothesis_selection(args.output_root, payload)
    print({"summary": f"{args.output_root}/hypothesis_selection_summary.json", "gate": payload["gate"], "metrics": payload["summary"]})


if __name__ == "__main__":
    main()
