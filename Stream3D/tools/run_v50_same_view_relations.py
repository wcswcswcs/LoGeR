from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_same_view_relations, write_v50_same_view_relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 2 relation graph audit.")
    parser.add_argument("--output-root", default="outputs/audit/v50_same_view_relations")
    parser.add_argument("--max-relation-rows", type=int, default=10000)
    args = parser.parse_args()
    payload = build_v50_same_view_relations(max_relation_rows=args.max_relation_rows)
    write_v50_same_view_relations(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/relation_summary.json",
            "gate": payload["gate"],
            "part_edge_count": payload["summary"]["part_edge_count"],
            "sibling_edge_count": payload["summary"]["sibling_edge_count"],
            "part_relation_precision": payload["summary"]["part_relation_precision"],
        }
    )


if __name__ == "__main__":
    main()
