from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import failure_autopsy, load_scene_artifacts, read_json, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v46 failure autopsy rows.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_semantic_part_graph_radio")
    parser.add_argument("--alignment-root", default="outputs/audit/v42_part_gated_alignment_combined_dino_prepared_allframe_r1")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--positive-root", default="outputs/audit/v46_positive_edges")
    parser.add_argument("--negative-root", default="outputs/audit/v46_negative_edges")
    parser.add_argument("--solver-root", default="outputs/audit/v46_solver_comparison")
    parser.add_argument("--output-root", default="outputs/audit/v46_failure_autopsy")
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, alignment_root=args.alignment_root, scenes=args.scenes)
    positive = read_json(ROOT / args.positive_root / "positive_edge_audit.json") or {}
    negative = read_json(ROOT / args.negative_root / "negative_edge_audit.json") or {}
    solver = read_json(ROOT / args.solver_root / "solver_comparison.json") or {}
    payload = failure_autopsy(artifacts, positive_payload=positive, negative_payload=negative, solver_payload=solver)
    out = ROOT / args.output_root
    write_json(out / "failure_autopsy.json", payload)
    write_csv(out / "per_scene_metric_table.csv", payload["per_scene_metric_table"])
    write_csv(out / "edge_error_rows.csv", payload["edge_error_rows"])
    write_csv(out / "top_false_positive_edges.csv", payload["top_false_positive_edges"])
    write_csv(out / "top_false_negative_edges.csv", payload["top_false_negative_edges"])
    write_csv(out / "underseg_supporter_failures.csv", payload["underseg_supporter_failures"])
    write_csv(out / "solver_merge_trace.csv", payload["solver_merge_trace"])
    write_csv(out / "cluster_conflict_rows.csv", payload["cluster_conflict_rows"])
    write_csv(out / "carrier_assignment_conflict_rows.csv", payload["carrier_assignment_conflict_rows"])
    write_json(out / "visualization_manifest.json", payload["visualization_manifest"])
    print(json.dumps({"summary": str(out / "failure_autopsy.json"), "final_failure_label": payload["final_failure_label"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
