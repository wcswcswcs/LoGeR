from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import build_final_decision, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v46 final decision from phase artifacts.")
    parser.add_argument("--fact-root", default="outputs/audit/v46_fact_lock")
    parser.add_argument("--incidence-root", default="outputs/audit/v46_incidence")
    parser.add_argument("--supporter-root", default="outputs/audit/v46_supporter_reliability")
    parser.add_argument("--positive-root", default="outputs/audit/v46_positive_edges")
    parser.add_argument("--negative-root", default="outputs/audit/v46_negative_edges")
    parser.add_argument("--solver-root", default="outputs/audit/v46_solver_comparison")
    parser.add_argument("--object-root", default="outputs/audit/v46_object_field_export")
    parser.add_argument("--stage1-root", default="outputs/audit/v46_full_stage1")
    parser.add_argument("--autopsy-root", default="outputs/audit/v46_failure_autopsy")
    parser.add_argument("--ap-root", default="outputs/audit/v46_eval_aligned_ap")
    parser.add_argument("--stage2-root", default="outputs/audit/v46_stage2")
    parser.add_argument("--output-root", default="outputs/audit/v46_final_decision")
    args = parser.parse_args()
    payload = build_final_decision(
        fact_payload=read_json(ROOT / args.fact_root / "fact_lock.json") or {},
        incidence_payload=read_json(ROOT / args.incidence_root / "incidence_audit.json") or {},
        supporter_payload=read_json(ROOT / args.supporter_root / "supporter_reliability_audit.json") or {},
        positive_payload=read_json(ROOT / args.positive_root / "positive_edge_audit.json") or {},
        negative_payload=read_json(ROOT / args.negative_root / "negative_edge_audit.json") or {},
        solver_payload=read_json(ROOT / args.solver_root / "solver_comparison.json") or {},
        object_payload=read_json(ROOT / args.object_root / "object_field_export.json") or {},
        stage1_payload=read_json(ROOT / args.stage1_root / "full_stage1_controls.json") or {},
        autopsy_payload=read_json(ROOT / args.autopsy_root / "failure_autopsy.json") or {},
        ap_payload=read_json(ROOT / args.ap_root / "eval_aligned_ap_summary.json") or {},
        stage2_payload=read_json(ROOT / args.stage2_root / "stage2_eligibility.json") or {},
    )
    out = ROOT / args.output_root
    write_json(out / "v46_final_decision.json", payload)
    print(json.dumps({"summary": str(out / "v46_final_decision.json"), "final_label": payload["final_label"], "answers": payload["answers"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
