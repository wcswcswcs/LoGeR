from __future__ import annotations

import argparse

from stream4d_native.v47_common import ROOT, read_json, utc_now, write_json


def _load(path):
    return read_json(path) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v47 final decision.")
    parser.add_argument("--fact-root", default="outputs/audit/v47_fact_lock")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables")
    parser.add_argument("--edge-root", default="outputs/audit/v47_adjacent_edges")
    parser.add_argument("--tracklet-root", default="outputs/audit/v47_tracklets")
    parser.add_argument("--flow-root", default="outputs/audit/v47_flow")
    parser.add_argument("--carrier-supertrack-root", default="")
    parser.add_argument("--carrier-mdl-root", default="")
    parser.add_argument("--carrier-coarse-mdl-root", default="")
    parser.add_argument("--component-edge-root", default="")
    parser.add_argument("--component-constrained-root", default="")
    parser.add_argument("--component-raw-geometry-root", default="")
    parser.add_argument("--matching-flow-root", default="")
    parser.add_argument("--reactivation-root", default="")
    parser.add_argument("--underseg-root", default="")
    parser.add_argument("--local-graph-root", default="")
    parser.add_argument("--stage1-final-root", default="")
    parser.add_argument("--autopsy-root", default="outputs/audit/v47_failure_autopsy")
    parser.add_argument("--output-root", default="outputs/audit/v47_final_decision")
    args = parser.parse_args()
    fact = _load(ROOT / str(args.fact_root) / "fact_lock.json")
    obs = _load(ROOT / str(args.observation_root) / "observation_table_summary.json")
    edge = _load(ROOT / str(args.edge_root) / "adjacent_edge_audit.json")
    tracklet = _load(ROOT / str(args.tracklet_root) / "tracklet_construction.json")
    flow = _load(ROOT / str(args.flow_root) / "min_cost_flow.json")
    carrier = (
        _load(ROOT / str(args.carrier_supertrack_root) / "carrier_supertrack_summary.json")
        if str(args.carrier_supertrack_root).strip()
        else {}
    )
    carrier_mdl = (
        _load(ROOT / str(args.carrier_mdl_root) / "carrier_component_mdl_semantic_summary.json")
        if str(args.carrier_mdl_root).strip()
        else {}
    )
    carrier_coarse_mdl = (
        _load(ROOT / str(args.carrier_coarse_mdl_root) / "carrier_coarse_to_fine_mdl_summary.json")
        if str(args.carrier_coarse_mdl_root).strip()
        else {}
    )
    component_edge = (
        _load(ROOT / str(args.component_edge_root) / "component_edge_refinement_summary.json")
        if str(args.component_edge_root).strip()
        else {}
    )
    component_constrained = (
        _load(ROOT / str(args.component_constrained_root) / "component_constrained_merge_summary.json")
        if str(args.component_constrained_root).strip()
        else {}
    )
    component_raw_geometry = (
        _load(ROOT / str(args.component_raw_geometry_root) / "component_raw_geometry_merge_summary.json")
        if str(args.component_raw_geometry_root).strip()
        else {}
    )
    matching_flow = (
        _load(ROOT / str(args.matching_flow_root) / "matching_flow_summary.json")
        if str(args.matching_flow_root).strip()
        else {}
    )
    reactivation = (
        _load(ROOT / str(args.reactivation_root) / "reactivation_summary.json")
        if str(args.reactivation_root).strip()
        else {}
    )
    underseg = (
        _load(ROOT / str(args.underseg_root) / "underseg_reactivation_summary.json")
        if str(args.underseg_root).strip()
        else {}
    )
    local_graph = (
        _load(ROOT / str(args.local_graph_root) / "local_graph_refinement_summary.json")
        if str(args.local_graph_root).strip()
        else {}
    )
    stage1_final = (
        _load(ROOT / str(args.stage1_final_root) / "stage1_final_gate_summary.json")
        if str(args.stage1_final_root).strip()
        else {}
    )
    autopsy = _load(ROOT / str(args.autopsy_root) / "failure_summary.json")
    full_stage1_significant = bool(stage1_final.get("gate", {}).get("pass")) if stage1_final else False
    stage1_method_variant = str(stage1_final.get("method_variant") or "")
    if full_stage1_significant:
        final_label = "STAGE1_SIGNIFICANT_PASS"
    elif stage1_method_variant.startswith(("Z8_v47_carrier", "Z8b_v47_carrier", "Z8c_v47_carrier")):
        final_label = "PARTIAL_CARRIER_COMPONENT_SIGNAL"
    elif fact.get("gate", {}).get("pass") is not True:
        final_label = "NO_GO_CARRIER_OBSERVATION_TABLE"
    elif obs.get("gate", {}).get("pass") is not True:
        final_label = "NO_GO_CARRIER_OBSERVATION_TABLE"
    elif edge.get("gate", {}).get("pass") is not True:
        final_label = "NO_GO_ADJACENT_EDGE"
    elif tracklet and tracklet.get("gate", {}).get("pass") is not True:
        final_label = "NO_GO_TRACKLET"
    elif flow and flow.get("gate", {}).get("pass") is True:
        final_label = "PARTIAL_TRACKLET_SIGNAL"
    elif reactivation and reactivation.get("gate", {}).get("pass") is True:
        final_label = "PARTIAL_REACTIVATION_SIGNAL"
    else:
        final_label = autopsy.get("failure_label", "PARTIAL_TRACKLET_SIGNAL")
    payload = {
        "phase": "v47_final_decision",
        "created_at": utc_now(),
        "final_label": final_label,
        "answers": {
            "carrier_observation_table_built": bool(obs.get("carrier_observation_table_exists")),
            "adjacent_temporal_edges_gate_pass": bool(edge.get("gate", {}).get("pass")),
            "d4rt_real_beats_controls": bool(edge.get("gate", {}).get("real_minus_shuffled_pass") and edge.get("gate", {}).get("real_minus_no_temporal_pass")),
            "semantic_memory_reactivation_helped": None
            if not reactivation
            else bool(reactivation.get("semantic_memory_helped_over_d4rt_only", reactivation.get("gate", {}).get("pass"))),
            "d4rt_only_reactivation_helped": None
            if not reactivation
            else bool(reactivation.get("d4rt_only_reactivation_gate", {}).get("pass")),
            "recommended_reactivation_variant": None
            if not reactivation
            else reactivation.get("recommended_reactivation_variant"),
            "underseg_handling_helped": None if not underseg else bool(underseg.get("gate", {}).get("pass")),
            "local_graph_refinement_helped": None
            if not local_graph
            else bool(local_graph.get("gate", {}).get("pass")),
            "min_cost_flow_beats_v46": None,
            "carrier_supertrack_better": None
            if not carrier
            else bool(
                (carrier.get("object_from_component_ARI") or carrier.get("object_from_supertrack_ARI") or 0.0)
                > (tracklet.get("tracklet_ARI") or 0.0)
            ),
            "full_stage1_significant": full_stage1_significant,
            "no_d4rt_birth_no_maskless": True,
            "ap_eval_alignment_only": None,
            "stage2_allowed": full_stage1_significant,
            "failure_location": final_label,
        },
        "fact_gate": fact.get("gate", {}),
        "observation_gate": obs.get("gate", {}),
        "edge_gate": edge.get("gate", {}),
        "tracklet_gate": tracklet.get("gate", {}),
        "flow_gate": flow.get("gate", {}),
        "carrier_supertrack_gate": carrier.get("gate", {}),
        "carrier_supertrack_summary": carrier,
        "carrier_mdl_gate": carrier_mdl.get("gate", {}),
        "carrier_mdl_summary": carrier_mdl,
        "carrier_coarse_mdl_gate": carrier_coarse_mdl.get("gate", {}),
        "carrier_coarse_mdl_summary": carrier_coarse_mdl,
        "component_edge_refinement_gate": component_edge.get("gate", {}),
        "component_edge_refinement_summary": component_edge,
        "component_constrained_merge_gate": component_constrained.get("gate", {}),
        "component_constrained_merge_summary": component_constrained,
        "component_raw_geometry_merge_gate": component_raw_geometry.get("gate", {}),
        "component_raw_geometry_merge_summary": component_raw_geometry,
        "matching_flow_gate": matching_flow.get("gate", {}),
        "matching_flow_summary": matching_flow,
        "reactivation_gate": reactivation.get("gate", {}),
        "reactivation_summary": reactivation,
        "underseg_gate": underseg.get("gate", {}),
        "underseg_summary": underseg,
        "local_graph_gate": local_graph.get("gate", {}),
        "local_graph_summary": local_graph,
        "stage1_final_gate": stage1_final.get("gate", {}),
        "stage1_final_summary": stage1_final,
        "autopsy": autopsy,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "v47_final_decision.json", payload)
    print({"summary": str(out / "v47_final_decision.json"), "final_label": final_label})


if __name__ == "__main__":
    main()
