from __future__ import annotations

import argparse
import json
from typing import Any

from stream4d_native.v47_common import ROOT, parse_bool, parse_float, read_csv, read_json, write_csv, write_json
from stream4d_native.v47_failure_autopsy import build_failure_summary


def _optional_json(path):
    return read_json(path) if path.exists() else None


def _optional_csv(path):
    return read_csv(path) if path.exists() else []


def _edge_error_rows(edge_rows, score_key="A5_d4rt_semantic_confirmation", topn=200):
    false_high = [
        dict(row, error_type="false_high_score_edge")
        for row in edge_rows
        if not parse_bool(row.get("diagnostic_same_gt"))
    ]
    false_high.sort(key=lambda row: parse_float(row.get(score_key)), reverse=True)
    missed_true = [
        dict(row, error_type="low_score_same_gt_edge")
        for row in edge_rows
        if parse_bool(row.get("diagnostic_same_gt"))
    ]
    missed_true.sort(key=lambda row: parse_float(row.get(score_key)))
    return false_high[:topn] + missed_true[:topn]


def _tracklet_error_rows(scan_rows, topn=200):
    rows = [dict(row, error_type="tracklet_control_gate_failed") for row in scan_rows]
    rows.sort(key=lambda row: parse_float(row.get("real_minus_no_temporal_tracklet_ARI")), reverse=True)
    return rows[:topn]


def _status_rows(status, reason):
    return [{"status": status, "reason": reason}]


def _pair_key(row):
    return (str(row.get("src_tracklet_id")), str(row.get("dst_tracklet_id")))


def _reactivation_error_rows(candidates, selected, topn=250):
    selected_keys = {_pair_key(row) for row in selected}
    false_selected = [
        dict(row, error_type="reactivation_false_merge_selected")
        for row in selected
        if str(row.get("diagnostic_src_gt")) and str(row.get("diagnostic_dst_gt")) and not parse_bool(row.get("diagnostic_same_gt"))
    ]
    false_selected.sort(key=lambda row: parse_float(row.get("selected_score")), reverse=True)
    missed_true = [
        dict(
            row,
            error_type="reactivation_true_pair_not_selected",
            candidate_score=parse_float(row.get("A5_d4rt_semantic_confirmation")) + parse_float(row.get("semantic_memory_similarity")),
        )
        for row in candidates
        if _pair_key(row) not in selected_keys and parse_bool(row.get("diagnostic_same_gt"))
    ]
    missed_true.sort(key=lambda row: parse_float(row.get("candidate_score")), reverse=True)
    return false_selected[:topn] + missed_true[:topn]


def _local_graph_error_rows(scan_rows, topn=250):
    rows = [dict(row, error_type="local_graph_gate_failed") for row in scan_rows]
    rows.sort(
        key=lambda row: (
            parse_float(row.get("ARI_change")),
            parse_float(row.get("purity_change")),
            -parse_float(row.get("hard_negative_violation_rate")),
        ),
        reverse=True,
    )
    return rows[:topn]


def _underseg_rows(underseg_rows, topn=500):
    rows = [dict(row, error_type="underseg_shared_observation") for row in underseg_rows]
    rows.sort(
        key=lambda row: (
            parse_float(row.get("supporting_primary_track_count")),
            parse_float(row.get("supporting_carrier_count")),
            -parse_float(row.get("dominant_primary_track_fraction")),
        ),
        reverse=True,
    )
    return rows[:topn]


def _per_scene_rows(tracklet_summary, control_summary):
    rows = []
    if tracklet_summary:
        rows.extend(
            [
                {
                    "source": "tracklet_main",
                    "variant": tracklet_summary.get("score_key"),
                    "scene": "scene0081_01",
                    "tracklet_purity": tracklet_summary.get("scene0081_tracklet_purity"),
                },
                {
                    "source": "tracklet_main",
                    "variant": tracklet_summary.get("score_key"),
                    "scene": "scene0591_00",
                    "tracklet_purity": tracklet_summary.get("scene0591_tracklet_purity"),
                },
            ]
        )
    best = (control_summary or {}).get("best_row") or {}
    for prefix in ["real", "shuffled", "no_temporal"]:
        for scene in ["scene0081", "scene0591"]:
            rows.append(
                {
                    "source": "tracklet_control_scan_best",
                    "variant": prefix,
                    "scene": f"{scene}_00" if scene == "scene0591" else "scene0081_01",
                    "tracklet_purity": best.get(f"{prefix}_{scene}_tracklet_purity"),
                    "tracklet_ARI": best.get(f"{prefix}_tracklet_ARI"),
                    "selected_edge_count": best.get(f"{prefix}_selected_edge_count"),
                }
            )
    return rows


def _reactivation_selected_pairs_file(reactivation_summary, explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    variant = str((reactivation_summary or {}).get("recommended_reactivation_variant") or "")
    if variant.startswith("R1"):
        return "reactivation_best_R1_selected_pairs.csv"
    if variant.startswith("R2"):
        return "reactivation_best_R2_selected_pairs.csv"
    if variant.startswith("R5"):
        return "reactivation_best_R5_selected_pairs.csv"
    return "reactivation_best_R5_selected_pairs.csv"


def _extend_per_scene_rows(rows, matching_summary, reactivation_summary, underseg_summary, local_graph_summary, stage1_final_summary):
    if matching_summary:
        for key in ["best_real_row", "best_no_temporal_row", "best_shuffled_row"]:
            item = matching_summary.get(key) or {}
            if item:
                rows.append(
                    {
                        "source": "matching_flow",
                        "variant": key,
                        "score_key": item.get("score_key"),
                        "4D_ARI": item.get("ARI"),
                        "4D_purity": item.get("purity"),
                        "4D_completeness": item.get("completeness"),
                        "temporal_span_mean": item.get("temporal_span_mean"),
                        "scene0081_ARI": item.get("scene0081_ARI"),
                        "scene0011_purity": item.get("scene0011_purity"),
                        "scene0050_purity": item.get("scene0050_purity"),
                        "scene0591_completeness": item.get("scene0591_completeness"),
                    }
                )
    if reactivation_summary:
        item = reactivation_summary.get("best_R5_row") or {}
        rows.append(
            {
                "source": "reactivation",
                "variant": item.get("variant"),
                "4D_ARI": item.get("ARI"),
                "4D_purity": item.get("purity"),
                "4D_completeness": item.get("completeness"),
                "temporal_span_mean": item.get("temporal_span_mean"),
                "ARI_change": item.get("ARI_gain"),
                "purity_change": item.get("purity_change"),
                "completeness_change": item.get("completeness_change"),
            }
        )
    if underseg_summary:
        item = underseg_summary.get("after") or {}
        rows.append(
            {
                "source": "underseg_refined_veto",
                "variant": "best_scan_row",
                "4D_ARI": item.get("ARI"),
                "4D_purity": item.get("purity"),
                "4D_completeness": item.get("completeness"),
                "temporal_span_mean": item.get("temporal_span_mean"),
                "ARI_change": underseg_summary.get("ARI_change"),
                "purity_change": underseg_summary.get("purity_gain"),
                "completeness_change": -parse_float(underseg_summary.get("completeness_drop")),
            }
        )
    if local_graph_summary:
        item = local_graph_summary.get("recommended_local_row") or local_graph_summary.get("best_local_row") or {}
        rows.append(
            {
                "source": "local_graph_refinement",
                "variant": item.get("variant"),
                "4D_ARI": item.get("ARI"),
                "4D_purity": item.get("purity"),
                "4D_completeness": item.get("completeness"),
                "temporal_span_mean": item.get("temporal_span_mean"),
                "ARI_change": item.get("ARI_change"),
                "purity_change": item.get("purity_change"),
                "completeness_change": item.get("completeness_change"),
                "hard_negative_violation_rate": item.get("hard_negative_violation_rate"),
            }
        )
    if stage1_final_summary:
        item = stage1_final_summary.get("method_row") or {}
        rows.append(
            {
                "source": "stage1_final_gate",
                "variant": stage1_final_summary.get("method_variant"),
                "4D_ARI": item.get("ARI"),
                "4D_purity": item.get("purity"),
                "4D_completeness": item.get("completeness"),
                "temporal_span_mean": item.get("temporal_span_mean"),
                "scene0081_ARI": item.get("scene0081_ARI"),
                "scene0011_purity": item.get("scene0011_purity"),
                "scene0050_purity": item.get("scene0050_purity"),
                "scene0591_completeness": item.get("scene0591_completeness"),
                "stage1_gate_pass": stage1_final_summary.get("gate", {}).get("pass"),
            }
        )
    return rows


def _extended_failure_payload(payload, matching_summary, reactivation_summary, underseg_summary, local_graph_summary, stage1_final_summary):
    layers = []
    if matching_summary:
        layers.append(
            {
                "layer": "matching_flow",
                "gate": matching_summary.get("gate", {}),
                "best_real_row": matching_summary.get("best_real_row"),
                "best_real_minus_best_no_temporal_ARI": matching_summary.get("best_real_minus_best_no_temporal_ARI"),
                "best_real_minus_best_shuffled_ARI": matching_summary.get("best_real_minus_best_shuffled_ARI"),
            }
        )
    if reactivation_summary:
        layers.append(
            {
                "layer": "reactivation",
                "gate": reactivation_summary.get("gate", {}),
                "best_R5_row": reactivation_summary.get("best_R5_row"),
            }
        )
    if underseg_summary:
        layers.append(
            {
                "layer": "underseg",
                "gate": underseg_summary.get("gate", {}),
                "best_scan_row": underseg_summary.get("best_scan_row"),
            }
        )
    if local_graph_summary:
        layers.append(
            {
                "layer": "local_graph_refinement",
                "gate": local_graph_summary.get("gate", {}),
                "recommended_local_row": local_graph_summary.get("recommended_local_row") or local_graph_summary.get("best_local_row"),
                "best_raw_local_row": local_graph_summary.get("best_raw_local_row"),
                "best_safe_local_row": local_graph_summary.get("best_safe_local_row"),
            }
        )
    if stage1_final_summary:
        layers.append(
            {
                "layer": "stage1_final_gate",
                "gate": stage1_final_summary.get("gate", {}),
                "method_variant": stage1_final_summary.get("method_variant"),
                "method_row": stage1_final_summary.get("method_row"),
            }
        )
    payload = dict(payload)
    payload["failure_layers"] = layers
    reactivation_pass = bool((reactivation_summary or {}).get("gate", {}).get("pass"))
    later_gate_failed = any(
        summary and summary.get("gate", {}).get("pass") is False
        for summary in [matching_summary, underseg_summary, local_graph_summary]
    )
    if reactivation_pass and later_gate_failed:
        payload["prior_failure_label"] = payload.get("failure_label")
        payload["failure_label"] = "NO_GO_STAGE1_NOT_SIGNIFICANT"
        payload["blocker"] = (
            "R5 reactivation is a real partial signal, but matching-flow/component, underseg, "
            "and local-graph refinements do not pass Stage-1 gates."
        )
    if stage1_final_summary and stage1_final_summary.get("gate", {}).get("pass") is not True:
        method = stage1_final_summary.get("method_row") or {}
        gate = stage1_final_summary.get("gate") or {}
        payload["prior_failure_label"] = payload.get("failure_label")
        payload["failure_label"] = "NO_GO_STAGE1_NOT_SIGNIFICANT"
        payload["blocker"] = (
            "Phase9 Stage-1 final gate failed: "
            f"4D_ARI={method.get('ARI')}, 4D_completeness={method.get('completeness')}, "
            f"threshold_failed={gate.get('threshold_failed_metric_names')}, "
            f"unavailable={gate.get('unavailable_metric_names')}."
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v47 failure autopsy summary.")
    parser.add_argument("--edge-root", default="outputs/audit/v47_adjacent_edges")
    parser.add_argument("--tracklet-root", default="outputs/audit/v47_tracklets")
    parser.add_argument("--tracklet-control-root", default="")
    parser.add_argument("--matching-flow-root", default="")
    parser.add_argument("--reactivation-root", default="")
    parser.add_argument("--reactivation-selected-pairs-file", default="")
    parser.add_argument("--underseg-root", default="")
    parser.add_argument("--local-graph-root", default="")
    parser.add_argument("--stage1-final-root", default="")
    parser.add_argument("--output-root", default="outputs/audit/v47_failure_autopsy")
    args = parser.parse_args()
    edge_root = ROOT / str(args.edge_root)
    tracklet_root = ROOT / str(args.tracklet_root)
    control_root = ROOT / str(args.tracklet_control_root) if str(args.tracklet_control_root).strip() else None
    edge_summary = _optional_json(edge_root / "adjacent_edge_audit.json")
    tracklet_summary = _optional_json(tracklet_root / "tracklet_construction.json")
    control_summary = _optional_json(control_root / "tracklet_filter_scan_summary.json") if control_root else None
    matching_root = ROOT / str(args.matching_flow_root) if str(args.matching_flow_root).strip() else None
    reactivation_root = ROOT / str(args.reactivation_root) if str(args.reactivation_root).strip() else None
    underseg_root = ROOT / str(args.underseg_root) if str(args.underseg_root).strip() else None
    local_graph_root = ROOT / str(args.local_graph_root) if str(args.local_graph_root).strip() else None
    stage1_final_root = ROOT / str(args.stage1_final_root) if str(args.stage1_final_root).strip() else None
    matching_summary = _optional_json(matching_root / "matching_flow_summary.json") if matching_root else None
    reactivation_summary = _optional_json(reactivation_root / "reactivation_summary.json") if reactivation_root else None
    underseg_summary = _optional_json(underseg_root / "underseg_reactivation_summary.json") if underseg_root else None
    local_graph_summary = _optional_json(local_graph_root / "local_graph_refinement_summary.json") if local_graph_root else None
    stage1_final_summary = _optional_json(stage1_final_root / "stage1_final_gate_summary.json") if stage1_final_root else None
    payload = build_failure_summary(
        edge_summary=edge_summary,
        tracklet_summary=tracklet_summary,
        tracklet_control_summary=control_summary,
    )
    payload = _extended_failure_payload(payload, matching_summary, reactivation_summary, underseg_summary, local_graph_summary, stage1_final_summary)
    out = ROOT / str(args.output_root)
    blocker = payload["blocker"]
    edge_rows = _optional_csv(edge_root / "temporal_candidate_edge_table.csv")
    scan_rows = _optional_csv(control_root / "tracklet_filter_scan_rows.csv") if control_root else []
    matching_rows = _optional_csv(matching_root / "matching_flow_best_real_selected_edges.csv") if matching_root else []
    reactivation_candidates = _optional_csv(reactivation_root / "reactivation_candidate_rows.csv") if reactivation_root else []
    reactivation_pairs_file = _reactivation_selected_pairs_file(reactivation_summary, args.reactivation_selected_pairs_file)
    reactivation_selected = _optional_csv(reactivation_root / reactivation_pairs_file) if reactivation_root else []
    underseg_rows = _optional_csv(underseg_root / "underseg_shared_observation_rows.csv") if underseg_root else []
    local_graph_rows = _optional_csv(local_graph_root / "local_graph_refinement_scan_rows.csv") if local_graph_root else []
    stage1_gate_rows = _optional_csv(stage1_final_root / "stage1_final_gate_rows.csv") if stage1_final_root else []
    stage1_metric_rows = _optional_csv(stage1_final_root / "stage1_final_gate_metric_rows.csv") if stage1_final_root else []
    edge_error_rows = _edge_error_rows(edge_rows)
    tracklet_error_rows = _tracklet_error_rows(scan_rows)
    reactivation_error_rows = _reactivation_error_rows(reactivation_candidates, reactivation_selected) if reactivation_root else _status_rows("not_run", "No reactivation root provided.")
    underseg_shared_rows = _underseg_rows(underseg_rows) if underseg_root else _status_rows("not_run", "No underseg root provided.")
    local_graph_error_rows = _local_graph_error_rows(local_graph_rows) if local_graph_root else _status_rows("not_run", "No local graph root provided.")
    write_json(out / "failure_summary.json", payload)
    write_csv(out / "edge_error_rows.csv", edge_error_rows)
    write_csv(out / "tracklet_error_rows.csv", tracklet_error_rows)
    write_csv(out / "flow_selected_edge_rows.csv", matching_rows if matching_rows else _status_rows("not_run", "No matching-flow selected edge rows available."))
    write_csv(out / "reactivation_error_rows.csv", reactivation_error_rows)
    write_csv(out / "underseg_shared_observation_rows.csv", underseg_shared_rows)
    write_csv(out / "local_graph_error_rows.csv", local_graph_error_rows)
    write_csv(out / "stage1_final_gate_rows.csv", stage1_gate_rows if stage1_gate_rows else _status_rows("not_run", "No Stage-1 final gate root provided."))
    write_csv(out / "stage1_final_metric_rows.csv", stage1_metric_rows if stage1_metric_rows else _status_rows("not_run", "No Stage-1 final gate root provided."))
    write_csv(out / "mask_atom_rows.csv", _status_rows("not_implemented", "Phase6 tried shared-observation veto scan; mask atomization was not promoted into the method."))
    write_csv(
        out / "per_scene_metric_table.csv",
        _extend_per_scene_rows(
            _per_scene_rows(tracklet_summary, control_summary),
            matching_summary,
            reactivation_summary,
            underseg_summary,
            local_graph_summary,
            stage1_final_summary,
        ),
    )
    false_reactivation = [row for row in reactivation_error_rows if row.get("error_type") == "reactivation_false_merge_selected"]
    missed_reactivation = [row for row in reactivation_error_rows if row.get("error_type") == "reactivation_true_pair_not_selected"]
    write_json(
        out / "false_merge_casebook.json",
        {
            "status": "partial_rows_with_reactivation_underseg_local_graph",
            "reason": blocker,
            "top_false_high_score_edges": edge_error_rows[:20],
            "reactivation_false_merge_selected": false_reactivation[:20],
            "reactivation_selected_pairs_file": reactivation_pairs_file,
            "underseg_shared_observation_top": underseg_shared_rows[:20],
            "local_graph_top_rows": local_graph_error_rows[:20],
            "stage1_final_gate_rows": stage1_gate_rows[:20],
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    )
    write_json(
        out / "false_cut_casebook.json",
        {
            "status": "partial_rows_with_reactivation_underseg_local_graph",
            "reason": blocker,
            "top_low_score_same_gt_edges": [row for row in edge_error_rows if row.get("error_type") == "low_score_same_gt_edge"][:20],
            "reactivation_true_pair_not_selected": missed_reactivation[:20],
            "reactivation_selected_pairs_file": reactivation_pairs_file,
            "local_graph_top_rows": local_graph_error_rows[:20],
            "stage1_final_gate_rows": stage1_gate_rows[:20],
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    )
    best = (control_summary or {}).get("best_row") or {}
    best_r5 = (reactivation_summary or {}).get("best_R5_row") or {}
    recommended_reactivation = (reactivation_summary or {}).get("recommended_reactivation_row") or best_r5
    best_underseg = (underseg_summary or {}).get("best_scan_row") or {}
    best_local = (local_graph_summary or {}).get("recommended_local_row") or (local_graph_summary or {}).get("best_local_row") or {}
    stage1_gate = (stage1_final_summary or {}).get("gate") or {}
    stage1_method = (stage1_final_summary or {}).get("method_row") or {}
    (out / "failure_summary.md").write_text(
        "\n".join(
            [
                "# v47 Failure Summary",
                "",
                f"label: {payload['failure_label']}",
                "",
                blocker,
                "",
                f"prior_failure_label: {payload.get('prior_failure_label')}",
                "",
                "## Key Evidence",
                "",
                f"- edge_gate_pass: {(edge_summary or {}).get('gate', {}).get('pass')}",
                f"- tracklet_basic_gate_pass: {(tracklet_summary or {}).get('gate', {}).get('pass')}",
                f"- tracklet_control_gate_pass: {(control_summary or {}).get('gate', {}).get('pass')}",
                f"- control_rows_scanned: {(control_summary or {}).get('result_rows')}",
                f"- best_real_minus_shuffled_tracklet_ARI: {best.get('real_minus_shuffled_tracklet_ARI')}",
                f"- best_real_minus_no_temporal_tracklet_ARI: {best.get('real_minus_no_temporal_tracklet_ARI')}",
                f"- matching_flow_gate_pass: {(matching_summary or {}).get('gate', {}).get('pass')}",
                f"- reactivation_gate_pass: {(reactivation_summary or {}).get('gate', {}).get('pass')}",
                f"- recommended_reactivation_variant: {(reactivation_summary or {}).get('recommended_reactivation_variant')}",
                f"- recommended_reactivation_ARI_gain: {recommended_reactivation.get('ARI_gain')}",
                f"- recommended_reactivation_purity_change: {recommended_reactivation.get('purity_change')}",
                f"- R5_ARI_gain: {best_r5.get('ARI_gain')}",
                f"- R5_purity_change: {best_r5.get('purity_change')}",
                f"- reactivation_selected_pairs_file: {reactivation_pairs_file}",
                f"- underseg_gate_pass: {(underseg_summary or {}).get('gate', {}).get('pass')}",
                f"- underseg_purity_gain: {(underseg_summary or {}).get('purity_gain')}",
                f"- underseg_ARI_change: {(underseg_summary or {}).get('ARI_change')}",
                f"- local_graph_gate_pass: {(local_graph_summary or {}).get('gate', {}).get('pass')}",
                f"- local_graph_best_ARI_change: {best_local.get('ARI_change')}",
                f"- local_graph_best_purity_change: {best_local.get('purity_change')}",
                f"- stage1_final_gate_pass: {stage1_gate.get('pass')}",
                f"- stage1_method_ARI: {stage1_method.get('ARI')}",
                f"- stage1_method_completeness: {stage1_method.get('completeness')}",
                f"- stage1_threshold_failed_metric_names: {stage1_gate.get('threshold_failed_metric_names')}",
                f"- stage1_unavailable_metric_names: {stage1_gate.get('unavailable_metric_names')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print({"summary": str(out / "failure_summary.json"), "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
