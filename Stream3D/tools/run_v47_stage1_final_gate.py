from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_float,
    parse_int,
    read_csv,
    read_json,
    safe_mean,
    utc_now,
    write_csv,
    write_json,
)


def _load(path):
    return read_json(path) if path.exists() else {}


def _metric(name: str, value: Any, *, source: str, note: str = "") -> dict[str, Any]:
    return {
        "metric": name,
        "value": value,
        "available": value is not None,
        "source": source if value is not None else "",
        "note": note,
    }


def _threshold_gate(row: dict[str, Any], *, op: str, threshold: float | int) -> dict[str, Any]:
    value = row.get("value")
    if value is None:
        return {
            "metric": row["metric"],
            "pass": False,
            "reason": "unavailable",
            "value": None,
            "op": op,
            "threshold": threshold,
        }
    if op == ">=":
        passed = float(value) >= float(threshold)
    elif op == "<=":
        passed = float(value) <= float(threshold)
    elif op == "==":
        passed = float(value) == float(threshold)
    else:
        raise ValueError(f"unsupported gate op: {op}")
    return {
        "metric": row["metric"],
        "pass": bool(passed),
        "reason": "ok" if passed else "threshold_not_met",
        "value": value,
        "op": op,
        "threshold": threshold,
    }


def _scene_metric(rows: list[dict[str, Any]], scene_prefix: str, metric: str) -> float | None:
    true_labels = [str(row.get("diagnostic_gt_instance")) for row in rows if str(row.get("scene", "")).startswith(scene_prefix) and str(row.get("diagnostic_gt_instance", ""))]
    pred_labels = [str(row.get("predicted_component_object_id")) for row in rows if str(row.get("scene", "")).startswith(scene_prefix) and str(row.get("diagnostic_gt_instance", ""))]
    if not true_labels:
        return None
    if metric == "ARI":
        return adjusted_rand_score(true_labels, pred_labels)
    if metric == "purity":
        return cluster_purity(true_labels, pred_labels)
    if metric == "completeness":
        return cluster_completeness(true_labels, pred_labels)
    raise ValueError(metric)


def _carrier_component_row(carrier: dict[str, Any], carrier_vote_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not carrier:
        return {}
    component_ids_by_scene: dict[str, set[str]] = defaultdict(set)
    frames_by_component: dict[str, set[int]] = defaultdict(set)
    for row in carrier_vote_rows:
        component = str(row.get("predicted_component_object_id") or "")
        if not component:
            continue
        scene = str(row.get("scene"))
        component_ids_by_scene[scene].add(component)
        frames_by_component[component].add(parse_int(row.get("frame_id")))
    return {
        "variant": "Z8_carrier_supertrack_component_union32",
        "ARI": carrier.get("object_from_component_ARI"),
        "purity": carrier.get("object_from_component_purity"),
        "completeness": carrier.get("object_from_component_completeness"),
        "temporal_span_mean": safe_mean(len(frames) for frames in frames_by_component.values()),
        "mean_predictions_per_scene": safe_mean(len(ids) for ids in component_ids_by_scene.values()),
        "scene0081_ARI": _scene_metric(carrier_vote_rows, "scene0081", "ARI"),
        "scene0011_purity": _scene_metric(carrier_vote_rows, "scene0011", "purity"),
        "scene0050_purity": _scene_metric(carrier_vote_rows, "scene0050", "purity"),
        "scene0591_completeness": _scene_metric(carrier_vote_rows, "scene0591", "completeness"),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "real_minus_shuffled_ARI": carrier.get("real_minus_shuffled_component_ARI"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "source_summary_phase": carrier.get("phase"),
        "component_count": carrier.get("component_count"),
        "max_union_unique_carriers": carrier.get("max_union_unique_carriers"),
    }


def _carrier_mdl_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary.get("best_real_row") or {})
    if not row:
        return {}
    shuffled_deltas = [
        summary.get("best_real_minus_best_shuffled_semantic_ARI"),
        summary.get("best_real_minus_best_shuffled_d4rt_ARI"),
    ]
    available_shuffled = [parse_float(value) for value in shuffled_deltas if value is not None]
    row.update(
        {
            "variant": f"Z8b_carrier_component_mdl_semantic:{row.get('variant', '')}",
            "real_minus_shuffled_ARI": min(available_shuffled) if available_shuffled else None,
            "real_minus_no_temporal_ARI": summary.get("best_real_minus_best_no_temporal_ARI"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "source_summary_phase": summary.get("phase"),
            "source_gate": summary.get("gate", {}),
        }
    )
    return row


def _carrier_coarse_mdl_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary.get("best_real_row") or {})
    if not row:
        return {}
    shuffled_deltas = [
        summary.get("best_real_minus_best_shuffled_coarse_ARI"),
        summary.get("best_real_minus_best_shuffled_semantic_ARI"),
    ]
    available_shuffled = [parse_float(value) for value in shuffled_deltas if value is not None]
    row.update(
        {
            "variant": f"Z8c_carrier_coarse_to_fine_mdl:{row.get('variant', '')}",
            "real_minus_shuffled_ARI": min(available_shuffled) if available_shuffled else None,
            "real_minus_no_temporal_ARI": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "source_summary_phase": summary.get("phase"),
            "source_gate": summary.get("gate", {}),
        }
    )
    return row


def _best_method_row(
    *,
    carrier_component: dict[str, Any],
    carrier_mdl: dict[str, Any],
    carrier_coarse_mdl: dict[str, Any],
    local_graph: dict[str, Any],
    reactivation: dict[str, Any],
    matching_flow: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any], bool]] = []
    if carrier_component:
        candidates.append(("Z8_v47_carrier_supertrack_component_union32", carrier_component, True))
    if carrier_mdl:
        candidates.append(("Z8b_v47_carrier_component_mdl_semantic", carrier_mdl, True))
    if carrier_coarse_mdl:
        candidates.append(("Z8c_v47_carrier_coarse_to_fine_mdl", carrier_coarse_mdl, True))
    if local_graph.get("gate", {}).get("pass") and (local_graph.get("recommended_local_row") or local_graph.get("best_local_row")):
        candidates.append(
            (
                "Z9_v47_r1_d4rt_only_plus_g3_local_hard_negative_veto",
                local_graph.get("recommended_local_row") or local_graph["best_local_row"],
                True,
            )
        )
    if reactivation.get("gate", {}).get("pass") and reactivation.get("recommended_reactivation_row"):
        candidates.append(("Z6_v47_recommended_reactivation", reactivation["recommended_reactivation_row"], True))
    if matching_flow.get("best_real_row"):
        candidates.append(("Z5_v47_matching_flow_best_real", matching_flow["best_real_row"], True))
    candidates = [(name, row, ok) for name, row, ok in candidates if ok and row]
    if not candidates:
        return "none", {}
    name, row, _ok = max(
        candidates,
        key=lambda item: (
            parse_float(item[1].get("ARI")),
            parse_float(item[1].get("purity")),
            parse_float(item[1].get("completeness")),
        ),
    )
    return name, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v47 Phase9 full Stage-1 final gate from existing audit artifacts.")
    parser.add_argument("--matching-flow-root", default="outputs/audit/v47_matching_flow_gap2_global_proxy")
    parser.add_argument("--carrier-supertrack-root", default="")
    parser.add_argument("--carrier-mdl-root", default="")
    parser.add_argument("--carrier-coarse-mdl-root", default="")
    parser.add_argument("--reactivation-root", default="outputs/audit/v47_reactivation_gap2_tracklet_strict_continued12_r1")
    parser.add_argument("--underseg-root", default="outputs/audit/v47_underseg_reactivation_veto_scan_r1_continued12")
    parser.add_argument("--local-graph-root", default="outputs/audit/v47_local_graph_refinement_r1_continued12_medium")
    parser.add_argument("--output-root", default="outputs/audit/v47_stage1_final_gate")
    args = parser.parse_args()

    matching_flow = _load(ROOT / str(args.matching_flow_root) / "matching_flow_summary.json")
    carrier = (
        _load(ROOT / str(args.carrier_supertrack_root) / "carrier_supertrack_summary.json")
        if str(args.carrier_supertrack_root).strip()
        else {}
    )
    carrier_vote_rows = (
        read_csv(ROOT / str(args.carrier_supertrack_root) / "carrier_supertrack_mask_vote_rows.csv")
        if str(args.carrier_supertrack_root).strip()
        else []
    )
    carrier_component = _carrier_component_row(carrier, carrier_vote_rows)
    carrier_mdl_summary = (
        _load(ROOT / str(args.carrier_mdl_root) / "carrier_component_mdl_semantic_summary.json")
        if str(args.carrier_mdl_root).strip()
        else {}
    )
    carrier_coarse_mdl_summary = (
        _load(ROOT / str(args.carrier_coarse_mdl_root) / "carrier_coarse_to_fine_mdl_summary.json")
        if str(args.carrier_coarse_mdl_root).strip()
        else {}
    )
    carrier_mdl = _carrier_mdl_row(carrier_mdl_summary)
    carrier_coarse_mdl = _carrier_coarse_mdl_row(carrier_coarse_mdl_summary)
    reactivation = _load(ROOT / str(args.reactivation_root) / "reactivation_summary.json")
    underseg = _load(ROOT / str(args.underseg_root) / "underseg_reactivation_summary.json")
    local_graph = _load(ROOT / str(args.local_graph_root) / "local_graph_refinement_summary.json")
    method_name, method = _best_method_row(
        carrier_component=carrier_component,
        carrier_mdl=carrier_mdl,
        carrier_coarse_mdl=carrier_coarse_mdl,
        local_graph=local_graph,
        reactivation=reactivation,
        matching_flow=matching_flow,
    )
    dense_semantic = local_graph.get("best_dense_control_row", {}) if local_graph else {}

    metrics = [
        _metric("4D_ARI", method.get("ARI"), source=method_name),
        _metric("4D_purity", method.get("purity"), source=method_name),
        _metric("4D_completeness", method.get("completeness"), source=method_name),
        _metric("3D_ARI", None, source="", note="not produced by current v47 Stage-1 artifacts; AP/eval-aligned adapter is blocked until Stage-1 passes"),
        _metric("3D_purity", None, source="", note="not produced by current v47 Stage-1 artifacts; AP/eval-aligned adapter is blocked until Stage-1 passes"),
        _metric("3D_completeness", None, source="", note="not produced by current v47 Stage-1 artifacts; AP/eval-aligned adapter is blocked until Stage-1 passes"),
        _metric("temporal_span_mean", method.get("temporal_span_mean"), source=method_name),
        _metric("scene0081_ARI", method.get("scene0081_ARI"), source=method_name),
        _metric("scene0011_purity", method.get("scene0011_purity"), source=method_name),
        _metric("scene0050_purity", method.get("scene0050_purity"), source=method_name),
        _metric("scene0591_completeness", method.get("scene0591_completeness"), source=method_name),
        _metric("mean_predictions_per_scene", method.get("mean_predictions_per_scene"), source=method_name),
        _metric("duplicate_rate", None, source="", note="requires object-field/probe5 export; not available for current Stage-1 candidate artifact"),
        _metric("conflict_rate", None, source="", note="requires object-field/probe5 export; not available for current Stage-1 candidate artifact"),
        _metric("unknown_tube_ratio", None, source="", note="requires tube/object-field assignment export; not available for current Stage-1 candidate artifact"),
        _metric("birth_from_d4rt_tube_count", method.get("birth_from_d4rt_tube_count", 0), source=method_name, note="candidate objects are generated from prepared masks or carrier-supported masks; no D4RT-only birth operation"),
        _metric("maskless_object_count", method.get("maskless_object_count", 0), source=method_name, note="candidate objects are generated from prepared masks or carrier-supported masks; no maskless object operation"),
        _metric(
            "real_minus_shuffled_ARI",
            method.get("real_minus_shuffled_ARI"),
            source=method_name if method.get("real_minus_shuffled_ARI") is not None else "",
            note="uses paired shuffled component control when the selected method is carrier component; otherwise unavailable",
        ),
        _metric(
            "real_minus_no_temporal_ARI",
            method.get("real_minus_no_temporal_ARI"),
            source=method_name if method.get("real_minus_no_temporal_ARI") is not None else "",
            note="uses paired no-temporal control when the selected method artifact provides it; matching-flow control is recorded as context only",
        ),
        _metric(
            "real_minus_mask_only_ARI",
            method.get("real_minus_mask_only_ARI"),
            source=method_name if method.get("real_minus_mask_only_ARI") is not None else "",
            note="mask-only Z12 control was not produced unless provided by the selected method artifact",
        ),
        _metric(
            "real_minus_semantic_only_ARI",
            None
            if not dense_semantic
            else float(parse_float(method.get("ARI")) - parse_float(dense_semantic.get("ARI"))),
            source="Z9 best local minus G5_dense_semantic_control" if dense_semantic else "",
            note="G5 is diagnostic semantic-only local graph control and is not promotable when purity drops",
        ),
        _metric("bootstrap_delta_ARI_lower95", None, source="", note="blocked until a full Stage-1 method with required controls exists"),
        _metric("bootstrap_delta_completeness_lower95", None, source="", note="blocked until a full Stage-1 method with required controls exists"),
    ]
    metric_by_name = {row["metric"]: row for row in metrics}
    gate_specs = [
        ("4D_ARI", ">=", 0.485),
        ("4D_purity", ">=", 0.875),
        ("4D_completeness", ">=", 0.555),
        ("temporal_span_mean", ">=", 1.70),
        ("scene0081_ARI", ">=", 0.270),
        ("scene0011_purity", ">=", 0.84),
        ("scene0050_purity", ">=", 0.84),
        ("mean_predictions_per_scene", "<=", 150),
        ("duplicate_rate", "<=", 0.05),
        ("conflict_rate", "<=", 0.10),
        ("unknown_tube_ratio", "<=", 0.35),
        ("birth_from_d4rt_tube_count", "==", 0),
        ("maskless_object_count", "==", 0),
        ("real_minus_shuffled_ARI", ">=", 0.30),
        ("real_minus_no_temporal_ARI", ">=", 0.25),
        ("real_minus_mask_only_ARI", ">=", 0.25),
        ("bootstrap_delta_ARI_lower95", ">=", 0.025),
        ("bootstrap_delta_completeness_lower95", ">=", 0.020),
    ]
    gate_rows = [_threshold_gate(metric_by_name[name], op=op, threshold=threshold) for name, op, threshold in gate_specs]
    gate = {
        "pass": bool(gate_rows and all(row["pass"] for row in gate_rows)),
        "passed_metric_count": sum(1 for row in gate_rows if row["pass"]),
        "failed_metric_count": sum(1 for row in gate_rows if not row["pass"]),
        "unavailable_metric_names": [row["metric"] for row in gate_rows if row["reason"] == "unavailable"],
        "threshold_failed_metric_names": [row["metric"] for row in gate_rows if row["reason"] == "threshold_not_met"],
        "rows": gate_rows,
    }
    matching_flow_context = {}
    if matching_flow:
        matching_flow_context = {
            "best_real_ARI": matching_flow.get("best_real_row", {}).get("ARI"),
            "best_no_temporal_ARI": matching_flow.get("best_no_temporal_row", {}).get("ARI"),
            "best_shuffled_ARI": matching_flow.get("best_shuffled_row", {}).get("ARI"),
            "best_real_minus_best_no_temporal_ARI": matching_flow.get("best_real_minus_best_no_temporal_ARI"),
            "best_real_minus_best_shuffled_ARI": matching_flow.get("best_real_minus_best_shuffled_ARI"),
            "gate": matching_flow.get("gate", {}),
        }
    payload = {
        "phase": "v47_stage1_final_gate",
        "created_at": utc_now(),
        "method_variant": method_name,
        "method_row": method,
        "metric_rows": metrics,
        "gate": gate,
        "matching_flow_preliminary_control_context": matching_flow_context,
        "carrier_component_context": {
            "summary_root": str(ROOT / str(args.carrier_supertrack_root)) if str(args.carrier_supertrack_root).strip() else "",
            "row": carrier_component,
            "gate": carrier.get("gate", {}),
        },
        "carrier_mdl_context": {
            "summary_root": str(ROOT / str(args.carrier_mdl_root)) if str(args.carrier_mdl_root).strip() else "",
            "row": carrier_mdl,
            "gate": carrier_mdl_summary.get("gate", {}),
        },
        "carrier_coarse_mdl_context": {
            "summary_root": str(ROOT / str(args.carrier_coarse_mdl_root)) if str(args.carrier_coarse_mdl_root).strip() else "",
            "row": carrier_coarse_mdl,
            "gate": carrier_coarse_mdl_summary.get("gate", {}),
        },
        "reactivation_gate": reactivation.get("gate", {}),
        "underseg_gate": underseg.get("gate", {}),
        "local_graph_gate": local_graph.get("gate", {}),
        "stage2_allowed": bool(gate["pass"]),
        "status": "stage1_significant_pass" if gate["pass"] else "blocked_stage1_not_significant",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "stage1_final_gate_summary.json", payload)
    write_csv(out / "stage1_final_gate_metric_rows.csv", metrics)
    write_csv(out / "stage1_final_gate_rows.csv", gate_rows)
    print({"summary": str(out / "stage1_final_gate_summary.json"), "status": payload["status"], "gate": gate})


if __name__ == "__main__":
    main()
