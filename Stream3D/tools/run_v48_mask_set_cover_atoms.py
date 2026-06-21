from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Callable

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    parse_bool,
    parse_float,
    parse_int,
    read_csv,
    read_json,
    safe_mean,
    write_csv,
    write_json,
)
from stream4d_native.v48_data_contract import utc_now


def _component_id(row: dict[str, Any]) -> str:
    return str(row.get("predicted_component_object_id") or row.get("mask_observation_id") or row.get("node_id"))


def _is_real_component(component: str) -> bool:
    return bool(component) and not component.startswith("uncovered:")


def _quality(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        parse_float(row.get("supporting_unique_carrier_count")),
        parse_float(row.get("supporting_carrier_observation_count")),
        -parse_float(row.get("node_id")),
    )


def _evaluate(mask_vote_rows: list[dict[str, Any]], pred_fn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    frames: dict[str, set[int]] = defaultdict(set)
    object_gt: dict[str, Counter[str]] = defaultdict(Counter)
    for row in mask_vote_rows:
        gt = str(row.get("diagnostic_gt_instance") or "")
        pred = pred_fn(row)
        frames[pred].add(parse_int(row.get("frame_id")))
        if gt:
            true_labels.append(gt)
            pred_labels.append(pred)
            object_gt[pred][gt] += 1
            scene = str(row.get("scene"))
            scene_true[scene].append(gt)
            scene_pred[scene].append(pred)
    conflict_objects = sum(1 for counts in object_gt.values() if len(counts) > 1)
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "purity": cluster_purity(true_labels, pred_labels),
        "completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(v) for v in frames.values()),
        "cluster_count": len(frames),
        "mean_predictions_per_scene": safe_mean(len(set(scene_pred[scene])) for scene in sorted(scene_pred)),
        "conflict_rate": float(conflict_objects / max(len(object_gt), 1)),
        "scene0081_ARI": adjusted_rand_score(scene_true["scene0081_01"], scene_pred["scene0081_01"])
        if scene_true.get("scene0081_01")
        else None,
        "scene0011_purity": cluster_purity(scene_true["scene0011_00"], scene_pred["scene0011_00"])
        if scene_true.get("scene0011_00")
        else None,
        "scene0050_purity": cluster_purity(scene_true["scene0050_00"], scene_pred["scene0050_00"])
        if scene_true.get("scene0050_00")
        else None,
        "scene0591_completeness": cluster_completeness(scene_true["scene0591_00"], scene_pred["scene0591_00"])
        if scene_true.get("scene0591_00")
        else None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _component_support(mask_vote_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    support: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        comp = _component_id(row)
        if not _is_real_component(comp):
            continue
        item = support.setdefault(
            comp,
            {
                "component_id": comp,
                "mask_count": 0,
                "supporting_unique_carrier_count": 0.0,
                "supporting_carrier_observation_count": 0.0,
                "best_node_id": "",
            },
        )
        item["mask_count"] += 1
        item["supporting_unique_carrier_count"] += parse_float(row.get("supporting_unique_carrier_count"))
        item["supporting_carrier_observation_count"] += parse_float(row.get("supporting_carrier_observation_count"))
        if not item["best_node_id"] or _quality(row) > _quality({"node_id": item["best_node_id"]}):
            item["best_node_id"] = row.get("node_id")
    return support


def _selected_mask_candidates(mask_vote_rows: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in mask_vote_rows
        if _is_real_component(_component_id(row)) and parse_float(row.get("supporting_unique_carrier_count")) > 0
    ]
    candidates.sort(key=_quality, reverse=True)
    selected: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for row in candidates:
        node = str(row.get("node_id"))
        if node in seen_nodes:
            continue
        selected.append(row)
        seen_nodes.add(node)
        if len(selected) >= int(max_candidates):
            break
    return selected


def _selected_components(mask_vote_rows: list[dict[str, Any]], max_candidates: int | None) -> list[dict[str, Any]]:
    support = _component_support(mask_vote_rows)
    rows = sorted(
        support.values(),
        key=lambda row: (
            parse_float(row.get("supporting_unique_carrier_count")),
            parse_float(row.get("supporting_carrier_observation_count")),
            str(row.get("component_id")),
        ),
        reverse=True,
    )
    return rows if max_candidates is None else rows[: int(max_candidates)]


def _variant_row(
    *,
    variant: str,
    prediction_domain: str,
    mask_vote_rows: list[dict[str, Any]],
    selected_components: set[str],
    selected_nodes: set[str],
    total_components: int,
    baseline: dict[str, Any],
    selected_candidate_count: int,
    mask_atom_count: int = 0,
    atoms_per_mask_mean: float | None = None,
    underseg_false_merge_reduction: float | None = None,
    duplicate_rate: float = 0.0,
    note: str = "",
) -> dict[str, Any]:
    def pred(row: dict[str, Any]) -> str:
        comp = _component_id(row)
        node = str(row.get("node_id"))
        if node in selected_nodes:
            return f"{variant}:mask:{node}"
        if comp in selected_components:
            return f"{variant}:component:{comp}"
        return f"{variant}:uncovered:{row.get('mask_observation_id') or node}"

    metrics = _evaluate(mask_vote_rows, pred)
    covered_components = set(selected_components)
    for row in mask_vote_rows:
        if str(row.get("node_id")) in selected_nodes:
            comp = _component_id(row)
            if _is_real_component(comp):
                covered_components.add(comp)
    component_coverage_ratio = float(len(covered_components) / max(total_components, 1))
    row = {
        "variant": variant,
        "prediction_domain": prediction_domain,
        "selected_candidate_count": selected_candidate_count,
        "component_coverage_ratio": component_coverage_ratio,
        "mask_atom_count": mask_atom_count,
        "atoms_per_mask_mean": atoms_per_mask_mean,
        "underseg_false_merge_reduction": underseg_false_merge_reduction,
        "duplicate_rate": duplicate_rate,
        **metrics,
        "delta_ARI_vs_component_baseline": metrics["ARI"] - baseline["ARI"],
        "purity_gain_vs_component_baseline": metrics["purity"] - baseline["purity"],
        "completeness_gain_vs_component_baseline": metrics["completeness"] - baseline["completeness"],
        "gate_component_coverage_pass": component_coverage_ratio >= 0.75,
        "gate_purity_pass": metrics["purity"] >= 0.875,
        "gate_completeness_gain_pass": metrics["completeness"] >= baseline["completeness"] + 0.08,
        "gate_selected_count_pass": selected_candidate_count <= 150,
        "gate_underseg_reduction_pass": underseg_false_merge_reduction is not None and underseg_false_merge_reduction >= 0.20,
        "note": note,
    }
    row["gate_pass"] = bool(
        row["gate_component_coverage_pass"]
        and row["gate_purity_pass"]
        and row["gate_completeness_gain_pass"]
        and row["gate_selected_count_pass"]
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v48 mask set cover and atom repair diagnostics.")
    parser.add_argument("--mask-vote-rows", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--underseg-summary", default="outputs/audit/v47_underseg_reactivation_veto_scan_r1_continued12/underseg_reactivation_summary.json")
    parser.add_argument("--underseg-shared-rows", default="outputs/audit/v47_underseg_reactivation_veto_scan_r1_continued12/underseg_shared_observation_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v48_mask_set_cover_atoms")
    args = parser.parse_args()

    mask_vote_rows = read_csv(ROOT / str(args.mask_vote_rows))
    underseg_summary = read_json(ROOT / str(args.underseg_summary))
    underseg_shared_rows = read_csv(ROOT / str(args.underseg_shared_rows))
    components = sorted({_component_id(row) for row in mask_vote_rows if _is_real_component(_component_id(row))})
    total_components = len(components)

    baseline = _evaluate(mask_vote_rows, lambda row: f"component:{_component_id(row)}")
    shared_rows = [row for row in underseg_shared_rows if parse_bool(row.get("shared_observation"))]
    mask_atom_count = sum(parse_int(row.get("supporting_primary_track_count")) for row in shared_rows)
    atoms_per_mask_mean = safe_mean(parse_int(row.get("supporting_primary_track_count")) for row in shared_rows)
    underseg_reduction = parse_float(underseg_summary.get("underseg_false_merge_reduction"))

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    rows.append(
        _variant_row(
            variant="F0_component_baseline_no_set_cover_no_atoms",
            prediction_domain="carrier_component_mask_vote",
            mask_vote_rows=mask_vote_rows,
            selected_components=set(components),
            selected_nodes=set(),
            total_components=total_components,
            baseline=baseline,
            selected_candidate_count=total_components,
            note="Component baseline reused for comparison; not a set-cover success row.",
        )
    )

    selected_masks_150 = _selected_mask_candidates(mask_vote_rows, 150)
    selected_mask_nodes_150 = {str(row.get("node_id")) for row in selected_masks_150}
    rows.append(
        _variant_row(
            variant="F1_d4rt_native_mask_set_cover_budget150",
            prediction_domain="mask_observation_set_cover",
            mask_vote_rows=mask_vote_rows,
            selected_components=set(),
            selected_nodes=selected_mask_nodes_150,
            total_components=total_components,
            baseline=baseline,
            selected_candidate_count=len(selected_mask_nodes_150),
            note="Top mask observations by carrier support under selected_candidate_count<=150.",
        )
    )

    selected_component_150 = {str(row["component_id"]) for row in _selected_components(mask_vote_rows, 150)}
    rows.append(
        _variant_row(
            variant="F2_carrier_component_coverage_budget150",
            prediction_domain="component_support_set_cover",
            mask_vote_rows=mask_vote_rows,
            selected_components=selected_component_150,
            selected_nodes=set(),
            total_components=total_components,
            baseline=baseline,
            selected_candidate_count=len(selected_component_150),
            note="Object-count penalty keeps <=150 component candidates.",
        )
    )

    selected_component_full = {str(row["component_id"]) for row in _selected_components(mask_vote_rows, None)}
    rows.append(
        _variant_row(
            variant="F2b_carrier_component_coverage_full_no_count_penalty",
            prediction_domain="component_support_set_cover",
            mask_vote_rows=mask_vote_rows,
            selected_components=selected_component_full,
            selected_nodes=set(),
            total_components=total_components,
            baseline=baseline,
            selected_candidate_count=len(selected_component_full),
            note="Full component coverage upper check; fails selected_candidate_count<=150.",
        )
    )

    atom_row = {
        "variant": "F3_mask_atoms_by_previous_track_component_support",
        "prediction_domain": "tracklet_reactivation_diagnostic",
        "selected_candidate_count": parse_int((underseg_summary.get("after") or {}).get("selected_pair_count")),
        "component_coverage_ratio": None,
        "mask_atom_count": mask_atom_count,
        "atoms_per_mask_mean": atoms_per_mask_mean,
        "underseg_false_merge_reduction": underseg_reduction,
        "duplicate_rate": None,
        "conflict_rate": parse_float((underseg_summary.get("after") or {}).get("false_merge_rate")),
        "ARI": parse_float((underseg_summary.get("after") or {}).get("ARI")),
        "purity": parse_float((underseg_summary.get("after") or {}).get("purity")),
        "completeness": parse_float((underseg_summary.get("after") or {}).get("completeness")),
        "delta_ARI_vs_component_baseline": parse_float((underseg_summary.get("after") or {}).get("ARI")) - baseline["ARI"],
        "purity_gain_vs_component_baseline": parse_float((underseg_summary.get("after") or {}).get("purity")) - baseline["purity"],
        "completeness_gain_vs_component_baseline": parse_float((underseg_summary.get("after") or {}).get("completeness")) - baseline["completeness"],
        "gate_component_coverage_pass": False,
        "gate_purity_pass": parse_float((underseg_summary.get("after") or {}).get("purity")) >= 0.875,
        "gate_completeness_gain_pass": False,
        "gate_selected_count_pass": parse_int((underseg_summary.get("after") or {}).get("selected_pair_count")) <= 150,
        "gate_underseg_reduction_pass": underseg_reduction >= 0.20,
        "gate_pass": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "note": "Uses existing v47 shared-observation veto scan; diagnostic domain is tracklet reactivation, not direct component object birth.",
    }
    rows.append(atom_row)

    rows.append(
        _variant_row(
            variant="F6_budget150_set_cover_plus_atoms",
            prediction_domain="component_support_set_cover_plus_tracklet_atom_diagnostic",
            mask_vote_rows=mask_vote_rows,
            selected_components=selected_component_150,
            selected_nodes=set(),
            total_components=total_components,
            baseline=baseline,
            selected_candidate_count=len(selected_component_150),
            mask_atom_count=mask_atom_count,
            atoms_per_mask_mean=atoms_per_mask_mean,
            underseg_false_merge_reduction=underseg_reduction,
            note="Budgeted set-cover candidate with atom diagnostic attached; atom signal is not used to birth objects.",
        )
    )
    rows.append(
        _variant_row(
            variant="F6_fullcoverage_set_cover_plus_atoms",
            prediction_domain="component_support_set_cover_plus_tracklet_atom_diagnostic",
            mask_vote_rows=mask_vote_rows,
            selected_components=selected_component_full,
            selected_nodes=set(),
            total_components=total_components,
            baseline=baseline,
            selected_candidate_count=len(selected_component_full),
            mask_atom_count=mask_atom_count,
            atoms_per_mask_mean=atoms_per_mask_mean,
            underseg_false_merge_reduction=underseg_reduction,
            note="Full-coverage upper check with atom diagnostic; fails object-count budget.",
        )
    )

    for row in selected_masks_150:
        candidate_rows.append(
            {
                "candidate_type": "mask_observation",
                "node_id": row.get("node_id"),
                "mask_observation_id": row.get("mask_observation_id"),
                "component_id": _component_id(row),
                "supporting_unique_carrier_count": row.get("supporting_unique_carrier_count"),
                "supporting_carrier_observation_count": row.get("supporting_carrier_observation_count"),
                "selected_in": "F1_d4rt_native_mask_set_cover_budget150",
                "uses_gt_for_prediction": False,
            }
        )
    for row in _selected_components(mask_vote_rows, None):
        candidate_rows.append(
            {
                "candidate_type": "component_support",
                "component_id": row.get("component_id"),
                "mask_count": row.get("mask_count"),
                "supporting_unique_carrier_count": row.get("supporting_unique_carrier_count"),
                "supporting_carrier_observation_count": row.get("supporting_carrier_observation_count"),
                "selected_in": "F2/F6 component variants",
                "uses_gt_for_prediction": False,
            }
        )

    best = max(rows, key=lambda row: parse_float(row.get("ARI")))
    f6_rows = [row for row in rows if str(row.get("variant")).startswith("F6")]
    gate = {
        "pass": any(parse_bool(row.get("gate_pass")) for row in f6_rows),
        "failure_label": None,
        "best_variant": best["variant"],
        "best_ARI": best["ARI"],
        "best_purity": best["purity"],
        "best_completeness": best["completeness"],
        "component_baseline_ARI": baseline["ARI"],
        "component_baseline_purity": baseline["purity"],
        "component_baseline_completeness": baseline["completeness"],
        "underseg_false_merge_reduction": underseg_reduction,
        "mask_atom_rows_domain": "tracklet_reactivation_diagnostic",
    }
    if not gate["pass"]:
        gate["failure_label"] = "NO_GO_MASK_ATOM"

    payload = {
        "phase": "v48_mask_set_cover_atoms",
        "created_at": utc_now(),
        "total_real_components": total_components,
        "mask_vote_row_count": len(mask_vote_rows),
        "shared_observation_count": len(shared_rows),
        "mask_atom_count": mask_atom_count,
        "atoms_per_mask_mean": atoms_per_mask_mean,
        "rows": rows,
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "limitations": [
            "Current v47 artifacts expose mask-to-component aggregate votes, not per-carrier component membership.",
            "Tracklet-domain underseg atom diagnostics are not promoted into method prediction or used as GT-informed object birth.",
        ],
    }
    out = ROOT / str(args.output_root)
    write_json(out / "mask_set_cover_atoms_summary.json", payload)
    write_csv(out / "mask_set_cover_atoms_rows.csv", rows)
    write_csv(out / "mask_set_cover_candidate_rows.csv", candidate_rows)
    write_csv(out / "mask_atom_support_rows.csv", shared_rows)
    write_json(out / "mask_set_cover_config.json", vars(args))
    print({"summary": str(out / "mask_set_cover_atoms_summary.json"), "gate": gate})


if __name__ == "__main__":
    main()
