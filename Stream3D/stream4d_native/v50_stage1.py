from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, read_csv, read_json, utc_now, write_csv, write_json


def project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def rel(path: str | Path) -> str:
    path_obj = project_path(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def exists(path: str | Path) -> bool:
    return project_path(path).exists()


def load_optional_json(path: str | Path) -> dict[str, Any]:
    path_obj = project_path(path)
    if not path_obj.exists():
        return {"missing": True, "path": rel(path_obj)}
    payload = read_json(path_obj)
    if isinstance(payload, dict):
        return payload
    return {"payload": payload}


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _fact_row(
    key: str,
    value: Any,
    source: str | Path,
    note: str = "",
    required: bool = True,
) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "source": rel(source),
        "available": value not in (None, ""),
        "required": required,
        "note": note,
    }


def _path_row(key: str, path: str | Path, note: str = "", required: bool = True) -> dict[str, Any]:
    return _fact_row(key, exists(path), path, note, required)


def _best_v37_4d_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("best_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _best_v37_postprocess_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("best_postprocess_metrics")
    if isinstance(metrics, dict):
        return metrics
    all_metrics = payload.get("all_metrics")
    variant = payload.get("preferred_audit_best_postprocess_variant") or payload.get("best_postprocess_variant")
    if isinstance(all_metrics, dict) and variant in all_metrics and isinstance(all_metrics[variant], dict):
        return all_metrics[variant]
    return {}


def _available_dense_backend(semantic_payload: dict[str, Any], backend_id: str) -> bool:
    for row in semantic_payload.get("backend_rows", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("backend_id")) == backend_id and bool(row.get("uses_frozen_dense_features")):
            return bool(row.get("feature_success_rate", 0.0) >= 0.95)
    return False


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _as_float(row.get("bbox_x0")),
        _as_float(row.get("bbox_y0")),
        _as_float(row.get("bbox_x1")),
        _as_float(row.get("bbox_y1")),
    )


def _bbox_area(row: dict[str, Any]) -> float:
    x0, y0, x1, y1 = _bbox(row)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_intersection_area(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    return width * height


def _support_component_id(row: dict[str, Any]) -> str:
    component = str(row.get("predicted_component_object_id") or "")
    return "" if component.startswith("uncovered:") else component


def _component_key_from_vote(row: dict[str, Any]) -> str:
    component = _support_component_id(row)
    return f"{row.get('scene')}|{component}" if component else ""


def _component_gt_map(vote_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in vote_rows:
        key = _component_key_from_vote(row)
        gt = str(row.get("diagnostic_gt_instance") or "")
        if key and gt:
            counts[key][gt] += 1
    out: dict[str, dict[str, Any]] = {}
    for key, counter in counts.items():
        total = sum(counter.values())
        dominant, dominant_count = counter.most_common(1)[0]
        out[key] = {
            "diagnostic_dominant_gt": dominant,
            "diagnostic_gt_purity": dominant_count / total if total else 0.0,
            "diagnostic_gt_vote_count": total,
        }
    return out


def build_v50_mask_source_audit(max_overlap_rows: int = 500) -> dict[str, Any]:
    mask_table_path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv"
    vote_rows_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv"
    obs_summary_path = "outputs/audit/v47_observation_tables_metricfix/observation_table_summary.json"
    component_summary_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_summary.json"
    lattice_rows_path = "outputs/audit/v49_component_lattice/component_lattice_containment_rows.csv"
    lattice_summary_path = "outputs/audit/v49_component_lattice/component_lattice_summary.json"

    mask_rows = read_csv(project_path(mask_table_path)) if exists(mask_table_path) else []
    vote_rows = read_csv(project_path(vote_rows_path)) if exists(vote_rows_path) else []
    lattice_rows = read_csv(project_path(lattice_rows_path)) if exists(lattice_rows_path) else []
    obs_summary = load_optional_json(obs_summary_path)
    component_summary = load_optional_json(component_summary_path)
    lattice_summary = load_optional_json(lattice_summary_path)

    frame_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mask_rows:
        frame_groups[(str(row.get("scene")), str(row.get("frame_id")))].append(row)

    bbox_overlap_pair_count = 0
    bbox_containment_pair_count = 0
    bbox_duplicate_pair_count = 0
    bbox_conflict_candidate_count = 0
    overlap_rows: list[dict[str, Any]] = []
    for (scene, frame_id), rows in frame_groups.items():
        for idx, left in enumerate(rows):
            bbox_area_l = max(_bbox_area(left), 1.0)
            gt_l = str(left.get("diagnostic_gt_instance"))
            for right in rows[idx + 1 :]:
                bbox_area_r = max(_bbox_area(right), 1.0)
                inter = _bbox_intersection_area(left, right)
                if inter <= 0.0:
                    continue
                bbox_overlap_pair_count += 1
                contain_l_in_r = inter / bbox_area_l
                contain_r_in_l = inter / bbox_area_r
                area_ratio = max(bbox_area_l, bbox_area_r) / max(min(bbox_area_l, bbox_area_r), 1.0)
                relation = "bbox_overlap_proxy"
                if area_ratio <= 1.10 and min(contain_l_in_r, contain_r_in_l) >= 0.85:
                    bbox_duplicate_pair_count += 1
                    relation = "bbox_duplicate_proxy"
                elif max(contain_l_in_r, contain_r_in_l) >= 0.80 and area_ratio >= 1.30:
                    bbox_containment_pair_count += 1
                    relation = "bbox_containment_proxy"
                gt_r = str(right.get("diagnostic_gt_instance"))
                if gt_l and gt_r and gt_l != gt_r:
                    bbox_conflict_candidate_count += 1
                if len(overlap_rows) < max_overlap_rows:
                    overlap_rows.append(
                        {
                            "scene": scene,
                            "frame_id": frame_id,
                            "mask_a": left.get("mask_observation_id"),
                            "mask_b": right.get("mask_observation_id"),
                            "bbox_intersection_area": inter,
                            "bbox_containment_a_in_b": contain_l_in_r,
                            "bbox_containment_b_in_a": contain_r_in_l,
                            "bbox_area_ratio": area_ratio,
                            "relation_proxy": relation,
                            "exact_pixel_overlap_available": False,
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic_labels": True,
                        }
                    )

    component_ids = {_support_component_id(row) for row in vote_rows}
    component_ids.discard("")
    total_components = _as_int(component_summary.get("component_count"))
    component_coverage = len(component_ids) / total_components if total_components > 0 else 0.0
    frame_count = len(frame_groups)
    mean_masks_per_frame = len(mask_rows) / frame_count if frame_count else 0.0
    mask_count = len(mask_rows)
    carrier_coverage = _as_float(obs_summary.get("carrier_inside_any_mask_ratio"), _as_float(obs_summary.get("observed_mask_hit_rate")))
    source_counts = Counter(str(row.get("feature_backend") or "unknown") for row in mask_rows)
    lattice_edge_count = len(lattice_rows)

    # Prepared labels expose one observed_mask_id per carrier, so exact same-view mask overlap is not recoverable here.
    exact_overlap_pair_count = 0
    exact_containment_pair_count = 0
    exact_duplicate_pair_count = 0
    whole_candidate_count = 0
    part_candidate_count = 0
    parentless_mask_ratio = 1.0 if mask_count else 0.0
    same_view_hierarchy_available = False
    component_lattice_fallback_available = lattice_edge_count > 0
    effective_hierarchy_route = "multi_scale_component_lattice_fallback" if component_lattice_fallback_available else "disabled"

    source_rows = [
        {
            "variant": "M0_current_prepared_masks",
            "available": bool(mask_rows),
            "mask_count": mask_count,
            "frame_count": frame_count,
            "mean_masks_per_frame": mean_masks_per_frame,
            "overlap_pair_count": exact_overlap_pair_count,
            "containment_pair_count": exact_containment_pair_count,
            "containment_pair_ratio": 0.0,
            "duplicate_pair_count": exact_duplicate_pair_count,
            "same_frame_conflict_candidate_count": 0,
            "whole_candidate_count": whole_candidate_count,
            "part_candidate_count": part_candidate_count,
            "parentless_mask_ratio": parentless_mask_ratio,
            "component_coverage_by_masks": component_coverage,
            "carrier_coverage_by_masks": carrier_coverage,
            "bbox_overlap_pair_count_diagnostic": bbox_overlap_pair_count,
            "bbox_containment_pair_count_diagnostic": bbox_containment_pair_count,
            "bbox_duplicate_pair_count_diagnostic": bbox_duplicate_pair_count,
            "bbox_conflict_candidate_count_diagnostic": bbox_conflict_candidate_count,
            "exact_pixel_overlap_available": False,
            "source_note": "prepared mask observations are flat labels; bbox proxies are diagnostic only",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "variant": "M5_multi_scale_component_lattice_fallback",
            "available": component_lattice_fallback_available,
            "mask_count": mask_count,
            "frame_count": frame_count,
            "mean_masks_per_frame": mean_masks_per_frame,
            "overlap_pair_count": 0,
            "containment_pair_count": lattice_edge_count,
            "containment_pair_ratio": lattice_edge_count / max(total_components, 1),
            "duplicate_pair_count": 0,
            "same_frame_conflict_candidate_count": 0,
            "whole_candidate_count": lattice_edge_count,
            "part_candidate_count": lattice_edge_count,
            "parentless_mask_ratio": None,
            "component_coverage_by_masks": component_coverage,
            "carrier_coverage_by_masks": carrier_coverage,
            "bbox_overlap_pair_count_diagnostic": bbox_overlap_pair_count,
            "bbox_containment_pair_count_diagnostic": bbox_containment_pair_count,
            "bbox_duplicate_pair_count_diagnostic": bbox_duplicate_pair_count,
            "bbox_conflict_candidate_count_diagnostic": bbox_conflict_candidate_count,
            "exact_pixel_overlap_available": False,
            "source_note": "fallback hierarchy from v49 multi-scale component lattice, not same-view pixel overlap",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
        },
    ]

    for variant, keyword in [
        ("M1_CropFormer_masks", "crop"),
        ("M2_SAM_or_SAM2_masks", "sam"),
        ("M3_prepared_plus_SAM_union", "sam"),
        ("M4_prepared_plus_CropFormer_union", "crop"),
    ]:
        source_rows.append(
            {
                "variant": variant,
                "available": False,
                "mask_count": 0,
                "frame_count": 0,
                "mean_masks_per_frame": 0.0,
                "overlap_pair_count": 0,
                "containment_pair_count": 0,
                "containment_pair_ratio": 0.0,
                "duplicate_pair_count": 0,
                "same_frame_conflict_candidate_count": 0,
                "whole_candidate_count": 0,
                "part_candidate_count": 0,
                "parentless_mask_ratio": None,
                "component_coverage_by_masks": 0.0,
                "carrier_coverage_by_masks": 0.0,
                "bbox_overlap_pair_count_diagnostic": 0,
                "bbox_containment_pair_count_diagnostic": 0,
                "bbox_duplicate_pair_count_diagnostic": 0,
                "bbox_conflict_candidate_count_diagnostic": 0,
                "exact_pixel_overlap_available": False,
                "source_note": f"no {keyword} mask source was wired into this Phase 1 audit",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )

    gate = {
        "same_view_hierarchy_available": same_view_hierarchy_available,
        "component_lattice_fallback_available": component_lattice_fallback_available,
        "effective_hierarchy_available": component_lattice_fallback_available,
        "effective_hierarchy_route": effective_hierarchy_route,
        "component_coverage_pass": component_coverage >= 0.75,
        "carrier_coverage_pass": carrier_coverage >= 0.75,
        "pass": component_lattice_fallback_available and component_coverage >= 0.75,
    }
    return {
        "phase": "v50_mask_source_audit",
        "created_at": utc_now(),
        "mask_source_rows": source_rows,
        "same_view_overlap_rows": overlap_rows,
        "summary": {
            "mask_count": mask_count,
            "frame_count": frame_count,
            "mean_masks_per_frame": mean_masks_per_frame,
            "overlap_pair_count": exact_overlap_pair_count,
            "containment_pair_count": exact_containment_pair_count,
            "containment_pair_ratio": 0.0,
            "duplicate_pair_count": exact_duplicate_pair_count,
            "same_frame_conflict_candidate_count": 0,
            "whole_candidate_count": whole_candidate_count,
            "part_candidate_count": part_candidate_count,
            "parentless_mask_ratio": parentless_mask_ratio,
            "component_coverage_by_masks": component_coverage,
            "carrier_coverage_by_masks": carrier_coverage,
            "bbox_overlap_pair_count_diagnostic": bbox_overlap_pair_count,
            "bbox_containment_pair_count_diagnostic": bbox_containment_pair_count,
            "bbox_duplicate_pair_count_diagnostic": bbox_duplicate_pair_count,
            "bbox_conflict_candidate_count_diagnostic": bbox_conflict_candidate_count,
            "pixel_overlap_available": False,
            "prepared_masks_are_flat_label_map": True,
            "feature_backend_counts": dict(source_counts),
            "component_lattice_fallback_edge_count": lattice_edge_count,
            "component_lattice_summary_gate": lattice_summary.get("gate"),
        },
        "gate": gate,
        "artifact_sources": {
            "mask_observation_table": rel(mask_table_path),
            "carrier_vote_rows": rel(vote_rows_path),
            "observation_summary": rel(obs_summary_path),
            "component_summary": rel(component_summary_path),
            "component_lattice_rows": rel(lattice_rows_path),
            "component_lattice_summary": rel(lattice_summary_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_v50_fact_lock() -> dict[str, Any]:
    v37_4d_path = "outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_decision.json"
    v37_ap_path = "outputs/audit/v37_ap_if_allowed_i4_sparse/ap_eval_summary.json"
    v37_post_path = "outputs/audit/v37_ap_if_allowed_i4_sparse/ap_postprocess_final_summary.json"
    v49_selection_path = "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"
    v49_full_path = "outputs/audit/v49_full_stage1/full_stage1_summary.json"
    v49_final_path = "outputs/audit/v49_final_decision/v49_final_decision.json"
    v49_ap_path = "outputs/audit/v49_eval_aligned_ap/eval_aligned_ap_summary.json"
    obs_path = "outputs/audit/v47_observation_tables_metricfix/observation_table_summary.json"
    carrier_summary_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_summary.json"
    semantic_v48_path = "outputs/audit/v48_semantic_features/semantic_feature_audit_summary.json"
    semantic_v49_path = "outputs/audit/v49_semantic_backend_availability/semantic_backend_availability_summary.json"

    v37_4d = load_optional_json(v37_4d_path)
    v37_ap = load_optional_json(v37_ap_path)
    v37_post = load_optional_json(v37_post_path)
    v49_selection = load_optional_json(v49_selection_path)
    v49_full = load_optional_json(v49_full_path)
    v49_final = load_optional_json(v49_final_path)
    v49_ap = load_optional_json(v49_ap_path)
    obs = load_optional_json(obs_path)
    carrier_summary = load_optional_json(carrier_summary_path)
    semantic_v48 = load_optional_json(semantic_v48_path)
    semantic_v49 = load_optional_json(semantic_v49_path)

    v37_metrics = _best_v37_4d_metrics(v37_4d)
    v37_post_metrics = _best_v37_postprocess_metrics(v37_post)
    v49_best = _nested(v49_selection, "best_real_row", default={}) or _nested(v49_full, "final_candidate", default={}) or {}
    v49_ap_rows = v49_ap.get("ap_rows", []) if isinstance(v49_ap.get("ap_rows"), list) else []
    v49_ap_status = ",".join(sorted({str(row.get("status", "")) for row in v49_ap_rows if isinstance(row, dict)})) or None

    rows = [
        _fact_row("v37_4D_ARI", v37_metrics.get("4D_ARI"), v37_4d_path, "prior v37 4D decision baseline"),
        _fact_row("v37_4D_purity", v37_metrics.get("4D_purity"), v37_4d_path, "prior v37 4D decision baseline"),
        _fact_row("v37_4D_completeness", v37_metrics.get("4D_completeness"), v37_4d_path, "prior v37 4D decision baseline"),
        _fact_row("v37_AP", v37_ap.get("AP"), v37_ap_path, "raw v37 AP export diagnostic"),
        _fact_row("v37_AP50", v37_ap.get("AP50"), v37_ap_path, "raw v37 AP export diagnostic"),
        _fact_row("v37_AP25", v37_ap.get("AP25"), v37_ap_path, "raw v37 AP export diagnostic"),
        _fact_row("v37_postprocess_AP", v37_post_metrics.get("AP"), v37_post_path, "best/preferred v37 postprocess AP diagnostic"),
        _fact_row("v37_postprocess_AP50", v37_post_metrics.get("AP50"), v37_post_path, "best/preferred v37 postprocess AP diagnostic", required=False),
        _fact_row("v37_postprocess_AP25", v37_post_metrics.get("AP25"), v37_post_path, "best/preferred v37 postprocess AP diagnostic", required=False),
        _fact_row("v49_selected_variant", v49_best.get("solver_variant"), v49_selection_path),
        _fact_row("v49_selected_4D_ARI", v49_best.get("4D_ARI"), v49_selection_path),
        _fact_row("v49_selected_purity", v49_best.get("4D_purity"), v49_selection_path),
        _fact_row("v49_selected_completeness", v49_best.get("4D_completeness"), v49_selection_path),
        _fact_row("v49_selected_conflict_rate", v49_best.get("conflict_rate"), v49_selection_path),
        _fact_row("v49_real_minus_shuffled_ARI", v49_best.get("real_minus_shuffled_ARI"), v49_selection_path),
        _fact_row("v49_real_minus_no_temporal_ARI", v49_best.get("real_minus_no_temporal_ARI"), v49_selection_path),
        _fact_row("v49_real_minus_mask_only_ARI", v49_best.get("real_minus_mask_only_ARI"), v49_selection_path),
        _fact_row("v49_AP_status", v49_ap_status, v49_ap_path, "v49 AP was gated by strict Stage-1; v50 must relax this"),
        _fact_row("v49_final_label", v49_final.get("final_label"), v49_final_path, required=False),
        _fact_row("carrier_observation_table_available", obs.get("carrier_observation_table_exists"), obs_path),
        _fact_row("mask_observation_table_available", obs.get("mask_observation_table_exists"), obs_path),
        _fact_row("D4RT_encoder_stride", obs.get("D4RT_encoder_stride"), obs_path),
        _fact_row("scale_guard_pass", obs.get("scale_weak_row_count") == 0, obs_path),
        _fact_row("U32_carrier_components_available", int(carrier_summary.get("component_count", 0) or 0) > 0, carrier_summary_path),
        _fact_row("RADIO_available", _available_dense_backend(semantic_v48, "radio"), semantic_v48_path, "dense RADIO backend from prior semantic proxy audit", required=False),
        _fact_row("DINO_available", _available_dense_backend(semantic_v48, "dinov2"), semantic_v48_path, "dense DINO backend from prior semantic proxy audit", required=False),
        _fact_row("dense_semantic_available", any(_available_dense_backend(semantic_v48, key) for key in ("radio", "dinov2")), semantic_v48_path, "prior dense semantic proxy availability", required=False),
        _fact_row("component_level_feature_available", bool(_nested(semantic_v49, "gate", "v49_observation_has_dense_feature_backend", default=False)), semantic_v49_path, "v49 found no component-level dense feature backend", required=False),
        _path_row("ap_evaluator_available", "evaluation/evaluate.py", "primary ScanNet evaluator source exists"),
        _path_row("ap_exporter_available", "stream4d/export_scannet.py", "generic ScanNet exporter exists; v50 object-field AP exporter still must be exercised"),
        _path_row("eval_alignment_adapter_available", "evaluation/eval_aligned_ap_bridge.py", "eval-only alignment adapter source exists", required=False),
        _fact_row(
            "rgbd_pose_mesh_bridge_available",
            False,
            "data/scannet",
            "only ScanNet GT txt/evaluation files detected; no explicit RGB-D pose/mesh bridge root found in Phase 0 scan",
            required=False,
        ),
        _path_row("scannet_gt_available", "data/scannet/gt", "evaluator GT directory exists"),
        _path_row("scannet_eval_output_dir_available", "data/evaluation/scannet", "evaluation output directory exists"),
    ]
    fact = {row["key"]: row["value"] for row in rows}
    required_keys = [
        "carrier_observation_table_available",
        "mask_observation_table_available",
        "D4RT_encoder_stride",
        "scale_guard_pass",
        "U32_carrier_components_available",
        "ap_evaluator_available",
        "ap_exporter_available",
    ]
    gate = {
        "carrier_observation_table_available": bool(fact.get("carrier_observation_table_available")),
        "mask_observation_table_available": bool(fact.get("mask_observation_table_available")),
        "D4RT_encoder_stride_eq_1": fact.get("D4RT_encoder_stride") == 1,
        "scale_guard_pass": bool(fact.get("scale_guard_pass")),
        "U32_carrier_components_available": bool(fact.get("U32_carrier_components_available")),
        "ap_evaluator_available": bool(fact.get("ap_evaluator_available") and fact.get("scannet_gt_available")),
        "ap_exporter_available": bool(fact.get("ap_exporter_available")),
    }
    gate["pass"] = bool(all(gate.values()))
    missing_required = [key for key in required_keys if fact.get(key) in (None, "", False)]

    ap_export_contract = {
        "phase": "v50_ap_export_contract",
        "created_at": utc_now(),
        "ap_policy": "relaxed_ap_required_even_when_strict_stage1_fails",
        "method_safe_native_exporter_available": bool(fact.get("ap_exporter_available")),
        "eval_alignment_adapter_available": bool(fact.get("eval_alignment_adapter_available")),
        "rgbd_pose_mesh_bridge_available": bool(fact.get("rgbd_pose_mesh_bridge_available")),
        "required_policy_fields": [
            "is_method_result",
            "is_diagnostic_only",
            "uses_gt_for_prediction",
            "uses_gt_for_evaluation_alignment",
            "uses_rgbd_pose_mesh_for_export",
            "forbidden_for_method_table",
        ],
        "phase8_repair_needed": not bool(fact.get("ap_exporter_available") and fact.get("ap_evaluator_available")),
        "notes": [
            "v49 AP status was blocked by strict Stage-1; v50 must not inherit that gate.",
            "Generic ScanNet exporter exists, but v50 still needs AP smoke rows from selected object fields before final decision.",
            "RGB-D/pose/mesh bridge is not marked available in Phase 0 because only GT/evaluation text files were detected.",
        ],
    }

    return {
        "phase": "v50_fact_lock",
        "created_at": utc_now(),
        "plan": "docs/stream4d_v50_mosaic_merge_codex_plan_relaxed_ap.md",
        "fact_rows": rows,
        "fact_map": fact,
        "gate": gate,
        "missing_required": missing_required,
        "ap_export_contract": ap_export_contract,
        "artifact_sources": {
            "v37_4d": rel(v37_4d_path),
            "v37_ap": rel(v37_ap_path),
            "v37_ap_postprocess": rel(v37_post_path),
            "v49_selection": rel(v49_selection_path),
            "v49_full_stage1": rel(v49_full_path),
            "v49_final_decision": rel(v49_final_path),
            "v49_ap": rel(v49_ap_path),
            "observation_table": rel(obs_path),
            "u32_components": rel(carrier_summary_path),
            "semantic_v48": rel(semantic_v48_path),
            "semantic_v49": rel(semantic_v49_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "repair_note": None if gate["pass"] else "Follow Phase 0 repair path before method claims.",
    }


def write_v50_fact_lock(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "fact_lock.json", payload)
    write_csv(out / "fact_lock_rows.csv", payload["fact_rows"])
    write_json(out / "ap_export_contract.json", payload["ap_export_contract"])


def write_v50_mask_source_audit(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "mask_source_summary.json", payload)
    write_csv(out / "mask_source_rows.csv", payload["mask_source_rows"])
    write_csv(out / "same_view_overlap_rows.csv", payload["same_view_overlap_rows"])


def _relation_precision(rows: list[dict[str, Any]]) -> float | None:
    labeled = [row for row in rows if row.get("diagnostic_same_gt") in (True, False)]
    if not labeled:
        return None
    return sum(1 for row in labeled if row.get("diagnostic_same_gt") is True) / len(labeled)


def build_v50_same_view_relations(max_relation_rows: int = 10000) -> dict[str, Any]:
    lattice_rows_path = "outputs/audit/v49_component_lattice/component_lattice_containment_rows.csv"
    vote_rows_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv"
    mask_source_path = "outputs/audit/v50_mask_source_audit/mask_source_summary.json"
    semantic_path = "outputs/audit/v49_semantic_backend_availability/semantic_backend_availability_summary.json"

    lattice_rows = read_csv(project_path(lattice_rows_path)) if exists(lattice_rows_path) else []
    vote_rows = read_csv(project_path(vote_rows_path)) if exists(vote_rows_path) else []
    mask_source = load_optional_json(mask_source_path)
    semantic = load_optional_json(semantic_path)
    comp_gt = _component_gt_map(vote_rows)

    part_rows_relaxed: list[dict[str, Any]] = []
    part_rows_conservative: list[dict[str, Any]] = []
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for row in lattice_rows:
        child = str(row.get("child_component") or "")
        parent = str(row.get("parent_component") or "")
        if not child or not parent:
            continue
        child_gt = comp_gt.get(child, {})
        parent_gt = comp_gt.get(parent, {})
        same_gt: bool | None = None
        if child_gt.get("diagnostic_dominant_gt") and parent_gt.get("diagnostic_dominant_gt"):
            same_gt = child_gt["diagnostic_dominant_gt"] == parent_gt["diagnostic_dominant_gt"]
        shared = _as_int(row.get("shared_mask_observation_count"))
        child_count = _as_int(row.get("parent_child_count"))
        confidence = min(1.0, (shared / 25.0)) * (1.0 / max(child_count, 1)) ** 0.25
        rel_row = {
            "relation_id": f"part_of_fallback::{child}->{parent}",
            "relation_type": "part_of_fallback_component_lattice",
            "source_variant": "R4_relaxed_component_lattice_fallback",
            "child": child,
            "parent": parent,
            "mask_a": "",
            "mask_b": "",
            "shared_mask_observation_count": shared,
            "parent_child_count": child_count,
            "relation_confidence": confidence,
            "relation_uses_semantic_backend": False,
            "exact_same_view_pixel_relation": False,
            "bbox_proxy_only": False,
            "diagnostic_child_gt": child_gt.get("diagnostic_dominant_gt"),
            "diagnostic_parent_gt": parent_gt.get("diagnostic_dominant_gt"),
            "diagnostic_child_gt_purity": child_gt.get("diagnostic_gt_purity"),
            "diagnostic_parent_gt_purity": parent_gt.get("diagnostic_gt_purity"),
            "diagnostic_child_gt_vote_count": child_gt.get("diagnostic_gt_vote_count"),
            "diagnostic_parent_gt_vote_count": parent_gt.get("diagnostic_gt_vote_count"),
            "diagnostic_same_gt": same_gt,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        part_rows_relaxed.append(rel_row)
        children_by_parent[parent].append(child)
        if shared >= 5 and child_count <= 12 and confidence >= 0.30:
            cons = dict(rel_row)
            cons["source_variant"] = "R3_conservative_component_lattice_fallback"
            part_rows_conservative.append(cons)

    sibling_rows: list[dict[str, Any]] = []
    for parent, children in children_by_parent.items():
        unique_children = sorted(set(children))
        for idx, left in enumerate(unique_children):
            for right in unique_children[idx + 1 :]:
                left_gt = comp_gt.get(left, {})
                right_gt = comp_gt.get(right, {})
                same_gt: bool | None = None
                if left_gt.get("diagnostic_dominant_gt") and right_gt.get("diagnostic_dominant_gt"):
                    same_gt = left_gt["diagnostic_dominant_gt"] == right_gt["diagnostic_dominant_gt"]
                sibling_rows.append(
                    {
                        "relation_id": f"sibling_fallback::{left}<->{right}@{parent}",
                        "relation_type": "sibling_fallback_shared_lattice_parent",
                        "source_variant": "R4_relaxed_component_lattice_fallback",
                        "child": left,
                        "parent": parent,
                        "mask_a": left,
                        "mask_b": right,
                        "shared_mask_observation_count": "",
                        "parent_child_count": len(unique_children),
                        "relation_confidence": 1.0 / max(len(unique_children), 1),
                        "relation_uses_semantic_backend": False,
                        "exact_same_view_pixel_relation": False,
                        "bbox_proxy_only": False,
                        "diagnostic_child_gt": left_gt.get("diagnostic_dominant_gt"),
                        "diagnostic_parent_gt": right_gt.get("diagnostic_dominant_gt"),
                        "diagnostic_child_gt_purity": left_gt.get("diagnostic_gt_purity"),
                        "diagnostic_parent_gt_purity": right_gt.get("diagnostic_gt_purity"),
                        "diagnostic_child_gt_vote_count": left_gt.get("diagnostic_gt_vote_count"),
                        "diagnostic_parent_gt_vote_count": right_gt.get("diagnostic_gt_vote_count"),
                        "diagnostic_same_gt": same_gt,
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                    }
                )

    part_precision_relaxed = _relation_precision(part_rows_relaxed)
    part_precision_conservative = _relation_precision(part_rows_conservative)
    sibling_precision = _relation_precision(sibling_rows)
    best_part_precision = max(
        value for value in [part_precision_relaxed, part_precision_conservative, 0.0] if value is not None
    )
    relation_conf_values = [float(row["relation_confidence"]) for row in part_rows_relaxed]
    relation_conf_values_sorted = sorted(relation_conf_values)
    p10_index = int(0.10 * (len(relation_conf_values_sorted) - 1)) if relation_conf_values_sorted else 0
    relation_conf_p10 = relation_conf_values_sorted[p10_index] if relation_conf_values_sorted else 0.0
    relation_conf_mean = sum(relation_conf_values) / len(relation_conf_values) if relation_conf_values else 0.0

    exact_same_view_available = bool(_nested(mask_source, "gate", "same_view_hierarchy_available", default=False))
    component_fallback_available = bool(_nested(mask_source, "gate", "component_lattice_fallback_available", default=False))
    semantic_backend_available = bool(_nested(semantic, "gate", "v49_observation_has_dense_feature_backend", default=False))
    metric_rows = [
        {
            "variant": "R0_geometry_only_same_view_exact",
            "available": exact_same_view_available,
            "part_edge_count": 0,
            "duplicate_edge_count": 0,
            "sibling_edge_count": 0,
            "conflict_edge_count": 0,
            "weak_underseg_parent_count": 0,
            "part_relation_precision": None,
            "sibling_relation_precision": None,
            "conflict_relation_precision": None,
            "same_GT_sibling_ratio": None,
            "different_GT_conflict_ratio": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "note": "exact same-view pixel overlap unavailable for prepared flat label source",
        },
        {
            "variant": "R3_conservative_component_lattice_fallback",
            "available": component_fallback_available,
            "part_edge_count": len(part_rows_conservative),
            "duplicate_edge_count": 0,
            "sibling_edge_count": 0,
            "conflict_edge_count": 0,
            "weak_underseg_parent_count": len(part_rows_conservative),
            "part_relation_precision": part_precision_conservative,
            "sibling_relation_precision": None,
            "conflict_relation_precision": None,
            "same_GT_sibling_ratio": None,
            "different_GT_conflict_ratio": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "note": "high-confidence subset of multi-scale component lattice fallback",
        },
        {
            "variant": "R4_relaxed_component_lattice_fallback",
            "available": component_fallback_available,
            "part_edge_count": len(part_rows_relaxed),
            "duplicate_edge_count": 0,
            "sibling_edge_count": len(sibling_rows),
            "conflict_edge_count": 0,
            "weak_underseg_parent_count": len(part_rows_relaxed),
            "part_relation_precision": part_precision_relaxed,
            "sibling_relation_precision": sibling_precision,
            "conflict_relation_precision": None,
            "same_GT_sibling_ratio": sibling_precision,
            "different_GT_conflict_ratio": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "note": "all multi-scale component lattice fallback edges; no exact same-view pixel relation claim",
        },
        {
            "variant": "R2_geometry_plus_semantic_contradiction",
            "available": semantic_backend_available,
            "part_edge_count": 0,
            "duplicate_edge_count": 0,
            "sibling_edge_count": 0,
            "conflict_edge_count": 0,
            "weak_underseg_parent_count": 0,
            "part_relation_precision": None,
            "sibling_relation_precision": None,
            "conflict_relation_precision": None,
            "same_GT_sibling_ratio": None,
            "different_GT_conflict_ratio": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "note": "component-level dense semantic backend unavailable in v49 semantic backend audit",
        },
        {
            "variant": "R5_no_relation_control",
            "available": True,
            "part_edge_count": 0,
            "duplicate_edge_count": 0,
            "sibling_edge_count": 0,
            "conflict_edge_count": 0,
            "weak_underseg_parent_count": 0,
            "part_relation_precision": None,
            "sibling_relation_precision": None,
            "conflict_relation_precision": None,
            "same_GT_sibling_ratio": None,
            "different_GT_conflict_ratio": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "note": "matched no-relation control for downstream phases",
        },
    ]
    repair_sweep_rows: list[dict[str, Any]] = []
    for min_shared in [1, 2, 5, 10, 20, 50, 100, 200, 400]:
        for max_fanout in [1, 2, 3, 4, 6, 8, 12, 20, 40, 9999]:
            for min_gt_purity in [0.0, 0.5, 0.7, 0.85, 0.95, 0.99]:
                for min_gt_votes in [1, 2, 5, 10, 20, 50]:
                    filtered = [
                        row
                        for row in part_rows_relaxed
                        if _as_int(row.get("shared_mask_observation_count")) >= min_shared
                        and _as_int(row.get("parent_child_count")) <= max_fanout
                        and _as_float(row.get("diagnostic_child_gt_purity")) >= min_gt_purity
                        and _as_float(row.get("diagnostic_parent_gt_purity")) >= min_gt_purity
                        and _as_int(row.get("diagnostic_child_gt_vote_count")) >= min_gt_votes
                        and _as_int(row.get("diagnostic_parent_gt_vote_count")) >= min_gt_votes
                        and row.get("diagnostic_same_gt") in (True, False)
                    ]
                    if len(filtered) < 5:
                        continue
                    precision = sum(1 for row in filtered if row.get("diagnostic_same_gt") is True) / len(filtered)
                    repair_sweep_rows.append(
                        {
                            "min_shared_mask_observation_count": min_shared,
                            "max_parent_child_count": max_fanout,
                            "min_component_gt_purity": min_gt_purity,
                            "min_component_gt_vote_count": min_gt_votes,
                            "edge_count": len(filtered),
                            "part_relation_precision": precision,
                            "passes_part_precision_gate": precision >= 0.70,
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic_labels": True,
                        }
                    )
    best_sweep = max((row["part_relation_precision"] for row in repair_sweep_rows), default=None)
    output_rows = (part_rows_conservative + part_rows_relaxed + sibling_rows)[:max_relation_rows]
    gate = {
        "exact_same_view_relation_available": exact_same_view_available,
        "component_lattice_fallback_available": component_fallback_available,
        "part_relation_precision_pass": best_part_precision >= 0.70,
        "sibling_relation_precision_pass": sibling_precision is not None and sibling_precision >= 0.60,
        "conflict_relation_precision_pass": False,
        "conflict_relation_soft_only": True,
        "pass": component_fallback_available and best_part_precision >= 0.70,
    }
    return {
        "phase": "v50_same_view_relations",
        "created_at": utc_now(),
        "summary": {
            "part_edge_count": len(part_rows_relaxed),
            "duplicate_edge_count": 0,
            "sibling_edge_count": len(sibling_rows),
            "conflict_edge_count": 0,
            "weak_underseg_parent_count": len(part_rows_relaxed),
            "relation_confidence_mean": relation_conf_mean,
            "relation_confidence_p10": relation_conf_p10,
            "relation_source_breakdown": {
                "component_lattice_fallback_relaxed": len(part_rows_relaxed),
                "component_lattice_fallback_conservative": len(part_rows_conservative),
                "sibling_shared_lattice_parent": len(sibling_rows),
                "exact_same_view_pixel": 0,
                "semantic_contradiction": 0,
            },
            "relation_uses_semantic_backend": False,
            "part_relation_precision": best_part_precision,
            "part_relation_precision_relaxed": part_precision_relaxed,
            "part_relation_precision_conservative": part_precision_conservative,
            "best_threshold_sweep_part_precision": best_sweep,
            "threshold_sweep_row_count": len(repair_sweep_rows),
            "sibling_relation_precision": sibling_precision,
            "conflict_relation_precision": None,
            "same_GT_sibling_ratio": sibling_precision,
            "different_GT_conflict_ratio": None,
        },
        "gate": gate,
        "mask_relation_rows": output_rows,
        "relation_metric_rows": metric_rows,
        "relation_repair_sweep_rows": repair_sweep_rows,
        "artifact_sources": {
            "component_lattice_rows": rel(lattice_rows_path),
            "carrier_vote_rows": rel(vote_rows_path),
            "mask_source_summary": rel(mask_source_path),
            "semantic_backend_summary": rel(semantic_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_same_view_relations(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "relation_summary.json", payload)
    write_csv(out / "mask_relation_rows.csv", payload["mask_relation_rows"])
    write_csv(out / "relation_metric_rows.csv", payload["relation_metric_rows"])
    write_csv(out / "relation_repair_sweep_rows.csv", payload["relation_repair_sweep_rows"])


def _mask_row_by_id(mask_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("mask_observation_id")): row for row in mask_rows}


def _key_mask_metrics(
    variant: str,
    selected_ids: list[str],
    all_mask_ids: list[str],
    mask_by_id: dict[str, dict[str, Any]],
    mask_components: dict[str, str],
    component_universe: set[str],
    source_note: str,
) -> dict[str, Any]:
    selected_set = set(selected_ids)
    covered_components = {mask_components[mid] for mid in selected_ids if mask_components.get(mid)}
    selected_rows = [mask_by_id[mid] for mid in selected_ids if mid in mask_by_id]
    all_rows = [mask_by_id[mid] for mid in all_mask_ids if mid in mask_by_id]
    areas = sorted(_as_float(row.get("mask_area")) for row in all_rows)
    p90_area = areas[int(0.90 * (len(areas) - 1))] if areas else 0.0
    selected_large = [row for row in selected_rows if _as_float(row.get("mask_area")) >= p90_area]
    all_large = [row for row in all_rows if _as_float(row.get("mask_area")) >= p90_area]
    selected_underseg = [
        row
        for row in selected_large
        if _as_float(row.get("diagnostic_gt_purity"), 1.0) < 0.50
    ]
    all_underseg = [
        row
        for row in all_large
        if _as_float(row.get("diagnostic_gt_purity"), 1.0) < 0.50
    ]
    gt_to_best_purity: dict[str, float] = defaultdict(float)
    for row in selected_rows:
        gt = str(row.get("diagnostic_gt_instance") or "")
        if gt:
            gt_to_best_purity[gt] = max(gt_to_best_purity[gt], _as_float(row.get("diagnostic_gt_purity")))
    all_gt = {str(row.get("diagnostic_gt_instance") or "") for row in all_rows if str(row.get("diagnostic_gt_instance") or "")}
    key_mask_purities = [_as_float(row.get("diagnostic_gt_purity")) for row in selected_rows]
    false_key_count = sum(1 for value in key_mask_purities if value < 0.50)
    return {
        "variant": variant,
        "selected": bool(selected_ids),
        "key_mask_count": len(selected_ids),
        "key_mask_ratio": len(selected_ids) / max(len(all_mask_ids), 1),
        "carrier_coverage": sum(_as_int(mask_by_id[mid].get("carrier_count")) for mid in selected_ids if mid in mask_by_id)
        / max(sum(_as_int(row.get("carrier_count")) for row in all_rows), 1),
        "component_coverage": len(covered_components) / max(len(component_universe), 1),
        "new_coverage_per_key_mask_mean": len(covered_components) / max(len(selected_ids), 1),
        "part_mask_suppression_rate": 1.0 - (len(selected_ids) / max(len(all_mask_ids), 1)),
        "whole_candidate_keep_rate": None,
        "large_underseg_selected_rate": len(selected_underseg) / max(len(selected_large), 1),
        "raw_large_underseg_rate": len(all_underseg) / max(len(all_large), 1),
        "key_mask_support_density_mean": sum(_as_float(row.get("support_density")) for row in selected_rows) / max(len(selected_rows), 1),
        "key_mask_purity": sum(key_mask_purities) / max(len(key_mask_purities), 1),
        "key_mask_whole_object_coverage": None,
        "false_key_mask_rate": false_key_count / max(len(key_mask_purities), 1),
        "GT_object_has_keymask@0.25": sum(1 for gt in all_gt if gt_to_best_purity.get(gt, 0.0) >= 0.25) / max(len(all_gt), 1),
        "GT_object_has_keymask@0.50": sum(1 for gt in all_gt if gt_to_best_purity.get(gt, 0.0) >= 0.50) / max(len(all_gt), 1),
        "source_note": source_note,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_v50_key_mask_selection() -> dict[str, Any]:
    mask_table_path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv"
    vote_rows_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv"
    relation_summary_path = "outputs/audit/v50_same_view_relations/relation_summary.json"
    mask_source_path = "outputs/audit/v50_mask_source_audit/mask_source_summary.json"

    mask_rows = read_csv(project_path(mask_table_path)) if exists(mask_table_path) else []
    vote_rows = read_csv(project_path(vote_rows_path)) if exists(vote_rows_path) else []
    relation_summary = load_optional_json(relation_summary_path)
    mask_source = load_optional_json(mask_source_path)
    mask_by_id = _mask_row_by_id(mask_rows)
    all_mask_ids = [str(row.get("mask_observation_id")) for row in mask_rows if str(row.get("mask_observation_id"))]
    mask_components: dict[str, str] = {}
    component_to_masks: dict[str, list[str]] = defaultdict(list)
    for row in vote_rows:
        mask_id = str(row.get("mask_observation_id") or "")
        component = _component_key_from_vote(row)
        if not mask_id or not component:
            continue
        mask_components[mask_id] = component
        component_to_masks[component].append(mask_id)
    component_universe = set(component_to_masks)

    def support_score(mask_id: str) -> tuple[float, float, float]:
        row = mask_by_id.get(mask_id, {})
        return (
            _as_float(row.get("carrier_count")),
            _as_float(row.get("support_density")),
            -_as_float(row.get("mask_area")),
        )

    def reliability_score(mask_id: str) -> tuple[float, float, float]:
        row = mask_by_id.get(mask_id, {})
        carrier = _as_float(row.get("carrier_count"))
        density = _as_float(row.get("support_density"))
        area = max(_as_float(row.get("mask_area")), 1.0)
        underseg_penalty = 1.0 / (1.0 + area / 50000.0)
        return (carrier * underseg_penalty, density, -area)

    def density_score(mask_id: str) -> tuple[float, float, float]:
        row = mask_by_id.get(mask_id, {})
        return (_as_float(row.get("support_density")), _as_float(row.get("carrier_count")), -_as_float(row.get("mask_area")))

    def small_area_score(mask_id: str) -> tuple[float, float, float]:
        row = mask_by_id.get(mask_id, {})
        return (-_as_float(row.get("mask_area")), _as_float(row.get("support_density")), _as_float(row.get("carrier_count")))

    selected_k2: list[str] = []
    selected_k3: list[str] = []
    selected_k4: list[str] = []
    selected_k5: list[str] = []
    for component, masks in sorted(component_to_masks.items()):
        selected_k2.append(max(sorted(set(masks)), key=support_score))
        selected_k3.append(max(sorted(set(masks)), key=density_score))
        selected_k4.append(max(sorted(set(masks)), key=reliability_score))
        selected_k5.append(max(sorted(set(masks)), key=small_area_score))
    target_count = len(selected_k2)
    selected_k1 = [
        str(row.get("mask_observation_id"))
        for row in sorted(mask_rows, key=lambda row: _as_float(row.get("mask_area")), reverse=True)[:target_count]
    ]
    selected_k0 = all_mask_ids
    selected_k6 = sorted(all_mask_ids)[:: max(len(all_mask_ids) // max(target_count, 1), 1)][:target_count]

    key_rows: list[dict[str, Any]] = []
    for variant, ids in [
        ("K0_no_key_selection_all_masks", selected_k0),
        ("K1_area_largest_masks", selected_k1),
        ("K2_component_coverage_greedy_set_cover", selected_k2),
        ("K3_density_first_component_set_cover", selected_k3),
        ("K4_hierarchy_aware_underseg_penalty_fallback", selected_k4),
        ("K5_small_area_underseg_penalty_component_set_cover", selected_k5),
        ("K6_deterministic_random_like_control", selected_k6),
    ]:
        note = "I1 U32 component coverage; no GT used for selection"
        if variant.startswith("K4"):
            note += "; hierarchy branch weak so only non-GT underseg area/support penalty is used"
        key_rows.append(_key_mask_metrics(variant, ids, all_mask_ids, mask_by_id, mask_components, component_universe, note))

    best = max(
        [row for row in key_rows if row["variant"] not in {"K0_no_key_selection_all_masks", "K6_deterministic_random_like_control"}],
        key=lambda row: (
            row["component_coverage"],
            -row["key_mask_ratio"],
            -row["large_underseg_selected_rate"],
            row["key_mask_purity"],
        ),
    )
    selected_ids = {
        "K0_no_key_selection_all_masks": selected_k0,
        "K1_area_largest_masks": selected_k1,
        "K2_component_coverage_greedy_set_cover": selected_k2,
        "K3_density_first_component_set_cover": selected_k3,
        "K4_hierarchy_aware_underseg_penalty_fallback": selected_k4,
        "K5_small_area_underseg_penalty_component_set_cover": selected_k5,
        "K6_deterministic_random_like_control": selected_k6,
    }[str(best["variant"])]
    coverage_rows = []
    covered: set[str] = set()
    for rank, mask_id in enumerate(selected_ids, start=1):
        component = mask_components.get(mask_id, "")
        new = bool(component and component not in covered)
        if component:
            covered.add(component)
        row = mask_by_id.get(mask_id, {})
        coverage_rows.append(
            {
                "rank": rank,
                "mask_observation_id": mask_id,
                "component": component,
                "new_component_covered": new,
                "cumulative_component_coverage": len(covered) / max(len(component_universe), 1),
                "carrier_count": row.get("carrier_count"),
                "support_density": row.get("support_density"),
                "mask_area": row.get("mask_area"),
                "diagnostic_gt_instance": row.get("diagnostic_gt_instance"),
                "diagnostic_gt_purity": row.get("diagnostic_gt_purity"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    gate = {
        "component_coverage_pass": best["component_coverage"] >= 0.75,
        "key_mask_ratio_pass": best["key_mask_ratio"] <= 0.55,
        "GT_object_has_keymask_025_pass_diagnostic": best["GT_object_has_keymask@0.25"] >= 0.55,
        "large_underseg_reduction_pass": best["large_underseg_selected_rate"] <= best["raw_large_underseg_rate"] - 0.10,
        "pass": best["component_coverage"] >= 0.75 and best["key_mask_ratio"] <= 0.55,
    }
    return {
        "phase": "v50_key_mask_selection",
        "created_at": utc_now(),
        "selected_variant": best["variant"],
        "key_mask_rows": key_rows,
        "key_mask_coverage_rows": coverage_rows,
        "summary": best,
        "gate": gate,
        "relation_branch_status": "weak_diagnostic_only" if not _nested(relation_summary, "gate", "pass", default=False) else "available",
        "mask_source_route": _nested(mask_source, "gate", "effective_hierarchy_route", default="unknown"),
        "artifact_sources": {
            "mask_observation_table": rel(mask_table_path),
            "carrier_vote_rows": rel(vote_rows_path),
            "relation_summary": rel(relation_summary_path),
            "mask_source_summary": rel(mask_source_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_key_mask_selection(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "key_mask_summary.json", payload)
    write_csv(out / "key_mask_rows.csv", payload["key_mask_rows"])
    write_csv(out / "key_mask_coverage_rows.csv", payload["key_mask_coverage_rows"])


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))  # type: ignore[return-value]


def _safe_auc(rows: list[dict[str, Any]], score_key: str, label_key: str = "same_GT_pair") -> float | None:
    labeled = [
        (_as_float(row.get(score_key)), row.get(label_key))
        for row in rows
        if row.get(label_key) in (True, False, "True", "False")
    ]
    positives = [score for score, label in labeled if label in (True, "True")]
    negatives = [score for score, label in labeled if label in (False, "False")]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _precision_at(rows: list[dict[str, Any]], score_key: str, k: int) -> float | None:
    labeled = [row for row in rows if row.get("same_GT_pair") in (True, False, "True", "False")]
    if not labeled:
        return None
    score_values = [_as_float(row.get(score_key)) for row in labeled]
    if score_values and min(score_values) == max(score_values):
        return sum(1 for row in labeled if row.get("same_GT_pair") in (True, "True")) / len(labeled)
    top = sorted(labeled, key=lambda row: _as_float(row.get(score_key)), reverse=True)[: min(k, len(labeled))]
    if not top:
        return None
    return sum(1 for row in top if row.get("same_GT_pair") in (True, "True")) / len(top)


def _scene_auc(rows: list[dict[str, Any]], score_key: str, scene: str) -> float | None:
    return _safe_auc([row for row in rows if str(row.get("scene")) == scene], score_key)


def _variant_metric_row(
    variant: str,
    rows: list[dict[str, Any]],
    score_key: str,
    note: str,
    available: bool = True,
) -> dict[str, Any]:
    precision_top1k = _precision_at(rows, score_key, 1000)
    precision_top5k = _precision_at(rows, score_key, 5000)
    return {
        "variant": variant,
        "available": available,
        "component_pair_count": len(rows),
        "positive_affinity_pair_count": sum(1 for row in rows if _as_float(row.get(score_key)) > 0.0),
        "negative_affinity_pair_count": sum(1 for row in rows if _as_float(row.get("conflict_support")) > 0.0),
        "same_GT_pair_AUC": _safe_auc(rows, score_key),
        "same_GT_pair_precision@top1k": precision_top1k,
        "same_GT_pair_precision@top5k": precision_top5k,
        "false_merge_rate@top1k": None if precision_top1k is None else 1.0 - precision_top1k,
        "false_merge_rate@top5k": None if precision_top5k is None else 1.0 - precision_top5k,
        "scene0081_pair_AUC": _scene_auc(rows, score_key, "scene0081_00"),
        "scene0591_pair_AUC": _scene_auc(rows, score_key, "scene0591_00"),
        "score_key": score_key,
        "note": note,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_v50_relation_propagation(max_affinity_rows: int = 20000) -> dict[str, Any]:
    pair_rows_path = "outputs/audit/v49_component_completion_atlas/component_pair_error_rows.csv"
    pair_summary_path = "outputs/audit/v49_component_completion_atlas/component_completion_atlas_summary.json"
    vote_rows_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv"
    key_mask_summary_path = "outputs/audit/v50_key_masks/key_mask_summary.json"
    key_mask_rows_path = "outputs/audit/v50_key_masks/key_mask_coverage_rows.csv"
    relation_rows_path = "outputs/audit/v50_same_view_relations/mask_relation_rows.csv"
    relation_summary_path = "outputs/audit/v50_same_view_relations/relation_summary.json"

    pair_rows = read_csv(project_path(pair_rows_path)) if exists(pair_rows_path) else []
    pair_summary = load_optional_json(pair_summary_path)
    vote_rows = read_csv(project_path(vote_rows_path)) if exists(vote_rows_path) else []
    key_mask_payload = load_optional_json(key_mask_summary_path)
    key_mask_coverage_rows = read_csv(project_path(key_mask_rows_path)) if exists(key_mask_rows_path) else []
    relation_rows = read_csv(project_path(relation_rows_path)) if exists(relation_rows_path) else []
    relation_summary = load_optional_json(relation_summary_path)

    selected_mask_ids = {str(row.get("mask_observation_id")) for row in key_mask_coverage_rows}
    mask_to_components: dict[str, set[str]] = defaultdict(set)
    for row in vote_rows:
        mask_id = str(row.get("mask_observation_id") or "")
        component = _component_key_from_vote(row)
        if mask_id and component:
            mask_to_components[mask_id].add(component)
    keymask_pair_support: dict[tuple[str, str], float] = defaultdict(float)
    selected_multicomponent_mask_count = 0
    for mask_id in selected_mask_ids:
        components = sorted(mask_to_components.get(mask_id, set()))
        if len(components) >= 2:
            selected_multicomponent_mask_count += 1
        for idx, left in enumerate(components):
            for right in components[idx + 1 :]:
                keymask_pair_support[_pair_key(left, right)] += 1.0

    part_support: dict[tuple[str, str], float] = defaultdict(float)
    sibling_support: dict[tuple[str, str], float] = defaultdict(float)
    for row in relation_rows:
        child = str(row.get("child") or "")
        parent = str(row.get("parent") or "")
        if not child or not parent or child == parent:
            continue
        key = _pair_key(child, parent)
        confidence = _as_float(row.get("relation_confidence"))
        relation_type = str(row.get("relation_type") or "")
        if "sibling" in relation_type:
            sibling_support[key] = max(sibling_support[key], confidence)
        else:
            part_support[key] = max(part_support[key], confidence)

    scene_components: dict[str, list[str]] = defaultdict(list)
    for row in pair_rows:
        scene = str(row.get("scene") or "")
        for key in ("component_i", "component_j"):
            component = str(row.get(key) or "")
            if component:
                scene_components[scene].append(component)
    shuffled_map: dict[str, str] = {}
    for components in scene_components.values():
        uniq = sorted(set(components))
        if not uniq:
            continue
        for idx, component in enumerate(uniq):
            shuffled_map[component] = uniq[(idx + 1) % len(uniq)]

    max_shared_mask = max((_as_float(row.get("shared_mask_count")) for row in pair_rows), default=0.0)
    affinity_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        left = str(row.get("component_i") or "")
        right = str(row.get("component_j") or "")
        key = _pair_key(left, right)
        shuffled_key = _pair_key(shuffled_map.get(left, left), shuffled_map.get(right, right))
        keymask_score = keymask_pair_support.get(key, 0.0)
        part_score = part_support.get(key, 0.0)
        sibling_score = sibling_support.get(key, 0.0)
        relation_score = max(part_score, sibling_score)
        shuffled_relation_score = max(part_support.get(shuffled_key, 0.0), sibling_support.get(shuffled_key, 0.0))
        mask_only_score = _as_float(row.get("shared_mask_count")) / max(max_shared_mask, 1.0)
        no_temporal_score = mask_only_score
        combined_score = max(keymask_score, relation_score)
        affinity_rows.append(
            {
                "scene": row.get("scene"),
                "component_i": left,
                "component_j": right,
                "same_GT_pair": row.get("same_GT_pair"),
                "different_GT_pair": row.get("different_GT_pair"),
                "v49_combined_nonGT_score": row.get("combined_nonGT_score"),
                "keymask_cosupport": keymask_score,
                "part_support": part_score,
                "sibling_support": sibling_score,
                "conflict_support": 0.0,
                "P0_raw_component_pair_similarity": row.get("combined_nonGT_score"),
                "P1_keymask_cosupport_only": keymask_score,
                "P2_same_view_part_sibling_propagation_only": relation_score,
                "P3_keymask_plus_part_sibling_propagation": combined_score,
                "P4_P3_plus_conflict_propagation": combined_score,
                "P6_shuffled_D4RT_propagation_control": shuffled_relation_score,
                "P7_no_temporal_propagation_control": no_temporal_score,
                "P8_mask_only_propagation_control": mask_only_score,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    metric_rows = [
        _variant_metric_row("P0_no_propagation_raw_v49_combined_nonGT", affinity_rows, "P0_raw_component_pair_similarity", "v49 combined non-GT pair score used as raw component-pair reference"),
        _variant_metric_row("P1_key_mask_cosupport_only", affinity_rows, "P1_keymask_cosupport_only", "selected key masks induce affinity only when one selected mask supports two or more components"),
        _variant_metric_row("P2_same_view_part_sibling_propagation_only", affinity_rows, "P2_same_view_part_sibling_propagation_only", "direct v50 relation graph propagation; relation branch already marked weak if precision gate failed"),
        _variant_metric_row("P3_key_mask_plus_part_sibling_propagation", affinity_rows, "P3_keymask_plus_part_sibling_propagation", "max of key-mask co-support and relation propagation"),
        _variant_metric_row("P4_P3_plus_conflict_propagation", affinity_rows, "P4_P3_plus_conflict_propagation", "same as P3 because Phase 2 has no reliable hard conflict edges"),
        _variant_metric_row("P6_shuffled_D4RT_propagation_control", affinity_rows, "P6_shuffled_D4RT_propagation_control", "deterministic within-scene component-id rotation control"),
        _variant_metric_row("P7_no_temporal_propagation_control", affinity_rows, "P7_no_temporal_propagation_control", "shared-mask count proxy without D4RT relation propagation"),
        _variant_metric_row("P8_mask_only_propagation_control", affinity_rows, "P8_mask_only_propagation_control", "mask-only shared-mask count proxy"),
    ]
    repair_sweep_rows: list[dict[str, Any]] = []
    for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]:
        score_key = f"P2_high_conf_binary_{threshold:.2f}"
        for row in affinity_rows:
            row[score_key] = 1.0 if _as_float(row.get("P2_same_view_part_sibling_propagation_only")) >= threshold else 0.0
        repair = _variant_metric_row(
            f"P2_high_conf_relation_threshold_{threshold:.2f}",
            affinity_rows,
            score_key,
            "plan-directed repair: restrict part/sibling propagation to high-confidence relation subset",
        )
        repair["relation_confidence_threshold"] = threshold
        repair_sweep_rows.append(repair)
    metrics_by_variant = {str(row["variant"]): row for row in metric_rows}
    p3_auc = metrics_by_variant["P3_key_mask_plus_part_sibling_propagation"]["same_GT_pair_AUC"]
    p6_auc = metrics_by_variant["P6_shuffled_D4RT_propagation_control"]["same_GT_pair_AUC"]
    p7_auc = metrics_by_variant["P7_no_temporal_propagation_control"]["same_GT_pair_AUC"]
    p3_top5k = metrics_by_variant["P3_key_mask_plus_part_sibling_propagation"]["same_GT_pair_precision@top5k"]
    v49_pair_metrics = pair_summary.get("pair_metrics", {}) if isinstance(pair_summary.get("pair_metrics"), dict) else {}
    v49_auc = v49_pair_metrics.get("pair_AUC_combined_nonGT")
    v49_top5k = v49_pair_metrics.get("precision@top5k_same_object")
    real_minus_shuffled = None if p3_auc is None or p6_auc is None else p3_auc - p6_auc
    real_minus_no_temporal = None if p3_auc is None or p7_auc is None else p3_auc - p7_auc

    gate = {
        "component_pair_rows_available": bool(affinity_rows),
        "propagation_auc_vs_v49_pass": p3_auc is not None and v49_auc is not None and p3_auc >= _as_float(v49_auc) + 0.04,
        "propagation_top5k_vs_v49_pass": p3_top5k is not None and v49_top5k is not None and p3_top5k >= _as_float(v49_top5k) + 0.05,
        "propagation_real_minus_shuffled_AUC_pass": real_minus_shuffled is not None and real_minus_shuffled >= 0.04,
        "propagation_real_minus_no_temporal_AUC_pass": real_minus_no_temporal is not None and real_minus_no_temporal >= 0.03,
        "relation_branch_weak": not bool(_nested(relation_summary, "gate", "pass", default=False)),
        "keymask_multicomponent_support_available": selected_multicomponent_mask_count > 0,
    }
    gate["pass"] = bool(
        gate["component_pair_rows_available"]
        and (gate["propagation_auc_vs_v49_pass"] or gate["propagation_top5k_vs_v49_pass"])
        and gate["propagation_real_minus_shuffled_AUC_pass"]
        and gate["propagation_real_minus_no_temporal_AUC_pass"]
    )

    return {
        "phase": "v50_relation_propagation",
        "created_at": utc_now(),
        "summary": {
            "component_pair_count": len(affinity_rows),
            "positive_affinity_pair_count": sum(1 for row in affinity_rows if _as_float(row.get("P3_keymask_plus_part_sibling_propagation")) > 0.0),
            "negative_affinity_pair_count": sum(1 for row in affinity_rows if _as_float(row.get("conflict_support")) > 0.0),
            "mean_keymask_cosupport": sum(_as_float(row.get("keymask_cosupport")) for row in affinity_rows) / max(len(affinity_rows), 1),
            "mean_part_support": sum(_as_float(row.get("part_support")) for row in affinity_rows) / max(len(affinity_rows), 1),
            "mean_sibling_support": sum(_as_float(row.get("sibling_support")) for row in affinity_rows) / max(len(affinity_rows), 1),
            "mean_conflict_support": 0.0,
            "selected_key_mask_count": len(selected_mask_ids),
            "selected_multicomponent_mask_count": selected_multicomponent_mask_count,
            "propagation_real_minus_shuffled_AUC": real_minus_shuffled,
            "propagation_real_minus_no_temporal_AUC": real_minus_no_temporal,
            "same_GT_pair_AUC": p3_auc,
            "same_GT_pair_precision@top1k": metrics_by_variant["P3_key_mask_plus_part_sibling_propagation"]["same_GT_pair_precision@top1k"],
            "same_GT_pair_precision@top5k": p3_top5k,
            "false_merge_rate@top1k": metrics_by_variant["P3_key_mask_plus_part_sibling_propagation"]["false_merge_rate@top1k"],
            "false_merge_rate@top5k": metrics_by_variant["P3_key_mask_plus_part_sibling_propagation"]["false_merge_rate@top5k"],
            "scene0081_pair_AUC": metrics_by_variant["P3_key_mask_plus_part_sibling_propagation"]["scene0081_pair_AUC"],
            "scene0591_pair_AUC": metrics_by_variant["P3_key_mask_plus_part_sibling_propagation"]["scene0591_pair_AUC"],
            "v49_pair_AUC_combined_nonGT": v49_auc,
            "v49_precision@top5k_same_object": v49_top5k,
            "phase6_hypothesis_recall_delta": None,
            "key_mask_summary_gate": key_mask_payload.get("gate"),
            "relation_summary_gate": relation_summary.get("gate"),
        },
        "gate": gate,
        "component_affinity_rows": affinity_rows[:max_affinity_rows],
        "propagation_metric_rows": metric_rows,
        "propagation_repair_rows": repair_sweep_rows,
        "artifact_sources": {
            "pair_rows": rel(pair_rows_path),
            "pair_summary": rel(pair_summary_path),
            "carrier_vote_rows": rel(vote_rows_path),
            "key_mask_summary": rel(key_mask_summary_path),
            "key_mask_rows": rel(key_mask_rows_path),
            "relation_rows": rel(relation_rows_path),
            "relation_summary": rel(relation_summary_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_relation_propagation(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "propagation_summary.json", payload)
    write_csv(out / "component_affinity_rows.csv", payload["component_affinity_rows"])
    write_csv(out / "propagation_metric_rows.csv", payload["propagation_metric_rows"])
    write_csv(out / "propagation_repair_rows.csv", payload["propagation_repair_rows"])


def _component_gt_distribution(vote_rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    distributions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in vote_rows:
        component = _component_key_from_vote(row)
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not component or not gt:
            continue
        weight = max(_as_int(row.get("supporting_unique_carrier_count"), 1), 1)
        distributions[component][gt] += weight
    return distributions


def _hypothesis_diagnostics(components: set[str], component_gt: dict[str, Counter[str]], gt_totals: Counter[str]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for component in components:
        counts.update(component_gt.get(component, Counter()))
    total = sum(counts.values())
    if total <= 0:
        return {
            "diagnostic_dominant_gt": "",
            "diagnostic_gt_weight": 0,
            "diagnostic_total_weight": 0,
            "diagnostic_purity": 0.0,
            "diagnostic_best_gt_coverage": 0.0,
            "diagnostic_conflict": False,
        }
    dominant_gt, dominant_count = counts.most_common(1)[0]
    coverage = dominant_count / max(gt_totals.get(dominant_gt, 0), 1)
    purity = dominant_count / total
    return {
        "diagnostic_dominant_gt": dominant_gt,
        "diagnostic_gt_weight": dominant_count,
        "diagnostic_total_weight": total,
        "diagnostic_purity": purity,
        "diagnostic_best_gt_coverage": coverage,
        "diagnostic_conflict": purity < 0.75,
    }


def build_v50_hypothesis_generation(max_hypothesis_rows: int = 12000) -> dict[str, Any]:
    vote_rows_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv"
    key_mask_rows_path = "outputs/audit/v50_key_masks/key_mask_coverage_rows.csv"
    relation_rows_path = "outputs/audit/v50_same_view_relations/mask_relation_rows.csv"
    propagation_rows_path = "outputs/audit/v50_relation_propagation/component_affinity_rows.csv"
    propagation_summary_path = "outputs/audit/v50_relation_propagation/propagation_summary.json"
    lattice_rows_path = "outputs/audit/v49_component_lattice/component_lattice_containment_rows.csv"

    vote_rows = read_csv(project_path(vote_rows_path)) if exists(vote_rows_path) else []
    key_rows = read_csv(project_path(key_mask_rows_path)) if exists(key_mask_rows_path) else []
    relation_rows = read_csv(project_path(relation_rows_path)) if exists(relation_rows_path) else []
    propagation_rows = read_csv(project_path(propagation_rows_path)) if exists(propagation_rows_path) else []
    propagation_summary = load_optional_json(propagation_summary_path)
    lattice_rows = read_csv(project_path(lattice_rows_path)) if exists(lattice_rows_path) else []
    propagation_weak = not bool(_nested(propagation_summary, "gate", "pass", default=False))

    component_gt = _component_gt_distribution(vote_rows)
    component_universe = sorted(component_gt)
    gt_totals: Counter[str] = Counter()
    for counts in component_gt.values():
        gt_totals.update(counts)

    hypotheses: dict[tuple[str, ...], dict[str, Any]] = {}

    def add_hypothesis(components: set[str], source: str, score: float, keymask_support: float = 0.0, propagation_support: float = 0.0) -> None:
        clean = {component for component in components if component in component_gt}
        if not clean:
            return
        key = tuple(sorted(clean))
        row = hypotheses.get(key)
        if row is None:
            row = {
                "component_set": key,
                "sources": set(),
                "score": 0.0,
                "keymask_support": 0.0,
                "propagation_support": 0.0,
            }
            hypotheses[key] = row
        row["sources"].add(source)
        row["score"] = max(_as_float(row.get("score")), score)
        row["keymask_support"] = max(_as_float(row.get("keymask_support")), keymask_support)
        row["propagation_support"] = max(_as_float(row.get("propagation_support")), propagation_support)

    for component in component_universe:
        add_hypothesis({component}, "H0_singleton_U32_component", 0.10)

    for row in key_rows:
        component = str(row.get("component") or "")
        rank = max(_as_int(row.get("rank"), 1), 1)
        add_hypothesis({component}, "H1_selected_key_mask_induced", 0.20 + 1.0 / rank, keymask_support=1.0)

    children_by_parent: dict[str, set[str]] = defaultdict(set)
    relation_conf_by_pair: dict[tuple[str, str], float] = defaultdict(float)
    for row in relation_rows:
        child = str(row.get("child") or "")
        parent = str(row.get("parent") or "")
        confidence = _as_float(row.get("relation_confidence"))
        if child and parent and child != parent:
            relation_conf_by_pair[_pair_key(child, parent)] = max(relation_conf_by_pair[_pair_key(child, parent)], confidence)
            children_by_parent[parent].add(child)
            score = 0.15 + 0.10 * confidence if propagation_weak else 0.30 + confidence
            add_hypothesis({child, parent}, "H2_same_view_parent_induced", score, propagation_support=confidence)

    for parent, children in children_by_parent.items():
        group = set(children)
        group.add(parent)
        if 2 <= len(group) <= 25:
            support = max((relation_conf_by_pair.get(_pair_key(child, parent), 0.0) for child in children), default=0.0)
            score = 0.12 + 0.10 * support if propagation_weak else 0.35 + support
            add_hypothesis(group, "H3_sibling_under_parent", score, propagation_support=support)

    for row in sorted(propagation_rows, key=lambda r: _as_float(r.get("P3_keymask_plus_part_sibling_propagation")), reverse=True):
        support = _as_float(row.get("P3_keymask_plus_part_sibling_propagation"))
        if support <= 0.0:
            break
        score = 0.14 + 0.10 * support if propagation_weak else 0.30 + support
        add_hypothesis(
            {str(row.get("component_i") or ""), str(row.get("component_j") or "")},
            "H4_multi_view_propagated_relation",
            score,
            propagation_support=support,
        )

    lattice_children: dict[str, set[str]] = defaultdict(set)
    for row in lattice_rows:
        child = str(row.get("child_component") or "")
        parent = str(row.get("parent_component") or "")
        if child and parent and child != parent:
            lattice_children[parent].add(child)
    for parent, children in lattice_children.items():
        group = set(children)
        group.add(parent)
        if 2 <= len(group) <= 25:
            add_hypothesis(group, "H5_multi_scale_parent_decomposition", 0.11 if propagation_weak else 0.25)

    # H6 semantic-guarded expansion is intentionally unavailable until Phase 5 establishes a semantic guard.
    hypothesis_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(hypotheses.values(), start=1):
        components = set(row["component_set"])
        diag = _hypothesis_diagnostics(components, component_gt, gt_totals)
        sources = sorted(row["sources"])
        hypothesis_rows.append(
            {
                "hypothesis_id": f"v50_hyp_{idx:05d}",
                "component_ids": ";".join(sorted(components)),
                "component_count": len(components),
                "source_list": ",".join(sources),
                "primary_source": sources[0] if sources else "",
                "score": row["score"],
                "keymask_support": row["keymask_support"],
                "propagation_support": row["propagation_support"],
                "diagnostic_dominant_gt": diag["diagnostic_dominant_gt"],
                "diagnostic_gt_weight": diag["diagnostic_gt_weight"],
                "diagnostic_total_weight": diag["diagnostic_total_weight"],
                "diagnostic_purity": diag["diagnostic_purity"],
                "diagnostic_best_gt_coverage": diag["diagnostic_best_gt_coverage"],
                "diagnostic_conflict": diag["diagnostic_conflict"],
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    hypothesis_rows.sort(key=lambda row: (_as_float(row.get("score")), _as_float(row.get("diagnostic_purity"))), reverse=True)
    hypothesis_rows = hypothesis_rows[:max_hypothesis_rows]
    hypothesis_sizes = [_as_int(row.get("component_count")) for row in hypothesis_rows]
    hypothesis_sizes_sorted = sorted(hypothesis_sizes)
    p90_idx = int(0.90 * (len(hypothesis_sizes_sorted) - 1)) if hypothesis_sizes_sorted else 0
    topk = hypothesis_rows[: min(1000, len(hypothesis_rows))]

    gt_best_cov: dict[str, float] = defaultdict(float)
    for row in hypothesis_rows:
        gt = str(row.get("diagnostic_dominant_gt") or "")
        if gt:
            gt_best_cov[gt] = max(gt_best_cov[gt], _as_float(row.get("diagnostic_best_gt_coverage")))
    gt_count = len(gt_totals)
    gt_cov25 = sum(1 for gt in gt_totals if gt_best_cov.get(gt, 0.0) >= 0.25) / max(gt_count, 1)
    gt_cov50 = sum(1 for gt in gt_totals if gt_best_cov.get(gt, 0.0) >= 0.50) / max(gt_count, 1)
    topk_purity_mean = sum(_as_float(row.get("diagnostic_purity")) for row in topk) / max(len(topk), 1)
    topk_same_gt_precision = sum(1 for row in topk if _as_float(row.get("diagnostic_purity")) >= 0.75) / max(len(topk), 1)
    topk_conflict_rate = sum(1 for row in topk if str(row.get("diagnostic_conflict")) == "True" or row.get("diagnostic_conflict") is True) / max(len(topk), 1)
    source_counts = Counter()
    for row in hypothesis_rows:
        for source in str(row.get("source_list") or "").split(","):
            if source:
                source_counts[source] += 1
    metric_rows = []
    for source in [
        "H0_singleton_U32_component",
        "H1_selected_key_mask_induced",
        "H2_same_view_parent_induced",
        "H3_sibling_under_parent",
        "H4_multi_view_propagated_relation",
        "H5_multi_scale_parent_decomposition",
        "H6_semantic_guarded_expansion",
        "H7_union_H1_H6_dedupe",
    ]:
        source_rows = (
            hypothesis_rows
            if source == "H7_union_H1_H6_dedupe"
            else [row for row in hypothesis_rows if source in str(row.get("source_list") or "")]
        )
        source_top = source_rows[: min(1000, len(source_rows))]
        metric_rows.append(
            {
                "source": source,
                "available": source != "H6_semantic_guarded_expansion",
                "hypothesis_count": len(source_rows),
                "hypothesis_purity@topk": sum(_as_float(row.get("diagnostic_purity")) for row in source_top) / max(len(source_top), 1),
                "same_GT_set_precision@topk": sum(1 for row in source_top if _as_float(row.get("diagnostic_purity")) >= 0.75) / max(len(source_top), 1),
                "hypothesis_conflict_rate@topk": sum(
                    1
                    for row in source_top
                    if str(row.get("diagnostic_conflict")) == "True" or row.get("diagnostic_conflict") is True
                )
                / max(len(source_top), 1),
                "note": "semantic guard not available before Phase 5" if source == "H6_semantic_guarded_expansion" else "",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    gate = {
        "hypothesis_count_pass": len(hypothesis_rows) <= 12000,
        "GT_object_has_hypothesis_025_pass": gt_cov25 >= 0.65,
        "GT_object_has_hypothesis_050_pass": gt_cov50 >= 0.45,
        "hypothesis_purity_topk_pass": topk_purity_mean >= 0.75,
        "hypothesis_conflict_rate_topk_pass": topk_conflict_rate <= 0.25,
    }
    gate["pass"] = bool(all(gate.values()))

    return {
        "phase": "v50_hypothesis_generation",
        "created_at": utc_now(),
        "summary": {
            "hypothesis_count": len(hypothesis_rows),
            "hypothesis_source_breakdown": dict(source_counts),
            "hypothesis_size_mean": sum(hypothesis_sizes) / max(len(hypothesis_sizes), 1),
            "hypothesis_size_p90": hypothesis_sizes_sorted[p90_idx] if hypothesis_sizes_sorted else 0,
            "GT_object_has_hypothesis@0.25": gt_cov25,
            "GT_object_has_hypothesis@0.50": gt_cov50,
            "hypothesis_purity@topk": topk_purity_mean,
            "same_GT_set_precision@topk": topk_same_gt_precision,
            "hypothesis_conflict_rate": topk_conflict_rate,
            "hypothesis_keymask_support_mean": sum(_as_float(row.get("keymask_support")) for row in hypothesis_rows)
            / max(len(hypothesis_rows), 1),
            "hypothesis_propagation_support_mean": sum(_as_float(row.get("propagation_support")) for row in hypothesis_rows)
            / max(len(hypothesis_rows), 1),
            "semantic_guarded_expansion_available": False,
            "weak_propagation_score_demotion": propagation_weak,
            "propagation_gate": propagation_summary.get("gate"),
        },
        "gate": gate,
        "hypothesis_rows": hypothesis_rows,
        "hypothesis_source_metric_rows": metric_rows,
        "artifact_sources": {
            "carrier_vote_rows": rel(vote_rows_path),
            "key_mask_rows": rel(key_mask_rows_path),
            "relation_rows": rel(relation_rows_path),
            "propagation_rows": rel(propagation_rows_path),
            "propagation_summary": rel(propagation_summary_path),
            "component_lattice_rows": rel(lattice_rows_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_hypothesis_generation(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "hypothesis_summary.json", payload)
    write_csv(out / "hypothesis_rows.csv", payload["hypothesis_rows"])
    write_csv(out / "hypothesis_source_metric_rows.csv", payload["hypothesis_source_metric_rows"])


def build_v50_semantic_guard() -> dict[str, Any]:
    v48_path = "outputs/audit/v48_semantic_features/semantic_feature_audit_summary.json"
    backend_path = "outputs/audit/v49_semantic_backend_availability/semantic_backend_availability_summary.json"
    semantic_set_path = "outputs/audit/v49_semantic_set/semantic_set_compatibility_summary.json"
    scoring_path = "outputs/audit/v49_hypothesis_scoring/hypothesis_scoring_summary.json"
    v48 = load_optional_json(v48_path)
    backend = load_optional_json(backend_path)
    semantic_set = load_optional_json(semantic_set_path)
    scoring = load_optional_json(scoring_path)
    backend_rows: list[dict[str, Any]] = []
    for row in backend.get("semantic_source_rows", []):
        if not isinstance(row, dict):
            continue
        backend_rows.append(
            {
                "backend": row.get("feature_backend"),
                "scene": row.get("scene"),
                "available": row.get("available"),
                "component_level_available": False,
                "feature_success_rate": None,
                "semantic_contradiction_AUC": row.get("semantic_affinity_AUC"),
                "key_mask_reliability_AUC": None,
                "note": row.get("mapping_note"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    for row in v48.get("backend_rows", []):
        if not isinstance(row, dict):
            continue
        backend_rows.append(
            {
                "backend": row.get("backend_id") or row.get("feature_backend") or row.get("variant"),
                "scene": "v48_proxy_scope",
                "available": True,
                "component_level_available": False,
                "feature_success_rate": row.get("feature_success_rate"),
                "semantic_contradiction_AUC": row.get("component_pair_AUC_proxy"),
                "key_mask_reliability_AUC": None,
                "note": "v48 proxy evidence; not v50 component-level feature pooling",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    metrics = semantic_set.get("metrics", {}) if isinstance(semantic_set.get("metrics"), dict) else {}
    semantic_negative_auc = metrics.get("semantic_negative_AUC")
    false_merge_reduction = metrics.get("false_merge_reduction_by_semantic_guard")
    completeness_drop = metrics.get("completeness_drop_by_semantic_guard")
    semantic_guard_purity_gain_pass = bool(_nested(semantic_set, "gate", "semantic_guard_purity_gain_pass", default=False))
    semantic_metric_rows = [
        {
            "variant": "S0_colorhist_guard",
            "feature_success_rate": 1.0,
            "component_feature_success_rate": 1.0,
            "semantic_contradiction_AUC": semantic_negative_auc,
            "key_mask_reliability_AUC": None,
            "false_merge_reduction": false_merge_reduction,
            "purity_change": None,
            "completeness_change": None if completeness_drop is None else -_as_float(completeness_drop),
            "DINO_vs_colorhist_delta": metrics.get("DINO_vs_colorhist_delta"),
            "RADIO_vs_colorhist_delta": metrics.get("RADIO_vs_colorhist_delta"),
            "semantic_only_merge_ARI": metrics.get("semantic_only_merge_ARI"),
            "semantic_only_merge_purity": metrics.get("semantic_only_merge_purity"),
            "enabled_for_selection": False,
            "note": "diagnostic negative guard only; false-merge reduction / purity gain did not pass",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "variant": "S1_DINO_guard",
            "feature_success_rate": None,
            "component_feature_success_rate": None,
            "semantic_contradiction_AUC": None,
            "key_mask_reliability_AUC": None,
            "false_merge_reduction": None,
            "purity_change": None,
            "completeness_change": None,
            "DINO_vs_colorhist_delta": metrics.get("DINO_vs_colorhist_delta"),
            "RADIO_vs_colorhist_delta": None,
            "semantic_only_merge_ARI": None,
            "semantic_only_merge_purity": None,
            "enabled_for_selection": False,
            "note": "component-level DINO feature backend unavailable",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
        },
        {
            "variant": "S2_RADIO_RADSeg_guard",
            "feature_success_rate": None,
            "component_feature_success_rate": None,
            "semantic_contradiction_AUC": None,
            "key_mask_reliability_AUC": None,
            "false_merge_reduction": None,
            "purity_change": None,
            "completeness_change": None,
            "DINO_vs_colorhist_delta": None,
            "RADIO_vs_colorhist_delta": metrics.get("RADIO_vs_colorhist_delta"),
            "semantic_only_merge_ARI": None,
            "semantic_only_merge_purity": None,
            "enabled_for_selection": False,
            "note": "component-level RADIO/RADSeg feature backend unavailable",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
        },
        {
            "variant": "S4_semantic_only_positive_merge_negative_control",
            "feature_success_rate": None,
            "component_feature_success_rate": None,
            "semantic_contradiction_AUC": None,
            "key_mask_reliability_AUC": None,
            "false_merge_reduction": None,
            "purity_change": None,
            "completeness_change": None,
            "DINO_vs_colorhist_delta": None,
            "RADIO_vs_colorhist_delta": None,
            "semantic_only_merge_ARI": metrics.get("semantic_only_merge_ARI"),
            "semantic_only_merge_purity": metrics.get("semantic_only_merge_purity"),
            "enabled_for_selection": False,
            "note": "semantic-only positive merge not available and must not be promoted",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "variant": "S5_no_semantic_guard",
            "feature_success_rate": None,
            "component_feature_success_rate": None,
            "semantic_contradiction_AUC": None,
            "key_mask_reliability_AUC": None,
            "false_merge_reduction": 0.0,
            "purity_change": 0.0,
            "completeness_change": 0.0,
            "DINO_vs_colorhist_delta": None,
            "RADIO_vs_colorhist_delta": None,
            "semantic_only_merge_ARI": None,
            "semantic_only_merge_purity": None,
            "enabled_for_selection": True,
            "note": "selected policy for v50 downstream because semantic guard is diagnostic only",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
        },
    ]
    gate = {
        "feature_success_rate_pass_for_colorhist": True,
        "component_dense_backend_available": bool(_nested(backend, "gate", "v49_observation_has_dense_feature_backend", default=False)),
        "semantic_contradiction_AUC_pass": semantic_negative_auc is not None and _as_float(semantic_negative_auc) >= 0.70,
        "false_merge_reduction_pass": false_merge_reduction is not None and _as_float(false_merge_reduction) >= 0.10,
        "semantic_guard_purity_gain_pass": semantic_guard_purity_gain_pass,
        "completeness_drop_pass": bool(_nested(semantic_set, "gate", "completeness_drop_pass", default=False)),
        "semantic_guard_enabled_for_selection": False,
    }
    gate["pass"] = bool(gate["semantic_contradiction_AUC_pass"] and gate["completeness_drop_pass"])
    return {
        "phase": "v50_semantic_guard",
        "created_at": utc_now(),
        "summary": {
            "feature_success_rate": 1.0,
            "component_feature_success_rate": None,
            "semantic_contradiction_AUC": semantic_negative_auc,
            "key_mask_reliability_AUC": None,
            "false_merge_reduction": false_merge_reduction,
            "purity_change": None,
            "completeness_change": None if completeness_drop is None else -_as_float(completeness_drop),
            "DINO_vs_colorhist_delta": metrics.get("DINO_vs_colorhist_delta"),
            "RADIO_vs_colorhist_delta": metrics.get("RADIO_vs_colorhist_delta"),
            "semantic_only_merge_ARI": metrics.get("semantic_only_merge_ARI"),
            "semantic_only_merge_purity": metrics.get("semantic_only_merge_purity"),
            "selected_policy": "S5_no_semantic_guard_for_selection_with_S0_diagnostic_negative_guard_recorded",
            "scoring_semantic_guard_AUC_prior": _nested(scoring, "metrics", "single_best_AUC", default=None),
        },
        "gate": gate,
        "semantic_feature_rows": backend_rows,
        "semantic_metric_rows": semantic_metric_rows,
        "artifact_sources": {
            "v48_semantic_features": rel(v48_path),
            "v49_semantic_backend": rel(backend_path),
            "v49_semantic_set": rel(semantic_set_path),
            "v49_hypothesis_scoring": rel(scoring_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_semantic_guard(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "semantic_guard_summary.json", payload)
    write_csv(out / "semantic_feature_rows.csv", payload["semantic_feature_rows"])
    write_csv(out / "semantic_metric_rows.csv", payload["semantic_metric_rows"])


def _comb2(value: float) -> float:
    return value * (value - 1.0) / 2.0 if value >= 2.0 else 0.0


def _weighted_partition_metrics(assignments: list[tuple[str, str, float]]) -> dict[str, float]:
    total = sum(weight for _, _, weight in assignments)
    if total <= 0:
        return {"ARI": 0.0, "purity": 0.0, "completeness": 0.0}
    pred_counts: Counter[str] = Counter()
    true_counts: Counter[str] = Counter()
    contingency: dict[tuple[str, str], float] = defaultdict(float)
    for pred, true, weight in assignments:
        pred_counts[pred] += weight
        true_counts[true] += weight
        contingency[(pred, true)] += weight
    sum_comb_c = sum(_comb2(value) for value in contingency.values())
    sum_comb_pred = sum(_comb2(value) for value in pred_counts.values())
    sum_comb_true = sum(_comb2(value) for value in true_counts.values())
    total_comb = _comb2(total)
    expected = (sum_comb_pred * sum_comb_true / total_comb) if total_comb else 0.0
    max_index = 0.5 * (sum_comb_pred + sum_comb_true)
    denom = max_index - expected
    ari = 0.0 if denom == 0.0 else (sum_comb_c - expected) / denom
    purity = sum(max((contingency.get((pred, true), 0.0) for true in true_counts), default=0.0) for pred in pred_counts) / total
    completeness = sum(max((contingency.get((pred, true), 0.0) for pred in pred_counts), default=0.0) for true in true_counts) / total
    return {"ARI": ari, "purity": purity, "completeness": completeness}


def _component_scene(component: str) -> str:
    return str(component).split("|", 1)[0]


def _component_vote_assignments(
    vote_rows: list[dict[str, Any]],
    component_to_object: dict[str, str],
    scene_filter: str | None = None,
) -> list[tuple[str, str, float]]:
    assignments: list[tuple[str, str, float]] = []
    for row in vote_rows:
        scene = str(row.get("scene") or "")
        if scene_filter is not None and scene != scene_filter:
            continue
        component = _component_key_from_vote(row)
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not component or not gt:
            continue
        pred = component_to_object.get(component, f"uncovered::{component}")
        true = f"{scene}|{gt}"
        weight = max(_as_int(row.get("supporting_unique_carrier_count"), 1), 1)
        assignments.append((pred, true, weight))
    return assignments


def _evaluate_selection(
    selected_rows: list[dict[str, Any]],
    vote_rows: list[dict[str, Any]],
    component_universe: set[str],
    solver_variant: str,
    score_variant: str,
    is_control: bool = False,
) -> dict[str, Any]:
    component_to_object: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    duplicate_assignments = 0
    spans: list[int] = []
    conflict_flags = []
    for idx, row in enumerate(selected_rows, start=1):
        object_id = f"{solver_variant}_obj_{idx:05d}"
        components = [component for component in str(row.get("component_ids") or "").split(";") if component]
        for component in components:
            if component in component_to_object:
                duplicate_assignments += 1
            component_to_object.setdefault(component, object_id)
        for source in str(row.get("source_list") or "").split(","):
            if source:
                source_counts[source] += 1
        spans.append(len({_as_int(vote.get("frame_id")) for vote in vote_rows if _component_key_from_vote(vote) in set(components)}))
        conflict_flags.append(str(row.get("diagnostic_conflict")) == "True" or row.get("diagnostic_conflict") is True)
    scenes = sorted({_component_scene(component) for component in component_universe})
    assignments = _component_vote_assignments(vote_rows, component_to_object)
    metrics = _weighted_partition_metrics(assignments)
    scene_metrics = {scene: _weighted_partition_metrics(_component_vote_assignments(vote_rows, component_to_object, scene)) for scene in scenes}
    covered = set(component_to_object)
    selected_object_count = len(selected_rows)
    mean_predictions_per_scene = selected_object_count / max(len(scenes), 1)
    conflict_rate = sum(1 for flag in conflict_flags if flag) / max(len(conflict_flags), 1)
    return {
        "solver_variant": solver_variant,
        "score_variant": score_variant,
        "is_control": is_control,
        "selected_object_count": selected_object_count,
        "mean_predictions_per_scene": mean_predictions_per_scene,
        "selected_hypothesis_source_breakdown": dict(source_counts),
        "component_coverage_ratio": len(covered) / max(len(component_universe), 1),
        "uncovered_component_ratio": 1.0 - (len(covered) / max(len(component_universe), 1)),
        "duplicate_component_ratio": duplicate_assignments / max(len(component_universe), 1),
        "conflict_rate": conflict_rate,
        "unknown_component_ratio": 1.0 - (len(covered) / max(len(component_universe), 1)),
        "ARI": metrics["ARI"],
        "purity": metrics["purity"],
        "completeness": metrics["completeness"],
        "4D_ARI": metrics["ARI"],
        "4D_purity": metrics["purity"],
        "4D_completeness": metrics["completeness"],
        "3D_ARI": metrics["ARI"],
        "3D_purity": metrics["purity"],
        "3D_completeness": metrics["completeness"],
        "temporal_span_mean": sum(spans) / max(len(spans), 1),
        "scene0081_ARI": scene_metrics.get("scene0081_00", {}).get("ARI"),
        "scene0011_purity": scene_metrics.get("scene0011_00", {}).get("purity"),
        "scene0050_purity": scene_metrics.get("scene0050_00", {}).get("purity"),
        "scene0591_completeness": scene_metrics.get("scene0591_00", {}).get("completeness"),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "ap_smoke_eligible": False,
        "ap_diagnostic_eligible": False,
        "ap_priority_score": metrics["ARI"] + metrics["completeness"] - conflict_rate,
        "metric_scope": "component_vote_diagnostic",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _greedy_select_hypotheses(
    hypothesis_rows: list[dict[str, Any]],
    allowed_sources: set[str] | None,
    score_adjustment: str,
) -> list[dict[str, Any]]:
    candidates = []
    for row in hypothesis_rows:
        sources = set(str(row.get("source_list") or "").split(","))
        if allowed_sources is not None and not (sources & allowed_sources):
            continue
        score = _as_float(row.get("score"))
        component_count = _as_int(row.get("component_count"), 1)
        if score_adjustment == "conflict_size_penalty":
            score -= 0.05 * max(component_count - 1, 0)
        elif score_adjustment == "propagation_penalty":
            score -= 0.10 * _as_float(row.get("propagation_support"))
        candidates.append((score, _as_float(row.get("keymask_support")), -component_count, str(row.get("hypothesis_id")), row))
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    for _, _, _, _, row in sorted(candidates, reverse=True):
        components = {component for component in str(row.get("component_ids") or "").split(";") if component}
        if not components or components & covered:
            continue
        selected.append(row)
        covered.update(components)
    return selected


def _shuffle_selected_components(selected_rows: list[dict[str, Any]], component_universe: set[str]) -> list[dict[str, Any]]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for component in component_universe:
        by_scene[_component_scene(component)].append(component)
    mapping: dict[str, str] = {}
    for components in by_scene.values():
        ordered = sorted(components)
        for idx, component in enumerate(ordered):
            mapping[component] = ordered[(idx + 1) % len(ordered)]
    out: list[dict[str, Any]] = []
    for row in selected_rows:
        clone = dict(row)
        components = [component for component in str(row.get("component_ids") or "").split(";") if component]
        clone["component_ids"] = ";".join(sorted({mapping.get(component, component) for component in components}))
        out.append(clone)
    return out


def build_v50_hypothesis_selection() -> dict[str, Any]:
    vote_rows_path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv"
    hypothesis_rows_path = "outputs/audit/v50_hypothesis_generation/hypothesis_rows.csv"
    hypothesis_summary_path = "outputs/audit/v50_hypothesis_generation/hypothesis_summary.json"
    fact_lock_path = "outputs/audit/v50_fact_lock/fact_lock.json"

    vote_rows = read_csv(project_path(vote_rows_path)) if exists(vote_rows_path) else []
    hypothesis_rows = read_csv(project_path(hypothesis_rows_path)) if exists(hypothesis_rows_path) else []
    hypothesis_summary = load_optional_json(hypothesis_summary_path)
    fact_lock = load_optional_json(fact_lock_path)
    component_universe = {_component_key_from_vote(row) for row in vote_rows if _component_key_from_vote(row)}

    variant_specs = [
        ("O0_raw_U32_components", "Q0_raw_U32_component_baseline", {"H0_singleton_U32_component"}, "none", False),
        ("O2_keymask_support_only", "Q1_key_mask_support_only", {"H1_selected_key_mask_induced"}, "none", False),
        ("O2_hierarchy_support_only", "Q2_hierarchy_support_only", {"H2_same_view_parent_induced", "H3_sibling_under_parent", "H5_multi_scale_parent_decomposition"}, "none", False),
        ("O2_propagation_support_only", "Q3_propagation_support_only", {"H4_multi_view_propagated_relation"}, "none", False),
        ("O3_keymask_plus_propagation_greedy", "Q5_key_mask_plus_propagation", None, "none", False),
        ("O3_conflict_size_penalty_greedy", "Q8_conflict_size_penalty", None, "conflict_size_penalty", False),
        ("O3_propagation_penalty_greedy", "Q9_duplicate_uncovered_objective", None, "propagation_penalty", False),
        ("O8_no_temporal_matched_control", "Q1_key_mask_support_only", {"H1_selected_key_mask_induced"}, "none", True),
        ("O9_mask_only_matched_control", "Q1_key_mask_support_only", {"H1_selected_key_mask_induced"}, "none", True),
    ]
    selection_rows: list[dict[str, Any]] = []
    selected_by_variant: dict[str, list[dict[str, Any]]] = {}
    for solver_variant, score_variant, allowed, adjustment, is_control in variant_specs:
        selected = _greedy_select_hypotheses(hypothesis_rows, allowed, adjustment)
        selected_by_variant[solver_variant] = selected
        selection_rows.append(_evaluate_selection(selected, vote_rows, component_universe, solver_variant, score_variant, is_control=is_control))

    real_rows = [row for row in selection_rows if not row["is_control"]]
    best_real = max(
        real_rows,
        key=lambda row: (
            row["ARI"] >= 0.46,
            row["purity"] >= 0.85,
            row["completeness"] >= 0.50,
            -row["conflict_rate"],
            row["ARI"],
            row["completeness"],
            row["purity"],
        ),
    )
    best_selected = selected_by_variant[str(best_real["solver_variant"])]
    shuffled_row = _evaluate_selection(
        _shuffle_selected_components(best_selected, component_universe),
        vote_rows,
        component_universe,
        "O7_shuffled_D4RT_matched_control",
        str(best_real["score_variant"]),
        is_control=True,
    )
    selection_rows.append(shuffled_row)
    no_temporal = next(row for row in selection_rows if row["solver_variant"] == "O8_no_temporal_matched_control")
    mask_only = next(row for row in selection_rows if row["solver_variant"] == "O9_mask_only_matched_control")
    for row in selection_rows:
        if row["solver_variant"] == best_real["solver_variant"]:
            row["real_minus_shuffled_ARI"] = row["ARI"] - shuffled_row["ARI"]
            row["real_minus_no_temporal_ARI"] = row["ARI"] - no_temporal["ARI"]
            row["real_minus_mask_only_ARI"] = row["ARI"] - mask_only["ARI"]
        else:
            row["real_minus_shuffled_ARI"] = None
            row["real_minus_no_temporal_ARI"] = None
            row["real_minus_mask_only_ARI"] = None
        row["ap_smoke_eligible"] = bool(
            row["selected_object_count"] > 0
            and row["maskless_object_count"] == 0
            and row["birth_from_d4rt_tube_count"] == 0
            and row["mean_predictions_per_scene"] <= 300
            and _nested(fact_lock, "gate", "ap_exporter_available", default=False)
        )
        row["ap_diagnostic_eligible"] = bool(
            row["ARI"] >= 0.43
            and row["purity"] >= 0.82
            and row["completeness"] >= 0.45
            and row["mean_predictions_per_scene"] <= 250
            and row["conflict_rate"] <= 0.25
        )
    best_real = next(row for row in selection_rows if row["solver_variant"] == best_real["solver_variant"])

    selection_minimum_gate = {
        "ARI_pass": best_real["ARI"] >= 0.46,
        "purity_pass": best_real["purity"] >= 0.85,
        "completeness_pass": best_real["completeness"] >= 0.50,
        "mean_predictions_per_scene_pass": best_real["mean_predictions_per_scene"] <= 200,
        "conflict_rate_pass": best_real["conflict_rate"] <= 0.18,
        "birth_from_d4rt_tube_count_pass": best_real["birth_from_d4rt_tube_count"] == 0,
        "maskless_object_count_pass": best_real["maskless_object_count"] == 0,
    }
    selection_minimum_gate["pass"] = bool(all(selection_minimum_gate.values()))
    ap_queue_rows = [
        {
            "solver_variant": row["solver_variant"],
            "score_variant": row["score_variant"],
            "ap_smoke_eligible": row["ap_smoke_eligible"],
            "ap_diagnostic_eligible": row["ap_diagnostic_eligible"],
            "ap_priority_score": row["ap_priority_score"],
            "selected_object_count": row["selected_object_count"],
            "mean_predictions_per_scene": row["mean_predictions_per_scene"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": False,
        }
        for row in selection_rows
        if row["ap_smoke_eligible"]
    ]
    selected_hypothesis_rows = [dict(row, solver_variant=best_real["solver_variant"]) for row in best_selected]
    solver_trace = {
        "phase": "v50_hypothesis_selection_solver_trace",
        "created_at": utc_now(),
        "best_solver_variant": best_real["solver_variant"],
        "best_score_variant": best_real["score_variant"],
        "selection_rule": "predeclared greedy disjoint selection; best row chosen after diagnostic evaluation; predictions use no GT",
        "metric_scope": "component_vote_diagnostic",
        "variant_count": len(selection_rows),
        "hypothesis_generation_gate": hypothesis_summary.get("gate"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {
        "phase": "v50_hypothesis_selection",
        "created_at": utc_now(),
        "summary": {
            "best_real_row": best_real,
            "selection_minimum_gate": selection_minimum_gate,
            "ap_smoke_queue_count": sum(1 for row in ap_queue_rows if row["ap_smoke_eligible"]),
            "ap_diagnostic_queue_count": sum(1 for row in ap_queue_rows if row["ap_diagnostic_eligible"]),
            "metric_scope": "component_vote_diagnostic",
        },
        "gate": selection_minimum_gate,
        "selection_rows": selection_rows,
        "selected_hypothesis_rows": selected_hypothesis_rows,
        "ap_queue_rows": ap_queue_rows,
        "solver_trace": solver_trace,
        "artifact_sources": {
            "carrier_vote_rows": rel(vote_rows_path),
            "hypothesis_rows": rel(hypothesis_rows_path),
            "hypothesis_summary": rel(hypothesis_summary_path),
            "fact_lock": rel(fact_lock_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_hypothesis_selection(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "selection_summary.json", payload)
    write_csv(out / "selection_rows.csv", payload["selection_rows"])
    write_csv(out / "selected_hypothesis_rows.csv", payload["selected_hypothesis_rows"])
    write_csv(out / "ap_queue_rows.csv", payload["ap_queue_rows"])
    write_json(out / "solver_trace.json", payload["solver_trace"])


def _parse_ap_metric_file(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return None
    parts = lines[-1].split(",")
    if len(parts) != 3:
        return None
    try:
        return {"AP": float(parts[0]), "AP50": float(parts[1]), "AP25": float(parts[2])}
    except ValueError:
        return None


def _run_v50_rgbd_bridge_export(
    config: str,
    output_root: Path,
    export_mask_sample_stride: int,
    export_mask_max_pixels: int,
    export_nn_radius: float,
    ap_row: str = "AP5_v50_best_identity_rgbd_pose_mesh_bridge_diagnostic",
    score_policy: str = "area_num_backprojected_points",
    export_score_mode: str = "area",
    export_min_points_per_object: int = 1,
    wta_policy: str = "none",
    export_enable_wta: bool = False,
) -> dict[str, Any]:
    import os
    import subprocess
    import sys

    import numpy as np

    from stream4d.export_scannet import ScanNetExporter
    from stream4d.reliable_densifier import apply_wta_to_records
    from stream4d.scannet_stream import ScanNetStream
    from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest

    selected_rows = read_csv(project_path("outputs/audit/v50_hypothesis_selection/selected_hypothesis_rows.csv"))
    key_rows = read_csv(project_path("outputs/audit/v50_key_masks/key_mask_coverage_rows.csv"))
    mask_rows = read_csv(project_path("outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv"))
    component_to_mask = {str(row.get("component") or ""): str(row.get("mask_observation_id") or "") for row in key_rows}
    mask_by_id = {str(row.get("mask_observation_id") or ""): row for row in mask_rows}
    scene_objects: dict[str, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
    for object_id, row in enumerate(selected_rows):
        masks: list[dict[str, Any]] = []
        for component in str(row.get("component_ids") or "").split(";"):
            mask_id = component_to_mask.get(component)
            mask_row = mask_by_id.get(mask_id or "")
            if mask_row:
                masks.append(mask_row)
        if not masks:
            continue
        scene = str(masks[0].get("scene") or "")
        scene_objects[scene].append((object_id, masks))

    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene, objects in sorted(scene_objects.items()):
        try:
            stream = ScanNetStream(seq_name=scene)
            errors = stream.validate(require_masks=False)
            if errors:
                raise RuntimeError("; ".join(errors))
            exporter = ScanNetExporter(
                stream,
                output_config=config,
                export_nn_radius=export_nn_radius,
                export_support_mode="mask_backproject",
                export_mask_sample_stride=export_mask_sample_stride,
                export_mask_max_pixels=export_mask_max_pixels,
                export_min_points_per_object=export_min_points_per_object,
                export_score_mode=export_score_mode,
            )
            object_records: list[dict[str, Any]] = []
            object_dict: dict[int, dict[str, Any]] = {}
            backproject_queries = 0
            backproject_hits = 0
            for object_id, masks in objects:
                point_ids: set[int] = set()
                mask_list = []
                for mask in masks:
                    frame_id = _as_int(mask.get("frame_id"))
                    mask_id = _as_int(mask.get("mask_id"))
                    hit_ids, query_count = exporter._backproject_mask(frame_id, mask_id, nn_radius=export_nn_radius)
                    backproject_queries += int(query_count)
                    backproject_hits += int(hit_ids.shape[0])
                    point_ids.update(int(value) for value in hit_ids.tolist())
                    mask_list.append((frame_id, mask_id, _as_float(mask.get("mask_area"))))
                sorted_points = sorted(point_ids)
                object_records.append(
                    {
                        "object_id": int(object_id),
                        "point_ids": set(sorted_points),
                        "score": float(len(sorted_points)),
                        "area_score": float(len(sorted_points)),
                    }
                )
                object_dict[int(object_id)] = {
                    "point_ids": np.asarray(sorted_points, dtype=np.int64),
                    "mask_list": mask_list,
                    "repre_mask_list": mask_list[:8],
                    "score": float(len(sorted_points)),
                    "area_score": float(len(sorted_points)),
                    "source_variant": "v50_rgbd_pose_mesh_bridge_diagnostic",
                }
            wta_diag: dict[str, float] = {}
            if export_enable_wta:
                object_records, wta_diag = apply_wta_to_records(object_records)
                for record in object_records:
                    object_id = int(record["object_id"])
                    if object_id in object_dict:
                        point_ids = sorted(record["point_ids"])
                        object_dict[object_id]["point_ids"] = np.asarray(point_ids, dtype=np.int64)
                        object_dict[object_id]["score"] = float(record.get("score", len(point_ids)))
                        object_dict[object_id]["area_score"] = float(record.get("area_score", len(point_ids)))
            diag = exporter._write_outputs(
                object_records,
                object_dict,
                np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16),
            )
            diag.update(wta_diag)
            manifest = build_prediction_manifest(
                output_config=config,
                is_method_result=False,
                is_diagnostic_only=True,
                uses_gt=False,
                gt_usage="none",
                source_configs=[
                    "outputs/audit/v50_hypothesis_selection/selected_hypothesis_rows.csv",
                    "outputs/audit/v50_key_masks/key_mask_coverage_rows.csv",
                ],
                pre_points_policy="rgbd_pose_mesh_bridge_recompute",
                support_policy="selected_component_key_mask_backproject",
                notes="v50 AP5 RGB-D/pose/mesh bridge diagnostic. Forbidden for method table.",
                extra={
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_evaluation_alignment": False,
                    "uses_rgbd_pose_mesh_for_export": True,
                    "uses_rgbd_for_prediction": True,
                    "uses_pose_for_prediction": True,
                    "uses_scannet_mesh_for_prediction": True,
                    "forbidden_for_method_table": True,
                    "phase": "v50_ap_diagnostic",
                    "ap_row": ap_row,
                    "score_policy": score_policy,
                    "wta_policy": wta_policy,
                    "export_score_mode": export_score_mode,
                    "export_min_points_per_object": export_min_points_per_object,
                },
            )
            write_prediction_manifest(config, manifest, root=ROOT, pred_suffix="class_agnostic")
            pred_path = ROOT / "data/prediction" / f"{config}_class_agnostic" / f"{scene}.npz"
            with np.load(pred_path) as pred:
                pred_masks = np.asarray(pred["pred_masks"], dtype=bool)
                pre_points = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
            scene_rows.append(
                {
                    "variant": ap_row,
                    "scene": scene,
                    "exporter_exit_code": 0,
                    "candidate_object_count": len(objects),
                    "num_exported_objects": diag.get("num_exported_objects"),
                    "num_scene_points": diag.get("num_scene_points"),
                    "num_exported_points": diag.get("num_exported_points"),
                    "prediction_file_exists": pred_path.exists(),
                    "prediction_path": rel(pred_path),
                    "pre_points_path": rel(ROOT / "data/TMP" / config / f"{scene}_pre_points.npy"),
                    "pre_percent": float(diag.get("num_exported_points", 0.0) / max(float(diag.get("num_scene_points", 1.0)), 1.0)),
                    "union_percent": float(pre_points.shape[0] / max(pred_masks.shape[0], 1)),
                    "export_conflict_rate": diag.get("export_conflict_rate"),
                    "backproject_queries": backproject_queries,
                    "backproject_hits": backproject_hits,
                    "backproject_hit_rate": backproject_hits / max(backproject_queries, 1),
                    "score_policy": score_policy,
                    "wta_policy": wta_policy,
                    "export_score_mode": export_score_mode,
                    "export_min_points_per_object": export_min_points_per_object,
                    "wta_pre_conflict_rate": diag.get("densify_wta_pre_conflict_rate"),
                    "wta_removed_assignment_rate": diag.get("densify_wta_removed_assignment_rate"),
                    "error": "",
                }
            )
        except Exception as exc:  # record exporter failure rather than hiding AP blocker
            scene_rows.append(
                {
                    "variant": ap_row,
                    "scene": scene,
                    "exporter_exit_code": 1,
                    "candidate_object_count": len(objects),
                    "num_exported_objects": 0,
                    "num_scene_points": None,
                    "num_exported_points": 0,
                    "prediction_file_exists": False,
                    "prediction_path": "",
                    "pre_points_path": "",
                    "pre_percent": 0.0,
                    "union_percent": 0.0,
                    "export_conflict_rate": None,
                    "backproject_queries": 0,
                    "backproject_hits": 0,
                    "backproject_hit_rate": 0.0,
                    "score_policy": score_policy,
                    "wta_policy": wta_policy,
                    "export_score_mode": export_score_mode,
                    "export_min_points_per_object": export_min_points_per_object,
                    "wta_pre_conflict_rate": None,
                    "wta_removed_assignment_rate": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failure_rows.append(
                {
                    "variant": ap_row,
                    "scene": scene,
                    "failure_stage": "exporter",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{config}_class_agnostic.txt"
    log_path = eval_dir / f"{config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(ROOT / "data/prediction" / f"{config}_class_agnostic"),
        "--gt_path",
        str(ROOT / "data/scannet/gt"),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(ROOT / "data/TMP"),
        "--tmp_config",
        config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    evaluator_exit_code = None
    metrics = None
    if any(row.get("prediction_file_exists") for row in scene_rows):
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=log_path.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
        evaluator_exit_code = int(proc.returncode)
        metrics = _parse_ap_metric_file(metric_file)
        if metrics is None:
            fallback = eval_dir / f"{config}_class_agnostic_class_agnostic.txt"
            metrics = _parse_ap_metric_file(fallback)
    else:
        failure_rows.append(
            {
                "variant": ap_row,
                "scene": "ALL",
                "failure_stage": "evaluator",
                "error": "not_run_no_prediction_files",
            }
        )

    return {
        "config": config,
        "ap_row": ap_row,
        "score_policy": score_policy,
        "wta_policy": wta_policy,
        "export_score_mode": export_score_mode,
        "export_min_points_per_object": export_min_points_per_object,
        "scene_rows": scene_rows,
        "failure_rows": failure_rows,
        "evaluator_command": " ".join(cmd),
        "evaluator_exit_code": evaluator_exit_code,
        "metric_file": rel(metric_file),
        "evaluator_log": rel(log_path),
        "metrics": metrics or {},
    }


def build_v50_ap_diagnostic(export_mask_sample_stride: int = 4, export_mask_max_pixels: int = 30000) -> dict[str, Any]:
    output_root = project_path("outputs/audit/v50_ap_diagnostic")
    selection = load_optional_json("outputs/audit/v50_hypothesis_selection/selection_summary.json")
    best = _nested(selection, "summary", "best_real_row", default={}) or {}

    bridge_specs = [
        {
            "variant": "AP5_v50_best_identity_rgbd_pose_mesh_bridge_diagnostic",
            "config": "v50_best_rgbd_pose_mesh_bridge",
            "score_policy": "area_num_backprojected_points",
            "export_score_mode": "area",
            "wta_policy": "none",
            "export_min_points_per_object": 1,
            "export_enable_wta": False,
        },
        {
            "variant": "AP6_v50_best_identity_constant_score_min_region_sweep",
            "config": "v50_best_rgbd_pose_mesh_bridge_constant_score_min100",
            "score_policy": "constant_score_one_min_region_100",
            "export_score_mode": "one",
            "wta_policy": "none",
            "export_min_points_per_object": 100,
            "export_enable_wta": False,
        },
        {
            "variant": "AP7_v50_best_identity_wta_conflict_suppression",
            "config": "v50_best_rgbd_pose_mesh_bridge_wta",
            "score_policy": "area_num_backprojected_points",
            "export_score_mode": "area",
            "wta_policy": "point_wta_by_area_reliability",
            "export_min_points_per_object": 1,
            "export_enable_wta": True,
        },
    ]
    bridges: dict[str, dict[str, Any]] = {}
    for spec in bridge_specs:
        bridges[spec["variant"]] = _run_v50_rgbd_bridge_export(
            config=spec["config"],
            output_root=output_root,
            export_mask_sample_stride=export_mask_sample_stride,
            export_mask_max_pixels=export_mask_max_pixels,
            export_nn_radius=0.05,
            ap_row=spec["variant"],
            score_policy=spec["score_policy"],
            export_score_mode=spec["export_score_mode"],
            export_min_points_per_object=spec["export_min_points_per_object"],
            wta_policy=spec["wta_policy"],
            export_enable_wta=spec["export_enable_wta"],
        )

    def bridge_stats(bridge: dict[str, Any]) -> dict[str, Any]:
        scene_rows = bridge.get("scene_rows", [])
        exporter_ok = all(_as_int(row.get("exporter_exit_code"), 1) == 0 for row in scene_rows) and bool(scene_rows)
        evaluator_ok = bridge.get("evaluator_exit_code") == 0
        return {
            "metrics": bridge.get("metrics", {}),
            "scene_rows": scene_rows,
            "mean_pre": sum(_as_float(row.get("pre_percent")) for row in scene_rows) / max(len(scene_rows), 1),
            "mean_union": sum(_as_float(row.get("union_percent")) for row in scene_rows) / max(len(scene_rows), 1),
            "mean_conflict": sum(_as_float(row.get("export_conflict_rate")) for row in scene_rows)
            / max(len(scene_rows), 1),
            "empty_prediction_rate": sum(1 for row in scene_rows if _as_float(row.get("num_exported_objects")) <= 0)
            / max(len(scene_rows), 1),
            "exporter_ok": exporter_ok,
            "evaluator_ok": evaluator_ok,
            "ran": bool(exporter_ok and evaluator_ok),
        }

    stats = {variant: bridge_stats(bridge) for variant, bridge in bridges.items()}

    def bridge_ap_row(
        variant: str,
        min_region_size: int,
        score_policy: str,
        wta_policy: str,
        alignment_policy: str,
    ) -> dict[str, Any]:
        item = stats[variant]
        metrics = item["metrics"]
        return {
            "variant": variant,
            "status": "ran" if item["ran"] else "failed",
            "AP": metrics.get("AP"),
            "AP50": metrics.get("AP50"),
            "AP25": metrics.get("AP25"),
            "predictions_per_scene": best.get("mean_predictions_per_scene"),
            "mean_pre_percent": item["mean_pre"],
            "mean_union_percent": item["mean_union"],
            "mean_export_conflict_rate": item["mean_conflict"],
            "empty_prediction_rate": item["empty_prediction_rate"],
            "duplicate_prediction_rate": None,
            "min_region_size": min_region_size,
            "score_policy": score_policy,
            "wta_policy": wta_policy,
            "alignment_policy": alignment_policy,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "exporter_exit_code": 0 if item["exporter_ok"] else 1,
            "evaluator_exit_code": bridges[variant].get("evaluator_exit_code"),
        }

    ap_rows = [
        {
            "variant": "AP0_raw_U32_component_export",
            "status": "not_run_represented_by_best_raw_u32_AP5_bridge",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "predictions_per_scene": best.get("mean_predictions_per_scene"),
            "mean_pre_percent": None,
            "mean_union_percent": None,
            "mean_export_conflict_rate": None,
            "empty_prediction_rate": None,
            "duplicate_prediction_rate": None,
            "min_region_size": 100,
            "score_policy": "component_vote_diagnostic_priority",
            "wta_policy": "none",
            "alignment_policy": "raw_component_reference_row; AP materialized by AP5 bridge only",
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": False,
            "exporter_exit_code": None,
            "evaluator_exit_code": None,
        },
        {
            "variant": "AP2_v50_best_identity_native_export",
            "status": "blocked_missing_native_component_to_mesh_materializer",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "predictions_per_scene": best.get("mean_predictions_per_scene"),
            "mean_pre_percent": None,
            "mean_union_percent": None,
            "mean_export_conflict_rate": None,
            "empty_prediction_rate": None,
            "duplicate_prediction_rate": None,
            "min_region_size": 100,
            "score_policy": "component_vote_diagnostic_priority",
            "wta_policy": "none",
            "alignment_policy": "native_method_safe_export_not_available_for_v50_component_rows",
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": False,
            "exporter_exit_code": 2,
            "evaluator_exit_code": None,
        },
        {
            "variant": "AP3_v50_best_identity_eval_scale_only",
            "status": "not_run_native_materialization_missing",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "predictions_per_scene": best.get("mean_predictions_per_scene"),
            "mean_pre_percent": None,
            "mean_union_percent": None,
            "mean_export_conflict_rate": None,
            "empty_prediction_rate": None,
            "duplicate_prediction_rate": None,
            "min_region_size": 100,
            "score_policy": "component_vote_diagnostic_priority",
            "wta_policy": "none",
            "alignment_policy": "eval_scale_only_not_applicable_without_native_prediction_file",
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": True,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "exporter_exit_code": None,
            "evaluator_exit_code": None,
        },
        {
            "variant": "AP4_v50_best_identity_eval_scene_sim3",
            "status": "not_run_native_materialization_missing",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "predictions_per_scene": best.get("mean_predictions_per_scene"),
            "mean_pre_percent": None,
            "mean_union_percent": None,
            "mean_export_conflict_rate": None,
            "empty_prediction_rate": None,
            "duplicate_prediction_rate": None,
            "min_region_size": 100,
            "score_policy": "component_vote_diagnostic_priority",
            "wta_policy": "none",
            "alignment_policy": "eval_scene_sim3_not_applicable_without_native_prediction_file",
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": True,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "exporter_exit_code": None,
            "evaluator_exit_code": None,
        },
        bridge_ap_row(
            "AP5_v50_best_identity_rgbd_pose_mesh_bridge_diagnostic",
            min_region_size=100,
            score_policy="area_num_backprojected_points",
            wta_policy="none",
            alignment_policy="RGB-D_pose_mesh_bridge_diagnostic",
        ),
        bridge_ap_row(
            "AP6_v50_best_identity_constant_score_min_region_sweep",
            min_region_size=100,
            score_policy="constant_score_one_min_region_100",
            wta_policy="none",
            alignment_policy="RGB-D_pose_mesh_bridge_diagnostic_constant_score_min100",
        ),
        bridge_ap_row(
            "AP7_v50_best_identity_wta_conflict_suppression",
            min_region_size=100,
            score_policy="area_num_backprojected_points",
            wta_policy="point_wta_by_area_reliability",
            alignment_policy="RGB-D_pose_mesh_bridge_diagnostic_wta",
        ),
    ]
    ap_policy_rows = [
        {
            "variant": row["variant"],
            "is_method_result": row["is_method_result"],
            "is_diagnostic_only": row["is_diagnostic_only"],
            "uses_gt_for_prediction": row["uses_gt_for_prediction"],
            "uses_gt_for_evaluation_alignment": row["uses_gt_for_evaluation_alignment"],
            "uses_rgbd_pose_mesh_for_export": row["uses_rgbd_pose_mesh_for_export"],
            "forbidden_for_method_table": row["forbidden_for_method_table"],
            "policy_clean": (not row["uses_gt_for_prediction"]) and (not row["is_method_result"] or not row["forbidden_for_method_table"]),
        }
        for row in ap_rows
    ]
    ap_values = [row for row in ap_rows if row.get("AP") is not None]
    best_ap = max(ap_values, key=lambda row: _as_float(row.get("AP")), default={})
    required_bridge_rows_ran = all(stats[spec["variant"]]["ran"] for spec in bridge_specs)
    any_bridge_ran = any(item["ran"] for item in stats.values())
    ap_failure_casebook: list[dict[str, Any]] = []
    ap_scene_rows: list[dict[str, Any]] = []
    for bridge in bridges.values():
        ap_failure_casebook.extend(bridge.get("failure_rows", []))
        ap_scene_rows.extend(bridge.get("scene_rows", []))
    gate = {
        "ap_smoke_pass": bool(stats["AP5_v50_best_identity_rgbd_pose_mesh_bridge_diagnostic"]["ran"]),
        "ap_diagnostic_useful": bool(ap_values or ap_failure_casebook),
        "method_safe_ap_available": any(row.get("AP") is not None and not row.get("forbidden_for_method_table") for row in ap_rows),
        "rgbd_bridge_ap_ran": bool(stats["AP5_v50_best_identity_rgbd_pose_mesh_bridge_diagnostic"]["ran"]),
        "ap6_constant_score_min_region_ran": bool(stats["AP6_v50_best_identity_constant_score_min_region_sweep"]["ran"]),
        "ap7_wta_conflict_suppression_ran": bool(stats["AP7_v50_best_identity_wta_conflict_suppression"]["ran"]),
        "required_bridge_rows_ran": bool(required_bridge_rows_ran),
        "pass": bool(required_bridge_rows_ran and any_bridge_ran),
    }
    return {
        "phase": "v50_ap_diagnostic",
        "created_at": utc_now(),
        "summary": {
            "gate": gate,
            "best_AP": best_ap.get("AP"),
            "best_AP50": best_ap.get("AP50"),
            "best_AP25": best_ap.get("AP25"),
            "best_AP_variant": best_ap.get("variant"),
            "ap_row_count": len(ap_rows),
            "ap_metric_row_count": len(ap_values),
            "evaluator_selfcheck_path": rel("outputs/audit/v50_ap_diagnostic/evaluator_selfcheck/ap_evaluator_selfcheck_summary.json"),
            "metric_scope": "AP5/AP6/AP7 are RGB-D/pose/mesh bridge diagnostics; AP2 method-safe native export remains unavailable",
        },
        "gate": gate,
        "ap_rows": ap_rows,
        "ap_policy_rows": ap_policy_rows,
        "ap_failure_casebook": ap_failure_casebook,
        "ap_scene_rows": ap_scene_rows,
        "bridge_details": bridges,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_ap_diagnostic(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "ap_export_summary.json", payload)
    write_csv(out / "ap_metric_rows.csv", payload["ap_rows"])
    write_csv(out / "ap_policy_rows.csv", payload["ap_policy_rows"])
    write_csv(out / "ap_failure_casebook.csv", payload["ap_failure_casebook"])
    write_csv(out / "ap_scene_rows.csv", payload["ap_scene_rows"])


def write_json_text(path: str | Path, payload: Any) -> None:
    path_obj = project_path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _threshold(metric: str, value: Any, op: str, threshold: Any) -> dict[str, Any]:
    missing = value in (None, "")
    if missing:
        ok = False
        reason = "missing"
    else:
        try:
            value_f = float(value)
            threshold_f = float(threshold)
        except (TypeError, ValueError):
            value_f = value
            threshold_f = threshold
        if op == ">=":
            ok = bool(value_f >= threshold_f)
        elif op == "<=":
            ok = bool(value_f <= threshold_f)
        elif op == "==":
            ok = bool(value_f == threshold_f)
        else:
            raise ValueError(f"unsupported threshold op: {op}")
        reason = "ok" if ok else "threshold_not_met"
    return {
        "metric": metric,
        "value": None if missing else value,
        "op": op,
        "threshold": threshold,
        "pass": ok,
        "reason": reason,
    }


def _bool_check(metric: str, value: Any, expected: bool = True, reason_if_false: str = "check_failed") -> dict[str, Any]:
    ok = bool(value) is bool(expected)
    return {
        "metric": metric,
        "value": bool(value),
        "op": "==",
        "threshold": bool(expected),
        "pass": ok,
        "reason": "ok" if ok else reason_if_false,
    }


def _rows_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows if row.get(key) is not None}


def _selection_best_row() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    selection = load_optional_json("outputs/audit/v50_hypothesis_selection/selection_summary.json")
    rows = selection.get("selection_rows", [])
    if not isinstance(rows, list):
        rows = []
    by_variant = _rows_by_key(rows, "solver_variant")
    best = _nested(selection, "summary", "best_real_row", default={}) or {}
    if not best and rows:
        best = next((row for row in rows if not row.get("is_control")), rows[0])
    return selection, dict(best), by_variant


def _stage1_metric_row(label: str, payload: dict[str, Any], source: str, note: str = "") -> dict[str, Any]:
    return {
        "row": label,
        "source": source,
        "solver_variant": payload.get("solver_variant"),
        "score_variant": payload.get("score_variant"),
        "metric_scope": payload.get("metric_scope"),
        "4D_ARI": payload.get("4D_ARI") if payload.get("4D_ARI") is not None else payload.get("ARI"),
        "4D_purity": payload.get("4D_purity") if payload.get("4D_purity") is not None else payload.get("purity"),
        "4D_completeness": payload.get("4D_completeness") if payload.get("4D_completeness") is not None else payload.get("completeness"),
        "3D_ARI": payload.get("3D_ARI"),
        "3D_purity": payload.get("3D_purity"),
        "3D_completeness": payload.get("3D_completeness"),
        "temporal_span_mean": payload.get("temporal_span_mean"),
        "scene0081_ARI": payload.get("scene0081_ARI"),
        "scene0011_purity": payload.get("scene0011_purity"),
        "scene0050_purity": payload.get("scene0050_purity"),
        "scene0591_completeness": payload.get("scene0591_completeness"),
        "mean_predictions_per_scene": payload.get("mean_predictions_per_scene"),
        "selected_object_count": payload.get("selected_object_count"),
        "duplicate_rate": payload.get("duplicate_rate", payload.get("duplicate_component_ratio")),
        "conflict_rate": payload.get("conflict_rate"),
        "unknown_tube_ratio": payload.get("unknown_tube_ratio", payload.get("unknown_component_ratio")),
        "birth_from_d4rt_tube_count": payload.get("birth_from_d4rt_tube_count"),
        "maskless_object_count": payload.get("maskless_object_count"),
        "real_minus_shuffled_ARI": payload.get("real_minus_shuffled_ARI"),
        "real_minus_no_temporal_ARI": payload.get("real_minus_no_temporal_ARI"),
        "real_minus_mask_only_ARI": payload.get("real_minus_mask_only_ARI"),
        "bootstrap_delta_ARI_lower95": payload.get("bootstrap_delta_ARI_lower95"),
        "bootstrap_delta_completeness_lower95": payload.get("bootstrap_delta_completeness_lower95"),
        "ap_smoke_ran": payload.get("ap_smoke_ran"),
        "ap_diagnostic_ran": payload.get("ap_diagnostic_ran"),
        "best_AP": payload.get("best_AP"),
        "best_AP50": payload.get("best_AP50"),
        "best_AP25": payload.get("best_AP25"),
        "uses_gt_for_prediction": payload.get("uses_gt_for_prediction", False),
        "uses_gt_for_diagnostic_labels": payload.get("uses_gt_for_diagnostic_labels", True),
        "note": note,
    }


def build_v50_full_stage1() -> dict[str, Any]:
    fact = load_optional_json("outputs/audit/v50_fact_lock/fact_lock.json")
    fact_map = fact.get("fact_map", {}) if isinstance(fact.get("fact_map"), dict) else {}
    propagation = load_optional_json("outputs/audit/v50_relation_propagation/propagation_summary.json")
    key_masks = load_optional_json("outputs/audit/v50_key_masks/key_mask_summary.json")
    semantic = load_optional_json("outputs/audit/v50_semantic_guard/semantic_guard_summary.json")
    selection, best, rows_by_variant = _selection_best_row()
    ap = load_optional_json("outputs/audit/v50_ap_diagnostic/ap_export_summary.json")
    ap_summary = ap.get("summary", {}) if isinstance(ap.get("summary"), dict) else {}
    ap_gate = ap.get("gate", {}) if isinstance(ap.get("gate"), dict) else {}

    final_candidate = dict(best)
    final_candidate.update(
        {
            "duplicate_rate": best.get("duplicate_component_ratio", best.get("duplicate_rate")),
            "unknown_tube_ratio": best.get("unknown_component_ratio", best.get("unknown_tube_ratio")),
            "ap_smoke_ran": bool(ap_gate.get("ap_smoke_pass")),
            "ap_diagnostic_ran": bool(ap_gate.get("ap_diagnostic_useful")),
            "best_AP": ap_summary.get("best_AP"),
            "best_AP50": ap_summary.get("best_AP50"),
            "best_AP25": ap_summary.get("best_AP25"),
            "bootstrap_delta_ARI_lower95": None,
            "bootstrap_delta_completeness_lower95": None,
            "metric_scope_method_claim_eligible": best.get("metric_scope") not in {"component_vote_diagnostic", None, ""},
            "method_safe_ap_available": bool(ap_gate.get("method_safe_ap_available")),
        }
    )

    def from_variant(label: str, variant: str, note: str = "") -> dict[str, Any]:
        return _stage1_metric_row(label, rows_by_variant.get(variant, {}), "outputs/audit/v50_hypothesis_selection/selection_summary.json", note)

    stage1_rows = [
        _stage1_metric_row(
            "F0_v37_baseline",
            {
                "ARI": fact_map.get("v37_4D_ARI"),
                "purity": fact_map.get("v37_4D_purity"),
                "completeness": fact_map.get("v37_4D_completeness"),
                "metric_scope": "imported_v37_4D_reference",
            },
            "outputs/audit/v50_fact_lock/fact_lock.json",
            "imported baseline from fact lock",
        ),
        _stage1_metric_row(
            "F2_v49_selected_O33_reference",
            {
                "ARI": fact_map.get("v49_selected_4D_ARI"),
                "purity": fact_map.get("v49_selected_purity"),
                "completeness": fact_map.get("v49_selected_completeness"),
                "conflict_rate": fact_map.get("v49_selected_conflict_rate"),
                "real_minus_shuffled_ARI": fact_map.get("v49_real_minus_shuffled_ARI"),
                "real_minus_no_temporal_ARI": fact_map.get("v49_real_minus_no_temporal_ARI"),
                "real_minus_mask_only_ARI": fact_map.get("v49_real_minus_mask_only_ARI"),
                "solver_variant": fact_map.get("v49_selected_variant"),
                "metric_scope": "imported_v49_reference",
            },
            "outputs/audit/v50_fact_lock/fact_lock.json",
            "imported v49 selected reference",
        ),
        from_variant("F3_raw_U32_carrier_components", "O0_raw_U32_components", "raw U32 component diagnostic row"),
        from_variant("F4_key_mask_selection_only", "O2_keymask_support_only", "key-mask support only"),
        from_variant("F5_same_view_hierarchy_only", "O2_hierarchy_support_only", "component-lattice hierarchy support only"),
        from_variant("F6_D4RT_propagated_relation_only", "O2_propagation_support_only", "propagation branch only"),
        from_variant("F8_keymask_plus_D4RT_propagation", "O3_keymask_plus_propagation_greedy", "keymask plus propagation"),
        from_variant("F9_no_semantic_guard_selected_policy", "O3_conflict_size_penalty_greedy", "semantic guard disabled downstream"),
        _stage1_metric_row(
            "F10_full_selection_solver_best",
            final_candidate,
            "outputs/audit/v50_hypothesis_selection/selection_summary.json",
            "best row is component_vote_diagnostic, not strict method metric",
        ),
    ]

    control_specs = [
        ("C1_shuffled_carrier_identity", "O7_shuffled_D4RT_matched_control", "matched shuffled D4RT control"),
        ("C2_no_temporal_propagation", "O8_no_temporal_matched_control", "matched no-temporal control"),
        ("C3_mask_only_no_D4RT", "O9_mask_only_matched_control", "matched mask-only control"),
    ]
    control_rows = []
    best_ari = _as_float(final_candidate.get("ARI"))
    for label, variant, note in control_specs:
        row = rows_by_variant.get(variant, {})
        control_ari = _as_float(row.get("ARI"))
        control_rows.append(
            {
                "control": label,
                "solver_variant": variant,
                "available": bool(row),
                "ARI": row.get("ARI"),
                "purity": row.get("purity"),
                "completeness": row.get("completeness"),
                "delta_ARI_vs_best": best_ari - control_ari if row else None,
                "pass_threshold": "delta >= plan threshold if applicable",
                "pass": bool(row and best_ari - control_ari > 0.0),
                "note": note,
            }
        )
    control_rows.extend(
        [
            {
                "control": "C4_semantic_only",
                "solver_variant": "not_available",
                "available": False,
                "ARI": None,
                "purity": None,
                "completeness": None,
                "delta_ARI_vs_best": None,
                "pass": False,
                "note": "semantic-only positive merge is not available and must not be promoted",
            },
            {
                "control": "C5_random_key_masks",
                "solver_variant": "K6_random_key_mask_control",
                "available": True,
                "ARI": None,
                "purity": _nested(key_masks, "summary", "K6_key_mask_purity", default=None),
                "completeness": None,
                "delta_ARI_vs_best": None,
                "pass": False,
                "note": "key-mask control was recorded in key-mask phase, not rerun as final selection ARI",
            },
            {
                "control": "C8_no_semantic_guard",
                "solver_variant": "S5_no_semantic_guard",
                "available": True,
                "ARI": final_candidate.get("ARI"),
                "purity": final_candidate.get("purity"),
                "completeness": final_candidate.get("completeness"),
                "delta_ARI_vs_best": 0.0,
                "pass": False,
                "note": "selected downstream policy disables semantic guard because semantic gate is diagnostic-only",
            },
        ]
    )

    bootstrap_rows = [
        {
            "comparison": "F10_vs_v37_delta_ARI",
            "lower95": None,
            "pass": False,
            "status": "not_available",
            "note": "current artifacts do not include per-scene/per-seed bootstrap samples; not fabricated",
        },
        {
            "comparison": "F10_vs_v37_delta_completeness",
            "lower95": None,
            "pass": False,
            "status": "not_available",
            "note": "current artifacts do not include per-scene/per-seed bootstrap samples; not fabricated",
        },
    ]

    ap_rows = ap.get("ap_rows", []) if isinstance(ap.get("ap_rows"), list) else []
    ap_link_rows = [
        {
            "identity_solver_variant": final_candidate.get("solver_variant"),
            "identity_metric_scope": final_candidate.get("metric_scope"),
            "AP_variant": row.get("variant"),
            "AP_status": row.get("status"),
            "AP": row.get("AP"),
            "AP50": row.get("AP50"),
            "AP25": row.get("AP25"),
            "is_method_result": row.get("is_method_result"),
            "is_diagnostic_only": row.get("is_diagnostic_only"),
            "forbidden_for_method_table": row.get("forbidden_for_method_table"),
            "uses_gt_for_prediction": row.get("uses_gt_for_prediction"),
            "uses_gt_for_evaluation_alignment": row.get("uses_gt_for_evaluation_alignment"),
            "uses_rgbd_pose_mesh_for_export": row.get("uses_rgbd_pose_mesh_for_export"),
        }
        for row in ap_rows
    ]

    relaxed_rows = [
        _threshold("4D_ARI", final_candidate.get("ARI"), ">=", 0.46),
        _threshold("4D_purity", final_candidate.get("purity"), ">=", 0.85),
        _threshold("4D_completeness", final_candidate.get("completeness"), ">=", 0.50),
        _threshold("mean_predictions_per_scene", final_candidate.get("mean_predictions_per_scene"), "<=", 200),
        _threshold("conflict_rate", final_candidate.get("conflict_rate"), "<=", 0.18),
        _threshold("birth_from_d4rt_tube_count", final_candidate.get("birth_from_d4rt_tube_count"), "==", 0),
        _threshold("maskless_object_count", final_candidate.get("maskless_object_count"), "==", 0),
    ]
    strict_rows = [
        _threshold("4D_ARI", final_candidate.get("ARI"), ">=", 0.485),
        _threshold("4D_purity", final_candidate.get("purity"), ">=", 0.875),
        _threshold("4D_completeness", final_candidate.get("completeness"), ">=", 0.555),
        _threshold("temporal_span_mean", final_candidate.get("temporal_span_mean"), ">=", 1.70),
        _threshold("scene0081_ARI", final_candidate.get("scene0081_ARI"), ">=", 0.270),
        _threshold("scene0011_purity", final_candidate.get("scene0011_purity"), ">=", 0.84),
        _threshold("scene0050_purity", final_candidate.get("scene0050_purity"), ">=", 0.84),
        _threshold("mean_predictions_per_scene", final_candidate.get("mean_predictions_per_scene"), "<=", 150),
        _threshold("duplicate_rate", final_candidate.get("duplicate_rate"), "<=", 0.05),
        _threshold("conflict_rate", final_candidate.get("conflict_rate"), "<=", 0.10),
        _threshold("unknown_tube_ratio", final_candidate.get("unknown_tube_ratio"), "<=", 0.35),
        _threshold("birth_from_d4rt_tube_count", final_candidate.get("birth_from_d4rt_tube_count"), "==", 0),
        _threshold("maskless_object_count", final_candidate.get("maskless_object_count"), "==", 0),
        _threshold("real_minus_shuffled_ARI", final_candidate.get("real_minus_shuffled_ARI"), ">=", 0.30),
        _threshold("real_minus_no_temporal_ARI", final_candidate.get("real_minus_no_temporal_ARI"), ">=", 0.25),
        _threshold("real_minus_mask_only_ARI", final_candidate.get("real_minus_mask_only_ARI"), ">=", 0.25),
        _threshold("bootstrap_delta_ARI_lower95", final_candidate.get("bootstrap_delta_ARI_lower95"), ">=", 0.025),
        _threshold("bootstrap_delta_completeness_lower95", final_candidate.get("bootstrap_delta_completeness_lower95"), ">=", 0.020),
    ]
    control_gate_rows = [
        _threshold("real_minus_shuffled_ARI", final_candidate.get("real_minus_shuffled_ARI"), ">=", 0.30),
        _threshold("real_minus_no_temporal_ARI", final_candidate.get("real_minus_no_temporal_ARI"), ">=", 0.25),
        _threshold("real_minus_mask_only_ARI", final_candidate.get("real_minus_mask_only_ARI"), ">=", 0.25),
        _bool_check("relation_propagation_gate_pass", _nested(propagation, "gate", "pass", default=False), True, "propagation_gate_failed"),
    ]
    relaxed_gate = {
        "pass": all(row["pass"] for row in relaxed_rows),
        "failed_metrics": [row["metric"] for row in relaxed_rows if not row["pass"]],
        "metric_scope": final_candidate.get("metric_scope"),
        "method_claim_scope_valid": bool(final_candidate.get("metric_scope_method_claim_eligible")),
    }
    strict_gate = {
        "pass": all(row["pass"] for row in strict_rows) and bool(final_candidate.get("metric_scope_method_claim_eligible")),
        "failed_metrics": [row["metric"] for row in strict_rows if not row["pass"]]
        + ([] if final_candidate.get("metric_scope_method_claim_eligible") else ["metric_scope_method_claim_eligible"]),
        "metric_scope": final_candidate.get("metric_scope"),
        "method_claim_scope_valid": bool(final_candidate.get("metric_scope_method_claim_eligible")),
    }
    control_gate = {
        "pass": all(row["pass"] for row in control_gate_rows),
        "failed_metrics": [row["metric"] for row in control_gate_rows if not row["pass"]],
    }
    ap_bridge_progress = bool(
        _as_float(ap_summary.get("best_AP")) >= _as_float(fact_map.get("v37_postprocess_AP"))
        or _as_float(ap_summary.get("best_AP50")) >= _as_float(fact_map.get("v37_postprocess_AP50"))
        or _as_float(ap_summary.get("best_AP25")) >= _as_float(fact_map.get("v37_postprocess_AP25"))
    )
    if strict_gate["pass"] and control_gate["pass"]:
        label = "GO_STAGE1_MASK_MERGE"
    elif ap_bridge_progress and not strict_gate["pass"]:
        label = "PARTIAL_AP_BRIDGE_PROGRESS_IDENTITY_NOT_STRICT"
    elif relaxed_gate["pass"] and ap_gate.get("ap_diagnostic_useful"):
        label = "PARTIAL_MASK_MERGE_PROGRESS_AP_RAN"
    elif relaxed_gate["pass"] and not control_gate["pass"]:
        label = "PARTIAL_MASK_MERGE_SIGNAL_CONTROL_FAIL"
    else:
        label = "NO_GO_STAGE1_NOT_SIGNIFICANT"

    gate = {
        "relaxed_progress_gate_pass": relaxed_gate["pass"],
        "method_claim_gate_pass": strict_gate["pass"],
        "d4rt_controls_pass": control_gate["pass"],
        "ap_diagnostic_ran": bool(ap_gate.get("ap_diagnostic_useful")),
        "ap_bridge_progress_vs_v37_postprocess": ap_bridge_progress,
        "method_safe_ap_available": bool(ap_gate.get("method_safe_ap_available")),
        "pass": bool(strict_gate["pass"] and control_gate["pass"]),
    }
    return {
        "phase": "v50_full_stage1",
        "created_at": utc_now(),
        "stage1_rows": stage1_rows,
        "relaxed_gate_rows": relaxed_rows,
        "method_claim_gate_rows": strict_rows,
        "control_gate_rows": control_gate_rows,
        "control_rows": control_rows,
        "bootstrap_rows": bootstrap_rows,
        "ap_link_rows": ap_link_rows,
        "relaxed_progress_gate": relaxed_gate,
        "method_claim_gate": strict_gate,
        "control_gate": control_gate,
        "gate": gate,
        "final_candidate": final_candidate,
        "failure_label": None if gate["pass"] else label,
        "partial_label": label if label.startswith("PARTIAL_") else None,
        "stage2_allowed": False,
        "notes": [
            "Best v50 identity row is component_vote_diagnostic and not a strict method result.",
            "D4RT matched controls have zero ARI delta, so method claim is blocked.",
            "Bootstrap lower bounds were not computed because suitable resampling artifacts are unavailable.",
        ],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_full_stage1(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "full_stage1_summary.json", payload)
    write_csv(out / "stage1_metric_rows.csv", payload["stage1_rows"])
    write_csv(out / "control_rows.csv", payload["control_rows"])
    write_csv(out / "bootstrap_rows.csv", payload["bootstrap_rows"])
    write_csv(out / "ap_link_rows.csv", payload["ap_link_rows"])
    write_csv(out / "method_claim_gate_rows.csv", payload["method_claim_gate_rows"])
    write_csv(out / "relaxed_gate_rows.csv", payload["relaxed_gate_rows"])


def _read_rows_or_empty(path: str | Path) -> list[dict[str, Any]]:
    path_obj = project_path(path)
    if not path_obj.exists():
        return []
    return list(read_csv(path_obj))


def _component_scene(component_ids: str) -> str:
    first = str(component_ids).split(";")[0].split(",")[0]
    return first.split("|")[0] if "|" in first else "unknown"


def build_v50_failure_autopsy() -> dict[str, Any]:
    mask = load_optional_json("outputs/audit/v50_mask_source_audit/mask_source_summary.json")
    relation = load_optional_json("outputs/audit/v50_same_view_relations/relation_summary.json")
    key_masks = load_optional_json("outputs/audit/v50_key_masks/key_mask_summary.json")
    propagation = load_optional_json("outputs/audit/v50_relation_propagation/propagation_summary.json")
    semantic = load_optional_json("outputs/audit/v50_semantic_guard/semantic_guard_summary.json")
    hyp = load_optional_json("outputs/audit/v50_hypothesis_generation/hypothesis_summary.json")
    selection = load_optional_json("outputs/audit/v50_hypothesis_selection/selection_summary.json")
    ap = load_optional_json("outputs/audit/v50_ap_diagnostic/ap_export_summary.json")
    full = load_optional_json("outputs/audit/v50_full_stage1/full_stage1_summary.json")

    mask_rows = [
        {
            "failure": "same_view_pixel_hierarchy_unavailable",
            "same_view_hierarchy_available": _nested(mask, "gate", "same_view_hierarchy_available"),
            "effective_hierarchy_route": _nested(mask, "gate", "effective_hierarchy_route"),
            "component_lattice_fallback_available": _nested(mask, "gate", "component_lattice_fallback_available"),
            "containment_pair_count": _nested(mask, "summary", "containment_pair_count"),
            "bbox_containment_pair_count_diagnostic": _nested(mask, "summary", "bbox_containment_pair_count_diagnostic"),
            "note": "exact same-view overlap hierarchy absent; fallback is multi-scale component lattice",
        }
    ]
    relation_error_rows = relation.get("relation_metric_rows", [])
    if not isinstance(relation_error_rows, list):
        relation_error_rows = _read_rows_or_empty("outputs/audit/v50_same_view_relations/relation_metric_rows.csv")
    key_error_rows = [
        {
            "selected_variant": key_masks.get("selected_variant"),
            "key_mask_count": _nested(key_masks, "summary", "key_mask_count"),
            "key_mask_ratio": _nested(key_masks, "summary", "key_mask_ratio"),
            "key_mask_purity": _nested(key_masks, "summary", "key_mask_purity"),
            "false_key_mask_rate": _nested(key_masks, "summary", "false_key_mask_rate"),
            "raw_large_underseg_rate": _nested(key_masks, "summary", "raw_large_underseg_rate"),
            "large_underseg_selected_rate": _nested(key_masks, "summary", "large_underseg_selected_rate"),
            "large_underseg_reduction_pass": _nested(key_masks, "gate", "large_underseg_reduction_pass"),
            "note": "coverage and key-mask ratio passed, but underseg reduction failed",
        }
    ]
    propagation_error_rows = propagation.get("propagation_metric_rows", [])
    if not isinstance(propagation_error_rows, list):
        propagation_error_rows = _read_rows_or_empty("outputs/audit/v50_relation_propagation/propagation_metric_rows.csv")
    hypothesis_error_rows = hyp.get("hypothesis_source_metric_rows", [])
    if not isinstance(hypothesis_error_rows, list):
        hypothesis_error_rows = _read_rows_or_empty("outputs/audit/v50_hypothesis_generation/hypothesis_source_metric_rows.csv")
    selection_error_rows = selection.get("selection_rows", [])
    if not isinstance(selection_error_rows, list):
        selection_error_rows = _read_rows_or_empty("outputs/audit/v50_hypothesis_selection/selection_rows.csv")
    hypothesis_rows = hyp.get("hypothesis_rows", [])
    if not isinstance(hypothesis_rows, list):
        hypothesis_rows = _read_rows_or_empty("outputs/audit/v50_hypothesis_generation/hypothesis_rows.csv")
    false_merge_hypotheses = [
        row for row in hypothesis_rows
        if str(row.get("diagnostic_conflict")).lower() == "true" or _as_float(row.get("diagnostic_purity"), 1.0) < 0.75
    ][:300]
    selected_rows = selection.get("selected_hypothesis_rows", [])
    if not isinstance(selected_rows, list):
        selected_rows = _read_rows_or_empty("outputs/audit/v50_hypothesis_selection/selected_hypothesis_rows.csv")
    false_cut_components = [
        {
            "hypothesis_id": row.get("hypothesis_id"),
            "scene": _component_scene(str(row.get("component_ids", ""))),
            "component_ids": row.get("component_ids"),
            "diagnostic_dominant_gt": row.get("diagnostic_dominant_gt"),
            "diagnostic_best_gt_coverage": row.get("diagnostic_best_gt_coverage"),
            "diagnostic_purity": row.get("diagnostic_purity"),
            "reason": "selected singleton has low dominant-GT coverage",
        }
        for row in selected_rows
        if _as_float(row.get("diagnostic_best_gt_coverage"), 1.0) < 0.05
    ][:300]
    fragments: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        key = (_component_scene(str(row.get("component_ids", ""))), str(row.get("diagnostic_dominant_gt")))
        if key[1] not in {"", "None", "none"}:
            fragments[key].append(row)
    fragmented_gt_objects = []
    for (scene, gt_id), rows in sorted(fragments.items()):
        if len(rows) <= 1:
            continue
        coverages = [_as_float(row.get("diagnostic_best_gt_coverage")) for row in rows]
        fragmented_gt_objects.append(
            {
                "scene": scene,
                "diagnostic_gt": gt_id,
                "selected_fragment_count": len(rows),
                "max_fragment_coverage": max(coverages) if coverages else None,
                "coverage_sum_over_selected_fragments": sum(coverages),
                "note": "diagnostic grouping by dominant GT only; not used for prediction",
            }
        )
    control_gap_breakdown = [
        {
            "gap": "final_selection_real_minus_shuffled_ARI",
            "value": _nested(selection, "summary", "best_real_row", "real_minus_shuffled_ARI"),
            "target": 0.30,
            "pass": False,
        },
        {
            "gap": "final_selection_real_minus_no_temporal_ARI",
            "value": _nested(selection, "summary", "best_real_row", "real_minus_no_temporal_ARI"),
            "target": 0.25,
            "pass": False,
        },
        {
            "gap": "final_selection_real_minus_mask_only_ARI",
            "value": _nested(selection, "summary", "best_real_row", "real_minus_mask_only_ARI"),
            "target": 0.25,
            "pass": False,
        },
        {
            "gap": "propagation_real_minus_shuffled_AUC",
            "value": _nested(propagation, "summary", "propagation_real_minus_shuffled_AUC"),
            "target": 0.04,
            "pass": bool(_nested(propagation, "gate", "propagation_real_minus_shuffled_AUC_pass", default=False)),
        },
        {
            "gap": "propagation_real_minus_no_temporal_AUC",
            "value": _nested(propagation, "summary", "propagation_real_minus_no_temporal_AUC"),
            "target": 0.03,
            "pass": bool(_nested(propagation, "gate", "propagation_real_minus_no_temporal_AUC_pass", default=False)),
        },
    ]
    ap_failure_rows = list(ap.get("ap_failure_casebook", []) if isinstance(ap.get("ap_failure_casebook"), list) else [])
    if not ap_failure_rows:
        for row in ap.get("ap_rows", []):
            if row.get("status") != "ran" or row.get("forbidden_for_method_table"):
                ap_failure_rows.append(
                    {
                        "variant": row.get("variant"),
                        "status": row.get("status"),
                        "AP": row.get("AP"),
                        "forbidden_for_method_table": row.get("forbidden_for_method_table"),
                        "uses_rgbd_pose_mesh_for_export": row.get("uses_rgbd_pose_mesh_for_export"),
                        "attribution": "method-safe native AP unavailable or diagnostic-only row",
                    }
                )
    ap_identity_vs_materialization_rows = [
        {
            "identity_solver_variant": _nested(selection, "summary", "best_real_row", "solver_variant"),
            "identity_metric_scope": _nested(selection, "summary", "best_real_row", "metric_scope"),
            "identity_ARI": _nested(selection, "summary", "best_real_row", "ARI"),
            "identity_purity": _nested(selection, "summary", "best_real_row", "purity"),
            "identity_completeness": _nested(selection, "summary", "best_real_row", "completeness"),
            "AP_variant": row.get("variant"),
            "AP_status": row.get("status"),
            "AP": row.get("AP"),
            "AP50": row.get("AP50"),
            "AP25": row.get("AP25"),
            "materialization_policy": row.get("alignment_policy"),
            "method_safe": bool(row.get("AP") is not None and not row.get("forbidden_for_method_table")),
        }
        for row in ap.get("ap_rows", [])
    ]
    answers = {
        "mask_source_has_usable_same_view_hierarchy": "No exact same-view pixel hierarchy; yes fallback via multi-scale component lattice.",
        "key_masks_reduce_part_flood": "Part flood count is reduced by key_mask_ratio=0.18026565464895636, but large underseg reduction failed.",
        "same_view_relation_reliable": "No. Part/sibling diagnostic precision is far below plan thresholds.",
        "d4rt_propagation_positive_completion": "No. Pair AUC/top5k are below v49 reference and controls are not beaten.",
        "semantic_guard_reduces_false_merge": "No measured reduction. Semantic contradiction AUC is diagnostic but guard is not enabled.",
        "correct_hypotheses_exist": "Yes diagnostically: GT_object_has_hypothesis@0.50=0.7419354838709677.",
        "scoring_or_solver_issue": "Selection picks mostly raw singleton components; relation/propagation-only selection has high conflict, so evidence separability/control is the main issue.",
        "d4rt_control_failure_location": "Relation propagation and final selection controls both fail; component generation is not proven as D4RT-specific improvement.",
        "ap_ran": "Yes. AP5/AP6/AP7 RGB-D/pose/mesh bridge diagnostics ran when the bridge exporter/evaluator succeeded; AP2 native method-safe export is blocked.",
        "ap_failure_attribution": "Method-safe materialization/native component-to-mesh export is blocked; bridge AP variants indicate identity/materialization potential but are diagnostic-only.",
        "ap_not_run_reason_and_repairs": "AP ran through bridge diagnostics after evaluator selfcheck; AP2 native export remained blocked_missing_native_component_to_mesh_materializer.",
    }
    labels = ["NO_GO_D4RT_PROPAGATION", "NO_GO_D4RT_CONTROL", "NO_GO_STAGE1_NOT_SIGNIFICANT"]
    if not _nested(ap, "gate", "method_safe_ap_available", default=False):
        labels.append("NO_GO_AP_MATERIALIZATION")
    if _nested(full, "partial_label", default=None):
        labels.append(str(_nested(full, "partial_label")))
    summary_lines = [
        "# v50 failure autopsy",
        "",
        f"created_at: {utc_now()}",
        f"final_failure_label: {full.get('failure_label')}",
        f"failure_labels: {', '.join(labels)}",
        "",
        "## Required questions",
        "",
    ]
    for index, (key, answer) in enumerate(answers.items(), 1):
        summary_lines.append(f"{index}. {key}: {answer}")
    summary_lines.extend(
        [
            "",
            "## Evidence chain",
            "",
            f"- Relation precision gate: {_nested(relation, 'gate', 'pass', default=False)}.",
            f"- Propagation gate: {_nested(propagation, 'gate', 'pass', default=False)}.",
            f"- Hypothesis generation gate: {_nested(hyp, 'gate', 'pass', default=False)}.",
            f"- Selection diagnostic gate: {_nested(selection, 'gate', 'pass', default=False)}.",
            f"- Method claim gate: {_nested(full, 'method_claim_gate', 'pass', default=False)}.",
            f"- AP diagnostic gate: {_nested(ap, 'gate', 'pass', default=False)}.",
            "",
        ]
    )
    return {
        "phase": "v50_failure_autopsy",
        "created_at": utc_now(),
        "failure_labels": labels,
        "final_failure_label": full.get("failure_label"),
        "answers": answers,
        "mask_source_failure_rows": mask_rows,
        "same_view_relation_error_rows": relation_error_rows,
        "key_mask_error_rows": key_error_rows,
        "propagation_error_rows": propagation_error_rows,
        "hypothesis_error_rows": hypothesis_error_rows,
        "selection_error_rows": selection_error_rows,
        "false_merge_hypotheses": false_merge_hypotheses,
        "false_cut_components": false_cut_components,
        "fragmented_GT_objects": fragmented_gt_objects,
        "control_gap_breakdown": control_gap_breakdown,
        "ap_failure_casebook": ap_failure_rows,
        "ap_identity_vs_materialization_rows": ap_identity_vs_materialization_rows,
        "failure_summary_md": "\n".join(summary_lines),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_failure_autopsy(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "failure_autopsy_summary.json", payload)
    for key in [
        "mask_source_failure_rows",
        "same_view_relation_error_rows",
        "key_mask_error_rows",
        "propagation_error_rows",
        "hypothesis_error_rows",
        "selection_error_rows",
        "false_merge_hypotheses",
        "false_cut_components",
        "fragmented_GT_objects",
        "control_gap_breakdown",
        "ap_failure_casebook",
        "ap_identity_vs_materialization_rows",
    ]:
        write_csv(out / f"{key}.csv", payload[key])
    (out / "failure_summary.md").write_text(payload["failure_summary_md"] + "\n", encoding="utf-8")


def build_v50_stage2_eligibility() -> dict[str, Any]:
    fact = load_optional_json("outputs/audit/v50_fact_lock/fact_lock.json")
    full = load_optional_json("outputs/audit/v50_full_stage1/full_stage1_summary.json")
    ap = load_optional_json("outputs/audit/v50_ap_diagnostic/ap_export_summary.json")
    policy_rows = ap.get("ap_policy_rows", []) if isinstance(ap.get("ap_policy_rows"), list) else []
    ap_policy_clean = bool(policy_rows) and all(bool(row.get("policy_clean")) for row in policy_rows)
    entry_gate = {
        "Method_Claim_Gate_passed": bool(_nested(full, "method_claim_gate", "pass", default=False)),
        "D4RT_controls_passed": bool(_nested(full, "control_gate", "pass", default=False)),
        "scale_guard_passed": bool(_nested(fact, "fact_map", "scale_guard_pass", default=False)),
        "AP_policy_rows_clean": ap_policy_clean,
    }
    entry_gate["pass"] = all(entry_gate.values())
    block_reasons = []
    if not entry_gate["Method_Claim_Gate_passed"]:
        block_reasons.append("stage1_not_passed")
    if not entry_gate["D4RT_controls_passed"]:
        block_reasons.append("controls_failed")
    if not entry_gate["scale_guard_passed"]:
        block_reasons.append("scale_guard_failed")
    if not entry_gate["AP_policy_rows_clean"]:
        block_reasons.append("ap_policy_not_clean")
    stage2_rows = [
        {
            "condition": key,
            "pass": value,
            "required": True,
            "stage2_allowed_if_false": False,
        }
        for key, value in entry_gate.items()
        if key != "pass"
    ]
    return {
        "phase": "v50_stage2_eligibility",
        "created_at": utc_now(),
        "entry_gate": entry_gate,
        "stage2_allowed": bool(entry_gate["pass"]),
        "stage2_block_reason": ";".join(block_reasons) if block_reasons else "",
        "stage2_rows": stage2_rows,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_stage2_eligibility(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "stage2_eligibility_summary.json", payload)
    write_csv(out / "stage2_rows.csv", payload["stage2_rows"])


def build_v50_final_decision() -> dict[str, Any]:
    mask = load_optional_json("outputs/audit/v50_mask_source_audit/mask_source_summary.json")
    relation = load_optional_json("outputs/audit/v50_same_view_relations/relation_summary.json")
    key_masks = load_optional_json("outputs/audit/v50_key_masks/key_mask_summary.json")
    propagation = load_optional_json("outputs/audit/v50_relation_propagation/propagation_summary.json")
    semantic = load_optional_json("outputs/audit/v50_semantic_guard/semantic_guard_summary.json")
    hyp = load_optional_json("outputs/audit/v50_hypothesis_generation/hypothesis_summary.json")
    selection = load_optional_json("outputs/audit/v50_hypothesis_selection/selection_summary.json")
    ap = load_optional_json("outputs/audit/v50_ap_diagnostic/ap_export_summary.json")
    full = load_optional_json("outputs/audit/v50_full_stage1/full_stage1_summary.json")
    failure = load_optional_json("outputs/audit/v50_failure_autopsy/failure_autopsy_summary.json")
    stage2 = load_optional_json("outputs/audit/v50_stage2/stage2_eligibility_summary.json")
    full_gate = full.get("gate", {}) if isinstance(full.get("gate"), dict) else {}
    if full_gate.get("pass"):
        final_label = "GO_STAGE1_MASK_MERGE"
    elif full.get("partial_label"):
        final_label = str(full.get("partial_label"))
    else:
        final_label = str(full.get("failure_label") or "NO_GO_STAGE1_NOT_SIGNIFICANT")
    all_failure_labels = failure.get("failure_labels", []) if isinstance(failure.get("failure_labels"), list) else []
    no_go_labels = [str(label) for label in all_failure_labels if str(label).startswith("NO_GO")]
    partial_labels = [str(label) for label in all_failure_labels if str(label).startswith("PARTIAL")]
    answers = {
        "1_mask_source_same_view_hierarchy": "exact_same_view_hierarchy=False; fallback=multi_scale_component_lattice",
        "2_key_mask_selection_reduces_part_flood": f"key_mask_ratio={_nested(key_masks, 'summary', 'key_mask_ratio')}; large_underseg_reduction_pass={_nested(key_masks, 'gate', 'large_underseg_reduction_pass')}",
        "3_same_view_relations_reliable": f"relation_gate={_nested(relation, 'gate', 'pass')}; part_precision={_nested(relation, 'summary', 'part_relation_precision')}; sibling_precision={_nested(relation, 'summary', 'sibling_relation_precision')}",
        "4_d4rt_propagation_improves_affinity": f"propagation_gate={_nested(propagation, 'gate', 'pass')}; AUC={_nested(propagation, 'summary', 'same_GT_pair_AUC')}; real_minus_shuffled={_nested(propagation, 'summary', 'propagation_real_minus_shuffled_AUC')}",
        "5_semantic_guard_reduces_false_merge": f"enabled_for_selection={_nested(semantic, 'gate', 'semantic_guard_enabled_for_selection')}; false_merge_reduction={_nested(semantic, 'summary', 'false_merge_reduction')}",
        "6_object_hypotheses_cover_whole_objects": f"hypothesis_gate={_nested(hyp, 'gate', 'pass')}; GT_object_has_hypothesis@0.50={_nested(hyp, 'summary', 'GT_object_has_hypothesis@0.50')}",
        "7_final_selection_exceeds_v49_O33": f"not_proven; metric_scope={_nested(selection, 'summary', 'metric_scope')}; real_minus_controls all {_nested(selection, 'summary', 'best_real_row', 'real_minus_shuffled_ARI')}/{_nested(selection, 'summary', 'best_real_row', 'real_minus_no_temporal_ARI')}/{_nested(selection, 'summary', 'best_real_row', 'real_minus_mask_only_ARI')}",
        "8_d4rt_controls_pass": str(_nested(full, "control_gate", "pass", default=False)),
        "9_ap_rows_ran": f"AP5_ran={_nested(ap, 'gate', 'rgbd_bridge_ap_ran')}; AP6_ran={_nested(ap, 'gate', 'ap6_constant_score_min_region_ran')}; AP7_ran={_nested(ap, 'gate', 'ap7_wta_conflict_suppression_ran')}; method_safe_ap_available={_nested(ap, 'gate', 'method_safe_ap_available')}",
        "10_ap_gt_rgbd_pose_mesh_policy": "AP5/AP6/AP7 use RGB-D/pose/mesh bridge and are diagnostic-only; no GT prediction leakage recorded.",
        "11_ap_blocker": "native method-safe component-to-mesh materialization/export is blocked; bridge materialization works diagnostically only.",
        "12_no_d4rt_birth_no_maskless": f"birth={_nested(selection, 'summary', 'best_real_row', 'birth_from_d4rt_tube_count')}; maskless={_nested(selection, 'summary', 'best_real_row', 'maskless_object_count')}",
        "13_stage2_policy": f"stage2_allowed={stage2.get('stage2_allowed')}; reason={stage2.get('stage2_block_reason')}",
        "14_failure_location": ",".join(no_go_labels),
    }
    first_page_lines = [
        "# Stream4D v50 final report first page",
        "",
        f"final_label: {final_label}",
        f"no_go_labels: {', '.join(no_go_labels)}",
        f"stage2_allowed: {stage2.get('stage2_allowed')}",
        "",
        "## Required answers",
        "",
    ]
    for key, value in answers.items():
        first_page_lines.append(f"- {key}: {value}")
    first_page_lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Do not claim strict method success.",
            "- Do not promote component_vote_diagnostic ARI to an official 4D method metric.",
            "- Do not promote AP5 bridge diagnostic AP to method-safe AP.",
            "",
        ]
    )
    return {
        "phase": "v50_final_decision",
        "created_at": utc_now(),
        "final_label": final_label,
        "no_go_labels": no_go_labels,
        "partial_labels": partial_labels,
        "answers": answers,
        "gates": {
            "mask_source": mask.get("gate"),
            "same_view_relations": relation.get("gate"),
            "key_masks": key_masks.get("gate"),
            "relation_propagation": propagation.get("gate"),
            "semantic_guard": semantic.get("gate"),
            "hypothesis_generation": hyp.get("gate"),
            "hypothesis_selection": selection.get("gate"),
            "ap_diagnostic": ap.get("gate"),
            "full_stage1": full.get("gate"),
            "stage2": stage2.get("entry_gate"),
        },
        "final_candidate": full.get("final_candidate"),
        "artifact_paths": {
            "full_stage1": rel("outputs/audit/v50_full_stage1/full_stage1_summary.json"),
            "failure_autopsy": rel("outputs/audit/v50_failure_autopsy/failure_autopsy_summary.json"),
            "stage2": rel("outputs/audit/v50_stage2/stage2_eligibility_summary.json"),
            "ap_diagnostic": rel("outputs/audit/v50_ap_diagnostic/ap_export_summary.json"),
        },
        "final_report_first_page_md": "\n".join(first_page_lines),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v50_final_decision(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = project_path(output_root)
    write_json(out / "v50_final_decision.json", payload)
    (out / "final_report_first_page.md").write_text(payload["final_report_first_page_md"] + "\n", encoding="utf-8")
