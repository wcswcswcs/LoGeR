from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _comb2(value: float) -> float:
    return value * (value - 1.0) / 2.0 if value >= 2.0 else 0.0


def weighted_partition_metrics(assignments: list[tuple[str, str, float]]) -> dict[str, float]:
    total = sum(weight for _, _, weight in assignments)
    if total <= 0:
        return {"ARI": 0.0, "purity": 0.0, "completeness": 0.0}
    pred_counts: Counter[str] = Counter()
    true_counts: Counter[str] = Counter()
    contingency: dict[tuple[str, str], float] = defaultdict(float)
    for pred, true, weight in assignments:
        if not true or weight <= 0:
            continue
        pred_counts[pred] += weight
        true_counts[true] += weight
        contingency[(pred, true)] += weight
    sum_comb_c = sum(_comb2(value) for value in contingency.values())
    sum_comb_pred = sum(_comb2(value) for value in pred_counts.values())
    sum_comb_true = sum(_comb2(value) for value in true_counts.values())
    total_comb = _comb2(sum(true_counts.values()))
    expected = (sum_comb_pred * sum_comb_true / total_comb) if total_comb else 0.0
    max_index = 0.5 * (sum_comb_pred + sum_comb_true)
    denom = max_index - expected
    ari = 0.0 if denom == 0.0 else (sum_comb_c - expected) / denom
    purity = sum(max((contingency.get((pred, true), 0.0) for true in true_counts), default=0.0) for pred in pred_counts) / max(sum(true_counts.values()), 1.0)
    completeness = sum(max((contingency.get((pred, true), 0.0) for pred in pred_counts), default=0.0) for true in true_counts) / max(sum(true_counts.values()), 1.0)
    return {"ARI": float(ari), "purity": float(purity), "completeness": float(completeness)}


def _load_component_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(";") if item.strip()]
    return [str(item) for item in payload]


def _component_scene(component_rows: list[dict[str, Any]]) -> dict[str, str]:
    scene_by_component: dict[str, Counter[str]] = defaultdict(Counter)
    for row in component_rows:
        scene_by_component[str(row["component_id"])][str(row["scene"])] += parse_int(row.get("support_count"), 1)
    return {component: counts.most_common(1)[0][0] for component, counts in scene_by_component.items() if counts}


def _support_assignments(
    support_rows: list[dict[str, Any]],
    component_to_object: dict[str, str],
    *,
    scene_filter: str | None = None,
    raw_component_fallback: bool = False,
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for row in support_rows:
        if scene_filter is not None and str(row.get("scene")) != scene_filter:
            continue
        component_id = str(row.get("component_id"))
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not gt:
            continue
        if component_id in component_to_object:
            pred = component_to_object[component_id]
        elif raw_component_fallback:
            pred = f"{row.get('scene')}|raw:{component_id}"
        else:
            pred = f"{row.get('scene')}|unknown:{component_id}"
        rows.append((pred, f"{row.get('scene')}|{gt}", float(parse_int(row.get("support_count"), 1))))
    return rows


def _evaluate_variant(
    variant: str,
    support_rows: list[dict[str, Any]],
    component_to_object: dict[str, str],
    object_rows: list[dict[str, Any]],
    all_components: set[str],
    component_scene: dict[str, str],
    *,
    raw_component_fallback: bool = False,
) -> dict[str, Any]:
    metrics = weighted_partition_metrics(
        _support_assignments(support_rows, component_to_object, raw_component_fallback=raw_component_fallback)
    )
    scenes = sorted({str(row.get("scene")) for row in support_rows})
    scene_metrics: dict[str, dict[str, float]] = {}
    for scene in scenes:
        scene_metrics[scene] = weighted_partition_metrics(
            _support_assignments(
                support_rows,
                component_to_object,
                scene_filter=scene,
                raw_component_fallback=raw_component_fallback,
            )
        )
    selected_components = set(component_to_object)
    object_frames: dict[str, set[int]] = defaultdict(set)
    object_scenes: dict[str, str] = {}
    for row in object_rows:
        object_id = str(row["objectlet_id"])
        object_scenes[object_id] = str(row["scene"])
        for frame_id in row.get("target_frame_ids", []):
            object_frames[object_id].add(int(frame_id))
    scene_object_counts = Counter(object_scenes.values())
    unknown_components = all_components - selected_components
    unknown_by_scene = Counter(component_scene.get(component, "") for component in unknown_components)
    return {
        "variant": variant,
        "selected_objectlet_count": len(object_rows),
        "selected_component_count": len(selected_components),
        "mean_components_per_objectlet": _mean([float(row.get("component_count", 0)) for row in object_rows]),
        "component_coverage_ratio": float(len(selected_components) / max(len(all_components), 1)),
        "uncovered_component_ratio": float(len(unknown_components) / max(len(all_components), 1)),
        "duplicate_component_ratio": _mean([float(row.get("duplicate_component_ratio", 0.0)) for row in object_rows]) or 0.0,
        "conflict_rate": _mean([parse_float(row.get("same_frame_exclusion_violation_rate")) for row in object_rows]) or 0.0,
        "outside_residual_mean": _mean([parse_float(row.get("outside_all_related_masks_ratio_mean")) for row in object_rows]),
        "underseg_objectlet_rate": _mean([1.0 if parse_bool(row.get("underseg_proxy", False)) else 0.0 for row in object_rows]) or 0.0,
        "semantic_contradiction_rate": None,
        "unknown_component_ratio": float(len(unknown_components) / max(len(all_components), 1)),
        "4D_ARI": metrics["ARI"],
        "4D_purity": metrics["purity"],
        "4D_completeness": metrics["completeness"],
        "temporal_span_mean": _mean([float(len(frames)) for frames in object_frames.values()]),
        "scene0081_ARI": scene_metrics.get("scene0081_01", {}).get("ARI"),
        "scene0591_completeness": scene_metrics.get("scene0591_00", {}).get("completeness"),
        "mean_predictions_per_scene": _mean([float(value) for value in scene_object_counts.values()]),
        "unknown_components_by_scene": dict(unknown_by_scene),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _gate(row: dict[str, Any], *, strict: bool) -> dict[str, bool]:
    if strict:
        gate = {
            "4D_ARI_ge_0.485": parse_float(row.get("4D_ARI")) >= 0.485,
            "4D_purity_ge_0.875": parse_float(row.get("4D_purity")) >= 0.875,
            "4D_completeness_ge_0.555": parse_float(row.get("4D_completeness")) >= 0.555,
            "conflict_rate_le_0.10": parse_float(row.get("conflict_rate")) <= 0.10,
            "mean_predictions_per_scene_le_150": parse_float(row.get("mean_predictions_per_scene"), 9999.0) <= 150,
            "real_minus_shuffled_ARI_ge_0.20": parse_float(row.get("real_minus_shuffled_ARI"), -9999.0) >= 0.20,
            "real_minus_no_temporal_ARI_ge_0.10": parse_float(row.get("real_minus_no_temporal_ARI"), -9999.0)
            >= 0.10,
            "real_minus_mask_only_ARI_ge_0.10": parse_float(row.get("real_minus_mask_only_ARI"), -9999.0) >= 0.10,
        }
    else:
        gate = {
            "4D_ARI_ge_0.46": parse_float(row.get("4D_ARI")) >= 0.46,
            "4D_purity_ge_0.85": parse_float(row.get("4D_purity")) >= 0.85,
            "4D_completeness_ge_0.50": parse_float(row.get("4D_completeness")) >= 0.50,
            "conflict_rate_le_0.18": parse_float(row.get("conflict_rate")) <= 0.18,
            "outside_residual_mean_le_0.35": parse_float(row.get("outside_residual_mean"), 9999.0) <= 0.35,
            "mean_predictions_per_scene_le_220": parse_float(row.get("mean_predictions_per_scene"), 9999.0) <= 220,
            "birth_from_d4rt_tube_count_eq_0": parse_int(row.get("birth_from_d4rt_tube_count")) == 0,
            "maskless_object_count_eq_0": parse_int(row.get("maskless_object_count")) == 0,
        }
    gate["pass"] = bool(all(gate.values()))
    return gate


def _shuffled_component_assignment(component_to_object: dict[str, str], component_scene: dict[str, str]) -> dict[str, str]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for component in component_to_object:
        by_scene[component_scene.get(component, "")].append(component)
    out: dict[str, str] = {}
    for scene, components in by_scene.items():
        components = sorted(components)
        if len(components) <= 1:
            for component in components:
                out[component] = component_to_object[component]
            continue
        rotated = components[1:] + components[:1]
        for src_component, dst_component in zip(components, rotated):
            out[dst_component] = component_to_object[src_component]
    return out


def _filter_single_frame_assignment(
    component_to_object: dict[str, str],
    object_rows: list[dict[str, Any]],
) -> dict[str, str]:
    single_frame_objects = {
        str(row["objectlet_id"]) for row in object_rows if len(row.get("target_frame_ids", [])) <= 1
    }
    return {
        component: object_id
        for component, object_id in component_to_object.items()
        if object_id in single_frame_objects
    }


def _mask_only_candidates(
    support_rows: list[dict[str, Any]],
    representative_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    support_by_mask: dict[str, set[str]] = defaultdict(set)
    for row in support_rows:
        support_by_mask[str(row.get("mask_observation_id"))].add(str(row.get("component_id")))
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for row in representative_rows:
        mask_id = str(row.get("mask_observation_id"))
        components = sorted(support_by_mask.get(mask_id, set()))
        if not components:
            continue
        key = (str(row.get("scene")), tuple(components))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "candidate_id": f"maskonly{len(candidates):05d}",
                "candidate_source": "L9_mask_only_representative_support",
                "scene": row.get("scene"),
                "chunk_id": row.get("chunk_id"),
                "source_mask_observation_id": mask_id,
                "source_frame_id": row.get("frame_id"),
                "source_mask_id": row.get("mask_id"),
                "component_count": len(components),
                "component_ids": json.dumps(components),
                "candidate_success_rate": 0.0,
                "outside_all_related_masks_ratio_mean": 0.0,
                "same_frame_exclusion_violation_rate": 0.0,
            }
        )
    return candidates


def _select_objectlets(
    candidate_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    representative_by_mask: dict[str, dict[str, Any]],
    *,
    variant: str,
    max_components_per_objectlet: int | None,
    min_new_component_ratio: float,
    sort_mode: str = "success_outside_large",
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    frames_by_candidate: dict[str, set[int]] = defaultdict(set)
    for row in ledger_rows:
        frames_by_candidate[str(row["candidate_id"])].add(parse_int(row.get("target_frame_id")))
    if sort_mode == "coverage_first":
        sort_key = lambda row: (
            -parse_int(row.get("component_count")),
            -parse_float(row.get("candidate_success_rate")),
            parse_float(row.get("outside_all_related_masks_ratio_mean"), 1.0),
            str(row.get("candidate_id")),
        )
    elif sort_mode == "repeated_signature_first":
        sort_key = lambda row: (
            0 if str(row.get("candidate_source")) == "R5_repeated_support_signature" else 1,
            -parse_int(row.get("repeated_support_signature_len")),
            -parse_float(row.get("candidate_success_rate")),
            parse_float(row.get("outside_all_related_masks_ratio_mean"), 1.0),
            -parse_int(row.get("component_count")),
            str(row.get("candidate_id")),
        )
    elif sort_mode == "success_large":
        sort_key = lambda row: (
            -parse_float(row.get("candidate_success_rate")),
            -parse_int(row.get("component_count")),
            parse_float(row.get("outside_all_related_masks_ratio_mean"), 1.0),
            str(row.get("candidate_id")),
        )
    else:
        sort_key = lambda row: (
            -parse_float(row.get("candidate_success_rate")),
            parse_float(row.get("outside_all_related_masks_ratio_mean"), 1.0),
            -parse_int(row.get("component_count")),
            str(row.get("candidate_id")),
        )
    sorted_candidates = sorted(candidate_rows, key=sort_key)
    component_to_object: dict[str, str] = {}
    object_rows: list[dict[str, Any]] = []
    for candidate in sorted_candidates:
        components = _load_component_ids(candidate.get("component_ids"))
        if max_components_per_objectlet is not None and len(components) > int(max_components_per_objectlet):
            continue
        new_components = [component for component in components if component not in component_to_object]
        if not new_components:
            continue
        if len(new_components) / max(len(components), 1) < float(min_new_component_ratio):
            continue
        objectlet_id = f"{candidate.get('scene')}|{variant}|obj{len(object_rows):05d}"
        for component in new_components:
            component_to_object[component] = objectlet_id
        representative = representative_by_mask.get(str(candidate.get("source_mask_observation_id")), {})
        duplicate_ratio = 1.0 - (len(new_components) / max(len(components), 1))
        object_rows.append(
            {
                "variant": variant,
                "objectlet_id": objectlet_id,
                "scene": candidate.get("scene"),
                "chunk_id": candidate.get("chunk_id"),
                "candidate_id": candidate.get("candidate_id"),
                "source_mask_observation_id": candidate.get("source_mask_observation_id"),
                "objectlet_source": candidate.get("candidate_source"),
                "selection_sort_mode": sort_mode,
                "component_count": len(new_components),
                "original_candidate_component_count": len(components),
                "component_ids": json.dumps(new_components),
                "original_candidate_component_ids": json.dumps(components),
                "duplicate_component_ratio": duplicate_ratio,
                "target_frame_ids": sorted(frames_by_candidate.get(str(candidate.get("candidate_id")), set())),
                "candidate_success_rate": candidate.get("candidate_success_rate"),
                "outside_all_related_masks_ratio_mean": candidate.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": candidate.get("same_frame_exclusion_violation_rate"),
                "underseg_proxy": representative.get("underseg_proxy", False),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return component_to_object, object_rows


def _method_variant_name(value: Any) -> bool:
    text = str(value)
    return text.startswith("L4_") or text.startswith("L6_") or text.startswith("L11_") or text.startswith("L12_")


def _select_objectlets_dynamic_uncovered_gain(
    candidate_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    representative_by_mask: dict[str, dict[str, Any]],
    *,
    variant: str,
    duplicate_penalty: float,
    outside_weight: float,
    conflict_weight: float,
    object_penalty: float,
    max_objectlets_per_scene: int = 150,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    frames_by_candidate: dict[str, set[int]] = defaultdict(set)
    for row in ledger_rows:
        frames_by_candidate[str(row["candidate_id"])].add(parse_int(row.get("target_frame_id")))
    remaining = list(candidate_rows)
    component_to_object: dict[str, str] = {}
    object_rows: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    while remaining:
        best_index = -1
        best_key: tuple[float, int, int, str] | None = None
        best_payload: tuple[list[str], list[str], list[str], float] | None = None
        for index, candidate in enumerate(remaining):
            scene = str(candidate.get("scene"))
            if scene_counts[scene] >= int(max_objectlets_per_scene):
                continue
            components = _load_component_ids(candidate.get("component_ids"))
            if not components:
                continue
            new_components = [component for component in components if component not in component_to_object]
            if not new_components:
                continue
            duplicate_count = len(components) - len(new_components)
            success = parse_float(candidate.get("candidate_success_rate"))
            outside = parse_float(candidate.get("outside_all_related_masks_ratio_mean"), 1.0)
            conflict = parse_float(candidate.get("same_frame_exclusion_violation_rate"), 1.0)
            score = (
                len(new_components) * max(success, 0.0)
                - float(duplicate_penalty) * duplicate_count
                - float(outside_weight) * outside * len(components)
                - float(conflict_weight) * conflict * len(components)
                - float(object_penalty)
            )
            key = (float(score), len(new_components), -duplicate_count, str(candidate.get("candidate_id")))
            if best_key is None or key > best_key:
                best_index = index
                best_key = key
                best_payload = (components, new_components, [component for component in components if component in component_to_object], score)
        if best_index < 0 or best_key is None or best_payload is None:
            break
        if best_key[0] <= 0.0:
            break
        candidate = remaining.pop(best_index)
        components, new_components, duplicate_components, score = best_payload
        objectlet_id = f"{candidate.get('scene')}|{variant}|obj{len(object_rows):05d}"
        for component in new_components:
            component_to_object[component] = objectlet_id
        scene_counts[str(candidate.get("scene"))] += 1
        representative = representative_by_mask.get(str(candidate.get("source_mask_observation_id")), {})
        duplicate_ratio = len(duplicate_components) / max(len(components), 1)
        object_rows.append(
            {
                "variant": variant,
                "objectlet_id": objectlet_id,
                "scene": candidate.get("scene"),
                "chunk_id": candidate.get("chunk_id"),
                "candidate_id": candidate.get("candidate_id"),
                "source_mask_observation_id": candidate.get("source_mask_observation_id"),
                "objectlet_source": candidate.get("candidate_source"),
                "selection_sort_mode": "dynamic_uncovered_gain",
                "selection_score": score,
                "component_count": len(new_components),
                "original_candidate_component_count": len(components),
                "component_ids": json.dumps(new_components),
                "original_candidate_component_ids": json.dumps(components),
                "duplicate_component_count": len(duplicate_components),
                "duplicate_component_ratio": duplicate_ratio,
                "target_frame_ids": sorted(frames_by_candidate.get(str(candidate.get("candidate_id")), set())),
                "candidate_success_rate": candidate.get("candidate_success_rate"),
                "outside_all_related_masks_ratio_mean": candidate.get("outside_all_related_masks_ratio_mean"),
                "same_frame_exclusion_violation_rate": candidate.get("same_frame_exclusion_violation_rate"),
                "underseg_proxy": representative.get("underseg_proxy", False),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return component_to_object, object_rows


def build_local_objectlets(
    support_rows_path: str | Path = "outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv",
    candidate_rows_path: str | Path = "outputs/audit/v53_reprojection_ledger_conflict_veto/candidate_rows.csv",
    ledger_rows_path: str | Path = "outputs/audit/v53_reprojection_ledger_conflict_veto/reprojection_ledger_rows.csv",
    representative_rows_path: str | Path = "outputs/audit/v53_representative_observations_k8_underseg_cap_fixed/representative_mask_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    representative_variant: str = "K8_underseg_capped_partial_repair",
) -> dict[str, Any]:
    support_rows = [row for row in read_csv(_project(support_rows_path)) if str(row.get("variant")) == support_variant]
    candidate_rows = read_csv(_project(candidate_rows_path))
    ledger_rows = read_csv(_project(ledger_rows_path))
    representative_rows = [
        row for row in read_csv(_project(representative_rows_path)) if str(row.get("variant")) == representative_variant
    ]
    representative_by_mask = {str(row.get("mask_observation_id")): row for row in representative_rows}
    all_components = {str(row["component_id"]) for row in support_rows}
    scene_by_component = _component_scene(support_rows)

    rows: list[dict[str, Any]] = []
    object_rows_all: list[dict[str, Any]] = []
    component_maps: dict[str, dict[str, str]] = {}
    object_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    raw_component_to_object = {component: f"{scene_by_component.get(component, '')}|raw:{component}" for component in all_components}
    raw_row = _evaluate_variant(
        "L0_raw_U32_components",
        support_rows,
        raw_component_to_object,
        [],
        all_components,
        scene_by_component,
        raw_component_fallback=True,
    )
    rows.append(raw_row)

    variants = [
        ("L4_conflict_veto_all_disjoint", None, 0.50, "success_outside_large"),
        ("L4_conflict_veto_cap64_disjoint", 64, 0.50, "success_outside_large"),
        ("L4_conflict_veto_cap16_disjoint", 16, 0.50, "success_outside_large"),
        ("L4_conflict_veto_cap4_disjoint", 4, 0.50, "success_outside_large"),
        ("L6_coverage_first_minnew025", None, 0.25, "coverage_first"),
        ("L6_coverage_first_minnew010", None, 0.10, "coverage_first"),
        ("L6_coverage_first_minnew000", None, 0.00, "coverage_first"),
        ("L6_coverage_first_minnew050", None, 0.50, "coverage_first"),
        ("L6_coverage_first_minnew075", None, 0.75, "coverage_first"),
        ("L6_success_large_minnew025", None, 0.25, "success_large"),
        ("L12_repeated_signature_first_minnew025", None, 0.25, "repeated_signature_first"),
        ("L12_repeated_signature_first_minnew010", None, 0.10, "repeated_signature_first"),
    ]
    for variant, cap, min_new_ratio, sort_mode in variants:
        component_to_object, object_rows = _select_objectlets(
            candidate_rows,
            ledger_rows,
            representative_by_mask,
            variant=variant,
            max_components_per_objectlet=cap,
            min_new_component_ratio=min_new_ratio,
            sort_mode=sort_mode,
        )
        component_maps[variant] = component_to_object
        object_rows_by_variant[variant] = object_rows
        object_rows_all.extend(object_rows)
        rows.append(
            _evaluate_variant(
                variant,
                support_rows,
                component_to_object,
                object_rows,
                all_components,
                scene_by_component,
            )
        )
    dynamic_specs = [
        ("L11_dynamic_uncovered_gain_dup010", 0.10, 2.0, 4.0, 0.10),
        ("L11_dynamic_uncovered_gain_dup025", 0.25, 2.0, 4.0, 0.10),
        ("L11_dynamic_uncovered_gain_dup050", 0.50, 2.0, 4.0, 0.10),
    ]
    for variant, duplicate_penalty, outside_weight, conflict_weight, object_penalty in dynamic_specs:
        component_to_object, object_rows = _select_objectlets_dynamic_uncovered_gain(
            candidate_rows,
            ledger_rows,
            representative_by_mask,
            variant=variant,
            duplicate_penalty=duplicate_penalty,
            outside_weight=outside_weight,
            conflict_weight=conflict_weight,
            object_penalty=object_penalty,
        )
        component_maps[variant] = component_to_object
        object_rows_by_variant[variant] = object_rows
        object_rows_all.extend(object_rows)
        rows.append(
            _evaluate_variant(
                variant,
                support_rows,
                component_to_object,
                object_rows,
                all_components,
                scene_by_component,
            )
        )
    mask_only_candidate_rows = _mask_only_candidates(support_rows, representative_rows)
    mask_only_map, mask_only_objects = _select_objectlets(
        mask_only_candidate_rows,
        [],
        representative_by_mask,
        variant="L9_mask_only_representative_support",
        max_components_per_objectlet=None,
        min_new_component_ratio=0.50,
        sort_mode="coverage_first",
    )
    component_maps["L9_mask_only_representative_support"] = mask_only_map
    object_rows_by_variant["L9_mask_only_representative_support"] = mask_only_objects
    object_rows_all.extend(mask_only_objects)
    rows.append(
        _evaluate_variant(
            "L9_mask_only_representative_support",
            support_rows,
            mask_only_map,
            mask_only_objects,
            all_components,
            scene_by_component,
        )
    )
    best_name_for_controls = max(
        [
            row
            for row in rows
            if _method_variant_name(row["variant"])
        ],
        key=lambda row: (parse_float(row.get("4D_ARI")), parse_float(row.get("4D_completeness"))),
        default={},
    ).get("variant")
    if best_name_for_controls:
        best_map = component_maps.get(str(best_name_for_controls), {})
        best_objects = object_rows_by_variant.get(str(best_name_for_controls), [])
        control_specs = [
            ("L7_shuffled_D4RT_assignment_control", _shuffled_component_assignment(best_map, scene_by_component), best_objects),
            ("L8_no_temporal_single_frame_control", _filter_single_frame_assignment(best_map, best_objects), best_objects),
        ]
        for variant, control_map, control_objects in control_specs:
            rows.append(
                _evaluate_variant(
                    variant,
                    support_rows,
                    control_map,
                    control_objects,
                    all_components,
                    scene_by_component,
                )
            )
        row_by_variant = {str(row["variant"]): row for row in rows}
        real_row = row_by_variant.get(str(best_name_for_controls), {})
        shuffled_row = row_by_variant.get("L7_shuffled_D4RT_assignment_control", {})
        no_temporal_row = row_by_variant.get("L8_no_temporal_single_frame_control", {})
        mask_only_row = row_by_variant.get("L9_mask_only_representative_support", {})
        real_row["real_minus_shuffled_ARI"] = parse_float(real_row.get("4D_ARI")) - parse_float(shuffled_row.get("4D_ARI"))
        real_row["real_minus_no_temporal_ARI"] = parse_float(real_row.get("4D_ARI")) - parse_float(
            no_temporal_row.get("4D_ARI")
        )
        real_row["real_minus_mask_only_ARI"] = parse_float(real_row.get("4D_ARI")) - parse_float(mask_only_row.get("4D_ARI"))
    for row in rows:
        row["relaxed_gate"] = _gate(row, strict=False)
        row["success_gate"] = _gate(row, strict=True)
    best_real = max(
        [
            row
            for row in rows
            if _method_variant_name(row["variant"])
        ],
        key=lambda row: (parse_float(row.get("4D_ARI")), parse_float(row.get("4D_completeness"))),
        default={},
    )
    raw = next((row for row in rows if row["variant"] == "L0_raw_U32_components"), {})
    summary = {
        "phase": "v53_local_objectlets",
        "created_at": utc_now(),
        "support_variant": support_variant,
        "candidate_rows_path": str(candidate_rows_path),
        "ledger_rows_path": str(ledger_rows_path),
        "variant_rows": rows,
        "best_real_variant": best_real.get("variant"),
        "best_real_row": best_real,
        "raw_row": raw,
        "real_minus_raw_ARI": parse_float(best_real.get("4D_ARI")) - parse_float(raw.get("4D_ARI")),
        "real_minus_raw_completeness": parse_float(best_real.get("4D_completeness")) - parse_float(raw.get("4D_completeness")),
        "any_relaxed_gate_pass": any(bool(row.get("relaxed_gate", {}).get("pass")) for row in rows if _method_variant_name(row["variant"])),
        "any_success_gate_pass": any(bool(row.get("success_gate", {}).get("pass")) for row in rows if _method_variant_name(row["variant"])),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "selection_metric_rows": rows, "objectlet_rows": object_rows_all}


def write_local_objectlets(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "local_objectlet_summary.json", payload["summary"])
    write_csv(out / "selection_metric_rows.csv", payload["selection_metric_rows"])
    write_csv(out / "objectlet_rows.csv", payload["objectlet_rows"])


__all__ = ["build_local_objectlets", "weighted_partition_metrics", "write_local_objectlets"]
