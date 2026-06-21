from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json
from .v53_local_objectlets import _load_component_ids, weighted_partition_metrics


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [parse_float(row.get(key)) for row in rows if row.get(key) not in ("", None)]
    return _mean(values)


def _method_variant_name(value: Any) -> bool:
    text = str(value)
    return text.startswith("L4_") or text.startswith("L6_") or text.startswith("L11_") or text.startswith("L12_")


def _support_assignments(
    support_rows: list[dict[str, Any]],
    component_to_object: dict[str, str],
    *,
    chunk_id: str,
    raw_component_fallback: bool = False,
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for row in support_rows:
        component_id = str(row.get("component_id"))
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not gt:
            continue
        if component_id in component_to_object:
            pred = component_to_object[component_id]
        elif raw_component_fallback:
            pred = f"{chunk_id}|raw:{component_id}"
        else:
            pred = f"{chunk_id}|unknown:{component_id}"
        rows.append((pred, f"{chunk_id}|gt:{gt}", float(parse_int(row.get("support_count"), 1))))
    return rows


def _evaluate_chunk_variant(
    *,
    chunk_id: str,
    scene: str,
    variant: str,
    support_rows: list[dict[str, Any]],
    component_to_object: dict[str, str],
    object_rows: list[dict[str, Any]],
    raw_component_fallback: bool = False,
) -> dict[str, Any]:
    metrics = weighted_partition_metrics(
        _support_assignments(
            support_rows,
            component_to_object,
            chunk_id=chunk_id,
            raw_component_fallback=raw_component_fallback,
        )
    )
    all_components = {str(row.get("component_id")) for row in support_rows}
    selected_components = set(component_to_object)
    object_frames: dict[str, set[int]] = defaultdict(set)
    for row in object_rows:
        object_id = str(row.get("objectlet_id") or "")
        for frame_id in _load_component_ids(row.get("target_frame_ids")):
            try:
                object_frames[object_id].add(int(frame_id))
            except ValueError:
                continue
    unknown_components = all_components - selected_components
    return {
        "variant": variant,
        "scene": scene,
        "chunk_id": chunk_id,
        "support_row_count": len(support_rows),
        "support_component_count": len(all_components),
        "selected_objectlet_count": len(object_rows),
        "selected_component_count": len(selected_components),
        "component_coverage_ratio": float(len(selected_components) / max(len(all_components), 1)),
        "uncovered_component_ratio": float(len(unknown_components) / max(len(all_components), 1)),
        "mean_components_per_objectlet": _mean([float(parse_int(row.get("component_count"))) for row in object_rows]),
        "duplicate_component_ratio": _safe_mean(object_rows, "duplicate_component_ratio") or 0.0,
        "conflict_rate": _safe_mean(object_rows, "same_frame_exclusion_violation_rate") or 0.0,
        "outside_residual_mean": _safe_mean(object_rows, "outside_all_related_masks_ratio_mean"),
        "temporal_span_mean": _mean([float(len(frames)) for frames in object_frames.values()]),
        "local_ARI": metrics["ARI"],
        "local_purity": metrics["purity"],
        "local_completeness": metrics["completeness"],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _objectlet_component_map(object_rows: list[dict[str, Any]]) -> dict[str, str]:
    component_to_object: dict[str, str] = {}
    for objectlet in object_rows:
        objectlet_id = str(objectlet.get("objectlet_id") or "")
        for component_id in _load_component_ids(objectlet.get("component_ids")):
            component_to_object[str(component_id)] = objectlet_id
    return component_to_object


def _clone_weak_object_row(
    row: dict[str, Any],
    *,
    variant: str,
    components: list[str],
) -> dict[str, Any]:
    cloned = dict(row)
    cloned["variant"] = variant
    cloned["objectlet_id"] = f"{row.get('objectlet_id')}|weak_uncovered"
    cloned["objectlet_source"] = "W1_mask_only_uncovered_restitution"
    cloned["component_count"] = len(components)
    cloned["component_ids"] = json.dumps(components)
    cloned["original_candidate_component_ids"] = row.get("component_ids")
    cloned["weak_support_restitution"] = True
    cloned["uses_gt_for_prediction"] = False
    cloned["uses_gt_for_diagnostic_labels"] = True
    return cloned


def _build_weak_support_rows(
    *,
    base_variant: str,
    mask_only_variant: str,
    base_rows: list[dict[str, Any]],
    mask_only_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, str], list[dict[str, Any]], int]:
    weak_variant = f"W1_{base_variant}_plus_mask_only_uncovered"
    component_to_object = _objectlet_component_map(base_rows)
    weak_rows: list[dict[str, Any]] = [dict(row, variant=weak_variant) for row in base_rows]
    weak_component_count = 0
    for row in mask_only_rows:
        new_components = [
            str(component_id)
            for component_id in _load_component_ids(row.get("component_ids"))
            if str(component_id) not in component_to_object
        ]
        if not new_components:
            continue
        weak_object_id = f"{row.get('objectlet_id')}|weak_uncovered"
        for component_id in new_components:
            component_to_object[component_id] = weak_object_id
        weak_rows.append(_clone_weak_object_row(row, variant=weak_variant, components=new_components))
        weak_component_count += len(new_components)
    return weak_variant, component_to_object, weak_rows, weak_component_count


def _build_weak_expansion_rows(
    *,
    base_variant: str,
    base_rows: list[dict[str, Any]],
    mask_only_rows: list[dict[str, Any]],
    include_uncovered_mask_only: bool,
) -> tuple[str, dict[str, str], list[dict[str, Any]], int, int]:
    suffix = "mask_expand_plus_uncovered" if include_uncovered_mask_only else "mask_expand_same_source"
    weak_variant = f"W{'3' if include_uncovered_mask_only else '2'}_{base_variant}_{suffix}"
    mask_only_components_by_source: dict[str, set[str]] = defaultdict(set)
    for row in mask_only_rows:
        source_mask = str(row.get("source_mask_observation_id") or "")
        if not source_mask:
            continue
        mask_only_components_by_source[source_mask].update(_load_component_ids(row.get("component_ids")))

    component_to_object: dict[str, str] = {}
    weak_rows: list[dict[str, Any]] = []
    expanded_component_count = 0
    for row in base_rows:
        objectlet_id = str(row.get("objectlet_id") or "")
        method_components = [str(component) for component in _load_component_ids(row.get("component_ids"))]
        expanded_components = list(method_components)
        source_mask = str(row.get("source_mask_observation_id") or "")
        for component_id in sorted(mask_only_components_by_source.get(source_mask, set())):
            if component_id not in expanded_components:
                expanded_components.append(component_id)
                expanded_component_count += 1
        for component_id in expanded_components:
            component_to_object[component_id] = objectlet_id
        cloned = dict(row)
        cloned["variant"] = weak_variant
        cloned["component_count"] = len(expanded_components)
        cloned["component_ids"] = json.dumps(expanded_components)
        cloned["method_component_count_before_weak_expansion"] = len(method_components)
        cloned["weak_expanded_component_count"] = len(expanded_components) - len(method_components)
        cloned["weak_support_restitution"] = True
        cloned["weak_support_mode"] = suffix
        cloned["uses_gt_for_prediction"] = False
        cloned["uses_gt_for_diagnostic_labels"] = True
        weak_rows.append(cloned)

    uncovered_component_count = 0
    if include_uncovered_mask_only:
        for row in mask_only_rows:
            new_components = [
                str(component_id)
                for component_id in _load_component_ids(row.get("component_ids"))
                if str(component_id) not in component_to_object
            ]
            if not new_components:
                continue
            weak_object_id = f"{row.get('objectlet_id')}|weak_uncovered"
            for component_id in new_components:
                component_to_object[component_id] = weak_object_id
            weak_rows.append(_clone_weak_object_row(row, variant=weak_variant, components=new_components))
            uncovered_component_count += len(new_components)
    return weak_variant, component_to_object, weak_rows, expanded_component_count, uncovered_component_count


def _gate_from_row(row: dict[str, Any]) -> dict[str, bool]:
    gate = {
        "available": bool(row),
        "local_purity_mean_ge_0.86": parse_float(row.get("local_purity_mean")) >= 0.86,
        "local_completeness_mean_ge_0.70": parse_float(row.get("local_completeness_mean")) >= 0.70,
        "local_conflict_rate_mean_le_0.10": parse_float(row.get("local_conflict_rate_mean"), 9999.0) <= 0.10,
    }
    gate["pass"] = bool(all(gate.values()))
    return gate


def _load_native_available(path: str | Path | None) -> tuple[bool | None, dict[str, Any] | None]:
    if not path:
        return None, None
    native_path = _project(path)
    if not native_path.exists():
        return False, {"missing_native_summary": _rel(native_path)}
    payload = read_json(native_path)
    available = bool(
        payload.get("method_safe_native_support_available")
        or payload.get("native_carrier_materialization_pass")
    )
    return available, payload


def build_v54_local_reproduction(
    *,
    support_rows_path: str | Path,
    chunk_component_rows_path: str | Path,
    chunk_mask_rows_path: str | Path,
    objectlet_summary_path: str | Path,
    objectlet_rows_path: str | Path,
    support_variant: str = "R0_visible_tau0.05",
    native_summary_path: str | Path | None = None,
    enable_weak_support_restitution: bool = False,
) -> dict[str, Any]:
    support_rows_all = [
        row for row in read_csv(_project(support_rows_path)) if str(row.get("variant")) == str(support_variant)
    ]
    chunk_component_rows = read_csv(_project(chunk_component_rows_path))
    chunk_mask_rows = read_csv(_project(chunk_mask_rows_path))
    objectlet_rows_all = read_csv(_project(objectlet_rows_path))
    objectlet_summary = read_json(_project(objectlet_summary_path)) if _project(objectlet_summary_path).exists() else {}

    chunk_scene: dict[str, str] = {}
    components_by_chunk: dict[str, set[str]] = defaultdict(set)
    for row in chunk_component_rows:
        chunk_id = str(row.get("chunk_id"))
        chunk_scene[chunk_id] = str(row.get("scene"))
        components_by_chunk[chunk_id].add(str(row.get("component_id")))

    masks_by_chunk: dict[str, set[str]] = defaultdict(set)
    for row in chunk_mask_rows:
        chunk_id = str(row.get("chunk_id"))
        chunk_scene.setdefault(chunk_id, str(row.get("scene")))
        masks_by_chunk[chunk_id].add(str(row.get("mask_observation_id")))

    support_by_chunk: dict[str, list[dict[str, Any]]] = {}
    for chunk_id in sorted(chunk_scene):
        component_ids = components_by_chunk.get(chunk_id, set())
        mask_ids = masks_by_chunk.get(chunk_id, set())
        support_by_chunk[chunk_id] = [
            row
            for row in support_rows_all
            if str(row.get("component_id")) in component_ids and str(row.get("mask_observation_id")) in mask_ids
        ]

    object_rows_by_variant_chunk: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    variants: set[str] = set()
    for row in objectlet_rows_all:
        variant = str(row.get("variant"))
        chunk_id = str(row.get("chunk_id"))
        variants.add(variant)
        object_rows_by_variant_chunk[(variant, chunk_id)].append(row)

    metric_rows: list[dict[str, Any]] = []
    for chunk_id in sorted(chunk_scene):
        scene = chunk_scene[chunk_id]
        support_rows = support_by_chunk.get(chunk_id, [])
        all_components = {str(row.get("component_id")) for row in support_rows}
        raw_map = {component_id: f"{chunk_id}|raw:{component_id}" for component_id in all_components}
        metric_rows.append(
            _evaluate_chunk_variant(
                chunk_id=chunk_id,
                scene=scene,
                variant="L0_raw_U32_components",
                support_rows=support_rows,
                component_to_object=raw_map,
                object_rows=[],
                raw_component_fallback=True,
            )
        )
        for variant in sorted(variants):
            object_rows = object_rows_by_variant_chunk.get((variant, chunk_id), [])
            component_to_object = _objectlet_component_map(object_rows)
            metric_rows.append(
                _evaluate_chunk_variant(
                    chunk_id=chunk_id,
                    scene=scene,
                    variant=variant,
                    support_rows=support_rows,
                    component_to_object=component_to_object,
                    object_rows=object_rows,
                )
            )
        if enable_weak_support_restitution:
            mask_only_rows = object_rows_by_variant_chunk.get(("L9_mask_only_representative_support", chunk_id), [])
            for variant in sorted(item for item in variants if _method_variant_name(item)):
                object_rows = object_rows_by_variant_chunk.get((variant, chunk_id), [])
                weak_variant, weak_map, weak_rows, weak_component_count = _build_weak_support_rows(
                    base_variant=variant,
                    mask_only_variant="L9_mask_only_representative_support",
                    base_rows=object_rows,
                    mask_only_rows=mask_only_rows,
                )
                weak_metric = _evaluate_chunk_variant(
                    chunk_id=chunk_id,
                    scene=scene,
                    variant=weak_variant,
                    support_rows=support_rows,
                    component_to_object=weak_map,
                    object_rows=weak_rows,
                )
                weak_metric["base_method_variant"] = variant
                weak_metric["weak_support_variant"] = "L9_mask_only_representative_support"
                weak_metric["weak_support_component_count"] = weak_component_count
                weak_metric["weak_support_restitution"] = True
                metric_rows.append(weak_metric)
                for include_uncovered in (False, True):
                    (
                        expanded_variant,
                        expanded_map,
                        expanded_rows,
                        expanded_component_count,
                        uncovered_component_count,
                    ) = _build_weak_expansion_rows(
                        base_variant=variant,
                        base_rows=object_rows,
                        mask_only_rows=mask_only_rows,
                        include_uncovered_mask_only=include_uncovered,
                    )
                    expanded_metric = _evaluate_chunk_variant(
                        chunk_id=chunk_id,
                        scene=scene,
                        variant=expanded_variant,
                        support_rows=support_rows,
                        component_to_object=expanded_map,
                        object_rows=expanded_rows,
                    )
                    expanded_metric["base_method_variant"] = variant
                    expanded_metric["weak_support_variant"] = "L9_mask_only_representative_support"
                    expanded_metric["weak_expanded_component_count"] = expanded_component_count
                    expanded_metric["weak_support_component_count"] = uncovered_component_count
                    expanded_metric["weak_support_restitution"] = True
                    metric_rows.append(expanded_metric)

    summary_rows: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_variant[str(row.get("variant"))].append(row)
    for variant, rows in sorted(by_variant.items()):
        summary_rows.append(
            {
                "variant": variant,
                "chunk_count": len(rows),
                "local_ARI_mean": _safe_mean(rows, "local_ARI"),
                "local_purity_mean": _safe_mean(rows, "local_purity"),
                "local_completeness_mean": _safe_mean(rows, "local_completeness"),
                "local_conflict_rate_mean": _safe_mean(rows, "conflict_rate"),
                "local_outside_residual_mean": _safe_mean(rows, "outside_residual_mean"),
                "temporal_span_mean": _safe_mean(rows, "temporal_span_mean"),
                "selected_objectlet_count_total": sum(parse_int(row.get("selected_objectlet_count")) for row in rows),
                "selected_component_count_total": sum(parse_int(row.get("selected_component_count")) for row in rows),
                "component_coverage_ratio_mean": _safe_mean(rows, "component_coverage_ratio"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    scene_chunk_counts = Counter(chunk_scene.values())
    chunks_per_scene = [float(value) for value in scene_chunk_counts.values()]
    method_rows = [row for row in summary_rows if _method_variant_name(row.get("variant"))]
    weak_support_rows = [row for row in summary_rows if str(row.get("variant", "")).startswith("W")]
    best_method = max(
        method_rows,
        key=lambda row: (
            parse_float(row.get("local_ARI_mean")),
            parse_float(row.get("local_completeness_mean")),
        ),
        default={},
    )
    best_weak_support = max(
        weak_support_rows,
        key=lambda row: (
            parse_float(row.get("local_ARI_mean")),
            parse_float(row.get("local_completeness_mean")),
        ),
        default={},
    )
    mask_only = next((row for row in summary_rows if row.get("variant") == "L9_mask_only_representative_support"), {})
    native_available, native_payload = _load_native_available(native_summary_path)

    multi_chunk_scene_count_ge_3_value = sum(1 for count in scene_chunk_counts.values() if count >= 3)
    local_gate = {
        "multi_chunk_scene_count_ge_3": multi_chunk_scene_count_ge_3_value >= 3,
        "chunks_per_scene_mean_ge_3": (_mean(chunks_per_scene) or 0.0) >= 3.0,
        "local_purity_mean_ge_0.86": parse_float(best_method.get("local_purity_mean")) >= 0.86,
        "local_completeness_mean_ge_0.70": parse_float(best_method.get("local_completeness_mean")) >= 0.70,
        "local_conflict_rate_mean_le_0.10": parse_float(best_method.get("local_conflict_rate_mean"), 9999.0) <= 0.10,
    }
    if native_available is not None:
        local_gate["native_carrier_field_available"] = bool(native_available)
    local_gate["pass"] = bool(all(local_gate.values()))

    weak_gate = {
        "mask_only_restitution_available": bool(mask_only),
        "mask_only_purity_mean_ge_0.86": parse_float(mask_only.get("local_purity_mean")) >= 0.86,
        "mask_only_completeness_mean_ge_0.70": parse_float(mask_only.get("local_completeness_mean")) >= 0.70,
        "mask_only_conflict_rate_mean_le_0.10": parse_float(mask_only.get("local_conflict_rate_mean"), 9999.0) <= 0.10,
    }
    weak_gate["pass"] = bool(all(weak_gate.values()))
    weak_support_gate = _gate_from_row(best_weak_support)
    weak_support_gate["note"] = (
        "Diagnostic repair only: method objectlets keep priority, mask-only support fills only components "
        "uncovered by the method variant."
    )

    summary = {
        "phase": "v54_local_reproduction",
        "created_at": utc_now(),
        "support_rows_path": _rel(support_rows_path),
        "chunk_component_rows_path": _rel(chunk_component_rows_path),
        "chunk_mask_rows_path": _rel(chunk_mask_rows_path),
        "objectlet_summary_path": _rel(objectlet_summary_path),
        "objectlet_rows_path": _rel(objectlet_rows_path),
        "support_variant": support_variant,
        "objectlet_global_best_variant": objectlet_summary.get("best_real_variant"),
        "objectlet_global_best_row": objectlet_summary.get("best_real_row"),
        "multi_chunk_scene_count": multi_chunk_scene_count_ge_3_value,
        "multi_chunk_scene_count_definition": "number of scenes with at least 3 chunks",
        "chunks_per_scene_mean": _mean(chunks_per_scene),
        "chunks_by_scene": dict(sorted(scene_chunk_counts.items())),
        "best_method_variant": best_method.get("variant"),
        "best_method_row": best_method,
        "enable_weak_support_restitution": bool(enable_weak_support_restitution),
        "best_weak_support_variant": best_weak_support.get("variant"),
        "best_weak_support_row": best_weak_support,
        "mask_only_restitution_row": mask_only,
        "local_gate": local_gate,
        "weak_mask_only_restitution_gate": weak_gate,
        "weak_support_restitution_gate": weak_support_gate,
        "native_summary_path": _rel(native_summary_path) if native_summary_path else None,
        "native_summary": native_payload,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "scoring_note": (
            "Chunk-local metrics filter support rows by chunk mask ids and chunk component ids; "
            "cross-chunk identity continuity is reserved for v54 history memory."
        ),
    }
    return {"summary": summary, "local_metric_rows": metric_rows, "local_variant_summary_rows": summary_rows}


def write_v54_local_reproduction(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "local_reproduction_summary.json", payload["summary"])
    write_csv(out / "local_metric_rows.csv", payload["local_metric_rows"])
    write_csv(out / "local_variant_summary_rows.csv", payload["local_variant_summary_rows"])


__all__ = ["build_v54_local_reproduction", "write_v54_local_reproduction"]
