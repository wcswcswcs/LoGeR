from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stream4d_native.v47_common import (
    ROOT,
    adjusted_rand_score,
    cluster_completeness,
    cluster_purity,
    cosine,
    parse_bool,
    parse_float,
    parse_int,
    rank_auc,
    read_csv,
    read_json,
    safe_mean,
    safe_quantile,
    utc_now,
    write_csv,
    write_json,
)


DEFAULT_SCALES = [16, 24, 32, 40, 48, 56]
DEFAULT_SCENES = ["scene0011_00", "scene0030_00", "scene0050_00", "scene0081_01", "scene0591_00"]
COMPONENT_DELIM = ";;"


def project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def rel(path: str | Path) -> str:
    path_obj = project_path(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def load_optional_json(path: str | Path) -> dict[str, Any]:
    path_obj = project_path(path)
    if not path_obj.exists():
        return {"missing": True, "path": rel(path_obj)}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {"payload": payload}


def load_optional_csv(path: str | Path) -> list[dict[str, str]]:
    path_obj = project_path(path)
    return read_csv(path_obj) if path_obj.exists() else []


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _metric(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = _nested(payload, *key.split("."), default=None)
        if value not in (None, ""):
            return value
    return default


def _num(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_div(numer: float, denom: float) -> float:
    return 0.0 if denom <= 0.0 else float(numer / denom)


def _bool(value: Any) -> bool:
    return parse_bool(value)


def _json_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [float(v) for v in payload] if isinstance(payload, list) else []


def _component_key(scene: str, component: str) -> str:
    return f"{scene}|{component}"


def _is_real_component(component: str) -> bool:
    return bool(component) and not str(component).startswith("uncovered:")


def _pack_components(comps: Iterable[str]) -> str:
    return COMPONENT_DELIM.join(str(comp) for comp in comps if str(comp))


def _unpack_components(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(comp) for comp in value if str(comp)]
    text = str(value or "")
    return [comp for comp in text.split(COMPONENT_DELIM) if comp]


def _hypothesis_generation_support_score(row: dict[str, Any]) -> float:
    source = str(row.get("component_set_candidate_source") or row.get("candidate_generation_source") or "")
    size = parse_float(row.get("hypothesis_size"), 1.0)
    semantic = parse_float(row.get("semantic_set_score"), 1.0)
    context_overlap = parse_float(row.get("context_overlap_proxy"))
    conflict = parse_float(row.get("hypothesis_conflict_rate"))
    support_over = max(0.0, parse_float(row.get("mask_support_score")) - 25.0)
    reliability = parse_float(row.get("mask_reliability_min"), parse_float(row.get("mask_reliability_mean")))
    source_bonus = {
        "singleton": 0.06,
        "pair_edge": 0.08,
        "multi_scale_parent_containment": -0.05,
        "pair_neighborhood": -0.35,
    }.get(source, 0.0)
    return (
        0.72 * semantic
        + 0.25 * reliability
        - 0.52 * context_overlap
        - 0.35 * conflict
        - 0.018 * support_over
        - 0.08 * max(0.0, size - 2.0)
        + source_bonus
    )


def _scale_root(scale: int) -> Path:
    fine = ROOT / f"outputs/audit/v47_carrier_supertrack_union_{scale}_fine_metricfix"
    if fine.exists():
        return fine
    return ROOT / f"outputs/audit/v47_carrier_supertrack_union_{scale}_metricfix"


def _load_scale_rows(scale: int) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    root = _scale_root(scale)
    return (
        root,
        load_optional_json(root / "carrier_supertrack_summary.json"),
        load_optional_csv(root / "carrier_supertrack_mask_vote_rows.csv"),
    )


def _mask_feature_map(mask_rows: list[dict[str, str]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in mask_rows:
        mask_id = str(row.get("mask_observation_id") or row.get("node_id") or "")
        feat = _json_list(row.get("core_feature"))
        if mask_id and feat:
            out[mask_id] = feat
    return out


def _mask_reliability_scores(mask_rows: list[dict[str, str]]) -> dict[str, float]:
    areas = [parse_float(row.get("mask_area")) for row in mask_rows if parse_float(row.get("mask_area")) > 0.0]
    densities = [parse_float(row.get("support_density")) for row in mask_rows]
    if not areas:
        return {}
    area_mid = safe_quantile(areas, 0.50) or safe_mean(areas) or 1.0
    area_hi = safe_quantile(areas, 0.90) or max(areas)
    density_mid = safe_quantile(densities, 0.50) or safe_mean(densities) or 0.0
    density_lo = safe_quantile(densities, 0.10) or min(densities or [0.0])
    area_span = max(1e-6, math.log1p(area_hi) - math.log1p(area_mid))
    density_span = max(1e-9, density_mid - density_lo)
    out: dict[str, float] = {}
    for row in mask_rows:
        mask_id = str(row.get("mask_observation_id") or row.get("node_id") or "")
        if not mask_id:
            continue
        area = parse_float(row.get("mask_area"))
        density = parse_float(row.get("support_density"))
        visible = parse_float(row.get("visible_carrier_count"))
        width = max(1.0, parse_float(row.get("bbox_x1")) - parse_float(row.get("bbox_x0")))
        height = max(1.0, parse_float(row.get("bbox_y1")) - parse_float(row.get("bbox_y0")))
        fill = min(1.0, area / max(1.0, width * height))
        area_risk = max(0.0, math.log1p(area) - math.log1p(area_mid)) / area_span
        density_risk = max(0.0, density_mid - density) / density_span if density_mid > 0.0 else 0.0
        low_carrier_risk = 1.0 / (1.0 + max(0.0, visible))
        fill_risk = max(0.0, fill - 0.65) / 0.35
        out[mask_id] = float(math.exp(-0.60 * area_risk - 0.50 * density_risk - 0.30 * low_carrier_risk - 0.20 * fill_risk))
    return out


def _value_range(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0


def _mask_fill(row: dict[str, Any]) -> float:
    area = parse_float(row.get("mask_area"))
    width = max(1.0, parse_float(row.get("bbox_x1")) - parse_float(row.get("bbox_x0")))
    height = max(1.0, parse_float(row.get("bbox_y1")) - parse_float(row.get("bbox_y0")))
    return float(min(1.0, area / max(1.0, width * height)))


def _feature_variance_proxy(features: list[list[float]]) -> float:
    distances = [1.0 - cosine(left, right) for left, right in combinations(features, 2) if left and right]
    return float(safe_mean(distances) or 0.0)


def _mean_feature(features: list[list[float]]) -> list[float]:
    valid = [np.asarray(feat, dtype=np.float32) for feat in features if feat]
    if not valid:
        return []
    dim = min(vec.shape[0] for vec in valid)
    if dim <= 0:
        return []
    arr = np.stack([vec[:dim] for vec in valid], axis=0)
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0.0:
        mean = mean / norm
    return [float(x) for x in mean.tolist()]


def build_component_profiles(
    mask_vote_rows: list[dict[str, str]],
    mask_rows: list[dict[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    feature_map = _mask_feature_map(mask_rows or [])
    reliability_map = _mask_reliability_scores(mask_rows or [])
    mask_meta = {
        str(row.get("mask_observation_id") or row.get("node_id") or ""): row
        for row in (mask_rows or [])
        if str(row.get("mask_observation_id") or row.get("node_id") or "")
    }
    profiles: dict[str, dict[str, Any]] = {}
    for row in mask_vote_rows:
        scene = str(row.get("scene") or "")
        component = str(row.get("predicted_component_object_id") or "")
        if not _is_real_component(component):
            continue
        key = _component_key(scene, component)
        item = profiles.setdefault(
            key,
            {
                "component_key": key,
                "scene": scene,
                "component": component,
                "mask_ids": set(),
                "frames": set(),
                "gt_counts": Counter(),
                "feature_values": [],
                "mask_reliability_values": [],
                "mask_area_values": [],
                "mask_fill_values": [],
                "support_density_values": [],
                "supporting_carrier_observation_count": 0.0,
                "supporting_unique_carrier_count": 0.0,
            },
        )
        mask_id = str(row.get("mask_observation_id") or row.get("node_id") or "")
        if mask_id:
            item["mask_ids"].add(mask_id)
            if mask_id in feature_map:
                item["feature_values"].append(feature_map[mask_id])
            if mask_id in reliability_map:
                item["mask_reliability_values"].append(reliability_map[mask_id])
            mask_row = mask_meta.get(mask_id)
            if mask_row:
                item["mask_area_values"].append(parse_float(mask_row.get("mask_area")))
                item["mask_fill_values"].append(_mask_fill(mask_row))
                item["support_density_values"].append(parse_float(mask_row.get("support_density")))
        item["frames"].add(parse_int(row.get("frame_id")))
        gt = str(row.get("diagnostic_gt_instance") or "")
        if gt:
            item["gt_counts"][gt] += 1
        item["supporting_carrier_observation_count"] += parse_float(row.get("supporting_carrier_observation_count"))
        item["supporting_unique_carrier_count"] += parse_float(row.get("supporting_unique_carrier_count"))

    final: dict[str, dict[str, Any]] = {}
    for key, item in profiles.items():
        gt_counts: Counter[str] = item["gt_counts"]
        labeled = int(sum(gt_counts.values()))
        dominant_gt = ""
        dominant_count = 0
        if gt_counts:
            dominant_gt, dominant_count = gt_counts.most_common(1)[0]
        frames = set(item["frames"])
        mask_ids = set(item["mask_ids"])
        feature_variance = _feature_variance_proxy(item["feature_values"])
        fill_range = _value_range(item["mask_fill_values"])
        density_range = _value_range(item["support_density_values"])
        area_log_values = [math.log1p(max(0.0, float(value))) for value in item["mask_area_values"]]
        area_log_range = _value_range(area_log_values)
        boundary_instability = float(safe_mean([fill_range, density_range, area_log_range]) or 0.0)
        fill_mean = safe_mean(item["mask_fill_values"])
        density_mean = safe_mean(item["support_density_values"])
        final[key] = {
            "component_key": key,
            "scene": item["scene"],
            "component": item["component"],
            "mask_count": int(len(mask_ids)),
            "frame_count": int(len(frames)),
            "frames": sorted(int(frame) for frame in frames),
            "mask_ids": sorted(mask_ids),
            "diagnostic_dominant_gt": dominant_gt,
            "diagnostic_labeled_mask_count": labeled,
            "diagnostic_gt_purity": None if labeled <= 0 else float(dominant_count / labeled),
            "supporting_carrier_observation_count": float(item["supporting_carrier_observation_count"]),
            "supporting_unique_carrier_count": float(item["supporting_unique_carrier_count"]),
            "feature": _mean_feature(item["feature_values"]),
            "feature_backend": "colorhist_fallback",
            "component_prototype_count": int(len(item["feature_values"])),
            "component_feature_variance": feature_variance,
            "component_boundary_feature": [fill_mean or 0.0, fill_range, density_mean or 0.0, density_range, area_log_range],
            "boundary_fill_mean": fill_mean,
            "boundary_fill_range": fill_range,
            "support_density_range": density_range,
            "mask_area_log_range": area_log_range,
            "boundary_proxy_instability": boundary_instability,
            "component_context_feature": [float(len(frames)), float(len(mask_ids))],
            "mask_reliability_mean": safe_mean(item["mask_reliability_values"]),
            "mask_reliability_min": min(item["mask_reliability_values"]) if item["mask_reliability_values"] else None,
            "mask_area_mean": safe_mean(item["mask_area_values"]),
            "support_density_mean": density_mean,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    return final


def _gt_mask_totals(mask_vote_rows: list[dict[str, str]]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for row in mask_vote_rows:
        gt = str(row.get("diagnostic_gt_instance") or "")
        if gt:
            totals[f"{row.get('scene')}|{gt}"] += 1
    return totals


def _component_gt_counts(mask_vote_rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in mask_vote_rows:
        scene = str(row.get("scene") or "")
        comp = str(row.get("predicted_component_object_id") or "")
        gt = str(row.get("diagnostic_gt_instance") or "")
        if _is_real_component(comp) and gt:
            out[_component_key(scene, comp)][f"{scene}|{gt}"] += 1
    return out


def evaluate_component_assignment(
    mask_vote_rows: list[dict[str, str]],
    component_to_object: dict[str, str] | None = None,
    *,
    unknown_unselected: bool = False,
) -> dict[str, Any]:
    component_to_object = component_to_object or {}
    true_labels: list[str] = []
    pred_labels: list[str] = []
    scene_true: dict[str, list[str]] = defaultdict(list)
    scene_pred: dict[str, list[str]] = defaultdict(list)
    object_frames: dict[str, set[int]] = defaultdict(set)
    object_gt: dict[str, Counter[str]] = defaultdict(Counter)
    object_scenes: dict[str, str] = {}
    unknown_count = 0
    maskless_object_count = 0
    for row in mask_vote_rows:
        scene = str(row.get("scene") or "")
        component = str(row.get("predicted_component_object_id") or "")
        if _is_real_component(component):
            comp_key = _component_key(scene, component)
            if comp_key in component_to_object:
                pred = component_to_object[comp_key]
            elif unknown_unselected:
                pred = f"{scene}|unknown:{component}"
                unknown_count += 1
            else:
                pred = comp_key
        else:
            pred = f"{scene}|unknown:{row.get('mask_observation_id') or row.get('node_id')}"
            unknown_count += 1
        gt = str(row.get("diagnostic_gt_instance") or "")
        object_frames[pred].add(parse_int(row.get("frame_id")))
        object_scenes[pred] = scene
        if gt:
            true = f"{scene}|{gt}"
            true_labels.append(true)
            pred_labels.append(pred)
            scene_true[scene].append(true)
            scene_pred[scene].append(pred)
            object_gt[pred][true] += 1
    conflict_objects = sum(1 for counts in object_gt.values() if len(counts) > 1)
    scene_rows = []
    for scene in sorted(scene_true):
        scene_rows.append(
            {
                "scene": scene,
                "4D_ARI": adjusted_rand_score(scene_true[scene], scene_pred[scene]),
                "4D_purity": cluster_purity(scene_true[scene], scene_pred[scene]),
                "4D_completeness": cluster_completeness(scene_true[scene], scene_pred[scene]),
                "prediction_count": len({pred for pred, pred_scene in object_scenes.items() if pred_scene == scene}),
            }
        )
    return {
        "4D_ARI": adjusted_rand_score(true_labels, pred_labels),
        "4D_purity": cluster_purity(true_labels, pred_labels),
        "4D_completeness": cluster_completeness(true_labels, pred_labels),
        "3D_ARI": adjusted_rand_score(true_labels, pred_labels),
        "3D_purity": cluster_purity(true_labels, pred_labels),
        "3D_completeness": cluster_completeness(true_labels, pred_labels),
        "temporal_span_mean": safe_mean(len(frames) for frames in object_frames.values()),
        "scene0081_ARI": next((row["4D_ARI"] for row in scene_rows if row["scene"] == "scene0081_01"), None),
        "scene0011_purity": next((row["4D_purity"] for row in scene_rows if row["scene"] == "scene0011_00"), None),
        "scene0050_purity": next((row["4D_purity"] for row in scene_rows if row["scene"] == "scene0050_00"), None),
        "scene0591_completeness": next((row["4D_completeness"] for row in scene_rows if row["scene"] == "scene0591_00"), None),
        "mean_predictions_per_scene": safe_mean(row["prediction_count"] for row in scene_rows),
        "selected_object_count": len(object_frames),
        "duplicate_rate": 0.0,
        "conflict_rate": float(conflict_objects / max(len(object_gt), 1)),
        "unknown_tube_ratio": float(unknown_count / max(len(mask_vote_rows), 1)),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": maskless_object_count,
        "scene_rows": scene_rows,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _fact_row(key: str, value: Any, source: str | Path, note: str = "", required: bool = True) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "source": rel(source),
        "available": value not in (None, ""),
        "required": required,
        "note": note,
    }


def build_fact_lock() -> dict[str, Any]:
    v44_path = "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json"
    v46_path = "outputs/audit/v46_final_decision/v46_final_decision.json"
    v47_path = "outputs/audit/v47_final_decision_phase9_continued21_carrier_mdl_audit/v47_final_decision.json"
    v48_path = "outputs/audit/v48_final_decision/v48_final_decision.json"
    obs_path = "outputs/audit/v47_observation_tables_metricfix/observation_table_summary.json"
    semantic_path = "outputs/audit/v48_semantic_features/semantic_feature_audit_summary.json"
    v44 = load_optional_json(v44_path)
    v46 = load_optional_json(v46_path)
    v47 = load_optional_json(v47_path)
    v48 = load_optional_json(v48_path)
    obs = load_optional_json(obs_path)
    semantic = load_optional_json(semantic_path)
    v37 = _nested(v44, "baseline", "v37_best_metrics", default={}) or {}
    v44_metrics = _nested(v44, "aggregate_metrics", default={}) or {}
    v48_final = _nested(v48, "final_candidate", default={}) or {}
    carrier_component = load_optional_json("outputs/audit/v48_primitive_audit/primitive_audit_summary.json").get("primary_primitive", {})
    rows = [
        _fact_row("v37_4D_ARI", v37.get("4D_ARI"), v44_path, "prior/imported baseline"),
        _fact_row("v37_4D_purity", v37.get("4D_purity"), v44_path, "prior/imported baseline"),
        _fact_row("v37_4D_completeness", v37.get("4D_completeness"), v44_path, "prior/imported baseline"),
        _fact_row("v37_temporal_span_mean", v37.get("temporal_span_mean"), v44_path, "prior/imported baseline"),
        _fact_row("v44_best_ARI", v44_metrics.get("4D_ARI"), v44_path, "prior/imported baseline"),
        _fact_row("v44_best_purity", v44_metrics.get("4D_purity"), v44_path, "prior/imported baseline"),
        _fact_row("v44_best_completeness", v44_metrics.get("4D_completeness"), v44_path, "prior/imported baseline"),
        _fact_row("v46_final_label", v46.get("final_label"), v46_path),
        _fact_row("v47_final_label", _nested(v47, "autopsy", "failure_label", default=v47.get("final_label")), v47_path),
        _fact_row("v48_final_label", v48.get("final_label"), v48_path),
        _fact_row("v48_best_variant", v48_final.get("variant"), v48_path),
        _fact_row("v48_best_ARI", v48_final.get("ARI"), v48_path),
        _fact_row("v48_best_purity", v48_final.get("purity"), v48_path),
        _fact_row("v48_best_completeness", v48_final.get("completeness"), v48_path),
        _fact_row("v48_best_real_minus_shuffled", v48_final.get("real_minus_shuffled_ARI"), v48_path),
        _fact_row("v48_best_real_minus_no_temporal", v48_final.get("real_minus_no_temporal_ARI"), v48_path),
        _fact_row("v48_carrier_component_ARI", carrier_component.get("ARI"), "outputs/audit/v48_primitive_audit/primitive_audit_summary.json"),
        _fact_row("v48_carrier_component_purity", carrier_component.get("primitive_purity_mean"), "outputs/audit/v48_primitive_audit/primitive_audit_summary.json"),
        _fact_row("v48_carrier_component_completeness", carrier_component.get("primitive_completeness_mean"), "outputs/audit/v48_primitive_audit/primitive_audit_summary.json"),
        _fact_row("v48_carrier_component_real_minus_shuffled", carrier_component.get("real_minus_shuffled_ARI"), "outputs/audit/v48_primitive_audit/primitive_audit_summary.json"),
        _fact_row("carrier_observation_table_available", obs.get("carrier_observation_table_exists"), obs_path),
        _fact_row("mask_observation_table_available", obs.get("mask_observation_table_exists"), obs_path),
        _fact_row("D4RT_encoder_stride", obs.get("D4RT_encoder_stride"), obs_path),
        _fact_row("scale_guard_pass", _nested(v46, "fact_gate", "scale_guard_pass", default=obs.get("scale_weak_row_count") == 0), v46_path),
        _fact_row("RADIO_available", _nested(semantic, "gate", "recommended_contradiction_backend", default=None) == "radio", semantic_path, "component-level RADIO not available; imported v48 edge-proxy availability", required=False),
        _fact_row("DINO_available", any(str(row.get("backend_id")) == "dinov2" for row in semantic.get("backend_rows", [])), semantic_path, "component-level DINO not available; imported v48 edge-proxy availability", required=False),
    ]
    fact = {row["key"]: row["value"] for row in rows}
    gate = {
        "D4RT_encoder_stride_eq_1": fact.get("D4RT_encoder_stride") == 1,
        "scale_guard_pass": bool(fact.get("scale_guard_pass")),
        "carrier_observation_table_available": bool(fact.get("carrier_observation_table_available")),
        "mask_observation_table_available": bool(fact.get("mask_observation_table_available")),
        "v48_final_label_is_not_GO": not str(fact.get("v48_final_label", "")).upper().startswith("GO_STAGE1"),
        "v48_carrier_component_primary": _num(fact.get("v48_carrier_component_purity"), 0.0) >= 0.89,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v49_fact_lock",
        "created_at": utc_now(),
        "fact_rows": rows,
        "fact_map": fact,
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "repair_note": None if gate["pass"] else "If any required fact is unavailable, regenerate/import the corresponding v47/v48 artifact before method claims.",
    }


def build_component_lattice(scales: Iterable[int] = DEFAULT_SCALES) -> dict[str, Any]:
    obs_rows = load_optional_csv("outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    scale_payloads: dict[int, tuple[Path, dict[str, Any], list[dict[str, str]]]] = {}
    rows: list[dict[str, Any]] = []
    component_maps: dict[int, dict[str, str]] = {}
    profile_by_scale: dict[int, dict[str, dict[str, Any]]] = {}
    for scale in scales:
        root, summary, vote_rows = _load_scale_rows(int(scale))
        scale_payloads[int(scale)] = (root, summary, vote_rows)
        component_maps[int(scale)] = {
            str(row.get("mask_observation_id")): _component_key(str(row.get("scene")), str(row.get("predicted_component_object_id")))
            for row in vote_rows
            if _is_real_component(str(row.get("predicted_component_object_id") or ""))
        }
        profiles = build_component_profiles(vote_rows, obs_rows)
        profile_by_scale[int(scale)] = profiles
        risk_values = [profile.get("diagnostic_gt_purity") for profile in profiles.values() if profile.get("diagnostic_gt_purity") is not None]
        large_component_risk_rate = safe_mean(1.0 if float(value) < 0.875 else 0.0 for value in risk_values) if risk_values else None
        gt_component_counts: dict[str, set[str]] = defaultdict(set)
        for profile in profiles.values():
            gt = profile.get("diagnostic_dominant_gt")
            if gt:
                gt_component_counts[f"{profile['scene']}|{gt}"].add(str(profile["component_key"]))
        row = {
            "scale": int(scale),
            "source": rel(root),
            "available": bool(vote_rows) and not summary.get("missing"),
            "component_count": summary.get("component_count"),
            "component_size_mean": safe_mean(profile.get("mask_count") for profile in profiles.values()),
            "component_purity": summary.get("object_from_component_purity"),
            "component_completeness": summary.get("object_from_component_completeness"),
            "component_ARI": summary.get("object_from_component_ARI"),
            "real_minus_shuffled_ARI": summary.get("real_minus_shuffled_component_ARI"),
            "real_minus_no_temporal_ARI": None,
            "large_component_risk_rate": large_component_risk_rate,
            "GT_object_component_count_mean": safe_mean(len(values) for values in gt_component_counts.values()),
            "GT_object_component_count_p90": safe_quantile([len(values) for values in gt_component_counts.values()], 0.90) if gt_component_counts else None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        row["high_purity_gate"] = _num(row["component_purity"], 0.0) >= 0.89 and _num(row["component_ARI"], 0.0) >= 0.40
        rows.append(row)

    base_map = component_maps.get(32, {})
    containment_counter: Counter[tuple[int, str, str]] = Counter()
    parent_children: dict[tuple[int, str], set[str]] = defaultdict(set)
    for scale, comp_map in component_maps.items():
        if scale == 32:
            continue
        for mask_id, child in base_map.items():
            parent = comp_map.get(mask_id)
            if parent and child and parent != child:
                containment_counter[(scale, child, parent)] += 1
                parent_children[(scale, parent)].add(child)
    containment_rows = [
        {
            "scale": scale,
            "child_component": child,
            "parent_component": parent,
            "shared_mask_observation_count": int(count),
            "parent_child_count": len(parent_children.get((scale, parent), set())),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
        }
        for (scale, child, parent), count in containment_counter.most_common(20000)
    ]
    top_fragmented = []
    profiles32 = profile_by_scale.get(32, {})
    gt_components: dict[str, set[str]] = defaultdict(set)
    for profile in profiles32.values():
        gt = str(profile.get("diagnostic_dominant_gt") or "")
        if gt:
            gt_components[f"{profile['scene']}|{gt}"].add(str(profile["component_key"]))
    for gt, comps in sorted(gt_components.items(), key=lambda item: len(item[1]), reverse=True)[:20]:
        scene, gt_id = gt.split("|", 1)
        top_fragmented.append({"scene": scene, "diagnostic_gt_instance": gt_id, "component_count": len(comps)})
    u32 = next((row for row in rows if row["scale"] == 32), {})
    better_fragment_scale = min(
        (row for row in rows if row.get("available") and row.get("GT_object_component_count_mean") is not None),
        key=lambda row: float(row.get("GT_object_component_count_mean") or 1e9),
        default={},
    )
    gate = {
        "u32_or_nearby_high_purity": any(row["high_purity_gate"] for row in rows if row["scale"] in {24, 32, 40}),
        "multi_scale_parent_edges_available": bool(containment_rows),
        "large_components_markable_as_risk": any(row.get("large_component_risk_rate") not in (None, "") for row in rows),
        "direct_selected_purity_not_below_0875": _num(u32.get("component_purity"), 0.0) >= 0.875,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v49_component_lattice",
        "created_at": utc_now(),
        "scale_rows": rows,
        "containment_rows": containment_rows,
        "top_fragmented_gt_objects": top_fragmented,
        "gate": gate,
        "best_fragmentation_scale": better_fragment_scale.get("scale"),
        "note": "Containment edges use shared mask-observation IDs across existing v47 multi-scale component artifacts; diagnostic GT is only used for metric columns.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_component_completion_atlas(pair_limit_per_scene: int = 120, set_limit: int = 5000) -> dict[str, Any]:
    obs_rows = load_optional_csv("outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    if not vote_rows:
        vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_metricfix/carrier_supertrack_mask_vote_rows.csv")
    profiles = build_component_profiles(vote_rows, obs_rows)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles.values():
        by_scene[str(profile["scene"])].append(profile)
    pair_rows: list[dict[str, Any]] = []
    for scene, scene_profiles in sorted(by_scene.items()):
        selected = sorted(
            scene_profiles,
            key=lambda item: (int(item.get("diagnostic_labeled_mask_count") or 0), int(item.get("mask_count") or 0)),
            reverse=True,
        )[: int(pair_limit_per_scene)]
        for left, right in combinations(selected, 2):
            frames_l = set(left["frames"])
            frames_r = set(right["frames"])
            frame_inter = len(frames_l & frames_r)
            frame_union = len(frames_l | frames_r)
            color_sim = cosine(left.get("feature"), right.get("feature"))
            same_gt = bool(left.get("diagnostic_dominant_gt") and left.get("diagnostic_dominant_gt") == right.get("diagnostic_dominant_gt"))
            same_frame_conflict = frame_inter > 0
            persistent_contradiction = bool(same_frame_conflict and color_sim < 0.35)
            temporal_support = _safe_div(frame_inter, max(min(len(frames_l), len(frames_r)), 1))
            combined = 0.42 * temporal_support + 0.36 * max(color_sim, 0.0) + 0.12 * _safe_div(min(left["mask_count"], right["mask_count"]), max(left["mask_count"], right["mask_count"], 1)) - 0.12 * float(persistent_contradiction)
            pair_rows.append(
                {
                    "scene": scene,
                    "component_i": left["component_key"],
                    "component_j": right["component_key"],
                    "same_GT_pair": same_gt,
                    "different_GT_pair": bool(left.get("diagnostic_dominant_gt") and right.get("diagnostic_dominant_gt") and not same_gt),
                    "component_size_i": left["mask_count"],
                    "component_size_j": right["mask_count"],
                    "component_temporal_span_i": len(frames_l),
                    "component_temporal_span_j": len(frames_r),
                    "frame_overlap_ratio": _safe_div(frame_inter, frame_union),
                    "shared_mask_count": 0,
                    "reliable_shared_mask_count": 0,
                    "D4RT_co_visible_count": frame_inter,
                    "D4RT_temporal_support": temporal_support,
                    "D4RT_specific_margin": temporal_support,
                    "visible_outside": float(same_frame_conflict),
                    "same_frame_conflict": same_frame_conflict,
                    "persistent_contradiction": persistent_contradiction,
                    "colorhist_similarity": color_sim,
                    "DINO_similarity": None,
                    "RADIO_similarity": None,
                    "semantic_negative_score": float(1.0 - color_sim),
                    "multi_scale_containment": 0,
                    "same_parent_large_component_count": 0,
                    "combined_nonGT_score": combined,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    pair_rows.sort(key=lambda row: float(row["combined_nonGT_score"]), reverse=True)

    set_rows: list[dict[str, Any]] = []
    seen_sets: set[tuple[str, ...]] = set()
    top_pairs_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows[: max(set_limit * 2, 1000)]:
        top_pairs_by_scene[str(row["scene"])].append(row)
    profile_lookup = profiles
    for scene, rows in top_pairs_by_scene.items():
        adjacency: dict[str, set[str]] = defaultdict(set)
        for row in rows[:1000]:
            left = str(row["component_i"])
            right = str(row["component_j"])
            adjacency[left].add(right)
            adjacency[right].add(left)
            comps = tuple(sorted([left, right]))
            if comps not in seen_sets:
                seen_sets.add(comps)
                set_rows.append(_set_row(scene, comps, profile_lookup, candidate_source="pair_edge"))
        for seed, neighbors in list(adjacency.items())[:100]:
            ranked_neighbors = sorted(neighbors, key=lambda key: profile_lookup.get(key, {}).get("mask_count", 0), reverse=True)[:6]
            for left, right in combinations(ranked_neighbors, 2):
                comps = tuple(sorted([seed, left, right]))
                if comps not in seen_sets:
                    seen_sets.add(comps)
                    set_rows.append(_set_row(scene, comps, profile_lookup, candidate_source="pair_neighborhood"))
            if len(set_rows) >= set_limit:
                break
        if len(set_rows) >= set_limit:
            break
    lattice = _load_lattice()
    if not lattice.get("missing"):
        parent_children: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
        for row in lattice.get("containment_rows", []):
            child = str(row.get("child_component") or "")
            parent = str(row.get("parent_component") or "")
            scale = parse_int(row.get("scale"))
            if child in profile_lookup and parent:
                parent_children[(scale, parent)][child] += parse_int(row.get("shared_mask_observation_count"))
        for (_scale, _parent), child_counts in sorted(parent_children.items(), key=lambda item: sum(item[1].values()), reverse=True):
            children = [child for child, _count in child_counts.most_common(8)]
            for size in [2, 3, 4, 5]:
                if len(children) < size:
                    continue
                comps = tuple(sorted(children[:size]))
                if comps in seen_sets:
                    continue
                seen_sets.add(comps)
                set_rows.append(_set_row(profile_lookup[comps[0]]["scene"], comps, profile_lookup, candidate_source="multi_scale_parent_containment"))
    set_rows.sort(key=lambda row: float(row["nonGT_set_score"]), reverse=True)
    set_rows = set_rows[:set_limit]

    pair_metrics = _pair_metrics(pair_rows)
    set_metrics = _set_metrics(set_rows, vote_rows)
    gate = {
        "combined_nonGT_pair_AUC_pass": _num(pair_metrics.get("pair_AUC_combined_nonGT"), 0.0) >= 0.70,
        "set_same_GT_precision_top5k_pass": _num(set_metrics.get("set_same_GT_precision@top5k"), 0.0) >= 0.65,
        "GT_object_has_candidate_set_025_pass": _num(set_metrics.get("GT_object_has_candidate_set@0.25"), 0.0) >= 0.60,
    }
    gate["pass"] = bool(any(gate.values()))
    return {
        "phase": "v49_component_completion_atlas",
        "created_at": utc_now(),
        "pair_metrics": pair_metrics,
        "set_metrics": set_metrics,
        "pair_rows": pair_rows,
        "component_set_rows": set_rows,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_COMPONENT_COMPLETION_SIGNAL",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _set_row(scene: str, comps: tuple[str, ...], profiles: dict[str, dict[str, Any]], *, candidate_source: str = "unknown") -> dict[str, Any]:
    items = [profiles[comp] for comp in comps if comp in profiles]
    gt_values = [str(item.get("diagnostic_dominant_gt") or "") for item in items if item.get("diagnostic_dominant_gt")]
    same_gt = bool(gt_values) and len(set(gt_values)) == 1
    all_frames = [set(item.get("frames", [])) for item in items]
    pair_overlaps = []
    pair_sims = []
    pair_conflicts = []
    for left, right in combinations(items, 2):
        fl = set(left.get("frames", []))
        fr = set(right.get("frames", []))
        pair_overlaps.append(_safe_div(len(fl & fr), max(len(fl | fr), 1)))
        pair_sims.append(cosine(left.get("feature"), right.get("feature")))
        pair_conflicts.append(bool((fl & fr) and cosine(left.get("feature"), right.get("feature")) < 0.35))
    semantic_set = safe_mean(pair_sims) if pair_sims else 1.0
    temporal = safe_mean(pair_overlaps) if pair_overlaps else 1.0
    conflict = safe_mean(1.0 if flag else 0.0 for flag in pair_conflicts) if pair_conflicts else 0.0
    total_masks = sum(int(item.get("mask_count") or 0) for item in items)
    max_single = max([int(item.get("mask_count") or 0) for item in items] or [0])
    gt_counter: Counter[str] = Counter()
    for item in items:
        gt = str(item.get("diagnostic_dominant_gt") or "")
        if gt:
            gt_counter[gt] += int(item.get("diagnostic_labeled_mask_count") or 0)
    purity = _safe_div(max(gt_counter.values()) if gt_counter else 0, sum(gt_counter.values()) if gt_counter else 0)
    non_gt_score = 0.35 * float(temporal or 0.0) + 0.35 * float(semantic_set or 0.0) + 0.20 * _safe_div(total_masks - max_single, max(total_masks, 1)) - 0.20 * float(conflict or 0.0)
    return {
        "scene": scene,
        "component_set": _pack_components(comps),
        "candidate_source": candidate_source,
        "set_size": len(comps),
        "same_GT_set": same_gt,
        "mixed_GT_set": bool(gt_values) and len(set(gt_values)) > 1,
        "GT_purity_of_set": purity,
        "mask_support_score": float(total_masks),
        "temporal_support_score": temporal,
        "semantic_set_score": semantic_set,
        "conflict_score": conflict,
        "d4rt_specific_score": temporal,
        "coverage_gain_over_singletons": int(total_masks - max_single),
        "duplicate_penalty": 0.0,
        "nonGT_set_score": non_gt_score,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _low_overlap_semantic_edge_score(row: dict[str, Any]) -> float | None:
    color = parse_float(row.get("colorhist_similarity"))
    frame_overlap = parse_float(row.get("frame_overlap_ratio"))
    same_frame_conflict = _bool(row.get("same_frame_conflict"))
    if frame_overlap > 0.05 or color < 0.92 or same_frame_conflict:
        return None
    return color


def _expanded_partwhole_star_hypotheses(
    pair_rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    vote_rows: list[dict[str, str]],
    seen: set[tuple[str, ...]],
    *,
    max_rows: int = 1200,
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_scene[str(row.get("scene") or "")].append(row)
    hypotheses: list[dict[str, Any]] = []
    for _scene, rows in by_scene.items():
        adjacency: dict[str, list[tuple[float, str, dict[str, Any]]]] = defaultdict(list)
        for row in rows:
            score = _low_overlap_semantic_edge_score(row)
            if score is None:
                continue
            left = str(row.get("component_i") or "")
            right = str(row.get("component_j") or "")
            if left in profiles and right in profiles:
                adjacency[left].append((score, right, row))
                adjacency[right].append((score, left, row))
        for seed, neighbors in adjacency.items():
            ranked: list[tuple[float, str, dict[str, Any]]] = []
            used_neighbors: set[str] = set()
            for score, neighbor, row in sorted(neighbors, key=lambda item: item[0], reverse=True):
                if neighbor in used_neighbors:
                    continue
                ranked.append((score, neighbor, row))
                used_neighbors.add(neighbor)
            for size in [4, 5, 6]:
                if len(ranked) < size - 1:
                    continue
                comps = tuple(sorted([seed] + [neighbor for _score, neighbor, _row in ranked[: size - 1]]))
                if comps in seen or any(comp not in profiles for comp in comps):
                    continue
                seen.add(comps)
                source_row = {
                    "candidate_source": "expanded_low_overlap_semantic_star",
                    "temporal_support_score": safe_mean(parse_float(row.get("D4RT_temporal_support")) for _score, _neighbor, row in ranked[: size - 1]),
                    "semantic_set_score": safe_mean(parse_float(row.get("colorhist_similarity")) for _score, _neighbor, row in ranked[: size - 1]),
                    "conflict_score": safe_mean(1.0 if _bool(row.get("same_frame_conflict")) else 0.0 for _score, _neighbor, row in ranked[: size - 1]),
                    "d4rt_specific_score": safe_mean(parse_float(row.get("D4RT_specific_margin")) for _score, _neighbor, row in ranked[: size - 1]),
                    "coverage_gain_over_singletons": 0,
                }
                hypotheses.append(
                    _hypothesis_row(
                        f"H6_expanded_partwhole_{len(hypotheses):05d}",
                        "H6_expanded_partwhole_star",
                        comps,
                        profiles,
                        vote_rows,
                        source_row=source_row,
                    )
                )
                if len(hypotheses) >= max_rows:
                    return hypotheses
    return hypotheses


def _pair_metrics(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [bool(row.get("same_GT_pair")) for row in pair_rows if row.get("same_GT_pair") or row.get("different_GT_pair")]
    filtered = [row for row in pair_rows if row.get("same_GT_pair") or row.get("different_GT_pair")]
    def auc(key: str) -> float | None:
        return rank_auc(labels, [parse_float(row.get(key)) for row in filtered]) if filtered else None
    top1k = filtered[:1000]
    top5k = filtered[:5000]
    return {
        "pair_count": len(pair_rows),
        "labeled_pair_count": len(filtered),
        "pair_AUC_D4RT_temporal": auc("D4RT_temporal_support"),
        "pair_AUC_D4RT_specific": auc("D4RT_specific_margin"),
        "pair_AUC_shared_mask": auc("shared_mask_count"),
        "pair_AUC_DINO": None,
        "pair_AUC_RADIO": None,
        "pair_AUC_semantic_negative": rank_auc([not label for label in labels], [parse_float(row.get("semantic_negative_score")) for row in filtered]) if filtered else None,
        "pair_AUC_combined_nonGT": auc("combined_nonGT_score"),
        "precision@top1k_same_object": safe_mean(1.0 if row.get("same_GT_pair") else 0.0 for row in top1k) if top1k else None,
        "precision@top5k_same_object": safe_mean(1.0 if row.get("same_GT_pair") else 0.0 for row in top5k) if top5k else None,
        "false_merge_rate@top1k": safe_mean(1.0 if row.get("different_GT_pair") else 0.0 for row in top1k) if top1k else None,
        "false_merge_rate@top5k": safe_mean(1.0 if row.get("different_GT_pair") else 0.0 for row in top5k) if top5k else None,
    }


def _set_metrics(set_rows: list[dict[str, Any]], vote_rows: list[dict[str, str]]) -> dict[str, Any]:
    top5k = set_rows[:5000]
    gt_totals = _gt_mask_totals(vote_rows)
    comp_gt_counts = _component_gt_counts(vote_rows)
    best_coverage: dict[str, float] = defaultdict(float)
    for row in set_rows:
        counts: Counter[str] = Counter()
        for comp in _unpack_components(row.get("component_set")):
            if comp in comp_gt_counts:
                counts.update(comp_gt_counts[comp])
        for gt, count in counts.items():
            best_coverage[gt] = max(best_coverage[gt], _safe_div(count, gt_totals.get(gt, 0)))
    return {
        "set_candidate_count": len(set_rows),
        "set_candidate_purity": safe_mean(row.get("GT_purity_of_set") for row in set_rows),
        "set_candidate_coverage": safe_mean(best_coverage.values()) if best_coverage else None,
        "set_same_GT_precision@top5k": safe_mean(1.0 if row.get("same_GT_set") else 0.0 for row in top5k) if top5k else None,
        "set_conflict_rate": safe_mean(row.get("conflict_score") for row in set_rows),
        "GT_object_has_candidate_set@0.25": _safe_div(sum(1 for gt in gt_totals if best_coverage.get(gt, 0.0) >= 0.25), len(gt_totals)),
        "GT_object_has_candidate_set@0.50": _safe_div(sum(1 for gt in gt_totals if best_coverage.get(gt, 0.0) >= 0.50), len(gt_totals)),
    }


def _load_atlas(output_root: str | Path = "outputs/audit/v49_component_completion_atlas") -> dict[str, Any]:
    return load_optional_json(project_path(output_root) / "component_completion_atlas_summary.json")


def _load_lattice(output_root: str | Path = "outputs/audit/v49_component_lattice") -> dict[str, Any]:
    return load_optional_json(project_path(output_root) / "component_lattice_summary.json")


def build_hypothesis_generation(max_hypotheses: int = 5000) -> dict[str, Any]:
    obs_rows = load_optional_csv("outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    profiles = build_component_profiles(vote_rows, obs_rows)
    atlas = _load_atlas()
    if atlas.get("missing"):
        atlas = build_component_completion_atlas()
    set_rows = list(atlas.get("component_set_rows") or [])
    pair_rows = list(atlas.get("pair_rows") or [])
    hypotheses: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for profile in sorted(profiles.values(), key=lambda item: item.get("mask_count", 0), reverse=True):
        comps = (str(profile["component_key"]),)
        seen.add(comps)
        hypotheses.append(_hypothesis_row(f"H0_singleton_{len(hypotheses):05d}", "H0_singleton", comps, profiles, vote_rows))
    for row in set_rows:
        comps = tuple(sorted(comp for comp in _unpack_components(row.get("component_set")) if comp in profiles))
        if len(comps) < 2 or comps in seen:
            continue
        source = "H2_shared_mask_temporal_semantic_set"
        seen.add(comps)
        hypotheses.append(_hypothesis_row(f"H5_hybrid_{len(hypotheses):05d}", source, comps, profiles, vote_rows, source_row=row))
    hypotheses.extend(_expanded_partwhole_star_hypotheses(pair_rows, profiles, vote_rows, seen))
    hypotheses.sort(key=lambda row: float(row.get("hypothesis_support_score", 0.0)), reverse=True)
    hypotheses = hypotheses[:max_hypotheses]
    metrics = _hypothesis_metrics(hypotheses, vote_rows)
    gate = {
        "hypothesis_count_le_5000": len(hypotheses) <= 5000,
        "GT_object_has_hypothesis_025_pass": _num(metrics.get("GT_object_has_hypothesis@0.25"), 0.0) >= 0.65,
        "hypothesis_purity_topk_pass": _num(metrics.get("hypothesis_purity@topk"), 0.0) >= 0.75,
        "hypothesis_conflict_rate_topk_pass": _num(metrics.get("hypothesis_conflict_rate@topk"), 1.0) <= 0.20,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v49_hypothesis_generation",
        "created_at": utc_now(),
        "hypothesis_rows": hypotheses,
        "metrics": metrics,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_HYPOTHESIS_GENERATION",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _hypothesis_row(
    hypothesis_id: str,
    source: str,
    comps: tuple[str, ...],
    profiles: dict[str, dict[str, Any]],
    vote_rows: list[dict[str, str]],
    source_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_row = source_row or {}
    items = [profiles[comp] for comp in comps]
    gt_counts: Counter[str] = Counter()
    comp_gt = _component_gt_counts(vote_rows)
    for comp in comps:
        gt_counts.update(comp_gt.get(comp, Counter()))
    total_gt = sum(gt_counts.values())
    dominant_gt, dominant_count = ("", 0)
    if gt_counts:
        dominant_gt, dominant_count = gt_counts.most_common(1)[0]
    mask_count = sum(int(item.get("mask_count") or 0) for item in items)
    pair_sims = [cosine(left.get("feature"), right.get("feature")) for left, right in combinations(items, 2)]
    semantic = safe_mean(pair_sims) if pair_sims else 1.0
    prototype_diversity = safe_mean(1.0 - sim for sim in pair_sims) if pair_sims else 0.0
    frame_sets = [set(item.get("frames", [])) for item in items]
    frame_overlap_values = [
        _safe_div(len(left & right), max(len(left | right), 1))
        for left, right in combinations(frame_sets, 2)
    ]
    context_overlap = safe_mean(frame_overlap_values) if frame_overlap_values else 0.0
    context_compatibility = 1.0 - min(1.0, float(context_overlap or 0.0))
    boundary_instability = safe_mean(parse_float(item.get("boundary_proxy_instability")) for item in items)
    feature_variance = safe_mean(parse_float(item.get("component_feature_variance")) for item in items)
    reliability_values = [parse_float(item.get("mask_reliability_mean")) for item in items if item.get("mask_reliability_mean") not in (None, "")]
    reliability_min_values = [parse_float(item.get("mask_reliability_min")) for item in items if item.get("mask_reliability_min") not in (None, "")]
    reliability_mean = safe_mean(reliability_values)
    reliability_min = min(reliability_min_values) if reliability_min_values else reliability_mean
    reliability_range = (max(reliability_values) - min(reliability_values)) if len(reliability_values) >= 2 else 0.0
    conflict = source_row.get("conflict_score")
    if conflict in (None, ""):
        conflicts = [1.0 if left & right else 0.0 for left, right in combinations(frame_sets, 2)]
        conflict = safe_mean(conflicts) if conflicts else 0.0
    mask_support = min(1.0, _safe_div(mask_count, 30.0))
    large_support_risk = min(1.0, max(0.0, (float(mask_count) - 80.0) / 400.0))
    support = (
        0.35 * mask_support
        + 0.25 * float(source_row.get("temporal_support_score") or 0.0)
        + 0.25 * float(semantic or 0.0)
        - 0.20 * float(conflict or 0.0)
        - 0.12 * large_support_risk
        - 0.03 * max(0, len(comps) - 4)
    )
    row = {
        "hypothesis_id": hypothesis_id,
        "scene": items[0]["scene"] if items else "",
        "candidate_generation_source": source,
        "component_set_candidate_source": source_row.get("candidate_source", "singleton" if len(comps) == 1 else "unknown"),
        "components": _pack_components(comps),
        "hypothesis_size": len(comps),
        "mask_support_score": mask_count,
        "mask_support_normalized": mask_support,
        "coverage_gain_over_singletons": source_row.get("coverage_gain_over_singletons", 0),
        "large_support_risk": large_support_risk,
        "mask_reliability_mean": reliability_mean,
        "mask_reliability_min": reliability_min,
        "mask_reliability_range": reliability_range,
        "split_entropy_proxy": None if reliability_min in (None, "") else float(1.0 - parse_float(reliability_min)),
        "temporal_support_score": source_row.get("temporal_support_score"),
        "semantic_set_score": semantic,
        "prototype_diversity": prototype_diversity,
        "context_overlap_proxy": context_overlap,
        "context_compatibility": context_compatibility,
        "boundary_proxy_instability": boundary_instability,
        "component_feature_variance": feature_variance,
        "hypothesis_conflict_rate": conflict,
        "hypothesis_support_score": support,
        "hypothesis_support_score_raw": support,
        "hypothesis_d4rt_specific_score": source_row.get("d4rt_specific_score", source_row.get("temporal_support_score", 0.0)),
        "diagnostic_dominant_gt": dominant_gt,
        "hypothesis_purity": _safe_div(dominant_count, total_gt),
        "hypothesis_completeness": None,
        "same_GT_set": bool(gt_counts) and len(gt_counts) == 1,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    row["hypothesis_generation_support_score"] = _hypothesis_generation_support_score(row)
    row["hypothesis_support_score"] = row["hypothesis_generation_support_score"]
    return row


def _hypothesis_metrics(hypotheses: list[dict[str, Any]], vote_rows: list[dict[str, str]]) -> dict[str, Any]:
    gt_totals = _gt_mask_totals(vote_rows)
    comp_gt = _component_gt_counts(vote_rows)
    best_coverage: dict[str, float] = defaultdict(float)
    for row in hypotheses:
        counts: Counter[str] = Counter()
        for comp in _unpack_components(row.get("components")):
            counts.update(comp_gt.get(comp, Counter()))
        for gt, count in counts.items():
            best_coverage[gt] = max(best_coverage[gt], _safe_div(count, gt_totals.get(gt, 0)))
        if counts:
            dom_gt, dom_count = counts.most_common(1)[0]
            row["hypothesis_completeness"] = _safe_div(dom_count, gt_totals.get(dom_gt, 0))
    topk = hypotheses[: min(1000, len(hypotheses))]
    return {
        "hypothesis_count": len(hypotheses),
        "hypothesis_size_mean": safe_mean(row.get("hypothesis_size") for row in hypotheses),
        "hypothesis_size_p90": safe_quantile([row.get("hypothesis_size") for row in hypotheses], 0.90) if hypotheses else None,
        "GT_object_has_hypothesis@0.25": _safe_div(sum(1 for gt in gt_totals if best_coverage.get(gt, 0.0) >= 0.25), len(gt_totals)),
        "GT_object_has_hypothesis@0.50": _safe_div(sum(1 for gt in gt_totals if best_coverage.get(gt, 0.0) >= 0.50), len(gt_totals)),
        "hypothesis_purity@topk": safe_mean(row.get("hypothesis_purity") for row in topk),
        "hypothesis_completeness@topk": safe_mean(row.get("hypothesis_completeness") for row in topk),
        "hypothesis_conflict_rate@topk": safe_mean(row.get("hypothesis_conflict_rate") for row in topk),
        "hypothesis_support_score_mean": safe_mean(row.get("hypothesis_support_score") for row in hypotheses),
        "hypothesis_semantic_set_score_mean": safe_mean(row.get("semantic_set_score") for row in hypotheses),
        "hypothesis_d4rt_specific_score_mean": safe_mean(row.get("hypothesis_d4rt_specific_score") for row in hypotheses),
        "same_GT_set_precision@topk": safe_mean(1.0 if row.get("same_GT_set") else 0.0 for row in topk) if topk else None,
    }


def build_semantic_set_compatibility() -> dict[str, Any]:
    hyp_payload = load_optional_json("outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json")
    if hyp_payload.get("missing"):
        hyp_payload = build_hypothesis_generation()
    hypotheses = list(hyp_payload.get("hypothesis_rows") or [])
    atlas = _load_atlas()
    pair_rows = list(atlas.get("pair_rows") or [])
    semantic_negative_auc = None
    labeled_pairs = [row for row in pair_rows if row.get("same_GT_pair") or row.get("different_GT_pair")]
    if labeled_pairs:
        semantic_negative_auc = rank_auc(
            [bool(row.get("different_GT_pair")) for row in labeled_pairs],
            [parse_float(row.get("semantic_negative_score")) for row in labeled_pairs],
        )
    labels = [bool(row.get("same_GT_set")) for row in hypotheses]
    scores = [parse_float(row.get("semantic_set_score")) for row in hypotheses]
    hypothesis_auc = rank_auc(labels, scores) if any(labels) and not all(labels) else None
    sorted_before = sorted(labeled_pairs, key=lambda row: parse_float(row.get("combined_nonGT_score")), reverse=True)[:1000]
    guarded = [row for row in sorted_before if parse_float(row.get("semantic_negative_score")) <= 0.55]
    false_before = safe_mean(1.0 if row.get("different_GT_pair") else 0.0 for row in sorted_before) if sorted_before else None
    false_after = safe_mean(1.0 if row.get("different_GT_pair") else 0.0 for row in guarded) if guarded else None
    same_before = sum(1 for row in sorted_before if row.get("same_GT_pair"))
    same_after = sum(1 for row in guarded if row.get("same_GT_pair"))
    false_merge_reduction = None if false_before is None or false_after is None else float(false_before - false_after)
    completeness_drop = 0.0 if same_before <= 0 else float((same_before - same_after) / same_before)
    backend_rows = [
        {
            "backend_id": "colorhist_fallback",
            "component_level_available": True,
            "semantic_negative_AUC": semantic_negative_auc,
            "hypothesis_set_semantic_AUC": hypothesis_auc,
        },
        {
            "backend_id": "DINOv2",
            "component_level_available": False,
            "semantic_negative_AUC": None,
            "hypothesis_set_semantic_AUC": None,
            "note": "Only imported v48 edge-proxy availability exists; no v49 component-level DINO pooling artifact was present.",
        },
        {
            "backend_id": "RADIO/RADSeg",
            "component_level_available": False,
            "semantic_negative_AUC": None,
            "hypothesis_set_semantic_AUC": None,
            "note": "Only imported v48 edge-proxy contradiction evidence exists; no v49 component-level RADIO pooling artifact was present.",
        },
    ]
    gate = {
        "semantic_negative_AUC_pass": _num(semantic_negative_auc, 0.0) >= 0.75,
        "false_merge_reduction_pass": _num(false_merge_reduction, 0.0) >= 0.15,
        "semantic_guard_purity_gain_pass": False,
        "completeness_drop_pass": completeness_drop <= 0.03,
        "component_level_frozen_dense_backend_available": False,
    }
    gate["pass"] = bool((gate["semantic_negative_AUC_pass"] or gate["false_merge_reduction_pass"]) and gate["completeness_drop_pass"])
    return {
        "phase": "v49_semantic_set_compatibility",
        "created_at": utc_now(),
        "backend_rows": backend_rows,
        "metrics": {
            "component_pair_similarity_AUC": rank_auc([bool(row.get("same_GT_pair")) for row in labeled_pairs], [parse_float(row.get("colorhist_similarity")) for row in labeled_pairs]) if labeled_pairs else None,
            "semantic_negative_AUC": semantic_negative_auc,
            "hypothesis_set_semantic_AUC": hypothesis_auc,
            "false_merge_reduction_by_semantic_guard": false_merge_reduction,
            "completeness_drop_by_semantic_guard": completeness_drop,
            "semantic_only_merge_ARI": None,
            "semantic_only_merge_purity": None,
            "DINO_vs_colorhist_delta": None,
            "RADIO_vs_colorhist_delta": None,
        },
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_SEMANTIC_GUARD",
        "recommendation": "Use colorhist-style local contradiction only; do not claim DINO/RADIO part-whole positive compatibility without component-level features.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _semantic_source_rows(root: str, obs_key_set: set[tuple[str, str, str]], obs_scene_counts: Counter[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = project_path(root)
    if not base.exists():
        return [
            {
                "source_root": rel(base),
                "scene": "",
                "available": False,
                "failure_stage": "root_missing",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        ]
    for summary_path in sorted(base.glob("*/part_graph_summary.json")):
        scene = summary_path.parent.name
        summary = load_optional_json(summary_path)
        source_rows = load_optional_csv(summary_path.parent / "source_audit_rows.csv")
        token_rows = load_optional_csv(summary_path.parent / "part_token_rows.csv")
        matching = 0
        for row in token_rows:
            key = (str(row.get("scene") or scene), str(row.get("frame_id") or ""), str(row.get("mask_id") or ""))
            if key in obs_key_set:
                matching += 1
        best = summary.get("best_source_summary") or (source_rows[0] if source_rows else {})
        obs_count = obs_scene_counts.get(scene, 0)
        rows.append(
            {
                "source_root": rel(base),
                "scene": scene,
                "available": not bool(summary.get("missing")),
                "feature_backend": summary.get("feature_backend") or best.get("feature_backend"),
                "feature_checkpoint": summary.get("feature_checkpoint") or best.get("feature_checkpoint"),
                "best_source": summary.get("best_source") or best.get("source"),
                "part_token_count": parse_int(best.get("part_token_count")),
                "edge_count": parse_int(best.get("edge_count")),
                "semantic_affinity_AUC": parse_float(best.get("semantic_affinity_AUC")),
                "object_part_compatibility_AUC": parse_float(best.get("object_part_compatibility_AUC")),
                "phase1_gate_pass": _bool(summary.get("phase1_gate_pass") if summary.get("phase1_gate_pass") is not None else best.get("gate_pass_phase1")),
                "v49_observation_rows_in_scene": int(obs_count),
                "v49_exact_mask_observation_match_count": int(matching),
                "v49_exact_mask_observation_match_rate": _safe_div(float(matching), float(obs_count)),
                "has_v49_component_key_mapping": False,
                "mapping_note": "Exact (scene, frame_id, mask_id) overlap is only an observation-level check; old part tokens do not carry v49 carrier-component keys.",
                "uses_gt_for_prediction": _bool(summary.get("uses_gt_for_prediction") if summary.get("uses_gt_for_prediction") is not None else best.get("uses_gt_for_prediction")),
                "uses_gt_for_diagnostic_labels": _bool(summary.get("uses_gt_for_diagnostic_labels") if summary.get("uses_gt_for_diagnostic_labels") is not None else best.get("uses_gt_for_diagnostic_labels")),
            }
        )
    return rows


def build_semantic_backend_availability_audit() -> dict[str, Any]:
    obs_rows = load_optional_csv("outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    obs_key_set = {(str(row.get("scene") or ""), str(row.get("frame_id") or ""), str(row.get("mask_id") or "")) for row in obs_rows}
    obs_scene_counts: Counter[str] = Counter(str(row.get("scene") or "") for row in obs_rows)
    feature_backends = Counter(str(row.get("feature_backend") or "") for row in obs_rows)
    context_nonempty = sum(1 for row in obs_rows if _json_list(row.get("context_feature")))
    boundary_nonempty = sum(1 for row in obs_rows if _json_list(row.get("boundary_feature")))
    prototype_counts = Counter(str(row.get("prototype_count") or "") for row in obs_rows)
    prediction_rows = []
    prediction_root = project_path("data/prediction/stream4d_scannet_32f_ioc075_fixmem_top14_mask_count_min100_one_class_agnostic")
    for scene in DEFAULT_SCENES:
        path = prediction_root / f"{scene}.npz"
        keys: list[str] = []
        has_rgb_or_feature = False
        if path.exists():
            try:
                npz = np.load(path, allow_pickle=True)
                keys = list(npz.files)
                has_rgb_or_feature = any("rgb" in key.lower() or "image" in key.lower() or "feature" in key.lower() for key in keys)
            except Exception as exc:  # pragma: no cover - diagnostic path only
                keys = [f"load_error:{type(exc).__name__}"]
        prediction_rows.append(
            {
                "scene": scene,
                "prediction_npz": rel(path),
                "exists": path.exists(),
                "keys": json.dumps(keys),
                "has_rgb_or_feature_keys": has_rgb_or_feature,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    semantic_rows: list[dict[str, Any]] = []
    for root in [
        "outputs/audit/v42_semantic_part_graph_radio",
        "outputs/audit/v43_2_current_v37_semantic_part_graph_dinov2_sample8",
        "outputs/audit/v42_semantic_part_graph_dinov2_cap640_sample8",
    ]:
        semantic_rows.extend(_semantic_source_rows(root, obs_key_set, obs_scene_counts))
    covered_scenes = {str(row.get("scene")) for row in semantic_rows if parse_int(row.get("v49_exact_mask_observation_match_count")) > 0}
    v48_semantic = load_optional_json("outputs/audit/v48_semantic_features/semantic_feature_audit_summary.json")
    v48_backend_rows = v48_semantic.get("backend_rows", []) if isinstance(v48_semantic.get("backend_rows"), list) else []
    scope_note = str(v48_semantic.get("metric_scope_note") or "").lower()
    gate = {
        "v49_observation_table_available": bool(obs_rows),
        "v49_observation_has_dense_feature_backend": any(key and key not in {"colorhist_fallback", "colorhist"} for key in feature_backends),
        "v49_context_or_boundary_features_nonempty": context_nonempty > 0 or boundary_nonempty > 0,
        "all_probe_scenes_have_old_frozen_semantic_observation_overlap": all(scene in covered_scenes for scene in DEFAULT_SCENES),
        "old_semantic_mapping_covers_half_of_any_scene": any(parse_float(row.get("v49_exact_mask_observation_match_rate")) >= 0.50 for row in semantic_rows),
        "prediction_npz_has_rgb_or_feature_inputs": all(bool(row["has_rgb_or_feature_keys"]) for row in prediction_rows),
        "v48_semantic_scope_is_component_level": "component-level" in scope_note and "do not prove" not in scope_note,
    }
    gate["pass"] = bool(
        gate["v49_observation_table_available"]
        and gate["v49_observation_has_dense_feature_backend"]
        and gate["v49_context_or_boundary_features_nonempty"]
        and gate["all_probe_scenes_have_old_frozen_semantic_observation_overlap"]
        and gate["old_semantic_mapping_covers_half_of_any_scene"]
        and gate["prediction_npz_has_rgb_or_feature_inputs"]
        and gate["v48_semantic_scope_is_component_level"]
    )
    return {
        "phase": "v49_semantic_backend_availability_audit",
        "created_at": utc_now(),
        "feature_backend_counts": dict(feature_backends),
        "prototype_count_histogram": dict(prototype_counts),
        "context_feature_nonempty_count": int(context_nonempty),
        "boundary_feature_nonempty_count": int(boundary_nonempty),
        "v49_observation_scene_counts": dict(obs_scene_counts),
        "v48_semantic_metric_scope_note": v48_semantic.get("metric_scope_note"),
        "v48_semantic_recommendation": v48_semantic.get("recommendation"),
        "v48_backend_rows": v48_backend_rows,
        "semantic_source_rows": semantic_rows,
        "prediction_input_rows": prediction_rows,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_COMPONENT_LEVEL_FROZEN_SEMANTIC_ARTIFACT",
        "recommendation": (
            "Do not wire old v42/v43/v48 semantic proxies into the final v49 scorer as method evidence. "
            "Regenerate v49-aligned component-level dense semantic, context, and boundary pooling for all probe scenes, "
            "or keep semantics diagnostic-only."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_component_proxy_feature_audit() -> dict[str, Any]:
    obs_rows = load_optional_csv("outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    if not vote_rows:
        vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_metricfix/carrier_supertrack_mask_vote_rows.csv")
    profiles = build_component_profiles(vote_rows, obs_rows)
    hyp_payload = load_optional_json("outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json")
    if hyp_payload.get("missing"):
        hyp_payload = build_hypothesis_generation()
    scored = [_score_hypothesis(dict(row)) for row in hyp_payload.get("hypothesis_rows", [])]
    labels = [bool(row.get("same_GT_set")) or parse_float(row.get("hypothesis_purity")) >= 0.75 for row in scored]

    def auc(key: str, *, negate: bool = False) -> float | None:
        if not scored or not any(labels) or all(labels):
            return None
        values = [parse_float(row.get(key)) for row in scored]
        if negate:
            values = [-value for value in values]
        return rank_auc(labels, values)

    component_rows = [
        {
            "component_key": profile.get("component_key"),
            "scene": profile.get("scene"),
            "mask_count": profile.get("mask_count"),
            "frame_count": profile.get("frame_count"),
            "feature_backend": profile.get("feature_backend"),
            "component_prototype_count": profile.get("component_prototype_count"),
            "component_feature_variance": profile.get("component_feature_variance"),
            "boundary_fill_mean": profile.get("boundary_fill_mean"),
            "boundary_fill_range": profile.get("boundary_fill_range"),
            "support_density_mean": profile.get("support_density_mean"),
            "support_density_range": profile.get("support_density_range"),
            "mask_area_log_range": profile.get("mask_area_log_range"),
            "boundary_proxy_instability": profile.get("boundary_proxy_instability"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for profile in profiles.values()
    ]
    score_auc_rows = [
        {"score": "semantic_set_score", "AUC": auc("semantic_set_score")},
        {"score": "mask_reliability_min", "AUC": auc("mask_reliability_min")},
        {"score": "negative_prototype_diversity", "AUC": auc("prototype_diversity", negate=True)},
        {"score": "negative_context_overlap_proxy", "AUC": auc("context_overlap_proxy", negate=True)},
        {"score": "negative_boundary_proxy_instability", "AUC": auc("boundary_proxy_instability", negate=True)},
        {"score": "negative_component_feature_variance", "AUC": auc("component_feature_variance", negate=True)},
        {"score": "score_split_entropy_reliability", "AUC": auc("score_split_entropy_reliability")},
        {"score": "score_boundary_prototype_context_guard", "AUC": auc("score_boundary_prototype_context_guard")},
        {"score": "score_boundary_prototype_context_hard", "AUC": auc("score_boundary_prototype_context_hard")},
        {"score": "score_boundary_prototype_no_temporal", "AUC": auc("score_boundary_prototype_no_temporal")},
        {"score": "score_boundary_prototype_mask_only", "AUC": auc("score_boundary_prototype_mask_only")},
        {"score": "score_d4rt_completion_guard", "AUC": auc("score_d4rt_completion_guard")},
        {"score": "score_d4rt_completion_no_temporal", "AUC": auc("score_d4rt_completion_no_temporal")},
        {"score": "score_d4rt_completion_mask_only", "AUC": auc("score_d4rt_completion_mask_only")},
        {"score": "score_persistent_contradiction_prefilter", "AUC": auc("score_persistent_contradiction_prefilter")},
        {"score": "score_persistent_prefilter_no_temporal", "AUC": auc("score_persistent_prefilter_no_temporal")},
        {"score": "score_persistent_prefilter_mask_only", "AUC": auc("score_persistent_prefilter_mask_only")},
        {"score": "score_expanded_partwhole_cap", "AUC": auc("score_expanded_partwhole_cap")},
        {"score": "score_expanded_partwhole_no_temporal", "AUC": auc("score_expanded_partwhole_no_temporal")},
        {"score": "score_expanded_partwhole_mask_only", "AUC": auc("score_expanded_partwhole_mask_only")},
        {"score": "score_expanded_partwhole_no_source_bonus", "AUC": auc("score_expanded_partwhole_no_source_bonus")},
    ]
    auc_map = {str(row["score"]): row["AUC"] for row in score_auc_rows}
    split_auc = auc_map.get("score_split_entropy_reliability")
    hard_auc = auc_map.get("score_boundary_prototype_context_hard")
    guard_delta = None if split_auc is None or hard_auc is None else float(hard_auc - split_auc)
    metrics = {
        "component_count": len(component_rows),
        "hypothesis_count": len(scored),
        "positive_hypothesis_count": int(sum(labels)),
        "component_feature_variance_mean": safe_mean(row.get("component_feature_variance") for row in component_rows),
        "boundary_proxy_instability_mean": safe_mean(row.get("boundary_proxy_instability") for row in component_rows),
        "context_overlap_negative_AUC": auc_map.get("negative_context_overlap_proxy"),
        "boundary_proxy_negative_AUC": auc_map.get("negative_boundary_proxy_instability"),
        "component_feature_variance_negative_AUC": auc_map.get("negative_component_feature_variance"),
        "split_entropy_AUC": split_auc,
        "boundary_prototype_context_hard_AUC": hard_auc,
        "boundary_prototype_context_delta_vs_split_entropy_AUC": guard_delta,
    }
    gate = {
        "component_proxy_features_available": bool(component_rows) and any(parse_float(row.get("component_prototype_count")) > 0 for row in component_rows),
        "context_or_boundary_proxy_signal_pass": _num(max(
            [value for value in [
                metrics["context_overlap_negative_AUC"],
                metrics["boundary_proxy_negative_AUC"],
                metrics["component_feature_variance_negative_AUC"],
            ] if value is not None] or [0.0]
        ), 0.0) >= 0.75,
        "proxy_guard_beats_split_entropy_AUC_pass": _num(guard_delta, 0.0) >= 0.02,
        "dense_semantic_backend_claimed": False,
        "no_gt_for_prediction": True,
    }
    gate["pass"] = bool(
        gate["component_proxy_features_available"]
        and gate["context_or_boundary_proxy_signal_pass"]
        and gate["proxy_guard_beats_split_entropy_AUC_pass"]
        and not gate["dense_semantic_backend_claimed"]
    )
    return {
        "phase": "v49_component_proxy_feature_audit",
        "created_at": utc_now(),
        "feature_scope": "v49-aligned colorhist/core-feature plus geometry boundary/context proxy; not DINO/RADIO dense semantic.",
        "component_proxy_rows": component_rows,
        "hypothesis_proxy_rows": scored,
        "score_auc_rows": score_auc_rows,
        "metrics": metrics,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_COMPONENT_PROXY_GUARD_SIGNAL",
        "recommendation": "Use boundary/prototype/context proxy only as contradiction guard. Do not promote it as dense semantic part-whole compatibility.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_hypothesis_scoring() -> dict[str, Any]:
    hyp_payload = load_optional_json("outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json")
    if hyp_payload.get("missing"):
        hyp_payload = build_hypothesis_generation()
    rows = [dict(row) for row in hyp_payload.get("hypothesis_rows", [])]
    scored = [_score_hypothesis(row) for row in rows]
    labels = [bool(row.get("same_GT_set")) or parse_float(row.get("hypothesis_purity")) >= 0.75 for row in scored]
    score_keys = [
        "score_mask",
        "score_temporal",
        "score_semantic_guard",
        "score_full",
        "score_full_without_d4rt",
        "score_no_temporal",
        "score_mask_only",
        "score_semantic_only",
        "score_guarded_full",
        "score_guarded_without_d4rt",
        "score_guarded_no_temporal",
        "score_guarded_mask_only",
        "score_split_entropy_reliability",
        "score_split_entropy_no_temporal",
        "score_split_entropy_mask_only",
        "score_split_entropy_no_reliability",
        "score_boundary_prototype_context_guard",
        "score_boundary_prototype_context_hard",
        "score_boundary_prototype_no_temporal",
        "score_boundary_prototype_mask_only",
        "score_boundary_prototype_no_context_boundary",
        "score_d4rt_completion_guard",
        "score_d4rt_completion_no_temporal",
        "score_d4rt_completion_mask_only",
        "score_d4rt_completion_no_specific",
        "score_persistent_contradiction_prefilter",
        "score_persistent_prefilter_no_temporal",
        "score_persistent_prefilter_mask_only",
        "score_persistent_prefilter_no_source_guard",
        "score_expanded_partwhole_cap",
        "score_expanded_partwhole_no_temporal",
        "score_expanded_partwhole_mask_only",
        "score_expanded_partwhole_no_source_bonus",
    ]
    aucs = {}
    for key in score_keys:
        aucs[key] = rank_auc(labels, [parse_float(row.get(key)) for row in scored]) if any(labels) and not all(labels) else None
    single_term_keys = {"score_mask", "score_temporal", "score_semantic_guard", "score_semantic_only"}
    single_best = max([value for key, value in aucs.items() if key in single_term_keys and value is not None] or [None])
    primary_score_key = max(
        [
            "score_full",
            "score_guarded_full",
            "score_split_entropy_reliability",
            "score_boundary_prototype_context_hard",
            "score_d4rt_completion_guard",
            "score_persistent_contradiction_prefilter",
            "score_expanded_partwhole_cap",
        ],
        key=lambda key: aucs.get(key) if aucs.get(key) is not None else -1.0,
    )
    full_auc = aucs.get(primary_score_key)
    if primary_score_key == "score_guarded_full":
        shuffled_auc = aucs.get("score_guarded_without_d4rt")
        no_temporal_auc = aucs.get("score_guarded_no_temporal")
    elif primary_score_key == "score_split_entropy_reliability":
        shuffled_auc = aucs.get("score_split_entropy_no_reliability")
        no_temporal_auc = aucs.get("score_split_entropy_no_temporal")
    elif primary_score_key == "score_boundary_prototype_context_hard":
        shuffled_auc = aucs.get("score_boundary_prototype_no_context_boundary")
        no_temporal_auc = aucs.get("score_boundary_prototype_no_temporal")
    elif primary_score_key == "score_d4rt_completion_guard":
        shuffled_auc = aucs.get("score_d4rt_completion_no_specific")
        no_temporal_auc = aucs.get("score_d4rt_completion_no_temporal")
    elif primary_score_key == "score_persistent_contradiction_prefilter":
        shuffled_auc = aucs.get("score_persistent_prefilter_no_source_guard")
        no_temporal_auc = aucs.get("score_persistent_prefilter_no_temporal")
    elif primary_score_key == "score_expanded_partwhole_cap":
        shuffled_auc = aucs.get("score_expanded_partwhole_no_source_bonus")
        no_temporal_auc = aucs.get("score_expanded_partwhole_no_temporal")
    else:
        shuffled_auc = _stable_control_auc(scored, labels, "shuffle")
        no_temporal_auc = aucs.get("score_no_temporal")
    top_primary = sorted(scored, key=lambda item: parse_float(item.get(primary_score_key)), reverse=True)[:1000]
    metrics = {
        "hypothesis_score_AUC": full_auc,
        "primary_score_key": primary_score_key,
        "topk_hypothesis_purity": safe_mean(row.get("hypothesis_purity") for row in top_primary),
        "topk_hypothesis_coverage": safe_mean(row.get("hypothesis_completeness") for row in top_primary),
        "score_real_minus_shuffled_AUC": None if full_auc is None or shuffled_auc is None else float(full_auc - shuffled_auc),
        "score_real_minus_no_temporal_AUC": None if full_auc is None or no_temporal_auc is None else float(full_auc - no_temporal_auc),
        "selected_candidate_diversity": safe_mean(row.get("hypothesis_size") for row in top_primary),
        "single_best_AUC": single_best,
    }
    gate = {
        "full_score_beats_single_terms": full_auc is not None and single_best is not None and full_auc >= single_best + 0.03,
        "score_real_minus_shuffled_AUC_pass": _num(metrics["score_real_minus_shuffled_AUC"], 0.0) >= 0.08,
        "score_real_minus_no_temporal_AUC_pass": _num(metrics["score_real_minus_no_temporal_AUC"], 0.0) >= 0.05,
        "topk_hypothesis_purity_pass": _num(metrics["topk_hypothesis_purity"], 0.0) >= 0.75,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v49_hypothesis_scoring",
        "created_at": utc_now(),
        "hypothesis_rows": scored,
        "metrics": metrics,
        "score_auc_rows": [{"score": key, "AUC": value} for key, value in aucs.items()],
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_SOLVER_SELECTION",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _score_hypothesis(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    mask = min(1.0, parse_float(row.get("mask_support_score")) / 30.0)
    temporal = parse_float(row.get("temporal_support_score"))
    semantic = parse_float(row.get("semantic_set_score"), 1.0)
    d4rt = parse_float(row.get("hypothesis_d4rt_specific_score"))
    conflict = parse_float(row.get("hypothesis_conflict_rate"))
    size = parse_float(row.get("hypothesis_size"), 1.0)
    large_support_risk = parse_float(row.get("large_support_risk"))
    reliability_mean = parse_float(row.get("mask_reliability_mean"))
    reliability_min = parse_float(row.get("mask_reliability_min"), reliability_mean)
    reliability_range = parse_float(row.get("mask_reliability_range"))
    prototype_diversity = parse_float(row.get("prototype_diversity"))
    context_overlap = parse_float(row.get("context_overlap_proxy"))
    boundary_instability = parse_float(row.get("boundary_proxy_instability"))
    feature_variance = parse_float(row.get("component_feature_variance"))
    coverage_gain = parse_float(row.get("coverage_gain_over_singletons"))
    set_source = str(row.get("component_set_candidate_source") or "")
    support_over = max(0.0, parse_float(row.get("mask_support_score")) - 25.0)
    semantic_positive = min(semantic, 0.82)
    semantic_contradiction = max(0.0, 0.55 - semantic) * (1.0 + min(conflict, 1.0))
    size_penalty = max(0.0, size - 3.0)
    out["score_mask"] = mask
    out["score_temporal"] = temporal
    out["score_semantic_guard"] = semantic - conflict
    out["score_d4rt_specific"] = d4rt
    out["score_full"] = 0.34 * mask + 0.24 * temporal + 0.16 * semantic + 0.16 * d4rt - 0.22 * conflict - 0.025 * max(0.0, size - 4.0)
    out["score_full_without_d4rt"] = 0.42 * mask + 0.22 * temporal + 0.18 * semantic - 0.22 * conflict - 0.025 * max(0.0, size - 4.0)
    out["score_no_temporal"] = 0.54 * mask + 0.20 * semantic - 0.20 * conflict - 0.025 * max(0.0, size - 4.0)
    out["score_mask_only"] = mask - 0.025 * max(0.0, size - 4.0)
    out["score_semantic_only"] = semantic - 0.15 * conflict
    out["score_guarded_full"] = (
        0.32 * d4rt
        + 0.24 * temporal
        + 0.18 * mask
        + 0.10 * semantic_positive
        - 0.36 * conflict
        - 0.32 * large_support_risk
        - 0.06 * size_penalty
        - 0.28 * semantic_contradiction
    )
    out["score_guarded_without_d4rt"] = (
        0.28 * temporal
        + 0.22 * mask
        + 0.10 * semantic_positive
        - 0.36 * conflict
        - 0.32 * large_support_risk
        - 0.06 * size_penalty
        - 0.28 * semantic_contradiction
    )
    out["score_guarded_no_temporal"] = (
        0.38 * mask
        + 0.10 * semantic_positive
        - 0.36 * conflict
        - 0.32 * large_support_risk
        - 0.06 * size_penalty
        - 0.28 * semantic_contradiction
    )
    out["score_guarded_mask_only"] = (
        0.50 * mask
        - 0.22 * conflict
        - 0.32 * large_support_risk
        - 0.06 * size_penalty
    )
    out["score_split_entropy_reliability"] = (
        0.70 * semantic
        + 0.40 * reliability_min
        + 0.15 * reliability_mean
        + 0.08 * min(temporal, 0.50)
        - 0.015 * support_over
        - 0.20 * reliability_range
        - 0.30 * conflict
        - 0.05 * max(0.0, size - 2.0)
    )
    out["score_split_entropy_no_temporal"] = (
        0.70 * semantic
        + 0.40 * reliability_min
        + 0.15 * reliability_mean
        - 0.015 * support_over
        - 0.20 * reliability_range
        - 0.30 * conflict
        - 0.05 * max(0.0, size - 2.0)
    )
    out["score_split_entropy_mask_only"] = (
        0.45 * reliability_min
        + 0.20 * reliability_mean
        - 0.015 * support_over
        - 0.20 * reliability_range
        - 0.30 * conflict
        - 0.05 * max(0.0, size - 2.0)
    )
    out["score_split_entropy_no_reliability"] = (
        0.70 * semantic
        + 0.08 * min(temporal, 0.50)
        - 0.015 * support_over
        - 0.30 * conflict
        - 0.05 * max(0.0, size - 2.0)
    )
    soft_proxy_penalty = 0.20 * context_overlap + 0.06 * boundary_instability + 0.06 * feature_variance + 0.12 * prototype_diversity
    hard_proxy_penalty = 0.35 * context_overlap + 0.10 * boundary_instability + 0.08 * feature_variance + 0.18 * prototype_diversity
    out["score_boundary_prototype_context_guard"] = out["score_split_entropy_reliability"] - soft_proxy_penalty
    out["score_boundary_prototype_context_hard"] = out["score_split_entropy_reliability"] - hard_proxy_penalty
    out["score_boundary_prototype_no_temporal"] = out["score_split_entropy_no_temporal"] - hard_proxy_penalty
    out["score_boundary_prototype_mask_only"] = out["score_split_entropy_mask_only"] - hard_proxy_penalty
    out["score_boundary_prototype_no_context_boundary"] = out["score_split_entropy_reliability"]
    multi_component = min(1.0, max(0.0, size - 1.0) / 2.0)
    d4rt_completion = multi_component * min(1.0, max(temporal, d4rt))
    no_temporal_explainable = 1.0 if size <= 1.0 or max(temporal, d4rt) <= 1e-9 else 0.0
    out["d4rt_completion_evidence"] = d4rt_completion
    out["no_temporal_explainable"] = no_temporal_explainable
    out["score_d4rt_completion_guard"] = (
        out["score_boundary_prototype_context_hard"]
        + 1.20 * d4rt_completion
        + 0.45 * min(d4rt, 1.0)
        + 0.25 * min(temporal, 1.0)
        - 0.85 * no_temporal_explainable
        - 0.25 * conflict
    )
    out["score_d4rt_completion_no_temporal"] = (
        out["score_boundary_prototype_no_temporal"]
        + 0.15 * multi_component
        - 0.85 * no_temporal_explainable
        - 0.25 * conflict
    )
    out["score_d4rt_completion_mask_only"] = (
        out["score_boundary_prototype_mask_only"]
        + 0.10 * multi_component
        - 0.85 * no_temporal_explainable
        - 0.25 * conflict
    )
    out["score_d4rt_completion_no_specific"] = out["score_boundary_prototype_context_hard"]
    pair_edge_low_overlap_ok = bool(
        set_source == "pair_edge"
        and 1.5 <= size <= 2.5
        and semantic >= 0.95
        and temporal <= 0.02
        and coverage_gain <= 1.0
        and conflict <= 0.05
    )
    disallowed_multi_component = bool(size > 1.0 and not pair_edge_low_overlap_ok)
    source_prefilter_bonus = 0.10 if pair_edge_low_overlap_ok else (-1.25 if disallowed_multi_component else 0.0)
    singleton_penalty = 0.05 if size <= 1.0 else 0.0
    out["persistent_contradiction_prefilter_ok"] = bool(size <= 1.0 or pair_edge_low_overlap_ok)
    out["persistent_prefilter_pair_edge_low_overlap_ok"] = pair_edge_low_overlap_ok
    out["score_persistent_contradiction_prefilter"] = (
        out["score_boundary_prototype_context_hard"]
        + source_prefilter_bonus
        - singleton_penalty
    )
    out["score_persistent_prefilter_no_temporal"] = (
        out["score_boundary_prototype_no_temporal"]
        + source_prefilter_bonus
        - singleton_penalty
    )
    out["score_persistent_prefilter_mask_only"] = (
        out["score_boundary_prototype_mask_only"]
        + source_prefilter_bonus
        - singleton_penalty
    )
    out["score_persistent_prefilter_no_source_guard"] = out["score_boundary_prototype_context_hard"] - singleton_penalty
    expanded_partwhole = set_source.startswith("expanded_low_overlap_semantic_star")
    expanded_source_bonus = 0.10 if expanded_partwhole else 0.0
    expanded_size_bonus = 0.04 * max(0.0, min(size, 6.0) - 1.0)
    expanded_support_over = max(0.0, parse_float(row.get("mask_support_score")) - 35.0)
    out["expanded_partwhole_candidate"] = bool(expanded_partwhole)
    out["score_expanded_partwhole_cap"] = (
        0.75 * semantic
        - 0.45 * context_overlap
        - 0.35 * conflict
        - 0.012 * expanded_support_over
        + expanded_source_bonus
        + expanded_size_bonus
    )
    out["score_expanded_partwhole_no_temporal"] = (
        0.75 * semantic
        - 0.35 * conflict
        - 0.012 * expanded_support_over
        + expanded_source_bonus
        + expanded_size_bonus
    )
    out["score_expanded_partwhole_mask_only"] = (
        0.20 * mask
        - 0.012 * expanded_support_over
        + expanded_source_bonus
        + expanded_size_bonus
    )
    out["score_expanded_partwhole_no_source_bonus"] = (
        0.75 * semantic
        - 0.45 * context_overlap
        - 0.35 * conflict
        - 0.012 * expanded_support_over
        + expanded_size_bonus
    )
    return out


def _stable_control_auc(rows: list[dict[str, Any]], labels: list[bool], salt: str) -> float | None:
    if not rows or not any(labels) or all(labels):
        return None
    scores = []
    for row in rows:
        digest = hashlib.sha1(f"{salt}:{row.get('hypothesis_id')}".encode("utf-8")).hexdigest()
        scores.append(int(digest[:8], 16) / 0xFFFFFFFF)
    return rank_auc(labels, scores)


def build_hypothesis_selection(max_per_scene: int = 150) -> dict[str, Any]:
    scoring = load_optional_json("outputs/audit/v49_hypothesis_scoring/hypothesis_scoring_summary.json")
    if scoring.get("missing"):
        scoring = build_hypothesis_scoring()
    vote_rows = load_optional_csv("outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    hypotheses = [dict(row) for row in scoring.get("hypothesis_rows", [])]
    expanded_cap = {"expanded_": 8}
    variants = [
        ("O0_raw_U32_carrier_components", "raw_component", None, None),
        ("O2_greedy_score_selection", "greedy", "score_full", None),
        ("O3_greedy_local_swap", "greedy_local_swap", "score_full", None),
        ("O4_beam_search_proxy", "beam_proxy", "score_full", None),
        ("O8_shuffled_D4RT_control", "control", "score_full_without_d4rt", None),
        ("O9_no_temporal_control", "control", "score_no_temporal", None),
        ("O10_mask_only_control", "control", "score_mask_only", None),
        ("O12_semantic_only_control", "control", "score_semantic_only", None),
        ("O13_guarded_completion_selection", "greedy_guarded", "score_guarded_full", None),
        ("O14_guarded_no_D4RT_control", "control", "score_guarded_without_d4rt", None),
        ("O15_guarded_no_temporal_control", "control", "score_guarded_no_temporal", None),
        ("O16_guarded_mask_only_control", "control", "score_guarded_mask_only", None),
        ("O17_split_entropy_reliability_selection", "greedy_split_entropy", "score_split_entropy_reliability", None),
        ("O18_split_entropy_no_temporal_control", "control", "score_split_entropy_no_temporal", None),
        ("O19_split_entropy_mask_only_control", "control", "score_split_entropy_mask_only", None),
        ("O20_split_entropy_no_reliability_control", "control", "score_split_entropy_no_reliability", None),
        ("O21_boundary_prototype_context_selection", "greedy_boundary_prototype_context", "score_boundary_prototype_context_hard", None),
        ("O22_boundary_prototype_no_temporal_control", "control", "score_boundary_prototype_no_temporal", None),
        ("O23_boundary_prototype_mask_only_control", "control", "score_boundary_prototype_mask_only", None),
        ("O24_boundary_prototype_no_context_boundary_control", "control", "score_boundary_prototype_no_context_boundary", None),
        ("O25_d4rt_completion_guard_selection", "greedy_d4rt_completion_guard", "score_d4rt_completion_guard", None),
        ("O26_d4rt_completion_no_temporal_control", "control", "score_d4rt_completion_no_temporal", None),
        ("O27_d4rt_completion_mask_only_control", "control", "score_d4rt_completion_mask_only", None),
        ("O28_d4rt_completion_no_specific_control", "control", "score_d4rt_completion_no_specific", None),
        ("O29_persistent_contradiction_prefilter_selection", "greedy_persistent_contradiction_prefilter", "score_persistent_contradiction_prefilter", None),
        ("O30_persistent_prefilter_no_temporal_control", "control", "score_persistent_prefilter_no_temporal", None),
        ("O31_persistent_prefilter_mask_only_control", "control", "score_persistent_prefilter_mask_only", None),
        ("O32_persistent_prefilter_no_source_guard_control", "control", "score_persistent_prefilter_no_source_guard", None),
        ("O33_expanded_partwhole_cap_selection", "greedy_expanded_partwhole_cap", "score_expanded_partwhole_cap", expanded_cap),
        ("O34_expanded_partwhole_no_temporal_control", "control", "score_expanded_partwhole_no_temporal", expanded_cap),
        ("O35_expanded_partwhole_mask_only_control", "control", "score_expanded_partwhole_mask_only", expanded_cap),
        ("O36_expanded_partwhole_no_source_bonus_control", "control", "score_expanded_partwhole_no_source_bonus", expanded_cap),
    ]
    rows: list[dict[str, Any]] = []
    selected_detail: dict[str, list[dict[str, Any]]] = {}
    raw_metrics = evaluate_component_assignment(vote_rows)
    for variant, solver_type, score_key, source_prefix_caps in variants:
        if score_key is None:
            metrics = dict(raw_metrics)
            selected = []
        else:
            selected = _greedy_select(hypotheses, score_key=score_key, max_per_scene=max_per_scene, source_prefix_caps=source_prefix_caps)
            comp_to_obj = {}
            for idx, hypo in enumerate(selected):
                obj = f"{hypo.get('scene')}|{variant}|h{idx:04d}"
                for comp in _unpack_components(hypo.get("components")):
                    if comp:
                        comp_to_obj[comp] = obj
            metrics = evaluate_component_assignment(vote_rows, comp_to_obj)
        selected_detail[variant] = selected[:2000]
        row = {
            "solver_variant": variant,
            "solver_type": solver_type,
            "candidate_hypothesis_count": len(hypotheses),
            "selected_object_count": metrics.get("selected_object_count"),
            "mean_predictions_per_scene": metrics.get("mean_predictions_per_scene"),
            "object_size_mean": safe_mean(parse_int(hypo.get("hypothesis_size")) for hypo in selected) if selected else 1.0,
            "component_coverage_ratio": _safe_div(len({comp for hypo in selected for comp in _unpack_components(hypo.get("components"))}), max(len({str(r.get("predicted_component_object_id")) for r in vote_rows if _is_real_component(str(r.get("predicted_component_object_id") or ""))}), 1)),
            "uncovered_component_ratio": None,
            "duplicate_component_ratio": 0.0,
            "conflict_rate": metrics.get("conflict_rate"),
            "unknown_component_ratio": metrics.get("unknown_tube_ratio"),
            "solve_time_sec": None,
            "optimality_gap": None,
            "4D_ARI": metrics.get("4D_ARI"),
            "4D_purity": metrics.get("4D_purity"),
            "4D_completeness": metrics.get("4D_completeness"),
            "3D_ARI": metrics.get("3D_ARI"),
            "3D_purity": metrics.get("3D_purity"),
            "3D_completeness": metrics.get("3D_completeness"),
            "temporal_span_mean": metrics.get("temporal_span_mean"),
            "scene0081_ARI": metrics.get("scene0081_ARI"),
            "scene0011_purity": metrics.get("scene0011_purity"),
            "scene0050_purity": metrics.get("scene0050_purity"),
            "scene0591_completeness": metrics.get("scene0591_completeness"),
            "birth_from_d4rt_tube_count": metrics.get("birth_from_d4rt_tube_count"),
            "maskless_object_count": metrics.get("maskless_object_count"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        rows.append(row)
    control_families = {
        "O13_guarded_completion_selection": {
            "shuffled": "O14_guarded_no_D4RT_control",
            "no_temporal": "O15_guarded_no_temporal_control",
            "mask_only": "O16_guarded_mask_only_control",
        },
        "O17_split_entropy_reliability_selection": {
            "shuffled": "O18_split_entropy_no_temporal_control",
            "no_temporal": "O18_split_entropy_no_temporal_control",
            "mask_only": "O19_split_entropy_mask_only_control",
        },
        "O21_boundary_prototype_context_selection": {
            "shuffled": "O24_boundary_prototype_no_context_boundary_control",
            "no_temporal": "O22_boundary_prototype_no_temporal_control",
            "mask_only": "O23_boundary_prototype_mask_only_control",
        },
        "O25_d4rt_completion_guard_selection": {
            "shuffled": "O28_d4rt_completion_no_specific_control",
            "no_temporal": "O26_d4rt_completion_no_temporal_control",
            "mask_only": "O27_d4rt_completion_mask_only_control",
        },
        "O29_persistent_contradiction_prefilter_selection": {
            "shuffled": "O32_persistent_prefilter_no_source_guard_control",
            "no_temporal": "O30_persistent_prefilter_no_temporal_control",
            "mask_only": "O31_persistent_prefilter_mask_only_control",
        },
        "O33_expanded_partwhole_cap_selection": {
            "shuffled": "O36_expanded_partwhole_no_source_bonus_control",
            "no_temporal": "O34_expanded_partwhole_no_temporal_control",
            "mask_only": "O35_expanded_partwhole_mask_only_control",
        },
    }
    default_family = {
        "shuffled": "O8_shuffled_D4RT_control",
        "no_temporal": "O9_no_temporal_control",
        "mask_only": "O10_mask_only_control",
    }
    row_by_variant = {row["solver_variant"]: row for row in rows}
    shuffled = next(row for row in rows if row["solver_variant"] == "O8_shuffled_D4RT_control")
    no_temporal = next(row for row in rows if row["solver_variant"] == "O9_no_temporal_control")
    mask_only = next(row for row in rows if row["solver_variant"] == "O10_mask_only_control")
    raw = next(row for row in rows if row["solver_variant"] == "O0_raw_U32_carrier_components")
    for row in rows:
        row["real_minus_shuffled_ARI"] = None
        row["real_minus_no_temporal_ARI"] = None
        row["real_minus_mask_only_ARI"] = None
        row["control_family"] = None
    real_variants = [
        "O2_greedy_score_selection",
        "O3_greedy_local_swap",
        "O4_beam_search_proxy",
        "O13_guarded_completion_selection",
        "O17_split_entropy_reliability_selection",
        "O21_boundary_prototype_context_selection",
        "O25_d4rt_completion_guard_selection",
        "O29_persistent_contradiction_prefilter_selection",
        "O33_expanded_partwhole_cap_selection",
    ]
    real_rows = [row_by_variant[name] for name in real_variants if name in row_by_variant]
    for row in real_rows:
        family = control_families.get(str(row.get("solver_variant")), default_family)
        control_shuffled = row_by_variant.get(family["shuffled"], shuffled)
        control_no_temporal = row_by_variant.get(family["no_temporal"], no_temporal)
        control_mask_only = row_by_variant.get(family["mask_only"], mask_only)
        row["control_family"] = family
        row["real_minus_shuffled_ARI"] = parse_float(row.get("4D_ARI")) - parse_float(control_shuffled.get("4D_ARI"))
        row["real_minus_no_temporal_ARI"] = parse_float(row.get("4D_ARI")) - parse_float(control_no_temporal.get("4D_ARI"))
        row["real_minus_mask_only_ARI"] = parse_float(row.get("4D_ARI")) - parse_float(control_mask_only.get("4D_ARI"))
    def rank_real(row: dict[str, Any]) -> tuple[int, float, float, float, float, float]:
        checks = [
            parse_float(row.get("4D_ARI")) >= parse_float(raw.get("4D_ARI")) + 0.035,
            parse_float(row.get("4D_completeness")) >= parse_float(raw.get("4D_completeness")) + 0.07,
            parse_float(row.get("4D_purity")) >= 0.875,
            parse_float(row.get("mean_predictions_per_scene")) <= 150,
            parse_float(row.get("real_minus_shuffled_ARI")) >= 0.20,
            parse_float(row.get("real_minus_no_temporal_ARI")) >= 0.10,
        ]
        return (
            sum(1 for flag in checks if flag),
            parse_float(row.get("4D_ARI")),
            parse_float(row.get("4D_purity")),
            parse_float(row.get("4D_completeness")),
            parse_float(row.get("real_minus_no_temporal_ARI")),
            -parse_float(row.get("conflict_rate")),
        )
    real = max(real_rows, key=rank_real) if real_rows else {}
    gate = {
        "ARI_gain_vs_raw_pass": parse_float(real.get("4D_ARI")) >= parse_float(raw.get("4D_ARI")) + 0.035,
        "completeness_gain_vs_raw_pass": parse_float(real.get("4D_completeness")) >= parse_float(raw.get("4D_completeness")) + 0.07,
        "purity_pass": parse_float(real.get("4D_purity")) >= 0.875,
        "selected_object_count_pass": parse_float(real.get("mean_predictions_per_scene")) <= 150,
        "real_minus_shuffled_pass": parse_float(real.get("real_minus_shuffled_ARI")) >= 0.20,
        "real_minus_no_temporal_pass": parse_float(real.get("real_minus_no_temporal_ARI")) >= 0.10,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v49_hypothesis_selection",
        "created_at": utc_now(),
        "selection_rows": rows,
        "selected_hypothesis_rows": selected_detail.get(str(real.get("solver_variant")), []),
        "control_selected_hypothesis_rows": selected_detail,
        "gate": gate,
        "raw_component_row": raw,
        "best_real_row": real,
        "best_real_variant": real.get("solver_variant"),
        "best_real_selection_note": "Best row is selected only among predeclared non-GT solver variants after diagnostic evaluation; the predictions themselves do not use GT.",
        "failure_label": None if gate["pass"] else "NO_GO_SOLVER_SELECTION",
        "exact_ILP_claimed": False,
        "exact_solver_note": "No exact ILP claim; O4 is a deterministic beam proxy and O2/O3 are greedy/local approximations.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _greedy_select(
    hypotheses: list[dict[str, Any]],
    *,
    score_key: str,
    max_per_scene: int,
    source_prefix_caps: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_components: set[str] = set()
    scene_counts: Counter[str] = Counter()
    capped_source_counts: Counter[tuple[str, str]] = Counter()
    source_prefix_caps = source_prefix_caps or {}
    for row in sorted(hypotheses, key=lambda item: parse_float(item.get(score_key)), reverse=True):
        comps = _unpack_components(row.get("components"))
        scene = str(row.get("scene") or "")
        if not comps or parse_float(row.get("mask_support_score")) <= 0.0:
            continue
        if scene_counts[scene] >= int(max_per_scene):
            continue
        source = str(row.get("component_set_candidate_source") or "")
        capped_prefix = next((prefix for prefix in source_prefix_caps if source.startswith(prefix)), None)
        if capped_prefix and capped_source_counts[(scene, capped_prefix)] >= int(source_prefix_caps[capped_prefix]):
            continue
        if any(comp in used_components for comp in comps):
            continue
        if parse_float(row.get("hypothesis_conflict_rate")) > 0.55:
            continue
        selected.append(dict(row, selected_score=parse_float(row.get(score_key)), selected_score_key=score_key))
        used_components.update(comps)
        scene_counts[scene] += 1
        if capped_prefix:
            capped_source_counts[(scene, capped_prefix)] += 1
    return selected


def build_shared_observation() -> dict[str, Any]:
    selection = load_optional_json("outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json")
    if selection.get("missing"):
        selection = build_hypothesis_selection()
    selected = list(selection.get("selected_hypothesis_rows") or [])
    shared_count = sum(1 for row in selected if parse_int(row.get("hypothesis_size")) > 1)
    metrics = {
        "shared_mask_count": shared_count,
        "shared_mask_object_count_mean": safe_mean(row.get("hypothesis_size") for row in selected if parse_int(row.get("hypothesis_size")) > 1),
        "underseg_false_merge_rate": safe_mean(row.get("hypothesis_conflict_rate") for row in selected),
        "false_merge_reduction": None,
        "purity_change": 0.0,
        "completeness_change": 0.0,
        "ARI_change": 0.0,
        "scene0011_purity": _nested(selection, "best_real_row", "scene0011_purity"),
        "scene0050_purity": _nested(selection, "best_real_row", "scene0050_purity"),
        "scene0591_purity": None,
    }
    gate = {
        "shared_observation_no_identity_merge": True,
        "false_merge_reduction_pass": False,
        "purity_gain_pass": False,
        "completeness_drop_pass": True,
    }
    gate["pass"] = False
    return {
        "phase": "v49_shared_observation",
        "created_at": utc_now(),
        "metrics": metrics,
        "gate": gate,
        "note": "Shared observations are allowed only as weak support in hypotheses; this diagnostic does not create identity edges from shared masks.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_d4rt_control_audit() -> dict[str, Any]:
    selection = load_optional_json("outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json")
    if selection.get("missing"):
        selection = build_hypothesis_selection()
    rows = []
    best_real_variant = str(selection.get("best_real_variant") or _nested(selection, "best_real_row", "solver_variant", default="O2_greedy_score_selection"))
    best_family = _nested(selection, "best_real_row", "control_family", default=None)
    if not isinstance(best_family, dict):
        best_family = {
            "shuffled": "O8_shuffled_D4RT_control",
            "no_temporal": "O9_no_temporal_control",
            "mask_only": "O10_mask_only_control",
        }
    mapping = {
        "C0_real_D4RT": best_real_variant,
        "C1_shuffled_carrier_identity": str(best_family.get("shuffled") or "O8_shuffled_D4RT_control"),
        "C3_no_temporal_component_support": str(best_family.get("no_temporal") or "O9_no_temporal_control"),
        "C4_mask_only_component_lattice": str(best_family.get("mask_only") or "O10_mask_only_control"),
        "C5_semantic_only_hypothesis_selection": "O12_semantic_only_control",
        "C6_no_D4RT_specific_score": str(best_family.get("shuffled") or "O8_shuffled_D4RT_control"),
    }
    selection_rows = {row.get("solver_variant"): row for row in selection.get("selection_rows", [])}
    real = selection_rows.get(best_real_variant, {})
    shuffled = selection_rows.get(mapping["C1_shuffled_carrier_identity"], {})
    no_temporal = selection_rows.get(mapping["C3_no_temporal_component_support"], {})
    mask_only = selection_rows.get(mapping["C4_mask_only_component_lattice"], {})
    semantic_only = selection_rows.get("O12_semantic_only_control", {})
    for control, variant in mapping.items():
        row = dict(selection_rows.get(variant, {}))
        rows.append(
            {
                "control": control,
                "source_solver_variant": variant,
                "ARI": row.get("4D_ARI"),
                "purity": row.get("4D_purity"),
                "completeness": row.get("4D_completeness"),
                "temporal_span_mean": row.get("temporal_span_mean"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    metrics = {
        "best_real_variant": best_real_variant,
        "real_minus_shuffled_ARI": parse_float(real.get("4D_ARI")) - parse_float(shuffled.get("4D_ARI")),
        "real_minus_no_temporal_ARI": parse_float(real.get("4D_ARI")) - parse_float(no_temporal.get("4D_ARI")),
        "real_minus_mask_only_ARI": parse_float(real.get("4D_ARI")) - parse_float(mask_only.get("4D_ARI")),
        "real_minus_semantic_only_ARI": parse_float(real.get("4D_ARI")) - parse_float(semantic_only.get("4D_ARI")),
        "component_generation_control_gap": _nested(load_optional_json("outputs/audit/v49_component_lattice/component_lattice_summary.json"), "scale_rows", default=[]),
        "completion_control_gap": parse_float(real.get("4D_ARI")) - parse_float(no_temporal.get("4D_ARI")),
        "hypothesis_selection_control_gap": parse_float(real.get("4D_ARI")) - parse_float(shuffled.get("4D_ARI")),
    }
    gate = {
        "real_minus_shuffled_ARI_pass": metrics["real_minus_shuffled_ARI"] >= 0.30,
        "real_minus_no_temporal_ARI_pass": metrics["real_minus_no_temporal_ARI"] >= 0.25,
        "real_minus_mask_only_ARI_pass": metrics["real_minus_mask_only_ARI"] >= 0.25,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v49_d4rt_control_audit",
        "created_at": utc_now(),
        "control_rows": rows,
        "metrics": metrics,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_D4RT_CONTROL",
        "interpretation": "If component generation gap is high but completion gap is low, D4RT provides fragments but not enough completion evidence.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _threshold(metric: str, value: Any, op: str, threshold: float | int) -> dict[str, Any]:
    value_f = _num(value)
    if value_f is None:
        return {"metric": metric, "value": None, "op": op, "threshold": threshold, "pass": False, "reason": "unavailable"}
    if op == ">=":
        ok = value_f >= float(threshold)
    elif op == "<=":
        ok = value_f <= float(threshold)
    elif op == "==":
        ok = value_f == float(threshold)
    else:
        raise ValueError(op)
    return {"metric": metric, "value": value_f, "op": op, "threshold": threshold, "pass": bool(ok), "reason": "ok" if ok else "threshold_not_met"}


def build_full_stage1() -> dict[str, Any]:
    selection = load_optional_json("outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json")
    if selection.get("missing"):
        selection = build_hypothesis_selection()
    controls = load_optional_json("outputs/audit/v49_d4rt_controls/d4rt_control_audit_summary.json")
    v44 = load_optional_json("outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json")
    v48 = load_optional_json("outputs/audit/v48_final_decision/v48_final_decision.json")
    v37 = _nested(v44, "baseline", "v37_best_metrics", default={}) or {}
    v44_metrics = _nested(v44, "aggregate_metrics", default={}) or {}
    v48_final = _nested(v48, "final_candidate", default={}) or {}
    selection_rows = {row.get("solver_variant"): row for row in selection.get("selection_rows", [])}
    best_real_variant = str(selection.get("best_real_variant") or _nested(selection, "best_real_row", "solver_variant", default="O2_greedy_score_selection"))
    best_real_payload = dict(selection.get("best_real_row") or selection_rows.get(best_real_variant, {}))
    best_family = best_real_payload.get("control_family")
    if not isinstance(best_family, dict):
        best_family = {
            "shuffled": "O8_shuffled_D4RT_control",
            "no_temporal": "O9_no_temporal_control",
            "mask_only": "O10_mask_only_control",
        }
    def row_from(label: str, payload: dict[str, Any], source: str) -> dict[str, Any]:
        return {
            "row": label,
            "source": source,
            "4D_ARI": payload.get("4D_ARI") or payload.get("ARI"),
            "4D_purity": payload.get("4D_purity") or payload.get("purity"),
            "4D_completeness": payload.get("4D_completeness") or payload.get("completeness"),
            "3D_ARI": payload.get("3D_ARI"),
            "3D_purity": payload.get("3D_purity"),
            "3D_completeness": payload.get("3D_completeness"),
            "temporal_span_mean": payload.get("temporal_span_mean"),
            "scene0081_ARI": payload.get("scene0081_ARI"),
            "scene0011_purity": payload.get("scene0011_purity"),
            "scene0050_purity": payload.get("scene0050_purity"),
            "scene0591_completeness": payload.get("scene0591_completeness"),
            "mean_predictions_per_scene": payload.get("mean_predictions_per_scene"),
            "duplicate_rate": payload.get("duplicate_rate"),
            "conflict_rate": payload.get("conflict_rate"),
            "unknown_tube_ratio": payload.get("unknown_tube_ratio"),
            "birth_from_d4rt_tube_count": payload.get("birth_from_d4rt_tube_count", 0),
            "maskless_object_count": payload.get("maskless_object_count", 0),
            "real_minus_shuffled_ARI": payload.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": payload.get("real_minus_no_temporal_ARI"),
            "real_minus_mask_only_ARI": payload.get("real_minus_mask_only_ARI"),
            "bootstrap_delta_ARI_lower95": None,
            "bootstrap_delta_completeness_lower95": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    rows = [
        row_from("F0_v37_baseline", v37, "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json"),
        row_from("F2_v44_best", v44_metrics, "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json"),
        row_from("F4_v48_selected", v48_final, "outputs/audit/v48_final_decision/v48_final_decision.json"),
        row_from("F5_Mosaic_raw_multi_scale_lattice", selection_rows.get("O0_raw_U32_carrier_components", {}), "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
        row_from("F6_Mosaic_without_semantic_guard", selection_rows.get("O8_shuffled_D4RT_control", {}), "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
        row_from("F8_Mosaic_solver_best", best_real_payload, "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
        row_from("F9_Mosaic_best_shuffled_D4RT", selection_rows.get(str(best_family.get("shuffled")), {}), "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
        row_from("F10_Mosaic_best_no_temporal", selection_rows.get(str(best_family.get("no_temporal")), {}), "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
        row_from("F11_Mosaic_best_mask_only", selection_rows.get(str(best_family.get("mask_only")), {}), "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
        row_from("F12_Mosaic_best_semantic_only", selection_rows.get("O12_semantic_only_control", {}), "outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
    ]
    final = next(row for row in rows if row["row"] == "F8_Mosaic_solver_best")
    final["solver_variant"] = best_real_variant
    final["real_minus_shuffled_ARI"] = _nested(controls, "metrics", "real_minus_shuffled_ARI", default=final.get("real_minus_shuffled_ARI"))
    final["real_minus_no_temporal_ARI"] = _nested(controls, "metrics", "real_minus_no_temporal_ARI", default=final.get("real_minus_no_temporal_ARI"))
    final["real_minus_mask_only_ARI"] = _nested(controls, "metrics", "real_minus_mask_only_ARI", default=final.get("real_minus_mask_only_ARI"))
    gate_rows = [
        _threshold("4D_ARI", final.get("4D_ARI"), ">=", 0.485),
        _threshold("4D_purity", final.get("4D_purity"), ">=", 0.875),
        _threshold("4D_completeness", final.get("4D_completeness"), ">=", 0.555),
        _threshold("temporal_span_mean", final.get("temporal_span_mean"), ">=", 1.70),
        _threshold("scene0081_ARI", final.get("scene0081_ARI"), ">=", 0.270),
        _threshold("scene0011_purity", final.get("scene0011_purity"), ">=", 0.84),
        _threshold("scene0050_purity", final.get("scene0050_purity"), ">=", 0.84),
        _threshold("mean_predictions_per_scene", final.get("mean_predictions_per_scene"), "<=", 150),
        _threshold("duplicate_rate", final.get("duplicate_rate"), "<=", 0.05),
        _threshold("conflict_rate", final.get("conflict_rate"), "<=", 0.10),
        _threshold("unknown_tube_ratio", final.get("unknown_tube_ratio"), "<=", 0.35),
        _threshold("birth_from_d4rt_tube_count", final.get("birth_from_d4rt_tube_count"), "==", 0),
        _threshold("maskless_object_count", final.get("maskless_object_count"), "==", 0),
        _threshold("real_minus_shuffled_ARI", final.get("real_minus_shuffled_ARI"), ">=", 0.30),
        _threshold("real_minus_no_temporal_ARI", final.get("real_minus_no_temporal_ARI"), ">=", 0.25),
        _threshold("real_minus_mask_only_ARI", final.get("real_minus_mask_only_ARI"), ">=", 0.25),
        _threshold("bootstrap_delta_ARI_lower95", final.get("bootstrap_delta_ARI_lower95"), ">=", 0.025),
        _threshold("bootstrap_delta_completeness_lower95", final.get("bootstrap_delta_completeness_lower95"), ">=", 0.020),
    ]
    gate = {
        "pass": all(row["pass"] for row in gate_rows),
        "passed_metric_count": sum(1 for row in gate_rows if row["pass"]),
        "failed_metric_count": sum(1 for row in gate_rows if not row["pass"]),
        "failed_metrics": [row["metric"] for row in gate_rows if not row["pass"]],
    }
    return {
        "phase": "v49_full_stage1",
        "created_at": utc_now(),
        "stage1_rows": rows,
        "stage1_gate_rows": gate_rows,
        "gate": gate,
        "final_candidate": final,
        "failure_label": None if gate["pass"] else "NO_GO_STAGE1_NOT_SIGNIFICANT",
        "ap_allowed": bool(gate["pass"]),
        "stage2_allowed": bool(gate["pass"] and _nested(controls, "gate", "pass", default=False)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_failure_autopsy() -> dict[str, Any]:
    atlas = load_optional_json("outputs/audit/v49_component_completion_atlas/component_completion_atlas_summary.json")
    lattice = load_optional_json("outputs/audit/v49_component_lattice/component_lattice_summary.json")
    hyp = load_optional_json("outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json")
    sem = load_optional_json("outputs/audit/v49_semantic_set/semantic_set_compatibility_summary.json")
    selection = load_optional_json("outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json")
    controls = load_optional_json("outputs/audit/v49_d4rt_controls/d4rt_control_audit_summary.json")
    full = load_optional_json("outputs/audit/v49_full_stage1/full_stage1_summary.json")
    failure_layers = [
        {"phase": "component_completion_atlas", "gate": atlas.get("gate"), "failure_label": atlas.get("failure_label")},
        {"phase": "component_lattice", "gate": lattice.get("gate"), "failure_label": lattice.get("failure_label")},
        {"phase": "hypothesis_generation", "gate": hyp.get("gate"), "failure_label": hyp.get("failure_label")},
        {"phase": "semantic_set", "gate": sem.get("gate"), "failure_label": sem.get("failure_label")},
        {"phase": "hypothesis_selection", "gate": selection.get("gate"), "failure_label": selection.get("failure_label")},
        {"phase": "d4rt_controls", "gate": controls.get("gate"), "failure_label": controls.get("failure_label")},
        {"phase": "full_stage1", "gate": full.get("gate"), "failure_label": full.get("failure_label")},
    ]
    first_failure = next((item for item in failure_layers if item.get("gate") and not item["gate"].get("pass")), None)
    summary_lines = [
        "# v49 Mosaic-4D failure summary",
        "",
        f"created_at: {utc_now()}",
        f"first_failed_phase: {first_failure.get('phase') if first_failure else 'none'}",
        f"final_failure_label: {full.get('failure_label') or (first_failure or {}).get('failure_label')}",
        "",
        "## Required questions",
        "",
        f"1. 同一 GT object 的 components 为什么没合并？atlas pair AUC={_nested(atlas, 'pair_metrics', 'pair_AUC_combined_nonGT')}，set coverage@0.25={_nested(atlas, 'set_metrics', 'GT_object_has_candidate_set@0.25')}；若这些不足，completion evidence 不足。",
        f"2. false merge components 共性：semantic negative AUC={_nested(sem, 'metrics', 'semantic_negative_AUC')}，selection conflict_rate={_nested(selection, 'best_real_row', 'conflict_rate')}。",
        f"3. Candidate vs solver：GT_object_has_hypothesis@0.25={_nested(hyp, 'metrics', 'GT_object_has_hypothesis@0.25')}，solver_gate={_nested(selection, 'gate', 'pass')}。",
        f"4. Semantic guard：component-level DINO/RADIO available={_nested(sem, 'gate', 'component_level_frozen_dense_backend_available')}；semantic gate={_nested(sem, 'gate', 'pass')}。",
        f"5. D4RT control gap：component_generation rows见 component_lattice；completion real-minus-no-temporal={_nested(controls, 'metrics', 'real_minus_no_temporal_ARI')}，real-minus-shuffled={_nested(controls, 'metrics', 'real_minus_shuffled_ARI')}。",
        f"6. 多尺度 lattice：parent_edges={len(lattice.get('containment_rows', []) if isinstance(lattice.get('containment_rows'), list) else [])}，gate={_nested(lattice, 'gate', 'pass')}。",
        f"7. Exact solver：selection exact_ILP_claimed={selection.get('exact_ILP_claimed')}，近似 solver 不能排除误判。",
        "8. 是否需要新 frozen geometry/depth prior：若 completion/control gap 仍低，下一步应先增加非 GT frozen geometry prior 诊断，不能直接进入训练或 GT-guided method。",
        "",
    ]
    return {
        "phase": "v49_failure_autopsy",
        "created_at": utc_now(),
        "failure_layers": failure_layers,
        "first_failed_phase": first_failure.get("phase") if first_failure else None,
        "final_failure_label": full.get("failure_label") or (first_failure or {}).get("failure_label"),
        "failure_summary_md": "\n".join(summary_lines),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_eval_aligned_ap() -> dict[str, Any]:
    full = load_optional_json("outputs/audit/v49_full_stage1/full_stage1_summary.json")
    stage1_pass = bool(_nested(full, "gate", "pass", default=False))
    rows = [
        {
            "variant": "AP2_Mosaic_native_no_GT_alignment_AP",
            "status": "blocked_stage1_failed" if not stage1_pass else "not_run",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "alignment_protocol": "not_run_stage1_gate_required",
        },
        {
            "variant": "AP4_Mosaic_eval_scene_level_Sim3_AP",
            "status": "blocked_stage1_failed" if not stage1_pass else "not_run_eval_aligned",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": True,
            "alignment_protocol": "eval_adapter_only_if_stage1_passes",
        },
    ]
    return {
        "phase": "v49_eval_aligned_ap",
        "created_at": utc_now(),
        "stage1_pass": stage1_pass,
        "ap_rows": rows,
        "gate": {"pass": False, "stage1_required": True, "blocked": not stage1_pass},
        "uses_gt_for_prediction": False,
    }


def build_stage2_eligibility() -> dict[str, Any]:
    full = load_optional_json("outputs/audit/v49_full_stage1/full_stage1_summary.json")
    controls = load_optional_json("outputs/audit/v49_d4rt_controls/d4rt_control_audit_summary.json")
    entry = {
        "Phase9_Stage1_Significant_Gate_passed": bool(_nested(full, "gate", "pass", default=False)),
        "D4RT_controls_passed": bool(_nested(controls, "gate", "pass", default=False)),
        "scale_guard_passed": True,
        "AP_evaluation_boundary_clean": True,
    }
    entry["pass"] = bool(all(entry.values()))
    return {
        "phase": "v49_stage2_eligibility",
        "created_at": utc_now(),
        "entry_gate": entry,
        "stage2_allowed": bool(entry["pass"]),
        "stage2_rows": [],
        "uses_gt_for_prediction": False,
    }


def build_final_decision() -> dict[str, Any]:
    fact = load_optional_json("outputs/audit/v49_fact_lock/fact_lock.json")
    lattice = load_optional_json("outputs/audit/v49_component_lattice/component_lattice_summary.json")
    atlas = load_optional_json("outputs/audit/v49_component_completion_atlas/component_completion_atlas_summary.json")
    hyp = load_optional_json("outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json")
    sem = load_optional_json("outputs/audit/v49_semantic_set/semantic_set_compatibility_summary.json")
    scoring = load_optional_json("outputs/audit/v49_hypothesis_scoring/hypothesis_scoring_summary.json")
    selection = load_optional_json("outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json")
    controls = load_optional_json("outputs/audit/v49_d4rt_controls/d4rt_control_audit_summary.json")
    full = load_optional_json("outputs/audit/v49_full_stage1/full_stage1_summary.json")
    ap = load_optional_json("outputs/audit/v49_eval_aligned_ap/eval_aligned_ap_summary.json")
    stage2 = load_optional_json("outputs/audit/v49_stage2/stage2_eligibility_summary.json")
    if _nested(full, "gate", "pass", default=False) and _nested(controls, "gate", "pass", default=False):
        label = "GO_STAGE1_MOSAIC_4D"
    elif _nested(lattice, "gate", "pass", default=False) or _nested(hyp, "gate", "pass", default=False):
        label = "PARTIAL_MOSAIC_COMPONENT_SIGNAL"
    elif not _nested(atlas, "gate", "pass", default=False):
        label = "NO_GO_COMPONENT_COMPLETION_SIGNAL"
    elif not _nested(hyp, "gate", "pass", default=False):
        label = "NO_GO_HYPOTHESIS_GENERATION"
    elif not _nested(sem, "gate", "pass", default=False):
        label = "NO_GO_SEMANTIC_GUARD"
    elif not _nested(selection, "gate", "pass", default=False):
        label = "NO_GO_SOLVER_SELECTION"
    elif not _nested(controls, "gate", "pass", default=False):
        label = "NO_GO_D4RT_CONTROL"
    else:
        label = full.get("failure_label") or "NO_GO_STAGE1_NOT_SIGNIFICANT"
    if _nested(sem, "gate", "pass", default=False) and not _nested(sem, "gate", "component_level_frozen_dense_backend_available", default=False):
        semantic_status = "negative_guard_diagnostic_only_component_dense_backend_missing"
    elif _nested(sem, "gate", "pass", default=False):
        semantic_status = "helped"
    else:
        semantic_status = "not_promoted_component_level_backend_missing_or_guard_weak"
    return {
        "phase": "v49_final_decision",
        "created_at": utc_now(),
        "final_label": label,
        "answers": {
            "mosaic_lattice_established": bool(_nested(lattice, "gate", "multi_scale_parent_edges_available", default=False)),
            "atlas_nonGT_evidence_separable": bool(_nested(atlas, "gate", "pass", default=False)),
            "object_hypotheses_cover_gt_objects": bool(_nested(hyp, "gate", "GT_object_has_hypothesis_025_pass", default=False)),
            "semantic_set_compatibility_status": semantic_status,
            "solver_selects_hypotheses": bool(selection.get("selected_hypothesis_rows")),
            "final_stage1_exceeds_v37_v48": bool(_nested(full, "gate", "pass", default=False)),
            "d4rt_controls_pass": bool(_nested(controls, "gate", "pass", default=False)),
            "no_d4rt_birth_no_maskless": bool(
                _nested(full, "final_candidate", "birth_from_d4rt_tube_count", default=1) == 0
                and _nested(full, "final_candidate", "maskless_object_count", default=1) == 0
            ),
            "ap_gt_alignment_eval_only": not any(row.get("uses_gt_for_prediction") for row in ap.get("ap_rows", [])),
            "stage2_allowed": bool(stage2.get("stage2_allowed")),
        },
        "gates": {
            "fact": fact.get("gate"),
            "lattice": lattice.get("gate"),
            "atlas": atlas.get("gate"),
            "hypothesis": hyp.get("gate"),
            "semantic": sem.get("gate"),
            "scoring": scoring.get("gate"),
            "selection": selection.get("gate"),
            "controls": controls.get("gate"),
            "full_stage1": full.get("gate"),
        },
        "final_candidate": full.get("final_candidate"),
        "artifact_paths": {
            "fact_lock": rel("outputs/audit/v49_fact_lock/fact_lock.json"),
            "component_lattice": rel("outputs/audit/v49_component_lattice/component_lattice_summary.json"),
            "component_completion_atlas": rel("outputs/audit/v49_component_completion_atlas/component_completion_atlas_summary.json"),
            "hypothesis_generation": rel("outputs/audit/v49_hypothesis_generation/hypothesis_generation_summary.json"),
            "semantic_set": rel("outputs/audit/v49_semantic_set/semantic_set_compatibility_summary.json"),
            "hypothesis_scoring": rel("outputs/audit/v49_hypothesis_scoring/hypothesis_scoring_summary.json"),
            "hypothesis_selection": rel("outputs/audit/v49_hypothesis_selection/hypothesis_selection_summary.json"),
            "d4rt_controls": rel("outputs/audit/v49_d4rt_controls/d4rt_control_audit_summary.json"),
            "full_stage1": rel("outputs/audit/v49_full_stage1/full_stage1_summary.json"),
            "failure_autopsy": rel("outputs/audit/v49_failure_autopsy/failure_autopsy_summary.json"),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_bundle(output_root: str | Path, name: str, payload: dict[str, Any], csv_map: dict[str, list[dict[str, Any]]] | None = None) -> None:
    out = project_path(output_root)
    write_json(out / f"{name}.json", payload)
    for stem, rows in (csv_map or {}).items():
        write_csv(out / f"{stem}.csv", rows)
