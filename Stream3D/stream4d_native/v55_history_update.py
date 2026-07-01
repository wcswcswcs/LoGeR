from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import (
    ROOT,
    color_feature,
    cosine,
    load_mask_label,
    load_rgb,
    parse_float,
    parse_int,
    read_csv,
    read_json,
    utc_now,
    write_csv,
    write_json,
)
from .v53_local_objectlets import weighted_partition_metrics


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


def _load_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _dominant(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return str(max(counter.items(), key=lambda item: (int(item[1]), str(item[0])))[0])


def _component_atom_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    return {(str(row.get("scene")), str(row.get("component_id"))): str(row.get("atom_id")) for row in rows}


def _support_component_gt(
    support_rows_path: Path,
    *,
    support_variant: str,
    scenes: set[str],
) -> dict[tuple[str, str], Counter[str]]:
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with support_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            scene = str(row.get("scene"))
            if scene not in scenes:
                continue
            gt = str(row.get("diagnostic_gt_instance") or "")
            if not gt or gt == "0":
                continue
            counters[(scene, str(row.get("component_id")))] [gt] += max(parse_int(row.get("support_count")), 1)
    return counters


def _chunk_ranges(
    chunk_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    roles: set[str],
) -> list[tuple[str, int, int, str]]:
    role_by_chunk = {str(row.get("chunk_id")): str(row.get("role")) for row in role_rows}
    ranges: list[tuple[str, int, int, str]] = []
    for row in chunk_rows:
        chunk_id = str(row.get("chunk_id"))
        if role_by_chunk.get(chunk_id, "") not in roles:
            continue
        ranges.append(
            (
                str(row.get("scene")),
                parse_int(row.get("raw_frame_start")),
                parse_int(row.get("raw_frame_end")),
                chunk_id,
            )
        )
    return ranges


def _objectlet_frame_mask_counters(
    native_carrier_rows_path: Path,
    needed_objectlet_ids: set[str],
) -> dict[str, Counter[tuple[int, int]]]:
    counters: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    if not native_carrier_rows_path.exists() or not needed_objectlet_ids:
        return counters
    with native_carrier_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            objectlet_id = str(row.get("objectlet_id") or "")
            if objectlet_id not in needed_objectlet_ids:
                continue
            observed_mask_id = parse_int(row.get("observed_mask_id"))
            if observed_mask_id <= 0:
                continue
            counters[objectlet_id][(parse_int(row.get("frame_id")), observed_mask_id)] += 1
    return counters


def _objectlet_frame_mask_component_counters(
    native_carrier_rows_path: Path,
    needed_objectlet_ids: set[str],
) -> dict[str, dict[tuple[int, int], Counter[str]]]:
    counters: dict[str, dict[tuple[int, int], Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    if not native_carrier_rows_path.exists() or not needed_objectlet_ids:
        return counters
    with native_carrier_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            objectlet_id = str(row.get("objectlet_id") or "")
            if objectlet_id not in needed_objectlet_ids:
                continue
            observed_mask_id = parse_int(row.get("observed_mask_id"))
            component_id = str(row.get("component_id") or "")
            if observed_mask_id <= 0 or not component_id:
                continue
            counters[objectlet_id][(parse_int(row.get("frame_id")), observed_mask_id)][component_id] += 1
    return counters


def _component_accumulation_pass(
    stats: dict[str, Any] | None,
    *,
    min_support: int,
    min_masks: int,
    min_frames: int,
) -> bool:
    if not stats:
        return False
    return (
        int(stats.get("support", 0)) >= int(min_support)
        and len(stats.get("masks", set())) >= int(min_masks)
        and len(stats.get("frames", set())) >= int(min_frames)
    )


def _component_support_gate_pass(
    meta: dict[str, Any] | None,
    *,
    max_selected_rank: int,
    min_w_visible: float,
    min_r_mask: float,
    require_dominant: bool,
) -> bool:
    if not meta:
        return False
    if int(max_selected_rank) > 0 and int(meta.get("selected_rank", 0)) > int(max_selected_rank):
        return False
    if float(min_w_visible) > 0.0 and float(meta.get("W_visible", 0.0)) < float(min_w_visible):
        return False
    if float(min_r_mask) > 0.0 and float(meta.get("R_mask", 0.0)) < float(min_r_mask):
        return False
    if require_dominant and not bool(meta.get("is_dominant_component", False)):
        return False
    return True


def _objectlet_frame_projection_stats(
    native_carrier_rows_path: Path,
    needed_objectlet_ids: set[str],
) -> dict[str, dict[int, dict[str, float]]]:
    stats: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    if not native_carrier_rows_path.exists() or not needed_objectlet_ids:
        return stats
    with native_carrier_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            objectlet_id = str(row.get("objectlet_id") or "")
            if objectlet_id not in needed_objectlet_ids:
                continue
            frame_id = parse_int(row.get("frame_id"))
            uv_x = parse_float(row.get("uv_x"), default=np.nan)
            uv_y = parse_float(row.get("uv_y"), default=np.nan)
            if not (np.isfinite(uv_x) and np.isfinite(uv_y)):
                continue
            frame_stats = stats[objectlet_id].get(frame_id)
            if frame_stats is None:
                frame_stats = {"n": 0.0, "sx": 0.0, "sy": 0.0, "minx": uv_x, "miny": uv_y, "maxx": uv_x, "maxy": uv_y}
                stats[objectlet_id][frame_id] = frame_stats
            frame_stats["n"] += 1.0
            frame_stats["sx"] += uv_x
            frame_stats["sy"] += uv_y
            frame_stats["minx"] = min(frame_stats["minx"], uv_x)
            frame_stats["miny"] = min(frame_stats["miny"], uv_y)
            frame_stats["maxx"] = max(frame_stats["maxx"], uv_x)
            frame_stats["maxy"] = max(frame_stats["maxy"], uv_y)
    return stats


def _parse_mask_observation_id(mask_observation_id: Any) -> tuple[str, int, int] | None:
    try:
        scene, frame_id, mask_id = str(mask_observation_id).rsplit(":", 2)
    except ValueError:
        return None
    return scene, parse_int(frame_id), parse_int(mask_id)


def _semantic_mask_feature(
    mask_observation_id: str,
    *,
    backend: str,
    device: str,
    checkpoint: str | None,
    short_side: int,
    adapter_cache: dict[str, Any],
    feature_map_cache: dict[tuple[str, int, str], Any],
    feature_cache: dict[tuple[str, str, str, int], tuple[list[float], dict[str, Any]]],
) -> tuple[list[float], dict[str, Any]]:
    cache_key = (str(mask_observation_id), str(backend), str(checkpoint or ""), int(short_side))
    if cache_key in feature_cache:
        return feature_cache[cache_key]
    parsed = _parse_mask_observation_id(mask_observation_id)
    diag: dict[str, Any] = {
        "mask_observation_id": str(mask_observation_id),
        "semantic_backend": str(backend),
        "semantic_checkpoint": checkpoint,
        "semantic_feature_available": False,
        "semantic_feature_missing_reason": "",
    }
    if parsed is None:
        diag["semantic_feature_missing_reason"] = "bad_mask_observation_id"
        feature_cache[cache_key] = ([], diag)
        return feature_cache[cache_key]
    scene, frame_id, mask_id = parsed
    diag.update({"scene": scene, "frame_id": int(frame_id), "mask_id": int(mask_id)})
    label = load_mask_label(scene, frame_id)
    if label is None:
        diag["semantic_feature_missing_reason"] = "missing_cropformer_mask_png"
        feature_cache[cache_key] = ([], diag)
        return feature_cache[cache_key]
    mask = label == int(mask_id)
    diag["semantic_mask_pixel_count"] = int(mask.sum())
    if not np.any(mask):
        diag["semantic_feature_missing_reason"] = "empty_mask"
        feature_cache[cache_key] = ([], diag)
        return feature_cache[cache_key]
    if str(backend) == "colorhist":
        feature, ok = color_feature(scene, frame_id, mask)
        diag["semantic_feature_available"] = bool(ok)
        if not ok:
            diag["semantic_feature_missing_reason"] = "colorhist_unavailable"
        diag["semantic_feature_dim"] = len(feature)
        feature_cache[cache_key] = (feature, diag)
        return feature_cache[cache_key]
    if str(backend) != "dinov2_timm":
        diag["semantic_feature_missing_reason"] = f"unsupported_backend:{backend}"
        feature_cache[cache_key] = ([], diag)
        return feature_cache[cache_key]
    rgb = load_rgb(scene, frame_id)
    if rgb is None:
        diag["semantic_feature_missing_reason"] = "missing_rgb_jpg"
        feature_cache[cache_key] = ([], diag)
        return feature_cache[cache_key]
    if rgb.shape[:2] != label.shape[:2]:
        import cv2

        rgb = cv2.resize(rgb, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
    try:
        from .frozen_feature_adapter import FrozenFeatureAdapter, locate_default_dinov2_checkpoint

        adapter_key = f"{backend}|{device}|{checkpoint or ''}|{int(short_side)}"
        adapter = adapter_cache.get(adapter_key)
        if adapter is None:
            resolved_checkpoint = checkpoint or locate_default_dinov2_checkpoint()
            adapter = FrozenFeatureAdapter(
                backend="dinov2_timm",
                device=str(device),
                checkpoint=resolved_checkpoint,
                short_side=int(short_side),
            )
            adapter_cache[adapter_key] = adapter
            diag["semantic_checkpoint"] = resolved_checkpoint
        frame_key = (scene, int(frame_id), adapter_key)
        feature_map = feature_map_cache.get(frame_key)
        if feature_map is None:
            feature_map = adapter.extract_dense_features(rgb)
            feature_map_cache[frame_key] = feature_map
        pooled = np.asarray(adapter.pool_mask_feature(feature_map, mask), dtype=np.float32).reshape(-1)
    except Exception as exc:  # pragma: no cover - depends on optional frozen backends
        diag["semantic_feature_missing_reason"] = f"feature_extract_failed:{type(exc).__name__}:{exc}"
        feature_cache[cache_key] = ([], diag)
        return feature_cache[cache_key]
    norm = float(np.linalg.norm(pooled))
    if pooled.size == 0 or norm <= 1e-8:
        diag["semantic_feature_missing_reason"] = "empty_feature"
        feature_cache[cache_key] = ([], diag)
        return feature_cache[cache_key]
    feature = [float(value) for value in (pooled / norm).tolist()]
    diag["semantic_feature_available"] = True
    diag["semantic_feature_dim"] = len(feature)
    diag["semantic_feature_missing_reason"] = ""
    feature_cache[cache_key] = (feature, diag)
    return feature_cache[cache_key]


def _bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    ix0 = max(left["minx"], right["minx"])
    iy0 = max(left["miny"], right["miny"])
    ix1 = min(left["maxx"], right["maxx"])
    iy1 = min(left["maxy"], right["maxy"])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, left["maxx"] - left["minx"]) * max(0.0, left["maxy"] - left["miny"])
    right_area = max(0.0, right["maxx"] - right["minx"]) * max(0.0, right["maxy"] - right["miny"])
    return float(inter / max(left_area + right_area - inter, 1e-12))


def _center_distance(left: dict[str, float], right: dict[str, float]) -> float:
    lx = left["sx"] / max(left["n"], 1.0)
    ly = left["sy"] / max(left["n"], 1.0)
    rx = right["sx"] / max(right["n"], 1.0)
    ry = right["sy"] / max(right["n"], 1.0)
    return float(np.hypot(lx - rx, ly - ry))


def _build_assignment_map(histories: dict[str, dict[str, Any]], key: str) -> tuple[dict[tuple[str, str], str], int]:
    mapping: dict[tuple[str, str], str] = {}
    duplicates = 0
    for history_id, history in histories.items():
        scene = str(history["scene"])
        for component_id in history[key]:
            comp_key = (scene, str(component_id))
            if comp_key in mapping and mapping[comp_key] != history_id:
                duplicates += 1
                continue
            mapping[comp_key] = history_id
    return mapping, duplicates


def _next_history_by_scene(histories: dict[str, dict[str, Any]]) -> dict[str, str]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for history_id, history in histories.items():
        by_scene[str(history["scene"])].append(history_id)
    next_history: dict[str, str] = {}
    for history_ids in by_scene.values():
        history_ids.sort()
        if len(history_ids) <= 1:
            continue
        for index, history_id in enumerate(history_ids):
            next_history[history_id] = history_ids[(index + 1) % len(history_ids)]
    return next_history


def _metrics_from_support(
    support_rows_path: Path,
    *,
    support_variant: str,
    scenes: set[str],
    component_to_history: dict[tuple[str, str], str],
) -> dict[str, float]:
    assignments: list[tuple[str, str, float]] = []
    with support_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            scene = str(row.get("scene"))
            if scene not in scenes:
                continue
            gt = str(row.get("diagnostic_gt_instance") or "")
            if not gt or gt == "0":
                continue
            component_id = str(row.get("component_id"))
            pred = component_to_history.get((scene, component_id), f"unknown:{scene}:{component_id}")
            assignments.append((pred, f"{scene}|gt:{gt}", float(max(parse_int(row.get("support_count")), 1))))
    return weighted_partition_metrics(assignments)


def build_v55_history_update(
    *,
    chunk_role_rows_path: str | Path = "outputs/audit/v55_chunk_roles/chunk_role_rows.csv",
    chunk_rows_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_rows.csv",
    anchor_birth_rows_path: str | Path = "outputs/audit/v55_anchor_birth/anchor_birth_rows.csv",
    objectlet_rows_path: str | Path = "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    local_summary_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json",
    component_atom_rows_path: str | Path = "outputs/audit/v55_atoms/component_atom_rows.csv",
    support_rows_path: str | Path = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    min_overlap_atoms: int = 1,
    confirmed_overlap_ratio: float = 0.10,
    history_evidence_roles: tuple[str, ...] = ("bridge", "update"),
    cosupport_seed_ratio_min: float = 0.38,
    cosupport_dominance_ratio_min: float = 1.0,
    enable_mask_cosupport: bool = True,
    enable_cosupport_native_gate: bool = False,
    cosupport_native_min_support: int = 20,
    native_carrier_rows_path: str | Path | None = None,
    enable_native_frame_mask_projection: bool = True,
    native_boundary_min_support: int = 100,
    native_boundary_min_candidate_ratio: float = 0.10,
    native_boundary_min_jaccard: float = 0.01,
    native_boundary_min_shared_frame_masks: int = 3,
    enable_native_uv_projection: bool = False,
    native_uv_min_support: int = 20,
    native_uv_min_candidate_ratio: float = 0.10,
    native_uv_min_jaccard: float = 0.0,
    native_uv_min_mean_iou: float = 0.05,
    native_uv_max_center_dist: float = 0.10,
    native_uv_min_shared_frames: int = 3,
    enable_native_history_mask_projection: bool = False,
    native_history_mask_min_support: int = 100,
    native_history_mask_min_ratio: float = 0.90,
    native_history_mask_min_dominance: float = 1.5,
    native_history_mask_min_mask_ratio: float = 0.0,
    enable_native_history_mask_component_gate: bool = False,
    native_history_mask_component_min_support: int = 1,
    enable_native_history_mask_component_accumulation_gate: bool = False,
    native_history_mask_component_accumulation_min_support: int = 1,
    native_history_mask_component_accumulation_min_masks: int = 2,
    native_history_mask_component_accumulation_min_frames: int = 2,
    enable_native_history_mask_component_support_gate: bool = False,
    native_history_mask_component_max_selected_rank: int = 0,
    native_history_mask_component_min_w_visible: float = 0.0,
    native_history_mask_component_min_r_mask: float = 0.0,
    native_history_mask_component_require_dominant: bool = False,
    enable_native_history_mask_cannot_link_guard: bool = False,
    native_history_mask_other_seed_min_support: int = 1,
    native_history_mask_other_seed_min_ratio: float = 0.05,
    native_history_mask_second_native_min_support: int = 1,
    native_history_mask_second_native_min_ratio: float = 0.05,
    enable_native_history_mask_semantic_guard: bool = False,
    native_history_mask_semantic_backend: str = "colorhist",
    native_history_mask_semantic_min_cosine: float = 0.94,
    native_history_mask_semantic_device: str = "cpu",
    native_history_mask_semantic_checkpoint: str | None = None,
    native_history_mask_semantic_short_side: int = 518,
    objectlet_variant_override: str | None = None,
) -> dict[str, Any]:
    role_rows = read_csv(_project(chunk_role_rows_path))
    chunk_rows = read_csv(_project(chunk_rows_path))
    anchor_birth_rows = read_csv(_project(anchor_birth_rows_path))
    objectlet_rows = read_csv(_project(objectlet_rows_path))
    local_summary = read_json(_project(local_summary_path))
    component_atom_rows = read_csv(_project(component_atom_rows_path))
    component_to_atom = _component_atom_map(component_atom_rows)
    best_variant = str(objectlet_variant_override or local_summary.get("best_method_variant") or "")
    update_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) == "update"}
    scenes = {str(row.get("scene")) for row in role_rows}
    component_gt = _support_component_gt(_project(support_rows_path), support_variant=support_variant, scenes=scenes)

    histories: dict[str, dict[str, Any]] = {}
    for row in anchor_birth_rows:
        if str(row.get("accepted_birth")).lower() != "true":
            continue
        history_id = str(row.get("birth_object_id"))
        components = set(_load_list(row.get("component_ids")))
        atoms = set(_load_list(row.get("atom_ids")))
        scene = str(row.get("scene"))
        gt_counter: Counter[str] = Counter()
        for component_id in components:
            gt_counter.update(component_gt.get((scene, component_id), Counter()))
        histories[history_id] = {
            "history_id": history_id,
            "scene": scene,
            "anchor_chunk_id": str(row.get("anchor_chunk_id")),
            "source_mask_observation_id": str(row.get("source_mask_observation_id") or ""),
            "anchor_components": set(components),
            "history_components": set(components),
            "anchor_atoms": set(atoms),
            "history_atoms": set(atoms),
            "chunks": {str(row.get("anchor_chunk_id"))},
            "dominant_gt": _dominant(gt_counter),
        }

    update_candidates = [
        row for row in objectlet_rows if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in update_chunks
    ]
    evidence_roles = {str(role) for role in history_evidence_roles if str(role)}
    evidence_chunks = {str(row.get("chunk_id")) for row in role_rows if str(row.get("role")) in evidence_roles}
    evidence_objectlet_candidates = [
        row for row in objectlet_rows if str(row.get("variant")) == best_variant and str(row.get("chunk_id")) in evidence_chunks
    ]
    update_rows: list[dict[str, Any]] = []
    confirmed_update_count = 0
    partial_update_count = 0
    conflict_reject_count = 0
    duplicate_component_updates = 0
    update_precision_hits = 0
    update_precision_total = 0
    added_components_by_history: dict[str, set[str]] = defaultdict(set)
    for candidate in update_candidates:
        scene = str(candidate.get("scene"))
        components = set(_load_list(candidate.get("component_ids")))
        atoms = {component_to_atom.get((scene, component_id), "") for component_id in components}
        atoms = {atom for atom in atoms if atom}
        scored: list[tuple[int, float, str, dict[str, Any]]] = []
        for history_id, history in histories.items():
            if history["scene"] != scene:
                continue
            overlap = atoms & history["history_atoms"]
            if len(overlap) < int(min_overlap_atoms):
                continue
            ratio = len(overlap) / max(len(atoms), 1)
            scored.append((len(overlap), ratio, history_id, history))
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        overlap_count, overlap_ratio, history_id, history = scored[0]
        conflict = parse_float(candidate.get("same_frame_exclusion_violation_rate"))
        outside = parse_float(candidate.get("outside_all_related_masks_ratio_mean"))
        if conflict > 0.08 or outside > 0.35:
            state = "conflict_reject"
            conflict_reject_count += 1
            accepted_components: set[str] = set()
        elif overlap_ratio >= float(confirmed_overlap_ratio):
            state = "confirmed_update"
            confirmed_update_count += 1
            accepted_components = components
        else:
            state = "partial_update"
            partial_update_count += 1
            accepted_components = components
        for component_id in accepted_components:
            if component_id in history["history_components"]:
                continue
            history_gt = history.get("dominant_gt")
            component_label = _dominant(component_gt.get((scene, component_id), Counter()))
            if history_gt and component_label:
                update_precision_total += 1
                if component_label == history_gt:
                    update_precision_hits += 1
            history["history_components"].add(component_id)
            added_components_by_history[history_id].add(component_id)
            atom_id = component_to_atom.get((scene, component_id), "")
            if atom_id:
                history["history_atoms"].add(atom_id)
        if accepted_components:
            history["chunks"].add(str(candidate.get("chunk_id")))
        duplicate_component_updates += max(len(scored) - 1, 0)
        update_rows.append(
            {
                "scene": scene,
                "chunk_id": candidate.get("chunk_id"),
                "history_id": history_id,
                "candidate_id": candidate.get("candidate_id"),
                "update_state": state,
                "overlap_atom_count": overlap_count,
                "overlap_atom_ratio": overlap_ratio,
                "accepted_component_count": len(accepted_components),
                "candidate_component_count": len(components),
                "same_frame_exclusion_violation_rate": conflict,
                "outside_all_related_masks_ratio_mean": outside,
                "update_source": "objectlet_atom_overlap",
                "seed_support_ratio": None,
                "seed_dominance_ratio": None,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    objectlet_confirmed_update_count = confirmed_update_count
    objectlet_partial_update_count = partial_update_count
    objectlet_conflict_reject_count = conflict_reject_count
    evidence_ranges = _chunk_ranges(chunk_rows, role_rows, evidence_roles)
    comp_to_seed_history: dict[tuple[str, str], list[str]] = defaultdict(list)
    for history_id, history in histories.items():
        for component_id in history["anchor_components"]:
            comp_to_seed_history[(str(history["scene"]), str(component_id))].append(history_id)
    histories_by_scene: dict[str, list[str]] = defaultdict(list)
    for history_id, history in histories.items():
        histories_by_scene[str(history["scene"])].append(history_id)

    support_by_mask: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    support_mask_meta: dict[tuple[str, str, str], tuple[int, int]] = {}
    support_component_meta: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    if enable_mask_cosupport or enable_native_history_mask_projection:
        with _project(support_rows_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("variant")) != support_variant:
                    continue
                scene = str(row.get("scene"))
                frame_id = parse_int(row.get("frame_id"))
                component_id = str(row.get("component_id"))
                support_count = max(parse_int(row.get("support_count")), 1)
                for range_scene, start, end, chunk_id in evidence_ranges:
                    if scene == range_scene and start <= frame_id <= end:
                        key = (scene, chunk_id, str(row.get("mask_observation_id")))
                        support_by_mask[key][component_id] += support_count
                        support_mask_meta[key] = (frame_id, parse_int(row.get("mask_id")))
                        meta = support_component_meta[key].setdefault(
                            component_id,
                            {
                                "support_count": 0,
                                "W_visible": 0.0,
                                "R_mask": 0.0,
                                "W_all_carrier": 0.0,
                                "selected_rank": parse_int(row.get("selected_rank")),
                                "is_dominant_component": False,
                            },
                        )
                        meta["support_count"] += support_count
                        meta["W_visible"] = max(float(meta["W_visible"]), parse_float(row.get("W_visible")))
                        meta["R_mask"] = max(float(meta["R_mask"]), parse_float(row.get("R_mask")))
                        meta["W_all_carrier"] = max(float(meta["W_all_carrier"]), parse_float(row.get("W_all_carrier")))
                        selected_rank = parse_int(row.get("selected_rank"))
                        if selected_rank > 0:
                            current_rank = int(meta.get("selected_rank", 0))
                            meta["selected_rank"] = selected_rank if current_rank <= 0 else min(current_rank, selected_rank)
                        meta["is_dominant_component"] = bool(meta["is_dominant_component"]) or (
                            str(row.get("is_dominant_component")).lower() == "true"
                        )

    native_boundary_candidate_count = 0
    native_boundary_accepted_count = 0
    native_boundary_added_component_count = 0
    native_boundary_conflict_count = 0
    native_uv_candidate_count = 0
    native_uv_accepted_count = 0
    native_uv_added_component_count = 0
    native_uv_duplicate_noop_count = 0
    native_history_mask_candidate_count = 0
    native_history_mask_accepted_count = 0
    native_history_mask_added_component_count = 0
    native_history_mask_duplicate_noop_count = 0
    native_history_mask_component_gate_candidate_component_count = 0
    native_history_mask_component_gate_direct_component_count = 0
    native_history_mask_component_gate_filtered_component_count = 0
    native_history_mask_component_accumulation_candidate_component_count = 0
    native_history_mask_component_accumulation_eligible_component_count = 0
    native_history_mask_component_accumulation_filtered_component_count = 0
    native_history_mask_component_support_gate_candidate_component_count = 0
    native_history_mask_component_support_gate_pass_component_count = 0
    native_history_mask_component_support_gate_filtered_component_count = 0
    native_history_mask_cannot_link_reject_count = 0
    native_history_mask_semantic_reject_count = 0
    native_history_mask_semantic_feature_attempt_count = 0
    native_history_mask_semantic_feature_success_count = 0
    native_boundary_available = False
    native_projection_accepted_objectlets: set[str] = set()
    native_frame_masks_by_objectlet: dict[str, Counter[tuple[int, int]]] = {}
    native_frame_mask_components_by_objectlet: dict[str, dict[tuple[int, int], Counter[str]]] = {}
    if native_carrier_rows_path is not None:
        native_path = _project(native_carrier_rows_path)
        native_boundary_available = native_path.exists()
        needed_objectlets = {str(row.get("objectlet_id") or "") for row in evidence_objectlet_candidates} | set(histories)
        if enable_native_frame_mask_projection or enable_cosupport_native_gate or enable_native_history_mask_projection:
            native_frame_masks_by_objectlet = _objectlet_frame_mask_counters(native_path, needed_objectlets)
        if enable_native_history_mask_projection and enable_native_history_mask_component_gate:
            native_frame_mask_components_by_objectlet = _objectlet_frame_mask_component_counters(
                native_path, needed_objectlets
            )
        if enable_native_frame_mask_projection:
            frame_masks_by_objectlet = native_frame_masks_by_objectlet
            for candidate in evidence_objectlet_candidates:
                scene = str(candidate.get("scene"))
                objectlet_id = str(candidate.get("objectlet_id") or "")
                candidate_counter = frame_masks_by_objectlet.get(objectlet_id, Counter())
                if not candidate_counter:
                    continue
                native_boundary_candidate_count += 1
                candidate_support = sum(candidate_counter.values())
                scored: list[tuple[int, int, float, float, float, str]] = []
                for history_id in histories_by_scene.get(scene, []):
                    history_counter = frame_masks_by_objectlet.get(history_id, Counter())
                    if not history_counter:
                        continue
                    shared_keys = set(candidate_counter) & set(history_counter)
                    if not shared_keys:
                        continue
                    shared_support = sum(min(candidate_counter[key], history_counter[key]) for key in shared_keys)
                    history_support = sum(history_counter.values())
                    candidate_ratio = float(shared_support / max(candidate_support, 1))
                    history_ratio = float(shared_support / max(history_support, 1))
                    jaccard_support = float(shared_support / max(candidate_support + history_support - shared_support, 1))
                    scored.append((shared_support, len(shared_keys), candidate_ratio, history_ratio, jaccard_support, history_id))
                if not scored:
                    continue
                scored.sort(key=lambda item: (-item[0], -item[2], -item[4], item[5]))
                shared_support, shared_frame_masks, candidate_ratio, history_ratio, jaccard_support, history_id = scored[0]
                if (
                    shared_support < int(native_boundary_min_support)
                    or candidate_ratio < float(native_boundary_min_candidate_ratio)
                    or jaccard_support < float(native_boundary_min_jaccard)
                    or shared_frame_masks < int(native_boundary_min_shared_frame_masks)
                ):
                    continue
                native_boundary_accepted_count += 1
                history = histories[history_id]
                components = set(_load_list(candidate.get("component_ids")))
                accepted_components: set[str] = set()
                for component_id in components:
                    if component_id in history["history_components"]:
                        continue
                    history_gt = history.get("dominant_gt")
                    component_label = _dominant(component_gt.get((scene, component_id), Counter()))
                    if history_gt and component_label:
                        update_precision_total += 1
                        if component_label == history_gt:
                            update_precision_hits += 1
                    history["history_components"].add(component_id)
                    added_components_by_history[history_id].add(component_id)
                    accepted_components.add(component_id)
                    atom_id = component_to_atom.get((scene, component_id), "")
                    if atom_id:
                        history["history_atoms"].add(atom_id)
                if accepted_components:
                    confirmed_update_count += 1
                    history["chunks"].add(str(candidate.get("chunk_id")))
                    native_projection_accepted_objectlets.add(objectlet_id)
                else:
                    native_boundary_conflict_count += 1
                    conflict_reject_count += 1
                native_boundary_added_component_count += len(accepted_components)
                update_rows.append(
                    {
                        "scene": scene,
                        "chunk_id": candidate.get("chunk_id"),
                        "history_id": history_id,
                        "candidate_id": candidate.get("candidate_id"),
                        "objectlet_id": objectlet_id,
                        "update_state": "confirmed_update" if accepted_components else "conflict_reject",
                        "overlap_atom_count": None,
                        "overlap_atom_ratio": None,
                        "accepted_component_count": len(accepted_components),
                        "candidate_component_count": len(components),
                        "same_frame_exclusion_violation_rate": candidate.get("same_frame_exclusion_violation_rate"),
                        "outside_all_related_masks_ratio_mean": candidate.get("outside_all_related_masks_ratio_mean"),
                        "update_source": "native_boundary_projection",
                        "seed_support_ratio": None,
                        "seed_dominance_ratio": None,
                        "native_shared_frame_mask_count": int(shared_frame_masks),
                        "native_shared_support_min_sum": int(shared_support),
                        "native_candidate_projection_ratio": candidate_ratio,
                        "native_history_projection_ratio": history_ratio,
                        "native_projection_jaccard_support": jaccard_support,
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                    }
                )

        if enable_native_uv_projection:
            projection_stats = _objectlet_frame_projection_stats(native_path, needed_objectlets)
            for candidate in evidence_objectlet_candidates:
                scene = str(candidate.get("scene"))
                objectlet_id = str(candidate.get("objectlet_id") or "")
                if objectlet_id in native_projection_accepted_objectlets:
                    native_uv_duplicate_noop_count += 1
                    continue
                candidate_stats = projection_stats.get(objectlet_id, {})
                if not candidate_stats:
                    continue
                native_uv_candidate_count += 1
                candidate_support = sum(frame_stats["n"] for frame_stats in candidate_stats.values())
                scored: list[tuple[float, int, float, float, float, float, float, str]] = []
                for history_id in histories_by_scene.get(scene, []):
                    history_stats = projection_stats.get(history_id, {})
                    shared_frames = set(candidate_stats) & set(history_stats)
                    if not shared_frames:
                        continue
                    uv_support = 0.0
                    weighted_iou = 0.0
                    min_center_dist = float("inf")
                    accepted_frame_count = 0
                    for frame_id in shared_frames:
                        candidate_frame = candidate_stats[frame_id]
                        history_frame = history_stats[frame_id]
                        iou = _bbox_iou(candidate_frame, history_frame)
                        center_dist = _center_distance(candidate_frame, history_frame)
                        # Candidate frame evidence is image-space only; keep frames with bbox contact
                        # or close centroids, then apply aggregate thresholds below.
                        if iou <= 0.0 and center_dist > 0.25:
                            continue
                        frame_support = min(candidate_frame["n"], history_frame["n"])
                        uv_support += frame_support
                        weighted_iou += iou * frame_support
                        min_center_dist = min(min_center_dist, center_dist)
                        accepted_frame_count += 1
                    if uv_support <= 0.0:
                        continue
                    history_support = sum(frame_stats["n"] for frame_stats in history_stats.values())
                    candidate_ratio = float(uv_support / max(candidate_support, 1.0))
                    history_ratio = float(uv_support / max(history_support, 1.0))
                    uv_jaccard = float(uv_support / max(candidate_support + history_support - uv_support, 1.0))
                    mean_iou = float(weighted_iou / max(uv_support, 1.0))
                    scored.append(
                        (
                            uv_support,
                            accepted_frame_count,
                            candidate_ratio,
                            history_ratio,
                            uv_jaccard,
                            mean_iou,
                            min_center_dist,
                            history_id,
                        )
                    )
                if not scored:
                    continue
                scored.sort(key=lambda item: (-item[0], -item[2], -item[5], item[6], item[7]))
                (
                    uv_support,
                    shared_frames,
                    candidate_ratio,
                    history_ratio,
                    uv_jaccard,
                    mean_iou,
                    min_center_dist,
                    history_id,
                ) = scored[0]
                if (
                    uv_support < float(native_uv_min_support)
                    or candidate_ratio < float(native_uv_min_candidate_ratio)
                    or uv_jaccard < float(native_uv_min_jaccard)
                    or mean_iou < float(native_uv_min_mean_iou)
                    or min_center_dist > float(native_uv_max_center_dist)
                    or shared_frames < int(native_uv_min_shared_frames)
                ):
                    continue
                history = histories[history_id]
                components = set(_load_list(candidate.get("component_ids")))
                accepted_components = set()
                for component_id in components:
                    if component_id in history["history_components"]:
                        continue
                    history_gt = history.get("dominant_gt")
                    component_label = _dominant(component_gt.get((scene, component_id), Counter()))
                    if history_gt and component_label:
                        update_precision_total += 1
                        if component_label == history_gt:
                            update_precision_hits += 1
                    history["history_components"].add(component_id)
                    added_components_by_history[history_id].add(component_id)
                    accepted_components.add(component_id)
                    atom_id = component_to_atom.get((scene, component_id), "")
                    if atom_id:
                        history["history_atoms"].add(atom_id)
                if accepted_components:
                    confirmed_update_count += 1
                    native_uv_accepted_count += 1
                    history["chunks"].add(str(candidate.get("chunk_id")))
                    native_projection_accepted_objectlets.add(objectlet_id)
                else:
                    native_uv_duplicate_noop_count += 1
                native_uv_added_component_count += len(accepted_components)
                update_rows.append(
                    {
                        "scene": scene,
                        "chunk_id": candidate.get("chunk_id"),
                        "history_id": history_id,
                        "candidate_id": candidate.get("candidate_id"),
                        "objectlet_id": objectlet_id,
                        "update_state": "confirmed_update" if accepted_components else "duplicate_noop",
                        "overlap_atom_count": None,
                        "overlap_atom_ratio": None,
                        "accepted_component_count": len(accepted_components),
                        "candidate_component_count": len(components),
                        "same_frame_exclusion_violation_rate": candidate.get("same_frame_exclusion_violation_rate"),
                        "outside_all_related_masks_ratio_mean": candidate.get("outside_all_related_masks_ratio_mean"),
                        "update_source": "native_uv_bbox_projection",
                        "seed_support_ratio": None,
                        "seed_dominance_ratio": None,
                        "native_uv_shared_frame_count": int(shared_frames),
                        "native_uv_support_min_sum": int(uv_support),
                        "native_uv_candidate_ratio": candidate_ratio,
                        "native_uv_history_ratio": history_ratio,
                        "native_uv_jaccard_support": uv_jaccard,
                        "native_uv_mean_bbox_iou": mean_iou,
                        "native_uv_min_center_dist": min_center_dist,
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                    }
                )

    native_history_mask_projection_active = bool(
        enable_native_history_mask_projection and native_carrier_rows_path is not None
    )
    native_history_mask_semantic_guard_active = bool(
        native_history_mask_projection_active and enable_native_history_mask_semantic_guard
    )
    native_history_mask_component_accumulation_gate_active = bool(
        native_history_mask_projection_active and enable_native_history_mask_component_accumulation_gate
    )
    native_history_mask_component_support_gate_active = bool(
        native_history_mask_projection_active and enable_native_history_mask_component_support_gate
    )
    if native_history_mask_projection_active:
        semantic_adapter_cache: dict[str, Any] = {}
        semantic_feature_map_cache: dict[tuple[str, int, str], Any] = {}
        semantic_feature_cache: dict[tuple[str, str, str, int], tuple[list[float], dict[str, Any]]] = {}
        accumulation_stats: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        accumulation_eligible_components: set[tuple[str, str, str, str]] = set()
        if native_history_mask_component_accumulation_gate_active:
            for (scene, chunk_id, mask_observation_id), component_counts in sorted(support_by_mask.items()):
                frame_id, mask_id = support_mask_meta.get((scene, chunk_id, mask_observation_id), (0, 0))
                scored_native = [
                    (
                        native_frame_masks_by_objectlet.get(history_id, Counter()).get((frame_id, mask_id), 0),
                        history_id,
                    )
                    for history_id in histories_by_scene.get(scene, [])
                ]
                scored_native = [(support, history_id) for support, history_id in scored_native if support > 0]
                if not scored_native:
                    continue
                scored_native.sort(key=lambda item: (-item[0], item[1]))
                native_support, history_id = scored_native[0]
                second_native_support = scored_native[1][0] if len(scored_native) > 1 else 0
                total_native_support = sum(support for support, _history_id in scored_native)
                native_ratio = float(native_support / max(total_native_support, 1))
                native_mask_ratio = float(native_support / max(sum(component_counts.values()), 1))
                native_dominance = (
                    float(native_support / max(second_native_support, 1)) if second_native_support else None
                )
                if (
                    native_support < int(native_history_mask_min_support)
                    or native_ratio < float(native_history_mask_min_ratio)
                    or native_mask_ratio < float(native_history_mask_min_mask_ratio)
                    or (
                        native_dominance is not None
                        and native_dominance < float(native_history_mask_min_dominance)
                    )
                ):
                    continue
                for component_id, support_count in component_counts.items():
                    if component_id in histories[history_id]["history_components"]:
                        continue
                    if any(
                        other_id != history_id
                        for other_id in comp_to_seed_history.get((scene, component_id), [])
                    ):
                        continue
                    key = (scene, chunk_id, history_id, str(component_id))
                    stats = accumulation_stats.setdefault(key, {"support": 0, "masks": set(), "frames": set()})
                    stats["support"] += int(support_count)
                    stats["masks"].add(mask_observation_id)
                    stats["frames"].add(int(frame_id))
            native_history_mask_component_accumulation_candidate_component_count = len(accumulation_stats)
            for key, stats in accumulation_stats.items():
                if _component_accumulation_pass(
                    stats,
                    min_support=int(native_history_mask_component_accumulation_min_support),
                    min_masks=int(native_history_mask_component_accumulation_min_masks),
                    min_frames=int(native_history_mask_component_accumulation_min_frames),
                ):
                    accumulation_eligible_components.add(key)
            native_history_mask_component_accumulation_eligible_component_count = len(
                accumulation_eligible_components
            )
            native_history_mask_component_accumulation_filtered_component_count = (
                native_history_mask_component_accumulation_candidate_component_count
                - native_history_mask_component_accumulation_eligible_component_count
            )
        for (scene, chunk_id, mask_observation_id), component_counts in sorted(support_by_mask.items()):
            frame_id, mask_id = support_mask_meta.get((scene, chunk_id, mask_observation_id), (0, 0))
            scored_native = [
                (
                    native_frame_masks_by_objectlet.get(history_id, Counter()).get((frame_id, mask_id), 0),
                    history_id,
                )
                for history_id in histories_by_scene.get(scene, [])
            ]
            scored_native = [(support, history_id) for support, history_id in scored_native if support > 0]
            if not scored_native:
                continue
            native_history_mask_candidate_count += 1
            mask_support_total = sum(component_counts.values())
            scored_native.sort(key=lambda item: (-item[0], item[1]))
            native_support, history_id = scored_native[0]
            second_native_support = scored_native[1][0] if len(scored_native) > 1 else 0
            total_native_support = sum(support for support, _history_id in scored_native)
            native_ratio = float(native_support / max(total_native_support, 1))
            native_mask_ratio = float(native_support / max(mask_support_total, 1))
            second_native_ratio = float(second_native_support / max(total_native_support, 1))
            native_dominance = (
                float(native_support / max(second_native_support, 1)) if second_native_support else None
            )
            if (
                native_support < int(native_history_mask_min_support)
                or native_ratio < float(native_history_mask_min_ratio)
                or native_mask_ratio < float(native_history_mask_min_mask_ratio)
                or (
                    native_dominance is not None
                    and native_dominance < float(native_history_mask_min_dominance)
                )
            ):
                continue
            other_seed_support = 0
            for component_id, support_count in component_counts.items():
                if any(
                    other_id != history_id
                    for other_id in comp_to_seed_history.get((scene, component_id), [])
                ):
                    other_seed_support += int(support_count)
            other_seed_ratio = float(other_seed_support / max(mask_support_total, 1))
            cannot_link_reason = ""
            if enable_native_history_mask_cannot_link_guard:
                other_seed_conflict = (
                    other_seed_support >= int(native_history_mask_other_seed_min_support)
                    and other_seed_ratio >= float(native_history_mask_other_seed_min_ratio)
                )
                second_native_conflict = (
                    second_native_support >= int(native_history_mask_second_native_min_support)
                    and second_native_ratio >= float(native_history_mask_second_native_min_ratio)
                )
                if other_seed_conflict:
                    cannot_link_reason = "other_anchor_seed_support"
                elif second_native_conflict:
                    cannot_link_reason = "second_history_native_support"
                if cannot_link_reason:
                    conflict_reject_count += 1
                    native_history_mask_cannot_link_reject_count += 1
                    update_rows.append(
                        {
                            "scene": scene,
                            "chunk_id": chunk_id,
                            "history_id": history_id,
                            "candidate_id": mask_observation_id,
                            "update_state": "cannot_link_reject",
                            "overlap_atom_count": None,
                            "overlap_atom_ratio": None,
                            "accepted_component_count": 0,
                            "candidate_component_count": len(component_counts),
                            "same_frame_exclusion_violation_rate": None,
                            "outside_all_related_masks_ratio_mean": None,
                            "update_source": "native_history_mask_projection",
                            "seed_support_ratio": None,
                            "seed_dominance_ratio": None,
                            "native_history_mask_support": int(native_support),
                            "native_history_mask_ratio": native_ratio,
                            "native_history_mask_dominance": native_dominance,
                            "native_history_mask_mask_ratio": native_mask_ratio,
                            "native_history_mask_other_seed_support": int(other_seed_support),
                            "native_history_mask_other_seed_ratio": other_seed_ratio,
                            "native_history_mask_second_native_support": int(second_native_support),
                            "native_history_mask_second_native_ratio": second_native_ratio,
                            "native_history_mask_cannot_link_reason": cannot_link_reason,
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic_labels": True,
                        }
                    )
                    continue
            semantic_cosine = None
            semantic_reject_reason = ""
            semantic_current_diag: dict[str, Any] = {}
            semantic_history_diag: dict[str, Any] = {}
            if native_history_mask_semantic_guard_active:
                source_mask_observation_id = str(histories[history_id].get("source_mask_observation_id") or "")
                current_feature, semantic_current_diag = _semantic_mask_feature(
                    mask_observation_id,
                    backend=str(native_history_mask_semantic_backend),
                    device=str(native_history_mask_semantic_device),
                    checkpoint=native_history_mask_semantic_checkpoint,
                    short_side=int(native_history_mask_semantic_short_side),
                    adapter_cache=semantic_adapter_cache,
                    feature_map_cache=semantic_feature_map_cache,
                    feature_cache=semantic_feature_cache,
                )
                history_feature, semantic_history_diag = _semantic_mask_feature(
                    source_mask_observation_id,
                    backend=str(native_history_mask_semantic_backend),
                    device=str(native_history_mask_semantic_device),
                    checkpoint=native_history_mask_semantic_checkpoint,
                    short_side=int(native_history_mask_semantic_short_side),
                    adapter_cache=semantic_adapter_cache,
                    feature_map_cache=semantic_feature_map_cache,
                    feature_cache=semantic_feature_cache,
                )
                native_history_mask_semantic_feature_attempt_count += 1
                if current_feature and history_feature:
                    native_history_mask_semantic_feature_success_count += 1
                    semantic_cosine = cosine(current_feature, history_feature)
                    if semantic_cosine < float(native_history_mask_semantic_min_cosine):
                        semantic_reject_reason = "semantic_drift"
                else:
                    missing_bits = [
                        str(semantic_current_diag.get("semantic_feature_missing_reason") or ""),
                        str(semantic_history_diag.get("semantic_feature_missing_reason") or ""),
                    ]
                    semantic_reject_reason = "semantic_unavailable:" + "|".join(bit for bit in missing_bits if bit)
                if semantic_reject_reason == "semantic_drift":
                    conflict_reject_count += 1
                    native_history_mask_semantic_reject_count += 1
                    update_rows.append(
                        {
                            "scene": scene,
                            "chunk_id": chunk_id,
                            "history_id": history_id,
                            "candidate_id": mask_observation_id,
                            "update_state": "semantic_drift_reject",
                            "overlap_atom_count": None,
                            "overlap_atom_ratio": None,
                            "accepted_component_count": 0,
                            "candidate_component_count": len(component_counts),
                            "same_frame_exclusion_violation_rate": None,
                            "outside_all_related_masks_ratio_mean": None,
                            "update_source": "native_history_mask_projection",
                            "seed_support_ratio": None,
                            "seed_dominance_ratio": None,
                            "native_history_mask_support": int(native_support),
                            "native_history_mask_ratio": native_ratio,
                            "native_history_mask_dominance": native_dominance,
                            "native_history_mask_mask_ratio": native_mask_ratio,
                            "native_history_mask_other_seed_support": int(other_seed_support),
                            "native_history_mask_other_seed_ratio": other_seed_ratio,
                            "native_history_mask_second_native_support": int(second_native_support),
                            "native_history_mask_second_native_ratio": second_native_ratio,
                            "native_history_mask_cannot_link_reason": cannot_link_reason,
                            "semantic_guard_backend": str(native_history_mask_semantic_backend),
                            "semantic_guard_min_cosine": float(native_history_mask_semantic_min_cosine),
                            "semantic_cosine_to_anchor_source": semantic_cosine,
                            "semantic_reject_reason": semantic_reject_reason,
                            "semantic_feature_available": bool(current_feature and history_feature),
                            "semantic_current_missing_reason": semantic_current_diag.get("semantic_feature_missing_reason"),
                            "semantic_history_missing_reason": semantic_history_diag.get("semantic_feature_missing_reason"),
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic_labels": True,
                        }
                    )
                    continue
            history = histories[history_id]
            accepted_components: set[str] = set()
            direct_component_support = (
                native_frame_mask_components_by_objectlet.get(history_id, {}).get((frame_id, mask_id), Counter())
                if enable_native_history_mask_component_gate
                else Counter()
            )
            direct_component_count = sum(
                1
                for component_id in component_counts
                if direct_component_support.get(str(component_id), 0) >= int(native_history_mask_component_min_support)
            )
            if enable_native_history_mask_component_gate:
                native_history_mask_component_gate_candidate_component_count += len(component_counts)
                native_history_mask_component_gate_direct_component_count += direct_component_count
                native_history_mask_component_gate_filtered_component_count += len(component_counts) - direct_component_count
            accumulation_row_eligible_count = 0
            accumulation_row_support_sum = 0
            if native_history_mask_component_accumulation_gate_active:
                for component_id in component_counts:
                    key = (scene, chunk_id, history_id, str(component_id))
                    if key in accumulation_eligible_components:
                        accumulation_row_eligible_count += 1
                        accumulation_row_support_sum += int(accumulation_stats.get(key, {}).get("support", 0))
            support_gate_row_pass_count = 0
            support_gate_row_support_sum = 0
            if native_history_mask_component_support_gate_active:
                native_history_mask_component_support_gate_candidate_component_count += len(component_counts)
                for component_id in component_counts:
                    component_meta = support_component_meta.get((scene, chunk_id, mask_observation_id), {}).get(
                        str(component_id)
                    )
                    if _component_support_gate_pass(
                        component_meta,
                        max_selected_rank=int(native_history_mask_component_max_selected_rank),
                        min_w_visible=float(native_history_mask_component_min_w_visible),
                        min_r_mask=float(native_history_mask_component_min_r_mask),
                        require_dominant=bool(native_history_mask_component_require_dominant),
                    ):
                        support_gate_row_pass_count += 1
                        support_gate_row_support_sum += int(component_meta.get("support_count", 0)) if component_meta else 0
                native_history_mask_component_support_gate_pass_component_count += support_gate_row_pass_count
                native_history_mask_component_support_gate_filtered_component_count += (
                    len(component_counts) - support_gate_row_pass_count
                )
            for component_id in component_counts:
                component_native_support = int(direct_component_support.get(str(component_id), 0))
                if (
                    enable_native_history_mask_component_gate
                    and component_native_support < int(native_history_mask_component_min_support)
                ):
                    continue
                if (
                    native_history_mask_component_accumulation_gate_active
                    and (scene, chunk_id, history_id, str(component_id)) not in accumulation_eligible_components
                ):
                    continue
                if native_history_mask_component_support_gate_active and not _component_support_gate_pass(
                    support_component_meta.get((scene, chunk_id, mask_observation_id), {}).get(str(component_id)),
                    max_selected_rank=int(native_history_mask_component_max_selected_rank),
                    min_w_visible=float(native_history_mask_component_min_w_visible),
                    min_r_mask=float(native_history_mask_component_min_r_mask),
                    require_dominant=bool(native_history_mask_component_require_dominant),
                ):
                    continue
                if component_id in history["history_components"]:
                    continue
                other_seed_histories = [
                    other_id for other_id in comp_to_seed_history.get((scene, component_id), []) if other_id != history_id
                ]
                if other_seed_histories:
                    continue
                history_gt = history.get("dominant_gt")
                component_label = _dominant(component_gt.get((scene, component_id), Counter()))
                if history_gt and component_label:
                    update_precision_total += 1
                    if component_label == history_gt:
                        update_precision_hits += 1
                history["history_components"].add(component_id)
                added_components_by_history[history_id].add(component_id)
                accepted_components.add(component_id)
                atom_id = component_to_atom.get((scene, component_id), "")
                if atom_id:
                    history["history_atoms"].add(atom_id)
            if accepted_components:
                confirmed_update_count += 1
                native_history_mask_accepted_count += 1
                history["chunks"].add(chunk_id)
            else:
                native_history_mask_duplicate_noop_count += 1
            native_history_mask_added_component_count += len(accepted_components)
            update_rows.append(
                {
                    "scene": scene,
                    "chunk_id": chunk_id,
                    "history_id": history_id,
                    "candidate_id": mask_observation_id,
                    "update_state": "confirmed_update" if accepted_components else "duplicate_noop",
                    "overlap_atom_count": None,
                    "overlap_atom_ratio": None,
                    "accepted_component_count": len(accepted_components),
                    "candidate_component_count": len(component_counts),
                    "same_frame_exclusion_violation_rate": None,
                    "outside_all_related_masks_ratio_mean": None,
                    "update_source": "native_history_mask_projection",
                    "seed_support_ratio": None,
                    "seed_dominance_ratio": None,
                    "native_history_mask_support": int(native_support),
                    "native_history_mask_ratio": native_ratio,
                    "native_history_mask_dominance": native_dominance,
                    "native_history_mask_mask_ratio": native_mask_ratio,
                    "native_history_mask_other_seed_support": int(other_seed_support),
                    "native_history_mask_other_seed_ratio": other_seed_ratio,
                    "native_history_mask_second_native_support": int(second_native_support),
                    "native_history_mask_second_native_ratio": second_native_ratio,
                    "native_history_mask_cannot_link_reason": cannot_link_reason,
                    "native_history_mask_component_gate_enabled": bool(enable_native_history_mask_component_gate),
                    "native_history_mask_component_min_support": int(native_history_mask_component_min_support),
                    "native_history_mask_direct_component_count": int(direct_component_count),
                    "native_history_mask_direct_component_support_sum": int(sum(direct_component_support.values())),
                    "native_history_mask_component_accumulation_gate_enabled": bool(
                        native_history_mask_component_accumulation_gate_active
                    ),
                    "native_history_mask_component_accumulation_min_support": int(
                        native_history_mask_component_accumulation_min_support
                    ),
                    "native_history_mask_component_accumulation_min_masks": int(
                        native_history_mask_component_accumulation_min_masks
                    ),
                    "native_history_mask_component_accumulation_min_frames": int(
                        native_history_mask_component_accumulation_min_frames
                    ),
                    "native_history_mask_accumulation_eligible_component_count": int(
                        accumulation_row_eligible_count
                    ),
                    "native_history_mask_accumulation_support_sum": int(accumulation_row_support_sum),
                    "native_history_mask_component_support_gate_enabled": bool(
                        native_history_mask_component_support_gate_active
                    ),
                    "native_history_mask_component_max_selected_rank": int(
                        native_history_mask_component_max_selected_rank
                    ),
                    "native_history_mask_component_min_w_visible": float(native_history_mask_component_min_w_visible),
                    "native_history_mask_component_min_r_mask": float(native_history_mask_component_min_r_mask),
                    "native_history_mask_component_require_dominant": bool(
                        native_history_mask_component_require_dominant
                    ),
                    "native_history_mask_support_gate_pass_component_count": int(support_gate_row_pass_count),
                    "native_history_mask_support_gate_support_sum": int(support_gate_row_support_sum),
                    "semantic_guard_backend": str(native_history_mask_semantic_backend)
                    if native_history_mask_semantic_guard_active
                    else None,
                    "semantic_guard_min_cosine": float(native_history_mask_semantic_min_cosine)
                    if native_history_mask_semantic_guard_active
                    else None,
                    "semantic_cosine_to_anchor_source": semantic_cosine,
                    "semantic_reject_reason": semantic_reject_reason,
                    "semantic_feature_available": None
                    if not native_history_mask_semantic_guard_active
                    else bool(semantic_current_diag.get("semantic_feature_available"))
                    and bool(semantic_history_diag.get("semantic_feature_available")),
                    "semantic_current_missing_reason": semantic_current_diag.get("semantic_feature_missing_reason"),
                    "semantic_history_missing_reason": semantic_history_diag.get("semantic_feature_missing_reason"),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )

    cosupport_native_gate_active = bool(
        enable_mask_cosupport and enable_cosupport_native_gate and native_carrier_rows_path is not None
    )

    cosupport_accepted_mask_count = 0
    cosupport_conflict_mask_count = 0
    cosupport_multihistory_seed_mask_count = 0
    cosupport_native_gate_reject_count = 0
    cosupport_confirmed_update_count = 0
    cosupport_partial_update_count = 0
    cosupport_added_component_count = 0
    cosupport_items = sorted(support_by_mask.items()) if enable_mask_cosupport else []
    for (scene, chunk_id, mask_observation_id), component_counts in cosupport_items:
        total_support = sum(component_counts.values())
        if total_support <= 0:
            continue
        seed_support: Counter[str] = Counter()
        for component_id, support_count in component_counts.items():
            for history_id in comp_to_seed_history.get((scene, component_id), []):
                seed_support[history_id] += int(support_count)
        if not seed_support:
            continue
        ranked = seed_support.most_common()
        history_id, top_seed_support = ranked[0]
        second_seed_support = ranked[1][1] if len(ranked) > 1 else 0
        seed_ratio = float(top_seed_support / max(total_support, 1))
        dominance_ratio = float(top_seed_support / max(second_seed_support, 1)) if second_seed_support else None
        if seed_ratio < float(cosupport_seed_ratio_min):
            continue
        if second_seed_support:
            cosupport_multihistory_seed_mask_count += 1
        if second_seed_support and dominance_ratio is not None and dominance_ratio < float(cosupport_dominance_ratio_min):
            conflict_reject_count += 1
            cosupport_conflict_mask_count += 1
            update_rows.append(
                {
                    "scene": scene,
                    "chunk_id": chunk_id,
                    "history_id": history_id,
                    "candidate_id": mask_observation_id,
                    "update_state": "conflict_reject",
                    "overlap_atom_count": int(top_seed_support),
                    "overlap_atom_ratio": seed_ratio,
                    "accepted_component_count": 0,
                    "candidate_component_count": len(component_counts),
                    "same_frame_exclusion_violation_rate": None,
                    "outside_all_related_masks_ratio_mean": None,
                    "update_source": "visible_mask_cosupport",
                    "seed_support_ratio": seed_ratio,
                    "seed_dominance_ratio": dominance_ratio,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
            continue

        native_support = None
        if cosupport_native_gate_active:
            frame_id, mask_id = support_mask_meta.get((scene, chunk_id, mask_observation_id), (0, 0))
            native_support = native_frame_masks_by_objectlet.get(history_id, Counter()).get((frame_id, mask_id), 0)
            if native_support < int(cosupport_native_min_support):
                cosupport_native_gate_reject_count += 1
                update_rows.append(
                    {
                        "scene": scene,
                        "chunk_id": chunk_id,
                        "history_id": history_id,
                        "candidate_id": mask_observation_id,
                        "update_state": "native_gate_reject",
                        "overlap_atom_count": int(top_seed_support),
                        "overlap_atom_ratio": seed_ratio,
                        "accepted_component_count": 0,
                        "candidate_component_count": len(component_counts),
                        "same_frame_exclusion_violation_rate": None,
                        "outside_all_related_masks_ratio_mean": None,
                        "update_source": "visible_mask_cosupport",
                        "seed_support_ratio": seed_ratio,
                        "seed_dominance_ratio": dominance_ratio,
                        "cosupport_native_history_mask_support": int(native_support),
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                    }
                )
                continue

        state = "confirmed_update" if seed_ratio >= 0.50 else "partial_update"
        if state == "confirmed_update":
            confirmed_update_count += 1
            cosupport_confirmed_update_count += 1
        else:
            partial_update_count += 1
            cosupport_partial_update_count += 1
        cosupport_accepted_mask_count += 1
        history = histories[history_id]
        accepted_components: set[str] = set()
        for component_id in component_counts:
            if component_id in history["history_components"]:
                continue
            other_seed_histories = [
                other_id for other_id in comp_to_seed_history.get((scene, component_id), []) if other_id != history_id
            ]
            if other_seed_histories:
                continue
            accepted_components.add(component_id)
            history_gt = history.get("dominant_gt")
            component_label = _dominant(component_gt.get((scene, component_id), Counter()))
            if history_gt and component_label:
                update_precision_total += 1
                if component_label == history_gt:
                    update_precision_hits += 1
            history["history_components"].add(component_id)
            added_components_by_history[history_id].add(component_id)
            atom_id = component_to_atom.get((scene, component_id), "")
            if atom_id:
                history["history_atoms"].add(atom_id)
        if accepted_components or top_seed_support > 0:
            history["chunks"].add(chunk_id)
        cosupport_added_component_count += len(accepted_components)
        update_rows.append(
            {
                "scene": scene,
                "chunk_id": chunk_id,
                "history_id": history_id,
                "candidate_id": mask_observation_id,
                "update_state": state,
                "overlap_atom_count": int(top_seed_support),
                "overlap_atom_ratio": seed_ratio,
                "accepted_component_count": len(accepted_components),
                "candidate_component_count": len(component_counts),
                "same_frame_exclusion_violation_rate": None,
                "outside_all_related_masks_ratio_mean": None,
                "update_source": "visible_mask_cosupport",
                "seed_support_ratio": seed_ratio,
                "seed_dominance_ratio": dominance_ratio,
                "cosupport_native_history_mask_support": None if native_support is None else int(native_support),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    for history in histories.values():
        for chunk_id in sorted(chunk for chunk in update_chunks if chunk.startswith(str(history["scene"]))):
            if not any(row["history_id"] == history["history_id"] and row["chunk_id"] == chunk_id for row in update_rows):
                update_rows.append(
                    {
                        "scene": history["scene"],
                        "chunk_id": chunk_id,
                        "history_id": history["history_id"],
                        "candidate_id": "",
                        "update_state": "occluded_or_absent",
                        "overlap_atom_count": 0,
                        "overlap_atom_ratio": 0.0,
                        "accepted_component_count": 0,
                        "candidate_component_count": 0,
                        "same_frame_exclusion_violation_rate": None,
                        "outside_all_related_masks_ratio_mean": None,
                        "update_source": "no_evidence",
                        "seed_support_ratio": None,
                        "seed_dominance_ratio": None,
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_diagnostic_labels": True,
                    }
                )

    anchor_assignment, anchor_duplicate_count = _build_assignment_map(histories, "anchor_components")
    history_assignment, history_duplicate_count = _build_assignment_map(histories, "history_components")
    anchor_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=anchor_assignment,
    )
    history_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=history_assignment,
    )
    shuffled_histories: dict[str, dict[str, Any]] = {}
    for history_id, history in histories.items():
        shuffled_histories[history_id] = {**history, "history_components": set(history["anchor_components"])}
    next_history = _next_history_by_scene(histories)
    for history_id, added_components in added_components_by_history.items():
        target_id = next_history.get(history_id)
        if not target_id:
            continue
        shuffled_histories[target_id]["history_components"].update(added_components)
    shuffled_assignment, shuffled_duplicate_count = _build_assignment_map(shuffled_histories, "history_components")
    shuffled_metrics = _metrics_from_support(
        _project(support_rows_path),
        support_variant=support_variant,
        scenes=scenes,
        component_to_history=shuffled_assignment,
    )
    temporal_spans = [float(len(history["chunks"])) for history in histories.values()]
    anchor_temporal_spans = [1.0 for _history in histories.values()]
    update_precision = float(update_precision_hits / max(update_precision_total, 1)) if update_precision_total else None
    duplicate_rate = float(history_duplicate_count / max(len(history_assignment), 1))
    conflict_values = [
        float(row["same_frame_exclusion_violation_rate"])
        for row in update_rows
        if row["update_state"] in {"confirmed_update", "partial_update", "conflict_reject"}
        and row["same_frame_exclusion_violation_rate"] not in (None, "")
    ]
    variant_parts: list[str] = []
    if native_carrier_rows_path is not None and enable_native_frame_mask_projection:
        variant_parts.append("U4_native_boundary_projection")
    if native_carrier_rows_path is not None and enable_native_uv_projection:
        variant_parts.append("U5_native_uv_bbox_projection")
    if native_history_mask_projection_active:
        variant_parts.append("U6_native_history_mask_projection")
        if native_history_mask_component_accumulation_gate_active:
            variant_parts.append("U6a_component_accumulation_gate")
        if native_history_mask_component_support_gate_active:
            variant_parts.append("U6b_component_support_gate")
    if enable_mask_cosupport:
        if cosupport_native_gate_active:
            variant_parts.append("U3g_native_carrier_gated_cosupport")
        else:
            variant_parts.append("U3_visible_mask_cosupport")
    if not variant_parts:
        variant_parts.append("U1_objectlet_atom_overlap")
    summary = {
        "phase": "v55_history_update",
        "created_at": utc_now(),
        "input_paths": {
            "chunk_role_rows_path": _rel(chunk_role_rows_path),
            "chunk_rows_path": _rel(chunk_rows_path),
            "anchor_birth_rows_path": _rel(anchor_birth_rows_path),
            "objectlet_rows_path": _rel(objectlet_rows_path),
            "local_summary_path": _rel(local_summary_path),
            "component_atom_rows_path": _rel(component_atom_rows_path),
            "support_rows_path": _rel(support_rows_path),
        },
        "objectlet_variant": best_variant,
        "objectlet_variant_override": objectlet_variant_override,
        "history_object_count": len(histories),
        "update_chunk_count": len(update_chunks),
        "history_evidence_roles": sorted(evidence_roles),
        "history_update_variant": "_plus_".join(variant_parts),
        "objectlet_atom_overlap_confirmed_update_count": objectlet_confirmed_update_count,
        "objectlet_atom_overlap_partial_update_count": objectlet_partial_update_count,
        "objectlet_atom_overlap_conflict_reject_count": objectlet_conflict_reject_count,
        "native_boundary_projection_enabled": native_carrier_rows_path is not None,
        "native_carrier_rows_path": None if native_carrier_rows_path is None else _rel(native_carrier_rows_path),
        "native_boundary_carrier_rows_available": native_boundary_available,
        "native_frame_mask_projection_enabled": bool(enable_native_frame_mask_projection and native_carrier_rows_path is not None),
        "native_boundary_min_support": int(native_boundary_min_support),
        "native_boundary_min_candidate_ratio": float(native_boundary_min_candidate_ratio),
        "native_boundary_min_jaccard": float(native_boundary_min_jaccard),
        "native_boundary_min_shared_frame_masks": int(native_boundary_min_shared_frame_masks),
        "native_boundary_candidate_count": native_boundary_candidate_count,
        "native_boundary_accepted_count": native_boundary_accepted_count,
        "native_boundary_added_component_count": native_boundary_added_component_count,
        "native_boundary_conflict_count": native_boundary_conflict_count,
        "native_uv_projection_enabled": bool(enable_native_uv_projection and native_carrier_rows_path is not None),
        "native_uv_min_support": int(native_uv_min_support),
        "native_uv_min_candidate_ratio": float(native_uv_min_candidate_ratio),
        "native_uv_min_jaccard": float(native_uv_min_jaccard),
        "native_uv_min_mean_iou": float(native_uv_min_mean_iou),
        "native_uv_max_center_dist": float(native_uv_max_center_dist),
        "native_uv_min_shared_frames": int(native_uv_min_shared_frames),
        "native_uv_candidate_count": native_uv_candidate_count,
        "native_uv_accepted_count": native_uv_accepted_count,
        "native_uv_added_component_count": native_uv_added_component_count,
        "native_uv_duplicate_noop_count": native_uv_duplicate_noop_count,
        "native_history_mask_projection_enabled": native_history_mask_projection_active,
        "native_history_mask_min_support": int(native_history_mask_min_support),
        "native_history_mask_min_ratio": float(native_history_mask_min_ratio),
        "native_history_mask_min_dominance": float(native_history_mask_min_dominance),
        "native_history_mask_min_mask_ratio": float(native_history_mask_min_mask_ratio),
        "native_history_mask_component_gate_enabled": bool(
            native_history_mask_projection_active and enable_native_history_mask_component_gate
        ),
        "native_history_mask_component_min_support": int(native_history_mask_component_min_support),
        "native_history_mask_component_gate_candidate_component_count": int(
            native_history_mask_component_gate_candidate_component_count
        ),
        "native_history_mask_component_gate_direct_component_count": int(
            native_history_mask_component_gate_direct_component_count
        ),
        "native_history_mask_component_gate_filtered_component_count": int(
            native_history_mask_component_gate_filtered_component_count
        ),
        "native_history_mask_component_accumulation_gate_enabled": bool(
            native_history_mask_component_accumulation_gate_active
        ),
        "native_history_mask_component_accumulation_min_support": int(
            native_history_mask_component_accumulation_min_support
        ),
        "native_history_mask_component_accumulation_min_masks": int(
            native_history_mask_component_accumulation_min_masks
        ),
        "native_history_mask_component_accumulation_min_frames": int(
            native_history_mask_component_accumulation_min_frames
        ),
        "native_history_mask_component_accumulation_candidate_component_count": int(
            native_history_mask_component_accumulation_candidate_component_count
        ),
        "native_history_mask_component_accumulation_eligible_component_count": int(
            native_history_mask_component_accumulation_eligible_component_count
        ),
        "native_history_mask_component_accumulation_filtered_component_count": int(
            native_history_mask_component_accumulation_filtered_component_count
        ),
        "native_history_mask_component_support_gate_enabled": bool(
            native_history_mask_component_support_gate_active
        ),
        "native_history_mask_component_max_selected_rank": int(native_history_mask_component_max_selected_rank),
        "native_history_mask_component_min_w_visible": float(native_history_mask_component_min_w_visible),
        "native_history_mask_component_min_r_mask": float(native_history_mask_component_min_r_mask),
        "native_history_mask_component_require_dominant": bool(
            native_history_mask_component_require_dominant
        ),
        "native_history_mask_component_support_gate_candidate_component_count": int(
            native_history_mask_component_support_gate_candidate_component_count
        ),
        "native_history_mask_component_support_gate_pass_component_count": int(
            native_history_mask_component_support_gate_pass_component_count
        ),
        "native_history_mask_component_support_gate_filtered_component_count": int(
            native_history_mask_component_support_gate_filtered_component_count
        ),
        "native_history_mask_cannot_link_guard_enabled": bool(
            native_history_mask_projection_active and enable_native_history_mask_cannot_link_guard
        ),
        "native_history_mask_other_seed_min_support": int(native_history_mask_other_seed_min_support),
        "native_history_mask_other_seed_min_ratio": float(native_history_mask_other_seed_min_ratio),
        "native_history_mask_second_native_min_support": int(native_history_mask_second_native_min_support),
        "native_history_mask_second_native_min_ratio": float(native_history_mask_second_native_min_ratio),
        "native_history_mask_semantic_guard_enabled": native_history_mask_semantic_guard_active,
        "native_history_mask_semantic_backend": str(native_history_mask_semantic_backend),
        "native_history_mask_semantic_min_cosine": float(native_history_mask_semantic_min_cosine),
        "native_history_mask_semantic_device": str(native_history_mask_semantic_device),
        "native_history_mask_semantic_short_side": int(native_history_mask_semantic_short_side),
        "native_history_mask_semantic_feature_attempt_count": native_history_mask_semantic_feature_attempt_count,
        "native_history_mask_semantic_feature_success_count": native_history_mask_semantic_feature_success_count,
        "native_history_mask_semantic_feature_success_rate": float(
            native_history_mask_semantic_feature_success_count
            / max(native_history_mask_semantic_feature_attempt_count, 1)
        )
        if native_history_mask_semantic_guard_active
        else None,
        "native_history_mask_candidate_count": native_history_mask_candidate_count,
        "native_history_mask_accepted_count": native_history_mask_accepted_count,
        "native_history_mask_added_component_count": native_history_mask_added_component_count,
        "native_history_mask_duplicate_noop_count": native_history_mask_duplicate_noop_count,
        "native_history_mask_cannot_link_reject_count": native_history_mask_cannot_link_reject_count,
        "native_history_mask_semantic_reject_count": native_history_mask_semantic_reject_count,
        "mask_cosupport_enabled": bool(enable_mask_cosupport),
        "cosupport_seed_ratio_min": float(cosupport_seed_ratio_min),
        "cosupport_dominance_ratio_min": float(cosupport_dominance_ratio_min),
        "cosupport_native_gate_enabled": cosupport_native_gate_active,
        "cosupport_native_min_support": int(cosupport_native_min_support),
        "cosupport_accepted_mask_count": cosupport_accepted_mask_count,
        "cosupport_conflict_mask_count": cosupport_conflict_mask_count,
        "cosupport_multihistory_seed_mask_count": cosupport_multihistory_seed_mask_count,
        "cosupport_native_gate_reject_count": cosupport_native_gate_reject_count,
        "cosupport_confirmed_update_count": cosupport_confirmed_update_count,
        "cosupport_partial_update_count": cosupport_partial_update_count,
        "cosupport_added_component_count": cosupport_added_component_count,
        "confirmed_update_count": confirmed_update_count,
        "partial_update_count": partial_update_count,
        "occluded_absent_count": sum(1 for row in update_rows if row["update_state"] == "occluded_or_absent"),
        "conflict_reject_count": conflict_reject_count,
        "drift_warning_count": 0,
        "update_precision_diagnostic": update_precision,
        "update_recall_proxy": float((confirmed_update_count + partial_update_count) / max(len(update_candidates), 1)),
        "anchor_only_temporal_span_mean": _mean(anchor_temporal_spans),
        "history_temporal_span_mean": _mean(temporal_spans),
        "anchor_only_ARI": anchor_metrics["ARI"],
        "anchor_only_purity": anchor_metrics["purity"],
        "anchor_only_completeness": anchor_metrics["completeness"],
        "history_ARI": history_metrics["ARI"],
        "history_purity": history_metrics["purity"],
        "history_completeness": history_metrics["completeness"],
        "duplicate_rate": duplicate_rate,
        "anchor_duplicate_component_count": anchor_duplicate_count,
        "history_duplicate_component_count": history_duplicate_count,
        "shuffled_duplicate_component_count": shuffled_duplicate_count,
        "duplicate_update_candidate_count": duplicate_component_updates,
        "conflict_rate": _mean(conflict_values),
        "id_switch_rate_diagnostic": None if update_precision is None else float(1.0 - update_precision),
        "shuffled_history_ARI": shuffled_metrics["ARI"],
        "shuffled_history_purity": shuffled_metrics["purity"],
        "shuffled_history_completeness": shuffled_metrics["completeness"],
        "real_minus_shuffled_ARI": float(history_metrics["ARI"] - shuffled_metrics["ARI"]),
        "real_minus_shuffled_ARI_status": "computed_by_scene_history_rotation_of_added_components",
        "real_minus_no_temporal_ARI": float(history_metrics["ARI"] - anchor_metrics["ARI"]),
        "real_minus_mask_only_ARI_static": None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "history_temporal_span_mean_ge_anchor_only_plus_0.30": (summary["history_temporal_span_mean"] or 0.0)
        >= (summary["anchor_only_temporal_span_mean"] or 0.0) + 0.30,
        "history_ARI_ge_anchor_only_minus_0.01": summary["history_ARI"] >= summary["anchor_only_ARI"] - 0.01,
        "history_purity_ge_anchor_only_minus_0.01": summary["history_purity"] >= summary["anchor_only_purity"] - 0.01,
        "history_completeness_ge_anchor_only": summary["history_completeness"] >= summary["anchor_only_completeness"],
        "update_precision_diagnostic_ge_0.85": (update_precision or 0.0) >= 0.85,
        "conflict_rate_le_anchor_only_plus_0.02": (summary["conflict_rate"] or 0.0) <= 0.02,
        "real_minus_shuffled_ARI_ge_0.30": summary["real_minus_shuffled_ARI"] >= 0.30,
        "real_minus_no_temporal_ARI_ge_0.25": summary["real_minus_no_temporal_ARI"] >= 0.25,
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    history_rows = [
        {
            "history_id": history_id,
            "scene": history["scene"],
            "anchor_chunk_id": history["anchor_chunk_id"],
            "history_chunk_count": len(history["chunks"]),
            "anchor_component_count": len(history["anchor_components"]),
            "history_component_count": len(history["history_components"]),
            "anchor_atom_count": len(history["anchor_atoms"]),
            "history_atom_count": len(history["history_atoms"]),
            "dominant_gt_diagnostic": history.get("dominant_gt"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for history_id, history in histories.items()
    ]
    history_component_rows = [
        {
            "history_id": history_id,
            "scene": history["scene"],
            "component_id": component_id,
            "is_anchor_component": component_id in history["anchor_components"],
            "is_added_component": component_id not in history["anchor_components"],
            "atom_id": component_to_atom.get((str(history["scene"]), component_id), ""),
            "component_dominant_gt_diagnostic": _dominant(
                component_gt.get((str(history["scene"]), component_id), Counter())
            ),
            "history_dominant_gt_diagnostic": history.get("dominant_gt"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        for history_id, history in histories.items()
        for component_id in sorted(history["history_components"])
    ]
    metric_rows = [
        {"row": "anchor_only", **anchor_metrics},
        {"row": "history_update", **history_metrics},
        {"row": "shuffled_update_control", **shuffled_metrics},
    ]
    return {
        "summary": summary,
        "history_update_rows": update_rows,
        "history_rows": history_rows,
        "history_component_rows": history_component_rows,
        "history_metric_rows": metric_rows,
    }


def _write_visualizations(out: Path, vis_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(out / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(out / "visualization_error.json"), "status": "matplotlib_unavailable"}]

    vis_root.mkdir(parents=True, exist_ok=True)
    state_counts = Counter(str(row["update_state"]) for row in payload["history_update_rows"])
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(state_counts)
    x = np.arange(len(labels))
    ax.bar(x, [state_counts[label] for label in labels])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("v55 history update states")
    path = vis_root / "history_update_timeline_all.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    manifest.append({"path": str(path), "kind": "history_update_timeline"})

    summary = payload["summary"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["ARI", "purity", "completeness"]
    anchor_vals = [summary["anchor_only_ARI"], summary["anchor_only_purity"], summary["anchor_only_completeness"]]
    history_vals = [summary["history_ARI"], summary["history_purity"], summary["history_completeness"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, anchor_vals, width=0.36, label="anchor_only")
    ax.bar(x + 0.18, history_vals, width=0.36, label="history_update")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.set_title("v55 anchor-only vs history metrics")
    path = vis_root / "update_evidence_panel_metrics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    manifest.append({"path": str(path), "kind": "update_evidence_panel"})
    return manifest


def write_v55_history_update(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v55_visualizations/history_update",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "history_update_summary.json", payload["summary"])
    write_csv(out / "history_update_rows.csv", payload["history_update_rows"])
    write_csv(out / "history_rows.csv", payload["history_rows"])
    write_csv(out / "history_component_rows.csv", payload["history_component_rows"])
    write_csv(out / "history_metric_rows.csv", payload["history_metric_rows"])
    manifest = _write_visualizations(out, vis, payload)
    write_json(out / "visualization_manifest.json", {"phase": "v55_history_update", "files": manifest})


__all__ = ["build_v55_history_update", "write_v55_history_update"]
