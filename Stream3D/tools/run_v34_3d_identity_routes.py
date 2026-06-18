from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v28_proposal_selection import (
    _float,
    _labels_from_selected,
    _load_gt_labels,
    _parse_core_tube_ids,
    _proposal_score,
    _rows_for_variant,
    _select_calibrated_ownership_expansion,
    _select_greedy,
    _shuffle_candidate_memberships,
)


LOCAL_GATE = {
    "local_ARI": 0.40,
    "local_purity": 0.85,
    "local_completeness": 0.50,
    "unknown_tube_ratio_max": 0.40,
    "scene0081_local_ARI": 0.20,
}

FEATURE_COLUMNS = [
    "appearance_variance",
    "boundary_contact_ratio",
    "confidence_mean",
    "core_tube_count",
    "eroded_interior_ratio",
    "image_gradient_boundary_score",
    "mask_area",
    "mask_distance_mean",
    "mask_distance_p10",
    "mask_distance_p50",
    "mask_distance_p90",
    "mask_temporal_repeat_score",
    "num_boundary_tubes",
    "num_core_tubes",
    "overlap_with_other_proposals",
    "proposal_area",
    "proposal_area_ratio",
    "region_area",
    "same_frame_cannot_link_rate",
    "tube_canonical_compactness",
    "tube_density",
    "tube_temporal_length_mean",
    "tube_xy_compactness",
    "visibility_mean",
    "visible_outside_negative_rate",
]

DIAGNOSTIC_POLICY = {
    "is_method_result": False,
    "is_diagnostic_only": True,
    "forbidden_for_method_table": True,
    "uses_gt_for_prediction": True,
    "uses_gt_for_diagnostic_labels": True,
    "uses_rgbd_for_prediction": False,
    "uses_pose_for_prediction": False,
    "uses_scannet_mesh_for_prediction": False,
    "uses_eval_sim3_for_prediction": False,
}

METHOD_POLICY = {
    "is_method_result": True,
    "is_diagnostic_only": False,
    "forbidden_for_method_table": False,
    "uses_gt_for_prediction": False,
    "uses_gt_for_diagnostic_labels": True,
    "uses_rgbd_for_prediction": False,
    "uses_pose_for_prediction": False,
    "uses_scannet_mesh_for_prediction": False,
    "uses_eval_sim3_for_prediction": False,
    "uses_d4rt_self_sim3": True,
    "geometry_field": "cached D4RT tube memberships and image-space proposal features",
}

POOL_TO_TYPES = {
    "O0_full_mask": {"R0_full_mask_region"},
    "O1_eroded": {"R1_boundary_eroded_interior", "R7_high_purity_core_region"},
    "O2_watershed": {"R2_distance_watershed_region"},
    "O3_d4rt_tube_seeded": {"R3_d4rt_tube_seeded_voronoi", "R5_d4rt_canonical_adjacency_split"},
    "O4_image_gradient": {"R4_image_gradient_split"},
    "O5_hybrid": set(),
}

SOURCE_LABELS = {
    "R0_full_mask_region": "M0_current_cropformer_full_mask",
    "R1_boundary_eroded_interior": "M1_boundary_eroded_interior",
    "R2_distance_watershed_region": "M1_watershed_region",
    "R3_d4rt_tube_seeded_voronoi": "M2_d4rt_tube_seeded",
    "R4_image_gradient_split": "M1_image_gradient_split",
    "R5_d4rt_canonical_adjacency_split": "M2_d4rt_canonical_adjacency",
    "R6_mask_overlap_consensus_region": "M0_mask_overlap_consensus",
    "R6_mask_overlap_consensus_union": "M0_mask_overlap_consensus",
    "R7_high_purity_core_region": "M1_high_purity_core",
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _all_row(rows: list[dict[str, Any]], variant: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("scene")) == "ALL" and (variant is None or str(row.get("variant")) == variant):
            return row
    return None


def _rows_by_scene(rows: list[dict[str, Any]], scenes: list[str]) -> dict[str, list[dict[str, Any]]]:
    out = {scene: [] for scene in scenes}
    for row in rows:
        scene = str(row.get("scene") or "")
        if scene in out:
            out[scene].append(row)
    return out


def _is_hybrid_type(proposal_type: str) -> bool:
    if proposal_type in {
        "R0_full_mask_region",
        "R1_boundary_eroded_interior",
        "R2_distance_watershed_region",
        "R3_d4rt_tube_seeded_voronoi",
        "R4_image_gradient_split",
        "R5_d4rt_canonical_adjacency_split",
        "R6_mask_overlap_consensus_region",
        "R6_mask_overlap_consensus_union",
        "R7_high_purity_core_region",
    }:
        return True
    return proposal_type.startswith(
        (
            "R8_temporal_tube_overlap_track_union",
            "R9_temporal_tube_overlap_track_consensus",
            "R10_temporal_tube_overlap_visible_negative_pruned",
            "R11_temporal_visible_negative_pruned_canonical_split",
            "R12_temporal_visible_negative_eroded_pruned",
        )
    )


def _source_for_type(proposal_type: str) -> str:
    if proposal_type in SOURCE_LABELS:
        return SOURCE_LABELS[proposal_type]
    if proposal_type.startswith(("R8_", "R9_", "R10_", "R11_", "R12_")):
        return "M2_d4rt_temporal_region"
    return f"unknown::{proposal_type}"


def _load_selection_rows(root: Path, label: str) -> list[dict[str, Any]]:
    return _read_json(root / f"{label}_selection_summary.json")


def _load_phase0_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _gate_status(row: dict[str, Any]) -> dict[str, Any]:
    ari = _safe_float(row.get("local_ARI"))
    purity = _safe_float(row.get("local_purity"))
    completeness = _safe_float(row.get("local_completeness"))
    unknown = _safe_float(row.get("unknown_tube_ratio"))
    scene0081 = _safe_float(row.get("scene0081_local_ARI"))
    checks = {
        "ari_pass": ari is not None and ari >= LOCAL_GATE["local_ARI"],
        "purity_pass": purity is not None and purity >= LOCAL_GATE["local_purity"],
        "completeness_pass": completeness is not None and completeness >= LOCAL_GATE["local_completeness"],
        "unknown_pass": unknown is not None and unknown <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": scene0081 is not None and scene0081 >= LOCAL_GATE["scene0081_local_ARI"],
    }
    return {
        **checks,
        "local_gate_pass": bool(all(checks.values())),
        "local_gate_thresholds": dict(LOCAL_GATE),
    }


def _score_kwargs() -> dict[str, float]:
    return {
        "boundary_weight": 0.20,
        "cannot_link_weight": 0.20,
        "small_proposal_bonus": 0.0,
        "visible_negative_weight": 0.14,
    }


def _select_variant(
    scene_rows: list[dict[str, Any]],
    *,
    variant: str,
    strict: bool,
    shuffled_proxy_seed: int,
    scene: str,
) -> list[dict[str, Any]]:
    candidates = _rows_for_variant(scene_rows, variant)
    if variant == "P8_shuffled_membership_control":
        candidates = _shuffle_candidate_memberships(candidates, seed=shuffled_proxy_seed, scene=scene)
    if variant in {"P0_full_mask", "P1_eroded_core", "P2_watershed", "P3_d4rt_seeded"}:
        return _select_greedy(
            candidates,
            min_new_tubes=3,
            max_overlap_ratio=0.05 if variant in {"P0_full_mask", "P1_eroded_core"} else 0.10,
            min_score=-999.0,
            score_kwargs=_score_kwargs(),
        )
    if variant == "P11_calibrated_ownership_expansion":
        return _select_calibrated_ownership_expansion(
            candidates,
            min_new_tubes=3,
            seed_max_overlap_ratio=0.10,
            seed_min_score=0.20,
            expand_min_score=-0.05,
            expand_min_overlap_ratio=0.25,
            expand_min_votes=20,
            expand_margin=1.25,
            max_expanded_core_ratio=1.25,
            score_kwargs=_score_kwargs(),
        )
    if variant in {"P9_no_temporal_control", "P10_mask_only_control"}:
        return _select_greedy(
            candidates,
            min_new_tubes=3,
            max_overlap_ratio=0.10 if strict else 0.20,
            min_score=0.20 if strict else -0.45,
            score_kwargs=_score_kwargs(),
        )
    return _select_greedy(
        candidates,
        min_new_tubes=3,
        max_overlap_ratio=0.35,
        min_score=0.20 if strict else -0.45,
        score_kwargs=_score_kwargs(),
    )


def _row_core_set(row: dict[str, Any]) -> set[int]:
    return {int(tid) for tid in _parse_core_tube_ids(row)}


def _row_risk(row: dict[str, Any]) -> float:
    return float(
        np.log1p(max(_float(row, "same_frame_cannot_link_rate"), 0.0))
        + 0.80 * np.log1p(max(_float(row, "visible_outside_negative_rate"), 0.0))
        + 0.50 * max(_float(row, "boundary_contact_ratio"), 0.0)
    )


def _allowed_child_type(proposal_type: str, prefixes: tuple[str, ...]) -> bool:
    return proposal_type.startswith(prefixes)


def _child_split_score(row: dict[str, Any], *, seed_size: int) -> float:
    core_size = max(len(_row_core_set(row)), 1)
    size_ratio = core_size / max(int(seed_size), 1)
    return float(
        _proposal_score(row, **_score_kwargs())
        - 0.55 * np.log1p(max(_float(row, "same_frame_cannot_link_rate"), 0.0))
        - 0.35 * np.log1p(max(_float(row, "visible_outside_negative_rate"), 0.0))
        - 0.10 * max(_float(row, "boundary_contact_ratio"), 0.0)
        - 0.10 * size_ratio
    )


def _candidate_children_for_seed(
    scene_rows: list[dict[str, Any]],
    seed_row: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    seed_core = _row_core_set(seed_row)
    if not seed_core:
        return []
    prefixes = tuple(str(v) for v in profile["child_type_prefixes"])
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for row in scene_rows:
        proposal_type = str(row.get("proposal_type") or "")
        if not _allowed_child_type(proposal_type, prefixes):
            continue
        core = _row_core_set(row)
        if len(core) < int(profile["child_min_tubes"]):
            continue
        inside = core & seed_core
        inside_ratio = len(inside) / max(len(core), 1)
        if inside_ratio < float(profile["child_min_inside_ratio"]):
            continue
        if len(inside) < int(profile["child_min_tubes"]):
            continue
        if len(inside) > len(seed_core) * float(profile["child_max_seed_ratio"]):
            continue
        if _float(row, "same_frame_cannot_link_rate") > float(profile["child_max_cannot_link"]):
            continue
        if _float(row, "visible_outside_negative_rate") > float(profile["child_max_visible_negative"]):
            continue
        item = dict(row)
        item["_core_tube_ids"] = tuple(sorted(inside))
        item["core_tube_ids"] = ";".join(str(tid) for tid in sorted(inside))
        item["num_core_tubes"] = int(len(inside))
        item["proposal_id"] = f"{row.get('proposal_id')}_{profile['name']}_clipped"
        split_score = _child_split_score(item, seed_size=len(seed_core))
        if split_score < float(profile["child_min_split_score"]):
            continue
        item["split_score"] = float(split_score)
        key = tuple(sorted(inside))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return sorted(out, key=lambda row: (float(row.get("split_score") or -999.0), len(_row_core_set(row))), reverse=True)


def _select_nonoverlap_children(children: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    owned: set[int] = set()
    for row in children:
        core = _row_core_set(row)
        if len(core) < int(profile["child_min_tubes"]):
            continue
        new = core - owned
        overlap_ratio = (len(core) - len(new)) / max(len(core), 1)
        if len(new) < int(profile["child_min_new_tubes"]):
            continue
        if overlap_ratio > float(profile["child_max_overlap_ratio"]):
            continue
        item = dict(row)
        item["_core_tube_ids"] = tuple(sorted(new))
        item["core_tube_ids"] = ";".join(str(tid) for tid in sorted(new))
        item["num_core_tubes"] = int(len(new))
        selected.append(item)
        owned.update(new)
    return selected


def _nested_split_selected(
    scene_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for idx, seed in enumerate(selected):
        seed_core = _row_core_set(seed)
        seed_risk = _row_risk(seed)
        should_try = len(seed_core) >= int(profile["seed_min_tubes"]) or seed_risk >= float(profile["seed_min_risk"])
        if not should_try:
            repaired.append(seed)
            continue
        children = _candidate_children_for_seed(scene_rows, seed, profile)
        chosen = _select_nonoverlap_children(children, profile)
        covered = {tid for row in chosen for tid in _row_core_set(row)}
        cover_ratio = len(covered & seed_core) / max(len(seed_core), 1)
        replaced = len(chosen) >= int(profile["replace_min_child_count"]) and cover_ratio >= float(profile["replace_min_cover_ratio"])
        audit_rows.append(
            {
                "profile": profile["name"],
                "seed_index": int(idx),
                "seed_proposal_id": seed.get("proposal_id"),
                "seed_type": seed.get("proposal_type"),
                "seed_tube_count": int(len(seed_core)),
                "seed_risk": float(seed_risk),
                "child_candidate_count": int(len(children)),
                "child_selected_count": int(len(chosen)),
                "child_cover_ratio": float(cover_ratio),
                "replaced": bool(replaced),
            }
        )
        if replaced:
            for rank, child in enumerate(chosen):
                item = dict(child)
                item["proposal_id"] = f"{seed.get('proposal_id')}_{profile['name']}_child{rank:03d}"
                item["proposal_type"] = f"{profile['name']}::{child.get('proposal_type')}"
                item["selection_transform"] = profile["name"]
                repaired.append(item)
        else:
            repaired.append(seed)
    return repaired, audit_rows


def _component_split_selected(
    scene_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for idx, seed in enumerate(selected):
        seed_core = _row_core_set(seed)
        seed_risk = _row_risk(seed)
        should_try = len(seed_core) >= int(profile["seed_min_tubes"]) or seed_risk >= float(profile["seed_min_risk"])
        if not should_try:
            repaired.append(seed)
            continue
        children = _candidate_children_for_seed(scene_rows, seed, profile)
        parent: dict[int, int] = {tid: tid for tid in seed_core}

        def find(tid: int) -> int:
            cur = int(tid)
            while parent[cur] != cur:
                parent[cur] = parent[parent[cur]]
                cur = parent[cur]
            return cur

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        used: set[int] = set()
        for child in children:
            core = sorted(_row_core_set(child))
            if len(core) < int(profile["child_min_tubes"]):
                continue
            if len(core) > int(profile["component_max_child_tubes"]):
                continue
            used.update(core)
            first = core[0]
            for tid in core[1:]:
                union(first, tid)
        comps: dict[int, set[int]] = defaultdict(set)
        for tid in used:
            comps[find(tid)].add(int(tid))
        component_rows: list[dict[str, Any]] = []
        for comp_idx, comp in enumerate(sorted(comps.values(), key=len, reverse=True)):
            if len(comp) < int(profile["component_min_tubes"]):
                continue
            item = dict(seed)
            item["_core_tube_ids"] = tuple(sorted(comp))
            item["core_tube_ids"] = ";".join(str(tid) for tid in sorted(comp))
            item["num_core_tubes"] = int(len(comp))
            item["proposal_id"] = f"{seed.get('proposal_id')}_{profile['name']}_comp{comp_idx:03d}"
            item["proposal_type"] = f"{profile['name']}::component"
            item["selection_transform"] = profile["name"]
            component_rows.append(item)
        covered = {tid for row in component_rows for tid in _row_core_set(row)}
        cover_ratio = len(covered & seed_core) / max(len(seed_core), 1)
        replaced = (
            len(component_rows) >= int(profile["replace_min_child_count"])
            and cover_ratio >= float(profile["replace_min_cover_ratio"])
        )
        audit_rows.append(
            {
                "profile": profile["name"],
                "seed_index": int(idx),
                "seed_proposal_id": seed.get("proposal_id"),
                "seed_type": seed.get("proposal_type"),
                "seed_tube_count": int(len(seed_core)),
                "seed_risk": float(seed_risk),
                "child_candidate_count": int(len(children)),
                "component_count": int(len(component_rows)),
                "component_cover_ratio": float(cover_ratio),
                "replaced": bool(replaced),
            }
        )
        if replaced:
            repaired.extend(component_rows)
        else:
            repaired.append(seed)
    return repaired, audit_rows


def _route_a_repair_profiles() -> list[dict[str, Any]]:
    common = {
        "seed_min_tubes": 80,
        "seed_min_risk": 0.35,
        "child_type_prefixes": (
            "R1_boundary_eroded_interior",
            "R2_distance_watershed_region",
            "R3_d4rt_tube_seeded_voronoi",
            "R4_image_gradient_split",
            "R5_d4rt_canonical_adjacency_split",
            "R7_high_purity_core_region",
            "R10_temporal_tube_overlap_visible_negative_pruned",
            "R12_temporal_visible_negative_eroded_pruned",
        ),
        "child_min_tubes": 4,
        "child_min_new_tubes": 4,
        "child_min_inside_ratio": 0.85,
        "child_max_seed_ratio": 0.75,
        "child_max_cannot_link": 0.55,
        "child_max_visible_negative": 0.45,
        "child_max_overlap_ratio": 0.12,
        "replace_min_child_count": 2,
    }
    return [
        {
            **common,
            "name": "A8_nested_split_conservative",
            "kind": "nested",
            "child_min_split_score": -0.05,
            "replace_min_cover_ratio": 0.45,
        },
        {
            **common,
            "name": "A9_nested_split_hard_purity",
            "kind": "nested",
            "child_max_cannot_link": 0.35,
            "child_max_visible_negative": 0.25,
            "child_min_split_score": 0.10,
            "replace_min_cover_ratio": 0.35,
        },
        {
            **common,
            "name": "A10_component_split_safe_children",
            "kind": "component",
            "child_max_cannot_link": 0.35,
            "child_max_visible_negative": 0.25,
            "child_min_split_score": 0.10,
            "replace_min_cover_ratio": 0.45,
            "component_min_tubes": 8,
            "component_max_child_tubes": 180,
        },
        {
            **common,
            "name": "A11_overlap_split_relaxed_risk",
            "kind": "nested",
            "child_max_cannot_link": 6.0,
            "child_max_visible_negative": 3.0,
            "child_max_seed_ratio": 0.55,
            "child_min_split_score": -1.00,
            "replace_min_cover_ratio": 0.30,
            "replace_min_child_count": 2,
        },
        {
            **common,
            "name": "A12_high_risk_trim_to_child",
            "kind": "nested",
            "child_max_cannot_link": 6.0,
            "child_max_visible_negative": 3.0,
            "child_max_seed_ratio": 0.95,
            "child_min_split_score": -0.45,
            "replace_min_cover_ratio": 0.55,
            "replace_min_child_count": 1,
        },
    ]


def _evaluate_selected_for_route(
    *,
    scenes: list[str],
    route: str,
    variant: str,
    selected_by_scene: dict[str, list[dict[str, Any]]],
    gt_by_scene: dict[str, dict[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        gt_labels = gt_by_scene.get(scene, {})
        labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
        selected = selected_by_scene.get(scene, [])
        labels_pred, unknown_count = _labels_from_selected(selected, labeled_tubes)
        metrics = _cluster_metrics(labels_pred, gt_labels)
        rows.append(
            {
                "scene": scene,
                "route": route,
                "variant": variant,
                "selected_proposal_count": int(len(selected)),
                "labeled_tube_count": int(len(labeled_tubes)),
                "unknown_tube_count": int(unknown_count),
                "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
                "local_ARI": metrics["ari"],
                "local_purity": metrics["purity"],
                "local_completeness": metrics["completeness"],
                "local_overmerge": metrics["overmerge"],
                "local_oversplit": metrics["oversplit"],
                **METHOD_POLICY,
            }
        )
    aggregate = {
        "scene": "ALL",
        "route": route,
        "variant": variant,
        "selected_proposal_count": int(sum(int(r["selected_proposal_count"]) for r in rows)),
        "labeled_tube_count": int(sum(int(r["labeled_tube_count"]) for r in rows)),
        "unknown_tube_count": int(sum(int(r["unknown_tube_count"]) for r in rows)),
        "unknown_tube_ratio": _mean([_safe_float(r["unknown_tube_ratio"]) for r in rows]),
        "local_ARI": _mean([_safe_float(r["local_ARI"]) for r in rows]),
        "local_purity": _mean([_safe_float(r["local_purity"]) for r in rows]),
        "local_completeness": _mean([_safe_float(r["local_completeness"]) for r in rows]),
        "local_overmerge": _mean([_safe_float(r["local_overmerge"]) for r in rows]),
        "local_oversplit": _mean([_safe_float(r["local_oversplit"]) for r in rows]),
        "scene0081_local_ARI": next((r["local_ARI"] for r in rows if str(r.get("scene")) == "scene0081_01"), None),
        **METHOD_POLICY,
    }
    return rows + [{**aggregate, **_gate_status(aggregate)}]


def _load_gt_by_scene(args: argparse.Namespace, scenes: list[str]) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for scene in scenes:
        cache_scene = Path(args.cache_root) / scene
        if not cache_scene.exists():
            continue
        out[scene] = _load_gt_labels(
            Path(args.cache_root),
            scene,
            int(args.max_tubes_per_window),
            int(args.image_width),
            int(args.image_height),
        )
    return out


def _materialize_object_outputs(
    *,
    output_dir: Path,
    rows_by_scene: dict[str, list[dict[str, Any]]],
    gt_by_scene: dict[str, dict[int, int]],
    variant: str,
    strict: bool,
    label: str,
    shuffled_proxy_seed: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_metric_rows: list[dict[str, Any]] = []
    for scene, scene_rows in rows_by_scene.items():
        gt_labels = gt_by_scene.get(scene, {})
        labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
        selected = _select_variant(
            scene_rows,
            variant=variant,
            strict=strict,
            shuffled_proxy_seed=shuffled_proxy_seed,
            scene=scene,
        )
        labels_pred, unknown_count = _labels_from_selected(selected, labeled_tubes)
        metrics = _cluster_metrics(labels_pred, gt_labels)

        assignments: dict[int, int] = {}
        objects: list[dict[str, Any]] = []
        for object_id, row in enumerate(selected):
            tube_ids = []
            for tid in _parse_core_tube_ids(row):
                if int(tid) not in assignments:
                    assignments[int(tid)] = int(object_id)
                    tube_ids.append(int(tid))
            objects.append(
                {
                    "object_id": int(object_id),
                    "proposal_id": row.get("proposal_id"),
                    "proposal_type": row.get("proposal_type"),
                    "source_route": label,
                    "tube_ids": tube_ids,
                    "tube_count": int(len(tube_ids)),
                    "selection_score": _proposal_score(row, **_score_kwargs()),
                    "method_policy": dict(METHOD_POLICY),
                }
            )
        unknown_tubes = [int(tid) for tid in labeled_tubes if int(tid) not in assignments]
        assignment_rows = [
            {
                "scene": scene,
                "tube_id": int(tid),
                "object_id": assignments.get(int(tid), ""),
                "assignment_status": "assigned" if int(tid) in assignments else "unknown",
                "source_route": label,
                "variant": variant,
            }
            for tid in labeled_tubes
        ]
        scene_dir = output_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            scene_dir / "objects.json",
            {
                "scene": scene,
                "route_label": label,
                "variant": variant,
                "objects": objects,
                "unknown_tubes": unknown_tubes,
                "method_policy": dict(METHOD_POLICY),
            },
        )
        _write_csv(scene_dir / "tube_assignment.csv", assignment_rows)
        metric_row = {
            "scene": scene,
            "variant": variant,
            "route_label": label,
            "selected_proposal_count": int(len(selected)),
            "labeled_tube_count": int(len(labeled_tubes)),
            "unknown_tube_count": int(unknown_count),
            "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
            "local_ARI": metrics["ari"],
            "local_purity": metrics["purity"],
            "local_completeness": metrics["completeness"],
            "local_overmerge": metrics["overmerge"],
            "local_oversplit": metrics["oversplit"],
            **METHOD_POLICY,
        }
        _write_json(scene_dir / "metrics_diagnostic.json", metric_row)
        _write_json(
            scene_dir / "visualization_manifest.json",
            {
                "scene": scene,
                "status": "metadata_only",
                "reason": "v34 route runner materializes object/tube ownership tables; no rendered images are generated here.",
                "objects_json": str(scene_dir / "objects.json"),
                "tube_assignment_csv": str(scene_dir / "tube_assignment.csv"),
            },
        )
        scene_metric_rows.append(metric_row)

    aggregate = {
        "scene": "ALL",
        "variant": variant,
        "route_label": label,
        "selected_proposal_count": int(sum(int(r["selected_proposal_count"]) for r in scene_metric_rows)),
        "labeled_tube_count": int(sum(int(r["labeled_tube_count"]) for r in scene_metric_rows)),
        "unknown_tube_count": int(sum(int(r["unknown_tube_count"]) for r in scene_metric_rows)),
        "unknown_tube_ratio": _mean([_safe_float(r["unknown_tube_ratio"]) for r in scene_metric_rows]),
        "local_ARI": _mean([_safe_float(r["local_ARI"]) for r in scene_metric_rows]),
        "local_purity": _mean([_safe_float(r["local_purity"]) for r in scene_metric_rows]),
        "local_completeness": _mean([_safe_float(r["local_completeness"]) for r in scene_metric_rows]),
        "local_overmerge": _mean([_safe_float(r["local_overmerge"]) for r in scene_metric_rows]),
        "local_oversplit": _mean([_safe_float(r["local_oversplit"]) for r in scene_metric_rows]),
        "scene0081_local_ARI": next(
            (r["local_ARI"] for r in scene_metric_rows if str(r.get("scene")) == "scene0081_01"),
            None,
        ),
        **METHOD_POLICY,
    }
    rows = scene_metric_rows + [aggregate]
    _write_csv(output_dir / f"{label}_object_metrics.csv", rows)
    _write_json(output_dir / f"{label}_object_metrics.json", rows)
    return rows


def run_phase2(args: argparse.Namespace, proposal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "v34_phase2_mask_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_rows = _read_csv_rows(Path(args.proposal_root) / f"{args.proposal_label}_oracle_summary.csv")

    source_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposal_rows:
        proposal_type = str(row.get("proposal_type") or "")
        grouped[_source_for_type(proposal_type)].append(row)

    for source, rows in sorted(grouped.items()):
        gt_best: dict[tuple[str, int], float] = {}
        purity_vals: list[float] = []
        iou_vals: list[float] = []
        mixed_count = 0
        for row in rows:
            purity = _safe_float(row.get("proposal_purity"))
            iou = _safe_float(row.get("proposal_best_IoU"))
            best_gt = row.get("proposal_best_GT")
            scene = str(row.get("scene") or "")
            if purity is not None:
                purity_vals.append(purity)
                if purity < 0.85:
                    mixed_count += 1
            if iou is not None:
                iou_vals.append(iou)
                try:
                    gt_id = int(best_gt)
                except (TypeError, ValueError):
                    gt_id = 0
                if gt_id > 0:
                    key = (scene, gt_id)
                    gt_best[key] = max(gt_best.get(key, 0.0), float(iou))
        source_rows.append(
            {
                "source": source,
                "proposal_count": int(len(rows)),
                "scene_count": int(len({str(r.get("scene")) for r in rows})),
                "mean_proposal_purity_diagnostic": _mean(purity_vals),
                "mixed_proposal_rate_purity_lt_085": float(mixed_count / max(len(rows), 1)),
                "mean_best_iou_diagnostic": _mean(iou_vals),
                "covered_gt_proxy_count": int(len(gt_best)),
                "gt_coverage_proxy_iou_ge_010": float(
                    sum(1 for value in gt_best.values() if value >= 0.10) / max(len(gt_best), 1)
                )
                if gt_best
                else None,
                "gt_coverage_proxy_iou_ge_025": float(
                    sum(1 for value in gt_best.values() if value >= 0.25) / max(len(gt_best), 1)
                )
                if gt_best
                else None,
                **DIAGNOSTIC_POLICY,
            }
        )

    oracle_all_rows = [row for row in oracle_rows if str(row.get("scene")) == "ALL"]
    external_mask_sources = []
    for name, patterns in {
        "SAM": ["*sam*.pth", "*sam*.pt", "*sam*.ckpt"],
        "SAM2": ["*sam2*.pth", "*sam2*.pt", "*sam2*.ckpt"],
        "SAM3_or_EfficientSAM3": ["*sam3*.pth", "*sam3*.pt", "*efficient*sam*.pth", "*efficient*sam*.pt"],
        "HQ_SAM": ["*hq*sam*.pth", "*hq*sam*.pt"],
        "MaskCut_DINO": ["*maskcut*.pth", "*dino*.pth", "*dinov2*.pth"],
    }.items():
        matches: list[str] = []
        for root in [Path("."), Path("checkpoints"), Path("weights"), Path("pretrained"), Path("../checkpoints")]:
            if root.exists():
                for pattern in patterns:
                    matches.extend(str(p) for p in root.glob(pattern))
        external_mask_sources.append(
            {
                "source": name,
                "available_checkpoint_count": int(len(sorted(set(matches)))),
                "available_checkpoints": sorted(set(matches))[:20],
                "used_in_v34_method": False,
                "not_used_reason": "no local integration/run artifact found in v34 route runner",
            }
        )

    _write_csv(out_dir / "mask_source_audit.csv", source_rows)
    _write_json(
        out_dir / "mask_source_audit.json",
        {
            "mask_source_rows": source_rows,
            "oracle_summary_all_rows": oracle_all_rows,
            "external_mask_source_scan": external_mask_sources,
            "policy": dict(DIAGNOSTIC_POLICY),
        },
    )
    return {"source_rows": source_rows, "oracle_all_rows": oracle_all_rows}


def run_route_a(
    args: argparse.Namespace,
    *,
    proposal_rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "v34_routeA_region_first"
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_rows = _load_selection_rows(Path(args.strict_selection_root), args.strict_selection_label)
    p11_rows = _load_selection_rows(Path(args.p11_selection_root), args.p11_selection_label)
    shuffle_rows = _load_selection_rows(Path(args.shuffle_selection_root), args.shuffle_selection_label)
    strict_map = {str(row.get("variant")): row for row in strict_rows if str(row.get("scene")) == "ALL"}
    p11_map = {str(row.get("variant")): row for row in p11_rows if str(row.get("scene")) == "ALL"}
    shuffle_map = {str(row.get("variant")): row for row in shuffle_rows if str(row.get("scene")) == "ALL"}

    route_rows: list[dict[str, Any]] = []
    mapping = [
        ("A0_region_only_full_mask", strict_map.get("P0_full_mask")),
        ("A0_region_only_eroded", strict_map.get("P1_eroded_core")),
        ("A1_region_plus_d4rt_strict", strict_map.get("P4_greedy_set_packing")),
        ("A3_boundary_cannot_link_strict", strict_map.get("P4_greedy_set_packing")),
        ("A4_ownership_expansion", p11_map.get("P11_calibrated_ownership_expansion")),
        ("A5_shuffled_d4rt_control_full_pool", shuffle_map.get("P4_greedy_set_packing")),
        ("A6_mask_only_control", strict_map.get("P10_mask_only_control")),
        ("A7_no_temporal_control", strict_map.get("P9_no_temporal_control")),
    ]
    for route, row in mapping:
        if row is None:
            route_rows.append({"route": route, "status": "missing_source_row"})
            continue
        gate = _gate_status(row)
        route_rows.append(
            {
                "route": route,
                **{k: row.get(k) for k in row.keys()},
                **gate,
                **METHOD_POLICY,
                "source_selection_root": (
                    str(args.p11_selection_root)
                    if route == "A4_ownership_expansion"
                    else str(args.shuffle_selection_root)
                    if route == "A5_shuffled_d4rt_control_full_pool"
                    else str(args.strict_selection_root)
                ),
            }
        )

    rows_by_scene = _rows_by_scene(proposal_rows, scenes)
    object_rows = _materialize_object_outputs(
        output_dir=out_dir / "object_outputs_A4_ownership_expansion",
        rows_by_scene=rows_by_scene,
        gt_by_scene=gt_by_scene,
        variant="P11_calibrated_ownership_expansion",
        strict=False,
        label="A4_ownership_expansion",
        shuffled_proxy_seed=int(args.shuffle_control_seed),
    )
    base_selected_by_scene = {
        scene: _select_variant(
            rows,
            variant="P11_calibrated_ownership_expansion",
            strict=False,
            shuffled_proxy_seed=int(args.shuffle_control_seed),
            scene=scene,
        )
        for scene, rows in rows_by_scene.items()
    }
    repair_metric_rows: list[dict[str, Any]] = []
    repair_audit_rows: list[dict[str, Any]] = []
    repair_selected_rows: list[dict[str, Any]] = []
    for profile in _route_a_repair_profiles():
        repaired_by_scene: dict[str, list[dict[str, Any]]] = {}
        for scene, scene_rows in rows_by_scene.items():
            selected = base_selected_by_scene.get(scene, [])
            if profile["kind"] == "component":
                repaired, audit = _component_split_selected(scene_rows, selected, profile)
            else:
                repaired, audit = _nested_split_selected(scene_rows, selected, profile)
            repaired_by_scene[scene] = repaired
            for audit_row in audit:
                repair_audit_rows.append({"scene": scene, **audit_row})
            for rank, row in enumerate(repaired):
                repair_selected_rows.append(
                    {
                        "scene": scene,
                        "profile": profile["name"],
                        "rank": int(rank),
                        "proposal_id": row.get("proposal_id"),
                        "proposal_type": row.get("proposal_type"),
                        "selection_transform": row.get("selection_transform", ""),
                        "num_core_tubes": int(row.get("num_core_tubes") or len(_row_core_set(row))),
                        "split_score": row.get("split_score", ""),
                    }
                )
        profile_rows = _evaluate_selected_for_route(
            scenes=scenes,
            route=str(profile["name"]),
            variant=str(profile["name"]),
            selected_by_scene=repaired_by_scene,
            gt_by_scene=gt_by_scene,
        )
        repair_metric_rows.extend(profile_rows)
        aggregate = next(row for row in profile_rows if str(row.get("scene")) == "ALL")
        route_rows.append({**aggregate, "source_selection_root": str(args.p11_selection_root)})

    _write_csv(out_dir / "routeA_repair_metrics.csv", repair_metric_rows)
    _write_json(
        out_dir / "routeA_repair_metrics.json",
        {
            "profiles": _route_a_repair_profiles(),
            "repair_metric_rows": repair_metric_rows,
            "policy": dict(METHOD_POLICY),
            "note": "A8-A10 are training-free post-A4 structural repairs using proposal features only; GT is used only for evaluation.",
        },
    )
    _write_csv(out_dir / "routeA_repair_audit.csv", repair_audit_rows)
    _write_csv(out_dir / "routeA_repair_selected_proposals.csv", repair_selected_rows)
    _write_csv(out_dir / "routeA_summary.csv", route_rows)
    _write_json(
        out_dir / "routeA_summary.json",
        {
            "routes": route_rows,
            "object_output_metrics": object_rows,
            "repair_metric_rows": repair_metric_rows,
            "repair_profiles": _route_a_repair_profiles(),
            "policy": dict(METHOD_POLICY),
            "source_note": "Route A reuses v28 cached proposal rows and selection summaries, then materializes v34 object/tube tables for A4.",
        },
    )
    return {"route_rows": route_rows, "object_rows": object_rows}


def run_route_b(args: argparse.Namespace, proposal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "v34_routeB_visual_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_columns = ["appearance_variance", "image_gradient_boundary_score", "confidence_mean", "visibility_mean"]
    feature_rows = []
    for col in visual_columns:
        vals = [_safe_float(row.get(col)) for row in proposal_rows]
        feature_rows.append(
            {
                "feature": col,
                "count": int(sum(v is not None for v in vals)),
                "mean": _mean(vals),
                "min": min([float(v) for v in vals if v is not None], default=None),
                "max": max([float(v) for v in vals if v is not None], default=None),
            }
        )

    checkpoint_patterns = ["*clip*.pt", "*clip*.pth", "*dinov2*.pt", "*dinov2*.pth", "*siglip*.pt", "*siglip*.pth"]
    matches: list[str] = []
    for root in [Path("."), Path("checkpoints"), Path("weights"), Path("pretrained"), Path("../checkpoints")]:
        if root.exists():
            for pattern in checkpoint_patterns:
                matches.extend(str(p) for p in root.glob(pattern))

    status = {
        "route": "B_frozen_visual_embedding_graph",
        "status": "not_run_full_embedding_graph",
        "embedding_checkpoint_count": int(len(sorted(set(matches)))),
        "embedding_checkpoints": sorted(set(matches))[:20],
        "not_run_reason": "No local frozen embedding extraction/checkpoint artifact was found by the route runner. Existing proposal rows only contain lightweight RGB/gradient proxy features.",
        "proxy_feature_audit_written": True,
        "recommended_next_action": "If Route D says appearance/boundary features are decisive, integrate a frozen embedding cache and re-run Route B.",
        **METHOD_POLICY,
    }
    _write_csv(out_dir / "proxy_visual_feature_audit.csv", feature_rows)
    _write_json(out_dir / "routeB_status.json", {"status": status, "proxy_visual_feature_audit": feature_rows})
    return {"status": status, "feature_rows": feature_rows}


def run_route_c(args: argparse.Namespace, phase2: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "v34_routeC_mask_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_rows = _load_selection_rows(Path(args.strict_selection_root), args.strict_selection_label)
    strict_map = {str(row.get("variant")): row for row in strict_rows if str(row.get("scene")) == "ALL"}
    rows = [
        {
            "route": "C0_current_cropformer_full_mask",
            **strict_map.get("P0_full_mask", {}),
            **_gate_status(strict_map.get("P0_full_mask", {})),
            **METHOD_POLICY,
        },
        {
            "route": "C1_eroded_boundary_source",
            **strict_map.get("P1_eroded_core", {}),
            **_gate_status(strict_map.get("P1_eroded_core", {})),
            **METHOD_POLICY,
        },
        {
            "route": "C2_watershed_source",
            **strict_map.get("P2_watershed", {}),
            **_gate_status(strict_map.get("P2_watershed", {})),
            **METHOD_POLICY,
        },
        {
            "route": "C3_d4rt_seeded_source",
            **strict_map.get("P3_d4rt_seeded", {}),
            **_gate_status(strict_map.get("P3_d4rt_seeded", {})),
            **METHOD_POLICY,
        },
    ]
    _write_csv(out_dir / "routeC_summary.csv", rows)
    _write_json(
        out_dir / "routeC_summary.json",
        {
            "routes": rows,
            "phase2_source_rows": phase2.get("source_rows", []),
            "policy": dict(METHOD_POLICY),
            "note": "Route C audit is limited to available cached v28 sources; no new external mask backbone was run.",
        },
    )
    return {"route_rows": rows}


def _proposal_target(row: dict[str, Any]) -> int:
    purity = _safe_float(row.get("proposal_purity"), 0.0) or 0.0
    completeness = _safe_float(row.get("proposal_completeness"), 0.0) or 0.0
    best_iou = _safe_float(row.get("proposal_best_IoU"), 0.0) or 0.0
    return int(purity >= 0.85 and completeness >= 0.25 and best_iou >= 0.25)


def _proposal_target_relaxed(row: dict[str, Any]) -> int:
    purity = _safe_float(row.get("proposal_purity"), 0.0) or 0.0
    completeness = _safe_float(row.get("proposal_completeness"), 0.0) or 0.0
    best_iou = _safe_float(row.get("proposal_best_IoU"), 0.0) or 0.0
    return int(purity >= 0.80 and completeness >= 0.20 and best_iou >= 0.20)


def _feature_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    data = []
    for row in rows:
        data.append([_safe_float(row.get(col), 0.0) or 0.0 for col in columns])
    return np.asarray(data, dtype=np.float64)


def _calibrate_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_t = 0.50
    best_f1 = -1.0
    for t in np.quantile(scores, np.linspace(0.05, 0.95, 19)).tolist():
        pred = scores >= float(t)
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        denom = (2.0 * tp + fp + fn)
        f1 = (2.0 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _select_learned(rows: list[dict[str, Any]], scores: np.ndarray, *, threshold: float) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row, score in zip(rows, scores.tolist()):
        item = dict(row)
        item["learned_score"] = float(score)
        enriched.append(item)
    selected: list[dict[str, Any]] = []
    owned: set[int] = set()
    for row in sorted(enriched, key=lambda r: (float(r.get("learned_score") or 0.0), len(_parse_core_tube_ids(r))), reverse=True):
        if float(row.get("learned_score") or 0.0) < float(threshold):
            continue
        core = set(_parse_core_tube_ids(row))
        if len(core) < 3:
            continue
        new = core - owned
        overlap = (len(core) - len(new)) / max(len(core), 1)
        if len(new) < 3 or overlap > 0.35:
            continue
        selected.append(row)
        owned.update(core)
    return selected


def _identity_calibrated_threshold(
    *,
    train_rows_by_scene: dict[str, list[dict[str, Any]]],
    train_scores_by_scene: dict[str, np.ndarray],
    gt_by_scene: dict[str, dict[int, int]],
) -> tuple[float, dict[str, Any]]:
    all_scores = np.concatenate([scores for scores in train_scores_by_scene.values() if len(scores) > 0])
    if all_scores.size == 0:
        return 0.50, {"reason": "no_train_scores"}
    thresholds = sorted(set(float(v) for v in np.quantile(all_scores, np.linspace(0.05, 0.95, 19)).tolist()))
    best_threshold = thresholds[0]
    best_tuple: tuple[float, float, float, float, float] | None = None
    best_metrics: dict[str, Any] = {}
    train_scenes = sorted(train_rows_by_scene)
    for threshold in thresholds:
        scene_rows: list[dict[str, Any]] = []
        for scene in train_scenes:
            rows = train_rows_by_scene[scene]
            scores = train_scores_by_scene[scene]
            selected = _select_learned(rows, scores, threshold=float(threshold))
            gt_labels = gt_by_scene.get(scene, {})
            labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
            labels_pred, unknown_count = _labels_from_selected(selected, labeled_tubes)
            metrics = _cluster_metrics(labels_pred, gt_labels)
            scene_rows.append(
                {
                    "scene": scene,
                    "local_ARI": metrics["ari"],
                    "local_purity": metrics["purity"],
                    "local_completeness": metrics["completeness"],
                    "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
                }
            )
        aggregate = {
            "local_ARI": _mean([_safe_float(row["local_ARI"]) for row in scene_rows]),
            "local_purity": _mean([_safe_float(row["local_purity"]) for row in scene_rows]),
            "local_completeness": _mean([_safe_float(row["local_completeness"]) for row in scene_rows]),
            "unknown_tube_ratio": _mean([_safe_float(row["unknown_tube_ratio"]) for row in scene_rows]),
        }
        ari = _safe_float(aggregate["local_ARI"], 0.0) or 0.0
        purity = _safe_float(aggregate["local_purity"], 0.0) or 0.0
        completeness = _safe_float(aggregate["local_completeness"], 0.0) or 0.0
        unknown = _safe_float(aggregate["unknown_tube_ratio"], 1.0) or 1.0
        gateish = (
            min(ari / LOCAL_GATE["local_ARI"], 1.0)
            + min(purity / LOCAL_GATE["local_purity"], 1.0)
            + min(completeness / LOCAL_GATE["local_completeness"], 1.0)
            + min(max(LOCAL_GATE["unknown_tube_ratio_max"] - unknown, 0.0) / LOCAL_GATE["unknown_tube_ratio_max"], 1.0)
        )
        candidate_tuple = (float(gateish), float(ari), float(purity), float(completeness), float(-unknown))
        if best_tuple is None or candidate_tuple > best_tuple:
            best_tuple = candidate_tuple
            best_threshold = float(threshold)
            best_metrics = dict(aggregate)
    return best_threshold, {"threshold": best_threshold, "train_identity_metrics": best_metrics, "selection_score_tuple": best_tuple}


def _select_learned_children_for_seed(
    *,
    scene_rows: list[dict[str, Any]],
    seed: dict[str, Any],
    score_by_proposal_id: dict[str, float],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    seed_core = _row_core_set(seed)
    if not seed_core:
        return []
    candidates: list[dict[str, Any]] = []
    for row in _rows_for_variant(scene_rows, "P4_greedy_set_packing"):
        proposal_id = str(row.get("proposal_id"))
        learned_score = float(score_by_proposal_id.get(proposal_id, -1.0))
        if learned_score < float(params["threshold"]):
            continue
        core = _row_core_set(row)
        if not core:
            continue
        inside = core & seed_core
        if len(inside) < int(params["child_min_tubes"]):
            continue
        if len(inside) / max(len(core), 1) < float(params["child_min_inside_ratio"]):
            continue
        if len(inside) > len(seed_core) * float(params["child_max_seed_ratio"]):
            continue
        item = dict(row)
        item["_core_tube_ids"] = tuple(sorted(inside))
        item["core_tube_ids"] = ";".join(str(tid) for tid in sorted(inside))
        item["num_core_tubes"] = int(len(inside))
        item["proposal_id"] = f"{seed.get('proposal_id')}_D7_child_{proposal_id}"
        item["proposal_type"] = f"D7_learned_a4_child::{row.get('proposal_type')}"
        item["learned_score"] = float(learned_score)
        item["split_score"] = float(learned_score - 0.05 * (len(inside) / max(len(seed_core), 1)) - 0.05 * _row_risk(row))
        candidates.append(item)
    selected: list[dict[str, Any]] = []
    owned: set[int] = set()
    for row in sorted(candidates, key=lambda r: (float(r.get("split_score") or 0.0), len(_row_core_set(r))), reverse=True):
        core = _row_core_set(row)
        new = core - owned
        overlap = (len(core) - len(new)) / max(len(core), 1)
        if len(new) < int(params["child_min_new_tubes"]):
            continue
        if overlap > float(params["child_max_overlap_ratio"]):
            continue
        item = dict(row)
        item["_core_tube_ids"] = tuple(sorted(new))
        item["core_tube_ids"] = ";".join(str(tid) for tid in sorted(new))
        item["num_core_tubes"] = int(len(new))
        selected.append(item)
        owned.update(new)
    return selected


def _apply_learned_a4_repair(
    *,
    scene: str,
    scene_rows: list[dict[str, Any]],
    base_selected: list[dict[str, Any]],
    score_by_proposal_id: dict[str, float],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for idx, seed in enumerate(base_selected):
        seed_core = _row_core_set(seed)
        should_try = len(seed_core) >= int(params["seed_min_tubes"]) or _row_risk(seed) >= float(params["seed_min_risk"])
        if not should_try:
            repaired.append(seed)
            continue
        children = _select_learned_children_for_seed(
            scene_rows=scene_rows,
            seed=seed,
            score_by_proposal_id=score_by_proposal_id,
            params=params,
        )
        covered = {tid for row in children for tid in _row_core_set(row)}
        cover_ratio = len(covered & seed_core) / max(len(seed_core), 1)
        replaced = len(children) >= int(params["replace_min_child_count"]) and cover_ratio >= float(params["replace_min_cover_ratio"])
        audit_rows.append(
            {
                "scene": scene,
                "seed_index": int(idx),
                "seed_proposal_id": seed.get("proposal_id"),
                "seed_tube_count": int(len(seed_core)),
                "seed_risk": float(_row_risk(seed)),
                "child_selected_count": int(len(children)),
                "child_cover_ratio": float(cover_ratio),
                "replaced": bool(replaced),
                "params": json.dumps(_json_safe(params), sort_keys=True),
            }
        )
        if replaced:
            repaired.extend(children)
        else:
            repaired.append(seed)
    return repaired, audit_rows


def _calibrate_learned_a4_repair_params(
    *,
    scenes_train: list[str],
    rows_by_scene: dict[str, list[dict[str, Any]]],
    base_selected_by_scene: dict[str, list[dict[str, Any]]],
    scores_by_scene: dict[str, dict[str, float]],
    gt_by_scene: dict[str, dict[int, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_scores = [score for scores in scores_by_scene.values() for score in scores.values()]
    if not all_scores:
        params = {
            "threshold": 0.5,
            "seed_min_tubes": 80,
            "seed_min_risk": 0.35,
            "child_min_tubes": 4,
            "child_min_new_tubes": 4,
            "child_min_inside_ratio": 0.50,
            "child_max_seed_ratio": 0.95,
            "child_max_overlap_ratio": 0.20,
            "replace_min_cover_ratio": 0.45,
            "replace_min_child_count": 1,
        }
        return params, {"reason": "no_scores"}
    thresholds = sorted(set(float(v) for v in np.quantile(np.asarray(all_scores), [0.55, 0.75]).tolist()))
    best_params: dict[str, Any] | None = None
    best_tuple: tuple[float, float, float, float, float] | None = None
    best_metrics: dict[str, Any] = {}
    for threshold in thresholds:
        for replace_cover in [0.45, 0.60]:
            for child_max_seed_ratio in [0.80, 0.95]:
                for replace_min_child_count in [1]:
                    params = {
                        "threshold": float(threshold),
                        "seed_min_tubes": 80,
                        "seed_min_risk": 0.35,
                        "child_min_tubes": 4,
                        "child_min_new_tubes": 4,
                        "child_min_inside_ratio": 0.50,
                        "child_max_seed_ratio": float(child_max_seed_ratio),
                        "child_max_overlap_ratio": 0.20,
                        "replace_min_cover_ratio": float(replace_cover),
                        "replace_min_child_count": int(replace_min_child_count),
                    }
                    selected_by_scene: dict[str, list[dict[str, Any]]] = {}
                    for scene in scenes_train:
                        repaired, _ = _apply_learned_a4_repair(
                            scene=scene,
                            scene_rows=rows_by_scene[scene],
                            base_selected=base_selected_by_scene[scene],
                            score_by_proposal_id=scores_by_scene[scene],
                            params=params,
                        )
                        selected_by_scene[scene] = repaired
                    rows = _evaluate_selected_for_route(
                        scenes=scenes_train,
                        route="D7_train_calibration",
                        variant="D7_train_calibration",
                        selected_by_scene=selected_by_scene,
                        gt_by_scene=gt_by_scene,
                    )
                    aggregate = next(row for row in rows if str(row.get("scene")) == "ALL")
                    ari = _safe_float(aggregate["local_ARI"], 0.0) or 0.0
                    purity = _safe_float(aggregate["local_purity"], 0.0) or 0.0
                    completeness = _safe_float(aggregate["local_completeness"], 0.0) or 0.0
                    unknown = _safe_float(aggregate["unknown_tube_ratio"], 1.0) or 1.0
                    gateish = (
                        min(ari / LOCAL_GATE["local_ARI"], 1.0)
                        + min(purity / LOCAL_GATE["local_purity"], 1.0)
                        + min(completeness / LOCAL_GATE["local_completeness"], 1.0)
                        + min(max(LOCAL_GATE["unknown_tube_ratio_max"] - unknown, 0.0) / LOCAL_GATE["unknown_tube_ratio_max"], 1.0)
                    )
                    candidate_tuple = (float(gateish), float(ari), float(purity), float(completeness), float(-unknown))
                    if best_tuple is None or candidate_tuple > best_tuple:
                        best_tuple = candidate_tuple
                        best_params = dict(params)
                        best_metrics = dict(aggregate)
    assert best_params is not None
    return best_params, {"train_identity_metrics": best_metrics, "selection_score_tuple": best_tuple}


def run_route_d(
    args: argparse.Namespace,
    *,
    proposal_rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "v34_routeD_learned_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_scene = _rows_by_scene(proposal_rows, scenes)
    mask_region_columns = [
        "mask_area",
        "mask_distance_mean",
        "mask_distance_p10",
        "mask_distance_p50",
        "mask_distance_p90",
        "num_boundary_tubes",
        "num_core_tubes",
        "proposal_area",
        "proposal_area_ratio",
        "region_area",
    ]
    variants = [
        {
            "name": "D0_full_non_gt_feature_scorer",
            "columns": FEATURE_COLUMNS,
            "model": "logistic",
            "target": "strict",
            "threshold_mode": "proposal_f1",
        },
        {
            "name": "D1_no_d4rt_geometry_features",
            "columns": [
                c
                for c in FEATURE_COLUMNS
                if c not in {"tube_canonical_compactness", "tube_temporal_length_mean", "tube_xy_compactness"}
            ],
            "model": "logistic",
            "target": "strict",
            "threshold_mode": "proposal_f1",
        },
        {
            "name": "D2_no_visual_proxy_features",
            "columns": [
                c
                for c in FEATURE_COLUMNS
                if c not in {"appearance_variance", "image_gradient_boundary_score", "confidence_mean", "visibility_mean"}
            ],
            "model": "logistic",
            "target": "strict",
            "threshold_mode": "proposal_f1",
        },
        {
            "name": "D3_mask_region_only_features",
            "columns": mask_region_columns,
            "model": "logistic",
            "target": "strict",
            "threshold_mode": "proposal_f1",
        },
        {
            "name": "D4_random_forest_identity_calibrated",
            "columns": FEATURE_COLUMNS,
            "model": "random_forest",
            "target": "strict",
            "threshold_mode": "train_identity_gate",
        },
        {
            "name": "D5_gradient_boost_identity_calibrated",
            "columns": FEATURE_COLUMNS,
            "model": "gradient_boosting",
            "target": "strict",
            "threshold_mode": "train_identity_gate",
        },
        {
            "name": "D6_random_forest_relaxed_target_identity_calibrated",
            "columns": FEATURE_COLUMNS,
            "model": "random_forest",
            "target": "relaxed",
            "threshold_mode": "train_identity_gate",
        },
    ]
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - dependency guard for audit machines
        status = {
            "route": "D_learned_diagnostic_scorer",
            "status": "not_run",
            "not_run_reason": f"scikit-learn import failed: {exc}",
            **DIAGNOSTIC_POLICY,
        }
        _write_json(out_dir / "routeD_status.json", status)
        return {"status": status, "rows": []}

    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for spec in variants:
        variant = str(spec["name"])
        columns = list(spec["columns"])
        scene_rows_for_variant: list[dict[str, Any]] = []
        for heldout in scenes:
            train_rows_by_scene = {
                scene: [row for row in _rows_for_variant(rows, "P4_greedy_set_packing") if _parse_core_tube_ids(row)]
                for scene, rows in rows_by_scene.items()
                if scene != heldout
            }
            train_rows = [row for rows in train_rows_by_scene.values() for row in rows]
            test_rows = _rows_for_variant(rows_by_scene.get(heldout, []), "P4_greedy_set_packing")
            test_rows = [row for row in test_rows if _parse_core_tube_ids(row)]
            if not train_rows or not test_rows:
                continue
            target_fn = _proposal_target_relaxed if spec["target"] == "relaxed" else _proposal_target
            y_train = np.asarray([target_fn(row) for row in train_rows], dtype=np.int64)
            y_test = np.asarray([target_fn(row) for row in test_rows], dtype=np.int64)
            if len(set(y_train.tolist())) < 2:
                continue
            scaler = StandardScaler()
            x_train = scaler.fit_transform(_feature_matrix(train_rows, columns))
            x_test = scaler.transform(_feature_matrix(test_rows, columns))
            if spec["model"] == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=80,
                    max_depth=8,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    random_state=34,
                    n_jobs=1,
                )
            elif spec["model"] == "gradient_boosting":
                model = GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=34,
                )
            else:
                model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=34)
            model.fit(x_train, y_train)
            train_scores = model.predict_proba(x_train)[:, 1]
            test_scores = model.predict_proba(x_test)[:, 1]
            calibration_info: dict[str, Any]
            if spec["threshold_mode"] == "train_identity_gate":
                train_scores_by_scene = {
                    scene: model.predict_proba(scaler.transform(_feature_matrix(rows, columns)))[:, 1]
                    for scene, rows in train_rows_by_scene.items()
                }
                threshold, calibration_info = _identity_calibrated_threshold(
                    train_rows_by_scene=train_rows_by_scene,
                    train_scores_by_scene=train_scores_by_scene,
                    gt_by_scene=gt_by_scene,
                )
            else:
                threshold = _calibrate_threshold(y_train, train_scores)
                calibration_info = {"threshold": float(threshold), "mode": "proposal_f1"}
            selected = _select_learned(test_rows, test_scores, threshold=threshold)
            gt_labels = gt_by_scene.get(heldout, {})
            labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
            labels_pred, unknown_count = _labels_from_selected(selected, labeled_tubes)
            metrics = _cluster_metrics(labels_pred, gt_labels)
            auc = None
            ap = None
            if len(set(y_test.tolist())) >= 2:
                auc = float(roc_auc_score(y_test, test_scores))
                ap = float(average_precision_score(y_test, test_scores))
            row = {
                "scene": heldout,
                "variant": variant,
                "feature_count": int(len(columns)),
                "model": str(spec["model"]),
                "target": str(spec["target"]),
                "threshold_mode": str(spec["threshold_mode"]),
                "train_proposal_count": int(len(train_rows)),
                "train_positive_count": int(y_train.sum()),
                "test_proposal_count": int(len(test_rows)),
                "test_positive_count": int(y_test.sum()),
                "threshold": float(threshold),
                "calibration_info": json.dumps(_json_safe(calibration_info), sort_keys=True),
                "diagnostic_auc": auc,
                "diagnostic_average_precision": ap,
                "selected_proposal_count": int(len(selected)),
                "labeled_tube_count": int(len(labeled_tubes)),
                "unknown_tube_count": int(unknown_count),
                "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
                "local_ARI": metrics["ari"],
                "local_purity": metrics["purity"],
                "local_completeness": metrics["completeness"],
                "local_overmerge": metrics["overmerge"],
                "local_oversplit": metrics["oversplit"],
                **DIAGNOSTIC_POLICY,
            }
            scene_rows_for_variant.append(row)
            summary_rows.append(row)
            for rank, item in enumerate(selected):
                selected_rows.append(
                    {
                        "scene": heldout,
                        "variant": variant,
                        "rank": int(rank),
                        "proposal_id": item.get("proposal_id"),
                        "proposal_type": item.get("proposal_type"),
                        "learned_score": item.get("learned_score"),
                        "num_core_tubes": int(item.get("num_core_tubes") or len(_parse_core_tube_ids(item))),
                    }
                )

        if scene_rows_for_variant:
            aggregate = {
                "scene": "ALL",
                "variant": variant,
                "feature_count": int(len(columns)),
                "model": str(spec["model"]),
                "target": str(spec["target"]),
                "threshold_mode": str(spec["threshold_mode"]),
                "train_proposal_count": int(sum(int(r["train_proposal_count"]) for r in scene_rows_for_variant)),
                "train_positive_count": int(sum(int(r["train_positive_count"]) for r in scene_rows_for_variant)),
                "test_proposal_count": int(sum(int(r["test_proposal_count"]) for r in scene_rows_for_variant)),
                "test_positive_count": int(sum(int(r["test_positive_count"]) for r in scene_rows_for_variant)),
                "diagnostic_auc": _mean([_safe_float(r["diagnostic_auc"]) for r in scene_rows_for_variant]),
                "diagnostic_average_precision": _mean(
                    [_safe_float(r["diagnostic_average_precision"]) for r in scene_rows_for_variant]
                ),
                "selected_proposal_count": int(sum(int(r["selected_proposal_count"]) for r in scene_rows_for_variant)),
                "labeled_tube_count": int(sum(int(r["labeled_tube_count"]) for r in scene_rows_for_variant)),
                "unknown_tube_count": int(sum(int(r["unknown_tube_count"]) for r in scene_rows_for_variant)),
                "unknown_tube_ratio": _mean([_safe_float(r["unknown_tube_ratio"]) for r in scene_rows_for_variant]),
                "local_ARI": _mean([_safe_float(r["local_ARI"]) for r in scene_rows_for_variant]),
                "local_purity": _mean([_safe_float(r["local_purity"]) for r in scene_rows_for_variant]),
                "local_completeness": _mean([_safe_float(r["local_completeness"]) for r in scene_rows_for_variant]),
                "local_overmerge": _mean([_safe_float(r["local_overmerge"]) for r in scene_rows_for_variant]),
                "local_oversplit": _mean([_safe_float(r["local_oversplit"]) for r in scene_rows_for_variant]),
                "scene0081_local_ARI": next(
                    (r["local_ARI"] for r in scene_rows_for_variant if str(r.get("scene")) == "scene0081_01"),
                    None,
                ),
                **DIAGNOSTIC_POLICY,
            }
            summary_rows.append({**aggregate, **_gate_status(aggregate)})

    d7_audit_rows: list[dict[str, Any]] = []
    d7_scene_rows: list[dict[str, Any]] = []
    for heldout in scenes:
        train_scenes = [scene for scene in scenes if scene != heldout]
        train_rows_by_scene = {
            scene: [row for row in _rows_for_variant(rows_by_scene.get(scene, []), "P4_greedy_set_packing") if _parse_core_tube_ids(row)]
            for scene in train_scenes
        }
        train_rows = [row for rows in train_rows_by_scene.values() for row in rows]
        test_rows = [row for row in _rows_for_variant(rows_by_scene.get(heldout, []), "P4_greedy_set_packing") if _parse_core_tube_ids(row)]
        if not train_rows or not test_rows:
            continue
        y_train = np.asarray([_proposal_target_relaxed(row) for row in train_rows], dtype=np.int64)
        y_test = np.asarray([_proposal_target_relaxed(row) for row in test_rows], dtype=np.int64)
        if len(set(y_train.tolist())) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(_feature_matrix(train_rows, FEATURE_COLUMNS))
        x_test = scaler.transform(_feature_matrix(test_rows, FEATURE_COLUMNS))
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=3407,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        train_scores_by_scene = {
            scene: model.predict_proba(scaler.transform(_feature_matrix(rows, FEATURE_COLUMNS)))[:, 1]
            for scene, rows in train_rows_by_scene.items()
        }
        train_score_maps = {
            scene: {str(row.get("proposal_id")): float(score) for row, score in zip(train_rows_by_scene[scene], train_scores_by_scene[scene].tolist())}
            for scene in train_scenes
        }
        base_selected_by_scene = {
            scene: _select_variant(
                rows_by_scene.get(scene, []),
                variant="P11_calibrated_ownership_expansion",
                strict=False,
                shuffled_proxy_seed=int(args.shuffle_control_seed),
                scene=scene,
            )
            for scene in scenes
        }
        params, calibration_info = _calibrate_learned_a4_repair_params(
            scenes_train=train_scenes,
            rows_by_scene=rows_by_scene,
            base_selected_by_scene=base_selected_by_scene,
            scores_by_scene=train_score_maps,
            gt_by_scene=gt_by_scene,
        )
        test_scores = model.predict_proba(x_test)[:, 1]
        test_score_map = {str(row.get("proposal_id")): float(score) for row, score in zip(test_rows, test_scores.tolist())}
        repaired, audit = _apply_learned_a4_repair(
            scene=heldout,
            scene_rows=rows_by_scene.get(heldout, []),
            base_selected=base_selected_by_scene.get(heldout, []),
            score_by_proposal_id=test_score_map,
            params=params,
        )
        for audit_row in audit:
            d7_audit_rows.append(
                {
                    "variant": "D7_learned_a4_child_repair",
                    "heldout_scene": heldout,
                    **audit_row,
                }
            )
        gt_labels = gt_by_scene.get(heldout, {})
        labeled_tubes = sorted(tid for tid, gt in gt_labels.items() if int(gt) > 0)
        labels_pred, unknown_count = _labels_from_selected(repaired, labeled_tubes)
        metrics = _cluster_metrics(labels_pred, gt_labels)
        auc = None
        ap = None
        if len(set(y_test.tolist())) >= 2:
            auc = float(roc_auc_score(y_test, test_scores))
            ap = float(average_precision_score(y_test, test_scores))
        row = {
            "scene": heldout,
            "variant": "D7_learned_a4_child_repair",
            "feature_count": int(len(FEATURE_COLUMNS)),
            "model": "random_forest",
            "target": "relaxed",
            "threshold_mode": "train_identity_gate_a4_child_repair",
            "train_proposal_count": int(len(train_rows)),
            "train_positive_count": int(y_train.sum()),
            "test_proposal_count": int(len(test_rows)),
            "test_positive_count": int(y_test.sum()),
            "threshold": float(params["threshold"]),
            "calibration_info": json.dumps(_json_safe({"params": params, **calibration_info}), sort_keys=True),
            "diagnostic_auc": auc,
            "diagnostic_average_precision": ap,
            "selected_proposal_count": int(len(repaired)),
            "labeled_tube_count": int(len(labeled_tubes)),
            "unknown_tube_count": int(unknown_count),
            "unknown_tube_ratio": float(unknown_count / max(len(labeled_tubes), 1)),
            "local_ARI": metrics["ari"],
            "local_purity": metrics["purity"],
            "local_completeness": metrics["completeness"],
            "local_overmerge": metrics["overmerge"],
            "local_oversplit": metrics["oversplit"],
            **DIAGNOSTIC_POLICY,
        }
        d7_scene_rows.append(row)
        summary_rows.append(row)
        for rank, item in enumerate(repaired):
            selected_rows.append(
                {
                    "scene": heldout,
                    "variant": "D7_learned_a4_child_repair",
                    "rank": int(rank),
                    "proposal_id": item.get("proposal_id"),
                    "proposal_type": item.get("proposal_type"),
                    "learned_score": item.get("learned_score", ""),
                    "num_core_tubes": int(item.get("num_core_tubes") or len(_parse_core_tube_ids(item))),
                }
            )

    if d7_scene_rows:
        d7_aggregate = {
            "scene": "ALL",
            "variant": "D7_learned_a4_child_repair",
            "feature_count": int(len(FEATURE_COLUMNS)),
            "model": "random_forest",
            "target": "relaxed",
            "threshold_mode": "train_identity_gate_a4_child_repair",
            "train_proposal_count": int(sum(int(r["train_proposal_count"]) for r in d7_scene_rows)),
            "train_positive_count": int(sum(int(r["train_positive_count"]) for r in d7_scene_rows)),
            "test_proposal_count": int(sum(int(r["test_proposal_count"]) for r in d7_scene_rows)),
            "test_positive_count": int(sum(int(r["test_positive_count"]) for r in d7_scene_rows)),
            "diagnostic_auc": _mean([_safe_float(r["diagnostic_auc"]) for r in d7_scene_rows]),
            "diagnostic_average_precision": _mean([_safe_float(r["diagnostic_average_precision"]) for r in d7_scene_rows]),
            "selected_proposal_count": int(sum(int(r["selected_proposal_count"]) for r in d7_scene_rows)),
            "labeled_tube_count": int(sum(int(r["labeled_tube_count"]) for r in d7_scene_rows)),
            "unknown_tube_count": int(sum(int(r["unknown_tube_count"]) for r in d7_scene_rows)),
            "unknown_tube_ratio": _mean([_safe_float(r["unknown_tube_ratio"]) for r in d7_scene_rows]),
            "local_ARI": _mean([_safe_float(r["local_ARI"]) for r in d7_scene_rows]),
            "local_purity": _mean([_safe_float(r["local_purity"]) for r in d7_scene_rows]),
            "local_completeness": _mean([_safe_float(r["local_completeness"]) for r in d7_scene_rows]),
            "local_overmerge": _mean([_safe_float(r["local_overmerge"]) for r in d7_scene_rows]),
            "local_oversplit": _mean([_safe_float(r["local_oversplit"]) for r in d7_scene_rows]),
            "scene0081_local_ARI": next((r["local_ARI"] for r in d7_scene_rows if str(r.get("scene")) == "scene0081_01"), None),
            **DIAGNOSTIC_POLICY,
        }
        summary_rows.append({**d7_aggregate, **_gate_status(d7_aggregate)})

    _write_csv(out_dir / "routeD_learned_diagnostic_summary.csv", summary_rows)
    _write_json(
        out_dir / "routeD_learned_diagnostic_summary.json",
        {
            "summary_rows": summary_rows,
            "feature_variants": variants,
            "target_definition": {
                "strict": "proposal_purity>=0.85 and proposal_completeness>=0.25 and proposal_best_IoU>=0.25",
                "relaxed": "proposal_purity>=0.80 and proposal_completeness>=0.20 and proposal_best_IoU>=0.20",
            },
            "policy": dict(DIAGNOSTIC_POLICY),
        },
    )
    _write_csv(out_dir / "routeD_selected_proposals.csv", selected_rows)
    _write_csv(out_dir / "routeD_learned_a4_repair_audit.csv", d7_audit_rows)
    return {"rows": summary_rows, "selected_rows": selected_rows}


def _control_gate(row: dict[str, Any], controls: dict[str, float | None], window0: float | None) -> dict[str, Any]:
    ari = _safe_float(row.get("local_ARI"))
    shuffled = controls.get("shuffled")
    no_temporal = controls.get("no_temporal")
    mask_only = controls.get("mask_only")
    checks = {
        "control_shuffled_pass": ari is not None and shuffled is not None and ari >= shuffled + 0.20,
        "control_no_temporal_pass": ari is not None and no_temporal is not None and ari >= no_temporal + 0.05,
        "control_mask_only_pass": ari is not None and mask_only is not None and ari >= mask_only + 0.05,
        "control_window0_baseline_pass": ari is not None and window0 is not None and ari >= window0,
    }
    return {
        **checks,
        "control_gate_pass": bool(all(checks.values())),
        "control_values": {
            "shuffled_ari": shuffled,
            "no_temporal_ari": no_temporal,
            "mask_only_ari": mask_only,
            "window0_baseline_ari": window0,
        },
    }


def run_final_decision(args: argparse.Namespace, route_a: dict[str, Any], route_d: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.output_root) / "v34_final_decision"
    out_dir.mkdir(parents=True, exist_ok=True)
    phase0 = _load_phase0_lock(Path(args.phase0_lock))
    window0 = None
    current_state = phase0.get("current_state", {}) if phase0 else {}
    v27_partition = current_state.get("v27_partition_and_memory", {}) if isinstance(current_state, dict) else {}
    s5_window0 = v27_partition.get("S5_window0_positive_only", {}) if isinstance(v27_partition, dict) else {}
    if isinstance(s5_window0, dict):
        window0 = _safe_float(s5_window0.get("ARI"))
    metrics = phase0.get("metrics", {}) if isinstance(phase0, dict) else {}
    if window0 is None and isinstance(metrics, dict):
        for key in ["v27_S5_window0_positive_only_ARI", "window0_baseline_ARI"]:
            value = metrics.get(key, {})
            if isinstance(value, dict) and value.get("value") is not None:
                window0 = _safe_float(value.get("value"))
                break

    routes = route_a.get("route_rows", [])
    strict_map = {str(row.get("route")): row for row in routes}
    controls = {
        "shuffled": _safe_float(strict_map.get("A5_shuffled_d4rt_control_full_pool", {}).get("local_ARI")),
        "no_temporal": _safe_float(strict_map.get("A7_no_temporal_control", {}).get("local_ARI")),
        "mask_only": _safe_float(strict_map.get("A6_mask_only_control", {}).get("local_ARI")),
    }

    decision_rows: list[dict[str, Any]] = []
    for row in routes:
        if str(row.get("status")) == "missing_source_row":
            decision_rows.append(row)
            continue
        gate = _gate_status(row)
        control = _control_gate(row, controls, window0)
        decision_rows.append(
            {
                "route": row.get("route"),
                "variant": row.get("variant"),
                "is_method_result": row.get("is_method_result"),
                "is_diagnostic_only": row.get("is_diagnostic_only"),
                "local_ARI": row.get("local_ARI"),
                "local_purity": row.get("local_purity"),
                "local_completeness": row.get("local_completeness"),
                "unknown_tube_ratio": row.get("unknown_tube_ratio"),
                "scene0081_local_ARI": row.get("scene0081_local_ARI"),
                **gate,
                **control,
                "method_gate_pass": bool(gate["local_gate_pass"] and control["control_gate_pass"] and row.get("is_method_result") is True),
            }
        )

    for row in route_d.get("rows", []):
        if str(row.get("scene")) != "ALL":
            continue
        gate = _gate_status(row)
        decision_rows.append(
            {
                "route": f"RouteD::{row.get('variant')}",
                "variant": row.get("variant"),
                "is_method_result": False,
                "is_diagnostic_only": True,
                "local_ARI": row.get("local_ARI"),
                "local_purity": row.get("local_purity"),
                "local_completeness": row.get("local_completeness"),
                "unknown_tube_ratio": row.get("unknown_tube_ratio"),
                "scene0081_local_ARI": row.get("scene0081_local_ARI"),
                "diagnostic_auc": row.get("diagnostic_auc"),
                "diagnostic_average_precision": row.get("diagnostic_average_precision"),
                **gate,
                "control_gate_pass": False,
                "method_gate_pass": False,
                "diagnostic_only_reason": "Route D uses GT-calibrated proposal targets and is forbidden for method table.",
            }
        )

    method_passes = [row for row in decision_rows if bool(row.get("method_gate_pass"))]
    final = {
        "final_status": "GO_LOCAL_3D_OBJECT_IDENTITY" if method_passes else "NO_GO_LOCAL_3D_OBJECT_IDENTITY",
        "method_pass_count": int(len(method_passes)),
        "local_gate": dict(LOCAL_GATE),
        "control_gate_definition": {
            "real_vs_shuffled_margin": 0.20,
            "real_vs_no_temporal_margin": 0.05,
            "real_vs_mask_only_margin": 0.05,
            "must_beat_window0_baseline": True,
        },
        "window0_baseline_ari": window0,
        "decision_rows": decision_rows,
        "no_4d_or_ap_reason": None if method_passes else "No method route passed the v34 local 3D object identity gate.",
    }
    _write_csv(out_dir / "decision_table.csv", decision_rows)
    _write_json(out_dir / "decision_summary.json", final)

    if not method_passes:
        _write_json(
            Path(args.output_root) / "v34_4d_memory_if_allowed" / "not_run_manifest.json",
            {
                "phase": "v34_4d_memory_if_allowed",
                "status": "not_run",
                "not_run_reason": final["no_4d_or_ap_reason"],
                "upstream_decision": str(out_dir / "decision_summary.json"),
            },
        )
        _write_json(
            Path(args.output_root) / "v34_ap_export_if_allowed" / "not_run_manifest.json",
            {
                "phase": "v34_ap_export_if_allowed",
                "status": "not_run",
                "not_run_reason": final["no_4d_or_ap_reason"],
                "upstream_decision": str(out_dir / "decision_summary.json"),
            },
        )
    return final


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_split(Path(args.split))
    proposal_rows = _read_json(Path(args.proposal_root) / f"{args.proposal_label}_proposal_rows.json")
    requested = {item.strip() for item in str(args.routes).split(",") if item.strip()}
    gt_by_scene: dict[str, dict[int, int]] = {}
    if requested & {"routeA", "routeD", "final"}:
        gt_by_scene = _load_gt_by_scene(args, scenes)

    result: dict[str, Any] = {
        "routes_requested": sorted(requested),
        "output_root": str(output_root),
        "proposal_root": str(args.proposal_root),
        "proposal_label": str(args.proposal_label),
        "split": str(args.split),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    phase2 = {}
    if "phase2" in requested or "all" in requested:
        phase2 = run_phase2(args, proposal_rows)
        result["phase2"] = phase2
    route_a = {}
    if "routeA" in requested or "all" in requested or "final" in requested:
        route_a = run_route_a(args, proposal_rows=proposal_rows, scenes=scenes, gt_by_scene=gt_by_scene)
        result["routeA"] = route_a
    route_b = {}
    if "routeB" in requested or "all" in requested:
        route_b = run_route_b(args, proposal_rows)
        result["routeB"] = route_b
    route_c = {}
    if "routeC" in requested or "all" in requested:
        if not phase2:
            phase2 = run_phase2(args, proposal_rows)
        route_c = run_route_c(args, phase2)
        result["routeC"] = route_c
    route_d = {}
    if "routeD" in requested or "all" in requested or "final" in requested:
        route_d = run_route_d(args, proposal_rows=proposal_rows, scenes=scenes, gt_by_scene=gt_by_scene)
        result["routeD"] = route_d
    if "final" in requested or "all" in requested:
        if not route_a:
            route_a = run_route_a(args, proposal_rows=proposal_rows, scenes=scenes, gt_by_scene=gt_by_scene)
        if not route_d:
            route_d = run_route_d(args, proposal_rows=proposal_rows, scenes=scenes, gt_by_scene=gt_by_scene)
        result["final"] = run_final_decision(args, route_a, route_d)

    route_tag = "_".join(sorted(requested)) or "none"
    _write_json(output_root / f"v34_run_manifest_{route_tag}.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stream4D v34 local 3D object identity route audits.")
    parser.add_argument("--proposal-root", default="outputs/audit/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--proposal-label", default="v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--output-root", default="outputs/audit/v34_3d_object_identity")
    parser.add_argument("--routes", default="all", help="Comma-separated subset: phase2,routeA,routeB,routeC,routeD,final,all")
    parser.add_argument("--strict-selection-root", default="outputs/audit/v28_proposal_selection_guard5_strict_score02_r3_with_p8_proxy")
    parser.add_argument("--strict-selection-label", default="v28_proposal_selection_guard5_strict_score02_r3_with_p8_proxy")
    parser.add_argument("--p11-selection-root", default="outputs/audit/v28_proposal_selection_guard5_p11_ownership_expansion_r1")
    parser.add_argument("--p11-selection-label", default="v28_proposal_selection_guard5_p11_ownership_expansion_r1")
    parser.add_argument("--shuffle-selection-root", default="outputs/audit/v28_proposal_selection_shuffle_d4rt_guard5_strict_score02_r1")
    parser.add_argument("--shuffle-selection-label", default="v28_proposal_selection_shuffle_d4rt_guard5_strict_score02_r1")
    parser.add_argument("--phase0-lock", default="outputs/audit/v34_phase0/current_state_lock.json")
    parser.add_argument("--shuffle-control-seed", type=int, default=2808)
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    payload = run(parsed)
    print(json.dumps(_json_safe({"output_root": payload["output_root"], "routes_requested": payload["routes_requested"]}), indent=2))
