#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_v103_phase9g_da3_seed_gaussian_neighborhood import (  # noqa: E402
    AUDIT_ROOT,
    DEFAULT_PHASE9E_ROOT,
    PLAN_DOC,
    SCENE_SPECS,
    _frame_label_lookup,
    _hit_counts,
    _jsonable,
    _load_or_project_da3,
    _load_phase9e_supports,
    _obs_meta_from_phase3,
    _project_phase,
    _rel,
    _summarize_obs,
    _write_csv,
    _write_json,
)
from diagnose_v103_phase9h_da3_object_local_components import (  # noqa: E402
    _component_labels,
    _component_obs_sets,
    _ratio,
)


PHASE_ID = "v103_phase9i_da3_seed_growth_objectlike_graph"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9i_da3_seed_growth_objectlike_graph_r1"

VARIANTS = [
    {
        "variant_id": "i1_objuniverse_broad050_seed1_knn8_q90_obs010_min16_cap250k",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.50,
        "max_universe_gaussians": 250000,
        "component_knn_k": 8,
        "edge_radius_quantile": 0.90,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 16,
        "max_component_gaussians": 50000,
        "min_seed_gaussians_per_component": 8,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "i2_objuniverse_broad035_seed1_knn8_q90_obs010_min16_cap250k",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.35,
        "max_universe_gaussians": 250000,
        "component_knn_k": 8,
        "edge_radius_quantile": 0.90,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 16,
        "max_component_gaussians": 50000,
        "min_seed_gaussians_per_component": 8,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "i3_objuniverse_broad050_seed2_knn8_q90_obs010_min16_cap250k",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.50,
        "max_universe_gaussians": 250000,
        "component_knn_k": 8,
        "edge_radius_quantile": 0.90,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 16,
        "max_component_gaussians": 50000,
        "min_seed_gaussians_per_component": 8,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "i4_objuniverse_broad035_seed1_knn12_q75_obs020_min16_cap250k",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.35,
        "max_universe_gaussians": 250000,
        "component_knn_k": 12,
        "edge_radius_quantile": 0.75,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 16,
        "max_component_gaussians": 50000,
        "min_seed_gaussians_per_component": 8,
        "obs_selected_ratio_min": 0.20,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "i5_objuniverse_broad050_seed1_veto050_knn8_q90_obs010_min16_cap250k",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 0.50,
        "broad_ratio_max": 0.50,
        "max_universe_gaussians": 250000,
        "component_knn_k": 8,
        "edge_radius_quantile": 0.90,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 16,
        "max_component_gaussians": 50000,
        "min_seed_gaussians_per_component": 8,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
]


def _rank_and_cap(
    universe_idx: np.ndarray,
    seed_idx: np.ndarray,
    max_count: int,
    object_hit: np.ndarray,
    object_ratio: np.ndarray,
    seed_hit: np.ndarray,
    broad_ratio: np.ndarray,
) -> np.ndarray:
    if max_count <= 0 or universe_idx.size <= max_count:
        return universe_idx.astype(np.int64, copy=False)
    order = np.lexsort(
        (
            broad_ratio[universe_idx],
            -seed_hit[universe_idx],
            -object_ratio[universe_idx],
            -object_hit[universe_idx],
        )
    )
    capped = universe_idx[order[: int(max_count)]]
    if seed_idx.size:
        capped = np.unique(np.concatenate([capped.astype(np.int64, copy=False), seed_idx.astype(np.int64, copy=False)]))
    return capped.astype(np.int64, copy=False)


def _obs_counts(prefix: str, obs_set: set[str], obs_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return _summarize_obs(prefix, obs_set, obs_meta)


def _score_variant(
    *,
    scene_id: str,
    variant: dict[str, Any],
    mask_by_frame: np.ndarray,
    xyz: np.ndarray,
    frame_df: pd.DataFrame,
    positive_obs: set[str],
    object_hit: np.ndarray,
    object_seed_hit: np.ndarray,
    veto_hit: np.ndarray,
    broad_hit: np.ndarray,
    visible_count: np.ndarray,
    obs_meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    object_ratio = _ratio(object_hit, visible_count)
    seed_ratio = _ratio(object_seed_hit, visible_count)
    veto_ratio = _ratio(veto_hit, visible_count)
    broad_ratio = _ratio(broad_hit, visible_count)
    universe = (
        (visible_count > 0)
        & (object_hit >= int(variant["object_hit_min"]))
        & (object_ratio >= float(variant["object_ratio_min"]))
        & (veto_ratio <= float(variant["veto_ratio_max"]))
        & (broad_ratio <= float(variant["broad_ratio_max"]))
    )
    seed = (
        universe
        & (object_seed_hit >= int(variant["seed_hit_min"]))
        & (seed_ratio >= float(variant["seed_ratio_min"]))
    )
    universe_idx_raw = np.flatnonzero(universe).astype(np.int64, copy=False)
    seed_idx = np.flatnonzero(seed).astype(np.int64, copy=False)
    universe_idx = _rank_and_cap(
        universe_idx=universe_idx_raw,
        seed_idx=seed_idx,
        max_count=int(variant["max_universe_gaussians"]),
        object_hit=object_hit,
        object_ratio=object_ratio,
        seed_hit=object_seed_hit,
        broad_ratio=broad_ratio,
    )
    labels, component_info = _component_labels(
        xyz=xyz,
        candidate_idx=universe_idx,
        k=int(variant["component_knn_k"]),
        edge_radius_quantile=float(variant["edge_radius_quantile"]),
        edge_radius_factor=float(variant["edge_radius_factor"]),
    )
    component_sizes = np.bincount(labels, minlength=int(component_info["component_count_total"])) if labels.size else np.asarray([], dtype=np.int64)
    seed_for_candidate = seed[universe_idx] if universe_idx.size else np.asarray([], dtype=bool)
    component_seed_counts = np.bincount(labels, weights=seed_for_candidate.astype(np.int32), minlength=int(component_info["component_count_total"])).astype(np.int64) if labels.size else np.asarray([], dtype=np.int64)
    keep_component = (
        (component_sizes >= int(variant["min_component_gaussians"]))
        & (component_sizes <= int(variant["max_component_gaussians"]))
        & (component_seed_counts >= int(variant["min_seed_gaussians_per_component"]))
    ) if component_sizes.size else np.asarray([], dtype=bool)
    comp_obs = _component_obs_sets(
        mask_by_frame=mask_by_frame,
        frame_df=frame_df,
        candidate_idx=universe_idx,
        labels=labels,
        keep_component=keep_component,
        scene_id=scene_id,
        obs_selected_ratio_min=float(variant["obs_selected_ratio_min"]),
        obs_selected_count_min=int(variant["obs_selected_count_min"]),
    )

    component_rows: list[dict[str, Any]] = []
    clean_selected_obs: set[str] = set()
    clean_induced_obs: set[str] = set()
    all_selected_obs: set[str] = set()
    all_induced_obs: set[str] = set()
    clean_component_count = 0
    clean_anchor_only_count = 0
    clean_with_induced_count = 0
    for component_id in np.flatnonzero(keep_component).astype(int).tolist():
        selected_obs = comp_obs.get(int(component_id), set())
        if not selected_obs:
            continue
        induced_obs = selected_obs - positive_obs
        all_selected_obs.update(selected_obs)
        all_induced_obs.update(induced_obs)
        bits = _obs_counts("selected", selected_obs, obs_meta)
        bits.update(_obs_counts("induced", induced_obs, obs_meta))
        selected_known = max(int(bits["selected_known_obs_count"]), 1)
        object_like_rate = float(bits["selected_object_like_obs_count"]) / float(selected_known)
        broad_rate = float(bits["selected_broad_obs_count"]) / float(selected_known)
        is_clean = object_like_rate >= 0.60 and broad_rate <= 0.30
        if is_clean:
            clean_component_count += 1
            clean_selected_obs.update(selected_obs)
            clean_induced_obs.update(induced_obs)
            if induced_obs:
                clean_with_induced_count += 1
            else:
                clean_anchor_only_count += 1
        row = {
            "schema_version": "stream4d_v103_phase9i_component_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "variant_id": str(variant["variant_id"]),
            "component_id": int(component_id),
            "component_gaussian_count": int(component_sizes[component_id]),
            "component_seed_gaussian_count": int(component_seed_counts[component_id]),
            "is_clean_component": bool(is_clean),
            "anchor_obs_count": int(len(selected_obs & positive_obs)),
            "induced_obs_count": int(len(induced_obs)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_coverage_gate": False,
        }
        row.update(bits)
        component_rows.append(row)

    clean_induced_bits = _obs_counts("clean_induced", clean_induced_obs, obs_meta)
    clean_induced_known = max(int(clean_induced_bits["clean_induced_known_obs_count"]), 1)
    clean_induced_broad_rate = float(clean_induced_bits["clean_induced_broad_obs_count"]) / float(clean_induced_known)
    gate = (
        int(clean_induced_bits["clean_induced_object_like_obs_count"]) >= 30
        and clean_induced_broad_rate <= 0.50
        and clean_component_count >= 5
    )
    variant_row: dict[str, Any] = {
        "schema_version": "stream4d_v103_phase9i_variant_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "variant_id": str(variant["variant_id"]),
        "object_hit_min": int(variant["object_hit_min"]),
        "object_ratio_min": float(variant["object_ratio_min"]),
        "seed_hit_min": int(variant["seed_hit_min"]),
        "seed_ratio_min": float(variant["seed_ratio_min"]),
        "veto_ratio_max": float(variant["veto_ratio_max"]),
        "broad_ratio_max": float(variant["broad_ratio_max"]),
        "max_universe_gaussians": int(variant["max_universe_gaussians"]),
        "component_knn_k": int(variant["component_knn_k"]),
        "edge_radius_quantile": float(variant["edge_radius_quantile"]),
        "edge_radius_factor": float(variant["edge_radius_factor"]),
        "min_component_gaussians": int(variant["min_component_gaussians"]),
        "max_component_gaussians": int(variant["max_component_gaussians"]),
        "min_seed_gaussians_per_component": int(variant["min_seed_gaussians_per_component"]),
        "obs_selected_ratio_min": float(variant["obs_selected_ratio_min"]),
        "obs_selected_count_min": int(variant["obs_selected_count_min"]),
        "universe_gaussian_count_raw": int(universe_idx_raw.size),
        "seed_gaussian_count_raw": int(seed_idx.size),
        "universe_gaussian_count_after_cap": int(universe_idx.size),
        "kept_component_count": int(np.count_nonzero(keep_component)),
        "observed_component_count": int(len(component_rows)),
        "clean_component_count": int(clean_component_count),
        "clean_anchor_only_component_count": int(clean_anchor_only_count),
        "clean_with_induced_component_count": int(clean_with_induced_count),
        "largest_component_gaussians": int(component_sizes.max()) if component_sizes.size else 0,
        "largest_component_seed_gaussians": int(component_seed_counts.max()) if component_seed_counts.size else 0,
        "component_size_p50": float(np.quantile(component_sizes, 0.50)) if component_sizes.size else 0.0,
        "component_size_p90": float(np.quantile(component_sizes, 0.90)) if component_sizes.size else 0.0,
        "component_size_p95": float(np.quantile(component_sizes, 0.95)) if component_sizes.size else 0.0,
        "phase9i_seed_growth_gate_pass": bool(gate),
        "uses_gt_for_prediction": False,
        "uses_gt_for_coverage_gate": False,
    }
    variant_row.update(component_info)
    variant_row.update(_obs_counts("all_selected", all_selected_obs, obs_meta))
    variant_row.update(_obs_counts("all_induced", all_induced_obs, obs_meta))
    variant_row.update(_obs_counts("clean_selected", clean_selected_obs, obs_meta))
    variant_row.update(clean_induced_bits)
    return variant_row, component_rows


def _process_scene(
    scene_id: str,
    phase9e_root: Path,
    out: Path,
    device_id: int,
    projection_cache_root: Path | None,
) -> dict[str, Any]:
    scene_out = out / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    positive_masks, veto_masks = _load_phase9e_supports(phase9e_root, scene_id)
    mask_by_frame, xyz, frame_df, _meta, manifest = _load_or_project_da3(scene_id, out, projection_cache_root)
    obs_meta = _obs_meta_from_phase3(scene_id, int(device_id), out)
    positive_obs = {f"{scene_id}:{frame_id}:{mask_id}" for frame_id, mask_id in positive_masks}
    object_positive_masks = set()
    object_candidate_masks = set()
    for bits in obs_meta.values():
        frame_id = int(bits["frame_id"])
        mask_id = int(bits["mask_id"])
        if bool(bits.get("is_object_like", False)) and not bool(bits.get("is_broad", True)):
            object_candidate_masks.add((frame_id, mask_id))
    for frame_id, mask_id in positive_masks:
        obs = f"{scene_id}:{frame_id}:{mask_id}"
        bits = obs_meta.get(obs, {})
        if bool(bits.get("is_object_like", False)) and not bool(bits.get("is_broad", True)):
            object_positive_masks.add((int(frame_id), int(mask_id)))
    broad_masks = {
        (int(bits["frame_id"]), int(bits["mask_id"]))
        for bits in obs_meta.values()
        if bool(bits.get("is_broad", False))
    }
    max_label = int(np.max(mask_by_frame)) if mask_by_frame.size else 0
    object_lookup = _frame_label_lookup(frame_df, object_candidate_masks, max_label)
    object_positive_lookup = _frame_label_lookup(frame_df, object_positive_masks, max_label)
    veto_lookup = _frame_label_lookup(frame_df, veto_masks, max_label)
    broad_lookup = _frame_label_lookup(frame_df, broad_masks, max_label)
    object_hit = _hit_counts(mask_by_frame, object_lookup)
    object_seed_hit = _hit_counts(mask_by_frame, object_positive_lookup)
    veto_hit = _hit_counts(mask_by_frame, veto_lookup)
    broad_hit = _hit_counts(mask_by_frame, broad_lookup)
    visible_count = np.sum(mask_by_frame > 0, axis=0).astype(np.int16)

    variant_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        row, rows = _score_variant(
            scene_id=scene_id,
            variant=variant,
            mask_by_frame=mask_by_frame,
            xyz=xyz,
            frame_df=frame_df,
            positive_obs=positive_obs,
            object_hit=object_hit,
            object_seed_hit=object_seed_hit,
            veto_hit=veto_hit,
            broad_hit=broad_hit,
            visible_count=visible_count,
            obs_meta=obs_meta,
        )
        variant_rows.append(row)
        component_rows.extend(rows)

    variant_path = scene_out / "seed_growth_variant_rows.csv"
    component_path = scene_out / "seed_growth_component_rows.csv"
    _write_csv(variant_path, variant_rows)
    _write_csv(component_path, component_rows)
    best = max(
        variant_rows,
        key=lambda r: (
            bool(r.get("phase9i_seed_growth_gate_pass", False)),
            int(r.get("clean_induced_object_like_obs_count", 0)),
            -float(r.get("clean_induced_broad_obs_rate", 1.0)),
            int(r.get("clean_component_count", 0)),
        ),
    )
    pass_any = any(bool(row["phase9i_seed_growth_gate_pass"]) for row in variant_rows)
    return {
        "schema_version": "stream4d_v103_phase9i_scene_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "phase9e_root": _rel(phase9e_root),
        "da3_projection_manifest": manifest,
        "positive_anchor_mask_observation_count": int(len(positive_obs)),
        "object_like_non_broad_positive_anchor_mask_observation_count": int(len(object_positive_masks)),
        "object_like_non_broad_candidate_mask_observation_count": int(len(object_candidate_masks)),
        "veto_mask_observation_count": int(len(veto_masks)),
        "broad_mask_observation_count": int(len(broad_masks)),
        "gaussian_count": int(mask_by_frame.shape[1]),
        "object_candidate_gaussian_hit1_count": int(np.count_nonzero(object_hit > 0)),
        "object_seed_gaussian_hit1_count": int(np.count_nonzero(object_seed_hit > 0)),
        "object_seed_gaussian_hit2_count": int(np.count_nonzero(object_seed_hit >= 2)),
        "variant_count": len(VARIANTS),
        "phase9i_seed_growth_gate_pass": pass_any,
        "best_variant_id": best["variant_id"],
        "best_clean_component_count": best["clean_component_count"],
        "best_clean_anchor_only_component_count": best["clean_anchor_only_component_count"],
        "best_clean_with_induced_component_count": best["clean_with_induced_component_count"],
        "best_clean_induced_obs_count": best["clean_induced_obs_count"],
        "best_clean_induced_object_like_obs_count": best["clean_induced_object_like_obs_count"],
        "best_clean_induced_broad_obs_count": best["clean_induced_broad_obs_count"],
        "best_clean_induced_broad_obs_rate": best["clean_induced_broad_obs_rate"],
        "blocker": "" if pass_any else "da3_seed_growth_objectlike_graph_does_not_add_reliable_object_like_coverage",
        "uses_gt_for_prediction": False,
        "uses_gt_for_coverage_gate": False,
        "outputs": {
            "seed_growth_variant_rows": _rel(variant_path),
            "seed_growth_component_rows": _rel(component_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose DA3 object-like candidate graph grown from reliable D4RT anchors.")
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument("--projection-cache-root", default=str(AUDIT_ROOT / "v103_phase9g_da3_seed_gaussian_neighborhood_r1"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    phase9e_root = _project_phase(args.phase9e_root)
    out = _project_phase(args.output_root)
    projection_cache_root = _project_phase(args.projection_cache_root) if str(args.projection_cache_root).strip() else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    scenes = list(SCENE_SPECS.keys()) if args.scene == "all" else [str(args.scene)]
    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene in scenes:
        try:
            scene_rows.append(_process_scene(scene, phase9e_root, out, int(args.cupy_device_id), projection_cache_root))
        except Exception as exc:
            failure = {
                "schema_version": "stream4d_v103_phase9i_failure_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "blocker": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "uses_gt_for_prediction": False,
            }
            scene_rows.append(failure)
            failure_rows.append(failure)

    pass_count = sum(bool(row.get("phase9i_seed_growth_gate_pass", False)) for row in scene_rows)
    _write_csv(out / "scene_summary_rows.csv", scene_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    decision = (
        "PASS_PHASE9I_DA3_SEED_GROWTH_OBJECTLIKE_GRAPH"
        if pass_count == len(scenes) and not failure_rows
        else "PARTIAL_PHASE9I_DA3_SEED_GROWTH_OBJECTLIKE_GRAPH"
        if pass_count > 0 and not failure_rows
        else "NO_GO_PHASE9I_DA3_SEED_GROWTH_OBJECTLIKE_GRAPH"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9i_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "scene_count": len(scenes),
        "pass_scene_count": pass_count,
        "failure_count": len(failure_rows),
        "phase9e_root": _rel(phase9e_root),
        "projection_cache_root": "" if projection_cache_root is None else _rel(projection_cache_root),
        "plan_doc": _rel(PLAN_DOC),
        "truthfulness_note": (
            "This diagnostic grows DA3 primitive components from reliable D4RT positive anchors into a GT-free "
            "object-like/non-broad DA3 Gaussian universe. DA3 components are treated as primitive support only, "
            "not as object predictions; no AP predictions are emitted."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scene_summary_rows": _rel(out / "scene_summary_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
