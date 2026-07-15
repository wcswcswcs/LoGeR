#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import (  # noqa: E402
    OBJECT_LIKE_AREA_MIN,
    _load_scene_summary_and_masks,
    _load_semantic,
    _project,
)
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
    _project_phase,
    _rel,
    _write_csv,
    _write_json,
)
from diagnose_v103_phase9i_da3_seed_growth_objectlike_graph import _score_variant as _score_seed_growth  # noqa: E402
from diagnose_v103_phase9j_broad_mask_threshold_relaxation import _mask_rows  # noqa: E402


PHASE_ID = "v103_phase9k_da3_semantic_soft_seed_growth"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9k_da3_semantic_soft_seed_growth_r1"

VARIANTS = [
    {
        "variant_id": "k1_semsoft_allposseed_area020_knn8_q90_obs010_min16",
        "seed_source": "all_positive",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 1.01,
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
        "variant_id": "k2_semsoft_candposseed_area020_knn8_q90_obs010_min16",
        "seed_source": "candidate_positive",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 1.01,
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
        "variant_id": "k3_semsoft_allposseed_area020_veto050_knn8_q90_obs010_min16",
        "seed_source": "all_positive",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 0.50,
        "broad_ratio_max": 1.01,
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
        "variant_id": "k4_semsoft_allposseed_area020_knn12_q75_obs020_min16",
        "seed_source": "all_positive",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 1.01,
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
        "variant_id": "k5_semsoft_allposseed_area020_veto050_knn8_q90_obs005_count4_min8_seed4",
        "seed_source": "all_positive",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 0.50,
        "broad_ratio_max": 1.01,
        "max_universe_gaussians": 250000,
        "component_knn_k": 8,
        "edge_radius_quantile": 0.90,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 8,
        "max_component_gaussians": 80000,
        "min_seed_gaussians_per_component": 4,
        "obs_selected_ratio_min": 0.05,
        "obs_selected_count_min": 4,
    },
    {
        "variant_id": "k6_semsoft_candposseed_area020_veto050_knn8_q90_obs005_count4_min8_seed4",
        "seed_source": "candidate_positive",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 0.50,
        "broad_ratio_max": 1.01,
        "max_universe_gaussians": 250000,
        "component_knn_k": 8,
        "edge_radius_quantile": 0.90,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 8,
        "max_component_gaussians": 80000,
        "min_seed_gaussians_per_component": 4,
        "obs_selected_ratio_min": 0.05,
        "obs_selected_count_min": 4,
    },
    {
        "variant_id": "k7_semsoft_allposseed_area020_veto075_knn8_q90_obs005_count4_min8_seed4",
        "seed_source": "all_positive",
        "object_hit_min": 1,
        "object_ratio_min": 0.03,
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 0.75,
        "broad_ratio_max": 1.01,
        "max_universe_gaussians": 250000,
        "component_knn_k": 8,
        "edge_radius_quantile": 0.90,
        "edge_radius_factor": 2.0,
        "min_component_gaussians": 8,
        "max_component_gaussians": 80000,
        "min_seed_gaussians_per_component": 4,
        "obs_selected_ratio_min": 0.05,
        "obs_selected_count_min": 4,
    },
]


def _semantic_soft_candidates(mask_rows: list[dict[str, Any]], area_max: float = 0.20) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for row in mask_rows:
        area = float(row["area_ratio"])
        if float(OBJECT_LIKE_AREA_MIN) <= area <= float(area_max):
            out.add((int(row["frame_id"]), int(row["mask_id"])))
    return out


def _area_broad_masks(mask_rows: list[dict[str, Any]], area_min: float = 0.20) -> set[tuple[int, int]]:
    return {
        (int(row["frame_id"]), int(row["mask_id"]))
        for row in mask_rows
        if float(row["area_ratio"]) >= float(area_min)
    }


def _obs_meta_soft(scene_id: str, mask_rows: list[dict[str, Any]], candidate_masks: set[tuple[int, int]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in mask_rows:
        key = (int(row["frame_id"]), int(row["mask_id"]))
        obs = f"{scene_id}:{key[0]}:{key[1]}"
        meta[obs] = {
            "frame_id": key[0],
            "mask_id": key[1],
            "is_object_like": bool(key in candidate_masks),
            "is_broad": bool(float(row["area_ratio"]) >= 0.20),
            "semantic_broad_risk": bool(row["semantic_broad_risk"]),
        }
    return meta


def _process_scene(scene_id: str, phase9e_root: Path, out: Path, projection_cache_root: Path | None) -> dict[str, Any]:
    spec = dict(SCENE_SPECS[scene_id])
    spec["phase2_root"] = _project(spec["phase2_root"])
    _summary, frame_ids, masks = _load_scene_summary_and_masks(scene_id, spec["phase2_root"])
    _feature_index, _features, semantic_meta, semantic_constants = _load_semantic(scene_id, spec)
    mask_rows = _mask_rows(scene_id, frame_ids, masks, semantic_meta)
    candidate_masks = _semantic_soft_candidates(mask_rows, area_max=0.20)
    broad_masks = _area_broad_masks(mask_rows, area_min=0.20)
    positive_masks, veto_masks = _load_phase9e_supports(phase9e_root, scene_id)
    positive_obs = {f"{scene_id}:{frame_id}:{mask_id}" for frame_id, mask_id in positive_masks}
    mask_by_frame, xyz, frame_df, _meta, manifest = _load_or_project_da3(scene_id, out, projection_cache_root)
    obs_meta = _obs_meta_soft(scene_id, mask_rows, candidate_masks)
    max_label = int(np.max(mask_by_frame)) if mask_by_frame.size else 0
    object_lookup = _frame_label_lookup(frame_df, candidate_masks, max_label)
    veto_lookup = _frame_label_lookup(frame_df, veto_masks, max_label)
    broad_lookup = _frame_label_lookup(frame_df, broad_masks, max_label)
    object_hit = _hit_counts(mask_by_frame, object_lookup)
    veto_hit = _hit_counts(mask_by_frame, veto_lookup)
    broad_hit = _hit_counts(mask_by_frame, broad_lookup)
    visible_count = np.sum(mask_by_frame > 0, axis=0).astype(np.int16)

    variant_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        seed_masks = positive_masks if str(variant["seed_source"]) == "all_positive" else (positive_masks & candidate_masks)
        seed_lookup = _frame_label_lookup(frame_df, seed_masks, max_label)
        seed_hit = _hit_counts(mask_by_frame, seed_lookup)
        row, rows = _score_seed_growth(
            scene_id=scene_id,
            variant=variant,
            mask_by_frame=mask_by_frame,
            xyz=xyz,
            frame_df=frame_df,
            positive_obs=positive_obs,
            object_hit=object_hit,
            object_seed_hit=seed_hit,
            veto_hit=veto_hit,
            broad_hit=broad_hit,
            visible_count=visible_count,
            obs_meta=obs_meta,
        )
        row["schema_version"] = "stream4d_v103_phase9k_variant_row_v1"
        row["phase_id"] = PHASE_ID
        row["semantic_broad_mode"] = "soft_candidate_risk_only"
        row["candidate_mask_observation_count"] = int(len(candidate_masks))
        row["candidate_positive_anchor_overlap_count"] = int(len(candidate_masks & positive_masks))
        row["candidate_not_positive_anchor_count"] = int(len({f"{scene_id}:{a}:{b}" for a, b in candidate_masks} - positive_obs))
        variant_rows.append(row)
        for comp in rows:
            comp["schema_version"] = "stream4d_v103_phase9k_component_row_v1"
            comp["phase_id"] = PHASE_ID
            component_rows.append(comp)

    scene_out = out / scene_id
    variant_path = scene_out / "semantic_soft_seed_growth_variant_rows.csv"
    component_path = scene_out / "semantic_soft_seed_growth_component_rows.csv"
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
        "schema_version": "stream4d_v103_phase9k_scene_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "phase9e_root": _rel(phase9e_root),
        "da3_projection_manifest": manifest,
        "semantic_constants": semantic_constants,
        "positive_anchor_mask_observation_count": int(len(positive_masks)),
        "candidate_mask_observation_count": int(len(candidate_masks)),
        "candidate_positive_anchor_overlap_count": int(len(candidate_masks & positive_masks)),
        "candidate_not_positive_anchor_count": int(len({f"{scene_id}:{a}:{b}" for a, b in candidate_masks} - positive_obs)),
        "veto_mask_observation_count": int(len(veto_masks)),
        "area_broad_mask_observation_count": int(len(broad_masks)),
        "gaussian_count": int(mask_by_frame.shape[1]),
        "variant_count": len(VARIANTS),
        "phase9k_semantic_soft_seed_growth_gate_pass": pass_any,
        "best_variant_id": best["variant_id"],
        "best_clean_component_count": best["clean_component_count"],
        "best_clean_anchor_only_component_count": best["clean_anchor_only_component_count"],
        "best_clean_with_induced_component_count": best["clean_with_induced_component_count"],
        "best_clean_induced_obs_count": best["clean_induced_obs_count"],
        "best_clean_induced_object_like_obs_count": best["clean_induced_object_like_obs_count"],
        "best_clean_induced_broad_obs_count": best["clean_induced_broad_obs_count"],
        "best_clean_induced_broad_obs_rate": best["clean_induced_broad_obs_rate"],
        "blocker": "" if pass_any else "semantic_soft_candidates_do_not_form_clean_induced_objectlike_components",
        "uses_gt_for_prediction": False,
        "uses_gt_for_gate": False,
        "outputs": {
            "semantic_soft_seed_growth_variant_rows": _rel(variant_path),
            "semantic_soft_seed_growth_component_rows": _rel(component_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DA3 seed-growth with semantic-broad masks treated as soft risk instead of hard veto.")
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument("--projection-cache-root", default=str(AUDIT_ROOT / "v103_phase9g_da3_seed_gaussian_neighborhood_r1"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
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
            scene_rows.append(_process_scene(scene, phase9e_root, out, projection_cache_root))
        except Exception as exc:
            failure = {
                "schema_version": "stream4d_v103_phase9k_failure_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "blocker": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "uses_gt_for_prediction": False,
            }
            scene_rows.append(failure)
            failure_rows.append(failure)
    pass_count = sum(bool(row.get("phase9k_semantic_soft_seed_growth_gate_pass", False)) for row in scene_rows)
    _write_csv(out / "scene_summary_rows.csv", scene_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    decision = (
        "PASS_PHASE9K_DA3_SEMANTIC_SOFT_SEED_GROWTH"
        if pass_count == len(scenes) and not failure_rows
        else "PARTIAL_PHASE9K_DA3_SEMANTIC_SOFT_SEED_GROWTH"
        if pass_count > 0 and not failure_rows
        else "NO_GO_PHASE9K_DA3_SEMANTIC_SOFT_SEED_GROWTH"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9k_summary_v1",
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
            "This diagnostic treats semantic broad/background flags as soft risk for the candidate mask universe, "
            "while keeping DA3 components as primitive support only. It does not emit AP predictions."
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
