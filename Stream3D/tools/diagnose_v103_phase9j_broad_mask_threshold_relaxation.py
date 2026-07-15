#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


PHASE_ID = "v103_phase9j_broad_mask_threshold_relaxation"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9j_broad_mask_threshold_relaxation_r1"

THRESHOLD_VARIANTS = [
    {"variant_id": "t012_orig_area012_objmax020_semhard", "broad_area_ratio": 0.12, "object_like_area_max": 0.20, "semantic_broad_hard": True},
    {"variant_id": "t016_area016_objmax020_semhard", "broad_area_ratio": 0.16, "object_like_area_max": 0.20, "semantic_broad_hard": True},
    {"variant_id": "t020_area020_objmax020_semhard", "broad_area_ratio": 0.20, "object_like_area_max": 0.20, "semantic_broad_hard": True},
    {"variant_id": "t025_area025_objmax025_semhard", "broad_area_ratio": 0.25, "object_like_area_max": 0.25, "semantic_broad_hard": True},
    {"variant_id": "t030_area030_objmax030_semhard", "broad_area_ratio": 0.30, "object_like_area_max": 0.30, "semantic_broad_hard": True},
    {"variant_id": "t020_area020_objmax020_semsoft", "broad_area_ratio": 0.20, "object_like_area_max": 0.20, "semantic_broad_hard": False},
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mask_rows(scene_id: str, frame_ids: list[int], masks: np.ndarray, semantic_meta: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h, w = masks.shape[1:]
    denom = float(max(h * w, 1))
    for fi, frame_id in enumerate(frame_ids):
        labels, counts = np.unique(masks[fi], return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            label = int(label)
            if label <= 0:
                continue
            area_ratio = float(count) / denom
            sem = semantic_meta.get((int(frame_id), label), {})
            semantic_broad = bool(sem.get("broad_background_risk")) or bool(sem.get("semantic_background_score_proxy"))
            rows.append(
                {
                    "scene_id": scene_id,
                    "frame_id": int(frame_id),
                    "mask_id": label,
                    "mask_observation_id": f"{scene_id}:{int(frame_id)}:{label}",
                    "pixel_count": int(count),
                    "area_ratio": area_ratio,
                    "semantic_broad_risk": bool(semantic_broad),
                    "broad_background_risk": bool(sem.get("broad_background_risk")),
                    "semantic_background_score_proxy": bool(sem.get("semantic_background_score_proxy")),
                    "semantic_boundary_variance": float(sem.get("semantic_boundary_variance") or 0.0),
                    "semantic_entropy": float(sem.get("semantic_entropy") or 0.0),
                }
            )
    return rows


def _candidate_set(mask_rows: list[dict[str, Any]], variant: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    broad_area = float(variant["broad_area_ratio"])
    obj_max = float(variant["object_like_area_max"])
    semantic_hard = bool(variant["semantic_broad_hard"])
    for row in mask_rows:
        area = float(row["area_ratio"])
        semantic_broad = bool(row["semantic_broad_risk"])
        broad = (area >= broad_area) or (semantic_broad if semantic_hard else False)
        object_like = (float(OBJECT_LIKE_AREA_MIN) <= area <= obj_max) and not broad
        if object_like:
            out.add((int(row["frame_id"]), int(row["mask_id"])))
    return out


def _obs_ids(scene_id: str, masks: set[tuple[int, int]]) -> set[str]:
    return {f"{scene_id}:{int(frame_id)}:{int(mask_id)}" for frame_id, mask_id in masks}


def _area_stats(mask_rows: list[dict[str, Any]], masks: set[tuple[int, int]]) -> dict[str, Any]:
    areas = [
        float(row["area_ratio"])
        for row in mask_rows
        if (int(row["frame_id"]), int(row["mask_id"])) in masks
    ]
    if not areas:
        return {"area_ratio_p50": 0.0, "area_ratio_p90": 0.0, "area_ratio_p95": 0.0, "area_ratio_max": 0.0}
    arr = np.asarray(areas, dtype=np.float64)
    return {
        "area_ratio_p50": float(np.quantile(arr, 0.50)),
        "area_ratio_p90": float(np.quantile(arr, 0.90)),
        "area_ratio_p95": float(np.quantile(arr, 0.95)),
        "area_ratio_max": float(np.max(arr)),
    }


def _process_scene(scene_id: str, phase9e_root: Path, out: Path, projection_cache_root: Path | None) -> dict[str, Any]:
    spec = dict(SCENE_SPECS[scene_id])
    spec["phase2_root"] = _project(spec["phase2_root"])
    _summary, frame_ids, masks = _load_scene_summary_and_masks(scene_id, spec["phase2_root"])
    _feature_index, _features, semantic_meta, semantic_constants = _load_semantic(scene_id, spec)
    rows = _mask_rows(scene_id, frame_ids, masks, semantic_meta)
    positive_masks, veto_masks = _load_phase9e_supports(phase9e_root, scene_id)
    mask_by_frame, _xyz, frame_df, _meta, manifest = _load_or_project_da3(scene_id, out, projection_cache_root)
    max_label = int(np.max(mask_by_frame)) if mask_by_frame.size else 0

    original_candidate = _candidate_set(rows, THRESHOLD_VARIANTS[0])
    original_candidate_obs = _obs_ids(scene_id, original_candidate)
    positive_obs = _obs_ids(scene_id, positive_masks)
    variant_rows: list[dict[str, Any]] = []
    candidate_detail_rows: list[dict[str, Any]] = []
    for variant in THRESHOLD_VARIANTS:
        cand = _candidate_set(rows, variant)
        cand_obs = _obs_ids(scene_id, cand)
        extra = cand - original_candidate
        extra_obs = _obs_ids(scene_id, extra)
        non_anchor_extra = extra_obs - positive_obs
        lookup = _frame_label_lookup(frame_df, cand, max_label)
        extra_lookup = _frame_label_lookup(frame_df, extra, max_label)
        hit = _hit_counts(mask_by_frame, lookup)
        extra_hit = _hit_counts(mask_by_frame, extra_lookup)
        semantic_broad_count = sum(
            1
            for row in rows
            if (int(row["frame_id"]), int(row["mask_id"])) in cand and bool(row["semantic_broad_risk"])
        )
        row = {
            "schema_version": "stream4d_v103_phase9j_threshold_variant_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "variant_id": str(variant["variant_id"]),
            "broad_area_ratio": float(variant["broad_area_ratio"]),
            "object_like_area_max": float(variant["object_like_area_max"]),
            "semantic_broad_hard": bool(variant["semantic_broad_hard"]),
            "candidate_mask_observation_count": int(len(cand)),
            "candidate_semantic_broad_count": int(semantic_broad_count),
            "positive_anchor_overlap_count": int(len(cand_obs & positive_obs)),
            "candidate_not_positive_anchor_count": int(len(cand_obs - positive_obs)),
            "extra_vs_original_candidate_count": int(len(extra)),
            "extra_vs_original_not_positive_anchor_count": int(len(non_anchor_extra)),
            "da3_candidate_gaussian_hit1_count": int(np.count_nonzero(hit > 0)),
            "da3_extra_gaussian_hit1_count": int(np.count_nonzero(extra_hit > 0)),
            "veto_overlap_count": int(len(cand & veto_masks)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_gate": False,
        }
        row.update({f"candidate_{k}": v for k, v in _area_stats(rows, cand).items()})
        row.update({f"extra_{k}": v for k, v in _area_stats(rows, extra).items()})
        variant_rows.append(row)
        for detail in rows:
            key = (int(detail["frame_id"]), int(detail["mask_id"]))
            if key not in extra:
                continue
            out_row = {
                "schema_version": "stream4d_v103_phase9j_extra_candidate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "variant_id": str(variant["variant_id"]),
                "is_positive_anchor": bool(f"{scene_id}:{key[0]}:{key[1]}" in positive_obs),
                "is_veto_support": bool(key in veto_masks),
                "uses_gt_for_prediction": False,
            }
            out_row.update(detail)
            candidate_detail_rows.append(out_row)

    scene_out = out / scene_id
    variant_path = scene_out / "threshold_variant_rows.csv"
    detail_path = scene_out / "extra_candidate_rows.csv"
    _write_csv(variant_path, variant_rows)
    _write_csv(detail_path, candidate_detail_rows)
    best = max(
        variant_rows,
        key=lambda r: (
            int(r["extra_vs_original_not_positive_anchor_count"]),
            int(r["da3_extra_gaussian_hit1_count"]),
            -int(r["candidate_semantic_broad_count"]),
        ),
    )
    method_candidate = bool(int(best["extra_vs_original_not_positive_anchor_count"]) >= 20 and int(best["da3_extra_gaussian_hit1_count"]) >= 10000)
    return {
        "schema_version": "stream4d_v103_phase9j_scene_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "phase9e_root": _rel(phase9e_root),
        "da3_projection_manifest": manifest,
        "mask_observation_count": int(len(rows)),
        "original_candidate_mask_observation_count": int(len(original_candidate)),
        "positive_anchor_mask_observation_count": int(len(positive_masks)),
        "veto_mask_observation_count": int(len(veto_masks)),
        "best_relaxation_variant_id": best["variant_id"],
        "best_extra_vs_original_candidate_count": best["extra_vs_original_candidate_count"],
        "best_extra_vs_original_not_positive_anchor_count": best["extra_vs_original_not_positive_anchor_count"],
        "best_da3_extra_gaussian_hit1_count": best["da3_extra_gaussian_hit1_count"],
        "phase9j_relaxed_objectlike_universe_has_actionable_extra_candidates": method_candidate,
        "blocker": "" if method_candidate else "relaxing_broad_area_threshold_does_not_create_enough_new_non_anchor_objectlike_candidates",
        "semantic_constants": semantic_constants,
        "uses_gt_for_prediction": False,
        "uses_gt_for_gate": False,
        "outputs": {
            "threshold_variant_rows": _rel(variant_path),
            "extra_candidate_rows": _rel(detail_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose whether v103 broad-mask area threshold hides actionable object-like candidates.")
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
    _write_text(out / "last_command.txt", " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n")
    scenes = list(SCENE_SPECS.keys()) if args.scene == "all" else [str(args.scene)]
    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene in scenes:
        try:
            scene_rows.append(_process_scene(scene, phase9e_root, out, projection_cache_root))
        except Exception as exc:
            failure = {
                "schema_version": "stream4d_v103_phase9j_failure_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "blocker": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "uses_gt_for_prediction": False,
            }
            scene_rows.append(failure)
            failure_rows.append(failure)
    pass_count = sum(bool(row.get("phase9j_relaxed_objectlike_universe_has_actionable_extra_candidates", False)) for row in scene_rows)
    _write_csv(out / "scene_summary_rows.csv", scene_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    decision = (
        "PASS_PHASE9J_RELAXED_BROAD_THRESHOLD_HAS_ACTIONABLE_CANDIDATES"
        if pass_count == len(scenes) and not failure_rows
        else "PARTIAL_PHASE9J_RELAXED_BROAD_THRESHOLD_HAS_ACTIONABLE_CANDIDATES"
        if pass_count > 0 and not failure_rows
        else "NO_GO_PHASE9J_RELAXED_BROAD_THRESHOLD_NO_ACTIONABLE_CANDIDATES"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9j_summary_v1",
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
            "This diagnostic is GT-free. It audits whether the v103 broad-area threshold hides enough new "
            "object-like/non-broad mask observations to justify a relaxed candidate universe. It does not emit AP predictions."
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
