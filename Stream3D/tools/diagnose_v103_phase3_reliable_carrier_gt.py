#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import (  # noqa: E402
    SEMANTIC_CONTRADICTION_THRESHOLD,
    ALL_SUPPORT_BALANCED_VARIANTS,
    _apply_support_balanced_backfill,
    _compute_scene_arrays,
    _ensure_mmap_cache,
    _load_cached,
    _project,
    _variant_hard_ok,
    _variant_scores_and_candidate,
)


PHASE_ID = "v103_phase3_reliable_carrier_gt_diagnostic"
DEFAULT_PHASE3_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase3_carrier_reliability_filter_q5c_objlike16384_fast_support_balanced"
DEFAULT_PHASE4_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase4_primitive_affinity_q5c_support_balanced_r7_object_preserve_downweight"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase3_reliable_carrier_gt_diagnostic"
DEFAULT_SCENE0011_PHASE2 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_SCENE0050_PHASE2 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0050_first32"
SOURCE_CODEBOOK = {
    1: "uniform_grid",
    2: "mask_interior",
    3: "mask_boundary_band",
    4: "competing_mask_boundary",
    5: "semantic_gradient",
    6: "high_risk_broad_mask_interior",
    7: "overlap_frame_anchor",
}


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_by_id(variant_id: str) -> dict[str, Any]:
    for variant in ALL_SUPPORT_BALANCED_VARIANTS:
        if str(variant["variant_id"]) == str(variant_id):
            return dict(variant)
    raise KeyError(f"unsupported support-balanced variant: {variant_id}")


def _retained_phase3_semantics(variant: dict[str, Any], arrays: dict[str, np.ndarray], diag: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    score_key = "reliability_s2" if bool(variant["semantic"]) else "reliability_s0"
    scores, candidate = _variant_scores_and_candidate(variant, arrays)

    n = int(scores.shape[0])
    keep_n = max(1, int(round(float(variant["top_rate"]) * n)))
    candidate_count_before_hard = int(np.count_nonzero(candidate))
    if keep_n >= n:
        retained = np.ones((n,), dtype=bool)
        threshold = float(np.min(scores))
    elif candidate_count_before_hard <= keep_n:
        retained = candidate.copy()
        threshold = float(np.min(scores[retained])) if np.any(retained) else -float("inf")
    else:
        order = np.argpartition(scores, n - keep_n)
        keep = order[n - keep_n :]
        retained = np.zeros((n,), dtype=bool)
        retained[keep] = True
        threshold = float(np.min(scores[keep]))
    if bool(variant.get("hard_veto")):
        hard_ok = _variant_hard_ok(variant, arrays)
        retained &= hard_ok
        candidate &= hard_ok
    retained_before_backfill = int(np.count_nonzero(retained))
    retained, added_object, added_boundary = _apply_support_balanced_backfill(
        diag=diag,
        scores=scores,
        candidate=candidate,
        retained=retained,
        min_object_like_support_per_mask=int(variant.get("min_object_like_support_per_mask", 0)),
        min_boundary_support_per_mask=int(variant.get("min_boundary_support_per_mask", 0)),
    )
    return retained, {
        "score_key": score_key,
        "threshold": threshold,
        "candidate_count_before_hard_veto": candidate_count_before_hard,
        "candidate_count_after_hard_veto": int(np.count_nonzero(candidate)),
        "retained_count_before_backfill": retained_before_backfill,
        "retained_count_after_backfill": int(np.count_nonzero(retained)),
        "support_backfill_added_object": int(added_object),
        "support_backfill_added_boundary": int(added_boundary),
    }


def _read_label_png(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != shape_hw:
        image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image, dtype=np.int64)


def _load_gt_stack(scene: str, frame_ids: list[int], shape_hw: tuple[int, int]) -> np.ndarray:
    frames: list[np.ndarray] = []
    root = STREAM3D_ROOT / "data/scannet/processed" / scene / "instance/instance"
    for frame_id in frame_ids:
        frames.append(_read_label_png(root / f"{int(frame_id)}.png", shape_hw))
    return np.stack(frames, axis=0).astype(np.int64, copy=False)


def _carrier_gt_stats(
    *,
    batch: dict[str, np.ndarray],
    gt_stack: np.ndarray,
    visible_threshold: float,
    confidence_threshold: float,
) -> dict[str, np.ndarray]:
    uv = np.asarray(batch["uv_pred"], dtype=np.float32)
    valid = np.asarray(batch["valid"], dtype=bool)
    visibility = np.asarray(batch["visibility_prob"], dtype=np.float32)
    confidence = np.asarray(batch["confidence_prob"], dtype=np.float32)
    frame_count, carrier_count = valid.shape
    height, width = gt_stack.shape[1:]
    carrier_parts: list[np.ndarray] = []
    gt_parts: list[np.ndarray] = []
    visible_count = np.zeros((carrier_count,), dtype=np.int32)
    for fi in range(frame_count):
        u = uv[fi, :, 0]
        v = uv[fi, :, 1]
        ok = (
            valid[fi]
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0.0)
            & (u <= 1.0)
            & (v >= 0.0)
            & (v <= 1.0)
            & (visibility[fi] >= float(visible_threshold))
            & (confidence[fi] >= float(confidence_threshold))
        )
        idx = np.flatnonzero(ok).astype(np.int64)
        if idx.size == 0:
            continue
        visible_count += np.bincount(idx, minlength=carrier_count).astype(np.int32)
        xs = np.rint(u[idx] * float(max(width - 1, 1))).astype(np.int64)
        ys = np.rint(v[idx] * float(max(height - 1, 1))).astype(np.int64)
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)
        gt = gt_stack[fi, ys, xs].astype(np.int64)
        positive = gt > 0
        if np.any(positive):
            carrier_parts.append(idx[positive].astype(np.int64, copy=False))
            gt_parts.append(gt[positive].astype(np.int64, copy=False))

    positive_count = np.zeros((carrier_count,), dtype=np.int32)
    unique_gt_count = np.zeros((carrier_count,), dtype=np.int16)
    dominant_count = np.zeros((carrier_count,), dtype=np.int32)
    dominant_gt = np.zeros((carrier_count,), dtype=np.int64)
    if carrier_parts:
        carrier_idx = np.concatenate(carrier_parts, axis=0)
        gt_label = np.concatenate(gt_parts, axis=0)
        positive_count = np.bincount(carrier_idx, minlength=carrier_count).astype(np.int32)
        base = int(np.max(gt_label)) + 1
        encoded = carrier_idx.astype(np.int64) * np.int64(base) + gt_label.astype(np.int64)
        uniq, counts = np.unique(encoded, return_counts=True)
        pair_carrier = (uniq // base).astype(np.int64)
        pair_gt = (uniq % base).astype(np.int64)
        unique_gt_count = np.bincount(pair_carrier, minlength=carrier_count).astype(np.int16)
        order = np.lexsort((-counts, pair_carrier))
        seen: set[int] = set()
        for idx in order.tolist():
            ci = int(pair_carrier[idx])
            if ci in seen:
                continue
            seen.add(ci)
            dominant_count[ci] = int(counts[idx])
            dominant_gt[ci] = int(pair_gt[idx])
    purity = np.divide(
        dominant_count.astype(np.float64),
        np.maximum(positive_count.astype(np.float64), 1.0),
        out=np.zeros((carrier_count,), dtype=np.float64),
        where=positive_count > 0,
    )
    return {
        "visible_count": visible_count,
        "gt_positive_count": positive_count,
        "unique_gt_count": unique_gt_count,
        "dominant_gt_count": dominant_count,
        "dominant_gt_id": dominant_gt,
        "dominant_gt_purity": purity.astype(np.float32),
    }


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _metric_for_scope(
    *,
    scene: str,
    scope_name: str,
    mask: np.ndarray,
    gt_stats: dict[str, np.ndarray],
    min_gt_positive_obs: int,
    all_clean_carriers: np.ndarray,
    all_positive_carriers: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    visible = np.asarray(gt_stats["visible_count"], dtype=np.int64)
    positive = np.asarray(gt_stats["gt_positive_count"], dtype=np.int64)
    unique_gt = np.asarray(gt_stats["unique_gt_count"], dtype=np.int64)
    purity = np.asarray(gt_stats["dominant_gt_purity"], dtype=np.float64)
    dominant_gt = np.asarray(gt_stats["dominant_gt_id"], dtype=np.int64)
    eligible = mask & (positive >= int(min_gt_positive_obs))
    clean = eligible & (unique_gt == 1)
    multi = eligible & (unique_gt >= 2)
    visible_obs = int(np.sum(visible[mask]))
    pos_obs = int(np.sum(positive[mask]))
    clean_counts = np.bincount(dominant_gt[clean], minlength=int(np.max(dominant_gt) + 1) if dominant_gt.size else 1)
    covered = clean_counts[clean_counts > 0]
    return {
        "schema_version": "stream4d_v103_phase3_gt_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "carrier_scope": scope_name,
        "carrier_count": int(np.count_nonzero(mask)),
        "gt_positive_carrier_count": int(np.count_nonzero(eligible)),
        "gt_positive_carrier_rate": float(np.count_nonzero(eligible) / max(1, int(np.count_nonzero(mask)))),
        "visible_observation_count": visible_obs,
        "gt_positive_observation_count": pos_obs,
        "gt_positive_observation_rate": float(pos_obs / max(1, visible_obs)),
        "dominant_gt_purity_mean": float(np.mean(purity[eligible])) if np.any(eligible) else 0.0,
        "dominant_gt_purity_p10": _percentile(purity[eligible], 10),
        "dominant_gt_purity_p50": _percentile(purity[eligible], 50),
        "carrier_multi_GT_rate": float(np.mean(unique_gt[eligible] >= 2)) if np.any(eligible) else 0.0,
        "carrier_clean_single_GT_rate": float(np.mean(unique_gt[eligible] == 1)) if np.any(eligible) else 0.0,
        "clean_same_GT_carrier_recall_proxy": float(np.count_nonzero(mask & all_clean_carriers) / max(1, int(np.count_nonzero(all_clean_carriers)))),
        "gt_positive_carrier_recall_proxy": float(np.count_nonzero(mask & all_positive_carriers) / max(1, int(np.count_nonzero(all_positive_carriers)))),
        "selected_clean_gt_instance_count": int(np.count_nonzero(clean_counts > 0)),
        "selected_clean_gt_instance_ge5_count": int(np.count_nonzero(clean_counts >= 5)),
        "selected_clean_gt_instance_ge10_count": int(np.count_nonzero(clean_counts >= 10)),
        "selected_clean_carriers_per_gt_p10": _percentile(covered, 10),
        "selected_clean_carriers_per_gt_p50": _percentile(covered, 50),
        "diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
    }


def _phase4_carrier_parity(phase4_root: Path, scene: str, selected_carrier_ids: np.ndarray) -> dict[str, Any]:
    path = phase4_root / scene / "primitive_affinity_feature.pt"
    if not path.exists():
        return {
            "schema_version": "stream4d_v103_phase3_phase4_retained_parity_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "phase4_feature_path": _rel(path),
            "phase4_feature_exists": False,
            "phase3_retained_count": int(selected_carrier_ids.shape[0]),
            "phase4_carrier_count": 0,
            "retained_set_exact_match": False,
        }
    import torch

    payload = torch.load(path, map_location="cpu")
    phase4_ids = payload["carrier_id"].cpu().numpy().astype(np.int64)
    selected = np.asarray(selected_carrier_ids, dtype=np.int64)
    missing = int(np.count_nonzero(~np.isin(selected, phase4_ids)))
    extra = int(np.count_nonzero(~np.isin(phase4_ids, selected)))
    return {
        "schema_version": "stream4d_v103_phase3_phase4_retained_parity_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "phase4_feature_path": _rel(path),
        "phase4_feature_exists": True,
        "phase3_retained_count": int(selected.shape[0]),
        "phase4_carrier_count": int(phase4_ids.shape[0]),
        "missing_phase3_retained_in_phase4_count": missing,
        "extra_phase4_not_in_phase3_retained_count": extra,
        "retained_set_exact_match": bool(missing == 0 and extra == 0),
        "phase4_selected_phase3_variant": payload.get("selected_phase3_variant", ""),
    }


def _scene_specs(scene0011_phase2: str, scene0050_phase2: str) -> dict[str, dict[str, Path]]:
    audit = STREAM3D_ROOT / "outputs/audit"
    return {
        "scene0011_00": {
            "phase2_root": _project(scene0011_phase2),
            "semantic_npz": audit / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
            "semantic_rows": audit / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
        },
        "scene0050_00": {
            "phase2_root": _project(scene0050_phase2),
            "semantic_npz": audit / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
            "semantic_rows": audit / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase3 reliable-carrier GT diagnostic. GT is diagnostic-only.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--scene0011-selected-variant-id", default="")
    parser.add_argument("--scene0050-selected-variant-id", default="")
    parser.add_argument("--visible-threshold", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-gt-positive-obs", type=int, default=2)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument(
        "--diagnose-all-phase3-variants",
        action="store_true",
        help="GT-only sweep over pre-registered Phase3 variants from the Phase3 summary; not used for method selection.",
    )
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase3_root = _project(args.phase3_root)
    phase4_root = _project(args.phase4_root)
    phase3_summary = _read_json(phase3_root / "summary.json")
    selected_by_scene = {str(k): str(v) for k, v in dict(phase3_summary["selected_variant_by_scene"]).items()}
    selected_override_by_scene = {
        "scene0011_00": str(args.scene0011_selected_variant_id),
        "scene0050_00": str(args.scene0050_selected_variant_id),
    }
    selected_override_by_scene = {scene: variant for scene, variant in selected_override_by_scene.items() if variant}
    selected_by_scene.update(selected_override_by_scene)
    summary_variant_ids = [str(v) for v in phase3_summary.get("variant_ids", phase3_summary.get("evaluated_variant_ids", []))]
    specs = _scene_specs(args.scene0011_phase2_root, args.scene0050_phase2_root)

    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    scene_summary_rows: list[dict[str, Any]] = []
    diagnosed_variant_ids_by_scene: dict[str, list[str]] = {}
    for scene, spec in specs.items():
        scene_out = out / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, spec, scene_out, int(args.cupy_device_id))
        variant_id = selected_by_scene[scene]
        if args.diagnose_all_phase3_variants:
            variant_ids = list(dict.fromkeys([*summary_variant_ids, variant_id]))
        else:
            variant_ids = [variant_id]
        diagnosed_variant_ids_by_scene[scene] = variant_ids
        cache_dir, _manifest = _ensure_mmap_cache(spec["phase2_root"])
        batch = _load_cached(cache_dir)
        gt_stack = _load_gt_stack(scene, [int(v) for v in diag["frame_ids"]], tuple(diag["masks"].shape[1:]))
        gt_stats = _carrier_gt_stats(
            batch=batch,
            gt_stack=gt_stack,
            visible_threshold=float(args.visible_threshold),
            confidence_threshold=float(args.confidence_threshold),
        )
        positive = np.asarray(gt_stats["gt_positive_count"], dtype=np.int64)
        unique_gt = np.asarray(gt_stats["unique_gt_count"], dtype=np.int64)
        all_positive = positive >= int(args.min_gt_positive_obs)
        all_clean = all_positive & (unique_gt == 1)
        all_mask = np.ones((positive.shape[0],), dtype=bool)
        metric_rows.append(
            _metric_for_scope(
                scene=scene,
                scope_name="unfiltered_all_q5c_carriers",
                mask=all_mask,
                gt_stats=gt_stats,
                min_gt_positive_obs=int(args.min_gt_positive_obs),
                all_clean_carriers=all_clean,
                all_positive_carriers=all_positive,
            )
        )
        query_source = np.asarray(arrays["query_source_code"], dtype=np.int16)
        broad_rate = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
        sem_bad = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
        jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
        for code in sorted(SOURCE_CODEBOOK):
            src_mask = all_mask & (query_source == int(code))
            if not np.any(src_mask):
                continue
            row = _metric_for_scope(
                scene=scene,
                scope_name=f"unfiltered_all_q5c_carriers__source_{SOURCE_CODEBOOK[code]}",
                mask=src_mask,
                gt_stats=gt_stats,
                min_gt_positive_obs=int(args.min_gt_positive_obs),
                all_clean_carriers=all_clean,
                all_positive_carriers=all_positive,
            )
            row["query_source_code"] = int(code)
            row["query_source"] = SOURCE_CODEBOOK[code]
            row["mean_broad_mask_participation_rate"] = float(np.mean(broad_rate[src_mask]))
            row["mean_semantic_contradiction_rate"] = float(np.mean(sem_bad[src_mask]))
            row["normalized_jitter_p90"] = float(np.percentile(jitter[src_mask], 90))
            source_rows.append(row)

        selected_retained: np.ndarray | None = None
        selected_retain_meta: dict[str, Any] | None = None
        for candidate_variant_id in variant_ids:
            variant = _variant_by_id(candidate_variant_id)
            retained, retain_meta = _retained_phase3_semantics(variant, arrays, diag)
            is_selected = str(candidate_variant_id) == str(variant_id)
            scope_prefix = "phase3_selected" if is_selected else "phase3_candidate"
            row = _metric_for_scope(
                scene=scene,
                scope_name=f"{scope_prefix}_{candidate_variant_id}",
                mask=retained,
                gt_stats=gt_stats,
                min_gt_positive_obs=int(args.min_gt_positive_obs),
                all_clean_carriers=all_clean,
                all_positive_carriers=all_positive,
            )
            row["phase3_variant_id"] = str(candidate_variant_id)
            row["is_selected_phase3_variant"] = bool(is_selected)
            metric_rows.append(row)

            for code in sorted(SOURCE_CODEBOOK):
                src_mask = np.asarray(retained, dtype=bool) & (query_source == int(code))
                if not np.any(src_mask):
                    continue
                source_row = _metric_for_scope(
                    scene=scene,
                    scope_name=f"{scope_prefix}_{candidate_variant_id}__source_{SOURCE_CODEBOOK[code]}",
                    mask=src_mask,
                    gt_stats=gt_stats,
                    min_gt_positive_obs=int(args.min_gt_positive_obs),
                    all_clean_carriers=all_clean,
                    all_positive_carriers=all_positive,
                )
                source_row["phase3_variant_id"] = str(candidate_variant_id)
                source_row["is_selected_phase3_variant"] = bool(is_selected)
                source_row["query_source_code"] = int(code)
                source_row["query_source"] = SOURCE_CODEBOOK[code]
                source_row["mean_broad_mask_participation_rate"] = float(np.mean(broad_rate[src_mask]))
                source_row["mean_semantic_contradiction_rate"] = float(np.mean(sem_bad[src_mask]))
                source_row["normalized_jitter_p90"] = float(np.percentile(jitter[src_mask], 90))
                source_rows.append(source_row)

            scene_summary_rows.append(
                {
                    "schema_version": "stream4d_v103_phase3_gt_scene_summary_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "selected_variant_id": variant_id,
                    "diagnosed_variant_id": str(candidate_variant_id),
                    "is_selected_phase3_variant": bool(is_selected),
                    "total_carrier_count": int(retained.shape[0]),
                    "phase3_retained_count": int(np.count_nonzero(retained)),
                    **retain_meta,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": True,
                    "diagnostic_only": True,
                }
            )
            if is_selected:
                selected_retained = retained
                selected_retain_meta = retain_meta

        if selected_retained is None or selected_retain_meta is None:
            raise RuntimeError(f"selected Phase3 variant {variant_id} was not diagnosed for {scene}")
        carrier_ids = np.asarray(batch["carrier_id"], dtype=np.int64)
        selected_ids = carrier_ids[np.flatnonzero(selected_retained)]
        parity_rows.append(_phase4_carrier_parity(phase4_root, scene, selected_ids))

    diagnostic_variant_rows = [
        row
        for row in metric_rows
        if row["carrier_scope"].startswith("phase3_selected_") or row["carrier_scope"].startswith("phase3_candidate_")
    ]
    selected_rows = [row for row in diagnostic_variant_rows if bool(row.get("is_selected_phase3_variant"))]
    unfiltered_rows = {row["scene_id"]: row for row in metric_rows if row["carrier_scope"] == "unfiltered_all_q5c_carriers"}
    gt_diag_ok_by_scene: dict[str, bool] = {}
    any_gt_diag_ok_by_scene: dict[str, bool] = {}
    best_diagnostic_variant_by_scene: dict[str, dict[str, Any]] = {}
    failure_rows: list[dict[str, Any]] = []
    for row in diagnostic_variant_rows:
        scene = row["scene_id"]
        base = unfiltered_rows[scene]
        purity_delta = float(row["dominant_gt_purity_mean"]) - float(base["dominant_gt_purity_mean"])
        multi_reduction = (
            (float(base["carrier_multi_GT_rate"]) - float(row["carrier_multi_GT_rate"])) / max(float(base["carrier_multi_GT_rate"]), 1e-9)
            if float(base["carrier_multi_GT_rate"]) > 0
            else 0.0
        )
        ok = purity_delta >= 0.0 and multi_reduction >= 0.0 and float(row["clean_same_GT_carrier_recall_proxy"]) > 0.0
        row["purity_delta_vs_unfiltered"] = purity_delta
        row["multi_GT_relative_reduction_vs_unfiltered"] = multi_reduction
        row["variant_gt_reliable_diag_ok"] = bool(ok)
        any_gt_diag_ok_by_scene[scene] = bool(any_gt_diag_ok_by_scene.get(scene, False) or ok)
        score = float(purity_delta + multi_reduction)
        best = best_diagnostic_variant_by_scene.get(scene)
        if best is None or (int(ok), score) > (int(bool(best.get("variant_gt_reliable_diag_ok"))), float(best.get("diagnostic_score", -1e9))):
            best_diagnostic_variant_by_scene[scene] = {
                "variant_id": str(row.get("phase3_variant_id", "")),
                "is_selected_phase3_variant": bool(row.get("is_selected_phase3_variant")),
                "variant_gt_reliable_diag_ok": bool(ok),
                "diagnostic_score": score,
                "purity_delta_vs_unfiltered": purity_delta,
                "multi_GT_relative_reduction_vs_unfiltered": multi_reduction,
                "clean_same_GT_carrier_recall_proxy": float(row["clean_same_GT_carrier_recall_proxy"]),
                "carrier_count": int(row["carrier_count"]),
            }
        if bool(row.get("is_selected_phase3_variant")):
            gt_diag_ok_by_scene[scene] = bool(ok)
        if bool(row.get("is_selected_phase3_variant")) and not ok:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase3_gt_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "failure_id": "gt_diagnostic_reliable_carrier_not_supported",
                    "severity": "diagnostic_blocking_for_interpretation",
                    "evidence": f"purity_delta={purity_delta} multi_reduction={multi_reduction} clean_recall={row['clean_same_GT_carrier_recall_proxy']}",
                    "repair_direction": "Follow Phase3 plan: adjust query strata/density, broaden non-core retained support, or tighten false-bridge filters before trusting downstream affinity.",
                }
            )
    for parity in parity_rows:
        if not bool(parity.get("retained_set_exact_match")):
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase3_gt_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": parity["scene_id"],
                    "failure_id": "phase4_retained_set_not_equal_phase3_selected",
                    "severity": "blocking_provenance_mismatch",
                    "evidence": f"phase3={parity.get('phase3_retained_count')} phase4={parity.get('phase4_carrier_count')} missing={parity.get('missing_phase3_retained_in_phase4_count')} extra={parity.get('extra_phase4_not_in_phase3_retained_count')}",
                    "repair_direction": "Fix Phase4 retained-set reconstruction to match Phase3 exactly, then rerun Phase4/Phase5 before using downstream results.",
                }
            )

    _write_csv(out / "carrier_gt_metric_rows.csv", metric_rows)
    _write_csv(out / "carrier_gt_source_rows.csv", source_rows)
    _write_csv(out / "phase4_retained_parity_rows.csv", parity_rows)
    _write_csv(out / "scene_summary_rows.csv", scene_summary_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase3_gt_diagnostic_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_PHASE3_RELIABLE_CARRIER_DIAGNOSTIC_AND_PHASE4_PARITY" if not failure_rows else "NO_GO_PHASE3_RELIABLE_CARRIER_DIAGNOSTIC_OR_PARITY",
        "phase3_gt_diagnostic_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "selected_variant_by_scene": selected_by_scene,
        "selected_variant_override_by_scene": selected_override_by_scene,
        "diagnose_all_phase3_variants": bool(args.diagnose_all_phase3_variants),
        "diagnosed_variant_ids_by_scene": diagnosed_variant_ids_by_scene,
        "gt_diag_ok_by_scene": gt_diag_ok_by_scene,
        "any_pre_registered_variant_gt_diag_ok_by_scene": any_gt_diag_ok_by_scene,
        "best_diagnostic_variant_by_scene": best_diagnostic_variant_by_scene,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "diagnostic_only": True,
        "truthfulness_note": "GT labels are used only after method selection to diagnose whether GT-free Phase3 filtering produced cleaner carriers; they are not used for thresholds, carrier selection, AP, or prediction.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "carrier_gt_metric_rows": _rel(out / "carrier_gt_metric_rows.csv"),
            "carrier_gt_source_rows": _rel(out / "carrier_gt_source_rows.csv"),
            "phase4_retained_parity_rows": _rel(out / "phase4_retained_parity_rows.csv"),
            "scene_summary_rows": _rel(out / "scene_summary_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
