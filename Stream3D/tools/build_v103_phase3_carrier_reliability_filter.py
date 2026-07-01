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
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

SCENE_INPUTS = {
    "scene0011_00": {
        "phase2_root": AUDIT_ROOT / "v103_phase2_stratified_q2_objbnd_scene0011_first32",
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    },
    "scene0050_00": {
        "phase2_root": AUDIT_ROOT / "v103_phase2_stratified_q2_objbnd_scene0050_first32",
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
    },
}

VARIANTS = [
    {"variant_id": "S0_no_semantic_top40", "semantic": False, "top_rate": 0.40, "hard_veto": False, "score_mode": "base"},
    {"variant_id": "S2_mask_pooled_top40", "semantic": True, "top_rate": 0.40, "hard_veto": False, "score_mode": "base"},
    {"variant_id": "S2_mask_pooled_top30", "semantic": True, "top_rate": 0.30, "hard_veto": False, "score_mode": "base"},
    {"variant_id": "S2_mask_pooled_top40_hardveto", "semantic": True, "top_rate": 0.40, "hard_veto": True, "score_mode": "base"},
    {"variant_id": "S2_clean_broad_jitter_top40", "semantic": True, "top_rate": 0.40, "hard_veto": False, "score_mode": "clean_broad_jitter"},
]

VISIBLE_THRESHOLD = 0.10
CONFIDENCE_THRESHOLD = 0.0
SEMANTIC_DELTA_LOCAL = 3
SEMANTIC_CONTRADICTION_THRESHOLD = 0.20
SELF_ERROR_SIGMA_NORM = 0.015
BROAD_AREA_RATIO = 0.12
OBJECT_LIKE_AREA_MIN = 0.001
OBJECT_LIKE_AREA_MAX = 0.20


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def resolve_repo_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["schema_version"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def load_semantic(scene: str, spec: dict[str, Path]) -> tuple[dict[tuple[int, int], int], np.ndarray, dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    pack = np.load(spec["semantic_npz"])
    features = normalize_rows(pack["features"].astype(np.float32))
    frame_ids = pack["frame_id"].astype(np.int64)
    mask_ids = pack["mask_id"].astype(np.int64)
    feature_index = {(int(f), int(m)): int(i) for i, (f, m) in enumerate(zip(frame_ids.tolist(), mask_ids.tolist()))}
    rows = pd.read_csv(spec["semantic_rows"])
    meta: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        if str(row.get("scene_id")) != scene:
            continue
        frame_id = int(row["frame_id"])
        mask_id = int(row["mask_id"])
        meta[(frame_id, mask_id)] = {
            "broad_background_risk": parse_bool(row.get("broad_background_risk")),
            "semantic_background_score_proxy": parse_bool(row.get("semantic_background_score_proxy")),
            "used_pixel_count": int(float(row.get("used_pixel_count") or 0)),
            "semantic_boundary_variance": float(row.get("semantic_boundary_variance") or 0.0),
            "semantic_entropy": float(row.get("semantic_entropy") or 0.0),
        }
    rng = np.random.default_rng(10303)
    pair_count = min(8192, max(0, features.shape[0] * 2))
    if features.shape[0] >= 2 and pair_count > 0:
        a = rng.integers(0, features.shape[0], size=pair_count)
        b = rng.integers(0, features.shape[0], size=pair_count)
        neq = a != b
        sims = np.sum(features[a[neq]] * features[b[neq]], axis=1)
    else:
        sims = np.asarray([], dtype=np.float32)
    constants = {
        "scene_id": scene,
        "semantic_source": rel(spec["semantic_npz"]),
        "semantic_pair_sample_count": int(sims.shape[0]),
        "mu_sem_random_mask_pair_mean": float(np.mean(sims)) if sims.size else 0.0,
        "mu_sem_random_mask_pair_p50": float(np.percentile(sims, 50)) if sims.size else 0.0,
        "mu_sem_used": float(np.mean(sims)) if sims.size else 0.0,
    }
    return feature_index, features, meta, constants


def load_scene_inputs(scene: str, spec: dict[str, Path]) -> dict[str, Any]:
    phase2_root = spec["phase2_root"]
    summary = json.loads((phase2_root / "summary.json").read_text(encoding="utf-8"))
    batch = np.load(phase2_root / "carrier_batch.npz", allow_pickle=False)
    frame_ids = [int(v) for v in summary["frame_ids"]]
    mask_root = REPO_ROOT / str(summary["mask_root"])
    masks = np.stack([load_mask(mask_root / f"{fid}.png") for fid in frame_ids], axis=0)
    return {"summary": summary, "batch": batch, "frame_ids": frame_ids, "mask_root": mask_root, "masks": masks}


def label_projections(batch: Any, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv = np.asarray(batch["uv_pred"], dtype=np.float32)
    xyz = np.asarray(batch["xyz_ref"], dtype=np.float32)
    valid = np.asarray(batch["valid"], dtype=bool)
    t, n = valid.shape
    height, width = masks.shape[1:]
    finite = np.isfinite(uv).all(axis=-1) & np.isfinite(xyz).all(axis=-1)
    in01 = finite & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    in_image = valid & in01
    xs = np.rint(np.clip(uv[..., 0], 0.0, 1.0) * float(max(width - 1, 1))).astype(np.int64)
    ys = np.rint(np.clip(uv[..., 1], 0.0, 1.0) * float(max(height - 1, 1))).astype(np.int64)
    labels = np.full((t, n), -1, dtype=np.int32)
    for fi in range(t):
        ok = in_image[fi]
        if np.any(ok):
            labels[fi, ok] = masks[fi, ys[fi, ok], xs[fi, ok]]
    return labels, in_image, xs, ys, finite


def build_mask_meta(scene: str, frame_ids: list[int], masks: np.ndarray, semantic_meta: dict[tuple[int, int], dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    _, height, width = masks.shape
    denom = float(max(height * width, 1))
    for local_idx, frame_id in enumerate(frame_ids):
        labels, counts = np.unique(masks[local_idx], return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            if int(label) <= 0:
                continue
            area_ratio = float(count) / denom
            sem = semantic_meta.get((int(frame_id), int(label)), {})
            broad = bool(sem.get("broad_background_risk")) or bool(sem.get("semantic_background_score_proxy")) or area_ratio >= BROAD_AREA_RATIO
            object_like = (OBJECT_LIKE_AREA_MIN <= area_ratio <= OBJECT_LIKE_AREA_MAX) and not broad
            out[(int(local_idx), int(label))] = {
                "scene_id": scene,
                "frame_id": int(frame_id),
                "frame_local_index": int(local_idx),
                "mask_id": int(label),
                "area": int(count),
                "area_ratio": area_ratio,
                "broad": broad,
                "object_like": object_like,
                "semantic_available": (int(frame_id), int(label)) in semantic_meta,
            }
    return out


def semantic_for_label(
    *,
    frame_id: int,
    label: int,
    feature_index: dict[tuple[int, int], int],
    features: np.ndarray,
) -> np.ndarray | None:
    idx = feature_index.get((int(frame_id), int(label)))
    if idx is None:
        return None
    return features[int(idx)]


def calibrated_similarity(vec_a: np.ndarray, vec_b: np.ndarray, mu_sem: float) -> float:
    cos = float(np.dot(vec_a, vec_b))
    denom = max(1.0 - float(mu_sem), 1e-6)
    return float(np.clip((cos - float(mu_sem)) / denom, 0.0, 1.0))


def compute_carrier_rows(scene: str, inputs: dict[str, Any], feature_index: dict[tuple[int, int], int], features: np.ndarray, semantic_constants: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    batch = inputs["batch"]
    frame_ids = inputs["frame_ids"]
    masks = inputs["masks"]
    labels, in_image, xs, ys, finite = label_projections(batch, masks)
    visibility = np.asarray(batch["visibility_prob"], dtype=np.float32)
    confidence = np.asarray(batch["confidence_prob"], dtype=np.float32)
    valid = np.asarray(batch["valid"], dtype=bool)
    src_frame = np.asarray(batch["src_frame"], dtype=np.int64)
    src_uv = np.asarray(batch["src_uv"], dtype=np.float32)
    source_code = np.asarray(batch["query_source_code"], dtype=np.int64)
    carrier_id = np.asarray(batch["carrier_id"], dtype=np.int64)
    height, width = masks.shape[1:]
    diag = float(math.sqrt(float(width * width + height * height)))
    semantic_meta = build_mask_meta(scene, frame_ids, masks, {})

    # Merge semantic broad/object-like metadata after geometry-derived area stats.
    _feature_idx, _features, full_sem_meta, _constants = load_semantic(scene, SCENE_INPUTS[scene])
    semantic_meta = build_mask_meta(scene, frame_ids, masks, full_sem_meta)

    n = int(carrier_id.shape[0])
    rows: list[dict[str, Any]] = []
    total_sem_pairs = 0
    total_sem_bad = 0
    mu_sem = float(semantic_constants["mu_sem_used"])
    self_idx = np.arange(n, dtype=np.int64)
    src_frame_clipped = np.clip(src_frame, 0, labels.shape[0] - 1)
    self_uv = np.asarray(batch["uv_pred"], dtype=np.float32)[src_frame_clipped, self_idx]
    self_error_px = np.linalg.norm((self_uv - src_uv) * np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32), axis=1)
    self_error_px[~np.isfinite(self_error_px)] = diag
    self_error_norm = np.clip(self_error_px / max(diag, 1.0), 0.0, 1.0)

    for ci in range(n):
        obs = np.flatnonzero(in_image[:, ci] & (labels[:, ci] > 0))
        obs_count = int(obs.shape[0])
        broad_flags: list[bool] = []
        object_flags: list[bool] = []
        sem_vecs: list[tuple[int, np.ndarray]] = []
        for fi in obs.tolist():
            label = int(labels[fi, ci])
            meta = semantic_meta.get((int(fi), label), {})
            broad_flags.append(bool(meta.get("broad", False)))
            object_flags.append(bool(meta.get("object_like", False)))
            vec = semantic_for_label(frame_id=frame_ids[int(fi)], label=label, feature_index=feature_index, features=features)
            if vec is not None:
                sem_vecs.append((int(fi), vec))
        broad_rate = float(np.mean(broad_flags)) if broad_flags else 1.0
        object_like_rate = float(np.mean(object_flags)) if object_flags else 0.0
        visible = valid[:, ci] & finite[:, ci] & (visibility[:, ci] >= VISIBLE_THRESHOLD) & (confidence[:, ci] >= CONFIDENCE_THRESHOLD)
        visibility_rate = float(np.mean(visible))
        in_image_rate = float(np.mean(in_image[:, ci]))
        conf_mean = float(np.mean(confidence[in_image[:, ci], ci])) if np.any(in_image[:, ci]) else 0.0

        sem_scores: list[float] = []
        sem_bad = 0
        pair_count = 0
        for idx_a in range(len(sem_vecs)):
            fi_a, vec_a = sem_vecs[idx_a]
            for idx_b in range(idx_a + 1, len(sem_vecs)):
                fi_b, vec_b = sem_vecs[idx_b]
                if abs(fi_a - fi_b) > SEMANTIC_DELTA_LOCAL:
                    continue
                sim = calibrated_similarity(vec_a, vec_b, mu_sem)
                sem_scores.append(sim)
                pair_count += 1
                if sim < SEMANTIC_CONTRADICTION_THRESHOLD:
                    sem_bad += 1
        sem_stability = float(np.median(sem_scores)) if sem_scores else 1.0
        sem_contradiction = float(sem_bad / pair_count) if pair_count else 0.0
        total_sem_pairs += pair_count
        total_sem_bad += sem_bad
        conflict_rate = sem_contradiction
        geo = conf_mean * visibility_rate * in_image_rate * math.exp(-float(self_error_norm[ci]) / SELF_ERROR_SIGMA_NORM)
        mask_rel = max(0.0, (1.0 - broad_rate) * (1.0 - conflict_rate) * object_like_rate)
        sem_rel = sem_stability
        reliability_s0 = float(geo * mask_rel)
        reliability_s2 = float(geo * mask_rel * sem_rel)
        rows.append(
            {
                "schema_version": "stream4d_v103_phase3_carrier_reliability_row_v1",
                "phase_id": "v103_phase3_carrier_reliability_filter",
                "scene_id": scene,
                "carrier_index": int(ci),
                "carrier_id": int(carrier_id[ci]),
                "query_source_code": int(source_code[ci]) if ci < source_code.shape[0] else -1,
                "src_frame_local": int(src_frame[ci]),
                "src_frame_global": int(batch["src_frame_global"][ci]) if "src_frame_global" in batch.files else "",
                "obs_in_image_count": obs_count,
                "in_image_rate": in_image_rate,
                "visibility_rate": visibility_rate,
                "confidence_mean_in_image": conf_mean,
                "self_uv_error_px": float(self_error_px[ci]),
                "normalized_jitter": float(self_error_norm[ci]),
                "broad_mask_participation_rate": broad_rate,
                "object_like_mask_rate": object_like_rate,
                "competing_mask_conflict_rate": conflict_rate,
                "semantic_short_range_stability": sem_stability,
                "semantic_contradiction_rate": sem_contradiction,
                "semantic_pair_count": int(pair_count),
                "r_geo": float(geo),
                "r_mask": float(mask_rel),
                "r_sem": float(sem_rel),
                "reliability_s0": reliability_s0,
                "reliability_s2": reliability_s2,
            }
        )
    df = pd.DataFrame(rows)
    diag_info = {
        "scene_id": scene,
        "total_semantic_pair_count": int(total_sem_pairs),
        "total_semantic_bad_pair_count": int(total_sem_bad),
        "unfiltered_semantic_contradiction_rate": float(total_sem_bad / total_sem_pairs) if total_sem_pairs else 0.0,
        "projection_label_shape": list(labels.shape),
    }
    # Store arrays needed for support metrics.
    diag_info["_labels"] = labels
    diag_info["_in_image"] = in_image
    diag_info["_xs"] = xs
    diag_info["_ys"] = ys
    diag_info["_mask_meta"] = semantic_meta
    return df, diag_info


def boundary_masks_for_frame(mask: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    kernel = np.ones((5, 5), dtype=np.uint8)
    for label in np.unique(mask).tolist():
        if int(label) <= 0:
            continue
        binary = (mask == int(label)).astype(np.uint8)
        grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel).astype(bool)
        out[int(label)] = grad
    return out


def support_metrics(inputs: dict[str, Any], diag_info: dict[str, Any], retained: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(diag_info["_labels"], dtype=np.int32)
    in_image = np.asarray(diag_info["_in_image"], dtype=bool)
    xs = np.asarray(diag_info["_xs"], dtype=np.int64)
    ys = np.asarray(diag_info["_ys"], dtype=np.int64)
    mask_meta = diag_info["_mask_meta"]
    masks = inputs["masks"]
    object_like_keys = [key for key, meta in mask_meta.items() if bool(meta.get("object_like"))]
    support_counts = {key: 0 for key in object_like_keys}
    boundary_counts = {key: 0 for key in object_like_keys}
    retained_idx = np.flatnonzero(retained)
    for fi in range(labels.shape[0]):
        if retained_idx.size:
            lab = labels[fi, retained_idx]
            ok = in_image[fi, retained_idx] & (lab > 0)
            if np.any(ok):
                uniq, counts = np.unique(lab[ok], return_counts=True)
                for label, count in zip(uniq.tolist(), counts.tolist()):
                    key = (int(fi), int(label))
                    if key in support_counts:
                        support_counts[key] += int(count)
                boundaries = boundary_masks_for_frame(masks[fi])
                ok_indices = retained_idx[ok]
                for ci in ok_indices.tolist():
                    label = int(labels[fi, ci])
                    key = (int(fi), label)
                    if key not in boundary_counts:
                        continue
                    band = boundaries.get(label)
                    if band is not None and bool(band[int(ys[fi, ci]), int(xs[fi, ci])]):
                        boundary_counts[key] += 1
    support_arr = np.asarray(list(support_counts.values()), dtype=np.float64)
    boundary_arr = np.asarray(list(boundary_counts.values()), dtype=np.float64)
    return {
        "object_like_mask_count": int(len(object_like_keys)),
        "object_like_mask_support_p10": float(np.percentile(support_arr, 10)) if support_arr.size else 0.0,
        "object_like_mask_support_p50": float(np.percentile(support_arr, 50)) if support_arr.size else 0.0,
        "boundary_band_support_p10": float(np.percentile(boundary_arr, 10)) if boundary_arr.size else 0.0,
        "boundary_band_support_p50": float(np.percentile(boundary_arr, 50)) if boundary_arr.size else 0.0,
        "mask_support_coverage_after_filter": float(np.mean(support_arr > 0)) if support_arr.size else 0.0,
    }


def evaluate_variant(scene: str, variant: dict[str, Any], df: pd.DataFrame, inputs: dict[str, Any], diag_info: dict[str, Any]) -> dict[str, Any]:
    score_key = "reliability_s2" if bool(variant["semantic"]) else "reliability_s0"
    scores = df[score_key].to_numpy(dtype=np.float64)
    if variant.get("score_mode") == "clean_broad_jitter":
        broad = df["broad_mask_participation_rate"].to_numpy(dtype=np.float64)
        contradiction = df["semantic_contradiction_rate"].to_numpy(dtype=np.float64)
        jitter = df["normalized_jitter"].to_numpy(dtype=np.float64)
        clean = np.square(np.clip(1.0 - broad, 0.0, 1.0))
        clean *= np.clip(1.0 - contradiction, 0.0, 1.0)
        clean *= np.exp(-jitter / 0.004)
        scores = scores * clean
    n = int(scores.shape[0])
    keep_n = max(1, int(round(float(variant["top_rate"]) * n)))
    if keep_n >= n:
        threshold = float(np.min(scores))
        retained = np.ones((n,), dtype=bool)
    else:
        order = np.argsort(scores)
        threshold = float(scores[order[-keep_n]])
        retained = np.zeros((n,), dtype=bool)
        retained[order[-keep_n:]] = True
    if bool(variant.get("hard_veto")):
        retained &= df["semantic_contradiction_rate"].to_numpy(dtype=np.float64) <= SEMANTIC_CONTRADICTION_THRESHOLD
    retained_count = int(np.count_nonzero(retained))
    retained_rate = float(retained_count / max(n, 1))
    support = support_metrics(inputs, diag_info, retained)
    unfiltered_broad = float(np.mean(df["broad_mask_participation_rate"].to_numpy(dtype=np.float64)))
    filtered_broad = float(np.mean(df.loc[retained, "broad_mask_participation_rate"].to_numpy(dtype=np.float64))) if retained_count else 1.0
    unfiltered_sem = float(diag_info["unfiltered_semantic_contradiction_rate"])
    pair_counts = df.loc[retained, "semantic_pair_count"].to_numpy(dtype=np.float64)
    bad_rates = df.loc[retained, "semantic_contradiction_rate"].to_numpy(dtype=np.float64)
    filtered_sem = float(np.sum(pair_counts * bad_rates) / max(np.sum(pair_counts), 1.0)) if retained_count else 1.0
    unfiltered_jitter_p90 = float(np.percentile(df["normalized_jitter"].to_numpy(dtype=np.float64), 90))
    filtered_jitter_p90 = float(np.percentile(df.loc[retained, "normalized_jitter"].to_numpy(dtype=np.float64), 90)) if retained_count else 1.0
    row = {
        "schema_version": "stream4d_v103_phase3_filter_metric_row_v1",
        "phase_id": "v103_phase3_carrier_reliability_filter",
        "scene_id": scene,
        "variant_id": variant["variant_id"],
        "score_key": score_key,
        "threshold": threshold,
        "retained_carrier_count": retained_count,
        "total_carrier_count": n,
        "retained_carrier_rate": retained_rate,
        "object_like_mask_support_p10": support["object_like_mask_support_p10"],
        "object_like_mask_support_p50": support["object_like_mask_support_p50"],
        "boundary_band_support_p10": support["boundary_band_support_p10"],
        "boundary_band_support_p50": support["boundary_band_support_p50"],
        "mask_support_coverage_after_filter": support["mask_support_coverage_after_filter"],
        "broad_mask_participation_rate": filtered_broad,
        "unfiltered_broad_mask_participation_rate": unfiltered_broad,
        "broad_relative_reduction": float((unfiltered_broad - filtered_broad) / max(unfiltered_broad, 1e-9)),
        "semantic_contradiction_rate": filtered_sem,
        "unfiltered_semantic_contradiction_rate": unfiltered_sem,
        "semantic_relative_reduction": float((unfiltered_sem - filtered_sem) / max(unfiltered_sem, 1e-9)) if unfiltered_sem > 0 else 0.0,
        "normalized_jitter_p90": filtered_jitter_p90,
        "unfiltered_normalized_jitter_p90": unfiltered_jitter_p90,
        "jitter_relative_reduction": float((unfiltered_jitter_p90 - filtered_jitter_p90) / max(unfiltered_jitter_p90, 1e-9)),
        "in_image_rate_mean": float(np.mean(df.loc[retained, "in_image_rate"].to_numpy(dtype=np.float64))) if retained_count else 0.0,
        "visibility_rate_mean": float(np.mean(df.loc[retained, "visibility_rate"].to_numpy(dtype=np.float64))) if retained_count else 0.0,
        "object_like_mask_count": support["object_like_mask_count"],
        "uses_gt_for_threshold": False,
        "uses_future": False,
    }
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase3 carrier reliability filtering.")
    parser.add_argument("--output-root", default=str(OUT_DIR))
    parser.add_argument("--scene0011-phase2-root", default=str(SCENE_INPUTS["scene0011_00"]["phase2_root"]))
    parser.add_argument("--scene0050-phase2-root", default=str(SCENE_INPUTS["scene0050_00"]["phase2_root"]))
    return parser


def main() -> int:
    global OUT_DIR, SCENE_INPUTS
    args = build_parser().parse_args()
    OUT_DIR = resolve_repo_path(args.output_root)
    scene_inputs = {scene: dict(spec) for scene, spec in SCENE_INPUTS.items()}
    scene_inputs["scene0011_00"]["phase2_root"] = resolve_repo_path(args.scene0011_phase2_root)
    scene_inputs["scene0050_00"]["phase2_root"] = resolve_repo_path(args.scene0050_phase2_root)
    SCENE_INPUTS = scene_inputs

    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_carrier_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    semantic_distribution_rows: list[dict[str, Any]] = []
    constants_by_scene: dict[str, Any] = {}
    casebook_rows: list[dict[str, Any]] = []

    for scene, spec in SCENE_INPUTS.items():
        inputs = load_scene_inputs(scene, spec)
        feature_index, features, _semantic_meta, constants = load_semantic(scene, spec)
        constants_by_scene[scene] = constants
        semantic_distribution_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_semantic_distribution_row_v1",
                "phase_id": "v103_phase3_carrier_reliability_filter",
                **constants,
            }
        )
        carrier_df, diag_info = compute_carrier_rows(scene, inputs, feature_index, features, constants)
        all_carrier_frames.append(carrier_df)
        for variant in VARIANTS:
            row = evaluate_variant(scene, variant, carrier_df, inputs, diag_info)
            metric_rows.append(row)
        worst = carrier_df.sort_values("reliability_s2", ascending=True).head(10)
        for item in worst.to_dict("records"):
            casebook_rows.append(
                {
                    "schema_version": "stream4d_v103_phase3_casebook_row_v1",
                    "phase_id": "v103_phase3_carrier_reliability_filter",
                    "scene_id": scene,
                    "case_type": "lowest_reliability_s2",
                    "carrier_index": item["carrier_index"],
                    "carrier_id": item["carrier_id"],
                    "query_source_code": item["query_source_code"],
                    "in_image_rate": item["in_image_rate"],
                    "visibility_rate": item["visibility_rate"],
                    "normalized_jitter": item["normalized_jitter"],
                    "broad_mask_participation_rate": item["broad_mask_participation_rate"],
                    "object_like_mask_rate": item["object_like_mask_rate"],
                    "semantic_contradiction_rate": item["semantic_contradiction_rate"],
                    "reliability_s2": item["reliability_s2"],
                }
            )

    carrier_df_all = pd.concat(all_carrier_frames, ignore_index=True)
    carrier_path = OUT_DIR / "carrier_reliability_rows.parquet"
    carrier_df_all.to_parquet(carrier_path, index=False)
    metric_csv = OUT_DIR / "carrier_filter_metric_rows.csv"
    semantic_csv = OUT_DIR / "semantic_distribution_rows.csv"
    casebook_csv = OUT_DIR / "carrier_filter_casebook_rows.csv"
    write_csv(metric_csv, metric_rows)
    write_csv(semantic_csv, semantic_distribution_rows)
    write_csv(casebook_csv, casebook_rows)
    write_json(OUT_DIR / "semantic_baseline_constants.json", constants_by_scene)

    # Select the best pre-registered variant by GT-free gate count, then margin.
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    selected_by_scene: dict[str, str] = {}
    for scene in SCENE_INPUTS:
        scene_rows = [r for r in metric_rows if r["scene_id"] == scene]

        def pass_count(row: dict[str, Any]) -> tuple[int, float]:
            checks = [
                0.05 <= float(row["retained_carrier_rate"]) <= 0.60,
                float(row["object_like_mask_support_p10"]) >= 50.0,
                float(row["boundary_band_support_p10"]) >= 10.0,
                float(row["broad_relative_reduction"]) >= 0.20,
                float(row["semantic_relative_reduction"]) >= 0.20 if float(row["unfiltered_semantic_contradiction_rate"]) > 0 else True,
                float(row["jitter_relative_reduction"]) >= 0.20,
            ]
            return (sum(bool(v) for v in checks), float(row["broad_relative_reduction"]) + float(row["jitter_relative_reduction"]) + float(row["semantic_relative_reduction"]))

        selected = max(scene_rows, key=pass_count)
        selected_by_scene[scene] = str(selected["variant_id"])
        gate_specs = [
            ("retained_carrier_rate_between_0p05_0p60", 0.05 <= float(selected["retained_carrier_rate"]) <= 0.60, selected["retained_carrier_rate"], "0.05..0.60"),
            ("object_like_mask_support_p10_ge_50", float(selected["object_like_mask_support_p10"]) >= 50.0, selected["object_like_mask_support_p10"], 50.0),
            ("boundary_band_support_p10_ge_10", float(selected["boundary_band_support_p10"]) >= 10.0, selected["boundary_band_support_p10"], 10.0),
            ("broad_mask_participation_relative_reduction_ge_0p20", float(selected["broad_relative_reduction"]) >= 0.20, selected["broad_relative_reduction"], 0.20),
            ("semantic_contradiction_relative_reduction_ge_0p20", (float(selected["semantic_relative_reduction"]) >= 0.20 if float(selected["unfiltered_semantic_contradiction_rate"]) > 0 else True), selected["semantic_relative_reduction"], 0.20),
            ("normalized_jitter_p90_relative_reduction_ge_0p20", float(selected["jitter_relative_reduction"]) >= 0.20, selected["jitter_relative_reduction"], 0.20),
        ]
        for name, ok, observed, required in gate_specs:
            gate = {
                "schema_version": "stream4d_v103_phase3_gate_row_v1",
                "phase_id": "v103_phase3_carrier_reliability_filter",
                "scene_id": scene,
                "selected_variant_id": selected["variant_id"],
                "gate_name": name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
            }
            gate_rows.append(gate)
            if not ok:
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase3_failure_row_v1",
                        "phase_id": "v103_phase3_carrier_reliability_filter",
                        "scene_id": scene,
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"selected_variant={selected['variant_id']} observed={observed} required={required}",
                        "repair_direction": "Follow Phase3 repair: increase query density/object-like sources, relax semantic with hard veto, or adjust boundary oversampling before entering Phase4.",
                    }
                )

    gate_csv = OUT_DIR / "gate_rows.csv"
    failure_csv = OUT_DIR / "failure_rows.csv"
    write_csv(gate_csv, gate_rows)
    write_csv(failure_csv, failure_rows)
    phase3_pass = len(failure_rows) == 0
    summary = {
        "schema_version": "stream4d_v103_phase3_carrier_reliability_filter_summary_v1",
        "phase_id": "v103_phase3_carrier_reliability_filter",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "phase3_pass": phase3_pass,
        "decision": "PASS_ENTER_PHASE4_PRIMITIVE_AFFINITY" if phase3_pass else "NO_GO_REPAIR_PHASE3_CARRIER_FILTERING",
        "failure_count": len(failure_rows),
        "selected_variant_by_scene": selected_by_scene,
        "variant_ids": [v["variant_id"] for v in VARIANTS],
        "visible_threshold": VISIBLE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "semantic_delta_local": SEMANTIC_DELTA_LOCAL,
        "semantic_contradiction_threshold": SEMANTIC_CONTRADICTION_THRESHOLD,
        "self_error_sigma_norm": SELF_ERROR_SIGMA_NORM,
        "truthfulness_note": "Phase3 gates use GT-free reliability/support metrics only. GT diagnostics are not used for threshold selection in this script.",
        "plan_doc": rel(PLAN_DOC),
        "outputs": {
            "summary": rel(OUT_DIR / "summary.json"),
            "carrier_reliability_rows": rel(carrier_path),
            "carrier_filter_metric_rows": rel(metric_csv),
            "semantic_baseline_constants": rel(OUT_DIR / "semantic_baseline_constants.json"),
            "semantic_distribution_rows": rel(semantic_csv),
            "carrier_filter_casebook_rows": rel(casebook_csv),
            "gate_rows": rel(gate_csv),
            "failure_rows": rel(failure_csv),
        },
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase3_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
