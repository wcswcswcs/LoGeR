#!/usr/bin/env python3
"""Chunk32 DA3-GIANT 3DGS primitive-support missing-frame diagnostic for v102."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plyfile import PlyData


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402
from tools import build_v102_phase7d_phase7c_materialized_ap_diagnostic as p7d  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase7h_chunk32_primitive_support_shape_diagnostic"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PHASE2B_DIR = AUDIT_ROOT / "v102_phase2b_da3_giant_chunk32_audit" / "chunk32_process252"
PLY_PATH = PHASE2B_DIR / "gs_ply" / "0000.ply"
MINI_NPZ = PHASE2B_DIR / "exports" / "mini_npz" / "results.npz"
PHASE3B_SUMMARY = AUDIT_ROOT / "v102_phase3b_da3_giant_chunk32_visual_audit" / "summary.json"
PHASE7D_DIR = AUDIT_ROOT / "v102_phase7d_phase7c_materialized_ap_diagnostic"
PHASE7E_SUMMARY = AUDIT_ROOT / "v102_phase7e_gtfree_score_calibration_diagnostic" / "summary.json"
PHASE7D_NODE_ROWS = PHASE7D_DIR / "mv_object_frame_mask_rows.parquet"
PHASE7D_COMPONENT_ROWS = PHASE7D_DIR / "materialized_component_rows.csv"
FEATURE_ROWS = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_feature_rows.csv"

PHASE_ID = "v102_phase7h_chunk32_primitive_support_shape_diagnostic"
VARIANT_PREFIX = "P2_v102_phase7h"

VARIANTS = [
    {
        "variant_id": "H0_no_expand_s8_score",
        "expand": False,
        "min_seed_support": 99,
        "min_vote_count": 10**9,
        "min_support_fraction": 2.0,
        "min_mask_coverage": 2.0,
        "entropy_max": 0.0,
        "drop_broad": True,
    },
    {
        "variant_id": "H1_seed2_count20_frac008_cover010_sem045",
        "expand": True,
        "min_seed_support": 2,
        "min_vote_count": 20,
        "min_support_fraction": 0.08,
        "min_mask_coverage": 0.010,
        "entropy_max": 0.45,
        "drop_broad": False,
    },
    {
        "variant_id": "H2_seed2_count8_frac003_cover005_sem050",
        "expand": True,
        "min_seed_support": 2,
        "min_vote_count": 8,
        "min_support_fraction": 0.03,
        "min_mask_coverage": 0.005,
        "entropy_max": 0.50,
        "drop_broad": False,
    },
    {
        "variant_id": "H3_seed1_count16_frac006_cover008_dropbroad",
        "expand": True,
        "min_seed_support": 1,
        "min_vote_count": 16,
        "min_support_fraction": 0.06,
        "min_mask_coverage": 0.008,
        "entropy_max": 0.45,
        "drop_broad": True,
    },
    {
        "variant_id": "H4_seed1_count5_frac002_cover003_sem060",
        "expand": True,
        "min_seed_support": 1,
        "min_vote_count": 5,
        "min_support_fraction": 0.02,
        "min_mask_coverage": 0.003,
        "entropy_max": 0.60,
        "drop_broad": False,
    },
    {
        "variant_id": "H5_seed3_count3_frac001_cover001_sem060",
        "expand": True,
        "min_seed_support": 3,
        "min_vote_count": 3,
        "min_support_fraction": 0.01,
        "min_mask_coverage": 0.001,
        "entropy_max": 0.60,
        "drop_broad": False,
    },
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _norm(values: dict[str, float]) -> dict[str, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {key: 0.0 for key in values}
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-12:
        return {key: 0.5 for key in values}
    return {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}


def _prop(vertex_data: np.ndarray, name: str) -> np.ndarray:
    if vertex_data.dtype.names and name in vertex_data.dtype.names:
        return np.asarray(vertex_data[name])
    return np.full(len(vertex_data), np.nan)


def _homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = extrinsic.astype(np.float64)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")


def _load_xyz_and_mini() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not PLY_PATH.exists():
        raise FileNotFoundError(PLY_PATH)
    if not MINI_NPZ.exists():
        raise FileNotFoundError(MINI_NPZ)
    ply = PlyData.read(str(PLY_PATH))
    vertex = ply["vertex"].data
    xyz = np.column_stack([_prop(vertex, "x"), _prop(vertex, "y"), _prop(vertex, "z")]).astype(np.float32)
    finite = np.all(np.isfinite(xyz), axis=1)
    if not np.all(finite):
        xyz = xyz[finite]
    with np.load(MINI_NPZ) as data:
        mini = {key: np.asarray(data[key]) for key in data.files}
    return xyz, mini


def _project_frame_mask_ids(
    xyz: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_shape: tuple[int, int],
    label_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    ext = _homogeneous_extrinsic(extrinsic)
    k = np.asarray(intrinsic, dtype=np.float64)
    image_h, image_w = int(depth_shape[0]), int(depth_shape[1])
    mask_h, mask_w = int(label_shape[0]), int(label_shape[1])
    cam = xyz.astype(np.float64) @ ext[:3, :3].T + ext[:3, 3]
    z = cam[:, 2]
    valid_z = z > 1e-6
    u = np.empty(len(xyz), dtype=np.float64)
    v = np.empty(len(xyz), dtype=np.float64)
    u.fill(np.nan)
    v.fill(np.nan)
    u[valid_z] = k[0, 0] * (cam[valid_z, 0] / z[valid_z]) + k[0, 2]
    v[valid_z] = k[1, 1] * (cam[valid_z, 1] / z[valid_z]) + k[1, 2]
    inside = valid_z & (u >= 0.0) & (u < image_w) & (v >= 0.0) & (v < image_h)
    primitive_indices = np.flatnonzero(inside).astype(np.int32, copy=False)
    xs = np.floor(np.clip(u[inside] / float(image_w) * mask_w, 0, mask_w - 1)).astype(np.int32)
    ys = np.floor(np.clip(v[inside] / float(image_h) * mask_h, 0, mask_h - 1)).astype(np.int32)
    linear = (ys * mask_w + xs).astype(np.int64, copy=False)
    return primitive_indices, linear, int(np.count_nonzero(valid_z))


def _feature_maps() -> tuple[dict[str, dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    feature_rows = pd.read_csv(FEATURE_ROWS)
    feature_by_node = feature_rows.set_index("mask_observation_id").to_dict(orient="index")
    feature_by_frame_mask: dict[tuple[int, int], dict[str, Any]] = {}
    for row in feature_rows.to_dict(orient="records"):
        if str(row.get("scene_id", "")) != p7d.SCENE_ID:
            continue
        frame = int(_num(row.get("frame_id")))
        if frame not in set(p7d._frame_universe()):
            continue
        feature_by_frame_mask[(frame, int(_num(row.get("mask_id"))))] = row
    return feature_by_node, feature_by_frame_mask


def _object_meta(base_rows: pd.DataFrame, component_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    diag = component_rows.set_index("mv_object_id").to_dict(orient="index")
    out: dict[str, dict[str, Any]] = {}
    for oid, rows in base_rows.groupby("mv_object_id"):
        oid = str(oid)
        frame_ids = sorted({int(v) for v in rows["frame_id"].tolist()})
        out[oid] = {
            "mv_object_id": oid,
            "source_component_id": str(rows.iloc[0].get("source_component_id", "")),
            "frames": frame_ids,
            "seed_node_count": int(len(rows)),
            "diagnostic_gt_dominant": str(diag.get(oid, {}).get("diagnostic_gt_dominant", "")),
            "diagnostic_gt_purity": _num(diag.get(oid, {}).get("diagnostic_gt_purity", "")),
        }
    return out


def _build_support_counts(
    xyz: np.ndarray,
    mini: dict[str, np.ndarray],
    base_rows: pd.DataFrame,
    object_ids: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    object_to_idx = {oid: idx for idx, oid in enumerate(object_ids)}
    support = np.zeros((len(object_ids), len(xyz)), dtype=np.uint8)
    projection_rows: list[dict[str, Any]] = []
    mask_path_by_frame, mask_source = p7d._mask_path_lookup()
    extrinsics = np.asarray(mini["extrinsics"])
    intrinsics = np.asarray(mini["intrinsics"])
    depths = np.asarray(mini["depth"])
    camera_count = min(len(extrinsics), len(intrinsics), len(depths), len(p7d._frame_universe()))

    for camera_index, frame_id in enumerate(p7d._frame_universe()[:camera_count]):
        mask_path = mask_path_by_frame.get((p7d.SCENE_ID, int(frame_id)))
        if mask_path is None or not mask_path.exists():
            projection_rows.append(
                {
                    "schema_version": "stream4d_v102_phase7h_projection_frame_v1",
                    "phase_id": PHASE_ID,
                    "camera_index": camera_index,
                    "frame_id": int(frame_id),
                    "mask_exists": False,
                    "seed_hit_primitive_count": 0,
                    "note": "source registry mask missing",
                }
            )
            continue
        label = p7d._read_label(mask_path)
        primitive_idx, linear, positive_z_count = _project_frame_mask_ids(
            xyz,
            extrinsics[camera_index],
            intrinsics[camera_index],
            tuple(depths[camera_index].shape),
            tuple(label.shape),
        )
        flat_label = label.reshape(-1)
        mask_ids = flat_label[linear].astype(np.int32, copy=False)
        foreground = mask_ids > 0
        frame_base = base_rows[base_rows["frame_id"].astype(int) == int(frame_id)]
        max_label = int(flat_label.max()) if flat_label.size else 0
        label_to_object = np.full(max_label + 1, -1, dtype=np.int32)
        seed_mask_ids = []
        for row in frame_base.to_dict(orient="records"):
            mask_id = int(_num(row.get("selected_mask_id")))
            if 0 <= mask_id <= max_label:
                label_to_object[mask_id] = object_to_idx[str(row["mv_object_id"])]
                seed_mask_ids.append(mask_id)
        eligible = (mask_ids >= 0) & (mask_ids <= max_label)
        obj_for_hit = np.full(len(mask_ids), -1, dtype=np.int32)
        obj_for_hit[eligible] = label_to_object[mask_ids[eligible]]
        seed_hit = obj_for_hit >= 0
        for obj_idx in np.unique(obj_for_hit[seed_hit]):
            prims = primitive_idx[seed_hit & (obj_for_hit == obj_idx)]
            if len(prims):
                support[int(obj_idx), prims] = np.minimum(support[int(obj_idx), prims] + 1, 255)
        projection_rows.append(
            {
                "schema_version": "stream4d_v102_phase7h_projection_frame_v1",
                "phase_id": PHASE_ID,
                "camera_index": camera_index,
                "frame_id": int(frame_id),
                "mask_path": _rel(mask_path),
                "mask_exists": True,
                "mask_height": int(label.shape[0]),
                "mask_width": int(label.shape[1]),
                "depth_height": int(depths[camera_index].shape[0]),
                "depth_width": int(depths[camera_index].shape[1]),
                "positive_z_primitive_count": int(positive_z_count),
                "inside_image_primitive_count": int(len(primitive_idx)),
                "foreground_hit_primitive_count": int(np.count_nonzero(foreground)),
                "seed_mask_count": int(len(seed_mask_ids)),
                "seed_hit_primitive_count": int(np.count_nonzero(seed_hit)),
                "unique_foreground_masks_hit": int(len(np.unique(mask_ids[foreground]))),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": False,
            }
        )
    object_support_rows = []
    for oid, obj_idx in object_to_idx.items():
        vals = support[obj_idx]
        object_support_rows.append(
            {
                "schema_version": "stream4d_v102_phase7h_object_primitive_support_v1",
                "phase_id": PHASE_ID,
                "mv_object_id": oid,
                "primitive_support_ge1_count": int(np.count_nonzero(vals >= 1)),
                "primitive_support_ge2_count": int(np.count_nonzero(vals >= 2)),
                "primitive_support_ge3_count": int(np.count_nonzero(vals >= 3)),
                "primitive_support_max": int(vals.max()) if len(vals) else 0,
                "primitive_count": int(len(vals)),
                "uses_gt_for_prediction": False,
            }
        )
    projection_rows.append(
        {
            "schema_version": "stream4d_v102_phase7h_projection_source_v1",
            "phase_id": PHASE_ID,
            "row_type": "mask_source_summary",
            **mask_source,
        }
    )
    return support, projection_rows, object_support_rows


def _candidate_row_from_mask(
    oid: str,
    obj_idx: int,
    frame_id: int,
    mask_id: int,
    spec: dict[str, Any],
    feature: dict[str, Any],
    vote_count: int,
    support_projected_count: int,
    mask_area: int,
    source_component_id: str,
) -> dict[str, Any]:
    support_fraction = vote_count / max(1, support_projected_count)
    mask_coverage = vote_count / max(1, mask_area)
    expansion_score = support_fraction + 0.10 * mask_coverage + 0.001 * math.log1p(vote_count)
    expansion_score += 0.01 * _num(feature.get("semantic_prototype_margin"))
    expansion_score -= 0.001 * _num(feature.get("semantic_entropy"))
    return {
        "schema_version": "stream4d_v102_phase7h_expansion_candidate_v1",
        "phase_id": PHASE_ID,
        "dataset_split": "dev",
        "variant_id": f"{VARIANT_PREFIX}_{spec['variant_id']}",
        "mv_object_id": oid,
        "object_id": oid,
        "source_component_id": source_component_id,
        "scene_id": p7d.SCENE_ID,
        "chunk_id": p7d.CHUNK_ID,
        "window_id": p7d.CHUNK_ID,
        "frame_id": int(frame_id),
        "selected_mask_id": int(mask_id),
        "mask_id_or_generated_id": int(mask_id),
        "source_mask_observation_id": str(feature.get("mask_observation_id", f"{p7d.SCENE_ID}:{frame_id}:{mask_id}")),
        "readout_mode": "phase7h_chunk32_primitive_support_missing_frame_fill",
        "score": 0.0,
        "object_score": 0.0,
        "score_scope": "current_chunk32_diagnostic",
        "score_policy": "phase7h_s8_node_area_semantic_after_primitive_support_expansion",
        "method_chunk_size": p7d.CHUNK_SIZE,
        "method_chunk_overlap": 0,
        "frame_stride": p7d.FRAME_STRIDE,
        "object_id_policy": "phase7c_component_identity_with_phase7h_primitive_support_fill",
        "object_birth_scope": "phase7h_chunk32_primitive_support_shape_diagnostic",
        "semantic_entropy": feature.get("semantic_entropy", ""),
        "semantic_prototype_margin": feature.get("semantic_prototype_margin", ""),
        "broad_background_risk": feature.get("broad_background_risk", ""),
        "used_pixel_count": feature.get("used_pixel_count", mask_area),
        "primitive_support_min_seed_support": int(spec["min_seed_support"]),
        "primitive_support_vote_count": int(vote_count),
        "primitive_support_projected_count": int(support_projected_count),
        "primitive_support_fraction": float(support_fraction),
        "primitive_support_mask_area": int(mask_area),
        "primitive_support_mask_coverage": float(mask_coverage),
        "primitive_support_expansion_score": float(expansion_score),
        "phase7h_row_source": "primitive_support_missing_frame_fill",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": False,
        "uses_future": False,
    }


def _expand_rows(
    xyz: np.ndarray,
    mini: dict[str, np.ndarray],
    base_rows: pd.DataFrame,
    support: np.ndarray,
    object_ids: list[str],
    meta: dict[str, dict[str, Any]],
    feature_by_frame_mask: dict[tuple[int, int], dict[str, Any]],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) | {"phase7h_row_source": "base_phase7d"} for row in base_rows.to_dict(orient="records")]
    if not bool(spec["expand"]):
        return rows, [], {
            "candidate_examined_count": 0,
            "proposed_expansion_count": 0,
            "accepted_expansion_count": 0,
            "wta_drop_count": 0,
            "rejected_missing_feature_count": 0,
            "rejected_broad_count": 0,
            "rejected_entropy_count": 0,
            "rejected_threshold_count": 0,
        }

    object_to_idx = {oid: idx for idx, oid in enumerate(object_ids)}
    existing_object_frame = {(str(row["mv_object_id"]), int(row["frame_id"])) for row in rows}
    occupied_original_mask = {(int(row["frame_id"]), int(row["selected_mask_id"])) for row in rows}
    primitive_sets = [
        np.flatnonzero(support[obj_idx] >= int(spec["min_seed_support"])).astype(np.int32, copy=False)
        for obj_idx in range(len(object_ids))
    ]

    mask_path_by_frame, _mask_source = p7d._mask_path_lookup()
    extrinsics = np.asarray(mini["extrinsics"])
    intrinsics = np.asarray(mini["intrinsics"])
    depths = np.asarray(mini["depth"])
    camera_count = min(len(extrinsics), len(intrinsics), len(depths), len(p7d._frame_universe()))
    proposed: list[dict[str, Any]] = []
    candidate_debug: list[dict[str, Any]] = []
    candidate_examined_count = 0
    rejected_missing_feature_count = 0
    rejected_broad_count = 0
    rejected_entropy_count = 0
    rejected_threshold_count = 0

    for camera_index, frame_id in enumerate(p7d._frame_universe()[:camera_count]):
        mask_path = mask_path_by_frame.get((p7d.SCENE_ID, int(frame_id)))
        if mask_path is None or not mask_path.exists():
            continue
        label = p7d._read_label(mask_path)
        flat_label = label.reshape(-1)
        area_by_label = np.bincount(flat_label.astype(np.int64), minlength=int(flat_label.max()) + 1)
        primitive_idx, linear, _positive_z_count = _project_frame_mask_ids(
            xyz,
            extrinsics[camera_index],
            intrinsics[camera_index],
            tuple(depths[camera_index].shape),
            tuple(label.shape),
        )
        frame_mask_by_primitive = np.zeros(len(xyz), dtype=np.int32)
        frame_mask_by_primitive[primitive_idx] = flat_label[linear].astype(np.int32, copy=False)
        for oid in object_ids:
            if (oid, int(frame_id)) in existing_object_frame:
                continue
            obj_idx = object_to_idx[oid]
            prims = primitive_sets[obj_idx]
            if len(prims) == 0:
                continue
            mids = frame_mask_by_primitive[prims]
            mids = mids[mids > 0]
            support_projected_count = int(len(mids))
            if support_projected_count == 0:
                continue
            counts = np.bincount(mids.astype(np.int64), minlength=len(area_by_label))
            candidate_mask_ids = np.flatnonzero(counts >= int(spec["min_vote_count"]))
            if len(candidate_mask_ids) == 0:
                rejected_threshold_count += 1
                continue
            best: dict[str, Any] | None = None
            best_score = -1e18
            for mask_id_raw in candidate_mask_ids:
                mask_id = int(mask_id_raw)
                if mask_id <= 0 or (int(frame_id), mask_id) in occupied_original_mask:
                    continue
                candidate_examined_count += 1
                vote_count = int(counts[mask_id])
                mask_area = int(area_by_label[mask_id]) if mask_id < len(area_by_label) else 0
                support_fraction = vote_count / max(1, support_projected_count)
                mask_coverage = vote_count / max(1, mask_area)
                feature = feature_by_frame_mask.get((int(frame_id), mask_id))
                if feature is None:
                    rejected_missing_feature_count += 1
                    continue
                if bool(spec["drop_broad"]) and _bool(feature.get("broad_background_risk")):
                    rejected_broad_count += 1
                    continue
                if _num(feature.get("semantic_entropy"), 999.0) > float(spec["entropy_max"]):
                    rejected_entropy_count += 1
                    continue
                if support_fraction < float(spec["min_support_fraction"]) or mask_coverage < float(spec["min_mask_coverage"]):
                    rejected_threshold_count += 1
                    continue
                cand = _candidate_row_from_mask(
                    oid=oid,
                    obj_idx=obj_idx,
                    frame_id=int(frame_id),
                    mask_id=mask_id,
                    spec=spec,
                    feature=feature,
                    vote_count=vote_count,
                    support_projected_count=support_projected_count,
                    mask_area=mask_area,
                    source_component_id=meta[oid]["source_component_id"],
                )
                candidate_debug.append(cand)
                score = float(cand["primitive_support_expansion_score"])
                if score > best_score:
                    best_score = score
                    best = cand
            if best is not None:
                proposed.append(best)

    by_mask: dict[tuple[int, int], dict[str, Any]] = {}
    wta_drop_count = 0
    for row in proposed:
        key = (int(row["frame_id"]), int(row["selected_mask_id"]))
        current = by_mask.get(key)
        if current is None or float(row["primitive_support_expansion_score"]) > float(
            current["primitive_support_expansion_score"]
        ):
            if current is not None:
                wta_drop_count += 1
            by_mask[key] = row
        else:
            wta_drop_count += 1
    accepted = list(by_mask.values())
    rows.extend(accepted)
    diag = {
        "candidate_examined_count": int(candidate_examined_count),
        "proposed_expansion_count": int(len(proposed)),
        "accepted_expansion_count": int(len(accepted)),
        "wta_drop_count": int(wta_drop_count),
        "rejected_missing_feature_count": int(rejected_missing_feature_count),
        "rejected_broad_count": int(rejected_broad_count),
        "rejected_entropy_count": int(rejected_entropy_count),
        "rejected_threshold_count": int(rejected_threshold_count),
    }
    return rows, candidate_debug, diag


def _scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["mv_object_id"])].append(row)
    raw_node = {oid: float(len(vals)) for oid, vals in by_object.items()}
    raw_area = {
        oid: math.log1p(sum(_num(row.get("used_pixel_count")) for row in vals)) for oid, vals in by_object.items()
    }
    raw_entropy = {
        oid: float(np.mean([_num(row.get("semantic_entropy")) for row in vals])) for oid, vals in by_object.items()
    }
    raw_margin = {
        oid: float(np.mean([_num(row.get("semantic_prototype_margin")) for row in vals])) for oid, vals in by_object.items()
    }
    node = _norm(raw_node)
    area = _norm(raw_area)
    entropy_good = {oid: 1.0 - val for oid, val in _norm(raw_entropy).items()}
    margin = _norm(raw_margin)
    sem = {oid: 0.60 * margin[oid] + 0.40 * entropy_good[oid] for oid in by_object}
    return {oid: 0.45 * node[oid] + 0.35 * area[oid] + 0.20 * sem[oid] for oid in by_object}


def _mask_diagnostic_gt(label: np.ndarray, gt: np.ndarray, mask_id: int) -> str:
    mask = label == int(mask_id)
    vals = gt[mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return ""
    ids, counts = np.unique(vals, return_counts=True)
    return str(int(ids[int(np.argmax(counts))]))


def _evaluate(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
    meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    object_ids = sorted({str(row["mv_object_id"]) for row in rows})
    object_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[int(row["frame_id"])].append(row)
    mask_path_by_frame, mask_source = p7d._mask_path_lookup()
    acc = SparseSceneIoU()
    pixel_collision_count = 0
    missing_mask_frame_count = 0
    selected_mask_missing_count = 0
    accepted_expansion_gt_checked = 0
    accepted_expansion_same_gt = 0
    total_pred_positive = 0
    total_gt_positive = 0
    duplicate_keys = Counter((int(row["frame_id"]), int(row["selected_mask_id"])) for row in rows)
    same_frame_duplicate_mask_count = sum(max(0, count - 1) for count in duplicate_keys.values())

    for frame in p7d._frame_universe():
        mask_path = mask_path_by_frame.get((p7d.SCENE_ID, int(frame)))
        if mask_path is None or not mask_path.exists():
            missing_mask_frame_count += 1
            continue
        label = p7d._read_label(mask_path)
        gt = _load_gt_2d(p7d.SCENE_ID, int(frame), label.shape)
        pred = np.zeros(label.shape, dtype=np.int64)
        for row in sorted(
            rows_by_frame.get(int(frame), []),
            key=lambda r: (-scores[str(r["mv_object_id"])], str(r["mv_object_id"])),
        ):
            mask = label == int(row["selected_mask_id"])
            if int(np.count_nonzero(mask)) <= 0:
                selected_mask_missing_count += 1
                continue
            if row.get("phase7h_row_source") == "primitive_support_missing_frame_fill":
                accepted_expansion_gt_checked += 1
                diag_gt = _mask_diagnostic_gt(label, gt, int(row["selected_mask_id"]))
                accepted_expansion_same_gt += int(
                    diag_gt != "" and diag_gt == meta[str(row["mv_object_id"])]["diagnostic_gt_dominant"]
                )
            occupied = (pred > 0) & mask
            pixel_collision_count += int(np.count_nonzero(occupied))
            pred[(pred == 0) & mask] = object_index[str(row["mv_object_id"])]
        acc.add(pred, gt)
        total_pred_positive += int(np.count_nonzero(pred > 0))
        total_gt_positive += int(np.count_nonzero(gt > 0))

    input_scores = np.ones((len(object_ids),), dtype=np.float32)
    for oid, idx in object_index.items():
        input_scores[idx - 1] = float(scores.get(oid, 0.0))
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=64,
        min_gt_pixels=64,
        score_mode="input",
        input_scores=input_scores,
    )
    diag = {
        "object_count": len(object_ids),
        "frame_mask_count": len(rows),
        "eval_frame_count": int(acc.frame_count),
        "missing_mask_frame_count": int(missing_mask_frame_count),
        "selected_mask_missing_count": int(selected_mask_missing_count),
        "same_frame_duplicate_mask_count": int(same_frame_duplicate_mask_count),
        "pixel_collision_count": int(pixel_collision_count),
        "accepted_expansion_gt_checked": int(accepted_expansion_gt_checked),
        "accepted_expansion_same_gt": int(accepted_expansion_same_gt),
        "accepted_expansion_same_gt_rate": float(accepted_expansion_same_gt / max(1, accepted_expansion_gt_checked)),
        "total_pred_positive_pixels": int(total_pred_positive),
        "total_gt_positive_pixels": int(total_gt_positive),
        "mask_source": mask_source,
    }
    return summary, diag


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase3b = _read_json(PHASE3B_SUMMARY)
    phase7e = _read_json(PHASE7E_SUMMARY)
    base_rows = pd.read_parquet(PHASE7D_NODE_ROWS)
    component_rows = pd.read_csv(PHASE7D_COMPONENT_ROWS)
    feature_by_node, feature_by_frame_mask = _feature_maps()
    meta = _object_meta(base_rows, component_rows)
    object_ids = sorted(str(v) for v in base_rows["mv_object_id"].unique())

    xyz, mini = _load_xyz_and_mini()
    support, projection_rows, object_support_rows = _build_support_counts(xyz, mini, base_rows, object_ids)

    variant_rows: list[dict[str, Any]] = []
    candidate_rows_all: list[dict[str, Any]] = []
    materialized_rows_all: list[dict[str, Any]] = []
    for spec in VARIANTS:
        rows, candidate_rows, expand_diag = _expand_rows(
            xyz=xyz,
            mini=mini,
            base_rows=base_rows,
            support=support,
            object_ids=object_ids,
            meta=meta,
            feature_by_frame_mask=feature_by_frame_mask,
            spec=spec,
        )
        scores = _scores(rows)
        for row in rows:
            row = dict(row)
            row["variant_id"] = f"{VARIANT_PREFIX}_{spec['variant_id']}"
            row["score"] = scores[str(row["mv_object_id"])]
            row["object_score"] = scores[str(row["mv_object_id"])]
            materialized_rows_all.append(row)
        candidate_rows_all.extend(candidate_rows)
        summary, eval_diag = _evaluate(rows, scores, meta)
        variant_rows.append(
            {
                "schema_version": "stream4d_v102_phase7h_variant_metric_v1",
                "phase_id": PHASE_ID,
                "variant_id": spec["variant_id"],
                "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
                "expand": bool(spec["expand"]),
                "min_seed_support": spec["min_seed_support"],
                "min_vote_count": spec["min_vote_count"],
                "min_support_fraction": spec["min_support_fraction"],
                "min_mask_coverage": spec["min_mask_coverage"],
                "entropy_max": spec["entropy_max"],
                "drop_broad": spec["drop_broad"],
                **expand_diag,
                **eval_diag,
                "MV_AP_window": summary.get("ap"),
                "MV_AP50_window": summary.get("ap50"),
                "MV_AP25_window": summary.get("ap25"),
                "MV_AP_scene": None,
                "MV_AP50_scene": None,
                "scene_metric_computed": False,
                "scene_metric_not_computed_reason": "Phase7h evaluates only scene0050_00/c0000 chunk32 frames 0..155 stride5; full-scene/local2history scene metric is not computed.",
                "ScoreFreeMatch50_window": (summary.get("score_free_match_at_050") or {}).get("recall"),
                "ScoreFreeMatch25_window": (summary.get("score_free_match_at_025") or {}).get("recall"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )

    phase7e_ap50 = _num(phase7e.get("best_MV_AP50_window"))
    phase7e_sf50 = _num(phase7e.get("best_ScoreFreeMatch50_window"))
    best = max(
        variant_rows,
        key=lambda row: (
            _num(row.get("ScoreFreeMatch50_window")),
            _num(row.get("MV_AP50_window")),
            _num(row.get("MV_AP_window")),
            _num(row.get("accepted_expansion_same_gt_rate")),
        ),
    )
    best_delta_ap50 = _num(best.get("MV_AP50_window")) - phase7e_ap50
    best_delta_sf50 = _num(best.get("ScoreFreeMatch50_window")) - phase7e_sf50
    improves = bool(best_delta_ap50 > 1e-12 or best_delta_sf50 > 1e-12)
    safe_fill = _num(best.get("accepted_expansion_same_gt_rate")) >= 0.80 or int(_num(best.get("accepted_expansion_count"))) == 0
    projection_valid_rate_any = float(np.mean([row.get("inside_image_primitive_count", 0) > 0 for row in projection_rows if "camera_index" in row]))
    support_ge1_objects = sum(int(row["primitive_support_ge1_count"]) > 0 for row in object_support_rows)
    decision = (
        "PASS_PHASE7H_CHUNK32_PRIMITIVE_SUPPORT_LOCAL_IMPROVES__FORMAL_TARGET_NOT_CLAIMED"
        if improves and safe_fill
        else "NO_GO_PHASE7H_CHUNK32_PRIMITIVE_SUPPORT_NO_SAFE_LOCAL_GAIN"
    )
    gate_rows = [
        {
            "schema_version": "stream4d_v102_phase7h_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "chunk32_3dgs_inputs_present",
            "pass": bool(PLY_PATH.exists() and MINI_NPZ.exists()),
            "observed": f"ply={_rel(PLY_PATH)}; mini_npz={_rel(MINI_NPZ)}",
            "required": "chunk32 DA3-GIANT PLY and mini_npz",
        },
        {
            "schema_version": "stream4d_v102_phase7h_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "object_primitive_support_available",
            "pass": bool(support_ge1_objects == len(object_support_rows) and support_ge1_objects > 0),
            "observed": f"{support_ge1_objects}/{len(object_support_rows)} objects have >=1 supported primitive",
            "required": "all materialized Phase7d objects should have at least one support primitive",
        },
        {
            "schema_version": "stream4d_v102_phase7h_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_local_diagnostic_improves_over_phase7e",
            "pass": bool(improves),
            "observed": f"delta_ap50={best_delta_ap50}; delta_sf50={best_delta_sf50}",
            "required": ">0 local AP50 or ScoreFreeMatch50 delta",
        },
        {
            "schema_version": "stream4d_v102_phase7h_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "best_expansion_same_gt_rate_ge_0p80_diagnostic",
            "pass": bool(safe_fill),
            "observed": best.get("accepted_expansion_same_gt_rate"),
            "required": ">=0.80 diagnostic same-GT for accepted primitive-support fill rows",
        },
        {
            "schema_version": "stream4d_v102_phase7h_gate_v1",
            "phase_id": PHASE_ID,
            "gate": "formal_v102_target_achieved",
            "pass": False,
            "observed": "not claimed from local chunk32 primitive-support diagnostic",
            "required": "full-dev/holdout formal AP repair gate",
        },
    ]

    _write_csv(OUT_DIR / "projection_frame_rows.csv", projection_rows)
    _write_csv(OUT_DIR / "object_primitive_support_rows.csv", object_support_rows)
    _write_csv(OUT_DIR / "expansion_candidate_rows.csv", candidate_rows_all)
    _write_csv(OUT_DIR / "materialized_expanded_rows.csv", materialized_rows_all)
    _write_csv(OUT_DIR / "expansion_variant_metric_rows.csv", variant_rows)
    _write_csv(OUT_DIR / "expansion_gate_rows.csv", gate_rows)
    summary = {
        "schema_version": "stream4d_v102_phase7h_chunk32_primitive_support_shape_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "metric_scope": "chunk32_scene0050_local_diagnostic_not_full_dev",
        "variant_count": len(VARIANTS),
        "primitive_count": int(len(xyz)),
        "camera_count": int(min(len(mini["extrinsics"]), len(mini["intrinsics"]), len(mini["depth"]))),
        "phase3b_gaussian_count": phase3b.get("gaussian_count"),
        "phase3b_frame_count": phase3b.get("frame_count"),
        "phase3b_reprojection_valid_rate": phase3b.get("reprojection_valid_rate"),
        "projection_valid_rate_any_frame_row_proxy": projection_valid_rate_any,
        "object_count": int(len(object_ids)),
        "objects_with_support_ge1": int(support_ge1_objects),
        "phase7e_best_MV_AP50_window": phase7e_ap50,
        "phase7e_best_ScoreFreeMatch50_window": phase7e_sf50,
        "best_variant_id": best["variant_id"],
        "best_MV_AP_window": best.get("MV_AP_window"),
        "best_MV_AP50_window": best.get("MV_AP50_window"),
        "best_MV_AP25_window": best.get("MV_AP25_window"),
        "best_MV_AP_scene": None,
        "best_MV_AP50_scene": None,
        "scene_metric_computed": False,
        "scene_metric_not_computed_reason": "Phase7h evaluates only scene0050_00/c0000 chunk32 frames 0..155 stride5; full-scene/local2history scene metric is not computed.",
        "best_ScoreFreeMatch50_window": best.get("ScoreFreeMatch50_window"),
        "best_ScoreFreeMatch25_window": best.get("ScoreFreeMatch25_window"),
        "best_delta_MV_AP50_window_vs_phase7e": best_delta_ap50,
        "best_delta_ScoreFreeMatch50_window_vs_phase7e": best_delta_sf50,
        "best_accepted_expansion_count": best.get("accepted_expansion_count"),
        "best_wta_drop_count": best.get("wta_drop_count"),
        "best_frame_mask_count": best.get("frame_mask_count"),
        "best_accepted_expansion_same_gt_rate": best.get("accepted_expansion_same_gt_rate"),
        "best_pixel_collision_count": best.get("pixel_collision_count"),
        "local_diagnostic_improves": improves,
        "best_expansion_diagnostic_safe": safe_fill,
        "formal_v102_target_achieved": False,
        "formal_target_blocker": "Phase7h is local chunk32 primitive-support missing-frame expansion; Phase6 full repair remains blocked by Phase1b.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": (
            "Primitive-support candidates use DA3-GIANT chunk32 Gaussian projections into CropFormer masks and fixed Phase7d seed objects. "
            "GT is used only after rows are fixed for SparseSceneIoU/AP and diagnostic same-GT analysis."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "inputs": {
            "ply": _rel(PLY_PATH),
            "mini_npz": _rel(MINI_NPZ),
            "phase7d_node_rows": _rel(PHASE7D_NODE_ROWS),
            "feature_rows": _rel(FEATURE_ROWS),
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "projection_frame_rows": _rel(OUT_DIR / "projection_frame_rows.csv"),
            "object_primitive_support_rows": _rel(OUT_DIR / "object_primitive_support_rows.csv"),
            "expansion_candidate_rows": _rel(OUT_DIR / "expansion_candidate_rows.csv"),
            "materialized_expanded_rows": _rel(OUT_DIR / "materialized_expanded_rows.csv"),
            "expansion_variant_metric_rows": _rel(OUT_DIR / "expansion_variant_metric_rows.csv"),
            "expansion_gate_rows": _rel(OUT_DIR / "expansion_gate_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
