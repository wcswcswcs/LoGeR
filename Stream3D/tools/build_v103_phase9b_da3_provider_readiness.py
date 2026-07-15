from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v103_phase9b_da3_provider_readiness"
PLAN_DOC = ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
PHASE3_SUMMARY = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter" / "summary.json"
PHASE9A_ROWS = AUDIT_ROOT / "v103_phase9a_da3_chunk32_provider_export" / "chunk32_export_rows.csv"

IMAGE_PIXEL_COUNT = 968 * 1296
BROAD_MASK_AREA_RATIO = 0.20
MAX_AUDIT_GAP = 4
PAIR_KEY_BASE = 100000


SCENES = {
    "scene0011_00": {
        "input_manifest": AUDIT_ROOT
        / "v98_phase1_provider_contract"
        / "da3_streaming_full_scene0011_input"
        / "frame_manifest_rows.csv",
        "mask_root": STREAM3D
        / "outputs"
        / "cache"
        / "v66_cropformer_chunk_masks"
        / "scene0011_00"
        / "stride_5"
        / "cropformer_conf_0p500"
        / "mask2former_hornet_3x"
        / "final_processed"
        / "scene0011_00"
        / "output_Cropformer"
        / "mask",
        "mask_observation_table": AUDIT_ROOT
        / "v66_soma_fullscene_pipeline_scene0011_00_stride5_conf02_integrated_d4rt"
        / "observation_tables"
        / "mask_observation_table.csv",
        "aggregation_json": STREAM3D / "data" / "scannet" / "processed" / "scene0011_00" / "scene0011_00.aggregation.json",
        "feature_store": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011" / "mask_features.npz",
        "feature_manifest": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011" / "feature_store_manifest.json",
        "provider_id": "P6_DA3_GIANT_1_1_3DGS_official_scene0011",
        "artifact_candidates": [
            (
                AUDIT_ROOT
                / "v103_phase9a_da3_chunk32_provider_export"
                / "scene0011_00_chunk32_process252"
                / "gs_ply"
                / "0000.ply",
                AUDIT_ROOT
                / "v103_phase9a_da3_chunk32_provider_export"
                / "scene0011_00_chunk32_process252"
                / "exports"
                / "mini_npz"
                / "results.npz",
            )
        ],
    },
    "scene0050_00": {
        "input_manifest": AUDIT_ROOT
        / "v98_phase1_provider_contract"
        / "da3_streaming_d4rt32o3_scene0050_input119"
        / "frame_manifest_rows.csv",
        "mask_root": STREAM3D
        / "outputs"
        / "cache"
        / "v65_cropformer_chunk_masks"
        / "scene0050_00"
        / "stride_5"
        / "cropformer_conf_0p500"
        / "mask2former_hornet_3x"
        / "final_processed"
        / "scene0050_00"
        / "output_Cropformer"
        / "mask",
        "mask_observation_table": AUDIT_ROOT
        / "v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt"
        / "observation_tables"
        / "mask_observation_table.csv",
        "aggregation_json": STREAM3D / "data" / "scannet" / "processed" / "scene0050_00" / "scene0050_00.aggregation.json",
        "feature_store": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_features.npz",
        "feature_manifest": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "feature_store_manifest.json",
        "provider_id": "P6_DA3_GIANT_1_1_3DGS_official_scene0050",
        "artifact_candidates": [
            (
                AUDIT_ROOT
                / "v103_phase9a_da3_chunk32_provider_export"
                / "scene0050_00_chunk32_process252"
                / "gs_ply"
                / "0000.ply",
                AUDIT_ROOT
                / "v103_phase9a_da3_chunk32_provider_export"
                / "scene0050_00_chunk32_process252"
                / "exports"
                / "mini_npz"
                / "results.npz",
            ),
            (
                AUDIT_ROOT / "v102_phase2b_da3_giant_chunk32_audit" / "chunk32_process252" / "gs_ply" / "0000.ply",
                AUDIT_ROOT
                / "v102_phase2b_da3_giant_chunk32_audit"
                / "chunk32_process252"
                / "exports"
                / "mini_npz"
                / "results.npz",
            ),
        ],
    },
}


GEOMETRIC_VARIANTS = [
    {
        "variant_id": "strict_gap1_min5_r005_broad020",
        "max_gap": 1,
        "min_shared": 5,
        "ratio_min": 0.05,
        "broad_limit": 0.20,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "relax_gap2_min5_r002_broad020",
        "max_gap": 2,
        "min_shared": 5,
        "ratio_min": 0.02,
        "broad_limit": 0.20,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "relax_gap4_min1_r001_broad020",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "viewset_top2_gap4_min1_r001_broad020",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
        "topk_per_mask": 2,
    },
    {
        "variant_id": "no_broad_veto_gap2_min1_r001",
        "max_gap": 2,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": None,
        "topk_per_mask": 0,
    },
]


SEMANTIC_VARIANTS = [
    {
        "variant_id": "semantic_tau0p4_gap4_missing_allow",
        "semantic_cosine_min": 0.40,
        "missing_feature_policy": "allow",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p5_gap4_missing_allow",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "allow",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p6_gap4_missing_allow",
        "semantic_cosine_min": 0.60,
        "missing_feature_policy": "allow",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p5_gap4_missing_block",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "block",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p5_gap2_missing_allow",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "allow",
        "max_gap": 2,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p75_gap4_missing_block_highbarrier",
        "semantic_cosine_min": 0.75,
        "missing_feature_policy": "block",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p80_gap4_missing_block_highbarrier",
        "semantic_cosine_min": 0.80,
        "missing_feature_policy": "block",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p85_gap4_missing_block_highbarrier",
        "semantic_cosine_min": 0.85,
        "missing_feature_policy": "block",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p90_gap4_missing_block_highbarrier",
        "semantic_cosine_min": 0.90,
        "missing_feature_policy": "block",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prop(vertex_data: np.ndarray, name: str) -> np.ndarray:
    if vertex_data.dtype.names and name in vertex_data.dtype.names:
        return np.asarray(vertex_data[name])
    return np.full(len(vertex_data), np.nan)


def _attempt_id(scene_id: str, frame_count: int, process_res: int, frame_start_index: int) -> str:
    base = f"{scene_id}_chunk{int(frame_count)}"
    if int(frame_start_index) != 0:
        base += f"_start{int(frame_start_index):03d}"
    return f"{base}_process{int(process_res)}"


def _artifact_paths(
    scene_id: str,
    spec: dict[str, Any],
    *,
    phase9a_root: Path,
    frame_count: int,
    process_res: int,
    frame_start_index: int,
) -> tuple[Path | None, Path | None]:
    attempt_id = _attempt_id(scene_id, frame_count, process_res, frame_start_index)
    dynamic = (
        phase9a_root / attempt_id / "gs_ply" / "0000.ply",
        phase9a_root / attempt_id / "exports" / "mini_npz" / "results.npz",
    )
    if dynamic[0].exists() and dynamic[1].exists():
        return dynamic
    if int(frame_start_index) != 0:
        return None, None
    for ply, mini_npz in spec["artifact_candidates"]:
        if ply.exists() and mini_npz.exists():
            return ply, mini_npz
    return None, None


def _load_xyz(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    xyz = np.column_stack([_prop(vertex, "x"), _prop(vertex, "y"), _prop(vertex, "z")]).astype(np.float64)
    finite = np.all(np.isfinite(xyz), axis=1)
    return xyz[finite]


def _homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = extrinsic.astype(np.float64)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")


def _frame_manifest(path: Path, camera_count: int, frame_start_index: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("da3_frame_index").iloc[int(frame_start_index) : int(frame_start_index) + int(camera_count)].copy()
    df["da3_frame_index"] = df["da3_frame_index"].astype(int)
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def _semantic_map(path: Path) -> dict[int, str]:
    data = _read_json(path)
    return {int(group["objectId"]) + 1: str(group["label"]) for group in data.get("segGroups", [])}


def _mask_meta(scene_id: str, frame_ids: list[int], spec: dict[str, Any]) -> pd.DataFrame:
    meta = pd.read_csv(spec["mask_observation_table"])
    meta = meta[(meta["scene_id"] == scene_id) & (meta["frame_id"].isin(frame_ids))].copy()
    sem_map = _semantic_map(spec["aggregation_json"])
    meta["frame_id"] = meta["frame_id"].astype(int)
    meta["mask_id"] = meta["mask_id"].astype(int)
    meta["diagnostic_gt_instance"] = pd.to_numeric(meta["diagnostic_gt_instance"], errors="coerce")
    meta["diagnostic_gt_purity"] = pd.to_numeric(meta["diagnostic_gt_purity"], errors="coerce")
    meta["mask_area"] = pd.to_numeric(meta["mask_area"], errors="coerce")
    meta["mask_area_ratio"] = meta["mask_area"] / float(IMAGE_PIXEL_COUNT)
    meta["diagnostic_semantic_label"] = meta["diagnostic_gt_instance"].apply(
        lambda v: sem_map.get(int(v), "") if np.isfinite(v) and int(v) > 0 else ""
    )
    return meta


def _project_masks(
    *,
    scene_id: str,
    xyz: np.ndarray,
    mini: dict[str, np.ndarray],
    frame_df: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    extrinsics = np.asarray(mini["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(mini["intrinsics"], dtype=np.float64)
    depth = np.asarray(mini["depth"])
    camera_count = int(min(len(extrinsics), len(intrinsics), len(depth), len(frame_df)))
    mask_by_frame = np.zeros((camera_count, len(xyz)), dtype=np.uint16)
    points_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    visible_any = np.zeros(len(xyz), dtype=bool)
    source_rows: list[dict[str, Any]] = []
    for camera_index in range(camera_count):
        frame_id = int(frame_df.iloc[camera_index]["frame_id"])
        mask_path = spec["mask_root"] / f"{frame_id}.png"
        if not mask_path.exists():
            source_rows.append(
                {
                    "schema_version": "stream4d_v103_phase9b_mask_projection_source_row_v1",
                    "phase_id": "v103_phase9b_da3_provider_readiness",
                    "scene_id": scene_id,
                    "camera_index": camera_index,
                    "frame_id": frame_id,
                    "mask_path": _rel(mask_path),
                    "mask_exists": False,
                    "projected_inside_count": 0,
                    "participant_count": 0,
                    "uses_gt_for_prediction": False,
                }
            )
            continue
        ext = _homogeneous_extrinsic(extrinsics[camera_index])
        k = intrinsics[camera_index]
        h, w = int(depth[camera_index].shape[0]), int(depth[camera_index].shape[1])
        cam = (ext @ points_h.T).T[:, :3]
        z = cam[:, 2]
        valid_z = z > 1e-6
        u = np.full(len(xyz), np.nan, dtype=np.float64)
        v = np.full(len(xyz), np.nan, dtype=np.float64)
        u[valid_z] = k[0, 0] * (cam[valid_z, 0] / z[valid_z]) + k[0, 2]
        v[valid_z] = k[1, 1] * (cam[valid_z, 1] / z[valid_z]) + k[1, 2]
        inside = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        visible_any |= inside
        mask = np.asarray(Image.open(mask_path))
        mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
        inside_idx = np.flatnonzero(inside)
        xs = np.floor(np.clip(u[inside_idx] / float(w) * mask_w, 0, mask_w - 1)).astype(np.int32)
        ys = np.floor(np.clip(v[inside_idx] / float(h) * mask_h, 0, mask_h - 1)).astype(np.int32)
        mask_ids = mask[ys, xs].astype(np.uint16)
        positive = mask_ids > 0
        mask_by_frame[camera_index, inside_idx[positive]] = mask_ids[positive]
        source_rows.append(
            {
                "schema_version": "stream4d_v103_phase9b_mask_projection_source_row_v1",
                "phase_id": "v103_phase9b_da3_provider_readiness",
                "scene_id": scene_id,
                "camera_index": camera_index,
                "frame_id": frame_id,
                "mask_path": _rel(mask_path),
                "mask_exists": True,
                "processed_image_height": h,
                "processed_image_width": w,
                "mask_height": mask_h,
                "mask_width": mask_w,
                "projected_inside_count": int(np.sum(inside)),
                "participant_count": int(np.sum(positive)),
                "unique_mask_ids_hit": int(len(np.unique(mask_ids[positive]))) if np.any(positive) else 0,
                "uses_gt_for_prediction": False,
            }
        )
    return mask_by_frame, source_rows, float(np.mean(visible_any)) if len(visible_any) else 0.0


def _mask_summary_rows(scene_id: str, mask_by_frame: np.ndarray, frame_df: pd.DataFrame, meta: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for camera_index in range(mask_by_frame.shape[0]):
        frame_id = int(frame_df.iloc[camera_index]["frame_id"])
        ids, counts = np.unique(mask_by_frame[camera_index][mask_by_frame[camera_index] > 0], return_counts=True)
        count_map = {int(mask_id): int(count) for mask_id, count in zip(ids, counts)}
        frame_meta = meta[meta["frame_id"] == frame_id].sort_values("mask_id")
        for row in frame_meta.itertuples(index=False):
            rows.append(
                {
                    "schema_version": "stream4d_v103_phase9b_mask_primitive_summary_row_v1",
                    "phase_id": "v103_phase9b_da3_provider_readiness",
                    "scene_id": scene_id,
                    "camera_index": camera_index,
                    "frame_id": frame_id,
                    "mask_id": int(row.mask_id),
                    "mask_observation_id": f"{scene_id}:{frame_id}:{int(row.mask_id)}",
                    "participating_primitive_count": count_map.get(int(row.mask_id), 0),
                    "mask_area": float(row.mask_area) if np.isfinite(row.mask_area) else "",
                    "mask_area_ratio": float(row.mask_area_ratio) if np.isfinite(row.mask_area_ratio) else "",
                    "diagnostic_gt_instance": int(row.diagnostic_gt_instance)
                    if np.isfinite(row.diagnostic_gt_instance)
                    else "",
                    "diagnostic_gt_purity": float(row.diagnostic_gt_purity)
                    if np.isfinite(row.diagnostic_gt_purity)
                    else "",
                    "diagnostic_semantic_label": str(row.diagnostic_semantic_label),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    return rows


def _pair_count_map(mask_i: np.ndarray, mask_j: np.ndarray) -> dict[tuple[int, int], int]:
    valid = (mask_i > 0) & (mask_j > 0)
    if not np.any(valid):
        return {}
    keys = mask_i[valid].astype(np.int64) * PAIR_KEY_BASE + mask_j[valid].astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    return {(int(key // PAIR_KEY_BASE), int(key % PAIR_KEY_BASE)): int(count) for key, count in zip(unique, counts)}


def _build_bridge_rows(scene_id: str, provider_id: str, mask_by_frame: np.ndarray, frame_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_index = {(int(row.frame_id), int(row.mask_id)): row for row in meta.itertuples(index=False)}
    support: dict[tuple[int, int], int] = {}
    for camera_index in range(mask_by_frame.shape[0]):
        frame_id = int(frame_df.iloc[camera_index]["frame_id"])
        ids, counts = np.unique(mask_by_frame[camera_index][mask_by_frame[camera_index] > 0], return_counts=True)
        for mask_id, count in zip(ids, counts):
            support[(frame_id, int(mask_id))] = int(count)

    rows: list[dict[str, Any]] = []
    for i in range(mask_by_frame.shape[0]):
        frame_i = int(frame_df.iloc[i]["frame_id"])
        masks_i = sorted(meta.loc[meta["frame_id"] == frame_i, "mask_id"].astype(int).tolist())
        for j in range(i + 1, min(mask_by_frame.shape[0], i + MAX_AUDIT_GAP + 1)):
            frame_j = int(frame_df.iloc[j]["frame_id"])
            masks_j = sorted(meta.loc[meta["frame_id"] == frame_j, "mask_id"].astype(int).tolist())
            pair_counts = _pair_count_map(mask_by_frame[i], mask_by_frame[j])
            for mask_a in masks_i:
                meta_a = meta_index.get((frame_i, mask_a))
                support_a = support.get((frame_i, mask_a), 0)
                for mask_b in masks_j:
                    meta_b = meta_index.get((frame_j, mask_b))
                    support_b = support.get((frame_j, mask_b), 0)
                    shared = pair_counts.get((mask_a, mask_b), 0)
                    min_support = min(support_a, support_b)
                    union_support = support_a + support_b - shared
                    ratio_min = float(shared / max(min_support, 1))
                    ratio_union = float(shared / max(union_support, 1))
                    gt_a = float(meta_a.diagnostic_gt_instance) if meta_a is not None else np.nan
                    gt_b = float(meta_b.diagnostic_gt_instance) if meta_b is not None else np.nan
                    label_available = bool(np.isfinite(gt_a) and np.isfinite(gt_b) and gt_a > 0 and gt_b > 0)
                    same_gt = bool(label_available and int(gt_a) == int(gt_b))
                    different_gt = bool(label_available and int(gt_a) != int(gt_b))
                    sem_a = str(meta_a.diagnostic_semantic_label) if meta_a is not None else ""
                    sem_b = str(meta_b.diagnostic_semantic_label) if meta_b is not None else ""
                    same_semantic = bool(sem_a and sem_b and sem_a == sem_b)
                    same_semantic_diff_gt = bool(different_gt and same_semantic)
                    area_a = float(meta_a.mask_area_ratio) if meta_a is not None and np.isfinite(meta_a.mask_area_ratio) else 0.0
                    area_b = float(meta_b.mask_area_ratio) if meta_b is not None and np.isfinite(meta_b.mask_area_ratio) else 0.0
                    purity_a = float(meta_a.diagnostic_gt_purity) if meta_a is not None and np.isfinite(meta_a.diagnostic_gt_purity) else np.nan
                    purity_b = float(meta_b.diagnostic_gt_purity) if meta_b is not None and np.isfinite(meta_b.diagnostic_gt_purity) else np.nan
                    rows.append(
                        {
                            "schema_version": "stream4d_v103_phase9b_da3_bridge_row_v1",
                            "phase_id": "v103_phase9b_da3_provider_readiness",
                            "provider_id": provider_id,
                            "scene_id": scene_id,
                            "candidate_source": "chunk32_short_range_gap_le_4_cropformer_masks",
                            "frame_a": frame_i,
                            "frame_b": frame_j,
                            "frame_gap_index": j - i,
                            "mask_a_id": int(mask_a),
                            "mask_b_id": int(mask_b),
                            "mask_a_observation_id": f"{scene_id}:{frame_i}:{int(mask_a)}",
                            "mask_b_observation_id": f"{scene_id}:{frame_j}:{int(mask_b)}",
                            "mask_a_primitive_count": support_a,
                            "mask_b_primitive_count": support_b,
                            "gs_shared_gaussian_count": shared,
                            "gs_bridge_ratio_min_support": ratio_min,
                            "gs_bridge_ratio_union": ratio_union,
                            "final_bridge_score": ratio_min,
                            "broad_contamination_score": max(area_a, area_b),
                            "broad_contamination_risk": max(area_a, area_b) > BROAD_MASK_AREA_RATIO,
                            "diagnostic_gt_a": int(gt_a) if np.isfinite(gt_a) else -1,
                            "diagnostic_gt_b": int(gt_b) if np.isfinite(gt_b) else -1,
                            "diagnostic_semantic_label_a": sem_a,
                            "diagnostic_semantic_label_b": sem_b,
                            "diagnostic_same_semantic": same_semantic,
                            "diagnostic_same_gt": same_gt,
                            "diagnostic_different_gt": different_gt,
                            "diagnostic_same_semantic_different_gt": same_semantic_diff_gt,
                            "diagnostic_purity_min": float(np.nanmin([purity_a, purity_b])),
                            "uses_gt_for_prediction": False,
                            "uses_gt_for_diagnostic_labels": True,
                        }
                    )
    return pd.DataFrame(rows)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | str:
    labels = labels.astype(bool)
    pos = int(np.sum(labels))
    neg = int(np.sum(~labels))
    if pos == 0 or neg == 0:
        return ""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    for group_id in range(len(unique_scores)):
        idx = np.where(inverse == group_id)[0]
        if len(idx) > 1:
            ranks[idx] = float(np.mean(ranks[idx]))
    rank_sum_pos = float(np.sum(ranks[labels]))
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _topk_accept_mask(df: pd.DataFrame, base_accept: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0:
        return base_accept
    accepted = np.zeros(len(df), dtype=bool)
    work = df.loc[base_accept, ["mask_a_observation_id", "mask_b_observation_id", "final_bridge_score"]].copy()
    if len(work) == 0:
        return accepted
    work["_row_index"] = work.index.to_numpy()
    for col in ["mask_a_observation_id", "mask_b_observation_id"]:
        ranked = work.sort_values([col, "final_bridge_score"], ascending=[True, False])
        ranked["_rank"] = ranked.groupby(col).cumcount() + 1
        accepted[ranked.loc[ranked["_rank"] <= topk, "_row_index"].to_numpy(dtype=np.int64)] = True
    return accepted & base_accept


def _bridge_totals(df: pd.DataFrame) -> tuple[np.ndarray, int, int, int, float | str]:
    label_mask = (df["diagnostic_same_gt"] | df["diagnostic_different_gt"]).to_numpy(dtype=bool)
    labels = df.loc[label_mask, "diagnostic_same_gt"].to_numpy(dtype=bool)
    scores = df.loc[label_mask, "final_bridge_score"].to_numpy(dtype=np.float64)
    return (
        label_mask,
        int(np.sum(df["diagnostic_same_gt"])),
        int(np.sum(df["diagnostic_different_gt"])),
        int(np.sum(df["diagnostic_same_semantic_different_gt"])),
        _auc(scores, labels),
    )


def _geometric_variant_rows(scene_id: str, df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_mask, positive_total, negative_total, same_sem_neg_total, auc = _bridge_totals(df)
    rows: list[dict[str, Any]] = []
    for spec in GEOMETRIC_VARIANTS:
        broad_ok = (
            np.ones(len(df), dtype=bool)
            if spec["broad_limit"] is None
            else df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(spec["broad_limit"])
        )
        base_accept = (
            (df["frame_gap_index"].to_numpy(dtype=np.int64) <= int(spec["max_gap"]))
            & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= int(spec["min_shared"]))
            & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= float(spec["ratio_min"]))
            & broad_ok
        )
        accepted = _topk_accept_mask(df, base_accept, int(spec["topk_per_mask"]))
        accepted_labeled = accepted & label_mask
        tp = int(np.sum(accepted & df["diagnostic_same_gt"].to_numpy(dtype=bool)))
        fp = int(np.sum(accepted & df["diagnostic_different_gt"].to_numpy(dtype=bool)))
        fp_same_sem = int(np.sum(accepted & df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)))
        accepted_count = int(np.sum(accepted))
        accepted_labeled_count = int(np.sum(accepted_labeled))
        recall = float(tp / max(positive_total, 1)) if positive_total else ""
        diff_false = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        same_sem_false = float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        hard_neg_false = float(fp / max(negative_total, 1)) if negative_total else ""
        formal = bool(
            recall != ""
            and diff_false != ""
            and same_sem_false != ""
            and hard_neg_false != ""
            and auc != ""
            and recall >= 0.35
            and diff_false <= 0.20
            and same_sem_false <= 0.20
            and hard_neg_false <= 0.20
            and auc >= 0.65
        )
        rows.append(
            {
                "schema_version": "stream4d_v103_phase9b_geometric_variant_row_v1",
                "phase_id": "v103_phase9b_da3_provider_readiness",
                "scene_id": scene_id,
                "variant_id": spec["variant_id"],
                "accepted_count": accepted_count,
                "accepted_labeled_count": accepted_labeled_count,
                "true_positive_same_gt_count": tp,
                "false_positive_different_gt_count": fp,
                "false_positive_same_semantic_different_gt_count": fp_same_sem,
                "diagnostic_positive_pair_count": positive_total,
                "diagnostic_negative_pair_count": negative_total,
                "same_semantic_different_gt_hard_negative_count": same_sem_neg_total,
                "same_object_bridge_recall": recall,
                "different_gt_false_bridge_among_accepted": diff_false,
                "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false,
                "hard_negative_false_accept_rate": hard_neg_false,
                "bridge_auc": auc,
                "phase5_formal_bridge_gate_pass": formal,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    best = max(rows, key=lambda row: float(row["same_object_bridge_recall"]) if row["same_object_bridge_recall"] != "" else -1)
    bits = {
        "bridge_auc": auc,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_negative_pair_count": negative_total,
        "same_semantic_different_gt_hard_negative_count": same_sem_neg_total,
        "best_geometric_variant_id": best["variant_id"],
        "best_geometric_same_object_bridge_recall": best["same_object_bridge_recall"],
        "best_geometric_different_gt_false_bridge_among_accepted": best["different_gt_false_bridge_among_accepted"],
        "best_geometric_same_semantic_different_gt_false_bridge_among_accepted": best[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "geometric_bridge_gate_pass": any(bool(row["phase5_formal_bridge_gate_pass"]) for row in rows),
    }
    return rows, bits


def _load_feature_map(spec: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    store = np.load(spec["feature_store"])
    features = store["features"].astype(np.float32)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    ids = [str(x) for x in store["mask_observation_id"]]
    return {mask_id: features[i] for i, mask_id in enumerate(ids)}, _read_json(spec["feature_manifest"])


def _add_semantic_scores(df: pd.DataFrame, feature_map: dict[str, np.ndarray]) -> pd.DataFrame:
    cosines: list[float] = []
    available: list[bool] = []
    for row in df[["mask_a_observation_id", "mask_b_observation_id"]].itertuples(index=False):
        fa = feature_map.get(str(row.mask_a_observation_id))
        fb = feature_map.get(str(row.mask_b_observation_id))
        if fa is None or fb is None:
            cosines.append(np.nan)
            available.append(False)
        else:
            cosines.append(float(np.dot(fa, fb)))
            available.append(True)
    out = df.copy()
    out["semantic_residual_cosine"] = cosines
    out["semantic_residual_available"] = available
    return out


def _semantic_variant_rows(scene_id: str, df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_mask, positive_total, negative_total, same_sem_neg_total, auc = _bridge_totals(df)
    semantic_available_pair_count = int(np.sum(df["semantic_residual_available"]))
    semantic_available_rate = float(np.mean(df["semantic_residual_available"])) if len(df) else 0.0
    rows: list[dict[str, Any]] = []
    for spec in SEMANTIC_VARIANTS:
        broad_ok = (
            np.ones(len(df), dtype=bool)
            if spec["broad_limit"] is None
            else df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(spec["broad_limit"])
        )
        semantic_ok = df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= float(spec["semantic_cosine_min"])
        if spec["missing_feature_policy"] == "allow":
            semantic_ok = semantic_ok | (~df["semantic_residual_available"].to_numpy(dtype=bool))
        accepted = (
            (df["frame_gap_index"].to_numpy(dtype=np.int64) <= int(spec["max_gap"]))
            & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= int(spec["min_shared"]))
            & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= float(spec["ratio_min"]))
            & broad_ok
            & semantic_ok
        )
        accepted_labeled = accepted & label_mask
        tp = int(np.sum(accepted & df["diagnostic_same_gt"].to_numpy(dtype=bool)))
        fp = int(np.sum(accepted & df["diagnostic_different_gt"].to_numpy(dtype=bool)))
        fp_same_sem = int(np.sum(accepted & df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)))
        accepted_count = int(np.sum(accepted))
        accepted_labeled_count = int(np.sum(accepted_labeled))
        recall = float(tp / max(positive_total, 1)) if positive_total else ""
        diff_false = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        same_sem_false = float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        hard_neg_false = float(fp / max(negative_total, 1)) if negative_total else ""
        formal = bool(
            recall != ""
            and diff_false != ""
            and same_sem_false != ""
            and hard_neg_false != ""
            and auc != ""
            and recall >= 0.35
            and diff_false <= 0.20
            and same_sem_false <= 0.20
            and hard_neg_false <= 0.20
            and auc >= 0.65
        )
        rows.append(
            {
                "schema_version": "stream4d_v103_phase9b_semantic_variant_row_v1",
                "phase_id": "v103_phase9b_da3_provider_readiness",
                "scene_id": scene_id,
                "variant_id": spec["variant_id"],
                "semantic_cosine_min": spec["semantic_cosine_min"],
                "missing_feature_policy": spec["missing_feature_policy"],
                "accepted_count": accepted_count,
                "accepted_labeled_count": accepted_labeled_count,
                "true_positive_same_gt_count": tp,
                "false_positive_different_gt_count": fp,
                "false_positive_same_semantic_different_gt_count": fp_same_sem,
                "diagnostic_positive_pair_count": positive_total,
                "diagnostic_negative_pair_count": negative_total,
                "same_semantic_different_gt_hard_negative_count": same_sem_neg_total,
                "same_object_bridge_recall": recall,
                "different_gt_false_bridge_among_accepted": diff_false,
                "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false,
                "hard_negative_false_accept_rate": hard_neg_false,
                "bridge_auc": auc,
                "semantic_available_pair_count": semantic_available_pair_count,
                "semantic_available_rate": semantic_available_rate,
                "phase5_formal_bridge_gate_pass": formal,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    passing = [row for row in rows if bool(row["phase5_formal_bridge_gate_pass"])]
    best = max(
        passing or rows,
        key=lambda row: float(row["same_object_bridge_recall"]) if row["same_object_bridge_recall"] != "" else -1,
    )
    bits = {
        "semantic_available_pair_count": semantic_available_pair_count,
        "semantic_available_rate": semantic_available_rate,
        "best_semantic_variant_id": best["variant_id"],
        "best_semantic_same_object_bridge_recall": best["same_object_bridge_recall"],
        "best_semantic_different_gt_false_bridge_among_accepted": best["different_gt_false_bridge_among_accepted"],
        "best_semantic_same_semantic_different_gt_false_bridge_among_accepted": best[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "semantic_bridge_gate_pass": any(bool(row["phase5_formal_bridge_gate_pass"]) for row in rows),
    }
    return rows, bits


def _scene_failure(scene_id: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_phase9b_provider_scene_summary_row_v1",
        "phase_id": "v103_phase9b_da3_provider_readiness",
        "scene_id": scene_id,
        "provider_artifact_available": False,
        "provider_ready": False,
        "blocker": reason,
        "uses_gt_for_prediction": False,
    }


def _process_scene(
    scene_id: str,
    spec: dict[str, Any],
    out_dir: Path,
    *,
    phase9a_root: Path,
    frame_count: int,
    process_res: int,
    frame_start_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ply_path, mini_npz = _artifact_paths(
        scene_id,
        spec,
        phase9a_root=phase9a_root,
        frame_count=frame_count,
        process_res=process_res,
        frame_start_index=frame_start_index,
    )
    if ply_path is None or mini_npz is None:
        row = _scene_failure(scene_id, "DA3_CHUNK32_3DGS_ARTIFACT_MISSING")
        return row, {"failure_rows": [row]}

    t0 = time.time()
    xyz = _load_xyz(ply_path)
    with np.load(mini_npz) as data:
        mini = {key: np.asarray(data[key]) for key in data.files}
    camera_count = int(min(len(mini["extrinsics"]), len(mini["intrinsics"]), len(mini["depth"])))
    frame_df = _frame_manifest(spec["input_manifest"], camera_count, int(frame_start_index))
    frame_ids = frame_df["frame_id"].astype(int).tolist()
    meta = _mask_meta(scene_id, frame_ids, spec)
    mask_by_frame, source_rows, reprojection_valid_any = _project_masks(
        scene_id=scene_id,
        xyz=xyz,
        mini=mini,
        frame_df=frame_df,
        spec=spec,
    )
    mask_summary_rows = _mask_summary_rows(scene_id, mask_by_frame, frame_df, meta)
    bridge_df = _build_bridge_rows(scene_id, spec["provider_id"], mask_by_frame, frame_df, meta)
    geom_rows, geom_bits = _geometric_variant_rows(scene_id, bridge_df)
    feature_map, feature_manifest = _load_feature_map(spec)
    bridge_sem_df = _add_semantic_scores(bridge_df, feature_map)
    sem_rows, sem_bits = _semantic_variant_rows(scene_id, bridge_sem_df)

    scene_dir = out_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = scene_dir / "mask_pair_primitive_bridge_rows.parquet"
    bridge_sem_path = scene_dir / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    source_path = scene_dir / "mask_projection_source_rows.csv"
    mask_summary_path = scene_dir / "chunk32_mask_primitive_summary_rows.csv"
    geom_path = scene_dir / "geometric_variant_rows.csv"
    sem_path = scene_dir / "semantic_barrier_variant_rows.csv"
    mask_by_frame_path = scene_dir / "mask_by_frame.npy"
    xyz_path = scene_dir / "xyz.npy"
    bridge_df.to_parquet(bridge_path, index=False)
    bridge_sem_df.to_parquet(bridge_sem_path, index=False)
    np.save(mask_by_frame_path, mask_by_frame.astype(np.uint16, copy=False))
    np.save(xyz_path, xyz.astype(np.float32, copy=False))
    _write_csv(source_path, source_rows)
    _write_csv(mask_summary_path, mask_summary_rows)
    _write_csv(geom_path, geom_rows)
    _write_csv(sem_path, sem_rows)

    participant_counts = [int(row["participating_primitive_count"]) for row in mask_summary_rows]
    supported_mask_count = int(sum(count > 0 for count in participant_counts))
    mask_support_coverage = float(supported_mask_count / max(len(participant_counts), 1))
    provider_ready = bool(
        sem_bits["semantic_bridge_gate_pass"]
        and geom_bits["bridge_auc"] != ""
        and float(geom_bits["bridge_auc"]) >= 0.65
        and reprojection_valid_any >= 0.80
        and mask_support_coverage >= 0.50
    )
    blocker = ""
    if not provider_ready:
        missing = []
        if not sem_bits["semantic_bridge_gate_pass"]:
            missing.append("semantic_bridge_gate_fail")
        if geom_bits["bridge_auc"] == "" or float(geom_bits["bridge_auc"]) < 0.65:
            missing.append("bridge_auc_low_or_unavailable")
        if reprojection_valid_any < 0.80:
            missing.append("reprojection_valid_rate_low")
        if mask_support_coverage < 0.50:
            missing.append("mask_support_coverage_low")
        blocker = "|".join(missing)

    row = {
        "schema_version": "stream4d_v103_phase9b_provider_scene_summary_row_v1",
        "phase_id": "v103_phase9b_da3_provider_readiness",
        "scene_id": scene_id,
        "provider_id": spec["provider_id"],
        "provider_name": "official_DA3_GIANT_1_1_3DGS",
        "model_variant": "depth-anything/DA3-GIANT-1.1",
        "ply_file": _rel(ply_path),
        "mini_npz_file": _rel(mini_npz),
        "frame_count": camera_count,
        "frame_start_index": int(frame_start_index),
        "frame_ids": frame_ids,
        "process_res": int(np.asarray(mini["depth"]).shape[-1]) if "depth" in mini else "",
        "gaussian_or_surfel_count": int(len(xyz)),
        "reprojection_valid_rate": reprojection_valid_any,
        "mask_observation_count": int(len(mask_summary_rows)),
        "mask_support_coverage": mask_support_coverage,
        "mask_participation_count_mean": float(np.mean(participant_counts)) if participant_counts else 0.0,
        "candidate_pair_count": int(len(bridge_df)),
        "bridge_auc": geom_bits["bridge_auc"],
        "geometric_bridge_gate_pass": geom_bits["geometric_bridge_gate_pass"],
        "best_geometric_variant_id": geom_bits["best_geometric_variant_id"],
        "best_geometric_same_object_bridge_recall": geom_bits["best_geometric_same_object_bridge_recall"],
        "best_geometric_different_gt_false_bridge_among_accepted": geom_bits[
            "best_geometric_different_gt_false_bridge_among_accepted"
        ],
        "semantic_backend": feature_manifest.get("backend"),
        "semantic_available_pair_count": sem_bits["semantic_available_pair_count"],
        "semantic_available_rate": sem_bits["semantic_available_rate"],
        "semantic_bridge_gate_pass": sem_bits["semantic_bridge_gate_pass"],
        "best_semantic_variant_id": sem_bits["best_semantic_variant_id"],
        "best_semantic_same_object_bridge_recall": sem_bits["best_semantic_same_object_bridge_recall"],
        "best_semantic_different_gt_false_bridge_among_accepted": sem_bits[
            "best_semantic_different_gt_false_bridge_among_accepted"
        ],
        "best_semantic_same_semantic_different_gt_false_bridge_among_accepted": sem_bits[
            "best_semantic_same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "provider_ready": provider_ready,
        "blocker": blocker,
        "runtime_sec": time.time() - t0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "outputs": {
            "mask_pair_primitive_bridge_rows": _rel(bridge_path),
            "mask_pair_primitive_bridge_rows_with_semantic": _rel(bridge_sem_path),
            "mask_projection_source_rows": _rel(source_path),
            "chunk32_mask_primitive_summary_rows": _rel(mask_summary_path),
            "geometric_variant_rows": _rel(geom_path),
            "semantic_barrier_variant_rows": _rel(sem_path),
            "mask_by_frame": _rel(mask_by_frame_path),
            "xyz": _rel(xyz_path),
        },
    }
    return row, {"failure_rows": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["all", *SCENES.keys()], default="all")
    parser.add_argument("--output-root", default=str(OUT_DIR))
    parser.add_argument("--phase9a-root", default=str(AUDIT_ROOT / "v103_phase9a_da3_chunk32_provider_export"))
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--frame-start-index", type=int, default=0)
    parser.add_argument("--process-res", type=int, default=252)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_root)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    phase9a_root = Path(args.phase9a_root)
    if not phase9a_root.is_absolute():
        phase9a_root = ROOT / phase9a_root
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_ids = list(SCENES) if args.scene == "all" else [args.scene]
    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        row, extra = _process_scene(
            scene_id,
            SCENES[scene_id],
            out_dir,
            phase9a_root=phase9a_root,
            frame_count=int(args.frame_count),
            process_res=int(args.process_res),
            frame_start_index=int(args.frame_start_index),
        )
        scene_rows.append(row)
        failure_rows.extend(extra["failure_rows"])

    scene_summary_path = out_dir / "provider_scene_summary_rows.csv"
    failure_path = out_dir / "failure_rows.csv"
    gate_path = out_dir / "provider_gate_rows.csv"
    _write_csv(scene_summary_path, scene_rows)
    _write_csv(failure_path, failure_rows)
    ready_scene_count = sum(bool(row.get("provider_ready")) for row in scene_rows)
    phase3 = _read_json(PHASE3_SUMMARY) if PHASE3_SUMMARY.exists() else {}
    phase9a_rows_available = (phase9a_root / "chunk32_export_rows.csv").exists()
    gate_rows = [
        {
            "gate_id": "d4rt_phase3_blocker_triggers_da3_branch",
            "pass": phase3.get("phase3_pass") is False,
            "expected": "phase3_pass=false",
            "observed": phase3.get("decision", ""),
            "scope": "v103_phase3",
        },
        {
            "gate_id": "phase9a_export_rows_available",
            "pass": phase9a_rows_available,
            "expected": True,
            "observed": phase9a_rows_available,
            "scope": "v103_phase9a",
        },
        {
            "gate_id": "all_requested_scenes_provider_ready",
            "pass": ready_scene_count == len(scene_rows),
            "expected": len(scene_rows),
            "observed": ready_scene_count,
            "scope": "v103_phase9b",
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": all(row.get("uses_gt_for_prediction") is False for row in scene_rows),
            "expected": False,
            "observed": False,
            "scope": "all",
        },
    ]
    _write_csv(gate_path, gate_rows)
    decision = (
        "PASS_DA3_PROVIDER_READY_FOR_UNIFIED_AFFINITY_PIPELINE"
        if ready_scene_count == len(scene_rows) and scene_rows
        else "PARTIAL_DA3_PROVIDER_READY__MISSING_OR_FAILED_SCENE"
        if ready_scene_count > 0
        else "NO_GO_DA3_PROVIDER_READINESS"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9b_da3_provider_readiness_summary_v1",
        "phase_id": "v103_phase9b_da3_provider_readiness",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "plan_doc": _rel(PLAN_DOC),
        "phase3_trigger_decision": phase3.get("decision", ""),
        "scene_count": len(scene_rows),
        "frame_count": int(args.frame_count),
        "frame_start_index": int(args.frame_start_index),
        "phase9a_root": _rel(phase9a_root),
        "ready_scene_count": ready_scene_count,
        "failure_count": len(failure_rows),
        "truthfulness_note": (
            "DA3 provider readiness is evaluated as a primitive-provider branch only. "
            "Diagnostic GT labels score bridge recall/false-bridge; they are not used to accept prediction edges or tune AP."
        ),
        "outputs": {
            "summary": _rel(out_dir / "summary.json"),
            "provider_scene_summary_rows": _rel(scene_summary_path),
            "provider_gate_rows": _rel(gate_path),
            "failure_rows": _rel(failure_path),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if ready_scene_count == len(scene_rows) and scene_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
