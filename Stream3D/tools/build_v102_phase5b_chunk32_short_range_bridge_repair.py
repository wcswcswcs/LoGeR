from __future__ import annotations

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
OUT_DIR = AUDIT_ROOT / "v102_phase5b_chunk32_short_range_bridge_repair"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PHASE1_DIR = AUDIT_ROOT / "v102_phase1_fragment_casebook"
CHUNK32_DIR = AUDIT_ROOT / "v102_phase2b_da3_giant_chunk32_audit" / "chunk32_process252"
PLY_PATH = CHUNK32_DIR / "gs_ply" / "0000.ply"
MINI_NPZ = CHUNK32_DIR / "exports" / "mini_npz" / "results.npz"
FRAME_MANIFEST = AUDIT_ROOT / "v98_phase1_provider_contract" / "da3_streaming_d4rt32o3_scene0050_input119" / "frame_manifest_rows.csv"
MASK_OBSERVATION_TABLE = (
    AUDIT_ROOT
    / "v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt"
    / "observation_tables"
    / "mask_observation_table.csv"
)
CROPFORMER_MASK_ROOT = (
    STREAM3D
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
    / "mask"
)
AGGREGATION_JSON = STREAM3D / "data" / "scannet" / "processed" / "scene0050_00" / "scene0050_00.aggregation.json"

IMAGE_PIXEL_COUNT = 968 * 1296
BROAD_MASK_AREA_RATIO = 0.20
MAX_AUDIT_GAP = 4
PAIR_KEY_BASE = 10000


VARIANTS = [
    {
        "variant_id": "strict_gap1_min5_r005_broad020",
        "repair_family": "chunk32_short_range_baseline",
        "max_gap": 1,
        "min_shared": 5,
        "ratio_min": 0.05,
        "broad_limit": 0.20,
        "topk_per_mask": 0,
        "description": "Direct adjacent-frame shared-Gaussian bridge with the previous fixed threshold.",
    },
    {
        "variant_id": "relax_gap2_min5_r002_broad020",
        "repair_family": "recall_low_relax_short_range_path",
        "max_gap": 2,
        "min_shared": 5,
        "ratio_min": 0.02,
        "broad_limit": 0.20,
        "topk_per_mask": 0,
        "description": "Allow a two-step short-range path and lower the ratio threshold.",
    },
    {
        "variant_id": "relax_gap4_min1_r001_broad020",
        "repair_family": "recall_low_relax_short_range_path",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
        "topk_per_mask": 0,
        "description": "Stress recall by allowing gap<=4 and any persistent shared Gaussian.",
    },
    {
        "variant_id": "viewset_top2_gap4_min1_r001_broad020",
        "repair_family": "viewset_semantic_proxy_topk_without_semantic_features",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
        "topk_per_mask": 2,
        "description": "Keep only top-2 geometric partners per mask as a view-set support proxy.",
    },
    {
        "variant_id": "no_broad_veto_gap2_min1_r001",
        "repair_family": "broad_veto_sensitivity",
        "max_gap": 2,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": None,
        "topk_per_mask": 0,
        "description": "Ablate the broad-mask veto to test whether recall is blocked by broad filtering.",
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
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


def _load_xyz() -> np.ndarray:
    ply = PlyData.read(str(PLY_PATH))
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


def _frame_manifest(camera_count: int) -> pd.DataFrame:
    df = pd.read_csv(FRAME_MANIFEST)
    df = df.sort_values("da3_frame_index").head(camera_count).copy()
    df["da3_frame_index"] = df["da3_frame_index"].astype(int)
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def _semantic_map() -> dict[int, str]:
    data = json.loads(AGGREGATION_JSON.read_text(encoding="utf-8"))
    # ScanNet 2D instance ids are 1-based while aggregation objectId is 0-based.
    return {int(group["objectId"]) + 1: str(group["label"]) for group in data.get("segGroups", [])}


def _mask_meta(frame_ids: list[int]) -> pd.DataFrame:
    meta = pd.read_csv(MASK_OBSERVATION_TABLE)
    meta = meta[(meta["scene_id"] == "scene0050_00") & (meta["frame_id"].isin(frame_ids))].copy()
    sem_map = _semantic_map()
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


def _project_masks(xyz: np.ndarray, mini: dict[str, np.ndarray], frame_df: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    extrinsics = np.asarray(mini["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(mini["intrinsics"], dtype=np.float64)
    depth = np.asarray(mini["depth"])
    camera_count = int(min(len(extrinsics), len(intrinsics), len(depth), len(frame_df)))
    mask_by_frame = np.zeros((camera_count, len(xyz)), dtype=np.uint16)
    points_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    source_rows: list[dict[str, Any]] = []
    for camera_index in range(camera_count):
        frame_id = int(frame_df.iloc[camera_index]["frame_id"])
        mask_path = CROPFORMER_MASK_ROOT / f"{frame_id}.png"
        if not mask_path.exists():
            source_rows.append(
                {
                    "schema_version": "stream4d_v102_phase5b_mask_projection_source_row_v1",
                    "phase_id": "v102_phase5b_chunk32_short_range_bridge_repair",
                    "camera_index": camera_index,
                    "frame_id": frame_id,
                    "mask_path": _rel(mask_path),
                    "mask_exists": False,
                    "projected_inside_count": 0,
                    "participant_count": 0,
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
                "schema_version": "stream4d_v102_phase5b_mask_projection_source_row_v1",
                "phase_id": "v102_phase5b_chunk32_short_range_bridge_repair",
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
            }
        )
    return mask_by_frame, source_rows


def _mask_summary_rows(mask_by_frame: np.ndarray, frame_df: pd.DataFrame, meta: pd.DataFrame) -> list[dict[str, Any]]:
    meta_index = {
        (int(row.frame_id), int(row.mask_id)): row
        for row in meta.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for camera_index in range(mask_by_frame.shape[0]):
        frame_id = int(frame_df.iloc[camera_index]["frame_id"])
        ids, counts = np.unique(mask_by_frame[camera_index][mask_by_frame[camera_index] > 0], return_counts=True)
        count_map = {int(mask_id): int(count) for mask_id, count in zip(ids, counts)}
        frame_meta = meta[meta["frame_id"] == frame_id].sort_values("mask_id")
        for row in frame_meta.itertuples(index=False):
            rows.append(
                {
                    "schema_version": "stream4d_v102_phase5b_mask_primitive_summary_row_v1",
                    "phase_id": "v102_phase5b_chunk32_short_range_bridge_repair",
                    "scene_id": "scene0050_00",
                    "camera_index": camera_index,
                    "frame_id": frame_id,
                    "mask_id": int(row.mask_id),
                    "mask_observation_id": f"scene0050_00:{frame_id}:{int(row.mask_id)}",
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
    # meta_index is intentionally built above to keep the row construction obvious for audit.
    _ = meta_index
    return rows


def _pair_count_map(mask_i: np.ndarray, mask_j: np.ndarray) -> dict[tuple[int, int], int]:
    valid = (mask_i > 0) & (mask_j > 0)
    if not np.any(valid):
        return {}
    keys = mask_i[valid].astype(np.int64) * PAIR_KEY_BASE + mask_j[valid].astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    return {(int(key // PAIR_KEY_BASE), int(key % PAIR_KEY_BASE)): int(count) for key, count in zip(unique, counts)}


def _build_bridge_rows(mask_by_frame: np.ndarray, frame_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_index = {
        (int(row.frame_id), int(row.mask_id)): row
        for row in meta.itertuples(index=False)
    }
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
                    purity_a = (
                        float(meta_a.diagnostic_gt_purity)
                        if meta_a is not None and np.isfinite(meta_a.diagnostic_gt_purity)
                        else np.nan
                    )
                    purity_b = (
                        float(meta_b.diagnostic_gt_purity)
                        if meta_b is not None and np.isfinite(meta_b.diagnostic_gt_purity)
                        else np.nan
                    )
                    rows.append(
                        {
                            "schema_version": "stream4d_v102_phase5b_chunk32_bridge_row_v1",
                            "phase_id": "v102_phase5b_chunk32_short_range_bridge_repair",
                            "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
                            "candidate_source": "chunk32_short_range_gap_le_4_cropformer_masks",
                            "scene_id": "scene0050_00",
                            "frame_a": frame_i,
                            "frame_b": frame_j,
                            "frame_gap_index": j - i,
                            "mask_a_id": int(mask_a),
                            "mask_b_id": int(mask_b),
                            "mask_a_observation_id": f"scene0050_00:{frame_i}:{int(mask_a)}",
                            "mask_b_observation_id": f"scene0050_00:{frame_j}:{int(mask_b)}",
                            "mask_a_primitive_count": support_a,
                            "mask_b_primitive_count": support_b,
                            "gs_shared_gaussian_count": shared,
                            "gs_bridge_ratio_min_support": ratio_min,
                            "gs_bridge_ratio_union": ratio_union,
                            "final_bridge_score": ratio_min,
                            "broad_contamination_score": max(area_a, area_b),
                            "broad_contamination_risk": max(area_a, area_b) > BROAD_MASK_AREA_RATIO,
                            "diagnostic_gt_a": int(gt_a) if np.isfinite(gt_a) else "",
                            "diagnostic_gt_b": int(gt_b) if np.isfinite(gt_b) else "",
                            "diagnostic_semantic_label_a": sem_a,
                            "diagnostic_semantic_label_b": sem_b,
                            "diagnostic_same_semantic": same_semantic,
                            "diagnostic_same_gt": same_gt,
                            "diagnostic_different_gt": different_gt,
                            "diagnostic_same_semantic_different_gt": same_semantic_diff_gt,
                            "diagnostic_purity_min": float(np.nanmin([purity_a, purity_b])),
                            "same_frame_competing_cannot_link": False,
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


def _variant_metrics(df: pd.DataFrame, phase1: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_mask = (df["diagnostic_same_gt"] | df["diagnostic_different_gt"]).to_numpy(dtype=bool)
    labels = df.loc[label_mask, "diagnostic_same_gt"].to_numpy(dtype=bool)
    scores = df.loc[label_mask, "final_bridge_score"].to_numpy(dtype=np.float64)
    auc = _auc(scores, labels)
    positive_total = int(np.sum(df["diagnostic_same_gt"]))
    negative_total = int(np.sum(df["diagnostic_different_gt"]))
    same_semantic_negative_total = int(np.sum(df["diagnostic_same_semantic_different_gt"]))
    variant_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    baseline_recall = 0.0
    baseline_false = 1.0

    for spec in VARIANTS:
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
        diff_false_among_accepted = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        same_sem_false_among_accepted = (
            float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        )
        same_sem_hn_false_accept = (
            float(fp_same_sem / max(same_semantic_negative_total, 1)) if same_semantic_negative_total else ""
        )
        hard_negative_false_accept = float(fp / max(negative_total, 1)) if negative_total else ""
        if spec["variant_id"] == "strict_gap1_min5_r005_broad020":
            baseline_recall = float(recall) if recall != "" else 0.0
            baseline_false = float(diff_false_among_accepted) if diff_false_among_accepted != "" else 1.0
        formal_bridge_gate_pass = bool(
            recall != ""
            and diff_false_among_accepted != ""
            and same_sem_false_among_accepted != ""
            and hard_negative_false_accept != ""
            and auc != ""
            and recall >= 0.35
            and same_sem_false_among_accepted <= 0.20
            and diff_false_among_accepted <= 0.20
            and hard_negative_false_accept <= 0.20
            and auc >= 0.65
        )
        phase6_allowed = bool(formal_bridge_gate_pass and int(phase1.get("repair_candidate_pair_count", 0)) >= 30)
        row = {
            "schema_version": "stream4d_v102_phase5b_repair_variant_row_v1",
            "phase_id": "v102_phase5b_chunk32_short_range_bridge_repair",
            "variant_id": spec["variant_id"],
            "repair_family": spec["repair_family"],
            "description": spec["description"],
            "max_gap": spec["max_gap"],
            "min_shared_gaussian_count": spec["min_shared"],
            "ratio_min_threshold": spec["ratio_min"],
            "broad_veto_area_ratio": "" if spec["broad_limit"] is None else spec["broad_limit"],
            "topk_per_mask": spec["topk_per_mask"],
            "accepted_count": accepted_count,
            "accepted_labeled_count": accepted_labeled_count,
            "true_positive_same_gt_count": tp,
            "false_positive_different_gt_count": fp,
            "false_positive_same_semantic_different_gt_count": fp_same_sem,
            "diagnostic_positive_pair_count": positive_total,
            "diagnostic_negative_pair_count": negative_total,
            "same_semantic_different_gt_hard_negative_count": same_semantic_negative_total,
            "same_object_bridge_recall": recall,
            "different_gt_false_bridge_among_accepted": diff_false_among_accepted,
            "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false_among_accepted,
            "same_semantic_hard_negative_false_accept_rate": same_sem_hn_false_accept,
            "hard_negative_false_accept_rate": hard_negative_false_accept,
            "bridge_auc": auc,
            "recall_delta_vs_strict_gap1": (float(recall) - baseline_recall) if recall != "" else "",
            "different_gt_false_delta_vs_strict_gap1": (
                float(diff_false_among_accepted) - baseline_false if diff_false_among_accepted != "" else ""
            ),
            "phase5_formal_bridge_gate_pass": formal_bridge_gate_pass,
            "phase6_ap_repair_allowed": phase6_allowed,
            "phase6_blocker": ""
            if phase6_allowed
            else "Phase5 variant gate failed or Phase1 repair_candidate_pair_count remains <30.",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        variant_rows.append(row)
        if best_row is None or (
            row["same_object_bridge_recall"] != ""
            and (
                best_row["same_object_bridge_recall"] == ""
                or float(row["same_object_bridge_recall"]) > float(best_row["same_object_bridge_recall"])
            )
        ):
            best_row = row

    assert best_row is not None
    summary_bits = {
        "bridge_auc": auc,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_negative_pair_count": negative_total,
        "same_semantic_different_gt_hard_negative_count": same_semantic_negative_total,
        "best_variant_id_by_recall": best_row["variant_id"],
        "best_variant_same_object_bridge_recall": best_row["same_object_bridge_recall"],
        "best_variant_different_gt_false_bridge_among_accepted": best_row[
            "different_gt_false_bridge_among_accepted"
        ],
        "best_variant_same_semantic_different_gt_false_bridge_among_accepted": best_row[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "any_phase5_formal_bridge_gate_pass": any(bool(r["phase5_formal_bridge_gate_pass"]) for r in variant_rows),
        "any_phase6_ap_repair_allowed": any(bool(r["phase6_ap_repair_allowed"]) for r in variant_rows),
    }
    return variant_rows, summary_bits


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(PHASE1_DIR / "summary.json")
    xyz = _load_xyz()
    with np.load(MINI_NPZ) as data:
        mini = {key: np.asarray(data[key]) for key in data.files}
    camera_count = int(min(len(mini["extrinsics"]), len(mini["intrinsics"]), len(mini["depth"])))
    frame_df = _frame_manifest(camera_count)
    frame_ids = frame_df["frame_id"].astype(int).tolist()
    meta = _mask_meta(frame_ids)
    mask_by_frame, source_rows = _project_masks(xyz, mini, frame_df)
    mask_summary_rows = _mask_summary_rows(mask_by_frame, frame_df, meta)
    bridge_df = _build_bridge_rows(mask_by_frame, frame_df, meta)
    variant_rows, summary_bits = _variant_metrics(bridge_df, phase1)

    bridge_path = OUT_DIR / "mask_pair_primitive_bridge_rows.parquet"
    variant_path = OUT_DIR / "repair_variant_rows.csv"
    mask_summary_path = OUT_DIR / "chunk32_mask_primitive_summary_rows.csv"
    source_path = OUT_DIR / "mask_projection_source_rows.csv"
    provider_path = OUT_DIR / "provider_bridge_summary_rows.csv"
    hard_negative_path = OUT_DIR / "hard_negative_rows.csv"
    pseudo_positive_path = OUT_DIR / "pseudo_positive_rows.csv"
    gate_path = OUT_DIR / "variant_gate_rows.csv"

    bridge_df.to_parquet(bridge_path, index=False)
    _write_csv(variant_path, variant_rows)
    _write_csv(mask_summary_path, mask_summary_rows)
    _write_csv(source_path, source_rows)
    bridge_df[bridge_df["diagnostic_different_gt"]].to_csv(hard_negative_path, index=False)
    bridge_df[bridge_df["diagnostic_same_gt"]].to_csv(pseudo_positive_path, index=False)

    best_variant = max(
        variant_rows,
        key=lambda r: float(r["same_object_bridge_recall"]) if r["same_object_bridge_recall"] != "" else -1.0,
    )
    provider_rows = [
        {
            "schema_version": "stream4d_v102_phase5b_provider_bridge_summary_row_v1",
            "phase_id": "v102_phase5b_chunk32_short_range_bridge_repair",
            "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
            "chunk_id": "scene0050_00_chunk32_process252",
            "gaussian_count": int(len(xyz)),
            "camera_pose_count": int(mask_by_frame.shape[0]),
            "mask_observation_count": int(len(mask_summary_rows)),
            "candidate_pair_count": int(len(bridge_df)),
            "semantic_mapping": "ScanNet aggregation objectId = diagnostic_gt_instance - 1",
            "bridge_auc": summary_bits["bridge_auc"],
            "best_variant_id_by_recall": best_variant["variant_id"],
            "best_variant_same_object_bridge_recall": best_variant["same_object_bridge_recall"],
            "best_variant_different_gt_false_bridge_among_accepted": best_variant[
                "different_gt_false_bridge_among_accepted"
            ],
            "best_variant_same_semantic_different_gt_false_bridge_among_accepted": best_variant[
                "same_semantic_different_gt_false_bridge_among_accepted"
            ],
            "phase1_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
            "phase1_broad_contamination_rate": phase1.get("broad_contamination_rate"),
            "phase5_formal_bridge_gate_pass": summary_bits["any_phase5_formal_bridge_gate_pass"],
            "phase6_ap_repair_allowed": summary_bits["any_phase6_ap_repair_allowed"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    ]
    _write_csv(provider_path, provider_rows)

    gate_rows = [
        {
            "gate_id": "same_object_bridge_recall_best_variant",
            "pass": best_variant["same_object_bridge_recall"] != ""
            and float(best_variant["same_object_bridge_recall"]) >= 0.35,
            "expected": ">=0.35",
            "observed": best_variant["same_object_bridge_recall"],
            "variant_id": best_variant["variant_id"],
        },
        {
            "gate_id": "different_gt_false_bridge_best_variant",
            "pass": best_variant["different_gt_false_bridge_among_accepted"] != ""
            and float(best_variant["different_gt_false_bridge_among_accepted"]) <= 0.20,
            "expected": "<=0.20",
            "observed": best_variant["different_gt_false_bridge_among_accepted"],
            "variant_id": best_variant["variant_id"],
        },
        {
            "gate_id": "same_semantic_different_gt_false_bridge_best_variant",
            "pass": best_variant["same_semantic_different_gt_false_bridge_among_accepted"] != ""
            and float(best_variant["same_semantic_different_gt_false_bridge_among_accepted"]) <= 0.20,
            "expected": "<=0.20",
            "observed": best_variant["same_semantic_different_gt_false_bridge_among_accepted"],
            "variant_id": best_variant["variant_id"],
        },
        {
            "gate_id": "bridge_auc",
            "pass": summary_bits["bridge_auc"] != "" and float(summary_bits["bridge_auc"]) >= 0.65,
            "expected": ">=0.65",
            "observed": summary_bits["bridge_auc"],
            "variant_id": "all_scores",
        },
        {
            "gate_id": "phase1_repair_candidate_pair_count",
            "pass": int(phase1.get("repair_candidate_pair_count", 0)) >= 30,
            "expected": ">=30 before Phase6 AP repair",
            "observed": phase1.get("repair_candidate_pair_count"),
            "variant_id": "phase1",
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
            "variant_id": "all",
        },
    ]
    _write_csv(gate_path, gate_rows)

    if summary_bits["any_phase6_ap_repair_allowed"]:
        decision = "PASS_CHUNK32_BRIDGE_REPAIR_ENTER_PHASE6"
    elif summary_bits["any_phase5_formal_bridge_gate_pass"]:
        decision = "PARTIAL_CHUNK32_BRIDGE_GATE_PASS__PHASE6_BLOCKED_BY_PHASE1"
    else:
        decision = "NO_GO_CHUNK32_BRIDGE_REPAIR_GATE_STILL_FAILS"

    summary = {
        "schema_version": "stream4d_v102_phase5b_chunk32_short_range_bridge_repair_summary_v1",
        "phase_id": "v102_phase5b_chunk32_short_range_bridge_repair",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "gaussian_count": int(len(xyz)),
        "camera_pose_count": int(mask_by_frame.shape[0]),
        "frame_ids": frame_ids,
        "mask_observation_count": int(len(mask_summary_rows)),
        "candidate_pair_count": int(len(bridge_df)),
        "variant_count": len(variant_rows),
        "semantic_mapping": "ScanNet aggregation objectId = diagnostic_gt_instance - 1",
        "diagnostic_positive_pair_count": summary_bits["diagnostic_positive_pair_count"],
        "diagnostic_negative_pair_count": summary_bits["diagnostic_negative_pair_count"],
        "same_semantic_different_gt_hard_negative_count": summary_bits[
            "same_semantic_different_gt_hard_negative_count"
        ],
        "bridge_auc": summary_bits["bridge_auc"],
        "best_variant_id_by_recall": best_variant["variant_id"],
        "best_variant_same_object_bridge_recall": best_variant["same_object_bridge_recall"],
        "best_variant_different_gt_false_bridge_among_accepted": best_variant[
            "different_gt_false_bridge_among_accepted"
        ],
        "best_variant_same_semantic_different_gt_false_bridge_among_accepted": best_variant[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "any_phase5_formal_bridge_gate_pass": summary_bits["any_phase5_formal_bridge_gate_pass"],
        "phase1_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
        "phase1_broad_contamination_rate": phase1.get("broad_contamination_rate"),
        "phase6_ap_repair_allowed": summary_bits["any_phase6_ap_repair_allowed"],
        "truthfulness_note": (
            "Chunk32 bridge variants use DA3-GIANT-1.1 Gaussian projection and CropFormer masks. "
            "GT instance/semantic labels are used only for diagnostic labels and gates, not prediction."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "mask_pair_primitive_bridge_rows": _rel(bridge_path),
            "repair_variant_rows": _rel(variant_path),
            "chunk32_mask_primitive_summary_rows": _rel(mask_summary_path),
            "mask_projection_source_rows": _rel(source_path),
            "provider_bridge_summary_rows": _rel(provider_path),
            "hard_negative_rows": _rel(hard_negative_path),
            "pseudo_positive_rows": _rel(pseudo_positive_path),
            "variant_gate_rows": _rel(gate_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
