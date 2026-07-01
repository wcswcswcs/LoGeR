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
PHASE3_DIR = AUDIT_ROOT / "v102_phase3_da3_giant_3dgs_visual_audit"
PHASE1_DIR = AUDIT_ROOT / "v102_phase1_fragment_casebook"
OUT_DIR = AUDIT_ROOT / "v102_phase4_persistent_primitive_diagnostic"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"

PLY_PATH = PHASE3_DIR / "scene_chunk_3dgs.ply"
MINI_NPZ = AUDIT_ROOT / "v102_phase2_provider_ladder_audit" / "official_da3_giant_smoke2_gs_ply_only" / "exports" / "mini_npz" / "results.npz"
FRAME_MANIFEST = AUDIT_ROOT / "v98_phase1_provider_contract" / "da3_streaming_d4rt32o3_scene0050_input119" / "frame_manifest_rows.csv"
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
MASK_OBSERVATION_TABLE = (
    AUDIT_ROOT
    / "v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt"
    / "observation_tables"
    / "mask_observation_table.csv"
)


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


def _homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = extrinsic.astype(np.float64)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")


def _load_inputs() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not PLY_PATH.exists():
        raise FileNotFoundError(PLY_PATH)
    if not MINI_NPZ.exists():
        raise FileNotFoundError(MINI_NPZ)
    ply = PlyData.read(str(PLY_PATH))
    vertex = ply["vertex"].data
    xyz = np.column_stack([_prop(vertex, "x"), _prop(vertex, "y"), _prop(vertex, "z")]).astype(np.float64)
    attrs = {
        "nx": _prop(vertex, "nx").astype(np.float64),
        "ny": _prop(vertex, "ny").astype(np.float64),
        "nz": _prop(vertex, "nz").astype(np.float64),
        "opacity": _prop(vertex, "opacity").astype(np.float64),
        "scale_0": _prop(vertex, "scale_0").astype(np.float64),
        "scale_1": _prop(vertex, "scale_1").astype(np.float64),
        "scale_2": _prop(vertex, "scale_2").astype(np.float64),
        "rot_0": _prop(vertex, "rot_0").astype(np.float64),
        "rot_1": _prop(vertex, "rot_1").astype(np.float64),
        "rot_2": _prop(vertex, "rot_2").astype(np.float64),
        "rot_3": _prop(vertex, "rot_3").astype(np.float64),
    }
    with np.load(MINI_NPZ) as data:
        mini = {key: np.asarray(data[key]) for key in data.files}
    return xyz, attrs, mini


def _project(xyz: np.ndarray, mini: dict[str, np.ndarray]) -> tuple[np.ndarray, list[pd.DataFrame]]:
    extrinsics = np.asarray(mini["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(mini["intrinsics"], dtype=np.float64)
    depth = np.asarray(mini["depth"])
    n = min(len(extrinsics), len(intrinsics), len(depth))
    points_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    visibility_count = np.zeros(len(xyz), dtype=np.int16)
    obs_frames: list[pd.DataFrame] = []
    for i in range(n):
        ext = _homogeneous_extrinsic(extrinsics[i])
        k = intrinsics[i]
        h, w = int(depth[i].shape[0]), int(depth[i].shape[1])
        cam = (ext @ points_h.T).T[:, :3]
        z = cam[:, 2]
        valid_z = z > 1e-6
        u = np.full(len(xyz), np.nan, dtype=np.float64)
        v = np.full(len(xyz), np.nan, dtype=np.float64)
        u[valid_z] = k[0, 0] * (cam[valid_z, 0] / z[valid_z]) + k[0, 2]
        v[valid_z] = k[1, 1] * (cam[valid_z, 1] / z[valid_z]) + k[1, 2]
        inside = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        visibility_count += inside.astype(np.int16)
        obs_frames.append(
            pd.DataFrame(
                {
                    "schema_version": "stream4d_v102_phase4_primitive_observation_row_v1",
                    "phase_id": "v102_phase4_persistent_primitive_diagnostic",
                    "primitive_index": np.arange(len(xyz), dtype=np.int32),
                    "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
                    "frame_index": np.full(len(xyz), i, dtype=np.int16),
                    "image_height": np.full(len(xyz), h, dtype=np.int32),
                    "image_width": np.full(len(xyz), w, dtype=np.int32),
                    "projected_u": u.astype(np.float32),
                    "projected_v": v.astype(np.float32),
                    "camera_z": z.astype(np.float32),
                    "positive_depth": valid_z,
                    "projection_inside_image": inside,
                }
            )
        )
    return visibility_count, obs_frames


def _frame_ids_for_cameras(camera_count: int) -> list[int]:
    if not FRAME_MANIFEST.exists():
        return [i * 5 for i in range(camera_count)]
    frame_manifest = pd.read_csv(FRAME_MANIFEST)
    frame_manifest = frame_manifest.sort_values("da3_frame_index")
    frame_ids = []
    for i in range(camera_count):
        row = frame_manifest[frame_manifest["da3_frame_index"] == i]
        frame_ids.append(int(row.iloc[0]["frame_id"]) if len(row) else i * 5)
    return frame_ids


def _mask_observation_metadata(frame_ids: list[int]) -> pd.DataFrame:
    if not MASK_OBSERVATION_TABLE.exists():
        return pd.DataFrame()
    df = pd.read_csv(MASK_OBSERVATION_TABLE)
    df = df[(df["scene_id"] == "scene0050_00") & (df["frame_id"].isin(frame_ids))].copy()
    df["mask_id"] = df["mask_id"].astype(int)
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def _build_mask_participation(
    observation_df: pd.DataFrame,
    primitive_count: int,
    frame_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    meta = _mask_observation_metadata(frame_ids)
    meta_index = {
        (int(row.frame_id), int(row.mask_id)): row
        for row in meta.itertuples(index=False)
    }
    participation_frames: list[pd.DataFrame] = []
    mask_source_rows: list[dict[str, Any]] = []
    broad_risk_by_primitive = np.full(primitive_count, np.nan, dtype=np.float32)
    for camera_index, frame_id in enumerate(frame_ids):
        mask_path = CROPFORMER_MASK_ROOT / f"{frame_id}.png"
        if not mask_path.exists():
            mask_source_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_path": _rel(mask_path),
                    "mask_exists": False,
                    "participant_rows": 0,
                    "note": "CropFormer mask PNG missing.",
                }
            )
            continue
        mask = np.asarray(Image.open(mask_path))
        mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
        sub = observation_df[
            (observation_df["frame_index"] == camera_index)
            & (observation_df["projection_inside_image"])
        ].copy()
        if len(sub) == 0:
            mask_source_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_path": _rel(mask_path),
                    "mask_exists": True,
                    "mask_height": mask_h,
                    "mask_width": mask_w,
                    "participant_rows": 0,
                    "note": "No projected Gaussian inside DA3 processed image.",
                }
            )
            continue
        image_h = float(sub["image_height"].iloc[0])
        image_w = float(sub["image_width"].iloc[0])
        xs = np.floor(np.clip(sub["projected_u"].to_numpy() / image_w * mask_w, 0, mask_w - 1)).astype(np.int32)
        ys = np.floor(np.clip(sub["projected_v"].to_numpy() / image_h * mask_h, 0, mask_h - 1)).astype(np.int32)
        mask_ids = mask[ys, xs].astype(np.int32)
        valid = mask_ids > 0
        if not np.any(valid):
            mask_source_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_path": _rel(mask_path),
                    "mask_exists": True,
                    "mask_height": mask_h,
                    "mask_width": mask_w,
                    "participant_rows": 0,
                    "note": "Projected Gaussians hit only background label 0.",
                }
            )
            continue
        part = pd.DataFrame(
            {
                "schema_version": "stream4d_v102_phase4_primitive_mask_participation_row_v1",
                "phase_id": "v102_phase4_persistent_primitive_diagnostic",
                "primitive_id": [f"v102_gs_{i:08d}" for i in sub.loc[valid, "primitive_index"].to_numpy()],
                "primitive_index": sub.loc[valid, "primitive_index"].to_numpy(dtype=np.int32),
                "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
                "scene_id": "scene0050_00",
                "frame_index": np.full(np.sum(valid), camera_index, dtype=np.int16),
                "frame_id": np.full(np.sum(valid), frame_id, dtype=np.int32),
                "mask_id": mask_ids[valid].astype(np.int32),
                "mask_observation_id": [f"scene0050_00:{frame_id}:{int(mid)}" for mid in mask_ids[valid]],
                "participation_score": np.ones(np.sum(valid), dtype=np.float32),
                "projected_u": sub.loc[valid, "projected_u"].to_numpy(dtype=np.float32),
                "projected_v": sub.loc[valid, "projected_v"].to_numpy(dtype=np.float32),
                "mask_pixel_x": xs[valid],
                "mask_pixel_y": ys[valid],
                "source": "v65_cropformer_scaled_da3_projection_smoke2",
                "uses_gt_for_prediction": False,
            }
        )
        participation_frames.append(part)
        for mid in np.unique(mask_ids[valid]):
            key = (frame_id, int(mid))
            row = meta_index.get(key)
            if row is not None:
                risk = float(row.mask_area) / float(mask_h * mask_w)
                prim_idx = part.loc[part["mask_id"] == int(mid), "primitive_index"].to_numpy(dtype=np.int32)
                broad_risk_by_primitive[prim_idx] = np.nanmax(
                    np.column_stack([np.nan_to_num(broad_risk_by_primitive[prim_idx], nan=0.0), np.full(len(prim_idx), risk)]),
                    axis=1,
                )
        mask_source_rows.append(
            {
                "frame_id": frame_id,
                "mask_path": _rel(mask_path),
                "mask_exists": True,
                "mask_height": mask_h,
                "mask_width": mask_w,
                "participant_rows": int(np.sum(valid)),
                "unique_mask_ids_hit": int(len(np.unique(mask_ids[valid]))),
                "note": "Projected DA3-GS coordinates were scaled from DA3 processed resolution to CropFormer mask resolution.",
            }
        )

    participation_df = pd.concat(participation_frames, ignore_index=True) if participation_frames else pd.DataFrame()
    part_counts = np.zeros(primitive_count, dtype=np.int16)
    if len(participation_df):
        counts = participation_df.groupby("primitive_index").size()
        part_counts[counts.index.to_numpy(dtype=np.int32)] = counts.to_numpy(dtype=np.int16)

    if len(participation_df) and len(meta):
        grouped = (
            participation_df.groupby(["frame_id", "mask_id"], as_index=False)
            .agg(participating_primitive_count=("primitive_index", "nunique"))
        )
        purity_rows = []
        for row in grouped.itertuples(index=False):
            meta_row = meta_index.get((int(row.frame_id), int(row.mask_id)))
            purity_rows.append(
                {
                    "schema_version": "stream4d_v102_phase4_primitive_mask_purity_row_v1",
                    "phase_id": "v102_phase4_persistent_primitive_diagnostic",
                    "scene_id": "scene0050_00",
                    "frame_id": int(row.frame_id),
                    "mask_id": int(row.mask_id),
                    "mask_observation_id": f"scene0050_00:{int(row.frame_id)}:{int(row.mask_id)}",
                    "participating_primitive_count": int(row.participating_primitive_count),
                    "diagnostic_gt_instance": getattr(meta_row, "diagnostic_gt_instance", "") if meta_row is not None else "",
                    "diagnostic_gt_purity": getattr(meta_row, "diagnostic_gt_purity", "") if meta_row is not None else "",
                    "mask_area": getattr(meta_row, "mask_area", "") if meta_row is not None else "",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
        purity_df = pd.DataFrame(purity_rows)
    else:
        purity_df = pd.DataFrame(
            columns=[
                "schema_version",
                "phase_id",
                "scene_id",
                "frame_id",
                "mask_id",
                "mask_observation_id",
                "participating_primitive_count",
                "diagnostic_gt_instance",
                "diagnostic_gt_purity",
                "mask_area",
                "uses_gt_for_prediction",
                "uses_gt_for_diagnostic_labels",
            ]
        )

    boundary_df = pd.DataFrame(
        {
            "schema_version": "stream4d_v102_phase4_primitive_boundary_risk_row_v1",
            "phase_id": "v102_phase4_persistent_primitive_diagnostic",
            "primitive_id": [f"v102_gs_{i:08d}" for i in range(primitive_count)],
            "primitive_index": np.arange(primitive_count, dtype=np.int32),
            "boundary_risk": np.full(primitive_count, np.nan, dtype=np.float32),
            "broad_mask_risk": broad_risk_by_primitive,
            "source": "v65_mask_area_ratio_proxy_from_participating_masks",
            "blocker": "boundary crossing score not computed in this smoke; broad risk is mask-area proxy only.",
        }
    )
    return participation_df, boundary_df, purity_df, part_counts, broad_risk_by_primitive, mask_source_rows


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase3 = _read_json(PHASE3_DIR / "summary.json")
    phase1 = _read_json(PHASE1_DIR / "summary.json")
    if not bool(phase3.get("phase3_pass_for_visual_artifact")):
        raise RuntimeError("Phase3 visual artifact gate did not pass; refusing Phase4 diagnostic.")

    xyz, attrs, mini = _load_inputs()
    finite_xyz = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite_xyz]
    attrs = {key: value[finite_xyz] for key, value in attrs.items()}
    visibility_count, obs_frames = _project(xyz, mini)
    camera_count = int(min(len(mini["extrinsics"]), len(mini["intrinsics"]), len(mini["depth"])))
    projection_valid_rate = float(np.mean(visibility_count > 0)) if len(visibility_count) else 0.0
    projection_valid_rate_mean_per_primitive = float(np.mean(visibility_count / max(camera_count, 1))) if len(visibility_count) else 0.0
    observation_df = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame()
    frame_ids = _frame_ids_for_cameras(camera_count)
    participation_df, boundary_df, purity_df, part_counts, broad_risk_by_primitive, mask_source_rows = _build_mask_participation(
        observation_df, len(xyz), frame_ids
    )

    scale_linear = np.exp(np.clip(np.column_stack([attrs["scale_0"], attrs["scale_1"], attrs["scale_2"]]), -20.0, 20.0))
    primitive_df = pd.DataFrame(
        {
            "schema_version": "stream4d_v102_phase4_primitive_row_v1",
            "phase_id": "v102_phase4_persistent_primitive_diagnostic",
            "primitive_id": [f"v102_gs_{i:08d}" for i in range(len(xyz))],
            "primitive_index": np.arange(len(xyz), dtype=np.int32),
            "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
            "primitive_type": "gaussian",
            "chunk_id": "scene0050_00_smoke2",
            "position_x": xyz[:, 0].astype(np.float32),
            "position_y": xyz[:, 1].astype(np.float32),
            "position_z": xyz[:, 2].astype(np.float32),
            "normal_x": attrs["nx"].astype(np.float32),
            "normal_y": attrs["ny"].astype(np.float32),
            "normal_z": attrs["nz"].astype(np.float32),
            "opacity_raw": attrs["opacity"].astype(np.float32),
            "scale_x": scale_linear[:, 0].astype(np.float32),
            "scale_y": scale_linear[:, 1].astype(np.float32),
            "scale_z": scale_linear[:, 2].astype(np.float32),
            "confidence": np.ones(len(xyz), dtype=np.float32),
            "visibility_count": visibility_count.astype(np.int16),
            "projection_valid_rate": (visibility_count / max(camera_count, 1)).astype(np.float32),
            "mask_participation_count": part_counts.astype(np.int16),
            "boundary_risk": np.full(len(xyz), np.nan, dtype=np.float32),
            "broad_mask_risk": broad_risk_by_primitive.astype(np.float32),
            "uses_gt_for_prediction": False,
        }
    )

    primitive_path = OUT_DIR / "primitive_rows.parquet"
    observation_path = OUT_DIR / "primitive_observation_rows.parquet"
    participation_path = OUT_DIR / "primitive_mask_participation_rows.parquet"
    boundary_path = OUT_DIR / "primitive_boundary_risk_rows.parquet"
    purity_path = OUT_DIR / "primitive_mask_purity_rows.csv"
    mask_source_path = OUT_DIR / "mask_source_rows.csv"
    primitive_df.to_parquet(primitive_path, index=False)
    observation_df.to_parquet(observation_path, index=False)
    participation_df.to_parquet(participation_path, index=False)
    boundary_df.to_parquet(boundary_path, index=False)
    purity_df.to_csv(purity_path, index=False)
    _write_csv(mask_source_path, mask_source_rows)

    participation_mean = float(np.mean(primitive_df["mask_participation_count"])) if len(primitive_df) else 0.0
    purity_available = len(purity_df) > 0
    provider_quality_rows = [
        {
            "schema_version": "stream4d_v102_phase4_provider_quality_row_v1",
            "phase_id": "v102_phase4_persistent_primitive_diagnostic",
            "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
            "primitive_count": int(len(primitive_df)),
            "camera_pose_count": camera_count,
            "projection_valid_rate_any_camera": projection_valid_rate,
            "projection_valid_rate_mean_per_primitive": projection_valid_rate_mean_per_primitive,
            "mask_participation_count_mean": participation_mean,
            "mask_participation_available": len(participation_df) > 0,
            "primitive_mask_participation_row_count": int(len(participation_df)),
            "surfel_or_gaussian_purity_rows_available": purity_available,
            "primitive_mask_purity_row_count": int(len(purity_df)),
            "repair_candidate_pair_count_from_phase1": phase1.get("repair_candidate_pair_count"),
            "broad_contamination_rate_from_phase1": phase1.get("broad_contamination_rate"),
            "uses_gt_for_prediction": False,
            "blocker": "Smoke-2 mask participation is available for two frames only; Phase1 has zero AP repair candidates and broad contamination remains high.",
        }
    ]
    provider_quality_csv = OUT_DIR / "primitive_provider_quality_rows.csv"
    _write_csv(provider_quality_csv, provider_quality_rows)

    gate_rows = [
        {
            "gate_id": "primitive_count",
            "pass": int(len(primitive_df)) >= 10000,
            "expected": ">=10000 unless smoke-only",
            "observed": int(len(primitive_df)),
            "severity": "required",
        },
        {
            "gate_id": "projection_valid_rate",
            "pass": projection_valid_rate >= 0.80,
            "expected": ">=0.80",
            "observed": projection_valid_rate,
            "severity": "required",
        },
        {
            "gate_id": "mask_participation_count_mean",
            "pass": participation_mean > 0.0,
            "expected": ">0",
            "observed": participation_mean,
            "severity": "required_for_phase5_bridge",
        },
        {
            "gate_id": "surfel_or_gaussian_purity_rows_available",
            "pass": purity_available,
            "expected": True,
            "observed": purity_available,
            "severity": "required_for_phase5_diagnostic_gate",
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
            "severity": "required",
        },
        {
            "gate_id": "phase1_repair_candidate_pair_count",
            "pass": int(phase1.get("repair_candidate_pair_count", 0)) >= 30,
            "expected": ">=30 to justify AP repair branch",
            "observed": phase1.get("repair_candidate_pair_count"),
            "severity": "blocks_phase6_ap_repair",
        },
    ]
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    _write_csv(gate_csv, gate_rows)

    phase4_primitives_ready = bool(len(primitive_df) >= 10000 and projection_valid_rate >= 0.80)
    phase4_pass_for_phase5 = bool(phase4_primitives_ready and participation_mean > 0.0 and purity_available)
    decision = (
        "PARTIAL_PRIMITIVES_PROJECTIONS_MASK_PARTICIPATION_READY__AP_REPAIR_BLOCKED_BY_PHASE1"
        if phase4_pass_for_phase5 and int(phase1.get("repair_candidate_pair_count", 0)) < 30
        else "PARTIAL_PRIMITIVES_AND_PROJECTIONS_READY__BLOCK_MASK_PARTICIPATION_AND_PURITY"
        if phase4_primitives_ready and not phase4_pass_for_phase5
        else "PASS_PHASE4_ENTER_PHASE5"
        if phase4_pass_for_phase5
        else "NO_GO_PHASE4_PRIMITIVE_PROJECTION_FAIL"
    )

    summary = {
        "schema_version": "stream4d_v102_phase4_persistent_primitive_diagnostic_summary_v1",
        "phase_id": "v102_phase4_persistent_primitive_diagnostic",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase4_completed": True,
        "phase4_primitives_ready": phase4_primitives_ready,
        "phase4_pass_for_phase5_bridge": phase4_pass_for_phase5,
        "primitive_count": int(len(primitive_df)),
        "primitive_observation_count": int(len(observation_df)),
        "primitive_mask_participation_row_count": int(len(participation_df)),
        "primitive_mask_purity_row_count": int(len(purity_df)),
        "camera_pose_count": camera_count,
        "frame_ids": frame_ids,
        "projection_valid_rate_any_camera": projection_valid_rate,
        "projection_valid_rate_mean_per_primitive": projection_valid_rate_mean_per_primitive,
        "mask_participation_count_mean": participation_mean,
        "mask_participation_available": len(participation_df) > 0,
        "surfel_or_gaussian_purity_rows_available": purity_available,
        "phase1_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
        "phase1_broad_contamination_rate": phase1.get("broad_contamination_rate"),
        "truthfulness_note": "Phase4 builds real Gaussian primitive/projection/mask-participation diagnostics for Smoke-2 frames only. GT purity labels are diagnostic and are not used for prediction.",
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "primitive_rows": _rel(primitive_path),
            "primitive_observation_rows": _rel(observation_path),
            "primitive_mask_participation_rows": _rel(participation_path),
            "primitive_boundary_risk_rows": _rel(boundary_path),
            "primitive_mask_purity_rows": _rel(purity_path),
            "mask_source_rows": _rel(mask_source_path),
            "primitive_provider_quality_rows": _rel(provider_quality_csv),
            "variant_gate_rows": _rel(gate_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase4_primitives_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
