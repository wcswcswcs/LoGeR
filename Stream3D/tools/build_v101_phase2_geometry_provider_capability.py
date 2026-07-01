from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v101_phase2_geometry_provider_capability"

PHASE1_DIR = AUDIT_ROOT / "v101_phase1_f2_fragmentation_casebook"
PHASE1B_DIR = AUDIT_ROOT / "v101_phase1b_fragment_quality_decomp"
PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
V98_PROVIDER = AUDIT_ROOT / "v98_phase1_provider_contract"
V99_D4RT_PROVIDER = AUDIT_ROOT / "v99_phase10ae_d4rt_da3grid_provider_audit"
V99_D4RT_PREFIX = AUDIT_ROOT / "v99_phase10ag_prefix_da3_d4rt_sim3_alignment"
V99_D4RT_ANCHOR = AUDIT_ROOT / "v99_phase10ah_prefix_sim3_aligned_anchor_scene_stitch"
V99_DA3_BASE = AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_provider_audit"
V100_D4RT_VERIFIER = AUDIT_ROOT / "v100_phase5_da3_d4rt_verifier_audit"

TAUS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
OBJECT_LIKE_MIN_AREA_RATIO = 0.005
BROAD_MASK_AREA_RATIO = 0.20


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if str(p).startswith("Stream3D/"):
        return ROOT / p
    return STREAM3D / p


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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n", "", "none", "nan"}:
        return False
    try:
        return float(text) > 0.0
    except Exception:
        return False


def _read_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label png: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    arr = np.asarray(image, dtype=np.int64)
    if shape_hw is not None and tuple(arr.shape[:2]) != tuple(shape_hw):
        h, w = shape_hw
        arr = cv2.resize(arr, (int(w), int(h)), interpolation=cv2.INTER_NEAREST).astype(np.int64, copy=False)
    return arr


def _chunk_index(chunk_id: str) -> int:
    m = re.search(r"(\d+)$", str(chunk_id))
    return int(m.group(1)) if m else 0


def _provider_ladder() -> list[dict[str, Any]]:
    return [
        {
            "provider_name": "G0_D4RT_reliable_anchors_only",
            "model_variant": "OpenD4RT_32CLIP_9Dataset_NoAUG_DA3grid_prefix_sim3",
            "implementation": "official_pytorch_opend4rt_with_local_adapter",
            "quantization": "none",
        },
        {"provider_name": "G1_DA3_STREAMING_SMALL", "model_variant": "DA3-SMALL", "implementation": "da3_streaming", "quantization": "none"},
        {"provider_name": "G2_DA3_STREAMING_BASE", "model_variant": "DA3-BASE", "implementation": "da3_streaming", "quantization": "none"},
        {"provider_name": "G3_DA3_STREAMING_LARGE", "model_variant": "DA3-LARGE", "implementation": "da3_streaming", "quantization": "none"},
        {"provider_name": "G4_DA3METRIC_LARGE", "model_variant": "DA3-METRIC-LARGE", "implementation": "official_pytorch", "quantization": "none"},
        {"provider_name": "G5_DA3NESTED_GIANT_LARGE", "model_variant": "DA3-NESTED-GIANT-LARGE", "implementation": "da3_streaming", "quantization": "none"},
        {"provider_name": "G6_DA3_GIANT_3DGS_DIAGNOSTIC", "model_variant": "DA3-GIANT", "implementation": "da3_streaming", "quantization": "none"},
        {"provider_name": "G7_depth_anything_cpp_BASE_q8_0", "model_variant": "DA3-SMALL-gguf-q8_0", "implementation": "depth_anything_cpp", "quantization": "q8_0"},
        {"provider_name": "G8_depth_anything_cpp_LARGE_q8_0_or_f32", "model_variant": "DA3-LARGE-gguf", "implementation": "depth_anything_cpp", "quantization": "q8_0_or_f32"},
        {"provider_name": "G9_depth_anything_cpp_METRIC_LARGE", "model_variant": "DA3-METRIC-LARGE-gguf", "implementation": "depth_anything_cpp", "quantization": "q8_0_or_f32"},
        {"provider_name": "G10_depth_anything_cpp_NESTED_anyview_metric", "model_variant": "DA3-NESTED-anyview-metric-gguf", "implementation": "depth_anything_cpp", "quantization": "q8_0_or_f32"},
    ]


def _contract_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    stitch_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    v98_contract = {r["provider_id"]: r for r in _read_csv(V98_PROVIDER / "provider_contract_rows.csv")}
    v98_metric = {r["provider_id"]: r for r in _read_csv(V98_PROVIDER / "provider_metric_rows.csv")}
    d4rt_ae_rows = _read_csv(V99_D4RT_PROVIDER / "provider_rows.csv")
    d4rt_ae = _read_json(V99_D4RT_PROVIDER / "summary.json")
    d4rt_ag = _read_json(V99_D4RT_PREFIX / "summary.json")
    d4rt_ah = _read_json(V99_D4RT_ANCHOR / "summary.json")
    da3_base = _read_json(V99_DA3_BASE / "summary.json")
    v100_d4rt = _read_json(V100_D4RT_VERIFIER / "summary.json")

    partial_roots = {
        "G1_DA3_STREAMING_SMALL": V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_small_input119",
        "G3_DA3_STREAMING_LARGE": V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_large_input119",
        "G5_DA3NESTED_GIANT_LARGE": V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_nested_giant_large_input119",
        "G6_DA3_GIANT_3DGS_DIAGNOSTIC": V98_PROVIDER / "da3_streaming_d4rt32o3_scene0050_giant_input119",
    }

    for provider in _provider_ladder():
        name = provider["provider_name"]
        row = {
            "schema_version": "stream4d_v101_phase2_provider_contract_row_v1",
            "phase_id": "v101_phase2_geometry_provider_capability",
            "provider_name": name,
            "model_variant": provider["model_variant"],
            "implementation": provider["implementation"],
            "quantization": provider["quantization"],
            "input_resolution": "",
            "method_chunk_size": 32,
            "frame_stride": 5,
            "overlap": 3,
            "supports_chunk32": False,
            "subchunk_size_if_any": "",
            "outputs_depth": False,
            "outputs_confidence": False,
            "outputs_pose_or_ray": False,
            "outputs_intrinsics": False,
            "outputs_extrinsics": False,
            "outputs_point_cloud": False,
            "outputs_3d_gaussians": False,
            "provider_scale_type": "unknown",
            "runtime_sec_per_chunk32": "",
            "runtime_sec_per_frame": "",
            "peak_gpu_memory_MB": "",
            "peak_cpu_memory_MB": "",
            "OOM_count": 0,
            "failed_chunk_count": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "contract_status": "provider_contract_fail",
            "contract_failure_reason": "",
            "source_artifact": "",
        }

        if name == "G0_D4RT_reliable_anchors_only":
            total_runtime = sum(_num(r.get("decode_runtime_total_sec")) + _num(r.get("stitch_runtime_sec")) for r in d4rt_ae_rows)
            total_chunks = sum(_num(r.get("decoded_group_count")) for r in d4rt_ae_rows)
            total_tracks = sum(_num(r.get("stitched_track_row_count")) for r in d4rt_ae_rows)
            row.update(
                {
                    "input_resolution": "504x378_external_DA3_grid_internal_checkpoint_256x256",
                    "supports_chunk32": bool(d4rt_ae.get("provider_gate_pass") and d4rt_ag.get("prefix_causal_alignment_gate_pass")),
                    "outputs_depth": False,
                    "outputs_confidence": True,
                    "outputs_pose_or_ray": False,
                    "outputs_point_cloud": True,
                    "outputs_intrinsics": False,
                    "outputs_extrinsics": False,
                    "provider_scale_type": "relative_prefix_sim3_aligned",
                    "runtime_sec_per_chunk32": float(total_runtime / max(1.0, total_chunks)),
                    "runtime_sec_per_frame": "",
                    "failed_chunk_count": int(sum(_num(r.get("decode_error_count")) for r in d4rt_ae_rows)),
                    "contract_status": "provider_contract_pass",
                    "contract_failure_reason": "AP method gate still failed in v99/v100; provider contract pass is not method success.",
                    "source_artifact": _rel(V99_D4RT_PROVIDER / "summary.json"),
                }
            )
            runtime_rows.append(
                {
                    "schema_version": "stream4d_v101_phase2_provider_runtime_row_v1",
                    "phase_id": "v101_phase2_geometry_provider_capability",
                    "provider_name": name,
                    "scene_count": len(d4rt_ae_rows),
                    "runtime_total_sec": total_runtime,
                    "runtime_sec_per_chunk32": row["runtime_sec_per_chunk32"],
                    "track_or_primitive_row_count": int(total_tracks),
                    "source_artifact": _rel(V99_D4RT_PROVIDER / "provider_rows.csv"),
                }
            )
            for r in d4rt_ae_rows:
                stitch_rows.append(
                    {
                        "schema_version": "stream4d_v101_phase2_stitch_metric_row_v1",
                        "phase_id": "v101_phase2_geometry_provider_capability",
                        "provider_name": name,
                        "scene_id": r.get("scene_id"),
                        "overlap_frame_count": 3,
                        "overlap_point_count": r.get("fit_kept_anchor_count_mean", ""),
                        "Sim3_scale": r.get("scale_curr_to_prev_mean", ""),
                        "scale_deviation_from_1": abs(_num(r.get("scale_curr_to_prev_mean"), 1.0) - 1.0),
                        "Sim3_residual_mean": "",
                        "Sim3_residual_p50": "",
                        "Sim3_residual_p90": r.get("residual_p90_curr_to_prev_mean", ""),
                        "Sim3_residual_max": r.get("residual_p90_curr_to_prev_max", ""),
                        "inlier_ratio": "",
                        "stitch_pass": _bool(r.get("provider_gate_pass")),
                        "source_artifact": _rel(V99_D4RT_PROVIDER / "provider_rows.csv"),
                    }
                )
            row["track_or_primitive_row_count"] = int(total_tracks)
            row["v100_verifier_result"] = v100_d4rt.get("decision")
            row["v100_d4rt_real_minus_control_MV_AP_scene"] = v100_d4rt.get("d4rt_real_minus_control_MV_AP_scene")
            row["v99_anchor_scene_best_MV_AP_scene"] = d4rt_ah.get("best_real_MV_AP_scene")

        elif name == "G2_DA3_STREAMING_BASE":
            scene_rows = da3_base.get("scene_rows", [])
            row.update(
                {
                    "input_resolution": "504x378",
                    "supports_chunk32": bool(da3_base.get("provider_gate_pass")),
                    "outputs_depth": True,
                    "outputs_confidence": True,
                    "outputs_pose_or_ray": True,
                    "outputs_intrinsics": True,
                    "outputs_extrinsics": True,
                    "outputs_point_cloud": True,
                    "provider_scale_type": "relative",
                    "runtime_sec_per_chunk32": "",
                    "runtime_sec_per_frame": "",
                    "failed_chunk_count": 0,
                    "contract_status": "provider_contract_pass_holdout_only",
                    "contract_failure_reason": "Passes chunk32/o3 holdout provider audit, but v101 dev-split bridge/purity contract is not yet proven.",
                    "source_artifact": _rel(V99_DA3_BASE / "summary.json"),
                }
            )
            for sr in scene_rows:
                stitch_rows.append(
                    {
                        "schema_version": "stream4d_v101_phase2_stitch_metric_row_v1",
                        "phase_id": "v101_phase2_geometry_provider_capability",
                        "provider_name": name,
                        "scene_id": sr.get("scene_id"),
                        "overlap_frame_count": 3,
                        "overlap_point_count": sr.get("combined_pcd_vertex_count"),
                        "Sim3_scale": sr.get("self_overlap_stitch_scale_mean"),
                        "scale_deviation_from_1": abs(_num(sr.get("self_overlap_stitch_scale_mean"), 1.0) - 1.0),
                        "Sim3_residual_mean": sr.get("self_overlap_stitch_mean_error_mean"),
                        "Sim3_residual_p50": "",
                        "Sim3_residual_p90": sr.get("self_overlap_stitch_mean_error_max"),
                        "Sim3_residual_max": sr.get("self_overlap_stitch_mean_error_max"),
                        "inlier_ratio": "",
                        "stitch_pass": bool(sr.get("provider_gate_pass")),
                        "source_artifact": _rel(V99_DA3_BASE / "summary.json"),
                    }
                )
            runtime_rows.append(
                {
                    "schema_version": "stream4d_v101_phase2_provider_runtime_row_v1",
                    "phase_id": "v101_phase2_geometry_provider_capability",
                    "provider_name": name,
                    "scene_count": len(scene_rows),
                    "runtime_total_sec": "",
                    "runtime_sec_per_chunk32": "",
                    "track_or_primitive_row_count": int(sum(int(sr.get("result_npz_count", 0)) for sr in scene_rows)),
                    "source_artifact": _rel(V99_DA3_BASE / "summary.json"),
                }
            )

        elif name in partial_roots:
            root = partial_roots[name]
            result_count = len(list((root / "results_output").glob("frame_*.npz"))) if (root / "results_output").exists() else 0
            row.update(
                {
                    "input_resolution": "504x378" if result_count else "",
                    "supports_chunk32": result_count > 0,
                    "outputs_depth": result_count > 0,
                    "outputs_confidence": result_count > 0,
                    "outputs_pose_or_ray": result_count > 0,
                    "outputs_intrinsics": result_count > 0,
                    "outputs_extrinsics": result_count > 0,
                    "outputs_point_cloud": (root / "pcd").exists(),
                    "provider_scale_type": "relative",
                    "contract_status": "provider_contract_partial_scene0050_only" if result_count else "provider_contract_fail",
                    "contract_failure_reason": "Only scene0050 partial artifact exists; full dev/holdout chunk32/o3 bridge/purity contract not proven.",
                    "source_artifact": _rel(root),
                    "track_or_primitive_row_count": result_count,
                }
            )
            failure_rows.append(_failure(name, "provider_contract_partial", row["contract_failure_reason"], "provider_ladder_required"))

        elif name == "G7_depth_anything_cpp_BASE_q8_0":
            metric = v98_metric.get("P3_depth_anything_cpp_q8_0_noavx512", {})
            row.update(
                {
                    "input_resolution": "smoke_only",
                    "supports_chunk32": False,
                    "outputs_depth": _bool(metric.get("depth_valid_rate", "0")),
                    "outputs_confidence": False,
                    "outputs_pose_or_ray": _num(metric.get("ray_or_pose_available_rate")) > 0,
                    "outputs_intrinsics": True,
                    "outputs_extrinsics": True,
                    "outputs_point_cloud": _num(metric.get("point_cloud_available_rate")) > 0,
                    "provider_scale_type": "unknown",
                    "contract_status": "provider_contract_fail",
                    "contract_failure_reason": "depth-anything.cpp q8 smoke exists but confidence and chunk32/o3 overlap-stitch bridge contract are not exposed.",
                    "source_artifact": _rel(V98_PROVIDER / "summary.json"),
                }
            )
            failure_rows.append(_failure(name, "provider_contract_fail", row["contract_failure_reason"], "provider_ladder_required"))

        else:
            row["contract_failure_reason"] = "No verified local artifact for this provider/model variant in current workspace."
            failure_rows.append(_failure(name, "provider_contract_fail", row["contract_failure_reason"], "provider_ladder_required"))

        rows.append(row)
        if row["contract_status"] == "provider_contract_fail" and name not in {fr["provider_name"] for fr in failure_rows}:
            failure_rows.append(_failure(name, "provider_contract_fail", row["contract_failure_reason"], "provider_ladder_required"))

    return rows, runtime_rows, stitch_rows, failure_rows


def _failure(provider_name: str, failure_type: str, reason: str, severity: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v101_phase2_provider_failure_row_v1",
        "phase_id": "v101_phase2_geometry_provider_capability",
        "provider_name": provider_name,
        "failure_type": failure_type,
        "reason": reason,
        "severity": severity,
    }


def _mask_support_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    da3_base = _read_json(V99_DA3_BASE / "summary.json")
    scene_specs = {row["scene_id"]: row for row in da3_base.get("scene_rows", [])}
    mask_rows = pd.read_parquet(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet")
    frame_eval = pd.read_csv(PHASE2C_DIR / "frame_eval_rows.csv")
    frame_mask_path = {
        (str(r["dataset_split"]), str(r["scene_id"]), int(r["frame_id"])): _project(str(r["mask_path"]))
        for r in frame_eval.to_dict("records")
    }
    hold = mask_rows[mask_rows["dataset_split"].astype(str) == "holdout"].copy()
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, Path]] = {}
    for row in hold.to_dict("records"):
        scene = str(row["scene_id"])
        frame_id = int(row["frame_id"])
        selected_mask_id = int(row["selected_mask_id"])
        spec = scene_specs.get(scene)
        if not spec:
            continue
        output_dir = _project(spec["output_dir"])
        npz_path = output_dir / "results_output" / f"frame_{int(row.get('da3_frame_index', frame_id // 5 if frame_id % 5 == 0 else frame_id))}.npz"
        if not npz_path.exists():
            # DA3 result names are manifest indices. Fall back through manifest by frame id.
            manifest = pd.read_csv(_project(spec["input_dir"]) / "frame_manifest_rows.csv")
            hit = manifest[manifest["frame_id"].astype(int) == frame_id]
            if hit.empty:
                continue
            npz_path = output_dir / "results_output" / f"frame_{int(hit.iloc[0]['da3_frame_index'])}.npz"
        key = (scene, frame_id)
        if key not in cache:
            z = np.load(npz_path)
            conf = np.asarray(z["conf"], dtype=np.float32)
            depth = np.asarray(z["depth"], dtype=np.float32)
            mask_path = frame_mask_path.get(("holdout", scene, frame_id))
            if mask_path is None or not mask_path.exists():
                failure_rows.append(_failure("G2_DA3_STREAMING_BASE", "missing_mask_raster_for_support", f"{scene} frame={frame_id}", "support_quality"))
                continue
            label = _read_label(mask_path, tuple(conf.shape[:2]))
            valid = np.isfinite(depth) & (depth > 0) & np.isfinite(conf) & (conf > 0)
            cache[key] = (label, valid, npz_path)
        label, valid, used_npz = cache[key]
        mask = label == selected_mask_id
        support = mask & valid
        area = int(np.count_nonzero(mask))
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = mask.astype(np.uint8)
        boundary = (cv2.dilate(mask_u8, kernel, iterations=1) != cv2.erode(mask_u8, kernel, iterations=1)) & mask
        boundary_support = boundary & valid
        ratio = float(area / max(1, label.size))
        support_rows.append(
            {
                "schema_version": "stream4d_v101_phase2_mask_support_row_v1",
                "phase_id": "v101_phase2_geometry_provider_capability",
                "provider_name": "G2_DA3_STREAMING_BASE",
                "dataset_split": "holdout",
                "scene_id": scene,
                "chunk_id": row["chunk_id"],
                "frame_id": frame_id,
                "selected_mask_id": selected_mask_id,
                "mv_object_id": row["mv_object_id"],
                "support_count": int(np.count_nonzero(support)),
                "support_density": float(np.count_nonzero(support) / max(1, area)),
                "boundary_band_support_count": int(np.count_nonzero(boundary_support)),
                "boundary_band_support_density": float(np.count_nonzero(boundary_support) / max(1, np.count_nonzero(boundary))),
                "mask_area": area,
                "mask_area_ratio": ratio,
                "mask_objectness_bucket": "broad" if ratio >= BROAD_MASK_AREA_RATIO else "small" if ratio < OBJECT_LIKE_MIN_AREA_RATIO else "object_like",
                "source_container_id": _rel(used_npz),
            }
        )
    return support_rows, failure_rows


def _support_gate_rows(support_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not support_rows:
        return []
    df = pd.DataFrame(support_rows)
    rows: list[dict[str, Any]] = []
    for provider, sub in df.groupby("provider_name"):
        obj = sub[sub["mask_objectness_bucket"] == "object_like"]
        small = sub[sub["mask_objectness_bucket"] == "small"]
        broad = sub[sub["mask_objectness_bucket"] == "broad"]
        boundary = sub["boundary_band_support_count"].astype(float)
        object_like_support_p10 = float(obj["support_count"].quantile(0.1)) if len(obj) else 0.0
        small_support_p10 = float(small["support_count"].quantile(0.1)) if len(small) else ""
        small_support_ok = True if not len(small) else float(small_support_p10) >= 2
        rows.append(
            {
                "schema_version": "stream4d_v101_phase2_mask_support_summary_row_v1",
                "phase_id": "v101_phase2_geometry_provider_capability",
                "provider_name": provider,
                "mask_support_count_mean": float(sub["support_count"].mean()),
                "mask_support_count_p10": float(sub["support_count"].quantile(0.1)),
                "mask_support_count_p50": float(sub["support_count"].quantile(0.5)),
                "mask_support_count_p90": float(sub["support_count"].quantile(0.9)),
                "object_like_mask_count": int(len(obj)),
                "object_like_mask_support_p10": object_like_support_p10,
                "small_mask_count": int(len(small)),
                "small_mask_support_p10": small_support_p10,
                "small_object_support_gate_status": "no_small_masks_in_scope" if not len(small) else "pass" if small_support_ok else "fail",
                "broad_mask_count": int(len(broad)),
                "boundary_band_support_mean": float(boundary.mean()),
                "boundary_band_support_p10": float(boundary.quantile(0.1)),
                "broad_mask_support_mean": float(broad["support_count"].mean()) if len(broad) else 0.0,
                "coverage_gate_pass": bool(
                    object_like_support_p10 >= 5
                    and small_support_ok
                    and float(boundary.quantile(0.1)) >= 1
                ),
                "scope_note": "holdout Phase2c masks only; DA3-BASE provider artifact is holdout chunk32/o3.",
            }
        )
    return rows


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [(s, l) for s, l in zip(scores, labels) if l == 1]
    neg = [(s, l) for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def _bridge_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bridge_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    object_rows = pd.read_parquet(PHASE2C_DIR / "mv_object_rows.parquet")
    hold_obj = object_rows[object_rows["dataset_split"].astype(str) == "holdout"].copy()
    source_to_mv = {str(r["source_mv_object_id"]): str(r["mv_object_id"]) for r in hold_obj.to_dict("records")}
    overlap = pd.read_csv(PHASE1_DIR / "pred_gt_overlap_rows.csv")
    hold_overlap = overlap[overlap["dataset_split"].astype(str) == "holdout"].copy()
    best_gt: dict[str, dict[str, Any]] = {}
    for oid, sub in hold_overlap.groupby("mv_object_id"):
        idx = sub["IoU"].astype(float).idxmax()
        row = sub.loc[idx].to_dict()
        best_gt[str(oid)] = {
            "raw_gt_object_id": int(row["raw_gt_object_id"]),
            "best_gt_iou": float(row["IoU"]),
            "scene_id": str(row["scene_id"]),
            "chunk_id": str(row["chunk_id"]),
        }
    candidate_path = V99_D4RT_ANCHOR / "local2history_candidate_rows.csv"
    candidates = pd.read_csv(candidate_path)
    pair_score: dict[tuple[str, str], float] = {}
    for r in candidates.to_dict("records"):
        a_source = str(r["mv_object_id_a"])
        b_source = str(r["mv_object_id_b"])
        a = source_to_mv.get(a_source, "")
        b = source_to_mv.get(b_source, "")
        if not a or not b:
            continue
        ga = best_gt.get(a, {})
        gb = best_gt.get(b, {})
        same = (
            ga.get("scene_id") == gb.get("scene_id")
            and ga.get("raw_gt_object_id") == gb.get("raw_gt_object_id")
            and _num(ga.get("best_gt_iou")) >= 0.05
            and _num(gb.get("best_gt_iou")) >= 0.05
        )
        score = float(_num(r.get("object_anchor_overlap")))
        key = tuple(sorted([a, b]))
        pair_score[key] = max(pair_score.get(key, 0.0), score)
        bridge_rows.append(
            {
                "schema_version": "stream4d_v101_phase2_mask_pair_bridge_row_v1",
                "phase_id": "v101_phase2_geometry_provider_capability",
                "provider_name": "G0_D4RT_reliable_anchors_only",
                "bridge_scope": "object_pair_proxy_from_D4RT_anchor_overlap_not_mask_pair",
                "scene_id": r.get("scene_id"),
                "left_chunk_id": r.get("left_chunk_id"),
                "right_chunk_id": r.get("right_chunk_id"),
                "obj_i": a,
                "obj_j": b,
                "mask_a": "",
                "mask_b": "",
                "Bridge": score,
                "same_object_GT_diagnostic": bool(same),
                "obj_i_best_gt": ga.get("raw_gt_object_id", ""),
                "obj_j_best_gt": gb.get("raw_gt_object_id", ""),
                "obj_i_best_gt_iou": ga.get("best_gt_iou", ""),
                "obj_j_best_gt_iou": gb.get("best_gt_iou", ""),
                "shared_anchor_count": r.get("shared_anchor_count"),
                "anchor_family": r.get("anchor_family"),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
            }
        )

    # Universe of adjacent holdout object pairs, scored by D4RT candidate overlap if present, else zero.
    universe_scores: list[float] = []
    universe_labels: list[int] = []
    by_scene_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in hold_obj.to_dict("records"):
        oid = str(row["mv_object_id"])
        by_scene_chunk[(str(row["scene_id"]), str(row["chunk_id"]))].append(oid)
    same_universe = 0
    diff_universe = 0
    for scene in sorted({k[0] for k in by_scene_chunk}):
        chunks = sorted([k[1] for k in by_scene_chunk if k[0] == scene], key=_chunk_index)
        for left, right in zip(chunks[:-1], chunks[1:]):
            for a in by_scene_chunk[(scene, left)]:
                for b in by_scene_chunk[(scene, right)]:
                    ga = best_gt.get(a, {})
                    gb = best_gt.get(b, {})
                    valid = _num(ga.get("best_gt_iou")) >= 0.05 and _num(gb.get("best_gt_iou")) >= 0.05
                    same = bool(valid and ga.get("raw_gt_object_id") == gb.get("raw_gt_object_id"))
                    if same:
                        same_universe += 1
                    else:
                        diff_universe += 1
                    key = tuple(sorted([a, b]))
                    universe_scores.append(float(pair_score.get(key, 0.0)))
                    universe_labels.append(1 if same else 0)
    auc = _roc_auc(universe_scores, universe_labels)
    for tau in TAUS:
        selected = [(s, l) for s, l in zip(universe_scores, universe_labels) if s >= tau]
        same_selected = sum(1 for _s, label in selected if label == 1)
        diff_selected = sum(1 for _s, label in selected if label == 0)
        recall = float(same_selected / max(1, same_universe))
        false_rate_selected = float(diff_selected / max(1, len(selected)))
        curve_rows.append(
            {
                "schema_version": "stream4d_v101_phase2_bridge_curve_row_v1",
                "phase_id": "v101_phase2_geometry_provider_capability",
                "provider_name": "G0_D4RT_reliable_anchors_only",
                "bridge_scope": "object_pair_proxy_from_D4RT_anchor_overlap_not_mask_pair",
                "tau": tau,
                "same_object_bridge_recall_at_tau": recall,
                "same_object_pair_universe_count": same_universe,
                "selected_pair_count": len(selected),
                "same_object_selected_count": same_selected,
                "false_bridge_rate_same_semantic_diff_GT_at_tau": false_rate_selected,
                "false_bridge_proxy_note": "No semantic category metadata; false rate is diff-best-GT among selected adjacent object pairs.",
                "bridge_AUC_diagnostic": auc,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
            }
        )
    if auc is None:
        failure_rows.append(_failure("G0_D4RT_reliable_anchors_only", "bridge_auc_unavailable", "No positive or negative adjacent-pair labels for AUC.", "bridge_quality"))
    return bridge_rows, curve_rows, failure_rows


def _surfel_purity_rows(provider_names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name in provider_names:
        reason = (
            "Phase2 could not compute v101 surfel purity because this provider artifact does not expose persistent primitive-to-GT projection rows in the required schema."
        )
        rows.append(
            {
                "schema_version": "stream4d_v101_phase2_surfel_purity_row_v1",
                "phase_id": "v101_phase2_geometry_provider_capability",
                "provider_name": name,
                "visibility_count": "",
                "valid_projection_rate": "",
                "GT_purity_diagnostic": "",
                "multi_GT_rate_diagnostic": "",
                "background_flip_rate_diagnostic": "",
                "semantic_residual_variance": "",
                "mask_membership_entropy": "",
                "provider_confidence_mean": "",
                "surfel_purity_available": False,
                "reason": reason,
                "uses_gt_for_prediction": False,
            }
        )
        failures.append(_failure(name, "surfel_purity_contract_fail", reason, "purity_required_before_phase3"))
    return rows, failures


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(PHASE1_DIR / "summary.json")
    phase1b = _read_json(PHASE1B_DIR / "summary.json")
    if not bool(phase1.get("phase1_pass")):
        raise RuntimeError("Phase1 did not pass; refusing Phase2 provider audit.")

    contract_rows, runtime_rows, stitch_rows, failure_rows = _contract_rows()
    support_rows, support_failures = _mask_support_rows()
    support_summary_rows = _support_gate_rows(support_rows)
    bridge_rows, curve_rows, bridge_failures = _bridge_rows()
    provider_names = [r["provider_name"] for r in contract_rows]
    purity_rows, purity_failures = _surfel_purity_rows(provider_names)
    failure_rows.extend(support_failures)
    failure_rows.extend(bridge_failures)
    failure_rows.extend(purity_failures)

    support_by_provider = {r["provider_name"]: r for r in support_summary_rows}
    best_curve_at_010: dict[str, dict[str, Any]] = {}
    auc_by_provider: dict[str, float] = {}
    for row in curve_rows:
        if abs(float(row["tau"]) - 0.10) < 1e-9:
            best_curve_at_010[row["provider_name"]] = row
        if row.get("bridge_AUC_diagnostic") not in {"", None}:
            auc_by_provider[row["provider_name"]] = float(row["bridge_AUC_diagnostic"])

    gate_rows: list[dict[str, Any]] = []
    provider_potential: list[str] = []
    for row in contract_rows:
        name = row["provider_name"]
        contract_ok = str(row["contract_status"]).startswith("provider_contract_pass")
        support_ok = bool(support_by_provider.get(name, {}).get("coverage_gate_pass", False))
        curve = best_curve_at_010.get(name, {})
        recall_ok = _num(curve.get("same_object_bridge_recall_at_tau")) >= 0.35
        false_ok = bool(curve) and _num(curve.get("false_bridge_rate_same_semantic_diff_GT_at_tau"), 1.0) <= 0.20
        auc_ok = _num(auc_by_provider.get(name), 0.0) >= 0.65
        purity_ok = False
        full_ok = bool(contract_ok and support_ok and recall_ok and false_ok and auc_ok and purity_ok)
        if full_ok:
            provider_potential.append(name)
        gate_rows.append(
            {
                "schema_version": "stream4d_v101_phase2_provider_gate_row_v1",
                "phase_id": "v101_phase2_geometry_provider_capability",
                "provider_name": name,
                "contract_ok": contract_ok,
                "coverage_gate_pass": support_ok,
                "same_object_bridge_recall_at_0p10": curve.get("same_object_bridge_recall_at_tau", ""),
                "bridge_recall_gate_pass": recall_ok,
                "false_bridge_rate_at_0p10": curve.get("false_bridge_rate_same_semantic_diff_GT_at_tau", ""),
                "false_bridge_gate_pass": false_ok,
                "bridge_AUC_diagnostic": auc_by_provider.get(name, ""),
                "bridge_auc_gate_pass": auc_ok,
                "surfel_purity_gate_pass": purity_ok,
                "provider_has_v101_bridge_potential": full_ok,
                "gate_note": "D4RT bridge is object-pair proxy; DA3/depth-anything providers lack persistent primitive bridge rows in current artifacts.",
            }
        )

    provider_bridge_potential_confirmed = bool(provider_potential)
    decision = (
        "PASS_PROVIDER_BRIDGE_AVAILABLE_ENTER_PHASE3"
        if provider_bridge_potential_confirmed
        else "NO_GO_PROVIDER_BRIDGE_INSUFFICIENT_BLOCK_PHASE3_FRAGMENT_REPAIR"
    )

    files = {
        "provider_contract_rows": OUT_DIR / "provider_contract_rows.csv",
        "provider_runtime_rows": OUT_DIR / "provider_runtime_rows.csv",
        "stitch_metric_rows": OUT_DIR / "stitch_metric_rows.csv",
        "mask_support_rows": OUT_DIR / "mask_support_rows.csv",
        "mask_support_summary_rows": OUT_DIR / "mask_support_summary_rows.csv",
        "mask_pair_bridge_rows": OUT_DIR / "mask_pair_bridge_rows.csv",
        "bridge_curve_rows": OUT_DIR / "bridge_curve_rows.csv",
        "surfel_purity_rows": OUT_DIR / "surfel_purity_rows.csv",
        "provider_gate_rows": OUT_DIR / "provider_gate_rows.csv",
        "provider_failure_rows": OUT_DIR / "provider_failure_rows.csv",
    }
    _write_csv(files["provider_contract_rows"], contract_rows)
    _write_csv(files["provider_runtime_rows"], runtime_rows)
    _write_csv(files["stitch_metric_rows"], stitch_rows)
    _write_csv(files["mask_support_rows"], support_rows)
    _write_csv(files["mask_support_summary_rows"], support_summary_rows)
    _write_csv(files["mask_pair_bridge_rows"], bridge_rows)
    _write_csv(files["bridge_curve_rows"], curve_rows)
    _write_csv(files["surfel_purity_rows"], purity_rows)
    _write_csv(files["provider_gate_rows"], gate_rows)
    _write_csv(files["provider_failure_rows"], failure_rows)

    summary = {
        "schema_version": "stream4d_v101_phase2_geometry_provider_capability_summary_v1",
        "phase_id": "v101_phase2_geometry_provider_capability",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase2_completed": True,
        "provider_bridge_potential_confirmed": provider_bridge_potential_confirmed,
        "provider_bridge_potential_providers": provider_potential,
        "fragmentation_context": {
            "phase1_decision": phase1.get("decision"),
            "phase1_fragmentation_confirmed": phase1.get("fragmentation_confirmed"),
            "phase1_merge_potential_confirmed": phase1.get("merge_potential_confirmed"),
            "phase1b_decision": phase1b.get("decision"),
            "phase1b_route": phase1b.get("analysis", {}).get("route"),
        },
        "provider_count": len(contract_rows),
        "provider_contract_pass_or_partial_count": sum(1 for r in contract_rows if str(r.get("contract_status", "")).startswith("provider_contract_pass") or "partial" in str(r.get("contract_status", ""))),
        "provider_failure_count": len(failure_rows),
        "mask_support_row_count": len(support_rows),
        "mask_pair_bridge_row_count": len(bridge_rows),
        "bridge_curve_row_count": len(curve_rows),
        "surfel_purity_available_provider_count": 0,
        "G0_D4RT_bridge_at_0p10": best_curve_at_010.get("G0_D4RT_reliable_anchors_only", {}),
        "G2_DA3_BASE_mask_support_summary": support_by_provider.get("G2_DA3_STREAMING_BASE", {}),
        "analysis": {
            "main_blocker": "No current provider satisfies the full v101 bridge contract. D4RT has an object-level anchor proxy but fails the false-bridge/AUC/purity requirements; DA3 has dense mask support coverage but no persistent bridge primitive in current artifacts.",
            "direct_merge_allowed": False,
            "reason": "Phase1b merge potential is low and Phase2 provider bridge potential is not confirmed.",
        },
        "outputs": {key: _rel(path) for key, path in files.items()} | {"summary": _rel(OUT_DIR / "summary.json")},
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if provider_bridge_potential_confirmed else 2


if __name__ == "__main__":
    raise SystemExit(main())
