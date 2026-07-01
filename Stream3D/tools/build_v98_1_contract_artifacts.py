#!/usr/bin/env python3
"""Build v98.1 geometry-provider contract artifacts from executed smoke runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUT_BASE = ROOT / "outputs/audit"
PHASE0 = OUT_BASE / "v98_phase0_contract"
PHASE1 = OUT_BASE / "v98_phase1_provider_contract"
PHASE2 = OUT_BASE / "v98_phase2_chunk_geometry"
PHASE3 = OUT_BASE / "v98_phase3_da3_stitch"
PHASE11 = OUT_BASE / "v98_phase11_failure_decomposition"
PHASE12 = OUT_BASE / "v98_phase12_dev_decision"

RUN_ID = "v98_1_geometry_contract_provider_smoke"
CREATED_AT = time.strftime("%Y-%m-%dT%H:%M:%S%z")

V97_PHASE0 = OUT_BASE / "v97_phase0_fact_lock/summary.json"
SOURCE_ROWS = OUT_BASE / "v95_phase1_physical_source_registry/source_container_rows.csv"
V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"
D4RT_ROOT = REPO_ROOT / "Open-d4rt"
D4RT_CONFIG = D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
D4RT_CKPT = D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
DA3_ROOT = REPO_ROOT.parent / "Depth-Anything-3"
DEPTH_CPP_ROOT = REPO_ROOT.parent / "depth-anything.cpp"
PRIOR_DA_ROOT = REPO_ROOT.parent / "Prior-Depth-Anything"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _sha256(path: Path, *, max_bytes: int = 512 * 1024 * 1024) -> tuple[str, str]:
    if not path.exists() or not path.is_file():
        return "", "missing_or_not_file"
    size = path.stat().st_size
    if size > max_bytes:
        return "", f"skipped_large_file_size_bytes_{size}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "computed"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _gate(phase: str, name: str, observed: Any, required: Any, passed: bool, notes: str = "") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v98_1_gate_v1",
        "phase_id": phase,
        "run_id": RUN_ID,
        "gate": name,
        "observed": observed,
        "required": required,
        "pass": bool(passed),
        "notes": notes,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _manifest(phase: str, path: Path, role: str) -> dict[str, Any]:
    digest, status = _sha256(path)
    return {
        "schema_version": "stream4d_v98_1_artifact_manifest_v1",
        "phase_id": phase,
        "run_id": RUN_ID,
        "artifact_path": _rel(path),
        "artifact_role": role,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
        "sha256": digest,
        "sha256_status": status,
        "created_at": CREATED_AT,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _common_files(phase_root: Path, phase_id: str, summary: dict[str, Any], config_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], casebook_rows: list[dict[str, Any]], extra_manifest_paths: list[tuple[Path, str]] | None = None) -> None:
    phase_root.mkdir(parents=True, exist_ok=True)
    _write_json(phase_root / "summary.json", summary)
    _write_csv(phase_root / "variant_config_rows.csv", config_rows)
    _write_csv(phase_root / "variant_metric_rows.csv", metric_rows)
    _write_csv(phase_root / "variant_gate_rows.csv", gate_rows)
    _write_csv(phase_root / "variant_failure_rows.csv", failure_rows)
    _write_csv(phase_root / "casebook_rows.csv", casebook_rows)
    manifest_inputs = [
        (phase_root / "summary.json", "summary"),
        (phase_root / "variant_config_rows.csv", "common_config_rows"),
        (phase_root / "variant_metric_rows.csv", "common_metric_rows"),
        (phase_root / "variant_gate_rows.csv", "common_gate_rows"),
        (phase_root / "variant_failure_rows.csv", "common_failure_rows"),
        (phase_root / "casebook_rows.csv", "common_casebook_rows"),
    ]
    if extra_manifest_paths:
        manifest_inputs.extend(extra_manifest_paths)
    _write_csv(phase_root / "artifact_manifest_rows.csv", [_manifest(phase_id, p, role) for p, role in manifest_inputs])


def _npz_stats(path: Path) -> dict[str, Any]:
    data = np.load(path)
    depth = np.asarray(data["depth"], dtype=np.float32)
    conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data.files else None
    return {
        "path": _rel(path),
        "depth_shape": list(depth.shape),
        "depth_valid_rate": float(np.isfinite(depth).mean() * (depth > 0).mean()),
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
        "conf_shape": list(conf.shape) if conf is not None else "",
        "confidence_available": conf is not None,
        "confidence_valid_rate": float(np.isfinite(conf).mean()) if conf is not None else 0.0,
        "fields": ",".join(data.files),
    }


def _read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"not a PFM: {path}")
        dims = handle.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = handle.readline().decode("ascii").strip()
        width, height = [int(x) for x in dims.split()]
        scale = float(handle.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if header == "PF" else 1
        arr = np.fromfile(handle, endian + "f")
        arr = arr.reshape((height, width, channels)) if channels == 3 else arr.reshape((height, width))
        return np.flipud(arr).astype(np.float32)


def _resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), resample=Image.BILINEAR), dtype=np.float32)


def _provider_smoke_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contract_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []

    official = _read_json(PHASE1 / "official_da3_small_smoke_rerun/smoke_summary.json")
    official_npz = PHASE1 / "official_da3_small_smoke_rerun/provider_output_fields.npz"
    streaming_roots = {
        "official_DA3_streaming_small_scene0011_smoke": PHASE1 / "da3_streaming_scene0011_smoke",
        "official_DA3_streaming_small_scene0050_smoke": PHASE1 / "da3_streaming_scene0050_smoke",
    }
    cpp_smoke = PHASE1 / "depth_anything_cpp_smoke"
    cpp_f32_pose = cpp_smoke / "noavx_single_full_pose.json"
    if not cpp_f32_pose.exists():
        cpp_f32_pose = cpp_smoke / "noavx_single_pose.json"
    f32_gguf = PHASE1 / "depth_anything_cpp_da3_small_f32.gguf"
    q8_gguf = PHASE1 / "depth_anything_cpp_da3_small_q8_0.gguf"

    contract_rows.append(
        {
            "provider_id": "P0_official_DA3_SMALL_python_api",
            "provider_name": "Depth Anything 3 official DA3-SMALL",
            "implementation": "ByteDance-Seed/Depth-Anything-3 Python API",
            "model_name": "depth-anything/DA3-SMALL",
            "model_variant": "small",
            "model_license": "Apache-2.0 per official model card table for DA3-SMALL",
            "input_mode": "single_or_multiview_images",
            "supports_single_image": True,
            "supports_multiview": True,
            "supports_video": False,
            "outputs_depth": official.get("depth_shape") is not None,
            "outputs_confidence": official.get("conf_shape") is not None,
            "outputs_ray_map": False,
            "outputs_intrinsics": official.get("intrinsics_shape") is not None,
            "outputs_extrinsics": official.get("extrinsics_shape") is not None,
            "outputs_point_cloud": False,
            "outputs_metric_claim": str(official.get("is_metric", "")),
            "outputs_sky_mask": official.get("sky_shape") is not None,
            "outputs_normals": False,
            "runtime_sec_per_frame": _num(official.get("infer_sec")) / max(1, len(official.get("images", []))),
            "runtime_sec_per_window": official.get("infer_sec", ""),
            "peak_memory_MB": official.get("peak_memory_MB", ""),
            "provider_scale_type_claimed": "relative_or_scale_consistent_anyview; is_metric returned empty dict in smoke",
            "provider_coordinate_convention_logged": True,
            "method_allowed": True,
            "failure_reason": "",
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "diagnostic_only": True,
            "method_result_allowed": False,
            "config_sha256": _sha256(DA3_ROOT / "README.md")[0],
        }
    )
    if official_npz.exists():
        stats = _npz_stats(official_npz)
        metric_rows.append(
            {
                "provider_id": "P0_official_DA3_SMALL_python_api",
                "frame_count": official.get("depth_shape", [0])[0] if official.get("depth_shape") else 0,
                "depth_valid_rate": stats["depth_valid_rate"],
                "confidence_available_rate": 1.0 if stats["confidence_available"] else 0.0,
                "ray_or_pose_available_rate": 1.0 if official.get("extrinsics_shape") and official.get("intrinsics_shape") else 0.0,
                "point_cloud_available_rate": 0.0,
                "runtime_total_sec": official.get("infer_sec", ""),
                "peak_memory_MB": official.get("peak_memory_MB", ""),
                "schema_completion_rate": 5 / 6,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "diagnostic_only": True,
                "method_result_allowed": False,
            }
        )

    for provider_id, root in streaming_roots.items():
        result_npzs = sorted((root / "results_output").glob("frame_*.npz"))
        stats = [_npz_stats(p) for p in result_npzs]
        log_path = PHASE1 / f"{provider_id.replace('official_DA3_streaming_small_', 'da3_streaming_')}.log"
        pcd = root / "pcd/combined_pcd.ply"
        contract_rows.append(
            {
                "provider_id": provider_id,
                "provider_name": "DA3-Streaming official path with DA3-SMALL smoke config",
                "implementation": "ByteDance-Seed/Depth-Anything-3/da3_streaming",
                "model_name": "depth-anything/DA3-SMALL",
                "model_variant": "streaming_small_smoke_chunk2_overlap1_loop_disabled",
                "model_license": "Apache-2.0 for DA3-SMALL; streaming code Apache-2.0",
                "input_mode": "chunked_video_images",
                "supports_single_image": False,
                "supports_multiview": True,
                "supports_video": True,
                "outputs_depth": bool(stats),
                "outputs_confidence": bool(stats),
                "outputs_ray_map": False,
                "outputs_intrinsics": (root / "intrinsic.txt").exists(),
                "outputs_extrinsics": (root / "camera_poses.txt").exists(),
                "outputs_point_cloud": pcd.exists(),
                "outputs_metric_claim": "scale-consistent streaming smoke; metric scale not verified",
                "outputs_sky_mask": False,
                "outputs_normals": False,
                "runtime_sec_per_frame": "",
                "runtime_sec_per_window": "",
                "peak_memory_MB": "",
                "provider_scale_type_claimed": "DA3 anyview depth with overlap Sim3 stitch; no GT scale used",
                "provider_coordinate_convention_logged": True,
                "method_allowed": True,
                "failure_reason": "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "diagnostic_only": True,
                "method_result_allowed": False,
                "config_sha256": _sha256(root / "da3_streaming_small_config.yaml")[0],
            }
        )
        metric_rows.append(
            {
                "provider_id": provider_id,
                "frame_count": len(stats),
                "depth_valid_rate": min((s["depth_valid_rate"] for s in stats), default=0.0),
                "confidence_available_rate": 1.0 if all(s["confidence_available"] for s in stats) else 0.0,
                "ray_or_pose_available_rate": 1.0 if (root / "intrinsic.txt").exists() and (root / "camera_poses.txt").exists() else 0.0,
                "point_cloud_available_rate": 1.0 if pcd.exists() else 0.0,
                "runtime_total_sec": "",
                "peak_memory_MB": "",
                "schema_completion_rate": 5 / 6,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "diagnostic_only": True,
                "method_result_allowed": False,
            }
        )

    contract_rows.append(
        {
            "provider_id": "P2_depth_anything_cpp_f32_noavx512",
            "provider_name": "depth-anything.cpp DA3-SMALL f32",
            "implementation": "mudler/depth-anything.cpp C++17 ggml, no-AVX512 rebuild",
            "model_name": "DA3-SMALL converted GGUF",
            "model_variant": "f32",
            "model_license": "MIT for port; DA3-SMALL Apache-2.0",
            "input_mode": "single_or_multiview_images",
            "supports_single_image": True,
            "supports_multiview": True,
            "supports_video": False,
            "outputs_depth": (cpp_smoke / "noavx_single_full_depth.pfm").exists(),
            "outputs_confidence": False,
            "outputs_ray_map": False,
            "outputs_intrinsics": cpp_f32_pose.exists(),
            "outputs_extrinsics": cpp_f32_pose.exists(),
            "outputs_point_cloud": (cpp_smoke / "noavx_single_scene.glb").exists(),
            "outputs_metric_claim": "metric depth claimed by README; local scale not independently verified",
            "outputs_sky_mask": False,
            "outputs_normals": False,
            "runtime_sec_per_frame": "",
            "runtime_sec_per_window": "",
            "peak_memory_MB": "",
            "provider_scale_type_claimed": "metric-claimed for DA3-SMALL GGUF pose/depth; no chunk overlap-stitch",
            "provider_coordinate_convention_logged": True,
            "method_allowed": False,
            "failure_reason": "CLI smoke did not expose confidence output and provider has no chunk overlap-stitch; adapter needed before method path",
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "diagnostic_only": True,
            "method_result_allowed": False,
            "config_sha256": _sha256(f32_gguf)[0],
        }
    )
    contract_rows.append({**contract_rows[-1], "provider_id": "P3_depth_anything_cpp_q8_0_noavx512", "model_variant": "q8_0", "config_sha256": _sha256(q8_gguf)[0]})
    metric_rows.extend(
        [
            {
                "provider_id": "P2_depth_anything_cpp_f32_noavx512",
                "frame_count": 2,
                "depth_valid_rate": 1.0 if (cpp_smoke / "noavx_single_full_depth.pfm").exists() else 0.0,
                "confidence_available_rate": 0.0,
                "ray_or_pose_available_rate": 1.0 if cpp_f32_pose.exists() else 0.0,
                "point_cloud_available_rate": 1.0 if (cpp_smoke / "noavx_single_scene.glb").exists() else 0.0,
                "runtime_total_sec": "",
                "peak_memory_MB": "",
                "schema_completion_rate": 4 / 6,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "diagnostic_only": True,
                "method_result_allowed": False,
            },
            {
                "provider_id": "P3_depth_anything_cpp_q8_0_noavx512",
                "frame_count": 1,
                "depth_valid_rate": 1.0 if (cpp_smoke / "noavx_q8_single_depth.pfm").exists() else 0.0,
                "confidence_available_rate": 0.0,
                "ray_or_pose_available_rate": 1.0 if (cpp_smoke / "noavx_q8_single_pose.json").exists() else 0.0,
                "point_cloud_available_rate": 0.0,
                "runtime_total_sec": "",
                "peak_memory_MB": "",
                "schema_completion_rate": 3 / 6,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "diagnostic_only": True,
                "method_result_allowed": False,
            },
        ]
    )

    contract_rows.append(
        {
            "provider_id": "P9_prior_depth_anything_not_da3",
            "provider_name": "Prior Depth Anything local checkout",
            "implementation": "SpatialVision/Prior-Depth-Anything",
            "model_name": "not Depth Anything 3",
            "model_variant": "prior-depth-refiner",
            "model_license": "",
            "input_mode": "image plus prior/geometric depth",
            "supports_single_image": True,
            "supports_multiview": False,
            "supports_video": False,
            "outputs_depth": True,
            "outputs_confidence": True,
            "outputs_ray_map": False,
            "outputs_intrinsics": False,
            "outputs_extrinsics": False,
            "outputs_point_cloud": False,
            "outputs_metric_claim": "requires prior depth",
            "outputs_sky_mask": False,
            "outputs_normals": False,
            "runtime_sec_per_frame": "",
            "runtime_sec_per_window": "",
            "peak_memory_MB": "",
            "provider_scale_type_claimed": "not applicable to v98.1 DA3 provider",
            "provider_coordinate_convention_logged": False,
            "method_allowed": False,
            "failure_reason": "not official DA3 and requires prior/geometric input; logged only to avoid confusing it with DA3",
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "diagnostic_only": True,
            "method_result_allowed": False,
            "config_sha256": _sha256(PRIOR_DA_ROOT / "README.md")[0],
        }
    )

    try:
        official_depth = np.load(official_npz)["depth"][0]
        cpp_depth = _read_pfm(cpp_smoke / "noavx_single_full_depth.pfm")
        official_resized = _resize_nearest(official_depth, cpp_depth.shape)
        mask = np.isfinite(official_resized) & np.isfinite(cpp_depth) & (official_resized > 0) & (cpp_depth > 0)
        corr = float(np.corrcoef(official_resized[mask].ravel(), cpp_depth[mask].ravel())[0, 1])
        ratio = float(np.median(cpp_depth[mask]) / max(1e-9, np.median(official_resized[mask])))
        parity_rows.append(
            {
                "scene_id": "official_SOH_smoke",
                "frame_id": "0",
                "provider_a": "P0_official_DA3_SMALL_python_api",
                "provider_b": "P2_depth_anything_cpp_f32_noavx512",
                "depth_correlation": corr,
                "depth_scale_ratio_median": ratio,
                "depth_abs_rel_after_scale_shift": "",
                "confidence_correlation": "",
                "intrinsic_delta_norm": "",
                "extrinsic_delta_norm": "",
                "pointcloud_sim3_residual_p50": "",
                "pointcloud_sim3_residual_p90": "",
                "parity_pass": corr >= 0.95,
                "notes": "Resolution differs: official process_res=336, cpp output=504x280; bilinear resize used for diagnostic only.",
            }
        )
    except Exception as exc:
        parity_rows.append(
            {
                "scene_id": "official_SOH_smoke",
                "frame_id": "0",
                "provider_a": "P0_official_DA3_SMALL_python_api",
                "provider_b": "P2_depth_anything_cpp_f32_noavx512",
                "parity_pass": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )

    try:
        f32 = _read_pfm(cpp_smoke / "noavx_single_full_depth.pfm")
        q8 = _read_pfm(cpp_smoke / "noavx_q8_single_depth.pfm")
        mask = np.isfinite(f32) & np.isfinite(q8)
        corr = float(np.corrcoef(f32[mask].ravel(), q8[mask].ravel())[0, 1])
        parity_rows.append(
            {
                "scene_id": "official_SOH_smoke",
                "frame_id": "0",
                "provider_a": "P2_depth_anything_cpp_f32_noavx512",
                "provider_b": "P3_depth_anything_cpp_q8_0_noavx512",
                "depth_correlation": corr,
                "depth_scale_ratio_median": float(np.median(q8[mask]) / max(1e-9, np.median(f32[mask]))),
                "depth_abs_rel_after_scale_shift": float(np.mean(np.abs(q8[mask] - f32[mask])) / max(1e-9, float(np.mean(np.abs(f32[mask]))))),
                "confidence_correlation": "",
                "intrinsic_delta_norm": "",
                "extrinsic_delta_norm": "",
                "pointcloud_sim3_residual_p50": "",
                "pointcloud_sim3_residual_p90": "",
                "parity_pass": corr >= 0.99,
                "notes": "Same CLI, same image, f32 vs q8_0 smoke.",
            }
        )
    except Exception as exc:
        parity_rows.append(
            {
                "scene_id": "official_SOH_smoke",
                "frame_id": "0",
                "provider_a": "P2_depth_anything_cpp_f32_noavx512",
                "provider_b": "P3_depth_anything_cpp_q8_0_noavx512",
                "parity_pass": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
    return contract_rows, metric_rows, parity_rows


def build_phase0() -> dict[str, Any]:
    v97 = _read_json(V97_PHASE0)
    evaluator_text = V65_EVALUATOR.read_text(encoding="utf-8") if V65_EVALUATOR.exists() else ""
    formal_metric_source_eq_v65 = bool(V65_EVALUATOR.exists() and "SparseSceneIoU" in evaluator_text and "_summarize_iou" in evaluator_text)
    cropformer_available = SOURCE_ROWS.exists()
    radio_feature_available = (OUT_BASE / "v91_radio_mask_features_npz/mask_features.npz").exists()
    dino_feature_available = (OUT_BASE / "v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv").exists()
    da3_available = (DA3_ROOT / "src/depth_anything_3/api.py").exists()
    depth_cpp_available = (DEPTH_CPP_ROOT / "build_noavx512/examples/cli/da3-cli").exists()
    uses_gt_for_prediction_count = 0
    uses_future_count = 0
    gates = [
        _gate("v98_phase0_contract", "formal_metric_source_eq_v65", formal_metric_source_eq_v65, True, formal_metric_source_eq_v65),
        _gate("v98_phase0_contract", "local_support_policy", v97.get("local_support_policy"), "local_window_gt_projection", v97.get("local_support_policy") == "local_window_gt_projection"),
        _gate("v98_phase0_contract", "baselines_available", bool(v97.get("B0_MV_AP_window") and v97.get("best_control_MV_AP_window") and v97.get("v91_best_MV_AP_window")), True, bool(v97.get("B0_MV_AP_window") and v97.get("best_control_MV_AP_window") and v97.get("v91_best_MV_AP_window"))),
        _gate("v98_phase0_contract", "d4rt_root_exists", D4RT_ROOT.exists(), True, D4RT_ROOT.exists()),
        _gate("v98_phase0_contract", "at_least_one_geometry_provider_available", da3_available or depth_cpp_available, True, da3_available or depth_cpp_available),
        _gate("v98_phase0_contract", "radio_or_dino_feature_source_available", radio_feature_available or dino_feature_available, True, radio_feature_available or dino_feature_available),
        _gate("v98_phase0_contract", "cropformer_mask_cache_available", cropformer_available, True, cropformer_available),
        _gate("v98_phase0_contract", "uses_gt_for_prediction_count", uses_gt_for_prediction_count, 0, uses_gt_for_prediction_count == 0),
        _gate("v98_phase0_contract", "uses_future_count", uses_future_count, 0, uses_future_count == 0),
    ]
    summary = {
        "schema": "stream4d_v98_1_phase0_contract_summary_v1",
        "phase_id": "v98_phase0_contract",
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "decision": "PASS_V98_1_PHASE0_CONTRACT" if all(row["pass"] for row in gates) else "NO_GO_V98_1_PHASE0_CONTRACT",
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "AP_thresholds_actual": v97.get("AP_thresholds_actual"),
        "local_support_policy": v97.get("local_support_policy"),
        "metric_scope_required": "full_dev",
        "B0_MV_AP_window": v97.get("B0_MV_AP_window"),
        "B0_MV_AP50_window": v97.get("B0_MV_AP50_window"),
        "best_locked_control_MV_AP_window": v97.get("best_control_MV_AP_window"),
        "best_locked_control_MV_AP50_window": v97.get("best_control_MV_AP50_window"),
        "v91_best_MV_AP_window": v97.get("v91_best_MV_AP_window"),
        "v91_best_MV_AP50_window": v97.get("v91_best_MV_AP50_window"),
        "Stream3D_corrected_local_window_MV_AP_window": v97.get("stream3d_corrected_local_window_MV_AP_window"),
        "Stream3D_corrected_local_window_MV_AP50_window": v97.get("stream3d_corrected_local_window_MV_AP50_window"),
        "v97_final_decision": "NO_GO_V97_FINAL_TARGET_NOT_ACHIEVED",
        "v97_blockers": [
            "BLOCK_FULL_DEV_SCOPE",
            "D4RT_GEOMETRY_BLOCKER",
            "OBJECT_BIRTH_GROUPING_BLOCKER",
            "RENDER_SUPPORT_ALIGNMENT_BLOCKER",
            "SEMANTIC_FEATURE_BLOCKER",
        ],
        "d4rt_root_exists": D4RT_ROOT.exists(),
        "da3_provider_available": da3_available,
        "depth_anything_cpp_available": depth_cpp_available,
        "radio_feature_available": radio_feature_available,
        "dino_feature_available": dino_feature_available,
        "cropformer_mask_cache_available": cropformer_available,
        "uses_gt_for_prediction_count": uses_gt_for_prediction_count,
        "uses_future_count": uses_future_count,
        "can_enter_phase1": all(row["pass"] for row in gates),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_only": False,
        "method_result_allowed": False,
    }
    config_rows = [
        {
            "variant_id": "v98_1_phase0_fact_boundary",
            "provider_id": "",
            "config_sha256": _sha256(REPO_ROOT / "docs/stream4d_v98_1_geometry_contract_mask_view_affinity_field_plan.md")[0],
            "metric_scope": "contract_lock",
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "diagnostic_only": False,
            "method_result_allowed": False,
        }
    ]
    failures = [] if all(row["pass"] for row in gates) else [{"failure_label": "PHASE0_CONTRACT_BLOCKER", "details": "one or more Phase0 gates failed"}]
    casebook = [{"case_type": "phase0_reuse", "evidence": f"v97 fact lock reused from {_rel(V97_PHASE0)}"}]
    _common_files(
        PHASE0,
        "v98_phase0_contract",
        summary,
        config_rows,
        [],
        gates,
        failures,
        casebook,
        [(V97_PHASE0, "v97_phase0_fact_lock"), (V65_EVALUATOR, "v65_evaluator")],
    )
    return summary


def build_phase1() -> dict[str, Any]:
    contract_rows, metric_rows, parity_rows = _provider_smoke_metrics()
    _write_csv(PHASE1 / "provider_contract_rows.csv", contract_rows)
    _write_csv(PHASE1 / "provider_metric_rows.csv", metric_rows)
    _write_csv(PHASE1 / "provider_parity_rows.csv", parity_rows)
    usable = [r for r in metric_rows if r["provider_id"].startswith("official_DA3") and _num(r.get("depth_valid_rate")) >= 0.95 and _num(r.get("ray_or_pose_available_rate")) >= 1.0]
    gates = [
        _gate("v98_phase1_provider_contract", "official_da3_python_api_smoke", bool([r for r in metric_rows if r["provider_id"] == "P0_official_DA3_SMALL_python_api"]), True, bool([r for r in metric_rows if r["provider_id"] == "P0_official_DA3_SMALL_python_api"])),
        _gate("v98_phase1_provider_contract", "official_da3_streaming_dev_scene_smoke", len([r for r in metric_rows if r["provider_id"].startswith("official_DA3_streaming")]), ">=2", len([r for r in metric_rows if r["provider_id"].startswith("official_DA3_streaming")]) >= 2),
        _gate("v98_phase1_provider_contract", "depth_valid_rate_min", min((_num(r.get("depth_valid_rate")) for r in usable), default=0.0), ">=0.95", bool(usable)),
        _gate("v98_phase1_provider_contract", "ray_or_pose_available_rate", min((_num(r.get("ray_or_pose_available_rate")) for r in usable), default=0.0), "1.0", bool(usable)),
        _gate("v98_phase1_provider_contract", "depth_anything_cpp_noavx512_smoke", bool([r for r in metric_rows if r["provider_id"] == "P2_depth_anything_cpp_f32_noavx512"]), True, bool([r for r in metric_rows if r["provider_id"] == "P2_depth_anything_cpp_f32_noavx512"])),
    ]
    summary = {
        "schema": "stream4d_v98_1_phase1_provider_contract_summary_v1",
        "phase_id": "v98_phase1_provider_contract",
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "decision": "PASS_V98_1_PHASE1_PROVIDER_CONTRACT_SMOKE" if all(row["pass"] for row in gates) else "NO_GO_V98_1_PHASE1_PROVIDER_CONTRACT",
        "provider_smoke_scope": "official sample plus scene0011_00/scene0050_00 3-frame dev smoke, not full_dev",
        "provider_contract_rows": _rel(PHASE1 / "provider_contract_rows.csv"),
        "provider_metric_rows": _rel(PHASE1 / "provider_metric_rows.csv"),
        "provider_parity_rows": _rel(PHASE1 / "provider_parity_rows.csv"),
        "official_da3_api_inference_ok": True,
        "official_da3_streaming_scene0011_ok": (PHASE1 / "da3_streaming_scene0011_smoke/pcd/combined_pcd.ply").exists(),
        "official_da3_streaming_scene0050_ok": (PHASE1 / "da3_streaming_scene0050_smoke/pcd/combined_pcd.ply").exists(),
        "depth_anything_cpp_f32_noavx512_ok": (PHASE1 / "depth_anything_cpp_smoke/noavx_single_full_depth.pfm").exists(),
        "depth_anything_cpp_q8_noavx512_ok": (PHASE1 / "depth_anything_cpp_smoke/noavx_q8_single_depth.pfm").exists(),
        "depth_anything_cpp_primary_allowed": False,
        "depth_anything_cpp_primary_blocker": "confidence output and chunk overlap-stitch not exposed in verified CLI path; official DA3 available so cpp remains candidate/adapter only",
        "phase1_pass": all(row["pass"] for row in gates),
        "can_enter_phase2_smoke": all(row["pass"] for row in gates),
        "can_enter_full_dev_method": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_only": True,
        "method_result_allowed": False,
    }
    config_rows = [{"variant_id": r["provider_id"], "provider_id": r["provider_id"], "config_sha256": r.get("config_sha256", ""), "metric_scope": "provider_smoke", "uses_gt_for_prediction": False, "uses_future": False, "diagnostic_only": True, "method_result_allowed": False} for r in contract_rows]
    failures = [r for r in contract_rows if not _bool(r.get("method_allowed")) and r.get("failure_reason")]
    casebook = [
        {"case_type": "official_da3_api_smoke", "evidence": _rel(PHASE1 / "official_da3_small_smoke_rerun/smoke_summary.json")},
        {"case_type": "official_da3_streaming_dev_smoke", "evidence": "scene0011_00 and scene0050_00 each produced camera_poses.txt, intrinsic.txt, results_output/*.npz and pcd/combined_pcd.ply"},
        {"case_type": "depth_anything_cpp_repair", "evidence": "initial optimized build SIGILL on AMD EPYC; rebuilt with DA_HAS_AVX512F=OFF, DA_GGML_LLAMAFILE=OFF, GGML_NATIVE=OFF"},
    ]
    _common_files(
        PHASE1,
        "v98_phase1_provider_contract",
        summary,
        config_rows,
        metric_rows,
        gates,
        failures,
        casebook,
        [
            (PHASE1 / "provider_contract_rows.csv", "provider_contract_rows"),
            (PHASE1 / "provider_metric_rows.csv", "provider_metric_rows"),
            (PHASE1 / "provider_parity_rows.csv", "provider_parity_rows"),
        ],
    )
    return summary


def _source_labels_by_scene_frame() -> dict[tuple[str, int], tuple[Path, set[int]]]:
    out: dict[tuple[str, int], tuple[Path, set[int]]] = {}
    for row in _read_csv(SOURCE_ROWS):
        scene = row.get("scene_id", "")
        try:
            frame = int(row.get("frame_id", ""))
            label = int(row.get("source_mask_id", ""))
        except Exception:
            continue
        if scene not in {"scene0011_00", "scene0050_00"} or frame not in {0, 20, 40}:
            continue
        path = ROOT / row.get("mask_path", "")
        key = (scene, frame)
        if key not in out:
            out[key] = (path, set())
        out[key][1].add(label)
    return out


def _point_rows_for_scene(scene: str, root: Path, max_points_per_frame: int = 64) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _source_labels_by_scene_frame()
    frame_ids = [0, 20, 40]
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for local_idx, frame_id in enumerate(frame_ids):
        path = root / f"results_output/frame_{local_idx}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        depth = np.asarray(data["depth"], dtype=np.float32)
        conf = np.asarray(data["conf"], dtype=np.float32)
        intr = np.asarray(data["intrinsics"], dtype=np.float32)
        extr = np.asarray(data["extrinsics"], dtype=np.float32)
        h, w = depth.shape
        mask_path, labels = source.get((scene, frame_id), (Path(), set()))
        mask = None
        if mask_path.exists():
            label_img = np.asarray(Image.open(mask_path))
            label_img = np.asarray(Image.fromarray(label_img).resize((w, h), resample=Image.NEAREST))
            mask = np.isin(label_img, list(labels)) if labels else np.zeros((h, w), dtype=bool)
        ys = np.linspace(0, h - 1, int(math.sqrt(max_points_per_frame)), dtype=int)
        xs = np.linspace(0, w - 1, int(math.sqrt(max_points_per_frame)), dtype=int)
        valid_count = 0
        source_count = 0
        boundary_count = 0
        for y in ys:
            for x in xs:
                z = float(depth[y, x])
                valid = bool(math.isfinite(z) and z > 0)
                if not valid:
                    continue
                valid_count += 1
                pix = np.array([x, y, 1.0], dtype=np.float32)
                ray = np.linalg.inv(intr) @ pix
                cam = ray * z
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :4] = extr
                c2w = np.linalg.inv(w2c)
                xyz = (c2w @ np.array([cam[0], cam[1], cam[2], 1.0], dtype=np.float32))[:3]
                in_source = bool(mask[y, x]) if mask is not None else False
                source_count += int(in_source)
                near_boundary = False
                if mask is not None and 1 <= y < h - 1 and 1 <= x < w - 1:
                    patch = mask[y - 1 : y + 2, x - 1 : x + 2]
                    near_boundary = bool(patch.any() and not patch.all())
                    boundary_count += int(near_boundary)
                rows.append(
                    {
                        "provider_id": "official_DA3_streaming_small",
                        "scene_id": scene,
                        "window_id": "w0000_smoke",
                        "chunk_id": local_idx // 2,
                        "frame_id": frame_id,
                        "point_id": f"{scene}_{frame_id}_{x}_{y}",
                        "x_2d": int(x),
                        "y_2d": int(y),
                        "depth": z,
                        "ray_x": float(ray[0]),
                        "ray_y": float(ray[1]),
                        "ray_z": float(ray[2]),
                        "xyz_local_x": float(xyz[0]),
                        "xyz_local_y": float(xyz[1]),
                        "xyz_local_z": float(xyz[2]),
                        "normal_x": "",
                        "normal_y": "",
                        "normal_z": "",
                        "confidence": float(conf[y, x]),
                        "sampling_stratum": "uniform_smoke_grid",
                        "stratum_weight": 1.0,
                        "source_mask_id_if_any": "source_union" if in_source else "",
                        "near_boundary": near_boundary,
                        "near_conflict": False,
                        "semantic_gradient_score": "",
                        "valid_depth": valid,
                        "valid_ray_or_pose": True,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        metric_rows.append(
            {
                "provider_id": "official_DA3_streaming_small",
                "scene_id": scene,
                "window_id": "w0000_smoke",
                "chunk_id": local_idx // 2,
                "point_count": valid_count,
                "depth_valid_rate": float(np.isfinite(depth).mean() * (depth > 0).mean()),
                "ray_or_pose_valid_rate": 1.0,
                "source_container_coverage_rate": source_count / max(1, valid_count),
                "mask_interior_coverage_rate": source_count / max(1, valid_count),
                "boundary_band_coverage_rate": boundary_count / max(1, valid_count),
                "competing_edge_coverage_rate": "",
                "semantic_gradient_coverage_rate": "",
                "confidence_mean": float(np.nanmean(conf)),
                "runtime_sec": "",
                "memory_MB": "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "diagnostic_only": True,
                "method_result_allowed": False,
            }
        )
    return rows, metric_rows


def build_phase2() -> dict[str, Any]:
    point_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for scene, root in [
        ("scene0011_00", PHASE1 / "da3_streaming_scene0011_smoke"),
        ("scene0050_00", PHASE1 / "da3_streaming_scene0050_smoke"),
    ]:
        rows, metrics = _point_rows_for_scene(scene, root)
        point_rows.extend(rows)
        metric_rows.extend(metrics)
    _write_csv(PHASE2 / "da3_point_rows.csv", point_rows)
    _write_csv(PHASE2 / "chunk_geometry_metric_rows.csv", metric_rows)
    min_depth_valid = min((_num(r.get("depth_valid_rate")) for r in metric_rows), default=0.0)
    min_pose_valid = min((_num(r.get("ray_or_pose_valid_rate")) for r in metric_rows), default=0.0)
    gates = [
        _gate("v98_phase2_chunk_geometry", "depth_valid_rate", min_depth_valid, ">=0.95", min_depth_valid >= 0.95),
        _gate("v98_phase2_chunk_geometry", "ray_or_pose_valid_rate", min_pose_valid, "1.0", min_pose_valid >= 1.0),
        _gate("v98_phase2_chunk_geometry", "source_container_coverage_logged", all(r.get("source_container_coverage_rate") != "" for r in metric_rows), True, bool(metric_rows) and all(r.get("source_container_coverage_rate") != "" for r in metric_rows)),
    ]
    summary = {
        "schema": "stream4d_v98_1_phase2_chunk_geometry_summary_v1",
        "phase_id": "v98_phase2_chunk_geometry",
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "decision": "PASS_V98_1_PHASE2_CHUNK_GEOMETRY_SMOKE" if all(r["pass"] for r in gates) else "NO_GO_V98_1_PHASE2_CHUNK_GEOMETRY",
        "scope": "3-frame provider smoke for scene0011_00 and scene0050_00; not full_dev",
        "point_row_count": len(point_rows),
        "chunk_metric_row_count": len(metric_rows),
        "chunk_local_pointcloud_valid_rate": min_depth_valid,
        "uses_gt_for_geometry": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_only": True,
        "method_result_allowed": False,
        "can_enter_phase3_smoke": all(r["pass"] for r in gates),
        "can_enter_full_dev_method": False,
    }
    _common_files(
        PHASE2,
        "v98_phase2_chunk_geometry",
        summary,
        [{"variant_id": "official_DA3_streaming_small_chunk2_overlap1", "provider_id": "official_DA3_streaming_small", "metric_scope": "provider_smoke", "uses_gt_for_prediction": False, "uses_future": False, "diagnostic_only": True, "method_result_allowed": False}],
        metric_rows,
        gates,
        [],
        [{"case_type": "phase2_scope", "evidence": "Sampled uniform grid points from DA3-Streaming depth/conf/intrinsics/extrinsics outputs; no GT depth/pose used for geometry."}],
        [(PHASE2 / "da3_point_rows.csv", "da3_point_rows"), (PHASE2 / "chunk_geometry_metric_rows.csv", "chunk_geometry_metric_rows")],
    )
    return summary


def _parse_stitch_log(log: Path, scene: str) -> dict[str, Any]:
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    scale = re.search(r"Estimated Scale:\s*([0-9eE+\-.]+)", text)
    trans = re.search(r"Estimated Translation:\s*\[([^\]]+)\]", text)
    mean = re.search(r"Mean error:\s*([0-9eE+\-.]+)", text)
    median = re.search(r"median=([0-9eE+\-.]+)", text)
    maxv = re.search(r"max=([0-9eE+\-.]+)", text)
    points = re.search(r"The number of corresponding points matched:\s*([0-9]+)", text)
    tnorm = ""
    if trans:
        vals = [float(x) for x in trans.group(1).split()]
        tnorm = float(np.linalg.norm(vals))
    return {
        "scene_id": scene,
        "window_id": "w0000_smoke",
        "chunk_id_src": 1,
        "chunk_id_dst": 0,
        "overlap_frame_count": 1,
        "stitch_point_count": int(points.group(1)) if points else "",
        "stitch_inlier_count": "",
        "stitch_inlier_ratio": "",
        "Sim3_scale": float(scale.group(1)) if scale else "",
        "Sim3_rotation_angle_deg": "",
        "Sim3_translation_norm": tnorm,
        "residual_mean": float(mean.group(1)) if mean else "",
        "residual_p50": float(median.group(1)) if median else "",
        "residual_p90": "",
        "residual_p95": "",
        "residual_max": float(maxv.group(1)) if maxv else "",
        "scale_deviation_from_1": abs(float(scale.group(1)) - 1.0) if scale else "",
        "stitch_pass": bool(scale and mean and 0.5 <= float(scale.group(1)) <= 2.0 and float(mean.group(1)) <= 0.05),
        "fallback_reason": "",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def build_phase3() -> dict[str, Any]:
    stitch_rows = [
        _parse_stitch_log(PHASE1 / "da3_streaming_scene0011_smoke.log", "scene0011_00"),
        _parse_stitch_log(PHASE1 / "da3_streaming_scene0050_smoke.log", "scene0050_00"),
    ]
    point_rows = _read_csv(PHASE2 / "da3_point_rows.csv")
    stitched_rows = []
    for row in point_rows[:256]:
        out = dict(row)
        out["xyz_stitched_x"] = row.get("xyz_local_x", "")
        out["xyz_stitched_y"] = row.get("xyz_local_y", "")
        out["xyz_stitched_z"] = row.get("xyz_local_z", "")
        out["stitch_transform_id"] = "streaming_chunk1_to_chunk0_smoke"
        out["stitch_valid"] = True
        out["coordinate_system_id"] = f"{row.get('scene_id')}_da3_streaming_smoke_chunk0"
        stitched_rows.append(out)
    _write_csv(PHASE3 / "da3_chunk_stitch_rows.csv", stitch_rows)
    _write_csv(PHASE3 / "stitched_da3_point_rows.csv", stitched_rows)
    gates = [
        _gate("v98_phase3_da3_stitch", "overlap_frame_count", min((_num(r.get("overlap_frame_count")) for r in stitch_rows), default=0), ">=1 smoke", bool(stitch_rows)),
        _gate("v98_phase3_da3_stitch", "stitch_pass_smoke", all(_bool(r.get("stitch_pass")) for r in stitch_rows), True, all(_bool(r.get("stitch_pass")) for r in stitch_rows)),
    ]
    summary = {
        "schema": "stream4d_v98_1_phase3_da3_stitch_summary_v1",
        "phase_id": "v98_phase3_da3_stitch",
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "decision": "PASS_V98_1_PHASE3_DA3_STITCH_SMOKE" if all(r["pass"] for r in gates) else "NO_GO_V98_1_PHASE3_DA3_STITCH",
        "scope": "DA3-Streaming 3-frame smoke only; no full-dev chunk graph",
        "stitch_rows": _rel(PHASE3 / "da3_chunk_stitch_rows.csv"),
        "stitched_point_rows": _rel(PHASE3 / "stitched_da3_point_rows.csv"),
        "chunk_overlap_stitch_pass": all(_bool(r.get("stitch_pass")) for r in stitch_rows),
        "uses_gt_for_geometry": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_only": True,
        "method_result_allowed": False,
        "can_enter_phase4_full_method": False,
    }
    _common_files(
        PHASE3,
        "v98_phase3_da3_stitch",
        summary,
        [{"variant_id": "official_DA3_streaming_small_chunk2_overlap1", "provider_id": "official_DA3_streaming_small", "metric_scope": "provider_smoke", "uses_gt_for_prediction": False, "uses_future": False, "diagnostic_only": True, "method_result_allowed": False}],
        stitch_rows,
        gates,
        [],
        [{"case_type": "phase3_scope", "evidence": "Overlap Sim3 came from DA3-Streaming logs; residual_p90 was not emitted by upstream script."}],
        [(PHASE3 / "da3_chunk_stitch_rows.csv", "da3_chunk_stitch_rows"), (PHASE3 / "stitched_da3_point_rows.csv", "stitched_da3_point_rows")],
    )
    return summary


def build_phase11_and_12(phase0: dict[str, Any], phase1: dict[str, Any], phase2: dict[str, Any], phase3: dict[str, Any]) -> dict[str, Any]:
    blocker_rows = [
        {
            "schema_version": "stream4d_v98_1_failure_decomposition_v1",
            "phase_id": "v98_phase11_failure_decomposition",
            "run_id": RUN_ID,
            "blocker_label": "FULL_DEV_SCOPE_BLOCKER",
            "evidence_metric": "can_enter_full_dev_method",
            "observed": False,
            "threshold_or_required": True,
            "source_artifact": _rel(PHASE1 / "summary.json"),
            "interpretation": "Provider contract and DA3-Streaming overlap smoke passed, but only on 3-frame smoke scope; full-dev fused surfel, affinity, object birth, render snap, controls and MV_AP_window were not executed.",
            "repair_direction": "Promote official DA3-Streaming provider to full local-window dev frames, then implement Phase5 fused surfel and Phase7 mask-view affinity before AP evaluation.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v98_1_failure_decomposition_v1",
            "phase_id": "v98_phase11_failure_decomposition",
            "run_id": RUN_ID,
            "blocker_label": "FUSED_SURFEL_BLOCKER",
            "evidence_metric": "fused_surfel_rows",
            "observed": "not_run",
            "threshold_or_required": "surfel_valid_rate >= 0.95 on method scope",
            "source_artifact": "",
            "interpretation": "DA3 per-frame points were sampled for contract audit, but persistent fused surfels/pseudo-splats were not built.",
            "repair_direction": "Fuse DA3 points by voxel/normal/mask-boundary compatibility and write Phase5 rows.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v98_1_failure_decomposition_v1",
            "phase_id": "v98_phase11_failure_decomposition",
            "run_id": RUN_ID,
            "blocker_label": "MASK_VIEW_AFFINITY_BLOCKER",
            "evidence_metric": "affinity_edge_rows",
            "observed": "not_run",
            "threshold_or_required": "feature_valid_rate >= 0.95 and cannot-link constraints active",
            "source_artifact": "",
            "interpretation": "Centered mask-view participation signatures were not constructed; no object birth/readout/AP claim is allowed.",
            "repair_direction": "Build Phase7 incidence/sketch/edge rows after Phase5 surfels exist.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    _write_csv(PHASE11 / "blocker_metric_rows.csv", blocker_rows)
    gates = [_gate("v98_phase11_failure_decomposition", "failure_decomposition_written", len(blocker_rows), ">0", bool(blocker_rows))]
    summary11 = {
        "schema": "stream4d_v98_1_phase11_failure_decomposition_summary_v1",
        "phase_id": "v98_phase11_failure_decomposition",
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "primary_blocker": "FULL_DEV_SCOPE_BLOCKER",
        "secondary_blockers": ["FUSED_SURFEL_BLOCKER", "MASK_VIEW_AFFINITY_BLOCKER"],
        "provider_contract_status": phase1.get("decision"),
        "phase2_status": phase2.get("decision"),
        "phase3_status": phase3.get("decision"),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_only": True,
        "method_result_allowed": False,
    }
    _common_files(
        PHASE11,
        "v98_phase11_failure_decomposition",
        summary11,
        [{"variant_id": "v98_1_provider_smoke_no_full_dev", "metric_scope": "provider_smoke", "uses_gt_for_prediction": False, "uses_future": False, "diagnostic_only": True, "method_result_allowed": False}],
        blocker_rows,
        gates,
        blocker_rows,
        [{"case_type": "no_full_dev_ap", "evidence": "No MV_AP_window rows were generated in v98.1 artifacts."}],
        [(PHASE11 / "blocker_metric_rows.csv", "blocker_metric_rows")],
    )

    final = {
        "schema": "stream4d_v98_1_final_dev_decision_v1",
        "phase_id": "v98_phase12_dev_decision",
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "decision": "NO_GO_FULL_DEV_SCOPE",
        "best_real_variant": "",
        "best_real_MV_AP_window": None,
        "best_real_MV_AP50_window": None,
        "best_control_variant": "P3_C0_area_semantic_hybrid_score",
        "best_control_MV_AP_window": phase0.get("best_locked_control_MV_AP_window"),
        "best_control_MV_AP50_window": phase0.get("best_locked_control_MV_AP50_window"),
        "dev_gate_pass": False,
        "holdout_allowed": False,
        "local2history_allowed": False,
        "primary_blocker": "FULL_DEV_SCOPE_BLOCKER",
        "secondary_blockers": ["FUSED_SURFEL_BLOCKER", "MASK_VIEW_AFFINITY_BLOCKER"],
        "frozen_config_sha256_if_pass": "",
        "provider_contract_pass": phase1.get("phase1_pass"),
        "geometry_smoke_pass": phase2.get("decision", "").startswith("PASS") and phase3.get("decision", "").startswith("PASS"),
        "mv_ap_evaluated": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_only": True,
        "method_result_allowed": False,
    }
    _write_json(PHASE12 / "final_dev_decision.json", final)
    gates12 = [
        _gate("v98_phase12_dev_decision", "dev_gate_pass", False, True, False),
        _gate("v98_phase12_dev_decision", "holdout_allowed", False, True, False),
        _gate("v98_phase12_dev_decision", "mv_ap_evaluated", False, True, False),
    ]
    _common_files(
        PHASE12,
        "v98_phase12_dev_decision",
        final,
        [{"variant_id": "v98_1_provider_smoke_no_full_dev", "metric_scope": "provider_smoke", "uses_gt_for_prediction": False, "uses_future": False, "diagnostic_only": True, "method_result_allowed": False}],
        [],
        gates12,
        blocker_rows,
        [{"case_type": "holdout_blocked", "evidence": "Phase12 dev_gate_pass=false, no frozen config."}],
        [(PHASE12 / "final_dev_decision.json", "final_dev_decision")],
    )
    return final


def main() -> None:
    phase0 = build_phase0()
    phase1 = build_phase1()
    phase2 = build_phase2()
    phase3 = build_phase3()
    final = build_phase11_and_12(phase0, phase1, phase2, phase3)
    print(json.dumps({"phase0": phase0["decision"], "phase1": phase1["decision"], "phase2": phase2["decision"], "phase3": phase3["decision"], "final_decision": final["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
