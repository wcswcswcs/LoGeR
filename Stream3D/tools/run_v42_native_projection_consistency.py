from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from tools.run_v26_object_quality_diagnostics import _json_safe


ROOT = Path(__file__).resolve().parents[1]

SCENE_TRANSFORM_KEYS = {
    "scene_xyz",
    "world_xyz",
    "mesh_vertex_id",
    "mesh_vertex_ids",
    "mesh_ids",
    "T_d4rt_to_scene",
    "T_canonical_to_scene",
    "camera_pose",
    "pose",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, float(q))) if values.size else None


def projection_consistency_stats(
    xyz: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    xyz = np.asarray(xyz, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)
    if xyz.shape != uv.shape[:2] + (3,):
        raise ValueError(f"xyz shape {xyz.shape} is incompatible with uv shape {uv.shape}")
    if valid_mask.shape != uv.shape[:2]:
        raise ValueError(f"valid shape {valid_mask.shape} is incompatible with uv shape {uv.shape}")
    finite = valid_mask & np.isfinite(xyz).all(axis=-1) & np.isfinite(uv).all(axis=-1)
    z = xyz[..., 2]
    projectable = finite & (np.abs(z) > 1e-6)
    if not np.any(projectable):
        return {
            "sample_count": 0,
            "projectable_ratio": 0.0,
            "positive_z_ratio": 0.0,
            "projection_error_mean": None,
            "projection_error_p50": None,
            "projection_error_p90": None,
            "projection_error_p99": None,
            "projection_within_0p02": 0.0,
            "projection_within_0p05": 0.0,
            "projection_consistency_gate": False,
        }
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = xyz[..., 0]
    y = xyz[..., 1]
    u = (fx * x / z + cx) / float(max(int(width) - 1, 1))
    v = (fy * y / z + cy) / float(max(int(height) - 1, 1))
    projected = np.stack([u, v], axis=-1)
    err = np.linalg.norm(projected[projectable] - uv[projectable], axis=1)
    finite_err = err[np.isfinite(err)]
    within_0p02 = float(np.mean(finite_err <= 0.02)) if finite_err.size else 0.0
    within_0p05 = float(np.mean(finite_err <= 0.05)) if finite_err.size else 0.0
    p90 = _safe_quantile(finite_err, 0.90)
    return {
        "sample_count": int(finite_err.size),
        "projectable_ratio": float(np.count_nonzero(projectable) / max(int(valid_mask.size), 1)),
        "positive_z_ratio": float(np.mean(z[finite] > 0.0)) if np.any(finite) else 0.0,
        "projection_error_mean": float(np.mean(finite_err)) if finite_err.size else None,
        "projection_error_p50": _safe_quantile(finite_err, 0.50),
        "projection_error_p90": p90,
        "projection_error_p99": _safe_quantile(finite_err, 0.99),
        "projection_within_0p02": within_0p02,
        "projection_within_0p05": within_0p05,
        "projection_consistency_gate": bool(
            p90 is not None
            and float(p90) <= 0.02
            and within_0p02 >= 0.90
        ),
    }


def _audit_scene(
    *,
    cache_root: Path,
    scene: str,
    projection_keys: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    npz_path = cache_root / scene / "carriers_window000.npz"
    manifest_path = cache_root / scene / "carriers_window000_manifest.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    stream = ScanNetStream(scene)
    intrinsics = stream.load_intrinsics()
    depth0 = stream.load_depth(0)
    height, width = depth0.shape
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    available_scene_transform_keys = sorted(keys & SCENE_TRANSFORM_KEYS)
    missing_scene_transform_keys = sorted(SCENE_TRANSFORM_KEYS - keys)
    uv = np.asarray(data["uv_pred"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)
    rows: list[dict[str, Any]] = []
    for key in projection_keys:
        if key not in keys:
            rows.append(
                {
                    "scene": scene,
                    "xyz_key": key,
                    "status": "missing_xyz_key",
                    "projection_consistency_gate": False,
                }
            )
            continue
        stats = projection_consistency_stats(
            np.asarray(data[key], dtype=np.float32),
            uv,
            valid,
            intrinsics,
            width=int(width),
            height=int(height),
        )
        rows.append(
            {
                "scene": scene,
                "xyz_key": key,
                "status": "ok_projection_audit",
                "depth_width": int(width),
                "depth_height": int(height),
                **stats,
                "uses_gt_for_prediction": False,
                "uses_rgbd_for_prediction": False,
                "uses_pose_for_prediction": False,
                "uses_scannet_mesh_for_prediction": False,
            }
        )
    summary = {
        "scene": scene,
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "npz_keys": sorted(keys),
        "available_scene_transform_keys": available_scene_transform_keys,
        "missing_scene_transform_keys": missing_scene_transform_keys,
        "has_scene_coordinate_transform": bool(available_scene_transform_keys),
        "has_mesh_vertex_ids": bool({"mesh_vertex_id", "mesh_vertex_ids", "mesh_ids"} & keys),
        "projection_gate_pass": bool(rows and any(bool(row.get("projection_consistency_gate")) for row in rows)),
        "method_ap_materializer_ready": False,
        "method_ap_materializer_blockers": [
            "missing_scene_coordinate_transform",
            "missing_mesh_vertex_ids",
            "standard_scannet_ap_mesh_masks_need_coordinate_bridge",
        ],
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v42 native D4RT projection consistency and AP materializer readiness.")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v42_semantic_occupancy_real_dino_q5_mf32_b1024/Q5")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--projection-keys", default="xyz_ref,xyz_local")
    parser.add_argument("--output-root", default="outputs/audit/v42_native_projection_consistency_allframe_r1")
    args = parser.parse_args()

    cache_root = ROOT / str(args.cache_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    projection_keys = [item.strip() for item in str(args.projection_keys).split(",") if item.strip()]
    output_root = ROOT / str(args.output_root)
    rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        scene_rows, scene_summary = _audit_scene(
            cache_root=cache_root,
            scene=scene,
            projection_keys=projection_keys,
        )
        rows.extend(scene_rows)
        scene_summaries.append(scene_summary)
    summary = {
        "phase": "v42_native_projection_consistency",
        "status": "NO_GO_NATIVE_PROJECTION_NOT_AP_MATERIALIZER_READY"
        if any(not bool(item["method_ap_materializer_ready"]) for item in scene_summaries)
        else "OK_NATIVE_PROJECTION_AP_MATERIALIZER_READY",
        "cache_root": str(cache_root),
        "scene_summaries": scene_summaries,
        "rows": rows,
        "projection_all_scenes_gate_pass": bool(rows and all(bool(row.get("projection_consistency_gate")) for row in rows)),
        "method_ap_materializer_ready": bool(scene_summaries and all(bool(item["method_ap_materializer_ready"]) for item in scene_summaries)),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "forbidden_for_method_table": False,
        "note": (
            "This audit uses D4RT cache xyz/uv plus ScanNet intrinsics only. It does not produce AP; "
            "it checks whether native fields are projection-consistent and whether cache carries a "
            "method-compatible scene/mesh materialization bridge."
        ),
    }
    _write_json(output_root / "native_projection_consistency_summary.json", summary)
    _write_csv(output_root / "native_projection_consistency_rows.csv", rows)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "status": summary["status"],
                    "projection_all_scenes_gate_pass": summary["projection_all_scenes_gate_pass"],
                    "method_ap_materializer_ready": summary["method_ap_materializer_ready"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
