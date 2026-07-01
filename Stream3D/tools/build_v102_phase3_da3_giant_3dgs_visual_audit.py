from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
PHASE2_DIR = AUDIT_ROOT / "v102_phase2_provider_ladder_audit"
OUT_DIR = AUDIT_ROOT / "v102_phase3_da3_giant_3dgs_visual_audit"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"

SRC_PLY = PHASE2_DIR / "official_da3_giant_smoke2_gs_ply_only" / "gs_ply" / "0000.ply"
MINI_NPZ = PHASE2_DIR / "official_da3_giant_smoke2_gs_ply_only" / "exports" / "mini_npz" / "results.npz"
SCENE_PLY = OUT_DIR / "scene_chunk_3dgs.ply"


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


def _file_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024)) if path.exists() else 0.0


def _prop(vertex_data: np.ndarray, name: str) -> np.ndarray | None:
    if vertex_data.dtype.names and name in vertex_data.dtype.names:
        return np.asarray(vertex_data[name], dtype=np.float64)
    return None


def _safe_stat(values: np.ndarray | None, fn: str) -> float | str:
    if values is None or values.size == 0:
        return ""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    if fn == "mean":
        return float(np.mean(finite))
    if fn == "p50":
        return float(np.quantile(finite, 0.50))
    if fn == "p90":
        return float(np.quantile(finite, 0.90))
    if fn == "min":
        return float(np.min(finite))
    if fn == "max":
        return float(np.max(finite))
    raise ValueError(fn)


def _sample_indices(count: int, max_points: int = 50000) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    rng = np.random.default_rng(102)
    return np.sort(rng.choice(count, size=max_points, replace=False))


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo = float(np.quantile(finite, 0.02))
    hi = float(np.quantile(finite, 0.98))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _save_projection(
    xyz: np.ndarray,
    color_value: np.ndarray,
    axes: tuple[int, int],
    path: Path,
) -> dict[str, Any]:
    idx = _sample_indices(len(xyz))
    pts = xyz[idx]
    colors = _normalize(color_value[idx])
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.scatter(
        pts[:, axes[0]],
        pts[:, axes[1]],
        c=colors,
        cmap="viridis",
        s=0.25,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    image = plt.imread(path)
    rgb = image[..., :3]
    nonblack = np.any(rgb > 0.02, axis=2)
    return {
        "path": _rel(path),
        "sampled_points": int(len(idx)),
        "pixel_nonblack_rate": float(np.mean(nonblack)),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }


def _save_overlay(snapshot_paths: list[Path], opacity: np.ndarray | None, path: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=140)
    fig.patch.set_facecolor("black")
    for ax, snap in zip(axes.flat[:3], snapshot_paths):
        ax.imshow(plt.imread(snap))
        ax.axis("off")
    ax = axes.flat[3]
    ax.set_facecolor("black")
    if opacity is not None and opacity.size:
        finite = opacity[np.isfinite(opacity)]
        ax.hist(finite, bins=80, color="#7ad7ff", alpha=0.9)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("white")
    fig.tight_layout(pad=0.0)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    image = plt.imread(path)
    nonblack = np.any(image[..., :3] > 0.02, axis=2)
    return {
        "path": _rel(path),
        "pixel_nonblack_rate": float(np.mean(nonblack)),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }


def _write_viewer_readme(path: Path) -> None:
    text = f"""# Stream4D v102 DA3-GIANT-1.1 3DGS Viewer README

Source PLY:

```text
{_rel(SRC_PLY)}
```

Audited copy:

```text
{_rel(SCENE_PLY)}
```

Manual viewer workflow:

1. Open PlayCanvas Model Viewer or SuperSplat Editor.
2. Drag in `scene_chunk_3dgs.ply`.
3. Inspect front/top/side camera angles for scale collapse, large floating clouds, severe ghosting, and object boundary visibility.
4. Record any manual label separately. This v102 artifact builder does not fabricate `visual_quality_manual_label`.

Generated reproducible snapshots:

```text
{_rel(OUT_DIR / "snapshot_front.png")}
{_rel(OUT_DIR / "snapshot_top.png")}
{_rel(OUT_DIR / "snapshot_side.png")}
{_rel(OUT_DIR / "snapshot_overlay.png")}
```

Boundary:

```text
These snapshots prove the exported 3DGS PLY is parseable and nonblank.
They do not prove mask-pair bridge recall, false-bridge rate, surfel/gaussian purity, or AP improvement.
Phase5 bridge gates remain required before any Phase6 F2 repair.
```
"""
    path.write_text(text, encoding="utf-8")


def _homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = extrinsic.astype(np.float64)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")


def _reprojection_diagnostic(xyz: np.ndarray, mini_npz: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {
        "mini_npz_path": _rel(mini_npz) if mini_npz.exists() else "",
        "mini_npz_exists": mini_npz.exists(),
        "mini_npz_keys": "",
        "camera_pose_count": "",
        "reprojection_valid_rate": "",
        "reprojection_note": "mini_npz sidecar unavailable; no camera projection diagnostic.",
    }
    rows: list[dict[str, Any]] = []
    if not mini_npz.exists():
        return metrics, rows

    with np.load(mini_npz) as data:
        keys = sorted(data.files)
        metrics["mini_npz_keys"] = "|".join(keys)
        if "extrinsics" not in keys or "intrinsics" not in keys or "depth" not in keys:
            metrics["reprojection_note"] = "mini_npz exists but lacks depth/extrinsics/intrinsics."
            return metrics, rows
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        depth = np.asarray(data["depth"])

    if extrinsics.ndim != 3 or intrinsics.ndim != 3 or depth.ndim != 3:
        metrics["reprojection_note"] = (
            f"Unexpected shapes: extrinsics={extrinsics.shape}, intrinsics={intrinsics.shape}, depth={depth.shape}."
        )
        return metrics, rows

    n = min(len(extrinsics), len(intrinsics), len(depth))
    metrics["camera_pose_count"] = int(n)
    if n == 0:
        metrics["reprojection_note"] = "No cameras in mini_npz."
        return metrics, rows

    points_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    visible_any = np.zeros(len(xyz), dtype=bool)
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
        visible_any |= inside
        rows.append(
            {
                "schema_version": "stream4d_v102_phase3_reprojection_row_v1",
                "phase_id": "v102_phase3_da3_giant_3dgs_visual_audit",
                "camera_index": i,
                "image_height": h,
                "image_width": w,
                "gaussian_count": int(len(xyz)),
                "projected_inside_count": int(np.sum(inside)),
                "projected_inside_rate": float(np.mean(inside)),
                "positive_depth_rate": float(np.mean(valid_z)),
            }
        )

    metrics["reprojection_valid_rate"] = float(np.mean(visible_any))
    metrics["reprojection_note"] = "Projection uses DA3 mini_npz extrinsics/intrinsics and PLY xyz; no mask support is inferred."
    return metrics, rows


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase2 = _read_json(PHASE2_DIR / "summary.json")
    if not bool(phase2.get("phase2_pass_for_3dgs_promotion")):
        raise RuntimeError("Phase2 did not produce a promoted 3DGS smoke artifact.")
    if not SRC_PLY.exists():
        raise FileNotFoundError(SRC_PLY)

    shutil.copy2(SRC_PLY, SCENE_PLY)
    ply = PlyData.read(str(SCENE_PLY))
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names or [])
    gaussian_count = int(len(vertex))
    xyz = np.column_stack([_prop(vertex, "x"), _prop(vertex, "y"), _prop(vertex, "z")]).astype(np.float64)
    finite_xyz = np.all(np.isfinite(xyz), axis=1)
    xyz_valid = xyz[finite_xyz]
    if xyz_valid.size == 0:
        raise RuntimeError("No finite xyz coordinates in 3DGS PLY.")

    opacity = _prop(vertex, "opacity")
    scale_props = [_prop(vertex, name) for name in ("scale_0", "scale_1", "scale_2")]
    scale_raw = None
    scale_linear = None
    scale_anisotropy = None
    if all(prop is not None for prop in scale_props):
        scale_raw = np.column_stack(scale_props).astype(np.float64)
        scale_linear = np.exp(np.clip(scale_raw, -20.0, 20.0))
        scale_anisotropy = np.max(scale_linear, axis=1) / np.maximum(np.min(scale_linear, axis=1), 1e-12)

    bbox_min = np.min(xyz_valid, axis=0)
    bbox_max = np.max(xyz_valid, axis=0)
    bbox_extent = bbox_max - bbox_min
    bbox_extent_ratio = float(np.max(bbox_extent) / max(float(np.min(bbox_extent)), 1e-12))
    reproj_metrics, reproj_rows = _reprojection_diagnostic(xyz_valid, MINI_NPZ)

    color_value = opacity if opacity is not None else xyz[:, 2]
    snapshot_specs = [
        ("snapshot_front.png", (0, 1)),
        ("snapshot_top.png", (0, 2)),
        ("snapshot_side.png", (1, 2)),
    ]
    snapshot_rows = []
    snapshot_paths = []
    for name, axes in snapshot_specs:
        path = OUT_DIR / name
        snapshot_paths.append(path)
        row = _save_projection(xyz_valid, color_value[finite_xyz], axes, path)
        row.update(
            {
                "schema_version": "stream4d_v102_phase3_snapshot_row_v1",
                "phase_id": "v102_phase3_da3_giant_3dgs_visual_audit",
                "snapshot_name": name,
                "axes": f"{axes[0]},{axes[1]}",
            }
        )
        snapshot_rows.append(row)
    overlay_row = _save_overlay(snapshot_paths, opacity, OUT_DIR / "snapshot_overlay.png")
    overlay_row.update(
        {
            "schema_version": "stream4d_v102_phase3_snapshot_row_v1",
            "phase_id": "v102_phase3_da3_giant_3dgs_visual_audit",
            "snapshot_name": "snapshot_overlay.png",
            "axes": "front/top/side/opacity_hist",
        }
    )
    snapshot_rows.append(overlay_row)

    nonblank_snapshot_count = sum(1 for row in snapshot_rows if float(row.get("pixel_nonblack_rate", 0.0)) > 0.0005)
    automated_scale_collapse_flag = bool(
        gaussian_count <= 0
        or float(np.mean(finite_xyz)) < 0.999
        or np.any(bbox_extent <= 1e-9)
        or bbox_extent_ratio > 1e6
    )

    max_scale_linear = np.max(scale_linear, axis=1) if scale_linear is not None else None
    median_max_scale = float(np.median(max_scale_linear[np.isfinite(max_scale_linear)])) if max_scale_linear is not None else 0.0
    large_gaussian_rate_proxy = (
        float(np.mean(max_scale_linear > max(median_max_scale * 10.0, 1e-12)))
        if max_scale_linear is not None and max_scale_linear.size
        else ""
    )
    needle_gaussian_rate = (
        float(np.mean(scale_anisotropy > 20.0)) if scale_anisotropy is not None and scale_anisotropy.size else ""
    )

    quality_row = {
        "schema_version": "stream4d_v102_phase3_3dgs_provider_quality_row_v1",
        "phase_id": "v102_phase3_da3_giant_3dgs_visual_audit",
        "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
        "source_phase2_decision": phase2.get("decision"),
        "input_frame_count": 2,
        "export_format": "gs_ply_only",
        "ply_path": _rel(SCENE_PLY),
        "mini_npz_path": reproj_metrics.get("mini_npz_path", ""),
        "mini_npz_keys": reproj_metrics.get("mini_npz_keys", ""),
        "ply_file_size_MB": _file_size_mb(SCENE_PLY),
        "ply_vertex_properties": "|".join(names),
        "gaussian_count": gaussian_count,
        "finite_xyz_rate": float(np.mean(finite_xyz)),
        "bbox_min_x": float(bbox_min[0]),
        "bbox_min_y": float(bbox_min[1]),
        "bbox_min_z": float(bbox_min[2]),
        "bbox_max_x": float(bbox_max[0]),
        "bbox_max_y": float(bbox_max[1]),
        "bbox_max_z": float(bbox_max[2]),
        "bbox_extent_x": float(bbox_extent[0]),
        "bbox_extent_y": float(bbox_extent[1]),
        "bbox_extent_z": float(bbox_extent[2]),
        "bbox_extent_ratio": bbox_extent_ratio,
        "opacity_nonzero_rate": float(np.mean(np.abs(opacity) > 1e-8)) if opacity is not None else "",
        "opacity_mass_total": float(np.sum(opacity[np.isfinite(opacity)])) if opacity is not None else "",
        "opacity_mean": _safe_stat(opacity, "mean"),
        "opacity_p90": _safe_stat(opacity, "p90"),
        "scale_raw_mean": _safe_stat(scale_raw, "mean") if scale_raw is not None else "",
        "scale_raw_p90": _safe_stat(scale_raw, "p90") if scale_raw is not None else "",
        "scale_linear_mean": _safe_stat(scale_linear, "mean") if scale_linear is not None else "",
        "scale_linear_p90": _safe_stat(scale_linear, "p90") if scale_linear is not None else "",
        "scale_anisotropy_p90": _safe_stat(scale_anisotropy, "p90") if scale_anisotropy is not None else "",
        "needle_gaussian_rate": needle_gaussian_rate,
        "large_gaussian_rate_proxy": large_gaussian_rate_proxy,
        "floating_gaussian_rate_proxy": "",
        "camera_pose_count": reproj_metrics.get("camera_pose_count", ""),
        "reprojection_valid_rate": reproj_metrics.get("reprojection_valid_rate", ""),
        "rendered_support_mask_iou_mean": "",
        "mask_support_coverage": "",
        "same_object_bridge_recall": "",
        "same_semantic_diff_GT_false_bridge": "",
        "surfel_or_gaussian_purity_available": False,
        "visual_quality_manual_label": "not_assessed",
        "automated_scale_collapse_flag": automated_scale_collapse_flag,
        "snapshot_nonblank_count": nonblank_snapshot_count,
        "provider_promotion_allowed": False,
        "provider_promotion_blocker": (
            "Smoke-2 3DGS PLY is exported and parseable, but reprojection, mask support, "
            "bridge recall/false-bridge, gaussian purity, and manual visual label are not all available."
        ),
        "reprojection_note": reproj_metrics.get("reprojection_note", ""),
    }
    quality_csv = OUT_DIR / "3dgs_provider_quality_rows.csv"
    snapshot_csv = OUT_DIR / "visualization_manifest_rows.csv"
    reproj_csv = OUT_DIR / "reprojection_diagnostic_rows.csv"
    _write_csv(quality_csv, [quality_row])
    _write_csv(snapshot_csv, snapshot_rows)
    _write_csv(reproj_csv, reproj_rows)
    _write_viewer_readme(OUT_DIR / "viewer_README.md")

    artifact_gate_pass = bool(
        SCENE_PLY.exists()
        and gaussian_count > 0
        and float(np.mean(finite_xyz)) >= 0.999
        and nonblank_snapshot_count >= 4
        and not automated_scale_collapse_flag
    )
    phase3_pass_for_provider_promotion = False
    decision = (
        "PASS_3DGS_ARTIFACT_AND_AUTOSNAPSHOT__BLOCK_PROVIDER_PROMOTION_PENDING_BRIDGE_PURITY"
        if artifact_gate_pass
        else "NO_GO_3DGS_ARTIFACT_OR_AUTOSNAPSHOT_FAIL"
    )
    gate_rows = [
        {
            "gate_id": "export_success",
            "pass": bool(SCENE_PLY.exists()),
            "expected": "scene_chunk_3dgs.ply exists",
            "observed": _rel(SCENE_PLY) if SCENE_PLY.exists() else "",
            "severity": "required",
        },
        {
            "gate_id": "gaussian_count_positive",
            "pass": gaussian_count > 0,
            "expected": ">0",
            "observed": gaussian_count,
            "severity": "required",
        },
        {
            "gate_id": "finite_xyz_rate",
            "pass": float(np.mean(finite_xyz)) >= 0.999,
            "expected": ">=0.999",
            "observed": float(np.mean(finite_xyz)),
            "severity": "required",
        },
        {
            "gate_id": "snapshot_nonblank_count",
            "pass": nonblank_snapshot_count >= 4,
            "expected": "front/top/side/overlay nonblank",
            "observed": nonblank_snapshot_count,
            "severity": "required_for_visual_artifact",
        },
        {
            "gate_id": "reprojection_valid_rate",
            "pass": (
                reproj_metrics.get("reprojection_valid_rate") != ""
                and float(reproj_metrics.get("reprojection_valid_rate")) >= 0.80
            ),
            "expected": ">=0.80 for provider promotion",
            "observed": reproj_metrics.get("reprojection_valid_rate", ""),
            "severity": "required_for_provider_promotion",
        },
        {
            "gate_id": "provider_promotion_bridge_metrics_available",
            "pass": False,
            "expected": "same_object_bridge_recall, false_bridge, purity available",
            "observed": "unavailable_in_phase3_smoke2_artifact",
            "severity": "required_for_provider_promotion_and_phase5",
        },
        {
            "gate_id": "visual_quality_manual_label_available",
            "pass": False,
            "expected": "pass/marginal/fail after manual viewer inspection",
            "observed": "not_assessed",
            "severity": "required_for_provider_promotion",
        },
    ]
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    _write_csv(gate_csv, gate_rows)

    summary = {
        "schema_version": "stream4d_v102_phase3_da3_giant_3dgs_visual_audit_summary_v1",
        "phase_id": "v102_phase3_da3_giant_3dgs_visual_audit",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase3_completed": True,
        "phase3_pass_for_visual_artifact": artifact_gate_pass,
        "phase3_pass_for_provider_promotion": phase3_pass_for_provider_promotion,
        "phase3_ready_for_phase4_diagnostic_only": artifact_gate_pass,
        "gaussian_count": gaussian_count,
        "ply_file_size_MB": _file_size_mb(SCENE_PLY),
        "finite_xyz_rate": float(np.mean(finite_xyz)),
        "snapshot_nonblank_count": nonblank_snapshot_count,
        "mini_npz_exists": bool(reproj_metrics.get("mini_npz_exists")),
        "camera_pose_count": reproj_metrics.get("camera_pose_count", ""),
        "reprojection_valid_rate": reproj_metrics.get("reprojection_valid_rate", ""),
        "surfel_or_gaussian_purity_available": False,
        "same_object_bridge_recall_available": False,
        "same_semantic_diff_GT_false_bridge_available": False,
        "visual_quality_manual_label": "not_assessed",
        "truthfulness_note": (
            "This phase confirms parseable/nonblank Smoke-2 3DGS artifacts only. It does not claim "
            "bridge recall, false-bridge rate, gaussian purity, manual visual quality, or AP improvement."
        ),
        "phase2_summary": _rel(PHASE2_DIR / "summary.json"),
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "quality_rows": _rel(quality_csv),
            "visualization_manifest_rows": _rel(snapshot_csv),
            "reprojection_diagnostic_rows": _rel(reproj_csv),
            "viewer_README": _rel(OUT_DIR / "viewer_README.md"),
            "scene_chunk_3dgs": _rel(SCENE_PLY),
            "snapshot_front": _rel(OUT_DIR / "snapshot_front.png"),
            "snapshot_top": _rel(OUT_DIR / "snapshot_top.png"),
            "snapshot_side": _rel(OUT_DIR / "snapshot_side.png"),
            "snapshot_overlay": _rel(OUT_DIR / "snapshot_overlay.png"),
            "variant_gate_rows": _rel(gate_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if artifact_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
