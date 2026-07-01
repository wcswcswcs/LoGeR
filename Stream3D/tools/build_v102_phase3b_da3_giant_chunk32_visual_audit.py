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
PHASE2B_DIR = AUDIT_ROOT / "v102_phase2b_da3_giant_chunk32_audit"
OUT_DIR = AUDIT_ROOT / "v102_phase3b_da3_giant_chunk32_visual_audit"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"


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


def _load_chunk_paths() -> tuple[Path, Path, dict[str, Any]]:
    summary = _read_json(PHASE2B_DIR / "summary.json")
    if not bool(summary.get("chunk32_success")):
        raise RuntimeError("Chunk-32 export did not pass; refusing visual audit.")
    ply = ROOT / str(summary["best_chunk32_ply_file"])
    mini_npz = ROOT / str(summary["best_chunk32_mini_npz_file"])
    if not ply.exists():
        raise FileNotFoundError(ply)
    if not mini_npz.exists():
        raise FileNotFoundError(mini_npz)
    return ply, mini_npz, summary


def _prop(vertex_data: np.ndarray, name: str) -> np.ndarray | None:
    if vertex_data.dtype.names and name in vertex_data.dtype.names:
        return np.asarray(vertex_data[name], dtype=np.float64)
    return None


def _sample_indices(count: int, max_points: int = 120000) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    rng = np.random.default_rng(10232)
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


def _save_projection(xyz: np.ndarray, color_value: np.ndarray, axes: tuple[int, int], path: Path) -> dict[str, Any]:
    idx = _sample_indices(len(xyz))
    pts = xyz[idx]
    colors = _normalize(color_value[idx])
    fig = plt.figure(figsize=(9, 9), dpi=160)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.scatter(
        pts[:, axes[0]],
        pts[:, axes[1]],
        c=colors,
        cmap="viridis",
        s=0.12,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    image = plt.imread(path)
    nonblack = np.any(image[..., :3] > 0.02, axis=2)
    return {
        "path": _rel(path),
        "sampled_points": int(len(idx)),
        "pixel_nonblack_rate": float(np.mean(nonblack)),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }


def _save_overlay(snapshot_paths: list[Path], opacity: np.ndarray | None, path: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=140)
    fig.patch.set_facecolor("black")
    for ax, snap in zip(axes.flat[:3], snapshot_paths):
        ax.imshow(plt.imread(snap))
        ax.axis("off")
    ax = axes.flat[3]
    ax.set_facecolor("black")
    if opacity is not None and opacity.size:
        finite = opacity[np.isfinite(opacity)]
        ax.hist(finite, bins=100, color="#7ad7ff", alpha=0.9)
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


def _homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = extrinsic.astype(np.float64)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")


def _reprojection_rows(xyz: np.ndarray, mini_npz: Path) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    with np.load(mini_npz) as data:
        keys = sorted(data.files)
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        depth = np.asarray(data["depth"])
    n = min(len(extrinsics), len(intrinsics), len(depth))
    points_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    visible_any = np.zeros(len(xyz), dtype=bool)
    rows: list[dict[str, Any]] = []
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
                "schema_version": "stream4d_v102_phase3b_chunk_reprojection_row_v1",
                "phase_id": "v102_phase3b_da3_giant_chunk32_visual_audit",
                "camera_index": i,
                "image_height": h,
                "image_width": w,
                "gaussian_count": int(len(xyz)),
                "projected_inside_count": int(np.sum(inside)),
                "projected_inside_rate": float(np.mean(inside)),
                "positive_depth_rate": float(np.mean(valid_z)),
            }
        )
    meta = {
        "mini_npz_keys": "|".join(keys),
        "depth_shape": list(depth.shape),
        "extrinsics_shape": list(extrinsics.shape),
        "intrinsics_shape": list(intrinsics.shape),
    }
    return float(np.mean(visible_any)), rows, meta


def _write_readme(path: Path, scene_ply: Path) -> None:
    text = f"""# Stream4D v102 DA3-GIANT-1.1 Chunk-32 3DGS Viewer README

This is the chunk-level 3DGS result, not the earlier two-frame smoke.

Audited PLY:

```text
{_rel(scene_ply)}
```

Primary viewer:

1. Open SuperSplat Editor or PlayCanvas Model Viewer.
2. Drag in `scene_chunk32_3dgs.ply`.
3. Inspect scale collapse, floating/ghosting, object boundary readability, and whether the chunk looks coherent across views.

Static previews:

```text
{_rel(OUT_DIR / "chunk32_snapshot_front.png")}
{_rel(OUT_DIR / "chunk32_snapshot_top.png")}
{_rel(OUT_DIR / "chunk32_snapshot_side.png")}
{_rel(OUT_DIR / "chunk32_snapshot_overlay.png")}
```

Boundary:

```text
This artifact confirms a real 32-frame DA3-GIANT-1.1 3DGS export.
It does not by itself prove Phase5 bridge pass or AP improvement.
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_ply, mini_npz, phase2b = _load_chunk_paths()
    scene_ply = OUT_DIR / "scene_chunk32_3dgs.ply"
    shutil.copy2(src_ply, scene_ply)

    ply = PlyData.read(str(scene_ply))
    vertex = ply["vertex"].data
    xyz = np.column_stack([_prop(vertex, "x"), _prop(vertex, "y"), _prop(vertex, "z")]).astype(np.float64)
    finite_xyz = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite_xyz]
    opacity = _prop(vertex, "opacity")
    if opacity is not None:
        opacity = opacity[finite_xyz]
    color_value = opacity if opacity is not None else xyz[:, 2]

    bbox_min = np.min(xyz, axis=0)
    bbox_max = np.max(xyz, axis=0)
    bbox_extent = bbox_max - bbox_min
    reproj_rate, reproj_rows, mini_meta = _reprojection_rows(xyz, mini_npz)

    snapshot_specs = [
        ("chunk32_snapshot_front.png", (0, 1)),
        ("chunk32_snapshot_top.png", (0, 2)),
        ("chunk32_snapshot_side.png", (1, 2)),
    ]
    snapshot_rows: list[dict[str, Any]] = []
    snapshot_paths: list[Path] = []
    for name, axes in snapshot_specs:
        path = OUT_DIR / name
        snapshot_paths.append(path)
        row = _save_projection(xyz, color_value, axes, path)
        row.update(
            {
                "schema_version": "stream4d_v102_phase3b_chunk_snapshot_row_v1",
                "phase_id": "v102_phase3b_da3_giant_chunk32_visual_audit",
                "snapshot_name": name,
                "axes": f"{axes[0]},{axes[1]}",
            }
        )
        snapshot_rows.append(row)
    overlay = _save_overlay(snapshot_paths, opacity, OUT_DIR / "chunk32_snapshot_overlay.png")
    overlay.update(
        {
            "schema_version": "stream4d_v102_phase3b_chunk_snapshot_row_v1",
            "phase_id": "v102_phase3b_da3_giant_chunk32_visual_audit",
            "snapshot_name": "chunk32_snapshot_overlay.png",
            "axes": "front/top/side/opacity_hist",
        }
    )
    snapshot_rows.append(overlay)

    nonblank_snapshot_count = sum(1 for row in snapshot_rows if float(row.get("pixel_nonblack_rate", 0.0)) > 0.0005)
    quality_rows = [
        {
            "schema_version": "stream4d_v102_phase3b_chunk_quality_row_v1",
            "phase_id": "v102_phase3b_da3_giant_chunk32_visual_audit",
            "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
            "source_attempt_id": phase2b.get("best_chunk32_attempt_id"),
            "frame_count": phase2b.get("best_chunk32_frame_count"),
            "process_res": phase2b.get("best_chunk32_process_res"),
            "ply_path": _rel(scene_ply),
            "mini_npz_path": _rel(mini_npz),
            "mini_npz_keys": mini_meta["mini_npz_keys"],
            "depth_shape": mini_meta["depth_shape"],
            "extrinsics_shape": mini_meta["extrinsics_shape"],
            "intrinsics_shape": mini_meta["intrinsics_shape"],
            "gaussian_count": int(len(xyz)),
            "ply_file_size_MB": float(scene_ply.stat().st_size / (1024 * 1024)),
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
            "opacity_nonzero_rate": float(np.mean(np.abs(opacity) > 1e-8)) if opacity is not None else "",
            "opacity_mean": float(np.mean(opacity[np.isfinite(opacity)])) if opacity is not None else "",
            "reprojection_valid_rate": reproj_rate,
            "snapshot_nonblank_count": nonblank_snapshot_count,
            "visual_quality_manual_label": "not_assessed",
            "uses_gt_for_prediction": False,
        }
    ]
    quality_csv = OUT_DIR / "chunk32_3dgs_provider_quality_rows.csv"
    snapshot_csv = OUT_DIR / "chunk32_visualization_manifest_rows.csv"
    reproj_csv = OUT_DIR / "chunk32_reprojection_diagnostic_rows.csv"
    _write_csv(quality_csv, quality_rows)
    _write_csv(snapshot_csv, snapshot_rows)
    _write_csv(reproj_csv, reproj_rows)
    _write_readme(OUT_DIR / "viewer_README.md", scene_ply)

    visual_artifact_pass = bool(scene_ply.exists() and len(xyz) > 0 and reproj_rate >= 0.80 and nonblank_snapshot_count >= 4)
    summary = {
        "schema_version": "stream4d_v102_phase3b_da3_giant_chunk32_visual_audit_summary_v1",
        "phase_id": "v102_phase3b_da3_giant_chunk32_visual_audit",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_CHUNK32_3DGS_VISUAL_ARTIFACT" if visual_artifact_pass else "NO_GO_CHUNK32_3DGS_VISUAL_ARTIFACT",
        "chunk32_visual_artifact_pass": visual_artifact_pass,
        "frame_count": phase2b.get("best_chunk32_frame_count"),
        "process_res": phase2b.get("best_chunk32_process_res"),
        "gaussian_count": int(len(xyz)),
        "ply_file_size_MB": float(scene_ply.stat().st_size / (1024 * 1024)),
        "reprojection_valid_rate": reproj_rate,
        "snapshot_nonblank_count": nonblank_snapshot_count,
        "visual_quality_manual_label": "not_assessed",
        "truthfulness_note": "This is a real 32-frame chunk visual artifact. Manual visual quality and AP/bridge success are not claimed.",
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "scene_chunk32_3dgs": _rel(scene_ply),
            "viewer_README": _rel(OUT_DIR / "viewer_README.md"),
            "snapshot_front": _rel(OUT_DIR / "chunk32_snapshot_front.png"),
            "snapshot_top": _rel(OUT_DIR / "chunk32_snapshot_top.png"),
            "snapshot_side": _rel(OUT_DIR / "chunk32_snapshot_side.png"),
            "snapshot_overlay": _rel(OUT_DIR / "chunk32_snapshot_overlay.png"),
            "quality_rows": _rel(quality_csv),
            "visualization_manifest_rows": _rel(snapshot_csv),
            "reprojection_diagnostic_rows": _rel(reproj_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if visual_artifact_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
