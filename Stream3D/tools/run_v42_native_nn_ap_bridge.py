from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _parse_radii(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _load_native_points(path: Path, *, scene: str, variant: str, source: str) -> dict[int, np.ndarray]:
    by_object: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in _read_csv(path):
        if str(row.get("scene", "")) != scene:
            continue
        if str(row.get("variant", "")) != variant:
            continue
        if str(row.get("source", "")) != source:
            continue
        by_object[int(row["object_id"])].append((float(row["x"]), float(row["y"]), float(row["z"])))
    return {object_id: np.asarray(points, dtype=np.float32) for object_id, points in by_object.items()}


def _load_scene_points(scene: str) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for native NN AP bridge diagnostics") from exc
    stream = ScanNetStream(seq_name=scene)
    points = np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"failed to load scene mesh points for {scene}: {stream.mesh_path}")
    return points


def _write_prediction(
    *,
    scene: str,
    config: str,
    object_vertices: dict[int, np.ndarray],
    scores: dict[int, float],
    scene_point_count: int,
    native_point_rows: Path,
    radius: float,
) -> dict[str, Any]:
    pred_dir = ROOT / "data/prediction" / f"{config}_class_agnostic"
    tmp_dir = ROOT / "data/TMP" / config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    kept = [(object_id, ids) for object_id, ids in sorted(object_vertices.items()) if np.asarray(ids).size > 0]
    masks = np.zeros((int(scene_point_count), len(kept)), dtype=bool)
    pred_scores = np.zeros((len(kept),), dtype=np.float32)
    for col, (object_id, ids_raw) in enumerate(kept):
        ids = np.asarray(ids_raw, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < int(scene_point_count))]
        masks[ids, col] = True
        pred_scores[col] = float(scores.get(int(object_id), float(ids.shape[0])))
    np.savez_compressed(
        pred_dir / f"{scene}.npz",
        pred_masks=masks,
        pred_score=pred_scores,
        pred_classes=np.zeros((len(kept),), dtype=np.int32),
    )
    pre_points = np.flatnonzero(np.any(masks, axis=1)).astype(np.int64)
    np.save(tmp_dir / f"{scene}_pre_points.npy", pre_points)
    manifest = build_prediction_manifest(
        output_config=config,
        root=ROOT,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(native_point_rows)],
        pre_points_policy="native_d4rt_xyz_nearest_mesh_vertex",
        support_policy="d4rt_native_xyz_nn_radius",
        notes=(
            "v42 native D4RT support point nearest-neighbor AP bridge diagnostic. "
            "No RGB-D/pose is used, but ScanNet mesh vertices are used as AP output canvas."
        ),
        extra={
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "alignment_source": "d4rt_native_xyz_to_mesh_nn",
            "alignment_used_for_prediction": True,
            "phase": "v42_native_nn_ap_bridge",
            "nn_radius": float(radius),
        },
    )
    write_prediction_manifest(config, manifest, root=ROOT, pred_suffix="class_agnostic")
    return {
        "num_predictions": int(len(kept)),
        "num_scene_points": int(scene_point_count),
        "num_exported_points": int(pre_points.shape[0]),
        "mesh_coverage": float(pre_points.shape[0] / max(int(scene_point_count), 1)),
        "prediction_path": str(pred_dir / f"{scene}.npz"),
        "pre_points_path": str(tmp_dir / f"{scene}_pre_points.npy"),
    }


def _run_eval(config: str, output_root: Path) -> dict[str, Any]:
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{config}_class_agnostic.txt"
    log_path = eval_dir / f"{config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(ROOT / "data/prediction" / f"{config}_class_agnostic"),
        "--gt_path",
        str(ROOT / "data/scannet/gt"),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(ROOT / "data/TMP"),
        "--tmp_config",
        config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    metrics = _parse_metric_file(metric_file) if metric_file.exists() else {}
    return {
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "metric_file": str(metric_file),
        "log_path": str(log_path),
        "metrics": metrics,
    }


def _scene_quality(scene: str, config: str, radius: float) -> dict[str, Any]:
    pred_path = ROOT / "data/prediction" / f"{config}_class_agnostic" / f"{scene}.npz"
    pre_path = ROOT / "data/TMP" / config / f"{scene}_pre_points.npy"
    gt_path = ROOT / "data/scannet/gt" / f"{scene}.txt"
    with np.load(pred_path) as pred:
        masks = np.asarray(pred["pred_masks"], dtype=bool)
        scores = np.asarray(pred["pred_score"], dtype=np.float32)
    pre_points = np.load(pre_path).astype(np.int64)
    gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
    stats = _candidate_oracle_stats(masks, scores, gt_eval, pre_points)
    row = _quality_stats(scene, f"native_nn_r{radius:g}", masks, scores, gt_eval, pre_points, stats)
    row["nn_radius"] = float(radius)
    return row


def _aggregate(rows: list[dict[str, Any]], eval_result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"scene_count": int(len(rows))}
    keys = sorted(
        key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.integer, np.floating)) and key != "scene"
    )
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.integer, np.floating))]
        if values:
            out[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    out.update(eval_result.get("metrics", {}))
    out["eval_exit_code"] = int(eval_result.get("exit_code", -1))
    return out


def _finite_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        current = float(value)
    except (TypeError, ValueError):
        return None
    return current if np.isfinite(current) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="v42 native D4RT xyz nearest-neighbor AP bridge diagnostic.")
    parser.add_argument("--native-point-rows", default="outputs/audit/v42_streaming_memory_unioncap320_r1/memory_native_point_rows.csv")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--source", default="dinov2_maskcut")
    parser.add_argument("--radii", default="0.02,0.05,0.10,0.25,0.50")
    parser.add_argument("--output-config-prefix", default="v42_native_nn_memory_r1")
    parser.add_argument("--output-root", default="outputs/audit/v42_native_nn_ap_bridge_r1")
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    radii = _parse_radii(str(args.radii))
    native_point_rows = ROOT / str(args.native_point_rows)
    output_root = ROOT / str(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    matrix: list[dict[str, Any]] = []
    scene_rows_all: list[dict[str, Any]] = []
    evals: dict[str, dict[str, Any]] = {}
    for radius in radii:
        radius_tag = str(f"{float(radius):.3f}").replace(".", "p")
        config = f"{args.output_config_prefix}_r{radius_tag}"
        scene_rows: list[dict[str, Any]] = []
        for scene in scenes:
            scene_points = _load_scene_points(scene)
            tree = cKDTree(scene_points)
            native_by_object = _load_native_points(
                native_point_rows,
                scene=scene,
                variant=str(args.variant),
                source=str(args.source),
            )
            object_vertices: dict[int, np.ndarray] = {}
            scores: dict[int, float] = {}
            query_count = 0
            hit_count = 0
            distance_values: list[float] = []
            for object_id, points in native_by_object.items():
                pts = np.asarray(points, dtype=np.float32)
                query_count += int(pts.shape[0])
                if pts.size == 0:
                    object_vertices[int(object_id)] = np.empty((0,), dtype=np.int64)
                    scores[int(object_id)] = 0.0
                    continue
                dist, idx = tree.query(pts, k=1, distance_upper_bound=float(radius))
                valid = np.isfinite(dist) & (idx < scene_points.shape[0])
                hit_count += int(np.count_nonzero(valid))
                distance_values.extend(float(v) for v in dist[valid].tolist())
                object_vertices[int(object_id)] = np.unique(idx[valid].astype(np.int64))
                scores[int(object_id)] = float(np.count_nonzero(valid))
            write_diag = _write_prediction(
                scene=scene,
                config=config,
                object_vertices=object_vertices,
                scores=scores,
                scene_point_count=int(scene_points.shape[0]),
                native_point_rows=native_point_rows,
                radius=float(radius),
            )
            quality = _scene_quality(scene, config, float(radius))
            row = {
                **quality,
                **write_diag,
                "config": config,
                "scene": scene,
                "native_point_query_count": int(query_count),
                "native_point_hit_count": int(hit_count),
                "native_point_hit_rate": float(hit_count / max(query_count, 1)),
                "nn_distance_mean": float(np.mean(np.asarray(distance_values, dtype=np.float64))) if distance_values else None,
                "nn_distance_p90": float(np.quantile(np.asarray(distance_values, dtype=np.float64), 0.90)) if distance_values else None,
                "uses_gt_for_prediction": False,
                "uses_rgbd_for_prediction": False,
                "uses_pose_for_prediction": False,
                "uses_scannet_mesh_for_prediction": True,
                "forbidden_for_method_table": True,
            }
            scene_rows.append(row)
            scene_rows_all.append(row)
        eval_result = _run_eval(config, output_root)
        evals[config] = eval_result
        agg = _aggregate(scene_rows, eval_result)
        agg.update({"config": config, "nn_radius": float(radius)})
        matrix.append(agg)

    finite_ap_rows = [row for row in matrix if _finite_metric(row, "AP") is not None]
    if finite_ap_rows:
        best = max(finite_ap_rows, key=lambda row: float(row.get("AP") or -1.0))
        status = "OK_NATIVE_NN_DIAGNOSTIC_AP_COMPUTED"
    elif matrix:
        best = max(matrix, key=lambda row: float(row.get("native_point_hit_rate") or -1.0))
        status = "NO_GO_NATIVE_NN_AP_NO_VALID_AP"
    else:
        best = {}
        status = "NO_GO_NATIVE_NN_DIAGNOSTIC_EMPTY"
    summary = {
        "phase": "v42_native_nn_ap_bridge",
        "status": status,
        "native_point_rows": str(native_point_rows),
        "radii": radii,
        "matrix": matrix,
        "scene_rows": scene_rows_all,
        "evals": evals,
        "best_by_AP": best,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": True,
        "forbidden_for_method_table": True,
        "phase8_gate_pass": False,
        "phase8_gate_blocker": "native NN bridge still uses ScanNet mesh as AP canvas and does not solve method-compatible native AP calibration",
    }
    _write_json(output_root / "native_nn_ap_bridge_summary.json", summary)
    _write_csv(output_root / "native_nn_ap_bridge_matrix.csv", matrix)
    _write_csv(output_root / "native_nn_ap_bridge_scene_rows.csv", scene_rows_all)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "status": summary["status"],
                    "best_config": best.get("config"),
                    "best_radius": best.get("nn_radius"),
                    "best_AP": best.get("AP"),
                    "best_AP50": best.get("AP50"),
                    "best_AP25": best.get("AP25"),
                    "best_native_hit_rate": best.get("native_point_hit_rate"),
                    "phase8_gate_pass": summary["phase8_gate_pass"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
