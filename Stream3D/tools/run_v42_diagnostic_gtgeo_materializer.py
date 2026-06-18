from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)
from tools.run_v42_full_factor_graph import _parse_json_list
from tools.run_v42_semantic_part_audit import _npz_source_masks


ROOT = Path(__file__).resolve().parents[1]
CLASS_AGNOSTIC_ID = 3


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


def _parse_csv_json_list(value: Any) -> list[int]:
    return _parse_json_list(value)


def _load_object_rows(path: Path, *, scene: str, variant: str, source: str) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(path):
        if str(row.get("scene", "")) != scene:
            continue
        if str(row.get("variant", "")) != variant:
            continue
        if str(row.get("source", "")) != source:
            continue
        rows.append(
            {
                **row,
                "object_id": int(row["object_id"]),
                "primary_field_id": int(row["primary_field_id"]),
                "semantic_masklet_ids": _parse_csv_json_list(row.get("semantic_masklet_ids", "")),
                "attached_tube_ids": _parse_csv_json_list(row.get("attached_tube_ids", "")),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
            }
        )
    return rows


def _load_token_rows(part_graph_root: Path, *, scene: str, variant: str, source: str) -> dict[int, dict[str, Any]]:
    rows = _read_csv(part_graph_root / variant / scene / "part_token_rows.csv")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("source", "")) != source:
            continue
        token_id = int(row["token_id"])
        out[token_id] = {
            "token_id": token_id,
            "frame_id": int(row["frame_id"]),
            "mask_id": int(row["mask_id"]),
            "area": int(float(row.get("area", 0.0) or 0.0)),
        }
    return out


def _mask_xy(mask: np.ndarray, *, depth_shape: tuple[int, int], stride: int, max_pixels: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != depth_shape:
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (int(depth_shape[1]), int(depth_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    stride = max(1, int(stride))
    if stride > 1:
        keep = ((xs % stride) == 0) & ((ys % stride) == 0)
        xs = xs[keep]
        ys = ys[keep]
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if int(max_pixels) > 0 and ys.size > int(max_pixels):
        keep = np.linspace(0, ys.size - 1, num=int(max_pixels), dtype=np.int64)
        xs = xs[keep]
        ys = ys[keep]
    return np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)


def _load_external_masks(
    *,
    external_root: Path,
    scene: str,
    source: str,
    frame_ids: list[int],
    min_area: int,
) -> dict[tuple[int, int], np.ndarray]:
    masks_by_frame = _npz_source_masks(
        external_root,
        scene,
        source,
        frame_ids,
        min_area,
        sample_count=max(len(frame_ids), 1),
    )
    out: dict[tuple[int, int], np.ndarray] = {}
    for frame_id, masks in masks_by_frame.items():
        for mask_id, mask in masks:
            out[(int(frame_id), int(mask_id))] = np.asarray(mask, dtype=bool)
    return out


def _run_eval(args: argparse.Namespace, config: str, output_root: Path) -> dict[str, Any]:
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
        str(ROOT / args.gt_path),
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


def _export_scene(
    args: argparse.Namespace,
    *,
    scene: str,
    object_rows: list[dict[str, Any]],
    token_by_id: dict[int, dict[str, Any]],
    masks_by_key: dict[tuple[int, int], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stream = ScanNetStream(seq_name=scene)
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=str(args.output_config),
        export_nn_radius=float(args.export_nn_radius),
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.export_min_points_per_object),
        export_score_mode="area",
    )
    depth_shape_by_frame: dict[int, tuple[int, int]] = {}
    object_records: list[dict[str, Any]] = []
    object_dict: dict[int, dict[str, Any]] = {}
    trace_rows: list[dict[str, Any]] = []
    missing_token_count = 0
    missing_mask_count = 0
    backproject_pixel_count = 0
    backproject_hit_count = 0
    mask_observation_count = 0
    for out_idx, row in enumerate(sorted(object_rows, key=lambda item: int(item["object_id"]))):
        object_id = int(row["object_id"])
        point_ids: set[int] = set()
        mask_list: list[tuple[int, int, float]] = []
        token_ids = [int(v) for v in row["semantic_masklet_ids"]]
        for token_id in token_ids:
            token = token_by_id.get(int(token_id))
            if token is None:
                missing_token_count += 1
                continue
            frame_id = int(token["frame_id"])
            mask_id = int(token["mask_id"])
            mask = masks_by_key.get((frame_id, mask_id))
            if mask is None:
                missing_mask_count += 1
                continue
            if frame_id not in depth_shape_by_frame:
                depth_shape_by_frame[frame_id] = exporter.stream.load_depth(frame_id).shape
            xy = _mask_xy(
                mask,
                depth_shape=depth_shape_by_frame[frame_id],
                stride=int(args.export_mask_sample_stride),
                max_pixels=int(args.export_mask_max_pixels),
            )
            if xy.size == 0:
                continue
            hit_ids, _dist = exporter._backproject_xy(frame_id, xy, nn_radius=float(args.export_nn_radius))
            backproject_pixel_count += int(xy.shape[0])
            backproject_hit_count += int(hit_ids.shape[0])
            point_ids.update(int(v) for v in hit_ids.tolist())
            mask_list.append((frame_id, mask_id, float(token.get("area", 0))))
            mask_observation_count += 1
        sorted_points = sorted(point_ids)
        object_dict[object_id] = {
            "point_ids": np.asarray(sorted_points, dtype=np.int64),
            "mask_list": mask_list,
            "repre_mask_list": mask_list[: min(8, len(mask_list))],
            "semantic_masklet_ids": np.asarray(token_ids, dtype=np.int64),
            "attached_tube_ids": np.asarray([int(v) for v in row["attached_tube_ids"]], dtype=np.int64),
            "score": float(len(sorted_points)),
        }
        object_records.append(
            {
                "object_id": object_id,
                "point_ids": set(sorted_points),
                "score": float(len(sorted_points)),
                "area_score": float(len(sorted_points)),
                "observations": float(len(mask_list)),
            }
        )
        trace_rows.append(
            {
                "scene": scene,
                "prediction_index": int(out_idx),
                "object_id": object_id,
                "semantic_masklet_count": int(len(token_ids)),
                "mask_observation_count": int(len(mask_list)),
                "attached_tube_count": int(len(row["attached_tube_ids"])),
                "exported_point_count": int(len(sorted_points)),
                "score": float(len(sorted_points)),
            }
        )
    diag = exporter._write_outputs(object_records, object_dict, np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16))
    manifest = build_prediction_manifest(
        output_config=str(args.output_config),
        root=ROOT,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.memory_object_rows), str(args.part_graph_root), str(args.external_source_root)],
        pre_points_policy="recompute_from_diagnostic_mesh_backprojection",
        support_policy="semantic_masklet_mask_backproject",
        notes=(
            "v42 diagnostic GT/RGB-D geometry materialization from semantic-born memory object fields. "
            "Forbidden for method table because it uses ScanNet RGB-D/pose/mesh backprojection."
        ),
        extra={
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
            "phase": "v42_diagnostic_gtgeo_materializer",
            "object_source": "v42_streaming_memory_object_fields",
        },
    )
    write_prediction_manifest(str(args.output_config), manifest, root=ROOT, pred_suffix="class_agnostic")
    pred_path = ROOT / "data/prediction" / f"{args.output_config}_class_agnostic" / f"{scene}.npz"
    pre_path = ROOT / "data/TMP" / str(args.output_config) / f"{scene}_pre_points.npy"
    gt_path = ROOT / args.gt_path / f"{scene}.txt"
    with np.load(pred_path) as pred:
        masks = np.asarray(pred["pred_masks"], dtype=bool)
        scores = np.asarray(pred["pred_score"], dtype=np.float32)
    pre_points = np.load(pre_path).astype(np.int64)
    gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
    oracle_stats = _candidate_oracle_stats(masks, scores, gt_eval, pre_points)
    quality = _quality_stats(scene, "v42_gtgeo_memory", masks, scores, gt_eval, pre_points, oracle_stats)
    scene_row = {
        **quality,
        **diag,
        "scene": scene,
        "object_field_count": int(len(object_rows)),
        "missing_token_count": int(missing_token_count),
        "missing_mask_count": int(missing_mask_count),
        "mask_observation_count": int(mask_observation_count),
        "backproject_pixel_count": int(backproject_pixel_count),
        "backproject_hit_count": int(backproject_hit_count),
        "backproject_hit_rate": float(backproject_hit_count / max(backproject_pixel_count, 1)),
        "prediction_path": str(pred_path),
        "pre_points_path": str(pre_path),
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "forbidden_for_method_table": True,
    }
    return scene_row, trace_rows


def _aggregate_scene_rows(scene_rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        key
        for row in scene_rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key != "scene"
    )
    out: dict[str, Any] = {"scene_count": int(len(scene_rows))}
    for key in numeric_keys:
        values = [float(row[key]) for row in scene_rows if isinstance(row.get(key), (int, float, np.integer, np.floating))]
        if values:
            out[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="v42 diagnostic GT/RGB-D mesh materializer for memory object fields.")
    parser.add_argument("--memory-object-rows", default="outputs/audit/v42_streaming_memory_unioncap320_r1/memory_object_field_rows.csv")
    parser.add_argument("--part-graph-root", default="outputs/audit/v42_semantic_occupancy_real_q0q5_mf32_b1024_part_graph")
    parser.add_argument("--external-source-root", default="outputs/audit/v42_source_audit_external_stride1_smoke")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--source", default="dinov2_maskcut")
    parser.add_argument("--output-config", default="v42_gtgeo_memory_r1")
    parser.add_argument("--output-root", default="outputs/audit/v42_gtgeo_materializer_memory_r1")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-min-points-per-object", type=int, default=1)
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    memory_object_rows = ROOT / str(args.memory_object_rows)
    part_graph_root = ROOT / str(args.part_graph_root)
    external_root = ROOT / str(args.external_source_root)
    output_root = ROOT / str(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for scene in scenes:
        object_rows = _load_object_rows(
            memory_object_rows,
            scene=scene,
            variant=str(args.variant),
            source=str(args.source),
        )
        token_by_id = _load_token_rows(part_graph_root, scene=scene, variant=str(args.variant), source=str(args.source))
        frame_ids = sorted({int(token_by_id[int(token_id)]["frame_id"]) for row in object_rows for token_id in row["semantic_masklet_ids"] if int(token_id) in token_by_id})
        masks_by_key = _load_external_masks(
            external_root=external_root,
            scene=scene,
            source=str(args.source),
            frame_ids=frame_ids,
            min_area=int(args.min_area),
        )
        scene_row, current_trace = _export_scene(
            args,
            scene=scene,
            object_rows=object_rows,
            token_by_id=token_by_id,
            masks_by_key=masks_by_key,
        )
        scene_rows.append(scene_row)
        trace_rows.extend(current_trace)

    eval_result = _run_eval(args, str(args.output_config), output_root)
    aggregate = _aggregate_scene_rows(scene_rows)
    aggregate.update(eval_result.get("metrics", {}))
    summary = {
        "phase": "v42_diagnostic_gtgeo_materializer",
        "status": "OK_DIAGNOSTIC_GTGEO_AP_COMPUTED" if eval_result["exit_code"] == 0 and eval_result.get("metrics") else "NO_GO_DIAGNOSTIC_GTGEO_AP_FAILED",
        "output_config": str(args.output_config),
        "memory_object_rows": str(memory_object_rows),
        "part_graph_root": str(part_graph_root),
        "external_source_root": str(external_root),
        "scene_rows": scene_rows,
        "aggregate": aggregate,
        "eval": eval_result,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "forbidden_for_method_table": True,
        "phase8_gate_pass": False,
        "phase8_gate_blocker": "diagnostic GT/RGB-D materialization is not method-compatible AP",
    }
    _write_json(output_root / "gtgeo_materializer_summary.json", summary)
    _write_csv(output_root / "gtgeo_materializer_scene_rows.csv", scene_rows)
    _write_csv(output_root / "gtgeo_materializer_trace_rows.csv", trace_rows)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "status": summary["status"],
                    "AP": aggregate.get("AP"),
                    "AP50": aggregate.get("AP50"),
                    "AP25": aggregate.get("AP25"),
                    "mean_predictions": aggregate.get("mean_num_predictions"),
                    "phase8_gate_pass": summary["phase8_gate_pass"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
