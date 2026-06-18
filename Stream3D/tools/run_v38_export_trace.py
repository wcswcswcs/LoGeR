from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.export_scannet import ScanNetExporter, score_export_record
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v37_4d_if_allowed import _build_scene_state, _merge_components_rgb_temporal_topk
from tools.run_v37_ap_if_allowed import _component_from_variant, _component_mask_xy, _gt_coverage_ratio
from tools.run_v37_temporal_curriculum import _load_masks


EXPECTED_RAW_AP = {
    "AP": 0.003937456854837711,
    "AP50": 0.012952410140378828,
    "AP25": 0.1194303308283155,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _parse_metric_file(path: Path) -> dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty metric file: {path}")
    parts = lines[-1].split(",")
    if len(parts) != 3:
        raise ValueError(f"Could not parse final AP row from {path}: {lines[-1]}")
    return {"AP": float(parts[0]), "AP50": float(parts[1]), "AP25": float(parts[2])}


def _sha1_ints(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _component_trace_record(
    state: Any,
    component: list[int],
    object_id: int,
    point_ids: set[int],
    source_regions: list[dict[str, Any]],
    score: float,
) -> dict[str, Any]:
    tube_counter: Counter[int] = Counter()
    frame_ranks: set[int] = set()
    frame_ids: set[int] = set()
    nodes_by_id = {int(node.node_id): node for node in state.nodes}
    for node_id in component:
        node = nodes_by_id[int(node_id)]
        tube_counter.update(state.support_by_region.get(int(node_id), Counter()))
        frame_id = int(node.frame_id)
        frame_ids.add(frame_id)
        frame_ranks.add(int(state.frame_rank.get(frame_id, frame_id)))
    temporal_span = int(max(frame_ranks) - min(frame_ranks) + 1) if frame_ranks else 0
    return {
        "scene": state.scene,
        "object_id": int(object_id),
        "source_object_id": f"{state.scene}:{int(object_id)}",
        "component_node_count": int(len(component)),
        "source_node_ids": [int(v) for v in component],
        "source_frame_ids": sorted(int(v) for v in frame_ids),
        "source_frame_ranks": sorted(int(v) for v in frame_ranks),
        "source_frame_or_region_ids": [item["region_key"] for item in source_regions],
        "source_regions": source_regions,
        "object_tube_ids": sorted(int(v) for v in tube_counter.keys()),
        "object_tube_count": int(len(tube_counter)),
        "object_tube_observation_count": int(sum(tube_counter.values())),
        "temporal_span": int(temporal_span),
        "total_vertex_union": int(len(point_ids)),
        "score": float(score),
        "num_export_candidates": 1,
        "num_exported_candidates": 0,
        "main_candidate_id": None,
        "duplicate_candidate_count": 0,
        "candidate_vertex_iou_pair_count": 0,
        "candidate_vertex_iou_mean": None,
        "candidate_vertex_iou_max": None,
    }


def _build_scene_records(args: argparse.Namespace, scene: str, state: Any, components: list[list[int]]) -> tuple[Any, list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stream = ScanNetStream(seq_name=scene)
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode="mask_backproject",
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.export_min_points_per_object),
        export_score_mode=args.export_score_mode,
    )
    nodes_by_id = {int(node.node_id): node for node in state.nodes}
    _, labels_by_frame, _manifest = _load_masks(Path(args.mask_root), scene, args.source, args.mode, int(args.min_region_area))
    depth_shape_by_frame: dict[int, tuple[int, int]] = {}
    object_records: list[dict[str, Any]] = []
    object_dict: dict[int, dict[str, Any]] = {}
    object_trace_rows: list[dict[str, Any]] = []
    build_diag = {
        "mask_observation_count": 0,
        "backproject_pixel_count": 0,
        "backproject_hit_count": 0,
    }
    for object_id, component in enumerate(components):
        point_ids: set[int] = set()
        mask_list = []
        source_regions: list[dict[str, Any]] = []
        for node_id in component:
            node = nodes_by_id[int(node_id)]
            frame_id = int(node.frame_id)
            if frame_id not in depth_shape_by_frame:
                depth_shape_by_frame[frame_id] = exporter.stream.load_depth(frame_id).shape
            xy = _component_mask_xy(labels_by_frame, frame_id, int(node.node_id), depth_shape_by_frame[frame_id])
            if xy.size == 0:
                source_regions.append(
                    {
                        "node_id": int(node.node_id),
                        "frame_id": int(frame_id),
                        "frame_rank": int(state.frame_rank.get(frame_id, frame_id)),
                        "mask_index": int(node.mask_index),
                        "region_key": f"{int(frame_id)}:{int(node.mask_index)}:{int(node.node_id)}",
                        "area": int(node.area),
                        "pixel_count": 0,
                        "backproject_hit_count": 0,
                    }
                )
                continue
            hit_ids, _dist = exporter._backproject_xy(frame_id, xy, nn_radius=float(args.export_nn_radius))
            point_ids.update(int(v) for v in hit_ids.tolist())
            mask_list.append((int(frame_id), int(node.mask_index), float(node.area)))
            build_diag["mask_observation_count"] += 1
            build_diag["backproject_pixel_count"] += int(xy.shape[0])
            build_diag["backproject_hit_count"] += int(hit_ids.shape[0])
            source_regions.append(
                {
                    "node_id": int(node.node_id),
                    "frame_id": int(frame_id),
                    "frame_rank": int(state.frame_rank.get(frame_id, frame_id)),
                    "mask_index": int(node.mask_index),
                    "region_key": f"{int(frame_id)}:{int(node.mask_index)}:{int(node.node_id)}",
                    "area": int(node.area),
                    "pixel_count": int(xy.shape[0]),
                    "backproject_hit_count": int(hit_ids.shape[0]),
                }
            )
        sorted_points = sorted(point_ids)
        record = {
            "object_id": int(object_id),
            "point_ids": set(sorted_points),
            "score": float(len(sorted_points)),
            "area_score": float(len(sorted_points)),
        }
        object_records.append(record)
        object_dict[int(object_id)] = {
            "point_ids": np.asarray(sorted_points, dtype=np.int64),
            "mask_list": mask_list,
            "repre_mask_list": mask_list[: min(len(mask_list), 8)],
            "score": float(len(sorted_points)),
            "area_score": float(len(sorted_points)),
            "source_variant": args.variant,
            "source_nodes": np.asarray([int(v) for v in component], dtype=np.int64),
        }
        object_trace_rows.append(
            _component_trace_record(
                state=state,
                component=component,
                object_id=object_id,
                point_ids=point_ids,
                source_regions=source_regions,
                score=score_export_record(record, args.export_score_mode),
            )
        )
    return exporter, object_records, object_dict, object_trace_rows, build_diag


def _prediction_trace(
    scene: str,
    exporter: ScanNetExporter,
    pred_masks: np.ndarray,
    pred_scores: np.ndarray,
    kept_records: list[dict[str, Any]],
    object_trace_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    object_by_id = {int(row["object_id"]): row for row in object_trace_rows}
    owner_counts = pred_masks.sum(axis=1).astype(np.int32)
    point_ids_by_prediction = [np.flatnonzero(pred_masks[:, idx]).astype(np.int64) for idx in range(pred_masks.shape[1])]
    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, point_ids in enumerate(point_ids_by_prediction):
        hash_to_indices[_sha1_ints(point_ids)].append(int(idx))
    duplicate_indices = {idx for indices in hash_to_indices.values() if len(indices) > 1 for idx in indices}

    prediction_rows: list[dict[str, Any]] = []
    for out_idx, record in enumerate(kept_records):
        object_id = int(record["object_id"])
        object_row = object_by_id[object_id]
        point_ids = point_ids_by_prediction[out_idx]
        vertex_count = int(point_ids.shape[0])
        conflict_count = int(np.count_nonzero(owner_counts[point_ids] > 1)) if vertex_count else 0
        duplicate_group = hash_to_indices[_sha1_ints(point_ids)]
        prediction_id = f"{scene}:{out_idx}"
        object_row["num_exported_candidates"] = 1
        object_row["main_candidate_id"] = prediction_id
        object_row["duplicate_candidate_count"] = max(0, int(len(duplicate_group) - 1))
        row = {
            "scene": scene,
            "prediction_id": prediction_id,
            "prediction_index": int(out_idx),
            "object_id": object_id,
            "source_object_id": object_row["source_object_id"],
            "source_frame_or_region_ids": object_row["source_frame_or_region_ids"],
            "source_node_ids": object_row["source_node_ids"],
            "score": float(pred_scores[out_idx]),
            "vertex_count": vertex_count,
            "mesh_vertex_count": int(exporter.scene_points.shape[0]),
            "object_tube_count": int(object_row["object_tube_count"]),
            "temporal_span": int(object_row["temporal_span"]),
            "is_duplicate_candidate": bool(out_idx in duplicate_indices),
            "duplicate_group_size": int(len(duplicate_group)),
            "overlap_with_other_predictions": float(conflict_count / max(vertex_count, 1)),
            "conflict_vertex_count": int(conflict_count),
            "point_ids_sha1": _sha1_ints(point_ids),
        }
        prediction_rows.append(row)

    traced_object_ids = [int(row["object_id"]) for row in prediction_rows]
    summary = {
        "prediction_trace_row_count": int(len(prediction_rows)),
        "exported_prediction_count": int(pred_masks.shape[1]),
        "object_trace_row_count": int(len(object_trace_rows)),
        "object_to_candidate_trace_complete": bool(
            len(prediction_rows) == pred_masks.shape[1]
            and len(traced_object_ids) == len(set(traced_object_ids))
            and all(str(row.get("source_object_id") or "") for row in prediction_rows)
        ),
        "duplicate_prediction_count": int(len(duplicate_indices)),
        "duplicate_prediction_rate": float(len(duplicate_indices) / max(pred_masks.shape[1], 1)),
        "mean_vertices_per_prediction": _mean([float(row["vertex_count"]) for row in prediction_rows]),
        "median_vertices_per_prediction": _median([float(row["vertex_count"]) for row in prediction_rows]),
        "mean_prediction_overlap_fraction": _mean([float(row["overlap_with_other_predictions"]) for row in prediction_rows]),
        "mean_conflict_vertex_count": _mean([float(row["conflict_vertex_count"]) for row in prediction_rows]),
        "exact_duplicate_definition": "identical exported mesh vertex id set within a scene",
    }
    return prediction_rows, object_trace_rows, summary


def _export_scene(args: argparse.Namespace, scene: str, state: Any, components: list[list[int]], out_root: Path) -> dict[str, Any]:
    exporter, object_records, object_dict, object_trace_rows, build_diag = _build_scene_records(args, scene, state, components)
    kept_records = [record for record in object_records if len(record["point_ids"]) >= int(args.export_min_points_per_object)]
    diag = exporter._write_outputs(object_records, object_dict, np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16))
    pred_path = Path("data/prediction") / f"{args.output_config}_class_agnostic" / f"{scene}.npz"
    with np.load(pred_path) as pred:
        pred_masks = np.asarray(pred["pred_masks"], dtype=bool)
        pred_scores = np.asarray(pred["pred_score"], dtype=np.float32)
        point_union = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
    prediction_rows, object_trace_rows, trace_summary = _prediction_trace(
        scene=scene,
        exporter=exporter,
        pred_masks=pred_masks,
        pred_scores=pred_scores,
        kept_records=kept_records,
        object_trace_rows=object_trace_rows,
    )
    gt_ratio, gt_covered, gt_total = _gt_coverage_ratio(Path(args.gt_path) / f"{scene}.txt", point_union)
    scene_dir = out_root / "scenes" / scene
    _write_json(scene_dir / "prediction_trace_rows.json", prediction_rows)
    _write_json(scene_dir / "object_trace_rows.json", object_trace_rows)
    _write_csv(scene_dir / "prediction_trace_rows.csv", prediction_rows)
    _write_csv(scene_dir / "object_trace_rows.csv", object_trace_rows)
    row = {
        "scene": scene,
        "variant": args.variant,
        "prediction_path": str(pred_path),
        "prediction_trace_json": str(scene_dir / "prediction_trace_rows.json"),
        "prediction_trace_csv": str(scene_dir / "prediction_trace_rows.csv"),
        "object_trace_json": str(scene_dir / "object_trace_rows.json"),
        "object_trace_csv": str(scene_dir / "object_trace_rows.csv"),
        "num_components": int(len(components)),
        "num_candidate_objects": int(len(object_records)),
        "num_exported_objects": int(pred_masks.shape[1]),
        "num_scene_points": int(pred_masks.shape[0]),
        "num_exported_points": int(point_union.shape[0]),
        "pre_percent": float(point_union.shape[0] / max(pred_masks.shape[0], 1)),
        "union_percent": float(point_union.shape[0] / max(pred_masks.shape[0], 1)),
        "mesh_coverage": float(point_union.shape[0] / max(pred_masks.shape[0], 1)),
        "covered_GT_instance_ratio": gt_ratio,
        "covered_GT_instances": int(gt_covered),
        "total_GT_instances": int(gt_total),
        "mask_observation_count": int(build_diag["mask_observation_count"]),
        "backproject_pixel_count": int(build_diag["backproject_pixel_count"]),
        "backproject_hit_count": int(build_diag["backproject_hit_count"]),
        "backproject_hit_rate": float(build_diag["backproject_hit_count"] / max(build_diag["backproject_pixel_count"], 1)),
        "export_conflict_rate": float(diag["export_conflict_rate"]),
        **trace_summary,
    }
    _write_json(scene_dir / "scene_trace_summary.json", row)
    return row


def _write_manifest(args: argparse.Namespace) -> None:
    manifest = build_prediction_manifest(
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.memory_decision],
        pre_points_policy="recompute",
        support_policy="v38_export_trace:mask_backproject",
        notes="v38 Phase B evaluation-only export runtime trace from frozen v37 F31/I4 components; uses ScanNet RGB-D/pose/mesh as materialization bridge.",
        extra={
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "phase": "v38_phaseB_export_trace",
            "temporal_stage": args.variant,
            "mask_source": f"{args.source}:{args.mode}",
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
            "runtime_trace_artifact": str(Path(args.output_root) / "export_trace_summary.json"),
        },
    )
    write_prediction_manifest(args.output_config, manifest, pred_suffix="class_agnostic")


def _write_callgraph(args: argparse.Namespace, out_root: Path, scene_rows: list[dict[str, Any]], eval_result: dict[str, Any] | None) -> None:
    payload = {
        "phase": "v38_phaseB_export_trace",
        "static_callgraph": [
            {"stage": "F31 object tubes", "producer": "tools/run_v37_temporal_curriculum.py", "artifact": "outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json"},
            {"stage": "I4 memory objects", "producer": "tools/run_v37_4d_if_allowed.py", "artifact": args.memory_decision},
            {"stage": "export candidates", "producer": "tools/run_v38_export_trace.py:_build_scene_records", "artifact": "scenes/<scene>/object_trace_rows.{json,csv}"},
            {"stage": "mesh/vertex assignment", "producer": "stream4d/export_scannet.py:ScanNetExporter._backproject_xy", "artifact": "scenes/<scene>/prediction_trace_rows.{json,csv}"},
            {"stage": "prediction masks", "producer": "stream4d/export_scannet.py:ScanNetExporter._write_outputs", "artifact": f"data/prediction/{args.output_config}_class_agnostic/*.npz"},
            {"stage": "scores", "producer": "stream4d/export_scannet.py:score_export_record", "artifact": "pred_score in exported npz plus prediction_trace_rows"},
            {"stage": "evaluator input files", "producer": "evaluation.evaluate", "artifact": f"data/TMP/{args.output_config}/*_pre_points.npy"},
            {"stage": "AP evaluator", "producer": "evaluation/evaluate.py", "artifact": None if eval_result is None else eval_result.get("metric_file")},
        ],
        "runtime_scene_rows": scene_rows,
        "eval_result": eval_result,
    }
    _write_json(out_root / "export_callgraph.json", payload)
    lines = [
        "# Stream4D v38 Phase B Export Callgraph",
        "",
        "```text",
        "F31 object tubes -> I4 memory objects -> export candidates -> mesh/vertex assignment -> prediction masks -> scores -> evaluator input files -> AP evaluator",
        "```",
        "",
        "## Runtime Scenes",
        "",
        "| scene | predictions | trace rows | complete | conflict | duplicate_rate | pre% |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in scene_rows:
        lines.append(
            "| {scene} | {pred} | {trace} | {complete} | {conflict} | {dup} | {pre} |".format(
                scene=row["scene"],
                pred=row["exported_prediction_count"],
                trace=row["prediction_trace_row_count"],
                complete=row["object_to_candidate_trace_complete"],
                conflict=row["export_conflict_rate"],
                dup=row["duplicate_prediction_rate"],
                pre=row["pre_percent"],
            )
        )
    if eval_result is not None:
        lines.extend(["", "## Evaluator", "", "```text"])
        lines.append(f"exit_code={eval_result.get('exit_code')}")
        lines.append(f"metric_file={eval_result.get('metric_file')}")
        lines.append(f"metrics={eval_result.get('metrics')}")
        lines.append("```")
    (out_root / "export_callgraph.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_eval(args: argparse.Namespace, root: Path, out_root: Path) -> dict[str, Any]:
    eval_dir = out_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{args.output_config}_class_agnostic.txt"
    log_path = eval_dir / f"{args.output_config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(root / "data/prediction" / f"{args.output_config}_class_agnostic"),
        "--gt_path",
        str(root / args.gt_path),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(root / "data/TMP"),
        "--tmp_config",
        args.output_config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    metrics = _parse_metric_file(metric_file) if metric_file.exists() else {}
    return {
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "log_path": str(log_path),
        "metric_file": str(metric_file),
        "metrics": metrics,
        "raw_v37_reference": EXPECTED_RAW_AP,
        "raw_v37_abs_diff": {
            key: abs(float(metrics[key]) - expected) if key in metrics else None
            for key, expected in EXPECTED_RAW_AP.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    if Path.cwd().resolve() != root:
        os.chdir(root)
    out_root = root / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))
    decision = json.loads((root / args.memory_decision).read_text(encoding="utf-8"))
    if decision.get("final_status") != "GO_4D_MEMORY":
        raise RuntimeError(f"4D memory decision is not eligible for AP export trace: {args.memory_decision}")
    if args.variant == "":
        args.variant = str(decision.get("best_variant"))
    scenes = _read_split(root / args.split)
    scene_rows = []
    pair_row_count = 0
    for scene in scenes:
        state, pair_row_count = _build_scene_state(scene, args, pair_row_count)
        components, _info = _component_from_variant(state, args.variant)
        scene_rows.append(_export_scene(args, scene, state, components, out_root))
    _write_manifest(args)
    eval_result = _run_eval(args, root, out_root) if bool(args.run_eval) else None
    prediction_count = int(sum(int(row["exported_prediction_count"]) for row in scene_rows))
    trace_count = int(sum(int(row["prediction_trace_row_count"]) for row in scene_rows))
    complete = bool(
        prediction_count == trace_count
        and all(bool(row["object_to_candidate_trace_complete"]) for row in scene_rows)
    )
    summary = {
        "phase": "v38_phaseB_export_trace",
        "output_config": args.output_config,
        "variant": args.variant,
        "is_diagnostic_only": True,
        "is_method_result": False,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "forbidden_for_method_table": True,
        "runtime_callgraph_generated": True,
        "prediction_trace_row_count": trace_count,
        "exported_prediction_count": prediction_count,
        "object_to_candidate_trace_complete": complete,
        "phaseB_export_trace_pass": bool(complete and (eval_result is None or int(eval_result["exit_code"]) == 0)),
        "scene_rows": scene_rows,
        "mean_pre_percent": _mean([float(row["pre_percent"]) for row in scene_rows]),
        "mean_union_percent": _mean([float(row["union_percent"]) for row in scene_rows]),
        "mean_mesh_coverage": _mean([float(row["mesh_coverage"]) for row in scene_rows]),
        "mean_export_conflict_rate": _mean([float(row["export_conflict_rate"]) for row in scene_rows]),
        "mean_num_predictions": _mean([float(row["exported_prediction_count"]) for row in scene_rows]),
        "median_num_predictions": _median([float(row["exported_prediction_count"]) for row in scene_rows]),
        "duplicate_prediction_rate": _mean([float(row["duplicate_prediction_rate"]) for row in scene_rows]),
        "mean_vertices_per_prediction": _mean([float(row["mean_vertices_per_prediction"] or 0.0) for row in scene_rows]),
        "mean_covered_GT_instance_ratio": _mean(
            [float(row["covered_GT_instance_ratio"]) for row in scene_rows if row["covered_GT_instance_ratio"] is not None]
        ),
        "eval_result": eval_result,
        "notes": [
            "Trace is evaluation-only and uses ScanNet RGB-D/pose/mesh materialization bridge.",
            "Each exported prediction is one kept Stream4D object candidate from frozen F31/I4 components.",
            "duplicate_prediction_rate is exact identical mesh-vertex-set duplicate rate, not high-IoU near-duplicate rate.",
        ],
    }
    _write_json(out_root / "export_trace_summary.json", summary)
    _write_csv(out_root / "export_trace_scene_rows.csv", scene_rows)
    _write_callgraph(args, out_root, scene_rows, eval_result)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v38 Phase B AP export runtime trace from frozen v37 F31/I4 objects.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--memory-decision", default="outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_decision.json")
    parser.add_argument("--variant", default="")
    parser.add_argument("--output-config", default="v38_i4_sparse_export_trace_probe5")
    parser.add_argument("--output-root", default="outputs/audit/v38_export_trace")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--max-same-frame-pairs-per-frame", type=int, default=4000)
    parser.add_argument("--max-allpair-samples-per-scene", type=int, default=30000)
    parser.add_argument("--max-shuffled-pair-rows-per-scene", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-min-points-per-object", type=int, default=1)
    parser.add_argument("--export-score-mode", default="area", choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"])
    parser.add_argument("--run-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-visible-devices", default="6")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
