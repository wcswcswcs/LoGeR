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

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v36_external_downstream_assignment import UnionFind
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)


CLASS_AGNOSTIC_ID = 3
DEFAULT_INPUT_CONFIG = "v39_purity_targeted_i4_gap2_rgb099_probe5"


VARIANTS: dict[str, dict[str, Any]] = {
    "K1_min_ioc090_softmerge_cap300": {
        "config": "v39_k1_i4_min_ioc090_softmerge_cap300_probe5",
        "mode": "soft_merge",
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.90,
        "max_instances": 300,
        "min_area": 100,
        "merge_after_cap_below_threshold": True,
        "description": "area-priority soft merge; min-IoC 0.90 duplicate groups; overflow masks merge into best overlapping kept object",
    },
    "K2_min_ioc050_softmerge_cap300": {
        "config": "v39_k2_i4_min_ioc050_softmerge_cap300_probe5",
        "mode": "soft_merge",
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "min_area": 100,
        "merge_after_cap_below_threshold": True,
        "description": "area-priority soft merge; min-IoC 0.50 duplicate groups; overflow masks merge into best overlapping kept object",
    },
    "K3_min_ioc025_softmerge_cap300": {
        "config": "v39_k3_i4_min_ioc025_softmerge_cap300_probe5",
        "mode": "soft_merge",
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.25,
        "max_instances": 300,
        "min_area": 100,
        "merge_after_cap_below_threshold": True,
        "description": "area-priority soft merge; min-IoC 0.25 larger duplicate groups; overflow masks merge into best overlapping kept object",
    },
    "K4_iou010_softmerge_cap300": {
        "config": "v39_k4_i4_iou010_softmerge_cap300_probe5",
        "mode": "soft_merge",
        "overlap_mode": "iou",
        "overlap_threshold": 0.10,
        "max_instances": 300,
        "min_area": 100,
        "merge_after_cap_below_threshold": True,
        "description": "area-priority soft merge; IoU 0.10 groups complementary overlapping fragments",
    },
    "K5_contact015_gap020_color093_cap300": {
        "config": "v39_k5_i4_contact015_gap020_color093_cap300_probe5",
        "mode": "geometry_contact",
        "centroid_radius": 0.15,
        "bbox_gap": 0.020,
        "min_color_similarity": 0.93,
        "max_instances": 300,
        "min_area": 100,
        "description": "3D centroid/bbox/color contact graph; conservative local merge of adjacent fragments",
    },
    "K6_contact030_gap050_color090_cap300": {
        "config": "v39_k6_i4_contact030_gap050_color090_cap300_probe5",
        "mode": "geometry_contact",
        "centroid_radius": 0.30,
        "bbox_gap": 0.050,
        "min_color_similarity": 0.90,
        "max_instances": 300,
        "min_area": 100,
        "description": "3D centroid/bbox/color contact graph; looser local merge of adjacent fragments",
    },
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


def _load_scene_points(root: Path, scene: str) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for compact object materialization geometry features") from exc
    mesh_path = root / "data/scannet/processed" / scene / f"{scene}_vh_clean_2.ply"
    cloud = o3d.io.read_point_cloud(str(mesh_path))
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32) if cloud.has_colors() else None
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Failed to load scene point cloud: {mesh_path}")
    if colors is not None and colors.shape != points.shape:
        colors = None
    return points, colors


def _order_indices(areas: np.ndarray) -> list[int]:
    return sorted(range(areas.shape[0]), key=lambda idx: (-float(areas[idx]), int(idx)))


def _overlap_score(intersection: int, kept_area: float, area: float, mode: str) -> float:
    if mode == "iou":
        denom = kept_area + area - float(intersection)
    elif mode == "min_ioc":
        denom = min(kept_area, area)
    elif mode == "candidate_ioc":
        denom = area
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")
    return float(intersection) / max(float(denom), 1.0)


def _soft_merge(
    masks: np.ndarray,
    *,
    min_area: int,
    max_instances: int,
    threshold: float,
    mode: str,
    merge_after_cap_below_threshold: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    areas = masks.sum(axis=0).astype(np.float64)
    kept_masks: list[np.ndarray] = []
    kept_sources: list[int] = []
    group_sizes: list[int] = []
    owners_by_point: dict[int, list[int]] = defaultdict(list)
    point_cache: dict[int, np.ndarray] = {}
    dropped_low_area = 0
    dropped_unmatched_after_cap = 0
    merged_count = 0

    for idx in _order_indices(areas):
        area = float(areas[idx])
        if area < float(min_area):
            dropped_low_area += 1
            continue
        points = point_cache.get(idx)
        if points is None:
            points = np.flatnonzero(masks[:, idx]).astype(np.int64)
            point_cache[idx] = points
        intersections: dict[int, int] = defaultdict(int)
        for point in points.tolist():
            for kept_idx in owners_by_point.get(int(point), ()):
                intersections[int(kept_idx)] += 1
        best_kept = -1
        best_overlap = 0.0
        for kept_idx, intersection in intersections.items():
            overlap = _overlap_score(
                int(intersection),
                float(np.count_nonzero(kept_masks[int(kept_idx)])),
                area,
                mode,
            )
            if overlap > best_overlap:
                best_overlap = float(overlap)
                best_kept = int(kept_idx)
        if best_kept >= 0 and best_overlap >= float(threshold):
            before = kept_masks[best_kept].copy()
            kept_masks[best_kept][points] = True
            new_points = np.flatnonzero(kept_masks[best_kept] & ~before).astype(np.int64)
            for point in new_points.tolist():
                owners_by_point.setdefault(int(point), []).append(best_kept)
            group_sizes[best_kept] += 1
            merged_count += 1
            continue
        if int(max_instances) > 0 and len(kept_masks) >= int(max_instances):
            if best_kept >= 0 and (bool(merge_after_cap_below_threshold) or best_overlap >= float(threshold)):
                before = kept_masks[best_kept].copy()
                kept_masks[best_kept][points] = True
                new_points = np.flatnonzero(kept_masks[best_kept] & ~before).astype(np.int64)
                for point in new_points.tolist():
                    owners_by_point.setdefault(int(point), []).append(best_kept)
                group_sizes[best_kept] += 1
                merged_count += 1
            else:
                dropped_unmatched_after_cap += 1
            continue
        out_mask = np.zeros((masks.shape[0],), dtype=bool)
        out_mask[points] = True
        kept_idx = len(kept_masks)
        kept_masks.append(out_mask)
        kept_sources.append(int(idx))
        group_sizes.append(1)
        for point in points.tolist():
            owners_by_point.setdefault(int(point), []).append(kept_idx)

    out = np.stack(kept_masks, axis=1) if kept_masks else masks[:, :0]
    source = np.asarray(kept_sources, dtype=np.int64)
    diag = {
        "kept_groups": int(out.shape[1]),
        "merged_count": int(merged_count),
        "dropped_low_area": int(dropped_low_area),
        "dropped_unmatched_after_cap": int(dropped_unmatched_after_cap),
        "mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
    }
    return out, source, diag


def _bbox_gap(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
) -> float:
    gap = np.maximum(0.0, np.maximum(left_min - right_max, right_min - left_max))
    return float(np.linalg.norm(gap))


def _color_similarity(left: np.ndarray, right: np.ndarray) -> float:
    dist = float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)))
    return float(max(0.0, 1.0 - dist / np.sqrt(3.0)))


def _geometry_contact_merge(
    masks: np.ndarray,
    scene_points: np.ndarray,
    scene_colors: np.ndarray | None,
    *,
    min_area: int,
    max_instances: int,
    centroid_radius: float,
    bbox_gap: float,
    min_color_similarity: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    areas = masks.sum(axis=0).astype(np.int64)
    eligible = np.flatnonzero(areas >= int(min_area)).astype(np.int64)
    if eligible.size == 0:
        return masks[:, :0], np.zeros((0,), dtype=np.int64), {
            "eligible_count": 0,
            "candidate_pairs": 0,
            "accepted_edges": 0,
            "kept_groups": 0,
        }

    centroids = np.zeros((eligible.shape[0], 3), dtype=np.float32)
    bbox_min = np.zeros((eligible.shape[0], 3), dtype=np.float32)
    bbox_max = np.zeros((eligible.shape[0], 3), dtype=np.float32)
    colors = np.zeros((eligible.shape[0], 3), dtype=np.float32)
    color_valid = np.zeros((eligible.shape[0],), dtype=bool)
    point_cache: dict[int, np.ndarray] = {}
    for pos, idx in enumerate(eligible.tolist()):
        pts_idx = np.flatnonzero(masks[:, int(idx)]).astype(np.int64)
        point_cache[int(idx)] = pts_idx
        pts = scene_points[pts_idx]
        centroids[pos] = np.mean(pts, axis=0)
        bbox_min[pos] = np.min(pts, axis=0)
        bbox_max[pos] = np.max(pts, axis=0)
        if scene_colors is not None:
            colors[pos] = np.mean(scene_colors[pts_idx], axis=0)
            color_valid[pos] = True

    tree = cKDTree(centroids)
    candidate_pairs = sorted(tree.query_pairs(r=float(centroid_radius)))
    uf = UnionFind(int(eligible.shape[0]))
    accepted = 0
    rejected_bbox = 0
    rejected_color = 0
    for left, right in candidate_pairs:
        if _bbox_gap(bbox_min[left], bbox_max[left], bbox_min[right], bbox_max[right]) > float(bbox_gap):
            rejected_bbox += 1
            continue
        if scene_colors is not None and bool(color_valid[left]) and bool(color_valid[right]):
            if _color_similarity(colors[left], colors[right]) < float(min_color_similarity):
                rejected_color += 1
                continue
        if uf.union(int(left), int(right)):
            accepted += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for pos, idx in enumerate(eligible.tolist()):
        groups[uf.find(int(pos))].append(int(idx))
    ordered_groups = sorted(groups.values(), key=lambda group: (-int(sum(int(areas[idx]) for idx in group)), min(group)))
    if int(max_instances) > 0:
        ordered_groups = ordered_groups[: int(max_instances)]

    out_masks = []
    source_indices = []
    group_sizes = []
    for group in ordered_groups:
        merged = np.zeros((masks.shape[0],), dtype=bool)
        for idx in group:
            merged[point_cache[int(idx)]] = True
        if int(np.count_nonzero(merged)) < int(min_area):
            continue
        out_masks.append(merged)
        source_indices.append(int(max(group, key=lambda idx: int(areas[idx]))))
        group_sizes.append(len(group))
    out = np.stack(out_masks, axis=1) if out_masks else masks[:, :0]
    source = np.asarray(source_indices, dtype=np.int64)
    diag = {
        "eligible_count": int(eligible.shape[0]),
        "candidate_pairs": int(len(candidate_pairs)),
        "accepted_edges": int(accepted),
        "rejected_bbox": int(rejected_bbox),
        "rejected_color": int(rejected_color),
        "kept_groups": int(out.shape[1]),
        "mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "has_vertex_color": bool(scene_colors is not None),
    }
    return out, source, diag


def _apply_variant(
    root: Path,
    scene: str,
    variant: str,
    meta: dict[str, Any],
    masks: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(meta["mode"])
    if mode == "soft_merge":
        out_masks, source, diag = _soft_merge(
            masks,
            min_area=int(meta["min_area"]),
            max_instances=int(meta["max_instances"]),
            threshold=float(meta["overlap_threshold"]),
            mode=str(meta["overlap_mode"]),
            merge_after_cap_below_threshold=bool(meta.get("merge_after_cap_below_threshold", True)),
        )
    elif mode == "geometry_contact":
        scene_points, scene_colors = _load_scene_points(root, scene)
        if scene_points.shape[0] != masks.shape[0]:
            raise ValueError(f"{scene}: point cloud has {scene_points.shape[0]} points but masks have {masks.shape[0]}")
        out_masks, source, diag = _geometry_contact_merge(
            masks,
            scene_points,
            scene_colors,
            min_area=int(meta["min_area"]),
            max_instances=int(meta["max_instances"]),
            centroid_radius=float(meta["centroid_radius"]),
            bbox_gap=float(meta["bbox_gap"]),
            min_color_similarity=float(meta["min_color_similarity"]),
        )
    else:
        raise ValueError(f"Unsupported variant mode: {mode}")
    scores = out_masks.sum(axis=0).astype(np.float32) if out_masks.shape[1] else np.zeros((0,), dtype=np.float32)
    out_classes = classes[source] if source.size else np.zeros((0,), dtype=np.int32)
    out_classes[:] = int(CLASS_AGNOSTIC_ID)
    diag.update({
        "variant": variant,
        "config": str(meta["config"]),
        "num_predictions_in": int(masks.shape[1]),
        "num_predictions_out": int(out_masks.shape[1]),
    })
    return out_masks, scores, out_classes, diag


def _write_prediction(root: Path, config: str, scene: str, masks: np.ndarray, scores: np.ndarray, classes: np.ndarray, raw_pre_points: np.ndarray) -> None:
    pred_dir = root / "data/prediction" / f"{config}_class_agnostic"
    tmp_dir = root / "data/TMP" / config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{scene}.npz",
        pred_masks=masks.astype(bool, copy=False),
        pred_score=scores.astype(np.float32, copy=False),
        pred_classes=classes.astype(np.int32, copy=False),
    )
    np.save(tmp_dir / f"{scene}_pre_points.npy", raw_pre_points.astype(np.int64, copy=False))


def _write_manifest(root: Path, variant: str, meta: dict[str, Any], input_config: str) -> None:
    config = str(meta["config"])
    manifest = build_prediction_manifest(
        root=root,
        output_config=config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[input_config, "outputs/audit/v39_purity_targeted_repair/v37_4d_if_allowed/4d_memory_decision.json"],
        pre_points_policy=f"copied_raw_support:{input_config}",
        support_policy=f"v39_compact_object_materializer:{variant}:{meta['mode']}",
        notes=f"v39 compact object materializer {variant}: {meta['description']}. Diagnostic-only because the input AP export uses ScanNet RGB-D/pose/mesh bridge.",
        extra={
            "phase": "v39_compact_object_materializer",
            "variant": variant,
            "forbidden_for_method_table": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
            "input_config": input_config,
            "input_is_diagnostic_only": True,
        },
    )
    write_prediction_manifest(config, manifest, root=root, pred_suffix="class_agnostic")


def _run_eval(args: argparse.Namespace, root: Path, config: str, output_root: Path) -> dict[str, Any]:
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{config}_class_agnostic.txt"
    log_path = eval_dir / f"{config}_evaluate_allow_oracle_eval.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(root / "data/prediction" / f"{config}_class_agnostic"),
        "--gt_path",
        str(root / args.gt_path),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(root / "data/TMP"),
        "--tmp_config",
        config,
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
        "config": config,
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "metric_file": str(metric_file),
        "log_path": str(log_path),
        **metrics,
    }


def _aggregate_scene_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    out = []
    for variant, items in sorted(by_variant.items()):
        out.append({
            "variant": variant,
            "config": str(items[0].get("config")),
            "scene_count": int(len(items)),
            "mean_num_predictions": float(np.mean([float(row.get("num_predictions") or 0.0) for row in items])),
            "mean_predictions_in": float(np.mean([float(row.get("num_predictions_in") or 0.0) for row in items])),
            "mean_predictions_out": float(np.mean([float(row.get("num_predictions_out") or 0.0) for row in items])),
            "mean_mesh_coverage": float(np.mean([float(row.get("mesh_coverage") or 0.0) for row in items])),
            "mean_export_conflict_rate": float(np.mean([float(row.get("mean_export_conflict_rate") or 0.0) for row in items])),
            "mean_per_GT_best_IoU_ge_50": float(np.mean([float(row.get("per_GT_best_IoU_ge_50") or 0.0) for row in items])),
            "mean_per_GT_best_IoU_mean": float(np.mean([float(row.get("per_GT_best_IoU_mean") or 0.0) for row in items])),
            "mean_group_size": float(np.mean([float(row.get("mean_group_size") or 0.0) for row in items])),
            "max_group_size": int(max(int(row.get("max_group_size") or 0) for row in items)),
        })
    return out


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v39 Compact Object Materializer Summary",
        "",
        "```text",
        f"final_status={summary['final_status']}",
        f"best_variant={summary.get('best_variant')}",
        f"best_AP={summary.get('best_AP')}",
        f"best_AP50={summary.get('best_AP50')}",
        f"best_AP25={summary.get('best_AP25')}",
        "```",
        "",
        "| variant | AP | AP50 | AP25 | mean predictions | conflict | coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    eval_by_config = {row["config"]: row for row in summary.get("eval_rows", [])}
    for row in summary.get("variant_rows", []):
        metrics = eval_by_config.get(row["config"], {})
        lines.append(
            "| {variant} | {AP} | {AP50} | {AP25} | {preds:.1f} | {conflict:.6f} | {coverage:.6f} |".format(
                variant=row["variant"],
                AP=metrics.get("AP"),
                AP50=metrics.get("AP50"),
                AP25=metrics.get("AP25"),
                preds=float(row.get("mean_predictions_out") or 0.0),
                conflict=float(row.get("mean_export_conflict_rate") or 0.0),
                coverage=float(row.get("mean_mesh_coverage") or 0.0),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_split(root / args.split)
    selected_variants = [name for name in args.variants.split(",") if name] if args.variants else list(VARIANTS)
    for variant in selected_variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}; choices={sorted(VARIANTS)}")
        _write_manifest(root, variant, VARIANTS[variant], args.input_config)

    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        pred_path = root / "data/prediction" / f"{args.input_config}_class_agnostic" / f"{scene}.npz"
        pre_path = root / "data/TMP" / args.input_config / f"{scene}_pre_points.npy"
        gt_path = root / args.gt_path / f"{scene}.txt"
        with np.load(pred_path) as pred:
            masks = np.asarray(pred["pred_masks"], dtype=bool)
            classes = np.asarray(pred["pred_classes"], dtype=np.int32)
        raw_pre_points = np.load(pre_path).astype(np.int64)
        gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
        for variant in selected_variants:
            meta = VARIANTS[variant]
            out_masks, scores, out_classes, diag = _apply_variant(root, scene, variant, meta, masks, classes)
            config = str(meta["config"])
            _write_prediction(root, config, scene, out_masks, scores, out_classes, raw_pre_points)
            stats = _candidate_oracle_stats(out_masks, scores, gt_eval, raw_pre_points)
            row = _quality_stats(scene, variant, out_masks, scores, gt_eval, raw_pre_points, stats)
            row.update(diag)
            scene_rows.append(row)
            del out_masks, scores, out_classes
        del masks, classes

    variant_rows = _aggregate_scene_rows(scene_rows)
    eval_rows = []
    for variant in selected_variants:
        eval_rows.append(_run_eval(args, root, str(VARIANTS[variant]["config"]), output_root))

    successful_eval_rows = [row for row in eval_rows if int(row.get("exit_code", 1)) == 0 and row.get("AP") is not None]
    best_eval = max(successful_eval_rows, key=lambda row: float(row.get("AP") or -1.0), default={})
    best_variant = next((name for name in selected_variants if VARIANTS[name]["config"] == best_eval.get("config")), None)
    best_ap = best_eval.get("AP")
    final_status = (
        "NO_GO_COMPACT_OBJECT_MATERIALIZER_AP_STILL_LOW"
        if best_ap is None or float(best_ap) < float(args.ap_gate)
        else "GO_COMPACT_OBJECT_MATERIALIZER_DIAGNOSTIC_AP_GATE"
    )
    summary = {
        "plan": "docs/stream4d_v39_object_identity_first_plan.md",
        "phase": "post_closeout_compact_object_materializer",
        "input_config": args.input_config,
        "is_diagnostic_only": True,
        "is_method_result": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "ap_gate": float(args.ap_gate),
        "final_status": final_status,
        "best_variant": best_variant,
        "best_config": best_eval.get("config"),
        "best_AP": best_eval.get("AP"),
        "best_AP50": best_eval.get("AP50"),
        "best_AP25": best_eval.get("AP25"),
        "scene_rows": scene_rows,
        "variant_rows": variant_rows,
        "eval_rows": eval_rows,
    }
    _write_json(output_root / "compact_object_materializer_summary.json", summary)
    _write_csv(output_root / "compact_object_materializer_scene_rows.csv", scene_rows)
    _write_csv(output_root / "compact_object_materializer_variant_rows.csv", variant_rows)
    _write_csv(output_root / "compact_object_materializer_eval_rows.csv", eval_rows)
    _write_markdown(output_root / "compact_object_materializer_summary.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="v39 compact object-level materialization from F31/I4 AP export masks.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--input-config", default=DEFAULT_INPUT_CONFIG)
    parser.add_argument("--output-root", default="outputs/audit/v39_compact_object_materializer")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--variants", default="")
    parser.add_argument("--ap-gate", type=float, default=0.06267906582522053)
    parser.add_argument("--cuda-visible-devices", default=None)
    summary = run(parser.parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
