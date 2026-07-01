from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v47_common import resolve_mask_dir
from stream4d_native.v65_d4rt_geometry import D4RT_COORDINATE_MODES, load_d4rt_geometry_frames
from stream4d_native.v65_visualization_export import _load_gt, _load_scene_mesh
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    path_obj = _project(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path_obj = _project(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sha256(path: str | Path) -> str:
    path_obj = _project(path)
    if not path_obj.exists():
        return ""
    digest = hashlib.sha256()
    with path_obj.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_mask_observation_id(value: str) -> tuple[str, int, int] | None:
    parts = str(value or "").split(":")
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _best_objectlet_variant(pipeline_root: Path, explicit: str) -> str:
    if explicit and explicit != "best":
        return explicit
    summary = _read_json(pipeline_root / "local_objectlets" / "local_objectlet_summary.json")
    variant = str(summary.get("best_real_variant") or "").strip()
    if not variant:
        raise RuntimeError(f"best_real_variant missing in {pipeline_root / 'local_objectlets/local_objectlet_summary.json'}")
    return variant


def _resolve_pipeline_mask_dir(
    *,
    scene: str,
    pipeline_summary: dict[str, Any],
    override_mask_root: str | Path | None,
) -> Path:
    if override_mask_root is not None and str(override_mask_root).strip():
        return resolve_mask_dir(override_mask_root, scene)
    coverage = pipeline_summary.get("mask_frame_coverage")
    if isinstance(coverage, dict):
        mask_dir = str(coverage.get("mask_dir") or "").strip()
        if mask_dir:
            return _project(mask_dir)
    mask_root = str(pipeline_summary.get("mask_root_for_pipeline") or pipeline_summary.get("mask_root") or "").strip()
    if mask_root:
        return resolve_mask_dir(mask_root, scene)
    return resolve_mask_dir(None, scene)


def _load_pipeline_support(
    *,
    pipeline_root: Path,
    scene: str,
    objectlet_variant: str,
    success_only: bool,
) -> tuple[dict[tuple[int, int], int], dict[int, str], dict[str, Any]]:
    objectlet_rows_path = pipeline_root / "local_objectlets" / "objectlet_rows.csv"
    ledger_rows_path = pipeline_root / "reprojection_ledger" / "reprojection_ledger_rows.csv"
    selected_by_candidate: dict[str, tuple[str, int]] = {}
    object_idx_to_id: dict[int, str] = {}
    object_to_idx: dict[str, int] = {}
    selected_rows = 0
    with objectlet_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != scene or row.get("variant") != objectlet_variant:
                continue
            object_id = str(row.get("objectlet_id") or "").strip()
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not object_id or not candidate_id:
                continue
            if object_id not in object_to_idx:
                object_idx = len(object_to_idx) + 1
                object_to_idx[object_id] = object_idx
                object_idx_to_id[object_idx] = object_id
            selected_by_candidate[candidate_id] = (object_id, object_to_idx[object_id])
            selected_rows += 1

    mask_to_object_idx: dict[tuple[int, int], int] = {}
    duplicate_frame_mask_conflicts = 0
    ledger_rows = 0
    used_ledger_rows = 0
    skipped_failed_rows = 0
    with ledger_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ledger_rows += 1
            selected = selected_by_candidate.get(str(row.get("candidate_id") or ""))
            if not selected:
                continue
            if success_only and not _parse_bool(row.get("reprojection_success")):
                skipped_failed_rows += 1
                continue
            parsed = _parse_mask_observation_id(str(row.get("best_mask_observation_id") or ""))
            if parsed is None:
                continue
            row_scene, frame_id, mask_id = parsed
            if row_scene != scene or mask_id <= 0:
                continue
            _object_id, object_idx = selected
            key = (int(frame_id), int(mask_id))
            if key in mask_to_object_idx and mask_to_object_idx[key] != object_idx:
                duplicate_frame_mask_conflicts += 1
                object_idx = min(mask_to_object_idx[key], object_idx)
            mask_to_object_idx[key] = int(object_idx)
            used_ledger_rows += 1

    diag = {
        "objectlet_variant": objectlet_variant,
        "objectlet_row_count": int(selected_rows),
        "object_count": int(len(object_idx_to_id)),
        "ledger_row_count": int(ledger_rows),
        "used_ledger_row_count": int(used_ledger_rows),
        "skipped_failed_ledger_row_count": int(skipped_failed_rows),
        "support_pair_count": int(len(mask_to_object_idx)),
        "support_frame_count": int(len({frame for frame, _mask in mask_to_object_idx})),
        "duplicate_frame_mask_conflicts": int(duplicate_frame_mask_conflicts),
        "support_contract": "pipeline local_objectlets selected candidate_id joined to same-root reprojection_ledger best_mask_observation_id",
    }
    return mask_to_object_idx, object_idx_to_id, diag


def _score_object(point_hits: int, vertex_count: int, mode: str) -> float:
    if mode == "vertex_count":
        return float(vertex_count)
    if mode == "point_hits":
        return float(point_hits)
    return float(point_hits * np.sqrt(max(vertex_count, 1)))


def _sample_mask_owner(
    *,
    mask: np.ndarray,
    uv: np.ndarray,
    frame_id: int,
    mask_to_object_idx: dict[tuple[int, int], int],
) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    h, w = mask.shape[:2]
    xy = np.rint(
        np.stack([uv[:, 0] * float(max(w - 1, 1)), uv[:, 1] * float(max(h - 1, 1))], axis=1)
    ).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
    xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
    mask_ids = mask[xy[:, 1], xy[:, 0]]
    return np.asarray([mask_to_object_idx.get((int(frame_id), int(mask_id)), 0) for mask_id in mask_ids.tolist()], dtype=np.int32)


def _materialize_predictions(
    *,
    scene: str,
    pipeline_root: Path,
    mask_dir: Path,
    mask_to_object_idx: dict[tuple[int, int], int],
    object_idx_to_id: dict[int, str],
    confidence_threshold: float,
    visibility_threshold: float,
    nn_radius: float,
    min_vertices: int,
    score_mode: str,
    d4rt_coordinate_mode: str,
    d4rt_stride_summary: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scene_points, _scene_rgb, mesh_path = _load_scene_mesh(scene)
    tree = cKDTree(scene_points.astype(np.float32, copy=False))
    geometry_frames, geometry_diag = load_d4rt_geometry_frames(
        pipeline_root=pipeline_root,
        scene=scene,
        stream3d_root=ROOT,
        coordinate_mode=d4rt_coordinate_mode,
        d4rt_stride_summary=d4rt_stride_summary or None,
    )

    object_vertices: dict[int, set[int]] = defaultdict(set)
    object_point_hits: Counter[int] = Counter()
    object_assigned_points: Counter[int] = Counter()
    frame_rows: list[dict[str, Any]] = []
    missing_mask_frames: set[int] = set()
    counters: Counter[str] = Counter()
    carrier_count = 0
    for geometry_payload in geometry_frames:
        carrier_count += 1
        carrier_path = geometry_payload["carrier_path"]
        frame_ids = [int(value) for value in geometry_payload["frame_ids"]]
        xyz = np.asarray(geometry_payload["xyz"], dtype=np.float32)
        uv = np.asarray(geometry_payload["uv"], dtype=np.float32)
        valid = np.asarray(geometry_payload["valid"], dtype=bool)
        confidence = np.asarray(geometry_payload["confidence"], dtype=np.float32)
        visibility = np.asarray(geometry_payload["visibility"], dtype=np.float32)
        if len(frame_ids) != xyz.shape[0]:
            raise ValueError(f"frame manifest length mismatch: {carrier_path}")
        for local_idx, frame_id in enumerate(frame_ids):
            ok = (
                valid[local_idx]
                & np.isfinite(xyz[local_idx]).all(axis=1)
                & np.isfinite(uv[local_idx]).all(axis=1)
                & (uv[local_idx, :, 0] >= 0.0)
                & (uv[local_idx, :, 0] <= 1.0)
                & (uv[local_idx, :, 1] >= 0.0)
                & (uv[local_idx, :, 1] <= 1.0)
                & (confidence[local_idx] >= float(confidence_threshold))
                & (visibility[local_idx] >= float(visibility_threshold))
            )
            raw_slots = int(ok.shape[0])
            valid_slots = int(np.count_nonzero(ok))
            counters["raw_slot_count"] += raw_slots
            counters["valid_slot_count"] += valid_slots
            if valid_slots == 0:
                frame_rows.append(
                    {
                        "carrier_npz": _rel(carrier_path),
                        "frame_id": int(frame_id),
                        "raw_slot_count": raw_slots,
                        "valid_slot_count": valid_slots,
                        "mask_assigned_point_count": 0,
                        "nn_hit_count": 0,
                    }
                )
                continue
            mask_path = mask_dir / f"{int(frame_id)}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                missing_mask_frames.add(int(frame_id))
                frame_rows.append(
                    {
                        "carrier_npz": _rel(carrier_path),
                        "frame_id": int(frame_id),
                        "raw_slot_count": raw_slots,
                        "valid_slot_count": valid_slots,
                        "mask_assigned_point_count": 0,
                        "nn_hit_count": 0,
                        "missing_mask": True,
                    }
                )
                continue
            cur_xyz = xyz[local_idx, ok]
            cur_uv = uv[local_idx, ok]
            owners = _sample_mask_owner(mask=mask, uv=cur_uv, frame_id=int(frame_id), mask_to_object_idx=mask_to_object_idx)
            assigned = owners > 0
            assigned_count = int(np.count_nonzero(assigned))
            counters["mask_assigned_point_count"] += assigned_count
            if assigned_count == 0:
                frame_rows.append(
                    {
                        "carrier_npz": _rel(carrier_path),
                        "frame_id": int(frame_id),
                        "raw_slot_count": raw_slots,
                        "valid_slot_count": valid_slots,
                        "mask_assigned_point_count": 0,
                        "nn_hit_count": 0,
                        "missing_mask": False,
                    }
                )
                continue
            assigned_xyz = cur_xyz[assigned]
            assigned_owners = owners[assigned]
            dist, vertex_idx = tree.query(assigned_xyz, k=1, distance_upper_bound=float(nn_radius))
            hit = np.isfinite(dist) & (vertex_idx >= 0) & (vertex_idx < scene_points.shape[0])
            hit_count = int(np.count_nonzero(hit))
            counters["nn_hit_count"] += hit_count
            for object_idx in np.unique(assigned_owners):
                if int(object_idx) <= 0:
                    continue
                owner_sel = assigned_owners == int(object_idx)
                object_assigned_points[int(object_idx)] += int(np.count_nonzero(owner_sel))
                owner_hit = owner_sel & hit
                if not np.any(owner_hit):
                    continue
                verts = vertex_idx[owner_hit].astype(np.int64, copy=False)
                object_point_hits[int(object_idx)] += int(verts.shape[0])
                object_vertices[int(object_idx)].update(int(v) for v in verts.tolist())
            frame_rows.append(
                {
                    "carrier_npz": _rel(carrier_path),
                    "frame_id": int(frame_id),
                    "raw_slot_count": raw_slots,
                    "valid_slot_count": valid_slots,
                    "mask_assigned_point_count": assigned_count,
                    "nn_hit_count": hit_count,
                    "missing_mask": False,
                }
            )

    kept_objects: list[tuple[int, str, np.ndarray, int, float]] = []
    dropped_tiny_objects = 0
    for object_idx, vertices in sorted(object_vertices.items()):
        vertex_array = np.asarray(sorted(vertices), dtype=np.int64)
        if vertex_array.shape[0] < int(min_vertices):
            dropped_tiny_objects += 1
            continue
        point_hits = int(object_point_hits[object_idx])
        score = _score_object(point_hits, int(vertex_array.shape[0]), score_mode)
        kept_objects.append((object_idx, object_idx_to_id.get(object_idx, str(object_idx)), vertex_array, point_hits, score))

    masks = np.zeros((scene_points.shape[0], len(kept_objects)), dtype=bool)
    scores = np.zeros((len(kept_objects),), dtype=np.float32)
    classes = np.zeros((len(kept_objects),), dtype=np.int32)
    object_rows: list[dict[str, Any]] = []
    vertex_owner_counter: Counter[int] = Counter()
    for out_idx, (object_idx, object_id, vertices, point_hits, score) in enumerate(kept_objects):
        masks[vertices, out_idx] = True
        scores[out_idx] = float(score)
        vertex_owner_counter.update(int(v) for v in vertices.tolist())
        object_rows.append(
            {
                "output_object_index": int(out_idx),
                "pipeline_object_idx": int(object_idx),
                "pipeline_object_id": object_id,
                "assigned_d4rt_point_count": int(object_assigned_points[object_idx]),
                "nn_hit_d4rt_point_count": int(point_hits),
                "mesh_vertex_count": int(vertices.shape[0]),
                "score": float(score),
            }
        )

    pre_points = np.flatnonzero(np.any(masks, axis=1)).astype(np.int64)
    diag = {
        "mesh_path": _rel(mesh_path),
        "mesh_vertex_count": int(scene_points.shape[0]),
        "carrier_cache_window_count": int(carrier_count),
        **geometry_diag,
        "raw_slot_count": int(counters["raw_slot_count"]),
        "valid_slot_count": int(counters["valid_slot_count"]),
        "mask_assigned_point_count": int(counters["mask_assigned_point_count"]),
        "nn_hit_count": int(counters["nn_hit_count"]),
        "nn_hit_ratio_over_assigned": float(counters["nn_hit_count"] / max(counters["mask_assigned_point_count"], 1)),
        "exported_object_count": int(len(kept_objects)),
        "dropped_tiny_object_count": int(dropped_tiny_objects),
        "union_pre_points_count": int(pre_points.shape[0]),
        "union_pre_points_ratio": float(pre_points.shape[0] / max(scene_points.shape[0], 1)),
        "multi_owner_vertex_count": int(sum(1 for count in vertex_owner_counter.values() if count > 1)),
        "missing_mask_frame_count": int(len(missing_mask_frames)),
        "missing_mask_frames_first20": sorted(missing_mask_frames)[:20],
        "confidence_threshold": float(confidence_threshold),
        "visibility_threshold": float(visibility_threshold),
        "nn_radius": float(nn_radius),
        "min_vertices": int(min_vertices),
        "score_mode": score_mode,
    }
    return masks, scores, classes, pre_points, object_rows, frame_rows, diag


def _write_prediction_files(
    *,
    output_config: str,
    scene: str,
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    pre_points: np.ndarray,
) -> dict[str, str]:
    pred_dir = ROOT / "data" / "prediction" / f"{output_config}_class_agnostic"
    tmp_dir = ROOT / "data" / "TMP" / output_config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{scene}.npz"
    tmp_path = tmp_dir / f"{scene}_pre_points.npy"
    np.savez_compressed(
        pred_path,
        pred_masks=masks.astype(bool, copy=False),
        pred_score=scores.astype(np.float32, copy=False),
        pred_classes=classes.astype(np.int32, copy=False),
    )
    np.save(tmp_path, pre_points.astype(np.int64, copy=False))
    return {"prediction_npz": _rel(pred_path), "pre_points_npy": _rel(tmp_path)}


def _write_manifest(args: argparse.Namespace, summary_path: Path, source_hashes: dict[str, str]) -> list[str]:
    manifest = build_prediction_manifest(
        output_config=args.output_config,
        root=ROOT,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.pipeline_root],
        pre_points_policy="union_of_D4RT_NN_mesh_vertices_from_pipeline_object_support",
        support_policy=f"D4RT {args.d4rt_coordinate_mode} nearest-neighbor to ScanNet evaluator mesh vertices; diagnostic AP adapter",
        notes=(
            "Pipeline-root AP adapter. SOMA object support is read only from the supplied full-scene pipeline root. "
            f"Prediction geometry is D4RT {args.d4rt_coordinate_mode}, then materialized to ScanNet evaluator vertices by NN. "
            "This uses ScanNet mesh for export/evaluation adapter only and is forbidden for fair method tables."
        ),
        extra={
            "algorithm": "v65_pipeline_d4rt_nn_ap",
            "prediction_config": args.output_config,
            "class_setting": "class_agnostic",
            "support_scope": "PREDICTION_UNION_ISLAND",
            "support_policy": "union_of_D4RT_NN_mesh_vertices_from_pipeline_object_support",
            "gt_crop_full_policy": "GT labels and prediction masks are cropped by evaluator TMP pre_points; this is not FULLMESH.",
            "score_protocol": args.score_mode,
            "comparison_status": "not_comparable",
            "support_source": "v65_soma_fullscene_pipeline_root",
            "geometry_source": f"pipeline D4RT geometry mode {args.d4rt_coordinate_mode}",
            "semantic_source": "pipeline chunked CropFormer masks",
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": False,
            "uses_scannet_mesh_for_export": True,
            "uses_mesh_nearest_neighbor_for_export": True,
            "uses_gt_for_diagnostic": bool(args.d4rt_coordinate_mode == "chunk_final_gt_sim3"),
            "alignment_source": "ScanNet depth/pose final_gt_sim3" if args.d4rt_coordinate_mode == "chunk_final_gt_sim3" else "D4RT chunk self-stitch",
            "alignment_used_for_diagnostic": bool(args.d4rt_coordinate_mode == "chunk_final_gt_sim3"),
            "uses_rgbd_pose_mesh_for_export": True,
            "forbidden_for_method_table": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "eval_policy": "ScanNet class-agnostic evaluator with TMP support equal to exported union vertices",
            "pipeline_root": _rel(args.pipeline_root),
            "summary_path": _rel(summary_path),
            "source_hashes": source_hashes,
        },
    )
    paths = write_prediction_manifest(args.output_config, manifest, root=ROOT, pred_suffix="class_agnostic")
    return [_rel(path) for path in paths]


def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    output_file = ROOT / "data" / "evaluation" / "scannet" / f"{args.output_config}_class_agnostic.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        f"data/prediction/{args.output_config}_class_agnostic",
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        str(output_file.relative_to(ROOT)),
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        args.output_config,
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    if args.eval_gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpus)
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    parsed = _parse_eval_file(output_file) if output_file.exists() else {}
    return {
        "command": " ".join(cmd),
        "cwd": str(ROOT),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(elapsed),
        "output_file": _rel(output_file),
        "output_file_sha256": _sha256(output_file),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "metrics": parsed,
    }


def _parse_eval_file(path: Path) -> dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return {}
    last = lines[-1].split(",")
    if len(last) == 3:
        try:
            return {"all_ap": float(last[0]), "all_ap_50": float(last[1]), "all_ap_25": float(last[2])}
        except ValueError:
            return {}
    return {}


def _compute_support_iou_diagnostics(
    *,
    scene: str,
    masks: np.ndarray,
    pre_points: np.ndarray,
    materialization_diag: dict[str, Any],
) -> dict[str, Any]:
    gt_ids_full = np.asarray(_load_gt(scene))
    if gt_ids_full.shape[0] != masks.shape[0]:
        raise ValueError(
            f"GT/mask vertex count mismatch for {scene}: gt={gt_ids_full.shape[0]} masks={masks.shape[0]}"
        )
    support = np.asarray(pre_points, dtype=np.int64)
    support = support[(support >= 0) & (support < gt_ids_full.shape[0])]
    support = np.unique(support)
    full_gt_instance_ids = np.unique(gt_ids_full[gt_ids_full > 0])
    gt_support_ids = np.unique(gt_ids_full[support][gt_ids_full[support] > 0]) if support.size else np.asarray([], dtype=gt_ids_full.dtype)

    diag: dict[str, Any] = {
        "iou_formula": "IoU_Omega(P_i,G_j)=|P_i & G_j & Omega|/|(P_i | G_j) & Omega|",
        "omega_support_scope": "PREDICTION_UNION_ISLAND",
        "omega_pre_points_count": int(support.shape[0]),
        "prediction_count": int(masks.shape[1]),
        "gt_instance_count_in_omega": int(gt_support_ids.shape[0]),
        "full_scene_gt_instance_count": int(full_gt_instance_ids.shape[0]),
        "gt_instance_count_mean": float(gt_support_ids.shape[0]),
        "full_scene_gt_instance_count_mean": float(full_gt_instance_ids.shape[0]),
    }
    if support.size == 0 or masks.shape[1] == 0 or gt_support_ids.shape[0] == 0:
        diag.update(
            {
                "per_GT_best_IoU_mean": None,
                "pred_best_IoU_median": None,
                "gt_best_IoU_median": None,
                "duplicate_predictions_per_GT": None,
                "duplicate_predictions_per_GT_mean": None,
                "conflict_rate": None,
                "metric_status": "not_computed_empty_support_or_prediction_or_gt",
            }
        )
        return diag

    pred_support = np.asarray(masks[support, :], dtype=bool)
    gt_support = gt_ids_full[support]
    pred_areas = pred_support.sum(axis=0).astype(np.float64)
    ious = np.zeros((int(pred_support.shape[1]), int(gt_support_ids.shape[0])), dtype=np.float64)
    for gt_col, gt_id in enumerate(gt_support_ids.tolist()):
        gt_mask = gt_support == gt_id
        gt_area = float(np.count_nonzero(gt_mask))
        if gt_area <= 0:
            continue
        inter = pred_support[gt_mask, :].sum(axis=0).astype(np.float64)
        union = pred_areas + gt_area - inter
        valid = union > 0
        ious[valid, gt_col] = inter[valid] / union[valid]

    pred_best = ious.max(axis=1) if ious.size else np.asarray([], dtype=np.float64)
    gt_best = ious.max(axis=0) if ious.size else np.asarray([], dtype=np.float64)
    duplicate_by_gt = np.maximum((ious >= 0.5).sum(axis=0).astype(np.int64) - 1, 0)
    union_count = max(int(materialization_diag.get("union_pre_points_count", support.shape[0]) or support.shape[0]), 1)
    conflict_rate = float(int(materialization_diag.get("multi_owner_vertex_count", 0) or 0) / union_count)
    diag.update(
        {
            "per_GT_best_IoU_mean": float(np.mean(gt_best)) if gt_best.size else None,
            "pred_best_IoU_median": float(np.median(pred_best)) if pred_best.size else None,
            "gt_best_IoU_median": float(np.median(gt_best)) if gt_best.size else None,
            "duplicate_predictions_per_GT": duplicate_by_gt.astype(int).tolist(),
            "duplicate_predictions_per_GT_mean": float(np.mean(duplicate_by_gt)) if duplicate_by_gt.size else None,
            "conflict_rate": conflict_rate,
            "conflict_rate_definition": "multi_owner_vertex_count / union_pre_points_count",
            "metric_status": "computed",
        }
    )
    return diag


def _build_ap_condition_declaration(
    *,
    args: argparse.Namespace,
    objectlet_variant: str,
    support_diag: dict[str, Any],
    materialization_diag: dict[str, Any],
    iou_diag: dict[str, Any],
    prediction_paths: dict[str, str],
    pred_hashes: dict[str, str],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
    prediction_count = int(materialization_diag.get("exported_object_count", 0) or 0)
    pre_points_count = int(materialization_diag.get("union_pre_points_count", 0) or 0)
    support_policy = "union_of_D4RT_NN_mesh_vertices_from_pipeline_object_support"
    support_scope = "PREDICTION_UNION_ISLAND"
    return {
        "row_id": args.output_config,
        "method_name": "SOMA full-scene pipeline + D4RT stride-5 NN diagnostic AP",
        "split": f"ScanNet single-scene diagnostic: {args.scene}",
        "scene_count": 1,
        "scene_ids": [args.scene],
        "class_setting": "class_agnostic",
        "evaluator_name": "Stream3D evaluation.evaluate ScanNet class-agnostic evaluator",
        "evaluator_command": eval_result.get("command", "") if isinstance(eval_result, dict) else "",
        "evaluator_output_file": eval_result.get("output_file", "") if isinstance(eval_result, dict) else "",
        "evaluator_output_hash": eval_result.get("output_file_sha256", "") if isinstance(eval_result, dict) else "",
        "evaluator_min_region_threshold": 100,
        "support_scope": support_scope,
        "support_policy": support_policy,
        "support_policy_hash": pred_hashes.get("pre_points_npy", ""),
        "support_policy_hash_source": prediction_paths.get("pre_points_npy", ""),
        "pre_points_policy": support_policy,
        "pre_points_file": prediction_paths.get("pre_points_npy", ""),
        "pre_points_file_hash": pred_hashes.get("pre_points_npy", ""),
        "pre_points_count_mean": float(pre_points_count),
        "pre_points_count_min": pre_points_count,
        "pre_points_count_max": pre_points_count,
        "gt_crop_full_policy": "GT labels and prediction masks are cropped by evaluator TMP pre_points; this is not FULLMESH.",
        "gt_instance_count_mean": iou_diag.get("gt_instance_count_mean"),
        "full_scene_gt_instance_count_mean": iou_diag.get("full_scene_gt_instance_count_mean"),
        "prediction_count": prediction_count,
        "mean_predictions_per_scene": float(prediction_count),
        "prediction_union_ratio": materialization_diag.get("union_pre_points_ratio"),
        "prediction_union_inside_support_ratio": 1.0 if pre_points_count > 0 else None,
        "AP": metrics.get("all_ap"),
        "AP50": metrics.get("all_ap_50"),
        "AP25": metrics.get("all_ap_25"),
        "per_GT_best_IoU_mean": iou_diag.get("per_GT_best_IoU_mean"),
        "pred_best_IoU_median": iou_diag.get("pred_best_IoU_median"),
        "gt_best_IoU_median": iou_diag.get("gt_best_IoU_median"),
        "duplicate_predictions_per_GT": iou_diag.get("duplicate_predictions_per_GT_mean"),
        "duplicate_predictions_per_GT_detail": iou_diag.get("duplicate_predictions_per_GT"),
        "conflict_rate": iou_diag.get("conflict_rate"),
        "score_protocol": str(args.score_mode),
        "score_protocol_detail": "prediction score = point_hits * sqrt(mesh_vertex_count) unless --score-mode changes it",
        "objectlet_variant": objectlet_variant,
        "success_only": bool(args.success_only),
        "confidence_threshold": float(args.confidence_threshold),
        "visibility_threshold": float(args.visibility_threshold),
        "nn_radius": float(args.nn_radius),
        "min_vertices": int(args.min_vertices),
        "d4rt_coordinate_mode": str(args.d4rt_coordinate_mode),
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": True,
        "uses_rgbd_pose_mesh_for_export": True,
        "uses_scannet_mesh_for_export": True,
        "uses_final_gt_sim3_for_export": bool(args.d4rt_coordinate_mode == "chunk_final_gt_sim3"),
        "forbidden_for_method_table": True,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "comparison_status": "not_comparable",
        "not_comparable_reasons": [
            "support_scope is PREDICTION_UNION_ISLAND",
            "export materializes D4RT points to ScanNet evaluator mesh vertices by NN",
            "d4rt_coordinate_mode chunk_final_gt_sim3 uses ScanNet RGB-D/pose final Sim3 when selected",
            "no Stream3D row with same split/evaluator/support/class setting is bundled in this AP row",
        ],
        "comparison_allowed_conditions": {
            "support_scope_same": False,
            "evaluator_same": None,
            "split_same": None,
            "class_setting_same": None,
            "score_policy_documented": True,
        },
        "support_diag_summary": {
            "support_pair_count": support_diag.get("support_pair_count"),
            "support_frame_count": support_diag.get("support_frame_count"),
            "duplicate_frame_mask_conflicts": support_diag.get("duplicate_frame_mask_conflicts"),
        },
        "iou_diagnostics": iou_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v65 full-scene pipeline D4RT/SOMA support to ScanNet AP via NN adapter.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--output-config", default="v65_pipeline_d4rt_nn_scene0050_stride5_conf02_r008")
    parser.add_argument("--output-root", default="outputs/audit/v65_pipeline_d4rt_nn_ap_scene0050_stride5_conf02_r008")
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--objectlet-variant", default="best")
    parser.add_argument("--success-only", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--visibility-threshold", type=float, default=0.0)
    parser.add_argument("--nn-radius", type=float, default=0.08)
    parser.add_argument("--min-vertices", type=int, default=1)
    parser.add_argument("--score-mode", choices=["point_hits", "vertex_count", "point_hits_sqrt_vertices"], default="point_hits_sqrt_vertices")
    parser.add_argument("--d4rt-coordinate-mode", choices=list(D4RT_COORDINATE_MODES), default="chunk_final_gt_sim3")
    parser.add_argument("--d4rt-stride-summary", default="")
    parser.add_argument("--run-eval", type=int, default=1)
    parser.add_argument("--eval-gpus", default="6,7")
    args = parser.parse_args()

    pipeline_root = _project(args.pipeline_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_summary_path = pipeline_root / "pipeline_summary.json"
    pipeline_summary = _read_json(pipeline_summary_path)
    mask_dir = _resolve_pipeline_mask_dir(
        scene=args.scene,
        pipeline_summary=pipeline_summary,
        override_mask_root=args.mask_root if args.mask_root else None,
    )
    objectlet_variant = _best_objectlet_variant(pipeline_root, args.objectlet_variant)
    mask_to_object_idx, object_idx_to_id, support_diag = _load_pipeline_support(
        pipeline_root=pipeline_root,
        scene=args.scene,
        objectlet_variant=objectlet_variant,
        success_only=bool(args.success_only),
    )
    masks, scores, classes, pre_points, object_rows, frame_rows, materialization_diag = _materialize_predictions(
        scene=args.scene,
        pipeline_root=pipeline_root,
        mask_dir=mask_dir,
        mask_to_object_idx=mask_to_object_idx,
        object_idx_to_id=object_idx_to_id,
        confidence_threshold=float(args.confidence_threshold),
        visibility_threshold=float(args.visibility_threshold),
        nn_radius=float(args.nn_radius),
        min_vertices=int(args.min_vertices),
        score_mode=args.score_mode,
        d4rt_coordinate_mode=str(args.d4rt_coordinate_mode),
        d4rt_stride_summary=str(args.d4rt_stride_summary),
    )
    prediction_paths = _write_prediction_files(
        output_config=args.output_config,
        scene=args.scene,
        masks=masks,
        scores=scores,
        classes=classes,
        pre_points=pre_points,
    )

    object_rows_path = output_root / "d4rt_nn_ap_object_rows.csv"
    frame_rows_path = output_root / "d4rt_nn_ap_frame_rows.csv"
    summary_path = output_root / "d4rt_nn_ap_summary.json"
    _write_csv(object_rows_path, object_rows)
    _write_csv(frame_rows_path, frame_rows)

    source_hashes = {
        "pipeline_summary": _sha256(pipeline_summary_path),
        "mask_materialization_final_hashes": _sha256(pipeline_root / "mask_materialization_final_mask_hashes.csv"),
        "d4rt_geometry_materialization_summary": _sha256(pipeline_root / "d4rt_geometry_materialization_summary.json"),
        "local_objectlet_summary": _sha256(pipeline_root / "local_objectlets" / "local_objectlet_summary.json"),
        "objectlet_rows": _sha256(pipeline_root / "local_objectlets" / "objectlet_rows.csv"),
        "reprojection_summary": _sha256(pipeline_root / "reprojection_ledger" / "reprojection_summary.json"),
        "reprojection_ledger_rows": _sha256(pipeline_root / "reprojection_ledger" / "reprojection_ledger_rows.csv"),
    }
    manifest_paths = _write_manifest(args, summary_path, source_hashes)
    pred_hashes = {
        "prediction_npz": _sha256(prediction_paths["prediction_npz"]),
        "pre_points_npy": _sha256(prediction_paths["pre_points_npy"]),
        "prediction_manifest": _sha256(manifest_paths[0]) if manifest_paths else "",
        "tmp_manifest": _sha256(manifest_paths[1]) if len(manifest_paths) > 1 else "",
        "object_rows": _sha256(object_rows_path),
        "frame_rows": _sha256(frame_rows_path),
    }
    eval_result = _run_eval(args) if int(args.run_eval) else {"skipped": True}
    iou_diag = _compute_support_iou_diagnostics(
        scene=args.scene,
        masks=masks,
        pre_points=pre_points,
        materialization_diag=materialization_diag,
    )
    ap_condition_declaration = _build_ap_condition_declaration(
        args=args,
        objectlet_variant=objectlet_variant,
        support_diag=support_diag,
        materialization_diag=materialization_diag,
        iou_diag=iou_diag,
        prediction_paths=prediction_paths,
        pred_hashes=pred_hashes,
        eval_result=eval_result,
    )

    summary = {
        "phase": "v65_pipeline_d4rt_nn_ap",
        "scene": args.scene,
        "pipeline_root": _rel(pipeline_root),
        "pipeline_summary": _rel(pipeline_summary_path),
        "pipeline_summary_sha256": source_hashes["pipeline_summary"],
        "pipeline_gate": pipeline_summary.get("pipeline_gate", {}),
        "mask_dir": _rel(mask_dir),
        "mask_coverage": pipeline_summary.get("mask_frame_coverage", {}),
        "objectlet_variant": objectlet_variant,
        "success_only": bool(args.success_only),
        "confidence_threshold": float(args.confidence_threshold),
        "visibility_threshold": float(args.visibility_threshold),
        "nn_radius": float(args.nn_radius),
        "d4rt_coordinate_mode": str(args.d4rt_coordinate_mode),
        "d4rt_stride_summary": str(args.d4rt_stride_summary),
        "support_diag": support_diag,
        "materialization_diag": materialization_diag,
        "prediction_paths": prediction_paths,
        "manifest_paths": manifest_paths,
        "object_rows": _rel(object_rows_path),
        "frame_rows": _rel(frame_rows_path),
        "source_hashes": source_hashes,
        "prediction_hashes": pred_hashes,
        "eval_result": eval_result,
        "support_iou_diagnostics": iou_diag,
        "ap_condition_declaration": ap_condition_declaration,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "contract_note": (
            "This AP row reads SOMA/D4RT predictions from one validated full-scene stride-5 pipeline root, "
            f"but uses ScanNet mesh nearest-neighbor only to adapt D4RT {args.d4rt_coordinate_mode} points to evaluator vertex ids."
        ),
        "gate": {
            "pipeline_ap_ready": bool(pipeline_summary.get("pipeline_gate", {}).get("ap_ready")),
            "mask_full_stride_coverage": float(pipeline_summary.get("mask_frame_coverage", {}).get("coverage_ratio", 0.0) or 0.0) == 1.0,
            "support_pairs_available": int(support_diag.get("support_pair_count", 0)) > 0,
            "exported_objects_available": int(materialization_diag.get("exported_object_count", 0)) > 0,
            "union_pre_points_available": int(materialization_diag.get("union_pre_points_count", 0)) > 0,
            "eval_returncode_zero": bool(eval_result.get("returncode", 0) == 0) if not eval_result.get("skipped") else True,
            "ap_condition_support_scope_present": bool(ap_condition_declaration.get("support_scope")),
            "ap_condition_manifest_present": bool(manifest_paths),
            "ap_condition_evaluator_output_hash_present": bool(ap_condition_declaration.get("evaluator_output_hash")),
            "ap_condition_diagnostic_forbidden_for_method_table": bool(
                ap_condition_declaration.get("is_diagnostic_only")
                and ap_condition_declaration.get("forbidden_for_method_table")
                and not ap_condition_declaration.get("is_method_result")
            ),
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": _rel(summary_path),
                "output_config": args.output_config,
                "pipeline_summary_sha256": summary["pipeline_summary_sha256"],
                "objectlet_variant": objectlet_variant,
                "exported_object_count": materialization_diag["exported_object_count"],
                "union_pre_points_count": materialization_diag["union_pre_points_count"],
                "eval_metrics": eval_result.get("metrics", {}),
                "gate": summary["gate"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
