#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v104_lingbot_map_only_phase10_temporal_track_local_mv_ap"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_FEATURE_ROOT = AUDIT_ROOT / "v104_lingbot_map_only_phase8_voxel_centroid_sigma050"
DEFAULT_PHASE2_SCENE0011 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_PHASE2_SCENE0050 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"
DEFAULT_BASELINE_ROWS = AUDIT_ROOT / "v103_phase0_contract/baseline_metric_rows.csv"

if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))
from tools import build_v103_phase6_mask_clustering_local_object_birth as eval_base  # noqa: E402


VARIANTS = [
    {
        "variant_id": "T0_temporal_lingbot_3d2d_s3d025_s2d006_thr055_gap1_min2",
        "sigma_3d": 0.25,
        "sigma_2d": 0.06,
        "sigma_log_area": 0.70,
        "w3d": 0.50,
        "w2d": 0.30,
        "warea": 0.20,
        "threshold": 0.55,
        "max_gap": 1,
        "min_frames": 2,
        "non_broad_only": False,
    },
    {
        "variant_id": "T1_temporal_lingbot_3d2d_s3d040_s2d008_thr050_gap1_min2",
        "sigma_3d": 0.40,
        "sigma_2d": 0.08,
        "sigma_log_area": 0.85,
        "w3d": 0.45,
        "w2d": 0.35,
        "warea": 0.20,
        "threshold": 0.50,
        "max_gap": 1,
        "min_frames": 2,
        "non_broad_only": False,
    },
    {
        "variant_id": "T2_temporal_lingbot_3d2d_nonbroad_s3d040_s2d008_thr045_gap1_min2",
        "sigma_3d": 0.40,
        "sigma_2d": 0.08,
        "sigma_log_area": 0.85,
        "w3d": 0.45,
        "w2d": 0.35,
        "warea": 0.20,
        "threshold": 0.45,
        "max_gap": 1,
        "min_frames": 2,
        "non_broad_only": True,
    },
    {
        "variant_id": "T3_temporal_lingbot_3d2d_s3d060_s2d010_thr045_gap2_min2",
        "sigma_3d": 0.60,
        "sigma_2d": 0.10,
        "sigma_log_area": 1.00,
        "w3d": 0.40,
        "w2d": 0.40,
        "warea": 0.20,
        "threshold": 0.45,
        "max_gap": 2,
        "min_frames": 2,
        "non_broad_only": False,
    },
    {
        "variant_id": "T4_temporal_lingbot_2dshape_control_s2d008_thr050_gap1_min2",
        "sigma_3d": 1.0,
        "sigma_2d": 0.08,
        "sigma_log_area": 0.85,
        "w3d": 0.00,
        "w2d": 0.70,
        "warea": 0.30,
        "threshold": 0.50,
        "max_gap": 1,
        "min_frames": 2,
        "non_broad_only": False,
    },
    {
        "variant_id": "T5_temporal_lingbot_3d_only_s3d040_thr050_gap1_min2",
        "sigma_3d": 0.40,
        "sigma_2d": 1.0,
        "sigma_log_area": 1.0,
        "w3d": 1.00,
        "w2d": 0.00,
        "warea": 0.00,
        "threshold": 0.50,
        "max_gap": 1,
        "min_frames": 2,
        "non_broad_only": False,
    },
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return np.empty((0, 0), dtype=np.int64)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _mask_stats(label: np.ndarray, mask_id: int) -> tuple[int, float, float]:
    pixels = label == int(mask_id)
    y, x = np.nonzero(pixels)
    if x.size == 0:
        return 0, 0.0, 0.0
    h, w = label.shape[:2]
    return int(x.size), float(np.mean(x) / max(w - 1, 1)), float(np.mean(y) / max(h - 1, 1))


def _baseline(path: Path) -> dict[str, float]:
    rows = _read_csv(path)
    for row in rows:
        if row.get("baseline_role") == "current_strong_local_baseline" and row.get("dataset_split") == "dev":
            return {
                "MV_AP_window": _num(row.get("MV_AP_window")),
                "MV_AP50_window": _num(row.get("MV_AP50_window")),
                "MV_AP25_window": _num(row.get("MV_AP25_window")),
                "ScoreFreeMatch50_window": _num(row.get("ScoreFreeMatch50_window")),
            }
    return {}


def _enrich_observations(feature_root: Path, phase2_summaries: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = _read_csv(feature_root / "mask_observation_rows.csv")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    for row in rows:
        scene = row.get("scene_id", "")
        if scene not in phase2_summaries:
            continue
        source_frame = int(_num(row.get("source_frame_id"), -1))
        mask_id = int(_num(row.get("mask_id"), -1))
        summary = phase2_summaries[scene]
        mask_root = _project(summary.get("mask_root", ""))
        key = (scene, source_frame)
        if key not in label_cache:
            label_cache[key] = _load_label(mask_root / f"{source_frame}.png")
        area, cx, cy = _mask_stats(label_cache[key], mask_id)
        if area <= 0:
            continue
        obs = {
            "scene_id": scene,
            "source_frame_id": source_frame,
            "frame_local_index": int(_num(row.get("frame_local_index"), -1)),
            "mask_id": mask_id,
            "mask_area": area,
            "cx": cx,
            "cy": cy,
            "centroid": np.asarray(
                [
                    _num(row.get("centroid_x")),
                    _num(row.get("centroid_y")),
                    _num(row.get("centroid_z")),
                ],
                dtype=np.float32,
            ),
            "support_point_count": int(_num(row.get("support_point_count"), 0)),
            "mask_is_broad": _as_bool(row.get("mask_is_broad", "False")),
            "mask_is_object_like": _as_bool(row.get("mask_is_object_like", "True")),
            "history_id": row.get("history_id", ""),
            "candidate_row_id": row.get("candidate_row_id", ""),
        }
        by_scene[scene].append(obs)
    for scene in by_scene:
        by_scene[scene].sort(key=lambda r: (int(r["frame_local_index"]), int(r["mask_id"]), int(_num(r.get("candidate_row_id"), 0))))
    return by_scene


def _link_score(a: dict[str, Any], b: dict[str, Any], variant: dict[str, Any]) -> float:
    d3 = float(np.linalg.norm(np.asarray(a["centroid"], dtype=np.float32) - np.asarray(b["centroid"], dtype=np.float32)))
    d2 = math.hypot(float(a["cx"]) - float(b["cx"]), float(a["cy"]) - float(b["cy"]))
    ar = math.log((float(b["mask_area"]) + 1.0) / (float(a["mask_area"]) + 1.0))
    s3 = math.exp(-d3 / max(float(variant["sigma_3d"]), 1e-6))
    s2 = math.exp(-d2 / max(float(variant["sigma_2d"]), 1e-6))
    sa = math.exp(-abs(ar) / max(float(variant["sigma_log_area"]), 1e-6))
    score = float(variant["w3d"]) * s3 + float(variant["w2d"]) * s2 + float(variant["warea"]) * sa
    if bool(b.get("mask_is_broad", False)) and not bool(b.get("mask_is_object_like", False)):
        score *= 0.75
    return float(score)


def _track_scene(scene: str, observations: list[dict[str, Any]], variant: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        if bool(variant.get("non_broad_only", False)) and bool(obs.get("mask_is_broad", False)):
            continue
        frames[int(obs["frame_local_index"])].append(obs)

    tracks: list[dict[str, Any]] = []
    active: list[int] = []
    edge_rows: list[dict[str, Any]] = []
    threshold = float(variant["threshold"])
    max_gap = int(variant["max_gap"])

    for frame in sorted(frames):
        candidates = frames[frame]
        pairs: list[tuple[float, int, int]] = []
        for ti in list(active):
            track = tracks[ti]
            gap = int(frame) - int(track["last_frame"])
            if gap <= 0 or gap > max_gap:
                continue
            last = track["last_obs"]
            for oi, obs in enumerate(candidates):
                score = _link_score(last, obs, variant)
                if score >= threshold:
                    pairs.append((score, ti, oi))
        pairs.sort(key=lambda item: item[0], reverse=True)
        used_tracks: set[int] = set()
        used_obs: set[int] = set()
        for rank, (score, ti, oi) in enumerate(pairs):
            if ti in used_tracks or oi in used_obs:
                continue
            obs = candidates[oi]
            track = tracks[ti]
            track["observations"].append(obs)
            track["last_obs"] = obs
            track["last_frame"] = int(frame)
            track["link_scores"].append(float(score))
            used_tracks.add(ti)
            used_obs.add(oi)
            edge_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_temporal_track_edge_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant["variant_id"],
                    "scene_id": scene,
                    "edge_rank": rank,
                    "track_index": ti,
                    "from_frame": int(track["observations"][-2]["source_frame_id"]),
                    "to_frame": int(obs["source_frame_id"]),
                    "from_mask_id": int(track["observations"][-2]["mask_id"]),
                    "to_mask_id": int(obs["mask_id"]),
                    "link_score": float(score),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        for oi, obs in enumerate(candidates):
            if oi in used_obs:
                continue
            tracks.append({"observations": [obs], "last_obs": obs, "last_frame": int(frame), "link_scores": []})
        active = [idx for idx, track in enumerate(tracks) if int(frame) - int(track["last_frame"]) <= max_gap]

    object_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    object_idx = 0
    for ti, track in enumerate(tracks):
        obs_rows = track["observations"]
        if len(obs_rows) < int(variant["min_frames"]):
            continue
        object_id = f"{variant['variant_id']}:{scene}:c0000:obj_{object_idx:05d}"
        object_idx += 1
        link_scores = track["link_scores"]
        mean_link = float(np.mean(link_scores)) if link_scores else 0.0
        broad_ratio = float(np.mean([bool(o.get("mask_is_broad", False)) for o in obs_rows]))
        frame_denominator = max(float(variant.get("score_frame_denominator", 32.0)), 1.0)
        score = (
            float(len(obs_rows) / frame_denominator)
            * max(0.05, mean_link if mean_link > 0 else 0.25)
            * max(0.10, 1.0 - 0.50 * broad_ratio)
        )
        object_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_temporal_track_object_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant["variant_id"],
                "scene_id": scene,
                "mv_object_id": object_id,
                "object_id": object_id,
                "track_index": ti,
                "frame_count": len(obs_rows),
                "mask_count": len(obs_rows),
                "mean_link_score": mean_link,
                "broad_mask_ratio": broad_ratio,
                "object_score": score,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for obs in obs_rows:
            frame_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_temporal_track_frame_mask_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant["variant_id"],
                    "mv_object_id": object_id,
                    "object_id": object_id,
                    "scene_id": scene,
                    "chunk_id": "c0000",
                    "window_id": "c0000",
                    "frame_local_index": int(obs["frame_local_index"]),
                    "selected_mask_id": int(obs["mask_id"]),
                    "mask_id_or_generated_id": int(obs["mask_id"]),
                    "object_score": score,
                    "score": score,
                    "support_count": int(obs["support_point_count"]),
                    "selected_mask_is_broad": bool(obs["mask_is_broad"]),
                    "selected_mask_is_object_like": bool(obs["mask_is_object_like"]),
                    "readout_mode": "v104_lingbot_temporal_track",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return object_rows, frame_rows, edge_rows


def _evaluate_variant_scene(
    *,
    variant_id: str,
    scene_rows: dict[str, list[dict[str, Any]]],
    phase2_summaries: dict[str, dict[str, Any]],
    min_pred_pixels: int,
    min_gt_pixels: int,
    use_cupy_iou: bool,
    cupy_device_id: int,
    scene_metric_scope: str = "first32_scene_raw_gt",
    scene_aggregate_scope: str = "first32_scene_raw_gt_mean",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], int, int, int]:
    scene_metric_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    missing_mask_raster_count = 0
    backend_used = ""
    for scene, rows in sorted(scene_rows.items()):
        summary = phase2_summaries[scene]
        frame_ids = [int(v) for v in summary["frame_ids"]]
        mask_root = _project(summary["mask_root"])
        acc, backend = eval_base._accumulator(use_cupy_iou, cupy_device_id)
        backend_used = backend
        object_index: dict[str, int] = {}
        scores: dict[str, float] = {}
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            local = int(row["frame_local_index"])
            if 0 <= local < len(frame_ids):
                new = dict(row)
                new["frame_id"] = int(frame_ids[local])
                by_frame[int(frame_ids[local])].append(new)
        for frame_id in frame_ids:
            mask_path = mask_root / f"{int(frame_id)}.png"
            if not mask_path.exists():
                missing_mask_raster_count += 1
                gt = eval_base._load_gt_2d(scene, frame_id, (968, 1296))
                acc.add(np.zeros(gt.shape, dtype=np.int64), gt)
                continue
            label = eval_base._load_label_png(mask_path)
            pred = np.zeros(label.shape, dtype=np.int64)
            emitted = 0
            for row in sorted(by_frame.get(int(frame_id), []), key=lambda r: (-float(r.get("object_score", 0.0)), str(r.get("mv_object_id", "")))):
                oid = str(row["mv_object_id"])
                if oid not in object_index:
                    object_index[oid] = len(object_index) + 1
                    scores[oid] = float(row.get("object_score", 0.0))
                else:
                    scores[oid] = max(scores[oid], float(row.get("object_score", 0.0)))
                mask_id = int(row["selected_mask_id"])
                pixels = label == mask_id
                if not np.any(pixels):
                    missing_mask_raster_count += 1
                    continue
                overlap = pixels & (pred > 0)
                pixel_collision_count += int(np.count_nonzero(overlap))
                pred[(pred == 0) & pixels] = int(object_index[oid])
                emitted += 1
                selected = dict(row)
                selected["frame_id"] = int(frame_id)
                selected["score"] = float(row.get("object_score", 0.0))
                selected["selected_mask_area"] = int(np.count_nonzero(pixels))
                selected_rows.append(selected)
            gt = eval_base._load_gt_2d(scene, frame_id, label.shape)
            acc.add(pred, gt)
            preview_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_temporal_track_scene_frame_eval_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "frame_id": int(frame_id),
                    "emitted_object_count": int(emitted),
                    "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                }
            )
        input_scores = np.ones((len(object_index),), dtype=np.float32)
        for oid, idx in object_index.items():
            input_scores[int(idx) - 1] = float(scores.get(oid, 1.0))
        metric, iou, _pred_ids, _gt_ids = eval_base._summarize_iou(
            accumulator=acc,
            min_pred_pixels=min_pred_pixels,
            min_gt_pixels=min_gt_pixels,
            score_mode="input",
            input_scores=input_scores,
        )
        scene_metric_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_temporal_track_scene_metric_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant_id,
                "scene_id": scene,
                "MV_AP_scene": metric.get("ap"),
                "MV_AP50_scene": metric.get("ap50"),
                "MV_AP25_scene": metric.get("ap25"),
                "ScoreFreeMatch50_scene": metric.get("score_free_match_at_050", {}).get("f1"),
                "evaluated_pred_count": metric.get("evaluated_pred_count"),
                "evaluated_gt_count": metric.get("evaluated_gt_count"),
                "gt_best_iou_mean": metric.get("gt_best_iou_mean"),
                "pred_best_iou_mean": metric.get("pred_best_iou_mean"),
                "iou_backend": backend,
                "metric_scope": scene_metric_scope,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    metric_keys = ["MV_AP_scene", "MV_AP50_scene", "MV_AP25_scene", "ScoreFreeMatch50_scene"]
    aggregate = {
        "schema_version": "stream4d_v104_lingbot_temporal_track_scene_aggregate_row_v1",
        "phase_id": PHASE_ID,
        "variant_id": variant_id,
        "scene_count": len(scene_metric_rows),
        "metric_scope": scene_aggregate_scope,
        "iou_backend": backend_used,
        "pixel_collision_count": int(pixel_collision_count),
        "pixel_collision_rate": 0.0,
        "missing_mask_raster_count": int(missing_mask_raster_count),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    for key in metric_keys:
        vals = [float(row[key]) for row in scene_metric_rows if row.get(key) not in {"", None}]
        aggregate[key] = float(np.mean(vals)) if vals else 0.0
    pred_pixels = sum(int(row.get("pred_positive_pixels", 0)) for row in preview_rows)
    aggregate["pixel_collision_rate"] = float(pixel_collision_count / max(1, pred_pixels))
    return scene_metric_rows, aggregate, selected_rows, pixel_collision_count, missing_mask_raster_count, len(preview_rows)


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    observations = _enrich_observations(_project(args.feature_root), phase2_summaries)
    metric_scope_label = str(getattr(args, "metric_scope_label", "first32_dev_subset_window_mean; not a full-dev claim"))
    scene_metric_scope_label = str(getattr(args, "scene_metric_scope_label", "first32_scene_raw_gt"))
    scene_aggregate_scope_label = str(getattr(args, "scene_aggregate_scope_label", "first32_scene_raw_gt_mean"))

    all_object_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_scene_metric_rows: list[dict[str, Any]] = []
    all_scene_aggregate_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    baseline = _baseline(_project(args.baseline_rows))

    variants = VARIANTS
    if str(args.variants).strip():
        requested = {v.strip() for v in str(args.variants).split(",") if v.strip()}
        variants = [v for v in VARIANTS if str(v["variant_id"]) in requested]

    for variant in variants:
        scene_frame_rows: dict[str, list[dict[str, Any]]] = {}
        for scene, obs in observations.items():
            object_rows, frame_rows, edge_rows = _track_scene(scene, obs, variant)
            scene_frame_rows[scene] = frame_rows
            all_object_rows.extend(object_rows)
            all_frame_rows.extend(frame_rows)
            all_edge_rows.extend(edge_rows[:20000])
        window_rows, metric_row, selected_rows, _pix, _missing, _frames = eval_base._evaluate_variant(
            variant_id=str(variant["variant_id"]),
            scene_rows=scene_frame_rows,
            phase2_summaries=phase2_summaries,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            use_cupy_iou=not bool(args.disable_cupy_iou),
            cupy_device_id=int(args.cupy_device_id),
        )
        metric_row.update(
            {
                "metric_scope": metric_scope_label,
                "sigma_3d": variant["sigma_3d"],
                "sigma_2d": variant["sigma_2d"],
                "sigma_log_area": variant["sigma_log_area"],
                "w3d": variant["w3d"],
                "w2d": variant["w2d"],
                "warea": variant["warea"],
                "threshold": variant["threshold"],
                "max_gap": variant["max_gap"],
                "min_frames": variant["min_frames"],
                "non_broad_only": variant["non_broad_only"],
                "readout_mode": "v104_lingbot_temporal_track",
            }
        )
        all_window_rows.extend(window_rows)
        all_metric_rows.append(metric_row)
        all_selected_rows.extend(selected_rows)
        scene_rows, scene_metric, scene_selected, _scene_pix, _scene_missing, _scene_frames = _evaluate_variant_scene(
            variant_id=str(variant["variant_id"]),
            scene_rows=scene_frame_rows,
            phase2_summaries=phase2_summaries,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            use_cupy_iou=not bool(args.disable_cupy_iou),
            cupy_device_id=int(args.cupy_device_id),
            scene_metric_scope=scene_metric_scope_label,
            scene_aggregate_scope=scene_aggregate_scope_label,
        )
        scene_metric.update(
            {
                "sigma_3d": variant["sigma_3d"],
                "sigma_2d": variant["sigma_2d"],
                "sigma_log_area": variant["sigma_log_area"],
                "w3d": variant["w3d"],
                "w2d": variant["w2d"],
                "warea": variant["warea"],
                "threshold": variant["threshold"],
                "max_gap": variant["max_gap"],
                "min_frames": variant["min_frames"],
                "non_broad_only": variant["non_broad_only"],
                "readout_mode": "v104_lingbot_temporal_track",
            }
        )
        all_scene_metric_rows.extend(scene_rows)
        all_scene_aggregate_rows.append(scene_metric)

    best = max(all_metric_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    lingbot_rows = [row for row in all_metric_rows if float(row.get("w3d", 0.0)) > 0.0]
    control_rows = [row for row in all_metric_rows if float(row.get("w3d", 0.0)) <= 0.0]
    best_lingbot = max(lingbot_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    best_control = max(control_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    scene_best_lingbot = max(
        [row for row in all_scene_aggregate_rows if float(row.get("w3d", 0.0)) > 0.0],
        key=lambda row: (float(row.get("MV_AP_scene", 0.0)), float(row.get("MV_AP50_scene", 0.0))),
        default={},
    )
    scene_best_control = max(
        [row for row in all_scene_aggregate_rows if float(row.get("w3d", 0.0)) <= 0.0],
        key=lambda row: (float(row.get("MV_AP_scene", 0.0)), float(row.get("MV_AP50_scene", 0.0))),
        default={},
    )
    baseline_ap = float(baseline.get("MV_AP_window", 0.0))
    baseline_ap50 = float(baseline.get("MV_AP50_window", 0.0))
    for row in all_metric_rows:
        checks = [
            ("same_frame_collision_count_eq_0", int(row["same_frame_collision_count"]) == 0, row["same_frame_collision_count"], 0),
            ("pixel_collision_rate_le_0p02", float(row["pixel_collision_rate"]) <= 0.02, row["pixel_collision_rate"], 0.02),
            ("missing_mask_raster_count_eq_0", int(row["missing_mask_raster_count"]) == 0, row["missing_mask_raster_count"], 0),
            ("uses_gt_for_prediction_false", not bool(row["uses_gt_for_prediction"]), row["uses_gt_for_prediction"], False),
            ("uses_future_false", not bool(row["uses_future"]), row["uses_future"], False),
            ("MV_AP_window_ge_baseline_minus_0p003", float(row["MV_AP_window"]) >= baseline_ap - 0.003, row["MV_AP_window"], baseline_ap - 0.003),
            ("MV_AP50_window_ge_baseline_minus_0p006", float(row["MV_AP50_window"]) >= baseline_ap50 - 0.006, row["MV_AP50_window"], baseline_ap50 - 0.006),
        ]
        for name, ok, observed, required in checks:
            gate_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_temporal_track_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": row["variant_id"],
                    "gate_name": name,
                    "pass": bool(ok),
                    "observed": observed,
                    "required": required,
                }
            )
            if row["variant_id"] == best_lingbot.get("variant_id") and not bool(ok):
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v104_lingbot_temporal_track_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": row["variant_id"],
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": "Repair temporal link scoring, candidate mask filtering, or support geometry stability before any Go claim.",
                    }
                )

    _write_csv(out / "temporal_track_object_rows.csv", all_object_rows)
    _write_csv(out / "temporal_track_frame_mask_rows.csv", all_selected_rows)
    _write_csv(out / "raw_temporal_track_frame_rows.csv", all_frame_rows)
    _write_csv(out / "temporal_track_edge_rows.csv", all_edge_rows)
    _write_csv(out / "local_mv_metric_rows.csv", all_metric_rows)
    _write_csv(out / "local_mv_metric_window_rows.csv", all_window_rows)
    _write_csv(out / "scene_mv_metric_rows.csv", all_scene_aggregate_rows)
    _write_csv(out / "scene_mv_metric_scene_rows.csv", all_scene_metric_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    if all_selected_rows:
        try:
            import pandas as pd

            pd.DataFrame(all_selected_rows).to_parquet(out / "temporal_track_frame_mask_rows.parquet", index=False)
        except Exception:
            pass

    lingbot_ap = float(best_lingbot.get("MV_AP_window", 0.0)) if best_lingbot else 0.0
    lingbot_ap50 = float(best_lingbot.get("MV_AP50_window", 0.0)) if best_lingbot else 0.0
    control_ap = float(best_control.get("MV_AP_window", 0.0)) if best_control else 0.0
    qualified_lingbot_pass = bool(best_lingbot) and lingbot_ap >= baseline_ap - 0.003 and lingbot_ap50 >= baseline_ap50 - 0.006
    control_beats_lingbot = bool(best_control and best_lingbot) and control_ap > lingbot_ap
    phase_pass = qualified_lingbot_pass and not failure_rows
    if phase_pass:
        decision = "PASS_LINGBOT_TEMPORAL_TRACK_LOCAL_AP_CONTROL_STRONGER" if control_beats_lingbot else "PASS_LINGBOT_TEMPORAL_TRACK_LOCAL_AP"
    else:
        decision = "NO_GO_REPAIR_LINGBOT_TEMPORAL_TRACK"
    summary = {
        "schema_version": "stream4d_v104_lingbot_temporal_track_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix": time.time(),
        "phase10_pass": phase_pass,
        "decision": decision,
        "best_overall_variant_id": best.get("variant_id", ""),
        "best_overall_MV_AP_window": best.get("MV_AP_window", ""),
        "best_overall_MV_AP50_window": best.get("MV_AP50_window", ""),
        "best_lingbot_variant_id": best_lingbot.get("variant_id", ""),
        "best_lingbot_MV_AP_window": best_lingbot.get("MV_AP_window", ""),
        "best_lingbot_MV_AP50_window": best_lingbot.get("MV_AP50_window", ""),
        "best_lingbot_MV_AP25_window": best_lingbot.get("MV_AP25_window", ""),
        "best_lingbot_ScoreFreeMatch50_window": best_lingbot.get("ScoreFreeMatch50_window", ""),
        "best_control_variant_id": best_control.get("variant_id", ""),
        "best_control_MV_AP_window": best_control.get("MV_AP_window", ""),
        "best_control_MV_AP50_window": best_control.get("MV_AP50_window", ""),
        "best_lingbot_MV_AP_scene": scene_best_lingbot.get("MV_AP_scene", ""),
        "best_lingbot_MV_AP50_scene": scene_best_lingbot.get("MV_AP50_scene", ""),
        "best_lingbot_scene_variant_id": scene_best_lingbot.get("variant_id", ""),
        "best_control_MV_AP_scene": scene_best_control.get("MV_AP_scene", ""),
        "best_control_MV_AP50_scene": scene_best_control.get("MV_AP50_scene", ""),
        "best_control_scene_variant_id": scene_best_control.get("variant_id", ""),
        "control_beats_lingbot_MV_AP_scene": (
            float(scene_best_control.get("MV_AP_scene", 0.0)) > float(scene_best_lingbot.get("MV_AP_scene", 0.0))
            if scene_best_control and scene_best_lingbot
            else ""
        ),
        "control_minus_lingbot_MV_AP_scene": (
            float(scene_best_control.get("MV_AP_scene", 0.0)) - float(scene_best_lingbot.get("MV_AP_scene", 0.0))
            if scene_best_control and scene_best_lingbot
            else ""
        ),
        "qualified_lingbot_local_ap_pass": qualified_lingbot_pass,
        "control_beats_lingbot_MV_AP_window": control_beats_lingbot,
        "control_minus_lingbot_MV_AP_window": control_ap - lingbot_ap if best_control and best_lingbot else "",
        "baseline_contract": baseline,
        "metric_scope": metric_scope_label,
        "feature_root": _rel(_project(args.feature_root)),
        "variant_count": len(variants),
        "scene_ids": sorted(observations),
        "observation_count": sum(len(v) for v in observations.values()),
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "Temporal links are built from LingBot support centroids plus mask geometry only; GT is used only by canonical AP evaluation.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "local_mv_metric_rows": _rel(out / "local_mv_metric_rows.csv"),
            "local_mv_metric_window_rows": _rel(out / "local_mv_metric_window_rows.csv"),
            "scene_mv_metric_rows": _rel(out / "scene_mv_metric_rows.csv"),
            "scene_mv_metric_scene_rows": _rel(out / "scene_mv_metric_scene_rows.csv"),
            "temporal_track_object_rows": _rel(out / "temporal_track_object_rows.csv"),
            "temporal_track_frame_mask_rows": _rel(out / "temporal_track_frame_mask_rows.csv"),
            "temporal_track_edge_rows": _rel(out / "temporal_track_edge_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LingBot-only temporal tracking readout and local MV_AP evaluation.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--variants", default="")
    parser.add_argument("--metric-scope-label", default="first32_dev_subset_window_mean; not a full-dev claim")
    parser.add_argument("--scene-metric-scope-label", default="first32_scene_raw_gt")
    parser.add_argument("--scene-aggregate-scope-label", default="first32_scene_raw_gt_mean")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    return 0 if summary.get("phase10_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
