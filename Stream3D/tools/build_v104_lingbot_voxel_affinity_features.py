#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v104_lingbot_map_only_phase8_voxel_affinity_features"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_SUPPORT_ROWS = AUDIT_ROOT / "v104_lingbot_map_only_phase7_real_mask_support_rows/real_mask_support_rows.csv"
DEFAULT_SELECTED_ROWS = AUDIT_ROOT / "v87_phase1_mv_input_generation/frame_mask_selected_rows.csv"
DEFAULT_PHASE2_SCENE0011 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_PHASE2_SCENE0050 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"


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


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_points_and_xy(points: np.ndarray, xy: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if xy is None:
        return points[:0], np.empty((0, 2), dtype=np.float32)
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    n = min(points.shape[0], xy.shape[0])
    points = points[:n]
    xy = xy[:n]
    finite = np.isfinite(points).all(axis=1) & np.isfinite(xy).all(axis=1)
    return points[finite], xy[finite]


def _stable_hash(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.uint64)
    vals ^= vals >> np.uint64(30)
    vals *= np.uint64(0xBF58476D1CE4E5B9)
    vals ^= vals >> np.uint64(27)
    vals *= np.uint64(0x94D049BB133111EB)
    vals ^= vals >> np.uint64(31)
    return vals


def _voxel_hash(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros((0,), dtype=np.uint64), np.zeros((0, 3), dtype=np.int64)
    q = np.floor(np.asarray(points, dtype=np.float64) / float(voxel_size)).astype(np.int64)
    uq, counts = np.unique(q, axis=0, return_counts=True)
    shifted = uq.astype(np.int64) + np.int64(2_000_000_000)
    raw = (
        shifted[:, 0].astype(np.uint64) * np.uint64(73856093)
        ^ shifted[:, 1].astype(np.uint64) * np.uint64(19349663)
        ^ shifted[:, 2].astype(np.uint64) * np.uint64(83492791)
    )
    return _stable_hash(raw) ^ counts.astype(np.uint64), uq


def _sketch_from_voxels(voxel_ids: np.ndarray, sketch_dim: int) -> np.ndarray:
    vec = np.zeros((int(sketch_dim),), dtype=np.float32)
    if voxel_ids.size == 0:
        return vec
    ids, counts = np.unique(voxel_ids.astype(np.uint64), return_counts=True)
    mixed = _stable_hash(ids)
    buckets = (mixed % np.uint64(sketch_dim)).astype(np.int64)
    signs = np.where(((mixed >> np.uint64(63)) & np.uint64(1)) > 0, 1.0, -1.0).astype(np.float32)
    weights = np.sqrt(counts.astype(np.float32))
    np.add.at(vec, buckets, signs * weights)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _centroid_rff(points: np.ndarray, omega: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vec = np.zeros((omega.shape[0],), dtype=np.float32)
    if points.size == 0:
        return vec, np.zeros((3,), dtype=np.float32)
    centroid = np.asarray(points, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    proj = omega @ centroid.astype(np.float32) + bias
    vec = np.cos(proj).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec, centroid.astype(np.float32)


def _pair_values(feature: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.sum(feature[pairs[:, 0]] * feature[pairs[:, 1]], axis=1).astype(np.float32, copy=False)


def _sample_pairs(rows: list[dict[str, Any]], max_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_history: dict[str, list[int]] = defaultdict(list)
    by_frame: dict[int, list[int]] = defaultdict(list)
    broad: list[int] = []
    for idx, row in enumerate(rows):
        hist = str(row.get("history_id", ""))
        if hist:
            by_history[hist].append(idx)
        by_frame[int(row["source_frame_id"])].append(idx)
        if bool(row.get("mask_is_broad", False)):
            broad.append(idx)

    pseudo: list[tuple[int, int]] = []
    for members in by_history.values():
        ordered = sorted(set(members), key=lambda i: (int(rows[i]["source_frame_id"]), int(rows[i]["mask_id"]), i))
        for pos, a in enumerate(ordered[:-1]):
            for b in ordered[pos + 1 :]:
                if rows[a]["source_frame_id"] == rows[b]["source_frame_id"]:
                    continue
                pseudo.append((a, b))
                if len(pseudo) >= max_pairs:
                    break
            if len(pseudo) >= max_pairs:
                break
        if len(pseudo) >= max_pairs:
            break

    same_frame: list[tuple[int, int]] = []
    for members in by_frame.values():
        ordered = sorted(set(members), key=lambda i: (int(rows[i]["mask_id"]), i))
        for pos, a in enumerate(ordered[:-1]):
            for b in ordered[pos + 1 :]:
                same_frame.append((a, b))
                if len(same_frame) >= max_pairs:
                    break
            if len(same_frame) >= max_pairs:
                break
        if len(same_frame) >= max_pairs:
            break

    broad_pairs: list[tuple[int, int]] = []
    broad_set = set(broad)
    for a in broad:
        for b in range(len(rows)):
            if a == b or b in broad_set:
                continue
            broad_pairs.append((min(a, b), max(a, b)))
            if len(broad_pairs) >= max_pairs:
                break
        if len(broad_pairs) >= max_pairs:
            break

    return (
        np.asarray(pseudo, dtype=np.int64).reshape(-1, 2),
        np.asarray(same_frame, dtype=np.int64).reshape(-1, 2),
        np.asarray(broad_pairs, dtype=np.int64).reshape(-1, 2),
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(STREAM3D_ROOT))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import torch

    from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider

    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    target_scenes = [part.strip() for part in str(args.target_scenes).split(",") if part.strip()]
    variants = {part.strip() for part in str(args.variants).split(",") if part.strip()}
    feature_mode = str(args.feature_mode)
    if feature_mode not in {"voxel", "centroid_rff", "voxel_centroid"}:
        raise ValueError(f"unsupported feature_mode={feature_mode}")
    rng = np.random.default_rng(int(args.rff_seed))
    rff_dim = int(args.sketch_dim)
    if feature_mode == "voxel_centroid":
        voxel_dim = max(2, int(args.sketch_dim) // 2)
        rff_dim = int(args.sketch_dim) - voxel_dim
    else:
        voxel_dim = int(args.sketch_dim)
    omega = rng.normal(0.0, 1.0 / max(float(args.rff_sigma), 1e-6), size=(rff_dim, 3)).astype(np.float32)
    bias = rng.uniform(0.0, 2.0 * np.pi, size=(rff_dim,)).astype(np.float32)
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    frame_to_local = {
        scene: {int(frame): idx for idx, frame in enumerate(summary.get("frame_ids", []))}
        for scene, summary in phase2_summaries.items()
    }
    selected_meta = {row.get("candidate_row_id", ""): row for row in _read_csv(_project(args.selected_rows))}
    support_rows_raw = [
        row
        for row in _read_csv(_project(args.support_rows))
        if row.get("scene_id") in target_scenes and (not variants or row.get("variant", "") in variants)
    ]

    dedup: dict[tuple[str, int, int, str], dict[str, str]] = {}
    duplicate_count = 0
    for row in support_rows_raw:
        scene = row.get("scene_id", "")
        source_frame = _as_int(row.get("source_frame_id"))
        mask_id = _as_int(row.get("mask_id"))
        if source_frame not in frame_to_local.get(scene, {}):
            continue
        key = (scene, source_frame, mask_id, row.get("variant", ""))
        old = dedup.get(key)
        if old is None or _as_int(row.get("support_point_count"), 0) > _as_int(old.get("support_point_count"), 0):
            if old is not None:
                duplicate_count += 1
            dedup[key] = row
        else:
            duplicate_count += 1
    support_rows = sorted(
        dedup.values(),
        key=lambda r: (r.get("scene_id", ""), _as_int(r.get("source_frame_id")), _as_int(r.get("mask_id")), _as_int(r.get("candidate_row_id"))),
    )

    providers: dict[Path, LingBotMapGeometryProvider] = {}
    frame_cache: dict[tuple[Path, int], np.ndarray] = {}

    def support_points_for_row(row: dict[str, str]) -> np.ndarray:
        root = _project(row.get("lingbot_root", ""))
        bss_frame_id = _as_int(row.get("bss_frame_id"))
        provider = providers.get(root)
        if provider is None:
            provider = LingBotMapGeometryProvider(
                geometry_root=root,
                max_points_per_frame=int(args.max_points_per_frame),
                min_confidence=args.min_confidence,
            )
            providers[root] = provider
        key = (root, bss_frame_id)
        if key not in frame_cache:
            samples = provider.load_frame_samples(bss_frame_id)
            points, _xy = _finite_points_and_xy(samples.points, samples.xy)
            frame_cache[key] = points
        points = frame_cache[key]
        support_ids = np.asarray(np.load(_project(row.get("support_point_ids_path", ""))), dtype=np.int64)
        support_ids = support_ids[(support_ids >= 0) & (support_ids < points.shape[0])]
        return points[support_ids]

    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for scene in target_scenes:
        rows = [row for row in support_rows if row.get("scene_id") == scene]
        features: list[np.ndarray] = []
        scene_obs: list[dict[str, Any]] = []
        for obs_idx, row in enumerate(rows):
            candidate_id = row.get("candidate_row_id", "")
            meta = selected_meta.get(candidate_id, {})
            source_frame = _as_int(row.get("source_frame_id"))
            mask_id = _as_int(row.get("mask_id"))
            support_points = support_points_for_row(row)
            voxel_ids, voxel_xyz = _voxel_hash(support_points, float(args.voxel_size))
            voxel_feat = _sketch_from_voxels(voxel_ids, int(voxel_dim))
            centroid_feat, centroid = _centroid_rff(support_points, omega, bias)
            if feature_mode == "voxel":
                feat = voxel_feat
            elif feature_mode == "centroid_rff":
                feat = centroid_feat
            else:
                feat = np.concatenate([voxel_feat, centroid_feat], axis=0)
                norm = float(np.linalg.norm(feat))
                if norm > 0.0:
                    feat = feat / norm
            features.append(feat)
            obs = {
                "schema_version": "stream4d_v104_lingbot_voxel_mask_observation_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "mask_observation_index": obs_idx,
                "candidate_row_id": candidate_id,
                "variant": row.get("variant", ""),
                "variant_family": row.get("variant_family", ""),
                "history_id": row.get("history_id", ""),
                "source_frame_id": source_frame,
                "frame_local_index": int(frame_to_local[scene][source_frame]),
                "mask_id": mask_id,
                "support_point_count": int(support_points.shape[0]),
                "unique_voxel_count": int(voxel_ids.shape[0]),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "feature_norm": float(np.linalg.norm(feat)),
                "mask_is_broad": _as_bool(meta.get("broad_mask_flag", "False")),
                "mask_is_object_like": _as_bool(meta.get("object_mask_ownership_allowed", "True")),
                "support_point_ids_path": row.get("support_point_ids_path", ""),
                "lingbot_root": row.get("lingbot_root", ""),
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            scene_obs.append(obs)
            feature_rows.append(obs)
        scene_rows[scene] = scene_obs

        feature = np.stack(features, axis=0).astype(np.float32) if features else np.zeros((0, int(args.sketch_dim)), dtype=np.float32)
        mask_frame = np.asarray([row["frame_local_index"] for row in scene_obs], dtype=np.int64)
        mask_label = np.asarray([row["mask_id"] for row in scene_obs], dtype=np.int64)
        mask_is_broad = np.asarray([row["mask_is_broad"] for row in scene_obs], dtype=bool)
        mask_is_object_like = np.asarray([row["mask_is_object_like"] for row in scene_obs], dtype=bool)
        support_count = np.asarray([row["support_point_count"] for row in scene_obs], dtype=np.int64)
        unique_voxels = np.asarray([row["unique_voxel_count"] for row in scene_obs], dtype=np.int64)

        scene_dir = out / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "stream4d_v104_lingbot_voxel_mask_level_feature_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "variant_id": "L0_lingbot_voxel_sketch",
                "static_feature_source": f"lingbot_world_points_{feature_mode}",
                "pair_affinity_mode": "static_feature_cosine",
                "mask_observation_index": torch.arange(feature.shape[0], dtype=torch.int64),
                "mask_frame": torch.as_tensor(mask_frame, dtype=torch.int64),
                "mask_label": torch.as_tensor(mask_label, dtype=torch.int64),
                "mask_is_object_like": torch.as_tensor(mask_is_object_like, dtype=torch.bool),
                "mask_is_broad": torch.as_tensor(mask_is_broad, dtype=torch.bool),
                "support_count": torch.as_tensor(support_count, dtype=torch.int64),
                "feature": torch.as_tensor(feature, dtype=torch.float16),
                "voxel_size": float(args.voxel_size),
                "sketch_dim": int(args.sketch_dim),
                "feature_mode": feature_mode,
                "rff_sigma": float(args.rff_sigma),
                "uses_gt": False,
                "uses_future": False,
            },
            scene_dir / "mask_level_feature.pt",
        )

        pseudo_pairs, same_frame_pairs, broad_pairs = _sample_pairs(scene_obs, int(args.max_pair_rows))
        pseudo_vals = _pair_values(feature, pseudo_pairs)
        same_vals = _pair_values(feature, same_frame_pairs)
        broad_vals = _pair_values(feature, broad_pairs)
        for pair_type, pairs, vals in [
            ("pseudo_same_history_cross_frame", pseudo_pairs, pseudo_vals),
            ("same_frame_competing", same_frame_pairs, same_vals),
            ("broad_vs_nonbroad", broad_pairs, broad_vals),
        ]:
            order = np.argsort(vals)[::-1] if vals.size else np.zeros((0,), dtype=np.int64)
            for rank, idx in enumerate(order[: int(args.max_pair_rows)]):
                a, b = pairs[int(idx)].tolist()
                ra = scene_obs[int(a)]
                rb = scene_obs[int(b)]
                pair_rows.append(
                    {
                        "schema_version": "stream4d_v104_lingbot_voxel_mask_pair_affinity_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "pair_type": pair_type,
                        "rank": rank,
                        "mask_a": int(a),
                        "mask_b": int(b),
                        "frame_a": int(ra["source_frame_id"]),
                        "frame_b": int(rb["source_frame_id"]),
                        "mask_id_a": int(ra["mask_id"]),
                        "mask_id_b": int(rb["mask_id"]),
                        "history_id_a": ra["history_id"],
                        "history_id_b": rb["history_id"],
                        "affinity": float(vals[int(idx)]),
                        "uses_d4rt_for_prediction": False,
                        "uses_da3_for_prediction": False,
                        "uses_gt_for_prediction": False,
                    }
                )

        feature_valid = feature.shape[0] > 0 and float(np.mean(np.linalg.norm(feature, axis=1) > 0.0))
        object_mask = mask_is_object_like.astype(bool)
        min_support_pass_rate = (
            float(np.mean(support_count[object_mask] >= int(args.min_support_points_per_mask)))
            if np.any(object_mask)
            else 0.0
        )
        metric = {
            "schema_version": "stream4d_v104_lingbot_voxel_affinity_metric_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "variant_filter": ",".join(sorted(variants)) if variants else "all",
            "feature_mode": feature_mode,
            "frame_scope": "v103_phase2_first32_local_window",
            "mask_observation_count": len(scene_obs),
            "feature_valid_rate": feature_valid,
            "object_like_mask_with_min_support_rate": min_support_pass_rate,
            "support_point_count_mean": float(np.mean(support_count)) if support_count.size else 0.0,
            "support_point_count_p05": float(np.percentile(support_count, 5)) if support_count.size else 0.0,
            "unique_voxel_count_mean": float(np.mean(unique_voxels)) if unique_voxels.size else 0.0,
            "unique_voxel_count_p05": float(np.percentile(unique_voxels, 5)) if unique_voxels.size else 0.0,
            "pseudo_positive_affinity_mean": _stats(pseudo_vals)["mean"],
            "same_frame_competing_affinity_p95": _stats(same_vals)["p95"],
            "broad_pair_affinity_p95": _stats(broad_vals)["p95"],
            "pseudo_positive_pair_count": int(pseudo_pairs.shape[0]),
            "same_frame_pair_count": int(same_frame_pairs.shape[0]),
            "broad_pair_count": int(broad_pairs.shape[0]),
            "uses_gt_for_prediction": False,
            "uses_gt_for_metric": False,
        }
        metric["pseudo_minus_same_frame_p95"] = float(metric["pseudo_positive_affinity_mean"]) - float(metric["same_frame_competing_affinity_p95"])
        metric_rows.append(metric)

        gate_specs = [
            ("feature_valid_rate_ge_0p90", float(metric["feature_valid_rate"]), 0.90, ">="),
            ("object_like_min_support_rate_ge_0p80", float(metric["object_like_mask_with_min_support_rate"]), 0.80, ">="),
            ("pseudo_pair_count_positive", int(metric["pseudo_positive_pair_count"]), 1, ">="),
        ]
        for gate_name, observed, required, op in gate_specs:
            passed = observed >= required
            gate_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_voxel_affinity_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "gate_name": gate_name,
                    "pass": bool(passed),
                    "observed": observed,
                    "required": required,
                    "operator": op,
                }
            )
            if not passed:
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v104_lingbot_voxel_affinity_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "failure_id": gate_name,
                        "severity": "blocking",
                        "observed": observed,
                        "required": required,
                        "repair_direction": "Increase LingBot support density, adjust voxel size/sketching, or revisit selected mask support before Phase6 AP.",
                    }
                )

    _write_csv(out / "mask_observation_rows.csv", feature_rows)
    _write_csv(out / "mask_pair_affinity_rows.csv", pair_rows)
    _write_csv(out / "mask_pooling_metric_rows.csv", metric_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    if pair_rows:
        try:
            import pandas as pd

            pd.DataFrame(pair_rows).to_parquet(out / "mask_pair_affinity_rows.parquet", index=False)
        except Exception as exc:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_voxel_affinity_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "failure_id": "PAIR_PARQUET_WRITE_SKIPPED",
                    "severity": "diagnostic",
                    "observed": type(exc).__name__,
                    "required": "csv output exists; parquet is optional for this v104 adapter",
                }
            )
            _write_csv(out / "failure_rows.csv", failure_rows)

    blocking = [row for row in failure_rows if row.get("severity") == "blocking"]
    summary = {
        "schema_version": "stream4d_v104_lingbot_voxel_affinity_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix": time.time(),
        "feature_build_pass": not blocking and bool(feature_rows),
        "phase5_pass": not blocking and bool(feature_rows),
        "taxonomy": "LINGBOT_VOXEL_MASK_LEVEL_FEATURES_READY_FOR_PHASE6" if not blocking and feature_rows else "LINGBOT_VOXEL_MASK_LEVEL_FEATURES_FAIL",
        "blocker": "" if not blocking and feature_rows else "LINGBOT_VOXEL_FEATURE_BUILD_BLOCKED",
        "target_scenes": target_scenes,
        "variant_filter": sorted(variants) if variants else ["all"],
        "frame_scope": "v103_phase2_first32_local_window",
        "voxel_size": float(args.voxel_size),
        "sketch_dim": int(args.sketch_dim),
        "feature_mode": feature_mode,
        "rff_sigma": float(args.rff_sigma),
        "rff_seed": int(args.rff_seed),
        "raw_support_row_count": len(support_rows_raw),
        "deduped_window_support_row_count": len(feature_rows),
        "dropped_duplicate_row_count": duplicate_count,
        "metric_rows": metric_rows,
        "failure_count": len(failure_rows),
        "blocking_failure_count": len(blocking),
        "stream4d_metric_ready": False,
        "stream4d_metric_note": "Phase8 builds LingBot voxel mask-level features and pair affinities only; AP/MV_AP requires Phase6 evaluator.",
        "compatibility_note": "phase5_pass is set for v103 Phase6 input compatibility; this is a v104 LingBot voxel feature build, not a v103 Phase5 success claim.",
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "mask_observation_rows": _rel(out / "mask_observation_rows.csv"),
            "mask_pair_affinity_rows": _rel(out / "mask_pair_affinity_rows.csv"),
            "mask_pooling_metric_rows": _rel(out / "mask_pooling_metric_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LingBot voxel mask-level affinity features from real selected-mask support rows.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--support-rows", default=str(DEFAULT_SUPPORT_ROWS))
    parser.add_argument("--selected-rows", default=str(DEFAULT_SELECTED_ROWS))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--target-scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--variants", default="B0_local_only")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--feature-mode", choices=["voxel", "centroid_rff", "voxel_centroid"], default="voxel")
    parser.add_argument("--rff-sigma", type=float, default=0.50)
    parser.add_argument("--rff-seed", type=int, default=10419)
    parser.add_argument("--max-pair-rows", type=int, default=4096)
    parser.add_argument("--min-support-points-per-mask", type=int, default=5)
    parser.add_argument("--max-points-per-frame", type=int, default=20000)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
