from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.measurement_bank import build_measurement_bank
from tools.run_v26_object_quality_diagnostics import _json_safe
from tools.run_v28_proposal_oracle import _tube_xy
from tools.run_v28_proposal_selection import _load_gt_labels
from tools.run_v29_constrained_ownership_solver import _quantile, _set_core_ids
from tools.run_v30_object_slot_ownership import _read_split
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


@dataclass(frozen=True)
class CliqueConfig:
    name: str
    link_mode: str
    px_thr: float
    color_thr: float
    max_neg: int
    min_core: int
    max_core: int
    safe_min: float
    boundary_p10_min: float


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _pair(a: int, b: int) -> tuple[int, int]:
    left, right = int(a), int(b)
    return (left, right) if left < right else (right, left)


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _scene_records_and_measurements(args: argparse.Namespace, scene: str):
    scene_dir = Path(args.cache_root) / scene
    chunks, _ = load_scene_chunks_from_cache(
        scene_dir,
        max_tubes_per_window=int(args.max_tubes_per_window),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
    )
    builder = D4RTNativeSceneBuilder(
        object(),
        {"model": {"input": {"clip_frames": 32}}},
        temporal_chunk_size=32,
        temporal_chunk_stride=16,
    )
    records = chunks_to_records(builder.stitch_to_canonical(chunks))
    frame_ids = sorted(
        {int(v) for tube in records for v in np.asarray(tube.target_frames_global, dtype=np.int64).tolist()}
    )
    stream = ScanNetStream(seq_name=scene)
    masks_by_frame = {frame_id: stream.load_mask(frame_id) for frame_id in frame_ids}
    measurements, meas_diag = build_measurement_bank(
        records,
        masks_by_frame=masks_by_frame,
        min_visibility=0.5,
        min_confidence=0.5,
    )
    return stream, records, masks_by_frame, measurements, meas_diag


def _negative_pair_counts(measurements) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for meas in measurements:
        for left, right in meas.same_frame_different_mask_cannot_link_pairs:
            counts[_pair(left, right)] += 1
        for left, right in meas.visible_outside_negative_pairs:
            counts[_pair(left, right)] += 1
        for left, right in meas.boundary_crossing_cut_pairs:
            counts[_pair(left, right)] += 1
    return counts


def _node_rows(stream, records_by_id: dict[int, Any], mask: np.ndarray, rgb: np.ndarray, meas) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_id = int(meas.frame_global)
    for tid in sorted(set(int(v) for v in meas.inside_tube_ids)):
        tube = records_by_id.get(int(tid))
        if tube is None:
            continue
        xy = _tube_xy(tube, frame_id, mask.shape)
        if xy is None:
            continue
        x, y = xy
        if not (0 <= int(y) < rgb.shape[0] and 0 <= int(x) < rgb.shape[1]):
            continue
        dist = float(meas.mask_distance_to_boundary_per_tube.get(int(tid), 0.0))
        rows.append(
            {
                "tid": int(tid),
                "x": float(x),
                "y": float(y),
                "dist": dist,
                "safe": bool(meas.mask_eroded_interior_flag_per_tube.get(int(tid), False)),
                "color": (rgb[y, x].astype(np.float32) / 255.0).tolist(),
            }
        )
    return rows


def _compatible(a: dict[str, Any], b: dict[str, Any], cfg: CliqueConfig, neg_counts: Counter[tuple[int, int]]) -> bool:
    if int(neg_counts.get(_pair(int(a["tid"]), int(b["tid"])), 0)) > int(cfg.max_neg):
        return False
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    if math.hypot(dx, dy) > float(cfg.px_thr):
        return False
    ac = np.asarray(a["color"], dtype=np.float32)
    bc = np.asarray(b["color"], dtype=np.float32)
    if float(np.linalg.norm(ac - bc)) > float(cfg.color_thr):
        return False
    return True


def _negative_compatible(a: dict[str, Any], b: dict[str, Any], cfg: CliqueConfig, neg_counts: Counter[tuple[int, int]]) -> bool:
    return int(neg_counts.get(_pair(int(a["tid"]), int(b["tid"])), 0)) <= int(cfg.max_neg)


def _cluster_features(cluster: list[dict[str, Any]], neg_counts: Counter[tuple[int, int]]) -> dict[str, Any]:
    dists = [float(row["dist"]) for row in cluster]
    colors = np.asarray([row["color"] for row in cluster], dtype=np.float32)
    xs = [float(row["x"]) for row in cluster]
    ys = [float(row["y"]) for row in cluster]
    neg = 0
    total = 0
    for idx, a in enumerate(cluster):
        for b in cluster[idx + 1 :]:
            total += 1
            neg += int(neg_counts.get(_pair(int(a["tid"]), int(b["tid"])), 0) > 0)
    return {
        "core_n": int(len(cluster)),
        "safe_ratio": float(sum(1 for row in cluster if bool(row["safe"])) / max(len(cluster), 1)),
        "boundary_p10": float(np.quantile(np.asarray(dists, dtype=np.float64), 0.10)) if dists else 0.0,
        "boundary_mean": _mean(dists),
        "color_std": float(np.mean(np.std(colors, axis=0))) if len(colors) else 0.0,
        "xy_diam": float(math.hypot(max(xs) - min(xs), max(ys) - min(ys))) if xs and ys else 0.0,
        "internal_neg_pair_rate": float(neg / max(total, 1)),
    }


def _candidate_score(features: dict[str, Any]) -> float:
    return float(
        0.04 * float(features["boundary_p10"])
        + 0.35 * float(features["safe_ratio"])
        + 0.14 * math.log1p(float(features["core_n"]))
        - 0.60 * float(features["color_std"])
        - 0.004 * float(features["xy_diam"])
        - 1.50 * float(features["internal_neg_pair_rate"])
    )


def _row_core_ids(row: dict[str, Any]) -> tuple[int, ...]:
    if "_core_tube_ids" in row:
        return tuple(sorted(int(v) for v in row.get("_core_tube_ids") or ()))
    return tuple(int(v) for v in str(row.get("core_tube_ids") or "").split(";") if str(v).strip())


def _row_quality(row: dict[str, Any], gt_labels: dict[int, int], gt_counts: Counter[int]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    labeled = 0
    for tid in _row_core_ids(row):
        gt = int(gt_labels.get(int(tid), 0))
        if gt > 0:
            counts[gt] += 1
            labeled += 1
    if not counts:
        return {
            "purity": None,
            "best_iou": None,
            "best_gt": 0,
            "labeled": int(labeled),
            "counts": counts,
        }
    best_gt, best_overlap = counts.most_common(1)[0]
    return {
        "purity": float(best_overlap / max(labeled, 1)),
        "best_iou": float(best_overlap / max(labeled + int(gt_counts.get(int(best_gt), 0)) - best_overlap, 1)),
        "best_gt": int(best_gt),
        "labeled": int(labeled),
        "counts": counts,
    }


def _phase_c_metrics(
    rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
) -> dict[str, Any]:
    purities: list[float] = []
    best_ious: list[float] = []
    source_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        gt_labels = gt_by_scene[scene]
        gt_counts = Counter(int(gt) for gt in gt_labels.values() if int(gt) > 0)
        gt_best_iou: dict[int, float] = {int(gt): 0.0 for gt in gt_counts}
        scene_items = [row for row in rows if str(row.get("scene")) == scene]
        scene_purities: list[float] = []
        for row in scene_items:
            quality = _row_quality(row, gt_labels, gt_counts)
            if quality["purity"] is not None:
                purities.append(float(quality["purity"]))
                scene_purities.append(float(quality["purity"]))
            if quality["best_iou"] is not None:
                best_ious.append(float(quality["best_iou"]))
            labeled = int(quality["labeled"])
            for gt, overlap in quality["counts"].items():
                iou = float(overlap / max(labeled + int(gt_counts[int(gt)]) - int(overlap), 1))
                gt_best_iou[int(gt)] = max(gt_best_iou[int(gt)], iou)
        scene_rows.append(
            {
                "scene": scene,
                "seed_count": int(len(scene_items)),
                "labeled_seed_count": int(len(scene_purities)),
                "seed_purity_mean": _mean(scene_purities),
                "seed_purity_p10": _quantile(scene_purities, 0.10) if scene_purities else None,
                "GT_with_seed_IoU_ge_0.10": float(sum(1 for val in gt_best_iou.values() if val >= 0.10) / max(len(gt_best_iou), 1)),
                "GT_with_seed_IoU_ge_0.25": float(sum(1 for val in gt_best_iou.values() if val >= 0.25) / max(len(gt_best_iou), 1)),
            }
        )
    for source in sorted({str(row.get("seed_source")) for row in rows}):
        items = [row for row in rows if str(row.get("seed_source")) == source]
        vals: list[float] = []
        ious: list[float] = []
        cores: list[int] = []
        broad_overlap = 0
        for row in items:
            quality = _row_quality(row, gt_by_scene[str(row["scene"])], Counter(int(gt) for gt in gt_by_scene[str(row["scene"])].values() if int(gt) > 0))
            if quality["purity"] is not None:
                vals.append(float(quality["purity"]))
            if quality["best_iou"] is not None:
                ious.append(float(quality["best_iou"]))
            cores.append(len(_row_core_ids(row)))
            broad_overlap += int(bool(row.get("overlaps_broad_observation", False)))
        source_rows.append(
            {
                "seed_source": source,
                "count": int(len(items)),
                "labeled_count": int(len(vals)),
                "purity_mean": _mean(vals),
                "purity_p10": _quantile(vals, 0.10) if vals else None,
                "best_IoU_mean": _mean(ious),
                "core_tube_count_p50": _quantile(cores, 0.50) if cores else None,
                "core_tube_count_p90": _quantile(cores, 0.90) if cores else None,
                "broad_overlap_count": int(broad_overlap),
            }
        )
    all_row = {
        "seed_count": int(len(rows)),
        "labeled_seed_count": int(len(purities)),
        "seed_count_per_scene": ";".join(f"{row['scene']}={row['seed_count']}" for row in scene_rows),
        "seed_purity_mean": _mean(purities),
        "seed_purity_p10": _quantile(purities, 0.10) if purities else None,
        "seed_best_IoU_mean": _mean(best_ious),
        "GT_with_seed_IoU_ge_0.10": _mean([float(row["GT_with_seed_IoU_ge_0.10"]) for row in scene_rows]),
        "GT_with_seed_IoU_ge_0.25": _mean([float(row["GT_with_seed_IoU_ge_0.25"]) for row in scene_rows]),
        "scene0081_GT_with_seed_IoU_ge_0.10": next(
            (row["GT_with_seed_IoU_ge_0.10"] for row in scene_rows if row["scene"] == "scene0081_01"),
            None,
        ),
    }
    gates = {
        "seed_purity_mean_ge_0.90": bool(float(all_row.get("seed_purity_mean") or 0.0) >= 0.90),
        "seed_purity_p10_ge_0.75": bool(float(all_row.get("seed_purity_p10") or 0.0) >= 0.75),
        "GT_seed_IoU_010_ge_0.70": bool(float(all_row.get("GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.70),
        "GT_seed_IoU_025_ge_0.45": bool(float(all_row.get("GT_with_seed_IoU_ge_0.25") or 0.0) >= 0.45),
        "scene0081_seed_IoU_010_ge_0.50": bool(float(all_row.get("scene0081_GT_with_seed_IoU_ge_0.10") or 0.0) >= 0.50),
    }
    return {
        "n": int(len(rows)),
        "labeled_n": int(len(purities)),
        "pass_count": int(sum(1 for ok in gates.values() if ok)),
        "gates": gates,
        "purity_mean": all_row.get("seed_purity_mean"),
        "purity_p10": all_row.get("seed_purity_p10"),
        "GT_with_seed_IoU_ge_0.10": all_row.get("GT_with_seed_IoU_ge_0.10"),
        "GT_with_seed_IoU_ge_0.25": all_row.get("GT_with_seed_IoU_ge_0.25"),
        "scene0081_GT_with_seed_IoU_ge_0.10": all_row.get("scene0081_GT_with_seed_IoU_ge_0.10"),
        "seed_best_IoU_mean": all_row.get("seed_best_IoU_mean"),
        "seed_count_per_scene": all_row.get("seed_count_per_scene"),
        "scene_rows": scene_rows,
        "source_rows": source_rows,
    }


def _generate_scene_candidates(args: argparse.Namespace, scene: str, configs: list[CliqueConfig]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stream, records, masks_by_frame, measurements, meas_diag = _scene_records_and_measurements(args, scene)
    records_by_id = {int(tube.tube_id): tube for tube in records}
    neg_counts = _negative_pair_counts(measurements)
    rgb_cache: dict[int, np.ndarray] = {}
    dedup: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    per_config_counts: Counter[str] = Counter()
    processed_measurements = 0

    for meas in measurements:
        if len(meas.inside_tube_ids) < int(args.global_min_core):
            continue
        frame_id = int(meas.frame_global)
        mask = masks_by_frame.get(frame_id)
        if mask is None:
            continue
        if frame_id not in rgb_cache:
            rgb_cache[frame_id] = stream.load_rgb(frame_id)
        nodes = _node_rows(stream, records_by_id, mask, rgb_cache[frame_id], meas)
        if len(nodes) < int(args.global_min_core):
            continue
        nodes = sorted(nodes, key=lambda row: (bool(row["safe"]), float(row["dist"])), reverse=True)[
            : int(args.max_nodes_per_mask)
        ]
        processed_measurements += 1
        center_nodes = nodes[: int(args.max_centers_per_mask)]
        for cfg in configs:
            filtered = [
                row
                for row in nodes
                if float(row["dist"]) >= max(0.0, float(cfg.boundary_p10_min) * 0.35)
            ]
            if len(filtered) < int(cfg.min_core):
                continue
            for center_idx, center in enumerate(center_nodes):
                if center not in filtered:
                    continue
                cluster = [center]
                ordered = sorted(
                    (row for row in filtered if int(row["tid"]) != int(center["tid"])),
                    key=lambda row: (
                        bool(row["safe"]),
                        -float(np.linalg.norm(np.asarray(row["color"], dtype=np.float32) - np.asarray(center["color"], dtype=np.float32))),
                        -math.hypot(float(row["x"]) - float(center["x"]), float(row["y"]) - float(center["y"])),
                        float(row["dist"]),
                    ),
                    reverse=True,
                )
                for row in ordered:
                    if len(cluster) >= int(cfg.max_core):
                        break
                    if cfg.link_mode == "complete":
                        ok = all(_compatible(row, existing, cfg, neg_counts) for existing in cluster)
                    elif cfg.link_mode == "star_negall":
                        ok = _compatible(row, center, cfg, neg_counts) and all(
                            _negative_compatible(row, existing, cfg, neg_counts) for existing in cluster
                        )
                    else:
                        raise ValueError(f"unknown link_mode: {cfg.link_mode}")
                    if ok:
                        cluster.append(row)
                if len(cluster) < int(cfg.min_core):
                    continue
                features = _cluster_features(cluster, neg_counts)
                if float(features["safe_ratio"]) < float(cfg.safe_min):
                    continue
                if float(features["boundary_p10"]) < float(cfg.boundary_p10_min):
                    continue
                tids = tuple(sorted(int(row["tid"]) for row in cluster))
                if len(tids) < int(cfg.min_core):
                    continue
                score = _candidate_score(features)
                key = (cfg.name, tids)
                old = dedup.get(key)
                if old is not None and float(old["seed_score"]) >= score:
                    continue
                item: dict[str, Any] = {
                    "scene": scene,
                    "proposal_id": f"{scene}_v30_clique_{cfg.name}_f{frame_id:06d}_m{int(meas.mask_id):04d}_c{center_idx:03d}",
                    "proposal_type": "R40_cannot_link_complete_link_clique",
                    "seed_source": f"S40_clique_{cfg.name}",
                    "seed_score": score,
                    "source_config": cfg.name,
                    "link_mode": cfg.link_mode,
                    "frame_id": frame_id,
                    "mask_id": int(meas.mask_id),
                    **features,
                }
                _set_core_ids(item, tids)
                dedup[key] = item
                per_config_counts[cfg.name] += 1

    rows = list(dedup.values())
    summary = {
        "scene": scene,
        "record_count": int(len(records)),
        "measurement_count": int(len(measurements)),
        "processed_measurement_count": int(processed_measurements),
        "candidate_count": int(len(rows)),
        "negative_pair_key_count": int(len(neg_counts)),
        "measurement_diag": meas_diag,
        "per_config_raw_candidate_count": dict(sorted(per_config_counts.items())),
    }
    return rows, summary


def _config_list() -> list[CliqueConfig]:
    configs: list[CliqueConfig] = []
    for px_thr, color_thr in ((48.0, 0.24), (64.0, 0.24), (72.0, 0.32), (96.0, 0.32), (144.0, 0.32)):
        for min_core, max_core in ((2, 8), (2, 16), (3, 16), (3, 32)):
            for safe_min, boundary_p10_min in ((0.50, 0.0), (0.75, 3.0), (1.0, 6.0)):
                name = f"clique_px{int(px_thr)}_c{int(color_thr*100):02d}_n0_m{min_core}_M{max_core}_s{int(safe_min*100)}_b{int(boundary_p10_min)}"
                configs.append(
                    CliqueConfig(
                        name=name,
                        link_mode="complete",
                        px_thr=px_thr,
                        color_thr=color_thr,
                        max_neg=0,
                        min_core=min_core,
                        max_core=max_core,
                        safe_min=safe_min,
                        boundary_p10_min=boundary_p10_min,
                    )
                )
    for px_thr, color_thr in ((96.0, 0.32), (144.0, 0.32), (192.0, 0.40)):
        for min_core, max_core in ((3, 24), (4, 48), (4, 64)):
            for safe_min, boundary_p10_min in ((0.50, 0.0), (0.75, 3.0)):
                name = f"star_px{int(px_thr)}_c{int(color_thr*100):02d}_n0_m{min_core}_M{max_core}_s{int(safe_min*100)}_b{int(boundary_p10_min)}"
                configs.append(
                    CliqueConfig(
                        name=name,
                        link_mode="star_negall",
                        px_thr=px_thr,
                        color_thr=color_thr,
                        max_neg=0,
                        min_core=min_core,
                        max_core=max_core,
                        safe_min=safe_min,
                        boundary_p10_min=boundary_p10_min,
                    )
                )
    return configs


def _evaluate_selection(
    rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
    topk_per_scene: int,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for scene in scenes:
        scene_rows = [row for row in rows if str(row["scene"]) == scene]
        ranked = sorted(scene_rows, key=lambda row: (float(row["seed_score"]), int(row["core_n"])), reverse=True)
        accepted: list[set[int]] = []
        for row in ranked:
            ids = set(int(v) for v in row.get("_core_tube_ids", ()))
            if any(len(ids & old) / max(min(len(ids), len(old)), 1) >= 0.88 for old in accepted):
                continue
            item = dict(row)
            item["seed_id"] = f"{scene}_{row['source_config']}_{len(accepted):05d}"
            accepted.append(ids)
            selected.append(item)
            if len(accepted) >= int(topk_per_scene):
                break
    if not selected:
        return {"n": 0, "topk_per_scene": int(topk_per_scene), "pass_count": 0}
    metrics = _phase_c_metrics(selected, scenes, gt_by_scene)
    metrics.update(
        {
        "topk_per_scene": int(topk_per_scene),
        }
    )
    return metrics


def _read_component_feature_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            core_ids = tuple(int(v) for v in str(row.get("core_ids") or "").split(";") if str(v).strip())
            if not core_ids:
                continue
            item = dict(row)
            item["proposal_id"] = f"component_feature_{idx:06d}_{row.get('source_config')}"
            item["proposal_type"] = "R39_cannot_link_component_feature_pool"
            item["seed_source"] = f"S39_component_{row.get('source_config')}"
            item["seed_score"] = _component_score(item)
            _set_core_ids(item, tuple(sorted(core_ids)))
            rows.append(item)
    return rows


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key)
        if val in {None, ""}:
            return float(default)
        out = float(val)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _component_score(row: dict[str, Any]) -> float:
    core = min(math.log1p(_f(row, "core_n")) / math.log(161.0), 1.0)
    checks = min(math.log1p(_f(row, "component_cannot_checks")) / math.log(2000.0), 1.0)
    rate = _f(row, "component_cannot_rate")
    return float(
        1.00 * _f(row, "safe_ratio")
        + 0.85 * core
        + 0.25 * checks
        - 2.50 * abs(rate - 0.10)
        - 3.00 * max(rate - 0.15, 0.0)
        - 0.25 * min(_f(row, "mean_xy_dist") / 96.0, 3.0)
        - 0.60 * min(_f(row, "mean_color_dist") / 0.32, 3.0)
        - 0.15 * _f(row, "edge_density")
    )


def _overlap_ratio(a: set[int], b: set[int]) -> float:
    return float(len(a & b) / max(min(len(a), len(b)), 1))


def _dedupe_ordered(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_by_scene: dict[str, list[set[int]]] = defaultdict(list)
    for row in rows:
        ids = set(int(v) for v in row.get("_core_tube_ids", ()))
        if not ids:
            continue
        scene = str(row["scene"])
        if any(_overlap_ratio(ids, old) >= 0.88 for old in seen_by_scene[scene]):
            continue
        item = dict(row)
        item["seed_id"] = f"{scene}_{item.get('seed_source', 'hybrid')}_{len(seen_by_scene[scene]):05d}"
        seen_by_scene[scene].append(ids)
        out.append(item)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _select_component_base(rows: list[dict[str, Any]], scenes: list[str], mode: str) -> list[dict[str, Any]]:
    if mode.startswith("nonsafe_alt"):
        selected = [row for row in rows if str(row.get("source_config")) == "nonsafe_4of5_alt"]
        if "rate030" in mode:
            selected = [row for row in selected if _f(row, "component_cannot_rate") <= 0.30]
        if "rate020" in mode:
            selected = [row for row in selected if _f(row, "component_cannot_rate") <= 0.20]
        if "rate010" in mode:
            selected = [row for row in selected if _f(row, "component_cannot_rate") <= 0.10]
        if "core020" in mode:
            selected = [row for row in selected if _f(row, "core_n") <= 20]
        return _dedupe_ordered(sorted(selected, key=lambda row: float(row["seed_score"]), reverse=True))

    if mode.startswith("all_score_top"):
        topk = int(mode.replace("all_score_top", ""))
        out: list[dict[str, Any]] = []
        for scene in scenes:
            scene_rows = [row for row in rows if str(row["scene"]) == scene]
            out.extend(sorted(scene_rows, key=lambda row: float(row["seed_score"]), reverse=True)[:topk])
        return _dedupe_ordered(out)

    raise ValueError(f"unknown base mode: {mode}")


def _evaluate_fixed_rows(
    rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "pass_count": 0}
    return _phase_c_metrics(rows, scenes, gt_by_scene)


def _evaluate_hybrids(
    *,
    component_rows: list[dict[str, Any]],
    clique_rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
) -> list[dict[str, Any]]:
    if not component_rows or not clique_rows:
        return []
    base_modes = [
        "nonsafe_alt_all",
        "nonsafe_alt_rate030",
        "nonsafe_alt_rate020",
        "nonsafe_alt_rate010",
        "nonsafe_alt_rate020_core020",
        "all_score_top240",
        "all_score_top300",
        "all_score_top420",
    ]
    clique_topks = [80, 160, 300, 600, 1000]
    out: list[dict[str, Any]] = []
    for base_mode in base_modes:
        base = _select_component_base(component_rows, scenes, base_mode)
        for clique_topk in clique_topks:
            clique_selected: list[dict[str, Any]] = []
            for scene in scenes:
                scene_rows = [row for row in clique_rows if str(row["scene"]) == scene]
                clique_selected.extend(sorted(scene_rows, key=lambda row: float(row["seed_score"]), reverse=True)[:clique_topk])
            combined = _dedupe_ordered(base + clique_selected)
            metric = _evaluate_fixed_rows(combined, scenes, gt_by_scene)
            metric.update(
                {
                    "base_mode": base_mode,
                    "base_count_after_dedupe": int(len(base)),
                    "clique_topk_per_scene": int(clique_topk),
                    "combined_count": int(len(combined)),
                }
            )
            out.append(metric)
    out.extend(
        _evaluate_replacement_hybrids(
            component_rows=component_rows,
            clique_rows=clique_rows,
            scenes=scenes,
            gt_by_scene=gt_by_scene,
        )
    )
    return sorted(
        out,
        key=lambda row: (
            int(row.get("pass_count", 0)),
            float(row.get("purity_p10") or 0.0),
            float(row.get("GT_with_seed_IoU_ge_0.25") or 0.0),
            float(row.get("GT_with_seed_IoU_ge_0.10") or 0.0),
        ),
        reverse=True,
    )


def _component_is_risky(row: dict[str, Any], mode: str) -> bool:
    if mode == "risk_rate_gt010":
        return _f(row, "component_cannot_rate") > 0.10
    if mode == "risk_rate_gt020":
        return _f(row, "component_cannot_rate") > 0.20
    if mode == "risk_rate_gt030":
        return _f(row, "component_cannot_rate") > 0.30
    if mode == "risk_rate_gt010_or_large20":
        return _f(row, "component_cannot_rate") > 0.10 or _f(row, "core_n") > 20
    if mode == "risk_rate_gt020_or_large20":
        return _f(row, "component_cannot_rate") > 0.20 or _f(row, "core_n") > 20
    if mode == "risk_large20":
        return _f(row, "core_n") > 20
    raise ValueError(f"unknown risk mode: {mode}")


def _evaluate_replacement_hybrids(
    *,
    component_rows: list[dict[str, Any]],
    clique_rows: list[dict[str, Any]],
    scenes: list[str],
    gt_by_scene: dict[str, dict[int, int]],
) -> list[dict[str, Any]]:
    base_modes = [
        "nonsafe_alt_all",
        "all_score_top240",
        "all_score_top300",
        "all_score_top420",
    ]
    risk_modes = [
        "risk_rate_gt010",
        "risk_rate_gt020",
        "risk_rate_gt030",
        "risk_rate_gt010_or_large20",
        "risk_rate_gt020_or_large20",
        "risk_large20",
    ]
    replace_limits = [1, 2, 3]
    clique_scan_limits = [2000, 5000]
    clique_by_scene: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        scene_rows = [row for row in clique_rows if str(row["scene"]) == scene]
        clique_by_scene[scene] = sorted(scene_rows, key=lambda row: (float(row["seed_score"]), len(_row_core_ids(row))), reverse=True)
    clique_ids_by_limit: dict[int, dict[str, list[set[int]]]] = {}
    clique_index_by_limit: dict[int, dict[str, dict[int, list[int]]]] = {}
    for scan_limit in clique_scan_limits:
        ids_by_scene: dict[str, list[set[int]]] = {}
        index_by_scene: dict[str, dict[int, list[int]]] = {}
        for scene in scenes:
            top_rows = clique_by_scene[scene][: int(scan_limit)]
            id_sets = [set(_row_core_ids(row)) for row in top_rows]
            index: dict[int, list[int]] = defaultdict(list)
            for idx, ids in enumerate(id_sets):
                for tid in ids:
                    index[int(tid)].append(int(idx))
            ids_by_scene[scene] = id_sets
            index_by_scene[scene] = index
        clique_ids_by_limit[int(scan_limit)] = ids_by_scene
        clique_index_by_limit[int(scan_limit)] = index_by_scene

    out: list[dict[str, Any]] = []
    for base_mode in base_modes:
        base = _select_component_base(component_rows, scenes, base_mode)
        for risk_mode in risk_modes:
            kept = [row for row in base if not _component_is_risky(row, risk_mode)]
            skipped = [row for row in base if _component_is_risky(row, risk_mode)]
            for scan_limit in clique_scan_limits:
                for replace_limit in replace_limits:
                    replacements: list[dict[str, Any]] = []
                    for comp in skipped:
                        comp_ids = set(_row_core_ids(comp))
                        if not comp_ids:
                            continue
                        candidates: list[tuple[float, dict[str, Any]]] = []
                        scene = str(comp["scene"])
                        hit_counts: Counter[int] = Counter()
                        for tid in comp_ids:
                            hit_counts.update(clique_index_by_limit[int(scan_limit)][scene].get(int(tid), []))
                        for clique_idx, inter in hit_counts.items():
                            clique = clique_by_scene[scene][int(clique_idx)]
                            clique_ids = clique_ids_by_limit[int(scan_limit)][scene][int(clique_idx)]
                            if not clique_ids:
                                continue
                            if inter < min(2, len(clique_ids)):
                                continue
                            clique_inside = inter / max(len(clique_ids), 1)
                            comp_covered = inter / max(len(comp_ids), 1)
                            if clique_inside < 0.80:
                                continue
                            score = (
                                1.00 * clique_inside
                                + 0.35 * comp_covered
                                + 0.08 * math.log1p(len(clique_ids))
                                + 0.05 * float(clique["seed_score"])
                            )
                            candidates.append((float(score), clique))
                        for _, clique in sorted(candidates, key=lambda item: item[0], reverse=True)[: int(replace_limit)]:
                            replacements.append(clique)
                    combined = _dedupe_ordered(kept + replacements)
                    metric = _evaluate_fixed_rows(combined, scenes, gt_by_scene)
                    metric.update(
                        {
                            "base_mode": base_mode,
                            "hybrid_mode": "risky_component_replacement",
                            "risk_mode": risk_mode,
                            "base_count_after_dedupe": int(len(base)),
                            "kept_component_count": int(len(kept)),
                            "skipped_component_count": int(len(skipped)),
                            "replacement_raw_count": int(len(replacements)),
                            "clique_scan_limit_per_scene": int(scan_limit),
                            "replace_limit_per_component": int(replace_limit),
                            "combined_count": int(len(combined)),
                        }
                    )
                    out.append(metric)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    configs = _config_list()
    gt_by_scene = {
        scene: _load_gt_labels(
            Path(args.cache_root),
            scene,
            int(args.max_tubes_per_window),
            int(args.image_width),
            int(args.image_height),
        )
        for scene in scenes
    }
    all_candidates: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        rows, summary = _generate_scene_candidates(args, scene, configs)
        all_candidates.extend(rows)
        scene_summaries.append(summary)
        print(scene, "candidates", len(rows), "processed_measurements", summary["processed_measurement_count"], flush=True)

    grid_rows: list[dict[str, Any]] = []
    for cfg in configs:
        cfg_rows = [row for row in all_candidates if str(row["source_config"]) == cfg.name]
        for topk in args.topk_per_scene:
            metric = _evaluate_selection(cfg_rows, scenes, gt_by_scene, int(topk))
            metric.update(
                {
                    "source_config": cfg.name,
                    "candidate_count": int(len(cfg_rows)),
                    "px_thr": float(cfg.px_thr),
                    "color_thr": float(cfg.color_thr),
                    "max_neg": int(cfg.max_neg),
                    "min_core": int(cfg.min_core),
                    "max_core": int(cfg.max_core),
                    "link_mode": str(cfg.link_mode),
                    "safe_min": float(cfg.safe_min),
                    "boundary_p10_min": float(cfg.boundary_p10_min),
                }
            )
            grid_rows.append(metric)

    component_rows = _read_component_feature_rows(Path(args.component_feature_csv))
    hybrid_top_grid = _evaluate_hybrids(
        component_rows=component_rows,
        clique_rows=all_candidates,
        scenes=scenes,
        gt_by_scene=gt_by_scene,
    )[:100]

    top_grid = sorted(
        grid_rows,
        key=lambda row: (
            int(row.get("pass_count", 0)),
            float(row.get("purity_p10") or 0.0),
            float(row.get("GT_with_seed_IoU_ge_0.25") or 0.0),
            float(row.get("GT_with_seed_IoU_ge_0.10") or 0.0),
        ),
        reverse=True,
    )[:100]

    out_dir = Path(args.out_dir)
    candidate_preview = sorted(all_candidates, key=lambda row: float(row["seed_score"]), reverse=True)[: int(args.preview_rows)]
    _write_csv(out_dir / "native_tube_component_cannot_link_clique_candidate_preview.csv", candidate_preview)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "is_method_result": False,
        "uses_gt_for_candidate_generation": False,
        "uses_gt_for_diagnostic_metrics": True,
        "uses_rgb_for_candidate_generation": True,
        "uses_image_masks_for_candidate_generation": True,
        "uses_rgbd_for_candidate_generation": False,
        "uses_pose_for_candidate_generation": False,
        "uses_scannet_mesh_for_candidate_generation": False,
        "uses_eval_sim3_for_candidate_generation": False,
        "candidate_generation": "per-mask greedy complete-link clique and center-star candidates over D4RT uv, RGB color, boundary distance, and same-frame cannot-link negatives",
        "config_count": int(len(configs)),
        "candidate_count": int(len(all_candidates)),
        "component_feature_csv": str(args.component_feature_csv),
        "component_feature_row_count": int(len(component_rows)),
        "scene_summaries": scene_summaries,
        "top_grid": top_grid,
        "hybrid_top_grid": hybrid_top_grid,
        "artifact_files": [
            "native_tube_component_cannot_link_clique_grid.json",
            "native_tube_component_cannot_link_clique_candidate_preview.csv",
        ],
    }
    _write_json(out_dir / "native_tube_component_cannot_link_clique_grid.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v30 cannot-link complete-link seed candidates.")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--out-dir", default="outputs/audit/v30_profiles/native_rgb_boundary_diagnostic")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--max-nodes-per-mask", type=int, default=64)
    parser.add_argument("--max-centers-per-mask", type=int, default=24)
    parser.add_argument("--global-min-core", type=int, default=2)
    parser.add_argument("--topk-per-scene", type=int, nargs="+", default=[160, 240, 300, 420])
    parser.add_argument("--preview-rows", type=int, default=500)
    parser.add_argument(
        "--component-feature-csv",
        default="outputs/audit/v30_profiles/native_rgb_boundary_diagnostic/native_tube_component_cannot_link_candidate_features.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    best = payload["top_grid"][0] if payload.get("top_grid") else {}
    print(json.dumps(_json_safe(best), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
