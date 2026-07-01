from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v67_local_baselines import _row_from_eval, _summarize_variant_all  # noqa: E402
from stream4d_native.v71_representative_setcover import _load_json, _load_pipeline_roots, _mean, _rel  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _frame_data  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


@dataclass(frozen=True)
class TrackConfig:
    name: str
    min_area: float
    max_area: float
    allow_broad: bool
    max_underseg: float
    same_proto_required: bool
    allow_signature_bridge: bool
    max_gap: int
    min_bbox_iou: float
    min_mask_iou: float
    max_center_norm: float
    max_area_ratio: float
    max_per_frame: int
    singleton_fill: bool
    singleton_score_floor: float
    link_score_floor: float


CONFIGS = [
    TrackConfig(
        "AMT0_clean_same_proto_bbox_iou",
        0.008,
        0.22,
        False,
        0.75,
        True,
        False,
        1,
        0.010,
        0.000,
        0.35,
        3.5,
        48,
        False,
        0.0,
        0.35,
    ),
    TrackConfig(
        "AMT1_clean_proto_center_area_gap2",
        0.006,
        0.22,
        False,
        0.75,
        True,
        False,
        2,
        0.000,
        0.000,
        0.48,
        4.0,
        64,
        True,
        1.10,
        0.25,
    ),
    TrackConfig(
        "AMT2_clean_proto_or_signature_bridge",
        0.006,
        0.24,
        False,
        0.78,
        False,
        True,
        2,
        0.000,
        0.000,
        0.55,
        4.5,
        72,
        True,
        1.00,
        0.20,
    ),
    TrackConfig(
        "AMT3_mid_objectness_singleton_fill",
        0.010,
        0.24,
        False,
        0.78,
        True,
        False,
        2,
        0.000,
        0.000,
        0.55,
        4.5,
        72,
        True,
        1.25,
        0.18,
    ),
    TrackConfig(
        "AMT4_risky_area_temporal_bridge",
        0.008,
        0.30,
        True,
        0.88,
        False,
        True,
        2,
        0.000,
        0.000,
        0.65,
        6.0,
        96,
        True,
        0.80,
        0.10,
    ),
]


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in ("", None):
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key)
        if value in ("", None):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _b(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key)).strip().lower() in {"1", "true", "yes", "y"}


def _read_candidates(path: Path, scenes: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if scenes and row.get("scene_id") not in scenes:
                continue
            out[str(row.get("chunk_id") or "")].append(row)
    return out


def _bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (_f(row, "bbox_x0"), _f(row, "bbox_y0"), _f(row, "bbox_x1"), _f(row, "bbox_y1"))


def _bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0 + 1.0)
    ih = max(0.0, iy1 - iy0 + 1.0)
    inter = iw * ih
    aa = max(0.0, ax1 - ax0 + 1.0) * max(0.0, ay1 - ay0 + 1.0)
    ba = max(0.0, bx1 - bx0 + 1.0) * max(0.0, by1 - by0 + 1.0)
    return float(inter / max(1e-6, aa + ba - inter))


def _center_norm(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    acx, acy = (ax0 + ax1) * 0.5, (ay0 + ay1) * 0.5
    bcx, bcy = (bx0 + bx1) * 0.5, (by0 + by1) * 0.5
    diag = math.sqrt(max(1.0, max(ax1, bx1, 1.0) ** 2 + max(ay1, by1, 1.0) ** 2))
    return float(math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / max(1e-6, diag))


def _area_ratio_continuity(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = max(1e-6, _f(a, "area_ratio"))
    ba = max(1e-6, _f(b, "area_ratio"))
    return float(max(aa, ba) / min(aa, ba))


def _mask_iou(
    frame_masks: dict[int, np.ndarray | None],
    a: dict[str, Any],
    b: dict[str, Any],
    cache: dict[tuple[int, int, int, int], float],
) -> float:
    key = (_i(a, "frame_id"), _i(a, "mask_id"), _i(b, "frame_id"), _i(b, "mask_id"))
    if key in cache:
        return cache[key]
    ma = frame_masks.get(key[0])
    mb = frame_masks.get(key[2])
    if ma is None or mb is None or ma.shape != mb.shape:
        cache[key] = 0.0
        return 0.0
    aa = ma == key[1]
    bb = mb == key[3]
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    out = float(inter / max(1, union))
    cache[key] = out
    return out


def _is_usable(row: dict[str, Any], cfg: TrackConfig) -> bool:
    area = _f(row, "area_ratio")
    if area < cfg.min_area or area > cfg.max_area:
        return False
    broad = _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or area >= 0.30
    if broad and not cfg.allow_broad:
        return False
    if _f(row, "underseg_proxy_score") >= cfg.max_underseg:
        return False
    if _b(row, "small_mask_risk") and area < cfg.min_area * 1.25:
        return False
    return True


def _objectness_score(row: dict[str, Any], cfg: TrackConfig) -> float:
    area = _f(row, "area_ratio")
    entropy = _f(row, "semantic_entropy", 1.0)
    margin = _f(row, "semantic_prototype_margin")
    rel = _f(row, "D4RT_carrier_reliability_mean", 0.0)
    overlap = _f(row, "same_frame_overlap_count") + _f(row, "same_frame_competing_mask_count")
    broad = _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or area >= 0.30
    under = _f(row, "underseg_proxy_score") >= cfg.max_underseg
    mid_bonus = 1.0 if 0.012 <= area <= 0.18 else 0.0
    rep_bonus = 0.25 if _b(row, "representative_available") else 0.0
    raw_bonus = 0.15 if _b(row, "raw_cropformer_available") else 0.0
    score = 1.00 * mid_bonus + 2.00 * margin + 0.35 * entropy + 0.20 * rel + rep_bonus + raw_bonus
    score -= 0.05 * overlap
    score -= 1.40 if broad else 0.0
    score -= 1.25 if under else 0.0
    return float(score)


def _semantic_compatible(a: dict[str, Any], b: dict[str, Any], cfg: TrackConfig) -> bool:
    proto_a = str(a.get("semantic_prototype_id") or "")
    proto_b = str(b.get("semantic_prototype_id") or "")
    sig_a = str(a.get("repeated_signature_id") or "")
    sig_b = str(b.get("repeated_signature_id") or "")
    same_proto = bool(proto_a and proto_a == proto_b)
    if cfg.same_proto_required:
        return same_proto
    if same_proto:
        return True
    if cfg.allow_signature_bridge and sig_a and sig_a == sig_b:
        return True
    return False


def _link_score(
    a: dict[str, Any],
    b: dict[str, Any],
    cfg: TrackConfig,
    frame_index_gap: int,
    frame_masks: dict[int, np.ndarray | None],
    mask_iou_cache: dict[tuple[int, int, int, int], float],
) -> tuple[float, dict[str, float]]:
    if frame_index_gap <= 0 or frame_index_gap > cfg.max_gap:
        return float("-inf"), {}
    if not _semantic_compatible(a, b, cfg):
        return float("-inf"), {}
    bbox_iou = _bbox_iou(a, b)
    miou = _mask_iou(frame_masks, a, b, mask_iou_cache)
    center = _center_norm(a, b)
    area_ratio = _area_ratio_continuity(a, b)
    if bbox_iou < cfg.min_bbox_iou and miou < cfg.min_mask_iou and center > cfg.max_center_norm:
        return float("-inf"), {}
    if area_ratio > cfg.max_area_ratio:
        return float("-inf"), {}
    score = (
        1.20 * bbox_iou
        + 0.80 * miou
        + 0.55 * max(0.0, 1.0 - center / max(1e-6, cfg.max_center_norm))
        + 0.35 * max(0.0, 1.0 - math.log(max(1.0, area_ratio)) / math.log(max(1.01, cfg.max_area_ratio)))
        + 0.10 * min(_objectness_score(a, cfg), _objectness_score(b, cfg))
        - 0.08 * (frame_index_gap - 1)
    )
    if score < cfg.link_score_floor:
        return float("-inf"), {}
    return float(score), {
        "bbox_iou": float(bbox_iou),
        "mask_iou": float(miou),
        "center_norm": float(center),
        "area_ratio_continuity": float(area_ratio),
        "link_score": float(score),
    }


def _select_tracks(
    rows: list[dict[str, Any]],
    cfg: TrackConfig,
    max_tracks: int,
    frame_data: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int], int], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_masks = {int(item["frame_id"]): item.get("mask") for item in frame_data}
    frame_order = sorted({int(float(row.get("frame_id") or -1)) for row in rows})
    frame_index = {frame: idx for idx, frame in enumerate(frame_order)}
    usable_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _is_usable(row, cfg):
            usable_by_frame[_i(row, "frame_id")].append(row)
    for frame, items in list(usable_by_frame.items()):
        usable_by_frame[frame] = sorted(items, key=lambda row: _objectness_score(row, cfg), reverse=True)[: cfg.max_per_frame]

    tracks: list[dict[str, Any]] = []
    next_track_id = 1
    mask_iou_cache: dict[tuple[int, int, int, int], float] = {}
    link_rows: list[dict[str, Any]] = []
    for frame in frame_order:
        detections = usable_by_frame.get(frame, [])
        if not detections:
            continue
        active = [
            track
            for track in tracks
            if int(frame_index[frame] - frame_index[int(track["last_frame"])]) > 0
            and int(frame_index[frame] - frame_index[int(track["last_frame"])]) <= cfg.max_gap
        ]
        proposals = []
        for track in active:
            gap = int(frame_index[frame] - frame_index[int(track["last_frame"])])
            prev = track["members"][-1]
            for det in detections:
                score, details = _link_score(prev, det, cfg, gap, frame_masks, mask_iou_cache)
                if math.isfinite(score):
                    proposals.append((score, int(track["track_id"]), det, details))
        proposals.sort(key=lambda item: item[0], reverse=True)
        used_tracks: set[int] = set()
        used_detections: set[str] = set()
        track_lookup = {int(track["track_id"]): track for track in tracks}
        for score, track_id, det, details in proposals:
            obs_id = str(det.get("mask_observation_id") or f"{det.get('frame_id')}:{det.get('mask_id')}")
            if track_id in used_tracks or obs_id in used_detections:
                continue
            track = track_lookup[track_id]
            track["members"].append(det)
            track["last_frame"] = frame
            track["score"] += score + _objectness_score(det, cfg)
            track["link_scores"].append(score)
            used_tracks.add(track_id)
            used_detections.add(obs_id)
            link_rows.append(
                {
                    "variant": cfg.name,
                    "track_id": track_id,
                    "from_frame_id": _i(track["members"][-2], "frame_id"),
                    "from_mask_id": _i(track["members"][-2], "mask_id"),
                    "to_frame_id": _i(det, "frame_id"),
                    "to_mask_id": _i(det, "mask_id"),
                    **details,
                    "uses_gt_for_prediction": False,
                    "diagnostic_only": False,
                    "forbidden_for_method_table": False,
                }
            )
        for det in detections:
            obs_id = str(det.get("mask_observation_id") or f"{det.get('frame_id')}:{det.get('mask_id')}")
            if obs_id in used_detections:
                continue
            tracks.append(
                {
                    "track_id": next_track_id,
                    "members": [det],
                    "last_frame": frame,
                    "score": _objectness_score(det, cfg),
                    "link_scores": [],
                }
            )
            next_track_id += 1

    scored_tracks = []
    for track in tracks:
        members = track["members"]
        frame_count = len({int(float(row.get("frame_id") or -1)) for row in members})
        if frame_count <= 1 and not cfg.singleton_fill:
            continue
        if frame_count <= 1 and float(track["score"]) < cfg.singleton_score_floor:
            continue
        span = max(_i(row, "frame_id") for row in members) - min(_i(row, "frame_id") for row in members)
        mean_link = _mean([float(x) for x in track["link_scores"]]) or 0.0
        final_score = float(track["score"]) + 0.75 * max(0, frame_count - 1) + 0.01 * span + 0.50 * mean_link
        scored_tracks.append((final_score, track))
    scored_tracks.sort(key=lambda item: (item[0], len(item[1]["members"]), int(item[1]["track_id"])), reverse=True)
    selected = scored_tracks[:max_tracks]

    mapping: dict[tuple[int, int], int] = {}
    object_rows: list[dict[str, Any]] = []
    for object_id, (final_score, track) in enumerate(selected, start=1):
        members = track["members"]
        link_scores = [float(x) for x in track["link_scores"]]
        for row in members:
            mapping[(_i(row, "frame_id"), _i(row, "mask_id"))] = object_id
        object_rows.append(
            {
                "variant": cfg.name,
                "local_object_id": object_id,
                "track_source_id": int(track["track_id"]),
                "track_score": float(final_score),
                "member_mask_count": len(members),
                "member_frame_count": len({int(float(row.get("frame_id") or -1)) for row in members}),
                "frame_min": min(_i(row, "frame_id") for row in members),
                "frame_max": max(_i(row, "frame_id") for row in members),
                "mean_member_mask_area": _mean([_f(row, "area_ratio") for row in members]),
                "semantic_entropy_mean": _mean([_f(row, "semantic_entropy") for row in members]),
                "semantic_prototype_margin_mean": _mean([_f(row, "semantic_prototype_margin") for row in members]),
                "mean_link_score": _mean(link_scores),
                "semantic_prototype_count": len({str(row.get("semantic_prototype_id") or "") for row in members}),
                "repeated_signature_count": len({str(row.get("repeated_signature_id") or "") for row in members}),
                "broad_large_member_rate": sum(
                    1
                    for row in members
                    if _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or _f(row, "area_ratio") >= 0.30
                )
                / max(1, len(members)),
                "underseg_proxy_member_rate": sum(1 for row in members if _f(row, "underseg_proxy_score") >= cfg.max_underseg)
                / max(1, len(members)),
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )

    selected_members = [row for _, track in selected for row in track["members"]]
    diag = {
        "candidate_usable_count": sum(len(items) for items in usable_by_frame.values()),
        "raw_track_count": len(tracks),
        "selected_track_count": len(selected),
        "support_pair_count": len(mapping),
        "selected_mask_count": len(mapping),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "selected_track_member_count_mean": _mean([float(row["member_mask_count"]) for row in object_rows]),
        "selected_track_frame_count_mean": _mean([float(row["member_frame_count"]) for row in object_rows]),
        "single_frame_track_rate": sum(1 for row in object_rows if int(row["member_frame_count"]) <= 1) / max(1, len(object_rows)),
        "broad_large_member_rate": sum(
            1
            for row in selected_members
            if _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or _f(row, "area_ratio") >= 0.30
        )
        / max(1, len(selected_members)),
        "underseg_proxy_member_rate": sum(1 for row in selected_members if _f(row, "underseg_proxy_score") >= cfg.max_underseg)
        / max(1, len(selected_members)),
        "mean_link_score": _mean([float(row["link_score"]) for row in link_rows if row["variant"] == cfg.name]),
        "mean_bbox_iou": _mean([float(row["bbox_iou"]) for row in link_rows if row["variant"] == cfg.name]),
        "mean_mask_iou": _mean([float(row["mask_iou"]) for row in link_rows if row["variant"] == cfg.name]),
    }
    return mapping, object_rows, link_rows, diag


def _summarize_with_diag(metric_rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    out = []
    extra_keys = [
        "candidate_usable_count",
        "raw_track_count",
        "selected_track_count",
        "selected_track_member_count_mean",
        "selected_track_frame_count_mean",
        "single_frame_track_rate",
        "broad_large_member_rate",
        "underseg_proxy_member_rate",
        "mean_link_score",
        "mean_bbox_iou",
        "mean_mask_iou",
    ]
    for variant in variants:
        row = _summarize_variant_all(metric_rows, variant)
        subset = [item for item in metric_rows if item.get("variant") == variant]
        for key in extra_keys:
            row[f"{key}_mean"] = _mean([float(item[key]) for item in subset if item.get(key) not in ("", None)])
        out.append(row)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_by_chunk = _read_candidates(_rooted(args.candidate_rows), set(scenes))
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    atom_summary = _load_json(_rooted(args.atom_root) / "atom_summary.json")
    atom_metrics = atom_summary.get("key_metrics") if isinstance(atom_summary.get("key_metrics"), dict) else atom_summary
    diagnostic_gt_mean = float(atom_metrics.get("diagnostic_GT_count_per_chunk_mean") or 21.515923566878982)
    max_tracks = int(args.max_tracks_per_chunk or max(1, math.floor(3.0 * diagnostic_gt_mean)))
    variant_names = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    configs = [cfg for cfg in CONFIGS if cfg.name in set(variant_names)]

    object_rows_all: list[dict[str, Any]] = []
    link_rows_all: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    processed = 0
    for scene in scenes:
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in _chunk_rows(pipeline_root, scene):
            if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
                break
            chunk_id = str(chunk.get("chunk_id"))
            rows = candidates_by_chunk.get(chunk_id, [])
            if not rows:
                continue
            processed += 1
            print(f"[v71-adjacent-track] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            for cfg in configs:
                mapping, object_rows, link_rows, diag = _select_tracks(rows, cfg, max_tracks=max_tracks, frame_data=frame_data)
                for row in object_rows:
                    row.update({"scene_id": scene, "chunk_id": chunk_id})
                for row in link_rows:
                    row.update({"scene_id": scene, "chunk_id": chunk_id})
                object_rows_all.extend(object_rows)
                link_rows_all.extend(link_rows)
                metric = _row_from_eval(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=cfg.name,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    raw_per_frame_masks=False,
                    diag=diag,
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=False,
                    pipeline_root=pipeline_root,
                )
                metric.update(diag)
                metric_rows.append(metric)
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break

    summary_rows = _summarize_with_diag(metric_rows, [cfg.name for cfg in configs])
    best = max(summary_rows, key=lambda row: float(row.get("local_SF50_mean") or row.get("local_score_free_match50_recall_mean") or 0.0), default={})
    summary = {
        "decision": "ADJACENT_MASK_TRACK_REPAIR_DIAGNOSTIC_DONE",
        "processed_chunk_count": processed,
        "max_tracks_per_chunk": max_tracks,
        "variants": [cfg.name for cfg in configs],
        "best_variant": best.get("variant"),
        "best_variant_local_SF50": best.get("local_SF50_mean") or best.get("local_score_free_match50_recall_mean"),
        "best_variant_GT_best_IoU_mean": best.get("local_GT_best_IoU_mean_mean"),
        "best_variant_single_frame_track_rate": best.get("single_frame_track_rate_mean"),
        "best_variant_broad_large_member_rate": best.get("broad_large_member_rate_mean"),
        "best_variant_underseg_proxy_member_rate": best.get("underseg_proxy_member_rate_mean"),
        "summary_rows": summary_rows,
    }
    _write_csv(output_root / "adjacent_track_object_rows.csv", object_rows_all)
    _write_csv(output_root / "adjacent_track_link_rows.csv", link_rows_all)
    _write_csv(output_root / "adjacent_track_metric_rows.csv", metric_rows)
    _write_csv(output_root / "adjacent_track_variant_summary_rows.csv", summary_rows)
    (output_root / "adjacent_track_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sha_rows = [
        {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_root.glob("*"))
        if path.is_file()
    ]
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--atom-root", default="outputs/audit/v71_d4rt_atoms")
    parser.add_argument("--output-root", default="outputs/audit/v71_adjacent_mask_track_repair")
    parser.add_argument(
        "--variants",
        default=",".join(cfg.name for cfg in CONFIGS),
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-tracks-per-chunk", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
